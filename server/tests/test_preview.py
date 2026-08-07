"""The software renderer and the two preview endpoints.

These tests are about whether a *defect stays visible*, not whether the picture
is pretty. Three of them exist because of specific ways a preview can quietly
stop being useful while still returning a plausible PNG:

- a part lifted off the floor renders identically to one resting on it,
- highlighting a part disturbs the rest of the image, so a before/after
  comparison stops meaning anything,
- the camera re-frames when parts are hidden, which is exactly how a floating
  part hides from the render harness that is supposed to catch it.
"""
import numpy as np
import pytest
import trimesh

import materials
import preview
import texturing


def _box(size=(1.0, 1.0, 1.0), at=(0.0, 0.0, 0.0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(at)
    return mesh


def _part(name, size=(1.0, 1.0, 1.0), at=(0.0, 0.0, 0.0)) -> preview.Part:
    mesh = _box(size, at)
    return preview.Part(
        name,
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces),
        np.tile([0.75, 0.75, 0.75], (len(mesh.faces), 1)),
    )


def _scene(parts, path):
    """parts: [(name, mesh)] -> a .glb on disk, one node per part."""
    scene = trimesh.Scene()
    for name, mesh in parts:
        scene.add_geometry(mesh, node_name=name, geom_name=name)
    scene.export(str(path))
    return path


def _render(parts, view="side", tile=180, framing=None, **kw):
    """One view as an (H, W, 3) uint8 array.

    `framing` is passed in wherever two renders have to be compared, so the
    comparison is about the geometry and never about the camera.
    """
    framing = framing or preview.Framing.of(parts)
    draw = preview.build_draw_list(parts, framing, **kw)
    cam = preview.camera_for(view, framing, tile, tile)
    return preview.render_view(draw, cam, tile, tile)


def _changed_fraction(a, b, tolerance=8):
    return float((np.abs(a.astype(int) - b.astype(int)).max(axis=2) > tolerance).mean())


# --------------------------------------------------------------------------
# the rasteriser
# --------------------------------------------------------------------------
def test_batched_rasterizer_agrees_with_the_reference_one():
    """The fast path must be the same algorithm, not merely a similar one.

    texturing.rasterize is the version that has been shipping against real
    reference photos; if the batched rewrite disagreed with it, every pixel this
    module draws would be suspect.
    """
    rng = np.random.default_rng(7)
    xy = rng.uniform(-20, 220, size=(300, 2))
    depth = rng.uniform(0.5, 4.0, size=300)
    faces = rng.integers(0, 300, size=(400, 3))
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])]

    want_z, want_f = texturing.rasterize(xy, depth, faces, 200, 200)
    got_z, got_f = preview._rasterize(xy, depth, faces, 200, 200)

    assert np.allclose(want_z, got_z, equal_nan=True)
    assert np.array_equal(want_f, got_f)


def test_rasterizer_keeps_the_nearer_triangle():
    xy = np.array([[10.0, 10.0], [90.0, 10.0], [10.0, 90.0]] * 2)
    xy[3:] += 2.0
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    depth = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])

    _, fbuf = preview._rasterize(xy, depth, faces, 100, 100)

    assert fbuf[40, 40] == 1                      # the near triangle wins
    assert (fbuf == 0).any() and (fbuf == 1).any()  # both are drawn somewhere


def test_rasterizer_survives_geometry_entirely_off_screen():
    xy = np.array([[-500.0, -500.0], [-490.0, -500.0], [-500.0, -490.0]])
    _, fbuf = preview._rasterize(xy, np.ones(3), np.array([[0, 1, 2]]), 32, 32)

    assert (fbuf == -1).all()


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------
def test_a_red_part_reads_red(tmp_path):
    mesh = _box()
    materials.apply_to_mesh(mesh, "body", color="#cc2222")
    parts = preview.load_parts(_scene([("body", mesh)], tmp_path / "red.glb"))

    rgb = parts[0].face_rgb
    assert (rgb[:, 0] > 0.6).all()
    assert (rgb[:, 1] < 0.3).all() and (rgb[:, 2] < 0.3).all()


def test_material_colour_is_gamma_encoded_for_display(tmp_path):
    """glTF baseColorFactor is linear. Showing it raw comes out visibly dark."""
    mesh = _box()
    materials.apply_to_mesh(mesh, "body", color="#808080")
    parts = preview.load_parts(_scene([("body", mesh)], tmp_path / "grey.glb"))

    # #808080 is 0.216 in linear light and must come back near 0.5, not 0.216.
    assert parts[0].face_rgb[0][0] == pytest.approx(0.5, abs=0.03)


def test_an_unpainted_part_is_tinted_rather_than_trimesh_grey(tmp_path):
    """Two touching grey parts merge into one silhouette; tinted ones do not."""
    parts = preview.load_parts(
        _scene([("wing", _box()), ("strut", _box(at=(2, 0, 0)))], tmp_path / "p.glb")
    )
    by_name = {p.name: p.face_rgb[0] for p in parts}

    assert not np.allclose(by_name["wing"], by_name["strut"])
    # ...but only just tinted. A saturated palette would claim a colour the
    # part does not have.
    for rgb in by_name.values():
        assert rgb.max() - rgb.min() < 0.15


def test_the_same_part_name_always_gets_the_same_tint():
    assert np.array_equal(preview._clay("wing"), preview._clay("wing"))
    assert not np.array_equal(preview._clay("wing"), preview._clay("fin"))


# --------------------------------------------------------------------------
# the floor: the thing that makes a floating part visible
# --------------------------------------------------------------------------
def test_a_part_above_the_floor_renders_differently_from_one_on_it():
    """The defect this whole module exists to catch.

    Both scenes contain the same two boxes; in one, the second box has been
    lifted. The camera is deliberately identical for the two renders, so this
    measures the picture and not the framing.
    """
    hull = _part("hull", (2, 1, 1))
    seated = _part("fin", (0.4, 0.6, 0.4), (0, 0.8, 0))
    lifted = _part("fin", (0.4, 0.6, 0.4), (0, 1.9, 0))
    framing = preview.Framing.of([hull, lifted])

    a = _render([hull, seated], "three_qtr", framing=framing)
    b = _render([hull, lifted], "three_qtr", framing=framing)

    assert _changed_fraction(a, b) > 0.02, "lifting a part barely changed it"


def test_a_lifted_part_casts_its_shadow_somewhere_else():
    """A shadow that has parted company with its part is the tell.

    Tested on the shadow geometry rather than on pixels, because "the picture
    changed" would also pass if the part simply drew a few rows higher, and the
    displaced shadow is the specific thing that makes floating unmistakable.
    """
    framing = preview.Framing(
        center=np.zeros(3), radius=2.0, ground_y=0.0, fit=1.0
    )
    seated = _part("fin", (0.5, 0.5, 0.5), (0, 0.25, 0))
    lifted = _part("fin", (0.5, 0.5, 0.5), (0, 3.0, 0))

    low, _, _ = preview._shadow_geometry(seated.vertices, seated.faces, framing)
    high, _, _ = preview._shadow_geometry(lifted.vertices, lifted.faces, framing)

    # Both land on the floor...
    assert np.allclose(low[:, 1], framing.ground_y, atol=1e-2)
    assert np.allclose(high[:, 1], framing.ground_y, atol=1e-2)
    # ...but the lifted one's is displaced by more than its own footprint, so
    # the two cannot be mistaken for each other.
    drift = np.linalg.norm(high[:, [0, 2]].mean(axis=0) - low[:, [0, 2]].mean(axis=0))
    assert drift > 0.5


def test_turning_the_floor_off_changes_the_picture():
    """Guards the floor actually being drawn rather than merely computed."""
    parts = [_part("hull", (2, 1, 1))]
    framing = preview.Framing.of(parts)

    with_floor = _render(parts, "side", framing=framing)
    without = _render(parts, "side", framing=framing, ground=False, shadows=False)

    assert _changed_fraction(with_floor, without) > 0.2


def test_ground_report_measures_the_gap(tmp_path):
    path = _scene(
        [("base", _box((2, 1, 2))), ("fin", _box((0.5, 0.5, 0.5), at=(0, 2, 0)))],
        tmp_path / "gap.glb",
    )
    report = preview.ground_report(preview.load_parts(path))

    assert report[0]["name"] == "fin"
    # base spans y -0.5..0.5, fin 1.75..2.25, so the floor is at -0.5.
    assert report[0]["gap"] == pytest.approx(2.25, abs=1e-3)
    assert report[-1]["name"] == "base" and report[-1]["gap"] == 0.0


# --------------------------------------------------------------------------
# highlighting and isolation
# --------------------------------------------------------------------------
def _part_pixels(parts, framing, view, tile, index):
    draw = preview.build_draw_list(parts, framing, list(range(len(parts))))
    cam = preview.camera_for(view, framing, tile, tile)
    xy, depth = cam.project(draw.vertices)
    _, fbuf = preview._rasterize(xy, depth, draw.faces, tile, tile)
    covered = fbuf >= 0
    mask = np.zeros((tile, tile), dtype=bool)
    mask[covered] = draw.part_id[fbuf[covered]] == index
    return mask


def test_highlighting_a_part_changes_only_that_part_s_pixels():
    """Otherwise a before/after pair proves nothing about where the part is."""
    parts = [_part("hull", (2, 1, 1)), _part("fin", (0.4, 0.6, 0.4), (0, 1.4, 0))]
    framing = preview.Framing.of(parts)
    tile = 180

    plain = _render(parts, "three_qtr", tile, framing=framing)
    lit = _render(parts, "three_qtr", tile, framing=framing, highlight=1)

    changed = (plain != lit).any(axis=2)
    fin = _part_pixels(parts, framing, "three_qtr", tile, 1)

    assert changed.any(), "highlighting did nothing"
    assert not (changed & ~fin).any(), "highlighting leaked outside the part"


def _magenta(image):
    img = image.astype(int)
    return (img[..., 0] > 140) & (img[..., 1] < 90) & (img[..., 2] > 60)


def test_isolating_a_part_leaves_it_where_it_was():
    """The point of isolation, checked through the sheet the endpoint returns.

    Every pixel the fin occupies in the full render must still be fin in the
    isolated one — not the reverse, since the full render may occlude some of
    it. Nothing here would hold if render_sheet re-framed on what it can see.
    """
    parts = [_part("hull", (2, 1, 1)), _part("fin", (0.4, 0.6, 0.4), (0.3, 0.9, 0))]

    def sheet(isolate):
        img = preview.render_sheet(
            parts, views=["three_qtr"], size=200, columns=1,
            highlight="fin", isolate=isolate,
        )
        return _magenta(np.asarray(img.convert("RGB")))

    full, alone = sheet(False), sheet(True)

    assert full.any() and alone.any()
    assert not (full & ~alone).any(), "the isolated part is not where it was"


def test_framing_a_part_alone_would_have_moved_the_camera():
    """Why isolation has to reuse the scene's framing rather than recompute it.

    The camera an isolated fin would get on its own is nowhere near the one the
    whole scene gets. Auto-framing per render is what let a floating part look
    fine in the harness that was meant to catch it.
    """
    parts = [_part("hull", (2, 1, 1)), _part("fin", (0.4, 0.4, 0.4), (0, 4, 0))]

    whole = preview.Framing.of(parts)
    alone = preview.Framing.of([parts[1]])

    assert alone.radius < whole.radius / 3
    assert alone.ground_y > whole.ground_y + 3
    assert not np.allclose(alone.center, whole.center)


def test_every_view_shares_one_camera_distance():
    parts = [_part("hull", (2, 1, 3))]
    framing = preview.Framing.of(parts)
    cams = [preview.camera_for(v, framing, 200, 200) for v in preview.VIEWS]

    assert len({round(c.scale, 9) for c in cams}) == 1
    assert len({round(c.persp, 9) for c in cams}) == 1
    assert len({round(c.radius, 9) for c in cams}) == 1


# --------------------------------------------------------------------------
# sheet assembly
# --------------------------------------------------------------------------
def test_preview_png_is_a_png_of_the_requested_shape(tmp_path):
    path = _scene([("hull", _box((2, 1, 1)))], tmp_path / "one.glb")

    data = preview.preview_png(path, views=["side", "top"], size=600, columns=2)

    assert data[:4] == b"\x89PNG"
    from io import BytesIO

    from PIL import Image
    assert Image.open(BytesIO(data)).size[0] == 600


def test_render_sheet_rejects_an_unknown_view(tmp_path):
    parts = preview.load_parts(_scene([("hull", _box())], tmp_path / "one.glb"))

    with pytest.raises(ValueError, match="unknown view"):
        preview.render_sheet(parts, views=["sideways"])


def test_render_sheet_rejects_an_unknown_part_name(tmp_path):
    parts = preview.load_parts(_scene([("hull", _box())], tmp_path / "one.glb"))

    with pytest.raises(ValueError, match="no part named"):
        preview.render_sheet(parts, highlight="nope")


def test_load_parts_places_nodes_in_world_space(tmp_path):
    scene = trimesh.Scene()
    scene.add_geometry(_box(), node_name="a", geom_name="a")
    scene.add_geometry(
        _box(), node_name="b", geom_name="b",
        transform=trimesh.transformations.translation_matrix([5, 0, 0]),
    )
    path = tmp_path / "two.glb"
    scene.export(str(path))

    parts = {p.name: p for p in preview.load_parts(path)}

    assert set(parts) == {"a", "b"}
    assert parts["b"].vertices[:, 0].mean() == pytest.approx(5.0, abs=1e-4)


def test_load_parts_reads_a_single_mesh_file(tmp_path):
    path = tmp_path / "solo.glb"
    _box().export(str(path))

    parts = preview.load_parts(path)

    assert len(parts) == 1 and len(parts[0].faces) == 12


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def test_job_preview_returns_a_png(client, finished_job):
    job_id = finished_job("hull")

    response = client.get(f"/jobs/{job_id}/preview?size=400&columns=2&views=side,top")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"


def test_job_preview_404s_and_409s_like_the_other_job_routes(client):
    queued = client.post("/jobs", json={"image_b64": "x"}).json()["id"]

    assert client.get("/jobs/nosuchjob123/preview").status_code == 404
    assert client.get(f"/jobs/{queued}/preview").status_code == 409


def test_scene_preview_returns_a_png(client, finished_job):
    scene_id = client.post("/assemble", json={"parts": [
        {"job_id": finished_job("hull"), "name": "hull"},
        {"job_id": finished_job("fin"), "name": "fin", "position": [0, 6, 0]},
    ]}).json()["scene_id"]

    response = client.get(f"/scenes/{scene_id}/preview?size=500&columns=2")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:4] == b"\x89PNG"


def test_scene_preview_404s_for_an_unknown_scene(client):
    assert client.get("/scenes/nosuchscene/preview").status_code == 404


def test_scene_preview_400s_on_a_view_or_part_that_does_not_exist(client, finished_job):
    scene_id = client.post("/assemble", json={
        "parts": [{"job_id": finished_job("hull"), "name": "hull"}]
    }).json()["scene_id"]

    assert client.get(f"/scenes/{scene_id}/preview?views=sideways").status_code == 400
    assert client.get(f"/scenes/{scene_id}/preview?highlight=nope").status_code == 400


def test_scene_preview_rejects_a_silly_image_size(client, finished_job):
    scene_id = client.post("/assemble", json={
        "parts": [{"job_id": finished_job("hull"), "name": "hull"}]
    }).json()["scene_id"]

    assert client.get(f"/scenes/{scene_id}/preview?size=4").status_code == 422
    assert client.get(f"/scenes/{scene_id}/preview?size=9000").status_code == 422


def test_scene_ground_reports_the_gap_under_each_part(client, finished_job):
    scene_id = client.post("/assemble", json={"parts": [
        {"job_id": finished_job("hull"), "name": "hull"},
        {"job_id": finished_job("fin"), "name": "fin", "position": [0, 6, 0]},
    ]}).json()["scene_id"]

    body = client.get(f"/scenes/{scene_id}/ground").json()

    assert body["parts"][0]["name"] == "fin"
    assert body["parts"][0]["gap"] == pytest.approx(6.0, abs=1e-3)
    assert body["parts"][-1]["gap"] == 0.0


def test_preview_views_lists_the_catalogue(client):
    body = client.get("/preview/views").json()

    assert set(body["default"]) <= set(body["views"])
    assert "top" in body["views"] and "three_qtr" in body["views"]
