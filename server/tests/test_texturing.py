"""Back-projection texturing, on a synthetic scene with a known answer.

The point of this module is that the colours it produces are *correct*, not
merely present, so these tests are built around cases where the right answer is
known in advance and can be asserted rather than eyeballed. The rig is a box
whose six faces are six different flat colours, rendered to a reference image by
the module's own projection maths; the texturing pass then has to recover those
colours on the faces the camera saw and not on the ones it did not.

A mean-and-variance check would pass on noise -- that is exactly how the texture
failure in docs/QUALITY-COMPARISON.md went undetected -- so nothing here asserts
on aggregate statistics of an atlas. Every assertion is either a specific colour
in a specific place or an exact geometric property.

No GPU, no network, no Blender. Whether the result *looks* right is a render,
and the ones that were looked at are in docs/TEXTURING.md.
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

import texturing


# --------------------------------------------------------------------------
# fixtures: a box with a known colour per face, and a reference image of it
# --------------------------------------------------------------------------
FACE_COLORS = {
    "+x": (220, 40, 40),
    "-x": (40, 160, 220),
    "+y": (40, 200, 80),
    "-y": (240, 200, 40),
    "+z": (150, 60, 200),
    "-z": (250, 140, 30),
}
_AXIS_ORDER = ["+x", "-x", "+y", "-y", "+z", "-z"]


def _side_of(normal):
    axis = int(np.argmax(np.abs(normal)))
    return "+-"[int(normal[axis] < 0)] + "xyz"[axis]


@pytest.fixture
def box():
    """A unit box, subdivided so each side is many triangles, not two."""
    mesh = trimesh.creation.box((1.0, 1.0, 1.0))
    verts, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=0.12
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


@pytest.fixture
def known_camera():
    """A 3/4 view that sees +x, +y and +z and cannot see the other three."""
    return texturing.Camera(
        yaw=np.deg2rad(-35.0),
        pitch=np.deg2rad(25.0),
        roll=0.0,
        persp=0.25,
        scale=110.0,
        cx=192.0,
        cy=192.0,
        center=np.zeros(3),
        radius=0.5,
    )


@pytest.fixture
def reference(box, known_camera):
    """Render `box` through `known_camera`, painting each side its own colour.

    Rendered with the module's own rasteriser on purpose: the test is about
    whether projection round-trips, and introducing a second, differently
    calibrated renderer would only ever test that the two agree.
    """
    xy, depth = known_camera.project(box.vertices)
    _, fbuf = texturing.rasterize(xy, depth, box.faces, 384, 384)
    canvas = np.full((384, 384, 3), 255, dtype=np.uint8)
    sides = np.array([_AXIS_ORDER.index(_side_of(n)) for n in box.face_normals])
    palette = np.array([FACE_COLORS[k] for k in _AXIS_ORDER], dtype=np.uint8)
    hit = fbuf >= 0
    canvas[hit] = palette[sides[fbuf[hit]]]
    return Image.fromarray(canvas)


def _face_sides(mesh):
    return np.array([_side_of(n) for n in mesh.face_normals])


def _sampled_face_colors(textured, n_faces):
    """Mean atlas colour under each of the first `n_faces` faces' UVs."""
    atlas = np.asarray(textured.visual.material.baseColorTexture.convert("RGB"))
    h, w = atlas.shape[:2]
    uv = textured.visual.uv[textured.faces]
    px = np.clip((uv[:, :, 0] * w).astype(int), 0, w - 1)
    py = np.clip(((1.0 - uv[:, :, 1]) * h).astype(int), 0, h - 1)
    return atlas[py, px].mean(axis=1)[:n_faces]


# --------------------------------------------------------------------------
# matting
# --------------------------------------------------------------------------
def test_matte_uses_a_real_alpha_channel_when_there_is_one():
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    arr[4:12, 4:12, 3] = 255

    mask = texturing.alpha_matte(Image.fromarray(arr, "RGBA"))

    assert mask.sum() == 64
    assert mask[8, 8] and not mask[0, 0]


def test_matte_floods_white_inward_from_the_border_when_alpha_is_opaque():
    arr = np.full((32, 32, 3), 255, dtype=np.uint8)
    arr[8:24, 8:24] = (10, 20, 30)

    mask = texturing.alpha_matte(Image.fromarray(arr))

    assert mask.sum() == 256
    assert not mask[0, 0]


def test_matte_keeps_white_that_the_background_cannot_reach():
    # docs/QUALITY-COMPARISON.md's reason for flood-filling rather than
    # thresholding: a white bumper on a white backdrop must survive.
    arr = np.full((32, 32, 3), 255, dtype=np.uint8)
    arr[8:24, 8:24] = (10, 20, 30)
    arr[12:20, 12:20] = 255  # an enclosed white region

    mask = texturing.alpha_matte(Image.fromarray(arr))

    assert mask[16, 16]
    assert mask.sum() == 256


def test_a_reference_that_mattes_to_nothing_is_rejected(box):
    blank = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="matted to nothing"):
        texturing.texture_from_reference(box, blank)


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
def test_projection_is_orthographic_when_perspective_is_zero():
    cam = texturing.Camera(persp=0.0, scale=100.0, cx=50.0, cy=50.0, radius=1.0)
    near = np.array([[0.5, 0.0, 1.0]])
    far = np.array([[0.5, 0.0, -1.0]])

    assert cam.project(near)[0][0, 0] == pytest.approx(cam.project(far)[0][0, 0])


def test_perspective_makes_nearer_things_bigger():
    cam = texturing.Camera(persp=0.4, scale=100.0, cx=50.0, cy=50.0, radius=1.0)
    near_x = cam.project(np.array([[0.5, 0.0, 1.0]]))[0][0, 0]
    far_x = cam.project(np.array([[0.5, 0.0, -1.0]]))[0][0, 0]

    assert near_x > far_x


def test_projection_depth_grows_away_from_the_camera():
    cam = texturing.Camera(persp=0.3, radius=1.0)
    _, depth = cam.project(np.array([[0, 0, 1.0], [0, 0, -1.0]]))

    assert depth[0] < depth[1]


def test_fit_camera_recovers_a_known_view(box, known_camera, reference):
    mask = texturing.alpha_matte(reference)

    fitted = texturing.fit_camera(box, mask, refine_iters=1200)

    # A cube has 48 poses that produce the same silhouette, so the recovered
    # *angles* are not comparable to the originals -- what has to match is where
    # the surface lands on the image, which is what the silhouette IoU measures,
    # and how big the fit thinks the object is.
    assert fitted.iou > 0.85
    assert fitted.scale == pytest.approx(known_camera.scale, rel=0.15)
    assert fitted.center == pytest.approx((box.bounds[0] + box.bounds[1]) / 2, abs=1e-9)


def test_fit_camera_reports_a_poor_score_for_the_wrong_object(box):
    # A thin ring cannot explain a cube's silhouette, and the caller needs to be
    # able to tell that from the number rather than from the render.
    ring = trimesh.creation.annulus(r_min=0.9, r_max=1.0, height=0.02)
    arr = np.zeros((128, 128, 4), dtype=np.uint8)
    arr[32:96, 32:96, 3] = 255

    fitted = texturing.fit_camera(ring, texturing.alpha_matte(Image.fromarray(arr)),
                                  refine_iters=600)

    assert fitted.iou < 0.7


def test_camera_as_dict_is_json_safe(known_camera):
    import json

    json.dumps(known_camera.as_dict())


# --------------------------------------------------------------------------
# rasteriser
# --------------------------------------------------------------------------
def test_rasterizer_z_buffers_the_nearer_triangle():
    verts = np.array([[10.0, 10.0], [90.0, 10.0], [50.0, 90.0]])
    xy = np.vstack([verts, verts])
    depth = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    zbuf, fbuf = texturing.rasterize(xy, depth, faces, 100, 100)

    assert fbuf[40, 50] == 1
    assert zbuf[40, 50] == pytest.approx(1.0)
    assert fbuf[0, 0] == -1


def test_rasterizer_ignores_triangles_that_are_entirely_off_screen():
    xy = np.array([[-50.0, -50.0], [-40.0, -50.0], [-45.0, -40.0]])
    faces = np.array([[0, 1, 2]])

    _, fbuf = texturing.rasterize(xy, np.ones(3), faces, 32, 32)

    assert (fbuf == -1).all()


def test_uv_rasterizer_reports_barycentrics_that_reconstruct_position(box):
    uv = np.zeros((len(box.vertices), 2))
    uv[:, 0] = (box.vertices[:, 0] + 0.5)
    uv[:, 1] = (box.vertices[:, 1] + 0.5)

    fbuf, bary = texturing.rasterize_uv(uv * 64, box.faces, 64, 64)

    covered = fbuf >= 0
    assert covered.any()
    assert bary[covered].sum(axis=1) == pytest.approx(np.ones(covered.sum()), abs=1e-4)


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------
def test_symmetry_is_found_on_a_plane_that_is_not_axis_aligned():
    # The case that matters: a generator emits its subject in the input
    # camera's frame, so the mirror plane sits at an angle in the mesh's own
    # bounding box and an X/Y/Z-only test misses it entirely.
    mesh = trimesh.creation.box((1.0, 0.4, 0.2))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 5, (0, 0, 1)))

    plane, score = texturing.detect_symmetry(mesh)

    assert plane is not None
    assert score > 0.8
    # Mirroring must map the surface back onto itself.
    reflected = plane.reflect(mesh.vertices)
    assert np.abs(reflected).max(axis=0) == pytest.approx(
        np.abs(mesh.vertices).max(axis=0), abs=0.06
    )


def test_symmetry_plane_reflection_is_an_involution():
    mesh = trimesh.creation.box((1.0, 0.5, 0.3))
    plane, _ = texturing.detect_symmetry(mesh)

    twice = plane.reflect(plane.reflect(mesh.vertices))

    assert twice == pytest.approx(mesh.vertices, abs=1e-9)


# --------------------------------------------------------------------------
# the projection: correct colours in the right places
# --------------------------------------------------------------------------
def test_uv_mode_paints_the_seen_sides_with_their_own_colours(
    box, known_camera, reference
):
    out, stats = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    sides = _face_sides(box)
    sampled = _sampled_face_colors(out, len(box.faces))
    for side in ("+x", "+y", "+z"):
        # The median, not the mean: a minority of grazing triangles round the
        # silhouette are deliberately *not* painted from the camera, and
        # averaging lets them drag an otherwise exact answer off.
        got = np.median(sampled[sides == side], axis=0)
        assert got == pytest.approx(FACE_COLORS[side], abs=5), side
    assert stats["direct"] > 0.4 * len(box.faces)


def test_no_hidden_face_is_ever_painted_from_the_camera(box, known_camera):
    # The depth test earning its keep: without it every -x/-y/-z triangle would
    # sample straight through the box and come back red/green/purple.
    xy, depth = known_camera.project(box.vertices)

    visible, _ = texturing._face_visibility(
        box, known_camera, xy, depth, 384, 384,
        known_camera.scale / known_camera.radius,
    )

    sides = _face_sides(box)
    for hidden in ("-x", "-y", "-z"):
        assert not visible[sides == hidden].any(), hidden
    for seen in ("+x", "+y", "+z"):
        assert visible[sides == seen].mean() > 0.7, seen


def test_projected_uvs_stay_inside_the_reference_and_out_of_the_swatch_strip(box):
    # A subject that overflows the frame projects silhouette corners past the
    # image edge. The atlas is taller than the image, so an unclamped UV walks
    # off the bottom of the photo into the fallback strip -- and renders as a
    # swatch, or as black where the strip is unused.
    big = texturing.Camera(yaw=0.4, pitch=0.3, persp=0.25, scale=260.0,
                           cx=192.0, cy=192.0, center=np.zeros(3), radius=0.5)
    xy, depth = big.project(box.vertices)
    _, fbuf = texturing.rasterize(xy, depth, box.faces, 384, 384)
    canvas = np.full((384, 384, 3), 255, dtype=np.uint8)
    canvas[fbuf >= 0] = (200, 30, 30)

    out, stats = texturing.texture_from_reference(
        box, Image.fromarray(canvas), mode="uv", camera=big
    )

    assert xy.max() > 384  # the rig really does overflow
    sampled = _sampled_face_colors(out, len(box.faces))
    assert sampled.min() > 0


def test_nothing_is_left_black(box, known_camera, reference):
    # The failure mode this replaces is an unlit black shell. Every face has to
    # end up with *some* colour off the subject, not the atlas's zero fill.
    out, _ = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    sampled = _sampled_face_colors(out, len(box.faces))
    assert sampled.min(axis=1).min() > 20


def test_uv_mode_unmerges_vertices_so_faces_do_not_share_uvs(box, known_camera,
                                                             reference):
    out, _ = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    assert len(out.vertices) == 3 * len(box.faces)
    assert len(out.faces) == len(box.faces)
    assert len(out.visual.uv) == len(out.vertices)
    assert (out.visual.uv >= 0).all() and (out.visual.uv <= 1).all()


def test_uv_mode_geometry_is_unchanged(box, known_camera, reference):
    out, _ = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    assert out.extents == pytest.approx(box.extents)
    assert out.area == pytest.approx(box.area)


def test_vertex_mode_paints_the_seen_sides_with_their_own_colours(
    box, known_camera, reference
):
    out, stats = texturing.texture_from_reference(
        box, reference, mode="vertex", camera=known_camera
    )

    colors = out.visual.vertex_colors[:, :3].astype(float)
    face_rgb = colors[box.faces].mean(axis=1)
    sides = _face_sides(box)
    for side in ("+x", "+y", "+z"):
        assert np.median(face_rgb[sides == side], axis=0) == pytest.approx(
            FACE_COLORS[side], abs=5
        ), side
    assert stats["mode"] == "vertex"
    assert len(out.vertices) == len(box.vertices)


def test_atlas_mode_needs_uvs_and_says_so(box, known_camera, reference):
    with pytest.raises(ValueError, match="already carry UVs"):
        texturing.texture_from_reference(
            box, reference, mode="atlas", camera=known_camera
        )


def test_atlas_mode_bakes_into_the_existing_unwrap(box, known_camera, reference):
    uv = np.column_stack([
        (box.vertices[:, 0] + 0.5) * 0.98 + 0.01,
        (box.vertices[:, 1] + 0.5) * 0.98 + 0.01,
    ])
    boxed = box.copy()
    boxed.visual = trimesh.visual.TextureVisuals(uv=uv)

    out, stats = texturing.texture_from_reference(
        boxed, reference, mode="atlas", camera=known_camera, texture_size=256
    )

    # The unwrap is preserved exactly; only the colour in it is new.
    assert out.visual.uv == pytest.approx(uv)
    assert out.visual.material.baseColorTexture.size == (256, 256)
    assert stats["texels"] > 0
    assert stats["direct"] > 0
    atlas = np.asarray(out.visual.material.baseColorTexture.convert("RGB"))
    assert atlas.max() > 0


def test_mirroring_paints_the_far_side_of_a_symmetric_subject(
    box, known_camera, reference
):
    with_mirror, stats_on = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera, mirror=True
    )
    _, stats_off = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera, mirror=False
    )

    assert stats_on["mirrored"] > 0
    assert stats_off["mirrored"] == 0
    assert stats_on["flooded"] < stats_off["flooded"]
    assert stats_on["symmetry_score"] > 0.8
    # Nothing the mirror touched may end up unpainted or black.
    assert _sampled_face_colors(with_mirror, len(box.faces)).min() > 0


def test_every_face_is_accounted_for_exactly_once(box, known_camera, reference):
    _, stats = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    assert stats["direct"] + stats["mirrored"] + stats["flooded"] == len(box.faces)
    assert stats["coverage"] == pytest.approx(
        (stats["direct"] + stats["mirrored"]) / len(box.faces), abs=1e-3
    )


# --------------------------------------------------------------------------
# fallback colour
# --------------------------------------------------------------------------
def test_dominant_colour_is_the_mode_not_the_mean():
    # The mean of a mostly-white subject with a navy stripe is pale blue-grey,
    # a colour that appears nowhere on it. The mode is white.
    rgb = np.full((64, 64, 3), 250, dtype=np.uint8)
    rgb[:, :8] = (10, 10, 90)
    mask = np.ones((64, 64), dtype=bool)

    dominant = texturing._dominant_color(rgb, mask)

    assert dominant[0] > 200 and dominant[2] > 200


def test_flood_fades_to_the_fallback_rather_than_smearing_a_seed_colour():
    # A chain of faces seeded from one dark neighbour must not come out dark all
    # the way along; deep in the unseen region it should be the fallback.
    n = 40
    colors = np.zeros((n, 3))
    colors[0] = (10, 10, 10)
    known = np.zeros(n, dtype=bool)
    known[0] = True
    a = np.arange(n - 1)
    b = a + 1

    out = texturing._flood(colors, known, a, b, np.array([250, 250, 250]))

    assert out[0] == pytest.approx([10, 10, 10])
    assert out[1].max() < 100          # continuous with its seed
    assert out[-1] == pytest.approx([250, 250, 250], abs=1)


def test_flood_falls_back_when_nothing_was_painted():
    colors = np.zeros((5, 3))
    known = np.zeros(5, dtype=bool)

    out = texturing._flood(colors, known, np.array([0]), np.array([1]),
                           np.array([7, 8, 9]))

    assert out == pytest.approx(np.tile([7, 8, 9], (5, 1)))


def test_quantize_does_not_average_across_hues():
    colors = np.array([[10, 10, 120]] * 50 + [[110, 60, 10]] * 50, dtype=float)

    palette, assign = texturing._quantize(colors, 256)

    assert len(np.unique(assign)) == 2
    for i, want in ((assign[0], [10, 10, 120]), (assign[-1], [110, 60, 10])):
        assert palette[i] == pytest.approx(want, abs=2)


# --------------------------------------------------------------------------
# entry point contract
# --------------------------------------------------------------------------
def test_unknown_mode_is_rejected(box, reference):
    with pytest.raises(ValueError, match="mode must be one of"):
        texturing.texture_from_reference(box, reference, mode="magic")


def test_a_mesh_with_no_faces_is_rejected(reference):
    empty = trimesh.Trimesh(vertices=np.zeros((3, 3)), faces=np.zeros((0, 3), int),
                            process=False)

    with pytest.raises(ValueError, match="no faces"):
        texturing.texture_from_reference(empty, reference)


def test_passing_a_camera_skips_the_fit(box, known_camera, reference, monkeypatch):
    # Every part of a multi-part build shares one reference and therefore one
    # camera; refitting per part is both slower and less consistent.
    def explode(*args, **kwargs):
        raise AssertionError("fit_camera should not have been called")

    monkeypatch.setattr(texturing, "fit_camera", explode)

    out, stats = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )

    assert stats["camera"]["scale_px"] == pytest.approx(known_camera.scale, abs=0.01)
    assert out.visual.material.baseColorTexture is not None


def test_the_result_survives_a_glb_round_trip(box, known_camera, reference, tmp_path):
    out, _ = texturing.texture_from_reference(
        box, reference, mode="uv", camera=known_camera
    )
    path = tmp_path / "textured.glb"
    out.export(path)

    reloaded = trimesh.load(path, force="mesh", process=False)

    assert reloaded.visual.material.baseColorTexture is not None
    assert len(reloaded.visual.uv) == len(reloaded.vertices)
    assert len(reloaded.faces) == len(box.faces)
