"""Scene composition: transforms, node naming, and the describe() report."""
import numpy as np
import pytest
import trimesh

import assemble
import materials


def apply_to(T, point):
    return (T @ np.array([*point, 1.0]))[:3]


def test_transform_is_identity_when_nothing_is_placed():
    assert np.array_equal(assemble._transform(None, None, None), np.eye(4))


def test_transform_accepts_a_scalar_scale_as_uniform():
    T = assemble._transform(None, None, 3)

    assert apply_to(T, (1, 1, 1)) == pytest.approx([3, 3, 3])


def test_transform_accepts_a_per_axis_scale():
    T = assemble._transform(None, None, [1, 2, 3])

    assert apply_to(T, (1, 1, 1)) == pytest.approx([1, 2, 3])


def test_transform_rotation_is_in_degrees():
    T = assemble._transform(None, [0, 0, 90], None)

    assert apply_to(T, (1, 0, 0)) == pytest.approx([0, 1, 0], abs=1e-9)


def test_transform_scales_before_rotating():
    # A non-uniform scale is the only thing that tells the two orders apart:
    # rotate-then-scale would put this point at (0, 1, 0).
    T = assemble._transform(None, [0, 0, 90], [2, 1, 1])

    assert apply_to(T, (1, 0, 0)) == pytest.approx([0, 2, 0], abs=1e-9)


def test_transform_translates_after_scaling_and_rotating():
    T = assemble._transform([1, 2, 3], [0, 0, 90], [2, 1, 1])

    assert apply_to(T, (1, 0, 0)) == pytest.approx([1, 4, 3], abs=1e-9)


def test_transform_translation_is_not_affected_by_scale():
    T = assemble._transform([1, 0, 0], None, 10)

    assert apply_to(T, (0, 0, 0)) == pytest.approx([1, 0, 0])


def test_describe_reports_bounds_size_and_centre(make_mesh):
    path = make_mesh(trimesh.creation.box(extents=(1, 2, 3)))

    info = assemble.describe(path)

    assert info["faces"] == 12
    assert info["bounds_min"] == [-0.5, -1.0, -1.5]
    assert info["bounds_max"] == [0.5, 1.0, 1.5]
    assert info["size"] == [1.0, 2.0, 3.0]
    assert info["center"] == [0.0, 0.0, 0.0]


def test_describe_reports_an_off_origin_centre(make_mesh):
    box = trimesh.creation.box(extents=(2, 2, 2))
    box.apply_translation([5, 0, -1])
    path = make_mesh(box)

    info = assemble.describe(path)

    assert info["center"] == [5.0, 0.0, -1.0]
    assert info["bounds_min"] == [4.0, -1.0, -2.0]


def test_assemble_rejects_an_empty_part_list(tmp_path):
    with pytest.raises(ValueError, match="no parts"):
        assemble.assemble([], tmp_path / "scene.glb")


def test_assemble_reports_the_missing_mesh_by_path(tmp_path):
    missing = tmp_path / "gone.glb"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        assemble.assemble(
            [{"name": "hull", "mesh_path": str(missing)}], tmp_path / "scene.glb"
        )


def test_assemble_writes_one_node_per_part(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())
    out = tmp_path / "scenes" / "abc" / "scene.glb"

    result = assemble.assemble(
        [
            {"name": "hull", "mesh_path": str(path)},
            {"name": "wing", "mesh_path": str(path), "position": [10, 0, 0]},
        ],
        out,
    )

    assert result["part_count"] == 2
    assert result["total_faces"] == 24
    assert [p["name"] for p in result["parts"]] == ["hull", "wing"]
    assert out.exists() and result["file_bytes"] == out.stat().st_size
    assert set(trimesh.load(str(out)).geometry) == {"hull", "wing"}


def test_assemble_places_parts_at_their_positions(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())
    out = tmp_path / "scene.glb"

    result = assemble.assemble(
        [
            {"name": "hull", "mesh_path": str(path)},
            {"name": "wing", "mesh_path": str(path), "position": [10, 0, 0]},
        ],
        out,
    )

    assert result["bounds_min"] == [-0.5, -0.5, -0.5]
    assert result["bounds_max"] == [10.5, 0.5, 0.5]
    assert result["size"] == [11.0, 1.0, 1.0]


def test_assemble_scales_and_rotates_a_part(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box(extents=(1, 2, 3)))

    result = assemble.assemble(
        [{"name": "hull", "mesh_path": str(path), "rotation": [0, 0, 90], "scale": 2}],
        tmp_path / "scene.glb",
    )

    # 1x2x3 scaled by two, then turned a quarter turn about Z: X and Y swap.
    assert result["size"] == pytest.approx([4.0, 2.0, 6.0], abs=1e-4)


def test_assemble_deduplicates_repeated_node_names(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())
    out = tmp_path / "scene.glb"

    result = assemble.assemble(
        [{"name": "wing", "mesh_path": str(path)} for _ in range(3)], out
    )

    assert [p["name"] for p in result["parts"]] == ["wing", "wing_2", "wing_3"]
    assert set(trimesh.load(str(out)).geometry) == {"wing", "wing_2", "wing_3"}


def test_assemble_names_unnamed_parts_by_index(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [{"mesh_path": str(path)}, {"name": "", "mesh_path": str(path)}],
        tmp_path / "scene.glb",
    )

    assert [p["name"] for p in result["parts"]] == ["part_0", "part_1"]


def test_assemble_records_the_source_mesh_of_each_part(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [{"name": "hull", "mesh_path": str(path)}], tmp_path / "scene.glb"
    )

    assert result["parts"][0] == {
        "name": "hull",
        "faces": 12,
        "material": "paint",
        "source": str(path),
    }
    assert result["scene_path"] == str(tmp_path / "scene.glb")


def test_assemble_picks_a_material_from_each_part_name(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [
            {"name": "canopy", "mesh_path": str(path)},
            {"name": "front_wheel", "mesh_path": str(path)},
            {"name": "engine", "mesh_path": str(path)},
        ],
        tmp_path / "scene.glb",
    )

    assert [p["material"] for p in result["parts"]] == ["glass", "rubber", "metal"]


def test_assemble_honours_an_explicit_material_over_the_name(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [{"name": "canopy", "mesh_path": str(path), "material": "gold"}],
        tmp_path / "scene.glb",
    )

    assert result["parts"][0]["material"] == "gold"


def test_assemble_can_skip_materials(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [{"name": "canopy", "mesh_path": str(path)}],
        tmp_path / "scene.glb",
        apply_materials=False,
    )

    assert result["parts"][0]["material"] is None


def test_the_default_material_is_neutral(make_mesh, tmp_path):
    """paint is both the body-panel material and the fallback, so a saturated
    default would turn most scenes an arbitrary colour."""
    r, g, b, _ = materials.PALETTE[materials.DEFAULT_MATERIAL]["baseColorFactor"]

    assert max(r, g, b) - min(r, g, b) < 0.1


@pytest.mark.parametrize(
    "value, expected_rgb",
    [
        ("#ffffff", [1.0, 1.0, 1.0]),
        ("#000000", [0.0, 0.0, 0.0]),
        ("ff0000", [1.0, 0.0, 0.0]),  # bare hex, no leading '#'
    ],
)
def test_parse_color_converts_srgb_to_linear(value, expected_rgb):
    assert materials.parse_color(value)[:3] == pytest.approx(expected_rgb, abs=1e-6)


def test_parse_color_is_not_a_plain_divide_by_255():
    """glTF baseColorFactor is linear; treating sRGB as linear comes out bright."""
    assert materials.parse_color("#808080")[0] < 0.5


def test_parse_color_reads_alpha():
    assert materials.parse_color("#ffffff80")[3] == pytest.approx(128 / 255)


@pytest.mark.parametrize("bad", ["#fff", "#gggggg", "nope"])
def test_parse_color_rejects_malformed_input(bad):
    with pytest.raises(ValueError):
        materials.parse_color(bad)


def test_an_explicit_color_keeps_the_material_family(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    result = assemble.assemble(
        [{"name": "hull", "mesh_path": str(path), "color": "#c41e1a"}],
        tmp_path / "scene.glb",
    )

    # Still paint — colour overrides the base colour, not the metallic/roughness.
    assert result["parts"][0]["material"] == "paint"


def test_assemble_rejects_an_unknown_material(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    with pytest.raises(ValueError, match="unknown material"):
        assemble.assemble(
            [{"name": "hull", "mesh_path": str(path), "material": "unobtainium"}],
            tmp_path / "scene.glb",
        )
