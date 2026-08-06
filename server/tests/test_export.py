"""Target constraints: triangle budget, stud scaling, pivot, and warnings."""
import numpy as np
import pytest
import trimesh
from PIL import Image

import export

# 20480 faces — just over Roblox's per-mesh cap, and cheap to decimate.
DENSE_SUBDIVISIONS = 5


def dense_sphere():
    return trimesh.creation.icosphere(subdivisions=DENSE_SUBDIVISIONS)


def textured_box(width=5000, height=64):
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(box.vertices), 2)),
        image=Image.new("RGB", (width, height), (10, 20, 30)),
    )
    return box


def test_export_rejects_an_unknown_target(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())

    with pytest.raises(ValueError, match="unknown target"):
        export.export_for(path, "unity", tmp_path / "exported")


def test_export_rejects_an_unknown_target_before_writing_anything(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box())
    out = tmp_path / "exported"

    with pytest.raises(ValueError):
        export.export_for(path, "unity", out)

    assert not out.exists()


def test_export_rejects_a_missing_source_mesh(tmp_path):
    with pytest.raises(FileNotFoundError):
        export.export_for(tmp_path / "gone.glb", "roblox", tmp_path / "exported")


def test_export_writes_a_glb_and_an_obj(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box(), name="part.glb")

    result = export.export_for(path, "dcc", tmp_path / "exported")

    assert result["primary"] == str(tmp_path / "exported" / "part.glb")
    assert result["files"]["obj"] == str(tmp_path / "exported" / "part.obj")
    assert (tmp_path / "exported" / "part.glb").is_file()
    assert (tmp_path / "exported" / "part.obj").is_file()
    assert result["file_bytes"]["glb"] > 0
    assert result["file_bytes"]["obj"] > 0


def test_roblox_export_decimates_a_mesh_over_the_per_mesh_budget(make_mesh, tmp_path):
    path = make_mesh(dense_sphere())

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["source_faces"] == 20480
    assert result["total_faces"] == export.ROBLOX_MAX_TRIANGLES
    assert (
        "geometry_0: 20480 faces exceeded Roblox's 20000 per-mesh limit, "
        "decimated to fit"
    ) in result["warnings"]


def test_roblox_triangle_budget_applies_per_geometry_not_per_scene(make_mesh, tmp_path):
    path = make_mesh(parts=[("hull", dense_sphere()), ("wing", dense_sphere())])

    result = export.export_for(path, "roblox", tmp_path / "exported")

    # Each MeshPart gets its own 20k allowance, so a two-part scene keeps 40k.
    assert [p["faces"] for p in result["parts"]] == [20_000, 20_000]
    assert result["total_faces"] == 40_000
    assert len([w for w in result["warnings"] if "exceeded" in w]) == 2


def test_roblox_export_leaves_a_mesh_under_the_budget_untouched(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.icosphere(subdivisions=3))

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["total_faces"] == result["source_faces"] == 1280
    assert not any("exceeded" in w for w in result["warnings"])


def test_dcc_export_does_not_apply_the_roblox_budget(make_mesh, tmp_path):
    path = make_mesh(dense_sphere())

    result = export.export_for(path, "dcc", tmp_path / "exported")

    assert result["total_faces"] == result["source_faces"] == 20480
    assert result["warnings"] == []


def test_height_studs_sets_an_exact_y_extent(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box(extents=(1, 2, 3)))

    result = export.export_for(path, "roblox", tmp_path / "exported", height_studs=8)

    assert result["size"][1] == 8.0


def test_height_studs_scales_the_other_axes_proportionally(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box(extents=(1, 2, 3)))

    result = export.export_for(path, "dcc", tmp_path / "exported", height_studs=8)

    assert result["size"] == pytest.approx([4.0, 8.0, 12.0], abs=1e-4)


def test_export_without_height_studs_keeps_the_generated_size(make_mesh, tmp_path):
    path = make_mesh(trimesh.creation.box(extents=(1, 2, 3)))

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["size"] == pytest.approx([1.0, 2.0, 3.0], abs=1e-4)


def test_roblox_export_sits_the_mesh_on_the_ground_centred_on_its_footprint(
    make_mesh, tmp_path
):
    box = trimesh.creation.box(extents=(2, 2, 2))
    box.apply_translation([10, 5, -3])
    path = make_mesh(box)

    result = export.export_for(path, "roblox", tmp_path / "exported")

    lo, hi = trimesh.load(result["primary"]).bounds
    assert lo == pytest.approx([-1, 0, -1], abs=1e-5)
    assert hi == pytest.approx([1, 2, 1], abs=1e-5)
    assert result["pivot"] == "base-centered"


def test_dcc_export_leaves_the_geometry_where_the_generator_put_it(
    make_mesh, tmp_path
):
    box = trimesh.creation.box(extents=(2, 2, 2))
    box.apply_translation([10, 5, -3])
    path = make_mesh(box)

    result = export.export_for(path, "dcc", tmp_path / "exported")

    lo, hi = trimesh.load(result["primary"]).bounds
    assert lo == pytest.approx([9, 4, -4], abs=1e-5)
    assert hi == pytest.approx([11, 6, -2], abs=1e-5)
    assert result["pivot"] == "source"


def test_roblox_export_grounds_the_mesh_after_rescaling_it(make_mesh, tmp_path):
    box = trimesh.creation.box(extents=(2, 2, 2))
    box.apply_translation([10, 5, -3])
    path = make_mesh(box)

    result = export.export_for(path, "roblox", tmp_path / "exported", height_studs=10)

    lo, hi = trimesh.load(result["primary"]).bounds
    assert lo[1] == pytest.approx(0, abs=1e-5)
    assert hi[1] == pytest.approx(10, abs=1e-5)


def test_roblox_export_warns_about_a_texture_studio_would_downsample(
    make_mesh, tmp_path
):
    path = make_mesh(parts=[("skin", textured_box())])

    result = export.export_for(path, "roblox", tmp_path / "exported")

    expected = (
        "skin: texture is 5000x64, Studio downsamples above "
        f"{export.ROBLOX_MAX_TEXTURE_PX}px"
    )
    assert result["warnings"] == [expected]


def test_roblox_export_does_not_warn_about_a_texture_within_limits(
    make_mesh, tmp_path
):
    path = make_mesh(parts=[("skin", textured_box(256, 256))])

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["warnings"] == []


def test_roblox_export_warns_that_vertex_colour_is_lost_by_the_obj(
    make_mesh, tmp_path
):
    box = trimesh.creation.box()
    box.visual.vertex_colors = np.tile([255, 0, 0, 255], (len(box.vertices), 1))
    path = make_mesh(parts=[("hull", box)])

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["warnings"] == ["hull: colour is per-vertex only, not carried by .obj"]


def test_roblox_export_warns_when_a_mesh_has_no_colour_at_all(make_mesh, tmp_path):
    path = make_mesh(parts=[("hull", trimesh.creation.box())])

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["warnings"] == ["hull: no texture or vertex colour, imports untextured"]


def test_dcc_export_skips_the_texture_checks(make_mesh, tmp_path):
    path = make_mesh(parts=[("skin", textured_box())])

    result = export.export_for(path, "dcc", tmp_path / "exported")

    assert result["warnings"] == []


def test_export_reports_the_obj_sidecars_a_textured_mesh_needs(make_mesh, tmp_path):
    path = make_mesh(parts=[("skin", textured_box(256, 256))], name="part.glb")

    result = export.export_for(path, "roblox", tmp_path / "exported")

    sidecars = [s.rsplit("/", 1)[-1] for s in result["files"]["obj_sidecars"]]
    assert "part.mtl" in sidecars
    assert any(s.endswith(".png") for s in sidecars)


def test_export_names_every_part_it_wrote(make_mesh, tmp_path):
    path = make_mesh(
        parts=[("hull", trimesh.creation.box()), ("wing", trimesh.creation.box())]
    )

    result = export.export_for(path, "roblox", tmp_path / "exported")

    assert result["part_count"] == 2
    assert {p["name"] for p in result["parts"]} == {"hull", "wing"}
    assert result["target"] == "roblox"
