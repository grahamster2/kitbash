"""The scripted primitive library: geometry, the schema, and validation.

These assert real geometric properties rather than face counts alone. The whole
claim of this module is that a scripted part is *better* geometry than a
generated one — watertight, dimensioned exactly as asked, and cheap — so those
are the things worth failing a build over.
"""
import numpy as np
import pytest
import trimesh

import config
import materials
import primitives


ALL_KINDS = primitives.kinds()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_kind_builds_a_closed_solid(kind):
    mesh = primitives.build(kind)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_kind_is_centred_on_the_origin(kind):
    # Generated parts arrive centred and /assemble's placement assumes it. A
    # library that sat its parts on Y=0 would offset every mixed scene.
    mesh = primitives.build(kind)

    assert mesh.bounding_box.centroid == pytest.approx([0, 0, 0], abs=1e-6)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_kind_is_far_under_the_engine_budget(kind):
    # An AI-generated equivalent lands at the 20,000-face decimation target.
    mesh = primitives.build(kind)

    assert 8 <= len(mesh.faces) <= 2500


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_kind_has_no_degenerate_triangles(kind):
    mesh = primitives.build(kind)

    assert mesh.area_faces.min() > 1e-9


def test_a_chamfered_box_is_a_closed_solid_of_exactly_sixty_triangles():
    mesh = primitives._box(2.0, 1.0, 3.0, chamfer=0.2)

    assert len(mesh.faces) == 60
    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([2.0, 1.0, 3.0])
    # The bevel removes material without moving the bounding box.
    assert mesh.volume < 6.0


def test_a_zero_chamfer_box_is_the_twelve_triangle_cube():
    mesh = primitives._box(2.0, 1.0, 3.0, chamfer=0.0)

    assert len(mesh.faces) == 12
    assert mesh.volume == pytest.approx(6.0)


def test_a_chamfer_wider_than_the_box_is_clamped_rather_than_inverted():
    mesh = primitives._box(0.2, 4.0, 4.0, chamfer=1.0)

    assert mesh.is_watertight
    assert mesh.volume > 0
    assert mesh.extents == pytest.approx([0.2, 4.0, 4.0])


# --- dimensions come out as asked -------------------------------------------

def test_crate_dimensions_are_exactly_what_was_requested():
    mesh = primitives.build("crate", {"width": 3.5, "height": 2.0, "depth": 1.25})

    assert mesh.extents == pytest.approx([3.5, 2.0, 1.25])


@pytest.mark.parametrize("style", ["planks", "frame", "plain"])
def test_every_crate_style_keeps_the_requested_box(style):
    mesh = primitives.build("crate", {"width": 3.0, "depth": 2.0, "style": style})

    assert mesh.extents == pytest.approx([3.0, 2.0, 2.0])
    assert mesh.is_watertight


def test_a_plain_crate_is_a_single_chamfered_box():
    plain = primitives.build("crate", {"style": "plain"})
    planks = primitives.build("crate", {"style": "planks"})

    assert len(plain.faces) < len(planks.faces)


def test_plank_dimensions_map_to_the_named_axes():
    mesh = primitives.build(
        "plank", {"length": 6.0, "width": 0.5, "thickness": 0.2}
    )

    assert mesh.extents == pytest.approx([6.0, 0.2, 0.5])


def test_cylinder_height_is_exact_and_radius_is_the_circumscribed_one():
    mesh = primitives.build(
        "cylinder", {"radius": 0.75, "height": 4.0, "chamfer": 0.0, "sections": 64}
    )

    assert mesh.extents[1] == pytest.approx(4.0)
    assert mesh.extents[0] == pytest.approx(1.5, rel=0.01)


def test_a_pipe_is_hollow_and_a_rod_is_not():
    rod = primitives.build("cylinder", {"radius": 1.0, "height": 2.0})
    pipe = primitives.build(
        "cylinder", {"radius": 1.0, "height": 2.0, "wall_thickness": 0.2}
    )

    assert pipe.is_watertight
    assert pipe.volume < rod.volume / 2
    assert pipe.extents == pytest.approx(rod.extents)


def test_wall_panel_openings_do_not_change_the_panel_envelope():
    solid = primitives.build("wall_panel", {"opening": "none", "trim": False})
    window = primitives.build("wall_panel", {"opening": "window", "trim": False})

    assert window.extents == pytest.approx(solid.extents)
    assert window.volume < solid.volume


def test_a_door_opening_removes_more_wall_than_a_window():
    window = primitives.build("wall_panel", {"opening": "window", "trim": False})
    door = primitives.build(
        "wall_panel",
        {"opening": "door", "opening_height": 4.0, "trim": False},
    )

    assert door.volume < window.volume


def test_door_trim_does_not_hang_below_the_wall():
    plain = primitives.build("wall_panel", {"opening": "door", "trim": False})
    trimmed = primitives.build("wall_panel", {"opening": "door", "trim": True})

    assert trimmed.bounds[0][1] == pytest.approx(plain.bounds[0][1])


def test_trim_stands_proud_of_the_wall_on_both_faces():
    plain = primitives.build("wall_panel", {"trim": False, "thickness": 0.5})
    trimmed = primitives.build(
        "wall_panel", {"trim": True, "thickness": 0.5, "trim_depth": 0.12}
    )

    assert trimmed.extents[2] == pytest.approx(plain.extents[2] + 0.24)


def test_an_opening_wider_than_the_wall_is_rejected():
    with pytest.raises(ValueError, match="no wall beside it"):
        primitives.build("wall_panel", {"width": 2.0, "opening_width": 2.0})


def test_an_opening_taller_than_the_wall_is_rejected():
    with pytest.raises(ValueError, match="no wall above it"):
        primitives.build(
            "wall_panel", {"height": 3.0, "sill_height": 1.0, "opening_height": 2.5}
        )


def test_stairs_span_the_rise_and_run_they_were_given():
    mesh = primitives.build(
        "stairs", {"steps": 8, "rise": 0.4, "run": 0.6, "width": 5.0}
    )

    assert mesh.extents == pytest.approx([5.0, 3.2, 4.8])


def test_open_stairs_use_less_material_than_solid_ones():
    blocks = primitives.build("stairs", {"style": "blocks"})
    open_flight = primitives.build("stairs", {"style": "open"})

    assert open_flight.volume < blocks.volume / 2


def test_table_height_is_measured_to_the_top_surface():
    mesh = primitives.build("table", {"height": 3.0, "width": 5.0, "depth": 2.0})

    assert mesh.extents == pytest.approx([5.0, 3.0, 2.0])


def test_a_backless_bench_is_only_as_tall_as_the_seat():
    mesh = primitives.build("bench", {"height": 1.5, "backrest": False})

    assert mesh.extents[1] == pytest.approx(1.5)


def test_ladder_rungs_cost_exactly_one_cylinder_each():
    rung = primitives.build("cylinder", {"sections": 24})
    few = primitives.build("ladder", {"rung_count": 3, "sections": 24})
    many = primitives.build("ladder", {"rung_count": 9, "sections": 24})

    assert len(many.faces) - len(few.faces) == 6 * len(rung.faces)


def test_ladder_rungs_are_evenly_spaced_between_the_rail_ends():
    mesh = primitives.build("ladder", {"height": 6.0, "rung_count": 5})

    assert mesh.extents[1] == pytest.approx(6.0)


def test_wedge_is_half_the_volume_of_its_bounding_box():
    mesh = primitives.build("wedge", {"width": 4.0, "height": 2.0, "depth": 3.0})

    assert mesh.extents == pytest.approx([4.0, 2.0, 3.0])
    assert mesh.volume == pytest.approx(4.0 * 2.0 * 3.0 / 2)


def test_a_flipped_wedge_mirrors_rather_than_changing_size():
    normal = primitives.build("wedge", {"flip": False})
    flipped = primitives.build("wedge", {"flip": True})

    assert flipped.extents == pytest.approx(normal.extents)
    assert flipped.volume == pytest.approx(normal.volume)


def test_a_spokeless_wheel_is_a_solid_disc():
    disc = primitives.build("wheel", {"spoke_count": 0})
    spoked = primitives.build("wheel", {"spoke_count": 8})

    assert disc.is_watertight and spoked.is_watertight
    assert disc.volume > spoked.volume


def test_a_wheel_lies_in_the_xz_plane_with_y_as_the_axle():
    mesh = primitives.build("wheel", {"radius": 1.5, "width": 0.5})

    assert mesh.extents[0] == pytest.approx(mesh.extents[2], rel=1e-6)
    assert mesh.extents[1] == pytest.approx(0.5)


def test_barrel_bellies_out_past_its_end_radius():
    mesh = primitives.build(
        "barrel", {"height": 2.0, "belly_radius": 1.0, "end_radius": 0.6,
                   "hoop_count": 0}
    )

    assert mesh.extents[1] == pytest.approx(2.0)
    assert mesh.extents[0] > 1.2


def test_barrel_hoops_add_geometry_outside_the_staves():
    bare = primitives.build("barrel", {"hoop_count": 0})
    banded = primitives.build("barrel", {"hoop_count": 3})

    assert len(banded.faces) > len(bare.faces)
    assert banded.extents[0] > bare.extents[0]


def test_a_fluted_column_is_narrower_than_a_plain_one_but_the_same_height():
    plain = primitives.build("column", {"style": "plain"})
    fluted = primitives.build("column", {"style": "fluted"})

    assert fluted.extents[1] == pytest.approx(plain.extents[1])
    assert fluted.volume < plain.volume


def test_a_tapered_column_loses_the_radius_it_was_told_to():
    mesh = primitives.build(
        "column",
        {"style": "tapered", "taper": 0.5, "radius": 1.0, "base_height": 0.0,
         "capital_height": 0.0, "base_overhang": 0.0, "sections": 64},
    )
    top = mesh.vertices[mesh.vertices[:, 1] > mesh.bounds[1][1] - 1e-6]
    top_radius = np.linalg.norm(top[:, [0, 2]], axis=1).max()

    assert top_radius == pytest.approx(0.5, rel=0.01)


def test_a_column_with_no_base_or_capital_is_a_bare_shaft():
    mesh = primitives.build(
        "column", {"style": "plain", "base_height": 0.0, "capital_height": 0.0,
                   "radius": 0.5, "height": 4.0, "sections": 64}
    )

    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([1.0, 4.0, 1.0], rel=0.01)


# --- schema and validation ---------------------------------------------------

def test_the_catalogue_covers_every_registered_kind():
    entries = primitives.catalogue()

    assert [e["kind"] for e in entries] == ALL_KINDS


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_catalogue_entry_is_self_describing(kind):
    entry = primitives.KINDS[kind].as_dict()

    assert entry["summary"]
    assert entry["material"] in materials.families()
    assert entry["params"], "a parametric primitive with no parameters is not one"
    for param in entry["params"]:
        assert param["type"] in ("number", "integer", "boolean", "choice")
        assert param["description"]
        if param["type"] == "choice":
            assert param["default"] in param["choices"]
        else:
            assert param.get("unit") in ("studs", "count", None)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_documented_default_survives_its_own_validation(kind):
    defaults = {p.name: p.default for p in primitives.KINDS[kind].params}

    assert primitives.resolve(kind, defaults) == defaults


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_choice_a_kind_advertises_actually_builds(kind):
    for param in primitives.KINDS[kind].params:
        if param.type != "choice":
            continue
        for choice in param.choices:
            mesh = primitives.build(kind, {param.name: choice})
            assert mesh.is_watertight, f"{kind}.{param.name}={choice}"


def test_an_unknown_kind_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown kind"):
        primitives.resolve("teapot", {})


def test_a_misspelled_parameter_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="widht"):
        primitives.resolve("crate", {"widht": 2.0})


def test_a_negative_dimension_is_rejected():
    with pytest.raises(ValueError, match=">= 0.01"):
        primitives.resolve("crate", {"width": -3.0})


def test_an_absurd_dimension_is_rejected():
    with pytest.raises(ValueError, match="<= 200"):
        primitives.resolve("crate", {"width": 5000.0})


def test_a_string_where_a_number_belongs_is_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        primitives.resolve("crate", {"width": "big"})


def test_a_boolean_is_not_accepted_as_a_number():
    # bool is an int in Python, so this sails through a naive isinstance check.
    with pytest.raises(ValueError, match="must be a number"):
        primitives.resolve("crate", {"width": True})


def test_a_number_is_not_accepted_as_a_boolean():
    with pytest.raises(ValueError, match="true or false"):
        primitives.resolve("wall_panel", {"trim": 1})


def test_a_fractional_count_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        primitives.resolve("crate", {"plank_count": 2.5})


def test_an_unlisted_choice_is_rejected_with_the_list():
    with pytest.raises(ValueError, match=r"planks.*frame.*plain"):
        primitives.resolve("crate", {"style": "wicker"})


def test_resolve_fills_in_every_documented_default():
    resolved = primitives.resolve("crate", {"width": 5.0})

    assert resolved["width"] == 5.0
    assert set(resolved) == {p.name for p in primitives.KINDS["crate"].params}


def test_a_parameter_set_that_would_blow_the_face_budget_is_refused(monkeypatch):
    monkeypatch.setattr(config, "PRIMITIVE_MAX_FACES", 200)

    with pytest.raises(ValueError, match="over the 200"):
        primitives.build("crate", {"plank_count": 8})


# --- materials and output ----------------------------------------------------

def test_a_crate_is_wood_without_being_told():
    mesh = primitives.build("crate")

    assert mesh.visual.material.name == "kitbash_wood"


def test_a_pipe_is_metal_and_a_wall_is_stone():
    assert primitives.build("cylinder").visual.material.name == "kitbash_metal"
    assert primitives.build("wall_panel").visual.material.name == "kitbash_stone"


def test_an_explicit_material_overrides_the_kind():
    mesh = primitives.build("crate", material="metal")

    assert mesh.visual.material.name == "kitbash_metal"


def test_a_colour_overrides_the_base_colour_but_keeps_the_family():
    mesh = primitives.build("crate", color="#ff0000")

    assert mesh.visual.material.name == "kitbash_wood"
    assert mesh.visual.material.baseColorFactor[0] > mesh.visual.material.baseColorFactor[1]


def test_an_unknown_material_is_rejected():
    with pytest.raises(ValueError, match="unknown material"):
        primitives.build("crate", material="unobtanium")


def test_uvs_are_off_by_default_so_the_mesh_stays_welded():
    mesh = primitives.build("crate")

    assert mesh.visual.uv is None
    assert mesh.is_watertight


def test_uv_scale_emits_one_uv_per_vertex_covering_the_mesh():
    mesh = primitives.build("plank", {"length": 4.0}, uv_scale=1.0)

    assert mesh.visual.uv is not None
    assert len(mesh.visual.uv) == len(mesh.vertices)
    assert np.ptp(mesh.visual.uv[:, 0]) == pytest.approx(4.0, rel=0.05)


def test_store_writes_a_glb_and_reports_it_like_a_generation(tmp_path):
    result = primitives.store("crate", {"width": 3.0}, tmp_path / "job")

    assert set(result) >= {
        "mesh_path", "generation_seconds", "peak_vram_gib", "vertices", "faces",
        "decimated_from", "watertight", "file_bytes", "params",
    }
    assert result["peak_vram_gib"] == 0.0
    assert result["decimated_from"] is None
    assert result["watertight"] is True
    assert result["size"] == [3.0, 2.0, 2.0]
    assert (tmp_path / "job" / "mesh.glb").exists()


def test_a_stored_primitive_reloads_as_the_same_geometry(tmp_path):
    result = primitives.store("barrel", None, tmp_path / "job")
    reloaded = trimesh.load(result["mesh_path"], force="mesh")

    assert len(reloaded.faces) == result["faces"]
    assert reloaded.extents == pytest.approx(result["size"], abs=1e-3)


def test_store_records_the_resolved_parameters_not_just_the_given_ones(tmp_path):
    result = primitives.store("crate", {"width": 3.0}, tmp_path / "job")

    assert result["params"]["kind"] == "crate"
    assert result["params"]["width"] == 3.0
    assert result["params"]["style"] == "planks"


def test_the_part_name_beats_the_kinds_default_material():
    """A bench called "front_left_seat" is a seat. The caller naming it that is
    a stronger signal than the kind's assumption that benches are wooden."""
    seat = primitives.build("bench", part_name="front_left_seat")

    assert seat.visual.material.name == "kitbash_fabric"


def test_the_kinds_default_applies_when_the_name_says_nothing():
    anonymous = primitives.build("bench", part_name="thing_47")

    assert anonymous.visual.material.name == f"kitbash_{primitives.KINDS['bench'].material}"


def test_an_explicit_material_beats_both():
    seat = primitives.build("bench", part_name="front_left_seat", material="gold")

    assert seat.visual.material.name == "kitbash_gold"
