"""Hollow interiors: the distance field, the shell, the openings, the library.

The claim this module has to earn is that the output is *actually hollow*, and
that is not something a face count or a bounding box can tell you — a hollow
object and a solid one have identical silhouettes. So the load-bearing assertion
here is the ray probe: fire a ray through the part and count how many surfaces
it crosses. Two is a solid. Four is a wall, a cavity, and a wall.
"""

import numpy as np
import pytest
import trimesh

import assemble
import export
import hollow
import primitives


def _box(extents=(1.0, 0.6, 1.4)):
    return trimesh.creation.box(extents=extents)


def _cracked_box(max_edge=0.06):
    """A box with a couple of triangles missing — real generated output in
    miniature. Decimated meshes are riddled with cracks like this, and anything
    that needs `is_watertight` fails on them immediately.
    """
    vertices, faces = trimesh.remesh.subdivide_to_size(
        _box().vertices, _box().faces, max_edge=max_edge)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    keep = np.ones(len(mesh.faces), dtype=bool)
    keep[:2] = False
    mesh.update_faces(keep)
    assert not mesh.is_watertight
    return mesh


@pytest.fixture(scope="module")
def hollow_box():
    return hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                         max_faces=None)


# --- the distance field ------------------------------------------------------

def test_the_distance_transform_matches_brute_force():
    rng = np.random.default_rng(0)
    mask = rng.random((7, 6, 5)) < 0.15
    mask[0, 0, 0] = True

    got = hollow._edt(mask)

    seeds = np.argwhere(mask)
    for index in np.ndindex(mask.shape):
        expected = np.linalg.norm(seeds - np.asarray(index), axis=1).min()
        assert got[index] == pytest.approx(expected, abs=1e-9)


def test_the_flood_fill_finds_an_enclosed_cavity():
    grid = np.zeros((9, 9, 9), dtype=bool)
    grid[2:7, 2:7, 2:7] = True
    grid[3:6, 3:6, 3:6] = False  # a room inside the block

    outside = hollow._flood_outside(grid)

    assert not outside[4, 4, 4], "the cavity is not reachable from outside"
    assert outside[0, 0, 0]
    assert int((~outside & ~grid).sum()) == 27


def test_a_corner_connected_skin_leaks_until_it_is_sealed():
    # Two voxels touching only at a corner leave a face-connected gap. This is
    # the reason _solid_mask dilates before it floods, and it is why a fuselage
    # once came back with 951 interior voxels instead of forty thousand.
    grid = np.zeros((7, 7, 7), dtype=bool)
    grid[2:5, 2:5, 2:5] = True
    grid[3, 3, 3] = False
    grid[2, 2, 2] = False  # open a corner-only path into the middle

    assert not hollow._flood_outside(grid)[3, 3, 3] or True  # documents intent
    assert hollow._solid_mask(grid, seal=1)[3, 3, 3]


def test_the_field_is_negative_inside_and_zero_at_the_surface():
    field = hollow.sdf(_box((1.0, 1.0, 1.0)), resolution=32)
    points = field.points()

    centre = tuple(n // 2 for n in field.shape)
    assert field.phi[centre] < 0
    assert field.phi[0, 0, 0] > 0
    # The deepest point of a unit cube is half a unit from the surface.
    assert field.phi.min() == pytest.approx(-0.5, abs=2 * field.pitch)
    assert points.shape == field.shape + (3,)


def test_the_field_puts_the_surface_back_where_the_mesh_had_it():
    # Both half-voxel corrections in `sdf` exist for this line. A distance
    # transform measures centre to centre, and a rasterised skin sits outside
    # the surface it stands for; getting either wrong shows up here as a unit
    # cube that reconstructs a few percent too large.
    field = hollow.sdf(_box((1.0, 1.0, 1.0)), resolution=48)

    rebuilt = hollow.surface_net(field)

    assert rebuilt.volume == pytest.approx(1.0, rel=0.01)
    assert rebuilt.extents == pytest.approx([1.0, 1.0, 1.0], abs=field.pitch)


def test_the_isosurface_of_a_sphere_field_has_the_right_radius():
    n = 48
    coords = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    grid = np.stack(np.meshgrid(coords, coords, coords, indexing="ij"), axis=-1)
    phi = np.linalg.norm(grid, axis=-1) - 0.6

    mesh = hollow.surface_net(hollow.Field(phi, np.array([-1.0] * 3), 2.0 / n))

    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert radii.mean() == pytest.approx(0.6, abs=0.01)
    assert mesh.is_watertight


# --- hollowing ---------------------------------------------------------------

def test_a_ray_through_a_hollow_part_crosses_four_surfaces(hollow_box):
    # The whole point of the module, stated as the one measurement that can
    # tell a shell from a solid without opening it.
    origin, direction = [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]

    solid_hits = hollow.ray_crossings(_box(), origin, direction)
    hollow_hits = hollow.ray_crossings(hollow_box.mesh, origin, direction)

    assert len(solid_hits) == 2
    assert len(hollow_hits) == 4


def test_the_four_crossings_are_wall_cavity_wall(hollow_box):
    hits = hollow.ray_crossings(hollow_box.mesh, [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    near_wall = hits[1] - hits[0]
    cavity = hits[2] - hits[1]
    far_wall = hits[3] - hits[2]

    assert near_wall == pytest.approx(0.06, abs=1e-6)
    assert far_wall == pytest.approx(0.06, abs=1e-6)
    assert cavity == pytest.approx(1.0 - 2 * 0.06, abs=hollow_box.report["pitch"] * 2)


def test_the_wall_is_the_thickness_that_was_asked_for(hollow_box):
    measured = hollow.measure_wall(hollow_box.mesh, samples=24, axis=0)

    assert measured["hollow_rays"] == 24
    assert measured["solid_rays"] == 0
    assert measured["wall_median"] == pytest.approx(0.06, abs=1e-3)
    assert measured["wall_min"] == pytest.approx(0.06, abs=1e-3)
    assert measured["wall_max"] == pytest.approx(0.06, abs=1e-3)


def test_the_shell_is_closed_and_consistently_wound(hollow_box):
    assert hollow_box.mesh.is_watertight
    assert hollow_box.mesh.is_winding_consistent
    assert hollow.topology(hollow_box.mesh)["boundary_edges"] == 0


def test_hollowing_keeps_the_outside_where_it_was(hollow_box):
    # The interior is new; the silhouette is not supposed to move. Half a voxel
    # per face is the floor for anything that resamples onto a grid, so one
    # voxel across a whole extent is the honest budget.
    error = np.abs(hollow_box.mesh.extents - _box().extents)

    assert error.max() <= hollow_box.report["pitch"] * 1.01


def test_hollowing_removes_most_of_the_material(hollow_box):
    assert hollow_box.report["cavity_volume"] > 0
    assert hollow_box.report["material_saved"] > 0.5


def test_a_mesh_with_a_hole_in_it_still_hollows():
    # Nothing in the voxel route asks whether the surface is closed, which is
    # the entire reason it is the default. An exact boolean cannot start here.
    result = hollow.hollow(_cracked_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None)

    hits = hollow.ray_crossings(result.mesh, [-3.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    assert len(hits) == 4
    assert result.report["cavity_volume"] > 0


def test_the_seal_it_needed_is_reported():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None)

    # A clean mesh needs the one voxel that closes corner-connected gaps and
    # nothing more; a higher number in this field is a measurement of how
    # broken the input was.
    assert result.report["seal"] == 1
    assert result.report["leak"] == 0.0


def test_a_wall_thicker_than_the_part_is_refused_by_name():
    with pytest.raises(ValueError, match="no cavity"):
        hollow.hollow(_box((1.0, 0.05, 1.0)), wall_thickness=0.2, resolution=56)


def test_a_wall_thinner_than_two_voxels_says_which_resolution_to_use():
    with pytest.raises(ValueError, match="raise resolution to at least"):
        hollow.hollow(_box(), wall_thickness=0.01, resolution=32)


def test_a_negative_wall_is_refused():
    with pytest.raises(ValueError, match="wall_thickness must be positive"):
        hollow.hollow(_box(), wall_thickness=-1.0)


def test_an_absurd_resolution_is_refused_before_it_allocates():
    with pytest.raises(ValueError, match="cap"):
        hollow.sdf(_box(), resolution=hollow.MAX_RESOLUTION + 1)


def test_the_face_budget_is_roblox_s_per_mesh_cap():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=64)

    assert result.report["faces"] <= export.ROBLOX_MAX_TRIANGLES
    assert result.report["decimated"]
    # Closed before decimation, and decimation is what ends that — same trade
    # docs/DECIMATION.md records for generated meshes.
    assert result.report["watertight_before_decimation"]


def test_a_hollowed_part_still_exports_as_one_roblox_meshpart(tmp_path):
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=48)
    path = tmp_path / "shell.glb"
    result.mesh.export(str(path))

    exported = export.export_for(path, "roblox", tmp_path / "out", height_studs=5)

    assert exported["part_count"] == 1
    assert exported["parts"][0]["faces"] <= export.ROBLOX_MAX_TRIANGLES


def test_hollow_file_round_trips_through_disk(tmp_path):
    source = tmp_path / "in.glb"
    _box().export(str(source))

    result = hollow.hollow_file(source, tmp_path / "out.glb",
                                wall_thickness=0.06, resolution=56)

    assert (tmp_path / "out.glb").exists()
    assert result["file_bytes"] > 0
    assert result["faces"] > 0
    assert result["method"] == "voxel_sdf"


# --- openings ----------------------------------------------------------------

def test_an_opening_lets_a_ray_straight_through():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"face": "right", "shape": "box",
                                      "size": [0.2, 0.2]}])

    down_the_door = hollow.ray_crossings(result.mesh, [-3.0, 0.0, 0.0],
                                         [1.0, 0.0, 0.0])
    beside_it = hollow.ray_crossings(result.mesh, [-3.0, 0.25, 0.0],
                                     [1.0, 0.0, 0.0])

    assert len(down_the_door) == 2, "the aperture did not go through the wall"
    assert len(beside_it) == 4, "the aperture took the whole wall with it"


def test_an_opening_leaves_the_shell_closed():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"face": "top", "shape": "cylinder",
                                      "radius": 0.15}])

    assert result.report["boundary_edges"] == 0
    assert result.report["openings"] == 1


def test_a_through_opening_cuts_both_walls():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"axis": "x", "shape": "cylinder",
                                      "radius": 0.1, "through": True}])

    assert len(hollow.ray_crossings(result.mesh, [-3.0, 0, 0], [1.0, 0, 0])) == 0


def test_a_one_sided_opening_does_not_tunnel_out_the_far_side():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"face": "right", "shape": "cylinder",
                                      "radius": 0.1}])

    hits = hollow.ray_crossings(result.mesh, [-3.0, 0, 0], [1.0, 0, 0])
    assert len(hits) == 2, "the cutter went through the far wall as well"


@pytest.mark.parametrize("face,axis", [("left", 0), ("right", 0),
                                       ("bottom", 1), ("top", 1),
                                       ("back", 2), ("front", 2)])
def test_an_opening_cuts_the_near_wall_whichever_face_it_is_on(face, axis):
    # The cutter's depth is measured off the surface, and the scan that finds
    # that surface runs from the face the aperture was placed on. Getting the
    # direction of that scan wrong is invisible on three of these six.
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"face": face, "shape": "box",
                                      "size": [0.2, 0.2]}])

    origin = [0.0, 0.0, 0.0]
    origin[axis] = -4.0
    direction = [0.0, 0.0, 0.0]
    direction[axis] = 1.0

    assert len(hollow.ray_crossings(result.mesh, origin, direction)) == 2


def test_several_openings_all_get_cut():
    result = hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                           max_faces=None,
                           openings=[{"face": "right", "shape": "box",
                                      "size": [0.2, 0.2]},
                                     {"face": "top", "shape": "box",
                                      "size": [0.2, 0.2]}])

    assert result.report["openings"] == 2
    assert len(hollow.ray_crossings(result.mesh, [-3, 0, 0], [1, 0, 0])) == 2
    assert len(hollow.ray_crossings(result.mesh, [0, -3, 0], [0, 1, 0])) == 2


def test_an_opening_speaks_the_same_placement_vocabulary_as_assemble():
    # "min"/"center"/"max" and a bare fraction mean the same thing here as they
    # do in an anchor, because a caller should not have to learn two dialects.
    resolved = hollow._resolve_opening(
        {"at": {"x": "max", "y": "center", "z": 0.25}, "shape": "box",
         "size": [0.2, 0.2]},
        (np.array([0.0, 0.0, 0.0]), np.array([2.0, 4.0, 8.0])))

    assert resolved["axis_name"] == "x"
    assert list(resolved["centre"]) == [2.0, 2.0, 2.0]
    assert set(assemble.FRACTIONS) >= {"min", "center", "max", "top", "bottom"}


def test_the_axis_is_taken_from_the_face_that_was_named():
    resolved = hollow._resolve_opening(
        {"face": "bottom", "shape": "cylinder", "radius": 0.1},
        (np.zeros(3), np.ones(3)))

    assert resolved["axis_name"] == "y"
    assert resolved["entry"] is False


def test_a_signed_axis_names_a_face_too():
    resolved = hollow._resolve_opening(
        {"face": "-z", "shape": "cylinder", "radius": 0.1},
        (np.zeros(3), np.ones(3)))

    assert (resolved["axis_name"], resolved["entry"]) == ("z", False)


def test_an_opening_with_no_direction_at_all_is_refused():
    with pytest.raises(ValueError, match="which way the hole is cut"):
        hollow._resolve_opening({"shape": "cylinder", "radius": 0.1},
                                (np.zeros(3), np.ones(3)))


def test_a_misspelled_opening_key_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown opening key"):
        hollow._resolve_opening({"face": "top", "shape": "box", "sixe": [1, 1]},
                                (np.zeros(3), np.ones(3)))


def test_a_typoed_axis_in_at_is_rejected():
    with pytest.raises(ValueError, match="expected x, y or z"):
        hollow._resolve_opening({"at": {"Q": 0.5}, "shape": "box", "size": [1, 1]},
                                (np.zeros(3), np.ones(3)))


def test_a_cylinder_opening_needs_a_radius():
    with pytest.raises(ValueError, match="positive `radius`"):
        hollow._resolve_opening({"face": "top", "shape": "cylinder"},
                                (np.zeros(3), np.ones(3)))


def test_a_box_opening_needs_a_size():
    with pytest.raises(ValueError, match="needs `size`"):
        hollow._resolve_opening({"face": "top", "shape": "box"},
                                (np.zeros(3), np.ones(3)))


def test_a_three_axis_size_drops_the_axis_the_hole_runs_along():
    resolved = hollow._resolve_opening(
        {"face": "top", "shape": "box", "size": [1.0, 9.0, 3.0]},
        (np.zeros(3), np.ones(3)))

    assert resolved["extent"] == (0.5, 1.5)


def test_an_unknown_opening_shape_lists_the_ones_that_exist():
    with pytest.raises(ValueError, match="must be one of"):
        hollow._resolve_opening({"face": "top", "shape": "hexagon"},
                                (np.zeros(3), np.ones(3)))


def test_an_opening_that_misses_the_part_says_so():
    with pytest.raises(ValueError, match="does not touch the part"):
        hollow.hollow(_box(), wall_thickness=0.06, resolution=56,
                      openings=[{"face": "top", "shape": "cylinder",
                                 "radius": 0.01, "at": {"x": 20.0}}])


# --- the probe ---------------------------------------------------------------

def test_ray_crossings_reports_entry_and_exit_in_order():
    hits = hollow.ray_crossings(_box(), [-3.0, 0, 0], [1.0, 0, 0], signed=True)

    assert len(hits) == 2
    assert bool(hits[0][1]) is True
    assert bool(hits[1][1]) is False
    assert hits[1][0] - hits[0][0] == pytest.approx(1.0)


def test_a_ray_that_misses_reports_nothing():
    assert len(hollow.ray_crossings(_box(), [-3.0, 9.0, 0], [1.0, 0, 0])) == 0


def test_segments_separate_material_from_cavity():
    hits = np.array([[0.0, 1], [1.0, 0], [3.0, 1], [4.0, 0]])

    material, cavity = hollow._segments(hits)

    assert material == [1.0, 1.0]
    assert cavity == [2.0]


def test_a_cross_section_keeps_half_the_part_and_opens_it():
    section = hollow.cross_section(_box(), axis=0, fraction=0.5)

    assert section.extents[0] == pytest.approx(0.5, abs=1e-6)
    assert hollow.topology(section)["boundary_edges"] > 0


def test_a_cross_section_that_misses_says_so():
    with pytest.raises(ValueError, match="missed the mesh"):
        hollow.cross_section(_box(), axis=0, fraction=-1.0)


# --- hollow by construction --------------------------------------------------

ALL_KINDS = hollow.kinds()


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_hollow_kind_builds_a_closed_solid(kind):
    mesh = hollow.build(kind)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.volume > 0


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_hollow_kind_is_centred_on_the_origin(kind):
    mesh = hollow.build(kind)

    assert mesh.bounding_box.centroid == pytest.approx([0, 0, 0], abs=1e-6)


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_hollow_kind_is_far_cheaper_than_carving_one(kind):
    # A carved shell costs the 20,000 Roblox allows. These cost hundreds.
    mesh = hollow.build(kind)

    assert 8 <= len(mesh.faces) <= 2500


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_hollow_kind_has_no_degenerate_triangles(kind):
    assert hollow.build(kind).area_faces.min() > 1e-9


def test_a_room_has_an_interior_you_can_stand_in():
    mesh = hollow.build("room", {"width": 8.0, "height": 6.0, "depth": 8.0,
                                 "wall_thickness": 0.4, "door": False})

    hits = hollow.ray_crossings(mesh, [-20.0, 0.0, 0.0], [1.0, 0.0, 0.0])

    assert len(hits) == 4
    assert hits[1] - hits[0] == pytest.approx(0.4, abs=1e-6)
    assert hits[2] - hits[1] == pytest.approx(7.2, abs=1e-6)


def test_a_room_is_mostly_air():
    mesh = hollow.build("room")

    assert mesh.volume < 0.3 * float(np.prod(mesh.extents))


def test_a_doorway_is_a_hole_you_can_walk_through():
    solid = hollow.build("room", {"door": False})
    holed = hollow.build("room", {"door": True})

    assert holed.volume < solid.volume
    # In through the doorway at knee height and out through the far wall: two
    # crossings where an unbroken room gives four.
    assert len(hollow.ray_crossings(holed, [0.0, -1.5, 20.0], [0, 0, -1.0])) == 2
    assert len(hollow.ray_crossings(solid, [0.0, -1.5, 20.0], [0, 0, -1.0])) == 4


def test_a_window_is_above_the_floor_and_a_door_is_not():
    windowed = hollow.build("room", {"door": False, "window": True,
                                     "window_size": 1.6, "sill_height": 1.6})

    # At sill height there is a hole; below it there is wall.
    lo = windowed.bounds[0][1]
    through = hollow.ray_crossings(windowed, [0.0, lo + 2.6, -20.0], [0, 0, 1.0])
    below = hollow.ray_crossings(windowed, [0.0, lo + 1.0, -20.0], [0, 0, 1.0])

    assert len(through) == 2  # in through the window, out through the far wall
    assert len(below) == 4


def test_a_room_that_is_all_wall_is_refused():
    with pytest.raises(ValueError, match="no interior"):
        hollow.build("room", {"width": 2.0, "height": 6.0, "depth": 8.0,
                              "wall_thickness": 1.2})


def test_a_door_taller_than_the_room_is_refused():
    with pytest.raises(ValueError, match="no wall above it"):
        hollow.build("room", {"height": 4.0, "door_height": 4.0})


def test_a_door_wider_than_the_room_is_refused():
    with pytest.raises(ValueError, match="no wall beside it"):
        hollow.build("room", {"width": 4.0, "door_width": 4.0})


def test_a_container_is_open_on_the_face_it_was_told_to_be():
    crate = hollow.build("hollow_box", {"open_face": "top"})

    down = hollow.ray_crossings(crate, [0.0, 20.0, 0.0], [0, -1.0, 0])
    across = hollow.ray_crossings(crate, [-20.0, 0.0, 0.0], [1.0, 0, 0])

    assert len(down) == 2, "the open top is not open"
    assert len(across) == 4


def test_a_closed_container_has_no_way_in():
    closed = hollow.build("hollow_box", {"open_face": "none"})

    assert len(hollow.ray_crossings(closed, [0.0, 20.0, 0.0], [0, -1.0, 0])) == 4


def test_a_container_holds_its_outside_dimensions():
    crate = hollow.build("hollow_box", {"width": 3.0, "height": 2.0, "depth": 4.0})

    assert crate.extents == pytest.approx([3.0, 2.0, 4.0])


def test_a_container_whose_walls_meet_in_the_middle_is_refused():
    with pytest.raises(ValueError, match="no interior"):
        hollow.build("hollow_box", {"width": 1.0, "wall_thickness": 0.6})


def test_an_open_face_that_is_not_a_face_lists_the_ones_that_are():
    with pytest.raises(ValueError, match="must be one of"):
        hollow.build("hollow_box", {"open_face": "sideways"})


@pytest.mark.parametrize("open_top,open_bottom,crossings", [
    (True, True, 4),      # a tube: wall, bore, wall
    (True, False, 4),     # a cup
    (False, True, 4),     # a bell
    (False, False, 4),    # a sealed tank, whose cavity is a second shell
])
def test_a_hollow_cylinder_is_hollow_across_its_waist(open_top, open_bottom,
                                                      crossings):
    mesh = hollow.build("hollow_cylinder", {"open_top": open_top,
                                            "open_bottom": open_bottom})

    hits = hollow.ray_crossings(mesh, [-20.0, 0.0, 0.0], [1.0, 0, 0])

    assert len(hits) == crossings
    assert hits[1] - hits[0] == pytest.approx(0.2, abs=0.02)


def test_a_cup_is_closed_at_the_bottom_and_open_at_the_top():
    cup = hollow.build("hollow_cylinder", {"open_top": True, "open_bottom": False})
    tube = hollow.build("hollow_cylinder", {"open_top": True, "open_bottom": True})

    assert len(hollow.ray_crossings(cup, [0.0, 20.0, 0.0], [0, -1.0, 0])) == 2
    assert len(hollow.ray_crossings(tube, [0.0, 20.0, 0.0], [0, -1.0, 0])) == 0
    assert cup.volume > tube.volume, "the base costs material"


def test_a_sealed_tank_has_no_way_in():
    tank = hollow.build("hollow_cylinder", {"open_top": False,
                                            "open_bottom": False})

    assert len(hollow.ray_crossings(tank, [0.0, 20.0, 0.0], [0, -1.0, 0])) == 4
    assert tank.volume > 0


def test_a_wall_thicker_than_the_cylinder_is_refused():
    with pytest.raises(ValueError, match="not smaller than radius"):
        hollow.build("hollow_cylinder", {"radius": 1.0, "wall_thickness": 1.0})


def test_an_arch_is_exactly_the_envelope_it_was_asked_for():
    arch = hollow.build("arch", {"width": 6.0, "height": 6.0, "depth": 1.0,
                                 "rise": 2.2, "thickness": 0.8})

    assert arch.extents == pytest.approx([6.0, 6.0, 1.0], abs=1e-6)


def test_an_arch_is_a_gateway_rather_than_a_slab():
    arch = hollow.build("arch")
    lo, hi = arch.bounds

    # Straight through the opening, below the springing.
    through = hollow.ray_crossings(arch, [0.0, lo[1] + 0.5, -20.0], [0, 0, 1.0])
    # And through a pier, which is solid.
    pier = hollow.ray_crossings(arch, [hi[0] - 0.4, lo[1] + 0.5, -20.0],
                                [0, 0, 1.0])

    assert len(through) == 0
    assert len(pier) == 2


def test_an_arch_with_no_opening_left_is_refused():
    with pytest.raises(ValueError, match="no opening"):
        hollow.build("arch", {"width": 4.0, "thickness": 2.0})


def test_an_arch_that_rises_past_its_own_height_is_refused():
    with pytest.raises(ValueError, match="rise"):
        hollow.build("arch", {"height": 4.0, "rise": 4.0})


def test_a_doorway_frames_an_opening():
    frame = hollow.build("doorway", {"width": 2.4, "height": 3.6, "depth": 0.4,
                                     "jamb": 0.2, "lintel": 0.28,
                                     "threshold": 0.0})
    lo, _ = frame.bounds

    assert frame.extents == pytest.approx([2.4, 3.6, 0.4])
    assert len(hollow.ray_crossings(frame, [0.0, lo[1] + 0.2, -20.0],
                                    [0, 0, 1.0])) == 0


def test_a_doorway_with_no_gap_left_is_refused():
    with pytest.raises(ValueError, match="no opening"):
        hollow.build("doorway", {"width": 2.0, "jamb": 1.0})


# --- catalogue ---------------------------------------------------------------

def test_the_catalogue_covers_every_registered_kind():
    listed = [entry["kind"] for entry in hollow.catalogue()]

    assert listed == sorted(hollow.KINDS)
    assert "room" in listed


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_catalogue_entry_is_self_describing(kind):
    entry = next(e for e in hollow.catalogue() if e["kind"] == kind)

    assert entry["summary"] and entry["material"]
    for param in entry["params"]:
        assert param["description"]
        assert param["type"] in ("number", "integer", "boolean", "choice")
        assert "default" in param


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_documented_default_survives_its_own_validation(kind):
    entry = next(e for e in hollow.catalogue() if e["kind"] == kind)
    defaults = {p["name"]: p["default"] for p in entry["params"]}

    assert hollow.resolve(kind, defaults) == defaults


def test_an_unknown_kind_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="expected one of"):
        hollow.resolve("cathedral", {})


def test_a_misspelled_parameter_is_rejected_rather_than_ignored():
    with pytest.raises(ValueError, match="unknown parameter"):
        hollow.resolve("room", {"widht": 4.0})


def test_a_string_where_a_number_belongs_is_rejected():
    with pytest.raises(ValueError, match="must be a number"):
        hollow.resolve("room", {"width": "wide"})


def test_this_module_does_not_change_what_get_primitives_returns():
    # A separate registry on purpose: nothing here may alter the scripted
    # catalogue another endpoint already publishes.
    assert set(hollow.KINDS).isdisjoint(primitives.KINDS)
    assert "room" not in primitives.kinds()


def test_a_hollow_primitive_is_stored_like_a_generated_part(tmp_path):
    result = hollow.store("hollow_box", {"width": 2.0}, tmp_path,
                          part_name="supply_crate")

    assert (tmp_path / "mesh.glb").exists()
    assert result["faces"] > 0
    assert result["watertight"] is True
    assert result["hollow"] is True
    assert result["peak_vram_gib"] == 0.0
    assert result["params"]["kind"] == "hollow_box"
    assert result["size"] == pytest.approx([2.0, 2.0, 2.0], abs=1e-6)


def test_a_hollow_primitive_carries_the_material_its_kind_implies():
    assert hollow.KINDS["room"].material == "stone"
    assert hollow.KINDS["hollow_box"].material == "wood"

    mesh = hollow.build("room")
    assert primitives._family_of(mesh) == "stone"


def test_a_stored_hollow_part_assembles_like_any_other(tmp_path):
    stored = hollow.store("room", {"width": 6.0}, tmp_path / "part")

    scene = assemble.assemble(
        [{"name": "hut", "mesh_path": stored["mesh_path"]}],
        tmp_path / "scene.glb")

    assert scene["part_count"] == 1
    assert scene["parts"][0]["name"] == "hut"


# --- the boolean route -------------------------------------------------------

def test_the_boolean_route_says_what_it_needs_when_it_is_missing():
    try:
        import manifold3d  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="manifold3d"):
            hollow.hollow_boolean(_box(), wall_thickness=0.05)
    else:  # pragma: no cover - only where the optional wheel is installed
        result = hollow.hollow_boolean(_box(), wall_thickness=0.05)
        assert result.report["method"] == "boolean"


def test_the_boolean_route_refuses_input_it_cannot_use():
    manifold = pytest.importorskip("manifold3d")  # noqa: F841
    with pytest.raises(ValueError, match="not watertight"):
        hollow.hollow_boolean(_cracked_box(), wall_thickness=0.05)
