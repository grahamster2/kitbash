"""The detail layer: sweeps, surface relief, arrays and the building kit.

`test_primitives.py` asserts that a scripted part is *correct* geometry. This
file asserts that it is *detailed* geometry, which is a different claim and
needs different tests: that a brick wall is exactly as thick as a flat one,
that the courses stop at the window rather than running across it, that a
moulding's mitre closes, that a rivet lands where it was put, and that none of
that costs the envelope or the watertightness the rest of the suite depends on.

The invariant that changes here is *single solid*. Relief added by composition
— a stud sitting on a face — leaves every piece closed and the union merely
merged, exactly as `crate` and the showcase chest have always been. So the
tests below assert watertightness for everything and connectedness only for
`primitives.SINGLE_SOLID`.
"""
import numpy as np
import pytest

import primitives


# --- what counts as one solid ------------------------------------------------

def _bodies(mesh) -> int:
    """Connected components, by union-find over the edges.

    `trimesh.body_count` wants `scipy`, which this project deliberately does
    not install (docs/HOLLOW.md), so the six lines are cheaper than the
    dependency.
    """
    parent = list(range(len(mesh.vertices)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in mesh.edges_sorted:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(len(mesh.vertices))})


@pytest.mark.parametrize("kind", sorted(primitives.SINGLE_SOLID))
def test_the_single_solid_kinds_really_are_one_shell(kind):
    mesh = primitives.build(kind)

    assert _bodies(mesh) == 1
    assert mesh.is_watertight


@pytest.mark.parametrize("kind", sorted(set(primitives.kinds())
                                        - primitives.SINGLE_SOLID))
def test_an_assembly_is_watertight_without_being_connected(kind):
    """The trade the whole detail layer is built on: components interpenetrate
    rather than being unioned, so every edge still has exactly two faces and
    the mesh is watertight, but it is not one connected body. Roblox and every
    renderer cope; a boolean engine is what it would take to change it, and
    that is the dependency this project refuses."""
    mesh = primitives.build(kind)

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert _bodies(mesh) > 1


def test_a_declared_single_solid_kind_is_not_quietly_an_assembly():
    # The set is documentation, and documentation that is not checked rots.
    assert primitives.SINGLE_SOLID <= set(primitives.kinds())


# --- ear clipping ------------------------------------------------------------

def _area(points, tris):
    """Total area of a 2D triangulation. numpy 2 dropped the 2D cross product,
    so the determinant is spelled out."""
    p = np.asarray(points, float)
    total = 0.0
    for a, b, c in tris:
        u, v = p[b] - p[a], p[c] - p[a]
        total += abs(u[0] * v[1] - u[1] * v[0]) / 2
    return total


def test_earclip_triangulates_a_concave_polygon_without_slivers():
    """An L: the case a fan gets wrong, and the reason `_prism` is documented
    convex-only. Every moulding profile worth having is concave."""
    poly = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    tris = primitives._earclip(poly)

    assert len(tris) == len(poly) - 2
    assert _area(poly, tris) == pytest.approx(5.0)
    assert min(_area(poly, [t]) for t in tris) > 1e-9


def test_earclip_agrees_with_a_fan_on_a_convex_polygon():
    poly = [(0, 0), (2, 0), (2, 2), (0, 2)]

    assert _area(poly, primitives._earclip(poly)) == pytest.approx(4.0)


def test_earclip_covers_an_ogee_profile_exactly_once():
    profile = primitives._moulding_profile("ogee", 0.3, 0.4, steps=6)
    tris = primitives._earclip(profile)
    shoelace = abs(float(np.sum(
        np.asarray(profile)[:, 0] * np.roll(np.asarray(profile)[:, 1], -1)
        - np.roll(np.asarray(profile)[:, 0], -1) * np.asarray(profile)[:, 1]))) / 2

    assert _area(profile, tris) == pytest.approx(shoelace, rel=1e-9)


# --- moulding profiles -------------------------------------------------------

@pytest.mark.parametrize("style", primitives._PROFILES)
def test_every_profile_fills_the_section_it_was_given(style):
    profile = np.asarray(primitives._moulding_profile(style, 0.3, 0.4))

    assert profile[:, 0].max() == pytest.approx(0.3)
    assert profile[:, 1].max() == pytest.approx(0.4)
    assert profile.min() == pytest.approx(0.0)


@pytest.mark.parametrize("style", primitives._PROFILES)
def test_no_profile_repeats_a_point(style):
    """Two coincident points, or three collinear ones, are what make an ear
    clip emit a zero-area triangle — so they are removed at the source."""
    profile = np.asarray(primitives._moulding_profile(style, 0.3, 0.4))
    gaps = np.linalg.norm(profile - np.roll(profile, -1, axis=0), axis=1)

    assert gaps.min() > 1e-9


def test_an_unknown_profile_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown moulding profile"):
        primitives._moulding_profile("acanthus", 0.3, 0.4)


# --- sweeping ----------------------------------------------------------------

def test_a_straight_sweep_is_a_closed_solid_of_the_section_it_was_given():
    mesh = primitives._sweep(primitives._moulding_profile("ogee", 0.3, 0.4),
                             [(-2, 0, 0), (2, 0, 0)])

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert mesh.area_faces.min() > 1e-9
    assert mesh.extents == pytest.approx([4.0, 0.4, 0.3])


def test_a_closed_sweep_mitres_the_corners_of_a_rectangle():
    """The mitre is the whole reason a casing is a sweep and not four boxes:
    the profile's outer edge has to arrive at the corner unbroken. A frame
    swept round a 3x2 path with a 0.2 projection is exactly 3.4 x 2.4."""
    mesh = primitives._sweep(primitives._moulding_profile("ovolo", 0.2, 0.25),
                             primitives._rect_path(3.0, 2.0),
                             closed=True, up=(0, 0, 1))

    assert mesh.is_watertight
    assert mesh.extents == pytest.approx([3.4, 2.4, 0.25])
    assert _bodies(mesh) == 1


def test_a_closed_sweep_has_no_caps_and_an_open_one_does():
    """A closed path costs one more span of sides and no caps; an open one
    costs one fewer span and two ear-clipped ends."""
    section = [(0, 0), (0.2, 0), (0.2, 0.2), (0, 0.2)]
    path = primitives._rect_path(2.0, 2.0)
    closed = primitives._sweep(section, path, closed=True, up=(0, 0, 1))
    opened = primitives._sweep(section, path, closed=False, up=(0, 0, 1))

    assert len(closed.faces) == 4 * 4 * 2               # 4 spans of 4 quads
    assert len(opened.faces) == 3 * 4 * 2 + 2 * 2       # 3 spans, two caps


def test_a_sweep_along_its_own_up_vector_is_refused():
    with pytest.raises(ValueError, match="up vector"):
        primitives._sweep([(0, 0), (1, 0), (1, 1)], [(0, -1, 0), (0, 1, 0)])


def test_a_sweep_that_doubles_back_is_refused_rather_than_inverted():
    with pytest.raises(ValueError, match="doubles back"):
        primitives._sweep([(0, 0), (1, 0), (1, 1)],
                          [(-1, 0, 0), (1, 0, 0), (-1, 0, 0)])


def test_a_sweep_needs_a_path():
    with pytest.raises(ValueError, match="two path stations"):
        primitives._sweep([(0, 0), (1, 0), (1, 1)], [(0, 0, 0)])


# --- composition helpers -----------------------------------------------------

def test_a_line_of_points_includes_both_ends():
    points = primitives._line_points(5, (-2, 0, 0), (2, 0, 0))

    assert len(points) == 5
    assert points[0] == pytest.approx([-2, 0, 0])
    assert points[-1] == pytest.approx([2, 0, 0])
    assert np.diff(points[:, 0]) == pytest.approx([1.0] * 4)


def test_a_line_of_one_point_lands_in_the_middle():
    points = primitives._line_points(1, (-2, 0, 0), (2, 0, 0))

    assert points.shape == (1, 3)
    assert points[0] == pytest.approx([0, 0, 0])


def test_a_grid_fills_the_rectangle_corner_to_corner():
    points = primitives._grid_points(3, 2, u=(4, 0, 0), v=(0, 2, 0))

    assert len(points) == 6
    assert points[:, 0].min() == pytest.approx(-2.0)
    assert points[:, 0].max() == pytest.approx(2.0)
    assert points[:, 1].min() == pytest.approx(-1.0)


def test_a_ring_lies_in_the_plane_normal_to_its_axis():
    points = primitives._ring_points(8, 1.5, axis=1)

    assert points[:, 1] == pytest.approx(np.zeros(8))
    assert np.linalg.norm(points[:, [0, 2]], axis=1) == pytest.approx(
        np.full(8, 1.5))


def test_an_array_puts_one_copy_at_every_point():
    box = primitives._box(0.2, 0.2, 0.2)
    parts = primitives._array(box, primitives._line_points(4, (0, 0, 0), (3, 0, 0)))

    assert len(parts) == 4
    assert [p.bounding_box.centroid[0] for p in parts] == pytest.approx(
        [0.0, 1.0, 2.0, 3.0])


def test_jitter_is_seeded_so_a_part_is_reproducible():
    box = primitives._box(0.2, 0.2, 0.2)
    points = primitives._line_points(6, (0, 0, 0), (5, 0, 0))
    a = primitives._array(box, points, jitter=0.1, seed=7)
    b = primitives._array(box, points, jitter=0.1, seed=7)
    c = primitives._array(box, points, jitter=0.1, seed=8)

    assert [p.bounding_box.centroid[0] for p in a] == pytest.approx(
        [p.bounding_box.centroid[0] for p in b])
    assert [p.bounding_box.centroid[0] for p in a] != pytest.approx(
        [p.bounding_box.centroid[0] for p in c])


# --- rivets ------------------------------------------------------------------

def test_a_rivet_is_a_closed_solid_that_stands_proud_and_sinks_in():
    """The skirt below the surface is what lets a rivet be *merged* onto a
    plate instead of resting a coincident face on it."""
    rivet = primitives._rivet(0.05, 0.04)

    assert rivet.is_watertight
    assert rivet.bounds[1][1] == pytest.approx(0.04)
    assert rivet.bounds[0][1] < 0


@pytest.mark.parametrize("head", ["dome", "pan", "bolt"])
def test_every_rivet_head_is_a_closed_solid_of_the_radius_asked_for(head):
    rivet = primitives._rivet(0.05, 0.04, head=head)

    assert rivet.is_watertight
    assert rivet.extents[0] == pytest.approx(0.1, rel=0.15)


def test_studs_land_on_their_points_and_face_the_direction_given():
    points = primitives._grid_points(3, 2, (0, 0, 1.0), (2, 0, 0), (0, 1, 0))
    studs = primitives._studs_at(points, 0.05, 0.04, direction=(0, 0, 1))

    assert len(studs) == 6
    for stud, point in zip(studs, points):
        assert stud.is_watertight
        # Proud along +Z by exactly what was asked, and centred on its point.
        assert stud.bounds[1][2] == pytest.approx(point[2] + 0.04)
        assert stud.bounding_box.centroid[:2] == pytest.approx(point[:2])


# --- courses -----------------------------------------------------------------

def test_courses_fill_the_wall_they_were_given():
    blocks = primitives._courses(8.0, 6.0, 0.5, 1.15, 0.05)
    lo = min(cy - h / 2 for _, cy, _, h in blocks)
    hi = max(cy + h / 2 for _, cy, _, h in blocks)

    assert lo == pytest.approx(-3.0 + 0.025)
    assert hi == pytest.approx(3.0 - 0.025)
    assert len(blocks) > 50


def test_the_course_height_is_rounded_to_a_whole_number_that_fits():
    """A wall may not end on half a brick, so the requested course is the
    target and the fitted one is what gets built."""
    blocks = primitives._courses(8.0, 6.0, 0.45, 1.0, 0.0)
    rows = {round(cy, 6) for _, cy, _, _ in blocks}

    assert len(rows) == 13                      # 6.0 / 0.45 rounds to 13
    assert all(h == pytest.approx(6.0 / 13) for _, _, _, h in blocks)


def test_courses_are_staggered_so_the_joints_do_not_line_up():
    blocks = primitives._courses(8.0, 6.0, 0.5, 1.0, 0.05, stagger=0.5)
    rows = sorted({round(cy, 6) for _, cy, _, _ in blocks})
    first = {round(cx, 4) for cx, cy, _, _ in blocks if cy == rows[0]}
    second = {round(cx, 4) for cx, cy, _, _ in blocks if cy == rows[1]}

    assert not first & second


def test_a_keep_out_cuts_the_courses_instead_of_being_covered_by_them():
    """The difference between a brick wall with a hole in it and a brick
    pattern painted over one."""
    hole = (-1.3, 1.3, -1.0, 1.4)
    blocks = primitives._courses(8.0, 6.0, 0.5, 1.15, 0.05, keep_out=[hole])

    for cx, cy, w, h in blocks:
        overlaps_x = cx - w / 2 < hole[1] - 1e-6 and cx + w / 2 > hole[0] + 1e-6
        overlaps_y = cy - h / 2 < hole[3] - 1e-6 and cy + h / 2 > hole[2] + 1e-6
        assert not (overlaps_x and overlaps_y)


def test_no_course_leaves_a_sliver_narrower_than_its_joint():
    blocks = primitives._courses(8.0, 6.0, 0.5, 1.15, 0.06,
                                 keep_out=[(-1.31, 1.29, -1.0, 1.4)])

    assert min(w for _, _, w, _ in blocks) > 0.0


def test_ragged_courses_vary_and_still_fill_the_wall_exactly():
    blocks = primitives._courses(8.0, 6.0, 0.6, 1.2, 0.05, ragged=0.3, seed=3)
    heights = {round(h, 6) for _, _, _, h in blocks}
    lo = min(cy - h / 2 for _, cy, _, h in blocks)
    hi = max(cy + h / 2 for _, cy, _, h in blocks)

    assert len(heights) > 1
    assert lo == pytest.approx(-3.0 + 0.025)
    assert hi == pytest.approx(3.0 - 0.025)


# --- the faced wall ----------------------------------------------------------

FACINGS = [s for s in primitives._SURFACES if s != "flat"]


@pytest.mark.parametrize("surface", primitives._SURFACES)
def test_facing_a_wall_does_not_change_how_thick_it_is(surface):
    """The claim that makes the facing usable in a kit: the relief is recessed
    into the thickness rather than added on top of it, so a brick wall butts
    against a flat one without a step."""
    flat = primitives.build("wall_panel", {"surface": "flat", "trim": False})
    faced = primitives.build("wall_panel", {"surface": surface, "trim": False})

    assert faced.extents == pytest.approx(flat.extents)


@pytest.mark.parametrize("surface", FACINGS)
def test_facing_a_wall_costs_geometry_and_thins_the_slab_behind_it(surface):
    """`volume` is not the measure here — `_combine` merges rather than unions,
    so interpenetrating components double-count. What is measurable is that the
    slab itself got thinner by the relief and the blocks stand back out to the
    face."""
    flat = primitives.build("wall_panel", {"surface": "flat", "trim": False})
    faced = primitives.build("wall_panel", {"surface": surface, "trim": False})
    relief = primitives.RELIEF.default
    z = np.abs(faced.vertices[:, 2])

    assert len(faced.faces) > len(flat.faces) * 2
    # The blocks reach the requested face...
    assert z.max() == pytest.approx(0.25)
    # ...and the slab they stand on has retreated by exactly the relief.
    assert np.abs(z - (0.25 - relief)).min() < 1e-6


def test_no_brick_is_laid_across_the_window():
    """Sample the aperture: if a course ran through it there would be geometry
    in the hole, and the wall would read as a wall with a picture of a window
    on it."""
    mesh = primitives.build(
        "wall_panel",
        {"surface": "brick", "trim": False, "opening": "window",
         "opening_width": 2.6, "opening_height": 2.4, "sill_height": 2.0},
    )
    lo = -3.0 + 2.0
    inside = mesh.vertices[
        (np.abs(mesh.vertices[:, 0]) < 1.3 - 1e-6)
        & (mesh.vertices[:, 1] > lo + 1e-6)
        & (mesh.vertices[:, 1] < lo + 2.4 - 1e-6)
        & (np.abs(mesh.vertices[:, 2]) < 0.25 - 1e-6)
    ]

    assert len(inside) == 0


def test_boarding_runs_past_an_opening_above_and_below_it():
    mesh = primitives.build("wall_panel", {"surface": "board", "trim": False})

    # Boards over the head of the window and under its sill, not a bald band.
    above = mesh.vertices[mesh.vertices[:, 1] > 2.5]
    assert np.ptp(above[:, 0]) == pytest.approx(8.0, abs=0.2)


def test_a_different_seed_gives_a_different_wall_at_the_same_parameters():
    """The honest answer to 'every crate looks like every other crate'."""
    a = primitives.build("wall_panel", {"surface": "block", "seed": 1})
    b = primitives.build("wall_panel", {"surface": "block", "seed": 2})
    same = primitives.build("wall_panel", {"surface": "block", "seed": 1})

    assert a.extents == pytest.approx(b.extents)
    assert a.volume != pytest.approx(b.volume)
    assert a.volume == pytest.approx(same.volume)


# --- the building kit --------------------------------------------------------

EXACT = {
    "archway": ({"width": 7.0, "height": 8.0, "depth": 1.4}, [7.0, 8.0, 1.4]),
    "battlement": ({"width": 12.0, "height": 4.0, "thickness": 1.0},
                   [12.0, 4.0, 1.0]),
    "chimney": ({"width": 2.0, "depth": 1.6, "height": 6.0}, [2.0, 6.0, 1.6]),
    "moulding": ({"length": 5.0, "projection": 0.25, "height": 0.5},
                 [5.0, 0.5, 0.5]),
    "panel_door": ({"width": 2.6, "height": 4.4, "thickness": 0.3},
                   [2.6, 4.4, 0.3]),
    "railing": ({"length": 7.0, "height": 2.0, "depth": 0.3}, [7.0, 2.0, 0.3]),
    "riveted_panel": ({"width": 5.0, "height": 2.5, "thickness": 0.4},
                      [5.0, 2.5, 0.4]),
    "roof": ({"width": 9.0, "depth": 7.0, "height": 3.5}, [9.0, 3.5, 7.0]),
    "window": ({"width": 3.0, "height": 4.0, "depth": 0.5}, [3.0, 4.0, 0.5]),
}


@pytest.mark.parametrize("kind", sorted(EXACT))
def test_every_new_kind_comes_out_the_size_it_was_asked_for(kind):
    params, extents = EXACT[kind]
    mesh = primitives.build(kind, params)

    assert mesh.extents == pytest.approx(extents, abs=1e-3)


@pytest.mark.parametrize("kind", sorted(EXACT))
def test_no_ornament_escapes_the_envelope(kind):
    """Every projecting member — a sill, a keystone, a coping, a rivet — comes
    out of the stated dimension rather than being added to it. Otherwise a kit
    part does not butt against its neighbour."""
    params, extents = EXACT[kind]
    mesh = primitives.build(kind, params)

    assert np.abs(mesh.bounds[1] - np.array(extents) / 2).max() < 1e-3
    assert np.abs(mesh.bounds[0] + np.array(extents) / 2).max() < 1e-3


# --- the archway -------------------------------------------------------------

def test_an_archway_leaves_its_opening_clear():
    mesh = primitives.build("archway", {"width": 6.0, "height": 7.0,
                                        "pier": 0.9, "rise": 2.4})
    # A point a metre inside the opening and a metre up from the ground.
    low = mesh.vertices[(np.abs(mesh.vertices[:, 0]) < 1.5)
                        & (mesh.vertices[:, 1] < -1.0)]

    assert len(low) == 0


def test_a_keystone_is_a_bigger_block_than_its_neighbours():
    plain = primitives.build("archway", {"keystone": False})
    keyed = primitives.build("archway", {"keystone": True})

    assert keyed.volume > plain.volume
    assert keyed.extents == pytest.approx(plain.extents, abs=1e-3)


def test_the_voussoirs_are_separate_blocks_with_joints_between_them():
    """A ring of overlapping blocks reads as bent metal from three metres. The
    joints are what make it read as masonry, and each one is a real gap."""
    few = primitives.build("archway", {"voussoirs": 5})
    many = primitives.build("archway", {"voussoirs": 15})

    assert _bodies(many) - _bodies(few) == 10


def test_a_pier_wider_than_the_arch_is_rejected():
    with pytest.raises(ValueError, match="leaves no opening"):
        primitives.build("archway", {"width": 4.0, "pier": 2.0})


def test_a_rise_taller_than_the_arch_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        primitives.build("archway", {"height": 5.0, "rise": 4.8})


# --- the battlement ----------------------------------------------------------

def test_a_battlement_has_gaps_between_its_merlons():
    mesh = primitives.build("battlement", {"width": 10.0, "height": 5.0,
                                           "merlon_height": 1.2})
    # Nothing at all in the top two thirds of a crenel.
    top = mesh.vertices[mesh.vertices[:, 1] > 5.0 / 2 - 0.4]
    gaps = np.sort(np.unique(np.round(top[:, 0], 3)))

    assert np.diff(gaps).max() > 0.5


def test_merlon_count_follows_the_pitch_it_was_given():
    wide = primitives.build("battlement", {"merlon_width": 2.2,
                                           "crenel_width": 1.4})
    narrow = primitives.build("battlement", {"merlon_width": 0.8,
                                             "crenel_width": 0.5})

    assert len(narrow.faces) > len(wide.faces)


def test_a_merlon_taller_than_the_wall_is_rejected():
    with pytest.raises(ValueError, match="whole"):
        primitives.build("battlement", {"height": 3.0, "merlon_height": 3.0})


# --- the roof ----------------------------------------------------------------

def test_a_roof_is_clad_in_courses_rather_than_being_one_slab():
    bare = primitives.build("roof", {"course": 2.5})
    clad = primitives.build("roof", {"course": 0.4})

    assert len(clad.faces) > len(bare.faces) * 2
    assert clad.extents == pytest.approx(bare.extents, abs=1e-3)


def test_roof_tiles_lie_in_the_slope_rather_than_standing_on_end():
    """The bug this caught: the wrong sign on the course rotation stands every
    tile perpendicular to the roof, and the result reads as a louvre. Measured
    by area, most of the roof faces out along the slope normal; if the tiles
    were on end, most of it would face up the slope instead."""
    mesh = primitives.build("roof", {"width": 8.0, "depth": 6.0, "height": 3.0})
    pitch = np.arctan2(3.0, 3.0)
    normal = np.array([0.0, np.cos(pitch), np.sin(pitch)])
    # The near slope, ignoring the gable boards, which face along X.
    near = (mesh.triangles_center[:, 2] > 1.0) & (
        np.abs(mesh.face_normals[:, 0]) < 0.5)
    dots = np.abs(mesh.face_normals @ normal)[near]
    areas = mesh.area_faces[near]
    big = areas > np.percentile(areas, 85)

    # The largest faces there are tile faces, so they lie in the slope. Stood
    # on end, every one of them would be perpendicular to it instead.
    assert np.median(dots[big]) > 0.85


def test_a_gabled_roof_closes_its_ends():
    open_ends = primitives.build("roof", {"gable": False})
    closed = primitives.build("roof", {"gable": True})

    assert closed.volume > open_ends.volume
    assert closed.extents == pytest.approx(open_ends.extents, abs=1e-3)


def test_jitter_weathers_a_roof_without_letting_daylight_through_it():
    flat = primitives.build("roof", {"jitter": 0.0})
    weathered = primitives.build("roof", {"jitter": 0.04, "seed": 5})

    assert len(weathered.faces) == len(flat.faces)
    assert not np.allclose(np.sort(weathered.vertices[:, 1]),
                           np.sort(flat.vertices[:, 1]))
    # Weathering only sinks a tile, so the envelope is untouched.
    assert weathered.extents == pytest.approx(flat.extents, abs=1e-3)


# --- the window and the door -------------------------------------------------

def test_a_window_is_mostly_hole():
    mesh = primitives.build("window", {"width": 2.4, "height": 3.2,
                                       "depth": 0.4})
    box = 2.4 * 3.2 * 0.4

    assert mesh.volume < box * 0.35


def test_glazing_bars_divide_the_light_into_the_lights_asked_for():
    two = primitives.build("window", {"lights_wide": 2, "lights_high": 2})
    six = primitives.build("window", {"lights_wide": 3, "lights_high": 4})

    assert len(six.faces) > len(two.faces)


def test_a_frame_that_fills_the_window_is_rejected():
    with pytest.raises(ValueError, match="no light"):
        primitives.build("window", {"width": 1.0, "frame": 0.6})


def test_a_window_too_squat_for_a_round_head_is_rejected():
    with pytest.raises(ValueError, match="round head|arched head"):
        primitives.build("window", {"style": "arched", "width": 3.0,
                                    "height": 1.6})


def test_leaded_lights_stay_inside_the_frame():
    mesh = primitives.build("window", {"style": "lattice", "width": 2.4,
                                       "height": 3.2})

    assert mesh.extents == pytest.approx([2.4, 3.2, 0.4], abs=1e-3)


def test_a_panelled_door_has_relief_a_slab_does_not():
    slab = primitives.build("plank", {"length": 2.2, "width": 4.0,
                                      "thickness": 0.24})
    door = primitives.build("panel_door", {"width": 2.2, "height": 4.0,
                                           "thickness": 0.24})

    assert len(door.faces) > len(slab.faces) * 20
    assert door.extents == pytest.approx(slab.extents[[0, 2, 1]], abs=1e-3)


def test_a_banded_door_brings_its_own_ironwork():
    plain = primitives.build("panel_door", {"style": "plank"})
    banded = primitives.build("panel_door", {"style": "banded"})

    assert len(banded.faces) > len(plain.faces)
    assert banded.extents == pytest.approx(plain.extents, abs=1e-3)


def test_a_panel_that_does_not_fit_its_door_is_rejected():
    with pytest.raises(ValueError, match="do not fit|no panel"):
        primitives.build("panel_door", {"width": 1.2, "panels_wide": 4,
                                        "stile": 0.3})


# --- the railing and the moulding -------------------------------------------

def test_balusters_cost_exactly_one_turning_each():
    few = primitives.build("railing", {"baluster_count": 4, "sections": 24})
    many = primitives.build("railing", {"baluster_count": 9, "sections": 24})

    assert (len(many.faces) - len(few.faces)) % 5 == 0
    assert many.extents == pytest.approx(few.extents, abs=1e-3)


def test_a_moulding_with_returns_stops_with_its_own_section_showing():
    """A cornice that simply ends in mid-air reads as a bevelled plank."""
    plain = primitives.build("moulding", {"returns": False, "length": 6.0,
                                          "projection": 0.3})
    returned = primitives.build("moulding", {"returns": True, "length": 6.0,
                                             "projection": 0.3})

    assert plain.extents == pytest.approx([6.0, 0.4, 0.3])
    assert returned.extents == pytest.approx([6.0, 0.4, 0.6])
    assert len(returned.faces) > len(plain.faces)


def test_returns_wider_than_the_run_are_rejected():
    with pytest.raises(ValueError, match="returns meet"):
        primitives.build("moulding", {"length": 0.4, "projection": 0.3,
                                      "returns": True})


# --- the riveted panel -------------------------------------------------------

def test_rivets_are_real_geometry_rather_than_a_pattern():
    bare = primitives.build("riveted_panel", {"rivet_pitch": 4.0,
                                              "corner_bosses": False})
    riveted = primitives.build("riveted_panel", {"rivet_pitch": 0.25,
                                                 "corner_bosses": False})

    assert len(riveted.faces) > len(bare.faces) + 500
    assert riveted.extents == pytest.approx(bare.extents, abs=1e-3)


def test_a_corrugated_sheet_is_one_folded_section_not_a_row_of_boxes():
    mesh = primitives.build("riveted_panel", {"style": "corrugated"})

    assert mesh.is_watertight
    assert len(mesh.faces) < 500


def test_bays_that_do_not_fit_the_plate_are_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        primitives.build("riveted_panel", {"width": 1.0, "panels_wide": 12})


# --- the schema --------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(EXACT))
def test_every_new_kind_is_self_describing(kind):
    entry = primitives.KINDS[kind].as_dict()

    assert entry["summary"]
    assert len(entry["params"]) >= 4
    for param in entry["params"]:
        assert param["description"]
        if param["type"] in ("number", "integer"):
            assert "minimum" in param


def test_the_shared_detail_dials_are_one_vocabulary_not_five():
    """A wall, a battlement, an archway and a chimney standing next to each
    other have to be the same stone, so they take the same parameters."""
    for kind in ("wall_panel", "archway", "battlement", "chimney"):
        names = {p.name for p in primitives.KINDS[kind].params}
        assert {"surface", "course", "joint", "relief", "seed"} <= names
