"""Relative placement: anchors, mirroring, resolution order and the bounds report.

The bug this exists to prevent is a part floating in space next to the thing it
was supposed to be bolted to, so nearly every test here asserts on the *world*
bounds of the result rather than on a matrix.
"""
import numpy as np
import pytest
import trimesh

import assemble


@pytest.fixture
def box(make_mesh):
    """A 1x1x1 box at the origin — the simplest thing with measurable faces."""
    return make_mesh(trimesh.creation.box(extents=(1, 1, 1)))


def place(parts, tmp_path):
    """Assemble and return the per-part report keyed by name."""
    result = assemble.assemble(parts, tmp_path / "scene.glb")
    return {p["name"]: p for p in result["parts"]}


# --- the report -------------------------------------------------------------


def test_every_part_reports_its_world_bounds(box, tmp_path):
    parts = place(
        [{"name": "hull", "mesh_path": str(box), "position": [2, 0, 0], "scale": 4}],
        tmp_path,
    )

    assert parts["hull"]["bounds_min"] == [0.0, -2.0, -2.0]
    assert parts["hull"]["bounds_max"] == [4.0, 2.0, 2.0]
    assert parts["hull"]["size"] == [4.0, 4.0, 4.0]
    assert parts["hull"]["center"] == [2.0, 0.0, 0.0]
    assert parts["hull"]["position"] == [2.0, 0.0, 0.0]


def test_reported_bounds_follow_a_rotation(box, tmp_path):
    """The bounds are of the placed geometry, not of the file's box pushed
    through the matrix — those differ the moment anything is rotated."""
    parts = place(
        [{"name": "fin", "mesh_path": str(box), "rotation": [0, 45, 0]}], tmp_path
    )

    assert parts["fin"]["size"] == pytest.approx([1.4142, 1.0, 1.4142], abs=1e-3)


def test_an_unanchored_part_reports_no_anchor(box, tmp_path):
    parts = place([{"name": "hull", "mesh_path": str(box)}], tmp_path)

    assert parts["hull"]["anchored_to"] is None
    assert parts["hull"]["mirrored_from"] is None


def test_a_mirrored_part_reports_where_it_came_from(box, tmp_path):
    """Otherwise a mirrored part is indistinguishable in the report from one
    placed at an absolute position."""
    parts = place(
        [
            {"name": "left", "mesh_path": str(box), "position": [1, 0, 0]},
            {"name": "right", "mesh_path": str(box), "mirror_of": "left"},
        ],
        tmp_path,
    )

    assert parts["right"]["mirrored_from"] == "left"
    assert parts["left"]["mirrored_from"] is None


# --- attaching one part to another ------------------------------------------


def test_a_part_hangs_off_the_bottom_face_of_another(box, tmp_path):
    """The wheel-on-a-strut case. `under` sets both sides of the join, so the
    faces touch instead of the two centres coinciding."""
    parts = place(
        [
            {"name": "strut", "mesh_path": str(box), "scale": [0.1, 2, 0.1]},
            {
                "name": "wheel",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "strut", "align": {"y": "under"}},
            },
        ],
        tmp_path,
    )

    assert parts["strut"]["bounds_min"][1] == -1.0
    assert parts["wheel"]["bounds_max"][1] == pytest.approx(-1.0)
    assert parts["wheel"]["bounds_min"][1] == pytest.approx(-1.5)
    assert parts["wheel"]["anchored_to"] == "strut"


def test_align_alone_centres_the_part_on_the_named_point(box, tmp_path):
    """Without a `my`, the part's centre lands on the target's point — which is
    what "put the wheel at the bottom of the strut" usually means."""
    parts = place(
        [
            {"name": "strut", "mesh_path": str(box), "scale": [0.1, 2, 0.1]},
            {
                "name": "wheel",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "strut", "align": {"y": "min"}},
            },
        ],
        tmp_path,
    )

    assert parts["wheel"]["center"][1] == pytest.approx(-1.0)


def test_my_overrides_the_point_that_lands_on_the_target(box, tmp_path):
    parts = place(
        [
            {"name": "strut", "mesh_path": str(box), "scale": [0.1, 2, 0.1]},
            {
                "name": "wheel",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "strut", "align": {"y": "min"}, "my": {"y": "max"}},
            },
        ],
        tmp_path,
    )

    assert parts["wheel"]["bounds_max"][1] == pytest.approx(-1.0)


def test_a_fraction_places_a_part_along_the_targets_length(box, tmp_path):
    """"under the airframe, a fifth of the way back from the nose"."""
    parts = place(
        [
            {"name": "airframe", "mesh_path": str(box), "scale": [10, 2, 8]},
            {
                "name": "nose_gear",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "airframe", "align": {"x": 0.2, "y": "under"}},
            },
        ],
        tmp_path,
    )

    # The airframe runs -5..+5, so a fifth along is x = -3.
    assert parts["nose_gear"]["center"][0] == pytest.approx(-3.0)
    assert parts["nose_gear"]["bounds_max"][1] == pytest.approx(-1.0)


def test_axes_the_anchor_does_not_name_are_centred_on_the_target(box, tmp_path):
    """An unnamed axis defaults to the target's centre, not to world zero.
    Falling back to zero is exactly how a part ends up floating beside the thing
    it was supposed to be attached to."""
    parts = place(
        [
            {"name": "airframe", "mesh_path": str(box), "scale": [10, 2, 8],
             "position": [0, 20, 30]},
            {
                "name": "left_gear",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "airframe", "align": {"y": "under"}},
            },
        ],
        tmp_path,
    )

    assert parts["left_gear"]["center"] == pytest.approx([0.0, 18.75, 30.0])


def test_an_offset_is_how_you_move_off_the_anchor(box, tmp_path):
    """The left/right gear case: centred fore-aft on the airframe, then moved
    out to the wing root."""
    parts = place(
        [
            {"name": "airframe", "mesh_path": str(box), "scale": [10, 2, 8]},
            {
                "name": "left_gear",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {
                    "to": "airframe", "align": {"y": "under"}, "offset": [0, 0, 3]
                },
            },
        ],
        tmp_path,
    )

    assert parts["left_gear"]["center"] == pytest.approx([0.0, -1.25, 3.0])


def test_an_anchor_to_a_part_refuses_to_share_with_position(box, tmp_path):
    with pytest.raises(ValueError, match=r"`position` cannot apply too"):
        assemble.assemble(
            [
                {"name": "airframe", "mesh_path": str(box)},
                {"name": "gear", "mesh_path": str(box), "position": [1, 2, 3],
                 "anchor": {"to": "airframe", "align": {"y": "under"}}},
            ],
            tmp_path / "scene.glb",
        )


def test_an_offset_nudges_the_part_after_alignment(box, tmp_path):
    parts = place(
        [
            {"name": "strut", "mesh_path": str(box)},
            {
                "name": "wheel",
                "mesh_path": str(box),
                "anchor": {
                    "to": "strut", "align": {"y": "under"}, "offset": [0, -0.25, 0]
                },
            },
        ],
        tmp_path,
    )

    assert parts["wheel"]["bounds_max"][1] == pytest.approx(-0.75)


def test_an_anchor_with_no_align_centres_the_part_inside_the_target(box, tmp_path):
    """"put the cabin inside the fuselage" — the request that has no coordinates
    at all."""
    parts = place(
        [
            {
                "name": "fuselage",
                "mesh_path": str(box),
                "scale": 10,
                "position": [4, 5, 6],
            },
            {
                "name": "cabin",
                "mesh_path": str(box),
                "anchor": {"to": "fuselage"},
            },
        ],
        tmp_path,
    )

    assert parts["cabin"]["center"] == pytest.approx([4.0, 5.0, 6.0])


def test_the_target_box_is_the_scaled_footprint_not_the_file_bounds(box, tmp_path):
    """The failure that motivated all of this. The mesh on disk is 1 unit; the
    part in the scene is a twentieth of that, and anchoring to the file's bounds
    would leave the wheel almost a whole unit away from the strut."""
    parts = place(
        [
            {"name": "strut", "mesh_path": str(box), "scale": 0.05},
            {
                "name": "wheel",
                "mesh_path": str(box),
                "scale": 0.05,
                "anchor": {"to": "strut", "align": {"y": "under"}},
            },
        ],
        tmp_path,
    )

    assert parts["wheel"]["bounds_max"][1] == pytest.approx(-0.025)
    assert parts["wheel"]["size"] == pytest.approx([0.05, 0.05, 0.05])


def test_anchoring_measures_the_target_after_its_rotation(box, tmp_path):
    parts = place(
        [
            {"name": "fin", "mesh_path": str(box), "scale": [2, 1, 1],
             "rotation": [0, 0, 90]},
            {
                "name": "cap",
                "mesh_path": str(box),
                "scale": 0.1,
                "anchor": {"to": "fin", "align": {"y": "above"}},
            },
        ],
        tmp_path,
    )

    # Turned a quarter turn about Z, the 2-long axis is now Y, so the top is +1.
    assert parts["cap"]["bounds_min"][1] == pytest.approx(1.0)


def test_a_part_can_sit_on_the_ground_plane(box, tmp_path):
    parts = place(
        [{"name": "hull", "mesh_path": str(box), "scale": 3,
          "anchor": {"to": "ground"}}],
        tmp_path,
    )

    assert parts["hull"]["bounds_min"][1] == pytest.approx(0.0)
    assert parts["hull"]["anchored_to"] == "ground"


def test_the_ground_leaves_x_and_z_to_position(box, tmp_path):
    parts = place(
        [{"name": "hull", "mesh_path": str(box), "position": [7, 99, -2],
          "anchor": {"to": "ground"}}],
        tmp_path,
    )

    assert parts["hull"]["center"] == pytest.approx([7.0, 0.5, -2.0])


def test_the_ground_rejects_a_horizontal_alignment(box, tmp_path):
    with pytest.raises(ValueError, match="constrains y only"):
        assemble.assemble(
            [{"name": "hull", "mesh_path": str(box),
              "anchor": {"to": "ground", "align": {"x": "min"}}}],
            tmp_path / "scene.glb",
        )


# --- resolution order -------------------------------------------------------


def test_a_part_may_be_listed_before_the_part_it_anchors_to(box, tmp_path):
    parts = place(
        [
            {
                "name": "wheel",
                "mesh_path": str(box),
                "scale": 0.5,
                "anchor": {"to": "strut", "align": {"y": "under"}},
            },
            {"name": "strut", "mesh_path": str(box), "scale": [0.1, 2, 0.1]},
        ],
        tmp_path,
    )

    assert parts["wheel"]["bounds_max"][1] == pytest.approx(-1.0)


def test_a_chain_of_anchors_resolves_through(box, tmp_path):
    """Three deep, listed backwards: hub -> strut -> airframe."""
    parts = place(
        [
            {
                "name": "hub",
                "mesh_path": str(box),
                "scale": 0.2,
                "anchor": {"to": "strut", "align": {"y": "under"}},
            },
            {
                "name": "strut",
                "mesh_path": str(box),
                "scale": [0.1, 1, 0.1],
                "anchor": {"to": "airframe", "align": {"y": "under"}},
            },
            {"name": "airframe", "mesh_path": str(box), "scale": 4,
             "position": [0, 10, 0]},
        ],
        tmp_path,
    )

    assert parts["strut"]["bounds_max"][1] == pytest.approx(8.0)
    assert parts["strut"]["bounds_min"][1] == pytest.approx(7.0)
    assert parts["hub"]["bounds_max"][1] == pytest.approx(7.0)


def test_the_report_stays_in_the_order_the_caller_gave(box, tmp_path):
    result = assemble.assemble(
        [
            {"name": "wheel", "mesh_path": str(box),
             "anchor": {"to": "strut", "align": {"y": "under"}}},
            {"name": "strut", "mesh_path": str(box)},
        ],
        tmp_path / "scene.glb",
    )

    assert [p["name"] for p in result["parts"]] == ["wheel", "strut"]


def test_a_two_part_cycle_is_an_error_and_names_the_loop(box, tmp_path):
    with pytest.raises(ValueError, match=r"placement cycle: .*wheel.*strut"):
        assemble.assemble(
            [
                {"name": "wheel", "mesh_path": str(box),
                 "anchor": {"to": "strut", "align": {"y": "under"}}},
                {"name": "strut", "mesh_path": str(box),
                 "anchor": {"to": "wheel", "align": {"y": "above"}}},
            ],
            tmp_path / "scene.glb",
        )


def test_a_longer_cycle_is_also_caught(box, tmp_path):
    parts = [
        {"name": n, "mesh_path": str(box), "anchor": {"to": t, "align": {"y": "above"}}}
        for n, t in [("a", "c"), ("b", "a"), ("c", "b")]
    ]

    with pytest.raises(ValueError, match="placement cycle"):
        assemble.assemble(parts, tmp_path / "scene.glb")


def test_a_cycle_does_not_stop_unrelated_parts_being_reported(box, tmp_path):
    """The error has to mention the loop, not the innocent bystander."""
    with pytest.raises(ValueError, match=r"placement cycle: (a -> b -> a|b -> a -> b)"):
        assemble.assemble(
            [
                {"name": "free", "mesh_path": str(box)},
                {"name": "a", "mesh_path": str(box), "anchor": {"to": "b"}},
                {"name": "b", "mesh_path": str(box), "anchor": {"to": "a"}},
            ],
            tmp_path / "scene.glb",
        )


def test_a_part_cannot_anchor_to_itself(box, tmp_path):
    with pytest.raises(ValueError, match="cannot anchor to itself"):
        assemble.assemble(
            [{"name": "wheel", "mesh_path": str(box), "anchor": {"to": "wheel"}}],
            tmp_path / "scene.glb",
        )


def test_an_unknown_target_lists_the_parts_that_do_exist(box, tmp_path):
    with pytest.raises(ValueError, match=r"unknown part 'strutt'.*'hull', 'wheel'"):
        assemble.assemble(
            [
                {"name": "hull", "mesh_path": str(box)},
                {"name": "wheel", "mesh_path": str(box),
                 "anchor": {"to": "strutt"}},
            ],
            tmp_path / "scene.glb",
        )


def test_anchoring_to_a_duplicated_name_is_rejected_as_ambiguous(box, tmp_path):
    with pytest.raises(ValueError, match="ambiguous"):
        assemble.assemble(
            [
                {"name": "wing", "mesh_path": str(box)},
                {"name": "wing", "mesh_path": str(box)},
                {"name": "tip", "mesh_path": str(box), "anchor": {"to": "wing"}},
            ],
            tmp_path / "scene.glb",
        )


def test_a_deduplicated_name_can_still_be_anchored_to(box, tmp_path):
    """Uniquification renames the collision to `wing_2`; that name is
    unambiguous, so it stays usable as a target."""
    parts = place(
        [
            {"name": "wing", "mesh_path": str(box)},
            {"name": "wing", "mesh_path": str(box), "position": [0, 6, 0]},
            {"name": "tip", "mesh_path": str(box), "scale": 0.5,
             "anchor": {"to": "wing_2"}},
        ],
        tmp_path,
    )

    assert parts["tip"]["center"] == pytest.approx([0.0, 6.0, 0.0])


def test_an_anchor_needs_a_target(box, tmp_path):
    with pytest.raises(ValueError, match="anchor needs `to`"):
        assemble.assemble(
            [{"name": "wheel", "mesh_path": str(box), "anchor": {"align": {"y": 0}}}],
            tmp_path / "scene.glb",
        )


# --- mirroring --------------------------------------------------------------


def test_mirror_of_reflects_a_whole_placement(box, tmp_path):
    """Left and right gear are the same part; placing one has to place both, or
    the second one is a second chance to get it wrong."""
    parts = place(
        [
            {"name": "left_strut", "mesh_path": str(box), "scale": 0.5,
             "position": [1, -2, 3]},
            {"name": "right_strut", "mesh_path": str(box),
             "mirror_of": "left_strut"},
        ],
        tmp_path,
    )

    assert parts["right_strut"]["center"] == pytest.approx([-1.0, -2.0, 3.0])
    assert parts["right_strut"]["size"] == pytest.approx([0.5, 0.5, 0.5])


def test_mirror_of_carries_the_anchored_position(box, tmp_path):
    parts = place(
        [
            {"name": "airframe", "mesh_path": str(box), "scale": [10, 2, 8]},
            {"name": "left_gear", "mesh_path": str(box), "scale": 0.5,
             "anchor": {"to": "airframe", "align": {"y": "under"},
                        "offset": [3, 0, 0]}},
            {"name": "right_gear", "mesh_path": str(box), "mirror_of": "left_gear"},
        ],
        tmp_path,
    )

    # The height came from the anchor and the mirror preserved it: both legs
    # reach the same distance below the airframe without either being measured.
    assert parts["left_gear"]["center"] == pytest.approx([3.0, -1.25, 0.0])
    assert parts["right_gear"]["center"] == pytest.approx([-3.0, -1.25, 0.0])


def test_mirror_of_can_reflect_across_another_axis(box, tmp_path):
    parts = place(
        [
            {"name": "top", "mesh_path": str(box), "position": [0, 4, 0]},
            {"name": "bottom", "mesh_path": str(box), "mirror_of": "top",
             "mirror": "y"},
        ],
        tmp_path,
    )

    assert parts["bottom"]["center"] == pytest.approx([0.0, -4.0, 0.0])


def test_mirror_of_reflects_about_a_chosen_plane(box, tmp_path):
    parts = place(
        [
            {"name": "left", "mesh_path": str(box), "position": [1, 0, 0]},
            {"name": "right", "mesh_path": str(box), "mirror_of": "left",
             "mirror": {"axis": "x", "about": 5}},
        ],
        tmp_path,
    )

    assert parts["right"]["center"] == pytest.approx([9.0, 0.0, 0.0])


def test_mirror_of_resolves_regardless_of_listing_order(box, tmp_path):
    parts = place(
        [
            {"name": "right_strut", "mesh_path": str(box), "mirror_of": "left_strut"},
            {"name": "left_strut", "mesh_path": str(box), "position": [2, 0, 0]},
        ],
        tmp_path,
    )

    assert parts["right_strut"]["center"] == pytest.approx([-2.0, 0.0, 0.0])


def test_a_mirrored_part_can_be_anchored_to(box, tmp_path):
    parts = place(
        [
            {"name": "left_strut", "mesh_path": str(box), "scale": [0.1, 2, 0.1],
             "position": [1, 0, 0]},
            {"name": "right_strut", "mesh_path": str(box),
             "mirror_of": "left_strut"},
            {"name": "right_wheel", "mesh_path": str(box), "scale": 0.5,
             "anchor": {"to": "right_strut", "align": {"y": "under"}}},
        ],
        tmp_path,
    )

    assert parts["right_wheel"]["center"] == pytest.approx([-1.0, -1.25, 0.0])


def test_mirroring_a_mirror_returns_to_the_original(box, tmp_path):
    parts = place(
        [
            {"name": "a", "mesh_path": str(box), "position": [3, 0, 0]},
            {"name": "b", "mesh_path": str(box), "mirror_of": "a"},
            {"name": "c", "mesh_path": str(box), "mirror_of": "b"},
        ],
        tmp_path,
    )

    assert parts["c"]["center"] == pytest.approx([3.0, 0.0, 0.0])


def test_mirror_flips_a_parts_own_placement(box, tmp_path):
    parts = place(
        [{"name": "fin", "mesh_path": str(box), "position": [2, 0, 0],
          "mirror": "x"}],
        tmp_path,
    )

    assert parts["fin"]["center"] == pytest.approx([-2.0, 0.0, 0.0])


def test_mirror_applies_after_an_anchor(box, tmp_path):
    parts = place(
        [
            {"name": "wing", "mesh_path": str(box), "scale": [4, 1, 1],
             "position": [6, 0, 0]},
            {"name": "tip", "mesh_path": str(box), "scale": 0.5, "mirror": "x",
             "anchor": {"to": "wing", "align": {"x": "max"}}},
        ],
        tmp_path,
    )

    assert parts["tip"]["center"] == pytest.approx([-8.0, 0.0, 0.0])


def test_a_mirrored_part_is_not_inside_out(box, tmp_path):
    """A reflection reverses winding, and a glTF viewer reads reversed winding
    as an inward-facing surface. Flipping the faces back cancels it."""
    out = tmp_path / "scene.glb"
    assemble.assemble(
        [
            {"name": "left", "mesh_path": str(box), "position": [1, 0, 0]},
            {"name": "right", "mesh_path": str(box), "mirror_of": "left"},
        ],
        out,
    )

    scene = trimesh.load(str(out))
    for name, geom in scene.geometry.items():
        node = scene.graph.get(name)[0]
        # Signed volume says which way the faces point; the reflection in the
        # node transform flips that sign again, so the product is what a viewer
        # ends up rendering. (trimesh's own apply_transform re-flips faces for a
        # negative determinant, which would hide the bug from a naive check.)
        outward = geom.volume * np.sign(np.linalg.det(node[:3, :3]))
        assert outward > 0, f"{name} is inside-out"


def test_mirror_of_refuses_to_share_a_part_with_position(box, tmp_path):
    with pytest.raises(ValueError, match="cannot also set position"):
        assemble.assemble(
            [
                {"name": "left", "mesh_path": str(box)},
                {"name": "right", "mesh_path": str(box), "mirror_of": "left",
                 "position": [1, 0, 0]},
            ],
            tmp_path / "scene.glb",
        )


def test_a_part_cannot_mirror_itself(box, tmp_path):
    with pytest.raises(ValueError, match="cannot mirror itself"):
        assemble.assemble(
            [{"name": "left", "mesh_path": str(box), "mirror_of": "left"}],
            tmp_path / "scene.glb",
        )


def test_an_unknown_mirror_axis_is_rejected(box, tmp_path):
    with pytest.raises(ValueError, match="mirror axis"):
        assemble.assemble(
            [{"name": "fin", "mesh_path": str(box), "mirror": "w"}],
            tmp_path / "scene.glb",
        )


# --- the placement vocabulary -----------------------------------------------


@pytest.mark.parametrize(
    "spec, expected_centre_y",
    [
        ("min", -1.0), ("bottom", -1.0),
        ("center", 0.0), ("centre", 0.0), ("middle", 0.0),
        ("max", 1.0), ("top", 1.0),
        (0.25, -0.5), (0.0, -1.0), (1.0, 1.0),
    ],
)
def test_align_accepts_names_and_fractions_alike(box, tmp_path, spec,
                                                 expected_centre_y):
    parts = place(
        [
            {"name": "post", "mesh_path": str(box), "scale": [1, 2, 1]},
            {"name": "clip", "mesh_path": str(box), "scale": 0.1,
             "anchor": {"to": "post", "align": {"y": spec}}},
        ],
        tmp_path,
    )

    assert parts["clip"]["center"][1] == pytest.approx(expected_centre_y)


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("under", (-1.1, -1.0)), ("below", (-1.1, -1.0)),
        ("above", (1.0, 1.1)), ("over", (1.0, 1.1)), ("on", (1.0, 1.1)),
        ("flush_min", (-1.0, -0.9)), ("flush_max", (0.9, 1.0)),
    ],
)
def test_attachment_keywords_make_faces_touch(box, tmp_path, spec, expected):
    parts = place(
        [
            {"name": "post", "mesh_path": str(box), "scale": [1, 2, 1]},
            {"name": "clip", "mesh_path": str(box), "scale": 0.1,
             "anchor": {"to": "post", "align": {"y": spec}}},
        ],
        tmp_path,
    )

    assert parts["clip"]["bounds_min"][1] == pytest.approx(expected[0])
    assert parts["clip"]["bounds_max"][1] == pytest.approx(expected[1])


def test_an_attachment_keyword_still_yields_to_an_explicit_my(box, tmp_path):
    parts = place(
        [
            {"name": "post", "mesh_path": str(box), "scale": [1, 2, 1]},
            {"name": "clip", "mesh_path": str(box), "scale": 0.1,
             "anchor": {"to": "post", "align": {"y": "under"}, "my": {"y": "center"}}},
        ],
        tmp_path,
    )

    assert parts["clip"]["center"][1] == pytest.approx(-1.0)


@pytest.mark.parametrize("bad", ["middleish", True, None, [0.5]])
def test_an_unusable_alignment_is_rejected_with_the_vocabulary(box, tmp_path, bad):
    with pytest.raises(ValueError, match="align.y"):
        assemble.assemble(
            [
                {"name": "post", "mesh_path": str(box)},
                {"name": "clip", "mesh_path": str(box),
                 "anchor": {"to": "post", "align": {"y": bad}}},
            ],
            tmp_path / "scene.glb",
        )


def test_an_attachment_keyword_in_my_says_where_it_belongs(box, tmp_path):
    with pytest.raises(ValueError, match=r"belong in `align`"):
        assemble.assemble(
            [
                {"name": "post", "mesh_path": str(box)},
                {"name": "clip", "mesh_path": str(box),
                 "anchor": {"to": "post", "my": {"y": "under"}}},
            ],
            tmp_path / "scene.glb",
        )


def test_a_misspelled_axis_is_rejected_rather_than_ignored(box, tmp_path):
    """`{"Y": ...}` is fine, `{"up": ...}` is not — silently dropping it would
    mean the anchor constrains nothing and the part sits at the origin."""
    with pytest.raises(ValueError, match="expected x, y or z"):
        assemble.assemble(
            [
                {"name": "post", "mesh_path": str(box)},
                {"name": "clip", "mesh_path": str(box),
                 "anchor": {"to": "post", "align": {"up": "min"}}},
            ],
            tmp_path / "scene.glb",
        )


def test_an_axis_name_is_case_insensitive(box, tmp_path):
    parts = place(
        [
            {"name": "post", "mesh_path": str(box), "scale": [1, 2, 1]},
            {"name": "clip", "mesh_path": str(box), "scale": 0.1,
             "anchor": {"to": "post", "align": {"Y": "MAX"}}},
        ],
        tmp_path,
    )

    assert parts["clip"]["center"][1] == pytest.approx(1.0)


def test_a_short_offset_is_rejected(box, tmp_path):
    with pytest.raises(ValueError, match=r"offset must be \[x, y, z\]"):
        assemble.assemble(
            [
                {"name": "post", "mesh_path": str(box)},
                {"name": "clip", "mesh_path": str(box),
                 "anchor": {"to": "post", "offset": [1, 2]}},
            ],
            tmp_path / "scene.glb",
        )


# --- over the API -----------------------------------------------------------


def test_the_api_accepts_an_anchor(client, finished_job):
    """The stubbed generator writes a 1x2x3 box, so both parts are that."""
    hull = finished_job("hull")
    fin = finished_job("fin")

    body = {
        "parts": [
            {"job_id": fin, "name": "fin", "scale": 0.5,
             "anchor": {"to": "hull", "align": {"y": "above"}}},
            {"job_id": hull, "name": "hull"},
        ]
    }
    result = client.post("/assemble", json=body).json()
    parts = {p["name"]: p for p in result["parts"]}

    assert parts["hull"]["bounds_max"][1] == 1.0
    assert parts["fin"]["bounds_min"][1] == 1.0
    assert parts["fin"]["anchored_to"] == "hull"


def test_the_api_accepts_mirror_of(client, finished_job):
    job = finished_job("gear")

    body = {
        "parts": [
            {"job_id": job, "name": "left_gear", "position": [2, 0, 0]},
            {"job_id": job, "name": "right_gear", "mirror_of": "left_gear"},
        ]
    }
    parts = {
        p["name"]: p for p in client.post("/assemble", json=body).json()["parts"]
    }

    assert parts["right_gear"]["center"] == [-2.0, 0.0, 0.0]


def test_the_api_reports_a_placement_cycle_as_a_bad_request(client, finished_job):
    job = finished_job("part")

    body = {
        "parts": [
            {"job_id": job, "name": "a", "anchor": {"to": "b"}},
            {"job_id": job, "name": "b", "anchor": {"to": "a"}},
        ]
    }
    response = client.post("/assemble", json=body)

    assert response.status_code == 400
    assert "placement cycle" in response.json()["detail"]


def test_the_api_reports_an_unknown_anchor_target_as_a_bad_request(
    client, finished_job
):
    job = finished_job("part")

    body = {"parts": [{"job_id": job, "name": "a", "anchor": {"to": "nope"}}]}
    response = client.post("/assemble", json=body)

    assert response.status_code == 400
    assert "unknown part 'nope'" in response.json()["detail"]


def test_the_api_can_still_place_by_absolute_position(client, finished_job):
    """The old shape has to keep working — this is an addition, not a swap."""
    job = finished_job("hull")

    body = {"parts": [{"job_id": job, "name": "hull", "position": [1, 2, 3]}]}
    parts = client.post("/assemble", json=body).json()["parts"]

    assert parts[0]["position"] == [1.0, 2.0, 3.0]
    assert parts[0]["anchored_to"] is None
