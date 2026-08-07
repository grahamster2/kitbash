"""Canonical orientation, on synthetic parts whose right way up is known.

Every part here is built in the frame it is supposed to end up in and then
*scrambled* by a known rotation, so "did orienting work" is a question with an
exact answer rather than something to eyeball. The scramble is deliberately not
axis-aligned: a generator hands back a part at whatever angle its reference
image was taken from, and a test that only ever turns things by 90 degrees
would pass on code that cannot do the real job.

The 180-degree cases get their own section. An oriented bounding box cannot
tell a nose from a tail, so those tests use parts that are asymmetric end to
end and assert on *which end* landed where — a wing whose root is at the tip's
end is exactly as broken as one standing on edge, and only this catches it.

No GPU, no network. Whether the result looks like an aeroplane is a render, and
the ones that were looked at are in docs/ORIENTATION.md.
"""
import numpy as np
import pytest
import trimesh

import assemble
import orient


# --------------------------------------------------------------------------
# synthetic parts, built in the frame they belong in
# --------------------------------------------------------------------------
def _hull(deform, subdivide=0.25):
    """A closed box pushed into shape by a per-vertex function.

    Deforming a box keeps the topology valid and the winding outward, which
    matters here: face normals are evidence, and an inside-out part would
    invert every one of them.
    """
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = np.array([deform(v) for v in box.vertices])
    mesh = trimesh.Trimesh(vertices=verts, faces=box.faces, process=False)
    verts, faces = trimesh.remesh.subdivide_to_size(
        mesh.vertices, mesh.faces, max_edge=subdivide
    )
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


@pytest.fixture
def wing():
    """A left wing: 4 m of span along x, root at +x, leading edge at +z.

    Tapered in plan (root chord 1.4, tip 0.4) and in section (thick at the
    leading edge, thin at the trailing edge), which is what makes both of its
    180-degree questions answerable.
    """

    def deform(v):
        x = np.sign(v[0]) * 2.0
        chord = 1.4 if v[0] > 0 else 0.4
        z = np.sign(v[2]) * chord / 2.0
        thickness = 0.25 if v[2] > 0 else 0.06
        return [x, np.sign(v[1]) * thickness / 2.0, z]

    return _hull(deform)


@pytest.fixture
def fin():
    """A tail fin: thin across x, 1.5 m tall, fat at the root and thin on top."""

    def deform(v):
        chord = 1.2 if v[1] < 0 else 0.5
        thickness = 0.18 if v[1] < 0 else 0.06
        return [np.sign(v[0]) * thickness / 2.0,
                np.sign(v[1]) * 0.75,
                np.sign(v[2]) * chord / 2.0]

    return _hull(deform)


@pytest.fixture
def slab():
    """A plate with no end to tell from the other: 3 x 0.1 x 1, symmetric."""
    return _hull(lambda v: [v[0] * 3.0, v[1] * 0.1, v[2] * 1.0])


@pytest.fixture
def blob():
    """The part with no answer: a wedge in a near-cubic box.

    Not a cube — a cube is *harmless* to turn, since every one of the 24
    orientations leaves it looking identical, and confidence is about
    consequences. This one's box says nothing and its contents say plenty, so
    getting it wrong shows.
    """
    return _hull(lambda v: [v[0], v[1], v[2] * (1.0 if v[1] < 0 else 0.05)],
                 subdivide=0.3)


def rotor(blades: int = 3, radius: float = 0.5, shaft: float = 1.6):
    """A spinner with blades round it, longer than it is wide.

    Deliberately the shape that defeats sorted extents: a real propeller is a
    disc, this reconstruction is a spindle, and only its axis of symmetry says
    which way it goes on the aeroplane.
    """
    parts = [trimesh.creation.cylinder(radius=0.16, height=shaft, sections=24)]
    for i in range(blades):
        blade = trimesh.creation.box(extents=(0.1, radius, 0.28))
        blade.apply_translation([0.0, radius / 2.0 + 0.15, 0.0])
        blade.apply_transform(
            trimesh.transformations.rotation_matrix(
                2.0 * np.pi * i / blades, [0, 0, 1]
            )
        )
        parts.append(blade)
    return trimesh.util.concatenate(parts)


def scramble(mesh, angles=(37.0, -114.0, 62.0)):
    """Put a part where a generator would have left it: nowhere in particular."""
    turned = mesh.copy()
    turned.apply_transform(
        trimesh.transformations.euler_matrix(*np.radians(angles), "sxyz")
    )
    turned.apply_translation([0.4, -1.2, 0.7])   # and off the origin, too
    return turned


def turn(mesh, spec):
    """Orient a part and hand back both the answer and the turned mesh."""
    result = orient.orient(mesh, spec)
    T = np.eye(4)
    T[:3, :3] = result.matrix
    turned = mesh.copy()
    turned.apply_transform(T)
    return result, turned


def end_extent(mesh, along: int, across: int, positive: bool, share: float = 0.2) -> float:
    """How wide the part is across one axis, at one end of another.

    The measurement the 180-degree tests turn on: a wing's root end is the one
    with the chord in it. Measured at the outer fifth rather than over half the
    part, because a tapered part is near its average everywhere in the middle
    and the ends are the whole question.
    """
    points = np.asarray(mesh.vertices)
    lo, hi = points[:, along].min(), points[:, along].max()
    cut = hi - share * (hi - lo) if positive else lo + share * (hi - lo)
    end = points[points[:, along] > cut] if positive else points[points[:, along] < cut]
    return float(np.ptp(end[:, across]))


# --- the box ----------------------------------------------------------------


def test_oriented_box_recovers_a_rotated_box():
    mesh = trimesh.creation.box(extents=(3.0, 1.0, 0.4))
    mesh.apply_transform(trimesh.transformations.euler_matrix(0.3, -0.9, 1.7))

    _, extents, _, _ = orient.oriented_box(mesh)

    assert sorted(extents, reverse=True) == pytest.approx([3.0, 1.0, 0.4], abs=0.05)


def test_oriented_box_prefers_a_frame_the_surfaces_agree_with():
    """A cruciform's smallest box is the one turned 45 degrees, and it is
    meaningless. This is the tail that made W_ALIGN necessary."""
    cross = trimesh.util.concatenate([
        trimesh.creation.box(extents=(2.0, 0.2, 0.6)),
        trimesh.creation.box(extents=(0.2, 2.0, 0.6)),
    ])

    axes, _, _, agreement = orient.oriented_box(cross)

    # Every box axis lands on a world axis, rather than halfway between two.
    assert np.abs(axes).max(axis=0) == pytest.approx([1, 1, 1], abs=0.02)
    assert agreement > 0.9


def test_spin_scores_find_the_axis_a_three_blade_rotor_turns_about():
    mesh = rotor(blades=3)
    axes, extents, centre, _ = orient.oriented_box(mesh)
    local = (orient._samples(mesh, 20000) - centre) @ axes / extents.max()

    scores = orient.spin_scores(local)

    # The shaft is the long axis of that box, and it is the only one that
    # survives a third of a turn.
    assert int(np.argmax(scores)) == int(np.argmax(extents))
    assert scores.max() > 2.0 * np.delete(scores, np.argmax(scores)).max()


# --- axis assignment --------------------------------------------------------


def test_a_scrambled_wing_comes_back_spanwise(wing):
    result, turned = turn(scramble(wing), "wing")

    span, thickness, chord = turned.extents
    assert span == pytest.approx(4.0, abs=0.05)
    assert chord == pytest.approx(1.4, abs=0.05)
    assert thickness == pytest.approx(0.25, abs=0.05)
    assert result.confidence > 0.9


def test_a_scrambled_fin_comes_back_upright(fin):
    _, turned = turn(scramble(fin), "fin")

    across, height, chord = turned.extents
    assert height == pytest.approx(1.5, abs=0.05)
    assert chord == pytest.approx(1.2, abs=0.05)
    assert across == pytest.approx(0.18, abs=0.05)


def test_extents_may_be_declared_without_a_role(wing):
    _, turned = turn(scramble(wing), {"extents": [4.4, 0.25, 1.4]})

    assert np.argmax(turned.extents) == 0
    assert np.argmin(turned.extents) == 1


def test_a_bare_list_is_target_extents(wing):
    _, turned = turn(scramble(wing), [4.4, 0.25, 1.4])

    assert np.argmax(turned.extents) == 0


def test_the_flat_axis_is_found_from_surface_normals_not_extents():
    """A plate with an upturned tip fence is taller than it is deep, so sorted
    extents stand it on its edge. Its faces still say which way it lies, and
    they outvote the box — the same shape as the generated tailplane, whose
    curled trailing edge makes its box a third deeper than the plate."""
    plate = trimesh.creation.box(extents=(4.0, 0.12, 0.7))
    fence = trimesh.creation.box(extents=(0.12, 0.9, 0.7))
    fence.apply_translation([-1.94, 0.39, 0.0])
    winged = trimesh.util.concatenate([plate, fence])
    assert winged.extents[1] > winged.extents[2]        # the trap

    _, turned = turn(scramble(winged), {"extents": [4.4, 0.25, 1.4],
                                        "taper": {"x": "-"}})

    assert turned.extents[1] > turned.extents[2]        # still lying flat


def test_a_spindly_rotor_is_placed_by_its_axis_not_its_extents():
    """Sorted extents would lay this propeller on its side: its shaft is longer
    than its disc is wide, so the longest axis is the one that must point
    forward, not sideways."""
    result, turned = turn(scramble(rotor()), "propeller")

    assert np.argmax(turned.extents) == 2       # the shaft, down the z axis
    assert turned.extents[2] == pytest.approx(1.6, abs=0.1)


# --- the 180-degree question ------------------------------------------------


def test_the_root_of_a_wing_ends_up_inboard(wing):
    """Nose-forward from nose-backward: the half with the chord in it is the
    root, and the role says a wing's root is at +x."""
    _, turned = turn(scramble(wing), "wing")

    root = end_extent(turned, along=0, across=2, positive=True)
    tip = end_extent(turned, along=0, across=2, positive=False)
    assert root > 2.0 * tip


def test_reversing_the_declared_taper_reverses_the_part(wing):
    """The same mesh, the other way round, because the caller said so."""
    _, turned = turn(scramble(wing), {"role": "wing", "taper": {"x": "+"}})

    root = end_extent(turned, along=0, across=2, positive=False)
    tip = end_extent(turned, along=0, across=2, positive=True)
    assert root > 2.0 * tip


def test_the_leading_edge_of_a_wing_ends_up_forward(wing):
    """The second 180: a wing is thick at the leading edge and thin at the
    trailing edge, and the leading edge belongs at +z."""
    _, turned = turn(scramble(wing), "wing")

    leading = end_extent(turned, along=2, across=1, positive=True)
    trailing = end_extent(turned, along=2, across=1, positive=False)
    assert leading > 2.0 * trailing


def test_a_fin_stands_on_its_root(fin):
    _, turned = turn(scramble(fin), "fin")

    base = end_extent(turned, along=1, across=2, positive=False)
    top = end_extent(turned, along=1, across=2, positive=True)
    assert base > 1.5 * top


def test_a_part_with_no_front_says_so_rather_than_inventing_one(slab):
    result, _ = turn(scramble(slab), {"extents": [3, 0.1, 1], "taper": {"x": "-"}})

    assert any("taper is unresolved" in note for note in result.notes)
    # But it is still confident, because a part that is symmetric end to end
    # looks the same either way round — an unanswerable question with no
    # consequence is not a reason to leave the part lying on its side.
    assert result.confidence > 0.9


def test_the_taper_cue_survives_a_part_being_upside_down(wing):
    """Rolled 180 about its span, a wing's box is identical and its mass is
    not. This is the case a box-only method gets wrong half the time."""
    flipped = wing.copy()
    flipped.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))

    _, turned = turn(flipped, "wing")

    root = end_extent(turned, along=0, across=2, positive=True)
    tip = end_extent(turned, along=0, across=2, positive=False)
    assert root > 2.0 * tip


# --- confidence -------------------------------------------------------------


def test_a_near_cubic_part_is_reported_as_uncertain(blob, wing):
    result = orient.orient(blob, "wing")

    # Against the part that really is a wing, on the same declaration. For
    # scale: the Bonanza parts in docs/ORIENTATION.md that came out right score
    # 0.86 and up, and the ones that are honestly ambiguous score 0.67-0.80.
    assert result.confidence < 0.7
    assert result.confidence < orient.orient(wing, "wing").confidence - 0.2
    assert result.notes


def test_an_unambiguous_part_is_reported_as_certain(wing):
    result = orient.orient(scramble(wing), "wing")

    assert result.confidence > 0.9
    assert result.notes == []


def test_a_symmetric_part_keeps_its_confidence_despite_the_ambiguity():
    """A wheel could be clocked any way round its axle and a cowl any way round
    its barrel. Both are ambiguous; neither is a risk, because every rival
    orientation is the same shape."""
    wheel = trimesh.creation.cylinder(radius=0.5, height=0.2, sections=48)

    result = orient.orient(scramble(wheel), "wheel")

    assert result.confidence > 0.9


def test_a_mesh_that_is_not_the_declared_shape_says_which_axis_is_wrong(blob):
    result = orient.orient(blob, {"extents": [4.0, 0.1, 1.0]})

    assert any("not the shape it was declared to be" in n for n in result.notes)


# --- the rotation itself ----------------------------------------------------


def test_the_rotation_never_mirrors_a_part(wing):
    """A reflection would turn a left wing into a right one, which is a
    different object, not a different orientation."""
    result = orient.orient(scramble(wing), "wing")

    assert np.linalg.det(result.matrix) == pytest.approx(1.0)
    assert result.matrix @ result.matrix.T == pytest.approx(np.eye(3), abs=1e-9)


def test_the_reported_euler_angles_reproduce_the_matrix(wing):
    result = orient.orient(scramble(wing), "wing")

    euler = trimesh.transformations.euler_matrix(*np.radians(result.rotation), "sxyz")
    assert euler[:3, :3] == pytest.approx(result.matrix, abs=1e-9)


def test_a_part_that_is_already_right_is_left_alone(wing):
    result = orient.orient(wing, "wing")

    assert result.degrees < 0.5      # the box search is numerical, not exact
    assert result.identity


def test_orienting_is_deterministic(wing):
    scrambled = scramble(wing)

    first = orient.orient(scrambled, "wing")
    second = orient.orient(scrambled, "wing")

    assert first.matrix == pytest.approx(second.matrix)
    assert first.confidence == pytest.approx(second.confidence)


def test_the_result_serialises_for_the_api(wing):
    payload = orient.orient(scramble(wing), "wing").as_dict()

    assert set(payload) == {"rotation", "confidence", "degrees", "extents",
                            "declared", "asymmetry", "notes"}
    assert len(payload["rotation"]) == 3


# --- the declaration --------------------------------------------------------


def test_an_unknown_role_is_rejected(wing):
    with pytest.raises(ValueError, match="unknown orient role"):
        orient.orient(wing, "spaceship")


def test_a_misspelled_key_is_rejected_rather_than_ignored(wing):
    with pytest.raises(ValueError, match="unknown key"):
        orient.orient(wing, {"role": "wing", "tapre": {"x": "-"}})


def test_a_taper_needs_a_direction(wing):
    with pytest.raises(ValueError, match="expected '\\+' or '-'"):
        orient.orient(wing, {"role": "wing", "taper": {"x": "outboard"}})


def test_a_taper_axis_must_be_an_axis(wing):
    with pytest.raises(ValueError, match="expected x, y or z"):
        orient.orient(wing, {"role": "wing", "taper": {"w": "-"}})


def test_extents_must_be_positive(wing):
    with pytest.raises(ValueError, match="positive"):
        orient.orient(wing, [4.0, 0.0, 1.0])


def test_a_declaration_needs_something_to_go_on(wing):
    with pytest.raises(ValueError, match="extents"):
        orient.orient(wing, {"taper": {"x": "-"}})


def test_roles_are_listed_for_a_caller():
    assert "wing" in orient.roles()
    assert orient.roles() == sorted(orient.ROLES)


# --------------------------------------------------------------------------
# assembly: orientation is resolved before anything measures anything
# --------------------------------------------------------------------------
def place(parts, tmp_path):
    result = assemble.assemble(parts, tmp_path / "scene.glb")
    return {p["name"]: p for p in result["parts"]}


@pytest.fixture
def wing_file(make_mesh, wing):
    return make_mesh(scramble(wing))


def test_assemble_applies_a_declared_orientation(wing_file, tmp_path):
    placed = place(
        [{"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing"}],
        tmp_path,
    )["left_wing"]

    assert placed["size"][0] == pytest.approx(4.0, abs=0.05)
    assert placed["orient"]["applied"] is True
    assert placed["orient"]["confidence"] > 0.9


def test_a_part_without_an_orient_key_reports_nothing(wing_file, tmp_path):
    placed = place(
        [{"name": "left_wing", "mesh_path": str(wing_file)}], tmp_path
    )["left_wing"]

    assert placed["orient"] is None


def test_anchors_measure_the_oriented_part(wing_file, make_mesh, wing, tmp_path):
    """The reason orientation is resolved first: anchoring to a part that is
    about to be turned upright would measure it lying down."""
    placed = place([
        {"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing",
         "position": [0, 0, 0]},
        {"name": "pod", "mesh_path": str(make_mesh(trimesh.creation.box())),
         "anchor": {"to": "left_wing", "align": {"x": "min"}, "my": {"x": "max"}}},
    ], tmp_path)

    # Face to face with the *oriented* wing's inboard end, and that end is 4 m
    # of span away from its tip rather than whatever the scrambled mesh spanned.
    assert placed["pod"]["bounds_max"][0] == pytest.approx(
        placed["left_wing"]["bounds_min"][0], abs=1e-6
    )
    assert placed["left_wing"]["size"][0] == pytest.approx(4.0, abs=0.05)


def test_a_caller_rotation_applies_on_top_of_the_orientation(wing_file, tmp_path):
    """`rotation` beside `orient` is dihedral, not a competing absolute."""
    placed = place([
        {"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing",
         "rotation": [0, 0, 90]},
    ], tmp_path)["left_wing"]

    assert placed["size"][1] == pytest.approx(4.0, abs=0.05)   # span now vertical


def test_orientation_is_applied_before_a_non_uniform_scale(wing_file, tmp_path):
    """So a scale is stated in the frame the caller was thinking in."""
    placed = place([
        {"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing",
         "scale": [2, 1, 1]},
    ], tmp_path)["left_wing"]

    assert placed["size"][0] == pytest.approx(8.0, abs=0.1)


def test_a_mirrored_part_inherits_the_orientation(wing_file, tmp_path):
    placed = place([
        {"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing",
         "position": [-2, 0, 0]},
        {"name": "right_wing", "mesh_path": str(wing_file),
         "mirror_of": "left_wing", "mirror": "x"},
    ], tmp_path)

    assert placed["right_wing"]["size"] == pytest.approx(
        placed["left_wing"]["size"], abs=1e-6
    )
    assert placed["right_wing"]["center"][0] == pytest.approx(
        -placed["left_wing"]["center"][0]
    )


def test_orienting_a_mirrored_part_is_an_error(wing_file, tmp_path):
    with pytest.raises(ValueError, match="cannot also set `orient`"):
        place([
            {"name": "left_wing", "mesh_path": str(wing_file), "orient": "wing"},
            {"name": "right_wing", "mesh_path": str(wing_file), "orient": "wing",
             "mirror_of": "left_wing", "mirror": "x"},
        ], tmp_path)


@pytest.mark.parametrize("nested", [False, True])
def test_a_confidence_floor_leaves_a_doubtful_part_alone(make_mesh, blob, tmp_path,
                                                         nested):
    """The floor may sit inside the declaration, as the API nests it, or beside
    it, which reads better in a hand-written part list."""
    spec = {"extents": [4, 0.2, 1]}
    part = {"name": "blob", "mesh_path": str(make_mesh(blob)), "orient": spec}
    if nested:
        spec["min_confidence"] = 0.9
    else:
        part["min_confidence"] = 0.9

    placed = place([part], tmp_path)["blob"]

    assert placed["orient"]["applied"] is False
    assert placed["orient"]["confidence"] < 0.9
    # Left exactly as generated, rather than turned on a coin flip.
    assert placed["size"] == pytest.approx(blob.extents, abs=1e-6)


def test_a_bad_declaration_names_the_part_it_came_from(wing_file, tmp_path):
    with pytest.raises(ValueError, match="left_wing: unknown orient role"):
        place(
            [{"name": "left_wing", "mesh_path": str(wing_file), "orient": "sausage"}],
            tmp_path,
        )


# --------------------------------------------------------------------------
# over HTTP
# --------------------------------------------------------------------------
def test_the_api_accepts_a_role_name_on_a_part(client, finished_job):
    job_id = finished_job()

    body = {"parts": [{"job_id": job_id, "name": "hull", "orient": "plate"}]}
    part = client.post("/assemble", json=body).json()["parts"][0]

    assert part["orient"]["applied"] is True
    assert 0.0 <= part["orient"]["confidence"] <= 1.0


def test_the_api_accepts_a_full_declaration(client, finished_job):
    job_id = finished_job()

    body = {"parts": [{
        "job_id": job_id, "name": "hull",
        "orient": {"extents": [3, 2, 1], "taper": {"z": "-"},
                   "min_confidence": 0.0},
    }]}
    response = client.post("/assemble", json=body)

    assert response.status_code == 200
    # The stubbed generator writes a 1x2x3 box, so the declaration turns it.
    assert response.json()["parts"][0]["size"] == pytest.approx([3, 2, 1], abs=1e-3)


def test_the_api_reports_a_part_it_declined_to_turn(client, finished_job):
    job_id = finished_job()

    body = {"parts": [{
        "job_id": job_id, "name": "hull",
        "orient": {"extents": [3, 2.9, 2.8], "min_confidence": 1.0},
    }]}
    part = client.post("/assemble", json=body).json()["parts"][0]

    assert part["orient"]["applied"] is False
    assert part["size"] == pytest.approx([1, 2, 3], abs=1e-6)  # untouched


def test_the_api_rejects_an_unknown_role(client, finished_job):
    job_id = finished_job()

    body = {"parts": [{"job_id": job_id, "name": "hull", "orient": "sausage"}]}
    response = client.post("/assemble", json=body)

    assert response.status_code == 400
    assert "unknown orient role" in response.json()["detail"]


def test_the_api_lists_the_orientation_roles(client):
    roles = client.get("/orient/roles").json()["roles"]

    assert "wing" in roles
    assert roles["wing"]["extents"] == orient.ROLES["wing"]["extents"]
