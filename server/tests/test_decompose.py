"""Plan format, prompt composition and execution. No network, no GPU.

The backend is injected everywhere, so nothing here can reach fal or CUDA; the
one test that uses the real ServerBackend only exercises its scripted path,
which is numpy.
"""
import pytest

import decompose
import jobs
import primitives


class FakeBackend:
    """Records what a real run would have spent money and VRAM on."""

    def __init__(self, fail_on: set[str] | None = None):
        self.images: list[tuple[str, int, str]] = []
        self.submissions: list[dict] = []
        self.scripted: list[str] = []
        self.fail_on = fail_on or set()
        self._n = 0

    def image(self, prompt, seed, image_size="square_hd"):
        if any(word in prompt for word in self.fail_on):
            raise RuntimeError("provider is down")
        self.images.append((prompt, seed, image_size))
        self._n += 1
        return f"img{self._n}"

    def submit(self, params, image_id):
        self._n += 1
        self.submissions.append({**params, "image_id": image_id})
        return {"id": f"job{self._n}", "status": jobs.QUEUED}

    def script(self, part):
        self._n += 1
        self.scripted.append(part.name)
        return {
            "id": f"prim{self._n}",
            "status": jobs.DONE,
            "result": {"faces": 1380, "mesh_path": "/nowhere/mesh.glb"},
        }


def plan(**overrides) -> dict:
    base = {
        "subject": "a wooden hand cart",
        "style": "weathered oak, black iron, soft overcast daylight",
        "parts": [{"name": "body", "prompt": "a plank box"}],
    }
    return {**base, **overrides}


# --- the format -------------------------------------------------------------


def test_a_plan_round_trips_through_plain_data():
    """The primary caller is an agent over MCP, so a plan is JSON both ways."""
    original = decompose.Plan.from_dict(decompose.BONANZA)

    assert decompose.Plan.from_dict(original.to_dict()).to_dict() == original.to_dict()


def test_a_plan_needs_a_subject_and_a_style():
    with pytest.raises(decompose.DecomposeError, match="'style'"):
        decompose.Plan.from_dict({"subject": "a cart", "style": "  ", "parts": []})


def test_an_unknown_plan_field_is_rejected_rather_than_ignored():
    with pytest.raises(decompose.DecomposeError, match="stlye"):
        decompose.Plan.from_dict(plan(stlye="oak"))


def test_an_unknown_part_field_names_the_part():
    with pytest.raises(decompose.DecomposeError, match="prompts"):
        decompose.Plan.from_dict(
            plan(parts=[{"name": "body", "prompts": "a plank box"}])
        )


def test_parts_default_to_generating():
    p = decompose.Plan.from_dict(plan())

    assert p.parts[0].mode == decompose.GENERATE


def test_the_defaults_are_the_measured_ones():
    """TRELLIS 2 untextured at 12k faces — see docs/QUALITY-COMPARISON.md."""
    p = decompose.Plan.from_dict(plan())

    assert (p.generator, p.textured, p.target_faces) == ("trellis2", False, 12000)


# --- prompt composition -----------------------------------------------------


def test_the_style_suffix_is_appended_to_every_part_prompt():
    p = decompose.Plan.from_dict(plan())

    assert decompose.part_prompt(p, p.parts[0]) == (
        "a plank box, weathered oak, black iron, soft overcast daylight"
    )


def test_the_composed_prompt_does_not_repeat_the_providers_framing():
    """imagegen.FRAMING already supplies isolation and background."""
    p = decompose.Plan.from_dict(plan())
    composed = decompose.part_prompt(p, p.parts[0])

    assert "plain flat white background" not in composed
    assert "single isolated" not in composed


def test_trailing_punctuation_does_not_produce_a_double_comma():
    p = decompose.Plan.from_dict(plan(style="weathered oak."))
    p.parts[0].prompt = "a plank box,"

    assert ",," not in decompose.part_prompt(p, p.parts[0])


def test_a_part_seed_overrides_the_plans_so_one_part_can_be_rerolled():
    p = decompose.Plan.from_dict(plan(seed=7))
    p.parts[0].seed = 99

    assert decompose.part_seed(p, p.parts[0]) == 99


def test_parts_share_the_plan_seed_by_default():
    p = decompose.Plan.from_dict(plan(seed=7))

    assert decompose.part_seed(p, p.parts[0]) == 7


# --- the leakage guard ------------------------------------------------------


def test_a_style_naming_the_whole_object_is_flagged():
    """Measured: "the same aircraft" in the suffix put a whole aeroplane behind
    a propeller that was asked for on its own."""
    p = decompose.Plan.from_dict({
        "subject": "a Beechcraft Bonanza light aircraft",
        "style": "parts of the same aircraft, white paint, studio light",
        "parts": [{"name": "prop", "prompt": "three blades on a hub"}],
    })

    assert decompose.style_leaks(p) == ["aircraft"]
    assert "object-completion prior" in decompose.validate(p)[0]


def test_a_material_only_style_is_not_flagged():
    p = decompose.Plan.from_dict(decompose.BONANZA)

    assert decompose.style_leaks(p) == []
    assert decompose.validate(p) == []


def test_incidental_shared_words_do_not_trip_the_guard():
    """"light aircraft" and "soft light" share a word that carries no identity."""
    p = decompose.Plan.from_dict({
        "subject": "a light aircraft",
        "style": "white paint, soft light from the left",
        "parts": [{"name": "prop", "prompt": "three blades"}],
    })

    assert decompose.style_leaks(p) == []


# --- real-world size --------------------------------------------------------
#
# The generator normalises every mesh to a unit box, so nothing downstream knows
# that a strut is smaller than a fuselage. `size_m` is the only place that fact
# can enter the system.


def test_a_size_may_be_a_single_longest_dimension():
    """The friendly form: a propeller is 2 m across and has no long axis."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "prop", "prompt": "three blades", "size_m": 2.0},
    ]))

    assert decompose.part_length(p.parts[0]) == 2.0
    assert decompose.part_extents(p.parts[0]) is None


def test_a_size_may_be_extents_and_the_longest_of_them_drives_the_scale():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": [4.4, 0.25, 1.4]},
    ]))

    assert decompose.part_extents(p.parts[0]) == [4.4, 0.25, 1.4]
    assert decompose.part_length(p.parts[0]) == 4.4


def test_one_unit_is_one_metre_unless_a_reference_part_says_otherwise():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": 4.4},
    ]))

    assert decompose.unit_metres(p) == 1.0
    assert decompose.part_scale(p, p.parts[0]) == 4.4


def test_a_scale_reference_makes_every_number_a_ratio_of_one_part():
    """"the wing is half a fuselage" is checkable by eye; "0.5238" is not."""
    p = decompose.Plan.from_dict(plan(scale_reference="fuselage", parts=[
        {"name": "fuselage", "prompt": "a shell", "size_m": 8.4},
        {"name": "wing", "prompt": "a panel", "size_m": 4.2},
    ]))

    assert decompose.scales(p) == {"fuselage": 1.0, "wing": 0.5}


def test_a_scripted_part_is_measured_rather_than_assumed_to_be_a_unit_box():
    """A primitive is built at whatever its params say. Declaring the same
    numbers as the params must therefore be a no-op, not a 3.2x blow-up."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "bed", "mode": "script", "kind": "crate",
         "params": {"width": 3.2, "height": 0.7, "depth": 1.8},
         "size_m": [3.2, 0.7, 1.8]},
    ]))

    assert decompose.part_scale(p, p.parts[0]) == 1.0


def test_a_primitive_drawn_at_unit_span_is_scaled_like_a_generated_part():
    """Which is what lets a plan draw its primitives at any convenient size —
    the Bonanza's gear is drawn 1.0 long, the cart's axle is drawn 2.1 m."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "strut", "mode": "script", "kind": "cylinder",
         "params": {"radius": 0.055, "height": 1.0}, "size_m": 0.9},
    ]))

    assert decompose.part_scale(p, p.parts[0]) == 0.9


def test_a_mirrored_part_gets_no_scale_because_it_inherits_one():
    """assemble.py takes a mirror's whole transform from its source. A scale of
    its own would be applied on top of the source's, squaring it."""
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())

    by_name = {p["name"]: p for p in result["assemble_request"]}
    assert "scale" not in by_name["right_wing"]
    assert by_name["left_wing"]["scale"] == round(4.4 / 8.4, 6)


def test_the_assemble_request_carries_the_scale_so_nobody_computes_it_by_hand():
    """The gap this closes: the Bonanza's twelve scales were supplied by a
    throwaway script, which meant every new object hit the same wall."""
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())

    by_name = {p["name"]: p for p in result["assemble_request"]}
    assert by_name["fuselage"]["scale"] == 1.0
    assert round(by_name["propeller"]["scale"], 4) == 0.2381
    assert round(by_name["left_gear_wheel"]["scale"], 4) == 0.0655


def test_a_part_with_no_size_gets_no_scale_rather_than_a_guessed_one():
    result = decompose.run(plan(), backend=FakeBackend())

    assert "scale" not in result["assemble_request"][0]


def test_an_orient_that_defers_to_the_declared_size_is_expanded():
    """orient.py takes a bare [x, y, z] of target extents, which is exactly what
    size_m already is — so the three numbers are written once."""
    result = decompose.run(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": [4.4, 0.25, 1.4],
         "placement": {"orient": True}},
    ]), backend=FakeBackend())

    assert result["assemble_request"][0]["orient"] == [4.4, 0.25, 1.4]


def test_orienting_to_a_single_length_says_why_it_cannot():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": 4.4,
         "placement": {"orient": True}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="ratios"):
        decompose.validate(p)


# --- validation -------------------------------------------------------------


def test_an_empty_plan_is_rejected():
    with pytest.raises(decompose.DecomposeError, match="at least one part"):
        decompose.validate(decompose.Plan.from_dict(plan(parts=[])))


def test_duplicate_part_names_are_rejected():
    """assemble.py resolves anchors by name; a duplicate makes them ambiguous."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel"},
        {"name": "wing", "prompt": "a panel"},
    ]))

    with pytest.raises(decompose.DecomposeError, match="duplicate part name"):
        decompose.validate(p)


def test_a_generated_part_without_a_prompt_is_rejected():
    p = decompose.Plan.from_dict(plan(parts=[{"name": "wing"}]))

    with pytest.raises(decompose.DecomposeError, match="no prompt"):
        decompose.validate(p)


def test_a_part_cannot_be_both_generated_and_scripted():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "box", "prompt": "a plank box", "kind": "crate"},
    ]))

    with pytest.raises(decompose.DecomposeError, match="pick one path"):
        decompose.validate(p)


def test_an_unknown_mode_lists_the_real_ones():
    p = decompose.Plan.from_dict(plan(parts=[{"name": "box", "mode": "conjure"}]))

    with pytest.raises(decompose.DecomposeError, match="expected one of"):
        decompose.validate(p)


def test_an_unknown_generator_is_rejected_before_anything_runs():
    p = decompose.Plan.from_dict(plan(generator="midjourney3d"))

    with pytest.raises(decompose.DecomposeError, match="midjourney3d"):
        decompose.validate(p)


def test_a_scripted_part_needs_a_kind():
    p = decompose.Plan.from_dict(plan(parts=[{"name": "box", "mode": "script"}]))

    with pytest.raises(decompose.DecomposeError, match="names no kind"):
        decompose.validate(p)


def test_a_misspelled_primitive_parameter_fails_the_plan_not_the_build():
    """A millisecond of validation against eight minutes of GPU time."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "box", "mode": "script", "kind": "crate", "params": {"widht": 2}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="widht"):
        decompose.validate(p)


def test_a_mirror_needs_a_source():
    p = decompose.Plan.from_dict(plan(parts=[{"name": "right_wing", "mode": "mirror"}]))

    with pytest.raises(decompose.DecomposeError, match="mirror_of"):
        decompose.validate(p)


def test_a_mirror_of_an_unknown_part_is_rejected():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "right_wing", "mode": "mirror",
         "placement": {"mirror_of": "lft_wing"}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="lft_wing"):
        decompose.validate(p)


def test_a_mirror_of_a_mirror_is_rejected():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel"},
        {"name": "b", "mode": "mirror", "placement": {"mirror_of": "wing"}},
        {"name": "c", "mode": "mirror", "placement": {"mirror_of": "b"}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="itself a mirror"):
        decompose.validate(p)


def test_anchoring_to_an_unknown_part_is_caught_here_not_in_assembly():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel",
         "placement": {"anchor": {"to": "fuselarge"}}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="fuselarge"):
        decompose.validate(p)


def test_anchoring_to_the_ground_is_allowed():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": 1.0,
         "placement": {"anchor": {"to": "ground"}}},
    ]))

    assert decompose.validate(p) == []


def test_a_size_in_millimetres_is_caught_in_a_millisecond_not_eight_minutes():
    """The mistake an LLM actually makes with a field called `size_m`."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": 4400},
    ]))

    with pytest.raises(decompose.DecomposeError, match="millimetres"):
        decompose.validate(p)


def test_a_size_of_zero_is_rejected():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": [4.4, 0, 1.4]},
    ]))

    with pytest.raises(decompose.DecomposeError, match="positive"):
        decompose.validate(p)


def test_a_two_number_size_is_rejected_rather_than_read_as_two_of_three():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": [4.4, 1.4]},
    ]))

    with pytest.raises(decompose.DecomposeError, match="2 number"):
        decompose.validate(p)


def test_a_mirror_cannot_state_its_own_size():
    """It takes its source's whole transform, so its own size is ignored."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "left_wing", "prompt": "a panel", "size_m": 4.4},
        {"name": "right_wing", "mode": "mirror", "size_m": 4.4,
         "placement": {"mirror_of": "left_wing"}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="silently ignored"):
        decompose.validate(p)


def test_stating_both_a_size_and_a_scale_is_a_contradiction():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "wing", "prompt": "a panel", "size_m": 4.4,
         "placement": {"scale": 0.5}},
    ]))

    with pytest.raises(decompose.DecomposeError, match="thrown away"):
        decompose.validate(p)


def test_a_scale_reference_naming_no_such_part_is_rejected():
    p = decompose.Plan.from_dict(plan(scale_reference="fuselarge"))

    with pytest.raises(decompose.DecomposeError, match="fuselarge"):
        decompose.validate(p)


def test_a_scale_reference_with_no_size_of_its_own_is_rejected():
    """It is the part every other size is divided by; it needs a size."""
    p = decompose.Plan.from_dict(plan(scale_reference="body"))

    with pytest.raises(decompose.DecomposeError, match="nothing to be the unit"):
        decompose.validate(p)


def test_a_generated_part_with_no_size_is_warned_about():
    """Not an error — half a plan still builds — but silence here is what made
    every new object need a hand-written scale script."""
    p = decompose.Plan.from_dict(plan())

    assert "unit box" in decompose.validate(p)[0]


def test_a_scripted_part_with_no_size_is_not_warned_about():
    """A primitive is already built at the size its params state."""
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "bed", "mode": "script", "kind": "crate"},
        {"name": "cloth", "prompt": "a rolled bundle", "size_m": 1.1},
    ]))

    assert decompose.validate(p) == []


def test_an_all_scripted_plan_says_it_needs_no_gpu():
    p = decompose.Plan.from_dict(plan(parts=[
        {"name": "box", "mode": "script", "kind": "crate"},
    ]))

    assert "no GPU" in decompose.validate(p)[0]


# --- execution --------------------------------------------------------------


def test_run_generates_one_image_per_generated_part():
    """The whole point: one reference per part, each from its own prompt."""
    backend = FakeBackend()

    result = decompose.run(decompose.BONANZA, backend=backend)

    generated = [p for p in result["parts"] if p["mode"] == decompose.GENERATE]
    assert len(backend.images) == len(generated) == 6
    assert len({prompt for prompt, _, _ in backend.images}) == 6


def test_every_generated_prompt_carries_the_shared_style():
    backend = FakeBackend()

    decompose.run(decompose.BONANZA, backend=backend)

    style = decompose.BONANZA["style"]
    assert all(prompt.endswith(style) for prompt, _, _ in backend.images)


def test_every_part_is_generated_at_the_same_seed():
    backend = FakeBackend()

    decompose.run(decompose.BONANZA, backend=backend)

    assert {seed for _, seed, _ in backend.images} == {decompose.BONANZA["seed"]}


def test_generated_parts_are_submitted_untextured_to_trellis2():
    """TRELLIS 2's texture path returns noise; geometry only is the working path."""
    backend = FakeBackend()

    decompose.run(decompose.BONANZA, backend=backend)

    assert all(s["generator"] == "trellis2" for s in backend.submissions)
    assert all(s["textured"] is False for s in backend.submissions)


def test_a_part_can_override_the_plans_face_budget():
    backend = FakeBackend()

    decompose.run(decompose.BONANZA, backend=backend)

    budgets = {s["part_name"]: s["target_faces"] for s in backend.submissions}
    assert budgets["fuselage"] == 16000  # the hero part
    assert budgets["propeller"] == 12000  # the plan default


def test_a_mirrored_part_reuses_its_sources_job_and_costs_nothing():
    backend = FakeBackend()

    result = decompose.run(decompose.BONANZA, backend=backend)

    by_name = {p["name"]: p for p in result["parts"]}
    assert by_name["right_wing"]["job_id"] == by_name["left_wing"]["job_id"]
    assert by_name["right_wing"]["image_id"] is None
    assert len(backend.images) == 6  # not 12


def test_a_mirror_listed_before_its_source_is_reported_not_guessed():
    backend = FakeBackend()

    result = decompose.run(plan(parts=[
        {"name": "right_wing", "mode": "mirror",
         "placement": {"mirror_of": "left_wing"}},
        {"name": "left_wing", "prompt": "a panel"},
    ]), backend=backend)

    assert result["failed"] == ["right_wing"]
    assert "listed after it" in result["parts"][0]["error"]


def test_scripted_parts_take_the_primitive_path():
    backend = FakeBackend()

    result = decompose.run(decompose.WOODEN_CART, backend=backend)

    assert backend.scripted == ["cart_bed", "axle", "left_wheel", "left_shaft"]
    assert len(backend.images) == 2  # the canvas and the lantern only
    assert not result["failed"]


def test_one_failed_part_does_not_abandon_the_others():
    """Seven good parts and a named failure is a build you can finish."""
    backend = FakeBackend(fail_on={"three-blade propeller"})

    result = decompose.run(decompose.BONANZA, backend=backend)

    assert result["failed"] == ["propeller"]
    assert len(result["job_ids"]) == 11  # 5 generated + 2 scripted + 4 mirrors
    assert "provider is down" in {p["name"]: p["error"] for p in result["parts"]}["propeller"]


def test_a_mirror_of_a_failed_part_fails_rather_than_pointing_at_nothing():
    backend = FakeBackend(fail_on={"blade-shaped panel"})

    result = decompose.run(decompose.BONANZA, backend=backend)

    assert result["failed"] == ["left_wing", "right_wing"]


def test_run_reports_progress_for_every_part():
    """A plan is many minutes of work; silence for five of them is a bug."""
    events = []
    decompose.run(decompose.BONANZA, backend=FakeBackend(), progress=events.append)

    assert {e["event"] for e in events} == {"image", "queued", "scripted", "mirrored"}
    assert [e["total"] for e in events] == [12] * len(events)
    assert max(e["index"] for e in events) == 12


def test_run_surfaces_validation_warnings_instead_of_only_logging_them():
    result = decompose.run(
        plan(subject="a wooden cart", style="wooden planks, soft light"),
        backend=FakeBackend(),
    )

    assert "wooden" in result["warnings"][0]


def test_run_rejects_an_invalid_plan_before_spending_anything():
    backend = FakeBackend()

    with pytest.raises(decompose.DecomposeError):
        decompose.run(plan(parts=[{"name": "a"}, {"name": "a"}]), backend=backend)

    assert backend.images == []


def test_the_assemble_request_carries_the_plans_placement_intent():
    """The plan already says where parts go; the caller should not retype it."""
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())

    by_name = {p["name"]: p for p in result["assemble_request"]}
    assert by_name["right_wing"]["mirror_of"] == "left_wing"
    assert by_name["propeller"]["anchor"]["to"] == "engine_cowl"
    assert all("job_id" in p and "name" in p for p in result["assemble_request"])


def test_the_assemble_request_carries_the_material_the_plan_stated():
    """assemble.py would otherwise re-derive it from the node name, which is how
    a `barrel` comes back metal."""
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())

    by_name = {p["name"]: p for p in result["assemble_request"]}
    assert by_name["propeller"]["material"] == "metal"
    assert by_name["left_gear_wheel"]["material"] == "rubber"

    cart = decompose.run(decompose.WOODEN_CART, backend=FakeBackend())
    # cart_bed states none, so primitives.py's "a crate is wood" survives.
    assert "material" not in {p["name"]: p for p in cart["assemble_request"]}["cart_bed"]


def test_failed_parts_are_left_out_of_the_assemble_request():
    result = decompose.run(
        decompose.BONANZA, backend=FakeBackend(fail_on={"three-blade propeller"})
    )

    assert "propeller" not in {p["name"] for p in result["assemble_request"]}


# --- status and wait --------------------------------------------------------


def test_status_reads_the_live_job_registry(monkeypatch):
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())
    monkeypatch.setattr(
        jobs, "get", lambda job_id: {"id": job_id, "status": jobs.DONE, "error": None}
    )

    state = decompose.status(result)

    assert state["done"] == state["total"] == 12
    assert state["finished"] is True


def test_status_keeps_a_part_that_never_started_as_an_error(monkeypatch):
    result = decompose.run(
        decompose.BONANZA, backend=FakeBackend(fail_on={"three-blade propeller"})
    )
    monkeypatch.setattr(
        jobs, "get", lambda job_id: {"id": job_id, "status": jobs.DONE, "error": None}
    )

    state = decompose.status(result)

    assert state["done"] == 11
    assert state["finished"] is True  # errors still count as settled


def test_wait_returns_as_soon_as_everything_has_settled(monkeypatch):
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())
    monkeypatch.setattr(
        jobs, "get", lambda job_id: {"id": job_id, "status": jobs.DONE, "error": None}
    )

    state = decompose.wait(result, timeout=1.0, poll=0.01)

    assert state["finished"] is True
    assert "timed_out" not in state


def test_wait_gives_up_rather_than_hanging(monkeypatch):
    result = decompose.run(decompose.BONANZA, backend=FakeBackend())
    monkeypatch.setattr(
        jobs, "get", lambda job_id: {"id": job_id, "status": jobs.QUEUED, "error": None}
    )

    state = decompose.wait(result, timeout=0.05, poll=0.01)

    assert state["timed_out"] is True


# --- the real backend, scripted half only -----------------------------------


def test_the_server_backend_files_a_primitive_as_a_finished_job(out_dir):
    """A scripted part has to be indistinguishable from a generated one, or
    /assemble and /export would have to branch on where it came from."""
    part = decompose.Part(name="supply_crate", mode=decompose.SCRIPT, kind="crate")

    job = decompose.ServerBackend(out_dir=out_dir).script(part)

    assert job["status"] == jobs.DONE
    assert job["type"] == "primitive"
    assert jobs.get(job["id"])["id"] == job["id"]
    assert (out_dir / job["id"] / "mesh.glb").exists()
    assert (out_dir / job["id"] / "job.json").exists()


def test_the_server_backend_does_not_need_an_image_provider_for_scripted_parts(out_dir):
    """A cart is all hardware; a missing FAL_KEY must not stop it building."""
    backend = decompose.ServerBackend(out_dir=out_dir)

    backend.script(decompose.Part(name="bed", mode=decompose.SCRIPT, kind="crate"))

    assert backend._provider is None


def test_a_scripted_part_keeps_the_material_its_kind_implies(out_dir):
    part = decompose.Part(name="bed", mode=decompose.SCRIPT, kind="crate")

    job = decompose.ServerBackend(out_dir=out_dir).script(part)

    assert job["result"]["material"] == "wood"


# --- the worked examples ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(decompose.EXAMPLES))
def test_every_example_validates(name):
    assert decompose.validate(decompose.example(name)) == []


@pytest.mark.parametrize("name", sorted(decompose.EXAMPLES))
def test_every_example_runs(name):
    result = decompose.run(decompose.example(name), backend=FakeBackend())

    assert not result["failed"]
    assert len(result["assemble_request"]) == len(decompose.EXAMPLES[name]["parts"])


def test_every_scripted_kind_in_the_examples_exists():
    for spec in decompose.EXAMPLES.values():
        for part in spec["parts"]:
            if part.get("mode") == decompose.SCRIPT:
                assert part["kind"] in primitives.KINDS


def test_the_cart_example_demonstrates_both_halves_of_the_routing_rule():
    """docs/PROCEDURAL.md: dimensioned hardware is scripted, soft irregular
    cargo is generated. The example is the argument, so it has to show both."""
    modes = {p.mode for p in decompose.example("wooden_cart").parts}

    assert modes == {decompose.SCRIPT, decompose.MIRROR, decompose.GENERATE}


def test_the_aircraft_example_names_the_viewpoint_on_its_thin_part():
    """Without it the wing reconstructed as two crossed slabs — a thin panel
    seen edge-on carries almost no depth information."""
    wing = decompose.example("bonanza").part("left_wing")

    assert decompose.THIN_PART_VIEW in wing.prompt


def test_the_aircraft_example_scripts_its_landing_gear():
    """Generated, the strut came back a spindle and the wheel not at all: two
    objects at very different scales in one frame. Two primitives are exact."""
    plan = decompose.example("bonanza")

    assert plan.part("left_gear_strut").mode == decompose.SCRIPT
    assert plan.part("left_gear_wheel").kind == "wheel"


@pytest.mark.parametrize("name", sorted(decompose.EXAMPLES))
def test_every_generated_part_in_the_examples_declares_its_real_size(name):
    """An example that does not is an example that teaches the parts-bin
    failure — every generated part comes back the same size as every other."""
    generated = [
        p for p in decompose.example(name).parts if p.mode == decompose.GENERATE
    ]

    assert generated and all(p.size_m is not None for p in generated)


def test_the_aircraft_examples_scales_are_the_ones_measured_on_the_real_build():
    """These eight numbers were supplied by hand to assemble the Bonanza that
    worked. Reproducing them from `size_m` is the whole point of the field: the
    hand step is what stood between "works for aircraft" and "works for
    anything"."""
    scales = decompose.scales(decompose.BONANZA)

    assert {k: round(v, 4) for k, v in scales.items()} == {
        "fuselage": 1.0, "left_wing": 0.5238, "tail_fin": 0.1786,
        "left_tailplane": 0.2024, "engine_cowl": 0.1667, "propeller": 0.2381,
        "left_gear_strut": 0.1071, "left_gear_wheel": 0.0655,
    }


def test_the_cart_example_needs_no_scaling_because_it_is_drawn_in_metres():
    """It has no scale_reference, so a unit is a metre — which is what
    primitives.py already builds in. Declaring the sizes must not move it."""
    scales = decompose.scales(decompose.WOODEN_CART)

    scripted = {
        p.name for p in decompose.example("wooden_cart").parts
        if p.mode == decompose.SCRIPT
    }
    assert all(scales[name] == 1.0 for name in scripted)
    assert scales["lantern"] == 0.35  # 35 cm, against a 3.2 m cart bed


def test_an_unknown_example_lists_the_real_ones():
    with pytest.raises(decompose.DecomposeError, match="bonanza"):
        decompose.example("spaceship")
