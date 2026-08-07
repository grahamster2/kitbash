"""The decision layer: what it recommends, what it costs, what it warns about.

No network, no GPU. `strategy.py` is pure arithmetic over the measurements and
`primitives.build`, so everything here runs on the laptop in milliseconds.

The interesting tests are the regressions against builds this project actually
measured. The showcase chest and the Bonanza are both documented end to end —
what they generated, what they scripted, how long they took — so they are used
here the way a golden file is used: if the recommender disagrees with what those
builds concluded, the recommender is wrong.
"""
import pytest

import decompose
import primitives
import strategy


# --- the three strategies on clear cases ------------------------------------


@pytest.mark.parametrize("subject, expected", [
    # One sculptural whole. Nine of ten organic subjects came back usable from
    # a single generation; a skull has no seams to split on.
    ("a skull", strategy.SINGLE),
    ("a dragon", strategy.SINGLE),
    ("a weathered boulder", strategy.SINGLE),
    ("a gargoyle statue", strategy.SINGLE),
    # Ornament that no formula writes, bolted to hardware that is nothing but
    # dimensions. Both halves of the routing rule in one object.
    ("a treasure chest", strategy.HYBRID),
    ("a Beechcraft Bonanza", strategy.HYBRID),
    ("a detailed castle gatehouse", strategy.HYBRID),
    # Dimensions, repetition, or an explicit triangle budget.
    ("a low-poly medieval house", strategy.SCRIPTED),
    ("a stone wall section", strategy.SCRIPTED),
    ("a greybox of a warehouse", strategy.SCRIPTED),
    ("a wooden crate", strategy.SCRIPTED),
])
def test_the_three_strategies_land_on_clear_cases(subject, expected):
    assert strategy.recommend({"subject": subject})["strategy"] == expected


def test_single_is_a_first_class_answer_not_a_fallback():
    """A skull recommends `single` with high confidence and cites the measurement.

    The failure mode this guards against is `single` becoming the thing that
    happens when nothing else matches. It has to be a positive verdict.
    """
    result = strategy.recommend({"subject": "a horned beast skull"})

    assert result["strategy"] == strategy.SINGLE
    assert result["confidence"]["level"] == "high"
    assert len(result["plan"]["parts"]) == 1
    assert result["cost"]["generations"] == 1
    assert any("0.867" in r["evidence"] for r in result["reasoning"])


def test_an_unrecognised_subject_says_so_rather_than_guessing_confidently():
    result = strategy.recommend({"subject": "a zorblatt manifold"})

    assert result["family"] == "unknown"
    assert result["confidence"]["level"] == "low"
    assert "matched no family" in result["confidence"]["why"]


def test_naming_parts_rules_single_out_whatever_the_subject_is():
    """The caller has decided the object has parts; that settles it."""
    result = strategy.recommend({
        "subject": "a dragon", "parts": ["body", "left wing", "horn"],
    })

    assert result["strategy"] != strategy.SINGLE
    assert result["scores"][strategy.SINGLE] is None  # ruled out, not outscored


def test_forbidding_generation_forces_scripted():
    result = strategy.recommend({"subject": "a dragon", "max_generations": 0})

    assert result["strategy"] == strategy.SCRIPTED
    assert result["cost"]["generations"] == 0
    assert result["cost"]["gpu_seconds"]["likely"] == 0.0


def test_forty_of_something_scripts_where_one_of_it_generates():
    """'If you need forty rocks, script them. If you need one hero rock, generate it.'"""
    one = strategy.recommend({"subject": "a weathered boulder"})
    forty = strategy.recommend({"subject": "a weathered boulder", "quantity": 40})

    assert one["strategy"] == strategy.SINGLE
    assert forty["strategy"] == strategy.SCRIPTED


# --- the reasoning carries its evidence -------------------------------------


def test_every_reason_names_a_measurement_and_a_source():
    result = strategy.recommend({"subject": "a treasure chest"})

    assert result["reasoning"]
    for reason in result["reasoning"]:
        assert reason["evidence"].strip()
        assert reason["source"].startswith("docs/")


def test_the_strategies_not_chosen_say_why_and_when_they_would_win():
    result = strategy.recommend({"subject": "a skull"})

    alternatives = {a["strategy"]: a for a in result["alternatives"]}
    assert set(alternatives) == {strategy.HYBRID, strategy.SCRIPTED}
    for alternative in alternatives.values():
        assert alternative["why_not"].strip()
        assert alternative["when_it_would_win"].strip()


def test_every_routed_part_reports_the_archetype_that_routed_it():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})

    routed = {r["part"]: r for r in result["routing"]}
    assert routed["left_wing"]["mode"] == strategy.SCRIPT
    assert routed["left_wing"]["archetype"] == "thin_panel"
    assert "crossed slabs" in routed["left_wing"]["why"]


# --- the archetype taxonomy -------------------------------------------------


@pytest.mark.parametrize("text, archetype, route", [
    ("an escutcheon", "ornament", strategy.GENERATE),
    ("a carved crest", "ornament", strategy.GENERATE),
    ("a dragon skull", "creature", strategy.GENERATE),
    ("a weathered boulder", "organic_mass", strategy.GENERATE),
    ("a marble statue", "sculpture", strategy.GENERATE),
    ("a gear strut", "strut", strategy.SCRIPT),
    ("an iron band", "band", strategy.SCRIPT),
    ("a wing panel", "thin_panel", strategy.SCRIPT),
    ("a front wheel", "wheel", strategy.SCRIPT),
    ("a floor plank", "plank", strategy.SCRIPT),
    ("a wall section", "wall", strategy.SCRIPT),
    ("a stone floor", "floor", strategy.SCRIPT),
    ("a flight of stairs", "stair", strategy.SCRIPT),
    ("a railing frame", "frame", strategy.SCRIPT),
    ("a fluted column", "column", strategy.SCRIPT),
    ("a shipping crate", "container", strategy.SCRIPT),
    ("a long table", "furniture", strategy.SCRIPT),
    ("the chest lid", "dimensioned_surface", strategy.SCRIPT),
    ("an oval porthole", "aperture", strategy.SCRIPT),
])
def test_the_taxonomy_encodes_the_measured_routing_verdicts(text, archetype, route):
    found = strategy.classify_part(text)

    assert found is not None, text
    assert found.name == archetype
    assert found.route == route


def test_the_taxonomy_has_no_verdict_rather_than_a_guess():
    """A part nobody recognises is the caller's problem, and saying so is honest."""
    assert strategy.classify_part("a bilge keel scarph") is None


def test_every_scripted_archetype_names_a_kind_the_catalogue_has():
    """The catalogue is under active development, so this is not a given.

    `Archetype.kind` resolves through `primitives.KINDS` at call time
    precisely so a preference for `window` degrades to `wall_panel` rather
    than exploding when the kind is not there yet.
    """
    for archetype in strategy.ARCHETYPES:
        if archetype.route != strategy.SCRIPT:
            continue
        assert archetype.kind in primitives.KINDS, archetype.name


def test_a_missing_preferred_kind_falls_through_to_the_next():
    assert strategy._kind("no_such_kind_at_all", "plank") == "plank"
    assert strategy._kind("no_such_kind_at_all") is None


def test_the_taxonomy_is_served_whole_so_an_agent_can_learn_it():
    taxonomy = strategy.taxonomy()

    assert {s["name"] for s in taxonomy["strategies"]} == set(strategy.STRATEGIES)
    assert taxonomy["routes"][strategy.GENERATE]
    assert taxonomy["routes"][strategy.SCRIPT]
    assert "skull" in taxonomy["do_not_decompose"]
    assert taxonomy["ceilings"]
    assert taxonomy["targets"]["default_is_an_assumption"]


# --- ground truth: the showcase chest ---------------------------------------


def test_the_chest_recommends_hybrid_as_the_showcase_built_it():
    result = strategy.recommend({"subject": "an ornate treasure chest"})

    assert result["strategy"] == strategy.HYBRID
    modes = {p["name"]: p["mode"] for p in result["plan"]["parts"]}
    # The four things the showcase generated were the carcass, the escutcheon,
    # a claw foot and the hoard; the ornament and the carcass are the two that
    # carry the argument.
    assert modes["carcass"] == strategy.GENERATE
    assert modes["escutcheon"] == strategy.GENERATE
    assert modes["foot_front_left"] == strategy.GENERATE


def test_the_chest_lid_is_scripted_which_is_the_expensive_lesson():
    """Twenty candidate references never produced a barrel vault. Do not retry it.

    This is the single sharpest regression in the file: the lid was *supposed*
    to be generated, and the only reason it is not is that somebody spent
    twenty reference images finding out.
    """
    result = strategy.recommend({"subject": "an ornate treasure chest"})

    lid = [p for p in result["plan"]["parts"] if p["name"].startswith("lid_")]
    assert lid, "the draft chest has no lid at all"
    assert all(p["mode"] == strategy.SCRIPT for p in lid)
    assert all(p["kind"] in primitives.KINDS for p in lid)
    assert any("twenty candidate" in (p.get("note") or "").lower() for p in lid)


def test_a_lid_routes_to_script_wherever_it_appears():
    archetype = strategy.classify_part("the coopered lid")

    assert archetype.name == "dimensioned_surface"
    assert archetype.route == strategy.SCRIPT
    assert "Twenty candidate reference images" in archetype.evidence


def test_the_chests_gpu_cost_matches_what_the_showcase_measured():
    """151 s of GPU for the four generations that shipped.

    Priced from a four-generation plan rather than from the draft, because the
    draft is a simplification of the shipped build — what is being regressed
    is the cost model, not the recipe.
    """
    plan = {
        "subject": "an ornate chest",
        "style": "aged oak, blackened iron, tarnished brass, soft warm light",
        "target_faces": 20000,
        "parts": [
            {"name": name, "prompt": f"a {name} shape", "size_m": 1.0}
            for name in ("carcass", "escutcheon", "claw_foot", "hoard")
        ],
    }

    costed = strategy.cost(plan)

    assert costed["generations"] == 4
    measured = 151.0
    assert abs(costed["gpu_seconds"]["likely"] - measured) / measured < 0.05
    assert costed["gpu_seconds"]["low"] <= measured <= costed["gpu_seconds"]["high"]


def test_the_chest_scene_clears_robloxs_per_meshpart_cap():
    """87 616 triangles across 88 parts, largest 19 694, zero over budget."""
    result = strategy.recommend({"subject": "an ornate treasure chest"})
    roblox = result["cost"]["roblox"]

    assert roblox["over_budget"] == []
    assert roblox["largest_part"] <= strategy.ROBLOX_TRIANGLE_CAP
    # And the fact that makes multi-part legal at all.
    assert roblox["welded_would_fail"] is True
    assert roblox["effective_budget"] > result["cost"]["triangles"]["total"]


# --- ground truth: the Beechcraft Bonanza -----------------------------------


def test_the_aircraft_recommends_hybrid():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza G36"})

    assert result["strategy"] == strategy.HYBRID
    assert result["family"] == "aircraft"


def test_the_aircraft_scripts_its_landing_gear():
    """Generated, the strut was a spindle and the wheel did not survive at all."""
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    modes = {p["name"]: p for p in result["plan"]["parts"]}

    assert modes["left_gear_strut"]["mode"] == strategy.SCRIPT
    assert modes["left_gear_strut"]["kind"] == "cylinder"
    assert modes["left_gear_wheel"]["mode"] == strategy.SCRIPT
    assert modes["left_gear_wheel"]["kind"] == "wheel"


def test_the_aircraft_scripts_its_flight_surfaces():
    """Thin flat panels are the generator's worst case, measured on this wing.

    This is where the recommender deliberately improves on `decompose.BONANZA`,
    which generates the wings: `tapered_panel` was added to primitives.py
    *because* an agent built an aircraft's flight surfaces from the library,
    and its documented worked example is this wing's dimensions exactly.
    """
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    modes = {p["name"]: p for p in result["plan"]["parts"]}

    for name in ("left_wing", "tail_fin", "left_tailplane"):
        assert modes[name]["mode"] == strategy.SCRIPT, name
        assert modes[name]["kind"] == "tapered_panel", name


def test_the_aircraft_mirrors_its_right_hand_side_rather_than_generating_it():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    modes = {p["name"]: p for p in result["plan"]["parts"]}

    for name in ("right_wing", "right_tailplane", "right_gear_strut",
                 "right_gear_wheel"):
        assert modes[name]["mode"] == strategy.MIRROR, name
    assert result["cost"]["parts"]["mirrored"] == 4


def test_the_aircraft_generates_only_the_sculptural_parts():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    generated = {p["name"] for p in result["plan"]["parts"]
                 if p["mode"] == strategy.GENERATE}

    assert generated == {"fuselage", "engine_cowl", "propeller"}
    # Half the shipped example's six, which is the point.
    assert result["cost"]["generations"] == 3


def test_the_bonanzas_wall_time_matches_what_the_first_run_measured():
    """22.3 s to queue seven parts, then 475 s for the seven meshes.

    Two different totals from the same build, and the cost model has to
    reproduce both: `gpu_seconds` is generation alone, `wall_seconds` is what
    the caller waited for.
    """
    plan = decompose.Plan.from_dict(decompose.BONANZA).to_dict()
    # The shipped example is the *second* version; the measured run was the
    # first, with seven generations rather than six.
    plan["parts"].append({
        "name": "left_main_gear", "prompt": "a telescopic strut and a wheel",
        "size_m": 0.9,
    })

    costed = strategy.cost(plan)

    assert costed["generations"] == 7
    measured = 22.3 + 475.0
    assert abs(costed["wall_seconds"]["likely"] - measured) / measured < 0.05
    assert costed["wall_seconds"]["low"] <= measured <= costed["wall_seconds"]["high"]


# --- the cost model ---------------------------------------------------------


def test_generation_cost_is_flat_in_subject_complexity():
    """A dragon and a barrel cost the same. That is the measured finding."""
    def gpu(subject):
        return strategy.cost({
            "subject": subject, "style": "soft neutral studio light",
            "parts": [{"name": "hero", "prompt": subject, "size_m": 1.0}],
        })["gpu_seconds"]["likely"]

    assert gpu("a winged reptilian beast") == gpu("a smooth ovoid pebble")


def test_a_solid_box_is_priced_on_the_expensive_curve():
    """Generation cost scales with occupied volume, so a crate is the worst case."""
    dragon = strategy.cost({
        "subject": "x", "style": "soft light",
        "parts": [{"name": "beast", "prompt": "a winged beast", "size_m": 1.0}],
    })
    crate = strategy.cost({
        "subject": "x", "style": "soft light",
        "parts": [{"name": "hero", "prompt": "a wooden crate", "size_m": 1.0}],
    })

    assert crate["gpu_seconds"]["likely"] > dragon["gpu_seconds"]["likely"] * 2
    assert crate["gpu_seconds"]["high"] == pytest.approx(151.2)
    assert crate["peak_vram_gib"] == pytest.approx(6.88)


def test_a_scripted_part_costs_milliseconds_and_no_gpu():
    costed = strategy.cost({
        "subject": "a cart", "style": "weathered oak, soft light",
        "parts": [{"name": "bed", "mode": "script", "kind": "crate", "params": {}}],
    })

    assert costed["generations"] == 0
    assert costed["gpu_seconds"]["likely"] == 0.0
    assert costed["wall_seconds"]["likely"] < 2.0  # only the assembly second
    assert costed["peak_vram_gib"] == 0.0


def test_a_scripted_parts_triangles_are_measured_not_estimated():
    """Built, at three milliseconds, so the number is exact and stays true.

    The docs' quoted counts have already drifted as the catalogue gained
    detail; building the part is the only estimate that cannot.
    """
    costed = strategy.cost({
        "subject": "a cart", "style": "weathered oak, soft light",
        "parts": [{"name": "bed", "mode": "script", "kind": "crate", "params": {}}],
    })

    assert costed["per_part"][0]["triangles"] == len(primitives.build("crate", {}).faces)


def test_a_mirror_is_free_but_still_occupies_a_meshpart():
    costed = strategy.cost({
        "subject": "a cart", "style": "weathered oak, soft light",
        "parts": [
            {"name": "left_wheel", "mode": "script", "kind": "wheel", "params": {}},
            {"name": "right_wheel", "mode": "mirror",
             "placement": {"mirror_of": "left_wheel", "mirror": "x"}},
        ],
    })

    left, right = costed["per_part"]
    assert right["wall_seconds"]["likely"] == 0.0
    assert right["gpu_seconds"]["likely"] == 0.0
    assert right["triangles"] == left["triangles"]  # it is still a MeshPart
    assert costed["parts"]["mirrored"] == 1


def test_the_cost_model_says_what_scripting_saved():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    savings = " ".join(result["cost"]["savings"])

    assert "scripted part(s) instead of generated" in savings
    assert "mirrored part(s)" in savings


def test_a_plan_with_no_savings_says_what_scripting_one_part_would_buy():
    savings = strategy.cost({
        "subject": "a skull", "style": "soft light",
        "parts": [{"name": "skull", "prompt": "a skull", "size_m": 0.3}],
    })["savings"]

    assert "3 ms" in " ".join(savings)


def test_a_cold_generator_is_priced_with_its_weight_load():
    plan = {
        "subject": "a skull", "style": "soft light", "generator": "hunyuan3d",
        "parts": [{"name": "skull", "prompt": "a skull", "size_m": 0.3}],
    }

    cold = strategy.cost(plan, model_resident=False)
    warm = strategy.cost(plan, model_resident=True)

    assert cold["wall_seconds"]["likely"] - warm["wall_seconds"]["likely"] == \
        pytest.approx(strategy.COLD_START_SECONDS)


def test_file_size_tracks_the_measured_decimation_ladder():
    """353 966 -> 6.2 MiB, 40 000 -> 704 KiB, 20 000 -> 352 KiB, 8 000 -> 141 KiB."""
    for faces, kib in ((353966, 6.2 * 1024), (40000, 704), (20000, 352), (8000, 141)):
        estimate = strategy.estimated_bytes(faces) / 1024
        assert abs(estimate - kib) / kib < 0.06, faces


def test_file_size_matches_the_scripted_parts_built_on_the_live_server():
    """Nine primitives built over HTTP against the reference box.

    (triangles, bytes) as the server reported them. The point of checking
    against real files rather than the docs is that the docs' counts have
    already drifted as the catalogue gained detail.
    """
    measured = [(60, 2076), (60, 2068), (60, 2072), (60, 2012), (192, 4452),
                (192, 4456), (872, 16852), (36, 1704), (180, 4284)]

    for faces, actual in measured:
        assert abs(strategy.estimated_bytes(faces) - actual) / actual < 0.05


def test_back_projected_colour_is_most_of_a_generated_parts_file():
    """Measured live: a 19 036-face skull came back 1 683 184 bytes.

    Geometry alone is 343 648 of that, so a caller sizing a download off the
    triangle count would be out by a factor of five.
    """
    assert strategy.estimated_bytes(19036) == pytest.approx(343648, rel=0.01)
    assert strategy.estimated_bytes(19036, coloured=True) == \
        pytest.approx(1683184, rel=0.02)


def test_a_generated_parts_size_includes_its_colour_and_a_scripted_one_does_not():
    costed = strategy.cost({
        "subject": "a cart", "style": "weathered oak, soft light",
        "parts": [
            {"name": "cargo", "prompt": "a bundle of cloth", "size_m": 1.0,
             "target_faces": 19036},
            {"name": "bed", "mode": "script", "kind": "crate", "params": {}},
        ],
    })
    generated, scripted = costed["per_part"]

    assert generated["estimated_bytes"] > strategy.COLOUR_ATLAS_BYTES
    assert scripted["estimated_bytes"] < 100_000


def test_generation_timing_covers_the_run_measured_live():
    """64.8 s of wall for one skull, whose own generation_seconds was 53.1."""
    costed = strategy.cost({
        "subject": "a skull", "style": "ivory with brown staining, soft light",
        "target_faces": 20000,
        "parts": [{"name": "skull", "prompt": "a horned skull", "size_m": 0.3}],
    })

    assert costed["wall_seconds"]["low"] <= 64.8 <= costed["wall_seconds"]["high"]
    assert costed["gpu_seconds"]["low"] <= 53.1 <= costed["gpu_seconds"]["high"]


def test_a_long_build_says_so_in_words_rather_than_only_in_a_number():
    slow = strategy.cost({
        "subject": "a fleet", "style": "soft light",
        "parts": [{"name": f"p{i}", "prompt": "a shape", "size_m": 1.0}
                  for i in range(15)],
    })

    assert "min" in slow["wall_human"]
    assert "worth checking" in slow["wall_human"]


def test_costing_rejects_a_plan_that_would_not_run():
    with pytest.raises(decompose.DecomposeError, match="widht"):
        strategy.cost({
            "subject": "a cart", "style": "oak",
            "parts": [{"name": "bed", "mode": "script", "kind": "crate",
                       "params": {"widht": 2.0}}],
        })


# --- ceilings ---------------------------------------------------------------


def test_an_asymmetric_surface_feature_warns_before_the_gpu_is_spent():
    """The generator returns a body of revolution. A hump comes back all the way round."""
    warnings = strategy.warnings_for({
        "subject": "an aircraft", "style": "glossy white paint, soft light",
        "parts": [{"name": "fuselage",
                   "prompt": "an elongated shell with a raised cabin hump",
                   "size_m": 8.0}],
    })
    hit = next(w for w in warnings if w["code"] == "body_of_revolution")

    assert hit["severity"] == "blocker"
    assert hit["part"] == "fuselage"
    assert "within 0.002" in hit["evidence"]


def test_an_aerofoil_section_warns():
    warnings = strategy.warnings_for({
        "subject": "an aircraft", "style": "glossy white paint, soft light",
        "parts": [{"name": "left_wing",
                   "prompt": "a panel with a thick rounded leading edge",
                   "size_m": 4.4}],
    })
    codes = {w["code"] for w in warnings}

    assert "aerofoil_section" in codes


def test_a_window_cutout_warns_that_it_is_below_the_noise_floor():
    warnings = strategy.warnings_for({
        "subject": "a shell", "style": "glossy white paint, soft light",
        "parts": [{"name": "fuselage",
                   "prompt": "a tapered shell with six oval portholes in a row",
                   "size_m": 8.0}],
    })
    hit = next(w for w in warnings if w["code"] == "cutouts_below_noise_floor")

    assert hit["severity"] == "blocker"
    assert "portholes" in hit["evidence"]


def test_a_bare_propeller_warns_that_it_returns_a_marine_one():
    warnings = strategy.warnings_for({
        "subject": "an aircraft", "style": "polished chrome, soft light",
        "parts": [{"name": "propeller", "prompt": "a propeller", "size_m": 2.0}],
    })

    assert "propeller_is_ambiguous" in {w["code"] for w in warnings}


def test_a_dense_organic_cluster_warns_to_switch_generator():
    warnings = strategy.warnings_for({
        "subject": "foliage", "style": "damp moss, soft overcast light",
        "parts": [{"name": "cap_cluster",
                   "prompt": "many thin stalks rising from a common base",
                   "size_m": 0.4}],
    })
    hit = next(w for w in warnings if w["code"] == "high_genus_cluster")

    assert "12 905 884" in hit["evidence"]
    assert "0.495" in hit["evidence"]


def test_the_plan_wide_ceilings_are_reported_whenever_anything_generates():
    warnings = strategy.warnings_for({
        "subject": "a skull", "style": "ivory with brown staining, soft light",
        "parts": [{"name": "skull", "prompt": "a horned skull", "size_m": 0.3}],
    })
    codes = {w["code"] for w in warnings}

    assert {"scale_is_destroyed", "colour_coverage", "not_watertight",
            "unpredictable_failure", "preview_double_darkens"} <= codes


def test_a_plan_that_generates_nothing_gets_no_generator_ceilings():
    warnings = strategy.warnings_for({
        "subject": "a cart", "style": "weathered oak, soft light",
        "parts": [{"name": "bed", "mode": "script", "kind": "crate", "params": {}}],
    })

    assert warnings == []


def test_blockers_are_listed_before_warnings_and_notes():
    result = strategy.recommend({"subject": "a Beechcraft Bonanza"})
    severities = [w["severity"] for w in result["warnings"]]
    order = {"blocker": 0, "warning": 1, "note": 2}

    assert severities == sorted(severities, key=lambda s: order[s])


def test_every_ceiling_carries_its_measurement_and_its_source():
    for ceiling in strategy.CEILINGS:
        assert ceiling.severity in ("blocker", "warning", "note")
        assert ceiling.evidence.strip()
        assert ceiling.source.startswith("docs/")


# --- delivery targets and triangle budgets ----------------------------------


def test_an_unstated_target_falls_back_to_roblox_and_says_it_assumed_that():
    """The bug being fixed: 20 000 is a Roblox cap that was a universal default."""
    result = strategy.recommend({"subject": "a dragon"})

    assert result["budget"]["target"] == "roblox"
    assert result["budget"]["target_assumed"] is True
    assert "target_assumed" in {w["code"] for w in result["warnings"]}
    assert "Say where this is going" in result["next_steps"][0]


def test_a_stated_target_is_not_flagged_as_an_assumption():
    result = strategy.recommend({"subject": "a dragon", "target": "game_realtime"})

    assert result["budget"]["target_assumed"] is False
    assert "target_assumed" not in {w["code"] for w in result["warnings"]}


@pytest.mark.parametrize("intent, target, faces", [
    ("a hero prop the player holds in Unreal, seen close up", "game_hero", 200000),
    ("distant background scenery for a mobile game", "scenery_lod", 500),
    ("a prop for a Unity game", "game_realtime", 12000),
    ("something for a Roblox obby", "roblox", 20000),
    ("a greybox I am going to delete", "blockout", 300),
    ("a film render in Blender", "offline_render", None),
    ("I want to 3D print this", "fabrication", None),
])
def test_the_budget_comes_from_stated_intent_in_prose(intent, target, faces):
    """The caller is an agent describing a need, not filling in a form."""
    budget = strategy.recommend({"subject": "a dragon", "intent": intent})["budget"]

    assert budget["target"] == target
    assert budget["faces_per_part"] == faces


def test_the_same_subject_is_the_same_prompt_at_different_budgets():
    """A background rock and a hero rock differ only in the number."""
    hero = strategy.recommend({"subject": "a weathered boulder", "detail": "hero",
                               "target": "game_hero"})
    far = strategy.recommend({"subject": "a weathered boulder",
                              "detail": "background", "target": "scenery_lod"})

    assert hero["plan"]["parts"][0]["prompt"] == far["plan"]["parts"][0]["prompt"]
    assert hero["cost"]["triangles"]["total"] > far["cost"]["triangles"]["total"] * 100


def test_a_no_decimation_target_asks_for_the_raw_mesh():
    result = strategy.recommend({"subject": "a dragon", "target": "offline_render"})

    assert result["budget"]["decimate"] is False
    # 0 at plan level is how a plan says "do not decimate": `_job_params` does
    # `part.target_faces or plan.target_faces`, so a part-level 0 is swallowed.
    assert result["plan"]["target_faces"] == 0
    assert all("target_faces" not in p for p in result["plan"]["parts"])
    assert "raw_mesh_wanted" in {w["code"] for w in result["warnings"]}
    assert any("use_raw" in step for step in result["next_steps"])


def test_a_no_decimation_target_is_priced_at_the_raw_face_count():
    result = strategy.recommend({"subject": "a dragon", "target": "offline_render"})
    part = result["cost"]["per_part"][0]

    assert part["decimated"] is False
    assert part["triangles"] == strategy.RAW_FACES_TYPICAL
    assert "0.48 M to 4.9 M" in part["triangles_note"]


def test_printing_warns_that_generated_geometry_is_never_watertight():
    result = strategy.recommend({"subject": "a dragon", "target": "fabrication"})
    hit = next(w for w in result["warnings"] if w["code"] == "watertight_needed")

    assert hit["severity"] == "blocker"
    assert "watertight: false" in hit["evidence"]


def test_asking_for_more_triangles_than_roblox_takes_is_a_blocker():
    result = strategy.recommend({
        "subject": "a dragon", "target": "roblox", "target_faces": 200000,
    })
    hit = next(w for w in result["warnings"] if w["code"] == "over_target_cap")

    assert hit["severity"] == "blocker"
    assert "rejected" in hit["message"]


def test_a_tiny_budget_on_an_ornate_subject_warns_that_it_defeats_the_point():
    result = strategy.recommend({
        "subject": "a gargoyle statue", "target": "scenery_lod",
    })
    hit = next(w for w in result["warnings"]
               if w["code"] == "budget_destroys_the_point")

    assert "deliberate low-poly" in hit["message"]
    assert "fine surface relief" in hit["evidence"]


def test_the_roblox_cap_is_labelled_as_robloxs_rather_than_universal():
    costed = strategy.recommend({"subject": "a dragon"})["cost"]

    assert "only if this is going to Roblox" in costed["roblox"]["applies"]


def test_the_targets_catalogue_is_served_for_discovery():
    catalogue = strategy.targets()
    names = {t["name"] for t in catalogue["targets"]}

    assert {"roblox", "game_realtime", "game_hero", "scenery_lod",
            "offline_render", "fabrication", "blockout"} <= names
    assert catalogue["default"] == "roblox"
    assert "universal" in catalogue["default_is_an_assumption"]
    assert set(catalogue["the_two_knobs"]) == {"target_faces", "resolution"}


# --- generation resolution --------------------------------------------------


def test_the_default_resolution_is_the_tier_measured_to_complete():
    result = strategy.recommend({"subject": "a dragon", "target": "roblox"})
    settings = result["budget"]["generated_parts"][0]["settings"]

    assert settings["pipeline_type"] == "512"


def test_a_high_budget_raises_the_resolution_and_says_what_it_risks():
    result = strategy.recommend({"subject": "a dragon", "target": "game_hero"})
    part = result["budget"]["generated_parts"][0]

    assert part["settings"]["pipeline_type"] == "1024_cascade"
    assert "21 minutes" in part["why"]
    assert "resolution_will_not_fit" in {w["code"] for w in result["warnings"]}


def test_a_solid_subject_stays_at_512_however_high_the_budget():
    """The crate at 1024_cascade was killed at 21 minutes at 96% of VRAM."""
    result = strategy.recommend({"subject": "a shipping crate",
                                 "target": "game_hero",
                                 "parts": ["crate carcass", "corner bracket"]})
    generated = [p for p in result["budget"]["generated_parts"]]

    assert generated
    assert all(p["settings"]["pipeline_type"] == "512" for p in generated)
    assert any("occupied volume" in p["why"] for p in generated)


def test_the_high_resolution_tier_is_priced_at_what_it_measured():
    result = strategy.recommend({"subject": "a dragon", "target": "game_hero"})

    assert result["cost"]["gpu_seconds"]["likely"] == pytest.approx(102.7)
    # The upper bound is the configured timeout, because the failure mode is a
    # stall that never terminates on its own.
    assert result["cost"]["gpu_seconds"]["high"] == pytest.approx(900.0)
    assert result["cost"]["peak_vram_gib"] == pytest.approx(9.69)


def test_the_two_knobs_are_explained_wherever_the_budget_is_reported():
    budget = strategy.recommend({"subject": "a dragon"})["budget"]

    assert "no budget recovers what was never generated" in budget["the_two_knobs"]
    assert "POST /jobs" in budget["generated_parts"][0]["how_to_apply"]


# --- LOD chains -------------------------------------------------------------


def test_an_lod_chain_descends_through_the_measured_ladder():
    result = strategy.recommend({"subject": "a dragon", "target": "roblox",
                                 "lod": True})

    assert result["budget"]["lod_chain"] == [20000, 8000, 2000]


def test_an_lod_chain_is_costed_as_cpu_seconds_rather_than_generations():
    result = strategy.recommend({"subject": "a dragon", "target": "roblox",
                                 "lod": True})
    part = result["budget"]["generated_parts"][0]

    assert result["cost"]["generations"] == 1
    assert "no GPU" in part["lod_cost"]
    assert "0.6 s" in part["lod_cost"]  # two extra levels at 0.3 s each


def test_no_lod_chain_is_offered_for_a_target_that_does_not_decimate():
    result = strategy.recommend({"subject": "a dragon", "target": "offline_render",
                                 "lod": True})

    assert result["budget"]["lod_chain"] is None


# --- low-poly is a parameter decision, not a decimation ---------------------


def test_low_poly_turns_the_decoration_off_rather_than_decimating_it():
    ornate = strategy.recommend({"subject": "a medieval house"})
    lean = strategy.recommend({"subject": "a low-poly medieval house"})

    assert lean["strategy"] == strategy.SCRIPTED
    assert lean["cost"]["triangles"]["total"] < \
        ornate["cost"]["triangles"]["total"] / 5


def test_lean_params_only_reduces_and_never_overrides_the_caller():
    params = strategy.lean_params("wall_panel", {"width": 4.0})
    full = len(primitives.build("wall_panel", {"width": 4.0}).faces)

    assert params["width"] == 4.0  # stated by the caller, left alone
    assert len(primitives.build("wall_panel", params).faces) < full


def test_lean_params_survives_a_kind_it_has_never_heard_of():
    assert strategy.lean_params("no_such_kind", {"a": 1}) == {"a": 1}


def test_every_lean_result_still_builds():
    """The reductions are tried and kept only if they work, so this must hold."""
    for kind in primitives.kinds():
        params = strategy.lean_params(kind, {})
        assert len(primitives.build(kind, params).faces) > 0, kind


# --- the draft plan ---------------------------------------------------------


@pytest.mark.parametrize("subject", [
    "a skull", "a dragon", "a treasure chest", "a Beechcraft Bonanza",
    "a low-poly medieval house", "a stone wall section",
    "a detailed castle gatehouse", "an ornate axe", "a wooden hand cart",
    "a long table", "a zorblatt manifold", "a gargoyle statue",
])
def test_every_draft_plan_validates_and_can_be_run_unchanged(subject):
    result = strategy.recommend({"subject": subject})

    plan = decompose.Plan.from_dict(result["plan"])
    decompose.validate(plan)  # raises if not
    assert plan.parts


def test_a_draft_plan_is_labelled_a_draft():
    result = strategy.recommend({"subject": "a dragon"})

    assert "DRAFT" in result["draft_disclaimer"]
    assert "size_m" in result["draft_disclaimer"]
    assert result["next_steps"]


def test_the_drafted_style_does_not_leak_the_subject_noun():
    """A suffix naming the object re-arms the completion prior through the text."""
    for subject in ("a stone wall section", "a treasure chest", "a wooden crate",
                    "a Beechcraft Bonanza", "an oak table"):
        plan = strategy.recommend({"subject": subject})["plan"]
        assert decompose.style_leaks(decompose.Plan.from_dict(plan)) == []


def test_a_scripted_strategy_carries_no_generated_parts():
    for subject in ("a low-poly medieval house", "a stone wall section",
                    "a greybox warehouse"):
        plan = strategy.recommend({"subject": subject})["plan"]
        assert all(p["mode"] != strategy.GENERATE for p in plan["parts"]), subject


def test_a_caller_supplied_part_list_is_routed_through_the_taxonomy():
    result = strategy.recommend({
        "subject": "a siege tower",
        "parts": ["carved lion crest", "wheel", "ladder", "wall panel", "axle"],
    })
    modes = {p["name"]: p["mode"] for p in result["plan"]["parts"]}

    assert modes["carved_lion_crest"] == strategy.GENERATE
    assert modes["wheel"] == strategy.SCRIPT
    assert modes["ladder"] == strategy.SCRIPT
    assert modes["wall_panel"] == strategy.SCRIPT
    assert modes["axle"] == strategy.SCRIPT


def test_an_unrecognised_part_name_generates_and_says_why():
    result = strategy.recommend({
        "subject": "a machine", "parts": ["frobnicator", "strut"],
    })
    parts = {p["name"]: p for p in result["plan"]["parts"]}

    assert parts["frobnicator"]["mode"] == strategy.GENERATE
    assert "No archetype matched" in parts["frobnicator"]["note"]


# --- the request ------------------------------------------------------------


def test_a_request_needs_a_subject():
    with pytest.raises(strategy.StrategyError, match="subject"):
        strategy.Request.from_dict({"target": "roblox"})


def test_an_unknown_request_field_is_rejected_rather_than_ignored():
    """Silently dropping `lowpoly` would return a hybrid plan and blame the caller."""
    with pytest.raises(strategy.StrategyError, match="lowpoly"):
        strategy.Request.from_dict({"subject": "a house", "lowpoly": True})


def test_a_misspelled_target_is_rejected_rather_than_falling_back_to_roblox():
    with pytest.raises(strategy.StrategyError, match="robox"):
        strategy.Request.from_dict({"subject": "a house", "target": "robox"})


def test_a_bad_detail_level_names_the_alternatives():
    with pytest.raises(strategy.StrategyError, match="background"):
        strategy.Request.from_dict({"subject": "a house", "detail": "medium"})


def test_a_negative_triangle_budget_is_rejected():
    with pytest.raises(strategy.StrategyError, match="target_faces"):
        strategy.Request.from_dict({"subject": "a house", "target_faces": -5})


def test_quantity_must_be_a_positive_whole_number():
    with pytest.raises(strategy.StrategyError, match="quantity"):
        strategy.Request.from_dict({"subject": "a rock", "quantity": 0})


# --- the endpoints ----------------------------------------------------------


def test_post_strategy_returns_a_recommendation_and_a_runnable_plan(client):
    response = client.post("/strategy", json={"subject": "a treasure chest"})

    assert response.status_code == 200
    body = response.json()
    assert body["strategy"] == strategy.HYBRID
    decompose.validate(decompose.Plan.from_dict(body["plan"]))


def test_post_strategy_reads_intent_prose(client):
    response = client.post("/strategy", json={
        "subject": "a dragon",
        "intent": "distant background scenery for a mobile game",
    })

    assert response.json()["budget"]["target"] == "scenery_lod"


def test_post_strategy_rejects_a_bad_request_with_a_message(client):
    response = client.post("/strategy", json={"subject": "a house",
                                              "target": "playstation"})

    assert response.status_code == 400
    assert "playstation" in response.json()["detail"]


def test_get_strategy_archetypes_teaches_the_taxonomy(client):
    body = client.get("/strategy/archetypes").json()

    assert body["archetypes"]
    assert body["no_llm"]
    assert {a["name"] for a in body["archetypes"]} >= {"ornament", "strut", "wheel"}


def test_get_strategy_targets_lists_the_budgets(client):
    body = client.get("/strategy/targets").json()

    assert {t["name"] for t in body["targets"]} >= {"roblox", "offline_render"}
    assert body["decimation_ladder"]["20000"]


def test_post_strategy_cost_prices_an_existing_plan(client):
    response = client.post("/strategy/cost", json={"plan": decompose.BONANZA})

    assert response.status_code == 200
    assert response.json()["generations"] == 6


def test_post_strategy_cost_rejects_an_invalid_plan(client):
    response = client.post("/strategy/cost", json={"plan": {"subject": "x"}})

    assert response.status_code == 400


def test_post_strategy_warnings_reports_the_ceilings_of_a_real_plan(client):
    response = client.post("/strategy/warnings", json=decompose.BONANZA)
    codes = {w["code"] for w in response.json()["warnings"]}

    # The shipped example's own prompts walk into three measured ceilings.
    assert {"aerofoil_section", "body_of_revolution",
            "propeller_is_ambiguous"} <= codes


def test_post_jobs_lod_builds_extra_levels_off_one_generation(client, finished_job):
    job_id = finished_job("hero", target_faces=8)

    response = client.post(f"/jobs/{job_id}/lod", json={"levels": [8, 6]})

    assert response.status_code == 200
    body = response.json()
    assert len(body["levels"]) == 2
    for level in body["levels"]:
        assert client.get(f"/jobs/{level['job_id']}").json()["status"] == "done"


def test_an_lod_level_is_an_ordinary_job_that_assembles(client, finished_job):
    job_id = finished_job("hero")
    lod_id = client.post(f"/jobs/{job_id}/lod",
                         json={"levels": [8]}).json()["levels"][0]["job_id"]

    response = client.post("/assemble", json={
        "parts": [{"job_id": lod_id, "name": "hero_lod0"}]
    })

    assert response.status_code == 200
    assert response.json()["part_count"] == 1


def test_asking_for_more_triangles_than_exist_returns_the_mesh_unchanged(
        client, finished_job):
    job_id = finished_job("hero")

    body = client.post(f"/jobs/{job_id}/lod", json={"levels": [999999]}).json()

    assert body["levels"][0]["faces"] == body["source_faces"]
    assert client.get(f"/jobs/{body['levels'][0]['job_id']}") \
        .json()["result"]["decimated_from"] is None


def test_lod_rejects_an_empty_or_nonsensical_ladder(client, finished_job):
    job_id = finished_job("hero")

    assert client.post(f"/jobs/{job_id}/lod", json={"levels": []}).status_code == 400
    assert client.post(f"/jobs/{job_id}/lod",
                       json={"levels": [2]}).status_code == 400


def test_lod_on_a_missing_job_is_a_404(client):
    assert client.post("/jobs/nope/lod", json={"levels": [8]}).status_code == 404
