"""Decide *how* to build an asset before spending anything on building it.

Every mode this project needs is built and measured. `pipeline.py` and
`trellis.py` generate, `primitives.py` scripts, `assemble.py` composes,
`decompose.py` executes a plan. What was missing is the step in front of all of
it: **nothing chose.** `decompose.run()` demands a plan that has already
committed to an approach, and there was no way to say "do not decompose this,
it is a skull" — which for a skull is the correct answer and splitting it would
ruin it.

This module chooses. It answers one question — `single`, `hybrid` or
`scripted` — and then shows its working:

    single    one generation, one part. A skull, a dragon, a boulder, a
              gargoyle. Measured: nine of ten organic subjects came back
              usable in 30-49 s (docs/WHAT-GENERATION-IS-FOR.md). Splitting
              one sculptural whole into parts buys nothing and costs a
              generation each.
    hybrid    generated sculptural parts plus scripted hardware. A plane, a
              building, a chest. Both halves of the routing rule in one
              object, which is what the showcase chest and the Bonanza both
              are.
    scripted  primitives only. Low-poly, greyboxing, modular kits, anything
              with a stated dimension. Measured: a scripted barrel is 840
              triangles in 3 ms against a generated one at 19 237 in 42 s.

`single` is a first-class answer and not a degenerate case. It is the right
answer for the largest single category of thing a Roblox developer asks for.

**There is no LLM call in here, deliberately** — the same constraint
`decompose.py` is written under, and for the same reason. The agent calling
this knows that a Bonanza is an aircraft, that a gatehouse has a portcullis,
and roughly how big a dragon is. The server knows none of that and cannot
learn it from a string. So this does the part the server *can* do better than
the agent: it carries every measured number in the repo, computes what a plan
will cost before it runs, and warns about the ceilings the generator was
measured to hit. The draft plan it returns is a **draft** — a strong starting
point stated in `decompose.Plan` format, to be revised by the caller with the
world knowledge the server does not have.

The one criticism this exists to answer: a build took forty minutes and nobody
saw the price until it had been paid. `cost()` computes wall time, GPU seconds,
triangles and generation count from a plan, in about a millisecond, and it is
computed for the draft before the draft is returned.
"""
import logging
import math
import re
from dataclasses import asdict, dataclass, field

import config
import decompose
import primitives

log = logging.getLogger("kitbash.strategy")

SINGLE = "single"
HYBRID = "hybrid"
SCRIPTED = "scripted"
STRATEGIES = (SINGLE, HYBRID, SCRIPTED)

# Routing verdicts an archetype can carry. These are decompose.py's modes, and
# deliberately the same strings, so a routing decision drops into a plan.
GENERATE = decompose.GENERATE
SCRIPT = decompose.SCRIPT
MIRROR = decompose.MIRROR


class StrategyError(Exception):
    pass


# --- measured costs ---------------------------------------------------------
#
# Everything below is a number somebody measured on the reference RTX 3080 and
# wrote down. Nothing here is an estimate from a datasheet. The sources are
# named on each block, because a cost model whose provenance is untraceable
# gets quietly wrong and nobody notices.
#
# Two different totals come out of this, and they are not the same thing:
#
#   gpu_seconds   pure generation. The chest showcase reports "GPU time 151 s
#                 for the four generations that shipped" — that number.
#   wall_seconds  what the caller waits: reference images, generation, colour,
#                 decimation, and the per-job overhead of a generator that
#                 reloads its weights from disk every run. The Bonanza reports
#                 22.3 s to queue plus 475 s to finish seven meshes — that one.
#
# Both are regression-tested against those two builds in tests/test_strategy.py.

# fal-ai/flux/schnell, one reference image. Measured: the Bonanza plan produced
# seven images and queued seven jobs in 22.3 s (docs/DECOMPOSITION.md).
IMAGE_SECONDS = 3.2

# Generation, per part, as (fastest, typical, slowest) measured seconds.
#
# The ten-subject organic set ran 30.0-49.4 s with a median of 36.5
# (docs/WHAT-GENERATION-IS-FOR.md); the four chest generations that shipped ran
# 32.5-42.3 (docs/SHOWCASE-CHEST.md). 38 s is the typical figure for both.
#
# The *solid* row is a different animal and that is the whole procedural
# argument: generation cost scales with occupied volume, so a crate — which
# fills its voxel grid — cost TRELLIS 2 151.2 s and Hunyuan3D 83.1 s, against
# 30-49 s for a dragon that is mostly empty space (docs/QUALITY-COMPARISON.md).
# The upper bound is 53.1 s, measured live on the reference box while writing
# docs/STRATEGY.md — a horned beast skull at 512, textured=false, 1 064 844 raw
# faces decimated to 19 036. Rounded out to 55 rather than fitted to it: one
# run above a ten-run band widens the band, it does not move the centre.
GENERATE_SECONDS = {
    "trellis2": (30.0, 38.0, 55.0),
    "hunyuan3d": (40.4, 41.0, 43.0),
}
GENERATE_SECONDS_SOLID = {
    "trellis2": (78.6, 110.0, 151.2),
    "hunyuan3d": (41.0, 62.0, 83.1),
}

# TRELLIS 2 at `1024_cascade`. One data point and one tripwire, and the gap
# between them is the honest shape of the risk: 102.7 s on the dragon (which is
# mostly empty space), and `config.TRELLIS_TIMEOUT` — 900 s — on anything
# whose occupied volume pushes it into the memory-thrash stall that never
# terminates on its own. The crate at these settings was killed at 21 minutes.
GENERATE_SECONDS_HIRES = {
    "trellis2": (102.7, 102.7, 900.0),
    "hunyuan3d": (40.4, 41.0, 43.0),
}
GENERATE_VRAM_GIB_HIRES = {"trellis2": 9.69, "hunyuan3d": 9.30}

# Everything a queued part costs that the generation timing does not: queue
# turnaround, the image handoff, polling latency, the mesh write. Backed out of
# the Bonanza's 475 s for seven meshes against ~38 s of generation each, and
# checked against a standalone live run that came in at 64.8 s of wall for a
# part whose own `generation_seconds` was 53.1.
#
# Not the subprocess spawn: TRELLIS 2 runs out of process with
# `keep_models_loaded=False`, and its reported `generation_seconds` already
# includes reloading its DiTs from disk.
JOB_OVERHEAD_SECONDS = {"trellis2": 21.0, "hunyuan3d": 2.0}

# First call only. Hunyuan3D loads ~70 s of weights and then stays resident;
# TRELLIS 2 never stays resident, which is what JOB_OVERHEAD_SECONDS above is.
COLD_START_SECONDS = 70.0

# Back-projecting the reference photograph onto the finished mesh
# (docs/TEXTURING.md). Measured 5.2-8.2 s across the ten-subject set. No VRAM —
# it is laptop CPU — but it is wall time the caller waits for.
COLOUR_SECONDS = (5.2, 5.7, 8.2)

# Raw meshes come back at 0.5-4.9 M faces; QEM decimation to 20 000 is 0.2-0.8 s
# (docs/DECIMATION.md, docs/WHAT-GENERATION-IS-FOR.md).
DECIMATE_SECONDS = (0.2, 0.5, 0.8)

# The thirteen scripted kinds build in 0.8-5.5 ms each (docs/PROCEDURAL.md).
# Rounded to the millisecond because at this scale the HTTP round trip
# dominates and pretending otherwise is false precision.
SCRIPT_SECONDS = 0.003

# A mirror is another part's mesh reflected. It costs nothing at all — no
# image, no GPU, no build — and it is the cheapest triangle in the system.
MIRROR_SECONDS = 0.0

# 88 parts assembled "well under a second" (docs/SHOWCASE-CHEST.md).
ASSEMBLE_SECONDS = 1.0

# Peak VRAM, device-wide, per generation. Flat in subject complexity: a dragon
# and a barrel cost the same. TRELLIS 2 climbs on solid subjects for the same
# occupied-volume reason its wall time does.
GENERATE_VRAM_GIB = {
    "trellis2": 3.93,
    "hunyuan3d": 9.30,
}
GENERATE_VRAM_GIB_SOLID = {
    "trellis2": 6.88,
    "hunyuan3d": 9.34,
}
USABLE_VRAM_GIB = 8.88  # 10 GB nominal, less what Windows holds

# Roblox's per-MeshPart triangle cap (docs/ROBLOX-EXPORT.md). Per *mesh*, not
# per file, which is the fact that makes multi-part assembly a budget multiplier
# rather than merely a convenience: the chest's carcass alone is 19 694, so a
# welded chest is rejected outright while the 88-part one has 88 budgets.
#
# **This is a Roblox number and nothing else's.** It has been used as a
# universal default throughout this project — `config.PRIMITIVE_MAX_FACES`,
# `config.TRELLIS_TARGET_FACES`, every generated part in every doc — without
# ever being marked as an assumption. It is one target's cap among several; see
# TARGETS below.
ROBLOX_TRIANGLE_CAP = 20000

# glTF size, from the measured decimation ladder: 353 966 faces -> 6.2 MiB,
# 40 000 -> 704 KiB, 20 000 -> 352 KiB, 8 000 -> 141 KiB, and the scripted
# catalogue's crate 1 380 -> 25.7 KiB, plank 60 -> 2.0 KiB. All of those land on
# ~18 bytes a face plus about a kilobyte of container, which is close enough to
# tell a caller whether a part is going to be a megabyte before it is built.
BYTES_PER_FACE = 18
GLB_OVERHEAD_BYTES = 1000

# What back-projected colour adds on top of the geometry. Measured once, live:
# a 19 036-face skull came back at 1 683 184 bytes against 343 648 of geometry.
# One data point, so it is a flat surcharge rather than a curve — the atlas is
# a fixed-resolution image and does not scale with the mesh, which is why a
# constant is the right shape even before there is a second measurement.
COLOUR_ATLAS_BYTES = 1_340_000

# What a generation returns before any decimation. "Hunyuan3D emits ~350k faces
# for a single object"; pre-decimation counts across the ten-subject set ran
# 0.48 M (ornate axe) to 4.9 M (ornate chest), with one 12.9 M outlier that was
# the mesh being born broken. Used to price a target that does not decimate.
RAW_FACES_TYPICAL = 350000

# Decimation itself, per level (docs/DECIMATION.md). This number is why an LOD
# chain is nearly free: the raw mesh is already on disk as `mesh_raw.glb`, so
# every extra level is a third of a second of CPU and no GPU at all.
DECIMATE_LEVEL_SECONDS = 0.3

# Past this, say so out loud in the summary rather than burying it in a number.
# The complaint this module exists to answer was a forty-minute build nobody
# priced first.
LONG_BUILD_SECONDS = 600.0


# --- delivery targets and triangle budgets ----------------------------------
#
# Two separate knobs, and conflating them is the mistake this table exists to
# stop:
#
#   target_faces  what the mesh is DECIMATED TO. Cheap, reversible, and a
#                 per-level 0.3 s — the raw mesh is kept, so you can have
#                 several.
#   resolution    how much detail EXISTS before any decimation, set by
#                 TRELLIS 2's `pipeline_type` or Hunyuan3D's
#                 `octree_resolution`. Expensive, one-shot, and **no budget
#                 recovers what was never generated.**
#
# The budgets below are per part. Which one applies is stated intent, not a
# constant: a background rock and a hero rock are the same prompt at different
# budgets, and only the caller knows which one this is.


@dataclass(frozen=True)
class Target:
    """Where the asset is going, and what that place can take."""

    name: str
    keywords: tuple[str, ...]
    summary: str
    # (lean, typical, generous) triangles per part. `detail` picks among them.
    faces: tuple[int, int, int]
    # A real engine limit that rejects the import, or None for a convention.
    hard_cap: int | None
    # False means ship `mesh_raw.glb` — do not decimate at all.
    decimate: bool
    # Whether decimation's known side effect actually costs anything here.
    watertight_matters: bool
    evidence: str
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


TARGETS: tuple[Target, ...] = (
    Target(
        "roblox",
        ("roblox", "rbx", "studio", "meshpart", "mesh part", "obby",
         "roblox game"),
        "Roblox Studio. The only target here with a cap that rejects the "
        "import, and it is per MeshPart rather than per file.",
        (8000, 20000, 20000), ROBLOX_TRIANGLE_CAP, True, False,
        "20 000 triangles per MeshPart, and it happens to coincide exactly "
        "with the measured decimation sweet spot: 18x smaller than raw with no "
        "visible loss. The chest's 88 parts total 87 616 triangles with zero "
        "over budget, because the cap is per mesh — a welded version of the "
        "same model is rejected outright.",
        "docs/ROBLOX-EXPORT.md",
    ),
    Target(
        "game_realtime",
        ("unity", "unreal", "godot", "game engine", "realtime", "real time",
         "real-time", "in game", "in-game", "gameplay", "playable", "fps",
         "third person"),
        "A general realtime engine — Unity, Unreal, Godot. No hard cap; the "
        "budget is a frame-time convention, not an importer rule.",
        (5000, 12000, 15000), None, True, False,
        "The measured degradation curve is the same one: at 8 000 the "
        "silhouette is perfect and fine relief is mush; at 20 000 embossed "
        "lettering is still legible. 5 000-15 000 is the standard prop band "
        "and is convention rather than a measurement made here.",
        "docs/DECIMATION.md",
    ),
    Target(
        "game_mobile",
        ("mobile", "phone", "android", "ios", "quest", "vr", "webgl", "web",
         "browser", "three js", "threejs", "low end", "low-end"),
        "Mobile, VR or the browser, where the budget is bandwidth and fill "
        "rate rather than the importer.",
        (1500, 4000, 8000), None, True, False,
        "8 000 faces is 141 KiB against 20 000's 352 KiB and 353 966's 6.2 "
        "MiB, and at 8 000 'the silhouette is perfect, fine relief lost' — "
        "which is the correct trade when the thing is 300 pixels tall.",
        "docs/DECIMATION.md",
    ),
    Target(
        "game_hero",
        ("hero asset", "close up", "close-up", "closeup", "first person",
         "held in hand", "in the hand", "showcase", "key art", "portrait",
         "inspect", "examined"),
        "A realtime asset the player will put their face against. High "
        "budget, and the one case where the generation resolution matters "
        "more than the decimation target.",
        (40000, 80000, 200000), None, True, False,
        "40 000 faces is 'indistinguishable from raw' and 704 KiB. Above that "
        "the rule is about detail type rather than count: '40 000+ only if the "
        "part carries text or fine relief that has to read up close'. Note "
        "that raw generated output is 0.5-4.9 M faces, so a 200 000 budget is "
        "usually still a decimation.",
        "docs/DECIMATION.md",
    ),
    Target(
        "scenery_lod",
        ("background", "scenery", "distant", "far away", "faraway", "lod",
         "filler", "set dressing", "backdrop", "skyline", "crowd", "impostor"),
        "Set dressing and distant LODs. Almost nothing but silhouette.",
        (500, 1500, 2000), None, True, False,
        "Quadric decimation spends its budget on curvature, so 'proportions "
        "and pose survive aggressive reduction' — at 8 000 the bird still "
        "looks fine while the embossed lettering on its sign is mush. Below a "
        "couple of thousand you keep the silhouette and nothing else, which "
        "is exactly what a distant asset needs.",
        "docs/DECIMATION.md",
    ),
    Target(
        "offline_render",
        ("offline", "render", "rendered", "film", "animation", "cinematic",
         "blender", "cycles", "ray trace", "path trace", "sculpt", "retopo",
         "retopology", "zbrush", "substance", "marmoset", "houdini", "maya"),
        "Anything not running at 60 fps: a render, a sculpt base, a "
        "retopology source. Do not decimate at all.",
        (0, 0, 0), None, False, False,
        "'The dense original is kept. Every job writes mesh_raw.glb alongside "
        "the decimated mesh.glb. It is the better input for retopology, and "
        "regenerating it would cost another 40 s.' Nothing downstream here "
        "cares about triangle count, and decimation is a lossy step taken for "
        "no reason.",
        "docs/DECIMATION.md",
    ),
    Target(
        "fabrication",
        ("3d print", "3d-print", "printed", "printing", "resin", "fdm", "sla",
         "cnc", "mill", "fabricate", "fabrication", "slicer", "stl"),
        "3D printing or machining. Triangle count is irrelevant and "
        "watertightness is everything — which is the one thing decimation "
        "destroys and the one thing generated meshes never had.",
        (0, 0, 0), None, False, True,
        "'Decimation breaks watertightness. Raw meshes come out watertight; "
        "decimated ones generally do not. Engines do not care. 3D printing "
        "and boolean operations do.' And separately: every mesh in the "
        "ten-subject organic set reports `watertight: false` even before "
        "decimation, so a generated part needs a repair pass regardless. Every "
        "scripted primitive is asserted watertight.",
        "docs/DECIMATION.md",
    ),
    Target(
        "blockout",
        ("greybox", "graybox", "grey box", "gray box", "blockout", "block out",
         "whitebox", "white box", "placeholder", "proxy geometry", "prototype"),
        "Blocking geometry: measured, disposable, and needed now. Script it "
        "rather than budgeting it.",
        (8, 300, 2000), None, True, False,
        "A `wedge` ramp is 8 triangles in 0.8 ms and is the most common "
        "blocking shape in a Roblox place; a `plank` is 60. The generator's "
        "floor is ~35 s a part whatever the subject, which is the wrong shape "
        "of cost for geometry you are going to delete.",
        "docs/PROCEDURAL.md",
    ),
    Target(
        "unspecified",
        (),
        "No target stated. Falls back to Roblox's numbers because Roblox is "
        "this project's primary consumer — but that is an assumption, not a "
        "measurement, and it is the assumption that has been silently baked "
        "into every asset here.",
        (8000, 20000, 20000), None, True, False,
        "20 000 is the measured decimation sweet spot and also Roblox's cap; "
        "the coincidence is why the two got conflated. Say where this is "
        "going and the budget changes — a film render wants no decimation at "
        "all and a distant LOD wants 1 500.",
        "docs/DECIMATION.md",
    ),
)

_TARGETS_BY_NAME = {t.name: t for t in TARGETS}
UNSPECIFIED_TARGET = _TARGETS_BY_NAME["unspecified"]

# How hard this particular part will be looked at, within its target's band.
DETAIL_LEVELS = ("background", "prop", "hero")
_DETAIL_INDEX = {"background": 0, "prop": 1, "hero": 2}

# Words that say how close the viewer gets, for callers describing intent in
# prose rather than filling in a form.
_DISTANCE_WORDS = {
    "hero": ("close up", "close-up", "closeup", "held", "in the hand",
             "first person", "inspect", "examined", "hero", "focal", "centrepiece",
             "centerpiece", "showcase"),
    "background": ("background", "distant", "far away", "faraway", "scenery",
                   "filler", "set dressing", "backdrop", "ambient", "clutter"),
}


# --- generation resolution --------------------------------------------------
#
# The knob that decides how much detail exists at all. It is one-shot and
# expensive, and its cost is set by *occupied volume* rather than by the
# subject's bounding box — which is the finding that makes a naive "turn it up
# for hero assets" rule dangerous.

# TRELLIS 2 pipeline tiers. `512` is what config.py ships and what completes.
TRELLIS_PIPELINES = {
    "512": {
        "pipeline_type": "512", "texture_size": 2048,
        "completes_on": "everything measured, 79-151 s at 3.58-6.88 GiB",
        "evidence": "The shipped default. Every result in this repo is at 512.",
    },
    "1024_cascade": {
        "pipeline_type": "1024_cascade", "texture_size": 4096,
        "completes_on": "spindly or hollow organic subjects only — 102.7 s at "
                        "5.03 GiB on the dragon",
        "evidence": "Run unchanged on a solid crate it was still inside the "
                    "generate stage after 21 minutes, having reached 9.69 GiB "
                    "device-wide — 96% of the usable budget — with power "
                    "dropping 314 W to 150 W while pinned at 100% utilisation. "
                    "That signature is memory pressure, and it never "
                    "terminates on its own. It was killed, not completed.",
    },
}

# Hunyuan3D's density knob. 256 is the shipped default and the ceiling worth
# trusting on a 10 GB card.
HUNYUAN_OCTREE = {
    256: "the default. Peaks near 7.63 GiB and returns ~350k faces.",
    128: "for a simple part, a low budget, or a smaller card. Less detail "
         "exists, and no decimation target recovers it.",
}


# --- the part archetype taxonomy --------------------------------------------
#
# The reusable core, and the thing worth keeping even if everything else here
# is rewritten. Each entry is a measured routing verdict: this kind of part goes
# to the GPU, or this kind of part is arithmetic. They were learned one wasted
# generation at a time — the Bonanza's landing gear, the chest's lid — and
# without them they get rediscovered the same way.
#
# Longest keyword wins, the same rule materials.py uses, so "wall_panel"
# resolves before a bare "wall" and "propeller blade" before "blade".


@dataclass(frozen=True)
class Archetype:
    """One measured routing verdict."""

    name: str
    route: str
    summary: str
    keywords: tuple[str, ...]
    evidence: str
    source: str
    # primitives.py kinds this becomes when the verdict is SCRIPT, best first.
    # A list rather than a name because the catalogue is under active
    # development: `window` and `roof` did not exist when the routing rule was
    # written and `wall_panel` had to stand in for both. Resolved by asking
    # primitives.KINDS at call time, the same way `GET /primitives` expects a
    # client to discover the library rather than be told about it.
    kinds: tuple[str, ...] = ()
    # Scripted parts are frequently identical pairs — two wings, four feet,
    # eight studs. Worth flagging, because the second one should be a mirror or
    # a reused job id rather than a second build.
    often_repeated: bool = False

    @property
    def kind(self) -> str | None:
        """The best kind for this archetype that the catalogue actually has."""
        return _kind(*self.kinds)

    def as_dict(self) -> dict:
        return {**asdict(self), "kind": self.kind,
                "kinds_available": [k for k in self.kinds if k in primitives.KINDS]}


def _kind(*candidates: str) -> str | None:
    """The first candidate the primitive catalogue actually offers.

    primitives.py is being extended while this runs, so naming a kind that may
    or may not exist has to be a preference rather than an assertion. A missing
    kind falls through to the next; nothing here fails because a library grew.
    """
    for name in candidates:
        if name in primitives.KINDS:
            return name
    return None


ARCHETYPES: tuple[Archetype, ...] = (
    # --- generate: detail volume nobody wants to write down ------------------
    Archetype(
        "ornament", GENERATE,
        "Applied surface relief with no formula behind it: escutcheons, "
        "scrollwork, cast faces, filigree, crests, finials, corbels.",
        ("ornament", "escutcheon", "lock plate", "lockplate", "scrollwork",
         "filigree", "carving", "carved", "relief", "engraving", "crest",
         "heraldry", "heraldic", "medallion", "boss", "finial", "corbel",
         "gargoyle", "knotwork", "frieze", "emblem"),
        "The chest's `lock_plate` is a snarling beast face in deep relief. One "
        "attempt, silhouette IoU 0.858, 13 941 faces in 38.4 s — the single "
        "most obviously generated thing on the model, and nothing in "
        "primitives.py comes near it.",
        "docs/SHOWCASE-CHEST.md",
    ),
    Archetype(
        "creature", GENERATE,
        "Anything anatomical: creatures, monsters, mounts, riders, limbs, "
        "heads, skulls.",
        ("creature", "monster", "beast", "dragon", "wyvern", "drake", "animal",
         "horse", "wolf", "bear", "bird", "fish", "serpent", "skull", "bone",
         "skeleton", "head", "claw", "paw", "talon", "wing membrane", "hide",
         "scales", "fur", "tentacle", "insect", "spider"),
        "Dragon: 30.9 s, IoU 0.830, four limbs planted, individually separated "
        "claw toes, scale relief as real geometry at 18 865 triangles. Horned "
        "beast skull: IoU 0.867, the best of the ten-subject set, with the "
        "orbital sockets reproduced as actual holes. There is no parametric "
        "`dragon` and there never will be.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Archetype(
        "organic_mass", GENERATE,
        "Irregular natural bulk: rock, terrain, bark, foliage, cloth, piles "
        "and hoards — things with no dimension anybody has to get right.",
        ("boulder", "rock", "stone outcrop", "cliff", "terrain", "stump",
         "log", "root", "bark", "tree", "bush", "shrub", "foliage", "vine",
         "moss", "coral", "mushroom", "fungus", "cloth", "canvas", "sack",
         "bundle", "drape", "hoard", "pile", "rubble", "debris", "ice",
         "crystal", "lava"),
        "Gnarled hollow stump: 39.1 s, fluted bark ridges, four splayed roots, "
        "a splintered break ring. Weathered boulder: IoU 0.870, the highest of "
        "the set. Caveat recorded honestly at the source — one hero rock, "
        "generate; forty rocks, script a noised icosphere.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Archetype(
        "sculpture", GENERATE,
        "A carved or cast figure that is irregular by nature: statues, idols, "
        "busts, totems, grave markers.",
        ("statue", "idol", "totem", "bust", "effigy", "figurine", "sculpture",
         "shrine", "icon", "gravestone", "headstone"),
        "Gargoyle statue: 36.9 s, folded ribbed wing membranes, wing claws "
        "hooked over the shoulders, weathered-limestone albedo with soot in "
        "the recesses. Nearly written off, because /preview double-darkens — "
        "judge these from an unlit render, not from the preview endpoint.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Archetype(
        "weapon_head", GENERATE,
        "The business end of a weapon, where the cast and etched detail lives.",
        ("axe head", "blade", "axehead", "sword blade", "spearhead", "mace "
         "head", "hammer head", "pommel", "hilt", "guard"),
        "Ornate axe: crescent blade with a scalloped edge, knotwork etched "
        "into the cheek, a beast head cast where the blade meets the haft. "
        "30.0 s, IoU 0.850. Its shaft, by contrast, came back lumpy and "
        "slightly banana-shaped — script the shaft, generate the head.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Archetype(
        "shell_body", GENERATE,
        "A large smooth hull or carcass whose surface carries irregular "
        "relief — a fuselage, a chest carcass, a boat hull, a cowling.",
        ("fuselage", "carcass", "hull", "cowl", "cowling", "nacelle", "canopy "
         "shell", "shell", "body"),
        "The chest carcass is carved staves, chamfers and mouldings: 42.3 s, "
        "IoU 0.863, 19 694 faces. The Bonanza fuselage generated too, but came "
        "back a smooth lozenge with its portholes and canopy gone — a shell "
        "generates, the holes in it do not.",
        "docs/SHOWCASE-CHEST.md",
    ),

    # --- script: dimensions somebody has to get right ------------------------
    Archetype(
        "strut", SCRIPT,
        "A rod, spindle, pipe, axle or leg. Round, straight, dimensioned.",
        ("strut", "axle", "rod", "spindle", "pipe", "tube", "shaft", "haft",
         "pole", "mast", "leg", "spar", "dowel", "hinge barrel", "bail",
         "peg", "stud", "bolt", "rivet"),
        "The Bonanza's gear strut generated as a featureless spindle, 7 984 "
        "triangles in 51 s. The same part as a `cylinder`: 192 triangles, "
        "1.5 ms, exactly 1.0 x 0.11 x 0.11. That is a factor of 34 000 on time "
        "and 41 on triangles, for a better part.",
        "docs/DECOMPOSITION.md",
        kinds=("cylinder",), often_repeated=True,
    ),
    Archetype(
        "band", SCRIPT,
        "Strap iron, hoops, bands, corner braces, trim runs — long thin "
        "constant-section metalwork.",
        ("band", "strap", "hoop", "brace", "bracket", "batten", "girdle",
         "binding", "reinforcement", "corner iron", "hasp", "hinge leaf",
         "moulding", "molding", "cornice", "skirting", "architrave"),
        "The chest's corner straps, rim and base bands, body straps, hinge "
        "leaves and hasp are 68 `plank` parts at 60 triangles each. The whole "
        "scripted library on that model is 2 268 triangles — less than a ninth "
        "of one generated carcass, for two-thirds of the parts.",
        "docs/SHOWCASE-CHEST.md",
        kinds=("moulding", "plank"), often_repeated=True,
    ),
    Archetype(
        "thin_panel", SCRIPT,
        "A flat or tapered plate: a wing, tailplane, fin, rotor blade, "
        "vehicle body side, door, shutter, sign.",
        ("wing", "tailplane", "stabiliser", "stabilizer", "elevator", "fin",
         "rudder", "aileron", "flap", "rotor blade", "vane", "panel", "plate",
         "sheet", "shutter", "door leaf", "sign", "board", "slab", "tile"),
        "The generator's worst case, measured directly. The Bonanza's wing "
        "came back as two crossed slabs; naming the viewpoint fixed it into "
        "one 11 892-triangle wing that the doc still calls chunky. "
        "`tapered_panel` exists because an agent built an aircraft's flight "
        "surfaces from primitives and could not say 'narrower at the far end'; "
        "measured at sixty stations across a 4.4 m semi-span its largest step "
        "is 0.0166 m against a two-plank fake's 0.70 m cliff. 60 triangles.",
        "docs/PROCEDURAL.md",
        kinds=("tapered_panel", "plank"), often_repeated=True,
    ),
    Archetype(
        "wheel", SCRIPT,
        "A wheel, tyre, disc, gear or pulley — nothing but dimensions.",
        ("wheel", "tyre", "tire", "disc", "disk", "pulley", "gear", "cog",
         "roller", "castor", "caster"),
        "Generated on the Bonanza, the wheel did not survive at all — a strut "
        "and a wheel at very different scales share one 1024 frame and the "
        "wheel gets a few dozen pixels of depth cue. `wheel`: 872 triangles, "
        "2.8 ms, with a hub and six spokes.",
        "docs/DECOMPOSITION.md",
        kinds=("wheel",), often_repeated=True,
    ),
    Archetype(
        "plank", SCRIPT,
        "A dimensioned board: decking, staves, floorboards, shelves, treads.",
        ("plank", "stave", "floorboard", "deck board", "shelf", "beam",
         "joist", "rafter", "timber", "lath", "tread", "sleeper"),
        "33 of the chest's 36 distinct primitive jobs are `plank`, at 60 "
        "triangles and about a millisecond each.",
        "docs/SHOWCASE-CHEST.md",
        kinds=("plank",), often_repeated=True,
    ),
    Archetype(
        "wall", SCRIPT,
        "A wall section, with or without a window or door aperture.",
        ("wall", "partition", "bulkhead", "facade", "parapet", "battlement",
         "curtain wall", "fence"),
        "`wall_panel` composes its aperture out of the four slabs around it "
        "rather than cutting a hole, so it needs no boolean engine and leaves "
        "no sliver triangles — and its facing, courses and joints are "
        "parameters rather than luck. A generated window is not an option at "
        "all: the fuselage's six portholes were in the reference and not in "
        "the mesh. Exact triangle counts are measured per parameter set by the "
        "cost model rather than quoted here, because the catalogue is growing.",
        "docs/PROCEDURAL.md",
        kinds=("wall_panel", "plank"), often_repeated=True,
    ),
    Archetype(
        "floor", SCRIPT,
        "A floor, platform, deck or slab — a flat plate with a stated size.",
        ("floor", "platform", "decking", "landing", "roadway", "walkway",
         "foundation", "base plate", "ceiling"),
        "A flat plate is the shape image-to-3D inflates and the shape a "
        "chamfered box gets exactly right, at 60 triangles.",
        "docs/PROCEDURAL.md",
        kinds=("plank",),
    ),
    Archetype(
        "stair", SCRIPT,
        "Steps, staircases and ramps.",
        ("stair", "stairs", "staircase", "step", "steps", "flight of steps",
         "flight of stairs", "ramp", "incline", "slope", "riser"),
        "`stairs` is 360 triangles and stacks boxes rather than extruding a "
        "silhouette, because `_prism` fan-triangulates and is convex-only. "
        "`wedge` is a ramp in 8 triangles and is the most common blocking "
        "shape in a Roblox place.",
        "docs/PROCEDURAL.md",
        kinds=("stairs", "wedge"),
    ),
    Archetype(
        "frame", SCRIPT,
        "Rails, ladders, railings, trusses, grilles, portcullises — repeated "
        "straight members.",
        ("frame", "rail", "railing", "ladder", "truss", "lattice", "grille",
         "grate", "portcullis", "scaffold", "trellis", "banister"),
        "`ladder` is two rails and N rungs at 1 656 triangles. Repeated "
        "straight members are the case a formula is *for*; the generator "
        "returns them as a mushy lump, which is what it did to the crate's "
        "corner brackets — 'mushy lumps with no defined border'.",
        "docs/PROCEDURAL.md",
        kinds=("railing", "ladder"), often_repeated=True,
    ),
    Archetype(
        "column", SCRIPT,
        "Columns, pillars, posts and chimneys — bodies of revolution with a "
        "base and a capital.",
        ("column", "pillar", "post", "chimney", "obelisk", "bollard",
         "buttress", "capital", "plinth", "pedestal"),
        "`column` is 1 600 triangles with a base, a capital and 20 flutes, and "
        "a fluted shaft comes out of a per-section radial multiplier with no "
        "boolean at all. Generated bodies of revolution also smear their "
        "back-projected colour, because one photograph covers a little under "
        "half of them.",
        "docs/PROCEDURAL.md",
        kinds=("column",), often_repeated=True,
    ),
    Archetype(
        "container", SCRIPT,
        "Crates, boxes, barrels, casks, chests-as-boxes, kegs.",
        ("crate", "box", "carton", "barrel", "cask", "keg", "tub", "bin",
         "chest box", "coffer"),
        "The single sharpest number in the repo. A crate is the most expensive "
        "thing either generator makes — 83.1 s and 9.27 GiB on Hunyuan3D, "
        "151.2 s on TRELLIS 2, and the recommended TRELLIS 2 settings were "
        "killed at 21 minutes at 96% of the VRAM budget — because generation "
        "cost scales with occupied volume and a box is solid. `POST /primitives "
        "{\"kind\": \"crate\"}` is 4.6 ms and 1 380 triangles. That is a factor "
        "of eighteen thousand.",
        "docs/PROCEDURAL.md",
        kinds=("crate",),
    ),
    Archetype(
        "furniture", SCRIPT,
        "Tables, benches, chairs, workbenches — legs and a top.",
        ("table", "bench", "chair", "stool", "desk", "workbench", "counter",
         "pew", "seat frame"),
        "`table` is 540 triangles and `bench` 600, both exact to the "
        "dimensions asked for and watertight. Against that, one generated "
        "crate spends 20 000 triangles and 83-151 s and still has rounded "
        "corners and undulating panels.",
        "docs/PROCEDURAL.md",
        kinds=("table", "bench"),
    ),
    Archetype(
        "dimensioned_surface", SCRIPT,
        "A surface whose whole identity is a shape you can state: a coopered "
        "lid, a hatch, a roof, a vault, an arch, a hull section.",
        ("lid", "cover", "hatch", "roof", "vault", "dome", "arch", "canopy "
         "frame", "awning", "gable", "spire", "turret roof"),
        "The chest lid was supposed to be generated. **Twenty candidate "
        "reference images across five prompt strategies** and the image model "
        "never once produced a long low barrel vault at a three-quarter angle; "
        "the one framing that worked produced a mesh that arched over its long "
        "axis, needing a (2.05, 0.76, 0.61) per-axis correction that would "
        "have smeared the staves into mush. Shipped as nine flat `plank` "
        "staves on an ellipse: 39 parts, 2 340 triangles, exact to the body's "
        "footprint.",
        "docs/SHOWCASE-CHEST.md",
        kinds=("roof", "plank"), often_repeated=True,
    ),
    Archetype(
        "aperture", SCRIPT,
        "A window, porthole, doorway, vent or seam — a part whose identity is "
        "a hole.",
        ("window", "porthole", "doorway", "opening", "vent", "louvre",
         "louver", "embrasure", "arrow slit", "murder hole", "panel seam",
         "door line"),
        "Below the reconstruction noise floor. The Bonanza fuselage's six oval "
        "portholes and its glass canopy were in the reference image and are "
        "not in the mesh: at the 512 pipeline and 16 000 faces they are below "
        "what comes back. Parts whose identity is a hole must be scripted, or "
        "the hole must be composed from the slabs around it.",
        "docs/DECOMPOSITION.md",
        kinds=("window", "panel_door", "wall_panel"),
    ),
)

# The verdict for a subject that is one sculptural whole. Not in ARCHETYPES
# because it is not a *part* routing — it is the decision not to have parts.
DO_NOT_DECOMPOSE = (
    "A subject that is one sculptural whole does not get decomposed. There are "
    "no seams in a skull, so splitting it invents them; each extra part is "
    "another 30-49 s generation, another unit-box scale to declare, and another "
    "join that can be wrong. Nine of ten single-generation organic subjects "
    "came back usable."
)

_ARCHETYPES_BY_NAME = {a.name: a for a in ARCHETYPES}


# --- ceilings ---------------------------------------------------------------
#
# Things the generator was measured to be unable to do. Every one of these cost
# real GPU time to find out, and every one of them is invisible until after the
# money is spent — which is the whole reason they are checked here, before it
# is. Severity `blocker` means the measured outcome was failure, not merely
# roughness.


@dataclass(frozen=True)
class Ceiling:
    code: str
    severity: str  # blocker | warning | note
    message: str
    evidence: str
    source: str
    # Words that, appearing in a *generated* part's name or prompt, mean the
    # part is about to walk into this. Empty means the ceiling is checked by
    # code rather than by keyword.
    triggers: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


CEILINGS: tuple[Ceiling, ...] = (
    Ceiling(
        "body_of_revolution", "blocker",
        "The generator returns a body of revolution, so an asymmetric surface "
        "feature has to be a separate part rather than a better prompt.",
        "Measured on an aircraft fuselage whose reference clearly showed a "
        "raised cabin hump with a raked windscreen. The result is vertically "
        "symmetric at every station, within 0.002 over the whole length — the "
        "hump came back as a bulge that goes all the way round. The prompt was "
        "not the problem. A re-prompt is worth at most one attempt for this "
        "class of defect.",
        "docs/MULTI-PART.md",
        ("hump", "bulge", "blister", "spoiler", "turret", "chimney", "dorsal",
         "cabin", "cockpit", "windscreen", "windshield", "raised", "asymmetric",
         "one side", "off-centre", "off-center", "conning tower"),
    ),
    Ceiling(
        "aerofoil_section", "blocker",
        "It will not produce an aerofoil section. A thick rounded leading edge "
        "running to a thin sharp trailing edge is below what the reconstruction "
        "represents; script a `tapered_panel` instead.",
        "The Bonanza's left wing was prompted with the leading and trailing "
        "edges described explicitly and came back as two crossed slabs. Adding "
        "the steep-viewpoint clause produced one solid wing at 11 892 "
        "triangles in 49 s, which docs/DECOMPOSITION.md calls chunky. The same "
        "wing as a `tapered_panel` is 60 triangles in 0.9 ms with a 0.0166 m "
        "worst chord step and a thickness taper for free.",
        "docs/DECOMPOSITION.md",
        ("aerofoil", "airfoil", "leading edge", "trailing edge", "wing "
         "section", "camber", "chord"),
    ),
    Ceiling(
        "thin_flat_panel", "blocker",
        "Thin flat panels are the generator's worst case, measured. A panel "
        "seen edge-on is close to information-free, and it comes back inflated, "
        "crossed or absent.",
        "The two failures in the Bonanza's first run were its two thinnest "
        "subjects — the wing (two crossed slabs) and the landing gear (a "
        "spindle with the wheel gone). Both references showed them near edge-on "
        "and small in frame. Hunyuan3D's sword scored 0.1% planar area; "
        "TRELLIS 2's 0.6%.",
        "docs/DECOMPOSITION.md",
        ("thin", "flat panel", "flat plate", "sheet metal", "plating",
         "veneer", "membrane panel"),
    ),
    Ceiling(
        "cutouts_below_noise_floor", "blocker",
        "Window cut-outs, panel seams and door lines are below the "
        "reconstruction noise floor. A part whose identity is a hole must be "
        "scripted, or the hole composed from the slabs around it.",
        "The Bonanza fuselage's six oval portholes and its rounded glass "
        "canopy are in the reference image and are not in the mesh — at 512 "
        "pipeline resolution and 16 000 faces they are below what comes back. "
        "The crate's corner brackets came back as 'mushy lumps with no defined "
        "border' on Hunyuan3D for the same reason.",
        "docs/DECOMPOSITION.md",
        ("porthole", "window", "seam", "panel line", "door line", "louvre",
         "louver", "grille", "slot", "keyhole", "perforation", "hole"),
    ),
    Ceiling(
        "propeller_is_ambiguous", "warning",
        "'Propeller' on its own returns a marine propeller — a squat hub with "
        "broad curved blades, not an aircraft's. Say how many blades, say they "
        "are long and narrow, and say what the hub is.",
        "The shipped Bonanza prompt is 'a three-blade propeller with a polished "
        "spinner hub' for exactly this reason, and the result is recorded as "
        "usable with blades softer than the reference. Recorded here rather "
        "than in DECOMPOSITION.md because it came out of the build log.",
        "docs/STRATEGY.md",
        ("propeller", "prop", "airscrew", "rotor",),
    ),
    Ceiling(
        "solid_box_cost", "warning",
        "Generation cost scales with occupied volume, so a solid box is the "
        "most expensive thing either generator makes — and the cheapest thing "
        "to write down.",
        "Crate: Hunyuan3D 83.1 s at 9.27 GiB; TRELLIS 2 151.2 s at 6.88 GiB; "
        "TRELLIS 2 at its own recommended `1024_cascade`/4096 settings was "
        "killed at 21 minutes having reached 9.69 GiB, 96% of the usable "
        "budget, with power dropping 314 W -> 150 W while pinned at 100% "
        "utilisation. That signature is memory pressure, not slowness. The "
        "scripted crate is 4.6 ms.",
        "docs/QUALITY-COMPARISON.md",
        ("crate", "box", "block", "cube", "brick", "slab", "container"),
    ),
    Ceiling(
        "high_genus_cluster", "warning",
        "Many thin separate stalks rising from a common base is the one "
        "subject class where TRELLIS 2 fails and Hunyuan3D does not. Route "
        "these to `hunyuan3d`.",
        "Mushroom cluster, identical reference image: TRELLIS 2 returned "
        "12 905 884 raw faces and IoU 0.495 — a shattered mess of flat black "
        "planes, not mushrooms — against 0.5-4.9 M raw faces for every other "
        "subject. Hunyuan3D built it correctly from the same image at IoU "
        "0.771. Six times the triangle budget bought 0.036 of IoU, so it is "
        "not decimation; the mesh was born broken. The cost is VRAM: 7.63 GiB "
        "against 2.66.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        ("cluster", "stalks", "bundle of stems", "thicket", "coral", "antlers",
         "branches", "roots", "chain", "netting", "rigging", "cables"),
    ),
    Ceiling(
        "hard_surface_generator", "note",
        "TRELLIS 2 wins hard-surface geometry 3-0 and does it in 3.58 GiB "
        "against 9.27; Hunyuan3D wins high-genus organic clusters and is "
        "robust to unprepared input. Neither wins texture — TRELLIS 2's baked "
        "colour is noise on this project's reference style.",
        "Crate, sword and truck side by side: TRELLIS 2 keeps rectangular "
        "things rectangular where Hunyuan3D rounds a sword's cross-guard into "
        "a dowel, and its corner brackets are distinct raised plates rather "
        "than lumps. Its texture came back as confetti on all three, confirmed "
        "in the baked 2048 atlas, and root-caused since to a broken K-quant "
        "GGUF of the 512 texture DiT.",
        "docs/QUALITY-COMPARISON.md",
    ),
    Ceiling(
        "colour_coverage", "warning",
        "Back-projection paints from one reference camera, so about half of "
        "every generated model has no real colour — and on a box it goes "
        "near-black.",
        "Coverage across the ten-subject set ran 0.171-0.817 with a median "
        "around 0.41. On the chest the lock plate came back at 0.015 and the "
        "hoard at 0.091; on a box, where one three-quarter view sees two of six "
        "faces and the interior is a dark cavity, the flood fill settles on a "
        "near-black dominant colour and three faces of four arrived burnt. The "
        "chest ships semantic PBR materials instead.",
        "docs/TEXTURING.md",
    ),
    Ceiling(
        "scale_is_destroyed", "blocker",
        "Every generated mesh comes back normalised to a unit box, so a "
        "generated part with no `size_m` assembles the same size as everything "
        "else. Nothing downstream can recover it.",
        "Measured on the six generated Bonanza parts, longest side: 0.9923, "
        "0.9989, 0.9997, 0.9936, 0.9989, 0.9921. An 8.4 m fuselage and a 0.9 m "
        "gear strut arrive identical. A plan that says the wing is 44 m "
        "validates, generates, assembles, and produces an aeroplane with a wing "
        "ten times too long.",
        "docs/DECOMPOSITION.md",
    ),
    Ceiling(
        "not_watertight", "note",
        "Generated meshes are not watertight, so `POST /hollow` will often "
        "refuse them and the boolean path is unavailable. Build interiors "
        "hollow by construction instead.",
        "Every mesh in the ten-subject set reports `watertight: false`. "
        "Carving the chest carcass at a 0.045 wall was correctly refused — the "
        "part is thinner than two walls everywhere; at 0.02 it produced a "
        "closed shell that removed 6.0% of the material and destroyed the UVs. "
        "The shipped liner is a `hollow_box`: 300 triangles in 9 ms including "
        "the HTTP round trip.",
        "docs/HOLLOW.md",
    ),
    Ceiling(
        "unpredictable_failure", "note",
        "About one subject in ten fails with no warning you can see beforehand. "
        "Route on the two numbers already in every job result.",
        "The mushroom cluster's prompt follows the same rules as the nine that "
        "worked. The only pre-generation signal was the raw face count. Treat "
        "`decimated_from` above ~8 M or `silhouette_iou` below 0.6 as an "
        "automatic reroll or a switch to Hunyuan3D — both numbers are in the "
        "job result and nothing currently reads them.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Ceiling(
        "preview_double_darkens", "note",
        "Judge generated parts from an unlit albedo render, not from "
        "`GET /jobs/{id}/preview`. It systematically under-reports quality.",
        "The gargoyle, stump, boulder and chest rendered as near-black "
        "silhouettes and four of the best assets in the set were initially "
        "written down as failures. It is not a UV bug — it is double-darkening: "
        "the back-projected albedo already contains the reference photograph's "
        "own shading, and `_shade()` multiplies it by a second lighting term "
        "with a floor of 0.186. The same thing will happen in Roblox.",
        "docs/WHAT-GENERATION-IS-FOR.md",
    ),
    Ceiling(
        "style_suffix_leak", "warning",
        "The shared style suffix must not name the whole object, or the "
        "object-completion prior comes back through the text encoder.",
        "'all parts of the same aircraft, general-aviation livery' rendered a "
        "propeller *attached to an aeroplane* and turned the landing gear into "
        "a wing with a wheel under it. Deleting the aircraft nouns and keeping "
        "only materials, palette and lighting fixed three of four immediately, "
        "at no cost to coherence: the suffix does essentially all the coherence "
        "work (max pairwise part-colour distance 240 without it, 142-169 with) "
        "and the seed does none of it.",
        "docs/DECOMPOSITION.md",
    ),
    Ceiling(
        "crops_do_not_work", "note",
        "Do not crop a reference of the whole object to get per-part "
        "references. Every part needs its own prompt.",
        "Tested on a photograph of a Beechcraft Bonanza, cropping tail, "
        "propeller, wing and cowl: every crop generated a complete aeroplane. "
        "Tightening the crops, keying the background to real alpha and padding "
        "generously all changed the result and none of them fixed it. Padding "
        "with opaque white reconstructed the padding as walls.",
        "docs/MULTI-PART.md",
    ),
)

_CEILINGS_BY_CODE = {c.code: c for c in CEILINGS}


# --- subject families -------------------------------------------------------
#
# What kind of thing is being asked for. This is the weakest part of the module
# and it is weak on purpose: a keyword table cannot know that a Beechcraft
# Bonanza is an aeroplane, and the caller can. The families cover the subjects
# this project has actually measured plus the obvious neighbours; everything
# else falls through to `unknown`, which recommends `single` and says so.


@dataclass(frozen=True)
class Family:
    name: str
    keywords: tuple[str, ...]
    strategy: str
    rationale: str
    evidence: str
    source: str
    # A plausible longest dimension in metres for a typical member. Present so
    # the draft plan carries a `size_m` that is the right order of magnitude
    # rather than none at all; always listed in `caller_must_revise`.
    size_m: float
    # Materials, palette and light for the shared style suffix. Never a noun
    # from the subject — see the style_suffix_leak ceiling.
    style: str
    single_defensible: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


FAMILIES: tuple[Family, ...] = (
    Family(
        "creature",
        ("dragon", "wyvern", "drake", "beast", "monster", "creature", "animal",
         "horse", "wolf", "bear", "lion", "bird", "eagle", "fish", "shark",
         "serpent", "snake", "spider", "insect", "golem", "troll", "ogre",
         "goblin", "zombie", "skeleton", "skull", "head", "mount", "pet",
         "critter", "alien", "demon"),
        SINGLE,
        "A creature is one sculptural whole with no seams to split on, and no "
        "parametric formula will ever write it.",
        "Dragon: 30.9 s, IoU 0.830, individually separated claw toes and scale "
        "relief as real geometry. Horned beast skull: IoU 0.867, the best of "
        "the ten-subject set. Both shipped unmodified.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        3.0,
        "matte organic surface, muted earth palette, soft neutral studio light "
        "from the upper left, photorealistic",
        single_defensible=True,
    ),
    Family(
        "natural_object",
        ("rock", "boulder", "stone block", "cliff", "outcrop", "stump", "log",
         "tree", "bush", "shrub", "plant", "mushroom", "coral", "crystal",
         "ice", "terrain", "rubble", "debris", "cloud", "fossil"),
        SINGLE,
        "Irregular natural bulk has no dimension anybody has to get right, "
        "which is exactly the half of the routing rule the generator owns.",
        "Weathered boulder IoU 0.870 — the highest of the set — with strata, "
        "sheared facets and lichen in the cracks. Gnarled hollow stump 39.1 s "
        "with fluted bark ridges and four splayed roots. The honest caveat is "
        "recorded at the source: one hero rock, generate; forty rocks, script.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        1.5,
        "weathered natural surface, desaturated grey and ochre palette, soft "
        "overcast daylight, photorealistic, matte finish",
        single_defensible=True,
    ),
    Family(
        "statue",
        ("statue", "gargoyle", "idol", "totem", "bust", "effigy", "figurine",
         "sculpture", "monument", "gravestone", "headstone", "shrine"),
        SINGLE,
        "A statue is irregular by nature and its plinth is the only part with "
        "a dimension — which is not enough decomposition to be worth a second "
        "generation.",
        "Gargoyle statue: 36.9 s, folded ribbed wing membranes, wing claws "
        "hooked over the shoulders, clawed hands gripping the plinth edge.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        2.0,
        "weathered limestone, soot in the recesses, desaturated grey palette, "
        "soft overcast daylight, photorealistic",
        single_defensible=True,
    ),
    Family(
        "ornate_prop",
        ("treasure chest", "chest", "casket", "coffer", "urn", "chalice",
         "goblet", "altar", "throne", "sarcophagus", "reliquary", "brazier",
         "cauldron", "lantern", "chandelier", "mirror frame", "clock",
         "music box", "jewellery box", "jewelry box"),
        HYBRID,
        "The ornament is the entire value of the prop and no formula supplies "
        "it — but everything holding the ornament is dimensioned hardware, and "
        "the two halves are measurably better built by different means.",
        "The chest: four generated meshes (carcass, escutcheon, claw foot, "
        "hoard) at 151 s of GPU total, against 80 scripted parts from 36 "
        "primitives at 2 268 triangles and about a millisecond each. The lid "
        "was supposed to be generated and was not — see the "
        "`dimensioned_surface` archetype.",
        "docs/SHOWCASE-CHEST.md",
        1.1,
        "aged oak with visible grain, blackened wrought iron, tarnished brass, "
        "soft warm directional light from the upper left, photorealistic",
        single_defensible=True,
    ),
    Family(
        "weapon",
        ("sword", "axe", "hammer", "mace", "spear", "halberd", "staff", "wand",
         "bow", "crossbow", "dagger", "knife", "scythe", "flail", "club",
         "pickaxe", "shield"),
        HYBRID,
        "Generate the head, script the shaft. The routing rule visible in a "
        "single asset.",
        "Ornate axe: crescent blade with a scalloped edge, knotwork etched "
        "into the cheek, a beast head cast at the blade root — 30.0 s, IoU "
        "0.850. And: 'the shaft is the weak part: it is lumpy and slightly "
        "banana-shaped where a real haft is straight'. A `cylinder` haft is "
        "192 triangles in 1.5 ms and is straight by construction.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        1.0,
        "forged steel with a scuffed edge, dark oiled hardwood, wrapped leather "
        "cord, soft neutral studio light, photorealistic",
        single_defensible=True,
    ),
    Family(
        "aircraft",
        ("aircraft", "aeroplane", "airplane", "plane", "jet", "biplane",
         "glider", "helicopter", "airship", "zeppelin", "rocket", "spacecraft",
         "spaceship", "shuttle", "drone", "bonanza", "cessna", "spitfire",
         "fighter", "bomber", "airliner"),
        HYBRID,
        "The worst case for image-to-3D and the best case for a parametric "
        "script: hard-surface, engineered, bilaterally symmetric, assembled "
        "from thin flat panels, and so familiar that every millimetre of error "
        "is legible. The body generates; every flight surface and every piece "
        "of gear is arithmetic.",
        "The Bonanza is twelve parts and six generations, and its two failures "
        "were its two thinnest subjects. `tapered_panel` was added to "
        "primitives.py *because* an agent built an aircraft's flight surfaces "
        "from this library; its documented worked example is a 4.4 m semi-span "
        "wing at 60 triangles, which is the Bonanza's wing exactly.",
        "docs/DECOMPOSITION.md",
        8.4,
        "glossy white painted aluminium, navy blue and gold accent stripe, "
        "polished chrome, matte black rubber, soft neutral studio light from "
        "the upper left, photorealistic",
    ),
    Family(
        "vehicle",
        ("car", "truck", "lorry", "van", "bus", "tractor", "tank", "cart",
         "wagon", "carriage", "chariot", "boat", "ship", "submarine", "canoe",
         "raft", "train", "locomotive", "motorcycle", "bicycle", "mech",
         "rover", "buggy", "sled", "sleigh"),
        HYBRID,
        "Script the wheels, the bed and the frame; generate the soft irregular "
        "cargo and any sculptural bodywork. Every wheel on a vehicle is "
        "nothing but dimensions.",
        "The `wooden_cart` worked example is scripted hardware — bed, axle, "
        "wheels, shafts — with only the canvas bundle and the lantern "
        "generated. On the truck comparison both generators returned wheels "
        "'fused into the bodywork rather than separate cylinders, with no gap "
        "between tyre and arch'.",
        "docs/PROCEDURAL.md",
        4.0,
        "chipped painted steel, bare weathered timber, blackened iron "
        "fittings, matte black rubber, soft overcast daylight, photorealistic",
    ),
    Family(
        "architecture",
        ("house", "cottage", "hut", "shack", "cabin", "building", "tower",
         "castle", "keep", "gatehouse", "fort", "fortress", "church", "temple",
         "chapel", "barn", "mill", "shop", "inn", "tavern", "warehouse",
         "bridge", "gate", "ruin", "dungeon", "room", "interior", "well",
         "watchtower", "lighthouse", "windmill"),
        HYBRID,
        "A building is walls, floors, stairs and openings — every one of them a "
        "stated dimension — with a small budget of generated ornament where the "
        "carving is the point. Ask for it low-poly and the generated budget "
        "goes to zero.",
        "`wall_panel` composes its aperture out of the four slabs around it, "
        "720 triangles, no boolean; `stairs` is 360 and `wedge` is 8. Against "
        "that, a generated window does not exist at all — the fuselage's "
        "portholes were in the reference and not in the mesh.",
        "docs/PROCEDURAL.md",
        8.0,
        "weathered grey limestone, dark oiled timber, blackened iron fittings, "
        "clay roof tiles, soft overcast daylight, photorealistic, matte finish",
    ),
    Family(
        "modular_kit",
        ("wall section", "floor tile", "road", "path", "pavement", "fence",
         "railing", "pillar", "column", "arch", "platform", "ramp",
         "staircase", "stairs", "doorway", "corridor", "modular", "tileset",
         "tile set", "kit", "greybox", "graybox", "blockout", "block out"),
        SCRIPTED,
        "A modular piece is defined by the dimension it has to tile at. That is "
        "the definition of a thing you write down, and generation cannot hit a "
        "dimension at all.",
        "A generated barrel 'arrives at an arbitrary size that has to be "
        "declared by hand'; the scripted one comes out 'at exactly the "
        "dimensions asked for, watertight, with clean symmetric hoops', at 840 "
        "triangles and 2.9 ms against 19 237 and 42.2 s.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        4.0,
        "weathered grey limestone, mortar joints, soft overcast daylight, "
        "matte finish",
    ),
    Family(
        "furniture",
        ("table", "bench", "chair", "stool", "desk", "shelf", "bookcase",
         "cupboard", "wardrobe", "bed", "workbench", "counter", "ladder",
         "crate", "barrel", "keg", "box", "chest of drawers", "cabinet",
         "sign", "signpost", "fence post", "torch bracket"),
        SCRIPTED,
        "Furniture and containers are the catalogue. There is a kind for it, it "
        "is exact, and it is three orders of magnitude cheaper.",
        "A scripted crate is 4.6 ms and 1 380 triangles against a generated "
        "one's 83-151 s and 20 000, and it comes out at exactly the dimensions "
        "asked for. Every kind is asserted watertight, winding-consistent and "
        "dimensioned to the request.",
        "docs/PROCEDURAL.md",
        2.0,
        "weathered oak with visible grain, black wrought iron fittings, soft "
        "overcast daylight, photorealistic, matte finish",
    ),
    Family(
        "unknown",
        (),
        SINGLE,
        "The server does not recognise this subject, and a keyword table is "
        "the wrong thing to recognise it with. Defaulting to one generation, "
        "because that is what nine of ten organic subjects wanted and it is the "
        "cheapest way to be wrong.",
        "Nine of ten subjects in the organic set came back usable from one "
        "generation at 30-49 s. The tenth was a generator-choice problem, not a "
        "decomposition problem.",
        "docs/WHAT-GENERATION-IS-FOR.md",
        1.5,
        "neutral matte surface, muted palette, soft neutral studio light from "
        "the upper left, photorealistic",
        single_defensible=True,
    ),
)

_FAMILIES_BY_NAME = {f.name: f for f in FAMILIES}
UNKNOWN_FAMILY = _FAMILIES_BY_NAME["unknown"]


# --- request modifiers ------------------------------------------------------
#
# Words in the request that override what the subject alone would say. "a house"
# is hybrid; "a low-poly house" is not, and the difference is four words the
# caller typed.

_LOW_POLY_WORDS = (
    "low poly", "low-poly", "lowpoly", "lo-fi", "flat shaded", "flat-shaded",
    "faceted", "blocky", "minimal", "stylised low", "stylized low", "retro",
    "n64", "ps1", "voxel",
)
_GREYBOX_WORDS = (
    "greybox", "graybox", "grey box", "gray box", "blockout", "block out",
    "blocking", "placeholder", "whitebox", "white box", "prototype", "mockup",
    "proxy geometry",
)
_EXACT_WORDS = (
    "exact", "exactly", "precise", "to scale", "dimension", "measured",
    "millimetre", "millimeter", "tolerance", "fits", "must be", "studs wide",
    "studs tall", "studs long",
)
_MANY_WORDS = (
    "modular", "tileable", "tiling", "repeating", "set of", "kit of", "batch",
    "variants", "each", "several", "dozens", "hundreds", "a bunch of",
)
_DETAILED_WORDS = (
    "detailed", "ornate", "intricate", "elaborate", "decorated", "carved",
    "engraved", "hero", "showcase", "photoreal", "photorealistic", "highly",
    "richly", "baroque", "gothic", "filigreed",
)
_INTERIOR_WORDS = (
    "interior", "inside", "hollow", "openable", "opens", "walk in", "enter",
    "enterable", "cutaway", "room inside", "you can go in",
)

# A number followed by a unit. "a 4 m wall section" is a scripted request and
# says so in four characters.
_DIMENSION_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:m|cm|mm|metre|meter|metres|meters|stud|studs|ft|"
    r"foot|feet|inch|inches)\b"
)
# "forty rocks", "12 crates" — a count in front of a plural.
_COUNT_RE = re.compile(
    r"\b(\d+|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|"
    r"fifty|hundred)\s+\w+s\b"
)


def _normalise(text: str) -> str:
    """Lowercase, collapse punctuation to spaces, and singularise.

    Two padded forms come back joined, the second with plurals stripped, so a
    keyword table written in the singular matches "six oval portholes" and "two
    gear struts" without every entry having to be listed twice. Only a trailing
    's' on a word longer than three characters and not already ending 'ss', so
    "glass" and "brass" survive.
    """
    words = re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
    singular = [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss")
                else w for w in words]
    return f" {' '.join(words)}  {' '.join(singular)} "


def _hits(text: str, words) -> list[str]:
    """Which of `words` appear in `text`, longest first.

    Longest-first is the same rule materials.py uses: "wall section" must beat
    "wall", and "axe head" must beat "head", or the routing is decided by
    whichever key the dict happened to iterate first.

    The membership test is a substring of the padded text rather than a regex
    per keyword: there are several hundred keywords and this runs on every
    request. Padding both sides is what keeps "art" out of "cart".
    """
    normal = _normalise(text)
    found = [w for w in words if f" {w} " in normal]
    return sorted(set(found), key=lambda w: (-len(w), w))


# --- classification ---------------------------------------------------------


def classify_part(text: str) -> Archetype | None:
    """Which archetype a part name or description falls under, or None.

    Longest keyword wins. None means "the taxonomy has no verdict", which is an
    honest answer and better than routing a part on a guess — the caller is the
    one who knows what a `bilge keel` is.
    """
    best: tuple[int, Archetype] | None = None
    normal = _normalise(text)
    for archetype in ARCHETYPES:
        for keyword in archetype.keywords:
            if f" {keyword} " in normal and (best is None or len(keyword) > best[0]):
                best = (len(keyword), archetype)
    return best[1] if best else None


def classify_subject(text: str) -> Family:
    """Which family a subject falls under. Falls through to `unknown`.

    Longest keyword across all families wins, so "wall section" resolves to the
    modular kit rather than "wall"-in-architecture, and "treasure chest" to the
    ornate prop rather than "chest"-in-furniture.
    """
    best: tuple[int, Family] | None = None
    normal = _normalise(text)
    for family in FAMILIES:
        for keyword in family.keywords:
            if f" {keyword} " in normal and (best is None or len(keyword) > best[0]):
                best = (len(keyword), family)
    return best[1] if best else UNKNOWN_FAMILY


def classify_target(text: str) -> Target | None:
    """Where the caller said this is going, or None if they did not say.

    None is a real answer and the caller of this function has to handle it,
    because "they did not say" is exactly the case that has been silently
    resolving to Roblox's 20 000 for this project's whole life.
    """
    best: tuple[int, Target] | None = None
    normal = _normalise(text)
    for target in TARGETS:
        for keyword in target.keywords:
            if f" {keyword} " in normal and (best is None or len(keyword) > best[0]):
                best = (len(keyword), target)
    return best[1] if best else None


def classify_detail(text: str) -> str | None:
    """How close the viewer gets, from prose. None if the text does not say."""
    normal = _normalise(text)
    for level, words in _DISTANCE_WORDS.items():
        if any(f" {w} " in normal for w in words):
            return level
    return None


def budget_for(target: Target, detail: str, mode: str = GENERATE) -> int:
    """Triangles for one part, given where it is going and how close.

    Returns 0 for a target that should not be decimated at all — that is not a
    budget of zero, it is the absence of a budget, and `cost()` and the draft
    plan both read it that way.
    """
    if not target.decimate:
        return 0
    lean, typical, generous = target.faces
    return (lean, typical, generous)[_DETAIL_INDEX[detail]]


def lod_chain(target: Target, detail: str) -> list[int]:
    """A descending ladder of budgets from one generated mesh.

    Nearly free, and the reason is arithmetic rather than cleverness: the raw
    mesh is already written to `mesh_raw.glb`, and quadric decimation is ~0.3 s.
    So a three-level chain costs one generation plus 0.9 s, against three
    generations for three separate assets.

    The levels are the measured ladder — 40 000 indistinguishable from raw,
    20 000 the sweet spot, 8 000 silhouette-perfect with the relief gone — cut
    off at the target's own budget.
    """
    if not target.decimate:
        return []
    top = budget_for(target, detail)
    ladder = [n for n in (200000, 80000, 40000, 20000, 8000, 2000, 500)
              if n <= top]
    # Always at least the target itself, plus a distant level if there is room
    # below it. Two levels is where the value is; five is a chore nobody wants.
    if not ladder:
        return [top]
    chain = [top]
    for n in ladder:
        if n <= chain[-1] / 2:
            chain.append(n)
        if len(chain) == 3:
            break
    return chain


def generation_settings(target: Target, detail: str, generator: str,
                        solid: bool, faces: int | None = None) -> dict:
    """Resolution and generator for one generated part, with the reason.

    This is the knob that cannot be undone. Decimation is reversible — the raw
    mesh is kept — and resolution is not: no budget recovers detail that was
    never generated, and turning it up on the wrong subject does not fail
    slowly, it stalls at 96% of VRAM and has to be killed.
    """
    faces = budget_for(target, detail) if faces is None else faces
    wants_detail = (not target.decimate) or faces >= 40000

    if generator == "hunyuan3d":
        octree = 128 if (target.decimate and faces and faces <= 2000) else 256
        settings = {"octree_resolution": octree}
        why = HUNYUAN_OCTREE[octree]
        if octree == 128:
            why += (" Chosen because this part's whole budget is "
                    f"{faces} triangles, so a 350k raw mesh would be generated "
                    f"and then thrown away — unless you want an LOD chain, in "
                    f"which case stay at 256 and decimate several times.")
    elif wants_detail and not solid:
        tier = TRELLIS_PIPELINES["1024_cascade"]
        settings = {"pipeline_type": tier["pipeline_type"],
                    "texture_size": tier["texture_size"]}
        why = (
            "The budget is high enough that 512 becomes the limiting factor "
            "rather than the decimation, and this subject is not solid, which "
            "is the case where the higher tier completes. It is still the "
            "risky option: " + tier["evidence"]
        )
    else:
        tier = TRELLIS_PIPELINES["512"]
        settings = {"pipeline_type": tier["pipeline_type"],
                    "texture_size": tier["texture_size"]}
        why = (
            "512 is the shipped default and the only tier measured to complete "
            "on solid subjects. " + (
                "This subject reads as solid, and generation cost scales with "
                "occupied volume: " + TRELLIS_PIPELINES["1024_cascade"]["evidence"]
                if solid else
                "The budget here does not exceed what 512 supplies, so the "
                "higher tier would cost VRAM and wall time for detail that "
                "decimation removes anyway."
            )
        )

    return {
        "generator": generator,
        "target_faces": faces or None,
        "keep_raw": True,
        "decimate": bool(faces),
        "settings": settings,
        "why": why,
        "how_to_apply": (
            "These are `POST /jobs` parameters. `decompose.run()` forwards only "
            "`generator`, `target_faces`, `textured` and `seed`, so a part that "
            "needs a non-default resolution has to be submitted directly — a "
            "one-line gap in `decompose._job_params`."
        ),
    }


def estimated_bytes(faces: int, coloured: bool = False) -> int:
    """glTF size for a mesh of this many triangles.

    ~18 bytes a face plus a kilobyte of container. Fitted to the measured
    decimation ladder (353 966 -> 6.2 MiB, 20 000 -> 352 KiB, 8 000 -> 141 KiB)
    and checked against nine scripted parts built on the live server, where it
    lands within 3.4%.

    `coloured` adds the back-projected atlas, which is most of the file on a
    generated part: the skull measured live was 1.68 MiB of which 0.34 MiB was
    geometry. A caller sizing a download needs the second number; a caller
    sizing a triangle budget needs the first.
    """
    return int(GLB_OVERHEAD_BYTES + BYTES_PER_FACE * max(0, faces)
               + (COLOUR_ATLAS_BYTES if coloured else 0))


# --- the request ------------------------------------------------------------


@dataclass
class Request:
    """What the caller wants, in the terms the server can actually act on.

    Everything except `subject` is optional and every default is the common
    case. The fields that matter most are the ones a subject string cannot
    carry: how many of these there are, how hard they will be looked at, and
    whether the caller has already decided what the parts are.
    """

    subject: str
    # Where this is going, which decides the triangle budget. Left None it is
    # read out of `intent`/`notes`, and failing that it falls back to Roblox
    # with `assumed` set — because Roblox's 20 000 is a Roblox number that this
    # project has been applying universally, and the fix is to say so rather
    # than to pretend there is no default.
    target: str | None = None
    # Free text: *why* the caller needs this. The primary caller is an LLM
    # describing a need, not filling in a form, so this is matched for target
    # and viewing-distance words rather than parsed.
    intent: str | None = None
    # hero | prop | background — how close the viewer gets, within the target's
    # band. Inferred from `intent` when not given.
    detail: str | None = None
    # Ask for a descending ladder of budgets off one generation. Nearly free:
    # the raw mesh is already on disk and each level is ~0.3 s of CPU.
    lod: bool = False
    # Override the computed budget outright. The caller knows things the table
    # does not; 0 means "do not decimate".
    target_faces: int | None = None
    # How many of this thing are needed. Forty rocks is a scripted request even
    # though one rock is a generated one.
    quantity: int = 1
    # Part names the caller has already decided on. Overrides the draft's own
    # decomposition, and any value above 1 rules `single` out — the caller has
    # said it wants parts.
    parts: list[str] = field(default_factory=list)
    # Hard overrides. None means "infer from the subject text".
    low_poly: bool | None = None
    interior: bool | None = None
    # A budget the caller is willing to spend. 0 forbids the GPU entirely, which
    # is the useful case: "I need this now."
    max_generations: int | None = None
    # Used verbatim if given; drafted from the family otherwise. Never allowed
    # to repeat a content word from the subject — see the style_suffix_leak
    # ceiling.
    style: str | None = None
    seed: int = 20260806
    name: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, data) -> "Request":
        if isinstance(data, Request):
            return data
        if not isinstance(data, dict):
            raise StrategyError(f"a request must be an object, got {type(data).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            # Same policy as decompose.Part.from_dict: silently dropping
            # `lowpoly` would return a hybrid plan and blame the caller.
            raise StrategyError(
                f"unknown request field(s) {unknown}; expected any of {sorted(known)}"
            )
        payload = dict(data)
        if not str(payload.get("subject") or "").strip():
            raise StrategyError("a request needs a non-empty 'subject'")
        detail = payload.get("detail")
        if detail is not None and detail not in DETAIL_LEVELS:
            raise StrategyError(
                f"detail {detail!r}; expected one of {list(DETAIL_LEVELS)}"
            )
        target = payload.get("target")
        if target is not None and target not in _TARGETS_BY_NAME:
            # Not fatal in spirit — but a typo silently resolving to Roblox's
            # cap is precisely the bug this field exists to fix.
            raise StrategyError(
                f"target {target!r}; expected one of {sorted(_TARGETS_BY_NAME)}, "
                f"or leave it out and describe the need in `intent`"
            )
        quantity = payload.get("quantity", 1)
        if not isinstance(quantity, int) or isinstance(quantity, bool) or quantity < 1:
            raise StrategyError(f"quantity must be a positive whole number, got {quantity!r}")
        faces = payload.get("target_faces")
        if faces is not None and (isinstance(faces, bool) or not isinstance(faces, int)
                                  or faces < 0):
            raise StrategyError(
                f"target_faces must be a whole number of triangles, or 0 for "
                f"'do not decimate'; got {faces!r}"
            )
        payload["parts"] = list(payload.get("parts") or [])
        return cls(**payload)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def text(self) -> str:
        """Everything the caller wrote, for keyword matching."""
        return " ".join(filter(None, [self.subject, self.intent or "",
                                      self.notes or ""]))

    def resolve_target(self) -> tuple[Target, bool]:
        """The delivery target and whether it was assumed rather than stated.

        The `assumed` flag is the whole point. Roblox's per-MeshPart cap is a
        Roblox number; it has been the universal default in this project
        without ever being marked as one, so when it is being applied on a
        guess the recommendation says so out loud.
        """
        if self.target:
            return _TARGETS_BY_NAME[self.target], False
        found = classify_target(self.text)
        if found is not None:
            return found, False
        return _TARGETS_BY_NAME["roblox"], True

    def resolve_detail(self) -> tuple[str, bool]:
        """How close the viewer gets, and whether it was stated."""
        if self.detail:
            return self.detail, False
        found = classify_detail(self.text)
        if found is not None:
            return found, False
        return "prop", True

    @property
    def face_budget(self) -> int:
        """Triangles for one generated part. 0 means do not decimate at all.

        An explicit `target_faces` always wins — the caller may know that this
        particular part carries engraved lettering, which is the one thing
        decimation destroys first and the one thing no table can see.
        """
        if self.target_faces is not None:
            return self.target_faces
        target, _ = self.resolve_target()
        detail, _ = self.resolve_detail()
        return budget_for(target, detail)

    def _sub_budget(self, fraction: float, floor: int) -> int:
        """A smaller budget for a secondary part, respecting 'do not decimate'."""
        budget = self.face_budget
        if not budget:
            return 0
        return max(min(floor, budget), int(budget * fraction))


# --- choosing -------------------------------------------------------------


@dataclass
class Signal:
    """One reason, with the measurement behind it.

    A recommendation without its evidence is an opinion, and an agent cannot
    argue with an opinion. Every signal names what it saw in the request, which
    strategy it argues for, and the measured fact that makes it an argument.
    """

    saw: str
    argues_for: str
    weight: float
    claim: str
    evidence: str
    source: str

    def as_dict(self) -> dict:
        return asdict(self)


def _signals(request: Request, family: Family) -> list[Signal]:
    """Every reason this request points where it points."""
    text = request.text
    out: list[Signal] = []

    out.append(Signal(
        f"subject reads as {family.name!r}",
        family.strategy, 3.0,
        family.rationale, family.evidence, family.source,
    ))

    low_poly = request.low_poly
    if low_poly is None:
        low_poly = bool(_hits(text, _LOW_POLY_WORDS))
    if low_poly:
        out.append(Signal(
            "the request asks for low-poly",
            SCRIPTED, 4.0,
            "Low-poly is a triangle budget, and a generated part cannot hit "
            "one: it arrives at 0.5-4.9 M faces and is decimated down, which "
            "spreads its budget over whatever survived. A scripted part is "
            "built at the count you asked for.",
            "A `wedge` ramp is 8 triangles and a `plank` is 60, and every "
            "kind that has a facing or a bevel takes a parameter to turn it "
            "off — 8-12 radial sections 'reads as deliberately faceted', which "
            "is low-poly as a decision rather than as damage. One generated "
            "crate is 20 000, decimated from 0.9-9.8 M.",
            "docs/PROCEDURAL.md",
        ))

    if _hits(text, _GREYBOX_WORDS):
        out.append(Signal(
            "the request asks for a greybox or blockout",
            SCRIPTED, 5.0,
            "Blocking geometry is measured, disposable and needed now. Waiting "
            "40 s per part for a mesh you are going to throw away is the "
            "opposite of what blocking is for.",
            "A `wedge` is 8 triangles in 0.8 ms and is the most common blocking "
            "shape in a Roblox place; a `crate` is 4.6 ms. The generator's "
            "floor is ~35 s a part whatever the subject.",
            "docs/PROCEDURAL.md",
        ))

    if _hits(text, _EXACT_WORDS) or _DIMENSION_RE.search(text.lower()):
        out.append(Signal(
            "the request states a dimension",
            SCRIPTED, 3.5,
            "Generation cannot hit a stated dimension at all — the mesh comes "
            "back normalised to a unit box and has to be scaled by hand "
            "afterwards. A primitive is built at the number you gave it.",
            "Longest side of the six generated Bonanza parts: 0.9923, 0.9989, "
            "0.9997, 0.9936, 0.9989, 0.9921. An 8.4 m fuselage and a 0.9 m "
            "strut arrive the same size.",
            "docs/DECOMPOSITION.md",
        ))

    many = request.quantity > 1 or bool(_hits(text, _MANY_WORDS)) \
        or bool(_COUNT_RE.search(text.lower()))
    if many:
        out.append(Signal(
            f"the request wants more than one ({request.quantity} asked for)"
            if request.quantity > 1 else "the request reads as a set or a kit",
            SCRIPTED, 4.0 if request.quantity > 1 else 3.0,
            "Generation cost is per asset and flat; scripted cost is per asset "
            "and negligible, and a scripted family varies by changing a number "
            "rather than by rolling the dice again.",
            "'If you need forty rocks, script them. If you need one hero rock, "
            "generate it.' Forty generated rocks is 40 x 35 s = 23 minutes and "
            "800 000 triangles; forty scripted ones are 0.12 s.",
            "docs/WHAT-GENERATION-IS-FOR.md",
        ))

    if _hits(text, _DETAILED_WORDS):
        out.append(Signal(
            "the request asks for detail or ornament",
            HYBRID if family.strategy != SINGLE else SINGLE, 2.0,
            "Detail volume — arbitrary quantities of small surface incident "
            "nobody wants to write down — is the one thing the generator does "
            "that no amount of parametric scripting reaches at any cost.",
            "The generated chest has a barrel-vaulted lid, iron bands wrapping "
            "over it, individually raised rivet heads, a cast lion-faced lock "
            "plate, plank seams and scrollwork claw feet, all modelled, at "
            "49 s. A scripted chest is a rounded box with a hinge.",
            "docs/WHAT-GENERATION-IS-FOR.md",
        ))

    interior = request.interior
    if interior is None:
        interior = bool(_hits(text, _INTERIOR_WORDS))
    if interior:
        out.append(Signal(
            "the request needs an interior",
            HYBRID, 2.0,
            "Every image-to-3D generator emits a solid, and carving one open "
            "is usually refused. An interior is a scripted liner nested inside "
            "the generated shell, which makes the asset a hybrid by "
            "construction.",
            "Carving the chest carcass at a 0.045 wall was refused — 'the part "
            "is thinner than two walls everywhere'. At 0.02 it removed 6.0% of "
            "the material and destroyed the UVs. The shipped liner is a "
            "`hollow_box`: 300 triangles in 9 ms.",
            "docs/HOLLOW.md",
        ))

    target, target_assumed = request.resolve_target()
    detail, _ = request.resolve_detail()

    if detail == "background":
        out.append(Signal(
            "this is background dressing",
            SCRIPTED, 1.5,
            "A background asset is never looked at closely enough for the "
            "detail volume to be worth 35 s, and its budget is small enough "
            "that decimation throws away most of what was generated.",
            "The scripted barrel is 840 triangles against the generated one's "
            "19 237. At 8 000 faces 'the silhouette is perfect, fine relief "
            "lost' — so a 1 500-triangle decimation of an ornate subject keeps "
            "the outline and discards the thing you generated it for.",
            "docs/DECIMATION.md",
        ))

    if target.name == "blockout":
        out.append(Signal(
            "the delivery target is blocking geometry",
            SCRIPTED, 5.0,
            target.summary, target.evidence, target.source,
        ))
    elif not target.decimate:
        # A no-decimation target does not argue for or against decomposition,
        # but it does mean the budget arithmetic below stops applying.
        out.append(Signal(
            f"the delivery target is {target.name!r}, which wants the raw mesh",
            family.strategy, 0.5,
            target.summary, target.evidence, target.source,
        ))
    elif target.name == "roblox":
        out.append(Signal(
            "the delivery target is Roblox"
            + (" (assumed — no target was stated)" if target_assumed else ""),
            HYBRID if family.strategy == HYBRID else family.strategy, 0.5,
            "Roblox's 20 000-triangle cap is per MeshPart, not per file, so "
            "every part you separate raises the effective budget by another "
            "20 000. That is a second, independent argument for multi-part on "
            "anything complex — and it is the one cap here that rejects the "
            "import rather than merely costing frame time.",
            "The chest's carcass alone is 19 694 triangles, so a welded "
            "version of that model would be rejected outright; the 88-part one "
            "totals 87 616 with zero parts over budget.",
            "docs/ROBLOX-EXPORT.md",
        ))

    if len(request.parts) > 1:
        out.append(Signal(
            f"the caller has already named {len(request.parts)} parts",
            HYBRID, 4.0,
            "The caller has decided this object has parts, which settles the "
            "question of whether to decompose it. What is left is routing each "
            "part, which the archetype taxonomy does.",
            "Placement and decomposition are deliberately the caller's: it "
            "knows this is a biplane and that the second wing goes above the "
            "first, and the server does not.",
            "docs/MULTI-PART.md",
        ))

    if request.max_generations == 0:
        out.append(Signal(
            "the caller has forbidden generation (max_generations=0)",
            SCRIPTED, 10.0,
            "No GPU means no generated parts. Everything is arithmetic or it "
            "does not exist.",
            "`POST /primitives` is synchronous and never enters the queue — "
            "the queue exists to serialise a GPU this path does not touch.",
            "docs/PROCEDURAL.md",
        ))

    return out


def _choose(request: Request, family: Family, signals: list[Signal]) -> tuple[str, dict]:
    """Sum the signals, apply the two hard rules, return the winner and the tally."""
    tally = {s: 0.0 for s in STRATEGIES}
    for signal in signals:
        tally[signal.argues_for] += signal.weight

    # Hard rule one: `single` is only on the table if the family says one
    # generation can actually be the whole asset. An aeroplane cannot.
    if not family.single_defensible:
        tally[SINGLE] = -math.inf
    # Hard rule two: a caller that named parts has ruled out `single` outright,
    # whatever the subject looks like.
    if len(request.parts) > 1:
        tally[SINGLE] = -math.inf
    # Hard rule three: no GPU, no generated parts, no argument.
    if request.max_generations == 0:
        tally[SINGLE] = -math.inf
        tally[HYBRID] = -math.inf

    winner = max(STRATEGIES, key=lambda s: (tally[s], STRATEGIES.index(s)))
    return winner, tally


# --- drafting a plan --------------------------------------------------------
#
# Recipes, as data where they can be. These are starting points and the module
# says so loudly: the server can put a `wall_panel` where a wall goes, and it
# cannot know that this particular gatehouse has a portcullis and a murder hole.

# Framing clause for a whole object generated in one piece. Naming the object
# is fine *here* and only here: the completion prior that ruins part prompts is
# working in your favour when the part is the whole object.
WHOLE_OBJECT_VIEW = (
    "seen from a three-quarter angle slightly above so its depth and its "
    "length are both visible, whole object large in frame"
)


# Parameters that add surface incident rather than shape. Turning them off is
# what makes a scripted part low-poly *as a decision* rather than as damage —
# which is the difference between deliberate low-poly and a smeared decimation.
# Named rather than inferred, because "apron" and "backrest" are also booleans
# and dropping those changes what the object is.
_DECORATION_OFF: dict[str, object] = {
    "trim": False, "chamfer": 0.0, "relief": 0.0, "tread_depth": 0.0,
    "hoop_thickness": 0.0, "joint": 0.0, "corner_bosses": False,
    "returns": False, "sill": False, "hood": False, "straps": False,
    "studs": False, "keystone": False, "impost": False, "corbel": False,
    "coping": False, "fascia": False, "ridge": False, "crown": False,
    "newel": False, "batten": 0.0,
}
# Choices that mean "no facing" wherever a kind offers one.
_PLAIN_CHOICES = ("flat", "plain", "none", "square", "smooth")
# "8-12 reads as deliberately faceted" — primitives.py's own words about
# `sections`, and exactly the look a low-poly request is asking for.
_FACETED_SECTIONS = 8


def lean_params(kind: str, params: dict) -> dict:
    """The same part with its decoration turned off, for a low-poly build.

    Discovery-driven rather than tabulated: it reads the kind's own parameter
    spec from `primitives.KINDS`, tries each reduction, and keeps the ones that
    actually lower the face count. That costs a few milliseconds and it is
    immune to the catalogue changing underneath — which it is doing, and which
    is why the triangle counts in the docs have already drifted.
    """
    spec = primitives.KINDS.get(kind)
    if spec is None:
        return dict(params)

    out = dict(params)
    names = {p.name: p for p in spec.params}

    def faces(candidate: dict) -> int | None:
        try:
            return int(len(primitives.build(kind, candidate).faces))
        except Exception:
            # An out-of-range or incompatible combination is not an error here,
            # it just means this reduction does not apply to this kind.
            return None

    base = faces(out)
    if base is None:
        return dict(params)

    for name, value in _DECORATION_OFF.items():
        param = names.get(name)
        if param is None or name in params:
            continue  # never override something the caller stated
        trial = {**out, name: value}
        got = faces(trial)
        if got is not None and got < base:
            out, base = trial, got

    for name, param in names.items():
        if name in params:
            continue
        if param.type == "choice":
            for choice in _PLAIN_CHOICES:
                if choice in (param.choices or ()):
                    trial = {**out, name: choice}
                    got = faces(trial)
                    if got is not None and got < base:
                        out, base = trial, got
                    break
        elif param.name == "sections":
            trial = {**out, name: max(param.minimum or 3, _FACETED_SECTIONS)}
            got = faces(trial)
            if got is not None and got < base:
                out, base = trial, got

    return out


def _slug(text: str) -> str:
    """A part-name-shaped token from free text. Node names, so keep it plain."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    skip = {"a", "an", "the", "of", "with", "and"}
    kept = [w for w in words if w not in skip][:3]
    return "_".join(kept) or "part"


def _strip_style_leaks(style: str, subject: str) -> str:
    """Drop style clauses that repeat a content word from the subject.

    `decompose.style_leaks` warns about this and it is worth not tripping in
    the first place: a suffix that names the object re-arms the completion
    prior through the text encoder and brings whole aeroplanes back behind
    single parts. Clauses are comma-separated, so a leak costs one clause
    rather than the whole suffix.
    """
    leaked = decompose._content_words(subject)
    if not leaked:
        return style
    clauses = [c.strip() for c in style.split(",")]
    kept = [c for c in clauses if not (decompose._content_words(c) & leaked)]
    return ", ".join(kept) if kept else style


def _generated(name, prompt, size_m, faces, material=None, placement=None,
               generator=None, note=None) -> dict:
    part = {
        "name": name, "mode": GENERATE, "prompt": prompt,
        "size_m": size_m,
        # 0 means "do not decimate", and the way to say that in a plan is to
        # leave the part's budget unset and let the plan's 0 fall through:
        # `decompose._job_params` does `part.target_faces or plan.target_faces`,
        # so a part-level 0 would be swallowed by the `or`.
        **({"target_faces": faces} if faces else {}),
        "placement": placement or {},
    }
    if material:
        part["material"] = material
    if generator:
        part["generator"] = generator
    if note:
        part["note"] = note
    return part


def _scripted(name, kind, params, size_m=None, material=None, placement=None,
              note=None) -> dict:
    part = {
        "name": name, "mode": SCRIPT, "kind": kind, "params": params,
        "placement": placement or {},
    }
    if size_m is not None:
        part["size_m"] = size_m
    if material:
        part["material"] = material
    if note:
        part["note"] = note
    return part


def _mirrored(name, of, axis="x", note=None) -> dict:
    part = {"name": name, "mode": MIRROR,
            "placement": {"mirror_of": of, "mirror": axis}}
    if note:
        part["note"] = note
    return part


def _draft_single(request: Request, family: Family) -> list[dict]:
    """One part. The whole subject, generated once."""
    return [_generated(
        _slug(request.subject),
        f"{request.subject.strip().rstrip('.,')}, {WHOLE_OBJECT_VIEW}",
        family.size_m,
        request.face_budget,
        note="One sculptural whole. " + DO_NOT_DECOMPOSE,
    )]


def _draft_weapon(request: Request, family: Family) -> list[dict]:
    """Generate the head, script the shaft. The routing rule in one asset."""
    length = family.size_m
    return [
        _scripted(
            "haft", "cylinder",
            {"radius": round(length * 0.022, 4), "height": round(length * 0.72, 4),
             "chamfer": 0.01},
            size_m=[round(length * 0.044, 4), round(length * 0.72, 4),
                    round(length * 0.044, 4)],
            material="wood",
            placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
            note="Straight by construction. The generated axe shaft came back "
                 "lumpy and banana-shaped; a cylinder is 192 triangles.",
        ),
        _generated(
            "head",
            "a heavy cast metal weapon head with a scalloped edge, knotwork "
            "etched into both cheeks and a cast beast face where it meets the "
            "handle socket, cut off flat at the socket, nothing attached, "
            + WHOLE_OBJECT_VIEW,
            round(length * 0.3, 4), request.face_budget, material="metal",
            placement={"anchor": {"to": "haft", "align": {"y": "on"}}},
            note="The detail volume. One attempt on the ornate axe, IoU 0.850.",
        ),
    ]


def _draft_ornate_prop(request: Request, family: Family) -> list[dict]:
    """The chest, generalised: generated carcass and ornament, scripted iron.

    Deliberately shaped like docs/SHOWCASE-CHEST.md, including the decision
    that cost the most to learn — the lid is scripted.
    """
    width = family.size_m
    stave = round(width / 9.0, 4)
    parts = [
        _generated(
            "carcass",
            "a wide rectangular box of thick carved staves with chamfered "
            "mouldings along every edge, open at the top, no lid and no cover, "
            + WHOLE_OBJECT_VIEW,
            [width, round(width * 0.45, 4), round(width * 0.55, 4)],
            request.face_budget, material="wood",
            placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
            note="Irregular surface relief: the thing a formula cannot write. "
                 "Measured 42.3 s, IoU 0.863, 19 694 faces.",
        ),
        _generated(
            "escutcheon",
            "a thick cast metal plate in deep relief with a snarling beast "
            "face at its centre and a keyhole below the chin, flat on the back, "
            "nothing attached, " + WHOLE_OBJECT_VIEW,
            round(width * 0.2, 4), request._sub_budget(0.5, 8000),
            material="dark_metal",
            placement={"anchor": {"to": "carcass",
                                  "align": {"z": "max", "y": 0.55}},
                       "my": {"z": "min"}},
            note="The ornament is the entire value of the prop. One attempt, "
                 "IoU 0.858.",
        ),
        _generated(
            "foot_front_left",
            "a single cast metal clawed animal foot, three splayed toes with "
            "hooked talons gripping downward, cut off flat at the ankle, "
            "nothing attached, " + WHOLE_OBJECT_VIEW,
            round(width * 0.14, 4), request._sub_budget(0.34, 6000),
            material="dark_metal",
            placement={"anchor": {"to": "carcass",
                                  "align": {"y": "under", "x": 0.12, "z": 0.15}}},
            note="An organic paw. Generated once and mirrored three times — "
                 "the other three feet are free.",
        ),
        _mirrored("foot_front_right", "foot_front_left", "x"),
        _mirrored("foot_back_left", "foot_front_left", "z",
                  note="The fourth foot is not here, and the reason is a real "
                       "limit: `mirror_of` takes one world plane and a mirror "
                       "of a mirror is rejected, so the back-right foot cannot "
                       "be reached by reflection from the front-left one. Add "
                       "it as one more entry in the `assemble_request` reusing "
                       "`foot_front_left`'s job id with an explicit position — "
                       "which is what the showcase chest did, and it still "
                       "costs no second generation."),
    ]
    # The lid. Nine flat staves on an ellipse rather than one generated vault,
    # which is the decision docs/SHOWCASE-CHEST.md paid twenty reference images
    # to learn. Five here rather than nine, because this is a draft.
    for i in range(5):
        parts.append(_scripted(
            f"lid_stave_{i + 1}", "plank",
            {"length": round(width * 0.98, 4), "width": stave,
             "thickness": round(width * 0.05, 4), "chamfer": 0.01},
            size_m=[round(width * 0.98, 4), round(width * 0.05, 4), stave],
            material="wood",
            placement={"anchor": {"to": "carcass",
                                  "align": {"y": "on", "z": 0.15 + i * 0.175}}},
            note="The lid is scripted, and that was not the plan: twenty "
                 "candidate references across five prompt strategies never "
                 "produced a barrel vault at a three-quarter angle.",
        ))
    parts += [
        _scripted(
            "band_left", "plank",
            {"length": round(width * 0.06, 4), "width": round(width * 0.5, 4),
             "thickness": round(width * 0.015, 4), "chamfer": 0.005},
            size_m=[round(width * 0.06, 4), round(width * 0.015, 4),
                    round(width * 0.5, 4)],
            material="dark_metal",
            placement={"anchor": {"to": "carcass", "align": {"x": 0.15}}},
            note="Strap iron: nothing but dimensions. 60 triangles.",
        ),
        _mirrored("band_right", "band_left", "x"),
        _scripted(
            "hinge_barrel", "cylinder",
            {"radius": round(width * 0.02, 4), "height": round(width * 0.5, 4),
             "chamfer": 0.005},
            size_m=[round(width * 0.04, 4), round(width * 0.5, 4),
                    round(width * 0.04, 4)],
            material="dark_metal",
            placement={"rotation": [0, 90, 0],
                       "anchor": {"to": "carcass",
                                  "align": {"y": "top", "z": "min"}}},
        ),
    ]
    return parts


def _draft_aircraft(request: Request, family: Family) -> list[dict]:
    """Generated body, scripted flight surfaces and gear, mirrored right side.

    This differs from `decompose.BONANZA`, which generates the wings, and the
    difference is deliberate: that plan predates `tapered_panel`, which was
    added to primitives.py *because* an agent built an aircraft's flight
    surfaces out of this library. Three generations instead of six.
    """
    length = family.size_m
    semi_span = round(length * 0.52, 4)
    root = round(length * 0.17, 4)
    tip = round(length * 0.11, 4)
    thick = round(root * 0.15, 4)
    return [
        _generated(
            "fuselage",
            "a hollow elongated shell with an oval cross section, tapered at "
            "both ends, a smooth unbroken flank, no wings, no fins, no wheels, "
            + WHOLE_OBJECT_VIEW,
            [round(length * 0.13, 4), round(length * 0.155, 4), length],
            request.face_budget, material="paint",
            placement={"position": [0, 0, 0]},
            note="The hero part; everything else anchors to it. Do not ask it "
                 "for portholes or a cabin hump — see the ceilings.",
        ),
        _generated(
            "engine_cowl",
            "a hollow tapered cowling shell open at both ends with a round air "
            "intake at the front, nothing attached, " + WHOLE_OBJECT_VIEW,
            round(length * 0.17, 4), request._sub_budget(0.5, 8000),
            material="paint",
            placement={"anchor": {"to": "fuselage", "align": {"z": "max"}},
                       "my": {"z": "min"}},
        ),
        _generated(
            "propeller",
            "a hub with three long narrow blades radiating from it, each blade "
            "twisted along its length and tapering to a rounded tip, a polished "
            "conical spinner over the hub, " + WHOLE_OBJECT_VIEW,
            round(length * 0.24, 4), request._sub_budget(0.5, 6000),
            material="metal",
            placement={"anchor": {"to": "engine_cowl", "align": {"z": "max"}},
                       "my": {"z": "min"}},
            note="'Propeller' alone returns a marine propeller; the blade "
                 "count and the word 'narrow' are load-bearing.",
        ),
        _scripted(
            "left_wing", "tapered_panel",
            {"span": semi_span, "root_chord": root, "tip_chord": tip,
             "thickness": thick, "thickness_taper": round(1 - tip / root, 3),
             "sweep": round(-(root - tip) / 2, 4)},
            size_m=[semi_span, thick, root], material="paint",
            placement={"anchor": {"to": "fuselage",
                                  "align": {"x": "min", "y": 0.25, "z": 0.45},
                                  "my": {"x": "max"}}},
            note="Scripted, not generated. The Bonanza's generated wing came "
                 "back as two crossed slabs; with the viewpoint clause it "
                 "became one chunky wing at 11 892 triangles in 49 s. This is "
                 "60 triangles in 0.9 ms with a 0.0166 m worst chord step.",
        ),
        _mirrored("right_wing", "left_wing", "x"),
        _scripted(
            "tail_fin", "tapered_panel",
            {"span": round(length * 0.18, 4), "root_chord": round(root * 0.8, 4),
             "tip_chord": round(tip * 0.6, 4), "thickness": round(thick * 0.7, 4)},
            size_m=[round(thick * 0.7, 4), round(length * 0.18, 4),
                    round(root * 0.8, 4)],
            material="paint",
            placement={"rotation": [0, 0, 90],
                       "anchor": {"to": "fuselage",
                                  "align": {"z": "min", "y": "top"},
                                  "my": {"y": "min"}}},
        ),
        _scripted(
            "left_tailplane", "tapered_panel",
            {"span": round(semi_span * 0.4, 4), "root_chord": round(root * 0.6, 4),
             "tip_chord": round(tip * 0.5, 4), "thickness": round(thick * 0.5, 4)},
            size_m=[round(semi_span * 0.4, 4), round(thick * 0.5, 4),
                    round(root * 0.6, 4)],
            material="paint",
            placement={"anchor": {"to": "fuselage",
                                  "align": {"x": "min", "y": 0.4, "z": 0.05},
                                  "my": {"x": "max"}}},
        ),
        _mirrored("right_tailplane", "left_tailplane", "x"),
        _scripted(
            "left_gear_strut", "cylinder",
            {"radius": round(length * 0.0065, 4), "height": round(length * 0.107, 4),
             "chamfer": 0.02},
            size_m=[round(length * 0.013, 4), round(length * 0.107, 4),
                    round(length * 0.013, 4)],
            material="metal",
            placement={"anchor": {"to": "left_wing",
                                  "align": {"y": "under", "x": 0.7, "z": 0.5}}},
            note="Generated, this was a featureless spindle at 7 984 triangles "
                 "in 51 s. Scripted: 192 triangles, 1.5 ms.",
        ),
        _scripted(
            "left_gear_wheel", "wheel",
            {"radius": round(length * 0.033, 4), "width": round(length * 0.014, 4),
             "hub_radius": round(length * 0.01, 4), "spoke_count": 6,
             "chamfer": 0.02},
            size_m=[round(length * 0.066, 4), round(length * 0.014, 4),
                    round(length * 0.066, 4)],
            material="rubber",
            # _revolve sweeps around +Y, so a wheel is built lying flat.
            placement={"rotation": [0, 0, 90],
                       "anchor": {"to": "left_gear_strut", "align": {"y": "under"}}},
            note="Generated, the wheel did not survive at all.",
        ),
        _mirrored("right_gear_strut", "left_gear_strut", "x"),
        _mirrored("right_gear_wheel", "left_gear_wheel", "x"),
    ]


def _draft_vehicle(request: Request, family: Family) -> list[dict]:
    """Scripted running gear and bed; generated bodywork and soft cargo."""
    length = family.size_m
    return [
        _scripted(
            "chassis", "crate",
            {"width": length, "height": round(length * 0.22, 4),
             "depth": round(length * 0.56, 4), "style": "planks",
             "plank_count": 5},
            size_m=[length, round(length * 0.22, 4), round(length * 0.56, 4)],
            material="wood",
            placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
            note="A generated crate cost 83-151 s and 20 000 faces and still "
                 "had rounded corners. This is 4.6 ms and 1 380.",
        ),
        _scripted(
            "axle", "cylinder",
            {"radius": round(length * 0.028, 4), "height": round(length * 0.66, 4)},
            size_m=[round(length * 0.056, 4), round(length * 0.66, 4),
                    round(length * 0.056, 4)],
            material="dark_metal",
            placement={"rotation": [0, 0, 90],
                       "anchor": {"to": "chassis", "align": {"y": "under", "z": 0.5}}},
        ),
        _scripted(
            "left_wheel", "wheel",
            {"radius": round(length * 0.26, 4), "width": round(length * 0.07, 4),
             "spoke_count": 8},
            size_m=[round(length * 0.52, 4), round(length * 0.07, 4),
                    round(length * 0.52, 4)],
            material="wood",
            placement={"rotation": [0, 0, 90],
                       "anchor": {"to": "axle", "align": {"x": "min"},
                                  "my": {"x": "max"}}},
            note="Both generators returned wheels fused into the bodywork with "
                 "no gap between tyre and arch. `wheel` is 872 triangles.",
        ),
        _mirrored("right_wheel", "left_wheel", "x"),
        _generated(
            "cargo",
            "a rolled bundle of coarse heavy cloth tied with three loops of "
            "rope, deep soft folds and creases along its length, "
            + WHOLE_OBJECT_VIEW,
            [round(length * 0.34, 4), round(length * 0.14, 4),
             round(length * 0.14, 4)],
            request._sub_budget(0.5, 8000), material="fabric",
            placement={"anchor": {"to": "chassis", "align": {"y": "on", "x": 0.3}}},
            note="Soft, irregular, no dimension anybody has to get right. The "
                 "one thing on this vehicle worth a generation.",
        ),
    ]


def _draft_architecture(request: Request, family: Family,
                        allow_generation: bool) -> list[dict]:
    """Walls, floor, stairs and openings, plus a small ornament budget."""
    width = family.size_m
    depth = round(width * 0.75, 4)
    height = round(width * 0.62, 4)
    thickness = round(width * 0.06, 4)
    parts = [
        _scripted(
            "floor", "plank",
            {"length": width, "width": depth, "thickness": thickness,
             "chamfer": 0.02},
            size_m=[width, thickness, depth], material="stone",
            placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
        ),
        _scripted(
            "wall_front", "wall_panel",
            {"width": width, "height": height, "thickness": thickness,
             "opening": "door", "opening_width": round(width * 0.22, 4),
             "opening_height": round(height * 0.55, 4)},
            size_m=[width, height, thickness], material="stone",
            placement={"anchor": {"to": "floor", "align": {"y": "on", "z": "max"},
                                  "my": {"z": "max"}}},
            note="The aperture is composed from the four slabs around it, so "
                 "there is no boolean and no sliver triangles. A generated "
                 "opening does not exist at all — the fuselage's portholes "
                 "were in the reference and not in the mesh.",
        ),
        _scripted(
            "wall_back", "wall_panel",
            {"width": width, "height": height, "thickness": thickness,
             "opening": "window", "opening_width": round(width * 0.18, 4),
             "opening_height": round(height * 0.28, 4),
             "sill_height": round(height * 0.45, 4)},
            size_m=[width, height, thickness], material="stone",
            placement={"anchor": {"to": "floor", "align": {"y": "on", "z": "min"},
                                  "my": {"z": "min"}}},
        ),
        _scripted(
            "wall_left", "wall_panel",
            {"width": depth, "height": height, "thickness": thickness,
             "opening": "window", "opening_width": round(depth * 0.2, 4),
             "opening_height": round(height * 0.28, 4),
             "sill_height": round(height * 0.45, 4)},
            size_m=[thickness, height, depth], material="stone",
            placement={"rotation": [0, 90, 0],
                       "anchor": {"to": "floor", "align": {"y": "on", "x": "min"},
                                  "my": {"x": "min"}}},
        ),
        _mirrored("wall_right", "wall_left", "x"),
    ]

    # `roof` and `chimney` are recent additions to the catalogue and may not be
    # there; `_kind` asks rather than assumes, and a wedge pair is the fallback
    # that has always worked.
    roof_kind = _kind("roof")
    if roof_kind:
        parts.append(_scripted(
            "roof", roof_kind,
            {"width": width, "depth": depth, "height": round(height * 0.45, 4)},
            size_m=[width, round(height * 0.45, 4), depth], material="stone",
            placement={"anchor": {"to": "wall_front", "align": {"y": "on"}}},
            note="A purpose-built kind beats two wedges: the tiles, the ridge "
                 "and the gable are parameters rather than a composition.",
        ))
    else:
        parts += [
            _scripted(
                "roof_left", "wedge",
                {"width": width, "height": round(height * 0.45, 4),
                 "depth": round(depth * 0.5, 4), "chamfer": 0.0},
                size_m=[width, round(height * 0.45, 4), round(depth * 0.5, 4)],
                material="stone",
                placement={"anchor": {"to": "wall_left",
                                      "align": {"y": "on", "z": "min"},
                                      "my": {"z": "min"}}},
                note="A `wedge` is 8 triangles and keeps its exact rise and "
                     "run, because it is the one kind whose chamfer is off by "
                     "default — cutting a ramp's apex shortens the rise it was "
                     "asked for.",
            ),
            _mirrored("roof_right", "roof_left", "z"),
        ]
    parts += [
        _scripted(
            "entry_steps", "stairs",
            {"steps": 3, "rise": round(height * 0.08, 4),
             "run": round(height * 0.11, 4), "width": round(width * 0.3, 4),
             "style": "blocks"},
            size_m=[round(width * 0.3, 4), round(height * 0.24, 4),
                    round(height * 0.33, 4)],
            material="stone",
            placement={"anchor": {"to": "wall_front", "align": {"y": "min", "z": "max"},
                                  "my": {"z": "min"}}},
        ),
    ]
    if allow_generation:
        parts.append(_generated(
            "corbel_carving",
            "a single carved stone bracket in deep relief, a snarling horned "
            "face with folded wings under a square abacus block, flat on the "
            "back, cut off flat at the wall face, nothing attached, "
            + WHOLE_OBJECT_VIEW,
            round(width * 0.09, 4), request._sub_budget(0.5, 8000),
            material="stone",
            placement={"anchor": {"to": "wall_front",
                                  "align": {"y": "top", "z": "max", "x": 0.2},
                                  "my": {"z": "min"}}},
            note="The only part here worth a generation: carving is detail "
                 "volume, and the gargoyle came back at 36.9 s with ribbed "
                 "wing membranes and claws hooked over its shoulders. Mirror "
                 "or reuse this job id for every other corbel — they are free.",
        ))
    return parts


def _draft_modular(request: Request, family: Family) -> list[dict]:
    """One scripted piece, dimensioned. The whole point is that it tiles."""
    archetype = classify_part(request.subject) or _ARCHETYPES_BY_NAME["wall"]
    kind = archetype.kind or "wall_panel"
    width = family.size_m
    presets = {
        "wall_panel": ({"width": width, "height": round(width * 0.75, 4),
                        "thickness": round(width * 0.12, 4), "opening": "none"},
                       [width, round(width * 0.75, 4), round(width * 0.12, 4)]),
        "plank": ({"length": width, "width": round(width * 0.5, 4),
                   "thickness": round(width * 0.05, 4)},
                  [width, round(width * 0.05, 4), round(width * 0.5, 4)]),
        "stairs": ({"steps": 6, "rise": round(width * 0.12, 4),
                    "run": round(width * 0.17, 4), "width": width},
                   [width, round(width * 0.72, 4), round(width * 1.02, 4)]),
        "column": ({"height": round(width * 2.0, 4), "radius": round(width * 0.15, 4)},
                   [round(width * 0.3, 4), round(width * 2.0, 4),
                    round(width * 0.3, 4)]),
        "ladder": ({"height": round(width * 1.5, 4), "width": round(width * 0.3, 4)},
                   [round(width * 0.3, 4), round(width * 1.5, 4),
                    round(width * 0.06, 4)]),
    }
    params, size_m = presets.get(kind, presets["wall_panel"])
    return [_scripted(
        _slug(request.subject), kind, params, size_m=size_m,
        placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
        note="Exact to the dimensions asked for, watertight, and it tiles. A "
             "generated one arrives at an arbitrary size that has to be "
             "declared by hand and is not watertight after decimation.",
    )]


def _draft_furniture(request: Request, family: Family) -> list[dict]:
    """Straight out of the catalogue. There is a kind for it."""
    archetype = classify_part(request.subject)
    kind = (archetype.kind if archetype and archetype.route == SCRIPT else None) \
        or "crate"
    spec = primitives.KINDS[kind]
    params = {p.name: p.default for p in spec.params
              if p.type in ("number", "integer")}
    mesh = primitives.build(kind, params)
    lo, hi = mesh.bounds
    return [_scripted(
        _slug(request.subject), kind, {},
        size_m=[round(float(v), 4) for v in (hi - lo)],
        placement={"anchor": {"to": "ground"}, "position": [0, 0, 0]},
        note=f"`{kind}` at its catalogue defaults, {len(mesh.faces)} triangles. "
             f"Change the numbers rather than rerolling: a reroll here is an "
             f"edit, not another 40 seconds.",
    )]


def _draft_parts_from_caller(request: Request, family: Family) -> list[dict]:
    """Route the part names the caller supplied through the taxonomy.

    This is the archetype table doing its actual job. Anything the taxonomy has
    no verdict on is generated and flagged, because a part nobody recognises is
    more likely to be the sculptural one.
    """
    unit = family.size_m
    parts = []
    for name in request.parts:
        archetype = classify_part(name)
        key = _slug(name)
        if archetype is None or archetype.route == GENERATE:
            parts.append(_generated(
                key,
                f"{name.strip().rstrip('.,')}, cut off flat where it joins, "
                f"nothing attached, {WHOLE_OBJECT_VIEW}",
                round(unit * 0.3, 4), request.face_budget,
                note=(f"archetype {archetype.name!r}: {archetype.evidence}"
                      if archetype else
                      "No archetype matched this name, so it defaults to "
                      "generation. Route it yourself if it is hardware."),
            ))
        else:
            kind = archetype.kind or "plank"
            parts.append(_scripted(
                key, kind, {},
                placement={"anchor": {"to": parts[0]["name"]}} if parts else {},
                note=f"archetype {archetype.name!r}: {archetype.evidence}",
            ))
    return parts


def _wants_lean(request: Request) -> bool:
    """Whether this request wants decoration parameters turned off.

    Three ways to arrive here, and they are the same request said differently:
    the words 'low poly', a blocking target, or a budget small enough that the
    decoration would be decimated away regardless.
    """
    if request.low_poly is not None:
        return request.low_poly
    if _hits(request.text, _LOW_POLY_WORDS) or _hits(request.text, _GREYBOX_WORDS):
        return True
    target, _ = request.resolve_target()
    if target.name in ("blockout", "scenery_lod"):
        return True
    budget = request.face_budget
    return bool(budget) and budget <= 2000


def draft_plan(request: Request, family: Family, strategy: str) -> dict:
    """A `decompose.Plan` dict for this request. A starting point, not an answer.

    It validates — `decompose.validate()` is run on it before it is returned,
    so a caller can execute it unchanged — and it is wrong in at least one way
    that only the caller can fix: every `size_m` here is a family default, and
    a wrong size validates, generates, assembles and produces an aeroplane with
    a wing ten times too long.
    """
    if request.parts:
        parts = _draft_parts_from_caller(request, family)
    elif strategy == SINGLE:
        parts = _draft_single(request, family)
    elif family.name == "weapon":
        parts = _draft_weapon(request, family)
    elif family.name == "ornate_prop":
        parts = _draft_ornate_prop(request, family)
    elif family.name == "aircraft":
        parts = _draft_aircraft(request, family)
    elif family.name == "vehicle":
        parts = _draft_vehicle(request, family)
    elif family.name == "architecture":
        parts = _draft_architecture(request, family, allow_generation=strategy == HYBRID)
    elif family.name == "furniture":
        parts = _draft_furniture(request, family)
    elif family.name == "modular_kit":
        parts = _draft_modular(request, family)
    elif strategy == SCRIPTED:
        parts = _draft_modular(request, family)
    else:
        parts = _draft_single(request, family)

    if strategy == SCRIPTED:
        # A scripted strategy that still carries generated parts is not a
        # scripted strategy. Drop them rather than quietly contradicting the
        # recommendation the caller is reading.
        keep = {p["name"] for p in parts if p["mode"] != GENERATE}
        parts = [p for p in parts
                 if p["mode"] != GENERATE
                 and p.get("placement", {}).get("mirror_of", "") in keep | {""}]

    if _wants_lean(request):
        # Low-poly is a decision about the parameters, not a decimation of the
        # result. Every kind that has a facing, a bevel or a course pattern
        # takes a parameter to turn it off, and turning them off is what makes
        # 500 triangles look deliberate instead of broken.
        for part in parts:
            if part["mode"] == SCRIPT:
                part["params"] = lean_params(part["kind"], part.get("params") or {})

    style = request.style or _strip_style_leaks(family.style, request.subject)
    plan = {
        "name": request.name or _slug(request.subject),
        "subject": request.subject.strip(),
        "style": style,
        "seed": request.seed,
        "generator": decompose.DEFAULT_GENERATOR,
        "target_faces": request.face_budget,
        "textured": decompose.DEFAULT_TEXTURED,
        "parts": parts,
    }
    # The scale reference is the first part with a size, which for every recipe
    # here is the hero part. Without one, a unit is a metre — fine for a plan
    # made of primitives, and misleading for one whose hero is a unit box.
    hero = next((p for p in parts if p.get("size_m") is not None), None)
    if hero is not None and any(p["mode"] == GENERATE for p in parts):
        plan["scale_reference"] = hero["name"]
    return plan


# --- cost -------------------------------------------------------------------


def _triple(low_typ_high, count: int = 1) -> dict:
    low, typ, high = low_typ_high
    return {"low": round(low * count, 2), "likely": round(typ * count, 2),
            "high": round(high * count, 2)}


def _add(a: dict, b: dict) -> dict:
    return {k: round(a[k] + b[k], 2) for k in ("low", "likely", "high")}


_ZERO = {"low": 0.0, "likely": 0.0, "high": 0.0}


def _is_solid_subject(text: str) -> bool:
    """Whether this generated part is a solid box, for the cost curve.

    Generation cost scales with occupied volume: a dragon is mostly empty space
    and cheap, a crate fills its voxel grid and is the most expensive thing
    either generator makes. This is the difference between a 38-second part and
    a 110-second one, so it belongs in the estimate rather than in a footnote.
    """
    return bool(_hits(text, _CEILINGS_BY_CODE["solid_box_cost"].triggers))


def _script_triangles(part: decompose.Part) -> int:
    """Exact, by building it. Three milliseconds, and no other estimate is."""
    try:
        return int(len(primitives.build(part.kind, part.params).faces))
    except Exception:  # validate() reports a bad kind properly; do not mask it
        return 0


def cost(plan, model_resident: bool = False,
         high_resolution: set | None = None) -> dict:
    """What a plan will cost, computed before anything runs.

    The whole point of this function is that it is free and the thing it prices
    is not. `wall_seconds` is what the caller waits for; `gpu_seconds` is the
    generation time alone, which is the figure builds get reported in.

    Wall time is a range because generation is: 30.0-49.4 s measured across ten
    organic subjects, 78.6-151.2 s on a solid box. A single number here would
    be a lie with a decimal point on it.
    """
    plan = decompose.Plan.from_dict(plan)
    decompose.validate(plan)
    high_resolution = set(high_resolution or ())

    per_part: list[dict] = []
    wall = dict(_ZERO)
    gpu = dict(_ZERO)
    generations = 0
    peak_vram = 0.0
    triangles = 0
    largest = 0
    counts = {GENERATE: 0, SCRIPT: 0, MIRROR: 0}
    by_name: dict[str, dict] = {}

    for part in plan.parts:
        counts[part.mode] += 1
        entry = {"name": part.name, "mode": part.mode}

        if part.mode == GENERATE:
            generator = part.generator or plan.generator
            solid = _is_solid_subject(f"{part.name} {part.prompt or ''}")
            hires = part.name in high_resolution
            # Order matters: a solid subject at the high tier is the run that
            # was killed at 21 minutes, so `solid` wins the timing table and
            # `hires` wins the VRAM one.
            table = (GENERATE_SECONDS_SOLID if solid
                     else GENERATE_SECONDS_HIRES if hires else GENERATE_SECONDS)
            vram = (GENERATE_VRAM_GIB_HIRES if hires
                    else GENERATE_VRAM_GIB_SOLID if solid else GENERATE_VRAM_GIB)
            gen = _triple(table.get(generator, GENERATE_SECONDS["trellis2"]))
            overhead = JOB_OVERHEAD_SECONDS.get(generator, 20.0)
            textured = plan.textured if part.textured is None else part.textured
            colour = _ZERO if textured else _triple(COLOUR_SECONDS)
            part_wall = {
                k: round(IMAGE_SECONDS + gen[k] + colour[k]
                         + _triple(DECIMATE_SECONDS)[k] + overhead, 2)
                for k in gen
            }
            wall = _add(wall, part_wall)
            gpu = _add(gpu, gen)
            generations += 1
            peak_vram = max(peak_vram, vram.get(generator, 3.93))
            budget = int(part.target_faces or plan.target_faces)
            # A budget of 0 is "do not decimate", not "no triangles": what the
            # caller gets is the raw mesh, and its size is the thing they need
            # to see before they ask for it.
            faces = budget or RAW_FACES_TYPICAL
            entry.update({
                "generator": generator, "solid_subject": solid,
                "high_resolution": hires,
                "wall_seconds": part_wall, "gpu_seconds": gen,
                "triangles": faces,
                "decimated": bool(budget),
                "triangles_note": None if budget else (
                    f"undecimated — ~{RAW_FACES_TYPICAL:,} faces is Hunyuan3D's "
                    f"typical raw output; measured pre-decimation counts across "
                    f"the ten-subject set ran 0.48 M to 4.9 M."
                ),
                # The number that makes the argument. A generated part that
                # could have been scripted is this much slower than the
                # scripted one it should have been.
                "vs_scripted": f"{part_wall['likely']:.0f} s against ~3 ms",
            })

        elif part.mode == SCRIPT:
            faces = _script_triangles(part)
            part_wall = {"low": SCRIPT_SECONDS, "likely": SCRIPT_SECONDS,
                         "high": SCRIPT_SECONDS}
            wall = _add(wall, part_wall)
            entry.update({"kind": part.kind, "wall_seconds": part_wall,
                          "gpu_seconds": dict(_ZERO), "triangles": faces})

        else:  # MIRROR — free, but it is still a MeshPart with a budget.
            source = by_name.get(part.placement.get("mirror_of", ""))
            faces = int((source or {}).get("triangles", 0))
            entry.update({
                "mirror_of": part.placement.get("mirror_of"),
                "wall_seconds": dict(_ZERO), "gpu_seconds": dict(_ZERO),
                "triangles": faces,
                "note": "free — the source part's mesh, reflected",
            })

        entry["estimated_bytes"] = estimated_bytes(
            faces, coloured=part.mode == GENERATE and not (
                plan.textured if part.textured is None else part.textured))
        triangles += faces
        largest = max(largest, faces)
        by_name[part.name] = entry
        per_part.append(entry)

    if generations and not model_resident:
        # Hunyuan3D loads ~70 s of weights on the first call and then stays
        # resident. TRELLIS 2 never does, which is already in the per-job
        # overhead, so this only applies to the in-process generator.
        if any((p.generator or plan.generator) == "hunyuan3d"
               for p in plan.parts if p.mode == GENERATE):
            wall = {k: round(v + COLD_START_SECONDS, 2) for k, v in wall.items()}

    wall = {k: round(v + ASSEMBLE_SECONDS, 2) for k, v in wall.items()}

    part_count = len(plan.parts)
    over_budget = [p["name"] for p in per_part if p["triangles"] > ROBLOX_TRIANGLE_CAP]
    total_bytes = sum(p["estimated_bytes"] for p in per_part)

    return {
        "wall_seconds": wall,
        "wall_human": _human(wall["likely"]),
        "gpu_seconds": gpu,
        "generations": generations,
        "estimated_bytes": total_bytes,
        "estimated_size": _bytes_human(total_bytes),
        "peak_vram_gib": round(peak_vram, 2),
        "vram_headroom_gib": round(USABLE_VRAM_GIB - peak_vram, 2) if generations else None,
        "parts": {"total": part_count, "generated": counts[GENERATE],
                  "scripted": counts[SCRIPT], "mirrored": counts[MIRROR]},
        "triangles": {
            "total": triangles,
            "largest_part": largest,
            "generated": sum(p["triangles"] for p in per_part if p["mode"] == GENERATE),
            "scripted": sum(p["triangles"] for p in per_part if p["mode"] == SCRIPT),
            "mirrored": sum(p["triangles"] for p in per_part if p["mode"] == MIRROR),
        },
        "roblox": {
            "cap_per_meshpart": ROBLOX_TRIANGLE_CAP,
            # The cap is per MeshPart, so separating parts *raises* the budget.
            "effective_budget": part_count * ROBLOX_TRIANGLE_CAP,
            "largest_part": largest,
            "over_budget": over_budget,
            "welded_would_fail": triangles > ROBLOX_TRIANGLE_CAP,
            "applies": "only if this is going to Roblox. The cap is an import "
                       "rule there and a convention nowhere else.",
            "note": (
                f"{part_count} parts x {ROBLOX_TRIANGLE_CAP} = "
                f"{part_count * ROBLOX_TRIANGLE_CAP} triangles of budget. A "
                f"welded version of this model would be "
                + ("rejected outright." if triangles > ROBLOX_TRIANGLE_CAP
                   else "legal too, so multi-part here is for editability, "
                        "not for the budget.")
            ),
        },
        "per_part": per_part,
        "savings": _savings(per_part),
        "basis": (
            "Measured on the reference RTX 3080. Generation 30.0-49.4 s and "
            "3.93 GiB device-wide on organic subjects, 78.6-151.2 s and 6.88 "
            "GiB on a solid box; reference image 3.2 s; back-projected colour "
            "5.2-8.2 s; decimation 0.2-0.8 s; TRELLIS 2 reloads its weights "
            "every job, ~20 s. Scripted parts 0.8-5.5 ms and no GPU. Mirrors "
            "free."
        ),
    }


def _savings(per_part: list[dict]) -> list[str]:
    """What the scripted and mirrored parts of this plan did not cost.

    Stated as time not spent rather than as a ratio, because the complaint this
    module answers was measured in minutes.
    """
    out = []
    scripted = [p for p in per_part if p["mode"] == SCRIPT]
    mirrored = [p for p in per_part if p["mode"] == MIRROR]
    typical = IMAGE_SECONDS + GENERATE_SECONDS["trellis2"][1] \
        + JOB_OVERHEAD_SECONDS["trellis2"] + COLOUR_SECONDS[1]
    if scripted:
        saved = typical * len(scripted)
        out.append(
            f"{len(scripted)} scripted part(s) instead of generated: "
            f"{_human(saved)} not spent, and "
            f"{sum(p['triangles'] for p in scripted)} triangles instead of "
            f"{len(scripted) * ROBLOX_TRIANGLE_CAP}."
        )
    if mirrored:
        out.append(
            f"{len(mirrored)} mirrored part(s): {_human(typical * len(mirrored))} "
            f"not spent. A mirror is the source mesh reflected — no image, no "
            f"GPU, no build."
        )
    if not out:
        out.append(
            "Nothing is scripted or mirrored here. If any part of this is a "
            "strut, a band, a panel, a wheel or a plank, scripting it turns "
            f"{typical:.0f} s into 3 ms."
        )
    return out


def _bytes_human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KiB"
    return f"{n / 1024 ** 2:.2f} MiB"


def _human(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    tail = " — long enough that it is worth checking what is generated before starting" \
        if seconds >= LONG_BUILD_SECONDS else ""
    return f"{minutes:.1f} min{tail}"


# --- warnings ---------------------------------------------------------------


def warnings_for(plan) -> list[dict]:
    """Every ceiling this plan is about to walk into, before it spends GPU.

    Checked against the *generated* parts only: a scripted band cannot hit the
    body-of-revolution ceiling because nothing is being reconstructed. The
    plan-wide ceilings — scale, coverage, watertightness, the style leak — are
    reported whenever the plan generates anything at all, because they are true
    of every generated part and a caller who has not read the docs has not read
    them.
    """
    plan = decompose.Plan.from_dict(plan)
    generated = [p for p in plan.parts if p.mode == GENERATE]
    out: list[dict] = []

    for part in generated:
        text = f"{part.name} {part.prompt or ''}"
        for ceiling in CEILINGS:
            if not ceiling.triggers:
                continue
            hit = _hits(text, ceiling.triggers)
            if hit:
                out.append({
                    **ceiling.as_dict(),
                    "part": part.name,
                    "matched": hit,
                })

    if generated:
        for code in ("scale_is_destroyed", "colour_coverage", "not_watertight",
                     "unpredictable_failure", "preview_double_darkens",
                     "hard_surface_generator"):
            out.append({**_CEILINGS_BY_CODE[code].as_dict(), "part": None,
                        "matched": []})
        if len(generated) > 1:
            out.append({**_CEILINGS_BY_CODE["crops_do_not_work"].as_dict(),
                        "part": None, "matched": []})
        leaked = decompose.style_leaks(plan)
        if leaked and len(generated) > 1:
            out.append({
                **_CEILINGS_BY_CODE["style_suffix_leak"].as_dict(),
                "part": None, "matched": leaked,
            })

    # Deduplicate on (code, part): a prompt that says "window" twice is one
    # warning, not two.
    seen = set()
    unique = []
    for w in out:
        key = (w["code"], w["part"])
        if key not in seen:
            seen.add(key)
            unique.append(w)
    return order_warnings(unique)


def order_warnings(items: list[dict]) -> list[dict]:
    """Blockers first. A caller reading top-down should hit the fatal ones."""
    order = {"blocker": 0, "warning": 1, "note": 2}
    return sorted(items, key=lambda w: (order.get(w["severity"], 3), w["code"]))


# --- the recommendation -----------------------------------------------------


def _alternatives(winner: str, tally: dict, signals: list[Signal],
                  family: Family) -> list[dict]:
    """The strategies not chosen, and what would have to be true for them."""
    out = []
    for name in STRATEGIES:
        if name == winner:
            continue
        supporting = [s.as_dict() for s in signals if s.argues_for == name]
        if tally[name] == -math.inf:
            why = _ruled_out(name, family)
        elif supporting:
            why = (f"argued for by {len(supporting)} signal(s) but outweighed "
                   f"{tally[name]:.1f} to {tally[winner]:.1f}.")
        else:
            why = "nothing in this request argues for it."
        out.append({
            "strategy": name,
            "score": None if tally[name] == -math.inf else round(tally[name], 1),
            "why_not": why,
            "supporting_signals": supporting,
            "when_it_would_win": _when(name),
        })
    return out


def _ruled_out(name: str, family: Family) -> str:
    if name == SINGLE:
        return (
            f"ruled out: a {family.name!r} subject is not one sculptural whole, "
            f"or the caller named parts, or generation was forbidden. One "
            f"generation of a whole aeroplane returns one welded blob — "
            f"`objects=1` in Blender, nothing addressable, and a wrong tail "
            f"costs a reroll of everything (docs/MULTI-PART.md)."
        )
    return "ruled out: this request forbids generation, and this strategy needs it."


def _when(name: str) -> str:
    return {
        SINGLE: "the subject is one sculptural whole — a skull, a dragon, a "
                "boulder, a statue — with no hardware bolted to it and no "
                "dimension that has to be right.",
        HYBRID: "the object has both an ornamental part no formula writes and "
                "hardware that is nothing but dimensions: a chest, an aircraft, "
                "a detailed building, a weapon.",
        SCRIPTED: "the request is low-poly, a greybox, a modular kit, a stated "
                  "dimension, or more than a handful of the same thing.",
    }[name]


def _next_steps(strategy: str, plan: dict, budget: dict | None = None) -> list[str]:
    budget = budget or {}
    steps = [
        "Revise the draft. Every `size_m` is a family default and the server "
        "cannot check it; every `prompt` describes a generic member of the "
        "class, not the thing you actually want.",
        "`POST /decompose` with the revised plan, or build the parts yourself "
        "with `POST /images` + `POST /jobs` and `POST /primitives`.",
        "`POST /assemble` with the `assemble_request` that comes back, then "
        "`GET /scenes/{id}/preview` and look at it before calling it finished.",
    ]
    if strategy != SCRIPTED:
        steps.insert(1, (
            "Check each generated part with `GET /jobs/{id}/preview` — but "
            "judge quality from an unlit render, because the preview endpoint "
            "double-darkens and four of the best assets in the ten-subject set "
            "were initially written off because of it."
        ))
    if any(p["mode"] == GENERATE for p in plan["parts"]):
        steps.append(
            "Reroll on the numbers, not on taste: `decimated_from` above ~8 M "
            "or `silhouette_iou` below 0.6 means the mesh was born broken, and "
            "switching that one part to `hunyuan3d` is the measured fix."
        )
    if any("interior" in (p.get("note") or "") for p in plan["parts"]):
        steps.append(
            "`POST /hollow/primitives` for the interior liner — carving a "
            "generated shell is usually refused and costs its UVs when it is "
            "not."
        )
    if budget.get("target_assumed"):
        steps.insert(0, (
            "Say where this is going. No target was stated, so the budget "
            "fell back to Roblox's 20 000 per MeshPart — which is a Roblox "
            "import cap, not a universal one. `GET /strategy/targets` lists "
            "the alternatives; a film render wants no decimation, a distant "
            "LOD wants 1 500."
        ))
    if budget.get("lod_chain") and len(budget["lod_chain"]) > 1:
        steps.append(
            f"`POST /jobs/{{id}}/lod` with {budget['lod_chain'][1:]} once each "
            f"generated part is done. Each level is ~0.3 s of CPU off the raw "
            f"mesh already on disk, lands as its own job id, and goes into "
            f"/assemble and /export unchanged."
        )
    if budget.get("decimate") is False:
        steps.append(
            "Assemble with `use_raw: true` on every generated part, or submit "
            "with `target_faces: 0`. The plan carries the 0, but TRELLIS 2 "
            "decimates inside the node pack so the `use_raw` route is the one "
            "that is certain on either generator."
        )
    return steps


def budget_report(request: Request, plan: dict) -> dict:
    """What each generated part is decimated to, at what resolution, and why.

    Separated from `cost()` because it is a *decision* rather than an
    arithmetic consequence: the budget is chosen from stated intent, and the
    resolution is chosen from the budget plus what the subject is made of.
    """
    request = Request.from_dict(request)
    target, target_assumed = request.resolve_target()
    detail, detail_assumed = request.resolve_detail()
    faces = request.face_budget
    chain = lod_chain(target, detail) if request.lod else []

    parts = []
    for part in plan["parts"]:
        if part["mode"] != GENERATE:
            continue
        solid = _is_solid_subject(f"{part['name']} {part.get('prompt') or ''}")
        generator = part.get("generator") or plan.get("generator") \
            or decompose.DEFAULT_GENERATOR
        part_faces = part.get("target_faces") or (0 if not faces else faces)
        settings = generation_settings(target, detail, generator, solid,
                                       faces=part_faces)
        settings["target_faces"] = part_faces or None
        entry = {
            "part": part["name"],
            **settings,
            "estimated_bytes": estimated_bytes(part_faces or RAW_FACES_TYPICAL,
                                              coloured=True),
            "lod_chain": [n for n in chain if n <= (part_faces or 10 ** 9)] or None,
        }
        if entry["lod_chain"] and len(entry["lod_chain"]) > 1:
            entry["lod_cost"] = (
                f"{(len(entry['lod_chain']) - 1) * DECIMATE_LEVEL_SECONDS:.1f} s "
                f"of CPU for {len(entry['lod_chain']) - 1} extra level(s), and "
                f"no GPU — the raw mesh is already on disk as mesh_raw.glb. "
                f"POST /jobs/{{id}}/lod builds them."
            )
        parts.append(entry)

    return {
        "target": target.name,
        "target_assumed": target_assumed,
        "target_summary": target.summary,
        "target_evidence": target.evidence,
        "target_source": target.source,
        "detail": detail,
        "detail_assumed": detail_assumed,
        "faces_per_part": faces or None,
        "faces_source": ("caller (target_faces)" if request.target_faces is not None
                         else f"{target.name} / {detail}"),
        "decimate": bool(faces),
        "hard_cap": target.hard_cap,
        "watertight_matters": target.watertight_matters,
        "lod_chain": chain or None,
        "generated_parts": parts,
        "the_two_knobs": (
            "`target_faces` is what the mesh is decimated TO — cheap, "
            "reversible, and you can have several because the raw mesh is "
            "kept. Generation resolution is how much detail EXISTS before any "
            "of that, and no budget recovers what was never generated. Raising "
            "resolution is also the one setting measured to fail catastrophically "
            "rather than slowly: 1024_cascade on a solid crate was killed at 21 "
            "minutes at 96% of VRAM."
        ),
        "assumption_note": (
            "No target was stated, so this fell back to Roblox's numbers. "
            "Roblox's 20 000 is a per-MeshPart *import cap* and nothing else's "
            "— a film render wants no decimation at all, a distant LOD wants "
            "1 500, and a hero asset seen in the hand wants 80 000. Say where "
            "this is going and every number here changes."
            if target_assumed else
            f"Budget comes from the stated target {target.name!r}."
        ),
    }


def budget_warnings(request: Request, plan: dict,
                    budget: dict | None = None) -> list[dict]:
    """Where the stated intent and the chosen budget contradict each other.

    Cheap, and the alternative is finding out at import time — or worse, not
    finding out at all and shipping a hero asset that is a smeared silhouette.
    """
    request = Request.from_dict(request)
    budget = budget or budget_report(request, plan)
    target, target_assumed = request.resolve_target()
    detail, _ = request.resolve_detail()
    faces = request.face_budget
    out: list[dict] = []

    def add(code, severity, message, evidence, source):
        out.append({"code": code, "severity": severity, "message": message,
                    "evidence": evidence, "source": source, "part": None,
                    "matched": []})

    if target.hard_cap and faces and faces > target.hard_cap:
        add("over_target_cap", "blocker",
            f"{faces} triangles a part exceeds {target.name}'s hard cap of "
            f"{target.hard_cap}. This is not a quality trade-off — the import "
            f"is rejected.",
            "20 000 triangles per MeshPart, enforced by the importer. "
            "`/export` decimates over-budget parts individually rather than "
            "failing, so what you actually get is a silent second decimation "
            "you did not choose.",
            "docs/ROBLOX-EXPORT.md")

    if target_assumed:
        add("target_assumed", "warning",
            "No delivery target was stated, so this used Roblox's 20 000 — "
            "which is a Roblox import cap, not a universal one. It has been "
            "the silent default for every asset in this project.",
            "20 000 is simultaneously Roblox's per-MeshPart cap and the "
            "measured decimation sweet spot, which is why the two got "
            "conflated. Offline render: do not decimate. Distant LOD: 1 500. "
            "Hero asset in the hand: 40 000-200 000. Mobile: 4 000.",
            "docs/DECIMATION.md")

    generated = [p for p in plan["parts"] if p["mode"] == GENERATE]

    if generated and faces and faces <= 2000:
        relief = [p["name"] for p in generated
                  if (classify_part(f"{p['name']} {p.get('prompt') or ''}") or
                      Archetype("", "", "", (), "", "")).name
                  in ("ornament", "sculpture", "creature", "weapon_head")]
        if relief:
            add("budget_destroys_the_point", "warning",
                f"{relief} are being generated for their surface relief and "
                f"then decimated to {faces} triangles, which is the budget at "
                f"which surface relief is exactly what disappears. Consider "
                f"scripting these instead: a scripted part at 500 triangles is "
                f"deliberate low-poly, a decimated one is a smeared hero asset.",
                "'What dies first is fine surface relief.' At 20 000 the "
                "embossed lettering is legible; at 8 000 it is mush while the "
                "bird itself still looks fine. The rule is about detail type, "
                "not object size: a smooth rock decimates to 4k without "
                "complaint, a control panel covered in switches does not.",
                "docs/DECIMATION.md")

    if generated and not target.decimate:
        add("raw_mesh_wanted", "note",
            "This target wants the undecimated mesh. The plan below sets "
            "`target_faces: 0`, which `pipeline.py` reads as 'skip decimation' "
            "— on Hunyuan3D `mesh.glb` is then the raw mesh. TRELLIS 2 "
            "decimates inside the node pack via `target_face_num`, so 0 is not "
            "verified there; the reliable route on either generator is to "
            "assemble with `use_raw: true` on every part.",
            "Every job writes `mesh_raw.glb` alongside the decimated "
            "`mesh.glb`, and `/assemble` takes `use_raw` per part. Raw output "
            "is 0.5-4.9 M faces at 6+ MiB, so expect the file to be large.",
            "docs/DECIMATION.md")

    if generated and target.watertight_matters:
        add("watertight_needed", "blocker",
            "This target needs watertight geometry and a generated part will "
            "not be watertight, before or after decimation. Script what you "
            "can; expect a repair pass on what you cannot.",
            "Every mesh in the ten-subject organic set reports `watertight: "
            "false`. Decimation then breaks it further — 'raw meshes come out "
            "watertight; decimated ones generally do not'. Every scripted "
            "primitive is asserted watertight, winding-consistent and free of "
            "degenerate triangles.",
            "docs/DECIMATION.md")

    if faces and faces >= 40000:
        add("resolution_is_the_limit", "note",
            f"At {faces} triangles a part, the decimation target has stopped "
            f"being the limiting factor and the generation resolution has "
            f"started. Raising it is the one knob measured to fail hard rather "
            f"than gradually.",
            TRELLIS_PIPELINES["1024_cascade"]["evidence"] +
            " Raw output is 0.5-4.9 M faces, so a 40 000-200 000 budget is "
            "still a decimation of a 512-pipeline mesh — it does not add "
            "detail that was never generated.",
            "docs/QUALITY-COMPARISON.md")

    hires = [p["part"] for p in budget["generated_parts"]
             if p["settings"].get("pipeline_type") == "1024_cascade"]
    if hires:
        peak = GENERATE_VRAM_GIB_HIRES["trellis2"]
        add("resolution_will_not_fit", "blocker" if peak > USABLE_VRAM_GIB
            else "warning",
            f"{hires} are recommended at `1024_cascade` because the budget "
            f"asks for detail 512 does not supply — and that tier peaked at "
            f"{peak} GiB against {USABLE_VRAM_GIB} GiB usable on the reference "
            f"card. It does not fail fast: it thrashes. Set a 900 s timeout, "
            f"watch it, and fall back to 512 if the subject turns out to be "
            f"more solid than it looked.",
            TRELLIS_PIPELINES["1024_cascade"]["evidence"],
            "docs/QUALITY-COMPARISON.md")

    if request.lod and generated:
        add("lod_is_nearly_free", "note",
            "An LOD chain off one generation costs ~0.3 s a level and no GPU, "
            "because the raw mesh is already on disk. Three levels from one "
            "generation against three generations for three assets is a "
            "hundredfold difference.",
            "Decimation is ~0.3 s against a 40 s generation, and every job "
            "already writes `mesh_raw.glb`. `POST /jobs/{id}/lod` builds the "
            "levels; each lands as its own job id, so each goes into "
            "`/assemble` and `/export` unchanged.",
            "docs/DECIMATION.md")

    return out


def recommend(request) -> dict:
    """Pick a strategy, justify it, draft a plan, and price the draft.

    Returns the recommendation *and* everything needed to argue with it: the
    signals that produced it, the strategies that lost and why, the archetype
    verdict behind every routed part, the ceilings the draft is about to walk
    into, and what it will cost. The caller is expected to disagree sometimes —
    it knows things the server cannot.
    """
    request = Request.from_dict(request)
    family = classify_subject(request.text)
    signals = _signals(request, family)
    strategy, tally = _choose(request, family, signals)

    plan = draft_plan(request, family, strategy)
    plan_warnings = decompose.validate(decompose.Plan.from_dict(plan))
    budget = budget_report(request, plan)
    # Price what was actually recommended, not the default tier: choosing
    # 1024_cascade turns a 38-second part into a 103-second one, or into a
    # 900-second timeout, and the caller should see that in the estimate rather
    # than discover it.
    costed = cost(plan, high_resolution={
        p["part"] for p in budget["generated_parts"]
        if p["settings"].get("pipeline_type") == "1024_cascade"
    })

    routed = []
    for part in plan["parts"]:
        archetype = classify_part(f"{part['name']} {part.get('kind') or ''}")
        routed.append({
            "part": part["name"],
            "mode": part["mode"],
            "archetype": archetype.name if archetype else None,
            "why": (archetype.evidence if archetype else
                    "no archetype matched; routed by the recipe for this family"),
            "source": archetype.source if archetype else family.source,
        })

    supporting = [s.as_dict() for s in signals if s.argues_for == strategy]
    return {
        "subject": request.subject,
        "strategy": strategy,
        "family": family.name,
        "headline": _headline(strategy, family, costed, budget),
        "confidence": _confidence(family, tally, strategy),
        "reasoning": supporting,
        "scores": {k: (None if v == -math.inf else round(v, 1))
                   for k, v in tally.items()},
        "alternatives": _alternatives(strategy, tally, signals, family),
        "routing": routed,
        "budget": budget,
        "warnings": order_warnings(
            budget_warnings(request, plan, budget) + warnings_for(plan)
        ),
        "plan_warnings": plan_warnings,
        "cost": costed,
        "plan": plan,
        "draft_disclaimer": (
            "This plan is a DRAFT and it is wrong in ways only you can fix. "
            "The server has no world knowledge: it does not know how big this "
            "subject really is, what its parts are called, or that this "
            "particular one has a feature the generic version does not. Three "
            "things to change before running it — (1) every `size_m`, which is "
            "a family default and which nothing downstream can check; (2) every "
            "`prompt`, which describes a generic member of the class; (3) the "
            "part list itself, which is a common decomposition and not yours. "
            "Everything here validates, so you can also just run it and see."
        ),
        "next_steps": _next_steps(strategy, plan, budget),
        "request": request.to_dict(),
    }


def _headline(strategy: str, family: Family, costed: dict, budget: dict) -> str:
    """One sentence a caller can act on, with the price and the budget in it."""
    parts = costed["parts"]
    faces = budget["faces_per_part"]
    to = (f"{budget['target']}"
          + ("*, assumed" if budget["target_assumed"] else "")
          + (f" at {faces} triangles a part" if faces else ", undecimated"))
    if strategy == SINGLE:
        return (
            f"single: one generation, one part. {DO_NOT_DECOMPOSE.split('.')[0]}. "
            f"{costed['wall_human']}, {costed['gpu_seconds']['likely']:.0f} GPU "
            f"seconds, {costed['triangles']['total']} triangles, "
            f"{costed['estimated_size']} — for {to}."
        )
    if strategy == SCRIPTED:
        return (
            f"scripted: {parts['scripted']} primitive(s), no GPU at all. "
            f"{costed['wall_human']}, {costed['triangles']['total']} triangles, "
            f"{costed['estimated_size']}, exact to the dimensions asked for and "
            f"watertight — for {to}."
        )
    return (
        f"hybrid: {parts['generated']} generated part(s) where the detail "
        f"volume is, {parts['scripted']} scripted where the dimensions are, "
        f"{parts['mirrored']} mirrored for free. {costed['wall_human']}, "
        f"{costed['gpu_seconds']['likely']:.0f} GPU seconds, "
        f"{costed['triangles']['total']} triangles across {parts['total']} "
        f"parts, {costed['estimated_size']} — for {to}."
    )


def _confidence(family: Family, tally: dict, winner: str) -> dict:
    """How much of this is measurement and how much is a keyword table.

    Honest rather than flattering: an unrecognised subject is `low` and says
    the reason, because a confident wrong answer is worse here than a hedged
    right one.
    """
    finite = [v for k, v in tally.items() if v != -math.inf and k != winner]
    margin = tally[winner] - (max(finite) if finite else 0.0)
    if family is UNKNOWN_FAMILY:
        level, why = "low", (
            "the subject matched no family, so this is the default rather than "
            "a verdict. Say what kind of thing it is, or name its parts."
        )
    elif margin >= 3.0:
        level, why = "high", "the signals agree and the family is one this project measured."
    elif margin >= 1.0:
        level, why = "medium", f"the runner-up is within {margin:.1f} of this."
    else:
        level, why = "low", (
            f"the runner-up is within {margin:.1f}; this is close enough that "
            f"your judgement should decide it."
        )
    return {"level": level, "margin": round(margin, 1), "why": why}


# --- discovery --------------------------------------------------------------


def targets() -> dict:
    """Where an asset can be going, and what triangle budget that implies.

    The companion to the archetype table, and the correction to a bug that was
    baked in globally: 20 000 triangles is Roblox's per-MeshPart *import cap*,
    and it has been this project's universal default without ever being marked
    as an assumption. A film render wants no decimation at all; a distant LOD
    wants 1 500; a hero asset seen in the hand wants 80 000.
    """
    return {
        "targets": [
            {
                **t.as_dict(),
                "faces": {"background": t.faces[0], "prop": t.faces[1],
                          "hero": t.faces[2]} if t.decimate else None,
                "estimated_bytes": {
                    level: estimated_bytes(n)
                    for level, n in zip(DETAIL_LEVELS, t.faces)
                } if t.decimate else None,
            }
            for t in TARGETS
        ],
        "default": "roblox",
        "default_is_an_assumption": (
            "Nothing about 20 000 is universal. It is Roblox's per-MeshPart cap "
            "and, coincidentally, the measured decimation sweet spot — which is "
            "how the two came to be conflated everywhere in this repo. State "
            "your target, or describe the need in `intent` and it will be read "
            "out of the prose."
        ),
        "detail_levels": list(DETAIL_LEVELS),
        "detail_means": "How close the viewer gets, within the target's band. "
                        "A background rock and a hero rock are the same prompt "
                        "at different budgets.",
        "the_two_knobs": {
            "target_faces": (
                "What the mesh is decimated TO. Cheap (~0.3 s), reversible "
                "(the raw mesh is kept as mesh_raw.glb), and you can have "
                "several of them off one generation."
            ),
            "resolution": (
                "How much detail EXISTS before decimation — TRELLIS 2's "
                "`pipeline_type`, Hunyuan3D's `octree_resolution`. One-shot, "
                "expensive, and no budget recovers what was never generated. "
                "Cost scales with occupied volume, so a solid subject is far "
                "more expensive than a spindly one at the same setting."
            ),
        },
        "generation_tiers": {
            "trellis2": TRELLIS_PIPELINES,
            "hunyuan3d": {str(k): v for k, v in HUNYUAN_OCTREE.items()},
        },
        "decimation_ladder": {
            "353966": "6.2 MiB, raw. Unusable in an engine.",
            "40000": "704 KiB, 9x. Indistinguishable from raw.",
            "20000": "352 KiB, 18x. The sweet spot — fine relief survives.",
            "8000": "141 KiB, 44x. Silhouette perfect, fine relief lost.",
            "rule": "It is about detail TYPE, not object size: a smooth rock "
                    "decimates to 4k without complaint, a control panel "
                    "covered in switches does not.",
            "source": "docs/DECIMATION.md",
        },
        "lod": (
            "An LOD chain off one generation is nearly free — the raw mesh is "
            "already on disk and each level is ~0.3 s of CPU with no GPU. Three "
            "levels from one generation against three generations for three "
            "assets is a hundredfold difference. POST /jobs/{id}/lod builds "
            "them; each level lands as its own job id and goes into /assemble "
            "and /export unchanged."
        ),
        "watertightness": (
            "Decimation breaks it, and generated meshes never had it — every "
            "mesh in the ten-subject set reports watertight: false. Engines do "
            "not care; 3D printing and boolean operations do. Every scripted "
            "primitive is asserted watertight."
        ),
    }


def taxonomy() -> dict:
    """The whole routing table, so an agent can learn it rather than be told it.

    Same policy as `GET /primitives`: the numbers travel with the API, because
    a rule an agent has to be told about is a rule it will get wrong once per
    session.
    """
    return {
        "strategies": [
            {"name": SINGLE,
             "means": "one generation, one part",
             "when": _when(SINGLE),
             "cost": f"~{IMAGE_SECONDS + GENERATE_SECONDS['trellis2'][1]:.0f} s "
                     f"and {GENERATE_VRAM_GIB['trellis2']} GiB",
             "evidence": "Nine of ten organic subjects came back usable from "
                         "one generation at 30-49 s: dragon IoU 0.830, skull "
                         "0.867, boulder 0.870.",
             "source": "docs/WHAT-GENERATION-IS-FOR.md"},
            {"name": HYBRID,
             "means": "generated sculptural parts plus scripted hardware",
             "when": _when(HYBRID),
             "cost": "one generation per sculptural part; the hardware is free",
             "evidence": "The chest: 4 generated meshes at 151 s of GPU, 80 "
                         "scripted parts at 2 268 triangles. The Bonanza: 12 "
                         "parts, 6 generations, 4 of them mirrored.",
             "source": "docs/SHOWCASE-CHEST.md"},
            {"name": SCRIPTED,
             "means": "primitives only, no GPU",
             "when": _when(SCRIPTED),
             "cost": f"~{SCRIPT_SECONDS * 1000:.0f} ms a part, no VRAM",
             "evidence": "A crate is 4.6 ms and 1 380 triangles scripted "
                         "against 83-151 s and 20 000 generated — a factor of "
                         "eighteen thousand — and it comes out at exactly the "
                         "dimensions asked for.",
             "source": "docs/PROCEDURAL.md"},
        ],
        "do_not_decompose": DO_NOT_DECOMPOSE,
        "archetypes": [a.as_dict() for a in ARCHETYPES],
        "routes": {
            GENERATE: [a.name for a in ARCHETYPES if a.route == GENERATE],
            SCRIPT: [a.name for a in ARCHETYPES if a.route == SCRIPT],
        },
        "repeated_parts": (
            "A part that appears more than once is a mirror or a reused job id, "
            "never a second generation. The Bonanza is 12 parts and 6 "
            "generations; the chest reuses 36 primitive jobs across 80 "
            "placements and mirrors 6 more."
        ),
        "families": [f.as_dict() for f in FAMILIES],
        "ceilings": [c.as_dict() for c in CEILINGS],
        "costs": {
            "image_seconds": IMAGE_SECONDS,
            "generate_seconds": {k: list(v) for k, v in GENERATE_SECONDS.items()},
            "generate_seconds_solid": {k: list(v)
                                       for k, v in GENERATE_SECONDS_SOLID.items()},
            "job_overhead_seconds": JOB_OVERHEAD_SECONDS,
            "cold_start_seconds": COLD_START_SECONDS,
            "colour_seconds": list(COLOUR_SECONDS),
            "decimate_seconds": list(DECIMATE_SECONDS),
            "script_seconds": SCRIPT_SECONDS,
            "mirror_seconds": MIRROR_SECONDS,
            "generate_vram_gib": GENERATE_VRAM_GIB,
            "usable_vram_gib": USABLE_VRAM_GIB,
        },
        "roblox": {
            "cap_per_meshpart": ROBLOX_TRIANGLE_CAP,
            "note": "The cap is per MeshPart, not per file, so an n-part model "
                    "has an n x 20 000 budget while one welded blob over 20 000 "
                    "is rejected. The chest's carcass alone is 19 694. This is "
                    "a Roblox rule and nothing else's — see GET "
                    "/strategy/targets for the other budgets.",
            "source": "docs/ROBLOX-EXPORT.md",
        },
        "targets": targets(),
        "no_llm": (
            "There is no LLM call in this module, deliberately. It carries the "
            "measurements and the arithmetic; you carry the world knowledge. "
            "Disagree with it when you know better — a plan is data and you can "
            "edit every field of it."
        ),
    }
