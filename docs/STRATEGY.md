# The decision layer

`server/strategy.py`. Every mode this project needs is built and measured —
generation, primitives, decomposition, assembly, export. What was missing is
the step in front of all of it: **nothing chose.**

`decompose.run()` demands a plan that has already committed to an approach, and
there was no way at all to say *"do not decompose this, it is a skull"* — which
for a skull is the correct answer, and splitting it would ruin it.

This is that step. One question, three answers, and the evidence attached:

| | means | when | cost |
| --- | --- | --- | --- |
| **`single`** | one generation, one part | the subject is one sculptural whole — a skull, a dragon, a boulder, a statue | ~69 s, ~3.9 GiB, one mesh |
| **`hybrid`** | generated sculptural parts + scripted hardware | the object has ornament no formula writes *and* hardware that is nothing but dimensions — a chest, an aircraft, a detailed building | one generation per sculptural part; the hardware is free |
| **`scripted`** | primitives only, no GPU | low-poly, greyboxing, modular kits, stated dimensions, or more than a handful | ~3 ms a part |

`single` is a first-class answer, not a degenerate case. It is the right answer
for the largest single category of thing a Roblox developer asks for, and
[WHAT-GENERATION-IS-FOR.md](WHAT-GENERATION-IS-FOR.md) is 363 lines of evidence
that it works: nine of ten organic subjects came back usable from one
generation in 30–49 s.

## No LLM in the server, deliberately

The same constraint [decompose.py](../server/decompose.py) is written under and
for the same reason. The agent calling this knows that a Beechcraft Bonanza is
an aircraft, that a gatehouse has a portcullis, and roughly how big a dragon
is. The server knows none of that and cannot learn it from a string.

So this does the part the server can do *better* than the agent:

- it carries every measured number in the repo and cites it,
- it computes what a plan will cost **before** it runs, in about a millisecond,
- it warns about the ceilings the generator was measured to hit, and
- it hands back a strong draft, explicitly labelled a draft.

The one criticism this exists to answer: a build took forty minutes and nobody
saw the price until it had been paid.

## The API

```
POST /strategy              request -> recommendation + draft plan + costs
GET  /strategy/archetypes   the routing taxonomy, so an agent can learn it
GET  /strategy/targets      delivery targets and their triangle budgets
POST /strategy/cost         price a plan you already have
POST /strategy/warnings     the ceilings a plan is about to walk into
POST /jobs/{id}/lod         extra detail levels off one generation
```

MCP: `plan_asset`, `part_archetypes`, `delivery_targets`, `estimate_plan_cost`,
`build_lods`.

A request is a subject plus, optionally, why you need it:

```json
{
  "subject": "an ornate treasure chest",
  "intent": "a hero prop the player opens, in Unreal, seen close up",
  "lod": true
}
```

`intent` is prose, not a form. The primary caller is an agent describing a need.

## What it recommends

Run on the seven subjects the brief named, with no `intent` and no `target`:

| subject | strategy | parts | generations | wall | triangles | confidence |
| --- | --- | --- | --- | --- | --- | --- |
| `a skull` | **single** | 1g / 0s / 0m | 1 | 69 s | 20,000 | high |
| `a dragon` | **single** | 1g / 0s / 0m | 1 | 69 s | 20,000 | high |
| `a treasure chest` | **hybrid** | 3g / 7s / 3m | 3 | 4.6 min | 51,012 | high |
| `a Beechcraft Bonanza` | **hybrid** | 3g / 5s / 4m | 3 | 3.4 min | 42,428 | high |
| `a detailed castle gatehouse` | **hybrid** | 1g / 6s / 1m | 1 | 2.4 min | 51,956 | high |
| `a low-poly medieval house` | **scripted** | 0g / 6s / 1m | 0 | 1 s | 1,756 | low |
| `a stone wall section` | **scripted** | 0g / 1s / 0m | 0 | 1 s | 2,940 | high |

`g/s/m` is generated / scripted / mirrored.

### Against measured ground truth

Two of these have documented end-to-end builds, and they are used as
regressions in `server/tests/test_strategy.py` the way a golden file is used.

**The chest** ([SHOWCASE-CHEST.md](SHOWCASE-CHEST.md)) shipped as 88 parts: 7
generated from 4 meshes, 80 scripted from 36 primitives, 6 mirrored, 151 s of
GPU. The recommender says hybrid, generates the carcass, the escutcheon and a
claw foot, and **scripts the lid** — which is the decision that cost the most
to learn. The lid was supposed to be generated; twenty candidate reference
images across five prompt strategies never once produced a long low barrel
vault at a three-quarter angle, and the one framing that worked produced a mesh
arching over its long axis that needed a (2.05, 0.76, 0.61) per-axis correction.
The archetype table encodes that under `dimensioned_surface`, so nobody
rediscovers it.

**The aircraft** ([DECOMPOSITION.md](DECOMPOSITION.md)) shipped as
`decompose.BONANZA`: 12 parts, 6 generations, 2 scripted, 4 mirrored. The
recommender says hybrid, mirrors the same 4 parts, scripts the same landing
gear — and **also scripts the flight surfaces**, taking 6 generations to 3.

That last one is a deliberate disagreement with the shipped example, and it is
the shipped example that is out of date. `tapered_panel` was added to
`primitives.py` *because* an agent built an aircraft's flight surfaces out of
this library, and its documented worked example is a 4.4 m semi-span, 2.1 m
root chord wing — the Bonanza's wing exactly. Measured at sixty stations across
the span, the panel's largest chord step is 0.0166 m; the generated wing came
back as two crossed slabs, and with the viewpoint clause became one wing that
DECOMPOSITION.md still calls "chunky", at 11,892 triangles in 49 s. The panel
is 60 triangles in 0.9 ms.

Running the recommender's own ceiling check against the *shipped* Bonanza plan
finds this without being told:

```
POST /strategy/warnings  <- decompose.BONANZA
  blocker  aerofoil_section        left_wing
  blocker  aerofoil_section        left_tailplane
  blocker  body_of_revolution      fuselage      ("rounded glass canopy")
  blocker  thin_flat_panel         left_tailplane
  warning  propeller_is_ambiguous  propeller
```

## Two knobs, and they are not the same knob

The scope of this layer grew once, and correctly:

> *"not every part needs to be under 20,000. That's pretty much only for
> Roblox designers. And that's the advantage of using an LLM in this process —
> the user describes WHY they need the parts and the LLM decides the voxels."*

That was right, and it exposed a bug baked in globally: **20,000 triangles is
Roblox's per-MeshPart import cap, and this project applied it as a universal
default without ever marking it as an assumption.** `config.PRIMITIVE_MAX_FACES`
is 20,000. `config.TRELLIS_TARGET_FACES` is 20,000. Every asset in every
document is decimated to it, including ones that were never going near Roblox.
It got that way honestly — 20,000 is *also* the measured decimation sweet spot
([DECIMATION.md](DECIMATION.md)) — and the coincidence is exactly why the two
became one number.

So the recommendation now chooses the budget too, and the budget is two
separate decisions:

| | | |
| --- | --- | --- |
| **`target_faces`** | what the mesh is decimated **to** | cheap (~0.3 s), reversible (the raw mesh is kept), and you can have several |
| **resolution** | how much detail **exists** before any of that | one-shot, expensive, and no budget recovers what was never generated |

### Targets

`GET /strategy/targets`. Per part.

| target | budget | notes |
| --- | --- | --- |
| `roblox` | 8k / 20k / 20k | the only **hard cap** here — the importer rejects over it, per MeshPart |
| `game_realtime` | 5k / 12k / 15k | Unity, Unreal, Godot. A frame-time convention, not a rule |
| `game_mobile` | 1.5k / 4k / 8k | 8k is 141 KiB against 20k's 352 KiB |
| `game_hero` | 40k / 80k / 200k | 40k is "indistinguishable from raw" |
| `scenery_lod` | 500 / 1.5k / 2k | silhouette only, which is all a distant asset needs |
| `offline_render` | — | **do not decimate.** `mesh_raw.glb` is the deliverable |
| `fabrication` | — | do not decimate, and watertightness is the whole problem |
| `blockout` | 8 / 300 / 2k | script it instead |
| `unspecified` | falls back to Roblox, **and says it assumed that** |

The three numbers are background / prop / hero — how close the viewer gets. A
background rock and a hero rock are the same prompt at different budgets, and
the recommender emits literally the same prompt for both.

`intent` is read for all of this in prose. "a film render in Blender" resolves
to `offline_render`; "distant background scenery for a mobile game" to
`scenery_lod` at 500.

### Resolution

Raising it is the one setting measured to fail *hard* rather than gradually.
TRELLIS 2's `1024_cascade` completed on the dragon at 102.7 s and 5.03 GiB —
and on a solid crate it was still inside the generate stage after **21 minutes**
at 9.69 GiB, 96% of the usable budget, with power dropping 314 W → 150 W while
pinned at 100% utilisation. That signature is memory pressure and it never
terminates on its own. It was killed, not completed.

So the recommender proposes `1024_cascade` only when the budget genuinely
exceeds what 512 supplies *and* the subject is not solid, and it attaches the
21-minute story and a `resolution_will_not_fit` blocker when it does. Cost is
priced at 102.7 s with `config.TRELLIS_TIMEOUT` (900 s) as the upper bound,
because the failure mode is a stall rather than a slowdown.

### Low-poly is a parameter decision, not a decimation

A scripted part at 500 triangles is deliberate low-poly. A generated one
decimated to 500 is a smeared hero asset — "what dies first is fine surface
relief", and at 8,000 the embossed lettering is already mush while the object
around it still looks fine.

Every kind that has a facing, a bevel or a course pattern takes a parameter to
turn it off, so for a low-poly request the draft turns them off. It does this by
**discovery** — reading the kind's own parameter spec from `primitives.KINDS`,
trying each reduction, and keeping the ones that actually lower the face count —
rather than from a table, because the catalogue is being extended and a table
would go stale. Measured effect on the draft medieval house: **34,756 → 1,756
triangles**, same parts, same dimensions.

### LOD chains are nearly free

Every job already writes `mesh_raw.glb`, and quadric decimation is ~0.3 s. So
three levels off one generation is one generation plus under a second, against
three generations for three assets — a hundredfold difference.

`POST /jobs/{id}/lod` builds them. Each level lands as an ordinary job id, so it
goes into `/assemble`, `/export` and `/jobs/{id}/describe` unchanged — the same
interchangeability rule that lets a scripted part stand beside a generated one.
Ask for `lod: true` and the recommendation returns the ladder cut to your
target: `[20000, 8000, 2000]` for Roblox, which is the measured ladder.

## The archetype taxonomy

`GET /strategy/archetypes`. The reusable core, and the thing worth keeping even
if everything else here is rewritten. Twenty archetypes, each a measured routing
verdict learned one wasted generation at a time.

**Generate** — detail volume nobody wants to write down:

| archetype | the measurement |
| --- | --- |
| `ornament` | the chest's escutcheon: a snarling beast face in deep relief, **one attempt**, IoU 0.858, 13,941 faces in 38.4 s |
| `creature` | dragon 30.9 s / IoU 0.830 with individually separated claw toes; skull IoU 0.867, best of the set |
| `organic_mass` | boulder IoU 0.870, the highest measured; stump with fluted bark ridges at 39.1 s |
| `sculpture` | gargoyle 36.9 s, folded ribbed wing membranes, claws hooked over its shoulders |
| `weapon_head` | ornate axe 30.0 s, knotwork etched into the cheek, cast beast head at the blade root |
| `shell_body` | the chest carcass, 42.3 s, IoU 0.863 |

**Script** — dimensions somebody has to get right:

`strut` · `band` · `thin_panel` · `wheel` · `plank` · `wall` · `floor` ·
`stair` · `frame` · `column` · `container` · `furniture` ·
`dimensioned_surface` · `aperture`

The sharpest three:

- **`container`** — a crate is the most expensive thing either generator makes
  (83.1 s / 9.27 GiB on Hunyuan3D, 151.2 s on TRELLIS 2, killed at 21 min at the
  recommended settings) because cost scales with occupied volume and a box is
  solid. Scripted: **4.6 ms**. A factor of eighteen thousand.
- **`thin_panel`** — the generator's worst case, measured on wings.
- **`dimensioned_surface`** — the chest lid, and twenty reference images.

**Repeated parts** are a mirror or a reused job id, never a second generation.
The Bonanza is 12 parts and 6 generations; the chest reuses 36 primitive jobs
across 80 placements.

**One sculptural whole does not get decomposed.** There are no seams in a skull,
so splitting it invents them, and each extra part is another 30–49 s generation,
another unit-box scale to declare, and another join that can be wrong.

The taxonomy has **no verdict** for a part it does not recognise, and says so
rather than guessing — the caller is the one who knows what a bilge keel is.

## Ceilings

Things the generator was measured to be unable to do at any prompt. Every one
cost real GPU time to find out and every one is invisible until after the money
is spent, which is why they are checked before it is.

| code | severity | |
| --- | --- | --- |
| `body_of_revolution` | blocker | an asymmetric surface feature must be a separate part. A fuselage's cabin hump came back as a bulge going all the way round — vertically symmetric at every station within 0.002 over the whole length |
| `aerofoil_section` | blocker | it will not produce one. Script a `tapered_panel` |
| `thin_flat_panel` | blocker | a panel seen edge-on is close to information-free |
| `cutouts_below_noise_floor` | blocker | windows, panel seams and door lines. The fuselage's six portholes were in the reference and not in the mesh |
| `scale_is_destroyed` | blocker | every mesh normalises to a unit box: 0.9923, 0.9989, 0.9997, 0.9936, 0.9989, 0.9921 |
| `propeller_is_ambiguous` | warning | "propeller" alone returns a *marine* propeller |
| `solid_box_cost` | warning | cost scales with occupied volume |
| `high_genus_cluster` | warning | TRELLIS 2 returned 12,905,884 raw faces and IoU 0.495 on the mushroom; Hunyuan3D built it correctly at 0.771. Route these to Hunyuan3D |
| `colour_coverage` | warning | median ~0.41, and near-black on a box — the chest's lock plate came back at 0.015 |
| `style_suffix_leak` | warning | naming the object in the shared suffix brings the whole object back |
| `hard_surface_generator` | note | TRELLIS 2 wins hard-surface 3–0 at 40% of the VRAM; neither wins texture |
| `not_watertight` | note | every mesh in the ten-subject set reports `watertight: false` |
| `unpredictable_failure` | note | one in ten, with `decimated_from > 8M` or `silhouette_iou < 0.6` as the tell |
| `preview_double_darkens` | note | judge from an unlit render; four of the best assets were nearly written off |
| `crops_do_not_work` | note | every crop of a Bonanza generated a complete aeroplane |

Two of these — `propeller_is_ambiguous` and `aerofoil_section` — come from the
build log rather than from a written-up experiment, and are recorded here so
they have a source to cite.

## The cost model

Computed before anything runs, from the plan alone, in about a millisecond.

Two totals, and they are not the same thing:

- **`gpu_seconds`** — generation alone. The chest reports "GPU time 151 s for
  the four generations that shipped"; that number.
- **`wall_seconds`** — what the caller waits: reference images, generation,
  colour, decimation, per-job overhead. The Bonanza reports 22.3 s to queue plus
  475 s to finish seven meshes; that number.

Both come back as a **range**, because generation is one. A single number here
would be a lie with a decimal point on it.

| | measured |
| --- | --- |
| reference image | 3.2 s (fal `flux/schnell`) |
| generation, organic | 30.0 / 38.0 / 55.0 s, 3.93 GiB device-wide |
| generation, solid box | 78.6 / 110.0 / 151.2 s, 6.88 GiB |
| generation, `1024_cascade` | 102.7 s, or the 900 s timeout, at 9.69 GiB |
| back-projected colour | 5.2–8.2 s, no VRAM |
| decimation | 0.2–0.8 s |
| per-job overhead | ~21 s (TRELLIS 2), ~2 s (Hunyuan3D) |
| cold weight load | 70 s, first call, Hunyuan3D only |
| scripted part | ~3 ms, no GPU |
| **mirror** | **nothing at all** |

Triangles for a scripted part are **measured, by building it** — three
milliseconds, and no other estimate stays true. The docs' quoted counts have
already drifted as the catalogue gained detail; `wall_panel` moved from 720 to
10,680 while this document was being written.

File size is ~18 bytes a face plus a kilobyte of container, plus ~1.34 MiB for
a back-projected atlas when there is one.

The most useful line in the output is `savings` — what the scripted and
mirrored parts of the plan *did not* cost:

```
5 scripted part(s) instead of generated: 1.1 min not spent, and 1436
  triangles instead of 100000.
4 mirrored part(s): 55 s not spent. A mirror is the source mesh reflected —
  no image, no GPU, no build.
```

## Verified against the live server

Run against the reference RTX 3080 at `100.93.141.72:8188`.

**Scripted parts.** Every scripted part of five draft plans was built over
HTTP. Where the deployed and local `primitives.py` agree, the cost model's
triangle prediction was **exact, 11 for 11** — `plank` 60, `cylinder` 192,
`tapered_panel` 60, `wheel` 872, `stairs` 36/180. File size landed within
**3.4%** on all nine.

The four that disagreed were all `wall_panel`, and that is the deployed server
running an older `primitives.py` than the working tree — which is itself the
argument for measuring rather than quoting.

**One real generation.** The `single` recommendation for "a horned beast skull"
was run end to end with its own drafted prompt and style:

| | predicted | actual |
| --- | --- | --- |
| reference image | 3.2 s | **3.3 s** |
| total wall | 60.6 / 69.4 / 89.2 s | **64.8 s** |
| `generation_seconds` | 30.0 / 38.0 / 55.0 | **53.1 s** |
| peak VRAM | 3.93 GiB | **3.93 GiB** |
| faces | 20,000 | 19,036 (from 1,064,844) |
| file size | 1.68 MiB | **1.683 MiB** |

Two corrections came out of it and are now in the model: the organic
generation band's upper bound moved from 49.4 s to 55 s to cover the 53.1 s
run, and file size gained the colour-atlas term — geometry alone would have
been out by a factor of five on a coloured part.

## Honest limits

- **The subject classifier is a keyword table**, and that is the weakest thing
  here. It covers the families this project measured plus the obvious
  neighbours; everything else falls through to `unknown`, which recommends
  `single`, reports `confidence: low`, and says why. It cannot know that a
  Beechcraft Bonanza is an aircraft except that "bonanza" is in a list. **This
  is the part the calling agent should override**, and the reason `parts` and
  `target` exist as explicit fields.
- **Every `size_m` in a draft is a family default.** Nothing downstream can
  check it. A plan that says a Bonanza's wing is 44 m validates, generates,
  assembles, and produces an aeroplane with a wing ten times too long.
- **Every drafted prompt describes a generic member of the class.** They follow
  DECOMPOSITION.md's rules — geometry not nouns, viewpoint named, style suffix
  free of subject words, verified by `decompose.style_leaks` — but a prompt the
  server wrote is a scaffold, not a description of the thing you want.
- **Placement in a draft is approximate.** Anchors are structurally correct and
  the fractions are guesses. Assemble, call `GET /scenes/{id}/ground`, and
  correct — which is what the Bonanza's gear legs needed.
- **The fourth foot problem.** `mirror_of` takes one world plane and a mirror
  of a mirror is rejected, so a four-footed object can only reach three of its
  feet by reflection. The fourth is one more entry in the `assemble_request`
  reusing the same job id, which is what the showcase chest did — but the draft
  cannot express it, so it says so in a note.
- **Resolution cannot travel in a plan.** `decompose._job_params` forwards only
  `generator`, `target_faces`, `textured` and `seed`, so a part needing a
  non-default `pipeline_type` or `octree_resolution` has to be submitted
  directly through `POST /jobs`. One line to fix, in a file this layer does not
  own.
- **"Do not decimate" is expressible but awkward.** A plan-level
  `target_faces: 0` works, because `part.target_faces or plan.target_faces`
  falls through to it and `pipeline.py` reads a falsy budget as "skip". TRELLIS 2
  decimates inside the node pack, so 0 is not verified there — the reliable
  route on either generator is `use_raw: true` at assembly.
- **The colour-atlas size is one data point.** Flat surcharge, not a curve.
- **Nothing here validates that the recommendation was *good*.** The tests
  check that it agrees with what the chest and the Bonanza measured. Whether it
  is right about a subject nobody has built is a question only building it
  answers.

## Tests

`server/tests/test_strategy.py`, 142 tests, CPU-only, no network. The
substantive ones:

- the three strategies land on eleven clear cases,
- the chest recommends hybrid **and scripts its lid**,
- the aircraft scripts its gear, its flight surfaces, and mirrors its right side,
- the chest's GPU cost lands within 0.7% of the measured 151 s,
- the Bonanza's wall time lands within 3.5% of the measured 497 s,
- the live skull's 64.8 s wall and 53.1 s generation both fall inside the
  predicted ranges,
- file size matches nine parts built on the live server within 5%,
- each ceiling fires on a prompt that walks into it,
- every draft plan validates through `decompose.validate()` and can be run
  unchanged,
- no drafted style leaks a subject word,
- `lean_params` only ever reduces, never overrides the caller, and every kind
  in the catalogue still builds after it.
