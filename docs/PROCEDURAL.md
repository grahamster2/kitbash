# Procedural parts

**A crate is the most expensive thing either generator makes, and the cheapest
thing to write down.** That sentence is the whole argument. `server/primitives.py`
is a library of parametric game props that produces a part record indistinguishable
from a generated one, so `/assemble` and `/export` cannot tell which half of the
pipeline a part came from.

![The catalogue](images/procedural-catalogue.png)

## Why this exists: the crate is the worst case

[QUALITY-COMPARISON.md](QUALITY-COMPARISON.md) benchmarked three hard-surface
props against both generators. The crate was the most expensive subject for
both, and by a wide margin:

| Crate, image-to-3D | Wall | Peak VRAM | Faces out |
| --- | --- | --- | --- |
| Hunyuan3D 2.1 | 83.1 s (41.0 s gen + cold load) | 9.27 GiB | 19 982, decimated from 888 558 |
| TRELLIS 2, `512` / 2048 | **151.2 s** | **6.88 GiB** | 20 691, decimated from 9 762 008 |
| TRELLIS 2, recommended `1024_cascade` / 4096 | **killed at 21 min** | 9.69 GiB — 96 % of budget | — |

That document also explains *why*, and the reason is structural rather than a
tuning problem: **generation cost scales with occupied volume.** A dragon is
mostly empty space — thin wings, thin limbs — and is cheap. A crate is solid,
fills its voxel grid, and saturates the token budget. The eval that recommended
TRELLIS 2's settings measured them on a dragon. Kitbash's actual workload is
Roblox props, which are overwhelmingly solid boxes.

And after all that, the geometry still has the failures QUALITY-COMPARISON
records by eye: "every edge is slightly rounded where the reference is a crisp
90°, the large panels visibly undulate, and the brackets are mushy lumps."

The same crate, written down:

![Scripted crate](images/procedural-crate.png)

| | Generated crate | `POST /primitives {"kind": "crate"}` |
| --- | --- | --- |
| Wall time | 83–151 s | **4.6 ms** |
| VRAM | 6.88–9.27 GiB | **none — no GPU involved** |
| Faces | 20 000 (decimated from 0.9–9.8 M) | **1 380** |
| File | ~1 MiB | **25.7 KiB** |
| Dimensions | whatever came out, measured afterwards | **exactly what was asked for** |
| Watertight | not after decimation | **yes** |
| Edges | rounded, panels undulate | **flat panels, 90° corners, uniform chamfer** |
| Material | inferred from the part name | inferred from the *kind* — a crate is wood |
| Reroll | another 83 s | change a number |

Four milliseconds against eighty-three seconds is a factor of eighteen thousand.
That is not an optimisation, it is a different category of operation.

## The routing rule

**Geometric, man-made, dimensioned → script it. Organic, sculptural,
irregular → generate it.**

The two halves fail in exactly opposite directions:

| | Procedural | Generative |
| --- | --- | --- |
| Best at | boxes, cylinders, repeated structure, anything with a measurement | faces, foliage, creatures, rock, cloth, ornament |
| Worst at | anything you cannot write a formula for | flat planes, sharp creases, thin plates, exact sizes |
| Cost driver | face count you chose | occupied volume you did not |
| Failure mode | looks generic | looks melted |

A multi-part build usually wants both. Script the cart's wheels, bed and
staves; generate the horse. That is why the two paths share a job record rather
than living in separate systems — see *Interchangeability* below.

There is also a second, quieter argument. Roblox enforces **20 000 triangles per
`MeshPart`**, so a generated part spends its entire budget and a scripted one
spends 0.3–8 % of it. The whole catalogue at default parameters — all twelve
kinds — is **8 828 triangles**, less than half of what one generated crate costs.

## build123d vs. trimesh

I evaluated `build123d` properly rather than dismissing it, and **did not adopt
it.** The library is genuinely good and it installs cleanly on Python 3.12 in
one command. Three findings sank it for this use:

**1. It is not the permissive stack it looks like.** `build123d` is Apache-2.0
and `cadquery-ocp`'s wheel metadata says `License: Apache-2.0`, but the wheel
ships `libTK*.so.7.9.3` — the Open CASCADE kernel, which is **LGPL-2.1 with the
Open CASCADE exception**. The Apache label describes the Python bindings, not
the binaries that do the work. [DECIMATION.md](DECIMATION.md) explains why this
project keeps `pymeshlab` off the import path despite it being installed and
capable; taking on a bundled LGPL CAD kernel to draw boxes is the same trade in
the other direction.

**2. It costs 592 MB and 40+ transitive dependencies** — scikit-learn, sympy,
scipy, IPython, ezdxf, svgwrite — installed onto the GPU box whose environment
is already carefully pinned around Hunyuan3D. The server currently needs
`trimesh` + `numpy` + `fastapi`.

**3. Its output is the wrong shape for a triangle budget.** B-rep tessellation
is driven by a chord tolerance, not by a face count. Measured here, one filleted
pipe (`Cylinder(1,2) - Cylinder(0.8,2)`, `fillet(r=0.05)`):

| Export tolerance | Triangles |
| --- | --- |
| 1e-3 | **33 264** — over Roblox's per-mesh cap, for a single pipe |
| 1e-2 | 2 280 |
| hand-written `_revolve` | **192** |

To hit a budget you would tessellate and then decimate — reintroducing exactly
the QEM step, and the topology damage, that procedural generation exists to
avoid. It also does not hand you UVs.

The one thing build123d does better is fillets. Its chamfered box came out at 44
triangles against the 60 here, and a true rounded fillet is not something
composition can fake. That is a real loss and it is the reason to revisit this
if the library ever needs curvature continuity rather than bevels.

**So: plain `trimesh` + `numpy`, both MIT, both already dependencies.** No
boolean engine either — the test environment has no `scipy`, so `trimesh`'s
convex hull and boolean backends are unavailable anyway, and the module is
written to run on bare `trimesh` + `numpy` deliberately. Everything is built one
of three ways:

- `_box(w, h, d, chamfer)` — a chamfered box constructed vertex by vertex. With
  a uniform bevel on all twelve edges the six faces stay *rectangles* pulled in
  by the chamfer, joined by hexagonal bevel facets, and the three bevels meeting
  at each corner intersect at a single point rather than leaving a corner facet.
  32 vertices, 60 triangles, exactly.
- `_revolve(profile, sections, modulation)` — a closed `(radius, height)`
  profile swept around +Y, with an optional per-section radial multiplier. That
  multiplier is what makes a barrel staved and a column fluted without a single
  boolean, and a per-point flag keeps the modulation off the rings that should
  stay round.
- `_prism(polygon, width)` — a convex polygon extruded along X.

**Detail comes from composition, not subtraction.** A crate is a recessed core
plus corner posts plus boards; a wall's window is the four slabs around the
aperture rather than a hole cut through a slab. Both are exact, need no boolean
engine, and leave quads where a mesh boolean leaves sliver triangles. It is also
the same kitbashing idea the rest of the project is built on.

Every kind is asserted watertight, winding-consistent, free of degenerate
triangles, and dimensioned to what was requested — see
`server/tests/test_primitives.py`.

## The catalogue

Twelve kinds. Every dimension is in **studs** (1 file unit = 1 stud, matching
`/export`), and every part is centred on its bounding-box origin, matching
generated parts so `/assemble`'s placement maths does not have to branch.

| kind | what it is | material | tris | glb | build |
| --- | --- | --- | --- | --- | --- |
| `crate` | recessed panels, corner posts, boards on every face; `planks` / `frame` / `plain` | wood | 1 380 | 25.7 KiB | 4.6 ms |
| `barrel` | staved, bellied, with metal hoops | wood | 840 | 15.7 KiB | 2.9 ms |
| `cylinder` | rod or pipe (`wall_thickness > 0`), chamfered rims | metal | 192 | 4.3 KiB | 1.5 ms |
| `plank` | a dimensioned board | wood | 60 | 2.0 KiB | 0.9 ms |
| `wall_panel` | wall section with an optional window or door, and trim | stone | 720 | 13.8 KiB | 2.7 ms |
| `wheel` | chamfered tyre with hub and spokes, or a solid disc | rubber | 872 | 16.4 KiB | 2.8 ms |
| `stairs` | `blocks` (masonry) or `open` (treads on two stringers) | stone | 360 | 7.4 KiB | 1.8 ms |
| `ladder` | two rails and N rungs | wood | 1 656 | 30.3 KiB | 5.5 ms |
| `column` | base and capital; `plain` / `tapered` / `fluted` | stone | 1 600 | 29.1 KiB | 3.3 ms |
| `table` | top, four legs, optional apron | wood | 540 | 10.6 KiB | 2.6 ms |
| `bench` | seat, legs, stretcher, optional back | wood | 600 | 11.7 KiB | 3.2 ms |
| `wedge` | a ramp — the most common blocking shape in a Roblox place | stone | 8 | 1.0 KiB | 0.8 ms |

Counts are at default parameters; they move with `sections`, `plank_count`,
`steps` and so on. `config.PRIMITIVE_MAX_FACES` (default 20 000, Roblox's
per-`MeshPart` cap) refuses a parameter set that would exceed it, so a scripted
part can never be the thing that fails an import.

**The material comes from the kind**, via `server/materials.py` — a crate is
wood, a pipe is metal, a wall is stone, a wheel is rubber. That is strictly more
information than the name-keyword guess a generated part gets, because the kind
*is* the material fact. `material` and `color` on the request override it.

## The API

`GET /primitives` returns the whole catalogue — kinds, parameters, types,
defaults, units, ranges and choices — so an agent discovers the library by
calling it rather than by being told about it. Same reason the MCP tool
descriptions carry numbers instead of prose.

```json
{
  "kind": "crate",
  "summary": "Shipping crate: recessed panels, corner posts and boards.",
  "material": "wood",
  "params": [
    {"name": "width",  "type": "number", "default": 2.0, "unit": "studs",
     "minimum": 0.01, "maximum": 200.0, "description": "X extent."},
    {"name": "style",  "type": "choice", "default": "planks",
     "choices": ["planks", "frame", "plain"], "description": "..."}
  ]
}
```

`GET /primitives/{kind}` returns one entry. `POST /primitives` builds one:

```json
{
  "kind": "crate",
  "params": {"width": 3.0, "height": 2.0, "depth": 2.0, "plank_count": 4},
  "part_name": "supply_crate",
  "material": null,
  "color": null,
  "uv_scale": null
}
```

It responds with a **finished job record**, not a queued one.

Validation is strict on purpose: an unknown kind, a misspelled parameter, a
value out of range, a string where a number belongs, a boolean where a number
belongs, a fraction where a count belongs, and an opening that does not fit its
wall are all `400` with a message that names the problem and the alternatives.
Silently ignoring `widht` would produce a default-sized crate and a confused
caller.

### Interchangeability is the point

`POST /primitives` is **synchronous and never enters the queue.** The queue
exists to serialise access to a GPU this path does not touch; putting five
milliseconds of numpy behind a 40-second generation would be a bug, not a
policy. What it *does* share is the job record — same `id`, `status`, `result`
shape, same `job.json` mirrored to disk, same rehydration after a restart, same
appearance in `GET /jobs`. `type` is `"primitive"` rather than `"image_to_3d"`,
which is the only difference a client can see.

So the id goes straight into everything downstream:

```
POST /primitives {"kind": "crate"}     -> job id
GET  /jobs/{id}/describe               -> exact bounds, no guessing
POST /assemble  [{"job_id": id, ...}]  -> a named node in the scene
POST /export    {"scene_id": ...}      -> Roblox .glb
```

Thirteen scripted parts assembled through the ordinary `/assemble` path —
`OBJECTS: 13` in Blender, 12 412 triangles for the entire scene, 187 KiB:

![Assembled scene](images/procedural-scene.png)

## Honest limits

- **`/assemble` re-derives materials from the node name**, so a part built as
  `barrel` (wood) comes out `metal` — `materials.KEYWORDS` maps "barrel" to the
  gun kind. Pass `material` explicitly in the assemble call, or name the node
  something the keyword table reads correctly. The primitive's own material is
  the right answer and assembly currently throws it away; fixing that belongs in
  `assemble.py`.
- **UVs are opt-in and cost watertightness.** `uv_scale` emits a correct
  box-projection unwrap, one tile per N studs, but a hard-surface unwrap needs a
  vertex split at every seam and that ends the welded topology. It is off by
  default because nothing downstream has a texture to put on it yet.
- **No fillets, only chamfers.** See the build123d section. A bevel reads well
  on props and costs 48 extra triangles; a true rounded edge does not come out
  of composition.
- **`_prism` fan-triangulates, so it is convex-only.** Anything concave has to
  be composed out of convex pieces — which is why `stairs` stacks boxes rather
  than extruding a staircase silhouette.
- **`_combine` is a merge, not a union.** Components interpenetrate and their
  interior faces survive. Each is independently closed so the result is still
  watertight by `trimesh`'s definition, and a renderer and Roblox both cope, but
  it is not a clean single-shell solid. Components are deliberately embedded in
  each other rather than placed flush, so no vertices coincide and nothing fuses
  into a non-manifold edge.
- **These are generic.** Every crate looks like every other crate at the same
  parameters. Variation has to come from the parameters, from `color`, or from
  kitbashing several kinds together — that is the honest cost of the 18 000×
  speedup, and the reason the generator is still there.
