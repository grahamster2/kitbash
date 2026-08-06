# Multi-part generation

The single feature this project exists for.

## The problem with one-shot generation

Ask any image-to-3D model for "a plane" and you get **one welded blob**. Import
it into Blender and it reports:

```
objects=1  materials=0
```

Nothing is addressable. You cannot move the wings, swap the engine, or fix the
tail, because there are no wings, engine or tail — there is one mesh. If any
part of it is wrong, your only option is to regenerate the whole thing and hope
the reroll is better everywhere else too.

## What we do instead

Generate each part separately, then compose them:

```
prompt → reference image → crop per part → image-to-3D per part → assemble
```

The assembled glTF has one **named node per part**:

```
OBJECTS: 4
  - body         polys=8000  loc=(+0.00,+0.00,+0.00)
  - left_wing    polys=8000  loc=(-1.20,-0.30,+0.00)
  - right_wing   polys=8000  loc=(+1.20,-0.30,+0.00)
  - tail         polys=8000  loc=(+0.00,-0.50,-1.10)
```

Which buys three things:

- **Per-part regeneration.** The tail is wrong? Regenerate the tail. 40 seconds,
  and everything else is untouched.
- **Parts are editable downstream.** Separate objects in Blender, separate
  MeshParts in Roblox.
- **Parts are reusable.** The same generated wheel goes on four vehicles.

## Placement is the caller's job

`assemble_parts` does not guess where parts go. A coding agent driving this over
MCP is already doing spatial reasoning, and it has context the server does not —
it knows this is a biplane and that the second wing goes above the first.

What the server provides is **measurements, not guesses**: `describe_part`
returns real bounds, size and center so placement is computed rather than
estimated.

## Coordinate convention

glTF: **+Y is up**, +X right, +Z toward the viewer.

Blender and Roblox are Z-up and convert on import, so a part placed at `y=2`
here lands at `z=2` there. This is consistent and predictable, but it does mean
a part placed at `[0, -1.1, 0.5]` shows up in Blender at `(0, -0.5, -1.1)`.

Stack along Y.

## Transform order

Scale, then rotate (XYZ euler degrees), then translate. The usual order, but
worth stating because getting it backwards is a silent failure — the model
looks wrong rather than erroring.

## Detail budget per part

Each part carries its own face budget, which is a real advantage: the part
someone will look at closely can be dense while background parts are cheap. Pass
`use_raw` on a part to assemble from its pre-decimation mesh.

See [DECIMATION.md](DECIMATION.md) for how far each part can be reduced.
