# Kitbash MCP server

Lets a coding agent generate 3D assets without leaving the editor.

No Python and no GPU needed on this side — all generation happens in the
Kitbash server process, which this talks to over HTTP. That server can be on
localhost or on a machine across the internet; only `KITBASH_SERVER_URL`
changes.

## Install

```bash
npm install
npm run build
```

## Register with Claude Code

```bash
claude mcp add kitbash --env KITBASH_SERVER_URL=http://<gpu-host>:8188 \
    -- node /absolute/path/to/kitbash/mcp/dist/index.js
```

`<gpu-host>` is `127.0.0.1` when the GPU is local, or the Tailscale address of
the GPU box when it is not.

## Tools

| Tool | Purpose |
| --- | --- |
| `check_gpu_server` | Reachability, free VRAM, queue depth |
| `generate_reference_options` | Several reference images for one subject, **for the user to choose between** |
| `get_reference_options` | Re-show an earlier batch, spending nothing |
| `choose_reference` | Pop a picker, where the client supports elicitation |
| `generate_part` | Reference image → `.glb` on disk |
| `describe_part` | Bounds, size and center of a finished part |
| `assemble_parts` | Compose parts into one scene, one named node each |
| `export_for_roblox` | Write out under Roblox's import constraints |
| `get_generation_job` | Status of a running job |
| `save_mesh` | Download a job that finished after a timeout |
| `list_generation_jobs` | Recent jobs |

The intended loop is `generate_reference_options` → the user picks →
`generate_part` per part → `describe_part` to get real dimensions →
`assemble_parts` to place them. See
[docs/MULTI-PART.md](../docs/MULTI-PART.md).

## Choosing the reference

The generator does not read the prompt, it reads the picture — so the reference
decides the mesh, and picking it is the highest-leverage thing a human does in
this pipeline. `generate_reference_options` generates four candidates
concurrently (~6 s, four billed image calls) and returns **all of them as
images** in one tool result, each labelled with its `image_id`.

The flow the tool description teaches, and the step to not skip is the third:

1. `generate_reference_options` — supply `variants`, one full prompt per
   candidate. Four different ideas beat four re-rolls of one, and you know what
   the subject could be; the server does not.
2. **Show them to the user**, with a sentence describing each.
3. **Ask which one, and wait.** `choose_reference` pops a selection list if the
   client supports MCP elicitation (Claude Code does); otherwise just ask in
   chat and map "the second one" back to an `image_id` from the labels.
4. `generate_part` with `image_id` set to their answer.

**Do not choose on the user's behalf when the user is there to ask.** An agent
that picks silently and reports a finished mesh has skipped the only judgement
call in the pipeline it is worse at than the human.

Full write-up, with measurements and the two variation modes:
[docs/REFERENCE-SELECTION.md](../docs/REFERENCE-SELECTION.md).

## Things worth knowing

**Generation outlives the default MCP timeout.** A warm generation is ~40s; a
cold one adds ~70s of weight loading. Most MCP clients time a request out at
60s, so `generate_part` emits progress notifications, which reset that timer.
Clients must request progress (`_meta.progressToken`) for this to work — Claude
Code does.

**A timeout does not cancel the job.** Work continues on the server. Poll with
`get_generation_job` and then pull the result with `save_mesh`.

**Geometry only, no textures.** The reference GPU cannot fit the texture
pipeline; see `docs/SETUP-GPU.md`.

**Meshes come out dense** — 300k+ faces, far too heavy for a game engine.
Decimation before import is not optional.

**Placement is glTF convention: +Y is up.** Roblox is Y-up too, so placement
carries over unchanged. Blender is Z-up and converts on import.

## Smoke tests

A real generation against the configured server, optionally decimated:

```bash
KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/smoke.mjs reference.png 20000
```

The reference-selection flow, checking that all four candidates really arrive as
MCP image content with their ids attached:

```bash
KITBASH_SERVER_URL=http://<gpu-host>:8188 \
    node scripts/reference-smoke.mjs "an ornate treasure chest"
```

Four billed image calls. `--batch <batch_id>` re-shows an existing batch instead
and spends nothing.

Assembly on its own, reusing already-completed jobs so it runs in a second:

```bash
KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/assemble-demo.mjs
```

Assemble and export for Roblox:

```bash
KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/export-demo.mjs
```
