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
| `generate_part` | Reference image → `.glb` on disk |
| `describe_part` | Bounds, size and center of a finished part |
| `assemble_parts` | Compose parts into one scene, one named node each |
| `get_generation_job` | Status of a running job |
| `save_mesh` | Download a job that finished after a timeout |
| `list_generation_jobs` | Recent jobs |

The intended loop is `generate_part` per part → `describe_part` to get real
dimensions → `assemble_parts` to place them. See
[docs/MULTI-PART.md](../docs/MULTI-PART.md).

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

Assembly on its own, reusing already-completed jobs so it runs in a second:

```bash
KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/assemble-demo.mjs
```
