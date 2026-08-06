# Kitbash

Local, open-source 3D model generation for people who don't want to learn Blender — driven by your coding agent.

> *Kitbashing* is the modeling technique of building one object out of many separate parts. That's the idea: a plane isn't one mesh, it's a fuselage plus seats plus a console — each one regenerable on its own.

## The problem

If you want AI-generated 3D assets today you have two options, and both are bad:

- **Blender MCP** — let an agent drive Blender. Slow, burns tokens, and the output only gets good if you already know Blender well enough to fix it.
- **Meshy and friends** — good quality, but expensive, per-generation billing, and web-only.

Meanwhile the open-source models that power a lot of this are free, MIT/Apache licensed, and run on a consumer GPU.

## What this is

Three processes that talk over HTTP:

- **`server/`** — a FastAPI process on the GPU machine. It owns the model, a single-worker job queue, and the assemble and export stages. It is the only thing that imports torch.
- **`mcp/`** — an MCP server your coding agent connects to. Node/TypeScript, no Python and no GPU on this side; it is a thin HTTP client.
- **`app/`** — a Tauri v2 desktop app with a three.js viewport, for looking at what came out.

Because the boundary is HTTP, the GPU does not have to be the machine you're sitting at. Local generation is the case where the base URL happens to be `127.0.0.1`. The reference setup runs the server on a Windows desktop and everything else on a laptop with no discrete GPU, over Tailscale.

The unit of work is a **part**, not a model. Ask a generator for "a plane" and you get one welded blob — `objects=1` in Blender, nothing addressable. Kitbash generates each part separately and assembles them into one glTF with a named node per part, so a wrong tail costs one 40-second regeneration instead of a reroll of everything.

## What works today

Every number below is measured on the reference hardware: an RTX 3080 with 10 GB nominal, ~8.88 GiB actually usable once Windows has its share.

| | Measured |
| --- | --- |
| Shape generation | Hunyuan3D 2.1, geometry only. **40.4 s** warm, peak **7.63 GiB** VRAM at `octree_resolution=256` |
| Cold start | ~70 s of weight loading on the first call; the model then stays resident |
| Raw mesh | ~350k faces, watertight, ~6 MiB |
| Decimation | 20,000 faces is the sweet spot — 18× smaller, ~0.3 s, no visible loss ([DECIMATION.md](docs/DECIMATION.md)) |
| Assembly | 4 parts → `OBJECTS: 4` in Blender with correct names and positions, against `objects=1` for a monolithic generation |
| Roblox export | `.glb` imports natively. The 20,000-triangle cap is **per mesh**, so a 10-part model has a 200k budget while one welded blob is rejected |
| Availability | Server runs as a Windows scheduled task at boot as SYSTEM — verified launching **16 s after boot**, no login |
| Exposure | Port 8188 is firewalled to `100.64.0.0/10`, the Tailscale CGNAT range. Not reachable from the LAN or the internet |
| Materials | Each part gets a PBR material inferred from its name — `canopy` → glass, `wheel` → rubber, `engine` → metal. No VRAM, ~1 ms |
| Agent interface | 8 MCP tools, registered and connected in Claude Code |
| Tests | 120, CPU-only, ~1 s |

**Not built yet:** there is no text-to-image stage. The pipeline is image-conditioned end to end, so today the caller supplies the reference image. Generations carry semantic materials rather than generated textures — Hunyuan3D's texture stage wants 12–16 GB and does not fit. The desktop app can submit and view single parts but cannot assemble or export; those run through MCP.

**Measured but not yet integrated:** TRELLIS 2 (GGUF) produces visibly better geometry on hard-surface props at roughly 40% of the VRAM — sharp crate corners, a rectangular sword cross-guard where Hunyuan3D makes a dowel. Its **texture output failed on all three test props** (rainbow noise, confirmed in the baked atlas), and it is slower than Hunyuan3D on solid objects. Side-by-side renders and numbers in [QUALITY-COMPARISON.md](docs/QUALITY-COMPARISON.md); the earlier single-subject benchmark it corrects is [TRELLIS2-EVAL.md](docs/TRELLIS2-EVAL.md).

## Quickstart

### 1. GPU machine

Full install, including which of Hunyuan3D's pins to ignore and why, is in [docs/SETUP-GPU.md](docs/SETUP-GPU.md). Then:

```powershell
cd server
$env:KITBASH_HY3D_REPO = "D:\models\Hunyuan3D-2.1"   # where you cloned it
$env:KITBASH_OUT_DIR   = "D:\kitbash-out"            # where meshes land
.\.venv\Scripts\python.exe -m uvicorn app:api --host 0.0.0.0 --port 8188
```

Check it: `curl http://<gpu-host>:8188/health` reports free VRAM, whether the model is resident, and queue depth. It answers even when CUDA is broken — that is the point of it.

To survive reboots, run `server/install-service.ps1` elevated. It registers a scheduled task at startup running as SYSTEM and opens 8188 to the Tailscale range only.

### 2. MCP server (on the machine your agent runs on)

```bash
cd mcp
npm install && npm run build
claude mcp add kitbash --env KITBASH_SERVER_URL=http://<gpu-host>:8188 \
    -- node /absolute/path/to/kitbash/mcp/dist/index.js
```

`<gpu-host>` is `127.0.0.1` when the GPU is local, or the Tailscale address otherwise. Ask your agent to `check_gpu_server` to confirm the link. Tool reference: [mcp/README.md](mcp/README.md).

### 3. Desktop app (optional)

```bash
cd app
npm install
npm run tauri dev
```

The server URL is a field in the UI and is remembered; it seeds from `KITBASH_SERVER_URL` on first run.

### The loop

`generate_part` once per part → `describe_part` for real bounds → `assemble_parts` to place them → `export_for_roblox` with a `height_studs`. Worked example in [docs/MULTI-PART.md](docs/MULTI-PART.md).

## Layout

| Path | What lives here |
| --- | --- |
| `server/app.py` | FastAPI routes: jobs, assemble, scenes, export, health |
| `server/jobs.py` | Single-worker queue; job records mirrored to disk and rehydrated at startup |
| `server/pipeline.py` | Hunyuan3D 2.1 shape generation and decimation — the only file that imports torch |
| `server/assemble.py` | Parts → one glTF, one named node per part |
| `server/export.py` | Roblox and DCC export; enforces the per-mesh triangle budget, scale and pivot |
| `server/run.ps1`, `install-service.ps1` | Boot-at-startup task and the Tailscale-scoped firewall rule |
| `mcp/src/index.ts` | The 8 MCP tools |
| `mcp/src/client.ts` | HTTP client for the GPU server |
| `app/src/` | Frontend: job list, submit form, three.js viewport |
| `app/src-tauri/src/lib.rs` | Rust commands — every request to the server goes through here |
| `docs/` | Architecture, plan, and the measured writeups below |

## Docs

| | |
| --- | --- |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | The three processes and why the boundaries are where they are |
| [PLAN.md](docs/PLAN.md) | What is done, what remains, deadline |
| [SETUP-GPU.md](docs/SETUP-GPU.md) | Installing the inference stack on Windows without fighting the upstream pins |
| [DECIMATION.md](docs/DECIMATION.md) | How far a mesh reduces before it shows, with renders |
| [MULTI-PART.md](docs/MULTI-PART.md) | Part decomposition, placement, coordinate conventions |
| [ROBLOX-EXPORT.md](docs/ROBLOX-EXPORT.md) | Roblox's real import constraints, with sources |
| [HARDWARE.md](docs/HARDWARE.md) | What runs at what VRAM — written to be read by your coding agent |

## Status

Early. Built for [Reverie Hacks 2026](https://reverie-hacks-2026.devpost.com/) (submission Aug 17, 2026).

## Platforms

The server is developed on Windows; nothing in it is Windows-specific except the `.ps1` helpers. The MCP server and app run on Linux and Windows. macOS is not planned for now.

## License

TBD — intended to be permissive. The dependency stack is deliberately kept MIT-clean, which is why decimation uses `trimesh` + `fast-simplification` rather than the GPL `pymeshlab` sitting right next to it.
