# Architecture

Three processes that talk over HTTP. They may or may not be on the same machine — that's the whole trick.

```
┌─────────────────┐        ┌─────────────────┐        ┌──────────────────┐
│  Claude Code    │  MCP   │   mcp/          │  HTTP  │   server/        │
│  (any client)   │ ─────► │  MCP server     │ ─────► │  FastAPI + queue │
└─────────────────┘ stdio  │  (Node/TS)      │        │  (Python, GPU)   │
                           └─────────────────┘        └────────┬─────────┘
┌─────────────────┐                                            │
│  app/           │   Rust (reqwest) ───── HTTP ────────────────┘
│  Tauri desktop  │        ▲
│  three.js view  │ ───────┘ webview never speaks to the server directly
└─────────────────┘
```

## Why HTTP between everything

The dev laptop has no usable GPU; the desktop does. Rather than treating "remote GPU" as a feature to add later, the network boundary exists from the first commit. Local generation is just the case where the base URL happens to be `127.0.0.1`.

This has held up. The reference deployment now runs the server on a Windows box reachable only over Tailscale, and neither the MCP server nor the app needed a code change for it — `KITBASH_SERVER_URL` is the entire difference. The same boundary is what a future hosted/rented-GPU mode would use.

The corollary is that **clients never assume a shared filesystem**. Everything the server produces — part meshes, assembled scenes, exported files — is fetched over HTTP and written locally by whoever asked for it. `/export` returns absolute server-side paths, and `/export/file?path=` serves them back, restricted to the output directory because that endpoint takes a caller-supplied path and would otherwise serve anything on the machine.

## `server/` — inference

Python + FastAPI on port 8188. Owns the GPU and everything that needs torch, which is confined to `pipeline.py`; nothing else in the tree imports it.

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Free VRAM, whether the model is resident, queue depth, running jobs |
| `POST /jobs` | Queue an image → mesh generation |
| `GET /jobs`, `GET /jobs/{id}` | Listing and one job |
| `GET /jobs/{id}/mesh` | The `.glb` |
| `GET /jobs/{id}/describe` | Bounds, size, center of a finished part |
| `POST /assemble` | Compose finished parts into one scene |
| `GET /scenes/{id}/mesh` | The assembled `.glb` |
| `POST /export` | Write a part or scene out for `roblox` or `dcc` |
| `GET /export/file?path=` | Fetch a file `/export` produced |
| `POST /admin/unload` | Drop the model from VRAM |

### The queue is single-worker on purpose

Two concurrent generations do not fit in 8.88 GiB. The alternative to a queue is rejecting work when busy, and that is the wrong shape for this project: a multi-part build submits six parts at once and expects all six back. Accepting everything and running it one at a time is far easier for a client to reason about than backpressure the client has to implement itself.

Submit returns a job id immediately. Clients poll.

### `/health` must never raise

It reports VRAM by asking torch, and torch may be missing or CUDA may be broken — which is exactly when you are calling it. So it degrades to `gpu: null` rather than 500ing. A health endpoint that fails because CUDA is missing tells you less than one that says so.

### Job records are mirrored to disk

Each job writes `job.json` next to its mesh, and startup rehydrates them. Without this, `/jobs` is empty after every restart even though the meshes are still on disk and `/jobs/{id}` can still find them — which makes the app's history panel look like it lost your work. It also means a reboot of the GPU box (which is unattended and remote) costs nothing but the in-flight job.

Jobs recorded as `queued` or `running` are marked failed on rehydrate. No process is working on them any more, so leaving them looking in-flight is a lie the client cannot recover from.

Input images are kept out of the job record entirely. A base64 PNG would bloat both `job.json` and every API response.

### Model residency

Loading the pipeline costs ~70 s. For a multi-part build — the whole point of the project — unloading between parts would dominate the run, so the model stays resident by default (`KITBASH_KEEP_RESIDENT`). `/admin/unload` exists because the GPU is shared with whatever else the desktop is doing, and because a second model in the pipeline would need the VRAM back.

### Decimation happens on the way out

Raw output is ~350k faces. No engine will take that, so `target_faces` decimates before the mesh is written, and the dense original is kept alongside as `mesh_raw.glb`. Keeping it is nearly free and regenerating it costs another 40 s; it is also the better input for retopology or a higher-quality re-export. `assemble` can pull it back in per part via `use_raw`, so one part can carry detail the rest of the scene does not.

The implementation is `trimesh` + `fast-simplification`, both MIT. `pymeshlab` is installed, does the same job, and is **GPL** — importing it would make the server a derivative work. That constraint shapes the export path too: no `.fbx`, because there is no permissively-licensed Python FBX writer. See [DECIMATION.md](DECIMATION.md).

## Assembly

`server/assemble.py`, pure CPU, pure MIT, no GPU involved. Each part becomes a **named glTF node**, which is the entire point — names are what make parts addressable to the engine, to a human in Blender, and to a later regeneration. Duplicate names are uniquified, because two nodes called `wing` stop being addressable and silently defeat the feature.

Placement is the caller's job, not the server's. A coding agent driving this over MCP is already doing spatial reasoning and knows things the server does not — that this is a biplane, that the second wing goes above the first. What the server provides is **measurements, not guesses**: `/jobs/{id}/describe` returns real bounds so placement is computed rather than estimated.

Transform order is scale → rotate (XYZ euler degrees) → translate. Worth stating because getting it backwards is a silent failure: the model looks wrong rather than erroring.

## Export

`server/export.py` writes for a target rather than converting a format. Roblox Studio already imports `.glb` natively, so there is nothing to convert; the work is in the constraints:

- **20,000 triangles per mesh, not per file.** Each mesh node becomes one `MeshPart` and each is checked separately. So the budget is applied per geometry — a 10-part model gets 200k and spending it evenly would throw away detail nobody asked to lose. This is a second, independent argument for the multi-part assembly the project does anyway.
- **1 file unit = 1 stud.** Generated meshes normalise to ~2 units, i.e. knee-high, so `height_studs` rescales.
- **Pivot on the ground plane.** Studio drops a `MeshPart` at its mesh origin, so a centre-origin model spawns half-buried. Applied as one transform over the whole scene, so relative part placement is untouched.

The `dcc` target skips the budget and the re-origin: a DCC tool has its own opinions about units and pivots and should win. An `.obj` is written alongside both, as the format every importer has always taken, at the cost of hierarchy and vertex colours. Full sourcing in [ROBLOX-EXPORT.md](ROBLOX-EXPORT.md).

## `mcp/` — the agent interface

Node/TypeScript, deliberately *not* Python. It runs on the client machine, which may have no Python environment and no GPU, and installing a torch-adjacent environment just to shell out HTTP requests would be absurd. Two runtime dependencies, no build step beyond `tsc`.

Eight tools: `check_gpu_server`, `generate_part`, `get_generation_job`, `save_mesh`, `describe_part`, `assemble_parts`, `export_for_roblox`, `list_generation_jobs`.

**Progress notifications are load-bearing.** A cold generation is ~110 s and most MCP clients time a request out at 60 s. `generate_part` polls the job and emits `notifications/progress`, which resets that timer — without it a first call reliably fails in Claude Code. A timeout does not cancel anything either way: the job keeps running server-side, and `get_generation_job` + `save_mesh` recover it.

Tool descriptions carry the numbers an agent needs to make a decision — that 20,000 is the decimation default, that raw output is ~350k faces, that +Y is up. The agent is the user of this interface, so the docs go where it will read them.

## `app/` — desktop

Tauri v2: a Rust shell around the OS webview, with a vanilla-TypeScript frontend and a three.js viewport. Tauri rather than Electron for ~10 MB instead of ~150 MB and roughly 50 MB of RAM instead of 300–400. The viewport wants a WebGL context anyway, so the webview is doing real work rather than just hosting a UI.

### Every request goes through Rust

Not a stylistic choice. The server is a plain FastAPI app with **no CORS middleware**, and the webview's origin is `tauri://localhost`, so a `fetch` from the frontend is blocked before it leaves the process. Proxying through `reqwest` also sidesteps the webview's private-network and mixed-content rules, which matter the moment the base URL is a tailnet address instead of localhost.

Meshes come back as a Tauri `Response`, which reaches the webview as an ArrayBuffer — the GLB is never base64'd on its way to three.js.

Timeouts are split: 20 s for JSON, 120 s for a mesh, because a mesh may be crossing a tailnet from a machine that is simultaneously generating. Transport errors are translated into the failure a user actually hits ("could not reach the Kitbash server at ...") rather than surfaced raw.

The base URL lives in the frontend, in `localStorage`, seeded from the shell's `KITBASH_SERVER_URL` on first run. Changing which machine owns the GPU is a text field, not a restart.

### Polling is deliberately lazy

The server is usually across a network on a machine that is busy generating. Idle polling runs at 15 s and only tightens to 2.5 s while a job is in flight. Health is separate, every 10 s.

The viewport is tuned for reading *shape*, because generations are untextured and often not watertight: an IBL for form, double-sided materials so holes read as holes instead of black voids, a grid that rescales with the model so it works as a ruler, and shape-only white meshes tinted grey because pure white blows out under IBL and hides the surface detail you are trying to judge.

Python is **not** bundled into the binary. Packaging PyTorch + CUDA with PyInstaller is multi-gigabyte and fragile, and it would defeat the split that makes remote GPUs work at all.

## Generation pipeline

Text is not how these models work. Every strong open 3D generator is image-conditioned, so the intended shape is:

```
prompt ──► reference image ──► [crop per part] ──► image-to-3D ──► parts ──► assembled scene
```

**Today the image stage does not exist.** The server takes an image and returns a mesh; the caller supplies the image. Everything to the right of `image-to-3D` is built and measured.

The design reason for keeping the image stage explicit rather than hiding it behind text-to-3D still holds:

- The reference image is **previewable and re-rollable** — cheap to redo, unlike a 40-second 3D generation.
- Cropping *one* image per part keeps style, palette and proportion consistent across parts. Independent per-part text prompts would not.

Whatever fills that slot has to share VRAM with the 3D model, and on the reference card they cannot be resident at once — shape generation alone peaks at 7.63 GiB of ~8.88 usable. Sequential load/unload is a requirement, which is why `/admin/unload` exists ahead of the model that will need it.

## Coordinate convention

glTF is **+Y up**, +X right, +Z toward the viewer. Roblox is Y-up too, so placement carries over unchanged and the importer's defaults are already correct — worth stating because nearly every other engine pairing needs an axis flip.

Blender is Z-up and converts on import, mapping glTF `(x, y, z)` to `(x, -z, y)`. This is expected, not a bug, and it is the thing that makes people verifying an assembly in Blender conclude it is broken.

## Deployment

The GPU box is unattended and remote, so the server is registered as a **scheduled task at startup running as SYSTEM** rather than a real Windows service — a service would need NSSM or WinSW to supervise a Python process, and the task gets boot survival plus auto-restart with nothing extra installed. Verified launching 16 s after boot with no login.

Because it runs as SYSTEM, `run.ps1` sets every environment variable it needs explicitly rather than inheriting them. SYSTEM does not see the per-user variables `setx` wrote, and a server that works interactively but dies at boot is almost always that.

Port 8188 is firewalled to `100.64.0.0/10`, the CGNAT range Tailscale allocates from. The inference server has no authentication, so scoping to the tailnet means it is unreachable from the LAN, a coffee-shop network, or the internet — Tailscale uses these addresses even when the underlying path is a direct LAN connection. Port forwarding would expose an unauthenticated inference server to the open internet and fight NAT and dynamic IPs for no benefit.
