# Plan

**Deadline: Aug 17, 2026 @ 12:00am CDT.** Last full working day is Sunday Aug 16.

Track: Software Development — requires a GitHub repo, a demo video, and documentation.
Judged on innovation, feasibility, scalability, UX/design.

## Win condition

A working MCP that connects to the app, where the app generates good-quality 3D models and assembles them into one object. Everything else is optional.

**Status as of Aug 6: the win condition is met in the pipeline, not yet in the app.** The MCP server is registered and connected in Claude Code, generates parts on the remote GPU, and assembles them into one addressable glTF — verified in Blender as `OBJECTS: 4` with correct names and positions. The desktop app can submit and view single parts but cannot yet drive assembly. Closing that gap is the highest-value work left.

## The hard constraint

The dev laptop shares a LAN with the GPU desktop **only through Thursday Aug 6** — today. From Friday Aug 7 onward the desktop is remote and physically unreachable.

- [x] Python env + CUDA + PyTorch working — see [SETUP-GPU.md](SETUP-GPU.md)
- [x] One 3D model generating a mesh end to end — Hunyuan3D 2.1 shape, 40.4 s, 7.63 GiB peak
- [x] FastAPI server running, reachable from the laptop
- [x] Tailscale up on both machines, verified over the tailnet
- [x] OpenSSH server enabled on Windows, key-based login from the laptop
- [x] Server starts automatically on boot — scheduled task as SYSTEM, launched 16 s after boot, no login required
- [x] Firewall scoped to `100.64.0.0/10` so the unauthenticated server is tailnet-only
- [ ] **Sleep and hibernate disabled** — the one item on this list with no artifact in the repo. Confirm today; if the box sleeps unattended, the project stops.

Use Tailscale, not port forwarding. Port forwarding exposes an unauthenticated inference server to the open internet and fights NAT and dynamic IPs for no benefit.

## Phases

| Phase | State |
| --- | --- |
| **0 — desktop days (Aug 5-6)** | Done, bar the sleep/hibernate check above |
| **1 — generation quality** | Partly done. Shape generation and decimation are tuned and measured; the prompt → reference image stage is **not built** |
| **2 — MCP** | Done. 8 tools, connected in Claude Code, progress notifications shipped |
| **3 — multi-part** | Server side done: `describe` → `assemble` → `export`. Decomposition and per-part crops are not |
| **4 — app** | Partly done. Tauri shell, three.js viewport, job list, submit form. No part tree, no export UI |
| **5 — submission** | Not started |

## What remains, Aug 6 → Aug 16

**The image stage (Phase 1's gap).** Everything downstream works and is measured, but a caller has to supply the reference image. This is the single biggest hole in the story: the pitch is that an agent asks for a plane and gets a plane, and right now the agent has to bring pictures. Options, cheapest first: accept an image URL or an agent-provided image and document that as the contract; or run a small local image model, which must load and unload around the 3D model because they cannot share 7.63 GiB of a ~8.88 GiB card.

**Decomposition (Phase 3's gap).** `assemble_parts` takes placements; nothing yet decides what the parts *are* or crops one reference image into them. The MCP tool descriptions currently push that reasoning onto the agent, which is defensible — the agent has context the server does not — but it needs to be demonstrated working end to end, not just described.

**App parity (Phase 4).** The app talks to `/jobs` only. `/assemble`, `/scenes/{id}/mesh` and `/export` exist and are exercised by MCP but have no Rust command and no UI. A part list plus an export button is the smallest change that makes the app match the win condition's wording.

**Submission (Phase 5).** Demo video, final docs pass, repo cleanup. Reserve the last day and a half; do not let this get squeezed. Documentation is an explicit judging criterion and is currently the project's strongest asset — the measured writeups ([DECIMATION.md](DECIMATION.md), [ROBLOX-EXPORT.md](ROBLOX-EXPORT.md), [MULTI-PART.md](MULTI-PART.md), [SETUP-GPU.md](SETUP-GPU.md)) should be visible from the README rather than buried.

**Risk.** The GPU box is now unreachable in person for the rest of the hackathon. Anything that requires a hands-on fix there is unrecoverable, which is why boot survival and auto-restart were finished first. Keep changes to `server/` conservative and always verify `/health` after a deploy.

## Explicitly cut

Not "later in the hackathon" — out of scope for the submission:

- Textures. Confirmed out of reach: shape alone peaks at 7.63 GiB of ~8.88 usable, and the PBR stage wants 12–16 GB.
- Bundled tiny LLM for hardware detection. The user's coding agent already is that LLM; ship `HARDWARE.md` and let it read.
- Model picker UI and a multi-model catalog. One model, documented.
- Quantized model variants.
- Fine-tuning.
- Rented GPU / hourly billing.
- Roblox Open Cloud upload. Export writes a file; the user uploads it.
- `.fbx` export. No permissively-licensed Python writer, and glTF already does everything needed here.
- macOS.
- Bundling Python into a single binary.

## Hardware reality

The GPU is an RTX 3080 with 10 GB nominal. `nvidia-smi` reporting 10240 MiB is misleading — the Windows desktop holds ~1 GB, leaving **~8.88 GiB usable**. Budget against the real number.

- Shape generation at `octree_resolution=256` peaks at **7.63 GiB**. That fits, with no room for a second resident model.
- PBR texture generation wants 12–16 GB. Cut, not deferred.
- Any image model added later must load and unload around the 3D model. Sequential residency is a requirement, not an optimization, which is why `/admin/unload` already exists.
