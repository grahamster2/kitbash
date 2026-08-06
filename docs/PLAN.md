# Plan

**Deadline: Aug 17, 2026 @ 12:00am CDT.** Last full working day is Sunday Aug 16.

Track: Software Development — requires a GitHub repo, a demo video, and documentation.
Judged on innovation, feasibility, scalability, UX/design.

## Win condition

A working MCP that connects to the app, where the app generates good-quality 3D models and assembles them into one object. Everything else is optional.

## The hard constraint

The dev laptop shares a LAN with the GPU desktop **only through Thursday Aug 6**. From Friday Aug 7 onward the desktop is remote and physically unreachable.

Everything requiring hands on the desktop must be finished before then:

- [ ] Python env + CUDA + PyTorch working
- [ ] One 3D model generating a mesh end to end
- [ ] FastAPI server running, reachable from the laptop
- [ ] Tailscale up on both machines, verified over the tailnet (not just LAN)
- [ ] OpenSSH server enabled on Windows, key-based login from the laptop
- [ ] **Sleep and hibernate disabled** on the desktop — if it sleeps while unattended, the project stops
- [ ] Server set to start automatically on boot, so a power blip is recoverable

Use Tailscale, not port forwarding. Port forwarding exposes an unauthenticated inference server to the open internet and fights NAT and dynamic IPs for no benefit.

## Phases

**Phase 0 — Aug 5-6 (desktop days).** Everything in the list above. This is the only irreversible window; nothing else competes with it for time.

**Phase 1 — generation quality.** One model, tuned until output is genuinely good. A single model that produces convincing meshes beats a catalog of five that produce mush. Prompt → reference image → mesh.

**Phase 2 — MCP.** Tools for submitting a generation, polling status, listing and fetching results, and reporting hardware. This is the win condition; it gets whatever time it needs.

**Phase 3 — multi-part.** Decompose a prompt into parts, generate from crops of one shared reference image, assemble into a scene, regenerate individual parts in place.

**Phase 4 — app.** Tauri shell, three.js viewport, part tree, export to file.

**Phase 5 — submission.** Demo video, documentation, repo cleanup. Reserve the last day and a half; do not let this get squeezed.

## Explicitly cut

Not "later in the hackathon" — out of scope for the submission:

- Bundled tiny LLM for hardware detection. The user's coding agent already is that LLM; ship `HARDWARE.md` and a `check_hardware` MCP tool instead.
- Model picker UI and a multi-model catalog. One model, documented.
- Quantized model variants.
- Fine-tuning.
- Rented GPU / hourly billing.
- Roblox Open Cloud upload. Export writes a file; the user uploads it.
- macOS.
- Bundling Python into a single binary.

## Hardware reality

The GPU is an RTX 3080 with 10GB.

- Shape generation fits comfortably (~6GB).
- PBR texture generation officially wants 12-16GB. **Treat textures as a stretch goal**, not a dependency. Offloading forks may fit it; don't build the demo on that assumption.
- The image model and the 3D model cannot be resident at once. Sequential load/unload is a requirement, not an optimization.
