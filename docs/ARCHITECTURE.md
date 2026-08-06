# Architecture

Three processes that talk over HTTP. They may or may not be on the same machine — that's the whole trick.

```
┌─────────────────┐        ┌─────────────────┐        ┌──────────────────┐
│  Claude Code    │  MCP   │   mcp/          │  HTTP  │   server/        │
│  (any client)   │ ─────► │  MCP server     │ ─────► │  FastAPI + queue │
└─────────────────┘ stdio  │  (Node/TS)      │        │  (Python, GPU)   │
                           └─────────────────┘        └────────┬─────────┘
┌─────────────────┐                                            │
│  app/           │ ───────────────── HTTP ────────────────────┘
│  Tauri desktop  │
│  viewport       │
└─────────────────┘
```

## Why HTTP between everything

The dev laptop has no usable GPU; the desktop does. Rather than treating "remote GPU" as a feature to add later, the network boundary exists from the first commit. Local generation is just the case where the base URL happens to be `127.0.0.1`.

This same boundary is what a future hosted/rented-GPU mode would use. One abstraction, two payoffs.

## `server/` — inference

Python + FastAPI. Owns the GPU and everything that needs PyTorch.

- **Job queue.** Generation is slow (tens of seconds to minutes) and clients must not block. Submit returns a job id; clients poll or stream progress.
- **One model resident at a time.** On a 10GB card you cannot hold a text-to-image model and a 3D model in VRAM simultaneously. The queue owns load/unload; nothing else touches the GPU.
- **Artifacts on disk**, served over HTTP. Clients never assume a shared filesystem.

## `mcp/` — the agent interface

Node/TypeScript, distributed so it can be run with `npx`. Deliberately *not* Python: it runs on the client machine, which may have no Python environment and no GPU. It is a thin HTTP client with no heavy dependencies.

## `app/` — desktop

Tauri v2 (Rust shell + OS webview) with a React frontend and a three.js viewport.

Tauri rather than Electron: ~10MB installer instead of ~150MB, and roughly 50MB of RAM instead of 300-400MB. The 3D viewport wants a WebGL context anyway, so the webview is doing real work rather than just hosting a UI.

Python is **not** bundled into the binary. Packaging PyTorch + CUDA with PyInstaller is multi-gigabyte and fragile. Onboarding provisions an environment with `uv` instead — the same approach ComfyUI's desktop app takes.

## Generation pipeline

Text is not actually how these models work. Essentially every strong open 3D generator is image-conditioned, so:

```
prompt ──► reference image ──► [crop per part] ──► image-to-3D ──► parts ──► assembled scene
```

The user experiences text-to-3D. Internally there's an image stage, and exposing it turns out to be an advantage:

- The reference image is **previewable and re-rollable** — cheap to redo, unlike a 3D generation.
- Cropping *one* image per part keeps style, palette, and proportion consistent across parts. Independent per-part text prompts would not.

The image stage is swappable: a local model by default, or bring your own image / API key.

## Part decomposition

The agent decides how many parts an object warrants — a sword is one, a plane is many. Each part is independently addressable so it can be regenerated in place without disturbing the rest of the scene.
