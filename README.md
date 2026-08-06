# Kitbash

Local, open-source 3D model generation for people who don't want to learn Blender — driven by your coding agent.

> *Kitbashing* is the modeling technique of building one object out of many separate parts. That's the idea: a plane isn't one mesh, it's a fuselage plus seats plus a console — each one regenerable on its own.

## The problem

If you want AI-generated 3D assets today you have two options, and both are bad:

- **Blender MCP** — let an agent drive Blender. Slow, burns tokens, and the output only gets good if you already know Blender well enough to fix it.
- **Meshy and friends** — good quality, but expensive, per-generation billing, and web-only.

Meanwhile the open-source models that power a lot of this are free, MIT/Apache licensed, and run on a consumer GPU.

## What this is

A desktop app plus an MCP server. Your coding agent asks for a model; the app generates it locally on your own GPU and hands back a mesh.

- **MCP-first.** Claude Code (or any MCP client) can request assets mid-build. This is the point of the project, not a bolt-on.
- **Multi-part generation.** "Build a plane" fans out into separate generations for the fuselage, the seats, the cockpit — assembled into one scene. Don't like the seats? Regenerate the seats, not the plane.
- **Local and free.** Open-source models on your own hardware. No per-generation billing.
- **Your hardware, matched.** The repo tells your coding agent what your machine can actually run.

## Status

Early. Built for [Reverie Hacks 2026](https://reverie-hacks-2026.devpost.com/) (submission Aug 17, 2026).

## Layout

| Path | What lives here |
| --- | --- |
| `server/` | Python inference server — job queue, model loading, generation. Runs on the GPU machine. |
| `mcp/` | MCP server. Runs on the client machine, talks HTTP to `server/`. |
| `app/` | Tauri v2 desktop app — 3D viewport, part tree, export. |
| `docs/` | Architecture, hardware compatibility, plan. |

The three pieces talk over HTTP, so the GPU does not have to be the machine you're sitting at.

## Platforms

Linux and Windows. macOS is not planned for now.

## License

TBD — intended to be permissive.
