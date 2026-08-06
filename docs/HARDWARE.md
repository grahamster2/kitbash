# Hardware

This file is written to be read by *your coding agent*. Point Claude Code (or similar) at this repo and ask it to set you up — it can detect your hardware and pick from the table below.

## Detecting what you have

```bash
# NVIDIA
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# System RAM
free -g              # Linux
wmic ComputerSystem get TotalPhysicalMemory   # Windows
```

VRAM is the number that matters. System RAM matters only if you intend to use CPU offloading, which is slow enough to be a last resort.

**Budget against usable VRAM, not the number on the box.** A 10 GB card with displays attached reports 10240 MiB and gives you about **8.88 GiB** — Windows keeps the rest. That gap is wide enough to change which row you are on.

## What runs at what VRAM

Usable VRAM, not nominal.

| Usable VRAM | Shape generation | Textures | Notes |
| --- | --- | --- | --- |
| < 6 GB | Offload forks only | No | Painful. Consider pointing at a remote GPU instead. |
| 6-9 GB | Yes | No | Untextured meshes. This is the measured reference case — see below. |
| 9-12 GB | Yes | No | Shape generation is comfortable. The PBR stage still does not fit. |
| 16 GB+ | Yes | Yes | Full pipeline, no compromises. |
| No NVIDIA GPU | — | — | Use remote mode against another machine. |

**Measured**, not estimated: Hunyuan3D 2.1 shape generation peaks at **7.63 GiB** at `octree_resolution=256` and takes 40.4 s warm. Its PBR texture stage wants 12–16 GB and is out of reach below that — an RTX 3080 does not get textures, however the marketing number reads. Drop `octree_resolution` to 128 if you are near the edge.

TRELLIS 2 officially asks for 24 GB; GGUF quantization reportedly brings that to ~6 GB at Q4 and ~9 GB at Q8, with textures and UVs included. That is unverified here — see [TRELLIS2-EVAL.md](TRELLIS2-EVAL.md) if it exists.

If you need the last gigabyte and your CPU has integrated graphics, driving the displays from the iGPU frees the discrete card entirely.

## No GPU? Remote mode

The inference server does not have to run on the machine you're using. It is an HTTP service — point the app and the MCP server at another machine's address.

The recommended way to connect two of your own machines is [Tailscale](https://tailscale.com/): install on both, and each gets a stable address that works from anywhere without router configuration. Do not port-forward the inference server to the public internet; it has no authentication.

## Reference setup

The configuration this was developed against:

- **GPU machine:** Windows 11, RTX 3080 10 GB nominal / ~8.88 GiB usable
- **Dev machine:** Pop!_OS 24.04, Intel Core Ultra 5 226V (integrated graphics, no discrete GPU), 16 GB RAM
- Connected over Tailscale

Every measurement in these docs comes from that pair, with the laptop doing no
GPU work at all.
