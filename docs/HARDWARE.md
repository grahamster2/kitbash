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
| < 4 GB | Offload forks only | No | Painful. Consider pointing at a remote GPU instead. |
| 4-6 GB | TRELLIS 2 only | No | Geometry fits at ~3.6 GiB. |
| 6-9 GB | Yes | No | The measured reference case. Both models fit; neither produces usable textures. |
| 9 GB+ | Yes | Unproven | Comfortable for geometry. Nothing here has produced a texture worth shipping. |
| No NVIDIA GPU | — | — | Use remote mode against another machine. |

**Measured**, not estimated, on an RTX 3080 with 8.88 GiB usable. Device-wide peaks, which include a ~1.12 GiB idle baseline:

| | Time | Peak VRAM | Output |
| --- | --- | --- | --- |
| Hunyuan3D 2.1 | 41-43 s | 9.27-9.34 GiB | Geometry only |
| TRELLIS 2 GGUF (Q6_K, 512) | 79-151 s | 3.58-6.88 GiB | Better geometry; **texture output failed** |

**Textures remain unsolved on this class of card.** Hunyuan3D's texture stage wants 12-16 GB and does not fit. TRELLIS 2 *fits* a texture bake in budget but produced rainbow noise on all three test props — see [QUALITY-COMPARISON.md](QUALITY-COMPARISON.md). Until that is root-caused, parts carry semantic materials instead.

Note the VRAM asymmetry: Hunyuan3D uses ~92% of the budget, TRELLIS 2 about 28% on two of three subjects. Cost scales with **occupied volume**, so a solid crate is far more expensive than an equally-sized creature that is mostly empty space.

Drop `octree_resolution` to 128 on Hunyuan3D if you are near the edge.

Drop `octree_resolution` to 128 on Hunyuan3D if you are near the edge.

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
