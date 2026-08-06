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
| 4-6 GB | TRELLIS 2 only | Marginal | Shape fits at 3.93 GiB; the texture bake needs ~5 GiB. |
| 6-9 GB | Yes | **Yes, via TRELLIS 2** | The measured reference case. Hunyuan3D geometry fits; its texture stage does not. |
| 9 GB+ | Yes | Yes | Comfortable either way. |
| No NVIDIA GPU | — | — | Use remote mode against another machine. |

**Measured**, not estimated, on an RTX 3080 with 8.88 GiB usable:

| | Time | Peak VRAM | Output |
| --- | --- | --- | --- |
| Hunyuan3D 2.1 | 40.4 s | 7.63 GiB | Geometry only |
| TRELLIS 2 GGUF, shape only | 21.5 s | 3.93 GiB | Geometry only |
| TRELLIS 2 GGUF, shape + PBR | ~100 s | 5.08 GiB | Geometry + real PBR textures |

**"Textures need 12–16 GB" is a fact about Hunyuan3D, not about your card.** TRELLIS 2 bakes real PBR in less VRAM than Hunyuan3D uses for geometry alone, because its three DiTs load sequentially. Details and the quantization comparison are in [TRELLIS2-EVAL.md](TRELLIS2-EVAL.md).

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
