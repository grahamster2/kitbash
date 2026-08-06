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

## What runs at what VRAM

| VRAM | Shape generation | Textures | Notes |
| --- | --- | --- | --- |
| < 6 GB | Offload forks only | No | Painful. Consider pointing at a remote GPU instead. |
| 6-8 GB | Yes | No | Untextured meshes, usable for greyboxing and Roblox parts. |
| 10-12 GB | Yes | Marginal | Textures may need offloading. Shape generation is comfortable. |
| 16 GB+ | Yes | Yes | Full pipeline, no compromises. |
| No NVIDIA GPU | — | — | Use remote mode against another machine. |

Reference: Hunyuan3D 2.1 shape generation needs roughly 6GB; its PBR texture stage wants 12-16GB. TRELLIS 2 wants 8GB+.

## No GPU? Remote mode

The inference server does not have to run on the machine you're using. It is an HTTP service — point the app and the MCP server at another machine's address.

The recommended way to connect two of your own machines is [Tailscale](https://tailscale.com/): install on both, and each gets a stable address that works from anywhere without router configuration. Do not port-forward the inference server to the public internet; it has no authentication.

## Reference setup

The configuration this was developed against:

- **GPU machine:** Windows, RTX 3080 10GB
- **Dev machine:** Pop!_OS 24.04, Intel Core Ultra 5 226V (integrated graphics, no discrete GPU), 16GB RAM
- Connected over Tailscale
