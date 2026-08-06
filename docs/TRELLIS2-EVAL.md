# TRELLIS 2 (GGUF) evaluation — RTX 3080 10GB

**Verdict: it works, and it is not close. TRELLIS 2 does shape *and* PBR texture in ~100 s at
a 5.08 GiB peak — half the VRAM Hunyuan3D uses for shape alone. Adopt it.**

Evaluated 2026-08-06 against `Aero-Ex/ComfyUI-Trellis2-GGUF` on the Windows GPU box.
Everything lives in `D:\trellis2` and touches nothing existing.

## Measurements

Test object: single dragon, `microsoft/TRELLIS` example image, 719×719 RGBA, 12 steps per
stage, seed 42, Xatlas UV unwrap, Cumesh simplify to ~200k faces. Peak VRAM is device-wide
(sampled from `torch.cuda.mem_get_info` at 20 Hz), so it includes the CUDA context, not just
tensors. Usable budget is **8.88 GiB** — Windows holds the rest.

| Config | Generate | Post+UV+bake | **Total** | **Peak VRAM** | Output |
| --- | --- | --- | --- | --- | --- |
| Q8, shape only, 512 | 21.3 s | 0.0 s | **21.5 s** | **3.93 GiB** | 771k faces, no texture |
| Q8, 512, tex 2048 | 77.1 s | 13.4 s | **90.7 s** | **3.93 GiB** | 196k faces + PBR |
| Q8, 1024_cascade, tex 4096 | 89.5 s | 39.7 s | **129.3 s** | **5.04 GiB** | 197k faces + PBR |
| Q6_K, 1024_cascade, tex 4096 | 62.0 s | 40.7 s | **102.7 s** | **5.03 GiB** | 197k faces + PBR |
| Q4_K_M, 1024_cascade, tex 4096 | 59.5 s | 40.6 s | **100.2 s** | **5.08 GiB** | 196k faces + PBR |

Against the incumbent: Hunyuan3D does **shape only, 40 s, 7.63 GiB**. TRELLIS 2 does shape
only in **21.5 s at 3.93 GiB**, and shape + full PBR in **~100 s at 5.08 GiB**.

### Does Q4/Q5/Q6/Q8 fit in 8.88 GiB?

**All of them, with ~3.8 GiB to spare.** The question turns out to be the wrong one:

> **Peak VRAM is set by the post-process/UV-bake stage (~5.0 GiB), not by the quantization
> level.** Q4 → Q8 moves peak by 0.05 GiB (5.03–5.08 GiB), because the three 1.3B DiTs load
> and unload **sequentially** under `low_vram=True`. Only one is resident at a time, and the
> generate stage peaks at 3.55 GiB (Q4) to 3.93 GiB (Q8) — below the bake stage either way.

So quantization is a **disk and speed** knob here, not a VRAM knob. Q8 costs 1.43 GB/model on
disk vs Q4's 0.79 GB. Q8 was *slower* to generate than Q4/Q6 (89.5 s vs ~60 s) — Q8_0 dequant
is not free — so **Q6_K is the sweet spot**: Q4 speed, closer-to-Q8 fidelity, same VRAM.

Unquantized BF16 was **not** measured. The GGUF loader's "GGUF BF16" option looks for files
that are not in the expected repo layout. Extrapolating from weights alone (2.58 vs 1.43 GB)
it would likely land near 5.1 GiB and still fit — but that is an inference, not a number.

### Mesh quality: yes, real PBR and real UVs

Verified by reloading the exported GLB from disk, not by trusting the in-memory object:

- `TextureVisuals` / `PBRMaterial`, UVs present and spanning the full 0–1 range.
- **baseColorTexture** 4096×4096, mean RGB (149, 87, 68), std (65, 55, 47) — real content.
- **metallicRoughnessTexture** 4096×4096, correctly glTF-packed: R=0 unused, G=roughness
  (mean 131), B=metallic (mean 144). Not a flat placeholder.
- No normal/emissive/occlusion maps are produced.
- Meshes are **not watertight**. `target_face_num` is a node input, so Roblox-facing
  decimation can happen in-pipeline rather than as a post-step.

## Install cost — the parts that actually hurt

### The version mismatch is real but not fatal

`D:\ComfyUI` runs **Python 3.12.10 + torch 2.6.0+cu124**. The node pack ships prebuilt wheels
for **torch 2.7/2.8**. A separate venv is required — but not for the reason the guides claim.

> **The READMEs say the wheels are Python 3.11 only. That is out of date.** The repo ships
> **cp312** wheels for Windows under `Torch270/`, `Torch280/`, and `Torch2100/`. Python 3.12
> is fine. Only the **torch minor version** actually forces the split.

Built `D:\trellis2\venv` on Python 3.12.10 + torch 2.8.0+cu128. All six compiled wheels
(`cumesh`, `nvdiffrast`, `nvdiffrec_render`, `flex_gemm`, `o_voxel`, `custom_rasterizer`)
installed and imported clean. **No CUDA toolkit or MSVC build step was needed.**

### The blocker that cost the most time

The node pack has an **undocumented hard dependency on
[city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)**. Neither README mentions it.

`gguf_utils.convert_to_ggml()` opens with `if not HAS_GGUF_OPS: return module` — and
`HAS_GGUF_OPS` is only set when `custom_nodes/ComfyUI-GGUF` exists on disk. Without it the
GGUF layers are **silently left unconverted**, and loading dies hundreds of lines deep in
shape-mismatch errors that look nonsensical:

```
"blocks.29.mlp.mlp.0.weight" ... dimensions in the model are torch.Size([8192, 1536])
and whose dimensions in the checkpoint are torch.Size([8192, 1536]),
an exception occurred : (The size of tensor a (1536) must match the size of tensor b (1632))
```

The stated shapes match. The numbers that don't (1536→1632, 8192→8704) are **Q8_0 block-padded
byte counts**: 1536/32 = 48 blocks × 34 bytes = 1632. That is the tell that dequantization
never ran. Fix is one `git clone` into `custom_nodes`.

### Undocumented dependencies, in discovery order

Beyond both `requirements.txt` files: `triton-windows`, `plyfile`, `zstandard`, `easydict`,
`igraph`, `xatlas`, `onnxruntime`. Also **`torchaudio` must be pinned to match torch** —
ComfyUI's `requirements.txt` pulls 2.11.0, which fails `ctypes` load against torch 2.8.0 with
`OSError: [WinError 127]` before any TRELLIS code runs.

### DINOv3 gating is a non-issue

The base README says you need the **gated** `facebook/dinov3-vitl16-pretrain-lvd1689m`.
You don't: `Aero-Ex/Trellis2-GGUF` re-hosts it ungated at
`Vision/dinov3-vitl16-pretrain-lvd1689m.safetensors`, and `model_manager.py` fetches it
automatically from the ungated `Aero-Ex/Dinov3`. **No HF token, no license click-through.**

### Fork choice

Use **`Aero-Ex/ComfyUI-Trellis2-GGUF`**. `JanskNeh/ComfyUI-Trellis2-GGUF` is stale (last commit
2026-04-13); Aero-Ex was active 4 days before this eval. Both READMEs' changelogs stop at
2026-02-21 and are simply not maintained — judge by commit date, not by changelog. Upstream
`visualbruno/ComfyUI-Trellis2` has no GGUF support but is the source of the Windows wheels,
which the GGUF forks omit (they ship Linux wheels only).

### Footprint

`D:\trellis2` totals **28.6 GB** — 19.2 GB models (Q4+Q6+Q8 sets, both 512 and 1024), 9.2 GB
venv. A single-quant deployment is ~7 GB of models. D: has 206 GB free.

## Integration path for Kitbash

**The nodes drive fine from plain Python — no ComfyUI server, no workflow JSON.** Every number
above came from a script that calls `nodes.init_extra_nodes()` and then invokes the node
classes directly out of `NODE_CLASS_MAPPINGS`. So this drops into the existing FastAPI worker
shape rather than requiring a ComfyUI graph runtime:

```python
pipe, = N["Trellis2LoadModel_GGUF"]().process(
    modelname="TRELLIS.2-4B", model_format="GGUF Q6_K",
    backend="sdpa", device="cuda", low_vram=True, keep_models_loaded=False)
img,  = N["Trellis2PreProcessImage_GGUF"]().process(image=t, padding=0, remove_background=False)
mesh, bvh = N["Trellis2MeshWithVoxelGenerator_GGUF"]().process(...)
tri, base, mr = N["Trellis2PostProcessAndUnWrapAndRasterizer_GGUF"]().process(...)
```

Use `backend="sdpa"` — flash-attn is not installed and is not needed.

Costs to accept:
- **A second venv and a second process.** transformers is pinned to `==5.2.0` here vs the
  Kitbash server's 5.14.1, and torch differs (2.8.0 vs 2.11.0). Not reconcilable in one env.
- **Both models cannot be resident at once.** 7.63 + 5.08 GiB > 8.88 GiB. The existing
  `/admin/unload` handshake is already the right mechanism; a TRELLIS 2 worker needs the same
  contract.
- Model load is ~0.2 s warm, so unload/reload between tiers is cheap.

## Recommendation

**Adopt TRELLIS 2 at Q6_K, and consider making it the default rather than a tier above
Hunyuan3D.** It is strictly better on every axis measured here: 2× faster on shape-only, half
the VRAM, and it produces the PBR textures and UVs that Hunyuan3D cannot deliver on this card
at all. The `docs/SETUP-GPU.md` note that "textures need 12–16 GB and are out of reach on a
3080" is true for Hunyuan3D's texture pipeline but **not** a general limit — TRELLIS 2 bakes
4096×4096 base-color + metallic-roughness at 5.08 GiB.

For Roblox specifically this removes the biggest gap in the pipeline: assets arrive textured
and UV-unwrapped, at a controllable face budget, without Blender.

Suggested defaults: `Q6_K`, `1024_cascade`, `texture_size=4096`, `target_face_num` set from the
Roblox budget, `low_vram=True`, `backend="sdpa"`.

The honest caveat: this is a **one-object, one-seed** benchmark. Timings are stable but visual
quality across Q4/Q6/Q8 was verified statistically (texture means and variances), not by eye.
Do a side-by-side render pass on real Kitbash inputs before locking the quant level.

### Reproducing

Isolated at `D:\trellis2` — `venv\`, `ComfyUI\` (fresh clone), `ComfyUI\custom_nodes\`
(`ComfyUI-Trellis2` = Aero-Ex fork, plus `ComfyUI-GGUF`), `models\Trellis2\`, `out\`.
`D:\ComfyUI`, its venv, `D:\models\Hunyuan3D-2.1`, and the Kitbash server were not modified —
verified after the run (`D:\ComfyUI` venv still on torch 2.6.0+cu124, its `custom_nodes` still
empty). Free the Kitbash server's VRAM first: `POST /admin/unload` on port 8188.
