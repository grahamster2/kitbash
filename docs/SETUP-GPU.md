# Setting up the GPU machine

Reference environment: Windows 11, RTX 3080 10GB, driver 595.79, Python 3.12.

## Do not use Hunyuan3D's `requirements.txt`

It does not install on Python 3.12, and most of what it pulls in is for the texture pipeline you probably can't run anyway. Specifically:

| Pin | Problem |
| --- | --- |
| `numpy==1.24.4` | No Python 3.12 wheels at all. Conflicts with the numpy PyTorch installs. |
| `basicsr==1.4.2` | Imports `torchvision.transforms.functional_tensor`, removed from torchvision. This is what their `torchvision_fix.py` monkey-patches around. |
| `bpy==4.0` | Blender-as-a-module. Huge, and only used by the offline render tools. |
| `deepspeed` | Training only. Painful to build on Windows. |
| `pymeshlab==2022.2.post3`, `realesrgan`, `tb_nightly` | All pre-3.12 era. |

Install the inference set directly instead.

## What shape generation actually needs

Scanning imports under `hy3dshape/hy3dshape/` separates real dependencies from training baggage:

- **Off the inference path entirely:** `wandb`, `matplotlib`, `pythreejs`, `ipywidgets` — they live under `utils/trainings/` and `utils/visualizers/`.
- **Function-level imports, never reached in the default config:** `diso`, `sageattention`, `torch_cluster`. This matters a lot — `diso` is a CUDA extension, and needing it would mean installing the CUDA toolkit and MSVC build tools. The default surface extractor is `mc_algo='mc'` (skimage marching cubes), so it is never imported.

**Stay on `mc_algo='mc'` unless you have a reason not to.** Switching to `dmc` drags in a compile step.

## Working versions

Hunyuan3D pins `transformers==4.46` / `diffusers==0.30` / `numpy<2`. The current stack works anyway:

```
torch          2.11.0+cu128     numpy      2.4.4
transformers   5.14.1           diffusers  0.39.0
trimesh        5.0.0            skimage    0.26.0
opencv         5.0.0            onnxruntime 1.28.0
```

## Install

```powershell
# Caches on D: — model weights are tens of GB and will fill a small C:
setx HF_HOME "D:\hf-cache"
setx UV_CACHE_DIR "D:\uv-cache"

uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

uv pip install `
    diffusers transformers accelerate huggingface-hub safetensors `
    einops omegaconf pyyaml tqdm `
    scipy scikit-image opencv-python-headless pillow `
    trimesh pymeshlab pygltflib xatlas `
    timm torchdiffeq peft `
    rembg onnxruntime `
    fastapi "uvicorn[standard]" pydantic python-multipart

git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git D:\models\Hunyuan3D-2.1
```

No CUDA toolkit and no MSVC build tools required — PyTorch ships its own CUDA runtime, and nothing on this path compiles.

## VRAM

`nvidia-smi` reporting 10240 MiB is misleading; the Windows desktop holds ~1 GB, leaving **~8.9 GiB usable**. Budget against the real number.

**Hunyuan3D's** texture stage needs 12-16 GB and is out of reach on a 3080 without heavy offloading. Shape generation fits comfortably.

TRELLIS 2 GGUF *fits* a texture bake on the same card, so the ceiling is Hunyuan3D's rather than the GPU's — but the textures it produced were rainbow noise on every hard-surface prop tested, so this is not yet a way to get textures. See [QUALITY-COMPARISON.md](QUALITY-COMPARISON.md).

If you need that last gigabyte back and your CPU has integrated graphics, driving the displays from the iGPU frees the 3080 entirely.

## PowerShell over SSH

Two things that will waste your time otherwise:

- `$ErrorActionPreference = "Stop"` turns *any* native-command stderr output into a fatal error. `git clone` and `uv` both write progress to stderr, so scripts abort on success. Use `Continue` and check `$LASTEXITCODE`.
- Don't try to inline multi-line PowerShell through `ssh`. Write a `.ps1`, `scp` it, run it with `powershell -ExecutionPolicy Bypass -File`.
