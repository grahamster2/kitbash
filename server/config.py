"""Server configuration, all overridable by environment variable.

Defaults match the reference GPU box (see docs/SETUP-GPU.md). Nothing here
should need editing to run on a different machine — set env vars instead.
"""
import os
from pathlib import Path

# Where the Hunyuan3D repo was cloned. `hy3dshape` is imported from inside it
# rather than pip-installed, so this has to go on sys.path.
HY3D_REPO = Path(os.environ.get("KITBASH_HY3D_REPO", r"D:\models\Hunyuan3D-2.1"))

# Job outputs: one directory per job, holding the mesh and its metadata.
OUT_DIR = Path(os.environ.get("KITBASH_OUT_DIR", r"D:\kitbash-out"))

HOST = os.environ.get("KITBASH_HOST", "0.0.0.0")
PORT = int(os.environ.get("KITBASH_PORT", "8188"))

# Shape-generation defaults. octree_resolution drives both mesh density and
# VRAM; 256 peaked at 7.63 GiB on a 3080, which is the ceiling worth trusting.
DEFAULT_OCTREE_RESOLUTION = int(os.environ.get("KITBASH_OCTREE_RESOLUTION", "256"))
DEFAULT_INFERENCE_STEPS = int(os.environ.get("KITBASH_INFERENCE_STEPS", "30"))
DEFAULT_GUIDANCE_SCALE = float(os.environ.get("KITBASH_GUIDANCE_SCALE", "5.0"))

# Keep the model in VRAM between jobs. Loading costs ~70s, so for a multi-part
# build (the whole point of this project) unloading between parts is a big loss.
# Set to 0 when a second model needs the VRAM.
KEEP_MODEL_RESIDENT = os.environ.get("KITBASH_KEEP_RESIDENT", "1") != "0"

# Which generator POST /jobs uses when the caller does not say. Hunyuan3D stays
# the default: docs/QUALITY-COMPARISON.md found it ~2x faster in wall time and
# tolerant of unprepared input, where TRELLIS 2 needs an alpha matte to behave.
DEFAULT_GENERATOR = os.environ.get("KITBASH_DEFAULT_GENERATOR", "hunyuan3d")

# --- TRELLIS 2 --------------------------------------------------------------
# TRELLIS 2 runs out-of-process because its pins (torch 2.8, transformers 5.2.0)
# cannot coexist with this server's (torch 2.11, transformers 5.14.1). These
# point at the separate install; see docs/TRELLIS2-EVAL.md.
TRELLIS_PYTHON = Path(
    os.environ.get("KITBASH_TRELLIS_PYTHON", r"D:\trellis2\venv\Scripts\python.exe")
)
TRELLIS_WORKER = Path(
    os.environ.get("KITBASH_TRELLIS_WORKER", r"D:\trellis2\trellis_worker.py")
)
# The node pack is only importable with ComfyUI's root on sys.path and as cwd.
TRELLIS_COMFY = Path(
    os.environ.get("KITBASH_TRELLIS_COMFY", r"D:\trellis2\ComfyUI")
)
# The worker fetches DINOv3 through huggingface_hub; without this it would
# populate C: instead of the drive the 19 GB of weights already live on.
TRELLIS_HF_HOME = os.environ.get("KITBASH_TRELLIS_HF_HOME", r"D:\hf-cache")

# Q6_K is Q4 speed at closer-to-Q8 fidelity for the same VRAM — quantization is
# a disk/speed knob here, not a VRAM one, because the three DiTs load serially.
# Q8_0, not a K-quant. The K-quantised 512 texture DiTs (Q4_K_M, Q6_K) decode
# to input-independent noise — the rainbow atlases — while Q8_0 of the same
# model is clean. The earlier "Q6_K is the sweet spot" finding compared a Q8
# dragon against Q6 props and blamed the subject. Costs ~55s more per textured
# generation; peak VRAM is set by the bake stage either way, so it still fits.
TRELLIS_QUANT = os.environ.get("KITBASH_TRELLIS_QUANT", "GGUF Q8_0")

# 512 + 2048, NOT the eval's 1024_cascade + 4096. docs/QUALITY-COMPARISON.md ran
# the recommended settings on a solid crate and killed it at 21 minutes, pinned
# at 96% of the VRAM budget: TRELLIS 2's cost scales with occupied volume, and
# Kitbash's props are solid where the eval's dragon was mostly empty space.
TRELLIS_PIPELINE_TYPE = os.environ.get("KITBASH_TRELLIS_PIPELINE", "512")
TRELLIS_TEXTURE_SIZE = int(os.environ.get("KITBASH_TRELLIS_TEXTURE_SIZE", "2048"))
TRELLIS_STEPS = int(os.environ.get("KITBASH_TRELLIS_STEPS", "12"))

# target_face_num is a node input, so decimation happens in-pipeline rather than
# as a post-step. Matches the Roblox per-MeshPart cap.
TRELLIS_TARGET_FACES = int(os.environ.get("KITBASH_TRELLIS_TARGET_FACES", "20000"))

# A completing run is 79-151s. This is not a performance budget, it is the
# tripwire for the memory-thrash stall above, which never terminates on its own.
TRELLIS_TIMEOUT = int(os.environ.get("KITBASH_TRELLIS_TIMEOUT", "900"))

# How many finished jobs to retain in memory before evicting the oldest.
MAX_JOB_HISTORY = int(os.environ.get("KITBASH_MAX_JOB_HISTORY", "200"))

# --- procedural primitives --------------------------------------------------
# Radial segments on round scripted parts. 24 is smooth at prop scale and still
# cheap; the point of scripting is that this is a dial rather than a surprise.
PRIMITIVE_SECTIONS = int(os.environ.get("KITBASH_PRIMITIVE_SECTIONS", "24"))

# Refuse to build a primitive heavier than this. Matches Roblox's per-MeshPart
# triangle cap, so a scripted part can never be the thing that fails an import
# — nothing here needs more, and a parameter combination that does is a mistake.
PRIMITIVE_MAX_FACES = int(os.environ.get("KITBASH_PRIMITIVE_MAX_FACES", "20000"))

# --- image generation -------------------------------------------------------
# Which provider turns a prompt into a reference image. "fal" uses the user's
# own fal.ai key and bills them directly; "local" is scaffolded. See imagegen.py.
IMAGE_PROVIDER = os.environ.get("KITBASH_IMAGE_PROVIDER", "fal")

# Read from the environment and never persisted — this is the user's key.
FAL_KEY = os.environ.get("FAL_KEY") or os.environ.get("FAL_API_KEY")

# schnell is the fast, cheap FLUX variant; reference images do not need dev.
FAL_MODEL = os.environ.get("KITBASH_FAL_MODEL", "fal-ai/flux/schnell")
