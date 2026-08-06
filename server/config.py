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
