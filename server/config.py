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

# Keep the model in VRAM between jobs. Loading costs ~30s, so for a multi-part
# build (the whole point of this project) unloading between parts is a big loss.
# Set to 0 when a second model needs the VRAM.
KEEP_MODEL_RESIDENT = os.environ.get("KITBASH_KEEP_RESIDENT", "1") != "0"

# How many finished jobs to retain in memory before evicting the oldest.
MAX_JOB_HISTORY = int(os.environ.get("KITBASH_MAX_JOB_HISTORY", "200"))
