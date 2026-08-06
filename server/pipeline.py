"""Hunyuan3D 2.1 shape generation, wrapped so the rest of the server never
imports torch directly.

Only shape is implemented. Texturing needs 12-16 GB and does not fit on the
reference card; see docs/SETUP-GPU.md.
"""
import base64
import io
import logging
import sys
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("kitbash.pipeline")

# hy3dshape lives in the cloned repo, not site-packages.
_HY3D_SHAPE = str(config.HY3D_REPO / "hy3dshape")
if _HY3D_SHAPE not in sys.path:
    sys.path.insert(0, _HY3D_SHAPE)

# Guards the pipeline object. The worker is single-threaded today, but /health
# reads model state concurrently and a second worker is a plausible change.
_lock = threading.Lock()
_pipeline = None


def vram_stats():
    """Free/total VRAM in GiB, or None when torch or CUDA is unavailable.

    Called by /health, which must never raise — a health endpoint that 500s
    because CUDA is missing tells you less than one that says so.
    """
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info()
    return {
        "device": torch.cuda.get_device_name(0),
        "free_gib": round(free / 1024**3, 2),
        "total_gib": round(total / 1024**3, 2),
        "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 2),
    }


def model_loaded():
    return _pipeline is not None


def load():
    """Load the shape pipeline, or return the resident one."""
    global _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        t0 = time.time()
        log.info("loading Hunyuan3D shape pipeline")
        _pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1"
        )
        log.info("pipeline loaded in %.1fs", time.time() - t0)
        return _pipeline


def unload():
    global _pipeline
    with _lock:
        if _pipeline is None:
            return
        _pipeline = None
    try:
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        log.warning("unload: could not empty CUDA cache", exc_info=True)


def _decode_image(image_b64: str):
    from PIL import Image

    # Tolerate data URLs; clients paste them constantly.
    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def generate_shape(image_b64: str, out_dir: Path, params: dict) -> dict:
    """Image -> mesh. Writes mesh.glb into out_dir and returns its stats."""
    import torch

    image = _decode_image(image_b64)
    pipe = load()

    octree = int(params.get("octree_resolution", config.DEFAULT_OCTREE_RESOLUTION))
    steps = int(params.get("num_inference_steps", config.DEFAULT_INFERENCE_STEPS))
    guidance = float(params.get("guidance_scale", config.DEFAULT_GUIDANCE_SCALE))
    seed = params.get("seed")

    kwargs = {
        "image": image,
        "octree_resolution": octree,
        "num_inference_steps": steps,
        "guidance_scale": guidance,
    }
    if seed is not None:
        kwargs["generator"] = torch.Generator(device="cuda").manual_seed(int(seed))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    mesh = pipe(**kwargs)[0]
    elapsed = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.glb"
    raw_faces = int(len(mesh.faces))
    decimated_from = None

    target_faces = params.get("target_faces")
    if target_faces and raw_faces > int(target_faces):
        # Keep the dense original: it is the better input for retopology or a
        # higher-quality re-export later, and regenerating it costs 40s.
        mesh.export(str(out_dir / "mesh_raw.glb"))
        mesh = _decimate(mesh, int(target_faces))
        decimated_from = raw_faces

    mesh.export(str(mesh_path))

    if not config.KEEP_MODEL_RESIDENT:
        unload()

    return {
        "mesh_path": str(mesh_path),
        "generation_seconds": round(elapsed, 1),
        "peak_vram_gib": round(peak_vram, 2),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "decimated_from": decimated_from,
        "watertight": bool(mesh.is_watertight),
        "file_bytes": mesh_path.stat().st_size,
        "params": {
            "octree_resolution": octree,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
            "target_faces": target_faces,
        },
    }


def _decimate(mesh, target_faces: int):
    """Quadric decimation down to target_faces.

    Uses trimesh + fast_simplification, both MIT. pymeshlab would also do this
    and is already installed, but it is GPL — keeping it off the server's
    import path keeps the whole stack permissively licensed.

    Decimation usually breaks watertightness. That is fine for a game engine,
    which does not care, but it is why the raw mesh is kept alongside.
    """
    log.info("decimating %d -> %d faces", len(mesh.faces), target_faces)
    t0 = time.time()
    out = mesh.simplify_quadric_decimation(face_count=target_faces)
    log.info("decimated to %d faces in %.2fs", len(out.faces), time.time() - t0)
    return out
