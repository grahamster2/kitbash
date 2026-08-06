"""TRELLIS 2 (GGUF) as a second generator, driven out-of-process.

TRELLIS 2 pins torch 2.8 and transformers 5.2.0; this server runs torch 2.11 and
transformers 5.14.1. That is not reconcilable in one environment, so the model
runs in its own interpreter and this module is the client: a JSON request on the
worker's stdin, a JSON result on its stdout, and the mesh written to a path this
side chose. One HTTP surface and one job queue across two hostile venvs, at the
cost of ~10s of interpreter and node-pack import per job.

`generate_shape` matches pipeline.generate_shape's signature and result keys, so
jobs.py dispatches to either without knowing which is which.
"""
import base64
import io
import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import config
import pipeline

log = logging.getLogger("kitbash.trellis")

RESULT_PREFIX = "<<<KITBASH_RESULT>>>"

# What docs/QUALITY-COMPARISON.md measured on three real props, surfaced through
# GET /generators so a caller can choose without reading the repo.
CHARACTERISTICS = {
    "shape": "sharper than Hunyuan3D on hard-surface props: flat panels stay "
             "flat, rectangular parts stay rectangular, small applied details "
             "(crate brackets, sword guard, wheel hubs) survive",
    "texture": "textured=true bakes a 2048 base-colour + metallic-roughness "
               "pair with valid UVs, but the baked colour came back as "
               "multicoloured noise on every Kitbash reference prop tried. "
               "Treat the UVs as the deliverable, not the colour, until that "
               "is root-caused. textured=false skips the bake entirely and "
               "returns geometry only, with no UVs",
    "wall_seconds": "33 untextured, 123-231 textured, end to end through this "
                    "server. Cost scales with how solid the subject is, not "
                    "with its bounding box",
    "peak_vram_gib": "3.6 untextured, 3.7-8.2 textured device-wide, against "
                     "Hunyuan3D's 9.3",
    "needs_alpha": "yes -- an opaque background is reconstructed as geometry "
                   "and can stall the run. This server mattes with rembg first",
}

# The worker holds VRAM only while it is alive, so residency is exactly "is a
# subprocess running". Guarded because /health reads it off the request thread.
_lock = threading.Lock()
_running = 0


def available() -> dict:
    """Whether this box actually has the TRELLIS install wired up."""
    missing = [
        str(p)
        for p in (config.TRELLIS_PYTHON, config.TRELLIS_WORKER, config.TRELLIS_COMFY)
        if not p.exists()
    ]
    return {"available": not missing, "missing": missing}


def model_loaded() -> bool:
    with _lock:
        return _running > 0


def unload():
    """Free TRELLIS 2's VRAM.

    Nothing to do: the worker exits after every job and takes its allocations
    with it, which is the whole reason the subprocess split is cheap rather than
    a tax. It exists so callers can free "whichever generator is resident"
    without branching, and so /admin/unload keeps meaning what it says.

    Deliberately does not kill a live worker. The queue is single-worker, so a
    live worker means a job is mid-flight, and unloading is not a cancel button.
    """
    if model_loaded():
        log.warning("unload() called while a TRELLIS worker is running; leaving it")


def _ensure_alpha(image_b64: str) -> tuple[str, bool]:
    """Guarantee an alpha matte before the image reaches the model.

    docs/QUALITY-COMPARISON.md finding 1: handed an opaque white-background
    render, TRELLIS 2 spends its voxel budget reconstructing the backdrop and
    the run degenerates -- where Hunyuan3D, given the identical image, does not
    care. rembg is already a dependency and POST /images already calls it, but
    POST /jobs with a raw image_b64 does not, so it has to happen here.

    Returns the image and whether we produced the matte; if we could not, the
    worker asks the node pack to do it instead.
    """
    from PIL import Image

    if "," in image_b64 and image_b64.strip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    raw = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(raw)).convert("RGBA")

    if image.getchannel("A").getextrema()[0] < 255:
        return base64.b64encode(raw).decode(), True

    try:
        import rembg

        image = rembg.remove(image)
    except Exception:
        log.warning("rembg failed; deferring background removal to the node",
                    exc_info=True)
        return base64.b64encode(raw).decode(), False

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), True


def _parse(stdout: str, stderr: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX):])
    tail = (stderr or stdout).strip().splitlines()[-15:]
    raise RuntimeError("TRELLIS worker produced no result: " + "\n".join(tail))


def generate_shape(image_b64: str, out_dir: Path, params: dict) -> dict:
    """Image -> mesh via the TRELLIS 2 worker. Same contract as pipeline's."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.glb"

    ready = available()
    if not ready["available"]:
        raise RuntimeError(f"TRELLIS 2 is not installed here: missing {ready['missing']}")

    image_b64, matted = _ensure_alpha(image_b64)
    textured = bool(params.get("textured", True))
    request = {
        "image_b64": image_b64,
        "remove_background": not matted,
        "mesh_path": str(mesh_path),
        "comfy_dir": str(config.TRELLIS_COMFY),
        "quant": params.get("quant", config.TRELLIS_QUANT),
        "pipeline_type": str(params.get("pipeline_type", config.TRELLIS_PIPELINE_TYPE)),
        "texture_size": int(params.get("texture_size", config.TRELLIS_TEXTURE_SIZE)),
        "steps": int(params.get("num_inference_steps", config.TRELLIS_STEPS)),
        "seed": int(params.get("seed", 42)),
        "target_faces": int(params.get("target_faces", config.TRELLIS_TARGET_FACES)),
        "textured": textured,
    }

    # 7.63 + 6.88 GiB does not fit in 8.88. Hunyuan3D has to be out of VRAM
    # before the worker starts, not merely idle.
    pipeline.unload()

    elapsed, payload = _run_worker(request)

    if "error" in payload:
        log.error("TRELLIS worker traceback:\n%s", payload.get("traceback", ""))
        raise RuntimeError(payload["error"])

    return {
        "mesh_path": str(mesh_path),
        "generation_seconds": round(elapsed, 1),
        # Same metric pipeline.py reports (this process's torch peak) so the two
        # generators' numbers compare; the device-wide figure that actually
        # decides whether a run fits is alongside it.
        "peak_vram_gib": payload["peak_vram_gib"],
        "device_peak_vram_gib": payload["device_peak_vram_gib"],
        "vertices": payload["vertices"],
        "faces": payload["faces"],
        "decimated_from": payload["decimated_from"],
        "watertight": payload["watertight"],
        "file_bytes": mesh_path.stat().st_size,
        "generator": "trellis2",
        "textured": textured,
        "has_uv": payload["has_uv"],
        "base_color_texture": payload["base_color_texture"],
        "stages": payload["stages"],
        "params": {
            "quant": request["quant"],
            "pipeline_type": request["pipeline_type"],
            "texture_size": request["texture_size"],
            "num_inference_steps": request["steps"],
            "seed": request["seed"],
            "target_faces": request["target_faces"],
            "textured": textured,
        },
    }


def _run_worker(request: dict) -> tuple[float, dict]:
    global _running
    with _lock:
        _running += 1
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(config.TRELLIS_PYTHON), str(config.TRELLIS_WORKER)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            # The node pack resolves its imports relative to ComfyUI's root.
            cwd=str(config.TRELLIS_COMFY),
            env=_env(),
            timeout=config.TRELLIS_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"TRELLIS worker exceeded {config.TRELLIS_TIMEOUT}s. This is the "
            f"memory-thrash stall from docs/QUALITY-COMPARISON.md: a solid "
            f"subject at too high a pipeline_type/texture_size saturates VRAM "
            f"and stops making progress rather than failing."
        ) from exc
    finally:
        with _lock:
            _running -= 1
    return time.time() - t0, _parse(proc.stdout, proc.stderr)


def _env() -> dict:
    import os

    env = dict(os.environ)
    env["HF_HOME"] = config.TRELLIS_HF_HOME
    env["PYTHONUNBUFFERED"] = "1"
    return env
