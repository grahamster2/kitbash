"""Kitbash GPU server.

The only process that touches the GPU. Everything else — the MCP server, the
Tauri app — talks to it over HTTP, which is what makes "GPU on another machine"
a change of base URL rather than a change of architecture.

Run:  python -m uvicorn app:api --host 0.0.0.0 --port 8188
"""
import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import assemble as assembly
import config
import jobs
import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("kitbash.app")

STARTED_AT = time.time()

api = FastAPI(title="Kitbash GPU server", version="0.1.0")


@api.on_event("startup")
def _startup():
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs.rehydrate()
    jobs.start_worker()
    log.info("listening; outputs -> %s", config.OUT_DIR)


class GenerateRequest(BaseModel):
    image_b64: str = Field(..., description="PNG/JPEG as base64, data URLs accepted")
    octree_resolution: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    # Raw meshes come out at 300k+ faces, far too heavy for a game engine.
    # Decimation is cheap (~0.3s) and happens before export.
    target_faces: int | None = None
    # Free-form label so a caller can tag which part of a multi-part build this
    # is. The server does not interpret it.
    part_name: str | None = None


@api.get("/health")
def health():
    """Always answers, even when CUDA is broken — that is the point of it."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "model_loaded": pipeline.model_loaded(),
        "gpu": pipeline.vram_stats(),
        **jobs.stats(),
    }


@api.post("/jobs")
def create_job(req: GenerateRequest):
    params = {
        k: v
        for k, v in {
            "octree_resolution": req.octree_resolution,
            "num_inference_steps": req.num_inference_steps,
            "guidance_scale": req.guidance_scale,
            "seed": req.seed,
            "target_faces": req.target_faces,
            "part_name": req.part_name,
        }.items()
        if v is not None
    }
    return jobs.submit("image_to_3d", params, req.image_b64)


@api.get("/jobs")
def list_jobs(limit: int = 50):
    return {"jobs": jobs.listing(limit)}


@api.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    return job


@api.get("/jobs/{job_id}/mesh")
def get_mesh(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    if job["status"] != jobs.DONE:
        raise HTTPException(409, f"job is {job['status']}, not done")
    return FileResponse(
        job["result"]["mesh_path"],
        media_type="model/gltf-binary",
        filename=f"{job_id}.glb",
    )


class PartPlacement(BaseModel):
    job_id: str = Field(..., description="A completed image_to_3d job")
    name: str = Field(..., description="glTF node name for this part")
    position: list[float] | None = Field(None, description="[x, y, z]")
    rotation: list[float] | None = Field(None, description="[rx, ry, rz] in degrees")
    scale: float | list[float] | None = None
    # Assemble from the dense original instead of the decimated export. Useful
    # when one part needs detail the rest of the scene does not.
    use_raw: bool = False


class AssembleRequest(BaseModel):
    parts: list[PartPlacement]
    scene_name: str | None = None


def _part_mesh_path(p: PartPlacement) -> Path:
    job = jobs.get(p.job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {p.job_id}")
    if job["status"] != jobs.DONE:
        raise HTTPException(409, f"job {p.job_id} is {job['status']}, not done")
    mesh_path = Path(job["result"]["mesh_path"])
    if p.use_raw:
        raw = mesh_path.parent / "mesh_raw.glb"
        if not raw.exists():
            raise HTTPException(
                404,
                f"job {p.job_id} has no mesh_raw.glb — it was not decimated, "
                f"so mesh.glb is already the dense original",
            )
        return raw
    return mesh_path


@api.post("/assemble")
def assemble_scene(req: AssembleRequest):
    """Compose finished parts into a single glTF with one node per part."""
    if not req.parts:
        raise HTTPException(400, "no parts given")

    resolved = [
        {
            "name": p.name,
            "mesh_path": str(_part_mesh_path(p)),
            "position": p.position,
            "rotation": p.rotation,
            "scale": p.scale,
        }
        for p in req.parts
    ]

    scene_id = uuid.uuid4().hex[:12]
    out = config.OUT_DIR / "scenes" / scene_id / f"{req.scene_name or 'scene'}.glb"
    try:
        result = assembly.assemble(resolved, out)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"scene_id": scene_id, **result}


@api.get("/scenes/{scene_id}/mesh")
def get_scene(scene_id: str):
    scene_dir = config.OUT_DIR / "scenes" / scene_id
    matches = sorted(scene_dir.glob("*.glb")) if scene_dir.exists() else []
    if not matches:
        raise HTTPException(404, f"no such scene: {scene_id}")
    return FileResponse(
        matches[0], media_type="model/gltf-binary", filename=matches[0].name
    )


@api.get("/jobs/{job_id}/describe")
def describe_part(job_id: str):
    """Bounds and size of a finished part, so a caller can place it."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    if job["status"] != jobs.DONE:
        raise HTTPException(409, f"job is {job['status']}, not done")
    return assembly.describe(Path(job["result"]["mesh_path"]))


@api.post("/admin/unload")
def unload_model():
    """Drop the model from VRAM. Needed before running anything else on the GPU."""
    pipeline.unload()
    return {"model_loaded": pipeline.model_loaded(), "gpu": pipeline.vram_stats()}
