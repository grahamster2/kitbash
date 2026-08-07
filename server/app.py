"""Kitbash GPU server.

The only process that touches the GPU. Everything else — the MCP server, the
Tauri app — talks to it over HTTP, which is what makes "GPU on another machine"
a change of base URL rather than a change of architecture.

Run:  python -m uvicorn app:api --host 0.0.0.0 --port 8188
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import assemble as assembly
import config
import decompose
import export
import hollow as hollowing
import imagegen
import jobs
import materials
import orient as orienting
import pipeline
import preview as previewing
import primitives
import trellis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("kitbash.app")

STARTED_AT = time.time()

@asynccontextmanager
async def _lifespan(_: FastAPI):
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs.rehydrate()
    jobs.start_worker()
    log.info("listening; outputs -> %s", config.OUT_DIR)
    yield


api = FastAPI(title="Kitbash GPU server", version="0.1.0", lifespan=_lifespan)


class GenerateRequest(BaseModel):
    image_b64: str | None = Field(
        None, description="PNG/JPEG as base64, data URLs accepted"
    )
    generator: str | None = Field(
        None,
        description=(
            f"Which model builds the shape; see GET /generators. One of "
            f"{sorted(jobs.GENERATORS)}, default {config.DEFAULT_GENERATOR!r}."
        ),
    )
    textured: bool | None = Field(
        None,
        description=(
            "Bake a base-colour and metallic-roughness atlas. TRELLIS 2 only — "
            "Hunyuan3D's texture stage does not fit on the reference card. "
            "Defaults on for TRELLIS 2; turn it off to skip the ~50s bake when "
            "only geometry is wanted."
        ),
    )
    # Reuse an image from POST /images. The same reference driving several parts
    # is what makes an assembled object look like one object rather than a pile
    # of separately-imagined pieces.
    image_id: str | None = Field(None, description="An image from POST /images")
    octree_resolution: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    # Raw meshes come out at 300k+ faces, far too heavy for a game engine.
    # Decimation is cheap (~0.3s) and happens before export.
    target_faces: int | None = None
    # Paint the finished mesh with the reference image it was generated from.
    # Defaults on wherever the generator supplies no colour of its own.
    texture: bool | None = None
    texture_mode: str | None = Field(None, description='"uv", "atlas" or "vertex"')
    # Free-form label so a caller can tag which part of a multi-part build this
    # is. The server does not interpret it.
    part_name: str | None = None


@api.get("/health")
def health():
    """Always answers, even when CUDA is broken — that is the point of it."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        # Stays a plain bool for the MCP and Tauri clients, but now means "some
        # generator is holding VRAM" rather than specifically Hunyuan3D.
        "model_loaded": pipeline.model_loaded() or trellis.model_loaded(),
        "generators_loaded": {
            "hunyuan3d": pipeline.model_loaded(),
            "trellis2": trellis.model_loaded(),
        },
        "gpu": pipeline.vram_stats(),
        **jobs.stats(),
    }


@api.get("/generators")
def list_generators():
    """The shape models available here and what they actually cost.

    The numbers are measured (docs/QUALITY-COMPARISON.md), not quoted from the
    upstream READMEs, and they include the failure modes — an agent choosing a
    generator needs to know TRELLIS 2's baked colour came back as noise on this
    project's own reference style far more than it needs a feature list.
    """
    return {
        "default": config.DEFAULT_GENERATOR,
        "generators": [
            {
                "name": "hunyuan3d",
                "available": True,
                "loaded": pipeline.model_loaded(),
                "textures": False,
                "in_process": True,
                "characteristics": {
                    "shape": "softer than TRELLIS 2 on hard surfaces — edges "
                             "round off and large panels undulate — but robust "
                             "to unprepared input and reliable",
                    "texture": "none. The texture stage needs 12-16 GiB and "
                               "does not fit on the reference card",
                    "wall_seconds": "~41 warm, ~83 including a cold load",
                    "peak_vram_gib": "9.3 device-wide, 92% of the usable budget",
                    "needs_alpha": "no",
                },
                "defaults": {
                    "octree_resolution": config.DEFAULT_OCTREE_RESOLUTION,
                    "num_inference_steps": config.DEFAULT_INFERENCE_STEPS,
                    "guidance_scale": config.DEFAULT_GUIDANCE_SCALE,
                },
            },
            {
                "name": "trellis2",
                **trellis.available(),
                "loaded": trellis.model_loaded(),
                "textures": True,
                # Worth advertising: it explains the ~10s constant overhead and
                # why this generator cannot be kept warm between jobs.
                "in_process": False,
                "characteristics": trellis.CHARACTERISTICS,
                "defaults": {
                    "quant": config.TRELLIS_QUANT,
                    "pipeline_type": config.TRELLIS_PIPELINE_TYPE,
                    "texture_size": config.TRELLIS_TEXTURE_SIZE,
                    "num_inference_steps": config.TRELLIS_STEPS,
                    "target_faces": config.TRELLIS_TARGET_FACES,
                    "textured": True,
                },
            },
        ],
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
            "texture": req.texture,
            "texture_mode": req.texture_mode,
            "part_name": req.part_name,
            "generator": req.generator,
            "textured": req.textured,
        }.items()
        if v is not None
    }
    if bool(req.image_b64) == bool(req.image_id):
        raise HTTPException(400, "give exactly one of image_b64 or image_id")
    if req.generator is not None and req.generator not in jobs.GENERATORS:
        # Rejected here rather than in the worker: a queued job that is going to
        # fail on a typo should fail at the call that made the typo.
        raise HTTPException(
            400, f"unknown generator {req.generator!r}, expected one of "
                 f"{sorted(jobs.GENERATORS)}"
        )

    image_b64 = req.image_b64
    if req.image_id:
        try:
            image_b64 = imagegen.load_b64(req.image_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    return jobs.submit("image_to_3d", params, image_b64)


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


# --------------------------------------------------------------------------
# preview
# --------------------------------------------------------------------------
# The one thing an agent assembling a scene over MCP could not previously do is
# look at it. See server/preview.py and docs/PREVIEW.md.
_PREVIEW_VIEWS = Query(
    None,
    description=(
        "Comma-separated view names, in sheet order. "
        f"Any of {sorted(previewing.VIEWS)}; "
        f"default {','.join(previewing.DEFAULT_VIEWS)}."
    ),
)
_PREVIEW_SIZE = Query(1200, ge=256, le=2400, description="Sheet width in pixels.")
_PREVIEW_COLUMNS = Query(3, ge=1, le=6, description="Tiles per row.")
_PREVIEW_HIGHLIGHT = Query(
    None, description="Paint this named part magenta; everything else is unchanged."
)
_PREVIEW_ISOLATE = Query(
    False,
    description=(
        "With `highlight`, hide every other part. Framing does not change, so "
        "the isolated part stays at the pixel it occupied in the full render."
    ),
)


def _render_preview(source: Path, views, size, columns, highlight, isolate) -> Response:
    """Shared body of both preview endpoints.

    Rendering is CPU-only and takes about a second, so it runs inline rather
    than through the job queue — the queue exists to serialise the GPU, which
    this never touches, and a preview an agent has to poll for is a preview it
    will stop calling.
    """
    names = [v.strip() for v in views.split(",") if v.strip()] if views else None
    try:
        png = previewing.preview_png(
            source,
            views=names or previewing.DEFAULT_VIEWS,
            size=size,
            columns=columns,
            highlight=highlight,
            isolate=isolate,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Not cacheable: the same scene id is re-rendered after a part is fixed.
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@api.get("/preview/views", responses={200: {"description": "Available view names"}})
def list_preview_views():
    """The named camera angles a preview can be composed from."""
    return {
        "views": {
            name: {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll}
            for name, (yaw, pitch, roll) in previewing.VIEWS.items()
        },
        "default": list(previewing.DEFAULT_VIEWS),
        "notes": (
            "One camera distance is derived from the whole scene's bounds and "
            "shared by every view, so tiles are directly comparable and a part "
            "that floats cannot hide behind a re-framed camera."
        ),
    }


@api.get("/jobs/{job_id}/preview", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
def preview_job(
    job_id: str,
    views: str | None = _PREVIEW_VIEWS,
    size: int = _PREVIEW_SIZE,
    columns: int = _PREVIEW_COLUMNS,
    highlight: str | None = _PREVIEW_HIGHLIGHT,
    isolate: bool = _PREVIEW_ISOLATE,
):
    """A shaded contact sheet of one finished part, on a ground plane."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {job_id}")
    if job["status"] != jobs.DONE:
        raise HTTPException(409, f"job is {job['status']}, not done")
    return _render_preview(
        Path(job["result"]["mesh_path"]), views, size, columns, highlight, isolate
    )


@api.get("/scenes/{scene_id}/preview", response_class=Response,
         responses={200: {"content": {"image/png": {}}}})
def preview_scene(
    scene_id: str,
    views: str | None = _PREVIEW_VIEWS,
    size: int = _PREVIEW_SIZE,
    columns: int = _PREVIEW_COLUMNS,
    highlight: str | None = _PREVIEW_HIGHLIGHT,
    isolate: bool = _PREVIEW_ISOLATE,
):
    """A shaded contact sheet of an assembled scene, on a ground plane.

    This is the check that closes the assembly loop: parts are placed from
    measurements, and this is where a caller finds out whether the measurements
    meant what it thought. Floating parts, parts that never joined and parts
    that overshoot all show here and nowhere else in the API.
    """
    scene_dir = config.OUT_DIR / "scenes" / scene_id
    matches = sorted(scene_dir.glob("*.glb")) if scene_dir.exists() else []
    if not matches:
        raise HTTPException(404, f"no such scene: {scene_id}")
    return _render_preview(matches[0], views, size, columns, highlight, isolate)


@api.get("/scenes/{scene_id}/ground")
def scene_ground(scene_id: str):
    """How far each part sits above the scene's floor, worst first.

    The number beside the picture. A preview shows that the fin is floating; it
    is this that says by how much, which is what a fix needs.
    """
    scene_dir = config.OUT_DIR / "scenes" / scene_id
    matches = sorted(scene_dir.glob("*.glb")) if scene_dir.exists() else []
    if not matches:
        raise HTTPException(404, f"no such scene: {scene_id}")
    parts = previewing.load_parts(matches[0])
    framing = previewing.Framing.of(parts)
    return {
        "scene_id": scene_id,
        "floor_y": round(framing.ground_y, 4),
        "parts": previewing.ground_report(parts, framing),
    }


class PrimitiveRequest(BaseModel):
    kind: str = Field(..., description=f"One of {primitives.kinds()}")
    params: dict | None = Field(
        None, description="Dimensions and options; see GET /primitives."
    )
    part_name: str | None = Field(None, description="Defaults to the kind.")
    material: str | None = Field(
        None,
        description=(
            "Override the material the kind implies — a crate is wood, a pipe "
            f"is metal. One of {materials.families()}."
        ),
    )
    color: str | None = Field(None, description='Base colour as "#rrggbb".')
    uv_scale: float | None = Field(
        None,
        description=(
            "Emit box-projection UVs at one texture tile per this many studs. "
            "Off by default: it splits vertices at every seam, which ends the "
            "welded topology, and nothing downstream has a texture yet."
        ),
    )


@api.get("/primitives")
def list_primitives():
    """The scripted catalogue: kinds, parameters, types, defaults and units.

    Exists so an agent can discover the library by calling it rather than by
    being told about it, the same reason the MCP tool descriptions carry
    numbers instead of prose.
    """
    return {
        "kinds": primitives.catalogue(),
        "units": "1 file unit = 1 stud, matching /export",
        "origin": "bounding-box centre, matching generated parts",
        "max_faces": config.PRIMITIVE_MAX_FACES,
    }


@api.get("/primitives/{kind}")
def get_primitive(kind: str):
    if kind not in primitives.KINDS:
        raise HTTPException(404, f"no such kind: {kind}")
    return primitives.KINDS[kind].as_dict()


@api.post("/primitives")
def create_primitive(req: PrimitiveRequest):
    """Build a scripted part and record it as a finished job.

    Deliberately synchronous and deliberately *not* queued: this is a
    millisecond of numpy, and the queue exists to serialise access to a GPU
    that this path never touches. Queueing it behind a 40-second generation
    would be a bug, not a policy.

    What it does share is the job record, so the id it returns goes into
    /assemble, /export and /jobs/{id}/describe unchanged — assembly cannot tell
    a scripted part from a generated one, which is the whole point.
    """
    job_id = uuid.uuid4().hex[:12]
    try:
        result = primitives.store(
            req.kind, req.params, config.OUT_DIR / job_id,
            part_name=req.part_name, material=req.material, color=req.color,
            uv_scale=req.uv_scale,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    now = time.time()
    job = {
        "id": job_id,
        "type": "primitive",
        "status": jobs.DONE,
        "created_at": now,
        "started_at": now,
        "finished_at": time.time(),
        "params": {"kind": req.kind, "part_name": req.part_name or req.kind},
        "result": result,
        "error": None,
    }
    _record(job)
    return job


def _record(job: dict):
    """File an already-finished job into the registry.

    jobs.submit() is the queued path and would hand this to the worker, which
    only knows how to generate. Everything else about the record — the disk
    mirror that survives a restart, the history cap, the /jobs listing — should
    apply identically, so it is reproduced here rather than skipped.
    """
    with jobs._jobs_lock:
        jobs._jobs[job["id"]] = job
        while len(jobs._jobs) > config.MAX_JOB_HISTORY:
            old_id, old = jobs._jobs.popitem(last=False)
            if old["status"] in (jobs.QUEUED, jobs.RUNNING):
                jobs._jobs[old_id] = old
                jobs._jobs.move_to_end(old_id, last=False)
                break
            jobs._images.pop(old_id, None)
    jobs._persist(job)


class Anchor(BaseModel):
    """Place a part against another part instead of at a coordinate.

    The server measures the target's *placed* bounds — after its own scale,
    rotation and anchor — so the caller never has to know what a part's mesh
    measures on disk. That is the whole point: a part scaled 0.05 occupies a
    twentieth of its file bounds, and guessing from the file is the bug.
    """

    to: str = Field(
        ...,
        description=(
            "The part name to measure against, or 'ground' for the y=0 plane. "
            "The target may itself be anchored; order in the list does not "
            "matter, and a cycle is an error rather than a hang."
        ),
    )
    align: dict[str, float | str] | None = Field(
        None,
        description=(
            "Per axis, where on the TARGET this part goes. A number is a "
            "fraction of the target's box (0 = low face, 0.2 = a fifth along, "
            f"1 = high face); or one of {sorted(set(assembly.FRACTIONS))}. "
            f"The attachment keywords {sorted(set(assembly.ATTACH))} also set "
            "the matching point on this part, so 'under' means this part's top "
            "face meets the target's bottom face. Axes left out fall through to "
            "`position`. Omit `align` entirely to centre this part inside the "
            "target."
        ),
    )
    my: dict[str, float | str] | None = Field(
        None,
        description=(
            "Per axis, which point of THIS part lands on that spot. Same "
            "vocabulary as `align`. Defaults to the part's centre, so "
            "align {'y': 'min'} hangs this part's centre off the target's "
            "bottom face and my {'y': 'max'} makes them touch instead."
        ),
    )
    offset: list[float] | None = Field(
        None, description="[x, y, z] in world units, added after alignment."
    )


class Orient(BaseModel):
    """Turn the part into the frame it is declared to belong in.

    A generator reconstructs an object in its reference image's camera frame,
    so parts arrive at arbitrary azimuth and an assembly of correctly-*placed*
    parts still looks like debris. Say what the part is and the server works
    out the rotation from its own geometry. See server/orient.py.
    """

    role: str | None = Field(
        None,
        description=(
            "A named shape, which supplies extents and taper together. "
            f"One of {orienting.roles()}. Left/right parts describe the LEFT "
            "one; the right one is `mirror_of` and inherits this."
        ),
    )
    extents: list[float] | None = Field(
        None,
        description=(
            "[x, y, z] the part should measure in the assembled scene — real "
            "metres are ideal but only the ratios are used. Overrides `role`. "
            "Declare two axes as equal where you are not sure which is bigger; "
            "an honest tie costs nothing and a wrong claim costs an axis."
        ),
    )
    taper: dict[str, str] | None = Field(
        None,
        description=(
            "Per axis, '+' or '-': the direction the part gets THINNER in. "
            "This is what resolves nose-forward from nose-backward, which no "
            "bounding box can. A wing tapering to a tip at -x is {'x': '-'}."
        ),
    )
    spin: str | None = Field(
        None,
        description=(
            "'x', 'y' or 'z': the axis the part is rotationally symmetric "
            "about, for wheels, cowls and propellers whose extents say nothing "
            "useful. Detected in the mesh, not assumed."
        ),
    )
    min_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Leave the part as generated if orientation is less certain than "
            "this. The result is reported either way, so a caller can see the "
            "number it declined."
        ),
    )


class PartPlacement(BaseModel):
    job_id: str = Field(..., description="A completed image_to_3d or primitive job")
    name: str = Field(..., description="glTF node name for this part")
    orient: Orient | str | list[float] | None = Field(
        None,
        description=(
            "Rotate this part into the declared frame before anything else "
            "happens — including anchors, which then measure the part as it "
            "will appear. A bare string is a role name and a bare [x, y, z] is "
            "target extents."
        ),
    )
    position: list[float] | None = Field(
        None,
        description=(
            "[x, y, z]. With an anchor, this supplies only the axes the anchor "
            "does not constrain."
        ),
    )
    rotation: list[float] | None = Field(None, description="[rx, ry, rz] in degrees")
    scale: float | list[float] | None = None
    anchor: Anchor | None = Field(
        None, description="Place this part relative to another part or the ground."
    )
    mirror: str | None = Field(
        None,
        description=(
            "Reflect this part's placement across the world plane 'x', 'y' or "
            "'z' = 0. Face winding is corrected, so a mirrored part is not "
            "inside-out."
        ),
    )
    mirror_of: str | None = Field(
        None,
        description=(
            "This part is another part reflected — the left/right pair of a "
            "gear leg or a wing. It takes that part's whole transform, so it "
            "cannot also set position or anchor, and placing the original "
            "correctly places both."
        ),
    )
    material: str | None = Field(
        None,
        description=(
            "Override the material guessed from the part name. "
            f"One of {materials.families()}."
        ),
    )
    color: str | None = Field(
        None, description='Base colour as "#rrggbb". Keeps the material family.'
    )
    # Assemble from the dense original instead of the decimated export. Useful
    # when one part needs detail the rest of the scene does not.
    use_raw: bool = False


class AssembleRequest(BaseModel):
    parts: list[PartPlacement]
    scene_name: str | None = None
    apply_materials: bool = Field(
        True,
        description=(
            "Assign a PBR material to each part from its name. On by default "
            "because the alternative is uniform grey; see server/materials.py."
        ),
    )


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


def _orient_spec(spec):
    """A validated Orient back to the plain dict assemble.py takes.

    exclude_none: a missing `extents` means "whatever the role says", which is
    not the same request as an explicit null — and orient.py rejects keys it
    does not recognise, so a typo is an error rather than a silently ignored
    declaration.
    """
    if not isinstance(spec, Orient):
        return spec
    return spec.model_dump(exclude_none=True)


def _recorded_material(job_id: str) -> str | None:
    job = jobs.get(job_id)
    result = (job or {}).get("result") or {}
    return result.get("material")


@api.post("/assemble")
def assemble_scene(req: AssembleRequest):
    """Compose finished parts into a single glTF with one node per part."""
    if not req.parts:
        raise HTTPException(400, "no parts given")

    resolved = [
        {
            "name": p.name,
            "mesh_path": str(_part_mesh_path(p)),
            "orient": _orient_spec(p.orient),
            "position": p.position,
            "rotation": p.rotation,
            "scale": p.scale,
            # exclude_none so an omitted `align` stays omitted: assemble reads
            # "no align key" as "centre me inside the target", which is not the
            # same request as an explicit null.
            "anchor": p.anchor.model_dump(exclude_none=True) if p.anchor else None,
            "mirror": p.mirror,
            "mirror_of": p.mirror_of,
            # A scripted part already knows its material; falling back to
            # guessing from the node name would turn a wooden barrel into a
            # metal one, since "barrel" reads as gun barrel.
            "material": p.material or _recorded_material(p.job_id),
            "color": p.color,
        }
        for p in req.parts
    ]

    scene_id = uuid.uuid4().hex[:12]
    out = config.OUT_DIR / "scenes" / scene_id / f"{req.scene_name or 'scene'}.glb"
    try:
        result = assembly.assemble(resolved, out, req.apply_materials)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"scene_id": scene_id, **result}


class ImageRequest(BaseModel):
    prompt: str = Field(..., description="What the object is, e.g. 'a wooden crate'")
    provider: str | None = None
    image_size: str = "square_hd"
    seed: int | None = None
    remove_background: bool = True


@api.post("/images")
def create_image(req: ImageRequest):
    """Prompt -> reference image, stored for reuse across parts."""
    try:
        provider = imagegen.get_provider(req.provider)
        raw = provider.generate(req.prompt, image_size=req.image_size, seed=req.seed)
        image_id, path = imagegen.store(raw, req.remove_background)
    except imagegen.ImageGenError as exc:
        raise HTTPException(502, str(exc)) from exc

    return {
        "image_id": image_id,
        "path": str(path),
        "provider": provider.name,
        "prompt": req.prompt,
        "bytes": path.stat().st_size,
    }


@api.get("/images/providers")
def image_providers():
    return {"providers": imagegen.provider_status()}


@api.get("/images/{image_id}")
def get_image(image_id: str):
    path = imagegen.image_path(image_id)
    if not path.exists():
        raise HTTPException(404, f"no such image: {image_id}")
    return FileResponse(path, media_type="image/png", filename=f"{image_id}.png")


@api.post("/decompose")
def run_decomposition(plan: dict):
    """Build every part of a plan, and return an /assemble request for them.

    Long-running — a plan is many image generations and many meshes. The parts
    land in the job registry as they finish, so /jobs shows progress while this
    is still open.
    """
    try:
        return decompose.run(decompose.Plan.from_dict(plan))
    except decompose.DecomposeError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/decompose/examples")
def decomposition_examples():
    """Worked plans. This is how a coding agent learns the format."""
    return {"examples": decompose.EXAMPLES}


class HollowRequest(BaseModel):
    job_id: str = Field(..., description="A completed job to hollow out")
    wall_thickness: float = Field(0.04, description="In the mesh's own units")
    resolution: int = Field(
        64,
        description=(
            "Voxels along the longest axis. Counter-intuitively a CRACKED mesh "
            "wants a COARSER grid: sealing a crack of width w needs a seal of "
            "w/2 voxels, and the seal displaces the skin."
        ),
    )
    openings: list[dict] | None = Field(
        None, description="Apertures to cut, so the cavity can be reached"
    )
    max_faces: int = Field(20000, description="Roblox's per-mesh cap")


@api.post("/hollow")
def hollow_job(req: HollowRequest):
    """Carve a cavity into a finished mesh, recorded as a new job."""
    job = jobs.get(req.job_id)
    if job is None:
        raise HTTPException(404, f"no such job: {req.job_id}")
    if job["status"] != jobs.DONE:
        raise HTTPException(409, f"job is {job['status']}, not done")

    job_id = uuid.uuid4().hex[:12]
    out_dir = config.OUT_DIR / job_id
    kwargs = {"wall_thickness": req.wall_thickness, "resolution": req.resolution,
              "max_faces": req.max_faces}
    if req.openings:
        kwargs["openings"] = req.openings
    try:
        result = hollowing.hollow_file(
            Path(job["result"]["mesh_path"]), out_dir / "mesh.glb", **kwargs
        )
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    now = time.time()
    record = {
        "id": job_id, "type": "hollow", "status": jobs.DONE,
        "created_at": now, "started_at": now, "finished_at": time.time(),
        "params": {"source_job": req.job_id, **kwargs},
        "result": result, "error": None,
    }
    _record(record)
    return record


@api.get("/hollow/primitives")
def hollow_catalogue():
    """Kinds that are hollow by construction — cheaper and cleaner than carving."""
    return {"kinds": hollowing.catalogue()}


@api.post("/hollow/primitives")
def create_hollow_primitive(req: PrimitiveRequest):
    job_id = uuid.uuid4().hex[:12]
    try:
        result = hollowing.store(
            req.kind, req.params, config.OUT_DIR / job_id,
            part_name=req.part_name, material=req.material, color=req.color,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    now = time.time()
    job = {
        "id": job_id, "type": "hollow_primitive", "status": jobs.DONE,
        "created_at": now, "started_at": now, "finished_at": time.time(),
        "params": {"kind": req.kind, "part_name": req.part_name or req.kind},
        "result": result, "error": None,
    }
    _record(job)
    return job


@api.get("/materials")
def list_materials():
    """The material families a part name can resolve to."""
    return {"families": materials.families(), "default": materials.DEFAULT_MATERIAL}


@api.get("/orient/roles")
def list_orient_roles():
    """Named shapes a part can be oriented into, with what each one declares."""
    return {"roles": orienting.ROLES}


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


class ExportRequest(BaseModel):
    job_id: str | None = Field(None, description="Export a single generated part")
    scene_id: str | None = Field(None, description="Export an assembled scene")
    target: str = Field("roblox", description=f"one of {export.TARGETS}")
    height_studs: float | None = Field(
        None,
        description=(
            "Rescale so the result is this tall. Generated meshes normalise to "
            "~2 units and one unit is one stud, so without this everything "
            "arrives knee-high."
        ),
    )


@api.post("/export")
def export_mesh(req: ExportRequest):
    """Write a part or scene out under a target's constraints."""
    if bool(req.job_id) == bool(req.scene_id):
        raise HTTPException(400, "give exactly one of job_id or scene_id")

    if req.job_id:
        job = jobs.get(req.job_id)
        if job is None:
            raise HTTPException(404, f"no such job: {req.job_id}")
        if job["status"] != jobs.DONE:
            raise HTTPException(409, f"job is {job['status']}, not done")
        source = Path(job["result"]["mesh_path"])
        out_dir = source.parent / "export" / req.target
    else:
        scene_dir = config.OUT_DIR / "scenes" / req.scene_id
        matches = sorted(scene_dir.glob("*.glb")) if scene_dir.exists() else []
        if not matches:
            raise HTTPException(404, f"no such scene: {req.scene_id}")
        source = matches[0]
        out_dir = scene_dir / "export" / req.target

    try:
        return export.export_for(source, req.target, out_dir, req.height_studs)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/export/file")
def get_exported_file(path: str):
    """Serve a file produced by /export, by the absolute path it reported.

    Restricted to OUT_DIR: this takes a caller-supplied path, and without the
    check it would happily serve anything on the machine.
    """
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(config.OUT_DIR.resolve()):
        raise HTTPException(403, "path is outside the output directory")
    if not resolved.is_file():
        raise HTTPException(404, f"no such file: {path}")
    return FileResponse(resolved, filename=resolved.name)


@api.post("/admin/unload")
def unload_model():
    """Drop every generator's model from VRAM.

    Unconditionally both, because the caller's actual intent is "give me the
    card back" and it should not have to know which model happens to be
    resident. Generators also do this to each other at the start of a job —
    7.63 + 6.88 GiB does not fit in 8.88 — so this endpoint is for handing the
    GPU to something outside the server.
    """
    for generator in jobs.GENERATORS.values():
        generator.unload()
    return {
        "model_loaded": pipeline.model_loaded() or trellis.model_loaded(),
        "generators_loaded": {
            name: generator.model_loaded()
            for name, generator in jobs.GENERATORS.items()
        },
        "gpu": pipeline.vram_stats(),
    }
