"""Single-worker job queue.

One worker, deliberately. Two concurrent generations do not fit in 8.88 GiB,
and a queue that accepts everything and runs one at a time is far easier for a
client to reason about than one that rejects work when busy — a multi-part
build submits six parts at once and expects all six back.

Job records are mirrored to disk as job.json so a server restart (or the
reboot-survival test) does not lose completed results.
"""
import json
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import config
import pipeline
import texturing
import trellis

log = logging.getLogger("kitbash.jobs")

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"

# Name -> the module that owns the model. Both expose generate_shape with the
# same signature and result keys, which is the entire interface; the modules are
# stored rather than the functions so a test can monkeypatch generate_shape.
GENERATORS = {"hunyuan3d": pipeline, "trellis2": trellis}

_jobs: "OrderedDict[str, dict]" = OrderedDict()
# Input images are kept out of the job record so they never reach job.json or an
# API response — a base64 PNG would bloat both enormously.
_images: dict[str, str | None] = {}
_jobs_lock = threading.Lock()
_queue: "queue.Queue[str]" = queue.Queue()
_worker: threading.Thread | None = None


def _job_dir(job_id: str) -> Path:
    return config.OUT_DIR / job_id


def _persist(job: dict):
    try:
        d = _job_dir(job["id"])
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job, indent=2))
    except Exception:
        # Persistence is a convenience; never fail a job over it.
        log.warning("could not persist job %s", job["id"], exc_info=True)


def _set(job_id: str, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job.update(fields)
        snapshot = dict(job)
    _persist(snapshot)
    return snapshot


def submit(job_type: str, params: dict, image_b64: str | None) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "type": job_type,
        "status": QUEUED,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "params": params,
        "result": None,
        "error": None,
    }
    with _jobs_lock:
        _jobs[job_id] = job
        _images[job_id] = image_b64
        while len(_jobs) > config.MAX_JOB_HISTORY:
            old_id, old = _jobs.popitem(last=False)
            if old["status"] in (QUEUED, RUNNING):
                # Never evict work that has not finished; put it back.
                _jobs[old_id] = old
                _jobs.move_to_end(old_id, last=False)
                break
            _images.pop(old_id, None)
        snapshot = dict(job)
    _persist(snapshot)
    _queue.put(job_id)
    log.info("queued job %s (%s), depth=%d", job_id, job_type, _queue.qsize())
    return snapshot


def get(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            return dict(job)
    # Not in memory — it may predate a restart.
    path = _job_dir(job_id) / "job.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            log.warning("unreadable job.json for %s", job_id, exc_info=True)
    return None


def listing(limit: int = 50) -> list[dict]:
    if limit <= 0:
        # [-0:] is the whole list, so ?limit=0 would return everything.
        return []
    with _jobs_lock:
        return [dict(j) for j in list(_jobs.values())[-limit:]][::-1]


def stats() -> dict:
    with _jobs_lock:
        running = [j["id"] for j in _jobs.values() if j["status"] == RUNNING]
    return {"queue_depth": _queue.qsize(), "running": running}


def _run_one(job_id: str):
    job = get(job_id)
    if job is None:
        return
    _set(job_id, status=RUNNING, started_at=time.time())
    try:
        with _jobs_lock:
            image_b64 = _images.get(job_id)

        if job["type"] != "image_to_3d":
            raise ValueError(f"unknown job type: {job['type']}")
        if not image_b64:
            raise ValueError("image_to_3d requires an image")

        name = job["params"].get("generator", config.DEFAULT_GENERATOR)
        generator = GENERATORS.get(name)
        if generator is None:
            raise ValueError(f"unknown generator: {name}")

        result = generator.generate_shape(image_b64, _job_dir(job_id), job["params"])

        # Paint here rather than in a later call: back-projection needs the
        # reference image, and the image is deliberately dropped the moment the
        # job ends so it never reaches disk or an API response.
        if _wants_texture(job["params"], result):
            result = _back_project(image_b64, result, job["params"])

        _set(job_id, status=DONE, finished_at=time.time(), result=result)
        log.info("job %s done in %ss", job_id, result["generation_seconds"])
    except Exception as exc:
        log.exception("job %s failed", job_id)
        _set(job_id, status=ERROR, finished_at=time.time(),
             error=f"{type(exc).__name__}: {exc}")
    finally:
        with _jobs_lock:
            _images.pop(job_id, None)


def _wants_texture(params: dict, result: dict) -> bool:
    """Default on wherever nothing else supplies colour.

    Hunyuan3D has no texture path at all, and TRELLIS 2 untextured deliberately
    skips its own bake, so both arrive grey. A mesh that already carries a
    base-colour texture is left alone.
    """
    asked = params.get("texture")
    if asked is not None:
        return bool(asked)
    return not result.get("base_color_texture")


def _back_project(image_b64: str, result: dict, params: dict) -> dict:
    """Project the reference photograph onto the finished mesh.

    Failure here must not fail the job — the mesh is still good untextured, and
    semantic materials remain available downstream.
    """
    import base64
    import io

    from PIL import Image

    mesh_path = Path(result["mesh_path"])
    try:
        t0 = time.time()
        image = Image.open(io.BytesIO(base64.b64decode(image_b64)))
        painted, stats = texturing.texture_from_reference(
            str(mesh_path), image, mode=params.get("texture_mode", "uv")
        )
        painted.export(str(mesh_path))
        result = {
            **result,
            "textured_by": "back_projection",
            "texture_seconds": round(time.time() - t0, 1),
            # Surfaced so a caller can fall back to semantic materials rather
            # than ship a mesh painted from a pose that never fitted.
            # Nested: the fit belongs to the camera, not to the paint pass.
            "silhouette_iou": round(
                float((stats.get("camera") or {}).get("silhouette_iou", 0.0)), 3
            ),
            "texture_coverage": round(float(stats.get("coverage", 0.0)), 3),
            "file_bytes": mesh_path.stat().st_size,
        }
        log.info("back-projected %s: iou=%.2f coverage=%.2f",
                 mesh_path.parent.name, result["silhouette_iou"],
                 result["texture_coverage"])
    except Exception:
        log.warning("back-projection failed; keeping the untextured mesh",
                    exc_info=True)
        result = {**result, "textured_by": None}
    return result


def _loop():
    log.info("worker started")
    while True:
        job_id = _queue.get()
        try:
            _run_one(job_id)
        finally:
            _queue.task_done()


def rehydrate():
    """Reload finished jobs from disk at startup.

    Without this, /jobs is empty after every restart even though the results
    are still on disk and /jobs/{id} can find them — which makes the app's
    history panel look like it lost your work.

    Jobs recorded as queued or running did not survive the restart; there is no
    process still working on them, so they are marked failed rather than left
    to look perpetually in-flight.
    """
    if not config.OUT_DIR.exists():
        return
    found = []
    for path in config.OUT_DIR.glob("*/job.json"):
        try:
            job = json.loads(path.read_text())
        except Exception:
            log.warning("skipping unreadable %s", path)
            continue
        if job.get("status") in (QUEUED, RUNNING):
            job["status"] = ERROR
            job["error"] = "server restarted while this job was in flight"
            # Write the correction back, or job.json still says "running" and
            # get() reads that stale status straight back off disk once this
            # job is evicted from memory.
            _persist(job)
        found.append(job)

    found.sort(key=lambda j: j.get("created_at") or 0)
    with _jobs_lock:
        for job in found[-config.MAX_JOB_HISTORY:]:
            _jobs.setdefault(job["id"], job)
    log.info("rehydrated %d jobs from %s", len(found), config.OUT_DIR)


def start_worker():
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    _worker = threading.Thread(target=_loop, name="kitbash-worker", daemon=True)
    _worker.start()
