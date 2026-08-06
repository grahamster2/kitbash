"""Fixtures for the server suite.

The suite runs on the dev laptop, which has no GPU and no CUDA. pipeline.py is
still importable there — torch is imported inside the functions that need it,
never at module scope — so the only thing that has to be faked is
`generate_shape`, and it is faked for every test so a stray worker thread can
never reach the GPU path.
"""
import itertools
import json
import os
import queue
import sys
import tempfile
import time
from pathlib import Path

import pytest
import trimesh

SERVER_DIR = Path(__file__).resolve().parents[1]
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

# config resolves OUT_DIR at import time and its default is the reference GPU
# box's D:\kitbash-out, which on Linux is a *relative* path that would be
# created inside the repo. Every test overrides it again via the out_dir
# fixture; this only keeps the import itself harmless.
os.environ.setdefault(
    "KITBASH_OUT_DIR", str(Path(tempfile.gettempdir()) / "kitbash-tests-import")
)

import config  # noqa: E402
import jobs  # noqa: E402
import pipeline  # noqa: E402

# A real 1x1 PNG. Nothing decodes it — generation is stubbed — but request
# bodies here should still look like the ones a client sends.
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def out_dir(tmp_path, monkeypatch):
    """Give each test its own OUT_DIR and an empty job registry.

    The queue is replaced rather than drained: a worker thread started by an
    earlier test is parked in `get()` on the *old* queue object forever, so it
    cannot steal work from this one.
    """
    d = tmp_path / "out"
    d.mkdir()
    monkeypatch.setattr(config, "OUT_DIR", d)
    jobs._jobs.clear()
    jobs._images.clear()
    jobs._queue = queue.Queue()
    jobs._worker = None
    return d


@pytest.fixture(autouse=True)
def stub_generate(monkeypatch):
    """Stand in for the one function that needs a GPU.

    Writes a real box mesh, so everything downstream of generation — describe,
    assemble, export — still runs against genuine geometry.
    """

    def fake_generate_shape(image_b64: str, out_dir: Path, params: dict) -> dict:
        if not image_b64:
            raise ValueError("image_to_3d requires an image")
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
        mesh_path = out_dir / "mesh.glb"
        mesh.export(str(mesh_path))
        return {
            "mesh_path": str(mesh_path),
            "generation_seconds": 0.0,
            "peak_vram_gib": 0.0,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "decimated_from": None,
            "watertight": True,
            "file_bytes": mesh_path.stat().st_size,
            "params": dict(params),
        }

    monkeypatch.setattr(pipeline, "generate_shape", fake_generate_shape)
    return fake_generate_shape


@pytest.fixture
def make_mesh(tmp_path):
    """Write geometry to a .glb and return its path.

    `make_mesh(mesh)` for a single-geometry file; `make_mesh(parts=[(name, mesh),
    (name, mesh, transform)])` for a scene with named nodes.
    """
    counter = itertools.count()

    def _make(mesh=None, parts=None, name=None) -> Path:
        path = tmp_path / (name or f"mesh_{next(counter)}.glb")
        if mesh is not None:
            mesh.export(str(path))
            return path
        scene = trimesh.Scene()
        for geom_name, geom, *rest in parts:
            scene.add_geometry(
                geom,
                node_name=geom_name,
                geom_name=geom_name,
                transform=rest[0] if rest else None,
            )
        scene.export(str(path))
        return path

    return _make


def write_job_json(out_dir: Path, job_id: str, status: str, created_at: float = 0.0,
                   **extra) -> dict:
    """Plant a job.json the way a previous server run would have left it."""
    d = out_dir / job_id
    d.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "type": "image_to_3d",
        "status": status,
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "params": {},
        "result": None,
        "error": None,
        **extra,
    }
    (d / "job.json").write_text(json.dumps(job))
    return job


def wait_for(job_id: str, status: str, timeout: float = 5.0) -> dict:
    """Poll until a job reaches `status`. Only the real-worker test needs this."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = jobs.get(job_id)
        if job and job["status"] == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {status}: {jobs.get(job_id)}")


@pytest.fixture
def client(monkeypatch):
    """TestClient with the background worker replaced by inline execution.

    The worker is a thread; letting it run would make every assertion about job
    state a race. The `finished_job` fixture drives the same `_run_one` body the
    thread would have called.
    """
    from fastapi.testclient import TestClient

    import app

    monkeypatch.setattr(jobs, "start_worker", lambda: None)
    with TestClient(app.api) as c:
        yield c


@pytest.fixture
def finished_job(client):
    """Submit a job over the API and run it to completion, returning its id."""

    def _submit(part_name: str = "part", **params) -> str:
        body = {"image_b64": PNG_B64, "part_name": part_name, **params}
        job_id = client.post("/jobs", json=body).json()["id"]
        jobs._run_one(job_id)
        assert jobs.get(job_id)["status"] == jobs.DONE
        return job_id

    return _submit
