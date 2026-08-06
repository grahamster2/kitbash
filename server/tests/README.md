# Server tests

CPU-only. They run on the dev laptop, which has no GPU and no CUDA.

## Running them

```bash
cd server
uv pip install pytest fastapi httpx trimesh numpy pillow   # into whatever env you use
python -m pytest
```

`pytest.ini` puts `server/` on the import path, because the modules import each
other as top-level names (`import config`) the way uvicorn runs them.

354 tests, ~2s.

## What is covered

| file | subject |
| --- | --- |
| `test_jobs.py` | submit/get/listing/stats, the job.json round trip, `rehydrate()`, history eviction, generator dispatch, and that the input image never reaches disk or an API response |
| `test_trellis.py` | the TRELLIS 2 subprocess boundary with the subprocess faked: the settings sent, the alpha guarantee, the result contract, timeouts and worker errors, and that Hunyuan3D is evicted before the worker spawns |
| `test_assemble.py` | `_transform` composition order, node-name deduplication, `describe()`, multi-part scenes |
| `test_export.py` | the per-geometry triangle budget, `height_studs`, the roblox vs dcc pivot, texture/vertex-colour warnings |
| `test_primitives.py` | the scripted library: every kind watertight, dimensioned as requested and under the face cap; the catalogue schema; parameter validation |
| `test_app.py` | every endpoint via `TestClient`, including the `/export/file` traversal guard and that a scripted part assembles and exports like a generated one |

## What is deliberately not covered

- **GPU generation.** `generate_shape` is stubbed on *both* generators for every
  test by an autouse fixture (`conftest.stub_generate`) that writes a real box
  mesh instead. Nothing here loads Hunyuan3D, imports torch, touches CUDA, or
  spawns the TRELLIS interpreter — those paths are exercised by running the
  server on the GPU box. `pipeline.py` and `trellis.py` are both import-safe
  without torch, so no import-level faking is needed.
- **`server/trellis_worker.py`.** It runs inside `D:\trellis2`'s venv and every
  line of it is a call into the node pack, so there is nothing to assert on
  without the GPU. `test_trellis.py` tests the protocol it speaks, not the
  script.
- **The worker thread's concurrency.** `test_jobs.py` starts the real thread once
  to prove a queued job gets run, but the endpoint tests call `jobs._run_one`
  inline instead; asserting on job state while a thread races you is how you get
  a flaky suite. Nothing here tests two generations overlapping, because the
  queue exists precisely to make that impossible.
- **Whether a primitive *looks* good.** The tests assert watertightness, exact
  dimensions and face counts, none of which distinguish a designed crate from a
  chamfered box. That check is a Blender render, and the ones that were looked
  at are in [docs/PROCEDURAL.md](../../docs/PROCEDURAL.md).
- **Whether the exported files import into Roblox Studio.** The triangle,
  scale and pivot constraints are asserted; that Studio then accepts the file is
  a manual check.
