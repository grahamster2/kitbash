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

114 tests, ~1s.

## What is covered

| file | subject |
| --- | --- |
| `test_jobs.py` | submit/get/listing/stats, the job.json round trip, `rehydrate()`, history eviction, and that the input image never reaches disk or an API response |
| `test_assemble.py` | `_transform` composition order, node-name deduplication, `describe()`, multi-part scenes |
| `test_export.py` | the per-geometry triangle budget, `height_studs`, the roblox vs dcc pivot, texture/vertex-colour warnings |
| `test_app.py` | every endpoint via `TestClient`, including the `/export/file` traversal guard |

## What is deliberately not covered

- **GPU generation.** `pipeline.generate_shape` is stubbed for every test by an
  autouse fixture (`conftest.stub_generate`) that writes a real box mesh instead.
  Nothing here loads Hunyuan3D, imports torch, or touches CUDA — that path is
  exercised by running the server on the GPU box. `pipeline.py` itself is
  import-safe without torch, so no import-level faking is needed.
- **The worker thread's concurrency.** `test_jobs.py` starts the real thread once
  to prove a queued job gets run, but the endpoint tests call `jobs._run_one`
  inline instead; asserting on job state while a thread races you is how you get
  a flaky suite. Nothing here tests two generations overlapping, because the
  queue exists precisely to make that impossible.
- **Whether the exported files import into Roblox Studio.** The triangle,
  scale and pivot constraints are asserted; that Studio then accepts the file is
  a manual check.
