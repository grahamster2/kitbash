"""The TRELLIS 2 subprocess boundary, with the subprocess faked.

Nothing here starts an interpreter or touches D:\\trellis2. What is worth testing
on a laptop is the contract either side of the pipe: that the request carries the
settings that were measured to work, that the result comes back with the keys
pipeline.py produces, and that the failure modes turn into readable errors rather
than a stack trace out of json.loads.
"""
import base64
import io
import json
import subprocess
import sys

import pytest
import trimesh
from PIL import Image

import config
import pipeline
import trellis

# Captured at import, before conftest's autouse fixture swaps in the stub that
# keeps every *other* test away from the GPU. This module wants the real one.
generate_shape = trellis.generate_shape


@pytest.fixture(autouse=True)
def installed(monkeypatch, tmp_path):
    """Make available() say yes without a TRELLIS install on this box."""
    for name in ("TRELLIS_PYTHON", "TRELLIS_WORKER", "TRELLIS_COMFY"):
        path = tmp_path / name.lower()
        path.mkdir()
        monkeypatch.setattr(config, name, path)


WORKER_RESULT = {
    "vertices": 10_442,
    "faces": 20_691,
    "decimated_from": 9_762_008,
    "watertight": False,
    "peak_vram_gib": 4.11,
    "device_peak_vram_gib": 6.88,
    "worker_seconds": 148.9,
    "stages": {"load_model": 8.2, "generate": 100.8, "postprocess_uv_bake": 50.2},
    "has_uv": True,
    "base_color_texture": "2048x2048",
}


@pytest.fixture
def fake_worker(monkeypatch):
    """Replace the subprocess with a recorder that writes a real mesh.

    Exposes `.requests` and `.calls` so a test can assert on what crossed the
    pipe, and `.payload` / `.emit_result` / `.stderr` so it can misbehave.
    """

    def run(argv, input=None, **kwargs):
        run.calls.append((argv, kwargs))
        request = json.loads(input)
        run.requests.append(request)
        payload = dict(run.payload)
        if "error" not in payload:
            trimesh.creation.box().export(request["mesh_path"])
        # The node loader writes hundreds of lines before the result appears.
        stdout = "ComfyUI node chatter\nmore chatter\n"
        if run.emit_result:
            stdout += trellis.RESULT_PREFIX + json.dumps(payload) + "\n"
        return subprocess.CompletedProcess(argv, 0, stdout, run.stderr)

    run.calls, run.requests = [], []
    run.payload, run.emit_result, run.stderr = WORKER_RESULT, True, ""
    monkeypatch.setattr(subprocess, "run", run)
    return run


def opaque_png(size=(8, 8)) -> str:
    buf = io.BytesIO()
    Image.new("RGBA", size, (200, 30, 30, 255)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def matted_png(size=(8, 8)) -> str:
    image = Image.new("RGBA", size, (200, 30, 30, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def decode(image_b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGBA")


def generate(out_dir, **params):
    return generate_shape(matted_png(), out_dir / "job", params)


# --- the request that crosses the pipe --------------------------------------

def test_the_defaults_sent_are_the_settings_that_actually_completed(fake_worker, out_dir):
    """1024_cascade/4096 was killed at 21 minutes on a solid crate; 512/2048
    completed. The defaults have to be the ones that finished."""
    generate(out_dir)

    request = fake_worker.requests[0]
    assert request["pipeline_type"] == "512"
    assert request["texture_size"] == 2048
    assert request["quant"] == "GGUF Q6_K"
    assert request["target_faces"] == 20_000
    assert request["steps"] == 12


def test_params_override_every_default(fake_worker, out_dir):
    generate(
        out_dir, quant="GGUF Q4_K_M", pipeline_type="1024_cascade",
        texture_size=4096, num_inference_steps=20, seed=7, target_faces=5000,
    )

    request = fake_worker.requests[0]
    assert request["quant"] == "GGUF Q4_K_M"
    assert request["pipeline_type"] == "1024_cascade"
    assert request["texture_size"] == 4096
    assert request["steps"] == 20
    assert request["seed"] == 7
    assert request["target_faces"] == 5000


def test_the_mesh_path_is_chosen_by_the_server_not_the_worker(fake_worker, out_dir):
    result = generate(out_dir)

    assert fake_worker.requests[0]["mesh_path"] == str(out_dir / "job" / "mesh.glb")
    assert result["mesh_path"] == str(out_dir / "job" / "mesh.glb")
    assert (out_dir / "job" / "mesh.glb").exists()


def test_texturing_is_on_by_default_and_switchable(fake_worker, out_dir):
    assert generate(out_dir)["textured"] is True
    assert generate(out_dir, textured=False)["textured"] is False

    assert [r["textured"] for r in fake_worker.requests] == [True, False]


def test_the_worker_runs_in_its_own_interpreter_against_the_comfy_tree(fake_worker, out_dir):
    generate(out_dir)

    argv, kwargs = fake_worker.calls[0]
    assert argv == [str(config.TRELLIS_PYTHON), str(config.TRELLIS_WORKER)]
    # The node pack resolves its imports relative to ComfyUI's root.
    assert kwargs["cwd"] == str(config.TRELLIS_COMFY)
    assert kwargs["timeout"] == config.TRELLIS_TIMEOUT
    assert kwargs["env"]["HF_HOME"] == config.TRELLIS_HF_HOME


# --- alpha, the documented input hazard -------------------------------------

def test_an_opaque_image_is_matted_before_it_reaches_the_model(fake_worker, out_dir,
                                                               monkeypatch):
    """An opaque background gets reconstructed as geometry, which is what sent
    the crate into the stall. Hunyuan3D, given the same image, does not care."""
    monkeypatch.setitem(sys.modules, "rembg", _FakeRembg)

    generate_shape(opaque_png(), out_dir / "job", {})

    sent = decode(fake_worker.requests[0]["image_b64"])
    assert sent.getchannel("A").getextrema()[0] == 0
    assert fake_worker.requests[0]["remove_background"] is False


def test_an_image_that_already_has_alpha_is_passed_through_untouched(fake_worker, out_dir):
    image_b64 = matted_png()

    generate_shape(image_b64, out_dir / "job", {})

    assert fake_worker.requests[0]["image_b64"] == image_b64
    assert fake_worker.requests[0]["remove_background"] is False


def test_the_node_is_asked_to_matte_when_rembg_is_unavailable(fake_worker, out_dir,
                                                              monkeypatch):
    monkeypatch.setitem(sys.modules, "rembg", None)

    generate_shape(opaque_png(), out_dir / "job", {})

    # Slower in-model removal beats handing TRELLIS 2 a white backdrop.
    assert fake_worker.requests[0]["remove_background"] is True


def test_a_data_url_is_accepted(fake_worker, out_dir):
    generate_shape("data:image/png;base64," + matted_png(), out_dir / "job", {})

    assert decode(fake_worker.requests[0]["image_b64"]).size == (8, 8)


class _FakeRembg:
    @staticmethod
    def remove(image):
        out = image.copy()
        out.putpixel((0, 0), (0, 0, 0, 0))
        return out


# --- the result contract ----------------------------------------------------

def test_the_result_carries_every_key_the_hunyuan_path_produces(fake_worker, out_dir):
    result = generate(out_dir)

    assert set(result) >= {
        "mesh_path", "generation_seconds", "peak_vram_gib", "vertices", "faces",
        "decimated_from", "watertight", "file_bytes", "params",
    }
    assert result["vertices"] == 10_442
    assert result["faces"] == 20_691
    assert result["decimated_from"] == 9_762_008
    assert result["watertight"] is False
    assert result["generator"] == "trellis2"


def test_file_bytes_is_measured_from_disk_not_reported_by_the_worker(fake_worker, out_dir):
    result = generate(out_dir)

    assert result["file_bytes"] == (out_dir / "job" / "mesh.glb").stat().st_size


def test_generation_seconds_covers_the_whole_subprocess(fake_worker, out_dir):
    """Interpreter start and the node-pack import are real cost to the caller;
    reporting only the worker's own clock would understate the tier."""
    result = generate(out_dir)

    assert result["generation_seconds"] != WORKER_RESULT["worker_seconds"]
    assert result["stages"] == WORKER_RESULT["stages"]


def test_both_vram_figures_survive(fake_worker, out_dir):
    result = generate(out_dir)

    assert result["peak_vram_gib"] == 4.11
    assert result["device_peak_vram_gib"] == 6.88


def test_the_texture_is_reported_as_measured_not_as_requested(fake_worker, out_dir):
    fake_worker.payload = {**WORKER_RESULT, "base_color_texture": None, "has_uv": False}

    result = generate(out_dir, textured=True)

    assert result["textured"] is True
    assert result["base_color_texture"] is None
    assert result["has_uv"] is False


def test_the_params_recorded_are_what_was_sent(fake_worker, out_dir):
    result = generate(out_dir, seed=11)

    assert result["params"] == {
        "quant": "GGUF Q6_K", "pipeline_type": "512", "texture_size": 2048,
        "num_inference_steps": 12, "seed": 11, "target_faces": 20_000,
        "textured": True,
    }


# --- failure modes ----------------------------------------------------------

def test_a_worker_error_becomes_a_runtime_error_carrying_its_message(fake_worker, out_dir):
    fake_worker.payload = {"error": "RuntimeError: CUDA out of memory",
                           "traceback": "Traceback..."}

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        generate(out_dir)


def test_a_worker_that_prints_no_result_reports_its_output(fake_worker, out_dir):
    fake_worker.emit_result = False
    fake_worker.stderr = "ImportError: No module named 'nodes'"

    with pytest.raises(RuntimeError, match="No module named"):
        generate(out_dir)


def test_the_result_is_found_under_the_node_packs_stdout_chatter(fake_worker, out_dir):
    """ComfyUI writes hundreds of lines to stdout, so the result cannot be
    "whatever was printed last"."""
    assert generate(out_dir)["faces"] == 20_691


def test_a_timeout_names_the_stall_it_is_there_to_catch(monkeypatch, out_dir):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", config.TRELLIS_TIMEOUT)

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(RuntimeError, match="memory-thrash"):
        generate(out_dir)


def test_a_missing_install_is_reported_before_anything_is_spawned(monkeypatch, out_dir,
                                                                  tmp_path):
    monkeypatch.setattr(config, "TRELLIS_PYTHON", tmp_path / "nope.exe")

    with pytest.raises(RuntimeError, match="not installed here"):
        generate(out_dir)


def test_available_lists_what_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "TRELLIS_WORKER", tmp_path / "gone.py")

    assert trellis.available() == {"available": False,
                                   "missing": [str(tmp_path / "gone.py")]}


def test_available_is_true_when_the_install_is_present():
    assert trellis.available() == {"available": True, "missing": []}


# --- residency --------------------------------------------------------------

def test_hunyuan3d_is_evicted_before_the_worker_starts(fake_worker, out_dir, monkeypatch):
    """7.63 + 6.88 GiB does not fit in 8.88, so this is not an optimisation."""
    order = []
    monkeypatch.setattr(pipeline, "unload", lambda: order.append("unload"))
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (order.append("spawn"), fake_worker(*a, **kw))[1],
    )

    generate(out_dir)

    assert order == ["unload", "spawn"]


def test_model_loaded_is_true_only_while_a_worker_is_running(fake_worker, out_dir,
                                                             monkeypatch):
    observed = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **kw: (observed.append(trellis.model_loaded()), fake_worker(*a, **kw))[1],
    )

    assert trellis.model_loaded() is False
    generate(out_dir)

    assert observed == [True]
    assert trellis.model_loaded() is False


def test_model_loaded_returns_to_false_after_a_failed_run(monkeypatch, out_dir):
    def boom(*args, **kwargs):
        raise OSError("cannot spawn")

    monkeypatch.setattr(subprocess, "run", boom)

    with pytest.raises(OSError):
        generate(out_dir)

    assert trellis.model_loaded() is False


def test_unload_is_a_no_op_because_the_worker_already_exited():
    """Nothing to free: the worker exits after every job and takes its
    allocations with it. It exists so callers need not branch on generator."""
    trellis.unload()

    assert trellis.model_loaded() is False
