"""Text-to-image providers. No network: every HTTP call is faked."""
import io
import json
import urllib.error

import pytest
from PIL import Image

import config
import imagegen


def png_bytes(size=(8, 8), color=(255, 0, 0)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeHTTP:
    """Stands in for urllib.request.urlopen.

    Keyed by a substring of the URL so a test can script the queue -> status ->
    result -> download sequence without caring about exact request ids.
    """

    def __init__(self, routes: dict, image: bytes | None = None):
        self.routes = routes
        self.image = image if image is not None else png_bytes()
        self.calls: list[str] = []

    def __call__(self, req, timeout=None):
        url = req if isinstance(req, str) else req.full_url
        self.calls.append(url)
        for key, value in self.routes.items():
            if key in url:
                if isinstance(value, Exception):
                    raise value
                body = value if isinstance(value, bytes) else json.dumps(value).encode()
                return _Resp(body)
        return _Resp(self.image)


class _Resp:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@pytest.fixture(autouse=True)
def no_rembg(monkeypatch):
    """rembg downloads a model on first use, which a test must never do."""
    monkeypatch.setitem(__import__("sys").modules, "rembg", None)


@pytest.fixture
def fal(monkeypatch):
    monkeypatch.setattr(config, "FAL_KEY", "test-key")
    monkeypatch.setattr(config, "FAL_MODEL", "fal-ai/flux/schnell")
    return imagegen.FalProvider()


def test_framing_is_added_to_the_prompt():
    framed = imagegen._framed("a wooden crate")

    # Image-to-3D reconstructs a scene as one fused mesh, so the framing that
    # suppresses background and ground plane is the point of this.
    assert "a wooden crate" in framed
    assert "plain flat white background" in framed
    assert "no ground plane" in framed


def test_framing_strips_a_trailing_period():
    assert "crate," in imagegen._framed("a crate.")


def test_the_queue_app_id_drops_the_model_variant(fal):
    """fal-ai/flux/schnell is polled under fal-ai/flux/requests/{id}."""
    assert fal._app_id() == "fal-ai/flux"


def test_the_queue_app_id_leaves_a_two_segment_model_alone(monkeypatch):
    monkeypatch.setattr(config, "FAL_MODEL", "fal-ai/fast-sdxl")

    assert imagegen.FalProvider(api_key="k")._app_id() == "fal-ai/fast-sdxl"


def test_fal_is_unavailable_without_a_key(monkeypatch):
    monkeypatch.setattr(config, "FAL_KEY", None)

    assert imagegen.FalProvider().available() is False


def test_generating_without_a_key_names_the_variable(monkeypatch):
    monkeypatch.setattr(config, "FAL_KEY", None)

    with pytest.raises(imagegen.ImageGenError, match="FAL_KEY"):
        imagegen.FalProvider().generate("a crate")


def test_generate_polls_the_queue_and_returns_the_image(fal, monkeypatch):
    image = png_bytes(color=(0, 128, 255))
    http = FakeHTTP(
        {
            "/status": {"status": "COMPLETED"},
            "requests/req-1": {"images": [{"url": "https://cdn/img.png"}]},
            "queue.fal.run/fal-ai/flux/schnell": {"request_id": "req-1"},
        },
        image=image,
    )
    monkeypatch.setattr(urllib.request, "urlopen", http)

    assert fal.generate("a crate") == image


def test_generate_sends_the_key_and_the_seed(fal, monkeypatch):
    captured = {}

    def urlopen(req, timeout=None):
        # _download passes a bare URL string; everything else a Request.
        if isinstance(req, str):
            return _Resp(png_bytes())
        if req.data:
            captured["headers"] = req.headers
            captured["body"] = json.loads(req.data)
            return _Resp(json.dumps({"request_id": "r"}).encode())
        if "/status" in req.full_url:
            return _Resp(json.dumps({"status": "COMPLETED"}).encode())
        return _Resp(json.dumps({"images": [{"url": "https://cdn/i.png"}]}).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    fal.generate("a crate", seed=7)

    assert captured["headers"]["Authorization"] == "Key test-key"
    assert captured["body"]["seed"] == 7
    assert captured["body"]["num_images"] == 1


@pytest.mark.parametrize("code", [401, 403])
def test_a_rejected_key_says_so_rather_than_repeating_the_status(fal, monkeypatch, code):
    err = urllib.error.HTTPError("u", code, "no", {}, io.BytesIO(b"nope"))
    monkeypatch.setattr(urllib.request, "urlopen", FakeHTTP({"queue.fal.run": err}))

    with pytest.raises(imagegen.ImageGenError, match="rejected the API key"):
        fal.generate("a crate")


def test_an_unreachable_host_is_reported_as_such(fal, monkeypatch):
    err = urllib.error.URLError("name resolution failed")
    monkeypatch.setattr(urllib.request, "urlopen", FakeHTTP({"queue.fal.run": err}))

    with pytest.raises(imagegen.ImageGenError, match="could not reach fal.ai"):
        fal.generate("a crate")


def test_a_failed_queue_job_raises(fal, monkeypatch):
    http = FakeHTTP({
        "/status": {"status": "FAILED"},
        "queue.fal.run/fal-ai/flux/schnell": {"request_id": "r"},
    })
    monkeypatch.setattr(urllib.request, "urlopen", http)

    with pytest.raises(imagegen.ImageGenError, match="FAILED"):
        fal.generate("a crate")


def test_polling_gives_up_at_the_deadline(fal, monkeypatch):
    http = FakeHTTP({
        "/status": {"status": "IN_QUEUE"},
        "queue.fal.run/fal-ai/flux/schnell": {"request_id": "r"},
    })
    monkeypatch.setattr(urllib.request, "urlopen", http)
    monkeypatch.setattr(imagegen.time if hasattr(imagegen, "time") else __import__("time"),
                        "sleep", lambda _: None)

    with pytest.raises(imagegen.ImageGenError, match="did not finish"):
        fal.generate("a crate", poll_seconds=0)


def test_a_missing_request_id_is_reported(fal, monkeypatch):
    monkeypatch.setattr(
        urllib.request, "urlopen", FakeHTTP({"queue.fal.run": {"detail": "busy"}})
    )

    with pytest.raises(imagegen.ImageGenError, match="request_id"):
        fal.generate("a crate")


def test_a_result_with_no_images_is_reported(fal, monkeypatch):
    http = FakeHTTP({
        "/status": {"status": "COMPLETED"},
        "requests/r": {"images": []},
        "queue.fal.run/fal-ai/flux/schnell": {"request_id": "r"},
    })
    monkeypatch.setattr(urllib.request, "urlopen", http)

    with pytest.raises(imagegen.ImageGenError, match="no image"):
        fal.generate("a crate")


def test_the_local_provider_is_honest_about_not_existing():
    provider = imagegen.LocalProvider()

    assert provider.available() is False
    with pytest.raises(imagegen.ImageGenError, match="not implemented"):
        provider.generate("a crate")


def test_get_provider_rejects_an_unknown_name():
    with pytest.raises(imagegen.ImageGenError, match="unknown image provider"):
        imagegen.get_provider("midjourney")


def test_provider_status_marks_the_selected_one(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_PROVIDER", "local")

    selected = [p["name"] for p in imagegen.provider_status() if p["selected"]]

    assert selected == ["local"]


def test_store_normalises_to_rgba_png(out_dir):
    image_id, path = imagegen.store(png_bytes(), remove_background=False)

    assert path == imagegen.image_path(image_id)
    with Image.open(path) as saved:
        assert saved.format == "PNG"
        assert saved.mode == "RGBA"


def test_store_survives_background_removal_failing(out_dir, monkeypatch):
    """A white background still works; losing the image would not."""
    image_id, path = imagegen.store(png_bytes(), remove_background=True)

    assert path.exists()


def test_load_b64_round_trips(out_dir):
    import base64

    image_id, path = imagegen.store(png_bytes(), remove_background=False)

    assert base64.b64decode(imagegen.load_b64(image_id)) == path.read_bytes()


def test_load_b64_names_the_missing_image(out_dir):
    with pytest.raises(FileNotFoundError, match="nope"):
        imagegen.load_b64("nope")
