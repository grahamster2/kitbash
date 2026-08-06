"""Text to reference image — the missing front of the pipeline.

Everything downstream is image-conditioned, so without this a caller has to
supply artwork before it can generate anything. "Build me a plane" only works
once something can turn that sentence into a picture of a plane.

Three provider tiers, one interface:

- **fal** — the user's own fal.ai key. No VRAM, works on any hardware, and the
  user is billed by fal directly, so Kitbash never touches money. This is the
  one that is actually implemented.
- **local** — an image model on the same GPU. Scaffolded; see LocalProvider for
  what it needs. It must free its VRAM before returning, because image and 3D
  generation are sequential stages that cannot both be resident on a 10 GB card.
- **credits** — a hosted account. Not built, and deliberately so: it means
  payments, billing and holding other people's money.

Only stdlib HTTP here. The install story is already the hardest part of this
project on Windows and a provider is not worth another dependency.
"""
import json
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import config

log = logging.getLogger("kitbash.imagegen")

FAL_QUEUE = "https://queue.fal.run"

# Image-to-3D is far more sensitive to framing than to prose. A single centred
# object on a plain background reconstructs well; a scene with ground plane,
# horizon and props produces a mesh with all of that fused into it. Callers
# describe the object and this supplies the framing.
FRAMING = (
    "single isolated {subject}, centered, three-quarter view, full object "
    "visible, plain flat white background, even studio lighting, no shadows on "
    "the background, no ground plane, no scenery, no text, no watermark"
)


class ImageGenError(Exception):
    pass


def _framed(prompt: str) -> str:
    return FRAMING.format(subject=prompt.strip().rstrip("."))


class FalProvider:
    """fal.ai's queue API.

    https://fal.ai/models/fal-ai/flux/schnell/api

    The status and result URLs drop the model's trailing path segment —
    `fal-ai/flux/schnell` polls under `fal-ai/flux/requests/{id}`. Getting that
    wrong 404s in a way that looks like the request vanished.
    """

    name = "fal"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or config.FAL_KEY
        self.model = model or config.FAL_MODEL

    def available(self) -> bool:
        return bool(self.api_key)

    def _app_id(self) -> str:
        parts = self.model.split("/")
        return "/".join(parts[:2]) if len(parts) > 2 else self.model

    def _request(self, url: str, payload: dict | None = None, timeout: int = 60):
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Key {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code in (401, 403):
                raise ImageGenError(
                    f"fal.ai rejected the API key ({exc.code}). Check FAL_KEY."
                ) from exc
            raise ImageGenError(f"fal.ai returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ImageGenError(f"could not reach fal.ai: {exc.reason}") from exc

    def generate(self, prompt: str, image_size: str = "square_hd",
                 seed: int | None = None, poll_seconds: int = 120) -> bytes:
        if not self.available():
            raise ImageGenError(
                "No fal.ai API key. Set FAL_KEY, or switch providers with "
                "KITBASH_IMAGE_PROVIDER."
            )

        payload = {"prompt": _framed(prompt), "image_size": image_size, "num_images": 1}
        if seed is not None:
            payload["seed"] = int(seed)

        queued = self._request(f"{FAL_QUEUE}/{self.model}", payload)
        request_id = queued.get("request_id")
        if not request_id:
            raise ImageGenError(f"fal.ai did not return a request_id: {queued}")

        base = f"{FAL_QUEUE}/{self._app_id()}/requests/{request_id}"
        result = self._await_result(base, poll_seconds)

        images = result.get("images") or []
        if not images or not images[0].get("url"):
            raise ImageGenError(f"fal.ai returned no image: {result}")
        return self._download(images[0]["url"])

    def _await_result(self, base: str, poll_seconds: int) -> dict:
        import time

        deadline = time.time() + poll_seconds
        while True:
            status = self._request(f"{base}/status")
            state = status.get("status")
            if state == "COMPLETED":
                return self._request(base)
            if state in ("FAILED", "CANCELLED"):
                raise ImageGenError(f"fal.ai job {state}: {status}")
            if time.time() >= deadline:
                raise ImageGenError(
                    f"fal.ai did not finish within {poll_seconds}s (last: {state})"
                )
            time.sleep(1.0)

    @staticmethod
    def _download(url: str) -> bytes:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except urllib.error.URLError as exc:
            raise ImageGenError(f"could not download the image: {exc}") from exc


class LocalProvider:
    """An image model on the same GPU. Not implemented.

    What it needs, so nobody has to rediscover it:

    - A diffusion model in the server's venv — SDXL wants ~8 GiB, SD 1.5 ~4 GiB,
      a quantized FLUX variant lands in between.
    - **It must free its VRAM before returning.** Image generation and 3D
      generation are sequential stages, never concurrent, and on a 10 GB card
      they do not both fit. Follow the contract `pipeline.unload()` already
      uses; TRELLIS 2 warm-loads in 0.2s, so swapping is cheap.
    - Weights outside the repo, under HF_HOME.

    Deliberately unimplemented rather than half-implemented: a provider that
    silently OOMs the 3D model is worse than one that says it is not set up.
    """

    name = "local"

    def available(self) -> bool:
        return False

    def generate(self, prompt: str, **_) -> bytes:
        raise ImageGenError(
            "The local image provider is not implemented yet. Use the fal "
            "provider with a FAL_KEY, or supply a reference image directly."
        )


PROVIDERS = {"fal": FalProvider, "local": LocalProvider}


def get_provider(name: str | None = None):
    name = name or config.IMAGE_PROVIDER
    if name not in PROVIDERS:
        raise ImageGenError(
            f"unknown image provider {name!r}, expected one of {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name]()


def provider_status() -> list[dict]:
    out = []
    for name, cls in PROVIDERS.items():
        try:
            available = cls().available()
        except Exception:
            available = False
        out.append({
            "name": name,
            "available": available,
            "selected": name == config.IMAGE_PROVIDER,
        })
    return out


def image_dir() -> Path:
    return config.OUT_DIR / "images"


def image_path(image_id: str) -> Path:
    return image_dir() / f"{image_id}.png"


def store(raw: bytes, remove_background: bool = True) -> tuple[str, Path]:
    """Normalise to RGBA PNG on disk and return its id.

    Providers return JPEG as often as PNG, and image-to-3D wants an alpha matte
    — a white background baked into RGB gets reconstructed as geometry. rembg is
    already a dependency because the generation pipeline uses it.
    """
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(raw))
    if remove_background:
        try:
            import rembg

            image = rembg.remove(image.convert("RGBA"))
        except Exception:
            # Worth continuing without: a plain white background still works,
            # just less cleanly than a real matte.
            log.warning("background removal failed, keeping the original", exc_info=True)
    image = image.convert("RGBA")

    image_id = uuid.uuid4().hex[:12]
    path = image_path(image_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return image_id, path


def load_b64(image_id: str) -> str:
    """The stored image as base64, for handing straight to a generation job."""
    import base64

    path = image_path(image_id)
    if not path.exists():
        raise FileNotFoundError(f"no such image: {image_id}")
    return base64.b64encode(path.read_bytes()).decode()
