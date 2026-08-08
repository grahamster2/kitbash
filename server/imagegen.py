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

The second half of this module is candidate batches: several references for one
prompt, generated concurrently and returned unchosen, because image-to-3D is
decided by its reference and picking it is the one judgement in this pipeline a
human is better at than the machine. See docs/REFERENCE-SELECTION.md.
"""
import json
import logging
import random
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import config

log = logging.getLogger("kitbash.imagegen")

FAL_QUEUE = "https://queue.fal.run"

# Image-to-3D is far more sensitive to framing than to prose. A single centred
# object on a plain background reconstructs well; a scene with ground plane,
# horizon and props produces a mesh with all of that fused into it. Callers
# describe the object and this supplies the framing.
#
# `{view}` is a slot rather than a literal so a candidate batch can move the
# camera without rewriting the negatives, which are the part that actually
# earns its place.
FRAMING = (
    "single isolated {subject}, centered, {view}, full object "
    "visible, plain flat white background, even studio lighting, no shadows on "
    "the background, no ground plane, no scenery, no text, no watermark"
)

DEFAULT_VIEW = "three-quarter view"


class ImageGenError(Exception):
    pass


def _framed(prompt: str, view: str | None = None, style: str | None = None) -> str:
    """Wrap a subject in the framing, optionally re-aiming the camera.

    `style` goes *after* the framing rather than beside the subject: dropped in
    next to `{subject}` it reads as part of the noun phrase, and the negatives
    ("no ground plane", "no scenery") stop applying to whatever it added.
    """
    framed = FRAMING.format(
        subject=prompt.strip().rstrip("."), view=view or DEFAULT_VIEW
    )
    return f"{framed}, {style.strip().rstrip('.')}" if style else framed


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
                 seed: int | None = None, poll_seconds: int = 120,
                 view: str | None = None, style: str | None = None) -> bytes:
        if not self.available():
            raise ImageGenError(
                "No fal.ai API key. Set FAL_KEY, or switch providers with "
                "KITBASH_IMAGE_PROVIDER."
            )

        payload = {
            "prompt": _framed(prompt, view=view, style=style),
            "image_size": image_size,
            "num_images": 1,
        }
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


# --------------------------------------------------------------------------
# candidate batches — several references for one prompt, so a human can choose
# --------------------------------------------------------------------------
# Image-to-3D is decided by its reference. One prompt giving one image means the
# reference is whatever the model happened to imagine, and the one place a human
# could exercise judgement — which of these is the object I meant? — never
# happens. See docs/REFERENCE-SELECTION.md.

# Mechanical variation, for when the caller supplies no `variants` of its own.
#
# These are *interpretations*, not seeds: a weathered crate and a sleek crate
# are different objects, and four seeds of one prompt are four photographs of
# the same one. They name only surface, construction and viewpoint — never a
# parent object — because naming the whole object in a suffix re-arms the
# completion prior that returns a propeller attached to an aeroplane
# (docs/DECOMPOSITION.md, "The suffix must not name the whole object").
#
# Every entry keeps a three-quarter-ish camera. A profile or a plan view varies
# the picture nicely and ruins it as a reconstruction input, which is the one
# thing these images are for.
VARIATIONS: list[dict] = [
    # First is deliberately the plain framing: one candidate is always exactly
    # what POST /images would have returned, so choosing costs nothing.
    {"label": None, "view": None, "style": None},
    {
        "label": "weathered",
        "view": None,
        "style": "heavily weathered and worn, chipped and scratched surfaces, "
                 "aged patina, grime in the recesses, photorealistic",
    },
    {
        "label": "ornate",
        "view": "three-quarter view from a slightly low angle",
        "style": "ornate and elaborately decorated, intricate carved relief, "
                 "inlaid trim, rich contrasting materials",
    },
    {
        "label": "stylised",
        "view": "three-quarter view from a slightly high angle",
        "style": "stylised low-poly game asset, simplified flat planes, bold "
                 "chunky silhouette, flat matte colours, clean edges",
    },
    {
        "label": "sleek",
        "view": None,
        "style": "sleek minimal modern design, smooth uninterrupted surfaces, "
                 "restrained detail, brushed metal and matte finish",
    },
    {
        "label": "rugged",
        "view": "three-quarter view from a slightly low angle",
        "style": "rugged heavy-duty construction, thick reinforced edges, "
                 "exposed fasteners and bracing, utilitarian and unpainted",
    },
]

# Four fal calls is four times the bill of one. A cap keeps a typo'd `count`
# from turning into a hundred of them.
MAX_CANDIDATES = 8


def batch_dir() -> Path:
    return image_dir() / "batches"


def batch_path(batch_id: str) -> Path:
    return batch_dir() / f"{batch_id}.json"


def load_batch(batch_id: str) -> dict:
    """A previously generated batch, for polling and re-display.

    On disk rather than in memory on purpose: the desktop app and the MCP server
    are separate processes from whatever generated the batch, and a batch that
    only one of them can see is a batch the user cannot choose from.
    """
    path = batch_path(batch_id)
    if not path.exists():
        raise FileNotFoundError(f"no such batch: {batch_id}")
    return json.loads(path.read_text())


def _candidate_specs(prompt: str, count: int, variants: list[str] | None,
                     seed: int | None) -> list[dict]:
    """What each of the N calls should ask for.

    Distinct seeds throughout. They buy almost nothing on their own — the same
    prompt at two seeds is the same object twice (docs/DECOMPOSITION.md) — but
    they cost nothing either, and every seed is reported back so a caller can
    re-roll exactly one candidate.
    """
    specs = []
    for i in range(count):
        if variants:
            variant = variants[i % len(variants)]
            spec = {"prompt": variant, "variant": variant, "view": None,
                    "style": None}
        else:
            v = VARIATIONS[i % len(VARIATIONS)]
            spec = {"prompt": prompt, "variant": v["label"], "view": v["view"],
                    "style": v["style"]}
        # A caller's seed makes the whole batch reproducible; without one the
        # candidates still have to differ from each other, hence per-candidate
        # randomness rather than a single None.
        spec["seed"] = int(seed) + i if seed is not None else random.randrange(2**31)
        specs.append(spec)
    return specs


def generate_candidates(prompt: str, count: int = 4,
                        variants: list[str] | None = None,
                        image_size: str = "square_hd", seed: int | None = None,
                        remove_background: bool = True,
                        provider=None) -> dict:
    """N reference images for one prompt, generated concurrently.

    Concurrency is the whole reason this is not a loop in the caller: a fal
    round trip is ~4 s of *waiting on a socket*, so four sequential calls are
    ~16 s and four threads are ~4 s. `imagegen` is stdlib urllib and blocks, so
    threads release the GIL for the entire call.

    Storage is deliberately serial. `store()` runs rembg, which lazily builds a
    process-wide inference session on first use; several threads racing to
    create it is a hazard for something that only costs a few hundred
    milliseconds per image.

    One candidate failing does not fail the batch — three usable references are
    worth having, and the caller is told which slot was lost and why.
    """
    if count < 1:
        raise ImageGenError("count must be at least 1")
    if count > MAX_CANDIDATES:
        raise ImageGenError(
            f"count must be at most {MAX_CANDIDATES}; each candidate is a "
            f"separate billed image call"
        )
    if variants is not None and not variants:
        raise ImageGenError("variants was given but empty; omit it instead")

    provider = provider or get_provider()
    specs = _candidate_specs(prompt, count, variants, seed)
    started = time.time()

    def _fetch(spec: dict):
        return provider.generate(
            spec["prompt"], image_size=image_size, seed=spec["seed"],
            view=spec["view"], style=spec["style"],
        )

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="imagegen") as pool:
        raws = list(pool.map(_lift_errors(_fetch), specs))

    candidates, failed = [], []
    for index, (spec, (raw, error)) in enumerate(zip(specs, raws)):
        if error is not None:
            failed.append({"index": index, "variant": spec["variant"],
                           "prompt": spec["prompt"], "seed": spec["seed"],
                           "error": str(error)})
            continue
        image_id, path = store(raw, remove_background)
        candidates.append({
            "image_id": image_id,
            "prompt": spec["prompt"],
            "variant": spec["variant"],
            "seed": spec["seed"],
            "bytes": path.stat().st_size,
            "path": str(path),
        })

    if not candidates:
        raise ImageGenError(
            f"every one of the {count} candidates failed: "
            + "; ".join(f["error"] for f in failed)
        )

    elapsed = round(time.time() - started, 2)
    batch_id = uuid.uuid4().hex[:12]
    batch = {
        "batch_id": batch_id,
        "prompt": prompt,
        "candidates": candidates,
        # Cost visibility. Four images is four billed calls, and nobody should
        # find that out from an invoice.
        "count": len(candidates),
        "requested": count,
        "elapsed_seconds": elapsed,
        "provider": provider.name,
        "mode": "variants" if variants else "mechanical",
        "image_size": image_size,
        "failed": failed,
        "created_at": time.time(),
    }
    path = batch_path(batch_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(batch, indent=2))
    log.info("batch %s: %d/%d candidates in %.2fs (%s)",
             batch_id, len(candidates), count, elapsed, batch["mode"])
    return batch


def _lift_errors(fn):
    """Turn a raising call into `(value, error)`.

    ThreadPoolExecutor.map re-raises the first exception and abandons the rest,
    which would throw away images that were already paid for.
    """

    def wrapped(spec):
        try:
            return fn(spec), None
        except Exception as exc:  # a provider error, or anything it wrapped
            log.warning("candidate failed: %s", exc)
            return None, exc

    return wrapped
