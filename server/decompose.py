"""Break a subject into parts, and build every part from its own prompt.

This is the front half of the thing the project exists for. `assemble.py` can
compose named parts into one glTF; what it needs is *parts*, and getting them
is harder than it looks.

**Cropping a photograph of the whole object does not produce parts.** That was
tested on a Beechcraft Bonanza — tail, propeller, wing and cowl were all cropped
out of one reference and every crop generated a complete aeroplane. Image-to-3D
models carry a strong object-completion prior: shown an ambiguous fragment of
something recognisable they reconstruct the nearest whole object they know.
docs/MULTI-PART.md has the full postmortem.

So a per-part reference has to *depict only that part*, which means generating
it from its own prompt. That works — but it moves the problem: eight prompts
produce eight separately-imagined objects, and a scene assembled from those
looks like a parts bin from eight different aircraft.

Two things fix that, and only the first is load-bearing:

- **A shared style suffix on every part prompt.** Measured on four Bonanza
  parts, the same prompts *without* a suffix came back as unrelated grey
  hardware; with one they came back as the same white/navy/gold airframe. This
  is what buys coherence.
- **A fixed seed.** Measured, it does *not* meaningfully tighten the palette —
  same-seed and varied-seed runs drifted about equally. It is here for
  reproducibility instead: the same prompt and seed comes back near-identical
  (4.6/255 mean pixel difference, against 27.4 for a fresh seed), so rerolling
  one bad part is a deliberate change rather than a dice throw. Near-identical,
  not identical — fal's workers are not bit-deterministic.

Numbers for both, and the prompt-shape findings behind the examples, are in
docs/DECOMPOSITION.md.

**The style suffix must not name the whole object.** Putting "same aircraft" and
"general-aviation livery" in the suffix re-armed the completion prior through
the text encoder and brought the whole aeroplane back — a propeller prompt
rendered a propeller *attached to a plane*. `validate()` warns about this,
because it is invisible until you look at the images.

The plan is data. A coding agent driving this over MCP already has the reasoning
to decide that a cart has two wheels and an axle, so there is deliberately **no
LLM call in here** — the agent authors the plan and this executes it. See
EXAMPLES for two worked ones.
"""
import logging
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import config
import imagegen
import jobs
import primitives

log = logging.getLogger("kitbash.decompose")

# A part is built one of three ways.
GENERATE = "generate"  # its own prompt -> image -> image-to-3D
SCRIPT = "script"      # a parametric primitive; no GPU, milliseconds
MIRROR = "mirror"      # another part's mesh, reflected. Costs nothing at all.
MODES = (GENERATE, SCRIPT, MIRROR)

# TRELLIS 2, geometry only. It beats Hunyuan3D on hard-surface props and runs at
# ~3.6 GiB, but its texture path returns rainbow noise on every subject tried
# here — colour comes from the livery via materials.py instead. See
# docs/QUALITY-COMPARISON.md.
DEFAULT_GENERATOR = "trellis2"
DEFAULT_TEXTURED = False

# Below Roblox's 20 000-per-MeshPart cap on purpose: a multi-part build spends
# that budget once per part, and 12 000 leaves headroom for the parts that
# deserve more without any single one being the thing that fails an import.
DEFAULT_TARGET_FACES = 12000

# Append to a part prompt whose subject is thin — a wing, a fin, a strut, a
# blade. Measured: the wing came back near edge-on and small in frame, and
# reconstructed as two crossed slabs, because a thin panel seen edge-on is close
# to information-free. With this clause the same prompt produced a wing across
# the frame and a single solid wing mesh. imagegen.FRAMING already asks for a
# three-quarter view; for thin subjects the model reads that as edge-on anyway,
# so it has to be said again and more specifically.
THIN_PART_VIEW = (
    "seen from above and to one side at a steep angle so its thickness and "
    "depth are clearly visible, large in frame"
)

# Words that carry no subject identity, so they cannot be evidence of the style
# suffix leaking the whole object back into a part prompt.
_STOPWORDS = frozenset(
    "a an the of and or with for in on at to from single small large "
    "light heavy old new one two three four".split()
)


class DecomposeError(Exception):
    pass


@dataclass
class Part:
    """One addressable piece of the finished object.

    `placement` is passed through untouched. It is assemble.py's vocabulary —
    `anchor`, `mirror`, `mirror_of`, `position`, `rotation`, `scale` — and
    placement is deliberately not decided here; the caller knows this is a
    biplane and that the second wing goes above the first.
    """

    name: str
    mode: str = GENERATE
    # GENERATE: what this part looks like, on its own. Describe the *geometry*
    # and leave the parent object unnamed — "a long tapered blade-shaped panel
    # with a hinged flap" survives where "an aircraft wing" renders an aircraft.
    prompt: str | None = None
    # SCRIPT: a kind from primitives.py, plus its parameters.
    kind: str | None = None
    params: dict = field(default_factory=dict)
    # MIRROR: nothing to build; placement.mirror_of names the source part.
    material: str | None = None
    color: str | None = None
    target_faces: int | None = None
    generator: str | None = None
    textured: bool | None = None
    # Override the plan's seed for this part only. The reroll knob: change one
    # part's dice without disturbing the seven that came out right.
    seed: int | None = None
    placement: dict = field(default_factory=dict)
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Part":
        if not isinstance(data, dict):
            raise DecomposeError(f"each part must be an object, got {type(data).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            # Silently ignoring `prompts` would build a part with no prompt and
            # blame the caller for the empty result.
            raise DecomposeError(
                f"part {data.get('name', '?')!r} has unknown field(s) {unknown}; "
                f"expected any of {sorted(known)}"
            )
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    """A subject, a shared style, and the parts it breaks into."""

    subject: str
    # The shared style suffix: materials, palette, finish, lighting. Appended to
    # every generated part's prompt. Name materials and light, never the object.
    style: str
    parts: list[Part] = field(default_factory=list)
    seed: int = 20260806
    name: str | None = None
    # Plan-wide defaults; a part overrides either.
    generator: str = DEFAULT_GENERATOR
    target_faces: int = DEFAULT_TARGET_FACES
    textured: bool = DEFAULT_TEXTURED
    image_size: str = "square_hd"
    note: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        if isinstance(data, Plan):
            return data
        if not isinstance(data, dict):
            raise DecomposeError(f"a plan must be an object, got {type(data).__name__}")
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise DecomposeError(
                f"unknown plan field(s) {unknown}; expected any of {sorted(known)}"
            )
        payload = dict(data)
        payload["parts"] = [
            p if isinstance(p, Part) else Part.from_dict(p)
            for p in payload.get("parts") or []
        ]
        for required in ("subject", "style"):
            if not str(payload.get(required) or "").strip():
                raise DecomposeError(f"a plan needs a non-empty {required!r}")
        return cls(**payload)

    def to_dict(self) -> dict:
        return {**asdict(self), "parts": [p.to_dict() for p in self.parts]}

    def part(self, name: str) -> Part | None:
        return next((p for p in self.parts if p.name == name), None)


# --- prompt composition -----------------------------------------------------


def part_prompt(plan: Plan, part: Part) -> str:
    """The exact text sent to the image provider for one part.

    Just the part description and the shared suffix. The provider adds its own
    framing — `imagegen.FRAMING` supplies "single isolated ..., plain flat white
    background, no ground plane" — so this must not repeat it.
    """
    if not part.prompt:
        raise DecomposeError(f"part {part.name!r} is mode {GENERATE!r} but has no prompt")
    body = part.prompt.strip().rstrip(".,")
    style = (plan.style or "").strip().rstrip(".,")
    return f"{body}, {style}" if style else body


def part_seed(plan: Plan, part: Part) -> int:
    return plan.seed if part.seed is None else int(part.seed)


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{4,}", (text or "").lower())
        if w not in _STOPWORDS
    }


def style_leaks(plan: Plan) -> list[str]:
    """Words from the subject that reappear in the shared style suffix.

    The measured failure: a suffix reading "all parts of the same aircraft,
    general-aviation livery" put an entire aeroplane behind a propeller that was
    asked for on its own. The subject noun in the suffix is the trigger, and
    nothing about the resulting image looks like a prompt bug, so this is worth
    saying out loud before eight minutes of GPU time.
    """
    return sorted(_content_words(plan.subject) & _content_words(plan.style))


# --- validation -------------------------------------------------------------


def validate(plan: Plan) -> list[str]:
    """Reject what cannot work; warn about what usually does not.

    Everything here is cheap and everything after it is not — one aircraft plan
    is eight image generations and eight GPU jobs. A misspelled primitive
    parameter should cost a millisecond, not eight minutes.
    """
    if not plan.parts:
        raise DecomposeError("a plan needs at least one part")

    names = [p.name for p in plan.parts]
    for i, name in enumerate(names):
        if not str(name or "").strip():
            raise DecomposeError(f"part {i} has no name; names become glTF node names")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        # Not cosmetic: assemble.py resolves anchors and mirrors by name, and a
        # duplicate makes those references ambiguous.
        raise DecomposeError(f"duplicate part name(s) {duplicates}; names must be unique")

    known = set(names)
    warnings: list[str] = []

    for part in plan.parts:
        if part.mode not in MODES:
            raise DecomposeError(
                f"part {part.name!r} has mode {part.mode!r}; expected one of {list(MODES)}"
            )
        generator = part.generator or plan.generator
        if generator not in jobs.GENERATORS:
            raise DecomposeError(
                f"part {part.name!r} names generator {generator!r}; expected one of "
                f"{sorted(jobs.GENERATORS)}"
            )

        if part.mode == GENERATE:
            if not str(part.prompt or "").strip():
                raise DecomposeError(f"part {part.name!r} is {GENERATE!r} but has no prompt")
            if part.kind:
                raise DecomposeError(
                    f"part {part.name!r} is {GENERATE!r} but also names kind "
                    f"{part.kind!r}; pick one path"
                )
        elif part.mode == SCRIPT:
            if not part.kind:
                raise DecomposeError(
                    f"part {part.name!r} is {SCRIPT!r} but names no kind; expected "
                    f"one of {primitives.kinds()}"
                )
            try:
                # Resolve now rather than at build time: this is the check that
                # catches `widht` before anything else in the plan has run.
                primitives.resolve(part.kind, part.params)
            except ValueError as exc:
                raise DecomposeError(f"part {part.name!r}: {exc}") from exc
        else:  # MIRROR
            source = part.placement.get("mirror_of")
            if not source:
                raise DecomposeError(
                    f"part {part.name!r} is {MIRROR!r} but placement has no "
                    f"`mirror_of` naming the part it copies"
                )
            if source not in known:
                raise DecomposeError(
                    f"part {part.name!r} mirrors unknown part {source!r}; "
                    f"known parts are {sorted(known)}"
                )
            mirrored = plan.part(source)
            if mirrored is not None and mirrored.mode == MIRROR:
                raise DecomposeError(
                    f"part {part.name!r} mirrors {source!r}, which is itself a "
                    f"mirror; mirror the original instead"
                )

        anchor = part.placement.get("anchor")
        if isinstance(anchor, dict):
            to = str(anchor.get("to") or "").strip()
            if to and to.lower() != "ground" and to not in known:
                raise DecomposeError(
                    f"part {part.name!r} anchors to unknown part {to!r}; "
                    f"known parts are {sorted(known)}"
                )

    leaked = style_leaks(plan)
    if leaked:
        warnings.append(
            f"the style suffix repeats subject word(s) {leaked}. Naming the whole "
            f"object in the shared suffix re-arms the object-completion prior and "
            f"parts come back attached to the thing they belong to — describe "
            f"materials, palette and lighting instead."
        )
    if not any(p.mode == GENERATE for p in plan.parts):
        warnings.append("no generated parts; this plan needs no image provider and no GPU")

    return warnings


# --- execution --------------------------------------------------------------


class ServerBackend:
    """The default: fal for images, the job queue for meshes.

    Injectable because the two sides have wildly different costs — images are a
    4-second HTTP round trip, meshes are ~33 seconds of GPU each — and because a
    test must reach neither.
    """

    def __init__(self, provider=None, out_dir: Path | None = None):
        self._provider = provider
        self.out_dir = out_dir or config.OUT_DIR

    @property
    def provider(self):
        # Resolved lazily so a plan made entirely of scripted parts never
        # touches the image provider, and a missing FAL_KEY is not an error
        # until something actually needs a picture.
        if self._provider is None:
            self._provider = imagegen.get_provider()
        return self._provider

    def image(self, prompt: str, seed: int, image_size: str = "square_hd") -> str:
        raw = self.provider.generate(prompt, image_size=image_size, seed=seed)
        image_id, _ = imagegen.store(raw)
        return image_id

    def submit(self, params: dict, image_id: str) -> dict:
        return jobs.submit("image_to_3d", params, imagegen.load_b64(image_id))

    def script(self, part: "Part") -> dict:
        """Build a primitive and file it as an already-finished job.

        A scripted part never enters the queue — the queue exists to serialise a
        GPU this path does not touch — but it has to end up in the registry with
        the same record shape, because /assemble and /export take a job id and
        must not be able to tell the two halves apart. This mirrors
        `app.create_primitive`; it cannot call it, because app.py imports this
        module to serve the endpoint.
        """
        job_id = uuid.uuid4().hex[:12]
        result = primitives.store(
            part.kind, part.params, self.out_dir / job_id,
            part_name=part.name, material=part.material, color=part.color,
        )
        now = time.time()
        job = {
            "id": job_id,
            "type": "primitive",
            "status": jobs.DONE,
            "created_at": now,
            "started_at": now,
            "finished_at": time.time(),
            "params": {"kind": part.kind, "part_name": part.name},
            "result": result,
            "error": None,
        }
        with jobs._jobs_lock:
            jobs._jobs[job_id] = job
        jobs._persist(job)
        return job


def _job_params(plan: Plan, part: Part) -> dict:
    params = {
        "generator": part.generator or plan.generator,
        "target_faces": part.target_faces or plan.target_faces,
        "part_name": part.name,
        "seed": part_seed(plan, part),
    }
    textured = plan.textured if part.textured is None else part.textured
    if textured is not None:
        params["textured"] = bool(textured)
    return params


def run(plan, backend=None, progress=None) -> dict:
    """Execute a plan: images now, meshes queued, ids back.

    Images are generated inline — four seconds each, and a part with no image
    has nothing to submit — while the meshes go onto the single-worker queue and
    are collected by the caller. That split is what keeps this a request rather
    than a ten-minute connection: measured on the bonanza example, ten parts
    returned in 22 s with seven job ids that finished over the following eight
    minutes.

    One part failing does not abandon the rest. Seven good parts and a named
    failure is a build you can finish by rerolling one prompt; an exception
    halfway through is seven wasted generations.
    """
    plan = Plan.from_dict(plan)
    warnings = validate(plan)
    for warning in warnings:
        log.warning("%s: %s", plan.name or plan.subject, warning)

    backend = backend or ServerBackend()
    started = time.time()
    total = len(plan.parts)
    records: dict[str, dict] = {}
    results: list[dict] = []

    def emit(**event):
        log.info("[%s] %s", plan.name or plan.subject, event)
        if progress is not None:
            progress(dict(event))

    for index, part in enumerate(plan.parts, start=1):
        record = {
            "name": part.name,
            "mode": part.mode,
            "job_id": None,
            "image_id": None,
            "prompt": None,
            "status": "pending",
            "error": None,
            "placement": dict(part.placement),
            # Carried through to assembly rather than re-guessed from the node
            # name: the plan already stated what this part is made of, and
            # materials.KEYWORDS reading "barrel" as a gun is the failure that
            # costs.
            "material": part.material,
            "color": part.color,
            "note": part.note,
        }
        results.append(record)
        records[part.name] = record

        try:
            if part.mode == GENERATE:
                prompt = part_prompt(plan, part)
                record["prompt"] = prompt
                record["seed"] = part_seed(plan, part)
                emit(event="image", part=part.name, index=index, total=total)
                record["image_id"] = backend.image(
                    prompt, part_seed(plan, part), plan.image_size
                )
                job = backend.submit(_job_params(plan, part), record["image_id"])
                record["job_id"] = job["id"]
                record["status"] = job["status"]
                emit(event="queued", part=part.name, index=index, total=total,
                     job_id=job["id"], image_id=record["image_id"])

            elif part.mode == SCRIPT:
                job = backend.script(part)
                record["job_id"] = job["id"]
                record["status"] = job["status"]
                record["faces"] = job["result"]["faces"]
                emit(event="scripted", part=part.name, index=index, total=total,
                     job_id=job["id"], faces=job["result"]["faces"])

            else:  # MIRROR — no build at all, it reuses the source part's mesh.
                source = records.get(part.placement["mirror_of"])
                if source is None:
                    raise DecomposeError(
                        f"{part.name!r} mirrors {part.placement['mirror_of']!r}, "
                        f"which is listed after it; list the original first"
                    )
                if not source["job_id"]:
                    raise DecomposeError(
                        f"{part.name!r} mirrors {source['name']!r}, which failed"
                    )
                record["job_id"] = source["job_id"]
                record["status"] = "mirrored"
                emit(event="mirrored", part=part.name, index=index, total=total,
                     job_id=source["job_id"], of=source["name"])

        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            log.exception("part %s failed", part.name)
            emit(event="error", part=part.name, index=index, total=total,
                 error=record["error"])

    built = [r for r in results if r["job_id"]]
    return {
        "plan": plan.to_dict(),
        "subject": plan.subject,
        "seed": plan.seed,
        "warnings": warnings,
        "parts": results,
        "job_ids": [r["job_id"] for r in built],
        "failed": [r["name"] for r in results if r["status"] == "error"],
        "elapsed_seconds": round(time.time() - started, 2),
        # Handed back ready to post: the plan already carries placement intent,
        # so the caller should not have to transcribe it into a second document.
        # assemble.py owns what these keys mean.
        "assemble_request": [
            {
                "job_id": r["job_id"],
                "name": r["name"],
                **({"material": r["material"]} if r["material"] else {}),
                **({"color": r["color"]} if r["color"] else {}),
                **r["placement"],
            }
            for r in built
        ],
    }


def status(result: dict) -> dict:
    """Where a running build has got to, from the job registry."""
    parts = []
    for record in result["parts"]:
        job = jobs.get(record["job_id"]) if record["job_id"] else None
        parts.append({
            "name": record["name"],
            "mode": record["mode"],
            "job_id": record["job_id"],
            "status": (record["error"] and "error")
                      or (job["status"] if job else record["status"]),
            "error": record["error"] or (job or {}).get("error"),
        })
    done = sum(1 for p in parts if p["status"] in (jobs.DONE, "mirrored"))
    return {
        "parts": parts,
        "done": done,
        "total": len(parts),
        "finished": all(
            p["status"] in (jobs.DONE, "mirrored", "error") for p in parts
        ),
    }


def wait(result: dict, timeout: float = 1800.0, poll: float = 5.0,
         progress=None) -> dict:
    """Block until every queued part has finished. For scripts and tests.

    Not what an HTTP endpoint should do — see `run` — but a command-line build
    wants one call.
    """
    deadline = time.time() + timeout
    while True:
        state = status(result)
        if progress is not None:
            progress(state)
        if state["finished"]:
            return state
        if time.time() >= deadline:
            state["timed_out"] = True
            return state
        time.sleep(poll)


# --- worked examples --------------------------------------------------------
#
# Data, not code. An agent over MCP authors plans in exactly this shape, so
# these are simultaneously the format's documentation and its test fixtures.
#
# The two of them are the two halves of the routing rule in docs/PROCEDURAL.md:
# the aircraft is sculptural and generates; the cart is dimensioned hardware and
# scripts, with the only generated parts being the soft irregular cargo.

BONANZA: dict = {
    "name": "bonanza",
    "subject": "a Beechcraft Bonanza G36 light aircraft",
    # No aircraft noun anywhere in here. Materials, palette, finish, light —
    # that is the whole job of the suffix, and adding "the same aeroplane" to it
    # measurably brought whole aeroplanes back into single-part images.
    "style": (
        "glossy white painted aluminium, navy blue and gold accent stripe, "
        "polished chrome, matte black rubber, soft neutral studio light from "
        "the upper left, photorealistic"
    ),
    "seed": 20260806,
    "parts": [
        {
            "name": "fuselage",
            # Geometry first, subject never. "a bare aeroplane fuselage" renders
            # a complete aeroplane every time; a tapered shell with portholes
            # renders a tapered shell with portholes.
            "prompt": (
                "a hollow elongated shell with an oval cross section, tapered at "
                "both ends, six oval portholes in a row along the side, a rounded "
                "glass canopy near one end, no wings, no fins, no wheels"
            ),
            "target_faces": 16000,
            "material": "paint",
            "placement": {"position": [0, 0, 0]},
            "note": "the hero part; everything else anchors to it",
        },
        {
            "name": "left_wing",
            # The trailing view clause is not decoration. Without it the model
            # drew the wing near edge-on and small, and TRELLIS 2 reconstructed
            # two crossed slabs — a thin panel seen edge-on carries almost no
            # depth information. Naming the viewpoint fixed it. Any thin part
            # wants this; see docs/DECOMPOSITION.md.
            "prompt": (
                "a long tapered blade-shaped panel, thick rounded leading edge "
                "and thin sharp trailing edge, a hinged flap along the back edge, "
                "a small orange light at the narrow tip, cut off flat at the wide "
                "end, one panel only, " + THIN_PART_VIEW
            ),
            "material": "paint",
            "placement": {
                "anchor": {"to": "fuselage", "align": {"x": "min", "y": 0.25, "z": 0.45},
                           "my": {"x": "max"}},
            },
        },
        {
            "name": "right_wing",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_wing", "mirror": "x"},
            "note": "free: the same mesh reflected, so both wings are the same part",
        },
        {
            "name": "tail_fin",
            "prompt": (
                "a single vertical tail fin with a rudder, cut off flat at the "
                "base, nothing attached"
            ),
            "material": "paint",
            "placement": {
                "anchor": {"to": "fuselage", "align": {"z": "min", "y": "top"},
                           "my": {"y": "min"}},
            },
        },
        {
            "name": "left_tailplane",
            "prompt": (
                "a flat tapered horizontal fin panel with a hinged trailing-edge "
                "flap, cut off flat at one end, nothing attached to it"
            ),
            "target_faces": 8000,
            "material": "paint",
            "placement": {
                "anchor": {"to": "fuselage", "align": {"x": "min", "y": 0.4, "z": 0.05},
                           "my": {"x": "max"}},
            },
        },
        {
            "name": "right_tailplane",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_tailplane", "mirror": "x"},
        },
        {
            "name": "engine_cowl",
            "prompt": (
                "a hollow engine cowling shell, open at both ends, with a round "
                "air intake"
            ),
            "material": "paint",
            "placement": {
                "anchor": {"to": "fuselage", "align": {"z": "max"}, "my": {"z": "min"}},
            },
        },
        {
            "name": "propeller",
            "prompt": "a three-blade propeller with a polished spinner hub",
            "material": "metal",
            "placement": {
                "anchor": {"to": "engine_cowl", "align": {"z": "max"}, "my": {"z": "min"}},
            },
        },
        # The landing gear is scripted, and that is the interesting decision in
        # this plan. Generated, it was a strut *and* a wheel at very different
        # scales in one 1024² frame: the strut came back as a featureless
        # spindle and the wheel did not survive at all, with or without the
        # view clause. Two primitives are exact, free, and correct — which is
        # the routing rule in docs/PROCEDURAL.md landing on an aircraft.
        {
            "name": "left_gear_strut",
            "mode": SCRIPT,
            "kind": "cylinder",
            "params": {"radius": 0.055, "height": 1.0, "chamfer": 0.02},
            "material": "metal",
            "placement": {
                "anchor": {"to": "left_wing", "align": {"y": "under", "x": 0.7, "z": 0.5}},
            },
        },
        {
            "name": "left_gear_wheel",
            "mode": SCRIPT,
            "kind": "wheel",
            "params": {"radius": 0.26, "width": 0.11, "hub_radius": 0.08,
                       "spoke_count": 6},
            "material": "rubber",
            # _revolve sweeps around +Y, so a wheel is built lying flat and has
            # to be stood up to roll along Z.
            "placement": {
                "rotation": [0, 0, 90],
                "anchor": {"to": "left_gear_strut", "align": {"y": "under"}},
            },
        },
        {
            "name": "right_gear_strut",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_gear_strut", "mirror": "x"},
        },
        {
            "name": "right_gear_wheel",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_gear_wheel", "mirror": "x"},
        },
    ],
}

WOODEN_CART: dict = {
    "name": "wooden_cart",
    "subject": "a rustic two-wheeled wooden hand cart",
    "style": (
        "weathered oak with visible grain, black wrought iron fittings, worn "
        "brown leather straps, soft overcast daylight, photorealistic, matte finish"
    ),
    "seed": 4711,
    "parts": [
        # Everything with a measurement is scripted. A generated crate cost
        # 83-151s and 20 000 faces and still had rounded corners; the scripted
        # one is 4.6ms, 1 380 faces, and exactly the size asked for.
        {
            "name": "cart_bed",
            "mode": SCRIPT,
            "kind": "crate",
            "params": {"width": 3.2, "height": 0.7, "depth": 1.8,
                       "style": "planks", "plank_count": 5},
            "placement": {"anchor": {"to": "ground"}, "position": [0, 0, 0]},
        },
        {
            "name": "axle",
            "mode": SCRIPT,
            "kind": "cylinder",
            "params": {"radius": 0.09, "height": 2.1},
            "material": "dark_metal",
            "placement": {
                "rotation": [0, 0, 90],
                "anchor": {"to": "cart_bed", "align": {"y": "under", "z": 0.5}},
            },
        },
        {
            "name": "left_wheel",
            "mode": SCRIPT,
            "kind": "wheel",
            "params": {"radius": 0.85, "width": 0.22, "spoke_count": 8},
            "material": "wood",
            "placement": {
                "rotation": [0, 0, 90],
                "anchor": {"to": "axle", "align": {"x": "min"}, "my": {"x": "max"}},
            },
        },
        {
            "name": "right_wheel",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_wheel", "mirror": "x"},
        },
        {
            "name": "left_shaft",
            "mode": SCRIPT,
            "kind": "plank",
            "params": {"length": 2.4, "width": 0.14, "thickness": 0.1},
            "placement": {
                "anchor": {"to": "cart_bed", "align": {"x": 0.15, "y": 0.6, "z": "max"},
                           "my": {"z": "min"}},
            },
        },
        {
            "name": "right_shaft",
            "mode": MIRROR,
            "placement": {"mirror_of": "left_shaft", "mirror": "x"},
        },
        # ...and the two things nobody can write a formula for are generated.
        {
            "name": "canvas_bundle",
            "prompt": (
                "a rolled bundle of coarse canvas cloth tied with three loops of "
                "rope, soft folds and creases"
            ),
            "material": "fabric",
            "target_faces": 8000,
            "placement": {"anchor": {"to": "cart_bed", "align": {"y": "on", "x": 0.3}}},
        },
        {
            "name": "lantern",
            "prompt": (
                "a small hand lantern with a punched metal top, four glass panes "
                "and a looped carrying handle"
            ),
            "material": "metal",
            "target_faces": 6000,
            "placement": {"anchor": {"to": "cart_bed", "align": {"y": "on", "x": 0.8}}},
        },
    ],
}

EXAMPLES: dict[str, dict] = {"bonanza": BONANZA, "wooden_cart": WOODEN_CART}


def example(name: str) -> Plan:
    if name not in EXAMPLES:
        raise DecomposeError(
            f"unknown example {name!r}; expected one of {sorted(EXAMPLES)}"
        )
    return Plan.from_dict(EXAMPLES[name])
