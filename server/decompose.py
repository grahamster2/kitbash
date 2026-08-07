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

**Every part also has to say how big it really is.** Measured on the six
generated Bonanza parts, the longest side of every returned mesh is 0.992-1.000:
the generator normalises its output to a unit box, so a landing-gear strut comes
back exactly as large as a fuselage and no amount of anchoring recovers the
difference. Nothing downstream can know — the mesh does not, the reference image
does not — so `Part.size_m` states it and `part_scale()` turns it into the
`scale` assemble.py already accepts. Without it every new object repeats the
Bonanza's hand-tuned scale script.

The plan is data. A coding agent driving this over MCP already has the reasoning
to decide that a cart has two wheels and an axle, so there is deliberately **no
LLM call in here** — the agent authors the plan and this executes it. See
EXAMPLES for two worked ones.
"""
import logging
import math
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

# Sanity bounds on a declared real-world size, in metres. Wide on purpose — a
# rivet and an airliner are both legitimate — but they catch the mistake that
# actually happens, which is an LLM writing millimetres or centimetres into a
# field named `_m` and a propeller arriving 2 000 units across.
MIN_SIZE_M = 0.001
MAX_SIZE_M = 1000.0

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
    #
    # How big this part is in the real world, in metres. Either the longest
    # dimension as one number — `2.0` for a propeller, which is all a simple
    # part needs — or `[x, y, z]` extents in the part's own frame, which says
    # the same thing and additionally tells orient.py which way round it goes.
    # This is the only place the information can come from; see the module
    # docstring. A mirror inherits its source's, and must not state its own.
    size_m: float | list[float] | None = None
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
    # The part that is 1.0 unit across in the assembled scene — "the fuselage is
    # the unit". Every `size_m` is then divided by that part's, so the numbers a
    # caller reads back are ratios it can check by eye (a wing is half a
    # fuselage) instead of absolute metres it has to trust. Left unset, one unit
    # is one metre, which is what a plan built from primitives already assumes.
    scale_reference: str | None = None
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


# --- real-world size --------------------------------------------------------
#
# The generator hands back a unit box. Measured on the six generated Bonanza
# parts, longest side: fuselage 0.9923, wing 0.9989, fin 0.9997, tailplane
# 0.9936, cowl 0.9989, propeller 0.9921. A 0.9 m strut and an 8.4 m fuselage
# arrive the same size, and an anchor cannot fix it because an anchor measures
# whatever box it is given. So the plan states the real size and the scale is
# arithmetic from there.


def part_extents(part: Part) -> list[float] | None:
    """`size_m` as [x, y, z], or None if it was given as a single number.

    This is exactly what orient.py wants for `orient.extents` — real metres in
    the part's own frame, of which it uses only the ratios.
    """
    values = _size_values(part)
    return values if len(values) == 3 else None


def part_length(part: Part) -> float | None:
    """The part's longest real dimension, in metres. None if it declared none."""
    values = _size_values(part)
    return max(values) if values else None


def _size_values(part: Part) -> list[float]:
    """Validate `size_m` into a list of one or three positive metre lengths.

    Cheap, so `validate()` can run it on every part before anything is spent.
    """
    value = part.size_m
    if value is None:
        return []
    # bool is an int in Python, and `"size_m": true` is a typo, not one metre.
    if isinstance(value, bool):
        raise DecomposeError(
            f"part {part.name!r} has size_m {value!r}; expected its longest "
            f"dimension in metres, or [x, y, z] extents in metres"
        )
    if isinstance(value, (int, float)):
        values = [float(value)]
    elif isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise DecomposeError(
                f"part {part.name!r} has size_m with {len(value)} number(s); "
                f"expected one — its longest dimension in metres — or three, "
                f"[x, y, z] extents in metres"
            )
        try:
            values = [float(v) for v in value]
        except (TypeError, ValueError) as exc:
            raise DecomposeError(
                f"part {part.name!r} has size_m {list(value)!r}; every extent "
                f"must be a number of metres"
            ) from exc
    else:
        raise DecomposeError(
            f"part {part.name!r} has size_m {value!r}; expected a number of "
            f"metres or [x, y, z] extents in metres"
        )

    for v in values:
        if not math.isfinite(v) or v <= 0:
            raise DecomposeError(
                f"part {part.name!r} has size_m {value!r}; every extent must be "
                f"a positive, finite number of metres"
            )
        if not MIN_SIZE_M <= v <= MAX_SIZE_M:
            raise DecomposeError(
                f"part {part.name!r} has size_m {value!r}, which is {v}m — "
                f"outside {MIN_SIZE_M}m to {MAX_SIZE_M}m. size_m is **metres**, "
                f"not millimetres or studs: a 55 cm wheel is 0.55, not 550"
            )
    return values


def unit_metres(plan: Plan) -> float:
    """How many real metres one assembled unit is worth.

    `scale_reference` names the part that is 1.0 unit; without one, a unit is a
    metre, which is the convention a plan made of primitives is already written
    in — primitives.py builds a 3.2 m cart bed 3.2 units wide.
    """
    if not plan.scale_reference:
        return 1.0
    reference = plan.part(plan.scale_reference)
    if reference is None:
        raise DecomposeError(
            f"scale_reference names unknown part {plan.scale_reference!r}; "
            f"known parts are {sorted(p.name for p in plan.parts)}"
        )
    length = part_length(reference)
    if length is None:
        raise DecomposeError(
            f"scale_reference names {plan.scale_reference!r} as the 1.0 unit, "
            f"but that part declares no size_m, so there is nothing to be the "
            f"unit *of* — give it one, or drop scale_reference and state every "
            f"size in metres"
        )
    return length


def _natural_span(part: Part) -> float:
    """How many units of its own mesh this part already spans.

    1.0 for anything the generator produced — that is the measured fact this
    whole field exists for. A scripted part is built at whatever its params say,
    so it is measured instead, which means a primitive may be drawn at any
    convenient size and `size_m` still lands it correctly: the Bonanza's gear is
    drawn at unit span like a generated part, the cart's is drawn in metres, and
    both come out the size they claim.
    """
    if part.mode != SCRIPT or not part.kind:
        return 1.0
    try:
        mesh = primitives.build(part.kind, part.params)
    except Exception:  # validate() reports this properly; do not mask it here
        return 1.0
    lo, hi = mesh.bounds
    span = float(max(hi - lo))
    return span or 1.0


def part_scale(plan: Plan, part: Part) -> float | None:
    """The uniform `scale` this part needs, or None if it declared no size.

    Uniform, not per-axis: the mesh already has the right proportions, and a
    non-uniform scale would stretch a part that is merely the wrong size. A
    mirror gets None because it inherits its source's whole transform — scaling
    it again would square the source's scale.
    """
    if part.mode == MIRROR:
        return None
    length = part_length(part)
    if length is None:
        return None
    return round(length / unit_metres(plan) / _natural_span(part), 6)


def scales(plan: Plan) -> dict[str, float]:
    """Every part's computed scale, by name. Parts without a size are absent."""
    plan = Plan.from_dict(plan)
    out = {}
    for part in plan.parts:
        scale = part_scale(plan, part)
        if scale is not None:
            out[part.name] = scale
    return out


def placement_of(plan: Plan, part: Part) -> dict:
    """The part's placement, with anything it defers to the plan filled in.

    Only one thing defers today: `"orient": true` means "orient me to the
    extents I already declared", which saves writing the same three numbers
    twice. orient.py takes a bare [x, y, z] of target extents, so this is a
    substitution rather than a translation.
    """
    placement = dict(part.placement)
    if placement.get("orient") is True:
        extents = part_extents(part)
        if extents is None:
            raise DecomposeError(
                f"part {part.name!r} asks to be oriented to its own size_m, but "
                f"size_m is a single length; orienting needs [x, y, z] extents "
                f"because it is the ratios between them that say which way the "
                f"part lies"
            )
        placement["orient"] = extents
    return placement


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

        # Every size is parsed here, where it costs a microsecond, rather than
        # at assembly — a plan whose propeller is 2 000 units across has already
        # spent eight minutes of GPU time by then.
        _size_values(part)
        if part.size_m is not None and part.mode == MIRROR:
            raise DecomposeError(
                f"part {part.name!r} is {MIRROR!r} and also states size_m; a "
                f"mirror takes its whole transform from "
                f"{part.placement.get('mirror_of')!r}, including that part's "
                f"scale, so its own size would be silently ignored"
            )
        if part.size_m is not None and part.placement.get("scale") is not None:
            raise DecomposeError(
                f"part {part.name!r} states both size_m and placement.scale; "
                f"size_m *is* how the scale is computed, so one of the two would "
                f"be thrown away — keep size_m"
            )
        placement_of(plan, part)  # expands `orient: true`, or says why it cannot

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

    # Raises if scale_reference names nothing, or names a part with no size to
    # be the unit of. Cheap, and the alternative is a whole build at unit scale.
    unit_metres(plan)

    sizeless = [
        p.name for p in plan.parts if p.mode == GENERATE and p.size_m is None
    ]
    if sizeless:
        # A warning, not an error: a caller may be scaling by hand downstream,
        # and half a plan is still worth building. But it is the difference
        # between a model and a parts bin, so it does not stay silent.
        warnings.append(
            f"part(s) {sizeless} declare no `size_m`. The generator normalises "
            f"every mesh to a unit box — measured, the longest side comes back "
            f"0.99-1.00 whatever the subject — so these will assemble the same "
            f"size as each other and as everything else. Give each its real "
            f"size in metres: one number for the longest dimension, or "
            f"[x, y, z] extents."
        )

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
            "placement": placement_of(plan, part),
            # The real size the plan declared, and the scale that turns this
            # part's unit box into it. Computed once here so the caller never
            # has to write the throwaway scale script again.
            "size_m": part.size_m,
            "scale": part_scale(plan, part),
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
                # Before the placement, so a plan that stated `scale` there
                # outright still wins — validate() has already rejected stating
                # both. A mirrored part has no scale of its own: it inherits the
                # source's transform, and scaling it again would square it.
                **({"scale": r["scale"]} if r["scale"] is not None else {}),
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
    # Every size_m below is a measurement off a real G36, and the fuselage is
    # the unit — so the scales this produces read as ratios anyone can check
    # without a tape measure: a wing is a bit over half a fuselage (0.5238), a
    # wheel is a fifteenth of one (0.0655).
    "scale_reference": "fuselage",
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
            # 8.4 m nose to tail, and the part every other size is stated
            # against. The full triple rather than the bare 8.4 because a
            # fuselage is 6:1 slender and that ratio is what tells orient.py
            # which of the mesh's three axes is the long one.
            "size_m": [1.1, 1.3, 8.4],
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
            # 4.4 m semi-span, 1.4 m root chord, 0.25 m thick.
            "size_m": [4.4, 0.25, 1.4],
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
            "size_m": [0.18, 1.5, 1.4],
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
            "size_m": [1.7, 0.12, 0.9],
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
            "size_m": [1.1, 1.1, 1.4],
            "material": "paint",
            "placement": {
                "anchor": {"to": "fuselage", "align": {"z": "max"}, "my": {"z": "min"}},
            },
        },
        {
            "name": "propeller",
            "prompt": "a three-blade propeller with a polished spinner hub",
            # 2.0 m disc, and the single-number form is the right one here: a
            # three-blade propeller is round, so there is no long axis to
            # declare and orient.py's `propeller` role deliberately carries no
            # extents either — it finds the spin axis instead.
            "size_m": 2.0,
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
        #
        # Both are drawn at unit span — a 1.0-long cylinder, a 1.0-wide wheel —
        # exactly like a generated part, and `size_m` says what they really
        # measure. That is a free choice, not a requirement: `_natural_span`
        # measures the primitive, so the wooden cart's are drawn in metres
        # instead and land just as correctly. What must not happen is stating a
        # size that disagrees with the drawing, which is how the wheel used to
        # arrive at 0.29 m.
        {
            "name": "left_gear_strut",
            "mode": SCRIPT,
            "kind": "cylinder",
            "params": {"radius": 0.055, "height": 1.0, "chamfer": 0.02},
            "size_m": [0.11, 0.9, 0.11],
            "material": "metal",
            "placement": {
                "anchor": {"to": "left_wing", "align": {"y": "under", "x": 0.7, "z": 0.5}},
            },
        },
        {
            "name": "left_gear_wheel",
            "mode": SCRIPT,
            "kind": "wheel",
            "params": {"radius": 0.5, "width": 0.21, "hub_radius": 0.15,
                       "rim_width": 0.42, "spoke_count": 6, "chamfer": 0.04},
            # In the frame the primitive is drawn in — a 0.55 m disc in x/z,
            # 0.12 m wide along y — not the frame the `rotation` below puts it
            # in. size_m describes the part, placement moves it.
            "size_m": [0.55, 0.12, 0.55],
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
    # No scale_reference, so one unit is one metre — which is what this plan was
    # already written in, because primitives.py builds a 3.2 m bed 3.2 units
    # wide. Every scripted part below therefore computes a scale of exactly 1.0
    # and nothing moves; the two *generated* parts are the ones that needed the
    # field, and they are the reason it exists at all.
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
            # The same three numbers the params already state, which is exactly
            # the point: a scripted part is drawn at its real size, so declaring
            # it changes nothing and the plan still reads as dimensioned.
            "size_m": [3.2, 0.7, 1.8],
            "placement": {"anchor": {"to": "ground"}, "position": [0, 0, 0]},
        },
        {
            "name": "axle",
            "mode": SCRIPT,
            "kind": "cylinder",
            "params": {"radius": 0.09, "height": 2.1},
            "size_m": [0.18, 2.1, 0.18],
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
            "size_m": [1.7, 0.22, 1.7],
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
            "size_m": [2.4, 0.1, 0.14],
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
            # A metre-long roll. Without this it would arrive as big as the
            # cart — the generator returns a unit box whatever it drew, and a
            # bundle of cloth has no dimensions of its own to fall back on.
            "size_m": [1.1, 0.45, 0.45],
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
            "size_m": [0.18, 0.35, 0.18],
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
