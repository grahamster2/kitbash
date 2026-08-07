"""Turn a generated part the right way round before it is placed.

Positions, proportions and colour can all be correct and the assembly still
looks like debris, because an image-to-3D generator has no reason to emit a
part in any particular frame. It reconstructs the object in its *input
camera's* frame, and the reference image was a three-quarter view — so the
mesh arrives rotated by whatever angle the image happened to use. Measured on
the Bonanza set: the airframe's own symmetry plane sat 318 degrees round on one
subject and 174 on another, and a twelve-part aircraft came out with the wings
vertical, the fin lying on its side and the tailplane pointing forward.

Roll was already fixed upstream (TRELLIS emits Z-up; the untextured path used
to skip the glTF Y-up conversion). Azimuth was not, and no amount of care in
the placement vocabulary can fix it: `anchor` measures a *box*, and a box
cannot tell that the thing inside it is lying down.

What the caller does know is what the part *is* — a Bonanza wing is 4.4 m of
span, 1.4 m of chord and 0.25 m thick — and that is enough. Fit an oriented
bounding box, decide which of its axes is span, chord and thickness, and the
rotation follows. Nothing is regenerated; this repairs meshes that already
exist.

Four things make that harder than sorting three numbers, and they are what
most of this module is about:

- **An OBB is blind to 180 degrees.** Nose-forward and nose-backward produce
  the same box. Resolved from where the *mass* sits along each axis: a wing is
  fatter at the root, a fuselage at the cabin, a fin at its base. The caller
  declares which way a part tapers and the mesh is asked whether it agrees.
- **A box is not the part.** The generated wing measures 1 : 0.22 : 0.17, so
  chord and thickness are nearly the same number where the real wing's differ
  by five times; and a plate with an upturned tip fence, or the tailplane's
  curled trailing edge, has a box deeper than the plate inside it, so sorting
  extents stands it on its edge. The second opinion is the *surface normals*: a
  slab spends most of its area on the two faces that look along its flat axis,
  whatever its box says.
- **Some parts have no long axis to sort.** A propeller is a spinner with
  blades round it; which way its blades happen to be clocked is meaningless,
  and its extents depend on where the blades stopped. Those parts are named by
  their axis of rotational symmetry instead, which is detected rather than
  measured off a box.
- **Some parts genuinely have no answer.** A near-cubic fin, a cowl that is a
  body of revolution. Guessing there is worse than doing nothing, so every
  result carries a `confidence` and the caller may set a floor.

Confidence is not a vibe. It is the expected *agreement* between the
orientation we picked and every other orientation still consistent with the
declaration, measured as voxel overlap. Which gives the right answer at both
ends: a cowl is symmetric, so its rivals look identical to the winner and
confidence stays high even though the match is ambiguous; a wing's rival is
edge-on, overlaps badly, and confidence collapses if the evidence is thin.

numpy and trimesh only — no scipy anywhere in this server, which rules out a
convex hull and therefore trimesh's own `oriented_bounds`. The box search here
is the substitute. Pure CPU; a part costs ~0.25 s.
"""
from dataclasses import dataclass, field
from itertools import permutations

import numpy as np
import trimesh

AXES = ("x", "y", "z")
_AXIS = {"x": 0, "y": 1, "z": 2}

# How the aircraft convention in docs/MULTI-PART.md falls out: +Y up, and the
# assembled Bonanza points its nose at +Z (the cowl anchors to the fuselage's
# z max, the fin to its z min). Every role below is written in that frame, and
# every left/right part is the *left* one — the right one is `mirror_of`, which
# inherits this rotation for free.
#
# `extents` are real metres in the target frame; only their ratios are used.
# `taper` names, per axis, the direction the part gets THINNER in — so a wing
# tapering to a tip at -x is {"x": "-"}, and the fat end is therefore at +x.
ROLES: dict[str, dict] = {
    # Beechcraft G36 dimensions, which is what the shipped decomposition plan
    # builds. Anything similar in proportion works; ratios are what matter.
    "fuselage": {"extents": [1.1, 1.3, 8.4], "taper": {"z": "-"}},
    "wing": {"extents": [4.4, 0.25, 1.4], "taper": {"x": "-", "z": "-"}},
    "tailplane": {"extents": [1.7, 0.12, 0.9], "taper": {"x": "-", "z": "-"}},
    # Height and chord are declared *equal* on purpose, and only the height is
    # given a taper. A generated fin is rarely the proportion a real one is —
    # this one came back squatter than it is long — so the extents cannot be
    # trusted to say which way up it goes, while "fat at the root, thin at the
    # tip" holds for every fin ever built. Declaring a fore-and-aft taper as
    # well would undo it: two identical claims on two equal axes cancel, and
    # the fin goes up on its leading edge.
    "fin": {"extents": [0.12, 1.0, 1.0], "taper": {"y": "+"}},
    # No taper: a cowling is a barrel open at both ends, and the generated one
    # measures within 5% of symmetric along its axis. Declaring a front for it
    # would be inventing evidence, and the module would duly find some — it put
    # the cowl in sideways, because the only real asymmetry in that mesh is
    # across the barrel rather than along it.
    "cowl": {"extents": [1.1, 1.1, 1.4], "spin": "z"},
    # No extents: a generated propeller is a spinner with blades round it, and
    # its box says whatever the blades were doing. Its shaft is the only thing
    # about it that is not arbitrary, and `spin` names it directly.
    "propeller": {"spin": "z", "taper": {"z": "+"}},
    # Generic hardware, useful well beyond aircraft.
    "strut": {"extents": [0.11, 0.9, 0.11], "spin": "y"},
    "wheel": {"extents": [0.2, 0.55, 0.55], "spin": "x"},
    "blade": {"extents": [1.0, 0.08, 0.3], "taper": {"x": "-"}},
    "plate": {"extents": [1.0, 0.06, 1.0]},
    "rod": {"extents": [0.1, 1.0, 0.1], "spin": "y"},
}

# Weights on the four pieces of evidence. Each is load-bearing for a different
# part and none of them is decoration: turn the extents off and nothing turns
# differently but every confidence collapses; turn the normals off and a plate
# with a tip fence goes on edge; turn `spin` off and the propeller goes on
# sideways; turn `taper` off and the fuselage flies backwards. The ablation is
# in docs/ORIENTATION.md.
W_EXTENT = 1.0
W_NORMAL = 1.6
W_TAPER = 1.0
W_SPIN = 2.0

# How much the box search prefers a frame its surfaces agree with over the
# smallest box. Not a nicety: the minimum-volume box of the generated tail —
# a fin with a stabiliser across its foot, so a cruciform in plan — is the one
# turned 45 degrees, because a diagonal rectangle round a plus sign is smaller
# than the upright one. On a clean cross that box is 40% tighter and completely
# meaningless. A hard-surface part's real frame is written in its face normals,
# and at this weight both the synthetic cross and the generated fin land square
# (surface agreement 0.99 against 0.80) while every organic part in the Bonanza
# set keeps the frame the volume alone chose, to within a percent.
W_ALIGN = 3.0

# The asymmetry (as a fraction of half the part's length) that counts as a
# decisive answer to "which end is the nose". Measured on the Bonanza parts, a
# genuinely tapered axis runs 0.09-0.29 and a symmetric one under 0.02, so this
# sits below the weakest real signal and above the strongest piece of noise.
TAPER_FULL = 0.06

# Softmax temperature over candidate costs. Set so that a candidate one
# "meaningful discrepancy" worse than the winner keeps ~5% of the weight, i.e.
# it still drags confidence down if it looks nothing like the winner.
TEMPERATURE = 0.34

# A candidate whose posterior weight is below this cannot move the confidence
# and is not worth voxelising.
_NEGLIGIBLE = 0.002

# Prefer the answer that moves the part least when nothing else separates two
# candidates, so a part that already sat correctly is not spun for nothing.
_IDLE_BIAS = 2e-3


@dataclass
class Orientation:
    """The rotation that puts one part into the declared frame."""

    rotation: list[float]                 # XYZ euler degrees, for assemble.py
    matrix: np.ndarray                    # the same thing as a 3x3
    confidence: float
    extents: list[float]                  # the part's box after rotating
    declared: list[float]                 # what the caller said it should be
    asymmetry: list[float]                # signed mass offset per world axis
    degrees: float                        # how far this turns the part
    notes: list[str] = field(default_factory=list)

    @property
    def identity(self) -> bool:
        """Did this leave the part where it was? The box search is numerical,
        so "unturned" is a fraction of a degree rather than exactly zero."""
        return self.degrees < 0.5

    def as_dict(self) -> dict:
        return {
            "rotation": [round(float(v), 3) for v in self.rotation],
            "confidence": round(float(self.confidence), 3),
            "degrees": round(float(self.degrees), 1),
            "extents": [round(float(v), 4) for v in self.extents],
            "declared": [round(float(v), 4) for v in self.declared],
            "asymmetry": [round(float(v), 3) for v in self.asymmetry],
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# the declaration
# --------------------------------------------------------------------------
def roles() -> list[str]:
    return sorted(ROLES)


@dataclass
class Declaration:
    """What the caller says the part is."""

    extents: np.ndarray | None          # target [x, y, z]; ratios only
    taper: dict[str, int]               # axis -> direction it gets thinner in
    spin: str | None                    # axis of rotational symmetry, if any


def resolve_spec(spec) -> Declaration:
    """A caller's `orient` value -> a Declaration.

    Accepts a role name, a bare [x, y, z], or a dict of both — because the
    interesting case is "a wing, but this one is mounted the other way", and
    that should be one key of override rather than a fresh set of numbers.
    """
    if spec is None:
        raise ValueError("orient needs a role name or target extents")

    if isinstance(spec, str):
        spec = {"role": spec}
    elif isinstance(spec, (list, tuple, np.ndarray)):
        spec = {"extents": list(spec)}
    if not isinstance(spec, dict):
        raise ValueError(f"orient must be a role name, [x, y, z] or an object, got {spec!r}")

    unknown = set(spec) - {"role", "extents", "taper", "spin"}
    if unknown:
        # A misspelled key would otherwise mean "no such declaration", which is
        # the same silent nothing a typo'd axis name used to be in an anchor.
        raise ValueError(
            f"orient has unknown key(s) {sorted(unknown)}; expected role, "
            f"extents, taper or spin"
        )

    extents, taper, spin = None, {}, None
    role = spec.get("role")
    if role is not None:
        key = str(role).strip().lower()
        if key not in ROLES:
            raise ValueError(f"unknown orient role {role!r}; known roles are {roles()}")
        extents = ROLES[key].get("extents")
        taper = dict(ROLES[key].get("taper") or {})
        spin = ROLES[key].get("spin")

    if spec.get("extents") is not None:
        extents = list(spec["extents"])
    if spec.get("taper") is not None:
        # An explicit taper replaces the role's rather than merging with it: a
        # part mounted backwards needs to *remove* the role's assumption, and
        # merging leaves no way to say that.
        taper = dict(spec["taper"])
    if "spin" in spec:
        spin = spec["spin"]

    if spin is not None:
        spin = str(spin).strip().lower()
        if spin not in _AXIS:
            raise ValueError(f"orient.spin is {spin!r}; expected x, y or z")

    if extents is not None:
        if len(extents) != 3:
            raise ValueError(f"orient.extents must be [x, y, z], got {extents!r}")
        extents = np.array([float(v) for v in extents], dtype=np.float64)
        if not np.all(np.isfinite(extents)) or np.any(extents <= 0):
            raise ValueError(f"orient.extents must all be positive, got {extents.tolist()}")
    elif spin is None:
        raise ValueError("orient needs `extents` [x, y, z], a `spin` axis, or a known `role`")

    signs: dict[str, int] = {}
    for key, value in taper.items():
        axis = str(key).strip().lower()
        if axis not in _AXIS:
            raise ValueError(f"orient.taper has axis {key!r}; expected x, y or z")
        text = str(value).strip().lower()
        if text in ("+", "+1", "max", "high", "positive"):
            signs[axis] = 1
        elif text in ("-", "-1", "min", "low", "negative"):
            signs[axis] = -1
        else:
            raise ValueError(
                f"orient.taper.{axis} is {value!r}; expected '+' or '-' — the "
                f"direction the part gets thinner in"
            )
    return Declaration(extents, signs, spin)


# --------------------------------------------------------------------------
# measuring the mesh
# --------------------------------------------------------------------------
def _samples(mesh: trimesh.Trimesh, count: int, seed: int = 0) -> np.ndarray:
    """Area-weighted points on the surface.

    Area-weighted rather than raw vertices because a remesher puts vertices
    where curvature is, not where the object is: the fuselage carries a third
    of its vertices on the canopy, which would drag every mass measurement
    forward and invent a taper that is not there.
    """
    faces = mesh.triangles
    if len(faces) == 0:
        raise ValueError("cannot orient a mesh with no faces")
    area = mesh.area_faces
    total = float(area.sum())
    if total <= 0:
        raise ValueError("cannot orient a mesh with zero surface area")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(faces), count, p=area / total)
    a, b, c = faces[idx, 0], faces[idx, 1], faces[idx, 2]
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    over = (u + v > 1).ravel()
    u[over], v[over] = 1.0 - u[over], 1.0 - v[over]
    return a + u * (b - a) + v * (c - a)


def _fibonacci_sphere(count: int) -> np.ndarray:
    """Directions spread over a hemisphere; a box face normal and its opposite
    describe the same box, so half the sphere is the whole search."""
    i = np.arange(count) + 0.5
    z = i / count
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _perp(n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = n / (np.linalg.norm(n) or 1.0)
    seed = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, seed)
    e1 /= np.linalg.norm(e1) or 1.0
    return n, e1, np.cross(n, e1)


def _normal_histogram(mesh: trimesh.Trimesh, buckets: int = 192) -> tuple:
    """Face normals collapsed to a few hundred area-weighted directions.

    The frame search asks "how well do the surfaces agree with these axes?"
    thousands of times, and asking it of every face would dominate the cost.
    Opposite normals share a bucket because a box face and its opposite are the
    same constraint.
    """
    directions = _fibonacci_sphere(buckets)
    idx = np.argmax(np.abs(mesh.face_normals @ directions.T), axis=1)
    weight = np.bincount(idx, weights=mesh.area_faces, minlength=buckets)
    keep = weight > 0
    total = weight.sum() or 1.0
    return directions[keep], weight[keep] / total


def _sweep(points, n, bins, weights, steps, around=None, span=None) -> tuple:
    """Best box with one face perpendicular to `n`, over the in-plane angle.

    With the box's third axis pinned the problem drops to two dimensions, and
    the remaining angle is one vectorised sweep over a quarter turn — a
    rectangle at 90 degrees is the same rectangle. Both terms of the objective
    are evaluated for every angle at once.
    """
    n, e1, e2 = _perp(n)
    depth = float(np.ptp(points @ n))
    if around is None:
        theta = np.linspace(0.0, np.pi / 2, steps, endpoint=False)
    else:
        theta = np.linspace(around - span, around + span, steps)
    cos, sin = np.cos(theta), np.sin(theta)

    a, b = points @ e1, points @ e2
    u = np.outer(cos, a) + np.outer(sin, b)
    v = -np.outer(sin, a) + np.outer(cos, b)
    width = u.max(axis=1) - u.min(axis=1)
    height = v.max(axis=1) - v.min(axis=1)
    volume = width * height * depth

    # Surface agreement: the area-weighted mean of how closely each normal
    # lines up with its nearest box axis. 1.0 is a shape whose faces are all
    # square to the frame.
    ba, bb = bins @ e1, bins @ e2
    along_n = np.abs(bins @ n)
    p = np.abs(np.outer(cos, ba) + np.outer(sin, bb))
    q = np.abs(-np.outer(sin, ba) + np.outer(cos, bb))
    align = (np.maximum(np.maximum(p, q), along_n) * weights).sum(axis=1)

    # Volume enters as a log so the trade against alignment is scale-free.
    cost = np.log(np.maximum(volume, 1e-12)) - W_ALIGN * align
    k = int(np.argmin(cost))
    axes = np.stack(
        [cos[k] * e1 + sin[k] * e2, -sin[k] * e1 + cos[k] * e2, n], axis=1
    )
    return (float(cost[k]), axes, np.array([width[k], height[k], depth]),
            float(align[k]), float(theta[k]))


def oriented_box(
    mesh: trimesh.Trimesh, coarse: int = 48, keep: int = 6, walks: int = 2,
    polish: int = 70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """The part's own frame: (axes as columns, extents, centre, agreement).

    trimesh's `oriented_bounds` needs a convex hull, which needs scipy, which
    this server does not have and does not want (see texturing.py making the
    same trade). So: sweep candidate box-face normals over a hemisphere, solve
    the remaining in-plane angle exactly, keep the best few and polish them.

    The candidate directions include the mesh's own dominant surface normals,
    not just a blind sphere, because those are where the answer usually is —
    and the objective is volume traded against surface agreement rather than
    volume alone. See W_ALIGN for the tail that taught us that.

    Deterministic: fixed direction set, fixed seed, no ties broken by chance.
    """
    points = np.asarray(mesh.vertices, dtype=np.float64)
    if len(points) < 4:
        raise ValueError("cannot orient a mesh with fewer than 4 vertices")

    bins, weights = _normal_histogram(mesh)
    dominant = bins[np.argsort(weights)[::-1][:8]]

    # A box only ever touches extreme points, so the search can run on a
    # fraction of them and the winner is re-measured against all of them.
    rng = np.random.default_rng(0)
    small = points
    if len(points) > 2000:
        small = points[rng.choice(len(points), 2000, replace=False)]

    # The whole search runs on the subsample, at a fine in-plane resolution.
    # Fine matters more than exhaustive: scored on a coarse angle grid, a
    # candidate direction is charged for the grid's error rather than its own,
    # and the true frame of a fin lost to one five degrees off it.
    directions = np.vstack([_fibonacci_sphere(coarse), np.eye(3), dominant])
    coarse_hits = [_sweep(small, n, bins, weights, 45) for n in directions]
    order = np.argsort([hit[0] for hit in coarse_hits])[:keep]

    seeds = [_sweep(small, coarse_hits[i][1][:, 2], bins, weights, 120) for i in order]
    seeds.sort(key=lambda s: s[0])

    # Walk downhill from more than one seed. A hemisphere of 48 directions is
    # 15 degrees apart, which is far enough that the true frame's neighbour can
    # rank second; refining only the leader found a wing 6 degrees crooked and
    # was perfectly happy with it.
    best = seeds[0]
    for seed in seeds[:walks]:
        current, step = seed, 0.15
        for _ in range(polish):
            n = current[1][:, 2] + rng.normal(0.0, step, 3)
            cand = _sweep(small, n / (np.linalg.norm(n) or 1.0), bins, weights, 120)
            if cand[0] < current[0]:
                current = cand
            step *= 0.97
        if current[0] < best[0]:
            best = current

    # Then one narrow pass over *every* vertex: the direction is settled, and
    # what remains is the box actually containing the part rather than the
    # 2000 points that stood in for it.
    best = _sweep(points, best[1][:, 2], bins, weights, 120,
                  around=best[4], span=np.radians(2.0))

    axes, extents = best[1], best[2]
    local = points @ axes
    centre = axes @ ((local.max(axis=0) + local.min(axis=0)) / 2.0)
    return axes, extents, centre, best[3]


def _normal_share(mesh: trimesh.Trimesh, axes: np.ndarray) -> np.ndarray:
    """How much of the surface area looks along each box axis, normalised.

    A slab keeps most of its area on the two faces whose normal is its flat
    axis, and it does so whatever its box says. That matters when the box is
    bigger than the part: a plate with an upturned tip fence measures taller
    than it is deep, so sorted extents stand it on its edge, while 69% of its
    surface still looks along the flat axis and says otherwise. The generated
    tailplane is that shape — a curled trailing edge makes its box a third
    deeper than the plate inside it.
    """
    weight = mesh.area_faces
    share = np.abs(mesh.face_normals @ axes) * weight[:, None]
    total = share.sum()
    if total <= 0:
        return np.full(3, 1.0 / 3.0)
    return share.sum(axis=0) / total


def _predicted_share(extents: np.ndarray) -> np.ndarray:
    """The normal share a *box* of these proportions would have.

    The faces perpendicular to axis i have area proportional to the other two
    extents, so the declaration predicts a distribution without the caller
    having to think about surfaces at all.
    """
    face = np.array([extents[1] * extents[2],
                     extents[0] * extents[2],
                     extents[0] * extents[1]])
    return face / face.sum()


def _asymmetry(points: np.ndarray, axes: np.ndarray, extents: np.ndarray,
               centre: np.ndarray) -> np.ndarray:
    """Signed mass offset along each box axis, as a fraction of half the box.

    Positive means the part is fatter at the +axis end. This is the whole
    answer to the 180-degree question: a box is symmetric, a wing is not.
    """
    local = (points - centre) @ axes
    safe = np.where(extents > 0, extents, 1.0)
    return 2.0 * local.mean(axis=0) / safe


# --------------------------------------------------------------------------
# candidates and confidence
# --------------------------------------------------------------------------
def _candidates(axes: np.ndarray) -> list[tuple[np.ndarray, tuple[int, int, int], np.ndarray]]:
    """The 24 rotations that put the box's own axes on the world axes.

    Every one of them is a legitimate reading of the same box; picking between
    them is the entire problem. Reflections are excluded — mirroring a part
    turns a left wing into a right one, which is a different object.
    """
    out = []
    for perm in permutations(range(3)):
        for bits in range(8):
            signs = np.array([1 - 2 * ((bits >> k) & 1) for k in range(3)], dtype=float)
            # Row k is "which box axis, and which way, becomes world axis k".
            R = np.stack([signs[k] * axes[:, perm[k]] for k in range(3)], axis=0)
            if np.linalg.det(R) > 0:
                out.append((R, perm, signs))
    return out


def _cost(perm, signs, extents, share, asym, spin, decl) -> float:
    """How badly one candidate contradicts what the caller declared."""
    total = 0.0

    if decl.extents is not None:
        target = decl.extents
        world_ext = np.array([extents[perm[k]] for k in range(3)])
        world_share = np.array([share[perm[k]] for k in range(3)])

        # Scale-free: the caller states metres and the mesh is in generator
        # units, so only the *shape* of the two extent vectors may be compared.
        ratio = np.log(world_ext / target)
        total += W_EXTENT * float(np.mean((ratio - ratio.mean()) ** 2))

        predicted = _predicted_share(target)
        total += W_NORMAL * float(
            np.sum((np.sqrt(world_share) - np.sqrt(predicted)) ** 2)
        )

    if decl.spin is not None:
        # The declared axis of revolution should land on the mesh axis that
        # most nearly has one. Scored against the best axis rather than against
        # a perfect 1.0, because a lumpy generated propeller only manages 0.47
        # on its own shaft and what matters is that the shaft beats the others.
        # If no axis stands out the whole term flattens and stops mattering.
        best = float(spin.max())
        if best > 1e-6:
            total += W_SPIN * float(best - spin[perm[_AXIS[decl.spin]]]) / best

    world_asym = np.array([signs[k] * asym[perm[k]] for k in range(3)])
    for axis, thins_toward in decl.taper.items():
        i = _AXIS[axis]
        wanted = -thins_toward           # the fat end is opposite the taper
        # A declared taper rewards an axis that really does taper, as well as
        # punishing one that tapers the wrong way. Punishment alone is not
        # enough: a symmetric axis is never *wrong*, so a fin whose chord and
        # height are declared equal would put its uniform chord upright and its
        # tapered height across, at no cost, on a coin flip.
        # The reward is capped below the penalty on purpose. Being fat at the
        # expected end is weak evidence that this is even the right axis —
        # every part is lopsided about *something* — while being fat at the
        # wrong end is strong evidence that it is not. Capped this low, a taper
        # can still break a tie between two equally plausible axes and cannot
        # outvote a rotational symmetry that was actually measured.
        agreement = wanted * world_asym[i] / TAPER_FULL
        total -= W_TAPER * float(np.clip(agreement, -2.0, 0.75))

    return total


def _occupancy(points: np.ndarray, res: int) -> np.ndarray:
    idx = np.clip(((points + 0.5) * res).astype(np.int64), 0, res - 1)
    grid = np.zeros((res,) * 3, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def _dilate(grid: np.ndarray) -> np.ndarray:
    """Grow a boolean grid by one cell each way. Same trick as texturing.py's
    symmetry test, and for the same reason: these are shells, and punishing a
    perfect match for landing half a voxel off measures the grid, not the mesh.
    """
    out = grid.copy()
    for axis in range(3):
        src = np.moveaxis(grid, axis, 0)
        dst = np.moveaxis(out, axis, 0)
        dst[1:] |= src[:-1]
        dst[:-1] |= src[1:]
    return out


def _agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided voxel coverage of one orientation against another: 1.0 when
    the part looks the same either way, near 0 when it plainly does not."""
    fat_a, fat_b = _dilate(a), _dilate(b)
    na, nb = np.count_nonzero(a), np.count_nonzero(b)
    if not na or not nb:
        return 0.0
    return min(np.count_nonzero(a & fat_b) / na, np.count_nonzero(b & fat_a) / nb)


def spin_scores(local: np.ndarray, resolution: int = 24,
                folds: tuple[int, ...] = (3, 4, 6)) -> np.ndarray:
    """Per box axis: does turning the part about it leave it unchanged?

    A propeller answers yes at a third of a turn, a wheel or a cowl at every
    turn tested, a fuselage at none — and that is a far better description of
    what those parts *are* than three numbers off a box. It also settles a
    question the box cannot: turning a symmetric part about its own axis is
    free, so an ambiguity there costs nothing and must not cost confidence.

    Half a turn is deliberately not tested. Almost everything is roughly
    two-fold symmetric about *something* — a wing turned upside down is still a
    wing-shaped lump — so it separates nothing, and measured on the Bonanza
    parts it scored 0.96 on the cowl's short axis and sent the cowl in
    sideways. The price is that a two-blade propeller is not recognisable this
    way and needs its extents declared instead.
    """
    base = _occupancy(local, resolution)
    fat = _dilate(base)
    n = np.count_nonzero(base)
    if not n:
        return np.zeros(3)

    scores = np.zeros(3)
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        # Turn about the part's centre of mass, not the centre of its box. A
        # three-blade rotor's box is not centred on its shaft — the blades are
        # 120 degrees apart, so the silhouette is lopsided — and rotating about
        # the box centre scored a perfectly symmetric propeller at 0.43.
        pivot_u, pivot_v = local[:, u].mean(), local[:, v].mean()
        du, dv = local[:, u] - pivot_u, local[:, v] - pivot_v
        for fold in folds:
            theta = 2.0 * np.pi / fold
            turned = local.copy()
            turned[:, u] = pivot_u + np.cos(theta) * du - np.sin(theta) * dv
            turned[:, v] = pivot_v + np.sin(theta) * du + np.cos(theta) * dv
            grid = _occupancy(turned, resolution)
            m = np.count_nonzero(grid)
            if not m:
                continue
            score = min(np.count_nonzero(base & _dilate(grid)) / n,
                        np.count_nonzero(grid & fat) / m)
            scores[axis] = max(scores[axis], score)
    return scores


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def orient(mesh: trimesh.Trimesh, spec=None, *, samples: int = 20000,
           resolution: int = 24, seed: int = 0) -> Orientation:
    """Work out the rotation that puts `mesh` into the frame `spec` declares.

    `spec` is a role name, an [x, y, z] of real target extents, or an object
    with `role` / `extents` / `taper` / `spin`. The returned rotation is XYZ euler
    degrees, ready for assemble.py's `rotation`, and the confidence is the
    expected voxel agreement between what was chosen and every rival reading
    of the same box that the declaration still permits.
    """
    decl = resolve_spec(spec)

    axes, extents, centre, agreement = oriented_box(mesh)
    points = _samples(mesh, samples, seed=seed)
    share = _normal_share(mesh, axes)
    asym = _asymmetry(points, axes, extents, centre)

    # Everything below scores candidates in the box's own frame, so the mesh is
    # sampled and normalised exactly once.
    local = (points - centre) @ axes
    scale = float(extents.max()) or 1.0
    local = local / scale

    spin = spin_scores(local, resolution) if decl.spin is not None else np.zeros(3)

    scored = []
    for R, perm, signs in _candidates(axes):
        cost = _cost(perm, signs, extents, share, asym, spin, decl)
        # Break exact ties toward leaving the part alone. Too small to overturn
        # any real evidence, big enough that an already-correct part that
        # measures identically both ways stays put.
        angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        scored.append((cost + _IDLE_BIAS * angle, cost, R, perm, signs, angle))

    scored.sort(key=lambda s: s[0])
    costs = np.array([s[0] for s in scored])
    weights = np.exp(-(costs - costs.min()) / TEMPERATURE)
    weights /= weights.sum()

    best = scored[0]
    R_best, perm, signs, angle = best[2], best[3], best[4], best[5]

    # Confidence: how much of the surviving probability mass lands on
    # orientations that *look like* the one we picked. A part that is symmetric
    # about the ambiguous axis scores high because its rivals are the same
    # shape; a wing that could be edge-on scores low because they are not.
    grids: dict[tuple, np.ndarray] = {}

    def grid_for(perm_, signs_):
        key = (perm_, tuple(signs_))
        if key not in grids:
            world = np.stack([signs_[k] * local[:, perm_[k]] for k in range(3)], axis=1)
            grids[key] = _occupancy(world, resolution)
        return grids[key]

    reference = grid_for(perm, tuple(signs))
    spin_axis = _AXIS[decl.spin] if decl.spin is not None else None
    confidence = 0.0
    rival = None
    for w, (_, _, _, perm_, signs_, _) in zip(weights, scored):
        if w < _NEGLIGIBLE:
            continue
        same = perm_ == perm and np.array_equal(signs_, signs)
        agree = 1.0 if same else _agreement(reference, grid_for(perm_, tuple(signs_)))
        if (not same and spin_axis is not None
                and perm_[spin_axis] == perm[spin_axis]
                and signs_[spin_axis] == signs[spin_axis]):
            # Same shaft, pointing the same way: the rival is this part clocked
            # round its own axis of symmetry, which is not a different answer.
            # Voxels disagree about where the blades are; nobody else does.
            agree = max(agree, float(spin[perm[spin_axis]]))
        confidence += w * agree
        if not same and (rival is None or w * (1.0 - agree) > rival[0]):
            rival = (w * (1.0 - agree), w, agree, perm_, signs_)

    notes = _notes(extents, decl, asym, perm, signs, agreement, rival)

    world_ext = np.array([extents[perm[k]] for k in range(3)])
    world_asym = np.array([signs[k] * asym[perm[k]] for k in range(3)])
    T = np.eye(4)
    T[:3, :3] = R_best
    rx, ry, rz = trimesh.transformations.euler_from_matrix(T, "sxyz")

    return Orientation(
        rotation=[float(np.degrees(a)) for a in (rx, ry, rz)],
        matrix=R_best,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        extents=[float(v) for v in world_ext],
        declared=[float(v) for v in (decl.extents if decl.extents is not None else [])],
        asymmetry=[float(v) for v in world_asym],
        degrees=float(np.degrees(angle)),
        notes=notes,
    )


def _notes(extents, decl, asym, perm, signs, agreement, rival) -> list[str]:
    """Say *why* a part is uncertain, because "0.42" on its own is not
    actionable — the caller has to decide whether to re-generate, re-declare or
    place it by hand."""
    notes: list[str] = []
    if decl.extents is not None:
        world_ext = np.array([extents[perm[k]] for k in range(3)])
        shape = world_ext / world_ext.max()
        want = np.asarray(decl.extents) / max(decl.extents)
        for k in range(3):
            if shape[k] > 3.0 * want[k] or want[k] > 3.0 * shape[k]:
                notes.append(
                    f"{AXES[k]} measures {shape[k]:.2f} of the part's length where "
                    f"the declaration says {want[k]:.2f} — the mesh is not the shape "
                    f"it was declared to be"
                )
    for axis, _ in decl.taper.items():
        i = _AXIS[axis]
        value = signs[i] * asym[perm[i]]
        if abs(value) < TAPER_FULL / 3.0:
            notes.append(
                f"{axis} taper is unresolved: the part is near-symmetric end to "
                f"end ({value:+.3f}), so which way round it faces is a guess"
            )
    if agreement < 0.75:
        notes.append(
            f"the part has no clear frame of its own — only {agreement:.0%} of its "
            f"surface is square to the box it was measured in"
        )
    if rival is not None and rival[1] > 0.15 and rival[2] < 0.8:
        notes.append(
            f"a rival orientation keeps {rival[1]:.0%} of the evidence and "
            f"overlaps this one only {rival[2]:.0%}"
        )
    return notes
