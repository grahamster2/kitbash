"""Hollow interiors — the one thing an image-to-3D generator will never give you.

Every generator here emits a **solid**. A generated fuselage is a filled lump of
geometry: there is no cabin inside it, only more fuselage. For Roblox that is
fatal in a way it is not for a render, because the whole point of a vehicle is
that you get in it and the whole point of a building is that you walk into it.

Two ways to get an interior, and this module does both:

**Carve one out of a generated mesh.** The textbook move is an inward surface
offset and a boolean difference, and it does not survive real input: the meshes
coming off decimation are not watertight, not winding-consistent, and report
negative volume (measured in docs/HOLLOW.md). An exact boolean engine is
entitled to refuse those, and `manifold3d` does. So the default route never
performs a boolean at all — it rasterises the mesh into a voxel grid, floods the
outside to decide what "inside" means, builds a signed distance field with an
exact Euclidean distance transform, and reads the shell back out as the
isosurface pair `{0, -wall}`. Rasterising is what makes it robust: by the time
the shell is computed there is no topology left to be wrong, only numbers on a
grid, and a crack narrower than a voxel has already closed itself. Cracks wider
than a voxel are bridged by dilating the skin before the flood and handing the
voxel back afterwards; how far that had to go is reported, because it is also
how far the skin moved.

**Or build it hollow in the first place.** A room, a crate with an open top or a
silo does not need carving; it needs arithmetic. Those are down at the bottom of
this file, built the same way `primitives.py` builds everything — composition of
closed solids, no boolean engine, exact dimensions, a few hundred triangles.

Pure numpy + trimesh, both MIT. The distance transform, the flood fill, the
isosurface extraction and the ray probe are all written out here rather than
imported, because the libraries that provide them are the ones this project
refuses to depend on: `scipy` is not installed on the test box at all,
`scikit-image` would be another 90 MB for one function, and `pymeshlab` and
`bpy` are GPL (docs/DECIMATION.md). `manifold3d` *is* licence-clean — Apache-2.0
with no bundled LGPL, unlike `cadquery-ocp`, and the check is written up in
docs/HOLLOW.md — it is simply not needed for this.
"""
import logging
import math
import os
import time
from dataclasses import dataclass, field as _field
from pathlib import Path

import numpy as np
import trimesh

import assemble
import export
import materials
import primitives

log = logging.getLogger("kitbash.hollow")

_EPS = 1e-9

# Axis name -> index, taken from assemble rather than restated, so an opening
# and an anchor can never drift apart on what "z" means.
_AXIS = {name: i for i, name in enumerate(assemble.AXES)}

# Voxels along the longest side of the mesh. 128 resolves a door frame on a
# 1-unit generated part and costs about a second; the cost is cubic, so this is
# the dial that matters most.
DEFAULT_RESOLUTION = int(os.environ.get("KITBASH_HOLLOW_RESOLUTION", "128"))

# Above this the grid stops fitting comfortably in the 16 GB dev laptop, and the
# subdivision that feeds it is what blows up first, not the grid itself.
MAX_RESOLUTION = 320

# Generated parts are normalised to roughly 1-2 units across, so a wall of 0.04
# is a plausible default hull. Everything is rescaled at export time.
DEFAULT_WALL_THICKNESS = 0.04

# Roblox's per-MeshPart cap; a shell is two surfaces, so it is easy to blow.
# Aliased rather than redeclared so there is one source for the number.
ROBLOX_MAX_TRIANGLES = export.ROBLOX_MAX_TRIANGLES

# The wall has to survive rasterisation. Below about two voxels the erosion that
# forms the cavity eats the wall entirely and you get your solid back.
MIN_WALL_VOXELS = 2.0


# --- signed distance field ---------------------------------------------------

@dataclass
class Field:
    """A signed distance field on a regular grid: negative inside the solid.

    `phi[i, j, k]` is the distance, in world units, from the centre of that
    voxel to the surface. Sample positions are `origin + (index + 0.5) * pitch`,
    which puts the zero crossing on the boundary between a solid voxel and its
    empty neighbour rather than half a voxel off it.
    """
    phi: np.ndarray
    origin: np.ndarray
    pitch: float
    seal: int = 1
    leak: float | None = None

    @property
    def shape(self) -> tuple:
        return self.phi.shape

    def points(self) -> np.ndarray:
        """World position of every sample, shaped (nx, ny, nz, 3)."""
        grids = np.meshgrid(*(np.arange(n) for n in self.phi.shape), indexing="ij")
        idx = np.stack(grids, axis=-1)
        return self.origin + (idx + 0.5) * self.pitch


def _edt_1d(f: np.ndarray) -> np.ndarray:
    """Felzenszwalb's exact squared distance transform along the last axis.

    Vectorised over every other axis, so a 128^3 grid costs 3 x 128 numpy
    operations rather than 3 x 128^3 Python ones. The inner `while` pops
    parabolas off the lower envelope and runs a couple of times per column.
    """
    n = f.shape[-1]
    lead = f.shape[:-1]
    idx = np.arange(n, dtype=np.float64)

    k = np.zeros(lead, dtype=np.int64)
    v = np.zeros(lead + (n,), dtype=np.int64)
    z = np.empty(lead + (n + 1,), dtype=np.float64)
    z[..., 0] = -np.inf
    z[..., 1] = np.inf

    def gather(arr, at):
        return np.take_along_axis(arr, at[..., None], axis=-1)[..., 0]

    for q in range(1, n):
        fq = f[..., q]
        s = np.zeros(lead, dtype=np.float64)
        while True:
            vk = gather(v, k)
            fvk = gather(f, vk)
            s = ((fq + q * q) - (fvk + vk * vk)) / (2.0 * (q - vk))
            pop = (k > 0) & (s <= gather(z, k))
            if not pop.any():
                break
            k = np.where(pop, k - 1, k)
        k = k + 1
        np.put_along_axis(v, k[..., None], q, axis=-1)
        np.put_along_axis(z, k[..., None], s[..., None], axis=-1)
        np.put_along_axis(z, (k + 1)[..., None], np.inf, axis=-1)

    out = np.empty_like(f)
    k = np.zeros(lead, dtype=np.int64)
    for q in range(n):
        while True:
            ahead = gather(z, k + 1) < q
            if not ahead.any():
                break
            k = np.where(ahead, k + 1, k)
        vk = gather(v, k)
        out[..., q] = (q - idx[vk]) ** 2 + gather(f, vk)
    return out


def _edt(mask: np.ndarray) -> np.ndarray:
    """Exact Euclidean distance, in voxels, from every cell to the nearest True.

    A cell that is itself True gets 0. Seeded with a large finite value rather
    than inf because the parabola arithmetic subtracts two of them.
    """
    big = float(sum(n * n for n in mask.shape)) * 4.0
    f = np.where(mask, 0.0, big)
    for axis in range(f.ndim):
        f = np.moveaxis(_edt_1d(np.moveaxis(f, axis, -1)), -1, axis)
    return np.sqrt(f)


def _sweep(reach: np.ndarray, free: np.ndarray, axis: int, reverse: bool) -> np.ndarray:
    """Propagate `reach` along one axis in one direction, blocked by non-`free`.

    A whole row is settled per call: `maximum.accumulate` carries the index of
    the last seed and the index of the last blocker forward together, and a cell
    is reached when the seed is the more recent of the two. Iterating this to a
    fixed point is an exact flood fill for a fraction of the cost of dilating the
    grid one voxel at a time.
    """
    sl = [slice(None)] * reach.ndim
    if reverse:
        sl[axis] = slice(None, None, -1)
    sl = tuple(sl)
    r, f = reach[sl], free[sl]

    shape = [1] * reach.ndim
    shape[axis] = reach.shape[axis]
    idx = np.broadcast_to(np.arange(reach.shape[axis]).reshape(shape), reach.shape)

    blocked = np.maximum.accumulate(np.where(~f, idx, -1), axis=axis)
    seeded = np.maximum.accumulate(np.where(r, idx, -1), axis=axis)
    return (r | (f & (seeded > blocked)))[sl]


def _dilate(mask: np.ndarray, rounds: int = 1) -> np.ndarray:
    """Grow a mask by one face-connected voxel per round."""
    for _ in range(rounds):
        grown = mask.copy()
        for axis in range(mask.ndim):
            lo = [slice(None)] * mask.ndim
            hi = [slice(None)] * mask.ndim
            lo[axis], hi[axis] = slice(0, -1), slice(1, None)
            grown[tuple(lo)] |= mask[tuple(hi)]
            grown[tuple(hi)] |= mask[tuple(lo)]
        mask = grown
    return mask


def _flood_outside(occupied: np.ndarray, max_sweeps: int = 64) -> np.ndarray:
    """Everything reachable from the border without crossing an occupied voxel.

    This is the step that decides what "inside" means for a mesh that has no
    opinion on the matter. It never asks whether the surface is closed; it asks
    whether the *rasterisation* is closed, and at a sensible pitch that is true
    of meshes with thousands of boundary edges.
    """
    free = ~occupied
    reach = np.zeros_like(free)
    for axis in range(free.ndim):
        for end in (0, -1):
            sl = [slice(None)] * free.ndim
            sl[axis] = end
            reach[tuple(sl)] = free[tuple(sl)]

    for _ in range(max_sweeps):
        before = int(reach.sum())
        for axis in range(free.ndim):
            for reverse in (False, True):
                reach = _sweep(reach, free, axis, reverse)
        if int(reach.sum()) == before:
            return reach
    log.warning("flood fill did not settle in %d sweeps", max_sweeps)
    return reach


def _occupancy(mesh: trimesh.Trimesh, origin: np.ndarray, pitch: float,
               shape: tuple) -> np.ndarray:
    """Rasterise the surface into the grid.

    The mesh is subdivided until no edge is longer than half a voxel, so
    consecutive samples on a triangle land in the same cell or a neighbouring
    one and the rasterised skin has no gaps for the flood fill to leak through.
    Nothing about this cares whether the mesh is watertight, which is the whole
    reason the voxel route exists.
    """
    verts, _ = trimesh.remesh.subdivide_to_size(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces),
        max_edge=pitch * 0.5,
    )
    idx = np.floor((verts - origin) / pitch).astype(np.int64)
    np.clip(idx, 0, np.array(shape) - 1, out=idx)
    grid = np.zeros(shape, dtype=bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return grid


def _solid_mask(occupied: np.ndarray, seal: int) -> np.ndarray:
    """Which voxels are material: the rasterised skin plus everything it encloses.

    The flood fill is face-connected, and a rasterised triangle is only
    *corner*-connected: two voxels meeting at a corner leave a face-connected
    gap that the fill pours through, which is why an early version found 951
    interior voxels in a fuselage that should have had forty thousand. Fatten
    the skin before flooding and the diagonal gaps close; hand the voxel back
    afterwards, everywhere it was not real surface, and the surface lands where
    it started. `seal` above 1 also bridges real cracks in the mesh — and those
    are the norm, not the exception, on decimated generated output.
    """
    fat = _dilate(occupied, seal)
    outside = _flood_outside(fat)
    outside = _dilate(outside, seal) & ~occupied
    return ~outside


def _block_any(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by OR-ing each factor^3 block — a coarse rasterisation."""
    pads = [(0, (-n) % factor) for n in mask.shape]
    padded = np.pad(mask, pads)
    blocks = padded.reshape(padded.shape[0] // factor, factor,
                            padded.shape[1] // factor, factor,
                            padded.shape[2] // factor, factor)
    return blocks.any(axis=(1, 3, 5))


def _referee(occupied: np.ndarray, factor: int) -> np.ndarray | None:
    """Voxels that are certainly interior, decided on a `factor`-times coarser grid.

    A coarse voxel that the coarse fill calls interior *and* that contains no
    surface at all is unambiguously inside the mesh, whatever the fine grid
    thinks — coarsening is exactly what makes a crack narrower than a voxel. So
    the coarse pass gets to referee the fine one.

    Comparing volumes instead does not work: a coarse grid over-reports volume
    by about half a voxel of surface, and that inflation is the same size as the
    leak being looked for.
    """
    coarse_occ = _block_any(occupied, factor)
    core = _solid_mask(coarse_occ, 1) & ~coarse_occ
    # Eroded by one coarse voxel, because a coarse rasterisation is fat: a cell
    # just *outside* the surface can end up walled in by the coarse skin and
    # called interior. Those cells are the referee's own error, and without this
    # they read as a 2-3% leak on a mesh that is provably watertight.
    core = ~_dilate(~core, 1)
    if not core.any():
        return None
    fine = np.kron(core, np.ones((factor,) * 3, dtype=bool))
    fine = fine[:occupied.shape[0], :occupied.shape[1], :occupied.shape[2]] & ~occupied
    return fine if fine.any() else None


def _leak_fraction(certain: np.ndarray | None, solid: np.ndarray) -> float:
    """Share of the certain interior that this fill wrongly called outside."""
    if certain is None:
        return 0.0
    return float((certain & ~solid).sum()) / float(certain.sum())


# Escalating dilation radii. A watertight mesh is done at 1; a decimated TRELLIS
# 2 part with cracks 5% of its own length needs 6 or more, and by then the grid
# is telling you to use a coarser one.
_SEAL_LADDER = (1, 2, 3, 4, 6, 8)

# Two percent of the certain interior lost is rounding at the coarse boundary;
# a real leak takes tens of percent.
LEAK_TOLERANCE = 0.02


def _seal_until_sound(occupied: np.ndarray, seal, resolution: int):
    """Raise the seal until the coarse referee agrees the fill did not leak."""
    if seal != "auto":
        return _solid_mask(occupied, int(seal)), int(seal), None

    certain = _referee(occupied, max(2, int(round(resolution / 32))))
    worst = 1.0
    for radius in _SEAL_LADDER:
        solid = _solid_mask(occupied, radius)
        leak = _leak_fraction(certain, solid)
        if leak <= LEAK_TOLERANCE:
            if radius > 1:
                log.info("sealed cracks with a %d-voxel dilation", radius)
            return solid, radius, round(leak, 4)
        worst = leak

    raise ValueError(
        f"the fill leaks through this mesh at resolution {resolution}: even a "
        f"{_SEAL_LADDER[-1]}-voxel seal loses {worst:.0%} of the interior. The "
        f"cracks in it are wider than {_SEAL_LADDER[-1]} voxels, so drop to "
        f"resolution {max(16, resolution // 2)} — a coarser voxel is what makes "
        f"a crack sub-voxel — or repair the mesh first"
    )


def sdf(mesh: trimesh.Trimesh, resolution: int = DEFAULT_RESOLUTION,
        pad: int = 3, seal: int | str = "auto") -> Field:
    """Signed distance field of `mesh`, negative inside.

    `resolution` is voxels along the longest extent. `pad` is empty voxels kept
    around the mesh so the outside is always reachable from the grid border and
    so the isosurface never runs into the wall of the array. `seal` is how many
    voxels of crack the rasterisation may bridge; "auto" raises it until the
    coarse referee stops finding a leak.
    """
    if resolution < 8:
        raise ValueError(f"resolution {resolution} is too coarse to resolve a wall")
    if resolution > MAX_RESOLUTION:
        raise ValueError(
            f"resolution {resolution} is above the {MAX_RESOLUTION} cap; the "
            f"grid is cubic in this number"
        )

    lo, hi = np.asarray(mesh.bounds, dtype=np.float64)
    pitch = float((hi - lo).max()) / float(resolution)
    if pitch <= _EPS:
        raise ValueError("mesh has no extent")

    origin = lo - pitch * pad
    shape = tuple(int(math.ceil((hi[i] - lo[i]) / pitch)) + 2 * pad for i in range(3))

    occupied = _occupancy(mesh, origin, pitch, shape)
    solid, seal, leak = _seal_until_sound(occupied, seal, resolution)

    # Distance out of the solid minus distance into it. Both terms are zero on
    # their own side, so exactly one of them is nonzero everywhere and the sign
    # is never ambiguous.
    #
    # The half-voxel is not cosmetic. A distance transform measures centre to
    # centre, so the first solid voxel reports a full pitch when the surface is
    # really half a pitch away, and every isosurface below zero comes out half a
    # voxel shallow — a 0.05 wall measured 0.039 before this line existed. The
    # zero crossing itself is unaffected, because both sides shift together.
    #
    # The second half-voxel is a de-biasing. A rasterised skin voxel is the one
    # the surface passes *through*, and the fill stops at its far face, so the
    # zero level lands up to a full voxel outside the real surface — a 1.0 box
    # measured 1.05. Its expected position is the middle of that voxel, so
    # pushing the whole field in by half a pitch turns a one-sided error into a
    # symmetric one. Both surfaces move together, so the wall is unaffected.
    raw = _edt(solid) - _edt(~solid)
    phi = (raw - 0.5 * np.sign(raw) + 0.5) * pitch
    return Field(phi=phi, origin=origin, pitch=pitch, seal=seal, leak=leak)


# --- isosurface --------------------------------------------------------------

# The twelve edges of a cell, as pairs of corner indices, where a corner is
# numbered by its (dx, dy, dz) bits.
_CORNERS = np.array([(i >> 2 & 1, i >> 1 & 1, i & 1) for i in range(8)])
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if bin(a ^ b).count("1") == 1]


def surface_net(field: Field, level: float = 0.0) -> trimesh.Trimesh:
    """Extract the `level` isosurface as a naive surface net.

    One vertex per cell that straddles the level, placed at the mean of the
    edge crossings, and one quad per straddling grid edge. Chosen over marching
    cubes because it is twenty lines of numpy instead of a 256-entry table, and
    because it is *manifold by construction*: every quad edge is shared by
    exactly two quads, so the output is watertight even though the input mesh
    was not. That is the single most useful property this module has.
    """
    phi = field.phi - level
    vals = np.stack([phi[dx:phi.shape[0] - 1 + dx,
                         dy:phi.shape[1] - 1 + dy,
                         dz:phi.shape[2] - 1 + dz] for dx, dy, dz in _CORNERS])
    inside = vals < 0
    active = inside.any(axis=0) & ~inside.all(axis=0)
    if not active.any():
        raise ValueError(
            f"nothing crosses the {level} isosurface — the wall is thicker than "
            f"the part, so there is no interior to open up"
        )

    accum = np.zeros((3,) + active.shape)
    count = np.zeros(active.shape)
    for a, b in _EDGES:
        va, vb = vals[a], vals[b]
        crossing = (va < 0) != (vb < 0)
        if not crossing.any():
            continue
        denom = np.where(np.abs(va - vb) < _EPS, _EPS, va - vb)
        t = np.clip(va / denom, 0.0, 1.0)
        delta = _CORNERS[b] - _CORNERS[a]
        for axis in range(3):
            offset = _CORNERS[a][axis] + t * delta[axis]
            accum[axis] += np.where(crossing, offset, 0.0)
        count += crossing

    cell_index = np.full(active.shape, -1, dtype=np.int64)
    cell_index[active] = np.arange(int(active.sum()))
    cells = np.stack(np.nonzero(active), axis=-1)
    offsets = accum[:, active].T / count[active][:, None]
    vertices = field.origin + (cells + offsets + 0.5) * field.pitch

    faces = []
    for axis in range(3):
        faces.extend(_quads_along(phi, cell_index, axis))
    if not faces:
        raise ValueError("isosurface produced no faces")

    mesh = trimesh.Trimesh(vertices=vertices,
                           faces=np.concatenate(faces, axis=0), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _quads_along(phi: np.ndarray, cell_index: np.ndarray, axis: int) -> list:
    """Two triangles per grid edge along `axis` whose ends straddle the level.

    The four cells sharing that edge each already carry a vertex; joining them
    in order is what makes the result closed. Winding follows the sign change,
    so the normal points out of the solid without a repair pass afterwards.
    """
    u, w = (axis + 1) % 3, (axis + 2) % 3

    def sl(a_from, a_to, u_from, u_to, w_from, w_to):
        s = [None] * 3
        s[axis] = slice(a_from, a_to)
        s[u] = slice(u_from, u_to)
        s[w] = slice(w_from, w_to)
        return tuple(s)

    # Grid edges whose four surrounding cells all exist: interior in u and w.
    n_u, n_w = phi.shape[u], phi.shape[w]
    lo = sl(0, phi.shape[axis] - 1, 1, n_u - 1, 1, n_w - 1)
    hi = sl(1, phi.shape[axis], 1, n_u - 1, 1, n_w - 1)
    a, b = phi[lo], phi[hi]
    crossing = (a < 0) != (b < 0)
    if not crossing.any():
        return []

    def cells(du, dw):
        s = [None] * 3
        s[axis] = slice(0, phi.shape[axis] - 1)
        s[u] = slice(1 - du, n_u - 1 - du)
        s[w] = slice(1 - dw, n_w - 1 - dw)
        return cell_index[tuple(s)][crossing]

    c00, c10, c11, c01 = cells(0, 0), cells(1, 0), cells(1, 1), cells(0, 1)
    quad = np.stack([c00, c10, c11, c01], axis=-1)
    # Flip when the solid is on the far side, so every normal points outward.
    flip = (a >= 0)[crossing]
    quad[flip] = quad[flip][:, ::-1]
    return [np.stack([quad[:, 0], quad[:, 1], quad[:, 2]], axis=-1),
            np.stack([quad[:, 0], quad[:, 2], quad[:, 3]], axis=-1)]


# --- openings ----------------------------------------------------------------

# A cut is expressed in the same vocabulary /assemble uses for placement, so
# "on the +Z face, a third of the way up" reads the same here as it does there.
FACES = {
    "front": ("z", 1.0), "back": ("z", 0.0),
    "right": ("x", 1.0), "left": ("x", 0.0),
    "top": ("y", 1.0), "bottom": ("y", 0.0),
}

SHAPES = ("box", "cylinder")


def _at_fractions(at) -> dict:
    """Validate an `at` mapping into {axis: fraction}, defaulting to centred.

    Deliberately the same words `assemble.FRACTIONS` accepts — min/center/max,
    top/bottom/left/right, or a bare number — because a caller who has placed a
    part already knows this vocabulary and should not have to learn a second one.
    """
    out = {}
    for key, value in (at or {}).items():
        axis = str(key).strip().lower()
        if axis not in _AXIS:
            raise ValueError(f"opening.at has axis {key!r}; expected x, y or z")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            name = str(value).strip().lower()
            if name not in assemble.FRACTIONS:
                raise ValueError(
                    f"opening.at.{axis} is {value!r}; expected a number where 0 "
                    f"is the low face and 1 the high face, or one of "
                    f"{sorted(set(assemble.FRACTIONS))}"
                )
            value = assemble.FRACTIONS[name]
        out[axis] = float(value)
    return out


def _resolve_opening(spec: dict, bounds: tuple) -> dict:
    """Turn one opening request into an axis, a centre point and a size."""
    if not isinstance(spec, dict):
        raise ValueError(f"opening must be an object, got {spec!r}")

    unknown = sorted(set(spec) - {"shape", "size", "radius", "at", "face",
                                  "axis", "depth", "through"})
    if unknown:
        raise ValueError(
            f"unknown opening key(s) {unknown}; expected any of "
            f"['at', 'axis', 'depth', 'face', 'radius', 'shape', 'size', 'through']"
        )

    shape = spec.get("shape", "box")
    if shape not in SHAPES:
        raise ValueError(f"opening.shape must be one of {list(SHAPES)}, got {shape!r}")

    at = _at_fractions(spec.get("at"))
    face = spec.get("face")
    if face is not None:
        key = str(face).strip().lower().lstrip("+")
        if key in FACES:
            axis_name, frac = FACES[key]
        elif key.lstrip("-") in _AXIS and len(key.lstrip("-")) == 1:
            axis_name, frac = key.lstrip("-"), (0.0 if key.startswith("-") else 1.0)
        else:
            raise ValueError(
                f"opening.face is {face!r}; expected one of {sorted(FACES)} or "
                f"a signed axis like '+z'"
            )
        at.setdefault(axis_name, frac)
    else:
        axis_name = None

    if spec.get("axis"):
        axis_name = str(spec["axis"]).strip().lower()
        if axis_name not in _AXIS:
            raise ValueError(f"opening.axis is {spec['axis']!r}; expected x, y or z")
    elif axis_name is None:
        # Nobody said which way the hole goes, so take the axis they pushed
        # furthest off centre: {"z": "max"} plainly means "through the +Z face".
        if not at:
            raise ValueError(
                "opening needs `face`, `axis`, or an `at` that names an axis, "
                "so it is clear which way the hole is cut"
            )
        axis_name = max(at, key=lambda a: abs(at[a] - 0.5))

    lo, hi = bounds
    axis = _AXIS[axis_name]
    size = np.asarray(hi) - np.asarray(lo)
    centre = np.asarray(lo) + np.array([at.get(a, 0.5) for a in assemble.AXES]) * size

    if shape == "cylinder":
        radius = spec.get("radius")
        bad = (isinstance(radius, bool) or not isinstance(radius, (int, float))
               or radius <= 0)
        if bad:
            raise ValueError("a cylinder opening needs a positive `radius`")
        extent = (float(radius), float(radius))
    else:
        given = spec.get("size")
        if given is None:
            raise ValueError("a box opening needs `size`, either [u, w] or [x, y, z]")
        given = [float(v) for v in given]
        if len(given) == 3:
            given = [given[i] for i in range(3) if i != axis]
        if len(given) != 2 or min(given) <= 0:
            raise ValueError(
                f"opening.size must be two positive numbers across the aperture "
                f"(or three, one per axis), got {spec['size']!r}"
            )
        extent = (given[0] / 2.0, given[1] / 2.0)

    return {
        "shape": shape,
        "axis": axis,
        "axis_name": axis_name,
        "entry": at.get(axis_name, 1.0) >= 0.5,
        "centre": centre,
        "extent": extent,
        "depth": spec.get("depth"),
        "through": bool(spec.get("through", False)),
    }


def _cut_field(field: Field, cut: dict, wall: float) -> np.ndarray:
    """A signed field for one aperture: negative inside the material to remove.

    The cross-section is analytic, but the *depth* is measured off the mesh: the
    cutter starts wherever the surface actually is under the aperture, taken as
    a low percentile of the first solid hit in each column, rather than at the
    bounding box. Otherwise a door on a curved hull either floats outside it or
    tunnels all the way through, depending on how round that spot happens to be.
    """
    axis = cut["axis"]
    u, w = (axis + 1) % 3, (axis + 2) % 3

    grids = np.meshgrid(*(np.arange(n) for n in field.shape), indexing="ij")
    coord = [field.origin[i] + (grids[i] + 0.5) * field.pitch for i in range(3)]

    du = coord[u] - cut["centre"][u]
    dw = coord[w] - cut["centre"][w]
    if cut["shape"] == "cylinder":
        profile = np.hypot(du, dw) - cut["extent"][0]
    else:
        profile = np.maximum(np.abs(du) - cut["extent"][0],
                             np.abs(dw) - cut["extent"][1])

    if cut["through"]:
        return profile

    inside_profile = profile < 0
    solid = field.phi < 0
    hits = solid & inside_profile
    if not hits.any():
        raise ValueError(
            "the opening does not touch the part — check `at`, or widen `size`"
        )

    # Scan from the face the aperture was placed on. argmax finds the first True
    # along the axis, and a low percentile over the aperture's footprint gives a
    # flat cut plane that clears the nearest part of the surface.
    scan = np.flip(hits, axis=axis) if cut["entry"] else hits
    first = np.argmax(scan, axis=axis)
    columns = hits.any(axis=axis)
    index = first[columns].astype(float)
    n = field.shape[axis]
    if cut["entry"]:
        depth_index = n - 1 - np.percentile(index, 10)
        surface = field.origin[axis] + (depth_index + 0.5) * field.pitch
    else:
        surface = field.origin[axis] + (np.percentile(index, 10) + 0.5) * field.pitch

    depth = cut["depth"]
    if depth is None:
        # Enough to breach the wall wherever it sits, plus slack for the
        # curvature the flat cut plane does not follow.
        depth = wall * 3.0 + field.pitch * 2.0
    depth = float(depth)
    if depth <= 0:
        raise ValueError(f"opening.depth must be positive, got {depth}")

    a = coord[axis]
    if cut["entry"]:
        near, far = surface - depth, a.max() + field.pitch
        slab = np.maximum(near - a, a - far)
    else:
        near, far = a.min() - field.pitch, surface + depth
        slab = np.maximum(near - a, a - far)
    return np.maximum(profile, slab)


# --- hollowing ---------------------------------------------------------------

@dataclass
class Hollowed:
    mesh: trimesh.Trimesh
    report: dict = _field(default_factory=dict)


def hollow(mesh: trimesh.Trimesh,
           wall_thickness: float = DEFAULT_WALL_THICKNESS,
           openings=(),
           resolution: int = DEFAULT_RESOLUTION,
           max_faces: int | None = ROBLOX_MAX_TRIANGLES,
           seal: int | str = "auto") -> Hollowed:
    """Turn a solid mesh into a shell of `wall_thickness`, with optional holes.

    The shell is the region between the surface and the surface offset inward,
    which as a distance field is just `max(phi, -wall - phi)` — a CSG difference
    that costs one array operation and cannot fail on bad topology, because by
    that point there is no topology left, only numbers on a grid.

    Openings are subtracted the same way. Each is `{"face": "front", "shape":
    "box", "size": [w, h]}` or the `at`/`axis` spelling; see `_resolve_opening`.
    """
    wall = float(wall_thickness)
    if wall <= 0:
        raise ValueError(f"wall_thickness must be positive, got {wall_thickness}")

    t0 = time.time()
    field = sdf(mesh, resolution=resolution, seal=seal)
    voxels_per_wall = wall / field.pitch
    if voxels_per_wall < MIN_WALL_VOXELS:
        raise ValueError(
            f"a {wall} wall is {voxels_per_wall:.1f} voxels at resolution "
            f"{resolution}; raise resolution to at least "
            f"{int(math.ceil(resolution * MIN_WALL_VOXELS / voxels_per_wall))} "
            f"or thicken the wall"
        )

    interior = field.phi + wall  # negative where the cavity goes
    if not (interior < 0).any():
        raise ValueError(
            f"a {wall} wall leaves no cavity — the part is thinner than two "
            f"walls everywhere. Thin parts like a wing stay solid, correctly."
        )

    shell = np.maximum(field.phi, -interior)
    cuts = [_resolve_opening(spec, mesh.bounds) for spec in (openings or ())]
    for cut in cuts:
        shell = np.maximum(shell, -_cut_field(field, cut, wall))

    out = surface_net(Field(shell, field.origin, field.pitch))
    faces_before = len(out.faces)
    watertight_before = bool(out.is_watertight)
    decimated = False
    if max_faces and faces_before > max_faces:
        out = out.simplify_quadric_decimation(face_count=int(max_faces))
        decimated = True

    elapsed = time.time() - t0
    solid_volume = float((field.phi < 0).sum()) * field.pitch ** 3
    cavity_volume = float((interior < 0).sum()) * field.pitch ** 3

    report = {
        "method": "voxel_sdf",
        "wall_thickness": wall,
        "resolution": resolution,
        "pitch": round(field.pitch, 6),
        "voxels_per_wall": round(voxels_per_wall, 2),
        "grid": list(field.shape),
        # How many voxels of crack had to be bridged to find an inside at all.
        # 1 means the mesh was sound; anything higher is a measurement of how
        # broken the input was, and of how far the skin moved near a crack.
        "seal": field.seal,
        "leak": field.leak,
        "faces": int(len(out.faces)),
        "faces_before_decimation": int(faces_before),
        "decimated": decimated,
        "watertight": bool(out.is_watertight),
        # The isosurface is closed by construction. Quadric decimation is what
        # ends that, exactly as docs/DECIMATION.md records for generated meshes;
        # engines do not care, and Roblox imports it either way.
        "watertight_before_decimation": watertight_before,
        "winding_consistent": bool(out.is_winding_consistent),
        **topology(out),
        "openings": len(cuts),
        # How far the outer skin moved. Voxelisation alone costs about a
        # percent; more than that is the seal bulging the surface over a crack,
        # and it is the number that says the input was too broken for this grid.
        "size_error": [round(float(v), 4) for v in
                       (out.extents - mesh.extents) / mesh.extents],
        "solid_volume": round(solid_volume, 6),
        "cavity_volume": round(cavity_volume, 6),
        "material_saved": (round(cavity_volume / solid_volume, 4)
                           if solid_volume else 0.0),
        "seconds": round(elapsed, 3),
    }
    log.info("hollowed to %d faces in %.2fs (wall %.3f, %d openings)",
             len(out.faces), elapsed, wall, len(cuts))
    return Hollowed(mesh=out, report=report)


def hollow_boolean(mesh: trimesh.Trimesh,
                   wall_thickness: float = DEFAULT_WALL_THICKNESS,
                   max_faces: int | None = ROBLOX_MAX_TRIANGLES) -> Hollowed:
    """The textbook route, kept for comparison: offset inward and subtract.

    Requires `manifold3d` and a mesh an exact boolean engine will accept. On
    generated input it usually raises, which is the finding rather than a bug —
    see docs/HOLLOW.md. The inward offset is per-vertex along the vertex normal,
    which is exact on a convex surface and self-intersects anywhere the local
    curvature radius is smaller than the wall.
    """
    try:
        import manifold3d  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "the boolean route needs manifold3d (Apache-2.0); the default "
            "voxel route needs nothing extra"
        ) from exc

    t0 = time.time()
    outer = mesh.copy()
    outer.merge_vertices(merge_tex=True, merge_norm=True)
    trimesh.repair.fill_holes(outer)
    trimesh.repair.fix_normals(outer)
    if not outer.is_watertight:
        raise ValueError(
            f"mesh is still not watertight after hole filling "
            f"({len(trimesh.grouping.group_rows(outer.edges_sorted, require_count=1))} "
            f"boundary edges) — an exact boolean has nothing to subtract from"
        )

    inner = outer.copy()
    inner.vertices = inner.vertices - inner.vertex_normals * float(wall_thickness)

    result = trimesh.boolean.difference([outer, inner], engine="manifold")
    if max_faces and len(result.faces) > max_faces:
        result = result.simplify_quadric_decimation(face_count=int(max_faces))

    return Hollowed(mesh=result, report={
        "method": "boolean",
        "wall_thickness": float(wall_thickness),
        "faces": int(len(result.faces)),
        "watertight": bool(result.is_watertight),
        "seconds": round(time.time() - t0, 3),
    })


# --- proving it is actually hollow -------------------------------------------

def ray_crossings(mesh: trimesh.Trimesh, origin, direction,
                  signed: bool = False) -> np.ndarray:
    """Distances along a ray at which it crosses the surface, sorted.

    The one measurement that distinguishes a hollow object from a solid one
    without opening it: a solid gives two crossings, a shell gives four — in,
    into the cavity, out of the cavity, out. Written out with Moller-Trumbore
    because trimesh's own intersector wants `rtree`, which is not installed
    here and is another wheel to justify.

    With `signed`, returns (distance, entering) pairs — entering meaning the
    surface faced the ray — which is what tells a wall from a cavity.
    """
    origin = np.asarray(origin, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / np.linalg.norm(direction)

    tri = np.asarray(mesh.triangles, dtype=np.float64)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    if not ok.any():
        return np.zeros(0)

    inv = 1.0 / det[ok]
    tvec = origin - tri[ok, 0]
    u = np.einsum("ij,ij->i", tvec, p[ok]) * inv
    q = np.cross(tvec, e1[ok])
    v = q @ direction * inv
    t = np.einsum("ij,ij->i", e2[ok], q) * inv

    hit = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    distance = t[hit]
    order = np.argsort(distance)
    distance = distance[order]

    keep = np.ones(len(distance), dtype=bool)
    if len(distance) > 1:
        # A ray through a shared edge is reported once per triangle; two
        # crossings a nanometre apart are one crossing.
        keep[1:] = np.diff(distance) > 1e-7
    if not signed:
        return distance[keep]

    normals = np.asarray(mesh.face_normals)[ok][hit][order]
    entering = (normals @ direction) < 0
    return np.stack([distance[keep], entering[keep]], axis=-1)


def topology(mesh: trimesh.Trimesh) -> dict:
    """Edges with the wrong number of faces on them.

    `is_watertight` collapses two different failures into one bool, and they
    matter differently: a *boundary* edge is a hole, which an engine will shade
    wrong; a *non-manifold* edge is two surface sheets meeting in one voxel,
    which nothing downstream of here notices but a boolean engine refuses.
    """
    _, counts = np.unique(np.asarray(mesh.edges_sorted), axis=0,
                          return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
    }


def _segments(hits: np.ndarray) -> tuple[list, list]:
    """Split a signed hit list into (material runs, cavity runs).

    Walking entry/exit rather than pairing crossings blindly is what makes this
    work on a shape with more than one lobe: a ray down the length of a
    fuselage may cross a dozen surfaces, and only the entry-to-exit runs are
    wall.
    """
    material, cavity, depth, start = [], [], 0, None
    for distance, entering in hits:
        if entering:
            if depth == 0:
                if start is not None:
                    cavity.append(distance - start)
                start = distance
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0:
                material.append(distance - start)
                start = distance
    return material, cavity


def measure_wall(mesh: trimesh.Trimesh, samples: int = 64,
                 axis: int = 0, seed: int = 0, minimum: float = 0.0) -> dict:
    """Measure the real wall thickness by firing rays through the part.

    Reports what was built rather than what was asked for, which is the only
    honest way to state the accuracy of a voxel-quantised offset. A ray that
    finds two material runs with a gap between them has found a hollow; one
    that finds a single run went through something that stayed solid, which is
    the correct outcome for anything thinner than two walls.
    """
    lo, hi = mesh.bounds
    u, w = (axis + 1) % 3, (axis + 2) % 3
    rng = np.random.default_rng(seed)

    direction = np.zeros(3)
    direction[axis] = 1.0
    span = hi[axis] - lo[axis]

    walls, hollow_rays, solid_rays, cavities = [], 0, 0, []
    for _ in range(samples):
        origin = np.array(lo, dtype=float)
        origin[axis] -= span
        # Stay off the silhouette, where a grazing ray reports nonsense.
        origin[u] = lo[u] + (0.2 + 0.6 * rng.random()) * (hi[u] - lo[u])
        origin[w] = lo[w] + (0.2 + 0.6 * rng.random()) * (hi[w] - lo[w])

        material, cavity = _segments(ray_crossings(mesh, origin, direction,
                                                   signed=True))
        material = [m for m in material if m > minimum]
        cavity = [c for c in cavity if c > minimum]
        if len(material) >= 2 and cavity:
            hollow_rays += 1
            walls.extend(material)
            cavities.extend(cavity)
        elif material:
            solid_rays += 1

    def stat(values, fn):
        return round(float(fn(values)), 4) if values else None

    return {
        "rays": samples,
        "hollow_rays": hollow_rays,
        "solid_rays": solid_rays,
        "walls_measured": len(walls),
        "wall_median": stat(walls, np.median),
        "wall_mean": stat(walls, np.mean),
        "wall_min": stat(walls, np.min),
        "wall_max": stat(walls, np.max),
        "wall_p10": stat(walls, lambda v: np.percentile(v, 10)),
        "wall_p90": stat(walls, lambda v: np.percentile(v, 90)),
        "cavity_median": stat(cavities, np.median),
    }


def cross_section(mesh: trimesh.Trimesh, axis: int = 0,
                  fraction: float = 0.5) -> trimesh.Trimesh:
    """Half the mesh, cut on an axis plane — the only way to *see* an interior.

    A hollow object photographed from outside is pixel-identical to a solid one,
    so every render in docs/HOLLOW.md is of one of these.

    Clipped here rather than with `trimesh.slice_plane`, which reaches for
    `scipy.spatial.cKDTree` and so does not run in this project's environment at
    all. Straddling triangles are cut with Sutherland-Hodgman; the cut face is
    left open, which is what makes the wall section visible.
    """
    lo, hi = mesh.bounds
    plane = lo[axis] + float(fraction) * (hi[axis] - lo[axis])
    verts = np.asarray(mesh.vertices, dtype=float)
    distance = verts[:, axis] - plane

    face_signs = distance[mesh.faces] <= 0
    inside_count = face_signs.sum(axis=1)
    if not inside_count.any():
        raise ValueError("the cutting plane missed the mesh")

    out_verts = list(verts)
    out_faces = [f for f in mesh.faces[inside_count == 3]]
    for face in mesh.faces[(inside_count > 0) & (inside_count < 3)]:
        polygon = []
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            if distance[a] <= 0:
                polygon.append(int(a))
            if (distance[a] <= 0) != (distance[b] <= 0):
                t = distance[a] / (distance[a] - distance[b])
                out_verts.append(verts[a] + t * (verts[b] - verts[a]))
                polygon.append(len(out_verts) - 1)
        out_faces.extend((polygon[0], polygon[i], polygon[i + 1])
                         for i in range(1, len(polygon) - 1))

    return trimesh.Trimesh(vertices=np.asarray(out_verts),
                           faces=np.asarray(out_faces, dtype=np.int64),
                           process=False)


# --- hollow by construction --------------------------------------------------
#
# Carving is the hard road. A room, a bin or a silo is *known* to be hollow, and
# building it that way gives exact walls, a few hundred triangles and no
# resampling of the surface. These are written in primitives.py's vocabulary —
# `_box` for chamfered slabs, `_revolve` for round shells, `_combine` to merge
# closed solids without welding them — so they can be moved into that file
# verbatim; see the note at the end of docs/HOLLOW.md.

_box = primitives._box
_revolve = primitives._revolve
_prism = primitives._prism
_combine = primitives._combine
_center = primitives._center
_studs = primitives._studs
_count = primitives._count
Param = primitives.Param
Kind = primitives.Kind
CHAMFER = primitives.CHAMFER
SECTIONS = primitives.SECTIONS


def _pierced_wall(width, height, thickness, z, y_centre, opening_width,
                  opening_height, sill, chamfer, what: str) -> list:
    """A Z-facing wall built as the slabs *around* an aperture, not as a cut.

    Same trick `primitives._wall_panel` uses: exact, boolean-free, and it leaves
    quads around the opening instead of the slivers a mesh boolean would.
    """
    above = height - sill - opening_height
    if above <= _EPS:
        raise ValueError(
            f"{what} {opening_height} high leaves no wall above it in a "
            f"{round(height, 4)}-high interior"
        )
    side = (width - opening_width) / 2
    if side <= _EPS:
        raise ValueError(
            f"{what} {opening_width} wide leaves no wall beside it in a "
            f"{round(width, 4)}-wide interior"
        )

    def slab(w, h, x, y):
        return _box(w, h, thickness, chamfer=chamfer,
                    center=(x, y_centre + y, z))

    parts = [slab(side, height, (width - side) / 2, 0.0),
             slab(side, height, -(width - side) / 2, 0.0),
             slab(opening_width, above, 0.0, height / 2 - above / 2)]
    if sill > _EPS:
        parts.append(slab(opening_width, sill, 0.0, -height / 2 + sill / 2))
    return parts


def _room(width, height, depth, wall_thickness, door, door_width, door_height,
          window, window_size, sill_height, ceiling, chamfer):
    """A building shell you can walk into: four walls, a floor, and a doorway.

    The walls overlap each other in the corners on purpose. That is what lets
    `_combine` merge them without a boolean — nothing coincides, so no edge ends
    up with four faces on it, and the result is watertight.
    """
    wall = wall_thickness
    if min(width, depth) <= 2 * wall or height <= 2 * wall:
        raise ValueError(
            f"wall_thickness {wall} leaves no interior in a {width}x{height}x"
            f"{depth} room"
        )

    parts = [_box(width, wall, depth, chamfer=chamfer,
                  center=(0.0, -height / 2 + wall / 2, 0.0))]
    if ceiling:
        parts.append(_box(width, wall, depth, chamfer=chamfer,
                          center=(0.0, height / 2 - wall / 2, 0.0)))

    # The side walls run the full depth; the end walls fit between them, so the
    # doorway's width is measured across the interior rather than the envelope.
    inner_h = height - wall - (wall if ceiling else 0.0)
    y_mid = -height / 2 + wall + inner_h / 2
    for sx in (1, -1):
        parts.append(_box(wall, inner_h, depth, chamfer=chamfer,
                          center=(sx * (width - wall) / 2, y_mid, 0.0)))

    end_width = width - 2 * wall
    ends = ((1, door, door_width, door_height, 0.0, "door"),
            (-1, window, window_size, window_size, sill_height, "window"))
    for sz, pierced, opening_w, opening_h, sill, what in ends:
        z = sz * (depth - wall) / 2
        if not pierced:
            parts.append(_box(end_width, inner_h, wall, chamfer=chamfer,
                              center=(0.0, y_mid, z)))
        else:
            parts += _pierced_wall(end_width, inner_h, wall, z, y_mid,
                                   opening_w, opening_h, sill, chamfer, what)
    return _combine(parts)


def _hollow_box(width, height, depth, wall_thickness, open_face, chamfer):
    """A container: a box with a wall thickness and one face left off.

    The crate you can open, the bin you can drop things in, the cargo hold. The
    open face is genuinely absent rather than cut away, so the mesh stays a
    closed solid with a cavity in it.
    """
    wall = wall_thickness
    if min(width, height, depth) <= 2 * wall:
        raise ValueError(
            f"wall_thickness {wall} leaves no interior in a {width}x{height}x"
            f"{depth} container"
        )

    faces = {"top": (1, 1), "bottom": (1, -1), "front": (2, 1), "back": (2, -1),
             "right": (0, 1), "left": (0, -1), "none": None}
    if open_face not in faces:
        raise ValueError(
            f"open_face must be one of {sorted(faces)}, got {open_face!r}")
    missing = faces[open_face]

    size = (width, height, depth)
    parts = []
    for axis in range(3):
        for sign in (1, -1):
            if missing == (axis, sign):
                continue
            extents = list(size)
            extents[axis] = wall
            # The slab spans the full box on the axes it is not normal to, so
            # neighbouring slabs overlap in the corners rather than meeting.
            centre = [0.0, 0.0, 0.0]
            centre[axis] = sign * (size[axis] - wall) / 2
            parts.append(_box(*extents, chamfer=chamfer, center=tuple(centre)))
    return _combine(parts)


def _hollow_cylinder(radius, height, wall_thickness, open_top, open_bottom,
                     chamfer, sections):
    """A silo, tank or fuselage section: a tube with optional end caps.

    One revolve rather than a boolean — the profile simply walks up the outside
    and back down the inside, which is how `primitives._cylinder` already makes
    a pipe. The caps close the ends of that profile instead.
    """
    wall = wall_thickness
    if wall >= radius:
        raise ValueError(f"wall_thickness {wall} is not smaller than radius {radius}")
    if not open_top and not open_bottom and height <= 2 * wall:
        raise ValueError(f"wall_thickness {wall} leaves no interior in a {height} tube")

    c = min(chamfer, wall * 0.45, height * 0.2)
    inner = radius - wall
    top, bottom = height / 2, -height / 2
    in_top, in_bottom = top - wall, bottom + wall

    # The profile is a closed (radius, height) loop revolved around +Y, so it
    # walks up the outside and back down the inside. Where a cap closes an end
    # the loop crosses the axis there instead; where both ends are capped the
    # cavity no longer touches the axis at all and cannot be one loop, so it
    # becomes a second, inverted shell inside the first.
    if open_top and open_bottom:
        profile = [(inner + c, bottom), (radius - c, bottom), (radius, bottom + c),
                   (radius, top - c), (radius - c, top), (inner + c, top),
                   (inner, top - c), (inner, bottom + c)]
    elif open_top:
        profile = [(0.0, bottom), (radius - c, bottom), (radius, bottom + c),
                   (radius, top - c), (radius - c, top), (inner + c, top),
                   (inner, top - c), (inner, in_bottom), (0.0, in_bottom)]
    elif open_bottom:
        profile = [(inner, bottom + c), (inner, in_top), (0.0, in_top),
                   (0.0, top), (radius - c, top), (radius, top - c),
                   (radius, bottom + c), (radius - c, bottom), (inner + c, bottom)]
    else:
        outer = _revolve([(0.0, bottom), (radius - c, bottom), (radius, bottom + c),
                          (radius, top - c), (radius - c, top), (0.0, top)], sections)
        cavity = _revolve([(0.0, in_bottom), (inner, in_bottom),
                           (inner, in_top), (0.0, in_top)], sections)
        # Normals have to point out of the *material*, which at a cavity wall
        # means into the void. Without this the volume comes out as the sum of
        # the two shells rather than the difference.
        cavity.invert()
        return _combine([outer, cavity])
    return _revolve(profile, sections)


def _arch(width, height, depth, thickness, rise, segments, chamfer):
    """A gateway: two piers carrying a segmented arch you can walk through.

    Each voussoir is an exact trapezoid between two radii and two angles, so the
    ring's outer corners land on the circle rather than bulging past it and the
    envelope is exactly the width and height asked for. `_prism` extrudes a
    convex polygon, and a trapezoid is convex, which is the whole reason this
    shape is buildable without a boolean.
    """
    if rise >= height:
        raise ValueError(f"rise {rise} must be less than height {height}")
    span = width - 2 * thickness
    if span <= _EPS:
        raise ValueError(f"thickness {thickness} leaves no opening in a {width} arch")

    pier_h = height - rise
    parts = [_box(thickness, pier_h, depth, chamfer=chamfer,
                  center=(sx * (width - thickness) / 2,
                          -height / 2 + pier_h / 2, 0.0))
             for sx in (1, -1)]

    r, R = span / 2.0, span / 2.0 + thickness
    step = math.pi / segments
    # Neighbouring voussoirs overlap slightly so no two corners coincide — a
    # merge is not a union, and coincident vertices are what would make it one.
    # The two springing ends are left alone so the arch still starts flat on the
    # piers and the ring's extreme corners stay at exactly +-R.
    overlap = step * 0.08
    spans = [(i * step - (0.0 if i == 0 else overlap),
              (i + 1) * step + (0.0 if i == segments - 1 else overlap))
             for i in range(segments)]
    # Squash the circle into the rise asked for, measured off the corner that
    # actually reaches highest rather than off the crown of the ideal circle.
    peak = max(math.sin(a) for pair in spans for a in pair)
    k = rise / (R * peak)

    for a0, a1 in spans:
        polygon = [(math.cos(a) * q, math.sin(a) * q * k)
                   for a, q in ((a0, r), (a0, R), (a1, R), (a1, r))]
        block = _prism(polygon, depth)
        # _prism builds in the (z, y) plane and extrudes along x; the arch wants
        # the curve in (x, y) with the gateway's depth along z.
        block.apply_transform(trimesh.transformations.rotation_matrix(
            math.pi / 2, (0, 1, 0)))
        block.apply_translation((0.0, -height / 2 + pier_h, 0.0))
        parts.append(block)
    return _combine(parts)


def _doorway(width, height, depth, jamb, lintel, threshold, chamfer):
    """A standalone door frame — the thing you drop into a hole you already cut.

    Useful on its own, and the reason it is here: an aperture cut into a
    generated hull has a raw voxel edge, and a frame around it is what makes the
    opening read as a door rather than as damage.
    """
    if 2 * jamb >= width:
        raise ValueError(f"jamb {jamb} leaves no opening in a {width} doorway")
    if lintel + threshold >= height:
        raise ValueError(f"lintel {lintel} and threshold {threshold} fill a "
                         f"{height} doorway")

    parts = []
    for sx in (1, -1):
        parts.append(_box(jamb, height, depth, chamfer=chamfer,
                          center=(sx * (width - jamb) / 2, 0.0, 0.0)))
    parts.append(_box(width - 2 * jamb, lintel, depth, chamfer=chamfer,
                      center=(0.0, (height - lintel) / 2, 0.0)))
    if threshold > _EPS:
        parts.append(_box(width - 2 * jamb, threshold, depth, chamfer=chamfer,
                          center=(0.0, -(height - threshold) / 2, 0.0)))
    return _combine(parts)


# --- catalogue ---------------------------------------------------------------
#
# A separate registry from primitives.KINDS on purpose: this module does not
# reach into that dict, so nothing here can change what GET /primitives already
# returns. Merging is one line when someone wants it —
# `primitives.KINDS.update(hollow.KINDS)`.

KINDS: dict[str, Kind] = {}


def _register(kind: Kind):
    KINDS[kind.name] = kind


_register(Kind(
    "room", "A building shell you can walk into: walls, floor, doorway.",
    "stone", _room,
    (
        _studs("width", 8.0, "X extent, outside face to outside face."),
        _studs("height", 6.0, "Y extent."),
        _studs("depth", 8.0, "Z extent."),
        _studs("wall_thickness", 0.4, "Thickness of every wall, floor and roof."),
        Param("door", "boolean", True, "Cut a doorway through the +Z wall."),
        _studs("door_width", 2.0, "Doorway width."),
        _studs("door_height", 3.2, "Doorway height above the floor."),
        Param("window", "boolean", False, "Punch a window through the -Z wall."),
        _studs("window_size", 1.6, "Window width and height."),
        _studs("sill_height", 1.6, "Wall below the window."),
        Param("ceiling", "boolean", False,
              "Roof it over. Off by default: an open top is what lets a camera "
              "see in, and Roblox interiors are usually roofed separately."),
        CHAMFER,
    ),
))

_register(Kind(
    "hollow_box", "Container with a wall thickness and one face left open.",
    "wood", _hollow_box,
    (
        _studs("width", 2.0, "X extent."),
        _studs("height", 2.0, "Y extent."),
        _studs("depth", 2.0, "Z extent."),
        _studs("wall_thickness", 0.12, "Wall, floor and lid thickness."),
        Param("open_face", "choice", "top", "Which face to leave off.",
              choices=("top", "bottom", "front", "back", "left", "right", "none")),
        CHAMFER,
    ),
))

_register(Kind(
    "hollow_cylinder", "Silo, tank or barrel shell, open at either end.",
    "metal", _hollow_cylinder,
    (
        _studs("radius", 2.0, "Outer radius."),
        _studs("height", 5.0, "Y extent."),
        _studs("wall_thickness", 0.2, "Wall thickness."),
        Param("open_top", "boolean", True, "Leave the top open."),
        Param("open_bottom", "boolean", False, "Leave the bottom open."),
        _studs("chamfer", 0.04, "Bevel on the rims.", minimum=0.0, maximum=10.0),
        SECTIONS,
    ),
))

_register(Kind(
    "arch", "Gateway: two piers carrying a segmented arch.",
    "stone", _arch,
    (
        _studs("width", 6.0, "X extent."),
        _studs("height", 6.0, "Y extent."),
        _studs("depth", 1.0, "Z extent — how deep the gateway is."),
        _studs("thickness", 0.8, "Pier and voussoir thickness."),
        _studs("rise", 2.2, "Height of the curved part."),
        _count("segments", 7, "Voussoirs in the arch.", minimum=3, maximum=24),
        CHAMFER,
    ),
))

_register(Kind(
    "doorway", "A standalone door frame to trim an opening with.",
    "wood", _doorway,
    (
        _studs("width", 2.4, "X extent."),
        _studs("height", 3.6, "Y extent."),
        _studs("depth", 0.4, "Z extent."),
        _studs("jamb", 0.2, "Width of each upright."),
        _studs("lintel", 0.28, "Height of the head."),
        _studs("threshold", 0.0, "Height of the sill. 0 for a clear floor.",
               minimum=0.0),
        CHAMFER,
    ),
))


def catalogue() -> list[dict]:
    return [KINDS[name].as_dict() for name in sorted(KINDS)]


def kinds() -> list[str]:
    return sorted(KINDS)


def resolve(kind: str, params: dict | None) -> dict:
    """Validate and default a parameter set, exactly as primitives.resolve does."""
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {kinds()}")
    spec = KINDS[kind]
    given = dict(params or {})

    known = {p.name for p in spec.params}
    unknown = sorted(set(given) - known)
    if unknown:
        raise ValueError(
            f"unknown parameter(s) {unknown} for kind {kind!r}; "
            f"expected any of {sorted(known)}"
        )
    return {
        p.name: (p.coerce(given[p.name]) if p.name in given else p.default)
        for p in spec.params
    }


def build(kind: str, params: dict | None = None, part_name: str | None = None,
          material: str | None = None, color: str | None = None,
          uv_scale: float | None = None) -> trimesh.Trimesh:
    """Build one hollow primitive, centred and materialled like a scripted part."""
    resolved = resolve(kind, params)
    spec = KINDS[kind]
    mesh = _center(spec.build(**resolved))

    if len(mesh.faces) > ROBLOX_MAX_TRIANGLES:
        raise ValueError(
            f"{kind} with these parameters is {len(mesh.faces)} faces, over the "
            f"{ROBLOX_MAX_TRIANGLES} cap — reduce the counts"
        )

    materials.apply_to_mesh(
        mesh, part_name or kind,
        material or primitives._material_for(spec, part_name, material), color)
    if uv_scale:
        mesh.visual.uv = primitives._unwrap(mesh, uv_scale)
    return mesh


def store(kind: str, params: dict | None, out_dir: Path, **kwargs) -> dict:
    """Build and write mesh.glb, in pipeline.generate_shape's result shape, so a
    hollow part is indistinguishable from a generated one downstream."""
    t0 = time.time()
    resolved = resolve(kind, params)
    mesh = build(kind, resolved, **kwargs)
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.glb"
    mesh.export(str(mesh_path))
    lo, hi = mesh.bounds

    log.info("built hollow %s: %d faces in %.3fs", kind, len(mesh.faces), elapsed)
    return {
        "mesh_path": str(mesh_path),
        "generation_seconds": round(elapsed, 3),
        "peak_vram_gib": 0.0,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "decimated_from": None,
        "material": primitives._family_of(mesh),
        "watertight": bool(mesh.is_watertight),
        "hollow": True,
        "file_bytes": mesh_path.stat().st_size,
        "size": [round(float(v), 4) for v in (hi - lo)],
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "params": {"kind": kind, **resolved},
    }


def hollow_file(mesh_path: Path, out_path: Path, **kwargs) -> dict:
    """Hollow a mesh on disk. The shape an endpoint would return."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    result = hollow(mesh, **kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.mesh.export(str(out_path))
    lo, hi = result.mesh.bounds
    return {
        "mesh_path": str(out_path),
        "source": str(mesh_path),
        "vertices": int(len(result.mesh.vertices)),
        "faces": int(len(result.mesh.faces)),
        "file_bytes": out_path.stat().st_size,
        "size": [round(float(v), 4) for v in (hi - lo)],
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        **result.report,
    }
