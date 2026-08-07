"""Scripted parts — the other half of the generator.

docs/QUALITY-COMPARISON.md measured a wooden crate as the *most expensive*
object either generator handled: 151 s and 6.88 GiB for TRELLIS 2, 83 s for
Hunyuan3D, and TRELLIS 2's recommended settings did not finish at all. The
reason is structural — generation cost scales with occupied volume, and a crate
is solid. A dragon is mostly empty space and is cheap; a box is the worst case.

A crate is also about forty lines of arithmetic. Scripting one gives exact
dimensions, watertight geometry, a few hundred triangles instead of 20,000, and
a material that is known rather than inferred. So the routing rule is:
geometric, man-made, dimensioned objects come from here; organic and sculptural
ones go to the GPU. This module covers the first half.

Pure CPU, pure numpy + trimesh (MIT). No booleans and no CAD kernel: every
shape is built vertex-by-vertex or by revolving a profile, which is why the
face counts are exact and predictable rather than a tessellation tolerance.
Detail is added by *composition* — a crate is a panel plus posts plus boards,
the same kitbashing idea the rest of the project is built on.
"""
import functools
import itertools
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

import config
import materials

log = logging.getLogger("kitbash.primitives")

_EPS = 1e-9


# --- low-level geometry ------------------------------------------------------

def _quad(a: int, b: int, c: int, d: int) -> list[tuple[int, int, int]]:
    """Two triangles for a quad, dropping any that collapsed to an edge.

    Collapse happens wherever a revolved ring sits on the axis, so the caller
    does not have to special-case cones and closed ends.
    """
    return [t for t in ((a, b, c), (a, c, d)) if len(set(t)) == 3]


def _fan(ring: list[int]) -> list[tuple[int, int, int]]:
    """Triangle fan over a convex ring of vertex indices."""
    return [(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cross product, spelled out.

    np.cross spends more time in axis bookkeeping than in arithmetic, and this
    runs once per face of every box in every crate.
    """
    return np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1],
                     a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], axis=1)


def _orient_convex(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip faces whose normal points inward. Only valid for a convex solid.

    The reference point is the mean of the vertices, which is interior for any
    convex body — so a swept panel that does not straddle the origin is judged
    as correctly as a box that does.
    """
    v = vertices[faces] - vertices.mean(axis=0)
    normals = _cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    inward = np.einsum("ij,ij->i", normals, v.mean(axis=1)) < 0
    faces = faces.copy()
    faces[inward] = faces[inward][:, ::-1]
    return faces


def _finish(vertices, faces) -> trimesh.Trimesh:
    """Build a mesh with the winding fixed globally.

    process=False on purpose: index sharing is already exact, and letting
    trimesh weld by position would fuse parts that merely touch.
    """
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    if mesh.volume < 0:
        mesh.invert()
    return mesh


@functools.lru_cache(maxsize=32)
def _bevel_topology(rings: tuple[tuple[int, ...], ...]):
    """Which faces meet at each corner, and the triangles a bevel makes of them.

    Connectivity depends only on the rings, and a crate builds twenty-three
    boxes with the same ones, so this is worth remembering. Returns the three
    face ids at each corner and the finished triangle list, indexed so that
    corner `i` keeps index `i` and its point on face `fa[i][k]` lands at
    `V * (1 + k) + i`.
    """
    n_v = 1 + max(j for ring in rings for j in ring)
    at: list[list[int]] = [[] for _ in range(n_v)]
    for i, ring in enumerate(rings):
        for j in ring:
            at[j].append(i)
    if any(len(f) != 3 for f in at):
        raise ValueError("_bevel needs every corner to meet exactly three faces")

    edges: dict[tuple[int, int], list[int]] = {}
    for i, ring in enumerate(rings):
        for k, a in enumerate(ring):
            b = ring[(k + 1) % len(ring)]
            edges.setdefault((min(a, b), max(a, b)), []).append(i)

    fa = [sorted(f) for f in at]
    slot = {(i, f): n_v * (1 + k) + i
            for i, faces in enumerate(fa) for k, f in enumerate(faces)}

    tris = []
    for i, ring in enumerate(rings):
        tris += _fan([slot[(j, i)] for j in ring])
    for (a, b), (i, j) in edges.items():
        tris += _fan([slot[(a, i)], a, slot[(a, j)],
                      slot[(b, j)], b, slot[(b, i)]])
    return np.array(fa), np.array(tris, dtype=np.int64)


def _bevel(vertices, polygons, chamfer: float = 0.0) -> trimesh.Trimesh:
    """Assemble a convex solid from planar faces, chamfering every edge.

    `polygons` are rings of indices into `vertices`, one per face, wound either
    way — winding is fixed globally at the end. Every corner must meet exactly
    three faces, which is true of a box, a triangular prism and a trapezoidal
    panel alike.

    Cutting each edge at the bisector leaves every original face *itself*,
    pulled in by `chamfer` along its own plane, joined by hexagonal bevels; the
    three bevels meeting at a corner intersect at a single point rather than
    leaving a corner facet. A box comes out 32 vertices and 60 triangles.

    The chamfer is what separates a prop from a placeholder — a bare cube reads
    as untextured level-blocking, while a 2 cm chamfer catches a highlight on
    every edge and the same mesh reads as a made object.
    """
    v = np.asarray(vertices, dtype=float)
    rings = tuple(tuple(p) for p in polygons)
    t = float(chamfer)

    if t <= _EPS:
        faces = [tri for ring in rings for tri in _fan(list(ring))]
        return trimesh.Trimesh(
            vertices=v, faces=_orient_convex(v, np.array(faces, dtype=np.int64)),
            process=False,
        )

    # One outward plane per face. The centroid of a convex body's vertices is
    # inside it, which is all "outward" needs to mean here.
    inside = v.mean(axis=0)
    corners = v[[r[0] for r in rings]]
    normals = _cross(v[[r[1] for r in rings]] - corners,
                     v[[r[2] for r in rings]] - corners)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    outward = np.einsum("ij,ij->i", normals, corners - inside)
    normals *= np.where(outward > 0, 1.0, -1.0)[:, None]
    offsets = np.einsum("ij,ij->i", normals, corners)

    n_v = len(v)
    fa, tris = _bevel_topology(rings)     # (V, 3) face ids per corner
    nn, dd = normals[fa], offsets[fa]

    # Four new points per corner: the one where its three bevels meet, and one
    # on each of its three faces. Each is the meeting of three planes whose
    # offsets are affine in t, so it travels in a straight line as the chamfer
    # widens — solving with t as a second right-hand side gives both where it
    # starts (the original corner) and the direction it leaves in, which is
    # what makes the clamp below exact rather than a guess.
    #
    # The bevel between faces i and j contains both of their inset lines, which
    # puts it at (n_i + n_j).p = d_i + d_j - t; three of those meet at a corner.
    # A face point is simpler: on its own face, inset from the other two.
    rows = [nn + nn[:, [1, 2, 0]]]
    base = [dd + dd[:, [1, 2, 0]]]
    slope = [np.full(3, -1.0)]
    for k in range(3):
        cyc = [k, (k + 1) % 3, (k + 2) % 3]
        rows.append(nn[:, cyc])
        base.append(dd[:, cyc])
        slope.append(np.array([0.0, -1.0, -1.0]))
    rhs = np.stack([np.concatenate(base), np.repeat(slope, n_v, axis=0)], axis=-1)
    moves = np.linalg.solve(np.concatenate(rows), rhs)

    # A chamfer wider than the solid's own waist eats the face it was supposed
    # to bevel. The largest t that keeps every new point inside the original
    # planes is a plain linear bound; 0.49 of half of it is the same margin the
    # chamfered box has always used, and unlike a bound taken from the bounding
    # box it tightens on its own at an acute corner.
    travel = moves[..., 1] @ normals.T
    slack = offsets - moves[..., 0] @ normals.T
    live = travel > 1e-9
    if live.any():
        t = min(t, float((np.maximum(slack[live], 0.0) / travel[live]).min()) * 0.245)

    verts = moves[..., 0] + t * moves[..., 1]
    return trimesh.Trimesh(
        vertices=verts, faces=_orient_convex(verts, tris), process=False,
    )


# The eight (x, y, z) sign triples, in one fixed order every hexahedron uses.
_HEX = np.array(list(itertools.product((1, -1), repeat=3)))


def _hex_rings() -> tuple[tuple[int, ...], ...]:
    """The six quads of a hexahedron, over corners in `_HEX` order.

    Face (axis, sign) is the four corners sharing that sign on that axis, taken
    round the other two axes so the ring is a genuine cycle.
    """
    index = {tuple(s): i for i, s in enumerate(_HEX.tolist())}
    rings = []
    for d in range(3):
        u, w = (d + 1) % 3, (d + 2) % 3
        for sd in (1, -1):
            ring = []
            for su, sw in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
                s = [0, 0, 0]
                s[d], s[u], s[w] = sd, su, sw
                ring.append(index[tuple(s)])
            rings.append(tuple(ring))
    return tuple(rings)


_HEX_RINGS = _hex_rings()


def _hexahedron(corners, chamfer: float = 0.0) -> trimesh.Trimesh:
    """A six-faced cell from eight corners given in `_HEX` order.

    A box with its corners moved: as long as each of the three pairs of
    opposite faces stays planar it is still six quads and still bevels. That is
    what lets a tapered panel reuse the box's chamfer instead of inventing one.
    """
    return _bevel(corners, _HEX_RINGS, chamfer)


def _box(width: float, height: float, depth: float, center=(0.0, 0.0, 0.0),
         chamfer: float = 0.0) -> trimesh.Trimesh:
    """A box, optionally chamfered on all twelve edges. 60 triangles bevelled,
    12 without — the 48 the bevel costs are the cheapest detail in the file."""
    if float(chamfer) <= _EPS:
        mesh = trimesh.creation.box(extents=(width, height, depth))
        mesh.apply_translation(center)
        return mesh

    extents = np.array([width, height, depth], dtype=float) / 2.0
    mesh = _hexahedron(_HEX * extents, chamfer)
    mesh.apply_translation(center)
    return mesh


def _revolve(profile, sections: int, modulation=None) -> trimesh.Trimesh:
    """Revolve a closed (radius, height) profile around +Y.

    `profile` points are (r, y) or (r, y, modulate). `modulation` is a
    per-section radial multiplier — that is what turns a smooth lathe into a
    staved barrel or a fluted column without a single boolean.
    """
    pts = [(float(p[0]), float(p[1]), bool(p[2]) if len(p) > 2 else True)
           for p in profile]
    angles = np.linspace(0.0, 2 * np.pi, sections, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)
    mod = np.ones(sections) if modulation is None else np.asarray(modulation, float)

    verts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, y, modulate in pts:
        if r <= _EPS:
            # On the axis: one shared vertex, so the cap closes as a fan.
            rings.append([len(verts)] * sections)
            verts.append((0.0, y, 0.0))
            continue
        radii = r * (mod if modulate else 1.0)
        base = len(verts)
        verts.extend(zip(radii * cos, np.full(sections, y), radii * sin))
        rings.append(list(range(base, base + sections)))

    faces: list[tuple[int, int, int]] = []
    for i in range(len(rings)):
        a, b = rings[i], rings[(i + 1) % len(rings)]
        for j in range(sections):
            k = (j + 1) % sections
            faces += _quad(a[j], a[k], b[k], b[j])

    return _finish(verts, faces)


def _prism(polygon, width: float, chamfer: float = 0.0) -> trimesh.Trimesh:
    """Extrude a convex (z, y) polygon along X. Fan-triangulated, so convex only.

    Every corner of an extrusion meets one cap and two sides, so `_bevel` can
    take it — which is how a ramp gets the same edge treatment as a slab
    instead of meeting one with a knife edge.
    """
    n = len(polygon)
    verts = [(-width / 2.0, y, z) for z, y in polygon]
    verts += [(width / 2.0, y, z) for z, y in polygon]
    if float(chamfer) > _EPS:
        rings = [list(range(n)), list(range(n, 2 * n))]
        rings += [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
        return _bevel(verts, rings, chamfer)
    # The -X cap is wound against the direction the side quads travel, which is
    # what makes the two agree on which way is out.
    faces = _fan(list(range(n))[::-1]) + _fan(list(range(n, 2 * n)))
    for i in range(n):
        j = (i + 1) % n
        faces += _quad(i, j, n + j, n + i)
    return _finish(verts, faces)


def _combine(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Merge components into one mesh without welding them.

    Each component is already a closed solid, so the result stays watertight by
    trimesh's definition (every edge bounded by exactly two faces) even though
    the components interpenetrate. That interpenetration is deliberate: a board
    embedded in a panel has no coincident vertices to fuse, which is what keeps
    the merge safe without a boolean engine.
    """
    return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]


def _center(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Put the bounding-box centre on the origin, matching generated parts.

    Generated meshes arrive centred, and /assemble's placement maths assumes it.
    A library that sat its parts on Y=0 instead would silently offset every
    scene that mixed the two sources.
    """
    mesh.apply_translation(-mesh.bounding_box.centroid)
    return mesh


def _unwrap(mesh: trimesh.Trimesh, scale: float) -> np.ndarray:
    """Box-projection UVs, one tile per `scale` units.

    Splits every vertex so each face gets the projection matching its own
    normal. That is a correct hard-surface unwrap and it is also why it is
    opt-in: splitting vertices ends the welded topology that makes the mesh
    watertight, and today nothing downstream has a texture to put on it.
    """
    mesh.unmerge_vertices()
    corners = mesh.vertices[mesh.faces]
    axis = np.abs(mesh.face_normals).argmax(axis=1)
    # Project onto the two axes the face is not facing along.
    u_axis, v_axis = (axis + 1) % 3, (axis + 2) % 3
    uv = np.zeros((len(mesh.vertices), 2))
    for corner in range(3):
        idx = mesh.faces[:, corner]
        pos = corners[:, corner, :]
        uv[idx, 0] = pos[np.arange(len(pos)), u_axis] / scale
        uv[idx, 1] = pos[np.arange(len(pos)), v_axis] / scale
    return uv


# --- parameter schema --------------------------------------------------------

@dataclass(frozen=True)
class Param:
    name: str
    type: str  # number | integer | boolean | choice
    default: Any
    description: str
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None

    def as_dict(self) -> dict:
        d = {
            "name": self.name,
            "type": self.type,
            "default": self.default,
            "description": self.description,
        }
        for key in ("unit", "minimum", "maximum"):
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        if self.choices:
            d["choices"] = list(self.choices)
        return d

    def coerce(self, value: Any) -> Any:
        if self.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{self.name} must be true or false, got {value!r}")
            return value
        if self.type == "choice":
            if value not in self.choices:
                raise ValueError(
                    f"{self.name} must be one of {list(self.choices)}, got {value!r}"
                )
            return value
        # bool is an int in Python and would sail through the numeric checks.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.name} must be a number, got {value!r}")
        if self.type == "integer":
            # Check before narrowing, or int() has already thrown away the
            # fraction that made the value wrong.
            if float(value) != int(value):
                raise ValueError(f"{self.name} must be a whole number, got {value}")
            value = int(value)
        else:
            value = float(value)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.name} must be >= {self.minimum}, got {value}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.name} must be <= {self.maximum}, got {value}")
        return value


def _studs(name, default, description, minimum=0.01, maximum=200.0):
    return Param(name, "number", default, description, unit="studs",
                 minimum=minimum, maximum=maximum)


def _count(name, default, description, minimum=1, maximum=64):
    return Param(name, "integer", default, description, unit="count",
                 minimum=minimum, maximum=maximum)


CHAMFER = _studs(
    "chamfer", 0.02,
    "Bevel on every edge. Nonzero is what makes a box read as a made object "
    "rather than a placeholder; 0 gives a hard cube.",
    minimum=0.0, maximum=10.0,
)

SECTIONS = Param(
    "sections", "integer", config.PRIMITIVE_SECTIONS,
    "Radial segments on round surfaces. 24 is smooth at prop scale; 8-12 reads "
    "as deliberately faceted.",
    unit="count", minimum=3, maximum=256,
)


@dataclass(frozen=True)
class Kind:
    name: str
    summary: str
    material: str
    build: Callable[..., trimesh.Trimesh]
    params: tuple[Param, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        return {
            "kind": self.name,
            "summary": self.summary,
            "material": self.material,
            "params": [p.as_dict() for p in self.params],
        }


# --- the primitives ----------------------------------------------------------

def _crate(width, height, depth, style, plank_count, batten, relief, chamfer):
    """A shipping crate: recessed core, corner posts, boards across every face.

    Built as separate boards rather than as a textured cube because the whole
    argument for scripting this is that the relief is real geometry — it still
    reads at a grazing angle and under any light, where a normal map would not.
    """
    parts = [_box(width - 2 * relief, height - 2 * relief, depth - 2 * relief,
                  chamfer=chamfer)]

    # Corner posts run the full height and stand proud of the recessed core.
    for sx, sz in itertools.product((1, -1), repeat=2):
        parts.append(_box(batten, height, batten, chamfer=chamfer,
                          center=((width - batten) / 2 * sx, 0.0,
                                  (depth - batten) / 2 * sz)))

    if style == "plain":
        return _combine(parts)

    if style == "frame":
        # Just a rail top and bottom, leaving the panel between them exposed.
        rows = [((height - batten) / 2 * s, batten) for s in (1, -1)]
    else:
        pitch = height / plank_count
        # The remaining 18% of the pitch is the gap between boards.
        rows = [(-height / 2 + pitch * (i + 0.5), pitch * 0.82)
                for i in range(plank_count)]

    for y, board_h in rows:
        for sx in (1, -1):
            parts.append(_box(batten, board_h, depth - 2 * batten, chamfer=chamfer,
                              center=((width - batten) / 2 * sx, y, 0.0)))
        for sz in (1, -1):
            parts.append(_box(width - 2 * batten, board_h, batten, chamfer=chamfer,
                              center=(0.0, y, (depth - batten) / 2 * sz)))

    # Lid and floor boards run the other way, so the two read as separate
    # carpentry rather than one extruded silhouette.
    lid_pitch = (depth - 2 * batten) / plank_count
    for i in range(plank_count):
        z = -(depth - 2 * batten) / 2 + lid_pitch * (i + 0.5)
        for sy in (1, -1):
            parts.append(_box(width - 2 * batten, batten, lid_pitch * 0.82,
                              chamfer=chamfer,
                              center=(0.0, (height - batten) / 2 * sy, z)))
    return _combine(parts)


def _barrel(height, belly_radius, end_radius, stave_count, hoop_count,
            hoop_thickness, hoop_width, rings):
    """A staved, bellied barrel.

    The staves are a per-section radial modulation of one revolved surface
    rather than N separate planks: same silhouette and the same visible seams,
    one closed shell, a quarter of the triangles.
    """
    sections = stave_count * 2
    # Alternate stave centre (proud) with stave seam (recessed).
    groove = min(0.035, (belly_radius - end_radius) / belly_radius * 0.35 + 0.012)
    modulation = np.where(np.arange(sections) % 2 == 0, 1.0, 1.0 - groove)

    profile = [(0.0, -height / 2)]
    for i in range(rings + 1):
        t = i / rings
        y = -height / 2 + t * height
        # A cosine belly, flat-topped enough that the ends stay parallel.
        r = end_radius + (belly_radius - end_radius) * np.sin(np.pi * t) ** 0.7
        profile.append((float(r), float(y)))
    profile.append((0.0, height / 2))
    body = _revolve(profile, sections, modulation)

    parts = [body]
    for i in range(hoop_count):
        t = 0.18 + 0.64 * (i / max(hoop_count - 1, 1)) if hoop_count > 1 else 0.5
        y = -height / 2 + t * height
        r = end_radius + (belly_radius - end_radius) * np.sin(np.pi * t) ** 0.7
        inner, outer = r - hoop_thickness, r + hoop_thickness
        parts.append(_revolve(
            [(inner, y - hoop_width / 2), (outer, y - hoop_width / 2),
             (outer, y + hoop_width / 2), (inner, y + hoop_width / 2)],
            sections,
        ))
    return _combine(parts)


def _cylinder(radius, height, wall_thickness, chamfer, sections):
    """Solid rod or hollow pipe, with chamfered rims either way."""
    c = min(chamfer, radius * 0.45, height * 0.45)
    top, bottom = height / 2, -height / 2
    if wall_thickness <= 0:
        profile = [(0.0, bottom), (radius - c, bottom), (radius, bottom + c),
                   (radius, top - c), (radius - c, top), (0.0, top)]
        return _revolve(profile, sections)

    inner = radius - wall_thickness
    profile = [
        (inner + c, bottom), (radius - c, bottom), (radius, bottom + c),
        (radius, top - c), (radius - c, top), (inner + c, top),
        (inner, top - c), (inner, bottom + c),
    ]
    return _revolve(profile, sections)


def _plank(length, width, thickness, chamfer):
    return _box(length, thickness, width, chamfer=chamfer)


def _tapered_panel(span, root_chord, tip_chord, thickness, sweep,
                   thickness_taper, chamfer):
    """A trapezoidal panel — a wing, tailplane, fin, rotor blade or body side.

    Same axes as `plank`: span along X with the root at -X, chord along Z,
    thickness along Y. With `root_chord == tip_chord` and no thickness taper it
    *is* a plank, which is the point — one kind covers the constant-section
    case and the tapered one, and nobody has to butt two planks together and
    live with the step in the planform where they meet.

    Chord and thickness both vary linearly with span, so all six faces stay
    planar and this is a `_hexahedron`. It is deliberately not a `_prism`: the
    moment `thickness_taper` is nonzero the top and bottom faces slant and the
    cross-section is no longer constant along any axis.
    """
    tip = _HEX[:, 0] > 0
    chord = np.where(tip, tip_chord, root_chord)
    thick = np.where(tip, thickness * (1.0 - thickness_taper), thickness)
    offset = np.where(tip, sweep, 0.0)
    return _hexahedron(np.stack([_HEX[:, 0] * span / 2.0,
                                 _HEX[:, 1] * thick / 2.0,
                                 offset + _HEX[:, 2] * chord / 2.0], axis=1),
                       chamfer)


def _wall_panel(width, height, thickness, opening, opening_width, opening_height,
                sill_height, trim, trim_depth, chamfer):
    """A wall section, optionally with a window or a door cut through it.

    The opening is not subtracted — the wall is built as the four slabs that
    surround it. That is exact, needs no boolean engine, and leaves quads
    instead of the sliver triangles a mesh boolean puts around an aperture.
    """
    if opening == "none":
        return _box(width, height, thickness, chamfer=chamfer)

    below = 0.0 if opening == "door" else sill_height
    above = height - below - opening_height
    if above <= _EPS:
        raise ValueError(
            f"opening_height {opening_height} leaves no wall above it in a "
            f"{height}-high panel (sill at {below})"
        )
    side = (width - opening_width) / 2
    if side <= _EPS:
        raise ValueError(
            f"opening_width {opening_width} leaves no wall beside it in a "
            f"{width}-wide panel"
        )

    parts = []
    for sx in (1, -1):
        parts.append(_box(side, height, thickness, chamfer=chamfer,
                          center=((width - side) / 2 * sx, 0.0, 0.0)))
    parts.append(_box(opening_width, above, thickness, chamfer=chamfer,
                      center=(0.0, height / 2 - above / 2, 0.0)))
    if below > _EPS:
        parts.append(_box(opening_width, below, thickness, chamfer=chamfer,
                          center=(0.0, -height / 2 + below / 2, 0.0)))

    if trim:
        # A reveal standing proud on both faces, which is what makes an opening
        # look framed rather than punched.
        t = min(trim_depth, thickness)
        jamb = min(0.12 * min(opening_width, opening_height), side, above)
        # A door's trim stops at the wall's own base; running it the full
        # opening height plus a head leaves it hanging in the air below.
        sill = below > _EPS
        jamb_h = opening_height + jamb * (2 if sill else 1)
        jamb_y = (-height / 2 + below + opening_height / 2 if sill
                  else -height / 2 + jamb_h / 2)
        for sz in (1, -1):
            z = (thickness + t) / 2 * sz
            for sx in (1, -1):
                parts.append(_box(jamb, jamb_h, t, chamfer=chamfer,
                                  center=((opening_width + jamb) / 2 * sx,
                                          jamb_y, z)))
            parts.append(_box(opening_width, jamb, t, chamfer=chamfer,
                              center=(0.0, -height / 2 + below + opening_height
                                      + jamb / 2, z)))
            if sill:
                parts.append(_box(opening_width, jamb, t, chamfer=chamfer,
                                  center=(0.0, -height / 2 + below - jamb / 2, z)))
    return _combine(parts)


def _wheel(radius, width, hub_radius, rim_width, spoke_count, tread_depth,
           chamfer, sections):
    """A wheel or disc: chamfered tyre, hub, and either spokes or a solid web."""
    c = min(chamfer, width * 0.4, radius * 0.2)
    solid = spoke_count == 0
    inner = hub_radius if solid else radius - rim_width
    half = width / 2

    # Only the outer rings carry the tread modulation — letting it reach the
    # inner rings scallops the whole rim and the wheel comes out as a flower.
    profile = [
        (inner, -half, False), (radius - c, -half, True), (radius, -half + c, True),
        (radius, half - c, True), (radius - c, half, True), (inner, half, False),
    ]
    if solid:
        # Close across the axis so a disc is one solid body.
        profile = [(0.0, -half, False)] + profile[1:-1] + [(0.0, half, False)]
    modulation = None
    if tread_depth > 0 and not solid:
        # Tread is cut into the carcass, so it competes with the facets of the
        # circle itself. Two sections proud and two recessed gives each block a
        # flat top; alternating every section just makes a gear.
        sections = max(sections, 48)
        modulation = np.where(np.arange(sections) % 4 < 2,
                              1.0, 1.0 - tread_depth / radius)
    parts = [_revolve(profile, sections, modulation)]

    if not solid:
        # Flush with the tyre: a protruding boss looks better but would make
        # `width` stop being the Y extent the caller asked for.
        parts.append(_revolve(
            [(0.0, -half), (hub_radius, -half),
             (hub_radius, half), (0.0, half)], sections))
        spoke_len = radius - rim_width * 0.5 - hub_radius * 0.5
        spoke_z = min(width * 0.55, spoke_len * 0.3)
        spoke_y = width * 0.7
        for i in range(spoke_count):
            # The wheel lies in XZ with Y as the axle, so spokes sweep about Y.
            spoke = _box(spoke_len, spoke_y, spoke_z,
                         chamfer=min(chamfer, spoke_z * 0.3))
            spoke.apply_translation(((hub_radius + radius - rim_width * 0.5) / 2, 0, 0))
            spoke.apply_transform(trimesh.transformations.rotation_matrix(
                2 * np.pi * i / spoke_count, (0, 1, 0)))
            parts.append(spoke)
    return _combine(parts)


def _stairs(steps, rise, run, width, style, tread_thickness, chamfer):
    """A staircase. `blocks` is masonry (each step solid to the ground);
    `open` is carpentry (treads on two stringers)."""
    total_run, total_rise = steps * run, steps * rise
    parts = []
    if style == "blocks":
        for i in range(steps):
            h = (i + 1) * rise
            parts.append(_box(width, h, run, chamfer=chamfer,
                              center=(0.0, h / 2, -total_run / 2 + run * (i + 0.5))))
        return _combine(parts)

    # Open treads and nothing behind them — adding risers would just rebuild
    # the `blocks` silhouette out of more triangles.
    stringer_t = max(tread_thickness * 0.9, 0.06)
    for i in range(steps):
        parts.append(_box(width, tread_thickness, run * 1.06, chamfer=chamfer,
                          center=(0.0, (i + 1) * rise - tread_thickness / 2,
                                  -total_run / 2 + run * (i + 0.5))))

    length = float(np.hypot(total_run, total_rise))
    angle = float(np.arctan2(total_rise, total_run))
    for sx in (1, -1):
        stringer = _box(stringer_t, rise * 1.1, length, chamfer=chamfer)
        stringer.apply_transform(
            trimesh.transformations.rotation_matrix(-angle, (1, 0, 0)))
        # Ride just below the step noses so the treads sit on the stringer
        # rather than inside a trough.
        stringer.apply_translation(((width - stringer_t) / 2 * sx,
                                    total_rise / 2 - rise * 0.35, 0.0))
        parts.append(stringer)
    return _combine(parts)


def _ladder(height, width, rung_count, rail_width, rail_thickness, rung_radius,
            chamfer, sections):
    parts = []
    for sx in (1, -1):
        parts.append(_box(rail_thickness, height, rail_width, chamfer=chamfer,
                          center=((width - rail_thickness) / 2 * sx, 0.0, 0.0)))
    span = height - 2 * rail_width
    pitch = span / max(rung_count - 1, 1) if rung_count > 1 else 0.0
    for i in range(rung_count):
        y = -span / 2 + pitch * i if rung_count > 1 else 0.0
        # Rungs run the full width so they are seated inside both rails.
        rung = _cylinder(rung_radius, width, 0.0, rung_radius * 0.25, sections)
        rung.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, (0, 0, 1)))
        rung.apply_translation((0.0, y, 0.0))
        parts.append(rung)
    return _combine(parts)


def _column(height, radius, style, base_height, capital_height, base_overhang,
            flute_count, taper, chamfer, sections):
    """A pillar with a base and capital, optionally tapered or fluted."""
    shaft_bottom = -height / 2 + base_height
    shaft_top = height / 2 - capital_height
    base_r = radius + base_overhang
    edge = min(chamfer, base_overhang * 0.8)
    top_r = radius * (1.0 - taper) if style == "tapered" else radius

    profile = [(0.0, -height / 2)]
    if base_height > 0:
        c = min(edge, base_height * 0.4)
        profile += [(base_r - c, -height / 2), (base_r, -height / 2 + c),
                    (base_r, shaft_bottom - c), (base_r - c, shaft_bottom)]
    # Only the shaft rings are flagged, so the flutes stop at base and capital.
    profile += [(radius, shaft_bottom, True), (top_r, shaft_top, True)]
    if capital_height > 0:
        c = min(edge, capital_height * 0.4)
        profile += [(base_r - c, shaft_top), (base_r, shaft_top + c),
                    (base_r, height / 2 - c), (base_r - c, height / 2)]
    profile.append((0.0, height / 2))

    # Everything not explicitly flagged is excluded from the flute modulation.
    profile = [p if len(p) > 2 else (p[0], p[1], False) for p in profile]

    modulation = None
    if style == "fluted" and flute_count > 0:
        # Four samples per flute: fewer and the scallop reads as facetting.
        sections = max(sections, flute_count * 4)
        phase = np.linspace(0, 2 * np.pi, sections, endpoint=False) * flute_count
        modulation = 1.0 - 0.075 * (0.5 - 0.5 * np.cos(phase))
    return _revolve(profile, sections, modulation)


def _table(width, depth, height, top_thickness, leg_thickness, leg_inset,
           apron, apron_height, chamfer):
    parts = [_box(width, top_thickness, depth, chamfer=chamfer,
                  center=(0.0, height - top_thickness / 2, 0.0))]
    leg_h = height - top_thickness
    for sx, sz in itertools.product((1, -1), repeat=2):
        parts.append(_box(leg_thickness, leg_h, leg_thickness, chamfer=chamfer,
                          center=((width / 2 - leg_inset - leg_thickness / 2) * sx,
                                  leg_h / 2,
                                  (depth / 2 - leg_inset - leg_thickness / 2) * sz)))
    if apron:
        y = leg_h - apron_height / 2
        for sz in (1, -1):
            parts.append(_box(width - 2 * leg_inset, apron_height, leg_thickness * 0.6,
                              chamfer=chamfer,
                              center=(0.0, y, (depth / 2 - leg_inset
                                               - leg_thickness / 2) * sz)))
        for sx in (1, -1):
            parts.append(_box(leg_thickness * 0.6, apron_height, depth - 2 * leg_inset,
                              chamfer=chamfer,
                              center=((width / 2 - leg_inset - leg_thickness / 2) * sx,
                                      y, 0.0)))
    return _combine(parts)


def _bench(length, depth, height, seat_thickness, leg_thickness, backrest,
           back_height, chamfer):
    parts = [_box(length, seat_thickness, depth, chamfer=chamfer,
                  center=(0.0, height - seat_thickness / 2, 0.0))]
    leg_h = height - seat_thickness
    inset = leg_thickness
    for sx, sz in itertools.product((1, -1), repeat=2):
        parts.append(_box(leg_thickness, leg_h, leg_thickness, chamfer=chamfer,
                          center=((length / 2 - inset) * sx, leg_h / 2,
                                  (depth / 2 - inset) * sz)))
    # A stretcher between the leg pairs; without it a bench reads as a plank on
    # four sticks.
    parts.append(_box(length - 2 * inset, leg_thickness * 0.7, leg_thickness * 0.7,
                      chamfer=chamfer, center=(0.0, leg_h * 0.35, 0.0)))
    if backrest:
        z = -depth / 2 + leg_thickness / 2
        for sx in (1, -1):
            parts.append(_box(leg_thickness, back_height, leg_thickness,
                              chamfer=chamfer,
                              center=((length / 2 - inset) * sx,
                                      height + back_height / 2 - leg_thickness, z)))
        for frac in (0.5, 0.85):
            parts.append(_box(length * 0.96, back_height * 0.26,
                              leg_thickness * 0.8, chamfer=chamfer,
                              center=(0.0, height + back_height * frac
                                      - leg_thickness, z)))
    return _combine(parts)


def _wedge(width, height, depth, flip, chamfer):
    """A ramp. The single most common blocking shape in a Roblox place."""
    z = depth / 2
    poly = [(-z, -height / 2), (z, -height / 2), (z, height / 2)]
    if flip:
        poly = [(-z, -height / 2), (z, -height / 2), (-z, height / 2)]
    return _prism(poly, width, chamfer)


# --- catalogue ---------------------------------------------------------------

KINDS: dict[str, Kind] = {}


def _register(kind: Kind):
    KINDS[kind.name] = kind


_register(Kind(
    "crate", "Shipping crate: recessed panels, corner posts and boards.",
    "wood", _crate,
    (
        _studs("width", 2.0, "X extent."),
        _studs("height", 2.0, "Y extent."),
        _studs("depth", 2.0, "Z extent."),
        Param("style", "choice", "planks",
              "planks = boarded on every face; frame = corner posts and top/bottom "
              "rails around recessed panels; plain = a chamfered box.",
              choices=("planks", "frame", "plain")),
        _count("plank_count", 3, "Boards per face.", minimum=1, maximum=16),
        _studs("batten", 0.16, "Thickness of the posts and boards."),
        _studs("relief", 0.05, "How far the boards stand proud of the panel."),
        CHAMFER,
    ),
))

_register(Kind(
    "barrel", "Staved wooden barrel with a bellied profile and metal hoops.",
    "wood", _barrel,
    (
        _studs("height", 2.4, "Y extent."),
        _studs("belly_radius", 0.95, "Radius at the widest point."),
        _studs("end_radius", 0.72, "Radius at the flat ends."),
        _count("stave_count", 14, "Staves around the circumference.",
               minimum=5, maximum=48),
        Param("hoop_count", "integer", 2, "Metal bands around the barrel.",
              unit="count", minimum=0, maximum=8),
        _studs("hoop_thickness", 0.04, "How far a hoop stands off the staves."),
        _studs("hoop_width", 0.18, "Hoop height along Y."),
        _count("rings", 6, "Vertical segments in the belly curve.",
               minimum=2, maximum=32),
    ),
))

_register(Kind(
    "cylinder", "Rod, pipe or tube with chamfered rims.",
    "metal", _cylinder,
    (
        _studs("radius", 0.5, "Outer radius."),
        _studs("height", 3.0, "Y extent."),
        _studs("wall_thickness", 0.0,
               "0 for a solid rod; above 0 makes it a pipe open at both ends.",
               minimum=0.0),
        _studs("chamfer", 0.04, "Bevel on the rims.", minimum=0.0, maximum=10.0),
        SECTIONS,
    ),
))

_register(Kind(
    "plank", "A dimensioned board. Length runs along X, width along Z.",
    "wood", _plank,
    (
        _studs("length", 4.0, "X extent."),
        _studs("width", 0.8, "Z extent."),
        _studs("thickness", 0.12, "Y extent."),
        CHAMFER,
    ),
))

_register(Kind(
    "tapered_panel",
    "Trapezoidal panel: wing, tailplane, fin, rotor blade or vehicle body side.",
    # Every word in that summary is a keyword materials.py already reads as
    # paint, so the kind's default agrees with the name-derived guess instead of
    # fighting it.
    "paint", _tapered_panel,
    (
        _studs("span", 4.0, "X extent. Root at -X, tip at +X."),
        _studs("root_chord", 1.2, "Z extent at the root."),
        _studs("tip_chord", 0.6,
               "Z extent at the tip. Equal to root_chord gives a plain plank."),
        _studs("thickness", 0.16, "Y extent at the root."),
        _studs("sweep", 0.0,
               "How far the tip is offset along +Z. 0 tapers symmetrically "
               "about the mid-chord; +(root_chord - tip_chord)/2 lines the +Z "
               "edges up and the negative of that lines up the -Z ones, which "
               "is how a wing keeps one edge dead straight. More than that "
               "rakes the whole panel back.",
               minimum=-200.0, maximum=200.0),
        Param("thickness_taper", "number", 0.0,
              "Fraction of the thickness lost at the tip. Holding thickness/"
              "chord constant on a wing wants 1 - tip_chord/root_chord.",
              minimum=0.0, maximum=0.9),
        CHAMFER,
    ),
))

_register(Kind(
    "wall_panel", "Wall section, optionally with a window or door opening.",
    "stone", _wall_panel,
    (
        _studs("width", 8.0, "X extent."),
        _studs("height", 6.0, "Y extent."),
        _studs("thickness", 0.5, "Z extent."),
        Param("opening", "choice", "window", "What to leave a hole for.",
              choices=("none", "window", "door")),
        _studs("opening_width", 2.6, "Aperture width."),
        _studs("opening_height", 2.4, "Aperture height."),
        _studs("sill_height", 2.0,
               "Wall below a window opening. Ignored for a door.", minimum=0.0),
        Param("trim", "boolean", True,
              "Add a reveal standing proud around the opening on both faces."),
        _studs("trim_depth", 0.12, "How far the trim stands off the wall."),
        CHAMFER,
    ),
))

_register(Kind(
    "wheel", "Wheel or disc: chamfered tyre with spokes, or a solid disc.",
    "rubber", _wheel,
    (
        _studs("radius", 1.0, "Outer radius."),
        _studs("width", 0.4, "Y extent — the wheel lies in the XZ plane."),
        _studs("hub_radius", 0.18, "Radius of the central hub."),
        _studs("rim_width", 0.22, "Radial depth of the tyre carcass."),
        Param("spoke_count", "integer", 6,
              "0 makes a solid disc with no hub or spokes.",
              unit="count", minimum=0, maximum=32),
        _studs("tread_depth", 0.0,
               "Blocky tread cut into the running surface. Off by default: at "
               "prop scale even a shallow tread reads as a lobed gear rather "
               "than a tyre. 0.03-0.06 suits a large tractor wheel.",
               minimum=0.0),
        CHAMFER,
        # A tyre is looked at end-on more than any other round prop here, so it
        # gets a finer default than the shared one.
        Param("sections", "integer", 32, SECTIONS.description,
              unit="count", minimum=3, maximum=256),
    ),
))

_register(Kind(
    "stairs", "A staircase, masonry or carpentry.",
    "stone", _stairs,
    (
        _count("steps", 6, "Number of steps.", minimum=1, maximum=64),
        _studs("rise", 0.5, "Height of one step."),
        _studs("run", 0.7, "Depth of one step along Z."),
        _studs("width", 4.0, "X extent."),
        Param("style", "choice", "blocks",
              "blocks = each step solid to the ground; open = treads on two "
              "stringers.", choices=("blocks", "open")),
        _studs("tread_thickness", 0.16, "Board thickness, `open` style only."),
        CHAMFER,
    ),
))

_register(Kind(
    "ladder", "Two rails and N rungs.",
    "wood", _ladder,
    (
        _studs("height", 6.0, "Y extent."),
        _studs("width", 1.2, "X extent, outside of rail to outside of rail."),
        _count("rung_count", 8, "Rungs.", minimum=1, maximum=64),
        _studs("rail_width", 0.24, "Rail extent along Z."),
        _studs("rail_thickness", 0.1, "Rail extent along X."),
        _studs("rung_radius", 0.06, "Rung radius."),
        CHAMFER,
        SECTIONS,
    ),
))

_register(Kind(
    "column", "Pillar with a base and capital; plain, tapered or fluted.",
    "stone", _column,
    (
        _studs("height", 8.0, "Y extent."),
        _studs("radius", 0.6, "Shaft radius."),
        Param("style", "choice", "fluted", "Shaft treatment.",
              choices=("plain", "tapered", "fluted")),
        _studs("base_height", 0.5, "Plinth height. 0 for none.", minimum=0.0),
        _studs("capital_height", 0.4, "Capital height. 0 for none.", minimum=0.0),
        _studs("base_overhang", 0.25, "How far base and capital overhang the shaft.",
               minimum=0.0),
        _count("flute_count", 20, "Flutes, `fluted` style only.",
               minimum=0, maximum=64),
        Param("taper", "number", 0.15,
              "Fraction of the radius lost at the top, `tapered` style only.",
              minimum=0.0, maximum=0.9),
        CHAMFER,
        SECTIONS,
    ),
))

_register(Kind(
    "table", "Table: top, four legs, optional apron.",
    "wood", _table,
    (
        _studs("width", 4.0, "X extent of the top."),
        _studs("depth", 2.4, "Z extent of the top."),
        _studs("height", 2.5, "Floor to top surface."),
        _studs("top_thickness", 0.16, "Top slab thickness."),
        _studs("leg_thickness", 0.22, "Square leg cross-section."),
        _studs("leg_inset", 0.12, "How far the legs sit in from the edge.",
               minimum=0.0),
        Param("apron", "boolean", True, "Add a rail under the top between the legs."),
        _studs("apron_height", 0.3, "Apron height."),
        CHAMFER,
    ),
))

_register(Kind(
    "bench", "Bench: seat, four legs, a stretcher, optional back.",
    "wood", _bench,
    (
        _studs("length", 5.0, "X extent."),
        _studs("depth", 1.2, "Z extent."),
        _studs("height", 1.6, "Floor to seat surface."),
        _studs("seat_thickness", 0.16, "Seat slab thickness."),
        _studs("leg_thickness", 0.2, "Square leg cross-section."),
        Param("backrest", "boolean", True, "Add uprights and two back slats."),
        _studs("back_height", 1.4, "Backrest height above the seat."),
        CHAMFER,
    ),
))

_register(Kind(
    "wedge", "A ramp — triangular prism, hypotenuse facing +Z.",
    "stone", _wedge,
    (
        _studs("width", 4.0, "X extent."),
        _studs("height", 2.0, "Y extent."),
        _studs("depth", 4.0, "Z extent, the run of the slope."),
        Param("flip", "boolean", False, "Put the high edge at -Z instead of +Z."),
        # Every other kind bevels by default. A ramp cannot, and the reason is
        # dimensional rather than aesthetic: its apex and its toe are the
        # extremes of the bounding box, so cutting them is the one chamfer in
        # this file that shortens the thing it is applied to — by t/tan(half the
        # edge angle), which on a shallow ramp is several times t. A ramp is
        # blocking geometry that has to meet a floor, so it keeps its exact rise
        # and run unless the caller says otherwise.
        _studs("chamfer", 0.0,
               "Bevel on every edge. Off by default because it truncates the "
               "ramp's apex and toe, which are its own extents — set it to the "
               "chamfer of the slab this abuts (0.02-0.04) when the ramp is a "
               "fin or a fairing rather than blocking.",
               minimum=0.0, maximum=10.0),
    ),
))


def catalogue() -> list[dict]:
    return [KINDS[name].as_dict() for name in sorted(KINDS)]


def kinds() -> list[str]:
    return sorted(KINDS)


def resolve(kind: str, params: dict | None) -> dict:
    """Validate and default a parameter set. Raises ValueError on nonsense."""
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


def _material_for(spec, part_name: str | None, explicit: str | None) -> str:
    """Explicit choice, then the part name, then the kind's default.

    The kind's default must not beat the name: a `bench` called
    "front_left_seat" is a seat, and the caller naming it that is a stronger
    signal than the kind's assumption that benches are wooden. The default is
    only there for parts nobody bothered to name.
    """
    if explicit:
        return explicit
    family, _ = materials.resolve(part_name or "")
    if family != materials.DEFAULT_MATERIAL:
        return family
    return spec.material


def build(kind: str, params: dict | None = None, part_name: str | None = None,
          material: str | None = None, color: str | None = None,
          uv_scale: float | None = None) -> trimesh.Trimesh:
    """Build one primitive as a finished, materialled, origin-centred mesh."""
    resolved = resolve(kind, params)
    spec = KINDS[kind]
    mesh = _center(spec.build(**resolved))

    if len(mesh.faces) > config.PRIMITIVE_MAX_FACES:
        raise ValueError(
            f"{kind} with these parameters is {len(mesh.faces)} faces, over the "
            f"{config.PRIMITIVE_MAX_FACES} cap — reduce the counts "
            f"(sections, plank_count, steps, ...)"
        )

    materials.apply_to_mesh(mesh, part_name or kind, _material_for(spec, part_name, material), color)
    if uv_scale:
        mesh.visual.uv = _unwrap(mesh, uv_scale)
    return mesh


def _family_of(mesh) -> str | None:
    """The material family already baked onto the mesh by apply_to_mesh."""
    name = getattr(getattr(mesh.visual, "material", None), "name", "") or ""
    return name.removeprefix("kitbash_") or None


def store(kind: str, params: dict | None, out_dir: Path, **kwargs) -> dict:
    """Build and write mesh.glb, returning the same result shape as
    pipeline.generate_shape so a scripted part is indistinguishable from a
    generated one everywhere downstream."""
    t0 = time.time()
    resolved = resolve(kind, params)
    mesh = build(kind, resolved, **kwargs)
    elapsed = time.time() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = out_dir / "mesh.glb"
    mesh.export(str(mesh_path))
    lo, hi = mesh.bounds

    log.info("built %s: %d faces in %.3fs", kind, len(mesh.faces), elapsed)
    return {
        "mesh_path": str(mesh_path),
        "generation_seconds": round(elapsed, 3),
        "peak_vram_gib": 0.0,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "decimated_from": None,
        # Recorded so assembly can reuse it. A primitive knows what it is made
        # of; re-deriving it from a node name later can only lose information —
        # "barrel" reads as gun barrel and would come back metal.
        "material": _family_of(mesh),
        "watertight": bool(mesh.is_watertight),
        "file_bytes": mesh_path.stat().st_size,
        "size": [round(float(v), 4) for v in (hi - lo)],
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "params": {"kind": kind, **resolved},
    }
