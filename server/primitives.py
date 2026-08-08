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

# Where `build` leaves "this was watertight before the UV split". Read and
# removed by `store` so it never reaches the exported file's glTF extras.
CLOSED_SOLID = "kitbash_closed_solid"


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


# --- composition -------------------------------------------------------------
#
# docs/SHOWCASE-CHEST.md placed 39 lid planks by hand, one call each. Everything
# in this section exists so the next one is a point set and an array over it.
# Detail stops being expensive the moment it stops being typed out — and Roblox
# allows 20 000 triangles per MeshPart against the 60-870 the old kinds spend,
# so there was never a budget reason to keep it cheap.

def _line_points(count, start, end):
    """`count` points from `start` to `end`, both ends included."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    if count <= 1:
        return ((a + b) / 2.0)[None, :]
    return a + (b - a) * np.linspace(0.0, 1.0, count)[:, None]


def _grid_points(cols, rows, centre=(0.0, 0.0, 0.0), u=(1.0, 0.0, 0.0),
                 v=(0.0, 1.0, 0.0)):
    """A cols x rows lattice filling the rectangle spanned by `u` and `v`.

    `u` and `v` are the *full* edge vectors, so the lattice runs corner to
    corner: rivets on a plate want one in each corner, not one half a pitch in.
    """
    c, u, v = (np.asarray(x, float) for x in (centre, u, v))
    su = np.linspace(-0.5, 0.5, cols) if cols > 1 else np.zeros(1)
    sv = np.linspace(-0.5, 0.5, rows) if rows > 1 else np.zeros(1)
    return np.array([c + a * u + b * v for b in sv for a in su])


def _ring_points(count, radius, centre=(0.0, 0.0, 0.0), axis=1, phase=0.0):
    """`count` points evenly round a circle normal to `axis`."""
    t = np.linspace(0.0, 2 * np.pi, count, endpoint=False) + phase
    pts = np.zeros((count, 3))
    u, w = (axis + 1) % 3, (axis + 2) % 3
    pts[:, u] = radius * np.cos(t)
    pts[:, w] = radius * np.sin(t)
    return pts + np.asarray(centre, float)


def _array(part, points, jitter=0.0, seed=0):
    """Copy `part` to every point. `part` may be a mesh or a callable of index.

    `jitter` is a seeded uniform displacement, which is the cheapest way out of
    the "every crate looks like every other crate" limit: a course of stones
    that all sit at exactly the same depth reads as a printed pattern.
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    if jitter:
        pts = pts + np.random.default_rng(seed).uniform(-jitter, jitter, pts.shape)
    out = []
    for i, p in enumerate(pts):
        mesh = part(i) if callable(part) else part.copy()
        mesh.apply_translation(p)
        out.append(mesh)
    return out


def _facing(normal) -> np.ndarray:
    """The frame of a wall face: X across it, Y still up, Z out of it.

    `_towards` alone would spin a course of bricks onto its side when it put
    +Y onto a horizontal normal; a face needs a whole basis, not one axis.
    """
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(n @ up)) > 0.9:
        right = np.array([1.0, 0.0, 0.0])
        up = np.cross(n, right)
    else:
        right = np.cross(up, n)
        right /= np.linalg.norm(right)
    frame = np.eye(4)
    frame[:3, 0], frame[:3, 1], frame[:3, 2] = right, up, n
    return frame


def _towards(direction) -> np.ndarray:
    """The rotation taking +Y onto `direction` — how a rivet finds its face."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    axis = np.cross((0.0, 1.0, 0.0), d)
    sin = float(np.linalg.norm(axis))
    if sin < _EPS:
        if d[1] > 0:
            return np.eye(4)
        return trimesh.transformations.rotation_matrix(np.pi, (1.0, 0.0, 0.0))
    return trimesh.transformations.rotation_matrix(
        float(np.arctan2(sin, d[1])), axis)


def _rivet(radius, proud, head="dome", sections=8, rings=3):
    """One rivet, bolt or boss, standing `proud` of a surface it is sunk into.

    The skirt below y=0 is what lets it be dropped onto a face and merged: it
    interpenetrates the plate instead of resting a coincident face on it.
    """
    sink = -max(proud, radius) * 0.8
    if head == "bolt":
        # Six sections is a hex head; anything else is a dome with a flat top.
        sections, chamfer = 6, radius * 0.25
        profile = [(0.0, sink), (radius, sink), (radius, proud - chamfer),
                   (radius - chamfer, proud), (0.0, proud)]
    elif head == "pan":
        chamfer = min(radius, proud) * 0.45
        profile = [(0.0, sink), (radius, sink), (radius, proud - chamfer),
                   (radius - chamfer, proud), (0.0, proud)]
    else:
        t = np.linspace(0.0, np.pi / 2, rings + 1)[1:]
        profile = [(0.0, sink), (radius, sink)]
        profile += [(float(radius * np.cos(a)), float(proud * np.sin(a))) for a in t]
    return _revolve(profile, sections)


def _studs_at(points, radius, proud, direction=(0.0, 1.0, 0.0), head="dome",
              sections=8, jitter=0.0, seed=0):
    """A rivet at every point, standing out along `direction`.

    The chest's brass studs, generalised: any face of anything can be greebled
    by handing this a point set from `_line_points`, `_grid_points` or
    `_ring_points`.
    """
    rivet = _rivet(radius, proud, head, sections)
    rivet.apply_transform(_towards(direction))
    return _array(rivet, points, jitter, seed)


def _earclip(polygon) -> list[tuple[int, int, int]]:
    """Triangulate a simple polygon, concave ones included.

    `_prism` fan-triangulates and is convex-only, which is the documented reason
    `stairs` stacks boxes. A moulding profile is *never* convex — an ogee is an
    S — so a sweep needs this. Ears are only cut where the corner is strictly
    convex, so no degenerate triangle is ever emitted.
    """
    pts = np.asarray(polygon, float)
    n = len(pts)
    if n < 3:
        return []
    idx = list(range(n))
    area = float(np.sum(pts[:, 0] * np.roll(pts[:, 1], -1)
                        - np.roll(pts[:, 0], -1) * pts[:, 1]))
    if area < 0:
        idx.reverse()

    def _contains(a, b, c, p):
        # Barycentric sign test, strict, so a vertex lying on an ear's edge
        # does not veto it.
        v0, v1, v2 = c - a, b - a, p - a
        d = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(d) < _EPS:
            return False
        s = (v2[0] * v1[1] - v1[0] * v2[1]) / d
        t = (v0[0] * v2[1] - v2[0] * v0[1]) / d
        return s > 1e-12 and t > 1e-12 and s + t < 1 - 1e-12

    tris: list[tuple[int, int, int]] = []
    guard = 2 * n
    while len(idx) > 3 and guard > 0:
        guard -= 1
        for k in range(len(idx)):
            a, b, c = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            ab, bc = pts[b] - pts[a], pts[c] - pts[b]
            if ab[0] * bc[1] - ab[1] * bc[0] <= _EPS:
                continue
            if any(_contains(pts[a], pts[b], pts[c], pts[j])
                   for j in idx if j not in (a, b, c)):
                continue
            tris.append((a, b, c))
            idx.pop(k)
            guard = 2 * len(idx)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    elif len(idx) > 3:
        # Nothing left is a valid ear, which means the profile self-intersects.
        # A fan still closes the surface, which keeps the solid watertight.
        tris += _fan(idx)
    return tris


def _sweep(profile, path, closed=False, up=(0.0, 1.0, 0.0)):
    """Sweep a closed 2D profile along a 3D polyline, mitring every corner.

    This is the single mechanism behind most architectural detail: a cornice, a
    skirting, a window casing, a handrail, a plinth and a coping are all one
    profile taken round one path. `_prism` is the two-station case of it and
    `_revolve` the circular one.

    The profile lives in `(u, v)`, where `v` runs along `up` and `u` runs across
    the path. At a corner the u-axis is `(r0 + r1) / (1 + r0.r1)` — the vector
    whose projection on *both* segment normals is still 1, which is a true mitre
    rather than a smeared average, so a rectangular casing closes on itself with
    the profile's outer edge unbroken.
    """
    prof = np.asarray(profile, float)
    area = float(np.sum(prof[:, 0] * np.roll(prof[:, 1], -1)
                        - np.roll(prof[:, 0], -1) * prof[:, 1]))
    if area < 0:
        prof = prof[::-1]
    pts = np.asarray(path, float)
    m, n = len(prof), len(pts)
    if n < 2:
        raise ValueError("a sweep needs at least two path stations")
    up = np.asarray(up, float)
    up = up / np.linalg.norm(up)

    seg = np.roll(pts, -1, axis=0) - pts
    if not closed:
        seg = seg[:-1]
    seg = seg / np.linalg.norm(seg, axis=1)[:, None]

    verts: list = []
    rings: list[list[int]] = []
    for i in range(n):
        d0 = seg[i - 1] if (closed or i > 0) else seg[0]
        d1 = seg[i] if (closed or i < n - 1) else seg[-1]
        r0, r1 = np.cross(d0, up), np.cross(d1, up)
        n0, n1 = np.linalg.norm(r0), np.linalg.norm(r1)
        if n0 < _EPS or n1 < _EPS:
            raise ValueError("a sweep's path may not run along its up vector")
        r0, r1 = r0 / n0, r1 / n1
        denom = 1.0 + float(r0 @ r1)
        if denom < 1e-6:
            raise ValueError("a sweep's path doubles back on itself")
        across = (r0 + r1) / denom
        base = len(verts)
        verts.extend(pts[i] + u * across + v * up for u, v in prof)
        rings.append(list(range(base, base + m)))

    faces: list[tuple[int, int, int]] = []
    for i in range(n if closed else n - 1):
        a, b = rings[i], rings[(i + 1) % n]
        for j in range(m):
            k = (j + 1) % m
            faces += _quad(a[j], a[k], b[k], b[j])
    if not closed:
        cap = _earclip(prof)
        faces += [(rings[0][c], rings[0][b], rings[0][a]) for a, b, c in cap]
        faces += [(rings[-1][a], rings[-1][b], rings[-1][c]) for a, b, c in cap]
    return _finish(verts, faces)


def _rect_path(width, height, z=0.0):
    """The four corners of a rectangle in XY, anticlockwise from -X-Y.

    Anticlockwise with `up = +Z` puts a sweep's u-axis outward, which is what a
    casing wants: the profile grows away from the opening it frames.
    """
    x, y = width / 2.0, height / 2.0
    return [(-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)]


# Profiles are given as the *face* of the moulding: a run of points from the
# bottom of the projecting edge up to the back of the top, which the polygon
# closes down the u=0 back edge. u is how far it projects, v is its height.
_PROFILES = ("square", "bevel", "ovolo", "cavetto", "ogee", "step", "round")


def _moulding_profile(style, projection, height, steps=5):
    """A (u, v) polygon for one of the classical section shapes.

    A cornice, an architrave, a skirting, a plinth, a coping and a handrail are
    the same seven curves at different sizes; carrying them as a table is what
    makes one `moulding` kind cover all of them.
    """
    u, v = float(projection), float(height)
    t = np.linspace(0.0, 1.0, max(steps, 2) + 1)[1:]
    if style == "square":
        face = [(u, 0.0), (u, v)]
    elif style == "bevel":
        face = [(u, 0.0)]
    elif style == "ovolo":                      # convex quarter round
        face = [(u, 0.0)] + [(u * np.cos(a), v * np.sin(a))
                             for a in t * np.pi / 2]
    elif style == "cavetto":                    # concave quarter hollow
        face = [(u, 0.0)] + [(u * (1 - np.sin(a)), v * (1 - np.cos(a)))
                             for a in t * np.pi / 2]
    elif style == "ogee":                       # cyma recta, the S
        face = [(u, 0.0)] + [(u * (1 - (0.5 - 0.5 * np.cos(np.pi * a))), v * a)
                             for a in t]
    elif style == "step":                       # corbelled, for a crown
        face = [(u, 0.0), (u, 0.42 * v), (0.55 * u, 0.42 * v),
                (0.55 * u, 0.78 * v), (0.22 * u, 0.78 * v), (0.22 * u, v)]
    elif style == "round":                      # half-round bead or handrail
        # Sampled with a point *on* the crown, or the bead comes out shy of the
        # projection it was asked for and the run stops being dimensioned.
        half = np.linspace(0.0, 1.0, 2 * max(steps, 2) + 1)[:-1]
        face = [(u * np.sin(np.pi * a), v * (0.5 - 0.5 * np.cos(np.pi * a)))
                for a in half]
    else:
        raise ValueError(f"unknown moulding profile {style!r}, expected one of "
                         f"{list(_PROFILES)}")

    poly = [(0.0, 0.0)] + [(float(a), float(b)) for a, b in face] + [(0.0, v)]
    # Collinear or repeated points are what make an ear-clip emit a degenerate
    # triangle, so they are dropped here rather than guarded against later.
    out = [poly[0]]
    for p in poly[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def _frame(width, height, member, depth, chamfer=0.0, centre=(0.0, 0.0, 0.0),
           overlap=0.5):
    """Four bars making a rectangular ring in XY, `depth` thick along Z.

    The stiles run the full height and the rails stop short inside them, so no
    two corners coincide — `_combine` merges rather than unions, and touching
    corners are the one thing that would turn that into a non-manifold edge.
    """
    cx, cy, cz = centre
    parts = []
    for sx in (1, -1):
        parts.append(_box(member, height, depth, chamfer=chamfer,
                          center=(cx + (width - member) / 2 * sx, cy, cz)))
    rail = width - 2 * member * (1.0 - overlap)
    for sy in (1, -1):
        parts.append(_box(rail, member, depth, chamfer=chamfer,
                          center=(cx, cy + (height - member) / 2 * sy, cz)))
    return parts


def _subtract_spans(span, cuts):
    """What is left of the interval `span` after removing every interval in
    `cuts`. Used to stop a course of bricks running across a window."""
    out = [span]
    for c0, c1 in cuts:
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return out


def _courses(width, height, course, run, joint, stagger=0.5, keep_out=(),
             ragged=0.0, seed=0):
    """Lay a rectangle out in staggered courses of blocks.

    Returns `(cx, cy, w, h)` per block over a `width` x `height` rectangle
    centred on the origin. `keep_out` rectangles — a window, a door — cut the
    courses instead of being covered by them, which is the difference between a
    brick wall with a hole in it and a brick pattern painted over one.
    """
    rows = max(int(round(height / course)), 1)
    rng = np.random.default_rng(seed)
    if ragged:
        # Courses of one height and blocks of one length read as a printed
        # pattern however deep the joint is. Rubble varies both, and the
        # heights are renormalised so they still fill the wall exactly.
        heights = rng.uniform(1.0 - ragged, 1.0 + ragged, rows)
        heights *= height / heights.sum()
    else:
        heights = np.full(rows, height / rows)

    blocks = []
    y0 = -height / 2.0
    for r in range(rows):
        ch = float(heights[r])
        y1 = y0 + ch
        x = -width / 2.0 - (stagger * run if r % 2 else 0.0)
        if ragged:
            x -= float(rng.uniform(0.0, run * 0.5))
        cuts = [(k[0], k[1]) for k in keep_out
                if k[3] > y0 + _EPS and k[2] < y1 - _EPS]
        while x < width / 2.0 - _EPS:
            length = run * (1.0 + float(rng.uniform(-ragged, ragged))) if ragged else run
            span = (max(x, -width / 2.0), min(x + length, width / 2.0))
            for a, b in _subtract_spans(span, cuts):
                if b - a > joint * 1.5:
                    blocks.append(((a + b) / 2.0, y0 + ch / 2.0,
                                   b - a - joint, ch - joint))
            x += length
        y0 = y1
    return blocks


_SURFACES = ("flat", "brick", "block", "board")


def _face_relief(width, height, surface, course, joint, relief, chamfer,
                 seed=0, keep_out=(), normal=(0.0, 0.0, 1.0)):
    """Blocks tiling a rectangle in XY, standing `relief` proud along `normal`.

    Every block's outer face lands on the plane through the origin, and each
    reaches back twice its relief, so a caller translates the whole set to where
    the *finished* surface goes and lets the slab behind it overlap. That is
    what keeps a brick wall exactly as thick as it was asked to be: the relief
    is recessed into the wall, not added on top of it.
    """
    if surface == "flat" or relief <= _EPS:
        return []
    if surface == "board":
        # Vertical boarding: one plank per bay with a V-joint between, cut
        # around an aperture above and below rather than stopping at it.
        count = max(int(round(width / course)), 1)
        bw = width / count
        blocks = []
        for i in range(count):
            cx = -width / 2 + bw * (i + 0.5)
            cuts = [(k[2], k[3]) for k in keep_out
                    if k[1] > cx - bw / 2 + _EPS and k[0] < cx + bw / 2 - _EPS]
            for a, b in _subtract_spans((-height / 2, height / 2), cuts):
                if b - a > joint * 1.5:
                    blocks.append((cx, (a + b) / 2, bw - joint, b - a))
    elif surface == "block":
        blocks = _courses(width, height, course * 1.35, course * 2.1, joint, 0.5,
                          keep_out, ragged=0.34, seed=seed)
    else:
        blocks = _courses(width, height, course, course * 2.3, joint, 0.5,
                          keep_out, seed=seed)

    rng = np.random.default_rng(seed + 1)
    depth = relief * 2.4
    parts = []
    for i, (cx, cy, bw, bh) in enumerate(blocks):
        # A course of stones all sitting at exactly one depth reads as a printed
        # pattern; a tenth of the relief in variation reads as masonry.
        d = depth - (float(rng.uniform(0.0, relief * 0.7)) if surface == "block" else 0.0)
        parts.append(_box(bw, bh, d, chamfer=min(chamfer, bw * 0.2, bh * 0.2,
                                                 relief * 0.6),
                          center=(cx, cy, -d / 2.0)))
    if tuple(np.asarray(normal, float)) != (0.0, 0.0, 1.0):
        frame = _facing(normal)
        for p in parts:
            p.apply_transform(frame)
    return parts


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

# --- the detail dials, shared by every kind that carries surface relief ------
#
# One vocabulary across masonry, cladding and roofing, so a wall, a battlement
# and a chimney standing next to each other are the same stone rather than
# three different guesses at it.

SURFACE = Param(
    "surface", "choice", "brick",
    "Facing relief. brick = staggered courses; block = irregular ashlar with "
    "jittered depth; board = vertical planking; flat = a bare slab. The relief "
    "is recessed into the thickness, so a faced wall is exactly as thick as a "
    "flat one.",
    choices=_SURFACES,
)

COURSE = _studs(
    "course", 0.5,
    "Course height — board width for `board`. The wall is divided into a whole "
    "number of courses, so the value asked for is rounded to what fits.",
)

JOINT = _studs(
    "joint", 0.05,
    "Mortar joint between blocks. This is the shadow line that makes courses "
    "read as masonry rather than as one bumpy surface.",
)

RELIEF = _studs(
    "relief", 0.06, "How far the facing stands out of the recessed core.",
)

SEED = Param(
    "seed", "integer", 0,
    "Seeds the jitter in irregular facings and roofing. Changing it gives a "
    "different wall at the same parameters, which is the cheapest answer to "
    "'every crate looks like every other crate'.",
    unit="count", minimum=0, maximum=99999,
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
                sill_height, trim, trim_depth, surface, course, joint, relief,
                seed, chamfer):
    """A wall section, optionally with a window or a door cut through it, and
    optionally faced in courses of masonry.

    The opening is not subtracted — the wall is built as the four slabs that
    surround it. That is exact, needs no boolean engine, and leaves quads
    instead of the sliver triangles a mesh boolean puts around an aperture.

    The facing is **recessed rather than added**: the slab behind is thinned by
    the relief and the blocks stand back out to the requested thickness, so a
    brick wall is exactly as thick as a flat one. The courses are cut around the
    aperture rather than run across it, which is the difference between a wall
    with a hole in it and a brick pattern painted over one.
    """
    faced = surface != "flat" and relief > _EPS
    relief = min(relief, thickness * 0.3) if faced else 0.0
    core_t = thickness - 2 * relief

    def _face(keep_out):
        parts = []
        for sz in (1, -1):
            blocks = _face_relief(width, height, surface, course, joint, relief,
                                  chamfer, seed, keep_out,
                                  normal=(0.0, 0.0, float(sz)))
            for b in blocks:
                b.apply_translation((0.0, 0.0, sz * thickness / 2))
            parts += blocks
        return parts

    if opening == "none":
        wall = _box(width, height, core_t, chamfer=chamfer)
        return _combine([wall] + _face(())) if faced else wall

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
        parts.append(_box(side, height, core_t, chamfer=chamfer,
                          center=((width - side) / 2 * sx, 0.0, 0.0)))
    parts.append(_box(opening_width, above, core_t, chamfer=chamfer,
                      center=(0.0, height / 2 - above / 2, 0.0)))
    if below > _EPS:
        parts.append(_box(opening_width, below, core_t, chamfer=chamfer,
                          center=(0.0, -height / 2 + below / 2, 0.0)))
    if faced:
        parts += _face(((-opening_width / 2, opening_width / 2,
                         -height / 2 + below,
                         -height / 2 + below + opening_height),))

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


# --- the detailed kinds ------------------------------------------------------
#
# Everything below is an *assembly*: closed solids merged by `_combine`, each
# watertight on its own, interpenetrating rather than unioned. That is the same
# thing `crate` has always been and the same thing the chest is; it is written
# down here because these are the kinds where it stops being an implementation
# note and becomes the design. See `SINGLE_SOLID` at the bottom of the file for
# which kinds are one shell and which are not.

def _moulding(style, length, projection, height, returns, steps):
    """A profile swept along a run — cornice, skirting, plinth, handrail, band.

    One mechanism, most of the architectural vocabulary. `returns` mitres both
    ends back through the wall so the run stops with its own section showing,
    which is the difference between a cornice and a plank with a bevel.
    """
    profile = _moulding_profile(style, projection, height, steps)
    if not returns:
        return _sweep(profile, [(-length / 2, 0.0, 0.0), (length / 2, 0.0, 0.0)])
    a = length / 2 - projection
    if a <= _EPS:
        raise ValueError(
            f"projection {projection} is more than half the {length} run, so "
            f"the two returns meet in the middle"
        )
    return _sweep(profile, [(-a, 0.0, -projection), (-a, 0.0, 0.0),
                            (a, 0.0, 0.0), (a, 0.0, -projection)])


def _chord(offset, direction, half_width, half_height):
    """Where a line crosses a centred rectangle: (midpoint, length).

    Slab clipping, two axes. It is what lets a diagonal glazing bar stop at the
    frame instead of running out through it — the clip a boolean would do,
    written as four divisions.
    """
    lo, hi = -1e9, 1e9
    for axis, half in ((0, half_width), (1, half_height)):
        d, o = direction[axis], offset[axis]
        if abs(d) < 1e-9:
            if abs(o) > half:
                return None
            continue
        t0, t1 = (-half - o) / d, (half - o) / d
        lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
    if hi - lo <= 1e-6:
        return None
    mid = np.asarray(offset, float) + np.asarray(direction, float) * (lo + hi) / 2
    return mid, hi - lo


def _band_path(width, depth, y):
    """A rectangle in the XZ plane at height `y`, wound so a sweep's profile
    grows outward — the path a string course takes round a stack."""
    x, z = width / 2.0, depth / 2.0
    return [(-x, y, -z), (-x, y, z), (x, y, z), (x, y, -z)]


def _riveted_panel(width, height, thickness, style, panels_wide, panels_high,
                   relief, rivet_radius, rivet_pitch, ribs, corner_bosses,
                   chamfer):
    """An industrial plate: recessed bays, seams, ribs and rows of rivets.

    The detail decorators applied to the plainest possible shape. A `plank` is
    60 triangles and reads as a placeholder next to anything generated; the same
    slab with real seams and real rivet heads catches a highlight on every one
    of them and reads as fabricated. It is also the answer for hulls, hatches,
    boilers, containers and sci-fi walls, which is most of what a builder
    kitbashes out of flat plates.
    """
    relief = min(relief, thickness * 0.4)
    face = thickness / 2.0                       # where the finished skin lands
    parts = [_box(width, height, thickness - relief, chamfer=chamfer,
                  center=(0.0, 0.0, -relief / 2.0))]

    if style == "corrugated":
        # A constant-thickness wavy sheet: a non-convex closed section swept up
        # the panel. `_prism` could not take this section; `_earclip` can.
        n = max(int(round(width / max(rivet_pitch * 2, _EPS))), 2)
        sheet = relief * 0.9
        xs = np.linspace(-width / 2, width / 2, 2 * n + 1)
        crest = np.where(np.arange(2 * n + 1) % 2 == 0, 0.0, relief)
        top = [(float(x), float(face - relief + c)) for x, c in zip(xs, crest)]
        section = top + [(float(x), float(face - relief + c - sheet))
                         for x, c in zip(xs[::-1], crest[::-1])]
        parts.append(_sweep(section, [(0.0, -height / 2, 0.0),
                                      (0.0, height / 2, 0.0)],
                            up=(0.0, 0.0, 1.0)))
        return _combine(parts)

    margin = max(rivet_radius * 3.0, min(width, height) * 0.06)
    seam = max(rivet_radius * 2.0, relief * 1.5)
    bay_w = (width - 2 * margin - (panels_wide - 1) * seam) / panels_wide
    bay_h = (height - 2 * margin - (panels_high - 1) * seam) / panels_high
    if bay_w <= _EPS or bay_h <= _EPS:
        raise ValueError(
            f"{panels_wide}x{panels_high} bays do not fit on a {width}x{height} "
            f"panel once the {seam:.3f} seams are allowed for"
        )

    centres = _grid_points(
        panels_wide, panels_high, (0.0, 0.0, face - relief * 1.2),
        ((bay_w + seam) * (panels_wide - 1), 0.0, 0.0),
        (0.0, (bay_h + seam) * (panels_high - 1), 0.0))
    if style == "ribbed":
        # Ribs instead of bays: a vertical stiffener over every seam line.
        rib = max(seam * 1.6, relief * 2.0)
        for x in np.linspace(-width / 2 + margin, width / 2 - margin,
                             max(ribs, 2)):
            parts.append(_box(rib, height - margin, relief * 2.4, chamfer=chamfer,
                              center=(float(x), 0.0, face - relief * 1.2)))
    else:
        for c in centres:
            parts.append(_box(bay_w, bay_h, relief * 2.4, chamfer=chamfer,
                              center=tuple(c)))

    # Rivets on the flange the bays leave exposed, plus one at each corner of
    # each bay: a line of heads is what makes a seam read as two plates joined.
    proud = min(relief * 0.9, rivet_radius * 1.1)
    z = face - relief
    for sy in (1, -1):
        n = max(int(round((width - 2 * margin) / rivet_pitch)) + 1, 2)
        parts += _studs_at(
            _line_points(n, (-width / 2 + margin * 0.45, sy * (height - margin) / 2, z),
                         (width / 2 - margin * 0.45, sy * (height - margin) / 2, z)),
            rivet_radius, proud, (0.0, 0.0, 1.0))
    for sx in (1, -1):
        n = max(int(round((height - 2 * margin) / rivet_pitch)) - 1, 2)
        parts += _studs_at(
            _line_points(n, (sx * (width - margin) / 2, -height / 2 + margin * 1.3, z),
                         (sx * (width - margin) / 2, height / 2 - margin * 1.3, z)),
            rivet_radius, proud, (0.0, 0.0, 1.0))
    if corner_bosses:
        for c in centres:
            parts += _studs_at(
                _grid_points(2, 2, (c[0], c[1], z),
                             (bay_w + seam * 0.7, 0.0, 0.0),
                             (0.0, bay_h + seam * 0.7, 0.0)),
                rivet_radius * 1.15, proud, (0.0, 0.0, 1.0), head="bolt")
    return _combine(parts)


def _window(width, height, depth, style, lights_wide, lights_high, frame, bar,
            sill, hood, glazed, chamfer, sections):
    """A framed window: casing, mullions and transoms, sill and hood.

    The frame is recessed and the sill and hood project to the full depth, so
    the part is exactly as deep as it was asked to be *and* the sill still
    throws a shadow line — the one detail that reads as a window from across a
    street rather than as a hole with a border.
    """
    if 2 * frame + bar >= min(width, height):
        raise ValueError(
            f"a {frame} frame leaves no light in a {width}x{height} window")
    band = depth * 0.42 if (sill or hood) else 0.0
    core = depth - band
    z = -band / 2.0                              # centre of the recessed frame

    parts = _frame(width, height, frame, core, chamfer, (0.0, 0.0, z))
    light_w = width - 2 * frame
    light_h = height - 2 * frame

    if style == "arched":
        # A round head. The arch ring is a rectangular section swept round a
        # semicircle — the same `_sweep` a cornice uses, on a curved path —
        # which is exact where a ring of straight boxes is a polygon.
        r = light_w / 2 - frame
        if r <= _EPS:
            raise ValueError(
                f"a {frame} frame leaves no room for an arched head in a "
                f"{width}-wide window")
        springing = height / 2 - frame - light_w / 2
        if springing <= -height / 2 + frame * 2:
            raise ValueError(
                f"a {width}x{height} window is too squat for a round head")
        arc = [(float(np.cos(a) * r), float(springing + np.sin(a) * r), z)
               for a in np.linspace(0.0, np.pi, max(sections, 12) + 1)]
        parts.append(_sweep(_moulding_profile("square", frame, core), arc,
                            up=(0.0, 0.0, 1.0)))
        parts.append(_box(light_w, frame * 0.8, core, chamfer=chamfer,
                          center=(0.0, springing, z)))
        # Bars radiating from the springing centre, plus mullions below it.
        for i in range(1, max(lights_wide * 2, 2)):
            a = np.pi * i / max(lights_wide * 2, 2)
            spoke = _box(r, bar, core * 0.85, chamfer=min(chamfer, bar * 0.3))
            spoke.apply_transform(trimesh.transformations.rotation_matrix(
                a, (0.0, 0.0, 1.0)))
            spoke.apply_translation((np.cos(a) * r / 2,
                                     springing + np.sin(a) * r / 2, z))
            parts.append(spoke)
        below = springing - (-height / 2 + frame)
        for i in range(1, lights_wide):
            x = -light_w / 2 + light_w * i / lights_wide
            parts.append(_box(bar, below, core, chamfer=chamfer,
                              center=(x, springing - below / 2, z)))
        for i in range(1, lights_high):
            parts.append(_box(light_w, bar, core, chamfer=chamfer,
                              center=(0.0, springing - below * i / lights_high, z)))
    elif style == "lattice":
        # Diamond leading. Each lead is clipped to the light by `_chord`, which
        # is what a boolean would have been for — and the clip allows for the
        # bar's own thickness, so a diagonal still stops at the frame.
        pitch = max(light_w, light_h) / max(lights_wide * 2, 2)
        for sign in (1.0, -1.0):
            d = np.array([np.cos(np.pi / 4), sign * np.sin(np.pi / 4), 0.0])
            n = int((light_w + light_h) / (pitch * 1.42)) + 1
            hw = light_w / 2 - bar * abs(d[1]) / 2
            hh = light_h / 2 - bar * abs(d[0]) / 2
            for i in range(-n, n + 1):
                offset = np.array([-d[1], d[0], 0.0]) * i * pitch
                hit = _chord(offset, d, hw, hh)
                if hit is None:
                    continue
                mid, span = hit
                lead = _box(span, bar, core * 0.7, chamfer=min(chamfer, bar * 0.3))
                lead.apply_transform(trimesh.transformations.rotation_matrix(
                    np.arctan2(d[1], d[0]), (0.0, 0.0, 1.0)))
                lead.apply_translation((mid[0], mid[1], z))
                parts.append(lead)
    else:
        for i in range(1, lights_wide):
            x = -light_w / 2 + light_w * i / lights_wide
            parts.append(_box(bar, light_h, core, chamfer=chamfer,
                              center=(x, 0.0, z)))
        for i in range(1, lights_high):
            y = -light_h / 2 + light_h * i / lights_high
            parts.append(_box(light_w, bar, core, chamfer=chamfer,
                              center=(0.0, y, z)))

    # A bead round the light on both faces. Two more sweeps, 100 triangles, and
    # the frame stops being four boxes and starts being a rebated section.
    for sz in (1, -1):
        # Anticlockwise about the sweep's own up vector is what puts the
        # profile outside the light rather than across it, so the -Z bead runs
        # the other way round.
        ring = _rect_path(light_w, light_h, z + sz * core * (0.5 - 0.28))
        parts.append(_sweep(
            _moulding_profile("ovolo", frame * 0.55, core * 0.28),
            ring if sz > 0 else ring[::-1],
            closed=True, up=(0.0, 0.0, float(sz))))

    if glazed:
        parts.append(_box(light_w, light_h, core * 0.18, chamfer=0.0,
                          center=(0.0, 0.0, z - core * 0.3)))
    if sill:
        # Projects forward to the full depth and stands on the bottom rail, so
        # the light above it is not shortened.
        parts.append(_sweep(_moulding_profile("ogee", band, frame * 1.1),
                            [(-width / 2, -height / 2, depth / 2 - band),
                             (width / 2, -height / 2, depth / 2 - band)],
                            up=(0.0, 1.0, 0.0)))
    if hood:
        parts.append(_sweep(_moulding_profile("step", band, frame * 0.95),
                            [(-width / 2, height / 2 - frame * 0.95,
                              depth / 2 - band),
                             (width / 2, height / 2 - frame * 0.95,
                              depth / 2 - band)],
                            up=(0.0, 1.0, 0.0)))
    return _combine(parts)


def _panel_door(width, height, thickness, style, panels_wide, panels_high,
                stile, rail, relief, studs, straps, handle, chamfer, sections):
    """A door with joinery: stiles, rails, muntins, panels and ironmongery.

    Three doors in one kind, because they are the same frame with different
    infill: `panelled` is joinery, `plank` is boards on ledges, `banded` is a
    plank door under strap hinges and clavos, which is every castle door in
    every Roblox dungeon.
    """
    core = thickness * 0.62
    face = thickness / 2.0
    relief = min(relief, (thickness - core) / 2.0)
    straps = straps or style == "banded"
    studs = studs or style == "banded"
    parts = []

    if style == "panelled":
        if 2 * stile + (panels_wide - 1) * stile >= width:
            raise ValueError(
                f"a {stile} stile leaves no panel in a {width} door")
        parts.append(_box(width, height, core, chamfer=chamfer))
        bay_w = (width - (panels_wide + 1) * stile) / panels_wide
        bay_h = (height - (panels_high + 1) * rail) / panels_high
        if bay_w <= _EPS or bay_h <= _EPS:
            raise ValueError(
                f"{panels_wide}x{panels_high} panels do not fit a "
                f"{width}x{height} door with a {stile} stile and a {rail} rail")
        for i in range(panels_wide):
            for j in range(panels_high):
                cx = -width / 2 + stile + bay_w / 2 + i * (bay_w + stile)
                cy = -height / 2 + rail + bay_h / 2 + j * (bay_h + rail)
                for sz in (1, -1):
                    # A bolection moulding round each panel, then the raised
                    # field inside it: the two together are what make a panel
                    # read as joinery instead of as a rectangle scored in a slab.
                    base = core / 2 - relief * 0.2
                    ring = _rect_path(bay_w, bay_h, sz * base)
                    bolection = _sweep(
                        _moulding_profile("ovolo", stile * 0.55, face - base),
                        ring if sz > 0 else ring[::-1],
                        closed=True, up=(0.0, 0.0, float(sz)))
                    bolection.apply_translation((cx, cy, 0.0))
                    parts.append(bolection)
                    parts.append(_box(bay_w * 0.92, bay_h * 0.95, relief * 1.4,
                                      chamfer=min(chamfer, relief * 0.5),
                                      center=(cx, cy,
                                              sz * (core / 2 + relief * 0.2))))
    else:
        boards = max(int(round(width / max(stile * 2.4, _EPS))), 2)
        bw = width / boards
        # The boards carry the front face and the sheathing behind them carries
        # the back one, so the V-joints between boards are real gaps rather
        # than scored lines and the slab is still exactly `thickness` deep.
        # Where there is ironwork the boards drop back by its relief, or the
        # straps sit flush with the planking and vanish into it.
        front = face - (relief if (straps or studs or handle) else 0.0)
        parts.append(_box(width, height, core * 0.8, chamfer=chamfer,
                          center=(0.0, 0.0, -face + core * 0.4)))
        for i in range(boards):
            parts.append(_box(bw * 0.94, height, core, chamfer=chamfer,
                              center=(-width / 2 + bw * (i + 0.5), 0.0,
                                      front - core / 2)))
        # Ledges and a brace on the back — the carpentry that stops a plank
        # door being a fence panel. Both sit against the back face rather than
        # outside it, so the door is exactly as thick as it was asked to be.
        for sy in (1, -1):
            parts.append(_box(width * 0.94, rail, relief * 2.0, chamfer=chamfer,
                              center=(0.0, sy * (height / 2 - rail),
                                      -(face - relief))))
        brace = float(np.hypot(width * 0.9, height - 3 * rail))
        angle = float(np.arctan2(height - 3 * rail, width * 0.9))
        bar = _box(brace, rail * 0.8, relief * 1.8, chamfer=chamfer)
        bar.apply_transform(trimesh.transformations.rotation_matrix(
            angle, (0.0, 0.0, 1.0)))
        bar.apply_translation((0.0, 0.0, -(face - relief * 0.9)))
        parts.append(bar)

    if straps:
        # Strap hinges across the face, each carrying its own row of rivets.
        for sy in (1, -1):
            y = sy * (height / 2 - rail * 1.6)
            parts.append(_box(width * 0.66, rail * 0.75, relief * 1.5,
                              chamfer=chamfer,
                              center=(-width * 0.14, y, face - relief * 0.75)))
            proud = relief * 0.7
            parts += _studs_at(
                _line_points(5, (-width / 2 + rail * 0.9, y, face - proud),
                             (width * 0.16, y, face - proud)),
                rail * 0.16, proud, (0.0, 0.0, 1.0), head="dome")
            parts.append(_cylinder(rail * 0.42, rail * 1.9, 0.0, rail * 0.1,
                                   max(sections // 3, 6)))
            parts[-1].apply_translation((-width / 2 + rail * 0.42, y, 0.0))
    if studs:
        cols = max(int(round(width / (rail * 2.2))), 2)
        rows = max(int(round(height / (rail * 2.6))), 2)
        parts += _studs_at(
            _grid_points(cols, rows, (0.0, 0.0, face - relief),
                         (width - rail * 2.2, 0.0, 0.0),
                         (0.0, height - rail * 3.4, 0.0)),
            rail * 0.2, relief, (0.0, 0.0, 1.0), head="dome")
    if handle:
        ring = _revolve(
            [(rail * 0.5, -rail * 0.11), (rail * 0.72, -rail * 0.11),
             (rail * 0.72, rail * 0.11), (rail * 0.5, rail * 0.11)],
            max(sections, 12))
        ring.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, (1.0, 0.0, 0.0)))
        # On the stile, not on a panel: a ring lying in the middle of a raised
        # panel field is buried by it and reads as a scratch.
        hx = width / 2 - max(stile, rail) * 0.8
        ring.apply_translation((hx, 0.0, face - rail * 0.11))
        parts.append(ring)
        parts.append(_box(rail * 0.85, rail * 1.5, (face - core / 2) * 1.6,
                          chamfer=chamfer,
                          center=(hx, rail * 0.6, core / 2)))
    return _combine(parts)


def _voussoirs(span, rise, thickness, depth, count, keystone, chamfer, z=0.0,
               key_depth=None, key_z=0.0):
    """The wedge blocks of a round arch, as exact trapezoids, with joints.

    `hollow._arch` overlaps neighbouring blocks so no two corners coincide, and
    the ring comes out as one smooth band — from three metres it reads as bent
    metal, not as masonry. Here the blocks are *separated* by a joint and a
    continuous soffit ring runs behind them, so every joint throws a shadow and
    none of them is a hole. The keystone stands proud of the face rather than
    proud of the extrados, which is where a hood mould has to go.
    """
    r, R = span / 2.0, span / 2.0 + thickness
    step = np.pi / count
    gap = step * 0.07
    k = rise / R
    # The soffit: a thin continuous ring the joints show, rather than daylight.
    arc = [(float(np.cos(a) * r), float(np.sin(a) * r * k), z - depth / 2)
           for a in np.linspace(0.0, np.pi, max(count * 3, 12) + 1)]
    blocks = [_sweep(_moulding_profile("square", thickness * 0.34, depth), arc,
                     up=(0.0, 0.0, 1.0))]
    for i in range(count):
        a0 = i * step + (0.0 if i == 0 else gap / 2)
        a1 = (i + 1) * step - (0.0 if i == count - 1 else gap / 2)
        d, dz = depth, z
        if keystone and 2 * i + 1 == count:
            a0, a1 = a0 - step * 0.24, a1 + step * 0.24
            d, dz = (key_depth or depth), key_z
        polygon = [(np.cos(a) * q, np.sin(a) * q * k)
                   for a, q in ((a0, r), (a0, R), (a1, R), (a1, r))]
        block = _prism(polygon, d, chamfer)
        block.apply_transform(trimesh.transformations.rotation_matrix(
            np.pi / 2, (0.0, 1.0, 0.0)))
        block.apply_translation((0.0, 0.0, dz))
        blocks.append(block)
    return blocks


def _archway(width, height, depth, pier, rise, voussoirs, keystone, impost,
             hood, surface, course, joint, relief, seed, chamfer):
    """A gateway with an order: plinth, impost band, voussoir ring, hood mould.

    `hollow.arch` is the same silhouette with none of the mouldings, and the
    difference is the point of this whole file — the ring, the keystone, the
    band the arch springs from and the archivolt over it are what a mason would
    have built, and each one is a sweep or an array.
    """
    # Every projecting member — plinth, impost, archivolt — comes out of the
    # depth rather than being added to it, and the hood mould comes out of the
    # width and the rise, so the whole gateway is exactly the box asked for.
    band = depth * 0.18 if (impost or hood) else 0.0
    hood_w = pier * 0.34 if hood else 0.0
    span = width - 2 * pier - 2 * hood_w
    if span <= _EPS:
        raise ValueError(f"pier {pier} leaves no opening in a {width} archway")
    ring_rise = rise - hood_w
    pier_h = height - rise
    if ring_rise <= _EPS or pier_h <= pier:
        raise ValueError(f"rise {rise} does not fit under a {height} archway")

    core = depth - band
    z = -band / 2.0
    front = depth / 2.0 - band
    jamb = pier + hood_w                      # opening edge to the outer face

    parts = []
    for sx in (1, -1):
        x = sx * (width - jamb) / 2
        parts.append(_box(jamb, pier_h, core - 2 * relief, chamfer=chamfer,
                          center=(x, -height / 2 + pier_h / 2, z)))
        for sz in (1, -1):
            blocks = _face_relief(jamb, pier_h, surface, course, joint, relief,
                                  chamfer, seed, (), normal=(0.0, 0.0, float(sz)))
            for b in blocks:
                b.apply_translation((x, -height / 2 + pier_h / 2,
                                     z + sz * core / 2))
            parts += blocks

    # The keystone spans the whole depth, so it stands `band` proud of the
    # arch face — where a hood mould leaves it room and a radial projection
    # would collide with the archivolt.
    ring = _voussoirs(span, ring_rise, pier, core, voussoirs, keystone, chamfer,
                      z, key_depth=depth, key_z=0.0)
    for v in ring:
        v.apply_translation((0.0, -height / 2 + pier_h, 0.0))
    parts += ring

    if impost:
        for sx in (1, -1):
            for style, h, y in (
                    ("step", pier * 0.34, -height / 2 + pier_h - pier * 0.34),
                    ("ovolo", pier * 0.32, -height / 2)):
                # Always run the path along +X: a sweep's profile grows to the
                # left of its direction of travel, so a run laid out backwards
                # projects into the wall instead of out of it.
                x0, x1 = sorted((sx * (width / 2 - jamb * 1.15), sx * width / 2))
                parts.append(_sweep(_moulding_profile(style, band, h),
                                    [(x0, y, front), (x1, y, front)],
                                    up=(0.0, 1.0, 0.0)))
    if hood:
        # An archivolt: the same profile taken round the extrados, which is a
        # polyline like any other as far as `_sweep` is concerned. Wound from
        # 0 to pi so its u-axis points radially out of the ring.
        R = span / 2 + pier
        k = ring_rise / R
        n = max(voussoirs * 2, 8)
        arc = [(float(np.cos(a) * R),
                float(-height / 2 + pier_h + np.sin(a) * R * k), front)
               for a in np.linspace(0.0, np.pi, n + 1)]
        parts.append(_sweep(_moulding_profile("round", hood_w, band), arc,
                            up=(0.0, 0.0, 1.0)))
    return _combine(parts)


def _battlement(width, height, thickness, merlon_width, crenel_width,
                merlon_height, coping, corbel, arrow_slit, surface, course,
                joint, relief, seed, chamfer):
    """A crenellated parapet: merlons, crenels, coping, corbel table, slits.

    The wall below the merlons is faced with the same course machinery a
    `wall_panel` uses, so a battlement and the wall under it are the same
    masonry rather than two different ideas about stone.
    """
    body_h = height - merlon_height
    if body_h <= _EPS:
        raise ValueError(
            f"merlon_height {merlon_height} is the whole {height} wall")
    # The corbels and the copings are what reach the requested thickness; the
    # masonry face is set back behind them, which is both how a parapet is
    # actually built and what keeps the envelope exact.
    proud = min(relief * 1.8, thickness * 0.14) if (corbel or coping) else 0.0
    skin = thickness - 2 * proud
    relief = min(relief, skin * 0.24)
    core = skin - 2 * relief
    parts = [_box(width, body_h, core, chamfer=chamfer,
                  center=(0.0, -height / 2 + body_h / 2, 0.0))]

    corbel_h = min(body_h * 0.16, thickness * 0.9)
    faced_h = body_h - (corbel_h if corbel else 0.0)
    for sz in (1, -1):
        blocks = _face_relief(width, faced_h, surface, course, joint, relief,
                              chamfer, seed, (), normal=(0.0, 0.0, float(sz)))
        for b in blocks:
            b.apply_translation((0.0, -height / 2 + faced_h / 2, sz * skin / 2))
        parts += blocks

    if corbel:
        # A corbel table under the parapet: the single most recognisable
        # medieval detail, and it is one array of one small step moulding.
        n = max(int(round(width / (thickness * 1.1))), 2)
        for sz in (1, -1):
            block = _sweep(
                _moulding_profile("step", proud + relief, corbel_h),
                [(-thickness * 0.30, 0.0, 0.0), (thickness * 0.30, 0.0, 0.0)],
                up=(0.0, 1.0, 0.0))
            if sz < 0:
                block.apply_transform(trimesh.transformations.rotation_matrix(
                    np.pi, (0.0, 1.0, 0.0)))
            parts += _array(block, _line_points(
                n, (-width / 2 + thickness * 0.6,
                    -height / 2 + body_h - corbel_h, sz * (skin / 2 - relief)),
                (width / 2 - thickness * 0.6,
                 -height / 2 + body_h - corbel_h, sz * (skin / 2 - relief))))

    pitch = merlon_width + crenel_width
    count = max(int(round((width + crenel_width) / pitch)), 1)
    mw = (width - (count - 1) * crenel_width) / count
    cap = min(merlon_height * 0.24, proud * 3.0) if coping else 0.0
    for i in range(count):
        cx = -width / 2 + mw / 2 + i * (mw + crenel_width)
        mh = merlon_height - cap
        if arrow_slit and mw > thickness * 1.2:
            # A slit is two piers and a head, the same construction
            # `wall_panel` uses for a window: exact, and no boolean.
            slit = min(mw * 0.14, thickness * 0.5)
            for sx in (1, -1):
                parts.append(_box((mw - slit) / 2, mh, skin, chamfer=chamfer,
                                  center=(cx + sx * (mw + slit) / 4,
                                          -height / 2 + body_h + mh / 2, 0.0)))
            parts.append(_box(slit * 1.2, mh * 0.28, skin, chamfer=chamfer,
                              center=(cx, -height / 2 + body_h + mh * 0.86, 0.0)))
        else:
            parts.append(_box(mw, mh, skin, chamfer=chamfer,
                              center=(cx, -height / 2 + body_h + mh / 2, 0.0)))
        if coping:
            coping_cap = _sweep(
                _moulding_profile("bevel", proud, cap),
                _band_path(mw - 2 * proud, skin, height / 2 - cap),
                closed=True, up=(0.0, 1.0, 0.0))
            coping_cap.apply_translation((cx, 0.0, 0.0))
            parts.append(coping_cap)
    return _combine(parts)


def _roof(width, depth, height, style, course, tile_width, gable, jitter, seed,
          chamfer):
    """A gabled roof clad in overlapping courses, with a ridge and eaves.

    The clue that a roof is a roof is the horizontal shadow line every course
    throws, so the tiles are laid as real overlapping plates rather than scored
    into a slab. Eleven courses of twelve tiles is about 3 000 triangles — a
    seventh of what one `MeshPart` may spend, for the single most legible
    surface on any building.

    The ridge cap and the eaves fascia are structure rather than options: they
    are the two members that land on the requested envelope, and the cladding
    behind them is deliberately set back inside it.
    """
    slope = float(np.hypot(depth / 2.0, height))
    pitch = float(np.arctan2(height, depth / 2.0))
    thick = min(course * 0.32, height * 0.12)
    ridge_h = min(course * 0.55, height * 0.16)
    eaves_h = min(course * 0.6, height * 0.2)
    normal = np.array([0.0, np.cos(pitch), np.sin(pitch)])

    # The carcass is the roof triangle pulled in along both slope normals, so
    # the cladding laid on it lands exactly on the requested envelope instead
    # of standing proud of it.
    inset = 5.4 * thick / max(np.cos(pitch), 0.15)
    parts = [_prism([(-depth / 2 + inset, -height / 2),
                     (depth / 2 - inset, -height / 2),
                     (0.0, height / 2 - inset)], width)]

    lap = 0.0 if style == "corrugated" else 0.62

    def _tilt(step):
        """How far a tile is canted out of the slope, and how deep that puts
        its tail. A tile's tail rests on the head of the one below it, so the
        cant is one tile thickness over the tile's lapped length."""
        angle = float(np.arcsin(min(0.55, 2.0 * thick / (step * (1 + lap)))))
        return angle, step / 2 * np.sin(angle)

    # The bottom course has to start far enough up the slope that its *back*
    # lower corner still clears the eaves line, or a roof quietly grows below
    # the height it was asked for. Solved twice because the clearance depends
    # on the course spacing and the spacing depends on the clearance.
    start = eaves_h * 0.5
    for _ in range(2):
        clear = (2.15 * thick + 2 * _tilt(max(start, course))[1]) \
            / max(np.tan(pitch), 0.2)
        start = max(eaves_h * 0.5, clear)
        rows = max(int(round((slope - ridge_h - start) / course)), 1)
        step = (slope - ridge_h - start) / rows
        clear = (2.15 * thick + 2 * _tilt(step)[1]) / max(np.tan(pitch), 0.2)
        start = max(eaves_h * 0.5, clear)
    rows = max(int(round((slope - ridge_h - start) / course)), 1)
    step = (slope - ridge_h - start) / rows
    # Staggered courses leave a stepped verge if they run to the very edge, so
    # the cladding stops short and the gable board takes the last inch. That is
    # also what a barge board is for.
    # The cladding sits a quarter of a tile below the roof plane so the gable
    # triangle, which *is* the plane, stands proud of it as a barge board.
    # Coplanar with the tiles it z-fought and the verge came out serrated.
    verge = thick * 1.8 if gable else 0.0
    clad = width - 2 * verge
    cols = max(int(round(clad / tile_width)), 1)
    tw = clad / cols
    tilt = _tilt(step)[0]
    for sz in (1, -1):
        for r in range(rows):
            # The tile's head is under the course above; only its tail shows,
            # and that exposure is what the eye reads as a course.
            length = step * (1.0 + (lap if r else 0.0))
            s = start + step * (r + 1) - length / 2
            # Each tile is canted so its tail lifts clear of the course below
            # and its head buries under the course above. Laid dead flat they
            # are all coplanar and the roof reads as one sanded plane with
            # lines scored on it; the cant is what puts a shadow under every
            # course, which is the only thing that says "roof" from a distance.
            lift = length / 2 * np.sin(tilt)
            centre = np.array([0.0, -height / 2, sz * depth / 2]) \
                + np.array([0.0, np.sin(pitch), -sz * np.cos(pitch)]) * s \
                - np.array([0.0, normal[1], sz * normal[2]]) * (thick * 1.15 + lift)
            offset = (0.5 if (r % 2 and style != "corrugated") else 0.0) * tw
            for c in range(cols + 1):
                x = -clad / 2 + tw * c + offset
                w = tw
                if x - tw / 2 < -clad / 2:
                    w = tw / 2 + (x + clad / 2)
                    x = -clad / 2 + w / 2
                if x + w / 2 > clad / 2:
                    w = clad / 2 - (x - w / 2)
                    x = clad / 2 - w / 2
                if w <= tw * 0.08:
                    continue
                tile = _box(w * 0.96, length, thick * 2.0,
                            chamfer=min(chamfer, thick * 0.5, w * 0.2))
                # Lie the tile *in* the slope: its length up the slope, its
                # thickness along the slope normal. The other sign of this
                # rotation stands every tile on end and the roof comes out as
                # a louvre.
                tile.apply_transform(trimesh.transformations.rotation_matrix(
                    -sz * (np.pi / 2 - pitch + tilt), (1.0, 0.0, 0.0)))
                # Weathering only ever sinks a tile. Lifting one would put it
                # outside the envelope the caller asked for, which is the one
                # thing a kit part may not do.
                d = float(np.random.default_rng(seed + r * 97 + c).uniform(
                    0.0, jitter)) if jitter else 0.0
                tile.apply_translation((x + centre[0], centre[1] - d, centre[2]))
                parts.append(tile)

    # The cap has to be wide enough to bury the last course on both slopes, or
    # the top tiles poke through it and the ridge line comes out serrated.
    cap_w = max(ridge_h * 2.2, 2.6 * step * np.cos(pitch) + 2.5 * thick)
    parts.append(_sweep(
        _moulding_profile("round", cap_w, ridge_h * 1.15),
        [(-width / 2, height / 2 - ridge_h * 1.15, -cap_w / 2),
         (width / 2, height / 2 - ridge_h * 1.15, -cap_w / 2)],
        up=(0.0, 1.0, 0.0)))
    for sz in (1, -1):
        parts.append(_box(width, eaves_h, thick * 1.8, chamfer=chamfer,
                          center=(0.0, -height / 2 + eaves_h / 2,
                                  sz * (depth / 2 - thick * 0.9))))
    if gable:
        for sx in (1, -1):
            end = _prism([(-depth / 2, -height / 2), (depth / 2, -height / 2),
                          (0.0, height / 2)], verge, chamfer)
            end.apply_translation((sx * (width - verge) / 2, 0.0, 0.0))
            parts.append(end)
    return _combine(parts)


def _railing(length, height, depth, style, baluster_count, post_count,
             rail_height, newel, chamfer, sections):
    """A balustrade: newels, a moulded handrail, a bottom rail and balusters."""
    post = depth * 0.85
    parts = []
    top = height / 2 - rail_height / 2
    parts.append(_sweep(_moulding_profile("round", depth, rail_height),
                        [(-length / 2, height / 2 - rail_height, -depth / 2),
                         (length / 2, height / 2 - rail_height, -depth / 2)],
                        up=(0.0, 1.0, 0.0)))
    bottom_y = -height / 2 + rail_height * 0.45
    parts.append(_box(length, rail_height * 0.9, depth * 0.75, chamfer=chamfer,
                      center=(0.0, bottom_y, 0.0)))

    posts = _line_points(max(post_count, 2), (-length / 2 + post / 2, 0.0, 0.0),
                         (length / 2 - post / 2, 0.0, 0.0))
    if newel:
        cap_proud = post * 0.16
        for p in posts:
            parts.append(_box(post - cap_proud, height - rail_height * 1.2,
                              post - cap_proud, chamfer=chamfer,
                              center=(p[0], -rail_height * 0.6, 0.0)))
            # A moulded cap, because a newel that just stops is a fencepost.
            # Its band is pulled in by its own projection so the cap lands on
            # the post's face rather than outside the run.
            cap = _sweep(_moulding_profile("ovolo", cap_proud, rail_height * 0.8),
                         _band_path(post - 2 * cap_proud, post - 2 * cap_proud,
                                    height / 2 - rail_height * 1.4),
                         closed=True, up=(0.0, 1.0, 0.0))
            cap.apply_translation((p[0], 0.0, 0.0))
            parts.append(cap)

    clear = height - rail_height * 2.2
    span = length - (post * 1.4 if newel else 0.0)
    n = max(baluster_count, 1)
    for i, p in enumerate(_line_points(n, (-span / 2 + span / (2 * n), 0.0, 0.0),
                                       (span / 2 - span / (2 * n), 0.0, 0.0))):
        y = -height / 2 + rail_height * 0.9 + clear / 2
        if style == "square":
            parts.append(_box(depth * 0.42, clear, depth * 0.42, chamfer=chamfer,
                              center=(p[0], y, 0.0)))
        elif style == "lattice":
            # Clipped to the bay the same way the window's leading is, so the
            # crossings stop at the newels instead of poking out of the run.
            t = depth * 0.3
            for sign in (1, -1):
                d = np.array([span / n * 2, sign * clear, 0.0])
                d /= np.linalg.norm(d)
                hit = _chord((p[0], 0.0, 0.0), d,
                             span / 2 - t * abs(d[1]) / 2, clear / 2)
                if hit is None:
                    continue
                mid, length = hit
                bar = _box(length, t, depth * 0.34, chamfer=chamfer)
                bar.apply_transform(trimesh.transformations.rotation_matrix(
                    float(np.arctan2(d[1], d[0])), (0.0, 0.0, 1.0)))
                bar.apply_translation((mid[0], y + mid[1], 0.0))
                parts.append(bar)
        else:
            # A turned baluster: a vase profile on the lathe, which is exactly
            # what `_revolve` is and costs 200 triangles.
            r = depth * 0.22
            profile = [(0.0, -clear / 2), (r * 1.5, -clear / 2),
                       (r * 1.5, -clear / 2 + r * 0.5), (r * 0.75, -clear / 2 + r * 1.1),
                       (r * 1.35, -clear / 2 + clear * 0.30),
                       (r * 1.05, -clear / 2 + clear * 0.52),
                       (r * 0.62, -clear / 2 + clear * 0.70),
                       (r * 0.95, -clear / 2 + clear * 0.84),
                       (r * 0.7, clear / 2 - r * 0.6),
                       (r * 1.4, clear / 2 - r * 0.45), (r * 1.4, clear / 2),
                       (0.0, clear / 2)]
            turned = _revolve(profile, max(sections // 2, 8))
            turned.apply_translation((p[0], y, 0.0))
            parts.append(turned)
    return _combine(parts)


def _chimney(width, depth, height, surface, course, joint, relief, crown,
             pots, pot_radius, seed, chamfer, sections):
    """A brick stack with a corbelled crown and pots on top."""
    crown_h = min(height * 0.12, course * 1.6) if crown else 0.0
    proud = min(relief * 2.2, width * 0.12)
    core_w, core_d = width - 2 * proud, depth - 2 * proud
    pot_h = min(height * 0.16, pot_radius * 3.2) if pots else 0.0
    stack_h = height - pot_h
    parts = [_box(core_w - 2 * relief, stack_h, core_d - 2 * relief,
                  chamfer=chamfer, center=(0.0, -height / 2 + stack_h / 2, 0.0))]

    faced_h = stack_h - crown_h
    for axis, sign in ((2, 1), (2, -1), (0, 1), (0, -1)):
        w = core_w if axis == 2 else core_d
        normal = [0.0, 0.0, 0.0]
        normal[axis] = float(sign)
        blocks = _face_relief(w, faced_h, surface, course, joint, relief,
                              chamfer, seed + axis, (), normal=tuple(normal))
        shift = [0.0, -height / 2 + faced_h / 2, 0.0]
        shift[axis] = sign * (core_d if axis == 2 else core_w) / 2
        for b in blocks:
            b.apply_translation(shift)
        parts += blocks

    if crown:
        # Two corbelled courses, the upper one reaching exactly the requested
        # width and depth — the crown is what the stack is measured by.
        y = -height / 2 + stack_h - crown_h
        for style, h, out in (("step", crown_h * 0.6, proud * 0.7),
                              ("bevel", crown_h * 0.4, proud)):
            parts.append(_sweep(_moulding_profile(style, out, h),
                                _band_path(core_w, core_d, y),
                                closed=True, up=(0.0, 1.0, 0.0)))
            y += h
    if pots:
        n = max(pots, 1)
        for p in _line_points(n, (-(core_w / 2 - pot_radius * 1.2), 0.0, 0.0),
                              (core_w / 2 - pot_radius * 1.2, 0.0, 0.0)):
            y0 = -height / 2 + stack_h - pot_h * 0.25
            pot = _revolve(
                [(0.0, y0), (pot_radius, y0),
                 (pot_radius, y0 + pot_h * 0.55),
                 (pot_radius * 1.18, y0 + pot_h * 0.72),
                 (pot_radius * 1.18, height / 2),
                 (pot_radius * 0.72, height / 2),
                 (pot_radius * 0.72, y0 + pot_h * 0.55), (0.0, y0 + pot_h * 0.5)],
                max(sections // 2, 8))
            pot.apply_translation((p[0], 0.0, 0.0))
            parts.append(pot)
    return _combine(parts)


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
    "wall_panel",
    "Wall section with an optional window or door, faced in real masonry.",
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
        SURFACE,
        COURSE,
        JOINT,
        RELIEF,
        SEED,
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


_register(Kind(
    "moulding",
    "A profile swept along a run: cornice, skirting, plinth, band or handrail.",
    "wood", _moulding,
    (
        Param("style", "choice", "ogee",
              "Section shape. square and bevel are flat cuts; ovolo is a convex "
              "quarter round and cavetto the concave one; ogee is the S; step is "
              "corbelled, for a crown; round is a half-round bead or handrail.",
              choices=_PROFILES),
        _studs("length", 6.0, "X extent — the length of the run."),
        _studs("projection", 0.3, "How far the section stands off its backing."),
        _studs("height", 0.4, "Y extent of the section."),
        Param("returns", "boolean", True,
              "Mitre both ends back through the wall so the run stops with its "
              "own section showing. Doubles the Z extent, and it is the "
              "difference between a cornice and a bevelled plank."),
        _count("steps", 5, "Samples along a curved section.", minimum=2,
               maximum=32),
    ),
))

_register(Kind(
    "riveted_panel",
    "Industrial plate: recessed bays, seams, ribs or corrugation, and rivets.",
    "metal", _riveted_panel,
    (
        _studs("width", 4.0, "X extent."),
        _studs("height", 3.0, "Y extent."),
        _studs("thickness", 0.3, "Z extent. All relief is recessed into it."),
        Param("style", "choice", "panelled",
              "panelled = raised bays divided by seams; ribbed = vertical "
              "stiffeners; corrugated = a folded sheet, swept as one section.",
              choices=("panelled", "ribbed", "corrugated")),
        _count("panels_wide", 3, "Bays across.", minimum=1, maximum=12),
        _count("panels_high", 2, "Bays up.", minimum=1, maximum=12),
        _studs("relief", 0.06, "How far a bay or rib stands proud."),
        _studs("rivet_radius", 0.05, "Rivet head radius."),
        _studs("rivet_pitch", 0.35, "Spacing between rivets along a seam."),
        _count("ribs", 5, "Ribs, `ribbed` style only.", minimum=2, maximum=24),
        Param("corner_bosses", "boolean", True,
              "Put a hex bolt at each corner of each bay."),
        CHAMFER,
    ),
))

_register(Kind(
    "window",
    "Framed window: mullions or leaded lattice, moulded sill and hood.",
    "wood", _window,
    (
        _studs("width", 2.4, "X extent."),
        _studs("height", 3.2, "Y extent."),
        _studs("depth", 0.4, "Z extent. The frame is recessed inside it and "
                             "the sill and hood project to the full depth."),
        Param("style", "choice", "mullion",
              "mullion = orthogonal glazing bars; lattice = diagonal leaded "
              "diamonds, each clipped to the light; arched = a round head with "
              "radiating bars.",
              choices=("mullion", "lattice", "arched")),
        _count("lights_wide", 2, "Panes across.", minimum=1, maximum=12),
        _count("lights_high", 3, "Panes up.", minimum=1, maximum=12),
        _studs("frame", 0.16, "Width of the outer frame members."),
        _studs("bar", 0.07, "Width of a glazing bar."),
        Param("sill", "boolean", True, "Moulded sill along the bottom."),
        Param("hood", "boolean", True, "Corbelled hood mould over the head."),
        Param("glazed", "boolean", False,
              "Fill the light with a thin pane. Off by default: one primitive "
              "carries one material, so a real build wants the glass as its own "
              "part."),
        CHAMFER,
        SECTIONS,
    ),
))

_register(Kind(
    "panel_door",
    "Door with joinery: stiles and rails, panels or planks, straps and studs.",
    "wood", _panel_door,
    (
        _studs("width", 2.2, "X extent."),
        _studs("height", 4.0, "Y extent."),
        _studs("thickness", 0.24, "Z extent. Ironmongery is recessed into it."),
        Param("style", "choice", "panelled",
              "panelled = stiles, rails and bolection-moulded panels; plank = "
              "vertical boards on ledges with a brace; banded = the same under "
              "iron straps.",
              choices=("panelled", "plank", "banded")),
        _count("panels_wide", 2, "Panels across, `panelled` only.",
               minimum=1, maximum=6),
        _count("panels_high", 3, "Panels up, `panelled` only.",
               minimum=1, maximum=8),
        _studs("stile", 0.18, "Width of an upright."),
        _studs("rail", 0.22, "Height of a cross member."),
        _studs("relief", 0.045, "Depth of the panel field and the ironwork."),
        Param("studs", "boolean", False,
              "Grid of clavos over the face. On for `banded` doors by taste."),
        Param("straps", "boolean", False, "Strap hinges with their own rivets."),
        Param("handle", "boolean", True, "Ring handle and its back plate."),
        CHAMFER,
        SECTIONS,
    ),
))

_register(Kind(
    "archway",
    "Gateway with an order: plinth, impost band, voussoir ring, hood mould.",
    "stone", _archway,
    (
        _studs("width", 6.0, "X extent."),
        _studs("height", 7.0, "Y extent."),
        _studs("depth", 1.2, "Z extent."),
        _studs("pier", 0.9, "Pier and voussoir thickness."),
        _studs("rise", 2.4, "Height of the curved part."),
        _count("voussoirs", 11, "Blocks in the ring. Odd numbers put a "
                                "keystone at the crown.", minimum=3, maximum=31),
        Param("keystone", "boolean", True,
              "Widen and project the central block."),
        Param("impost", "boolean", True,
              "Band the piers where the arch springs, and plinth them at the "
              "base."),
        Param("hood", "boolean", True,
              "Run an archivolt round the extrados."),
        SURFACE,
        COURSE,
        JOINT,
        RELIEF,
        SEED,
        CHAMFER,
    ),
))

_register(Kind(
    "battlement",
    "Crenellated parapet: merlons, crenels, coping, corbel table, arrow slits.",
    "stone", _battlement,
    (
        _studs("width", 10.0, "X extent."),
        _studs("height", 5.0, "Y extent, merlons included."),
        _studs("thickness", 0.9, "Z extent."),
        _studs("merlon_width", 1.1, "Width of a merlon."),
        _studs("crenel_width", 0.7, "Width of the gap between merlons."),
        _studs("merlon_height", 1.2, "How far a merlon stands above the wall."),
        Param("coping", "boolean", True, "Cap every merlon with a splayed coping."),
        Param("corbel", "boolean", True,
              "Corbel table under the parapet on both faces."),
        Param("arrow_slit", "boolean", True, "Slit every merlon."),
        SURFACE,
        COURSE,
        JOINT,
        RELIEF,
        SEED,
        CHAMFER,
    ),
))

_register(Kind(
    "roof",
    "Gabled roof clad in overlapping courses, with ridge, fascia and gables.",
    "wood", _roof,
    (
        _studs("width", 8.0, "X extent — along the ridge."),
        _studs("depth", 6.0, "Z extent — the full span, eaves to eaves."),
        _studs("height", 3.0, "Y extent — the rise from eaves to ridge."),
        Param("style", "choice", "tile",
              "tile = staggered courses; shingle = the same, jittered; "
              "corrugated = unstaggered, for sheet metal.",
              choices=("tile", "shingle", "corrugated")),
        _studs("course", 0.55, "Exposed height of a course up the slope."),
        _studs("tile_width", 0.7, "Width of one tile across the slope."),
        Param("gable", "boolean", True, "Close the two ends."),
        _studs("jitter", 0.0,
               "Random vertical displacement per tile. 0.02-0.05 turns a tile "
               "roof into a weathered shingle one; more than that lets daylight "
               "through.", minimum=0.0, maximum=1.0),
        SEED,
        CHAMFER,
    ),
))

_register(Kind(
    "railing",
    "Balustrade: newels, moulded handrail, bottom rail and turned balusters.",
    "wood", _railing,
    (
        _studs("length", 6.0, "X extent."),
        _studs("height", 2.4, "Y extent."),
        _studs("depth", 0.28, "Z extent — the thickness of the rail."),
        Param("style", "choice", "turned",
              "turned = a lathed vase baluster; square = plain sticks; "
              "lattice = crossed diagonals.",
              choices=("turned", "square", "lattice")),
        _count("baluster_count", 9, "Balusters between the newels.",
               minimum=1, maximum=48),
        _count("post_count", 2, "Newel posts, spread along the run.",
               minimum=2, maximum=12),
        _studs("rail_height", 0.26, "Height of the handrail section."),
        Param("newel", "boolean", True, "Post and cap at each end."),
        CHAMFER,
        SECTIONS,
    ),
))

_register(Kind(
    "chimney", "Brick stack with a corbelled crown and pots.",
    "stone", _chimney,
    (
        _studs("width", 1.8, "X extent."),
        _studs("depth", 1.4, "Z extent."),
        _studs("height", 5.0, "Y extent, pots included."),
        SURFACE,
        # A stack is a narrow thing seen close to, so its courses are finer than
        # the shared default: at 0.5 a 1.8-wide chimney is three bricks across
        # and reads as a stack of crates.
        _studs("course", 0.3, COURSE.description),
        _studs("joint", 0.035, JOINT.description),
        _studs("relief", 0.04, RELIEF.description),
        Param("crown", "boolean", True, "Corbel the top two courses out."),
        Param("pots", "integer", 2, "Chimney pots on the crown. 0 for none.",
              unit="count", minimum=0, maximum=8),
        _studs("pot_radius", 0.22, "Pot radius."),
        SEED,
        CHAMFER,
        SECTIONS,
    ),
))


# Which kinds are one closed shell and which are assemblies of them. Every kind
# here is watertight by `trimesh`'s definition — every edge bounded by exactly
# two faces — but only the first set is a single connected solid. The rest are
# components that interpenetrate, which is what `_combine` has always done and
# what all the detail below is built out of: a rivet sitting *on* a plate is a
# separate closed body, and that is the trade that makes relief free.
SINGLE_SOLID = frozenset({
    "cylinder", "plank", "tapered_panel", "wedge", "column", "moulding",
})


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
          uv_scale: float | None = None, texture: bool | None = None
          ) -> trimesh.Trimesh:
    """Build one primitive as a finished, materialled, origin-centred mesh.

    `texture=True` attaches the family's generated maps and box-projects UVs at
    the tile size the *material* asks for — a brick is the same size on a
    gatehouse as on a garden wall, so the scale cannot come from the part.
    `uv_scale` overrides that tile size for a caller who wants bigger bricks.

    It is **off** here and on in `store`, and the asymmetry is the point:
    `build` returns a *solid* and `store` writes an *asset*. Unwrapping splits
    every vertex at a projection seam, so a textured mesh is no longer welded —
    the solid is unchanged and no hole opens, but `is_watertight` goes false
    because it is an index-level test. Leaving the split until the mesh becomes
    a file keeps that property measurable where it means something.
    """
    resolved = resolve(kind, params)
    spec = KINDS[kind]
    mesh = _center(spec.build(**resolved))

    if len(mesh.faces) > config.PRIMITIVE_MAX_FACES:
        raise ValueError(
            f"{kind} with these parameters is {len(mesh.faces)} faces, over the "
            f"{config.PRIMITIVE_MAX_FACES} cap — reduce the counts "
            f"(sections, plank_count, steps, ...)"
        )

    family = materials.apply_to_mesh(
        mesh, part_name or kind, _material_for(spec, part_name, material), color,
        texture=bool(texture),
    )
    # An explicit uv_scale means "unwrap, at this size" whether or not the
    # family has a map — that was its meaning before and callers rely on it.
    scale = uv_scale or (materials.tile_studs(family) if texture else None)
    if scale:
        # Noted before the split, because it cannot be recovered after it.
        # `is_watertight` counts faces per *index pair*, so duplicating a vertex
        # to give it a second UV reads as an open edge even though no hole
        # opened — and re-welding by position afterwards is not the inverse:
        # it also fuses the deliberately-coincident vertices `_combine` leaves
        # where two closed components touch, which really does break the solid.
        mesh.metadata[CLOSED_SOLID] = bool(mesh.is_watertight)
        mesh.visual.uv = _unwrap(mesh, scale)
    return mesh


def _family_of(mesh) -> str | None:
    """The material family already baked onto the mesh by apply_to_mesh."""
    name = getattr(getattr(mesh.visual, "material", None), "name", "") or ""
    return name.removeprefix("kitbash_") or None


def store(kind: str, params: dict | None, out_dir: Path,
          texture: bool | None = None, **kwargs) -> dict:
    """Build and write mesh.glb, returning the same result shape as
    pipeline.generate_shape so a scripted part is indistinguishable from a
    generated one everywhere downstream.

    `texture` defaults to on wherever the family has a generated map. This is
    the file-writing path — what comes out is an asset somebody drops into a
    game engine, and an asset that arrives as one flat colour is the thing
    docs/SHOWCASE-CHEST.md called the biggest gap in the scripted layer. Pass
    False for the untextured solid.
    """
    t0 = time.time()
    resolved = resolve(kind, params)
    if texture is None:
        family, _ = materials.resolve(
            kwargs.get("part_name") or kind,
            _material_for(KINDS[kind], kwargs.get("part_name"), kwargs.get("material")),
        )
        texture = materials.has_texture(family)
    mesh = build(kind, resolved, texture=texture, **kwargs)
    elapsed = time.time() - t0
    closed = bool(mesh.metadata.pop(CLOSED_SOLID, mesh.is_watertight))

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
        "textured": bool(getattr(mesh.visual, "uv", None) is not None),
        # The solid, not the index buffer — see build(). A textured part has
        # split UV seams and no holes, and it is the holes anyone cares about.
        "watertight": closed,
        "file_bytes": mesh_path.stat().st_size,
        "size": [round(float(v), 4) for v in (hi - lo)],
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "params": {"kind": kind, **resolved},
    }
