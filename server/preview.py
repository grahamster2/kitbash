"""Render a mesh or an assembled scene to a shaded PNG, on the CPU.

An agent driving this server assembles parts it has never seen. It authors
placement intent, the server resolves it to coordinates, and then — until this
module existed — nothing looked at the result. Every defect found in
docs/MULTI-PART.md was found because a human opened Blender, rendered the scene
and handed the picture to a model. This is that picture, produced by the server
itself, so the loop closes: assemble, look, fix.

Design constraints, all of them load-bearing:

- **No GPU, no Blender, no OpenGL.** The box has a card but it is busy
  generating, and a preview that only works when the GPU is idle is a preview an
  agent cannot rely on. Pure numpy, so it also runs on the laptop and in CI.
- **A ground plane, always.** A part floating in space is the single most common
  assembly defect and it is invisible in a bare mesh render — everything hangs
  in a void, so nothing looks wrong. Against a floor with a cast shadow, a
  detached fin reads instantly: its shadow is somewhere else.
- **One fixed camera distance for every view.** Auto-framing per view re-frames
  on whatever that view happens to see, which is precisely how a floating part
  hides — the frame grows to include it and the gap stops looking like a gap.
  The distance comes from the whole scene's bounds and never changes, not
  between views, not when a part is isolated or highlighted.
- **A contact sheet, not one image.** A single view hides exactly the faults
  that matter: a wing detached along X is invisible from the side, a fin
  floating in Y is invisible from the top.

Projection is `texturing.Camera` — the same class, the same yaw/pitch/persp
convention, used in the opposite direction. Back-projection asks "which pixel
does this triangle come from"; a preview asks "which triangle does this pixel
show". There is exactly one camera model in this codebase and this is it.

The rasteriser is *not* `texturing.rasterize`, and that is a measurement rather
than a preference. That one loops in Python per triangle, which is right when
you rasterise once against a reference photo (~1.7 s for the 93k-face Bonanza).
A contact sheet rasterises six times, plus shadow geometry, and 20 s is not
something an agent will call in a loop. `_rasterize` below is the same
half-space/barycentric/z-buffer algorithm batched over triangles of similar
screen size, which is ~15x faster and returns the same buffers.
"""
from __future__ import annotations

import colorsys
import hashlib
import logging
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from texturing import Camera

log = logging.getLogger("kitbash.preview")

# yaw, pitch, roll in degrees, matching texturing.Camera: yaw turns about +Y,
# pitch lifts the camera above the horizon, roll spins the frame.
#
# "side" and "front" sit 8 deg up rather than dead level on purpose — at exactly
# 0 the floor collapses to a line and stops reading as a floor. 8 deg is enough
# floor to anchor the model without tilting the elevation enough to spoil
# reading heights off it.
VIEWS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 8.0, 0.0),
    "side": (90.0, 8.0, 0.0),
    # Straight down, rolled so +Z (the nose, by convention) points up the page.
    "top": (0.0, 90.0, 180.0),
    "three_qtr": (35.0, 22.0, 0.0),
    "rear_qtr": (215.0, 22.0, 0.0),
    # Nearly eye-level: the horizon cuts the model, so anything floating is
    # separated from the floor by visible background rather than by a gap you
    # have to judge. This is the view that catches a hovering part.
    "low": (60.0, 3.0, 0.0),
}

DEFAULT_VIEWS = ("side", "front", "top", "three_qtr", "rear_qtr", "low")

# Inverse camera distance in object radii (see Camera.persp). Mild perspective:
# enough that a 3/4 view has depth, little enough that the elevations stay
# close to measurable. Fixed across views so tiles are comparable.
PERSP = 0.26
# Fraction of a tile's half-width the scene's widest projection fills. The rest
# is margin for the floor and for parts further out than expected — a defect
# that pushed a part off the edge would otherwise be invisible.
FRAME_FILL = 0.88

# Light directions (surface -> light), in world space. Key from the upper left
# and slightly in front; fill from the opposite side, dimmer and flatter, so
# unlit sides keep their shape instead of going to a black silhouette.
_KEY_DIR = np.array([-0.50, 0.78, 0.38])
_FILL_DIR = np.array([0.62, 0.30, -0.55])
_KEY = 0.62
_FILL = 0.20
_AMBIENT = 0.30

_SHADOW_DARKEN = 0.40
# The floor spans this many bounding-sphere radii each way from the model's
# centre. Wide enough that a shadow thrown sideways by the key light still
# lands on it.
_GROUND_SPAN = 3.2
_GROUND_CELLS = 14
# Low contrast on purpose. The checker is there for scale and to keep the floor
# from reading as fog; a strong one competes with the shadow, which is the thing
# on the floor that actually carries information.
_GROUND_LIGHT = np.array([0.45, 0.46, 0.49])
_GROUND_DARK = np.array([0.40, 0.41, 0.44])

_BG_TOP = np.array([0.16, 0.17, 0.20])
_BG_BOTTOM = np.array([0.26, 0.27, 0.31])

# What `highlight` paints the named part. Chosen to be a colour no material in
# materials.py can produce, so "this is the highlight" is never ambiguous.
HIGHLIGHT_RGB = np.array([1.0, 0.18, 0.62])

# trimesh's stand-in when a glTF mesh carries no colour of its own. Parts that
# come back exactly this are unpainted, not painted grey, and get a clay tint.
_TRIMESH_DEFAULT = np.array([102, 102, 102], dtype=np.uint8)

# Cap on candidate pixels materialised per rasteriser batch. The batch holds
# about eight arrays of this size at once, so it sets peak memory; 1M keeps that
# under ~64 MB no matter how large a single triangle is on screen.
_MAX_CELLS = 1_000_000


def _pow2(n: np.ndarray) -> np.ndarray:
    """Smallest power of two >= n, for n >= 1."""
    return np.left_shift(1, np.ceil(np.log2(np.maximum(n, 1))).astype(np.int64))


# --------------------------------------------------------------------------
# scene -> draw list
# --------------------------------------------------------------------------
@dataclass
class Part:
    """One named node's geometry, already in world space.

    Colour is per *face*, not per vertex: shading here is flat (see `_shade`),
    the floor's checker is naturally per-face, and shadows are per-face, so
    carrying one array of face colours removes a conversion from every path.
    """

    name: str
    vertices: np.ndarray   # (V, 3) world space
    faces: np.ndarray      # (F, 3)
    face_rgb: np.ndarray   # (F, 3) float 0..1, display space


def _srgb(linear: np.ndarray) -> np.ndarray:
    """Linear -> sRGB. glTF baseColorFactor is linear; a screen is not.

    Skipping this is the single most common way a software render comes out
    looking muddy — materials.parse_color deliberately converts the other way
    when a caller types "#c81e1e", and this undoes it for display.
    """
    c = np.clip(linear, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _clay(name: str) -> np.ndarray:
    """A pale, faintly-tinted grey, stable for a given part name.

    Used when a part carries no material at all. It could be flat grey — the
    Blender ground truth is — but two grey parts that touch merge into one
    silhouette, and "did the wing detach" is exactly the question a preview has
    to answer. The tint is deliberately near-neutral: it separates parts without
    claiming the part is that colour, which a saturated palette would.
    """
    h = int(hashlib.sha1(name.encode()).hexdigest()[:8], 16)
    hue = (h % 3600) / 3600.0
    return np.array(colorsys.hsv_to_rgb(hue, 0.10, 0.84))


def _face_colors(geom: trimesh.Trimesh, name: str) -> np.ndarray:
    """Per-face display colour for one part, from whatever the glTF supplied.

    Three sources, in the order they carry real information:
    1. a base-colour texture plus UVs — the projective texture texturing.py
       bakes, sampled at each face's centroid UV;
    2. per-vertex or per-face colours — the `vertex` texture mode, and anything
       a generator painted;
    3. the material's baseColorFactor — what materials.py assigns from the part
       name, so a part called "wheel" reads black rubber.
    Anything else is unpainted and gets `_clay`.
    """
    faces = geom.faces
    visual = getattr(geom, "visual", None)
    material = getattr(visual, "material", None)

    image = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None and len(uv) == len(geom.vertices):
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        h, w = rgb.shape[:2]
        centroid_uv = np.asarray(uv, dtype=np.float64)[faces].mean(axis=1)
        # glTF UV origin is top-left; numpy's row 0 is the top row, so V maps to
        # the row index directly and only needs wrapping, not flipping.
        px = np.clip((centroid_uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int64)
        py = np.clip((centroid_uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(np.int64)
        return rgb[py, px]

    kind = getattr(visual, "kind", None)
    if kind == "face":
        return np.asarray(visual.face_colors, dtype=np.float64)[:, :3] / 255.0
    if kind == "vertex":
        vc = np.asarray(visual.vertex_colors, dtype=np.float64)[:, :3]
        # A mesh with no material at all still reports vertex colours — every
        # one of them trimesh's default grey. That is "unpainted", not "grey".
        if np.allclose(vc, _TRIMESH_DEFAULT.astype(np.float64)):
            return np.tile(_clay(name), (len(faces), 1))
        return vc[faces].mean(axis=1) / 255.0

    factor = getattr(material, "baseColorFactor", None)
    if factor is not None:
        f = np.asarray(factor, dtype=np.float64)[:3]
        # trimesh hands these back as uint8 when it round-trips a glTF and as
        # 0..1 floats when the material was built in-process.
        if f.max() > 1.0 + 1e-9:
            f = f / 255.0
        return np.tile(_srgb(f), (len(faces), 1))

    return np.tile(_clay(name), (len(faces), 1))


def load_parts(path: Path | str) -> list[Part]:
    """Read a .glb into one Part per named node, world-space.

    `process=False` because trimesh's default merges vertices on load, which
    renumbers them out of step with the visual arrays this then reads.
    """
    obj = trimesh.load(str(path), process=False)

    if isinstance(obj, trimesh.Trimesh):
        return [Part(Path(path).stem, np.asarray(obj.vertices, dtype=np.float64),
                     np.asarray(obj.faces), _face_colors(obj, Path(path).stem))]

    parts: list[Part] = []
    for node in obj.graph.nodes_geometry:
        transform, geom_name = obj.graph[node]
        geom = obj.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
            continue
        verts = trimesh.transformations.transform_points(
            np.asarray(geom.vertices, dtype=np.float64), transform
        )
        parts.append(Part(str(node), verts, np.asarray(geom.faces),
                          _face_colors(geom, str(node))))
    if not parts:
        raise ValueError(f"{path} contains no renderable geometry")
    return parts


# --------------------------------------------------------------------------
# framing
# --------------------------------------------------------------------------
@dataclass
class Framing:
    """Where the camera sits and where the floor is, for every view alike.

    Computed once from *all* parts and then never recomputed. Isolating or
    highlighting a part must not move the camera: an agent comparing the
    isolated render against the full one is comparing positions, and a preview
    that silently re-frames turns "the fin is 0.2 too high" into two pictures
    that both look fine.
    """

    center: np.ndarray
    radius: float
    ground_y: float
    fit: float

    @classmethod
    def of(cls, parts: list[Part]) -> "Framing":
        lo = np.min([p.vertices.min(axis=0) for p in parts], axis=0)
        hi = np.max([p.vertices.max(axis=0) for p in parts], axis=0)
        center = (lo + hi) / 2.0
        radius = float(np.linalg.norm(hi - lo) / 2.0) or 1.0
        # The floor goes at the bottom of the scene, not at y=0. Assembly puts
        # parts wherever the placement maths lands them and nothing guarantees
        # the origin is the ground; what a preview has to show is which parts
        # fail to reach the *lowest* thing in the scene.
        framing = cls(center=center, radius=radius, ground_y=float(lo[1]), fit=1.0)

        # How far, in unit-scale pixels, the scene reaches from the frame centre
        # in its *worst* direction — measured by projecting the bounding box's
        # corners through every view in the catalogue, not just the requested
        # ones. Two consequences, both deliberate:
        #   - the frame is tight. A bounding *sphere* over-estimates badly for
        #     anything that is not a ball: on the reference aircraft it left the
        #     subject filling a third of the tile, with the defects too small to
        #     see, which defeats the purpose.
        #   - the frame does not depend on which views were asked for, on which
        #     parts are visible, or on the tile size. It is a property of the
        #     scene alone, so two renders of the same scene are comparable pixel
        #     for pixel. Perspective is included by construction, which a plain
        #     radius/distance ratio would have got wrong by 35%.
        # Measured on the real vertices, not on the bounding box's corners. A
        # corner of an aircraft's bounding box is empty air, and under
        # perspective it sits nearer the camera than any actual geometry, which
        # inflated the frame by ~20% and shrank the subject for no reason.
        points = np.concatenate([p.vertices for p in parts])
        reach = 0.0
        for view in VIEWS:
            probe = _camera(view, framing, scale=1.0, cx=0.0, cy=0.0)
            xy, _ = probe.project(points)
            reach = max(reach, float(np.abs(xy).max()))
        framing.fit = reach or 1.0
        return framing


def _camera(view: str, framing: Framing, scale: float, cx: float, cy: float) -> Camera:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}, expected one of {sorted(VIEWS)}")
    yaw, pitch, roll = VIEWS[view]
    return Camera(
        yaw=math.radians(yaw), pitch=math.radians(pitch), roll=math.radians(roll),
        persp=PERSP, scale=scale, cx=cx, cy=cy,
        center=framing.center, radius=framing.radius,
    )


def camera_for(view: str, framing: Framing, width: int, height: int) -> Camera:
    """The camera for one named view. Distance and scale come from `framing`."""
    return _camera(
        view, framing,
        scale=FRAME_FILL * min(width, height) / 2.0 / framing.fit,
        cx=width / 2.0, cy=height / 2.0,
    )


def _camera_position(cam: Camera) -> np.ndarray:
    """Where the camera is in world space, for backface and view-vector maths."""
    axis = cam.matrix().T @ np.array([0.0, 0.0, 1.0])
    return cam.center + axis * cam.radius / max(cam.persp, 1e-6)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
def _rasterize(
    xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Software z-buffer. Returns (depth buffer, face-id buffer, -1 where empty).

    Same half-space test as texturing.rasterize, batched instead of looped.
    Triangles are grouped by the power-of-two that covers their screen bounding
    box, so every triangle in a group can share one (N, S, S) candidate grid and
    the whole group's inside/depth test is a handful of array ops. A preview is
    overwhelmingly sub-pixel triangles — a 93k-face mesh drawn 400 px wide puts
    almost everything in the S=1 and S=2 groups — so the per-triangle Python
    overhead that dominates the looped version disappears entirely.

    Depth is resolved with a lexsort rather than np.minimum.at: unbuffered
    ufunc.at is roughly an order of magnitude slower than a sort at these sizes.
    """
    zbuf = np.full(height * width, np.inf, dtype=np.float64)
    fbuf = np.full(height * width, -1, dtype=np.int64)
    if len(faces) == 0:
        return zbuf.reshape(height, width), fbuf.reshape(height, width)

    tri = xy[faces]                     # (F, 3, 2)
    tz = depth[faces]                   # (F, 3)
    ax, ay = tri[:, 0, 0], tri[:, 0, 1]
    bx, by = tri[:, 1, 0], tri[:, 1, 1]
    cx, cy = tri[:, 2, 0], tri[:, 2, 1]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    raw_lo = np.floor(tri.min(axis=1))
    raw_hi = np.ceil(tri.max(axis=1))
    with np.errstate(invalid="ignore"):
        onscreen = (
            (raw_lo[:, 0] <= width - 1) & (raw_hi[:, 0] >= 0)
            & (raw_lo[:, 1] <= height - 1) & (raw_hi[:, 1] >= 0)
            & (np.abs(area) > 1e-12)
            & np.isfinite(area)
            & np.isfinite(tz).all(axis=1)
        )
    idx = np.nonzero(onscreen)[0]
    if len(idx) == 0:
        return zbuf.reshape(height, width), fbuf.reshape(height, width)

    lo_x = np.clip(raw_lo[:, 0], 0, width - 1).astype(np.int64)
    lo_y = np.clip(raw_lo[:, 1], 0, height - 1).astype(np.int64)
    hi_x = np.clip(raw_hi[:, 0], 0, width - 1).astype(np.int64)
    hi_y = np.clip(raw_hi[:, 1], 0, height - 1).astype(np.int64)

    # Bucket width and height independently. A single square bucket is simpler
    # but pays for the worst axis on both: a floor quad 320 px wide and 6 px
    # tall would allocate a 512x512 grid, 99% of it outside the triangle, and
    # that alone made a 1.4k-face crate slower to draw than a 93k-face aircraft.
    bx_ = _pow2(hi_x[idx] - lo_x[idx] + 1)
    by_ = _pow2(hi_y[idx] - lo_y[idx] + 1)

    for key in np.unique(bx_ * 65536 + by_):
        bw, bh = int(key // 65536), int(key % 65536)
        group = idx[(bx_ * 65536 + by_) == key]
        step = max(1, _MAX_CELLS // (bw * bh))
        ox = np.arange(bw, dtype=np.int64)
        oy = np.arange(bh, dtype=np.int64)
        for start in range(0, len(group), step):
            f = group[start:start + step]
            px = lo_x[f][:, None, None] + ox[None, None, :]   # (N, 1, BW)
            py = lo_y[f][:, None, None] + oy[None, :, None]   # (N, BH, 1)
            gx, gy = px + 0.5, py + 0.5
            inv = (1.0 / area[f])[:, None, None]
            w0 = ((bx[f][:, None, None] - gx) * (cy[f][:, None, None] - gy)
                  - (by[f][:, None, None] - gy) * (cx[f][:, None, None] - gx)) * inv
            w1 = ((cx[f][:, None, None] - gx) * (ay[f][:, None, None] - gy)
                  - (cy[f][:, None, None] - gy) * (ax[f][:, None, None] - gx)) * inv
            w2 = 1.0 - w0 - w1
            hit = (
                (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
                & (px <= hi_x[f][:, None, None]) & (py <= hi_y[f][:, None, None])
            )
            if not hit.any():
                continue
            z = (w0 * tz[f, 0][:, None, None] + w1 * tz[f, 1][:, None, None]
                 + w2 * tz[f, 2][:, None, None])
            flat = np.broadcast_to(py * width + px, hit.shape)[hit]
            zc = z[hit]
            fid = np.broadcast_to(f[:, None, None], hit.shape)[hit]

            # One candidate per pixel: sort by pixel, then depth, and keep the
            # first of each run. Cheaper than resolving every candidate against
            # the buffer, and deterministic when two triangles tie.
            order = np.lexsort((zc, flat))
            flat, zc, fid = flat[order], zc[order], fid[order]
            first = np.empty(len(flat), dtype=bool)
            first[0] = True
            np.not_equal(flat[1:], flat[:-1], out=first[1:])
            flat, zc, fid = flat[first], zc[first], fid[first]

            win = zc < zbuf[flat]
            zbuf[flat[win]] = zc[win]
            fbuf[flat[win]] = fid[win]

    return zbuf.reshape(height, width), fbuf.reshape(height, width)


# --------------------------------------------------------------------------
# geometry assembly for one render
# --------------------------------------------------------------------------
def _checker(points_xz: np.ndarray, origin_xz: np.ndarray, cell: float) -> np.ndarray:
    """Alternating floor colour at world XZ. A plain floor gives no scale cue."""
    ij = np.floor((points_xz - origin_xz) / cell).astype(np.int64)
    dark = ((ij[:, 0] + ij[:, 1]) & 1).astype(bool)
    return np.where(dark[:, None], _GROUND_DARK, _GROUND_LIGHT)


def _ground_geometry(framing: Framing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A checkered quad under the model, tessellated so the checker is per-face."""
    half = _GROUND_SPAN * framing.radius
    cell = 2 * half / _GROUND_CELLS
    axis = np.linspace(-half, half, _GROUND_CELLS + 1)
    gx, gz = np.meshgrid(axis, axis, indexing="ij")
    verts = np.stack([
        (framing.center[0] + gx).ravel(),
        np.full(gx.size, framing.ground_y),
        (framing.center[2] + gz).ravel(),
    ], axis=1)

    n = _GROUND_CELLS + 1
    cells = np.arange(_GROUND_CELLS)
    i, j = np.meshgrid(cells, cells, indexing="ij")
    a = (i * n + j).ravel()
    quads = np.stack([a, a + n, a + n + 1, a + 1], axis=1)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=0)

    centroids = verts[faces][:, :, [0, 2]].mean(axis=1)
    origin = framing.center[[0, 2]] - half
    return verts, faces, _checker(centroids, origin, cell)


def _shadow_geometry(
    verts: np.ndarray, faces: np.ndarray, framing: Framing
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The model flattened onto the floor along the key light.

    This is what makes a floating part unmistakable rather than merely
    detectable: a part resting on the floor meets its own shadow, and a part
    hovering 0.2 units up has its shadow sitting somewhere else entirely. Purely
    geometric — no shadow map, no second depth pass.

    Every triangle is projected, not just the light-facing half. That halving is
    valid only for a closed surface; generated parts are open shells whose
    lit-facing set has holes in it, and the holes come through as a speckled
    shadow that reads as noise rather than as a footprint.
    """
    light = _unit(_KEY_DIR)
    edges = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                     verts[faces[:, 2]] - verts[faces[:, 0]])
    kept = faces[np.linalg.norm(edges, axis=1) > 1e-12]
    if len(kept) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64), np.zeros((0, 3))

    # Slide each vertex down the light ray until it reaches the floor. The lift
    # is a depth bias, not a real offset: without it the shadow and the floor
    # are coplanar and which one wins a pixel is down to float noise.
    lift = 1e-3 * framing.radius
    t = (verts[:, 1] - framing.ground_y) / max(light[1], 1e-6)
    flat = verts - t[:, None] * light[None, :]
    flat[:, 1] = framing.ground_y + lift

    half = _GROUND_SPAN * framing.radius
    cell = 2 * half / _GROUND_CELLS
    origin = framing.center[[0, 2]] - half
    centroids = flat[kept][:, :, [0, 2]].mean(axis=1)
    # A part far above the floor throws its shadow a long way from the model.
    # Nothing is clamped: that displacement *is* the signal.
    return flat, kept, _checker(centroids, origin, cell) * _SHADOW_DARKEN


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


@dataclass
class DrawList:
    """Everything one view rasterises, already flattened into single arrays."""

    vertices: np.ndarray
    faces: np.ndarray
    rgb: np.ndarray        # (F, 3) unshaded display colour
    shaded: np.ndarray     # (F,) bool — shadows and the floor take no key light
    part_id: np.ndarray    # (F,) index into the Part list, -1 for floor/shadow


def build_draw_list(
    parts: list[Part],
    framing: Framing,
    visible: list[int] | None = None,
    highlight: int | None = None,
    ground: bool = True,
    shadows: bool = True,
) -> DrawList:
    """Floor + shadows + the visible parts, concatenated into one draw call.

    One rasterisation pass for the lot. The floor and the shadows are ordinary
    triangles at ordinary depths, so the z-buffer handles occlusion between
    model and floor for free — a strut that pokes below the floor is clipped by
    it, which is exactly what a viewer expects to see.
    """
    visible = list(range(len(parts))) if visible is None else visible
    verts_all, faces_all, rgb_all, shaded_all, part_all = [], [], [], [], []
    offset = 0

    def add(v, f, c, shaded_flag, pid):
        nonlocal offset
        if len(f) == 0:
            return
        verts_all.append(v)
        faces_all.append(f + offset)
        rgb_all.append(c)
        shaded_all.append(np.full(len(f), shaded_flag, dtype=bool))
        part_all.append(np.full(len(f), pid, dtype=np.int64))
        offset += len(v)

    if ground:
        add(*_ground_geometry(framing), True, -1)

    shown = [parts[i] for i in visible]
    if shadows and shown:
        counts = [len(p.vertices) for p in shown]
        merged_v = np.concatenate([p.vertices for p in shown])
        bases = np.concatenate([[0], np.cumsum(counts)[:-1]])
        merged_f = np.concatenate([p.faces + b for p, b in zip(shown, bases)])
        add(*_shadow_geometry(merged_v, merged_f, framing), False, -1)

    for i in visible:
        p = parts[i]
        rgb = p.face_rgb
        if highlight is not None and i == highlight:
            rgb = np.tile(HIGHLIGHT_RGB, (len(p.faces), 1))
        add(p.vertices, p.faces, rgb, True, i)

    if not faces_all:
        return DrawList(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64),
                        np.zeros((0, 3)), np.zeros(0, bool), np.zeros(0, np.int64))
    return DrawList(
        np.concatenate(verts_all),
        np.concatenate(faces_all),
        np.concatenate(rgb_all),
        np.concatenate(shaded_all),
        np.concatenate(part_all),
    )


# --------------------------------------------------------------------------
# shading
# --------------------------------------------------------------------------
def _shade(draw: DrawList, cam: Camera) -> np.ndarray:
    """Per-face lit colour. Flat shading, two-sided.

    Face normals rather than smoothed vertex normals: generated meshes are dense
    enough that faces land sub-pixel and read smooth anyway, while scripted
    primitives have hard edges that vertex-normal averaging would round off into
    a blob. Normals are flipped toward the camera before lighting because
    materials.py marks everything doubleSided — generated shells are not
    watertight and their winding cannot be trusted.
    """
    if len(draw.faces) == 0:
        return np.zeros((0, 3))

    tri = draw.vertices[draw.faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(length, 1e-12)

    to_eye = _camera_position(cam) - tri.mean(axis=1)
    away = (np.sum(normals * to_eye, axis=1) < 0)[:, None]
    normals = np.where(away, -normals, normals)

    key = np.maximum(normals @ _unit(_KEY_DIR), 0.0)
    fill = np.maximum(normals @ _unit(_FILL_DIR), 0.0)
    # Hemisphere ambient: brighter looking up, darker looking down. Cheap, and
    # it is what stops a flat top face and a flat bottom face reading identical.
    sky = 0.62 + 0.38 * (normals[:, 1] * 0.5 + 0.5)
    intensity = _AMBIENT * sky + _KEY * key + _FILL * fill
    intensity = np.where(draw.shaded, intensity, 1.0)
    return np.clip(draw.rgb * intensity[:, None], 0.0, 1.0)


def _background(width: int, height: int) -> np.ndarray:
    """A vertical gradient. Flat black loses dark parts against the void."""
    t = np.linspace(0.0, 1.0, height)[:, None, None]
    return _BG_TOP[None, None, :] * (1 - t) + _BG_BOTTOM[None, None, :] * t


def render_view(
    draw: DrawList, cam: Camera, width: int, height: int
) -> np.ndarray:
    """Rasterise and shade one view. Returns a (H, W, 3) uint8 array."""
    img = np.broadcast_to(_background(width, height), (height, width, 3)).copy()
    if len(draw.faces):
        xy, depth = cam.project(draw.vertices)
        _, fbuf = _rasterize(xy, depth, draw.faces, width, height)
        lit = _shade(draw, cam)
        covered = fbuf >= 0
        img[covered] = lit[fbuf[covered]]
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------
def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                       # Pillow < 10.1
        return ImageFont.load_default()


def render_sheet(
    parts: list[Part],
    views: list[str] | tuple[str, ...] = DEFAULT_VIEWS,
    size: int = 1200,
    columns: int = 3,
    highlight: str | None = None,
    isolate: bool = False,
    title: str | None = None,
) -> Image.Image:
    """The contact sheet. `highlight` names a part; `isolate` hides the rest.

    Framing is taken from every part in the scene before `isolate` removes any,
    which is the whole reason isolation is safe to offer: the isolated part sits
    at the same pixel it occupied in the full render, so an agent can flip
    between the two and see whether it moved.
    """
    if not parts:
        raise ValueError("nothing to render")
    views = list(views) or list(DEFAULT_VIEWS)
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise ValueError(f"unknown view(s) {unknown}, expected from {sorted(VIEWS)}")

    names = [p.name for p in parts]
    highlight_idx = None
    if highlight is not None:
        matches = [i for i, n in enumerate(names) if n == highlight]
        if not matches:
            lowered = [i for i, n in enumerate(names) if n.lower() == highlight.lower()]
            matches = lowered
        if not matches:
            raise ValueError(f"no part named {highlight!r}; scene has {names}")
        highlight_idx = matches[0]

    framing = Framing.of(parts)
    visible = list(range(len(parts)))
    if isolate and highlight_idx is not None:
        visible = [highlight_idx]

    columns = max(1, min(columns, len(views)))
    rows = math.ceil(len(views) / columns)
    tile = max(64, size // columns)
    header = max(18, tile // 20)

    sheet = Image.new("RGB", (tile * columns, header + tile * rows), (18, 19, 22))
    draw_ops = ImageDraw.Draw(sheet)
    label_font = _font(max(11, tile // 30))

    draw = build_draw_list(parts, framing, visible, highlight_idx)
    for n, view in enumerate(views):
        cam = camera_for(view, framing, tile, tile)
        pixels = render_view(draw, cam, tile, tile)
        col, row = n % columns, n // columns
        x0, y0 = col * tile, header + row * tile
        sheet.paste(Image.fromarray(pixels), (x0, y0))
        draw_ops.text((x0 + 6, y0 + 4), view, fill=(210, 214, 222), font=label_font)
        draw_ops.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline=(38, 40, 46))

    if title:
        draw_ops.text((6, max(2, (header - label_font.size) // 2)), title,
                      fill=(180, 186, 196), font=label_font)
    return sheet


def ground_report(parts: list[Part], framing: Framing | None = None) -> list[dict]:
    """How far each part sits above the floor, sorted worst first.

    The picture is the deliverable, but a number beside it removes the argument:
    "tail_fin clears the floor by 0.45" is not a judgement call about a render.
    Reported as a fraction of the scene's own size, because the units of an
    assembled scene are whatever the parts happened to come out at.
    """
    framing = framing or Framing.of(parts)
    span = 2 * framing.radius
    out = []
    for p in parts:
        gap = float(p.vertices[:, 1].min() - framing.ground_y)
        out.append({
            "name": p.name,
            "gap": round(gap, 4),
            "gap_fraction": round(gap / span, 4) if span else 0.0,
            "faces": int(len(p.faces)),
        })
    return sorted(out, key=lambda r: -r["gap"])


def preview_png(
    source: Path | str,
    views: list[str] | tuple[str, ...] = DEFAULT_VIEWS,
    size: int = 1200,
    columns: int = 3,
    highlight: str | None = None,
    isolate: bool = False,
    title: str | None = None,
) -> bytes:
    """A .glb path in, PNG bytes out. The one call the HTTP layer makes."""
    parts = load_parts(source)
    if title is None:
        framing = Framing.of(parts)
        extent = 2 * framing.radius
        # ASCII only: the fallback bitmap font has no em dash and drops it as a
        # tofu box, which looks like a rendering bug in the thing being reviewed.
        title = (
            f"{Path(source).stem} - {len(parts)} part{'s' if len(parts) != 1 else ''}"
            f", extent {extent:.3f}, floor y={framing.ground_y:.3f}"
        )
        if highlight:
            title += f" - highlight: {highlight}{' (isolated)' if isolate else ''}"
    sheet = render_sheet(parts, views, size, columns, highlight, isolate, title)
    buf = BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
