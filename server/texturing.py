"""Colour a generated mesh by projecting its own reference image back onto it.

The generator was handed a photograph and asked for the shape in it. That means
the answer to "what colour is this surface" is already sitting in the input: for
every triangle the camera could see, the correct pixel is the one the triangle
projects to. Recovering it costs no VRAM, no model and about a second, and it is
*exactly* right rather than a generative guess -- a white aircraft with navy
stripes comes back white with navy stripes, registration and all.

This is what StableProjectorz does interactively; here it runs headless.

The whole technique turns on one thing we are not told: the camera the mesh was
generated under. There is no canonical answer to assume. Measured on this
stack's own output (docs/TEXTURING.md), TRELLIS 2's **textured** path returns a
+Y-up mesh and its **untextured** path returns the same subject +Z-up, and the
azimuth is not canonical on either -- two props generated from 3/4-view
references fitted 318 deg apart. So `fit_camera` *measures* the view instead:
it searches pose, perspective and framing for the one whose silhouette best
covers the reference's alpha matte, and reports the IoU it reached so a caller
can tell a good fit from a shrug.

Three output modes:

- `mode="uv"` (default) -- **projective UV mapping**. The atlas *is* the
  reference image, and each face's corners carry the pixel coordinates they
  project to. Nothing is resampled, so the texture is as sharp as the photo:
  lettering stays legible, a 3-pixel pinstripe stays 3 pixels. Requires no UV
  unwrap, which is what makes it available even on the untextured generator path
  that returns no UVs at all.
- `mode="atlas"` -- rebake into the UV unwrap the mesh already has. Even texel
  density and no projective stretch, at the cost of one resample. Use it when
  the output has to look like a normal texture to whatever consumes it.
- `mode="vertex"` -- per-vertex colours. Simpler and smaller, but its resolution
  is the mesh's vertex count, so a 7k-vertex aircraft turns the stripes to
  smears. Kept for viewers and exporters that will not take a texture.

What the camera could not see is handled in three passes, best first:

1. **Mirror.** Most props are roughly bilaterally symmetric. The far side of the
   fuselage is the near side flipped, so a hidden face borrows the UV of its
   mirror image -- but only after a shadow-map test proves the mirrored point
   actually lands on visible surface. The symmetry plane is detected, not
   assumed: its orientation is searched over the sphere, because a subject
   generated in its input camera's frame sits diagonally in its own bounding
   box and its mirror plane is nowhere near axis-aligned.
2. **Adjacency flood.** Whatever is still unpainted takes the colour of the
   nearest painted face across the face graph. Undersides come out the colour of
   the flanks they join, which is nearly always right and never black.
3. **Dominant colour.** Anything the flood cannot reach -- disconnected shells --
   gets the subject's modal colour.

Dependencies are trimesh, numpy and PIL, all already present and all
permissively licensed. Nothing here needs pymeshlab or bpy; see
docs/DECIMATION.md for why that matters.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

log = logging.getLogger("kitbash.texturing")

# Search sizes. The coarse sweep is over a grid so it has to be cheap; the
# refine is a few hundred perturbations so it can afford more of both.
_COARSE_SAMPLES = 12_000
_REFINE_SAMPLES = 60_000
_COARSE_RES = 96
_REFINE_RES = 192
# How many of the coarse grid's best nodes get a local refine. The silhouette
# score is riddled with near-ties -- a plane seen from above and from below have
# the same outline -- so committing to the single best node loses the fit.
_REFINE_STARTS = 24

# A face is painted from the camera only if it is this front-facing. Grazing
# faces project to a sliver of pixels and stretch that sliver across the whole
# triangle, which reads as a smear of streaks along the silhouette.
_MIN_FACING = 0.15
# ...and only if it won this much of its own projected area in the z-buffer, or
# else had all three corners land on the depth surface. See _face_visibility for
# why one of those two tests alone is not enough.
_MIN_VISIBLE_FRACTION = 0.6
# Shadow-map tolerances, in units of the mesh's bounding radius: how close to
# the depth buffer a point has to be to count as "on the visible surface".
# Mirroring gets the looser one -- a mesh is never exactly symmetric, and the
# far flank landing 2% of a radius off its near counterpart is normal.
_DEPTH_TOL = 0.02
_MIRROR_DEPTH_TOL = 0.04
# Below this the object is not symmetric enough to bother mirroring. Set
# permissively on purpose: a false positive here is nearly harmless, because
# every mirrored face still has to pass the shadow-map test below before it is
# used, and on a lopsided object almost none do. A false *negative* is what
# hurts -- it throws away the far side of every symmetric prop.
_SYMMETRY_MIN_SCORE = 0.55

# How many face-adjacency rings the flood keeps a copied colour before it has
# fully faded to the subject's dominant colour.
_FLOOD_FADE_RINGS = 8

_SWATCH_GRID = 16  # 16x16 cells; a 6x6x6 RGB cube (216 colours) is packed in


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------
@dataclass
class Camera:
    """The view a mesh was generated under, in the mesh's own coordinates.

    The mesh is first normalised to a unit sphere about its bounding-box centre,
    so `scale`, `cx` and `cy` are in pixels of the reference image and the pose
    is independent of how big the generator happened to make the thing.

    `persp` is inverse camera distance in object radii: 0 is orthographic, 0.5
    is a camera two radii out. Parameterising the reciprocal keeps ortho a
    reachable point in the search rather than a limit at infinity.
    """

    yaw: float = 0.0     # radians, about +Y (up)
    pitch: float = 0.0   # radians, about +X
    roll: float = 0.0    # radians, about the view axis
    persp: float = 0.0
    scale: float = 100.0
    cx: float = 0.0
    cy: float = 0.0
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 1.0
    iou: float = 0.0

    def matrix(self) -> np.ndarray:
        cy_, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        return rz @ rx @ ry

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> (pixel xy, depth). Depth grows away from the camera."""
        local = (np.asarray(points, dtype=np.float64) - self.center) / self.radius
        cam = local @ self.matrix().T
        if self.persp <= 1e-9:
            factor = np.ones(len(cam))
            depth = -cam[:, 2]
        else:
            dist = 1.0 / self.persp
            depth = dist - cam[:, 2]
            # Points behind the camera cannot be projected; clamp so they land
            # far off-screen instead of wrapping round to a plausible pixel.
            depth = np.maximum(depth, 1e-6)
            factor = dist / depth
        xy = np.empty((len(cam), 2))
        xy[:, 0] = self.cx + self.scale * cam[:, 0] * factor
        xy[:, 1] = self.cy - self.scale * cam[:, 1] * factor
        return xy, depth

    def as_dict(self) -> dict:
        return {
            "yaw_deg": round(math.degrees(self.yaw) % 360.0, 2),
            "pitch_deg": round(math.degrees(self.pitch), 2),
            "roll_deg": round(math.degrees(self.roll), 2),
            "perspective": round(self.persp, 4),
            "scale_px": round(self.scale, 2),
            "center_px": [round(self.cx, 1), round(self.cy, 1)],
            "silhouette_iou": round(self.iou, 4),
        }


# --------------------------------------------------------------------------
# matting
# --------------------------------------------------------------------------
def alpha_matte(image: Image.Image, threshold: int = 244) -> np.ndarray:
    """A boolean subject mask for a reference image.

    Uses the alpha channel when the image has a real one. Otherwise it floods
    "near white" inward from the border, so only background-*connected* white is
    removed -- the trick docs/QUALITY-COMPARISON.md used to keep a white bumper
    and a near-white blade from being matted away with the backdrop.
    """
    arr = np.asarray(image.convert("RGBA"))
    alpha = arr[..., 3]
    if alpha.min() < 250:
        return alpha >= 128

    h, w = alpha.shape
    bright = arr[..., :3].min(axis=2) >= threshold
    # Iterative dilation of the border seed, intersected with `bright`. A few
    # dozen passes of a 4-neighbour max is a flood fill, and stays in numpy.
    seed = np.zeros((h, w), dtype=bool)
    seed[0, :] = seed[-1, :] = True
    seed[:, 0] = seed[:, -1] = True
    seed &= bright
    while True:
        grown = seed.copy()
        grown[1:, :] |= seed[:-1, :]
        grown[:-1, :] |= seed[1:, :]
        grown[:, 1:] |= seed[:, :-1]
        grown[:, :-1] |= seed[:, 1:]
        grown &= bright
        if grown.sum() == seed.sum():
            break
        seed = grown
    return ~seed


def _downsample_mask(mask: np.ndarray, longest: int) -> tuple[np.ndarray, float]:
    h, w = mask.shape
    step = max(1, int(round(max(h, w) / longest)))
    return mask[::step, ::step], float(step)


# --------------------------------------------------------------------------
# camera fitting
# --------------------------------------------------------------------------
def _splat_iou(pts_px: np.ndarray, mask: np.ndarray) -> float:
    """IoU between a splat of projected surface samples and the subject mask."""
    h, w = mask.shape
    x = np.floor(pts_px[:, 0]).astype(np.int64)
    y = np.floor(pts_px[:, 1]).astype(np.int64)
    keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    hit = np.zeros((h, w), dtype=bool)
    hit[y[keep], x[keep]] = True
    inter = np.count_nonzero(hit & mask)
    union = np.count_nonzero(hit | mask)
    return inter / union if union else 0.0


def _fit_framing(cam: Camera, points_local: np.ndarray, mask: np.ndarray) -> None:
    """Set scale/cx/cy by matching image moments. Mutates `cam`.

    Solving framing analytically instead of searching it drops three of the
    seven parameters out of the grid, which is the difference between a sweep
    that runs in seconds and one that does not run.
    """
    cam.scale, cam.cx, cam.cy = 1.0, 0.0, 0.0
    xy, _ = cam.project(points_local)
    ys, xs = np.nonzero(mask)
    if len(xs) < 8 or len(xy) < 8:
        return
    spread_img = math.sqrt(xs.var() + ys.var())
    spread_mesh = math.sqrt(xy[:, 0].var() + xy[:, 1].var())
    if spread_mesh < 1e-9:
        return
    cam.scale = spread_img / spread_mesh
    xy, _ = cam.project(points_local)
    cam.cx = float(xs.mean() - xy[:, 0].mean())
    cam.cy = float(ys.mean() - xy[:, 1].mean())


def fit_camera(
    mesh: trimesh.Trimesh,
    mask: np.ndarray,
    *,
    seed: int = 0,
    refine_iters: int = 700,
    allow_roll: bool = True,
) -> Camera:
    """Recover the view the mesh was generated under, by silhouette agreement.

    Coarse sweep over yaw/pitch/perspective with framing solved analytically at
    each node, then a shrinking random-perturbation refine over all seven
    parameters starting from the best few nodes. The returned camera carries the
    IoU it achieved; anything above ~0.85 is a confident fit, and below ~0.7
    means the mesh and the image are not really the same object.
    """
    rng = np.random.default_rng(seed)
    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    radius = float(np.max(mesh.extents)) / 2.0 or 1.0

    coarse_pts = _surface_points(mesh, _COARSE_SAMPLES, rng)
    fine_pts = _surface_points(mesh, _REFINE_SAMPLES, rng)
    coarse_mask, coarse_step = _downsample_mask(mask, _COARSE_RES)
    fine_mask, fine_step = _downsample_mask(mask, _REFINE_RES)

    def make(yaw, pitch, roll, persp) -> Camera:
        return Camera(yaw=yaw, pitch=pitch, roll=roll, persp=persp,
                      center=center, radius=radius)

    def score(cam: Camera, pts, m) -> float:
        xy, _ = cam.project(pts)
        return _splat_iou(xy, m)

    # Sweep all of SO(3), not just a turntable. Generators do not agree on an up
    # axis -- TRELLIS 2 hands back a Z-up mesh where glTF convention is Y-up --
    # and a yaw-only sweep silently fits the wrong family of poses and then
    # spends roll and pitch apologising for it. Rz(roll)Rx(pitch)Ry(yaw) over
    # these ranges covers every orientation, so no assumption is needed.
    candidates = []
    for yaw in np.radians(np.arange(0, 360, 15)):
        for pitch in np.radians(np.arange(-90, 91, 15)):
            for roll in np.radians(np.arange(0, 360, 30)):
                cam = make(yaw, pitch, roll, 0.0)
                _fit_framing(cam, coarse_pts, coarse_mask)
                candidates.append((score(cam, coarse_pts, coarse_mask), cam))
    candidates.sort(key=lambda c: -c[0])

    best = None
    for _, coarse_cam in candidates[:_REFINE_STARTS]:
        cam = make(coarse_cam.yaw, coarse_cam.pitch, coarse_cam.roll, 0.15)
        _fit_framing(cam, fine_pts, fine_mask)
        cam.iou = score(cam, fine_pts, fine_mask)
        cam = _refine(cam, fine_pts, fine_mask, rng,
                      refine_iters // _REFINE_STARTS, allow_roll)
        if best is None or cam.iou > best.iou:
            best = cam

    # Framing was solved in the downsampled mask's pixels; lift it to full res.
    # `radius` is deliberately untouched -- the object-space normalisation is
    # the same in both, only the pixel grid changed.
    best.scale *= fine_step
    best.cx *= fine_step
    best.cy *= fine_step
    log.info("fitted camera %s", best.as_dict())
    return best


def _refine(cam, pts, mask, rng, iters, allow_roll) -> Camera:
    """Shrinking random perturbation. Cheap, derivative-free, good enough."""
    best, best_iou = cam, cam.iou
    for i in range(max(iters, 1)):
        t = 1.0 - i / max(iters, 1)
        trial = Camera(**{**best.__dict__})
        trial.yaw += rng.normal(0, 0.12 * t)
        trial.pitch += rng.normal(0, 0.10 * t)
        if allow_roll:
            trial.roll += rng.normal(0, 0.06 * t)
        trial.persp = float(np.clip(trial.persp + rng.normal(0, 0.06 * t), 0.0, 0.7))
        trial.scale *= math.exp(rng.normal(0, 0.05 * t))
        trial.cx += rng.normal(0, 2.5 * t)
        trial.cy += rng.normal(0, 2.5 * t)
        xy, _ = trial.project(pts)
        iou = _splat_iou(xy, mask)
        if iou > best_iou:
            trial.iou, best, best_iou = iou, trial, iou
    best.iou = best_iou
    return best


def _surface_points(mesh: trimesh.Trimesh, count: int, rng) -> np.ndarray:
    """Uniform-ish points on the surface, without trimesh's RNG plumbing.

    Silhouette IoU wants area-weighted coverage of the *surface*, not the
    vertices: a mesh with a dense nose and a sparse tail would otherwise be
    scored as if the tail barely existed.
    """
    tris = mesh.triangles
    areas = mesh.area_faces
    total = areas.sum()
    if total <= 0:
        return mesh.vertices.copy()
    idx = rng.choice(len(tris), size=count, p=areas / total)
    u = rng.random((count, 1))
    v = rng.random((count, 1))
    flip = (u + v) > 1
    u[flip] = 1 - u[flip]
    v[flip] = 1 - v[flip]
    a, b, c = tris[idx, 0], tris[idx, 1], tris[idx, 2]
    return a + u * (b - a) + v * (c - a)


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------
def rasterize(
    xy: np.ndarray, depth: np.ndarray, faces: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Software z-buffer. Returns (depth buffer, face-id buffer, -1 where empty).

    A per-face Python loop over numpy slices. It looks slow and is not: each
    triangle of a 20k-face mesh covers a few dozen pixels, so the loop is 20k
    iterations of tiny array work -- about a second. Bringing in a GPU
    rasteriser to save that would cost a dependency and a device.
    """
    zbuf = np.full((height, width), np.inf, dtype=np.float64)
    fbuf = np.full((height, width), -1, dtype=np.int64)

    tri_xy = xy[faces]        # (F, 3, 2)
    tri_z = depth[faces]      # (F, 3)
    lo = np.maximum(np.floor(tri_xy.min(axis=1)).astype(np.int64), 0)
    hi_x = np.minimum(np.ceil(tri_xy[:, :, 0].max(axis=1)).astype(np.int64), width - 1)
    hi_y = np.minimum(np.ceil(tri_xy[:, :, 1].max(axis=1)).astype(np.int64), height - 1)

    ax, ay = tri_xy[:, 0, 0], tri_xy[:, 0, 1]
    bx, by = tri_xy[:, 1, 0], tri_xy[:, 1, 1]
    cx_, cy_ = tri_xy[:, 2, 0], tri_xy[:, 2, 1]
    area = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)

    live = (lo[:, 0] <= hi_x) & (lo[:, 1] <= hi_y) & (np.abs(area) > 1e-12)
    for f in np.nonzero(live)[0]:
        x0, y0, x1, y1 = lo[f, 0], lo[f, 1], hi_x[f], hi_y[f]
        px = np.arange(x0, x1 + 1) + 0.5
        py = np.arange(y0, y1 + 1) + 0.5
        gx, gy = np.meshgrid(px, py)
        inv = 1.0 / area[f]
        w0 = ((bx[f] - gx) * (cy_[f] - gy) - (by[f] - gy) * (cx_[f] - gx)) * inv
        w1 = ((cx_[f] - gx) * (ay[f] - gy) - (cy_[f] - gy) * (ax[f] - gx)) * inv
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * tri_z[f, 0] + w1 * tri_z[f, 1] + w2 * tri_z[f, 2]
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = inside & (z < sub_z)
        if not win.any():
            continue
        sub_z[win] = z[win]
        fbuf[y0:y1 + 1, x0:x1 + 1][win] = f
    return zbuf, fbuf


def rasterize_uv(
    uv_px: np.ndarray, faces: np.ndarray, width: int, height: int, pad: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise the UV layout. Returns (face-id buffer, barycentric weights).

    No depth test -- a valid unwrap does not overlap itself. `pad` widens the
    inside test by a fraction of a texel so charts come out slightly fat;
    without it every chart edge leaves a one-texel unwritten gutter, and bilinear
    filtering pulls that gutter into the surface as a dark seam.
    """
    fbuf = np.full((height, width), -1, dtype=np.int64)
    bary = np.zeros((height, width, 3), dtype=np.float32)

    tri = uv_px[faces]
    lo = np.maximum(np.floor(tri.min(axis=1) - pad).astype(np.int64), 0)
    hi_x = np.minimum(np.ceil(tri[:, :, 0].max(axis=1) + pad).astype(np.int64), width - 1)
    hi_y = np.minimum(np.ceil(tri[:, :, 1].max(axis=1) + pad).astype(np.int64), height - 1)
    ax, ay = tri[:, 0, 0], tri[:, 0, 1]
    bx, by = tri[:, 1, 0], tri[:, 1, 1]
    cx_, cy_ = tri[:, 2, 0], tri[:, 2, 1]
    area = (bx - ax) * (cy_ - ay) - (by - ay) * (cx_ - ax)

    live = (lo[:, 0] <= hi_x) & (lo[:, 1] <= hi_y) & (np.abs(area) > 1e-12)
    for f in np.nonzero(live)[0]:
        x0, y0, x1, y1 = lo[f, 0], lo[f, 1], hi_x[f], hi_y[f]
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        inv = 1.0 / area[f]
        w0 = ((bx[f] - gx) * (cy_[f] - gy) - (by[f] - gy) * (cx_[f] - gx)) * inv
        w1 = ((cx_[f] - gx) * (ay[f] - gy) - (cy_[f] - gy) * (ax[f] - gx)) * inv
        w2 = 1.0 - w0 - w1
        slack = pad / max(math.sqrt(abs(area[f])), 1e-9)
        inside = (w0 >= -slack) & (w1 >= -slack) & (w2 >= -slack)
        if not inside.any():
            continue
        fbuf[y0:y1 + 1, x0:x1 + 1][inside] = f
        block = bary[y0:y1 + 1, x0:x1 + 1]
        block[inside] = np.stack([w0, w1, w2], axis=-1)[inside]
    # `pad` admits texels just outside the triangle, whose weights go slightly
    # negative. Clamping and renormalising snaps them to the nearest point *on*
    # the triangle rather than extrapolating the surface past its own edge.
    bary = np.clip(bary, 0.0, 1.0)
    total = bary.sum(axis=-1, keepdims=True)
    return fbuf, np.divide(bary, total, out=bary, where=total > 0)


# --------------------------------------------------------------------------
# symmetry
# --------------------------------------------------------------------------
def _dilate3(grid: np.ndarray) -> np.ndarray:
    """Grow a boolean voxel grid by one cell along each axis."""
    out = grid.copy()
    for axis in range(3):
        src = np.moveaxis(grid, axis, 0)     # views; the source stays the
        dst = np.moveaxis(out, axis, 0)      # original so growth is 1 cell, not 3
        dst[1:] |= src[:-1]
        dst[:-1] |= src[1:]
    return out


def _fibonacci_hemisphere(count: int) -> np.ndarray:
    i = np.arange(count) + 0.5
    z = i / count                      # 0..1: upper hemisphere only, since a
    r = np.sqrt(np.maximum(1 - z * z, 0))  # plane and its opposite are the same
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


@dataclass
class SymmetryPlane:
    """A mirror plane, as a unit normal and a point on it."""

    normal: np.ndarray
    point: np.ndarray
    score: float

    def reflect(self, points: np.ndarray) -> np.ndarray:
        local = np.asarray(points, dtype=np.float64) - self.point
        return local - 2.0 * np.outer(local @ self.normal, self.normal) + self.point


def detect_symmetry(
    mesh: trimesh.Trimesh, resolution: int = 40, directions: int = 160
) -> tuple[SymmetryPlane | None, float]:
    """Find the plane the mesh mirrors across.

    Searching plane *orientation* rather than testing the three axis planes is
    not optional here: a generator that emits the object in its input camera's
    frame leaves a 3/4-view subject sitting diagonally in its own bounding box,
    and its symmetry plane is nowhere near axis-aligned. Testing X/Y/Z only
    scores such a mesh at ~0.2 and concludes, wrongly, that an aeroplane is not
    symmetric.

    Scored by voxel-occupancy overlap rather than by nearest-neighbour
    distance: no KD-tree (so no scipy), tolerant of the vertex-count asymmetry
    remeshing always leaves, and it asks whether the *volume* mirrors rather
    than whether the triangulation does.

    The plane's *offset* is searched too, not pinned to the centroid. The
    centroid is the right answer for anything exactly symmetric, but these are
    generated meshes: a couple of percent of asymmetry moves it, and a plane a
    few percent of a radius off makes every mirrored point miss the surface it
    was supposed to land on. That shows up downstream as a collapse in how many
    faces the mirror pass can serve -- 6.8% of the airframe instead of 23%.
    """
    rng = np.random.default_rng(0)
    pts = _surface_points(mesh, 40_000, rng)
    center = pts.mean(axis=0)
    local = pts - center
    span = float(np.abs(local).max()) or 1.0

    def occupancy(p: np.ndarray, res: int) -> np.ndarray:
        idx = np.clip(((p / span * 0.5 + 0.5) * res).astype(np.int64), 0, res - 1)
        grid = np.zeros((res,) * 3, dtype=bool)
        grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return grid

    cache: dict[int, tuple] = {}

    def scorer(res: int):
        if res not in cache:
            base = occupancy(local, res)
            cache[res] = (base, _dilate3(base), np.count_nonzero(base))
        base, base_fat, base_n = cache[res]

        def score_for(n: np.ndarray, offset: float = 0.0) -> float:
            """Two-sided coverage within one voxel, not plain IoU.

            These are *shells*, one or two voxels thick. Plain IoU punishes a
            perfect mirror for landing half a voxel off, and does so harder the
            finer the grid -- a cube scored 0.86 at resolution 24 and 0.32 at
            56, which is a measurement artefact, not a fact about the cube.
            Allowing a one-voxel slop and taking the worse direction is stable
            across resolutions and still rejects genuinely lopsided shapes.
            """
            n = n / (np.linalg.norm(n) or 1.0)
            grid = occupancy(local - 2.0 * np.outer(local @ n - offset, n), res)
            m = np.count_nonzero(grid)
            if not m or not base_n:
                return 0.0
            return min(np.count_nonzero(base & _dilate3(grid)) / base_n,
                       np.count_nonzero(grid & base_fat) / m)
        return score_for

    coarse = scorer(resolution)
    best_n, best_t, best = None, 0.0, 0.0
    for n in _fibonacci_hemisphere(directions):
        s = coarse(n)
        if s > best:
            best_n, best = n, s
    if best_n is None:
        return None, 0.0

    # Polish on a finer grid than the sweep. The coarse sphere is ~15 deg apart
    # and only locates the plane to a voxel; the fine pass is what makes the
    # mirrored points actually land on the surface.
    for res, iters, sigma in ((resolution, 60, 0.12), (resolution * 2, 120, 0.05)):
        fine = scorer(res)
        best = fine(best_n, best_t)
        step = sigma
        for _ in range(iters):
            cand = best_n + rng.normal(0, step, 3)
            cand /= np.linalg.norm(cand) or 1.0
            cand_t = best_t + rng.normal(0, step * span * 0.5)
            s = fine(cand, cand_t)
            if s > best:
                best_n, best_t, best = cand, cand_t, s
            step *= 0.97

    if best < _SYMMETRY_MIN_SCORE:
        return None, best
    best_n = best_n / np.linalg.norm(best_n)
    return SymmetryPlane(best_n, center + best_t * best_n, best), best


# --------------------------------------------------------------------------
# image helpers
# --------------------------------------------------------------------------
def _dilate_rgb(rgb: np.ndarray, mask: np.ndarray, iters: int = 12) -> np.ndarray:
    """Bleed subject colour outward past the silhouette.

    Faces at the edge of the object project to pixels straddling the matte
    boundary, and the pixel just outside is backdrop. Without this every
    silhouette triangle picks up a rim of white, which on a render looks exactly
    like a lighting artefact and is impossible to argue with.
    """
    out = rgb.astype(np.float32).copy()
    filled = mask.copy()
    for _ in range(iters):
        if filled.all():
            break
        acc = np.zeros_like(out)
        cnt = np.zeros(filled.shape, dtype=np.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_v = np.roll(out, (dy, dx), axis=(0, 1))
            shifted_m = np.roll(filled, (dy, dx), axis=(0, 1)).astype(np.float32)
            acc += shifted_v * shifted_m[..., None]
            cnt += shifted_m
        grow = (~filled) & (cnt > 0)
        out[grow] = acc[grow] / cnt[grow][:, None]
        filled |= grow
    return np.clip(out, 0, 255).astype(np.uint8)


def _dominant_color(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Modal subject colour, from a coarse RGB histogram.

    The mean of a white aircraft with navy stripes is pale blue-grey, which is
    the colour of nothing on the aircraft. The mode is white, which is the
    colour of most of it.
    """
    px = rgb[mask]
    if not len(px):
        return np.array([128, 128, 128], dtype=np.uint8)
    q = (px // 32).astype(np.int64)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    top = np.bincount(keys).argmax()
    sel = keys == top
    return px[sel].mean(axis=0).astype(np.uint8)


# --------------------------------------------------------------------------
# the projection itself
# --------------------------------------------------------------------------
def _face_visibility(mesh, cam, xy, depth, width, height, world_scale):
    """Per-face: did it win the z-test, and is it square-on enough to trust?

    Deliberately does not look at `face_normals`. Meshes off the generator's
    untextured path are **not winding-consistent** -- roughly half the shell's
    normals point inward -- so a `dot(normal, view) > 0` front-face test throws
    away most of the visible surface and keeps a pile of back faces. The
    z-buffer already knows which surface is nearest, and foreshortening is
    |cos t| whichever way the normal happens to point:

        projected_area / (world_area * pixels_per_unit^2) = |cos t|

    which needs no orientation at all.
    """
    zbuf, fbuf = rasterize(xy, depth, mesh.faces, width, height)

    tri = xy[mesh.faces]
    signed = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1])
              - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    proj_area = np.abs(signed) / 2.0
    foreshorten = proj_area / np.maximum(mesh.area_faces * world_scale ** 2, 1e-12)

    seen = np.bincount(fbuf[fbuf >= 0].ravel(), minlength=len(mesh.faces))
    # A triangle smaller than a pixel can never fill its own area; judge those
    # on "did it win any pixel at all" instead.
    covered = np.where(proj_area > 1.0, seen / np.maximum(proj_area, 1e-9), seen)

    # Two ways to be visible, unioned, because either alone leaves half the
    # front surface unpainted on a remeshed shell:
    #  - it won most of its own pixels; or
    #  - all three of its corners sit on the depth surface. Neighbouring
    #    triangles in a generated shell overlap slightly, so a face can be
    #    entirely unoccluded and still lose most of its pixels to a neighbour
    #    a hair in front of it. On the airframe this test alone more than
    #    doubles the directly-painted count (2.6k faces -> 5.5k).
    px = np.clip(xy.astype(np.int64), 0, [width - 1, height - 1])
    near = zbuf[px[:, 1], px[:, 0]]
    corner_ok = np.isfinite(near) & ((depth - near) < _DEPTH_TOL)
    visible = (covered > _MIN_VISIBLE_FRACTION) | corner_ok[mesh.faces].all(axis=1)
    return visible & (foreshorten > _MIN_FACING), zbuf


def project_uv(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    camera: Camera,
    mask: np.ndarray,
    *,
    raster_size: int = 1024,
    mirror: bool = True,
) -> tuple[trimesh.Trimesh, dict]:
    """Projective UV mapping: the reference image becomes the atlas.

    Returns a mesh whose vertices have been unmerged (one per face corner) so
    every face can carry its own UVs without dragging its neighbours' across a
    seam, plus a report of how each face got painted.
    """
    rgb_full = np.asarray(image.convert("RGB"))
    img_h, img_w = rgb_full.shape[:2]
    faces = mesh.faces
    n_faces = len(faces)

    verts_xy, verts_depth = camera.project(mesh.vertices)
    scale = raster_size / max(img_w, img_h)
    # Pixels per world unit, for the foreshortening test: the camera's `scale`
    # is per *normalised* unit, so undo the radius normalisation.
    world_scale = camera.scale * scale / camera.radius
    visible, zbuf = _face_visibility(
        mesh, camera, verts_xy * scale, verts_depth,
        max(int(round(img_w * scale)), 1), max(int(round(img_h * scale)), 1),
        world_scale,
    )

    source = np.full(n_faces, -1, dtype=np.int8)  # 0 direct, 1 mirrored, 2 flood
    source[visible] = 0
    corner_xy = verts_xy[faces].astype(np.float64)  # (F,3,2) in image pixels

    stats = {"faces": n_faces, "direct": int(visible.sum())}

    plane, sym_score = (detect_symmetry(mesh) if mirror else (None, 0.0))
    stats["symmetry_score"] = round(float(sym_score), 3)
    if plane is not None:
        m_xy, m_depth = camera.project(plane.reflect(mesh.vertices))
        m_px = np.clip((m_xy * scale).astype(np.int64), 0,
                       [zbuf.shape[1] - 1, zbuf.shape[0] - 1])
        sampled = zbuf[m_px[:, 1], m_px[:, 0]]
        # Only trust the mirrored UV where the mirrored point lands *on* the
        # visible surface. Otherwise it is showing through the object and would
        # paint the far flank with whatever happens to be in front of it.
        on_surface = np.abs(sampled - m_depth) < _MIRROR_DEPTH_TOL
        # ...and all three corners have to agree, or the face stretches.
        take = (~visible) & on_surface[faces].all(axis=1)
        corner_xy[take] = m_xy[faces][take]
        source[take] = 1
        stats["mirrored"] = int(take.sum())
    else:
        stats["mirrored"] = 0

    # Projected corners can land *outside* the reference frame -- a subject that
    # fills the image has silhouette geometry whose corners project past the
    # edge -- and the atlas is taller than the image, so an unclamped UV walks
    # off the bottom of the photo and into the fallback swatch strip. Those
    # faces then render as whatever swatch they hit, or as black where the strip
    # is unused. Clamp to the photo instead; that is what a CLAMP sampler would
    # do and it keeps edge triangles the colour of the edge.
    np.clip(corner_xy[..., 0], 0, img_w - 1, out=corner_xy[..., 0])
    np.clip(corner_xy[..., 1], 0, img_h - 1, out=corner_xy[..., 1])

    # ---- atlas: reference on top, fallback swatches in a strip below --------
    subject_rgb = _dilate_rgb(rgb_full, mask)
    cell = max(8, int(round(max(img_w, img_h) / 128)))
    strip_h = cell * _SWATCH_GRID
    atlas = np.zeros((img_h + strip_h, max(img_w, cell * _SWATCH_GRID), 3), np.uint8)
    atlas_h, atlas_w = atlas.shape[:2]
    atlas[:img_h, :img_w] = subject_rgb
    if atlas_w > img_w:
        atlas[:img_h, img_w:] = subject_rgb[:, -1:]

    painted = source >= 0
    face_rgb = np.zeros((n_faces, 3), dtype=np.float64)
    if painted.any():
        cc = np.clip(np.round(corner_xy[painted]).astype(np.int64), 0,
                     [img_w - 1, img_h - 1])
        face_rgb[painted] = subject_rgb[cc[:, :, 1], cc[:, :, 0]].mean(axis=1)

    dominant = _dominant_color(rgb_full, mask)
    flood_rgb = _flood_face_colors(mesh, painted, face_rgb, dominant)
    stats["flooded"] = int((~painted).sum())

    # Seed the whole strip with the dominant colour, not black: the palette
    # rarely fills all the cells, and a bilinear sampler at a cell boundary will
    # happily pull an unused neighbour in as a dark fringe.
    atlas[img_h:] = dominant

    # Quantise the flood colours into a small swatch grid and point the
    # unpainted faces at their cell centres.
    unpainted = np.nonzero(~painted)[0]
    if len(unpainted):
        levels = _SWATCH_GRID * _SWATCH_GRID
        pal, assign = _quantize(flood_rgb[unpainted], levels)
        for i, colour in enumerate(pal):
            gx, gy = i % _SWATCH_GRID, i // _SWATCH_GRID
            atlas[img_h + gy * cell:img_h + (gy + 1) * cell,
                  gx * cell:(gx + 1) * cell] = colour
        gx = assign % _SWATCH_GRID
        gy = assign // _SWATCH_GRID
        centre = np.stack([(gx + 0.5) * cell, img_h + (gy + 0.5) * cell], axis=1)
        corner_xy[unpainted] = centre[:, None, :]
        source[unpainted] = 2

    # ---- build the output mesh --------------------------------------------
    out = trimesh.Trimesh(
        vertices=mesh.vertices[faces].reshape(-1, 3),
        faces=np.arange(n_faces * 3).reshape(-1, 3),
        process=False,
    )
    uv = np.empty((n_faces * 3, 2))
    flat = corner_xy.reshape(-1, 2)
    uv[:, 0] = flat[:, 0] / atlas_w
    # trimesh stores UV origin at the bottom-left and flips on glTF export, but
    # pixels are indexed from the top, so v has to be inverted here.
    uv[:, 1] = 1.0 - flat[:, 1] / atlas_h
    uv = np.clip(uv, 0.0, 1.0)

    material = PBRMaterial(
        name="kitbash_backprojected",
        baseColorTexture=Image.fromarray(atlas),
        metallicFactor=0.0,
        roughnessFactor=0.65,
        doubleSided=True,
    )
    out.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    stats["atlas"] = f"{atlas_w}x{atlas_h}"
    stats["camera"] = camera.as_dict()
    stats["coverage"] = round(float(visible.sum() + stats["mirrored"]) / n_faces, 4)
    return out, stats


def _flood(colors, known, a, b, fallback):
    """BFS a colour outward across an edge list, fading to `fallback` with depth.

    Two decisions here, both learned the hard way on the aircraft's underside --
    a region no camera saw and no mirror reaches, so it is *entirely* whatever
    this function decides:

    - **Copy a neighbour, do not average the frontier.** Averaging is the
      obvious thing and it is wrong: after a dozen rings every unpainted region
      converges on the mean of its surroundings, and the mean of a white
      aeroplane with navy stripes and black tyres is grey.
    - **Fade toward the dominant colour as the ring index grows.** A copy is
      only trustworthy next to what it copied. Undersides seeded from a
      shadowed wing edge otherwise inherit that shadow across the whole
      underside, which renders as dark camouflage blotches on a white
      aeroplane. Fading keeps the seam continuous and lets the interior settle
      on the colour the object mostly is.
    """
    colors = colors.copy()
    known = known.copy()
    ring = np.zeros(len(colors), dtype=np.int32)
    if not known.any() or not len(a):
        colors[~known] = fallback
        return colors
    for depth in range(1, 4097):
        if known.all():
            break
        grew = False
        for src, dst in ((a, b), (b, a)):
            sel = known[src] & ~known[dst]
            if sel.any():
                colors[dst[sel]] = colors[src[sel]]
                known[dst[sel]] = True
                ring[dst[sel]] = depth
                grew = True
        if not grew:
            break
    colors[~known] = fallback
    faded = np.clip(ring / _FLOOD_FADE_RINGS, 0.0, 1.0)[:, None]
    return colors * (1.0 - faded) + np.asarray(fallback, float) * faded


def _flood_face_colors(mesh, painted, face_rgb, fallback) -> np.ndarray:
    adj = mesh.face_adjacency
    if not len(adj):
        return _flood(face_rgb, painted, np.empty(0, int), np.empty(0, int), fallback)
    return _flood(face_rgb, painted, adj[:, 0], adj[:, 1], fallback)


def _quantize(colors: np.ndarray, cells: int) -> tuple[np.ndarray, np.ndarray]:
    """Bucket colours into a uniform RGB cube; each swatch is its members' mean.

    A cube rather than a luminance sort. Sorting by luminance is one line
    shorter and puts navy and dark brown in the same bucket, which then averages
    to a muddy purple that appears on the model as a colour the subject does not
    contain anywhere.
    """
    side = max(1, int(round(cells ** (1 / 3))))
    while side ** 3 > cells:
        side -= 1
    q = np.clip((np.asarray(colors) / 256.0 * side).astype(np.int64), 0, side - 1)
    bucket = q[:, 0] * side * side + q[:, 1] * side + q[:, 2]
    used, inverse = np.unique(bucket, return_inverse=True)
    palette = np.stack([
        np.asarray(colors)[inverse == i].mean(axis=0) for i in range(len(used))
    ]).astype(np.uint8)
    return palette, inverse


def project_atlas(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    camera: Camera,
    mask: np.ndarray,
    *,
    raster_size: int = 1024,
    mirror: bool = True,
    texture_size: int = 1024,
) -> tuple[trimesh.Trimesh, dict]:
    """Bake into the mesh's *existing* UV unwrap. Needs `mesh.visual.uv`.

    The conventional output: one square atlas laid out by whatever unwrapped the
    mesh, with even texel density and no projective stretch. `project_uv` is
    sharper where the camera looked -- it resamples nothing -- but it inherits
    the reference photo's perspective, so a surface seen edge-on gets a smear of
    texels and the unseen side gets flat swatches. This mode spends a resample
    to fix both, and it is the mode to use when the atlas is going somewhere
    that expects a normal texture.

    On this stack the UVs come from the generator's textured path (TRELLIS 2's
    Xatlas unwrap), whose *layout* was always fine even in the runs whose baked
    colour was noise -- so this replaces the colour and keeps the unwrap.
    """
    uv = getattr(mesh.visual, "uv", None)
    if uv is None or len(uv) != len(mesh.vertices):
        raise ValueError(
            "mode='atlas' needs the mesh to already carry UVs; generate with "
            "textured=true, or use mode='uv' which builds its own"
        )

    rgb_full = np.asarray(image.convert("RGB"))
    img_h, img_w = rgb_full.shape[:2]
    subject_rgb = _dilate_rgb(rgb_full, mask)
    faces = mesh.faces

    # UV origin is bottom-left in trimesh; texel rows count from the top.
    uv_px = np.empty_like(np.asarray(uv, dtype=np.float64))
    uv_px[:, 0] = np.asarray(uv)[:, 0] * texture_size
    uv_px[:, 1] = (1.0 - np.asarray(uv)[:, 1]) * texture_size
    fbuf, bary = rasterize_uv(uv_px, faces, texture_size, texture_size)
    covered = fbuf >= 0
    ty, tx = np.nonzero(covered)
    fid = fbuf[ty, tx]
    w = bary[ty, tx].astype(np.float64)
    tri = mesh.vertices[faces[fid]]
    pos = (tri * w[:, :, None]).sum(axis=1)

    scale = raster_size / max(img_w, img_h)
    world_scale = camera.scale * scale / camera.radius
    verts_xy, verts_depth = camera.project(mesh.vertices)
    visible, zbuf = _face_visibility(
        mesh, camera, verts_xy * scale, verts_depth,
        max(int(round(img_w * scale)), 1), max(int(round(img_h * scale)), 1),
        world_scale,
    )

    def sample(points, tol, gate):
        xy, depth = camera.project(points)
        px = np.clip((xy * scale).astype(np.int64), 0,
                     [zbuf.shape[1] - 1, zbuf.shape[0] - 1])
        near = zbuf[px[:, 1], px[:, 0]]
        ok = np.isfinite(near) & (np.abs(depth - near) < tol) & gate
        ip = np.clip(np.round(xy).astype(np.int64), 0, [img_w - 1, img_h - 1])
        return subject_rgb[ip[:, 1], ip[:, 0]].astype(np.float64), ok

    colors, painted = sample(pos, _DEPTH_TOL, visible[fid])
    stats = {"texels": int(covered.sum()), "direct": int(painted.sum())}

    plane, sym_score = (detect_symmetry(mesh) if mirror else (None, 0.0))
    stats["symmetry_score"] = round(float(sym_score), 3)
    if plane is not None:
        m_colors, m_ok = sample(plane.reflect(pos), _MIRROR_DEPTH_TOL, ~painted)
        colors[m_ok] = m_colors[m_ok]
        painted |= m_ok
        stats["mirrored"] = int(m_ok.sum())
    else:
        stats["mirrored"] = 0

    # Whatever is still blank gets the per-face flood, then the whole atlas is
    # dilated so the gutters between charts carry their neighbour's colour.
    dominant = _dominant_color(rgb_full, mask)
    face_painted = np.zeros(len(faces), dtype=bool)
    face_rgb = np.zeros((len(faces), 3))
    np.maximum.at(face_painted, fid[painted], True)
    if painted.any():
        acc = np.zeros((len(faces), 3))
        cnt = np.zeros(len(faces))
        np.add.at(acc, fid[painted], colors[painted])
        np.add.at(cnt, fid[painted], 1)
        nz = cnt > 0
        face_rgb[nz] = acc[nz] / cnt[nz][:, None]
    flood_rgb = _flood_face_colors(mesh, face_painted, face_rgb, dominant)
    colors[~painted] = flood_rgb[fid[~painted]]
    stats["flooded"] = int((~painted).sum())

    atlas = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    atlas[ty, tx] = np.clip(colors, 0, 255).astype(np.uint8)
    atlas = _dilate_rgb(atlas, covered, iters=max(4, texture_size // 256))

    out = mesh.copy()
    out.visual = trimesh.visual.TextureVisuals(
        uv=np.asarray(uv, dtype=np.float64),
        material=PBRMaterial(
            name="kitbash_backprojected",
            baseColorTexture=Image.fromarray(atlas),
            metallicFactor=0.0,
            roughnessFactor=0.65,
            doubleSided=True,
        ),
    )
    stats["atlas"] = f"{texture_size}x{texture_size}"
    stats["camera"] = camera.as_dict()
    stats["coverage"] = round(float(stats["direct"] + stats["mirrored"])
                              / max(stats["texels"], 1), 4)
    return out, stats


def project_vertex(
    mesh: trimesh.Trimesh,
    image: Image.Image,
    camera: Camera,
    mask: np.ndarray,
    *,
    raster_size: int = 1024,
    mirror: bool = True,
) -> tuple[trimesh.Trimesh, dict]:
    """Per-vertex colours. Same three passes, no atlas.

    Lower fidelity by construction -- colour resolution is the vertex count --
    but it survives every exporter and viewer, including ones that drop
    textures.
    """
    rgb_full = np.asarray(image.convert("RGB"))
    img_h, img_w = rgb_full.shape[:2]
    subject_rgb = _dilate_rgb(rgb_full, mask)

    xy, depth = camera.project(mesh.vertices)
    scale = raster_size / max(img_w, img_h)
    zbuf, _ = rasterize(xy * scale, depth, mesh.faces,
                        max(int(round(img_w * scale)), 1),
                        max(int(round(img_h * scale)), 1))

    def sample(points_xy, points_depth, tol):
        px = np.clip((points_xy * scale).astype(np.int64), 0,
                     [zbuf.shape[1] - 1, zbuf.shape[0] - 1])
        near = zbuf[px[:, 1], px[:, 0]]
        ok = np.isfinite(near) & (np.abs(points_depth - near) < tol)
        ip = np.clip(np.round(points_xy).astype(np.int64), 0, [img_w - 1, img_h - 1])
        return subject_rgb[ip[:, 1], ip[:, 0]].astype(np.float64), ok

    colors, seen = sample(xy, depth, _DEPTH_TOL)
    stats = {"vertices": len(mesh.vertices), "direct": int(seen.sum())}

    plane, sym_score = (detect_symmetry(mesh) if mirror else (None, 0.0))
    stats["symmetry_score"] = round(float(sym_score), 3)
    if plane is not None:
        m_xy, m_depth = camera.project(plane.reflect(mesh.vertices))
        m_colors, m_ok = sample(m_xy, m_depth, _MIRROR_DEPTH_TOL)
        take = (~seen) & m_ok
        colors[take] = m_colors[take]
        seen |= take
        stats["mirrored"] = int(take.sum())
    else:
        stats["mirrored"] = 0

    dominant = _dominant_color(rgb_full, mask)
    colors = _flood_vertex_colors(mesh, seen, colors, dominant)
    stats["flooded"] = int((~seen).sum())
    stats["coverage"] = round(float(seen.sum()) / max(len(mesh.vertices), 1), 4)
    stats["camera"] = camera.as_dict()

    out = mesh.copy()
    rgba = np.empty((len(colors), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(colors, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=rgba)
    return out, stats


def _flood_vertex_colors(mesh, known, colors, fallback) -> np.ndarray:
    edges = mesh.edges_unique
    if not len(edges):
        return _flood(colors, known, np.empty(0, int), np.empty(0, int), fallback)
    return _flood(colors, known, edges[:, 0], edges[:, 1], fallback)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def texture_from_reference(
    mesh: trimesh.Trimesh | str,
    image: Image.Image | str,
    *,
    mode: str = "uv",
    camera: Camera | None = None,
    mirror: bool = True,
    raster_size: int = 1024,
    texture_size: int = 1024,
    fit_seed: int = 0,
    refine_iters: int = 700,
) -> tuple[trimesh.Trimesh, dict]:
    """Paint `mesh` with the reference `image` it was generated from.

    `mode` is "uv" (projective, sharpest, works on a mesh with no UVs), "atlas"
    (rebake into the mesh's existing unwrap; needs one) or "vertex".

    Pass `camera` to skip the fit when the pose is already known (all the parts
    of one multi-part build share a reference and therefore a camera, so fitting
    once and reusing it is both faster and more consistent than fitting each
    part against a silhouette it only partly explains).
    """
    modes = {"uv": project_uv, "atlas": project_atlas, "vertex": project_vertex}
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, force="mesh", process=False)
    if isinstance(image, str):
        image = Image.open(image)
    if mode not in modes:
        raise ValueError(f"mode must be one of {sorted(modes)}, got {mode!r}")
    if not len(mesh.faces):
        raise ValueError("mesh has no faces")

    mask = alpha_matte(image)
    if not mask.any():
        raise ValueError("reference image matted to nothing; check the background")

    if camera is None:
        camera = fit_camera(mesh, mask, seed=fit_seed, refine_iters=refine_iters)

    extra = {"texture_size": texture_size} if mode == "atlas" else {}
    out, stats = modes[mode](mesh, image, camera, mask,
                             raster_size=raster_size, mirror=mirror, **extra)
    stats["mode"] = mode
    return out, stats
