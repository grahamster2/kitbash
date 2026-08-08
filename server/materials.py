"""Give parts materials — and, now, textures — without a GPU.

Texture *generation* (the diffusion kind) needs 12-16 GB and does not fit on the
reference card, so everything comes out of the generator as untextured grey.
That reads as unfinished even when the geometry is good.

But in a multi-part build we know something a texture model has to infer: the
caller already told us what each part *is*. An agent that names a part "canopy"
has said it is glass. Mapping that name to a PBR material costs no VRAM and
about a millisecond, and it gets a scene most of the way to looking deliberate.

That was the whole module. It was not enough. docs/SHOWCASE-CHEST.md measured
the gap and named it: *"The wood has no grain — flat PBR. Biggest gap."* A
scripted part with one `baseColorFactor` and nothing else reads as untextured
blocking the moment it stands next to a generated asset carrying a photographic
albedo, however good its geometry is.

So the second half of this module *draws* the material. Every family that has a
surface worth seeing carries a recipe — thirty-odd lines of numpy that produce a
tiling 256-512 px base colour and a matching roughness map. Wood gets rings and
knots, brick gets courses and mortar, rusted iron gets blooms that are rough
where the clean steel beside them is not. No image assets are bundled, nothing
is downloaded, nothing is added to requirements: it is `numpy` and the `PIL`
that `trimesh` already depends on, and a family's texture is built once, on
first use, and cached for the process.

Three tables carry the library and each reads on its own:

- `PALETTE` — the PBR factors. This is still the whole material for the flat
  families, and the *mean* of the texture for the rest, which is why a part
  built without UVs still comes out the right colour.
- `TEXTURE` — the recipe: which generator, its arguments, and how many studs
  one tile covers.
- `PACKS` / `ROBLOX` — a coherent set of families to build one thing out of,
  and the Roblox `Material` enum to set when the asset lands in Studio.

This is still not a replacement for generated textures. It is what you do when
you cannot afford them, and it is now a long way better than grey.
"""
import colorsys
import functools
import hashlib
import logging

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial

log = logging.getLogger("kitbash.materials")


# --------------------------------------------------------------------------
# noise toolkit
#
# Everything here is *tileable*, which is not a nicety: these textures are
# box-projected across parts that butt up against each other, and a seam at the
# tile boundary would be visible on every wall in a facade. Tileability comes
# from indexing the lattice modulo its own size — the right edge interpolates
# back into the left because it is the same column.
# --------------------------------------------------------------------------

def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _lattice(rng, size: int, fy: int, fx: int | None = None) -> np.ndarray:
    """Smooth tiling value noise: an `fy` x `fx` random lattice, upsampled.

    `fy` and `fx` are separate because most natural surfaces are anisotropic —
    wood grain, brushed metal and grass blades are all "fine across, coarse
    along", and stretching one axis of the lattice is how you say that without
    a second pass.
    """
    fx = fy if fx is None else fx
    fy, fx = max(1, min(fy, size)), max(1, min(fx, size))
    g = rng.random((fy, fx))

    def axis(n, freq):
        t = np.arange(n) * freq / n
        i0 = np.floor(t).astype(np.int64) % freq
        f = t - np.floor(t)
        return i0, (i0 + 1) % freq, f * f * (3 - 2 * f)  # smoothstep, C1 at the seam

    y0, y1, ty = axis(size, fy)
    x0, x1, tx = axis(size, fx)
    ty, tx = ty[:, None], tx[None, :]
    a = g[np.ix_(y0, x0)] * (1 - tx) + g[np.ix_(y0, x1)] * tx
    b = g[np.ix_(y1, x0)] * (1 - tx) + g[np.ix_(y1, x1)] * tx
    return a * (1 - ty) + b * ty


def _fbm(rng, size: int, freq: int = 4, octaves: int = 5, gain: float = 0.5,
         stretch: float = 1.0) -> np.ndarray:
    """Fractal sum of `_lattice`, normalised to 0..1.

    `stretch` > 1 makes the noise longer along X than Y — the anisotropy that
    turns generic mottling into grain, bedding or brushing.

    Keep `stretch` below `freq`. Past it the X lattice clamps to a single
    column, the field stops varying along X at all, and what you get is a
    perfectly straight ruled pattern — corrugated sheet, not brushed metal;
    printed veneer, not wood. Both of those were bugs here once.
    """
    out = np.zeros((size, size))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        f = freq * 2 ** o
        if f > size:
            break
        out += amp * _lattice(rng, size, f, max(1, int(round(f / stretch))))
        total += amp
        amp *= gain
    return out / max(total, 1e-9)


def _ridged(rng, size: int, **kw) -> np.ndarray:
    """fbm folded about its midpoint: creases rather than blobs.

    Bark, veins and cracks are all ridge features — the interesting part is
    where the noise crosses a value, not where it is high.
    """
    return 1.0 - np.abs(2.0 * _fbm(rng, size, **kw) - 1.0)


def _worley(rng, size: int, cells: int, jitter: float = 0.85, aspect: float = 1.0):
    """Tiling cellular noise. Returns (nearest distance, second, cell id).

    Only the 3x3 neighbourhood is searched, so cost is 9 * size^2 rather than
    size^2 * cells^2 — the difference between 0.6 ms and 0.6 s at 256 px.
    `d2 - d1` is the ridge *between* cells, which is the mortar joint in
    cobblestone and the fracture plane in ice.

    `aspect` > 1 puts more cells along X than Y, giving cells that are wider
    than they are tall — slate plates, riven flagstones, laid rubble.
    """
    cy = max(2, cells)
    cx = max(2, int(round(cells * aspect)))
    off = (1 - jitter) / 2 + rng.random((cy, cx, 2)) * jitter
    ty, tx = np.arange(size) * cy / size, np.arange(size) * cx / size
    iy0, ix0 = np.floor(ty).astype(np.int64), np.floor(tx).astype(np.int64)
    fy, fx = (ty - iy0)[:, None], (tx - ix0)[None, :]

    d1 = np.full((size, size), 9.0)
    d2 = np.full((size, size), 9.0)
    ident = np.zeros((size, size), np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            iy = ((iy0 + dy) % cy)[:, None]
            ix = ((ix0 + dx) % cx)[None, :]
            # Distances are measured in *Y* cell units so an anisotropic grid
            # still produces round-looking stones rather than smeared ones.
            d = np.hypot((off[iy, ix, 0] + dx - fx) * cy / cx,
                         off[iy, ix, 1] + dy - fy)
            closer = d < d1
            d2 = np.where(closer, d1, np.minimum(d2, d))
            ident = np.where(closer, iy * cx + ix, ident)
            d1 = np.where(closer, d, d1)
    return d1, d2, ident


def _by_cell(rng, ident: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """One random value per Worley cell, painted back over the pixels.

    This is what makes a cobbled street read as *stones* rather than as a
    pattern: each stone is a flat colour of its own and the eye groups it.
    """
    return (lo + rng.random(int(ident.max()) + 1) * (hi - lo))[ident]


def _warp(field: np.ndarray, rng, size: int, amount: float, freq: int = 3) -> np.ndarray:
    """Push a field around by low-frequency noise, wrapping at the edges.

    Straight lines are the tell of a procedural texture. Marble veins, wood
    rings and sand ripples are all a simple periodic function with this applied.
    """
    if amount <= 0:
        return field
    dy = (_lattice(rng, size, freq) - 0.5) * 2 * amount
    dx = (_lattice(rng, size, freq) - 0.5) * 2 * amount
    yy, xx = np.mgrid[0:size, 0:size]
    return field[(yy + dy).astype(np.int64) % size, (xx + dx).astype(np.int64) % size]


def _mix(a, b, t):
    """Lerp between two colours (length-3 sequences) by a 0..1 field."""
    t = np.clip(t, 0.0, 1.0)[..., None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _rows(size: int, count: int):
    """A stack of `count` horizontal rows: (row index, position within row 0..1).

    Both are broadcast to the full image, so a recipe can index a per-row array
    with the first and shade across the row with the second.
    """
    v = np.arange(size)[:, None] / size * count + np.zeros((1, size))
    row = np.floor(v).astype(np.int64)
    return row, v - row


# --------------------------------------------------------------------------
# texture recipes
#
# Each generator returns (albedo, roughness): albedo is HxWx3 of *linear* light
# in 0..1 (glTF base colour is linear, and the PNG is sRGB-encoded on the way
# out); roughness is HxW, already linear because glTF says that channel is not
# colour data. Neither is colour-corrected here — `_texture` rescales the albedo
# so its mean lands exactly on the family's `baseColorFactor`, which is what
# keeps a textured part and an untextured one the same colour.
# --------------------------------------------------------------------------

def _grain(rng, size, *, rings=9.0, warp=0.10, sharp=3.2, depth=0.72, fibre=0.30,
           pores=0.0, knots=1, knot_size=0.055, sweep=0.09,
           late=(0.30, 0.17, 0.08), early=(0.66, 0.45, 0.25), rough=(0.62, 0.85)):
    """Wood: flat-sawn growth rings, fibre along the board, pores and knots.

    The first version of this drew wavy stripes and looked like painted
    plywood. Three things fix it and each is doing a specific job:

    * **Arches.** The ring coordinate is `|y - c(x)|`, distance from a line that
      wanders across the board, so the rings *mirror* about it into the nested
      cathedral arches a flat-sawn board actually has. Parallel stripes are what
      a quarter-sawn board looks like and almost no board is quarter-sawn.
    * **A thin latewood line.** `(1 - cos)` raised to a power, so the dark band
      is narrow and the light earlywood between is wide. Equal bands read as a
      stripe pattern; wood is mostly pale with dark lines in it.
    * **Fibre and pores at the pixel scale.** Fine noise stretched ~30x along
      the board. Without it the surface is flat colour between the rings and the
      eye reads vector art, and this is also what makes the roughness vary.

    `sweep` is small and there is exactly one wave in it, which matters more
    than it sounds: two waves put four chevrons across the tile and the result
    reads as a contour map. A board is *nearly* parallel bands with one arch in
    it, so the pith line is allowed to move about a tenth of the tile, once.
    """
    y, x = np.mgrid[0:size, 0:size] / size
    # One slow wave in the pith line. Integer frequency keeps the tile seamless.
    centre = 0.5 + np.sin(2 * np.pi * (x + rng.random())) * sweep
    # Wrapped distance to the pith, not `abs(y - centre)`. The plain absolute
    # value is not periodic in Y: the top edge sits at `centre` rings and the
    # bottom at `1 - centre`, and with the pith wandering those differ by
    # several whole rings, which put a hard line across every board every tile.
    # Wrapping mirrors the arches a second time at the far edge instead.
    r = np.abs(((y - centre + 0.5) % 1.0) - 0.5) * rings * 2.0
    r = r + (_fbm(rng, size, freq=6, octaves=4, stretch=3.0) - 0.5) * warp * rings

    knot_mask = np.zeros_like(r)
    for _ in range(knots):
        ky, kx = rng.random(2)
        d = np.hypot(((y - ky + 0.5) % 1.0) - 0.5, ((x - kx + 0.5) % 1.0) - 0.5)
        # 1/d drags the ring coordinate towards the knot, so the rings crowd and
        # sweep around it — the whorl, rather than a decal of one. The Gaussian
        # is what keeps it a knot: 1/d alone still shifts the rings by a quarter
        # of a ring half a tile away, and every board came out a swirling maze.
        r = r + rings * 0.09 / (d + 0.045) * np.exp(-(d / 0.13) ** 2)
        knot_mask = np.maximum(knot_mask, 1 - _smoothstep(knot_size * 0.6, knot_size, d))

    band = (0.5 - 0.5 * np.cos(2 * np.pi * r)) ** sharp
    # Fibre is drawn straight off two lattices rather than out of `_fbm`,
    # because fbm normalises its octaves together and the near-pixel detail
    # ends up too weak to see. Wood's fibre is a *strong* signal one pixel wide
    # and forty long; without it the rings are clean gradients and the board
    # reads as printed veneer, which is precisely how this looked before.
    fine = (0.62 * _lattice(rng, size, size // 3, max(2, size // 48))
            + 0.38 * _lattice(rng, size, size // 9, max(2, size // 20)))
    t = np.clip(band * depth + fibre * 2.0 * (fine - 0.5) + 0.10, 0, 1)
    if pores:
        # Ring-porous species (oak, ash) show open vessels as short dark dashes
        # that follow the grain — the detail that survives being seen up close.
        p = _lattice(rng, size, size // 2, max(2, size // 26))
        t = np.clip(t + pores * _smoothstep(0.66, 0.90, p), 0, 1)
    t = np.clip(t + knot_mask * 0.9, 0, 1)
    albedo = _mix(early, late, t)
    return albedo, rough[0] + (rough[1] - rough[0]) * t


def _planks(rng, size, *, boards=4, gap=0.05, **grain_kw):
    """Wood grain cut into boards, each with its own tint and grain phase.

    Boards of identical colour read as a printed pattern. The per-board value
    jitter is the cheapest thing in this file and it does most of the work.
    """
    albedo, rough = _grain(rng, size, **grain_kw)
    row, v = _rows(size, boards)
    shade = (0.80 + rng.random(boards) * 0.40)[row][..., None]
    albedo = albedo * shade
    # Roll each board along U so two neighbours never share a ring pattern.
    shift = (rng.integers(0, size, boards))[row]
    xx = (np.arange(size)[None, :] + shift) % size
    yy = np.arange(size)[:, None] + np.zeros_like(xx)
    albedo, rough = albedo[yy, xx], rough[yy, xx]
    seam = _smoothstep(0.0, gap, np.minimum(v, 1 - v))
    return albedo * (0.18 + 0.82 * seam)[..., None], rough * (1 - 0.25 * (1 - seam))


def _masonry(rng, size, *, courses=6, per_course=3, joint=0.055, bond=0.5,
             mortar=(0.52, 0.50, 0.46), tones=((0.42, 0.17, 0.13), (0.60, 0.28, 0.20),
                                               (0.36, 0.20, 0.17), (0.55, 0.34, 0.25)),
             grit=0.22, bevel=0.10, rough=(0.82, 0.96)):
    """Brickwork: courses, a running bond, mortar joints, one tone per brick.

    Four tones rather than one is the whole difference between "brick" and "red
    wall". Real facing brick is fired unevenly and a bond of a single colour is
    the thing that reads as a texture map from twenty metres.
    """
    v = np.arange(size)[:, None] / size * courses
    row = np.floor(v).astype(np.int64)
    fv = v - row
    u = (np.arange(size)[None, :] / size + row * bond) * per_course
    col = np.floor(u).astype(np.int64)
    fu = u - col

    ident = (row * per_course * 3 + col) % (courses * per_course)
    pick = rng.integers(0, len(tones), courses * per_course)[ident]
    base = np.asarray(tones)[pick]
    base = base * (0.86 + rng.random(courses * per_course)[ident] * 0.28)[..., None]

    # Joint in both axes; the bevel term darkens the brick's own edge, which is
    # what gives the course a shadow line instead of a painted stripe.
    ju = _smoothstep(0.0, joint * per_course, np.minimum(fu, 1 - fu))
    jv = _smoothstep(0.0, joint * courses, np.minimum(fv, 1 - fv))
    inside = np.minimum(ju, jv)
    face = _smoothstep(0.0, bevel, inside)

    grit_n = _fbm(rng, size, freq=16, octaves=4)
    base = base * (1 - grit + grit * (0.5 + grit_n))[..., None]
    mort = np.asarray(mortar) * (0.80 + 0.40 * _fbm(rng, size, freq=24, octaves=3))[..., None]
    albedo = _mix(mort, base, inside) * (0.55 + 0.45 * face)[..., None]
    return albedo, rough[1] - (rough[1] - rough[0]) * inside


def _cells(rng, size, *, cells=7, jitter=0.85, gap=0.055, dome=0.45, spread=0.30,
           tones=((0.34, 0.33, 0.31),), gap_color=(0.16, 0.16, 0.15), grit=0.18,
           warp=0.0, rough=(0.70, 0.95), flat=False):
    """Worley stones: cobbles, gravel, granite grain, pebbled leather, facets.

    `dome` shades each cell by its distance from its own centre — that is the
    rounded top of a cobble. `flat` turns it off for anything faceted (crystal,
    beaten foil) where a flat plane per cell is the point.
    """
    d1, d2, ident = _worley(rng, size, cells, jitter)
    n = cells * cells
    tone = np.asarray(tones)[rng.integers(0, len(tones), n)[ident]]
    tone = tone * (1 - spread / 2 + _by_cell(rng, ident, 0, spread))[..., None]

    edge = _smoothstep(0.0, gap, d2 - d1)
    if warp:
        edge = _warp(edge, rng, size, warp * size, freq=6)
    shade = 1.0 if flat else (1.0 - dome * np.clip(d1 / (d1.max() + 1e-9), 0, 1) ** 1.5)
    grit_n = _fbm(rng, size, freq=20, octaves=4)
    tone = tone * (np.asarray(shade) * (1 - grit + grit * (0.5 + grit_n)))[..., None]
    albedo = _mix(np.asarray(gap_color), tone, edge)
    return albedo, rough[1] - (rough[1] - rough[0]) * edge


def _bedded(rng, size, *, beds=14, stretch=16.0, contrast=1.5, blotch=0.45,
            plates=0, plate_aspect=2.6, plate_spread=0.30, plate_gap=0.03,
            light=(0.72, 0.62, 0.46), dark=(0.48, 0.38, 0.26), grit=0.16,
            rough=(0.78, 0.94)):
    """Sedimentary and metamorphic rock: laminations, blotching, riven plates.

    Bedding is drawn from noise stretched hard along X rather than from a sine,
    because a sine wanders and comes out looking like wood grain — which is
    exactly what the first version of this did to sandstone and slate. Real
    bedding is irregular in thickness and broken along its length; stretched
    fractal noise is that, and a cosine is not.

    `plates` adds a Worley layer of wide, flat, individually-toned slabs on top,
    which is what makes roofing slate read as slate rather than as dark rock.
    """
    lam = _fbm(rng, size, freq=beds, octaves=4, gain=0.60, stretch=stretch)
    lam = np.clip(0.5 + (lam - 0.5) * contrast, 0, 1)
    body = _fbm(rng, size, freq=3, octaves=5, stretch=max(1.0, stretch / 4))
    t = np.clip(lam * (1 - blotch) + blotch * body, 0, 1)

    if plates:
        d1, d2, ident = _worley(rng, size, plates, 0.9, aspect=plate_aspect)
        edge = _smoothstep(0.0, plate_gap, d2 - d1)
        tone = _by_cell(rng, ident, 1 - plate_spread, 1 + plate_spread)
        t = np.clip(t * tone * (0.45 + 0.55 * edge), 0, 1)

    grit_n = _fbm(rng, size, freq=30, octaves=3, gain=0.62)
    albedo = _mix(dark, light, t) * (1 - grit + grit * (0.5 + grit_n))[..., None]
    return albedo, rough[1] - (rough[1] - rough[0]) * t


def _marble(rng, size, *, veins=(3, 2), fine_veins=(7, 5), turbulence=0.55,
            sharpness=14.0, secondary=0.45, halo=0.30,
            body=(0.80, 0.79, 0.76), vein=(0.24, 0.24, 0.28), rough=(0.16, 0.30)):
    """Marble: a few long, sharp, connected veins across a pale, near-uniform body.

    Written against two specific failure modes.

    *Grey noise*: marble is **mostly flat stone**. The whole read comes from a
    small number of high-contrast veins, so the body is nearly uniform and the
    vein term is a `(1 - |sin|)` raised to a high power — thin lines, not bands.

    *Tadpoles*: the first attempt turbulated a **sine**, which is not monotonic,
    so the level sets closed into little loops scattered over the slab. The
    phase here is a **linear ramp** `nx*x + ny*y` plus turbulence. Integer
    `nx, ny` keep it periodic across the tile, and a ramp's level sets are open
    curves that cross the whole slab, which is what a vein does.
    """
    y, x = np.mgrid[0:size, 0:size] / size
    turb = (_fbm(rng, size, freq=3, octaves=6, gain=0.58) - 0.5) * 2

    def net(nx, ny, k, t):
        p = nx * x + ny * y + turb * t
        return (1.0 - np.abs(np.sin(np.pi * p))) ** k

    v = net(*veins, sharpness, turbulence)
    v = np.clip(v + secondary * net(*fine_veins, sharpness * 2.2, turbulence * 0.7), 0, 1)

    # The halo is the diffuse stain marble carries either side of a vein; a vein
    # with a hard edge and nothing around it looks drawn on.
    soft = np.clip(net(*veins, sharpness / 5, turbulence), 0, 1) * halo
    haze = _fbm(rng, size, freq=2, octaves=4)
    ground = np.asarray(body) * (0.95 + 0.10 * haze)[..., None]
    albedo = _mix(_mix(ground, vein, soft), vein, v)
    return albedo, rough[0] + (rough[1] - rough[0]) * np.clip(v + soft, 0, 1)


def _speckle(rng, size, *, freq=5, octaves=6, gain=0.55, stretch=1.0, contrast=1.0,
             grain=0.0, grain_freq=40, pits=0.0, pit_cells=22, pit_size=0.16,
             flecks=0.0, fleck_dark=0.35,
             light=(0.62, 0.61, 0.58), dark=(0.42, 0.41, 0.39),
             rough=(0.85, 0.97)):
    """The general mineral/earth surface: fbm between two tones, plus damage.

    Concrete, plaster, terracotta, dirt, snow, basalt and asphalt are all this
    with different tones and different amounts of damage. Keeping them one
    generator is deliberate — what actually separates them is colour and pit
    density, and thirteen near-copies of the same twenty lines would hide that.

    `grain` is a separate near-pixel-scale layer rather than more fbm octaves,
    because the two want different contrast: a wall is broad soft blotching
    *and* a hard fine tooth, and one gain cannot give you both.
    """
    t = _fbm(rng, size, freq=freq, octaves=octaves, gain=gain, stretch=stretch)
    t = np.clip(0.5 + (t - 0.5) * contrast, 0, 1)
    if grain:
        g = _fbm(rng, size, freq=grain_freq, octaves=2, gain=0.5)
        t = np.clip(t + grain * (g - 0.5), 0, 1)
    albedo = _mix(dark, light, t)
    r = rough[0] + (rough[1] - rough[0]) * (1 - t)

    if pits:
        d1, _, ident = _worley(rng, size, pit_cells, 0.95)
        keep = _by_cell(rng, ident) < pits
        hole = keep * (1 - _smoothstep(pit_size * 0.4, pit_size, d1))
        albedo = albedo * (1 - 0.55 * hole)[..., None]
        r = np.clip(r + 0.10 * hole, 0, 1)
    if flecks:
        n = rng.random((size, size))
        dark_f = (n < flecks * 0.5) * fleck_dark
        light_f = (n > 1 - flecks * 0.5) * fleck_dark
        albedo = albedo * (1 - dark_f + light_f)[..., None]
    return albedo, r


def _brushed(rng, size, *, tint=(0.60, 0.62, 0.66), depth=0.13, freq=40,
             stretch=9.0, scratches=0.0, mottle=0.0, rough=(0.22, 0.48)):
    """Brushed / milled metal: fine directional streaks, and roughness follows them.

    The roughness variation is the point. Flat-roughness metal is a mirror or a
    matte slab and neither looks like a part; a metal whose roughness wobbles
    along the brush direction catches light in streaks, which is what tells you
    it is metal at all in a still image.

    `freq` starts high on purpose. At the fbm default of 4 the coarsest octave
    dominates and the result is six fat horizontal bands — venetian blinds, not
    brushing. Brushing lives at the pixel scale, so the *first* octave has to be
    already fine and the ones above it are the ones adding texture.

    `stretch` is finite for the same reason. Push it past `freq` and the lattice
    collapses to one column: perfectly straight lines from edge to edge, which
    is corrugated sheet rather than a brushed finish. Around 8-10 the streaks
    still start and stop along their length, which is what brushing looks like.
    """
    fine = _fbm(rng, size, freq=freq, octaves=3, gain=0.55, stretch=stretch)
    t = np.clip(0.5 + (fine - 0.5) * 2.6, 0, 1)
    v = 1.0 - depth + 2 * depth * t
    if mottle:
        v = v * (1 - mottle + 2 * mottle * _fbm(rng, size, freq=3, octaves=4))
    if scratches:
        # A handful of long, deliberate scratches over the brushing. Raising a
        # ridge to a high power keeps only the crests, so they read as isolated
        # marks rather than as a second, coarser brush.
        sc = _ridged(rng, size, freq=6, octaves=3, stretch=stretch * 2) ** 14
        v = v * (1 + scratches * 3.0 * sc)
        t = np.clip(t + sc, 0, 1)
    albedo = np.asarray(tint) * np.clip(v, 0, 2)[..., None]
    return albedo, rough[0] + (rough[1] - rough[0]) * t


def _rust(rng, size, *, coverage=0.55, metal=(0.42, 0.44, 0.47),
          tones=((0.34, 0.14, 0.05), (0.50, 0.23, 0.08), (0.24, 0.11, 0.06)),
          pits=0.35, rough=(0.30, 0.98)):
    """Corrosion: rust blooms eating into clean metal, at two scales.

    The two scales matter. One scale of blotch is camouflage; a large bloom with
    a fringe of small pitting around it is corrosion, because that is how it
    spreads. Roughness swings the full 0.30-0.98 across the boundary, so the
    clean steel still catches a highlight the rust does not.
    """
    big = _fbm(rng, size, freq=3, octaves=5)
    small = _fbm(rng, size, freq=9, octaves=4)
    mask = _smoothstep(0.5 - coverage * 0.5, 0.5 + coverage * 0.35, big * 0.7 + small * 0.3)

    tone = np.asarray(tones)
    which = np.clip((small * len(tones)).astype(np.int64), 0, len(tones) - 1)
    rusty = tone[which] * (0.75 + 0.5 * _fbm(rng, size, freq=14, octaves=3))[..., None]
    clean, clean_r = _brushed(rng, size, tint=metal, depth=0.10, rough=(0.30, 0.50))

    albedo = _mix(clean, rusty, mask)
    if pits:
        d1, _, ident = _worley(rng, size, 26, 0.95)
        hole = (_by_cell(rng, ident) < pits) * (1 - _smoothstep(0.05, 0.14, d1)) * mask
        albedo = albedo * (1 - 0.45 * hole)[..., None]
    return albedo, clean_r * (1 - mask) + (rough[1] - 0.12 * small) * mask


def _patina(rng, size, *, base=(0.44, 0.24, 0.13), patina=(0.22, 0.52, 0.44),
            pale=(0.55, 0.78, 0.68), coverage=0.72, rough=(0.35, 0.92)):
    """Verdigris: copper going green in blooms, with the pale crust on top.

    Two greens, not one. Fresh verdigris is dark and waxy; where it has been
    rained on it dries to a chalky mint, and the pale crust sitting inside the
    dark bloom is the read.
    """
    big = _fbm(rng, size, freq=3, octaves=5)
    fine = _fbm(rng, size, freq=11, octaves=4)
    mask = _smoothstep(0.55 - coverage * 0.5, 0.62, big * 0.75 + fine * 0.25)
    crust = _smoothstep(0.55, 0.80, fine * 0.6 + big * 0.4) * mask
    metal, metal_r = _brushed(rng, size, tint=base, depth=0.12, rough=(0.30, 0.55))
    green = _mix(patina, pale, crust)
    albedo = _mix(metal, green, mask)
    return albedo, metal_r * (1 - mask) + (rough[1] - 0.15 * crust) * mask


def _weave(rng, size, *, threads=26, warp_c=(0.36, 0.34, 0.40),
           weft_c=(0.30, 0.28, 0.34), depth=0.30, fuzz=0.30, rough=(0.86, 0.99)):
    """Woven cloth: over-under threads on a checker, plus fibre fuzz.

    Cloth reads by its weave shadow, so the thread that is *under* has to darken
    at the crossing. A flat checkerboard of two colours does not; a cosine
    profile across each thread does.
    """
    y, x = np.mgrid[0:size, 0:size] / size * threads
    fy, fx = y % 1.0, x % 1.0
    over = ((np.floor(y).astype(np.int64) + np.floor(x).astype(np.int64)) % 2) == 0
    ridge_y = 0.5 + 0.5 * np.cos(2 * np.pi * (fy - 0.5))
    ridge_x = 0.5 + 0.5 * np.cos(2 * np.pi * (fx - 0.5))
    lift = np.where(over, ridge_y, ridge_x)
    albedo = np.where(over[..., None], np.asarray(warp_c), np.asarray(weft_c))
    albedo = albedo * (1 - depth + 2 * depth * lift)[..., None]
    albedo = albedo * (1 - fuzz / 2 + fuzz * _fbm(rng, size, freq=26, octaves=3))[..., None]
    return albedo, rough[1] - (rough[1] - rough[0]) * lift


def _blades(rng, size, *, density=44, clump=4, stretch=0.10, contrast=0.85,
            tip=(0.32, 0.46, 0.13), root=(0.10, 0.17, 0.05),
            dead=(0.42, 0.36, 0.14), dead_mix=0.18, rough=(0.88, 0.99)):
    """Grass, moss, thatch: many thin strands, clumped, some of them dead.

    Strands come from noise stretched hard *across* the strand direction, so
    each one is a few pixels wide and many long. Clumping is a second, coarse
    noise multiplying the value — grass in one flat tone is a green rectangle.
    """
    strand = _fbm(rng, size, freq=density, octaves=3, stretch=stretch)
    strand = np.clip(0.5 + (strand - 0.5) * (1 + contrast * 3), 0, 1)
    clump_n = _fbm(rng, size, freq=clump, octaves=4)
    t = np.clip(strand * (0.45 + 0.75 * clump_n), 0, 1)
    albedo = _mix(root, tip, t)
    if dead_mix:
        d = _smoothstep(1 - dead_mix, 1.0, _fbm(rng, size, freq=5, octaves=4))
        albedo = _mix(albedo, np.asarray(dead) * (0.7 + 0.6 * t)[..., None], d)
    return albedo, rough[1] - (rough[1] - rough[0]) * t


def _bark(rng, size, *, ridges=9, low=0.30, high=0.80, flake=0.35, splits=0.35,
          light=(0.40, 0.31, 0.21), dark=(0.05, 0.04, 0.03), rough=(0.88, 0.99)):
    """Tree bark: vertical ridges split by deep, near-black fissures.

    The fissures have to go properly dark and cover only a fraction of the
    surface. The first version smoothstepped from zero, which saturated most of
    the image at "lit ridge" and left a flat brown rectangle with faint lines in
    it. Remapping between `low` and `high` puts the transition where the noise
    actually is, so the ridges are lit, the splits between them are black, and
    there is a real edge in between.
    """
    ridge = _ridged(rng, size, freq=ridges, octaves=5, gain=0.58, stretch=0.14)
    t = _smoothstep(low, high, ridge)
    if splits:
        # A few deeper vertical splits right through the ridges.
        deep = _ridged(rng, size, freq=3, octaves=3, stretch=0.05) ** 5
        t = t * (1 - splits * 2.5 * np.clip(deep, 0, 0.4))
    t = np.clip(t + flake * (_fbm(rng, size, freq=26, octaves=3, stretch=0.4) - 0.5), 0, 1)
    return _mix(dark, light, t), rough[1] - (rough[1] - rough[0]) * t


def _tread(rng, size, *, pitch=6, bar=0.30, tint=(0.55, 0.57, 0.60), lift=0.55,
           rough=(0.24, 0.55)):
    """Diamond plate: pairs of raised bars at opposing 45 degrees.

    Roblox has a DiamondPlate material and it is the single most recognisable
    industrial surface there is, so it is worth the twelve lines rather than
    being folded into brushed metal.
    """
    y, x = np.mgrid[0:size, 0:size] / size * pitch
    cell_y, cell_x = np.floor(y).astype(np.int64), np.floor(x).astype(np.int64)
    fy, fx = y % 1.0, x % 1.0
    # Alternate the diagonal per checker cell — that is the "diamond".
    d = np.where(((cell_y + cell_x) % 2) == 0, (fy + fx) % 1.0, (fy - fx) % 1.0)
    band = 1.0 - np.abs(d * 2 - 1)
    raised = _smoothstep(1 - bar * 2, 1.0, band)
    # Only the middle of each cell carries a bar, so they read as short studs.
    window = _smoothstep(0.0, 0.18, np.minimum(np.minimum(fy, 1 - fy),
                                               np.minimum(fx, 1 - fx)))
    raised = raised * window
    plate, plate_r = _brushed(rng, size, tint=tint, depth=0.08, rough=rough)
    albedo = plate * (1 - lift * 0.35 + lift * 0.7 * raised)[..., None]
    return albedo, plate_r * (1 - 0.35 * raised)


def _tiles(rng, size, *, across=5, grout=0.06, glaze=0.20,
           tones=((0.62, 0.58, 0.50),), grout_c=(0.42, 0.40, 0.37),
           rough=(0.12, 0.90)):
    """Glazed tile: a grid, a grout line, and a slight tone per tile.

    Roughness is nearly binary here — glaze is glossy and grout is not — and
    that contrast is most of why a tiled surface looks tiled.
    """
    y, x = np.mgrid[0:size, 0:size] / size * across
    fy, fx = y % 1.0, x % 1.0
    ident = (np.floor(y).astype(np.int64) * across + np.floor(x).astype(np.int64))
    tone = np.asarray(tones)[rng.integers(0, len(tones), across * across)[ident]]
    tone = tone * (0.90 + rng.random(across * across)[ident] * 0.22)[..., None]
    tone = tone * (1 - glaze / 2 + glaze * _fbm(rng, size, freq=12, octaves=3))[..., None]
    inside = _smoothstep(0.0, grout, np.minimum(np.minimum(fy, 1 - fy),
                                                np.minimum(fx, 1 - fx)))
    albedo = _mix(np.asarray(grout_c), tone, inside)
    return albedo, rough[1] - (rough[1] - rough[0]) * inside


def _cracked(rng, size, *, cells=6, width=0.035, body=(0.62, 0.76, 0.84),
             deep=(0.24, 0.42, 0.54), line=(0.90, 0.95, 0.98), haze=0.35,
             rough=(0.06, 0.30)):
    """Ice, obsidian, glassy fracture: internal cloud plus bright crack planes.

    Distance-to-cell-boundary gives a crack network for free, and lighting it
    *brighter* than the body is what makes it read as an internal fracture
    catching light rather than a painted-on line.
    """
    # The crack is `d2 - d1`, the ridge equidistant from two cells. `d2` alone —
    # which this used to sample — is the distance to the second-nearest seed and
    # is large nearly everywhere, so the cracks never appeared at all.
    d1, d2, ident = _worley(rng, size, cells, 0.9)
    e1, e2, _ = _worley(rng, size, cells * 3, 0.9)
    crack = 1 - _smoothstep(0.0, width, d2 - d1)
    fine = 1 - _smoothstep(0.0, width * 0.7, e2 - e1)
    cloud = _fbm(rng, size, freq=3, octaves=5)
    depth = _by_cell(rng, ident, 0.55, 1.0)
    albedo = _mix(deep, body, np.clip(cloud * haze + depth * (1 - haze), 0, 1))
    albedo = _mix(albedo, line, crack * 0.85)
    albedo = _mix(albedo, line, fine * 0.30)
    return albedo, rough[1] - (rough[1] - rough[0]) * np.clip(crack + fine * 0.5, 0, 1)


def _thatch(rng, size, *, courses=5, density=70, overhang=0.22,
            light=(0.60, 0.46, 0.20), dark=(0.22, 0.16, 0.06), rough=(0.90, 0.99)):
    """Thatch: courses of straw, each course shadowed by the one above it."""
    straw = _fbm(rng, size, freq=density, octaves=3, stretch=0.06)
    straw = np.clip(0.5 + (straw - 0.5) * 3.4, 0, 1)
    _, v = _rows(size, courses)
    shade = _smoothstep(0.0, overhang, v) * 0.55 + 0.45
    t = np.clip(straw * shade * (0.6 + 0.7 * _fbm(rng, size, freq=4, octaves=3)), 0, 1)
    return _mix(dark, light, t), rough[1] - (rough[1] - rough[0]) * t


def _leather(rng, size, *, cells=20, gap=0.20, creases=0.35,
             tone=(0.30, 0.17, 0.10), rough=(0.42, 0.72)):
    """Pebbled hide: fine Worley cells with soft valleys and a few long creases."""
    albedo, r = _cells(rng, size, cells=cells, jitter=0.9, gap=gap, dome=0.48,
                       spread=0.26, tones=(tone,),
                       gap_color=tuple(c * 0.40 for c in tone), grit=0.14,
                       rough=rough)
    if creases:
        cr = _ridged(rng, size, freq=3, octaves=4, stretch=3.0) ** 8
        albedo = albedo * (1 - creases * cr)[..., None]
        r = np.clip(r + 0.15 * cr, 0, 1)
    return albedo, r


# --------------------------------------------------------------------------
# the library
# --------------------------------------------------------------------------
#
# `PALETTE` is the PBR factor set — the whole material for a flat family, and
# the mean colour for a textured one. The original twelve keep their exact
# values: everything downstream (and a good few tests) knows what `wood` and
# `paint` look like, and a texture library is no reason to move them.
PALETTE: dict[str, dict] = {
    # --- the original twelve -------------------------------------------------
    "metal":      dict(baseColorFactor=[0.62, 0.65, 0.70, 1.0], metallicFactor=0.95, roughnessFactor=0.35),
    "dark_metal": dict(baseColorFactor=[0.28, 0.29, 0.32, 1.0], metallicFactor=0.90, roughnessFactor=0.45),
    "glass":      dict(baseColorFactor=[0.45, 0.68, 0.85, 0.45], metallicFactor=0.0, roughnessFactor=0.05),
    "rubber":     dict(baseColorFactor=[0.09, 0.09, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.92),
    "wood":       dict(baseColorFactor=[0.52, 0.34, 0.18, 1.0], metallicFactor=0.0, roughnessFactor=0.75),
    "stone":      dict(baseColorFactor=[0.48, 0.47, 0.44, 1.0], metallicFactor=0.0, roughnessFactor=0.88),
    "fabric":     dict(baseColorFactor=[0.35, 0.33, 0.40, 1.0], metallicFactor=0.0, roughnessFactor=0.95),
    "leather":    dict(baseColorFactor=[0.33, 0.20, 0.13, 1.0], metallicFactor=0.0, roughnessFactor=0.65),
    # Neutral on purpose. "paint" is both the body-panel material and the
    # fallback for anything unrecognised, so a saturated colour here would turn
    # most scenes an arbitrary red. We do not know what colour the thing is —
    # pass `color` when you do.
    "paint":      dict(baseColorFactor=[0.82, 0.82, 0.84, 1.0], metallicFactor=0.10, roughnessFactor=0.45),
    "plastic":    dict(baseColorFactor=[0.85, 0.85, 0.87, 1.0], metallicFactor=0.0, roughnessFactor=0.40),
    "gold":       dict(baseColorFactor=[0.85, 0.68, 0.25, 1.0], metallicFactor=1.0, roughnessFactor=0.30),
    "emissive":   dict(baseColorFactor=[0.95, 0.85, 0.55, 1.0], metallicFactor=0.0, roughnessFactor=0.20,
                       emissiveFactor=[0.9, 0.75, 0.35]),

    # --- masonry -------------------------------------------------------------
    "brick":       dict(baseColorFactor=[0.44, 0.22, 0.17, 1.0], metallicFactor=0.0, roughnessFactor=0.90),
    "cobblestone": dict(baseColorFactor=[0.33, 0.32, 0.30, 1.0], metallicFactor=0.0, roughnessFactor=0.88),
    "sandstone":   dict(baseColorFactor=[0.62, 0.50, 0.33, 1.0], metallicFactor=0.0, roughnessFactor=0.90),
    "limestone":   dict(baseColorFactor=[0.70, 0.68, 0.60, 1.0], metallicFactor=0.0, roughnessFactor=0.88),
    "granite":     dict(baseColorFactor=[0.42, 0.40, 0.40, 1.0], metallicFactor=0.0, roughnessFactor=0.62),
    "marble":      dict(baseColorFactor=[0.72, 0.71, 0.69, 1.0], metallicFactor=0.0, roughnessFactor=0.22),
    "slate":       dict(baseColorFactor=[0.22, 0.23, 0.26, 1.0], metallicFactor=0.0, roughnessFactor=0.70),
    "basalt":      dict(baseColorFactor=[0.14, 0.14, 0.15, 1.0], metallicFactor=0.0, roughnessFactor=0.86),
    "concrete":    dict(baseColorFactor=[0.55, 0.55, 0.53, 1.0], metallicFactor=0.0, roughnessFactor=0.92),
    "asphalt":     dict(baseColorFactor=[0.11, 0.11, 0.12, 1.0], metallicFactor=0.0, roughnessFactor=0.94),
    "plaster":     dict(baseColorFactor=[0.80, 0.77, 0.70, 1.0], metallicFactor=0.0, roughnessFactor=0.90),
    "stucco":      dict(baseColorFactor=[0.74, 0.70, 0.62, 1.0], metallicFactor=0.0, roughnessFactor=0.94),
    "terracotta":  dict(baseColorFactor=[0.60, 0.29, 0.17, 1.0], metallicFactor=0.0, roughnessFactor=0.84),
    "tile":        dict(baseColorFactor=[0.58, 0.56, 0.50, 1.0], metallicFactor=0.0, roughnessFactor=0.28),
    "gravel":      dict(baseColorFactor=[0.38, 0.36, 0.33, 1.0], metallicFactor=0.0, roughnessFactor=0.94),

    # --- timber --------------------------------------------------------------
    "timber":  dict(baseColorFactor=[0.36, 0.24, 0.13, 1.0], metallicFactor=0.0, roughnessFactor=0.88),
    "planks":  dict(baseColorFactor=[0.46, 0.31, 0.17, 1.0], metallicFactor=0.0, roughnessFactor=0.80),
    "oak":     dict(baseColorFactor=[0.44, 0.30, 0.16, 1.0], metallicFactor=0.0, roughnessFactor=0.66),
    "walnut":  dict(baseColorFactor=[0.20, 0.12, 0.07, 1.0], metallicFactor=0.0, roughnessFactor=0.48),
    "pine":    dict(baseColorFactor=[0.66, 0.50, 0.29, 1.0], metallicFactor=0.0, roughnessFactor=0.74),
    "bark":    dict(baseColorFactor=[0.19, 0.14, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.95),
    "thatch":  dict(baseColorFactor=[0.38, 0.28, 0.12, 1.0], metallicFactor=0.0, roughnessFactor=0.96),

    # --- metals --------------------------------------------------------------
    "steel_plate":    dict(baseColorFactor=[0.54, 0.56, 0.59, 1.0], metallicFactor=0.95, roughnessFactor=0.42),
    "diamond_plate":  dict(baseColorFactor=[0.50, 0.52, 0.55, 1.0], metallicFactor=0.95, roughnessFactor=0.38),
    "corroded_steel": dict(baseColorFactor=[0.34, 0.26, 0.20, 1.0], metallicFactor=0.60, roughnessFactor=0.85),
    "rusted_iron":    dict(baseColorFactor=[0.32, 0.17, 0.09, 1.0], metallicFactor=0.35, roughnessFactor=0.92),
    "wrought_iron":   dict(baseColorFactor=[0.10, 0.10, 0.11, 1.0], metallicFactor=0.80, roughnessFactor=0.58),
    "copper":         dict(baseColorFactor=[0.55, 0.29, 0.16, 1.0], metallicFactor=1.0, roughnessFactor=0.34),
    "verdigris":      dict(baseColorFactor=[0.26, 0.46, 0.40, 1.0], metallicFactor=0.30, roughnessFactor=0.78),
    "brass":          dict(baseColorFactor=[0.66, 0.52, 0.22, 1.0], metallicFactor=1.0, roughnessFactor=0.32),
    "bronze":         dict(baseColorFactor=[0.44, 0.30, 0.15, 1.0], metallicFactor=1.0, roughnessFactor=0.44),
    "lead":           dict(baseColorFactor=[0.34, 0.35, 0.37, 1.0], metallicFactor=0.85, roughnessFactor=0.66),
    "gold_leaf":      dict(baseColorFactor=[0.80, 0.63, 0.22, 1.0], metallicFactor=1.0, roughnessFactor=0.22),

    # --- nature --------------------------------------------------------------
    "grass": dict(baseColorFactor=[0.22, 0.32, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.95),
    "moss":  dict(baseColorFactor=[0.16, 0.26, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.96),
    "dirt":  dict(baseColorFactor=[0.29, 0.21, 0.13, 1.0], metallicFactor=0.0, roughnessFactor=0.94),
    "mud":   dict(baseColorFactor=[0.18, 0.13, 0.09, 1.0], metallicFactor=0.0, roughnessFactor=0.55),
    "sand":  dict(baseColorFactor=[0.72, 0.62, 0.42, 1.0], metallicFactor=0.0, roughnessFactor=0.93),
    "snow":  dict(baseColorFactor=[0.90, 0.92, 0.95, 1.0], metallicFactor=0.0, roughnessFactor=0.60),
    "ice":   dict(baseColorFactor=[0.60, 0.75, 0.84, 1.0], metallicFactor=0.0, roughnessFactor=0.12),
    "bone":  dict(baseColorFactor=[0.76, 0.72, 0.61, 1.0], metallicFactor=0.0, roughnessFactor=0.72),

    # --- fantasy and finishes ------------------------------------------------
    "obsidian": dict(baseColorFactor=[0.07, 0.06, 0.09, 1.0], metallicFactor=0.20, roughnessFactor=0.12),
    "crystal":  dict(baseColorFactor=[0.55, 0.72, 0.86, 0.60], metallicFactor=0.0, roughnessFactor=0.08),
    "neon":     dict(baseColorFactor=[0.35, 0.85, 0.95, 1.0], metallicFactor=0.0, roughnessFactor=0.25,
                     emissiveFactor=[0.30, 0.85, 0.98]),
    "canvas":   dict(baseColorFactor=[0.62, 0.58, 0.48, 1.0], metallicFactor=0.0, roughnessFactor=0.95),
}

# Which recipe draws each family, its arguments, and how many studs one tile
# covers. Absence from this table means "no map" — a flat factor is the honest
# material for glass, paint and plastic, and inventing a pattern for them would
# only add noise.
#
# The tile size is chosen so the *feature* is right: a brick is about 0.4 studs
# tall, so a 7-course tile wants ~3 studs. The pixel size follows from that —
# 256 px over ~2.5 studs is around 100 px per stud, which is roughly what a
# 1024-px projected atlas gives a part this size, matched on purpose so a
# scripted wall and a generated one do not read at different resolutions. The
# families whose pattern *is* the point (brick courses, cobbles, granite grain,
# marble veins) pay for 384.
TEXTURE: dict[str, tuple] = {
    # family: (generator, kwargs, studs per tile, pixels)
    "wood":   (_grain, dict(rings=15.0, knots=1, sharp=6.0, depth=0.80,
                            fibre=0.26, pores=0.12), 2.4, 384),
    "timber": (_grain, dict(rings=11.0, knots=1, warp=0.14, sharp=4.0, fibre=0.46,
                            depth=0.62, sweep=0.06, knot_size=0.075,
                            late=(0.17, 0.11, 0.06), early=(0.44, 0.31, 0.18),
                            rough=(0.78, 0.96)), 3.0, 384),
    "planks": (_planks, dict(boards=4, rings=18.0, knots=1, sharp=6.0,
                             pores=0.12, fibre=0.26, depth=0.76), 3.2, 384),
    "oak":    (_grain, dict(rings=13.0, knots=1, sharp=7.0, depth=0.72, pores=0.38,
                            fibre=0.22, sweep=0.11,
                            late=(0.24, 0.15, 0.06), early=(0.60, 0.44, 0.24),
                            rough=(0.46, 0.74)), 2.2, 384),
    "walnut": (_grain, dict(rings=12.0, knots=0, warp=0.16, sharp=4.6, depth=0.86,
                            fibre=0.36, sweep=0.13,
                            late=(0.06, 0.035, 0.02), early=(0.34, 0.21, 0.13),
                            rough=(0.32, 0.60)), 2.6, 384),
    "pine":   (_grain, dict(rings=19.0, knots=2, sharp=7.0, depth=0.66,
                            knot_size=0.065, sweep=0.06,
                            late=(0.42, 0.26, 0.11), early=(0.82, 0.66, 0.42),
                            rough=(0.60, 0.84)), 2.0, 384),
    "bark":   (_bark, dict(), 1.8, 256),
    "thatch": (_thatch, dict(courses=4, density=40, overhang=0.34), 2.6, 256),

    "brick":       (_masonry, dict(courses=7, per_course=3, joint=0.05), 3.0, 512),
    "cobblestone": (_cells, dict(cells=6, gap=0.07, dome=0.55, spread=0.34, warp=0.02,
                                 tones=((0.30, 0.29, 0.27), (0.38, 0.36, 0.32),
                                        (0.24, 0.24, 0.24), (0.34, 0.31, 0.26)),
                                 gap_color=(0.13, 0.13, 0.12)), 2.6, 384),
    "gravel":      (_cells, dict(cells=17, gap=0.10, dome=0.50, spread=0.40,
                                 tones=((0.38, 0.36, 0.32), (0.30, 0.29, 0.28),
                                        (0.46, 0.43, 0.38)),
                                 gap_color=(0.14, 0.13, 0.12), rough=(0.86, 0.99)), 1.4, 256),
    "granite":     (_cells, dict(cells=26, gap=0.16, dome=0.10, spread=0.55, flat=True,
                                 tones=((0.56, 0.52, 0.50), (0.30, 0.29, 0.30),
                                        (0.68, 0.62, 0.58), (0.09, 0.09, 0.10),
                                        (0.44, 0.36, 0.34)),
                                 gap_color=(0.20, 0.19, 0.20), grit=0.10,
                                 rough=(0.48, 0.72)), 1.6, 384),
    "stone":       (_cells, dict(cells=5, gap=0.055, dome=0.42, spread=0.26,
                                 tones=((0.46, 0.45, 0.42), (0.54, 0.52, 0.47),
                                        (0.38, 0.38, 0.36)),
                                 gap_color=(0.22, 0.22, 0.20)), 2.8, 384),
    "sandstone":   (_bedded, dict(beds=11, stretch=5.0, contrast=1.7, blotch=0.40,
                                  grit=0.22), 2.6, 384),
    "limestone":   (_bedded, dict(beds=7, stretch=3.0, contrast=1.2, blotch=0.55,
                                  plates=4, plate_aspect=1.8, plate_spread=0.10,
                                  plate_gap=0.05, grit=0.30,
                                  light=(0.84, 0.82, 0.74), dark=(0.56, 0.54, 0.47)), 2.8, 384),
    "slate":       (_bedded, dict(beds=18, stretch=7.0, contrast=1.9, blotch=0.30,
                                  plates=5, plate_aspect=2.8, plate_spread=0.26,
                                  plate_gap=0.035, grit=0.12,
                                  light=(0.34, 0.36, 0.42), dark=(0.09, 0.10, 0.13),
                                  rough=(0.50, 0.80)), 2.2, 384),
    "marble":      (_marble, dict(), 3.6, 384),
    "basalt":      (_speckle, dict(freq=5, contrast=1.5, gain=0.62, grain=0.30,
                                   pits=0.60, pit_cells=18, pit_size=0.26,
                                   light=(0.23, 0.23, 0.24),
                                   dark=(0.05, 0.05, 0.06)), 2.0, 256),
    "concrete":    (_speckle, dict(freq=4, contrast=1.15, gain=0.62, grain=0.26,
                                   pits=0.34, pit_cells=24, flecks=0.014,
                                   light=(0.68, 0.67, 0.64),
                                   dark=(0.40, 0.40, 0.39)), 3.0, 384),
    "asphalt":     (_speckle, dict(freq=8, contrast=1.4, gain=0.62, grain=0.45,
                                   flecks=0.13, fleck_dark=0.70,
                                   light=(0.18, 0.18, 0.19), dark=(0.04, 0.04, 0.05),
                                   rough=(0.88, 0.99)), 2.0, 256),
    "plaster":     (_speckle, dict(freq=3, octaves=5, contrast=0.85, stretch=2.4,
                                   grain=0.16, grain_freq=54,
                                   light=(0.90, 0.87, 0.80), dark=(0.66, 0.63, 0.56)), 3.4, 256),
    "stucco":      (_speckle, dict(freq=9, contrast=1.3, gain=0.60, grain=0.30,
                                   pits=0.26, pit_cells=28, pit_size=0.22,
                                   light=(0.86, 0.82, 0.74),
                                   dark=(0.56, 0.52, 0.45)), 2.4, 256),
    "terracotta":  (_speckle, dict(freq=5, contrast=0.95, gain=0.60, grain=0.24,
                                   flecks=0.020, pits=0.14, pit_cells=34, pit_size=0.14,
                                   light=(0.72, 0.38, 0.22), dark=(0.44, 0.20, 0.11),
                                   rough=(0.72, 0.92)), 2.4, 256),
    "tile":        (_tiles, dict(across=4, tones=((0.60, 0.56, 0.48),
                                                  (0.55, 0.53, 0.47),
                                                  (0.64, 0.58, 0.46))), 2.0, 256),

    "metal":          (_brushed, dict(scratches=0.18), 3.0, 256),
    "dark_metal":     (_brushed, dict(tint=(0.28, 0.29, 0.32), depth=0.18,
                                      scratches=0.22, rough=(0.34, 0.62)), 3.0, 256),
    "steel_plate":    (_brushed, dict(tint=(0.54, 0.56, 0.59), depth=0.12, mottle=0.10,
                                      scratches=0.25, rough=(0.30, 0.58)), 3.0, 256),
    "diamond_plate":  (_tread, dict(), 2.4, 384),
    "wrought_iron":   (_cells, dict(cells=13, gap=0.22, dome=0.35, spread=0.30, flat=False,
                                    tones=((0.11, 0.11, 0.12),), gap_color=(0.05, 0.05, 0.06),
                                    grit=0.22, rough=(0.42, 0.72)), 1.8, 256),
    "copper":         (_brushed, dict(tint=(0.60, 0.31, 0.17), mottle=0.12,
                                      rough=(0.22, 0.46)), 2.6, 256),
    "brass":          (_brushed, dict(tint=(0.72, 0.57, 0.24), mottle=0.10,
                                      scratches=0.14, rough=(0.20, 0.44)), 2.6, 256),
    "bronze":         (_brushed, dict(tint=(0.48, 0.33, 0.16), depth=0.18, mottle=0.20,
                                      rough=(0.32, 0.60)), 2.6, 256),
    "lead":           (_brushed, dict(tint=(0.36, 0.37, 0.39), depth=0.10, stretch=3.0,
                                      mottle=0.22, rough=(0.55, 0.80)), 2.6, 256),
    "gold":           (_brushed, dict(tint=(0.92, 0.74, 0.28), depth=0.10,
                                      rough=(0.18, 0.40)), 2.6, 256),
    "gold_leaf":      (_cells, dict(cells=11, gap=0.40, dome=0.20, spread=0.34, flat=True,
                                    tones=((0.88, 0.70, 0.26),), gap_color=(0.52, 0.40, 0.13),
                                    grit=0.14, rough=(0.14, 0.34)), 1.6, 256),
    "rusted_iron":    (_rust, dict(coverage=0.80, metal=(0.30, 0.30, 0.31)), 2.4, 384),
    "corroded_steel": (_rust, dict(coverage=0.50, metal=(0.48, 0.50, 0.53),
                                   tones=((0.36, 0.18, 0.08), (0.44, 0.24, 0.12),
                                          (0.26, 0.17, 0.12))), 2.4, 384),
    "verdigris":      (_patina, dict(), 2.4, 384),

    "grass": (_blades, dict(density=46, clump=4), 1.6, 256),
    "moss":  (_blades, dict(density=26, clump=7, stretch=0.55, contrast=0.55,
                            tip=(0.26, 0.40, 0.12), root=(0.06, 0.12, 0.04),
                            dead_mix=0.08), 1.2, 256),
    "dirt":  (_speckle, dict(freq=5, contrast=1.35, gain=0.62, grain=0.34,
                             pits=0.34, pit_cells=22, flecks=0.030,
                             light=(0.42, 0.31, 0.19), dark=(0.16, 0.11, 0.07)), 2.2, 256),
    "mud":   (_speckle, dict(freq=4, contrast=1.4, gain=0.60, grain=0.20,
                             pits=0.24, pit_cells=12, pit_size=0.34,
                             light=(0.30, 0.22, 0.15), dark=(0.08, 0.06, 0.04),
                             rough=(0.28, 0.72)), 2.6, 256),
    "sand":  (_speckle, dict(freq=3, octaves=6, contrast=0.85, stretch=6.0,
                             grain=0.30, grain_freq=64, flecks=0.070,
                             fleck_dark=0.26, light=(0.86, 0.76, 0.53),
                             dark=(0.56, 0.46, 0.29)), 1.8, 256),
    "snow":  (_speckle, dict(freq=3, octaves=5, contrast=0.55, stretch=2.0,
                             grain=0.14, grain_freq=70, flecks=0.045,
                             fleck_dark=0.18, light=(0.99, 0.99, 1.00),
                             dark=(0.74, 0.79, 0.88), rough=(0.40, 0.78)), 2.6, 256),
    "bone":  (_speckle, dict(freq=4, contrast=0.95, gain=0.60, grain=0.22,
                             pits=0.24, pit_cells=30, pit_size=0.14,
                             light=(0.88, 0.84, 0.72), dark=(0.58, 0.54, 0.44),
                             rough=(0.54, 0.86)), 2.0, 256),
    "ice":      (_cracked, dict(), 3.0, 384),
    # Obsidian is volcanic glass, so the "cracks" want to be conchoidal
    # highlights rather than a stone's joints: narrow, dim, and mostly hidden
    # under a heavy swirl (haze 0.85) of the melt it froze out of.
    "obsidian": (_cracked, dict(cells=4, width=0.012, body=(0.11, 0.10, 0.14),
                                deep=(0.02, 0.02, 0.035), line=(0.22, 0.19, 0.30),
                                haze=0.85, rough=(0.05, 0.18)), 3.2, 384),
    "crystal":  (_cells, dict(cells=6, gap=0.30, dome=0.0, spread=0.45, flat=True,
                              tones=((0.52, 0.72, 0.88), (0.62, 0.80, 0.92),
                                     (0.40, 0.60, 0.80)),
                              gap_color=(0.82, 0.92, 0.98), grit=0.06,
                              rough=(0.05, 0.18)), 2.2, 256),

    "fabric":  (_weave, dict(threads=26), 1.6, 256),
    "canvas":  (_weave, dict(threads=17, warp_c=(0.64, 0.60, 0.49),
                             weft_c=(0.57, 0.53, 0.43), depth=0.34, fuzz=0.40), 1.8, 256),
    "leather": (_leather, dict(), 1.4, 256),
    "rubber":  (_speckle, dict(freq=18, contrast=0.85, gain=0.60, grain=0.35,
                               light=(0.13, 0.13, 0.14), dark=(0.06, 0.06, 0.07),
                               rough=(0.86, 0.99)), 1.6, 256),
}

# Roblox's `Material` enum for the families that have a counterpart. A Roblox
# developer asking for "granite" should get a texture *and* the right enum, so
# the part behaves correctly for physics, footstep sound and the material
# variant system — which a TextureID alone does not do.
#
# Only real enum members appear here, and only where the match is honest. There
# is no Roblox "brass", so brass maps to Metal rather than to something close in
# colour but wrong in behaviour.
ROBLOX: dict[str, str] = {
    "metal": "Metal", "dark_metal": "Metal", "glass": "Glass", "rubber": "Rubber",
    "wood": "Wood", "stone": "Rock", "fabric": "Fabric", "leather": "Fabric",
    "paint": "SmoothPlastic", "plastic": "Plastic", "gold": "Foil",
    "emissive": "Neon",

    "brick": "Brick", "cobblestone": "Cobblestone", "sandstone": "Sandstone",
    "limestone": "Limestone", "granite": "Granite", "marble": "Marble",
    "slate": "Slate", "basalt": "Basalt", "concrete": "Concrete",
    "asphalt": "Asphalt", "plaster": "Plaster", "stucco": "Plaster",
    "terracotta": "Brick", "tile": "CeramicTiles", "gravel": "Pebble",

    "timber": "Wood", "planks": "WoodPlanks", "oak": "Wood", "walnut": "Wood",
    "pine": "Wood", "bark": "Wood", "thatch": "LeafyGrass",

    "steel_plate": "Metal", "diamond_plate": "DiamondPlate",
    "corroded_steel": "CorrodedMetal", "rusted_iron": "CorrodedMetal",
    "wrought_iron": "Metal", "copper": "Metal", "verdigris": "CorrodedMetal",
    "brass": "Metal", "bronze": "Metal", "lead": "Metal", "gold_leaf": "Foil",

    "grass": "Grass", "moss": "LeafyGrass", "dirt": "Ground", "mud": "Mud",
    "sand": "Sand", "snow": "Snow", "ice": "Ice", "bone": "Limestone",

    "obsidian": "Basalt", "crystal": "Glass", "neon": "Neon", "canvas": "Fabric",
}

# Themed packs. This is the asset-store experience the library is imitating: you
# do not want "a material", you want a coherent set that looks like it came from
# one place, so a facade built out of `medieval` cannot end up with a granite
# plinth under a plastic wall. Packs overlap deliberately — stone is in three of
# them because stone is in three of them.
PACKS: dict[str, list[str]] = {
    "medieval": [
        "brick", "cobblestone", "sandstone", "limestone", "granite", "marble",
        "slate", "stone", "plaster", "stucco", "terracotta", "timber", "planks",
        "oak", "wood", "thatch", "wrought_iron", "lead", "bronze", "canvas",
        "leather", "gold",
    ],
    "industrial": [
        "concrete", "asphalt", "steel_plate", "diamond_plate", "corroded_steel",
        "rusted_iron", "dark_metal", "metal", "copper", "brass", "lead",
        "rubber", "glass", "plastic", "paint", "tile", "gravel", "neon",
        "wood",
    ],
    "natural": [
        "grass", "moss", "dirt", "mud", "sand", "snow", "ice", "gravel",
        "basalt", "granite", "stone", "bark", "pine", "timber", "bone", "thatch",
    ],
    "fantasy": [
        "obsidian", "crystal", "gold_leaf", "gold", "marble", "verdigris",
        "bronze", "wrought_iron", "bone", "walnut", "canvas", "emissive", "neon",
    ],
    "interior": [
        "oak", "walnut", "pine", "planks", "marble", "tile", "plaster", "fabric",
        "canvas", "leather", "brass", "glass", "paint", "plastic", "gold",
    ],
}

# Substring -> material family. The longest matching keyword wins, so
# "windshield" resolves before a bare "shield" would — and so the new specific
# words ("cobble", "thatch") beat the general ones ("stone", "roof") they
# contain or overlap.
KEYWORDS: dict[str, str] = {
    "canopy": "glass", "windshield": "glass", "window": "glass", "glass": "glass",
    "lens": "glass", "screen": "glass", "visor": "glass",
    "tire": "rubber", "tyre": "rubber", "wheel": "rubber", "tread": "rubber",
    "grip": "rubber", "seal": "rubber",
    "engine": "metal", "exhaust": "metal", "turbine": "metal", "propeller": "metal",
    "blade": "metal", "barrel": "metal", "turret": "metal", "cannon": "metal",
    "gun": "metal", "strut": "metal", "frame": "metal", "rail": "metal",
    "pipe": "metal", "antenna": "metal", "bolt": "metal", "hinge": "metal",
    "track": "dark_metal", "chassis": "dark_metal", "undercarriage": "dark_metal",
    "grille": "dark_metal", "vent": "dark_metal",
    "handle": "wood", "stock": "wood", "crate": "wood", "plank": "wood",
    "barrel_wood": "wood", "mast": "wood", "deck": "wood",
    "rock": "stone", "boulder": "stone", "brick": "stone", "wall": "stone",
    "pillar": "stone", "statue": "stone",
    "seat": "fabric", "cushion": "fabric", "flag": "fabric", "banner": "fabric",
    "sail": "fabric", "curtain": "fabric",
    "belt": "leather", "strap": "leather", "saddle": "leather", "boot": "leather",
    "body": "paint", "hull": "paint", "fuselage": "paint", "wing": "paint",
    "door": "paint", "panel": "paint", "hood": "paint", "roof": "paint",
    "fender": "paint", "tail": "paint", "nose": "paint",
    "button": "plastic", "knob": "plastic", "console": "plastic",
    "dashboard": "plastic", "casing": "plastic",
    "trim": "gold", "emblem": "gold", "badge": "gold", "ornament": "gold",
    "light": "emissive", "lamp": "emissive", "glow": "emissive",
    "headlight": "emissive", "thruster": "emissive",

    # --- the new families ----------------------------------------------------
    # These are longer than the words above that they overlap, which is exactly
    # how the longest-match rule is meant to be used: "brickwork" is brick, a
    # bare "wall" is still generic stone.
    "brickwork": "brick", "masonry": "brick", "chimney": "brick",
    "cobble": "cobblestone", "cobblestone": "cobblestone", "street": "cobblestone",
    "sandstone": "sandstone", "limestone": "limestone", "granite": "granite",
    "marble": "marble", "slate": "slate", "basalt": "basalt",
    "concrete": "concrete", "kerb": "concrete", "curb": "concrete",
    "asphalt": "asphalt", "tarmac": "asphalt", "road": "asphalt",
    "plaster": "plaster", "stucco": "stucco", "render": "plaster",
    "terracotta": "terracotta", "tile": "tile", "tiling": "tile",
    "gravel": "gravel", "shingle_bed": "gravel", "ballast": "gravel",
    "timber": "timber", "beam": "timber", "joist": "timber", "rafter": "timber",
    "floorboard": "planks", "decking": "planks", "planking": "planks",
    "oak": "oak", "walnut": "walnut", "pine": "pine", "bark": "bark",
    "trunk": "bark", "thatch": "thatch", "straw": "thatch",
    "steel": "steel_plate", "plating": "steel_plate", "bulkhead": "steel_plate",
    "diamond_plate": "diamond_plate", "checker_plate": "diamond_plate",
    "corroded": "corroded_steel", "rust": "rusted_iron", "rusty": "rusted_iron",
    "wrought": "wrought_iron", "portcullis": "wrought_iron",
    "grate": "wrought_iron", "gate": "wrought_iron",
    "copper": "copper", "verdigris": "verdigris", "patina": "verdigris",
    "brass": "brass", "bronze": "bronze", "lead": "lead", "flashing": "lead",
    "gold_leaf": "gold_leaf", "gilding": "gold_leaf", "gilt": "gold_leaf",
    "grass": "grass", "turf": "grass", "lawn": "grass", "moss": "moss",
    "dirt": "dirt", "soil": "dirt", "earth": "dirt", "mud": "mud",
    "sand": "sand", "dune": "sand", "snow": "snow", "ice": "ice",
    "icicle": "ice", "bone": "bone", "skull": "bone", "tusk": "bone",
    "obsidian": "obsidian", "crystal": "crystal", "gem": "crystal",
    "shard": "crystal", "neon": "neon", "sign": "neon",
    "canvas": "canvas", "tarp": "canvas", "awning": "canvas", "tent": "canvas",
}

DEFAULT_MATERIAL = "paint"

# How far a seed is allowed to move a family's colour. Small on purpose: the
# point is that twenty walls in a facade are not clones, not that they are
# different materials. About the largest spread that still reads as one wall
# having weathered unevenly.
#
# Saturation moves least. On a saturated family it is the term that separates
# the channels rather than moving them together, so the same percentage there
# reads as roughly twice the change it does in value.
SEED_VALUE = 0.08
SEED_HUE = 0.025
SEED_SAT = 0.04


def resolve(part_name: str, explicit: str | None = None) -> tuple[str, dict]:
    """Pick a material for a part. Explicit choice always wins over the guess."""
    if explicit:
        if explicit not in PALETTE:
            raise ValueError(
                f"unknown material {explicit!r}, expected one of {sorted(PALETTE)}"
            )
        return explicit, PALETTE[explicit]

    name = part_name.lower()
    hits = [(len(kw), fam) for kw, fam in KEYWORDS.items() if kw in name]
    family = max(hits)[1] if hits else DEFAULT_MATERIAL
    return family, PALETTE[family]


def parse_color(value: str) -> list[float]:
    """"#rrggbb" or "#rrggbbaa" -> linear-ish float RGBA.

    glTF baseColorFactor is linear, while a hex colour anyone types is sRGB, so
    the channels are converted rather than divided by 255 — skipping that makes
    every colour come out visibly too bright.
    """
    h = value.strip().lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError(f"colour must be #rrggbb or #rrggbbaa, got {value!r}")
    try:
        channels = [int(h[i:i + 2], 16) / 255 for i in range(0, len(h), 2)]
    except ValueError:
        raise ValueError(f"colour is not valid hex: {value!r}") from None

    def to_linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    rgba = [to_linear(c) for c in channels[:3]]
    rgba.append(channels[3] if len(channels) == 4 else 1.0)  # alpha stays linear
    return rgba


# --------------------------------------------------------------------------
# building and caching the maps
# --------------------------------------------------------------------------

def _to_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear light -> the sRGB bytes a glTF base-colour PNG is defined to hold.

    Writing linear values into that PNG is the same mistake `parse_color` exists
    to avoid, in the other direction: everything comes out washed and pale.
    """
    c = np.clip(linear, 0.0, 1.0)
    out = np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _normalise(albedo: np.ndarray, target: list[float]) -> np.ndarray:
    """Scale each channel so the texture's mean is the family's flat colour.

    This is what keeps `PALETTE` honest. A part built without UVs gets the flat
    factor and a part built with them gets the map; if the two disagreed, turning
    texturing on would visibly recolour a scene, and `PALETTE` would stop being
    a description of the family. The correction is small by construction — the
    recipes are authored near their stated colour — so it is a nudge rather
    than a filter.
    """
    mean = albedo.reshape(-1, 3).mean(axis=0)
    gain = np.asarray(target[:3]) / np.maximum(mean, 1e-6)
    return np.clip(albedo * gain, 0.0, 1.0)


@functools.lru_cache(maxsize=None)
def _maps(family: str) -> tuple[Image.Image, Image.Image] | None:
    """Draw one family's maps, once per process. Returns (baseColor, metalRough).

    Cached rather than pre-computed at import: a build that uses four families
    should not pay for fifty-seven, and the whole library is still only about a
    second if something does want all of it.
    """
    entry = TEXTURE.get(family)
    if entry is None:
        return None
    generator, kwargs, _tile, size = entry
    albedo, rough = generator(_rng(_family_seed(family)), size, **kwargs)
    albedo = _normalise(np.asarray(albedo, dtype=float), PALETTE[family]["baseColorFactor"])

    base = Image.fromarray(_to_srgb(albedo), mode="RGB")
    # glTF packs this one channel-wise: G is roughness, B is metallic, and both
    # multiply the corresponding factor. B stays 255 so `metallicFactor` still
    # says whether the family is a metal; R is unused and left at 255.
    mr = np.stack([
        np.full(rough.shape, 255, np.uint8),
        np.clip(np.asarray(rough) * 255 + 0.5, 0, 255).astype(np.uint8),
        np.full(rough.shape, 255, np.uint8),
    ], axis=-1)
    # Half resolution, and it costs nothing visible. Roughness carries the broad
    # story — this patch is rusted, that one is not — while the pixel-scale
    # detail that needs full resolution lives in the base colour. Halving it
    # takes about three quarters off the second map, and every one of these gets
    # embedded in every GLB that uses the family.
    mr_img = Image.fromarray(mr, mode="RGB")
    mr_img = mr_img.resize((max(1, size // 2), max(1, size // 2)), Image.BOX)
    return base, mr_img


def _family_seed(family: str) -> int:
    """A stable seed per family, so a texture is byte-identical run to run.

    Hashing the name rather than counting entries means adding a family in the
    middle of the table does not redraw the ones after it.
    """
    return int(hashlib.sha1(family.encode()).hexdigest()[:8], 16)


def _jitter(rgba: list[float], seed: int) -> list[float]:
    """Move a colour a little, deterministically. See SEED_VALUE / SEED_HUE.

    Works in HSV because that is where "slightly different brick" lives: a
    per-channel RGB wobble desaturates as often as it shifts hue, and the result
    reads as dirt on the lens rather than as variation in the material.
    """
    r, g, b = (min(max(c, 0.0), 1.0) for c in rgba[:3])
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    rng = _rng(seed)
    h = (h + (rng.random() - 0.5) * 2 * SEED_HUE) % 1.0
    v = min(1.0, max(0.0, v * (1 + (rng.random() - 0.5) * 2 * SEED_VALUE)))
    s = min(1.0, max(0.0, s * (1 + (rng.random() - 0.5) * 2 * SEED_SAT)))
    return [*colorsys.hsv_to_rgb(h, s, v), *rgba[3:]]


def _tint(target: list[float], own: list[float]) -> list[float]:
    """The baseColorFactor that lands a map whose mean is `own` on `target`.

    A ratio rather than a replacement. The map already carries the family's
    colour, so setting the factor to `target` would apply it twice and come out
    nearly black; setting it to plain white would throw away both the seed
    variation and any colour the caller asked for.
    """
    factor = [t / max(o, 1e-3) for t, o in zip(target[:3], own[:3])]
    # glTF permits a factor above 1 and trimesh clamps it to 255, which would
    # quietly discard every variation that came out *brighter*. Rescaling keeps
    # the hue shift and spends the overflow on value rather than losing it.
    peak = max(factor)
    return [f / max(peak, 1.0) for f in factor] + [target[3]]


def has_texture(family: str) -> bool:
    return family in TEXTURE


def tile_studs(family: str) -> float | None:
    """How many studs one texture tile covers, or None if the family has no map.

    This is the number `primitives.build` passes to `_unwrap`, and it is a
    property of the *material* rather than of the part: a brick is the same size
    on a gatehouse as on a garden wall, so scaling the tile to the part would be
    wrong in exactly the way that gives you giant bricks on a big building.
    """
    entry = TEXTURE.get(family)
    return entry[2] if entry else None


def texture_maps(family: str) -> tuple[Image.Image, Image.Image] | None:
    """The (baseColor, metallicRoughness) PIL images for a family, or None.

    Public because the docs sheet and the tests want them without building a
    mesh, and because an exporter writing Roblox `SurfaceAppearance` sidecars
    needs the files rather than the material.
    """
    return _maps(family)


def roblox_material(family: str) -> str | None:
    """The Roblox `Material` enum name for a family, if one is honest.

    Returns None rather than guessing. See docs/ROBLOX-EXPORT.md: base colour
    lands on `MeshPart.TextureID` and the fuller set on a `SurfaceAppearance`,
    but the enum is what drives physics, footstep sound and material variants,
    so a wrong one is worse than none.
    """
    return ROBLOX.get(family)


def apply_to_mesh(
    mesh, part_name: str, explicit: str | None = None, color: str | None = None,
    seed: int | None = None, texture: bool | None = None,
) -> str:
    """Attach a PBR material to one mesh. Returns the family chosen.

    `color` overrides only the base colour, keeping the family's metallic and
    roughness — a red car body should still behave like paint. With a texture in
    play it becomes a *tint*: the factor is set to the ratio between the colour
    asked for and the family's own, so red brick is still brick with courses and
    mortar rather than a flat red slab.

    `texture` is tri-state. True attaches the family's maps and expects the
    caller to supply UVs — that is `primitives.build`, which unwraps right
    afterwards. False is the flat material this module used to be.

    None is "auto", and it is the case that matters for `assemble`, which
    re-materials meshes loaded back off disk. Auto only attaches a map when the
    mesh already has UVs, because a base-colour texture with no UVs samples one
    corner of the image and paints the whole part that colour — worse than the
    flat factor it replaced. Auto also leaves a base-colour texture the mesh
    already carries alone: a generated part arrives with a photographic albedo
    backprojected onto its own unwrap and that is better than anything drawn
    here.

    `seed` jitters the colour within the family. It defaults to a hash of the
    part name, so the twenty walls in a facade differ without anyone asking and
    two parts that really are the same part stay identical.
    """
    family, spec = resolve(part_name, explicit)
    spec = dict(spec)

    # What the mesh brought with it. `mesh.visual` is replaced wholesale below,
    # so anything worth keeping has to be read out first.
    old = getattr(mesh, "visual", None)
    had_uv = getattr(old, "uv", None)
    if had_uv is not None and len(had_uv) != len(mesh.vertices):
        had_uv = None
    had_map = getattr(getattr(old, "material", None), "baseColorTexture", None)

    # Keeping the mesh's own albedo beats drawing over it — but not if the
    # caller named a colour, which is them overriding the picture on purpose.
    keep = (texture is not False and had_map is not None and had_uv is not None
            and not color)
    if texture is None:
        want = family in TEXTURE and had_uv is not None and not keep
    else:
        want = texture
    if want and family not in TEXTURE:
        raise ValueError(f"material {family!r} has no texture map; it is a flat colour")

    own = spec["baseColorFactor"]
    # What colour this part should end up. An explicit `color` is never
    # jittered: the caller said exactly what they wanted, and moving it would
    # make the parameter non-deterministic.
    target = parse_color(color) if color else _jitter(
        own, _family_seed(part_name) if seed is None else seed
    )

    if want:
        spec["baseColorTexture"], spec["metallicRoughnessTexture"] = _maps(family)
        spec["baseColorFactor"] = _tint(target, own)
    elif keep:
        # The mesh's own map already *is* this part's colour, so take only the
        # family's metallic and roughness and leave the picture alone — white
        # factor, and no jitter to shift a photograph off its own white balance.
        spec["baseColorTexture"] = had_map
        spec["baseColorFactor"] = [1.0, 1.0, 1.0, own[3]]
    else:
        spec["baseColorFactor"] = target

    mesh.visual = trimesh.visual.TextureVisuals(
        # The unwrap that produced these is expensive and cannot be recovered
        # from the replaced visual, so carry it across. Without this, assembling
        # a scene silently threw away every scripted part's UVs and every part
        # came out one flat colour again.
        uv=had_uv,
        # doubleSided, always. glTF defaults it to false, and generated shells
        # are not watertight — a single-sided material lets the viewer cull
        # backfaces and you see straight through the model, which reads as a
        # shattered mesh rather than a material setting.
        material=PBRMaterial(name=f"kitbash_{family}", doubleSided=True, **spec)
    )
    return family


def families() -> list[str]:
    return sorted(PALETTE)


def textured_families() -> list[str]:
    return sorted(TEXTURE)


def packs() -> dict[str, list[str]]:
    """The themed sets, so an agent can pick a coherent palette in one call."""
    return {name: list(members) for name, members in PACKS.items()}


def pack(name: str) -> list[str]:
    if name not in PACKS:
        raise ValueError(f"unknown pack {name!r}, expected one of {sorted(PACKS)}")
    return list(PACKS[name])
