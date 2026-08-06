"""Give parts materials without generating textures.

Texture generation needs 12-16 GB and does not fit on the reference card, so
everything comes out of the generator as untextured grey. That reads as
unfinished even when the geometry is good.

But in a multi-part build we know something a texture model has to infer: the
caller already told us what each part *is*. An agent that names a part "canopy"
has said it is glass. Mapping that name to a PBR material costs no VRAM and
about a millisecond, and it gets a scene most of the way to looking deliberate.

This is not a replacement for generated textures. It is what you do when you
cannot afford them, and it is genuinely better than grey.
"""
import logging

import trimesh
from trimesh.visual.material import PBRMaterial

log = logging.getLogger("kitbash.materials")

# Deliberately small. These are the material families that read as distinct at a
# glance; more entries would add nuance nobody can see in a game engine.
PALETTE: dict[str, dict] = {
    "metal":   dict(baseColorFactor=[0.62, 0.65, 0.70, 1.0], metallicFactor=0.95, roughnessFactor=0.35),
    "dark_metal": dict(baseColorFactor=[0.28, 0.29, 0.32, 1.0], metallicFactor=0.90, roughnessFactor=0.45),
    "glass":   dict(baseColorFactor=[0.45, 0.68, 0.85, 0.45], metallicFactor=0.0, roughnessFactor=0.05),
    "rubber":  dict(baseColorFactor=[0.09, 0.09, 0.10, 1.0], metallicFactor=0.0, roughnessFactor=0.92),
    "wood":    dict(baseColorFactor=[0.52, 0.34, 0.18, 1.0], metallicFactor=0.0, roughnessFactor=0.75),
    "stone":   dict(baseColorFactor=[0.48, 0.47, 0.44, 1.0], metallicFactor=0.0, roughnessFactor=0.88),
    "fabric":  dict(baseColorFactor=[0.35, 0.33, 0.40, 1.0], metallicFactor=0.0, roughnessFactor=0.95),
    "leather": dict(baseColorFactor=[0.33, 0.20, 0.13, 1.0], metallicFactor=0.0, roughnessFactor=0.65),
    "paint":   dict(baseColorFactor=[0.75, 0.16, 0.14, 1.0], metallicFactor=0.10, roughnessFactor=0.45),
    "plastic": dict(baseColorFactor=[0.85, 0.85, 0.87, 1.0], metallicFactor=0.0, roughnessFactor=0.40),
    "gold":    dict(baseColorFactor=[0.85, 0.68, 0.25, 1.0], metallicFactor=1.0, roughnessFactor=0.30),
    "emissive": dict(baseColorFactor=[0.95, 0.85, 0.55, 1.0], metallicFactor=0.0, roughnessFactor=0.20,
                     emissiveFactor=[0.9, 0.75, 0.35]),
}

# Substring -> material family. The longest matching keyword wins, so
# "windshield" resolves before a bare "shield" would.
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
}

DEFAULT_MATERIAL = "paint"


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


def apply_to_mesh(mesh, part_name: str, explicit: str | None = None) -> str:
    """Attach a PBR material to one mesh. Returns the family chosen."""
    family, spec = resolve(part_name, explicit)
    mesh.visual = trimesh.visual.TextureVisuals(
        material=PBRMaterial(name=f"kitbash_{family}", **spec)
    )
    return family


def families() -> list[str]:
    return sorted(PALETTE)
