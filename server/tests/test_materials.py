"""The material library: the tables, the generated maps, and how they attach.

The texture recipes are numpy and their *output* is a judgement call — whether
brick reads as brick is settled by looking at docs/images/materials-library.png,
not by an assertion. What is testable, and what these cover, is everything that
would silently ruin that output: a map whose mean has drifted off the family's
stated colour, a texture that does not tile, a UV set thrown away on the way
into a scene, a Roblox enum that does not exist.
"""
import numpy as np
import pytest
import trimesh
from PIL import Image

import materials
import primitives


# --- the tables --------------------------------------------------------------

def test_the_original_twelve_families_are_all_still_here():
    """Everything downstream knows these names; a texture library is no reason
    to renumber them."""
    original = {"metal", "dark_metal", "glass", "rubber", "wood", "stone",
                "fabric", "leather", "paint", "plastic", "gold", "emissive"}

    assert original <= set(materials.PALETTE)


def test_the_library_is_big_enough_to_build_something_out_of():
    assert len(materials.PALETTE) >= 50
    assert len(materials.TEXTURE) >= 45


@pytest.mark.parametrize("family", sorted(materials.PALETTE))
def test_every_family_is_a_complete_pbr_spec(family):
    spec = materials.PALETTE[family]

    assert len(spec["baseColorFactor"]) == 4
    assert all(0.0 <= c <= 1.0 for c in spec["baseColorFactor"])
    assert 0.0 <= spec["metallicFactor"] <= 1.0
    assert 0.0 <= spec["roughnessFactor"] <= 1.0


def test_every_textured_family_exists_in_the_palette():
    assert set(materials.TEXTURE) <= set(materials.PALETTE)


def test_every_keyword_points_at_a_real_family():
    unknown = {kw: fam for kw, fam in materials.KEYWORDS.items()
               if fam not in materials.PALETTE}

    assert unknown == {}


def test_every_pack_is_made_of_real_families():
    for name, members in materials.PACKS.items():
        assert set(members) <= set(materials.PALETTE), name
        assert len(members) >= 8, f"{name} is too thin to build from"


def test_every_family_is_reachable_through_a_pack():
    """A family no pack contains is a family nobody finds. Packs are how an
    agent asks for a coherent palette in one move, so they have to be the whole
    library between them, not a sampler of it."""
    covered = {f for members in materials.PACKS.values() for f in members}

    assert sorted(set(materials.PALETTE) - covered) == []


def test_no_pack_lists_a_family_twice():
    for name, members in materials.PACKS.items():
        assert len(members) == len(set(members)), name


def test_pack_hands_back_a_copy():
    """PACKS is module state; a caller that sorts what it got back must not be
    editing the library."""
    got = materials.pack("medieval")
    got.append("unobtanium")

    assert "unobtanium" not in materials.PACKS["medieval"]


def test_pack_rejects_a_name_that_does_not_exist():
    with pytest.raises(ValueError, match="unknown pack"):
        materials.pack("steampunk")


# Every member of Roblox's `Material` enum that a MeshPart can be set to. A
# typo here is invisible until Studio throws at runtime, which is exactly the
# kind of thing a table wants a test for.
ROBLOX_ENUM = {
    "Asphalt", "Basalt", "Brick", "Cardboard", "Carpet", "CeramicTiles",
    "ClayRoofTiles", "Cobblestone", "Concrete", "CorrodedMetal", "CrackedLava",
    "DiamondPlate", "Fabric", "Foil", "ForceField", "Glacier", "Glass",
    "Granite", "Grass", "Ground", "Ice", "LeafyGrass", "Limestone", "Marble",
    "Metal", "Mud", "Neon", "Pavement", "Pebble", "Plaster", "Plastic", "Rock",
    "RoofShingles", "Rubber", "Salt", "Sand", "Sandstone", "Slate",
    "SmoothPlastic", "Snow", "Water", "Wood", "WoodPlanks",
}


def test_every_roblox_mapping_names_a_real_material_enum():
    bad = {fam: name for fam, name in materials.ROBLOX.items()
           if name not in ROBLOX_ENUM}

    assert bad == {}


def test_roblox_mappings_only_cover_families_that_exist():
    assert set(materials.ROBLOX) <= set(materials.PALETTE)


def test_a_roblox_dev_asking_for_granite_gets_a_texture_and_the_enum():
    assert materials.roblox_material("granite") == "Granite"
    assert materials.has_texture("granite")


def test_a_family_with_no_honest_roblox_counterpart_returns_none():
    assert materials.roblox_material("verdigris") == "CorrodedMetal"
    assert materials.roblox_material("nonsense") is None


# --- the generated maps ------------------------------------------------------

@pytest.mark.parametrize("family", sorted(materials.TEXTURE))
def test_every_recipe_produces_a_pair_of_usable_maps(family):
    base, mr = materials.texture_maps(family)

    assert isinstance(base, Image.Image) and base.mode == "RGB"
    assert isinstance(mr, Image.Image) and mr.mode == "RGB"
    assert 256 <= base.size[0] <= 512 and base.size[0] == base.size[1]
    # Roughness is halved deliberately; see _maps.
    assert mr.size[0] == base.size[0] // 2


@pytest.mark.parametrize("family", sorted(materials.TEXTURE))
def test_a_map_is_the_colour_its_family_says_it_is(family):
    """The textured and untextured paths must agree, or turning texturing on
    recolours a scene and PALETTE stops describing anything."""
    base, _ = materials.texture_maps(family)
    srgb = np.asarray(base, dtype=float) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

    mean = linear.reshape(-1, 3).mean(axis=0)
    want = materials.PALETTE[family]["baseColorFactor"][:3]
    assert mean == pytest.approx(want, abs=0.03)


@pytest.mark.parametrize("family", sorted(materials.TEXTURE))
def test_a_map_actually_has_something_on_it(family):
    """A recipe that collapsed to a flat fill is worse than no texture at all —
    it costs a PNG and a UV split and buys nothing."""
    base, _ = materials.texture_maps(family)
    pixels = np.asarray(base, dtype=float)

    assert pixels.std() > 3.0, f"{family} is nearly a flat colour"


@pytest.mark.parametrize("family", sorted(materials.TEXTURE))
def test_a_map_tiles_without_a_seam(family):
    """These are box-projected across parts that butt together, so a seam at the
    tile edge shows up on every wall in a facade rather than once.

    The comparison is against the *largest* interior steps, not the mean. A
    woven or tiled surface is mostly flat with a hard edge every few pixels, so
    its mean step is tiny and any real edge landing on the wrap looks like a
    seam against it. What matters is that the wrap is an ordinary edge.
    """
    pixels = np.asarray(materials.texture_maps(family)[0], dtype=float)

    for axis in (0, 1):
        steps = np.abs(np.diff(pixels, axis=axis)).mean(
            axis=tuple(i for i in range(3) if i != axis))
        wrap = np.abs(np.take(pixels, 0, axis) - np.take(pixels, -1, axis)).mean()
        assert wrap <= np.percentile(steps, 99) * 1.5 + 2.0, \
            f"{family} seams on axis {axis}"


def test_roughness_varies_where_the_material_says_it_should():
    """A scratched metal that is uniformly rough is a matte slab. The point of
    shipping a roughness map at all is that it moves."""
    _, mr = materials.texture_maps("rusted_iron")
    rough = np.asarray(mr, dtype=float)[:, :, 1]

    assert rough.std() > 12.0


def test_maps_are_built_once_and_reused():
    """A scene with twenty brick parts must not draw brick twenty times."""
    first = materials.texture_maps("brick")
    second = materials.texture_maps("brick")

    assert first[0] is second[0]


def test_a_flat_family_has_no_map_and_no_tile_size():
    assert not materials.has_texture("glass")
    assert materials.texture_maps("glass") is None
    assert materials.tile_studs("glass") is None
    assert "glass" not in materials.textured_families()


def test_the_discovery_lists_agree_with_the_tables():
    assert materials.families() == sorted(materials.PALETTE)
    assert materials.textured_families() == sorted(materials.TEXTURE)


def test_a_tile_is_sized_in_studs_not_in_parts():
    """A brick is the same size on a gatehouse as on a garden wall."""
    for family in materials.TEXTURE:
        assert 0.5 <= materials.tile_studs(family) <= 8.0, family


# --- attaching it to a mesh --------------------------------------------------

def _box():
    return trimesh.creation.box(extents=(2.0, 2.0, 2.0))


def _factor(mesh) -> list[int]:
    """baseColorFactor as a plain list. trimesh normalises it to a uint8 array,
    and comparing two of those with `==` gives an array, not an answer."""
    return np.asarray(mesh.visual.material.baseColorFactor).tolist()


def test_a_textured_part_carries_both_maps():
    mesh = _box()
    materials.apply_to_mesh(mesh, "wall", "brick", texture=True)
    material = mesh.visual.material

    assert material.name == "kitbash_brick"
    assert material.baseColorTexture is not None
    assert material.metallicRoughnessTexture is not None


def test_a_textured_part_keeps_a_near_white_factor_so_the_map_is_not_doubled():
    """The map already carries the family colour. A factor of the family colour
    on top of it would apply brick-red twice and come out nearly black."""
    mesh = _box()
    materials.apply_to_mesh(mesh, "wall", "brick", texture=True)

    # uint8, because trimesh normalises whatever it is handed. Not exactly 255:
    # the seed variation lives in this factor too, as a small per-channel tint.
    assert np.mean(_factor(mesh)[:3]) >= 225


def test_a_colour_on_a_textured_part_tints_it_rather_than_flattening_it():
    """Red brick should still have courses and mortar."""
    mesh = _box()
    materials.apply_to_mesh(mesh, "wall", "brick", color="#803020", texture=True)

    assert mesh.visual.material.baseColorTexture is not None
    assert mesh.visual.material.baseColorFactor[0] > mesh.visual.material.baseColorFactor[2]


def test_asking_a_flat_family_for_a_texture_says_so():
    with pytest.raises(ValueError, match="no texture map"):
        materials.apply_to_mesh(_box(), "canopy", "glass", texture=True)


def test_texture_false_gives_back_the_flat_material_it_always_did():
    mesh = _box()
    materials.apply_to_mesh(mesh, "crate", "wood", texture=False)

    assert mesh.visual.material.baseColorTexture is None


def test_auto_does_not_texture_a_mesh_with_no_uvs():
    """A base-colour map without UVs samples one corner and paints the whole
    part that colour, which is worse than the flat factor it replaced."""
    mesh = _box()
    materials.apply_to_mesh(mesh, "wall", "brick")

    assert mesh.visual.material.baseColorTexture is None


def test_auto_textures_a_mesh_that_already_has_uvs():
    mesh = _box()
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)))
    materials.apply_to_mesh(mesh, "wall", "brick")

    assert mesh.visual.material.baseColorTexture is not None


def test_applying_a_material_does_not_throw_away_the_meshs_uvs():
    """apply_to_mesh replaces mesh.visual outright, and assemble calls it on
    every part it loads — so without this a scene lost every scripted part's
    unwrap and came out flat again."""
    mesh = primitives.build("crate", texture=True)
    before = np.array(mesh.visual.uv)

    materials.apply_to_mesh(mesh, "crate", "wood")

    assert mesh.visual.uv == pytest.approx(before)


def test_an_explicit_colour_overrides_the_meshs_own_albedo():
    """Keeping the mesh's picture is the right default, not a rule — a caller
    naming a colour is naming a colour."""
    mesh = _box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (200, 30, 30))),
    )
    materials.apply_to_mesh(mesh, "hull", color="#2040c0")

    assert mesh.visual.material.baseColorTexture is None
    assert _factor(mesh)[2] > _factor(mesh)[0]


def test_a_kept_albedo_is_not_tinted_by_the_seed_jitter():
    """Jittering the factor over a photograph shifts its white balance."""
    mesh = _box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (200, 30, 30))),
    )
    materials.apply_to_mesh(mesh, "hull")

    assert _factor(mesh)[:3] == [255, 255, 255]


def test_a_meshs_own_albedo_beats_the_generated_one():
    """A generated part arrives with a photograph backprojected onto it. That is
    better than anything this module draws, so auto leaves it alone."""
    mesh = _box()
    own = Image.new("RGB", (8, 8), (200, 30, 30))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=own),
    )
    materials.apply_to_mesh(mesh, "wall", "brick")

    assert mesh.visual.material.baseColorTexture is own


# --- variation ---------------------------------------------------------------

def test_two_differently_named_parts_of_one_family_are_not_clones():
    """Twenty archways in a facade should not be twenty copies of one archway."""
    left, right = _box(), _box()
    materials.apply_to_mesh(left, "wall_left", "limestone", texture=False)
    materials.apply_to_mesh(right, "wall_right", "limestone", texture=False)

    assert _factor(left) != _factor(right)


def test_the_same_part_name_always_gets_the_same_colour():
    a, b = _box(), _box()
    materials.apply_to_mesh(a, "wall_left", "limestone", texture=False)
    materials.apply_to_mesh(b, "wall_left", "limestone", texture=False)

    assert _factor(a) == _factor(b)


def test_variation_stays_inside_the_family():
    """The point is weathering, not a different material. Anything bigger and a
    facade reads as twenty materials rather than one wall."""
    base = np.array(materials.PALETTE["brick"]["baseColorFactor"][:3])
    for name in (f"wall_{i}" for i in range(40)):
        mesh = _box()
        materials.apply_to_mesh(mesh, name, "brick", texture=False)
        got = np.asarray(_factor(mesh)[:3], dtype=float) / 255.0
        assert np.abs(got - base).max() < 0.12, name


def test_an_explicit_colour_is_never_jittered():
    """`color` is the caller saying exactly what they want; moving it would make
    the parameter non-deterministic."""
    mesh = _box()
    materials.apply_to_mesh(mesh, "body", color="#808080", texture=False)

    want = [round(c * 255) for c in materials.parse_color("#808080")[:3]]
    assert _factor(mesh)[:3] == pytest.approx(want, abs=1)


def test_seed_overrides_the_name_hash():
    a, b = _box(), _box()
    materials.apply_to_mesh(a, "wall_left", "brick", seed=7, texture=False)
    materials.apply_to_mesh(b, "wall_right", "brick", seed=7, texture=False)

    assert _factor(a) == _factor(b)
