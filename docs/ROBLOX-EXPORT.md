# Exporting for Roblox

Roblox Studio is the primary consumer of everything this project generates, so
the export path is written to Roblox's rules and everything else is treated as
the general case.

## Roblox accepts `.glb`

The one thing worth checking before building a converter: **Studio's 3D Importer
takes `.fbx`, `.obj`, `.gltf` and `.glb`.** glTF support left beta and shipped in
full, and the announcement is explicit that both flavours count:

> A glTF file can come in one of two flavors: a JSON format (.gltf) with binary
> data buffers and optional external files, or a pure binary format (.glb)
> packed into a smaller byte array. The latter is especially powerful as it
> allows you to pass around entire scenes in a single file (no more zipping and
> unzipping multiple files or dealing with incorrect texture paths)!

— [3D Importer — glTF file support (full release)](https://devforum.roblox.com/t/3d-importer-gltf-file-support-full-release/2584034)

The docs page prose says "`.fbx`, `.gltf`, and `.obj`" and never spells out
`.glb`, which is where the doubt comes from. `.glb` is the binary container for
exactly the same format and the importer reads it.
([Importer](https://create.roblox.com/docs/studio/importer))

So the pipeline's native output is already the right format, and `.glb` is
arguably the *best* of the four here: textures embed in the single file, and
unlike `.obj` it carries hierarchy, so a multi-part model stays multi-part.

`.obj` is still written alongside as a fallback — it is the format every
importer has always taken — but it flattens hierarchy and drops vertex colours.

No `.fbx`. There is no permissively-licensed Python FBX writer (Autodesk's SDK
is proprietary, `bpy` is GPL), and it would buy nothing that glTF does not
already do for this use case. See [DECIMATION.md](DECIMATION.md) for why the
stack stays MIT.

## The limits that actually bite

| Constraint | Value | Source |
| --- | --- | --- |
| Triangles **per mesh** | 20,000 | [Modeling specifications](https://create.roblox.com/docs/art/modeling/specifications) |
| Bone influences per vertex | 4 | same |
| Animation tracks per file | 1 | same |
| Texture resolution | up to 4096×4096; UV space up to 1024×1024 | [Texture specifications](https://create.roblox.com/docs/art/modeling/texture-specifications) |
| Materials per mesh object | 1 | same |
| Texture upload formats | `.png`, `.jpg`, `.tga`, `.bmp` | same |

The triangle cap is **per mesh, not per file**. Each mesh node in the glTF
becomes one `MeshPart` inside a `Model`, and each is measured on its own — so a
ten-part model has a 200k budget while a single welded 100k blob is rejected.
That is a second, independent argument for the multi-part assembly this project
does anyway (see `server/assemble.py`).

20,000 also happens to be the decimation sweet spot measured in
[DECIMATION.md](DECIMATION.md), which is convenient: the default target is
already exactly Roblox's ceiling.

## Materials and textures

- glTF PBR materials come through. A base colour map lands on
  `MeshPart.TextureID`; the fuller PBR set (albedo, normal, roughness,
  metalness, emissive mask) lands in a `SurfaceAppearance` child.
  ([Meshes](https://create.roblox.com/docs/parts/meshes))
- Normal maps must be **OpenGL-convention tangent space**. Roughness, metalness
  and emissive masks are single-channel 8-bit greyscale; albedo and normal are
  24-bit RGB.
- Vertex colours are supported by the importer for `.fbx`/`.gltf` — but they do
  not survive a trip through `.obj`, so the fallback file loses them.
- Shape-only generations have no material at all. They import as untextured grey
  geometry, which is correct, not an error.

## Scale, axis, origin

**Units.** The importer's `Scale Unit` defaults to **Studs**, meaning one unit in
the file is read as one stud. Generated meshes are normalised to roughly 2 units
across, so straight out of the generator an asset arrives about knee-high on a
character. `export_for(..., height_studs=N)` rescales; a Roblox character is
about 5 studs tall, and the community conversion is 1 stud ≈ 0.28 m if you are
matching real-world dimensions.

**Up axis.** glTF is Y-up by specification and Roblox is Y-up, so the importer
defaults (`World Up = Top`, `World Forward = Front`) are already right. No axis
flip, no conversion. Worth stating because almost every other engine pairing
needs one.

**Origin.** Studio places a `MeshPart` at its mesh origin, so a model centred on
its own middle spawns half-buried. The Roblox target therefore centres the
footprint in X/Z and puts the lowest point at Y=0, so the model sits on the
floor where you drop it. For a multi-part scene this is one transform applied to
the whole scene, so relative part placement is untouched.

## Also worth knowing

- Meshes should be watertight, without exposed holes or backfaces, and must have
  some thickness — zero-thickness geometry is rejected. Decimation breaks
  watertightness (see DECIMATION.md); engines do not care, and Studio imports
  decimated meshes fine, but it is the reason the dense original is kept.
- Studio has no importer support for Draco or meshopt compression, so the
  exporter emits plain uncompressed glTF buffers.
- Uncheck **Import Only As Model** in the importer if you want each mesh in the
  file as an individually addressable asset rather than one bundled model.

## Using it

```python
from pathlib import Path
from export import export_for

result = export_for(Path("mesh.glb"), "roblox", Path("out"), height_studs=5)
result["primary"]   # out/mesh.glb  — drag this into Studio
result["warnings"]  # anything Studio will complain about or silently change
```

Targets are `"roblox"` and `"dcc"`. Both write `.glb` plus `.obj` (with `.mtl`
and any texture PNGs listed under `files["obj_sidecars"]`). `"dcc"` skips the
triangle budget and the re-origin, because a DCC tool has its own opinions about
units and pivots and should win.
