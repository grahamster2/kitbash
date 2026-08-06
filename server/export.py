"""Turn a generated mesh into something the target program will actually accept.

Roblox Studio's 3D Importer takes `.fbx`, `.obj`, `.gltf` **and `.glb`** — the
binary glTF we already produce imports natively, so the interesting work is not
format conversion at all, it is Roblox's constraints:

- **20,000 triangles per mesh.** Each mesh node in the file becomes one
  `MeshPart`, and each is checked against that budget separately. A multi-part
  scene of five 20k parts is fine; one 100k blob is rejected outright.
- **1 file unit = 1 stud** at the importer's default `Scale Unit`. Generated
  meshes are normalised to roughly 2 units, i.e. knee-high, so the caller
  usually wants to say how tall the thing should be.
- **Y-up.** glTF is Y-up by spec and so is Roblox, so the importer's defaults
  (World Up = Top, World Forward = Front) are already correct. Nothing to do —
  worth stating, because every other engine pairing needs an axis flip.
- **Pivot at the origin.** Studio drops a `MeshPart` at its mesh origin, so a
  model whose origin sits in its middle spawns half-buried.

`.obj` is written alongside as the universal fallback: it is the one format
every importer on earth has always taken, at the cost of hierarchy, vertex
colours and PBR. No `.fbx` — FBX has no permissively-licensed Python writer
(the SDK is proprietary, `bpy` is GPL) and buys nothing over glTF here. See
docs/DECIMATION.md for why the stack stays MIT.
"""
import logging
from pathlib import Path

import numpy as np
import trimesh

log = logging.getLogger("kitbash.export")

# Per-mesh, enforced by Studio at import time.
# https://create.roblox.com/docs/art/modeling/specifications
ROBLOX_MAX_TRIANGLES = 20_000

# Studio downsamples anything larger.
# https://create.roblox.com/docs/art/modeling/texture-specifications
ROBLOX_MAX_TEXTURE_PX = 4096

TARGETS = ("roblox", "dcc")


def export_for(
    mesh_path: Path,
    target: str,
    out_dir: Path,
    height_studs: float | None = None,
) -> dict:
    """Write `mesh_path` out for `target`, returning the files and what changed.

    target="roblox" enforces the triangle budget and puts the pivot on the
    ground plane; the returned `size` is then in studs, since the importer reads
    one file unit as one stud. target="dcc" changes container format only and
    leaves the geometry where the generator put it, because a DCC tool has its
    own opinions about units and origin and should win.

    height_studs rescales the whole thing to that Y extent. Generated meshes are
    normalised to roughly 2 units, so without it everything arrives knee-high.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown target {target!r}, expected one of {TARGETS}")
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    scene = _as_scene(trimesh.load(str(mesh_path)))
    if not scene.geometry:
        raise ValueError(f"no geometry in {mesh_path}")

    warnings: list[str] = []
    source_faces = sum(len(g.faces) for g in scene.geometry.values())

    if target == "roblox":
        _fit_triangle_budget(scene, warnings)
        _check_textures(scene, warnings)

    if height_studs:
        scale = float(height_studs) / float(scene.extents[1])
        scene.apply_transform(trimesh.transformations.scale_matrix(scale))

    if target == "roblox":
        # Centre the footprint and sit the lowest point on Y=0, so dropping the
        # model into a place puts it on the floor rather than through it.
        lo, hi = scene.bounds
        offset = [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]
        scene.apply_transform(trimesh.transformations.translation_matrix(offset))

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = mesh_path.stem
    files = {}

    glb_path = out_dir / f"{stem}.glb"
    glb_path.write_bytes(scene.export(file_type="glb"))
    files["glb"] = str(glb_path)

    # .obj drags a .mtl and one .png per material along with it, under names
    # trimesh chooses. Snapshot the directory so the caller gets the whole set
    # and not just the file that is useless on its own.
    before = set(out_dir.iterdir())
    obj_path = out_dir / f"{stem}.obj"
    scene.export(str(obj_path), include_texture=True, mtl_name=f"{stem}.mtl")
    files["obj"] = str(obj_path)
    sidecars = sorted(p for p in set(out_dir.iterdir()) - before if p != obj_path)
    if sidecars:
        files["obj_sidecars"] = [str(p) for p in sidecars]

    parts = [
        {"name": name, "faces": int(len(geom.faces))}
        for name, geom in scene.geometry.items()
    ]
    total_faces = sum(p["faces"] for p in parts)
    if total_faces != source_faces:
        log.info("export: %d -> %d faces for %s", source_faces, total_faces, target)

    return {
        "target": target,
        "primary": files["glb"],
        "files": files,
        "parts": parts,
        "part_count": len(parts),
        "total_faces": total_faces,
        "source_faces": source_faces,
        "size": [round(float(v), 4) for v in scene.extents],
        "pivot": "base-centered" if target == "roblox" else "source",
        "file_bytes": {
            k: Path(v).stat().st_size for k, v in files.items() if isinstance(v, str)
        },
        "warnings": warnings,
    }


def _as_scene(loaded) -> trimesh.Scene:
    """A .glb loads as a Scene, a .obj or .stl as a bare Trimesh."""
    if isinstance(loaded, trimesh.Scene):
        return loaded
    return trimesh.Scene(loaded)


def _fit_triangle_budget(scene: trimesh.Scene, warnings: list[str]) -> None:
    """Decimate any geometry over Roblox's per-mesh cap.

    Done per geometry rather than over the whole scene: the cap applies to each
    MeshPart, so a 10-part model has a 200k budget and spending it evenly would
    throw away detail nobody asked to lose.
    """
    for name, geom in list(scene.geometry.items()):
        faces = len(geom.faces)
        if faces <= ROBLOX_MAX_TRIANGLES:
            continue
        log.info("part %s over budget: %d faces, decimating", name, faces)
        scene.geometry[name] = geom.simplify_quadric_decimation(
            face_count=ROBLOX_MAX_TRIANGLES
        )
        warnings.append(
            f"{name}: {faces} faces exceeded Roblox's {ROBLOX_MAX_TRIANGLES} "
            "per-mesh limit, decimated to fit"
        )


def _check_textures(scene: trimesh.Scene, warnings: list[str]) -> None:
    """Flag oversized maps and untextured-but-vertex-coloured meshes.

    Neither is fatal — Studio downsamples large maps, and it does read glTF
    vertex colours — but vertex colours are lost the moment anyone round-trips
    through the .obj, which is exactly what someone hitting an import problem
    will try next.
    """
    for name, geom in scene.geometry.items():
        visual = getattr(geom, "visual", None)
        material = getattr(visual, "material", None)
        image = getattr(material, "image", None) or getattr(
            material, "baseColorTexture", None
        )
        if image is not None:
            if max(image.size) > ROBLOX_MAX_TEXTURE_PX:
                warnings.append(
                    f"{name}: texture is {image.size[0]}x{image.size[1]}, "
                    f"Studio downsamples above {ROBLOX_MAX_TEXTURE_PX}px"
                )
            continue
        if getattr(visual, "kind", None) in ("vertex", "face"):
            warnings.append(f"{name}: colour is per-vertex only, not carried by .obj")
        elif np.asarray(getattr(geom, "faces", ())).size:
            warnings.append(f"{name}: no texture or vertex colour, imports untextured")
