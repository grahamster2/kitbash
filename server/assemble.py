"""Compose separately-generated parts into one scene.

This is the point of the project. A single generation gives you one welded blob
— import it into Blender and you get `objects=1, materials=0`, which is exactly
what you cannot work with. Generating parts separately and assembling them here
gives a glTF with one named node per part, so the engine, the artist, and a
later regeneration can all address parts individually.

Pure CPU work, and pure MIT (trimesh + numpy). No GPU is involved.
"""
import logging
import math
from pathlib import Path

import numpy as np
import trimesh

import materials

log = logging.getLogger("kitbash.assemble")


def _transform(position, rotation_deg, scale) -> np.ndarray:
    """Compose scale -> rotate (XYZ euler, degrees) -> translate."""
    T = np.eye(4)

    if scale is not None:
        s = [float(scale)] * 3 if isinstance(scale, (int, float)) else [float(v) for v in scale]
        T[:3, :3] = np.diag(s)

    if rotation_deg is not None:
        rx, ry, rz = (math.radians(float(a)) for a in rotation_deg)
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
        T[:3, :3] = R @ T[:3, :3]

    if position is not None:
        T[:3, 3] = [float(v) for v in position]

    return T


def describe(mesh_path: Path) -> dict:
    """Bounds and size of a part, so a caller can place it without guessing."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    lo, hi = mesh.bounds
    return {
        "faces": int(len(mesh.faces)),
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "size": [round(float(v), 4) for v in (hi - lo)],
        "center": [round(float(v), 4) for v in mesh.bounding_box.centroid],
    }


def assemble(parts: list[dict], out_path: Path, apply_materials: bool = True) -> dict:
    """Build one glTF from many part meshes.

    Each part: {name, mesh_path, position?, rotation?, scale?, material?}. Names
    become glTF node names, which is what makes the parts addressable
    downstream — and, when apply_materials is on, what picks each part's
    material. See materials.py for why that is worth doing.
    """
    if not parts:
        raise ValueError("no parts to assemble")

    scene = trimesh.Scene()
    placed = []
    used_names: set[str] = set()

    for i, part in enumerate(parts):
        mesh_path = Path(part["mesh_path"])
        if not mesh_path.exists():
            raise FileNotFoundError(f"part mesh missing: {mesh_path}")

        mesh = trimesh.load(str(mesh_path), force="mesh")

        # glTF node names must be unique or the parts stop being addressable,
        # which defeats the entire purpose of assembling them separately.
        name = str(part.get("name") or f"part_{i}")
        base, n = name, 2
        while name in used_names:
            name = f"{base}_{n}"
            n += 1
        used_names.add(name)

        material = None
        if apply_materials:
            material = materials.apply_to_mesh(
                mesh, name, part.get("material"), part.get("color")
            )

        T = _transform(part.get("position"), part.get("rotation"), part.get("scale"))
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=T)

        placed.append({
            "name": name,
            "faces": int(len(mesh.faces)),
            "material": material,
            "source": str(mesh_path),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(out_path))

    lo, hi = scene.bounds
    total_faces = sum(p["faces"] for p in placed)
    log.info("assembled %d parts, %d faces -> %s", len(placed), total_faces, out_path)

    return {
        "scene_path": str(out_path),
        "part_count": len(placed),
        "total_faces": total_faces,
        "parts": placed,
        "bounds_min": [round(float(v), 4) for v in lo],
        "bounds_max": [round(float(v), 4) for v in hi],
        "size": [round(float(v), 4) for v in (hi - lo)],
        "file_bytes": out_path.stat().st_size,
    }
