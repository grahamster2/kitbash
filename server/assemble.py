"""Compose separately-generated parts into one scene.

This is the point of the project. A single generation gives you one welded blob
— import it into Blender and you get `objects=1, materials=0`, which is exactly
what you cannot work with. Generating parts separately and assembling them here
gives a glTF with one named node per part, so the engine, the artist, and a
later regeneration can all address parts individually.

Placement used to be absolute coordinates only, and that does not survive
contact with a real build. Fourteen parts placed by hand against a generated
airframe put the wheels at heights unrelated to their struts and the cabin
outside the fuselage, because the caller was inventing numbers for geometry it
had never measured. So parts may instead state *intent* — "the wheel goes at the
bottom of that strut", "the gear goes under the airframe, a fifth of the way
back from the nose", "the right one is the left one mirrored" — and the numbers
are derived here from the transformed bounds of the parts already placed.

Deriving from *transformed* bounds is the whole trick: a part scaled 0.05 has a
footprint twenty times smaller than the mesh on disk, and the file's bounds are
the one measurement guaranteed to be wrong.

Pure CPU work, and pure MIT (trimesh + numpy). No GPU is involved.
"""
import logging
import math
from pathlib import Path

import numpy as np
import trimesh

import materials
import orient as orienting

log = logging.getLogger("kitbash.assemble")

AXES = ("x", "y", "z")
_AXIS = {"x": 0, "y": 1, "z": 2}

# Named points along one axis of a box, as a fraction of its extent: 0 is the
# low face, 1 the high face. A caller may pass a bare number instead, which is
# what makes "20% of its length from the nose" expressible at all.
FRACTIONS: dict[str, float] = {
    "min": 0.0, "center": 0.5, "centre": 0.5, "middle": 0.5, "max": 1.0,
    # Axis-flavoured aliases. They are the words people actually reach for, and
    # they cost nothing because every axis measures the same way.
    "bottom": 0.0, "top": 1.0, "left": 0.0, "right": 1.0,
    "start": 0.0, "end": 1.0,
}

# Face-to-face attachment: one keyword instead of two fractions, because "put
# the wheel under the strut" needs a point on the target *and* the matching
# point on the part being placed, and stating only one of them is the mistake
# that leaves parts floating. Value is (where on the target, where on me).
ATTACH: dict[str, tuple[float, float]] = {
    "under": (0.0, 1.0), "below": (0.0, 1.0), "beneath": (0.0, 1.0),
    "above": (1.0, 0.0), "over": (1.0, 0.0), "on": (1.0, 0.0),
    # Flush *inside* the target rather than outside it — a floor sitting on the
    # cabin's own lower surface, not hanging beneath the fuselage.
    "flush_min": (0.0, 0.0), "flush_max": (1.0, 1.0),
}

# A target that is not a part. The ground is the y=0 plane, which is where an
# exporter's pivot goes and where a vehicle's wheels belong.
GROUND = "ground"


def _transform(position, rotation_deg, scale, orient_matrix=None) -> np.ndarray:
    """Compose orient -> scale -> rotate (XYZ euler, degrees) -> translate.

    Orientation comes first so everything downstream is stated in the frame the
    caller was thinking in: a part is turned the right way round, and only then
    is it 4.4 m along x. A `rotation` alongside an `orient` is therefore a
    deliberate nudge on top of a canonical part — dihedral, an incidence angle —
    rather than a competing absolute.
    """
    T = np.eye(4)

    if orient_matrix is not None:
        T[:3, :3] = np.asarray(orient_matrix, dtype=np.float64)

    if scale is not None:
        s = [float(scale)] * 3 if isinstance(scale, (int, float)) else [float(v) for v in scale]
        T[:3, :3] = np.diag(s) @ T[:3, :3]

    if rotation_deg is not None:
        rx, ry, rz = (math.radians(float(a)) for a in rotation_deg)
        R = trimesh.transformations.euler_matrix(rx, ry, rz, "sxyz")[:3, :3]
        T[:3, :3] = R @ T[:3, :3]

    if position is not None:
        T[:3, 3] = [float(v) for v in position]

    return T


def world_bounds(mesh, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The axis-aligned bounds a part actually occupies once placed.

    From the real vertices, not from the file's AABB pushed through the matrix.
    Those agree for an axis-aligned transform and disagree for a rotated one,
    where the corner-box is larger than the part — and "larger than the part" in
    an anchor means a visible gap between two things that should be touching.
    """
    points = trimesh.transform_points(np.asarray(mesh.vertices), T)
    return points.min(axis=0), points.max(axis=0)


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


def _fraction(value, field: str, axis: str) -> float:
    """A named point or a number -> a fraction along one axis of a box."""
    # bool is an int in Python, and `True` as a placement is a typo, not 100%.
    if isinstance(value, bool):
        raise ValueError(f"{field}.{axis} is {value!r}; expected a name or a number")
    if isinstance(value, (int, float)):
        return float(value)
    key = str(value).strip().lower()
    if key in FRACTIONS:
        return FRACTIONS[key]
    if key in ATTACH and field == "my":
        raise ValueError(
            f"my.{axis} is {value!r}; attachment keywords like {key!r} belong in "
            f"`align`, which sets both sides of the join"
        )
    raise ValueError(
        f"{field}.{axis} is {value!r}; expected a number where 0 is the low face "
        f"and 1 the high face, or one of {sorted(set(FRACTIONS))}"
    )


def _align_pair(value, axis: str) -> tuple[float, float | None]:
    """One `align` entry -> (point on the target, point on me or None).

    None means "not stated here", and the caller's `my` — or the default,
    centred — fills it in.
    """
    if not isinstance(value, bool) and not isinstance(value, (int, float)):
        key = str(value).strip().lower()
        if key in ATTACH:
            return ATTACH[key]
    return _fraction(value, "align", axis), None


def _axis_keys(mapping, field: str) -> dict[str, object]:
    """Validate the axis names in an align/my dict up front.

    A typo'd axis silently placing nothing is the failure mode this prevents:
    `{"Y": "min"}` would otherwise mean "no constraint at all".
    """
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ValueError(f"anchor.{field} must be a mapping of axis -> placement")
    out = {}
    for key, value in mapping.items():
        axis = str(key).strip().lower()
        if axis not in _AXIS:
            raise ValueError(f"anchor.{field} has axis {key!r}; expected x, y or z")
        out[axis] = value
    return out


def _mirror_matrix(spec, default_axis: str = "x") -> np.ndarray:
    """Reflection about a world plane, as a 4x4."""
    about = 0.0
    if spec is None or spec is True:
        axis = default_axis
    elif isinstance(spec, dict):
        axis = str(spec.get("axis", default_axis)).strip().lower()
        about = float(spec.get("about", 0.0))
    else:
        axis = str(spec).strip().lower()
    if axis not in _AXIS:
        raise ValueError(f"mirror axis is {axis!r}; expected x, y or z")

    i = _AXIS[axis]
    M = np.eye(4)
    M[i, i] = -1.0
    M[i, 3] = 2.0 * about
    return M


def _anchor_translation(anchor: dict, own_lo, own_hi, target, position) -> np.ndarray:
    """Where a part's origin has to sit for its anchor to be satisfied.

    `own_lo`/`own_hi` are the part's bounds after scale and rotation but before
    translation; `target` is the (lo, hi) of the thing being anchored to, or
    None for the ground plane.
    """
    align = _axis_keys(anchor.get("align"), "align")
    mine = _axis_keys(anchor.get("my"), "my")

    offset = anchor.get("offset") or [0.0, 0.0, 0.0]
    if len(offset) != 3:
        raise ValueError(f"anchor.offset must be [x, y, z], got {offset!r}")
    offset = [float(v) for v in offset]

    if target is None:  # ground: a plane, so only its height means anything
        for axis in (*align, *mine):
            if axis != "y":
                raise ValueError(
                    f"anchor to {GROUND!r} constrains y only, but align/my "
                    f"names {axis!r} — the ground plane has no x or z extent"
                )
        # The overwhelmingly common intent, and the one nobody should have to
        # spell: the part rests on the plane rather than being centred in it.
        align = {"y": align.get("y", 0.0)}
        mine = {"y": mine.get("y", "min")}
        t_lo = t_hi = np.zeros(3)
        constrained = {"y"}  # x and z still come from `position`
    else:
        t_lo, t_hi = target
        # Every axis is constrained, and an axis nobody mentioned is centred on
        # the target. Leaving it at world zero instead is precisely how a part
        # ends up floating beside the thing it was meant to be attached to, and
        # it makes a bare {"to": "fuselage"} mean "centre this inside that" for
        # free. `offset` is how you move off centre.
        constrained = set(AXES)

    result = np.array([float(v) for v in (position or [0.0, 0.0, 0.0])])
    for axis in constrained:
        i = _AXIS[axis]
        target_frac, implied = (
            _align_pair(align[axis], axis) if axis in align else (0.5, None)
        )
        if axis in mine:
            my_frac = _fraction(mine[axis], "my", axis)
        elif implied is not None:
            my_frac = implied
        else:
            my_frac = 0.5  # centred on the point, unless told otherwise

        here = t_lo[i] + target_frac * (t_hi[i] - t_lo[i])
        there = own_lo[i] + my_frac * (own_hi[i] - own_lo[i])
        result[i] = here - there

    return result + np.array(offset)


def _reference(ref, by_name: dict[str, list[int]], names: list[str]) -> int:
    """Resolve a part name in an anchor or mirror_of to a part index."""
    key = str(ref)
    matches = by_name.get(key, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"{key!r} names {len(matches)} parts, so anchoring to it is ambiguous; "
            f"give the parts distinct names"
        )
    raise ValueError(f"unknown part {key!r}; known parts are {sorted(set(names))}")


def _resolution_order(deps: dict[int, set[int]], names: list[str]) -> list[int]:
    """Kahn's algorithm, so parts resolve after whatever they depend on.

    This is what makes the part list order-independent: a wheel may be listed
    before the strut it hangs from. Anything left over is in a cycle, which has
    to be an error — a self-referential placement has no answer, and hanging
    while it looks for one is the worst possible way to say so.
    """
    remaining = {i: set(d) for i, d in deps.items()}
    order: list[int] = []
    ready = [i for i in sorted(remaining) if not remaining[i]]

    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(remaining):
            if i in remaining[j]:
                remaining[j].discard(i)
                if not remaining[j]:
                    ready.append(j)
        remaining.pop(i, None)

    if remaining:
        raise ValueError(f"placement cycle: {_describe_cycle(remaining, names)}")
    return order


def _describe_cycle(remaining: dict[int, set[int]], names: list[str]) -> str:
    """Name the loop, because "there is a cycle" is not actionable on its own."""
    start = min(remaining)
    path = [start]
    seen = {start}
    node = start
    while True:
        node = min(remaining[node] & remaining.keys())
        if node in seen:
            path = path[path.index(node):] + [node]
            break
        seen.add(node)
        path.append(node)
    return " -> ".join(names[i] for i in path)


def _placement_deps(part: dict, index: int, by_name, names) -> set[int]:
    """Which already-placed parts this one's transform is measured against."""
    deps = set()
    anchor = part.get("anchor")
    if anchor is not None:
        if not isinstance(anchor, dict):
            raise ValueError(f"{names[index]}: anchor must be an object")
        to = anchor.get("to")
        if not to:
            raise ValueError(
                f"{names[index]}: anchor needs `to` — a part name or {GROUND!r}"
            )
        if str(to).strip().lower() != GROUND:
            j = _reference(to, by_name, names)
            if j == index:
                raise ValueError(f"{names[index]}: a part cannot anchor to itself")
            deps.add(j)
            # An anchor to a part fixes all three axes, so a world-space
            # position next to it is a contradiction — and the losing one wins
            # silently, which is the kind of bug you only find in a render.
            if part.get("position") is not None:
                raise ValueError(
                    f"{names[index]}: anchoring to {to!r} places all three axes, "
                    f"so `position` cannot apply too — use anchor.offset to move "
                    f"it off the anchor, or anchor.align to pick a different point"
                )

    mirror_of = part.get("mirror_of")
    if mirror_of:
        j = _reference(mirror_of, by_name, names)
        if j == index:
            raise ValueError(f"{names[index]}: a part cannot mirror itself")
        deps.add(j)
        # Copying a transform and then overriding half of it has no sane
        # meaning, and silently ignoring the other keys is how a left/right pair
        # ends up subtly asymmetric.
        conflicts = [k for k in ("position", "anchor") if part.get(k) is not None]
        if conflicts:
            raise ValueError(
                f"{names[index]}: mirror_of takes its whole transform from "
                f"{mirror_of!r}, so it cannot also set {', '.join(conflicts)}"
            )
    return deps


def _orientations(parts: list[dict], meshes: list, names: list[str]) -> list[dict | None]:
    """Resolve every part's `orient` before anything is placed.

    Before, and not during, because an anchor measures a *box*: anchoring a
    wheel to a strut that is about to be turned upright would measure the strut
    lying down. Orientation is a fact about the part itself, so it is settled
    first and everything else is derived from the part as it will appear.
    """
    resolved: list[dict | None] = [None] * len(parts)
    for i, part in enumerate(parts):
        spec = part.get("orient")
        if spec is None:
            continue
        if part.get("mirror_of"):
            raise ValueError(
                f"{names[i]}: mirror_of takes its whole transform from "
                f"{part['mirror_of']!r}, including that part's orientation, so it "
                f"cannot also set `orient`"
            )
        # The floor may ride inside the declaration or sit beside it, because
        # the API nests it and a hand-written part list reads better flat.
        floor = float(part.get("min_confidence") or 0.0)
        if isinstance(spec, dict) and "min_confidence" in spec:
            spec = dict(spec)
            floor = float(spec.pop("min_confidence") or 0.0)
        try:
            result = orienting.orient(meshes[i], spec)
        except ValueError as exc:
            raise ValueError(f"{names[i]}: {exc}") from exc

        applied = result.confidence >= floor
        if not applied:
            # Deliberately not an error. A caller who sets a floor is saying
            # "leave it alone rather than get it wrong", and the part still has
            # to be placed — with whatever rotation it already had.
            log.info("%s: orientation %.2f below floor %.2f, left as generated",
                     names[i], result.confidence, floor)
        resolved[i] = {"applied": applied, "result": result}
    return resolved


def _unique_names(parts: list[dict]) -> list[str]:
    """glTF node names must be unique or the parts stop being addressable,
    which defeats the entire purpose of assembling them separately."""
    names, used = [], set()
    for i, part in enumerate(parts):
        name = str(part.get("name") or f"part_{i}")
        base, n = name, 2
        while name in used:
            name = f"{base}_{n}"
            n += 1
        used.add(name)
        names.append(name)
    return names


def resolve_placements(parts: list[dict], meshes: list) -> list[dict]:
    """Turn every part's placement — absolute, anchored or mirrored — into a
    world transform and the world bounds it produces.

    Returned in input order; resolved in dependency order.
    """
    names = _unique_names(parts)

    # Anchors are written against the names the caller chose. Uniquification
    # only renames collisions, and a collided name is ambiguous anyway, so both
    # spellings resolve and only a genuine duplicate is rejected.
    by_name: dict[str, list[int]] = {}
    for i, (part, name) in enumerate(zip(parts, names)):
        by_name.setdefault(name, []).append(i)
        requested = str(part.get("name") or "")
        if requested and requested != name:
            by_name.setdefault(requested, []).append(i)

    deps = {
        i: _placement_deps(part, i, by_name, names) for i, part in enumerate(parts)
    }
    orientations = _orientations(parts, meshes, names)

    placed: list[dict | None] = [None] * len(parts)
    for i in _resolution_order(deps, names):
        part = parts[i]
        mirror_of = part.get("mirror_of")

        if mirror_of:
            source = placed[_reference(mirror_of, by_name, names)]
            T = _mirror_matrix(part.get("mirror")) @ source["transform"]
            anchored_to, mirrored_from = None, source["name"]
        else:
            mirrored_from = None
            # Orientation, scale and rotation first: an anchor is a statement
            # about the part as it will appear, so its bounds have to be
            # measured after them.
            turn = orientations[i]
            T = _transform(
                None, part.get("rotation"), part.get("scale"),
                turn["result"].matrix if turn and turn["applied"] else None,
            )
            anchor = part.get("anchor")
            if anchor is None:
                T[:3, 3] = [float(v) for v in (part.get("position") or [0, 0, 0])]
                anchored_to = None
            else:
                own_lo, own_hi = world_bounds(meshes[i], T)
                to = str(anchor["to"]).strip()
                if to.lower() == GROUND:
                    target, anchored_to = None, GROUND
                else:
                    j = _reference(to, by_name, names)
                    target = (placed[j]["bounds_min"], placed[j]["bounds_max"])
                    anchored_to = names[j]
                T[:3, 3] = _anchor_translation(
                    anchor, own_lo, own_hi, target, part.get("position")
                )
            if part.get("mirror") is not None:
                T = _mirror_matrix(part["mirror"]) @ T

        lo, hi = world_bounds(meshes[i], T)
        placed[i] = {
            "name": names[i],
            "transform": T,
            "bounds_min": lo,
            "bounds_max": hi,
            "anchored_to": anchored_to,
            "mirrored_from": mirrored_from,
            "orient": orientations[i],
        }

    return placed


def assemble(parts: list[dict], out_path: Path, apply_materials: bool = True) -> dict:
    """Build one glTF from many part meshes.

    Each part: {name, mesh_path, position?, rotation?, scale?, anchor?, mirror?,
    mirror_of?, orient?, material?}. Names become glTF node names, which is what makes
    the parts addressable downstream — and, when apply_materials is on, what
    picks each part's material. See materials.py for why that is worth doing.

    Every part reports the world-space bounds it ended up occupying, so a caller
    can tell that the wheel is at the bottom of the strut without downloading
    the scene and opening it.
    """
    if not parts:
        raise ValueError("no parts to assemble")

    meshes = []
    for part in parts:
        mesh_path = Path(part["mesh_path"])
        if not mesh_path.exists():
            raise FileNotFoundError(f"part mesh missing: {mesh_path}")
        meshes.append(trimesh.load(str(mesh_path), force="mesh"))

    placements = resolve_placements(parts, meshes)

    scene = trimesh.Scene()
    placed = []
    for part, mesh, p in zip(parts, meshes, placements):
        name = p["name"]
        material = None
        if apply_materials:
            material = materials.apply_to_mesh(
                mesh, name, part.get("material"), part.get("color")
            )

        # A reflection — or any negative scale — reverses face winding, and a
        # glTF viewer reads that as "this surface faces inward". Flipping the
        # faces back cancels it, so a mirrored part is lit like its original
        # instead of looking hollow.
        if np.linalg.det(p["transform"][:3, :3]) < 0:
            mesh.invert()

        scene.add_geometry(
            mesh, node_name=name, geom_name=name, transform=p["transform"]
        )

        lo, hi = p["bounds_min"], p["bounds_max"]
        turn = p["orient"]
        placed.append({
            "name": name,
            "faces": int(len(mesh.faces)),
            "material": material,
            "source": str(part["mesh_path"]),
            # How this part got where it is. Both None means it was placed at an
            # absolute position, which is worth being able to tell apart from an
            # anchor that quietly resolved to the origin.
            "anchored_to": p["anchored_to"],
            "mirrored_from": p["mirrored_from"],
            # What orienting decided, whether or not it was applied — a caller
            # that set a confidence floor needs to see the number it failed.
            "orient": (
                {"applied": turn["applied"], **turn["result"].as_dict()}
                if turn else None
            ),
            "position": [round(float(v), 4) for v in p["transform"][:3, 3]],
            "bounds_min": [round(float(v), 4) for v in lo],
            "bounds_max": [round(float(v), 4) for v in hi],
            "size": [round(float(v), 4) for v in (hi - lo)],
            "center": [round(float(v), 4) for v in (lo + hi) / 2],
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
