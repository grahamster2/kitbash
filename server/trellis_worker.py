"""TRELLIS 2 generation, run inside D:\\trellis2's interpreter.

This file is deployed to the TRELLIS install and never imported by the Kitbash
server -- it is the far side of the subprocess boundary in server/trellis.py,
and it is the only code here that may import torch 2.8 or the node pack.

Protocol: one JSON request on stdin, one result line on stdout prefixed with
RESULT_PREFIX. The prefix exists because ComfyUI's node loader writes hundreds
of lines to stdout, so "the result is the last thing printed" is not true.

The nodes drive fine from plain Python via NODE_CLASS_MAPPINGS -- no ComfyUI
server and no workflow JSON, which is what makes a plain subprocess enough.
"""
import base64
import io
import json
import math
import os
import sys
import threading
import time
import traceback

RESULT_PREFIX = "<<<KITBASH_RESULT>>>"

NODES = [
    "Trellis2LoadModel_GGUF",
    "Trellis2PreProcessImage_GGUF",
    "Trellis2MeshWithVoxelGenerator_GGUF",
    "Trellis2PostProcessAndUnWrapAndRasterizer_GGUF",
    "Trellis2SimplifyMeshAdvanced_GGUF",
]


class VramSampler(threading.Thread):
    """Device-wide peak, polled at 20 Hz.

    torch.max_memory_allocated only sees this process's tensors; the numbers in
    docs/QUALITY-COMPARISON.md that decide whether a run fits are total-free,
    which includes the CUDA context and whatever Windows is holding.
    """

    def __init__(self, torch):
        super().__init__(daemon=True)
        self.torch = torch
        self.stop = False
        self.peak_used = 0

    def run(self):
        while not self.stop:
            try:
                free, total = self.torch.cuda.mem_get_info()
                self.peak_used = max(self.peak_used, total - free)
            except Exception:
                pass
            time.sleep(0.05)


def _load_nodes(comfy_dir: str):
    sys.path.insert(0, comfy_dir)
    os.chdir(comfy_dir)
    import nodes as comfy

    # init_extra_nodes became a coroutine partway through ComfyUI's history and
    # gained init_api_nodes; both spellings are live in the wild.
    try:
        pending = comfy.init_extra_nodes(init_api_nodes=False)
    except TypeError:
        pending = comfy.init_extra_nodes()
    if hasattr(pending, "__await__"):
        import asyncio

        asyncio.run(pending)

    missing = [name for name in NODES if name not in comfy.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError(
            f"node pack did not register {missing}; found "
            f"{[k for k in comfy.NODE_CLASS_MAPPINGS if 'Trellis' in k]}"
        )
    return comfy.NODE_CLASS_MAPPINGS


def _face_count(obj):
    """Faces of the pre-simplification mesh, for reporting decimated_from.

    The generator node's return type is undocumented and has changed shape
    between forks, so this probes rather than assumes and gives up quietly --
    the number is metadata, not something a job should fail over.
    """
    for holder in (obj, getattr(obj, "mesh", None)):
        faces = getattr(holder, "faces", None)
        if faces is not None:
            try:
                return int(len(faces))
            except TypeError:
                pass
    return None


def generate(req: dict) -> dict:
    import numpy as np
    import torch
    from PIL import Image

    nodes = _load_nodes(req["comfy_dir"])

    sampler = VramSampler(torch)
    sampler.start()
    torch.cuda.reset_peak_memory_stats()

    stages = {}
    clock = [time.time()]

    def mark(name):
        torch.cuda.synchronize()
        stages[name] = round(time.time() - clock[0], 1)
        clock[0] = time.time()

    started = time.time()
    image = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGBA")
    tensor = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]

    # keep_models_loaded=False: this process exits after one job, so holding the
    # DiTs would only delay handing the VRAM back to Hunyuan3D. Warm reload is
    # ~0.2s. low_vram loads the three 1.3B DiTs serially, which is why peak is
    # set by the bake stage rather than by the quantization level.
    pipe, = nodes["Trellis2LoadModel_GGUF"]().process(
        modelname="TRELLIS.2-4B",
        model_format=req["quant"],
        backend="sdpa",  # flash-attn is not installed and is not needed
        device="cuda",
        low_vram=True,
        keep_models_loaded=False,
    )
    mark("load_model")

    prepared, = nodes["Trellis2PreProcessImage_GGUF"]().process(
        image=tensor, padding=0, remove_background=req["remove_background"]
    )
    mark("preprocess")

    textured = bool(req["textured"])
    mesh, bvh = nodes["Trellis2MeshWithVoxelGenerator_GGUF"]().process(
        pipeline=pipe,
        image=prepared,
        seed=req["seed"],
        pipeline_type=req["pipeline_type"],
        sparse_structure_steps=req["steps"],
        shape_steps=req["steps"],
        texture_steps=req["steps"],
        max_num_tokens=49152,
        sparse_structure_resolution=32,
        max_views=4,
        generate_texture_slat=textured,
        use_tiled_decoder=True,
        sampler="euler",
    )
    mark("generate")
    raw_faces = _face_count(mesh)

    tri = (_bake(nodes, mesh, bvh, req) if textured
           else _simplify_only(nodes, mesh, req))
    mark("postprocess_uv_bake" if textured else "simplify")

    mesh_path = req["mesh_path"]
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    tri.export(mesh_path, file_type="glb")

    sampler.stop = True
    time.sleep(0.1)

    faces = int(len(tri.faces))
    visual = tri.visual
    material = getattr(visual, "material", None)
    base_color = getattr(material, "baseColorTexture", None)
    uv = getattr(visual, "uv", None)

    return {
        "vertices": int(len(tri.vertices)),
        "faces": faces,
        # None rather than the raw count when nothing was removed, matching what
        # pipeline.py reports for a mesh that came in under budget.
        "decimated_from": raw_faces if raw_faces and raw_faces > faces else None,
        "watertight": bool(tri.is_watertight),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "device_peak_vram_gib": round(sampler.peak_used / 2**30, 2),
        "worker_seconds": round(time.time() - started, 1),
        "stages": stages,
        "has_uv": bool(uv is not None and len(uv) > 0),
        # Reported so the server never has to claim a texture exists on the
        # strength of having asked for one.
        "base_color_texture": f"{base_color.size[0]}x{base_color.size[1]}"
        if base_color is not None
        else None,
    }


def _bake(nodes, mesh, bvh, req):
    """Remesh, unwrap, simplify and bake both atlases. The textured path."""
    tri, _basecolor, _mr = nodes["Trellis2PostProcessAndUnWrapAndRasterizer_GGUF"]().process(
        mesh=mesh,
        mesh_cluster_threshold_cone_half_angle_rad=60.0,
        mesh_cluster_refine_iterations=0,
        mesh_cluster_global_iterations=1,
        mesh_cluster_smooth_strength=1,
        texture_size=req["texture_size"],
        remesh=True,
        remesh_band=1.0,
        remesh_project=0.0,
        target_face_num=req["target_faces"],
        simplify_method="Cumesh",
        fill_holes=True,
        texture_alpha_mode="OPAQUE",
        dual_contouring_resolution="512",
        double_side_material=False,
        remove_floaters=True,
        bake_on_vertices=False,
        use_custom_normals=False,
        uv_unwrap_method="Xatlas",
        bvh=bvh,
        remove_inner_faces=True,
    )
    return tri


def _simplify_only(nodes, mesh, req):
    """Decimate to the face budget and stop. The untextured path.

    It cannot go through the postprocess node: that node reads mesh.coords for
    its AABB, and coords only exists when the texture SLAT was generated, so
    with generate_texture_slat=False it dies on `NoneType has no attribute
    device` before it does any geometry work. Cumesh is the same simplifier the
    node would have used, so the face budget is honoured the same way; what is
    lost is the remesh, the hole fill and the UV unwrap, all of which are only
    there to serve a bake.
    """
    import trimesh

    simplified, = nodes["Trellis2SimplifyMeshAdvanced_GGUF"]().process(
        mesh=mesh, target_face_num=req["target_faces"],
        method="Cumesh", verbose=False,
    )
    tri = trimesh.Trimesh(
        vertices=simplified.vertices.cpu().numpy(),
        faces=simplified.faces.cpu().numpy(),
    )

    # The postprocess node the textured path uses also converts TRELLIS's Z-up
    # output to glTF's Y-up. Skipping that node skipped the conversion too, so
    # every untextured part shipped lying on its side — measured as a 91.5 deg
    # roll against the textured path's 0. Assembly cannot recover from this:
    # a sideways wing has a perfectly valid bounding box.
    tri.apply_transform(
        trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
    )
    return tri


def main():
    try:
        result = generate(json.loads(sys.stdin.read()))
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc()}
    sys.stdout.write("\n" + RESULT_PREFIX + json.dumps(result) + "\n")
    sys.stdout.flush()
    sys.exit(1 if "error" in result else 0)


if __name__ == "__main__":
    main()
