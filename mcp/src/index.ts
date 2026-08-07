#!/usr/bin/env node
/**
 * Kitbash MCP server.
 *
 * Gives a coding agent the ability to generate 3D assets without leaving the
 * editor: no browser, no Blender, no manual export step. Runs on the machine
 * the agent runs on and needs no Python — all GPU work happens in the server
 * process it talks to over HTTP.
 */
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

import * as api from "./client.js";
import { SERVER_URL } from "./client.js";

const server = new McpServer({ name: "kitbash", version: "0.1.0" });

/** MCP wants text content; this keeps the JSON shaping in one place. */
function json(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }] };
}

function pluralSeconds(ms: number) {
  const s = Math.round(ms / 1000);
  return `generating (${s}s elapsed)`;
}

function failure(message: string) {
  return {
    content: [{ type: "text" as const, text: message }],
    isError: true,
  };
}

/**
 * A tool result the model can actually look at.
 *
 * MCP image content is base64 in the result itself — no file path, no URL — so
 * the picture arrives in the same turn as the tool call and the model sees it
 * without a second round trip. This is the entire point of the preview tools:
 * an agent that assembles a scene and never looks at it is the open loop every
 * defect in docs/MULTI-PART.md came through.
 */
function image(png: Uint8Array, caption: string) {
  return {
    content: [
      {
        type: "image" as const,
        data: Buffer.from(png).toString("base64"),
        mimeType: "image/png",
      },
      { type: "text" as const, text: caption },
    ],
  };
}

const PREVIEW_VIEWS = [
  "side",
  "front",
  "top",
  "three_qtr",
  "rear_qtr",
  "low",
] as const;

const previewInputs = {
  views: z
    .array(z.enum(PREVIEW_VIEWS))
    .optional()
    .describe(
      "Which angles to put on the sheet, in order. Defaults to all six. " +
        "Keep at least one elevation and the top view: a wing detached along " +
        "X is invisible from the side and a fin floating in Y is invisible " +
        "from the top.",
    ),
  size: z
    .number()
    .int()
    .optional()
    .describe("Sheet width in pixels, 256-2400. Default 1200."),
  columns: z.number().int().optional().describe("Tiles per row. Default 3."),
  highlight: z
    .string()
    .optional()
    .describe(
      "Paint one named part magenta, leaving every other part exactly as it " +
        "renders without this. Use it when the sheet shows something wrong " +
        "and you need to know which part it is.",
    ),
  isolate: z
    .boolean()
    .optional()
    .describe(
      "With highlight, hide every other part. The camera does not move, so " +
        "the isolated part sits at the same pixel it occupied in the full " +
        "render — flip between the two to see where it actually is.",
    ),
};

/** What to tell the model to do with the picture it was just handed. */
const LOOK_AT_IT =
  "Look at the image before you say the build worked. Check, in this order: " +
  "(1) does every part touch the ground or the part it is meant to join, or " +
  "is something hovering — a shadow sitting away from the part that casts it " +
  "means it is floating; (2) are left/right pairs symmetric; (3) does " +
  "anything overshoot or intersect. If something is wrong, fix the placement " +
  "and render again.";

server.registerTool(
  "check_gpu_server",
  {
    title: "Check the Kitbash GPU server",
    description:
      "Reports whether the GPU server is reachable and how much VRAM is free. " +
      "Worth calling first when a generation fails, to tell 'server is down' " +
      "apart from 'generation went wrong'.",
    inputSchema: {},
  },
  async () => {
    try {
      const h = await api.health();
      return json({ server_url: SERVER_URL, ...h });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "generate_part",
  {
    title: "Generate a 3D part from a reference image",
    description:
      "Turns a single reference image into a 3D mesh and writes it to disk as " +
      "a .glb. Takes roughly 40 seconds on the reference GPU once the model is " +
      "loaded, plus about 70 seconds more on the very first call while weights " +
      "load.\n\n" +
      "Generates geometry only — the mesh has no textures or materials.\n\n" +
      "For a multi-part object, call this once per part with the same seed and " +
      "a distinct part_name, so parts can be regenerated individually later " +
      "without rerolling the whole object.",
    inputSchema: {
      image_path: z
        .string()
        .optional()
        .describe("Path to a PNG or JPEG reference image. Either this or image_b64."),
      image_b64: z
        .string()
        .optional()
        .describe("Base64-encoded image, if you do not have a file on disk."),
      output_path: z
        .string()
        .describe("Where to write the .glb, e.g. ./assets/fuselage.glb"),
      part_name: z
        .string()
        .optional()
        .describe("Label for this part in a multi-part build, e.g. 'fuselage'."),
      seed: z
        .number()
        .int()
        .optional()
        .describe("Fixed seed. Use the same seed across parts for consistency."),
      target_faces: z
        .number()
        .int()
        .optional()
        .describe(
          "Decimate the mesh to roughly this many faces before saving. Raw " +
            "output is ~350k faces, which no game engine will accept, so set " +
            "this for anything headed into Roblox or Unity.\n" +
            "20000 is the recommended default: 18x smaller with no visible " +
            "loss. Drop to 8000 for props with no fine surface detail. Stay " +
            "at 40000+ only when the part carries engraved or embossed detail " +
            "that must read up close, which is the first thing decimation " +
            "destroys. The dense original is kept server-side either way.",
        ),
      octree_resolution: z
        .number()
        .int()
        .optional()
        .describe(
          "Mesh density. 256 is the default and peaks near 7.6 GiB of VRAM; " +
            "lower it to 128 for simple parts or a smaller card.",
        ),
      num_inference_steps: z.number().int().optional().describe("Default 30."),
      generator: z
        .enum(["hunyuan3d", "trellis2"])
        .optional()
        .describe(
          "trellis2 gives markedly better hard-surface geometry at a third of " +
            "the VRAM; hunyuan3d is more tolerant of unprepared input. " +
            "Call check_gpu_server for what is available.",
        ),
      texture: z
        .boolean()
        .optional()
        .describe(
          "Paint the mesh by projecting the reference image back onto it. On " +
            "by default wherever the generator supplies no colour of its own. " +
            "Real colour from the real photo, ~6s, no VRAM.",
        ),
      guidance_scale: z
        .number()
        .optional()
        .describe(
          "How closely to follow the reference image. Default 5.0. Raise it " +
            "when the mesh drifts from the image, lower it when the result " +
            "looks over-constrained or noisy.",
        ),
      timeout_seconds: z
        .number()
        .int()
        .optional()
        .describe(
          "How long to wait before returning the job id instead of the mesh. " +
            "Default 300. The job keeps running past the timeout either way.",
        ),
    },
  },
  async (args, extra) => {
    try {
      let imageB64 = args.image_b64;
      if (!imageB64) {
        if (!args.image_path) {
          return failure("Provide either image_path or image_b64.");
        }
        const bytes = await readFile(resolve(args.image_path)).catch(() => null);
        if (!bytes) return failure(`Could not read image: ${args.image_path}`);
        imageB64 = bytes.toString("base64");
      }

      const job = await api.submitJob({
        image_b64: imageB64,
        part_name: args.part_name,
        seed: args.seed,
        target_faces: args.target_faces,
        octree_resolution: args.octree_resolution,
        num_inference_steps: args.num_inference_steps,
        guidance_scale: args.guidance_scale,
        generator: args.generator,
        texture: args.texture,
      });

      // A cold generation runs ~110s, well past the 60s default request
      // timeout in most MCP clients. Progress notifications reset that timer,
      // so without them a first call reliably fails in Claude Code.
      const progressToken = extra?._meta?.progressToken;
      const finished = await api.waitForJob(
        job.id,
        (args.timeout_seconds ?? 300) * 1000,
        async (j, elapsedMs) => {
          if (progressToken === undefined) return;
          await extra.sendNotification({
            method: "notifications/progress",
            params: {
              progressToken,
              progress: Math.round(elapsedMs / 1000),
              message:
                j.status === "queued"
                  ? "queued on the GPU server"
                  : pluralSeconds(elapsedMs),
            },
          });
        },
      );

      if (finished.status === "error") {
        return failure(`Generation failed: ${finished.error}`);
      }
      if (finished.status !== "done") {
        return json({
          job_id: finished.id,
          status: finished.status,
          note:
            "Still running past the timeout. The job continues on the server; " +
            "poll it with get_generation_job and then call save_mesh.",
        });
      }

      const glb = await api.downloadMesh(finished.id);
      const out = resolve(args.output_path);
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, glb);

      const r = finished.result!;
      return json({
        job_id: finished.id,
        part_name: args.part_name ?? null,
        output_path: out,
        faces: r.faces,
        vertices: r.vertices,
        decimated_from: r.decimated_from,
        watertight: r.watertight,
        generation_seconds: r.generation_seconds,
        seed: r.params.seed ?? null,
        note:
          r.faces > 50_000
            ? `${r.faces} faces is very dense for a game engine — pass target_faces to decimate.`
            : undefined,
      });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "get_generation_job",
  {
    title: "Check a generation job",
    description: "Status of a job previously started by generate_part.",
    inputSchema: { job_id: z.string() },
  },
  async ({ job_id }) => {
    try {
      return json(await api.getJob(job_id));
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "save_mesh",
  {
    title: "Save a finished mesh to disk",
    description:
      "Downloads the .glb for a completed job. Use this when generate_part " +
      "timed out and the job finished afterwards.",
    inputSchema: {
      job_id: z.string(),
      output_path: z.string().describe("Where to write the .glb"),
    },
  },
  async ({ job_id, output_path }) => {
    try {
      const job = await api.getJob(job_id);
      if (job.status !== "done") {
        return failure(`Job ${job_id} is ${job.status}, not done.`);
      }
      const glb = await api.downloadMesh(job_id);
      const out = resolve(output_path);
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, glb);
      return json({ job_id, output_path: out, bytes: glb.byteLength });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "decomposition_examples",
  {
    title: "Worked decomposition plans",
    description:
      "Returns example plans in the format decompose_object expects. Read one " +
      "before writing your own — the format carries several non-obvious rules " +
      "about prompt wording that decide whether it works.",
    inputSchema: {},
  },
  async () => {
    try {
      return json(await api.decomposeExamples());
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "decompose_object",
  {
    title: "Build every part of a multi-part object from a plan",
    description:
      "Takes a decomposition plan and produces one mesh per part, returning " +
      "job ids and a ready-to-use parts list for assemble_parts.\n\n" +
      "You author the plan — you know what you are building and the server " +
      "does not. Call decomposition_examples first for the format.\n\n" +
      "Each part is generate (its own image prompt -> 3D), script (a " +
      "parametric primitive) or mirror (another part reflected, free). A " +
      "12-part aircraft costs 6 generations.\n\n" +
      "Rules that decide whether this works, learned the hard way:\n" +
      "- Give each part its OWN prompt. Cropping a photo of the whole object " +
      "  returns whole objects — these models complete partial views.\n" +
      "- The shared style suffix does nearly all the work of keeping parts " +
      "  coherent, but must NOT name the whole object, or the completion prior " +
      "  returns and a propeller comes back attached to an aeroplane.\n" +
      "- Describe geometry, not the object. 'a bare fuselage with no wings' " +
      "  gives an aeroplane; 'a hollow elongated shell with six oval " +
      "  portholes' gives a shell.\n" +
      "- Script anything dimensioned and hard-surface. Generation is worst at " +
      "  exactly that, and a scripted strut is 192 triangles against 8,000.\n\n" +
      "This runs for many minutes. Generation cost is per generated part.",
    inputSchema: {
      plan: z
        .record(z.unknown())
        .describe("The plan object. See decomposition_examples."),
    },
  },
  async ({ plan }) => {
    try {
      const r = await api.decompose(plan);
      return json({
        subject: r.subject,
        part_count: r.parts.length,
        job_ids: r.job_ids,
        failed: r.failed,
        warnings: r.warnings,
        elapsed_seconds: r.elapsed_seconds,
        note:
          "Parts are queued, not finished. Poll get_generation_job until each " +
          "is done, then pass assemble_request to assemble_parts.",
        assemble_request: r.assemble_request,
      });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "describe_part",
  {
    title: "Measure a generated part",
    description:
      "Bounding box, size and center of a finished part, in the mesh's own " +
      "units. Call this before assemble_parts so placement is computed from " +
      "real dimensions rather than guessed.",
    inputSchema: { job_id: z.string() },
  },
  async ({ job_id }) => {
    try {
      return json(await api.describePart(job_id));
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "assemble_parts",
  {
    title: "Assemble parts into one scene",
    description:
      "Composes separately-generated parts into a single .glb with one named " +
      "node per part.\n\n" +
      "This is the point of generating parts separately. A single generation " +
      "produces one welded blob you cannot edit; an assembled scene keeps " +
      "every part addressable, so a part can be regenerated on its own later " +
      "without rerolling the whole object.\n\n" +
      "Positions are in the parts' own units — call describe_part first to " +
      "get real dimensions rather than guessing. Rotations are XYZ euler " +
      "degrees, applied after scale and before translation.\n\n" +
      "Coordinates are glTF convention: +Y is UP, +X right, +Z toward the " +
      "viewer. Stack parts along Y. Roblox is Y-up too, so placement carries " +
      "over unchanged. Blender is Z-up and converts on import, mapping " +
      "(x, y, z) to (x, -z, y) — expected, not a bug.\n\n" +
      "Call preview_scene on the result before reporting success. The part " +
      "list this returns describes a debris field exactly as convincingly as " +
      "it describes an aeroplane.",
    inputSchema: {
      parts: z
        .array(
          z.object({
            job_id: z.string().describe("A completed generate_part job"),
            name: z.string().describe("Node name, e.g. 'fuselage'"),
            position: z.array(z.number()).length(3).optional(),
            rotation: z
              .array(z.number())
              .length(3)
              .optional()
              .describe("XYZ euler degrees"),
            scale: z.union([z.number(), z.array(z.number()).length(3)]).optional(),
            material: z
              .string()
              .optional()
              .describe(
                "Override the material inferred from the part name. One of: " +
                  "metal, dark_metal, glass, rubber, wood, stone, fabric, " +
                  "leather, paint, plastic, gold, emissive.",
              ),
            color: z
              .string()
              .optional()
              .describe(
                'Base colour as "#rrggbb". Materials are neutral by default ' +
                  "because the generator does not know what colour the object " +
                  "is — set this when you do. Keeps the family's metallic and " +
                  "roughness, so a red car body still behaves like paint.",
              ),
            use_raw: z
              .boolean()
              .optional()
              .describe(
                "Use the dense pre-decimation mesh for this part, when one " +
                  "part needs detail the rest of the scene does not.",
              ),
          }),
        )
        .min(1),
      output_path: z.string().describe("Where to write the assembled .glb"),
      scene_name: z.string().optional(),
      apply_materials: z
        .boolean()
        .optional()
        .describe(
          "Assign each part a PBR material inferred from its name — 'canopy' " +
            "becomes glass, 'wheel' rubber, 'engine' metal. On by default, " +
            "because generations are otherwise uniform grey. Name parts for " +
            "what they are and this comes out looking deliberate.",
        ),
    },
  },
  async ({ parts, output_path, scene_name, apply_materials }) => {
    try {
      const scene = await api.assembleScene({ parts, scene_name, apply_materials });
      const glb = await api.downloadScene(scene.scene_id);
      const out = resolve(output_path);
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, glb);
      return json({
        scene_id: scene.scene_id,
        output_path: out,
        part_count: scene.part_count,
        total_faces: scene.total_faces,
        parts: scene.parts.map((p) => ({
          name: p.name,
          faces: p.faces,
          material: p.material,
        })),
        size: scene.size,
        bytes: glb.byteLength,
        next: `Call preview_scene with scene_id ${scene.scene_id} and look at ` +
          `the image before reporting this as finished.`,
      });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "preview_scene",
  {
    title: "Look at an assembled scene",
    description:
      "Renders an assembled scene to a shaded contact sheet — side, front, " +
      "top and three-quarter views on one image — and returns it as an image " +
      "you can look at.\n\n" +
      "CALL THIS AFTER EVERY assemble_parts, BEFORE reporting success. You " +
      "authored the placement; the server resolved it to coordinates; this is " +
      "the only way to find out whether the result is the object you meant. " +
      "The part list assemble_parts returns will look perfectly reasonable " +
      "for a scene that is a debris field.\n\n" +
      "The model sits on a ground plane and casts a shadow, because a part " +
      "floating in empty space looks fine and a part floating above a floor " +
      "does not. All views share one fixed camera, so a part that drifted out " +
      "of place cannot hide behind a re-framed shot.\n\n" +
      "Costs about a second and no VRAM — it is pure CPU, so it works while " +
      "the GPU is busy generating. Cheap enough to call in a fix-and-recheck " +
      "loop, which is how it is meant to be used.\n\n" +
      "Also returns each part's gap above the floor in the scene's own units, " +
      "so 'it looks like it is floating' can be checked against a number.",
    inputSchema: { scene_id: z.string().describe("From assemble_parts"), ...previewInputs },
  },
  async ({ scene_id, views, size, columns, highlight, isolate }) => {
    try {
      const png = await api.previewScene(scene_id, {
        views: views ? [...views] : undefined,
        size,
        columns,
        highlight,
        isolate,
      });
      const ground = await api.sceneGround(scene_id).catch(() => null);
      const floating = ground
        ? ground.parts.filter((p) => p.gap_fraction > 0.01)
        : [];
      const caption = [
        `scene ${scene_id}, ${ground?.parts.length ?? "?"} parts, ` +
          `floor y=${ground?.floor_y ?? "?"}`,
        floating.length
          ? `Clear of the floor (gap, and as a fraction of the scene's size): ` +
            floating
              .map((p) => `${p.name} ${p.gap} (${(p.gap_fraction * 100).toFixed(1)}%)`)
              .join(", ") +
            `. A gap is only a defect if the part is not held up by another ` +
            `part — a wing on a fuselage should clear the floor; a wheel ` +
            `should not.`
          : "Every part reaches the floor.",
        LOOK_AT_IT,
      ].join("\n\n");
      return image(png, caption);
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "preview_part",
  {
    title: "Look at a single generated part",
    description:
      "Renders one finished part — from generate_part, a scripted primitive, " +
      "or any completed job — to a shaded contact sheet and returns it as an " +
      "image you can look at.\n\n" +
      "Worth calling before assembling: generation returns whatever the model " +
      "made of the prompt, and 'a hollow elongated shell' comes back as a " +
      "whole aeroplane often enough that it is worth thirty seconds to check. " +
      "The part is drawn on a ground plane at its own scale, so its " +
      "proportions read directly.\n\n" +
      "Pure CPU, about a second, no VRAM.",
    inputSchema: { job_id: z.string().describe("A completed job"), ...previewInputs },
  },
  async ({ job_id, views, size, columns, highlight, isolate }) => {
    try {
      const png = await api.previewJob(job_id, {
        views: views ? [...views] : undefined,
        size,
        columns,
        highlight,
        isolate,
      });
      return image(
        png,
        `job ${job_id}. Check the shape is the part you asked for and not the ` +
          `whole object, that it is the right way up, and that it is one solid ` +
          `piece rather than fragments.`,
      );
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "export_for_roblox",
  {
    title: "Export a part or scene for Roblox",
    description:
      "Writes a .glb (and an .obj fallback) that satisfies Roblox Studio's " +
      "import constraints, and saves it locally.\n\n" +
      "Roblox Studio imports .glb natively, so this is not a format " +
      "conversion — it enforces the constraints:\n" +
      "- 20,000 triangles PER MESH. The cap is per MeshPart, not per file, so " +
      "  an assembled 10-part model has a 200k budget while one welded 100k " +
      "  blob is rejected. Over-budget parts are decimated individually.\n" +
      "- 1 file unit = 1 stud. Generated meshes normalise to ~2 units, so " +
      "  without height_studs everything arrives knee-high.\n" +
      "- Pivot on the ground plane, so the model does not spawn half-buried.\n\n" +
      "Give exactly one of job_id or scene_id.",
    inputSchema: {
      job_id: z.string().optional().describe("Export one generated part"),
      scene_id: z.string().optional().describe("Export an assembled scene"),
      output_path: z.string().describe("Where to write the .glb locally"),
      height_studs: z
        .number()
        .optional()
        .describe("Desired height in studs. A human character is about 5."),
      target: z
        .enum(["roblox", "dcc"])
        .optional()
        .describe(
          "'roblox' applies the constraints above. 'dcc' converts container " +
            "format only and leaves units and origin alone, for Blender or " +
            "similar. Default 'roblox'.",
        ),
    },
  },
  async ({ job_id, scene_id, output_path, height_studs, target }) => {
    try {
      if (!job_id === !scene_id) {
        return failure("Give exactly one of job_id or scene_id.");
      }
      const result = await api.exportMesh({
        job_id,
        scene_id,
        target: target ?? "roblox",
        height_studs,
      });
      const glb = await api.downloadExported(result.primary);
      const out = resolve(output_path);
      await mkdir(dirname(out), { recursive: true });
      await writeFile(out, glb);
      return json({
        output_path: out,
        target: result.target,
        parts: result.parts,
        total_faces: result.total_faces,
        source_faces: result.source_faces,
        size_studs: result.size,
        pivot: result.pivot,
        bytes: glb.byteLength,
        warnings: result.warnings,
      });
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

server.registerTool(
  "list_generation_jobs",
  {
    title: "List recent generation jobs",
    description: "Recent jobs, newest first.",
    inputSchema: {
      limit: z.number().int().optional().describe("Default 20."),
    },
  },
  async ({ limit }) => {
    try {
      const { jobs } = await api.listJobs(limit ?? 20);
      return json(
        jobs.map((j) => ({
          id: j.id,
          status: j.status,
          part_name: j.params.part_name ?? null,
          faces: j.result?.faces ?? null,
          seconds: j.result?.generation_seconds ?? null,
          error: j.error,
        })),
      );
    } catch (err) {
      return failure(err instanceof Error ? err.message : String(err));
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
// stdout is the MCP transport — anything logged there corrupts the protocol.
console.error(`kitbash-mcp ready (server: ${SERVER_URL})`);
