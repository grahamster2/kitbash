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
      octree_resolution: z
        .number()
        .int()
        .optional()
        .describe(
          "Mesh density. 256 is the default and peaks near 7.6 GiB of VRAM; " +
            "lower it to 128 for simple parts or a smaller card.",
        ),
      num_inference_steps: z.number().int().optional().describe("Default 30."),
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
        octree_resolution: args.octree_resolution,
        num_inference_steps: args.num_inference_steps,
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
        watertight: r.watertight,
        generation_seconds: r.generation_seconds,
        seed: r.params.seed ?? null,
        note:
          r.faces > 50_000
            ? `${r.faces} faces is very dense for a game engine — plan on decimating before import.`
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
