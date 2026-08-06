/**
 * Assembles already-generated parts into one scene, over MCP.
 *
 * Uses existing completed jobs rather than generating new ones, so it runs in
 * a second and exercises the assembly path on its own.
 *
 *   KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/assemble-demo.mjs
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { statSync } from "node:fs";

const transport = new StdioClientTransport({
  command: "node",
  args: ["dist/index.js"],
  env: { ...process.env },
});
const client = new Client({ name: "kitbash-assemble-demo", version: "0.1.0" });
await client.connect(transport);

const listed = await client.callTool({
  name: "list_generation_jobs",
  arguments: { limit: 10 },
});
const done = JSON.parse(listed.content[0].text).filter((j) => j.status === "done");
if (done.length === 0) {
  console.error("No completed jobs on the server. Run scripts/smoke.mjs first.");
  process.exit(1);
}
console.log(`using job ${done[0].id} (${done[0].faces} faces) for every part`);

const measured = await client.callTool({
  name: "describe_part",
  arguments: { job_id: done[0].id },
});
console.log("\ndescribe_part:\n" + measured.content[0].text);

// Placement is deliberately hardcoded here. In real use a coding agent decides
// this — it knows what it is building; the server only supplies measurements.
const assembled = await client.callTool({
  name: "assemble_parts",
  arguments: {
    output_path: ".local-out/assembled.glb",
    scene_name: "demo",
    parts: [
      { job_id: done[0].id, name: "hull", position: [0, 0, 0] },
      { job_id: done[0].id, name: "turret", position: [0, 1.8, 0], scale: 0.5 },
      {
        job_id: done[0].id,
        name: "cannon",
        position: [0, 1.8, 1.2],
        scale: 0.25,
        rotation: [90, 0, 0],
      },
    ],
  },
});
console.log("\nassemble_parts:\n" + assembled.content[0].text);
if (assembled.isError) process.exit(1);

console.log("\non disk:", statSync(".local-out/assembled.glb").size, "bytes");
await client.close();
console.log("ASSEMBLY DEMO PASSED");
