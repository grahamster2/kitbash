/**
 * Exports an assembled scene for Roblox, over MCP.
 *
 *   KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/export-demo.mjs
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { statSync } from "node:fs";

const transport = new StdioClientTransport({
  command: "node", args: ["dist/index.js"], env: { ...process.env },
});
const client = new Client({ name: "kitbash-export-demo", version: "0.1.0" });
await client.connect(transport);

const listed = await client.callTool({ name: "list_generation_jobs", arguments: { limit: 10 } });
const done = JSON.parse(listed.content[0].text).filter((j) => j.status === "done");
if (!done.length) { console.error("no completed jobs; run scripts/smoke.mjs"); process.exit(1); }

// Assemble first, so the export exercises the multi-part path where Roblox's
// per-mesh triangle budget actually matters.
const asm = await client.callTool({ name: "assemble_parts", arguments: {
  output_path: ".local-out/for-roblox-src.glb", scene_name: "robloxdemo",
  parts: [
    { job_id: done[0].id, name: "base", position: [0, 0, 0] },
    { job_id: done[0].id, name: "top", position: [0, 1.8, 0], scale: 0.5 },
  ],
}});
if (asm.isError) { console.error(asm.content[0].text); process.exit(1); }
const sceneId = JSON.parse(asm.content[0].text).scene_id;
console.log("assembled scene:", sceneId);

const exported = await client.callTool({ name: "export_for_roblox", arguments: {
  scene_id: sceneId, output_path: ".local-out/for-roblox.glb", height_studs: 6,
}});
console.log("\nexport_for_roblox:\n" + exported.content[0].text);
if (exported.isError) process.exit(1);
console.log("\non disk:", statSync(".local-out/for-roblox.glb").size, "bytes");
await client.close();
console.log("EXPORT DEMO PASSED");
