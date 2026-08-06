/**
 * End-to-end smoke test: drives the MCP server over stdio exactly as a coding
 * agent would, and generates a real mesh on the GPU.
 *
 *   KITBASH_SERVER_URL=http://<gpu-host>:8188 node scripts/smoke.mjs <image>
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { statSync } from "node:fs";

const image = process.argv[2];
if (!image) {
  console.error("usage: node scripts/smoke.mjs <reference-image>");
  process.exit(2);
}

const transport = new StdioClientTransport({
  command: "node",
  args: ["dist/index.js"],
  env: { ...process.env },
});

const client = new Client({ name: "kitbash-smoke", version: "0.1.0" });
await client.connect(transport);

const { tools } = await client.listTools();
console.log("tools:", tools.map((t) => t.name).join(", "));

const health = await client.callTool({ name: "check_gpu_server", arguments: {} });
console.log("\n--- check_gpu_server ---\n" + health.content[0].text);
if (health.isError) process.exit(1);

const out = ".local-out/mcp-part.glb";
console.log(`\n--- generate_part (${image}) ---`);
const t0 = Date.now();
const gen = await client.callTool(
  {
    name: "generate_part",
    arguments: {
      image_path: image,
      output_path: out,
      part_name: "smoke",
      seed: 7,
      timeout_seconds: 420,
    },
    // Asking for progress is what keeps the request alive past 60s.
    _meta: { progressToken: "smoke-1" },
  },
  undefined,
  {
    timeout: 120_000,
    maxTotalTimeout: 600_000,
    resetTimeoutOnProgress: true,
    onprogress: (p) => console.log(`  progress: ${p.message ?? p.progress}`),
  },
);
console.log(gen.content[0].text);
if (gen.isError) process.exit(1);

console.log(`\nwall clock: ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log(`file on disk: ${statSync(out).size} bytes`);

const jobs = await client.callTool({
  name: "list_generation_jobs",
  arguments: { limit: 3 },
});
console.log("\n--- list_generation_jobs ---\n" + jobs.content[0].text);

await client.close();
console.log("\nMCP SMOKE TEST PASSED");
