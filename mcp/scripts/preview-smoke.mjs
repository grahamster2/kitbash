/**
 * Drives preview_scene / preview_part over stdio, exactly as a coding agent
 * would, and checks the result really is MCP image content.
 *
 * The point of the preview tools is that the model *sees* the picture, and the
 * only thing that makes that happen is the content block being
 * {type:"image", data, mimeType} rather than a file path in some text. This
 * asserts that, then writes the PNG out so a human can look at it too.
 *
 *   KITBASH_SERVER_URL=http://<gpu-host>:8188 \
 *     node scripts/preview-smoke.mjs <scene_id> [job_id]
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { mkdirSync, writeFileSync } from "node:fs";

const sceneId = process.argv[2];
const jobId = process.argv[3];
if (!sceneId) {
  console.error("usage: node scripts/preview-smoke.mjs <scene_id> [job_id]");
  process.exit(2);
}

const transport = new StdioClientTransport({
  command: "node",
  args: ["dist/index.js"],
  env: { ...process.env },
});
const client = new Client({ name: "kitbash-preview-smoke", version: "0.1.0" });
await client.connect(transport);

mkdirSync(".local-out", { recursive: true });

async function shoot(name, args, out) {
  const t0 = Date.now();
  const res = await client.callTool({ name, arguments: args });
  const ms = Date.now() - t0;
  if (res.isError) {
    console.error(`${name} failed: ${res.content[0].text}`);
    process.exit(1);
  }
  const img = res.content.find((c) => c.type === "image");
  const text = res.content.find((c) => c.type === "text");
  if (!img) {
    console.error(`${name} returned no image content: ${JSON.stringify(res.content)}`);
    process.exit(1);
  }
  if (img.mimeType !== "image/png") {
    console.error(`${name} returned mimeType ${img.mimeType}`);
    process.exit(1);
  }
  const bytes = Buffer.from(img.data, "base64");
  // PNG magic. A base64 blob that decodes to anything else would still render
  // as a broken image in the client rather than as an error here.
  if (bytes.subarray(0, 4).toString("hex") !== "89504e47") {
    console.error(`${name} data is not a PNG`);
    process.exit(1);
  }
  writeFileSync(out, bytes);
  console.log(`--- ${name} ---`);
  console.log(`  ${ms} ms, ${bytes.length} bytes -> ${out}`);
  console.log(`  caption: ${(text?.text ?? "").split("\n")[0]}`);
}

const { tools } = await client.listTools();
console.log("tools:", tools.map((t) => t.name).join(", "));

await shoot("preview_scene", { scene_id: sceneId }, ".local-out/preview-scene.png");
await shoot(
  "preview_scene",
  { scene_id: sceneId, views: ["side", "top"], size: 800, columns: 2 },
  ".local-out/preview-scene-two.png",
);
if (jobId) {
  await shoot("preview_part", { job_id: jobId }, ".local-out/preview-part.png");
}

await client.close();
console.log("\nok");
