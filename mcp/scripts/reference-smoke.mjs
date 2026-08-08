/**
 * Drives the reference-selection flow over stdio, exactly as a coding agent
 * would, and checks that every candidate really is MCP image content.
 *
 * The whole feature rests on one thing: all N candidates arriving in a single
 * tool result as {type:"image", data, mimeType} blocks, each preceded by a text
 * block naming its image_id. That is what lets the model show them and lets the
 * user's "the second one" resolve back to an id. This asserts it, then writes
 * the PNGs out so a human can look too.
 *
 *   KITBASH_SERVER_URL=http://<gpu-host>:8188 \
 *     node scripts/reference-smoke.mjs "an ornate treasure chest"
 *
 * A batch id instead of a prompt re-shows an existing batch and spends nothing:
 *
 *   node scripts/reference-smoke.mjs --batch f0538374e11a
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { mkdirSync, writeFileSync } from "node:fs";

const args = process.argv.slice(2);
const batchMode = args[0] === "--batch";
const subject = batchMode ? args[1] : args[0];
if (!subject) {
  console.error(
    'usage: node scripts/reference-smoke.mjs "<prompt>" | --batch <batch_id>',
  );
  process.exit(2);
}

const transport = new StdioClientTransport({
  command: "node",
  args: ["dist/index.js"],
  env: { ...process.env },
});
const client = new Client({ name: "kitbash-reference-smoke", version: "0.1.0" });
await client.connect(transport);

const { tools } = await client.listTools();
console.log("tools:", tools.map((t) => t.name).join(", "));

mkdirSync(".local-out", { recursive: true });

const t0 = Date.now();
const res = batchMode
  ? await client.callTool({
      name: "get_reference_options",
      arguments: { batch_id: subject },
    })
  : await client.callTool({
      name: "generate_reference_options",
      arguments: {
        prompt: subject,
        count: 4,
        // Four different ideas rather than four re-rolls — the mode the tool
        // description says to use, so the smoke test should use it too.
        variants: [
          `${subject}, plain and undecorated, simple honest construction`,
          `${subject}, heavily weathered and salt-encrusted, barnacled and split`,
          `${subject}, gilded and jewelled, elaborate scrollwork`,
          `${subject}, stylised low-poly game asset, bold chunky shapes, flat colours`,
        ],
      },
    });
const ms = Date.now() - t0;

if (res.isError) {
  console.error(`failed: ${res.content[0].text}`);
  process.exit(1);
}

const images = res.content.filter((c) => c.type === "image");
if (images.length === 0) {
  console.error("no image content came back");
  process.exit(1);
}

// Each image must be preceded by the text block that names its image_id, or
// the user's answer has nothing to map back to.
const ids = [];
res.content.forEach((block, i) => {
  if (block.type !== "image") return;
  const label = res.content[i - 1];
  const id = label?.type === "text" && label.text.match(/image_id: (\w+)/)?.[1];
  if (!id) {
    console.error(`image ${ids.length + 1} has no image_id label before it`);
    process.exit(1);
  }
  if (block.mimeType !== "image/png") {
    console.error(`image ${id} has mimeType ${block.mimeType}`);
    process.exit(1);
  }
  const bytes = Buffer.from(block.data, "base64");
  if (bytes.subarray(0, 4).toString("hex") !== "89504e47") {
    console.error(`image ${id} is not a PNG`);
    process.exit(1);
  }
  const out = `.local-out/candidate-${ids.length + 1}-${id}.png`;
  writeFileSync(out, bytes);
  console.log(`  option ${ids.length + 1}: ${id}, ${bytes.length} bytes -> ${out}`);
  ids.push(id);
});

console.log(`\n${ids.length} candidates in ${ms} ms`);
console.log(res.content[0].text.split("\n")[0]);

const batchId = res.content[0].text.match(/[Bb]atch (\w+)/)?.[1];
if (batchId) {
  // Does not elicit unless the client declared the capability; this one does
  // not, so the expected answer is supported:false plus the list to ask from.
  const choose = await client.callTool({
    name: "choose_reference",
    arguments: { batch_id: batchId },
  });
  const body = JSON.parse(choose.content[0].text);
  console.log(
    `choose_reference: elicitation supported=${body.supported}` +
      (body.supported ? "" : `, ${body.options.length} options to ask about`),
  );
}

await client.close();
console.log("\nok");
