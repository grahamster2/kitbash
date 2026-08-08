/**
 * HTTP client for the Kitbash GPU server.
 *
 * The GPU may be on this machine or on a box across the internet; the only
 * difference is KITBASH_SERVER_URL. That is the entire reason the server is a
 * separate process speaking HTTP rather than something imported in-tree.
 */

export const SERVER_URL = (
  process.env.KITBASH_SERVER_URL ?? "http://127.0.0.1:8188"
).replace(/\/+$/, "");

export interface JobResult {
  mesh_path: string;
  generation_seconds: number;
  peak_vram_gib: number;
  vertices: number;
  faces: number;
  decimated_from: number | null;
  watertight: boolean;
  file_bytes: number;
  params: Record<string, unknown>;
}

export interface Job {
  id: string;
  type: string;
  status: "queued" | "running" | "done" | "error";
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  params: Record<string, unknown>;
  result: JobResult | null;
  error: string | null;
}

export interface Health {
  status: string;
  uptime_seconds: number;
  model_loaded: boolean;
  gpu: {
    device: string;
    free_gib: number;
    total_gib: number;
    allocated_gib: number;
  } | null;
  queue_depth: number;
  running: string[];
}

/** Turns the two failure modes an agent actually hits into readable advice. */
export class KitbashError extends Error {}

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 30_000,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${SERVER_URL}${path}`, {
      ...init,
      signal: controller.signal,
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new KitbashError(
        `${init?.method ?? "GET"} ${path} -> ${res.status} ${res.statusText}${
          body ? `: ${body.slice(0, 400)}` : ""
        }`,
      );
    }
    return (await res.json()) as T;
  } catch (err) {
    if (err instanceof KitbashError) throw err;
    const reason = err instanceof Error ? err.message : String(err);
    // A generic "fetch failed" here is the single most common thing an agent
    // will see, and it is almost never a bug in the request.
    throw new KitbashError(
      `Could not reach the Kitbash GPU server at ${SERVER_URL} (${reason}). ` +
        `Check that the server is running, and that KITBASH_SERVER_URL points at it.`,
    );
  } finally {
    clearTimeout(timer);
  }
}

export function health(): Promise<Health> {
  return request<Health>("/health", undefined, 10_000);
}

export function submitJob(body: {
  // Exactly one of these. image_id reuses a reference already on the server,
  // which is how a chosen candidate reaches generation without the picture
  // making a base64 round trip through the agent's context.
  image_b64?: string;
  image_id?: string;
  part_name?: string;
  seed?: number;
  target_faces?: number;
  octree_resolution?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
  generator?: string;
  textured?: boolean;
  texture?: boolean;
}): Promise<Job> {
  return request<Job>("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function decompose(plan: unknown): Promise<{
  subject: string;
  parts: unknown[];
  job_ids: string[];
  failed: unknown[];
  warnings: string[];
  elapsed_seconds: number;
  assemble_request: unknown[];
}> {
  // A plan is many image generations and many meshes; the server holds the
  // request open until every job is queued.
  return request("/decompose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(plan),
  }, 600_000);
}

export function decomposeExamples(): Promise<{ examples: Record<string, unknown> }> {
  return request("/decompose/examples", undefined, 20_000);
}

export interface StrategyRequest {
  subject: string;
  intent?: string;
  target?: string;
  detail?: string;
  target_faces?: number;
  lod?: boolean;
  quantity?: number;
  parts?: string[];
  low_poly?: boolean;
  interior?: boolean;
  max_generations?: number;
  style?: string;
  seed?: number;
  name?: string;
  notes?: string;
}

export interface Recommendation {
  subject: string;
  strategy: "single" | "hybrid" | "scripted";
  family: string;
  headline: string;
  confidence: { level: string; margin: number; why: string };
  reasoning: { saw: string; claim: string; evidence: string; source: string }[];
  scores: Record<string, number | null>;
  alternatives: { strategy: string; why_not: string; when_it_would_win: string }[];
  routing: { part: string; mode: string; archetype: string | null; why: string }[];
  budget: Record<string, unknown>;
  warnings: { code: string; severity: string; message: string; evidence: string;
              source: string; part: string | null }[];
  plan_warnings: string[];
  cost: Record<string, any>;
  plan: Record<string, unknown>;
  draft_disclaimer: string;
  next_steps: string[];
}

/** The decision layer. Pure CPU on the server — no GPU, milliseconds. */
export function chooseStrategy(body: StrategyRequest): Promise<Recommendation> {
  return request<Recommendation>("/strategy", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, 30_000);
}

export function strategyArchetypes(): Promise<Record<string, unknown>> {
  return request("/strategy/archetypes", undefined, 20_000);
}

export function strategyTargets(): Promise<Record<string, unknown>> {
  return request("/strategy/targets", undefined, 20_000);
}

export function costPlan(body: {
  plan: unknown;
  model_resident?: boolean;
  high_resolution?: string[];
}): Promise<Record<string, any>> {
  return request("/strategy/cost", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, 30_000);
}

export function planWarnings(plan: unknown): Promise<{
  warnings: { code: string; severity: string; message: string;
              evidence: string; source: string; part: string | null }[];
}> {
  return request("/strategy/warnings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(plan),
  }, 30_000);
}

export function buildLods(jobId: string, levels: number[], fromRaw = true): Promise<{
  source_job: string;
  source: string;
  source_faces: number;
  levels: { job_id: string; requested: number; faces: number; seconds: number;
            file_bytes: number; watertight: boolean }[];
  note: string;
}> {
  return request(`/jobs/${jobId}/lod`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ levels, from_raw: fromRaw }),
  }, 120_000);
}

export function generators(): Promise<unknown> {
  return request("/generators", undefined, 15_000);
}

export function getJob(id: string): Promise<Job> {
  return request<Job>(`/jobs/${id}`, undefined, 10_000);
}

export function listJobs(limit = 20): Promise<{ jobs: Job[] }> {
  return request<{ jobs: Job[] }>(`/jobs?limit=${limit}`, undefined, 10_000);
}

export interface PartPlacement {
  job_id: string;
  name: string;
  position?: number[];
  rotation?: number[];
  scale?: number | number[];
  material?: string;
  color?: string;
  use_raw?: boolean;
}

export interface AssembledScene {
  scene_id: string;
  scene_path: string;
  part_count: number;
  total_faces: number;
  parts: { name: string; faces: number; material: string | null; source: string }[];
  bounds_min: number[];
  bounds_max: number[];
  size: number[];
  file_bytes: number;
}

export function assembleScene(body: {
  parts: PartPlacement[];
  scene_name?: string;
  apply_materials?: boolean;
}): Promise<AssembledScene> {
  return request<AssembledScene>("/assemble", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function describePart(id: string): Promise<{
  faces: number;
  bounds_min: number[];
  bounds_max: number[];
  size: number[];
  center: number[];
}> {
  return request(`/jobs/${id}/describe`, undefined, 15_000);
}

export interface ExportResult {
  target: string;
  primary: string;
  files: Record<string, string | string[]>;
  parts: { name: string; faces: number }[];
  part_count: number;
  total_faces: number;
  source_faces: number;
  size: number[];
  pivot: string;
  warnings: string[];
}

export function exportMesh(body: {
  job_id?: string;
  scene_id?: string;
  target: string;
  height_studs?: number;
}): Promise<ExportResult> {
  return request<ExportResult>("/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function downloadExported(remotePath: string): Promise<Uint8Array> {
  const res = await fetch(
    `${SERVER_URL}/export/file?path=${encodeURIComponent(remotePath)}`,
  );
  if (!res.ok) {
    throw new KitbashError(
      `download of ${remotePath} failed: ${res.status} ${res.statusText}`,
    );
  }
  return new Uint8Array(await res.arrayBuffer());
}

export async function downloadScene(sceneId: string): Promise<Uint8Array> {
  const res = await fetch(`${SERVER_URL}/scenes/${sceneId}/mesh`);
  if (!res.ok) {
    throw new KitbashError(
      `download of scene ${sceneId} failed: ${res.status} ${res.statusText}`,
    );
  }
  return new Uint8Array(await res.arrayBuffer());
}

export interface PreviewOptions {
  views?: string[];
  size?: number;
  columns?: number;
  highlight?: string;
  isolate?: boolean;
}

function previewQuery(opts: PreviewOptions): string {
  const q = new URLSearchParams();
  if (opts.views?.length) q.set("views", opts.views.join(","));
  if (opts.size) q.set("size", String(opts.size));
  if (opts.columns) q.set("columns", String(opts.columns));
  if (opts.highlight) q.set("highlight", opts.highlight);
  if (opts.isolate) q.set("isolate", "true");
  const s = q.toString();
  return s ? `?${s}` : "";
}

/**
 * Renders a scene or a part to a PNG contact sheet. Returns raw PNG bytes.
 *
 * Not `request()`: that one parses JSON, and this is the one endpoint whose
 * whole value is that it comes back as an image.
 */
async function fetchPng(path: string): Promise<Uint8Array> {
  let res: Response;
  try {
    res = await fetch(`${SERVER_URL}${path}`, {
      signal: AbortSignal.timeout(120_000),
    });
  } catch (err) {
    throw new KitbashError(
      `Could not reach the Kitbash GPU server at ${SERVER_URL} ` +
        `(${err instanceof Error ? err.message : String(err)}).`,
    );
  }
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new KitbashError(
      `GET ${path} -> ${res.status} ${res.statusText}${
        body ? `: ${body.slice(0, 400)}` : ""
      }`,
    );
  }
  return new Uint8Array(await res.arrayBuffer());
}

export function previewScene(
  sceneId: string,
  opts: PreviewOptions = {},
): Promise<Uint8Array> {
  return fetchPng(`/scenes/${sceneId}/preview${previewQuery(opts)}`);
}

// --- candidate reference images --------------------------------------------

export interface Candidate {
  image_id: string;
  prompt: string;
  variant: string | null;
  seed: number;
  bytes: number;
  path: string;
}

export interface CandidateBatch {
  batch_id: string;
  prompt: string;
  candidates: Candidate[];
  count: number;
  requested: number;
  elapsed_seconds: number;
  provider: string;
  mode: "variants" | "mechanical";
  image_size: string;
  failed: { index: number; variant: string | null; prompt: string;
            seed: number; error: string }[];
  created_at: number;
}

/**
 * N reference images for one prompt, all of them unchosen.
 *
 * The server generates them concurrently, so four candidates cost four billed
 * image calls and roughly the wall time of one. 120s rather than the default
 * 30: a slow queue at the provider stalls the whole batch.
 */
export function generateCandidates(body: {
  prompt: string;
  count?: number;
  variants?: string[];
  image_size?: string;
  seed?: number;
  remove_background?: boolean;
}): Promise<CandidateBatch> {
  return request<CandidateBatch>("/images/candidates", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, 120_000);
}

export function getCandidateBatch(batchId: string): Promise<CandidateBatch> {
  return request<CandidateBatch>(`/images/batches/${batchId}`, undefined, 20_000);
}

/** One stored reference image as raw PNG bytes, for showing it to the user. */
export function downloadImage(imageId: string): Promise<Uint8Array> {
  return fetchPng(`/images/${imageId}`);
}

export function previewJob(
  jobId: string,
  opts: PreviewOptions = {},
): Promise<Uint8Array> {
  return fetchPng(`/jobs/${jobId}/preview${previewQuery(opts)}`);
}

export interface GroundReport {
  scene_id: string;
  floor_y: number;
  parts: { name: string; gap: number; gap_fraction: number; faces: number }[];
}

export function sceneGround(sceneId: string): Promise<GroundReport> {
  return request<GroundReport>(`/scenes/${sceneId}/ground`, undefined, 60_000);
}

/** Downloads the finished mesh. Returns raw GLB bytes. */
export async function downloadMesh(id: string): Promise<Uint8Array> {
  const res = await fetch(`${SERVER_URL}/jobs/${id}/mesh`);
  if (!res.ok) {
    throw new KitbashError(
      `download of job ${id} failed: ${res.status} ${res.statusText}`,
    );
  }
  return new Uint8Array(await res.arrayBuffer());
}

/**
 * Polls until the job leaves the queue, or the deadline passes.
 *
 * onPoll fires on every tick. Callers use it to emit MCP progress
 * notifications: generation outlives the 60s default request timeout in most
 * clients, and progress is what resets that timer.
 */
export async function waitForJob(
  id: string,
  timeoutMs: number,
  onPoll?: (job: Job, elapsedMs: number) => void | Promise<void>,
  pollMs = 2000,
): Promise<Job> {
  const start = Date.now();
  const deadline = start + timeoutMs;
  for (;;) {
    const job = await getJob(id);
    await onPoll?.(job, Date.now() - start);
    if (job.status === "done" || job.status === "error") return job;
    if (Date.now() >= deadline) return job;
    await new Promise((r) => setTimeout(r, pollMs));
  }
}
