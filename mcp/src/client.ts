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
  image_b64: string;
  part_name?: string;
  seed?: number;
  octree_resolution?: number;
  num_inference_steps?: number;
  guidance_scale?: number;
}): Promise<Job> {
  return request<Job>("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getJob(id: string): Promise<Job> {
  return request<Job>(`/jobs/${id}`, undefined, 10_000);
}

export function listJobs(limit = 20): Promise<{ jobs: Job[] }> {
  return request<{ jobs: Job[] }>(`/jobs?limit=${limit}`, undefined, 10_000);
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
