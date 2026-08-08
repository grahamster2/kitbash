/**
 * Typed front for the Rust commands in src-tauri/src/lib.rs.
 *
 * The base URL lives here rather than in the Rust side so that changing which
 * machine owns the GPU is a text field in the UI, not a restart.
 */
import { invoke } from "@tauri-apps/api/core";

const FALLBACK_BASE_URL = "http://127.0.0.1:8188";
const BASE_URL_KEY = "kitbash.baseUrl";

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

/** Exactly one of `image_b64` / `image_id` — a dropped file or a picked candidate. */
export interface SubmitBody {
  image_b64?: string;
  image_id?: string;
  part_name?: string;
  seed?: number;
  target_faces?: number;
  generator?: string;
  textured?: boolean;
}

/** One idea in a batch. `variant` is the server's angle on the prompt, if any. */
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
}

export interface CandidatesBody {
  prompt: string;
  count: number;
  variants?: string[] | null;
  image_size?: string | null;
  seed?: number | null;
  remove_background: boolean;
}

export interface SingleImage {
  image_id: string;
  path: string;
  provider: string;
  prompt: string;
  bytes: number;
}

export type Vec3 = [number, number, number];

export interface Describe {
  faces: number;
  bounds_min: Vec3;
  bounds_max: Vec3;
  size: Vec3;
  center: Vec3;
}

/**
 * One entry of an /assemble request.
 *
 * glTF is +Y up and so is Roblox, so a position written here is the position
 * you get in Studio — there is no axis conversion anywhere in this path.
 * Rotation is XYZ euler degrees; the server applies scale, then rotate, then
 * translate.
 */
export interface ScenePart {
  job_id: string;
  name: string;
  position?: Vec3;
  rotation?: Vec3;
  scale?: number | Vec3;
  use_raw?: boolean;
}

/**
 * What `/decompose` hands back ready to post. It carries more than a hand-built
 * `ScenePart` — anchors, mirrors, materials and colours the plan already stated
 * — and `assemble.py` owns what those keys mean, so this passes through opaque
 * rather than being retyped here and drifting.
 */
export type AssemblePart = ScenePart & Record<string, unknown>;

export interface Scene {
  scene_id: string;
  scene_path: string;
  part_count: number;
  total_faces: number;
  parts: { name: string; faces: number; source: string }[];
  bounds_min: Vec3;
  bounds_max: Vec3;
  size: Vec3;
  file_bytes: number;
}

/* ---------- the decision layer ---------- */

/** `generate` costs a GPU minute, `script` costs milliseconds, `mirror` is free. */
export type PartMode = "generate" | "script" | "mirror";

/**
 * One part of a decompose plan. Deliberately loose: the plan is written by the
 * server (or by a coding agent through MCP) and is posted back to `/decompose`
 * unchanged, so nothing here may narrow what survives the round trip.
 */
export interface PlanPart extends Record<string, unknown> {
  name: string;
  mode: PartMode;
  kind?: string;
  prompt?: string;
  target_faces?: number;
  note?: string;
}

export interface Plan extends Record<string, unknown> {
  name?: string;
  subject: string;
  seed?: number;
  generator?: string;
  target_faces?: number;
  textured?: boolean;
  parts: PlanPart[];
}

export interface Reason {
  saw: string;
  argues_for: string;
  claim: string;
  evidence: string;
  source: string;
}

export interface Ceiling {
  code: string;
  severity: "blocker" | "warning" | "note";
  message: string;
  evidence: string;
  source: string;
  part: string | null;
}

export interface Cost {
  wall_human: string;
  gpu_seconds: { low: number; likely: number; high: number };
  generations: number;
  estimated_size: string;
  parts: { total: number; generated: number; scripted: number; mirrored: number };
  triangles: { total: number; largest_part: number };
  savings: string[];
}

/** `POST /strategy` — a recommendation, its evidence, its price, and a draft plan. */
export interface Recommendation {
  subject: string;
  strategy: "single" | "hybrid" | "scripted";
  family: string;
  headline: string;
  confidence: { level: string; why: string };
  reasoning: Reason[];
  warnings: Ceiling[];
  plan_warnings: string[];
  cost: Cost;
  plan: Plan;
  draft_disclaimer: string;
  budget: { target: string; target_assumed: boolean; faces_per_part: number };
}

/** One part of a running build, as `/decompose` reports it. */
export interface PartRecord {
  name: string;
  mode: PartMode;
  job_id: string | null;
  image_id: string | null;
  prompt: string | null;
  status: string;
  error: string | null;
  faces?: number;
  note?: string | null;
}

export interface DecomposeResult {
  subject: string;
  parts: PartRecord[];
  job_ids: string[];
  failed: string[];
  warnings: string[];
  elapsed_seconds: number;
  assemble_request: AssemblePart[];
}

export interface ExportBody {
  job_id?: string;
  scene_id?: string;
  target: "roblox" | "dcc";
  height_studs?: number;
}

export interface ExportResult {
  target: string;
  primary: string;
  /** `glb` and `obj` are paths; `obj_sidecars` is a list of them. */
  files: Record<string, string | string[]>;
  parts: { name: string; faces: number }[];
  part_count: number;
  total_faces: number;
  source_faces: number;
  size: Vec3;
  pivot: string;
  file_bytes: Record<string, number>;
  warnings: string[];
}

let baseUrl = localStorage.getItem(BASE_URL_KEY) ?? FALLBACK_BASE_URL;

/** Until a URL has been chosen in the UI, the shell's KITBASH_SERVER_URL wins. */
export async function initBaseUrl(): Promise<string> {
  if (localStorage.getItem(BASE_URL_KEY) === null) {
    baseUrl = await invoke<string>("default_base_url");
  }
  return baseUrl;
}

export function getBaseUrl(): string {
  return baseUrl;
}

export function setBaseUrl(next: string): string {
  baseUrl = (next.trim() || FALLBACK_BASE_URL).replace(/\/+$/, "");
  localStorage.setItem(BASE_URL_KEY, baseUrl);
  return baseUrl;
}

export const health = () => invoke<Health>("health", { baseUrl });

export const listJobs = (limit = 30) =>
  invoke<{ jobs: Job[] }>("list_jobs", { baseUrl, limit });

export const getJob = (id: string) => invoke<Job>("get_job", { baseUrl, id });

export const submitJob = (body: SubmitBody) =>
  invoke<Job>("submit_job", { baseUrl, body });

export const describeJob = (id: string) =>
  invoke<Describe>("describe_job", { baseUrl, id });

export const createCandidates = (body: CandidatesBody) =>
  invoke<CandidateBatch>("create_candidates", { baseUrl, body });

export const getBatch = (id: string) =>
  invoke<CandidateBatch>("get_batch", { baseUrl, id });

export const createImage = (body: {
  prompt: string;
  seed?: number;
  image_size?: string;
  remove_background: boolean;
}) => invoke<SingleImage>("create_image", { baseUrl, body });

/**
 * The decision in front of everything else. No LLM and no GPU: it reads the
 * subject and the intent prose, and answers in about a millisecond with what it
 * would build, what that costs, and a draft plan that validates.
 */
export const strategy = (body: { subject: string; intent?: string }) =>
  invoke<Recommendation>("strategy", { baseUrl, body });

/** Executes a plan: images inline, meshes queued, ids and an /assemble request back. */
export const decompose = (plan: Plan) =>
  invoke<DecomposeResult>("decompose", { baseUrl, body: plan });

export const assemble = (parts: AssemblePart[], sceneName?: string) =>
  invoke<Scene>("assemble", { baseUrl, body: { parts, scene_name: sceneName } });

export const exportScene = (body: ExportBody) =>
  invoke<ExportResult>("export", { baseUrl, body });

/** `path` is a server-side path from an export result; `dest` is local. */
export const downloadExportedFile = (path: string, dest: string) =>
  invoke<string>("download_exported_file", { baseUrl, path, dest });

/**
 * A byte response arrives as an ArrayBuffer when the frontend is served over
 * the dev server, but as a plain number array from the packaged app. Normalise
 * here so the viewport only ever sees bytes.
 */
async function bytes(cmd: string, id: string): Promise<ArrayBuffer> {
  const raw = await invoke<ArrayBuffer | number[]>(cmd, { baseUrl, id });
  return raw instanceof ArrayBuffer ? raw : new Uint8Array(raw).buffer;
}

export const fetchMesh = (id: string) => bytes("fetch_mesh", id);
export const fetchScene = (id: string) => bytes("fetch_scene", id);
/** PNG bytes for a candidate — the caller wraps them in a blob URL. */
export const fetchImage = (id: string) => bytes("fetch_image", id);
