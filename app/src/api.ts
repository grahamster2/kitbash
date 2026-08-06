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

export interface SubmitBody {
  image_b64: string;
  part_name?: string;
  seed?: number;
  target_faces?: number;
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

export const assemble = (parts: ScenePart[], sceneName?: string) =>
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
