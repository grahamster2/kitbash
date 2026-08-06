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

/**
 * Raw GLB bytes.
 *
 * A byte response arrives as an ArrayBuffer when the frontend is served over
 * the dev server, but as a plain number array from the packaged app. Normalise
 * here so the viewport only ever sees bytes.
 */
export async function fetchMesh(id: string): Promise<ArrayBuffer> {
  const raw = await invoke<ArrayBuffer | number[]>("fetch_mesh", { baseUrl, id });
  return raw instanceof ArrayBuffer ? raw : new Uint8Array(raw).buffer;
}
