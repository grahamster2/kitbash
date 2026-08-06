import * as api from "./api";
import { Viewport } from "./viewport";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  baseUrl: $<HTMLInputElement>("base-url"),
  serverStatus: $<HTMLSpanElement>("server-status"),
  serverDetail: $<HTMLParagraphElement>("server-detail"),
  imageInput: $<HTMLInputElement>("image-input"),
  imagePreview: $<HTMLImageElement>("image-preview"),
  dropLabel: $<HTMLSpanElement>("drop-label"),
  partName: $<HTMLInputElement>("part-name"),
  seed: $<HTMLInputElement>("seed"),
  targetFaces: $<HTMLInputElement>("target-faces"),
  generate: $<HTMLButtonElement>("generate"),
  submitStatus: $<HTMLParagraphElement>("submit-status"),
  refresh: $<HTMLButtonElement>("refresh"),
  jobList: $<HTMLUListElement>("job-list"),
  hudTitle: $<HTMLSpanElement>("hud-title"),
  hudStats: $<HTMLSpanElement>("hud-stats"),
  viewportError: $<HTMLParagraphElement>("viewport-error"),
  wireframe: $<HTMLButtonElement>("toggle-wireframe"),
  rotate: $<HTMLButtonElement>("toggle-rotate"),
};

const viewport = new Viewport($("viewport"));

let jobs: api.Job[] = [];
let selectedId: string | null = null;
let pendingImage: string | null = null;
let watching: string | null = null;
// Mesh downloads cross a network; a slow one must not clobber a newer click.
let loadToken = 0;

const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const jobLabel = (j: api.Job) => (j.params.part_name as string) || j.id;
const isActive = (j: api.Job) => j.status === "queued" || j.status === "running";

function showError(msg: string | null) {
  els.viewportError.textContent = msg ?? "";
  els.viewportError.hidden = !msg;
}

/* ---------- server ---------- */

els.baseUrl.addEventListener("change", () => {
  els.baseUrl.value = api.setBaseUrl(els.baseUrl.value);
  jobs = [];
  selectedId = null;
  renderJobs();
  void pollHealth();
  void refreshJobs();
});

async function pollHealth() {
  try {
    const h = await api.health();
    const gpu = h.gpu;
    els.serverStatus.textContent = "online";
    els.serverStatus.className = "pill ok";
    els.serverDetail.className = "detail";
    els.serverDetail.textContent = gpu
      ? `${gpu.device} — ${gpu.free_gib.toFixed(1)}/${gpu.total_gib.toFixed(0)} GiB free` +
        `${h.model_loaded ? ", model resident" : ""}` +
        `${h.queue_depth ? `, ${h.queue_depth} queued` : ""}`
      : "no CUDA device reported";
  } catch (e) {
    els.serverStatus.textContent = "offline";
    els.serverStatus.className = "pill down";
    els.serverDetail.className = "detail error";
    els.serverDetail.textContent = errText(e);
  }
}

/* ---------- job list ---------- */

async function refreshJobs() {
  try {
    const res = await api.listJobs(30);
    jobs = res.jobs;
    renderJobs();
  } catch {
    // pollHealth already surfaces an unreachable server; don't say it twice.
  }
}

function renderJobs() {
  els.jobList.replaceChildren(
    ...jobs.map((job) => {
      const li = document.createElement("li");
      li.className = "job";
      if (job.id === selectedId) li.classList.add("active");
      if (job.status !== "done") li.classList.add("pending");

      const head = document.createElement("div");
      head.className = "job-head";
      const name = document.createElement("span");
      name.className = "job-name";
      name.textContent = jobLabel(job);
      const status = document.createElement("span");
      status.className = `job-status ${job.status}`;
      status.textContent = job.status;
      head.append(name, status);

      const meta = document.createElement("span");
      meta.className = "detail";
      meta.textContent = jobMeta(job);

      li.append(head, meta);
      if (job.status === "done") li.addEventListener("click", () => void selectJob(job.id));
      return li;
    }),
  );
  if (!jobs.length) {
    const empty = document.createElement("p");
    empty.className = "detail";
    empty.textContent = "No jobs yet.";
    els.jobList.append(empty);
  }
}

function jobMeta(job: api.Job): string {
  if (job.status === "error") return job.error ?? "failed";
  if (job.result) {
    const r = job.result;
    return `${r.faces.toLocaleString()} faces · ${r.generation_seconds.toFixed(0)}s · ${(
      r.file_bytes / 1024
    ).toFixed(0)} KB`;
  }
  const since = job.started_at ?? job.created_at;
  return `${job.status} · ${Math.max(0, Math.round(Date.now() / 1000 - since))}s`;
}

/* ---------- viewport ---------- */

async function selectJob(id: string) {
  const token = ++loadToken;
  selectedId = id;
  renderJobs();
  showError(null);

  const job = jobs.find((j) => j.id === id);
  els.hudTitle.textContent = job ? jobLabel(job) : id;
  els.hudStats.textContent = "loading mesh…";

  try {
    const glb = await api.fetchMesh(id);
    if (token !== loadToken) return;
    const stats = await viewport.load(glb);
    if (token !== loadToken) return;
    const [x, y, z] = stats.size;
    els.hudStats.textContent =
      `${stats.triangles.toLocaleString()} tris · ` +
      `${x.toFixed(2)} × ${y.toFixed(2)} × ${z.toFixed(2)} · ${id}`;
    // Wireframe is per-material, so a freshly loaded mesh has to be re-flagged.
    viewport.setWireframe(els.wireframe.classList.contains("on"));
  } catch (e) {
    if (token !== loadToken) return;
    els.hudStats.textContent = "";
    showError(`Could not load mesh for ${id}: ${errText(e)}`);
  }
}

els.wireframe.addEventListener("click", () => {
  const on = els.wireframe.classList.toggle("on");
  viewport.setWireframe(on);
});

els.rotate.addEventListener("click", () => {
  const on = els.rotate.classList.toggle("on");
  viewport.setAutoRotate(on);
});

/* ---------- submission ---------- */

function useImage(file: File) {
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result as string;
    pendingImage = dataUrl.slice(dataUrl.indexOf(",") + 1);
    els.imagePreview.src = dataUrl;
    els.imagePreview.hidden = false;
    els.dropLabel.textContent = file.name;
    els.generate.disabled = watching !== null;
  };
  reader.readAsDataURL(file);
}

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files?.[0];
  if (file) useImage(file);
});

const drop = $<HTMLLabelElement>("drop");
drop.addEventListener("dragover", (e) => e.preventDefault());
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer?.files?.[0];
  if (file?.type.startsWith("image/")) useImage(file);
});

els.generate.addEventListener("click", () => void submit());

async function submit() {
  if (!pendingImage) return;
  els.generate.disabled = true;
  els.submitStatus.className = "detail";
  els.submitStatus.textContent = "submitting…";

  const seed = Number.parseInt(els.seed.value, 10);
  const faces = Number.parseInt(els.targetFaces.value, 10);
  try {
    const job = await api.submitJob({
      image_b64: pendingImage,
      part_name: els.partName.value.trim() || undefined,
      seed: Number.isFinite(seed) ? seed : undefined,
      target_faces: Number.isFinite(faces) ? faces : undefined,
    });
    watching = job.id;
    els.submitStatus.textContent = `queued as ${job.id}`;
    await refreshJobs();
  } catch (e) {
    els.submitStatus.className = "detail error";
    els.submitStatus.textContent = errText(e);
    els.generate.disabled = false;
  }
}

/** Follows the job submitted from this window until it finishes, then shows it. */
async function pollWatched() {
  if (!watching) return;
  try {
    const job = await api.getJob(watching);
    if (job.status === "done") {
      els.submitStatus.textContent = `${job.id} done in ${job.result?.generation_seconds.toFixed(0)}s`;
      const id = job.id;
      watching = null;
      els.generate.disabled = !pendingImage;
      await refreshJobs();
      await selectJob(id);
    } else if (job.status === "error") {
      els.submitStatus.className = "detail error";
      els.submitStatus.textContent = job.error ?? "generation failed";
      watching = null;
      els.generate.disabled = !pendingImage;
    } else {
      const since = job.started_at ?? job.created_at;
      const secs = Math.max(0, Math.round(Date.now() / 1000 - since));
      els.submitStatus.textContent = `${job.id} ${job.status} · ${secs}s`;
    }
  } catch (e) {
    els.submitStatus.className = "detail error";
    els.submitStatus.textContent = errText(e);
  }
}

/* ---------- loops ---------- */

els.refresh.addEventListener("click", () => void refreshJobs());

// The server is often across a network on a machine that is busy generating,
// so idle polling stays slow and only tightens while work is in flight.
let idleTicks = 0;
setInterval(() => {
  const busy = watching !== null || jobs.some(isActive);
  if (busy) {
    void pollWatched();
    void refreshJobs();
    idleTicks = 0;
  } else if (++idleTicks >= 6) {
    idleTicks = 0;
    void refreshJobs();
  }
}, 2500);

setInterval(() => void pollHealth(), 10_000);

async function start() {
  els.baseUrl.value = await api.initBaseUrl();
  await Promise.all([pollHealth(), refreshJobs()]);
  // Opening onto the last finished part beats opening onto an empty void.
  const latest = jobs.find((j) => j.status === "done");
  if (latest) await selectJob(latest.id);
}

void start();
