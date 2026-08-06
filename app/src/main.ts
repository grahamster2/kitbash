import { save } from "@tauri-apps/plugin-dialog";

import * as api from "./api";
import { PartInfo, Viewport } from "./viewport";

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
  sceneName: $<HTMLInputElement>("scene-name"),
  clearScene: $<HTMLButtonElement>("clear-scene"),
  draftList: $<HTMLDivElement>("draft-list"),
  draftEmpty: $<HTMLParagraphElement>("draft-empty"),
  draftCount: $<HTMLSpanElement>("draft-count"),
  assemble: $<HTMLButtonElement>("assemble"),
  assembleStatus: $<HTMLParagraphElement>("assemble-status"),
  loadedPanel: $<HTMLElement>("loaded-panel"),
  loadedParts: $<HTMLUListElement>("loaded-parts"),
  showAll: $<HTMLButtonElement>("show-all"),
  exportTarget: $<HTMLSelectElement>("export-target"),
  heightStuds: $<HTMLInputElement>("height-studs"),
  export: $<HTMLButtonElement>("export"),
  exportStatus: $<HTMLParagraphElement>("export-status"),
  exportWarnings: $<HTMLDivElement>("export-warnings"),
  exportFiles: $<HTMLUListElement>("export-files"),
};

const viewport = new Viewport($("viewport"));

/** A part queued for assembly, before the server has been asked to build it. */
interface DraftPart {
  key: number;
  jobId: string;
  name: string;
  position: api.Vec3;
  rotation: api.Vec3;
  scale: number;
  useRaw: boolean;
}

const DRAFT_KEY = "kitbash.draft";

let jobs: api.Job[] = [];
let selectedId: string | null = null;
let pendingImage: string | null = null;
let watching: string | null = null;
let draft: DraftPart[] = [];
let nextKey = 1;
/** What the viewport is showing, and therefore what Export will act on. */
let viewing: { kind: "job" | "scene"; id: string; label: string } | null = null;
let isolated: string | null = null;
// Real bounds per job, so placement is computed from measurements not guesses.
const measured = new Map<string, api.Describe>();
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
  // Job and scene ids belong to one server; nothing carries across a switch.
  viewing = null;
  measured.clear();
  els.export.disabled = true;
  clearExport();
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
      if (job.status === "done") {
        li.addEventListener("click", () => void selectJob(job.id));
        const add = document.createElement("button");
        add.className = "ghost add-part";
        add.type = "button";
        add.textContent = "+";
        add.title = "Add to scene";
        add.addEventListener("click", (e) => {
          e.stopPropagation();
          addDraftPart(job);
        });
        head.append(add);
      }
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
  const job = jobs.find((j) => j.id === id);
  selectedId = id;
  renderJobs();
  await show({ kind: "job", id, label: job ? jobLabel(job) : id }, () => api.fetchMesh(id));
}

async function showScene(scene: api.Scene) {
  selectedId = null;
  renderJobs();
  const label = scene.scene_path.split(/[\\/]/).pop() ?? scene.scene_id;
  await show({ kind: "scene", id: scene.scene_id, label }, () => api.fetchScene(scene.scene_id));
}

async function show(next: NonNullable<typeof viewing>, fetch: () => Promise<ArrayBuffer>) {
  const token = ++loadToken;
  showError(null);
  els.hudTitle.textContent = next.label;
  els.hudStats.textContent = "loading mesh…";

  try {
    const glb = await fetch();
    if (token !== loadToken) return;
    const stats = await viewport.load(glb);
    if (token !== loadToken) return;
    viewing = next;
    isolated = null;
    const [x, y, z] = stats.size;
    els.hudStats.textContent =
      `${stats.triangles.toLocaleString()} tris · ` +
      `${x.toFixed(2)} × ${y.toFixed(2)} × ${z.toFixed(2)} · ${next.id}`;
    // Wireframe is per-material, so a freshly loaded mesh has to be re-flagged.
    viewport.setWireframe(els.wireframe.classList.contains("on"));
    renderLoadedParts(stats.parts);
    els.export.disabled = false;
  } catch (e) {
    if (token !== loadToken) return;
    els.hudStats.textContent = "";
    showError(`Could not load ${next.kind} ${next.id}: ${errText(e)}`);
  }
}

/* ---------- part isolation ---------- */

function renderLoadedParts(parts: PartInfo[]) {
  // A single generated part has nothing to isolate from; only a scene does.
  els.loadedPanel.hidden = parts.length < 2;
  els.loadedParts.replaceChildren(
    ...parts.map((part) => {
      const li = document.createElement("li");
      li.className = "loaded-part";
      const name = document.createElement("span");
      name.textContent = part.name;
      const tris = document.createElement("span");
      tris.className = "detail";
      tris.textContent = `${part.triangles.toLocaleString()} tris`;
      li.append(name, tris);
      li.addEventListener("mouseenter", () => viewport.highlight(part.name));
      li.addEventListener("mouseleave", () => viewport.highlight(isolated));
      li.addEventListener("click", () => {
        isolated = isolated === part.name ? null : part.name;
        viewport.isolate(isolated);
        viewport.highlight(isolated);
        markIsolated();
      });
      return li;
    }),
  );
  markIsolated();
}

function markIsolated() {
  for (const li of els.loadedParts.children) {
    li.classList.toggle("on", li.firstElementChild?.textContent === isolated);
  }
  els.showAll.classList.toggle("on", isolated !== null);
}

els.showAll.addEventListener("click", () => {
  isolated = null;
  viewport.isolate(null);
  viewport.highlight(null);
  markIsolated();
});

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

/* ---------- scene builder ---------- */

function saveDraft() {
  localStorage.setItem(DRAFT_KEY, JSON.stringify({ name: els.sceneName.value, parts: draft }));
}

function loadDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw) as { name?: string; parts?: DraftPart[] };
    els.sceneName.value = saved.name ?? "";
    draft = saved.parts ?? [];
    nextKey = Math.max(0, ...draft.map((p) => p.key)) + 1;
  } catch {
    // A scene in progress is cheap to rebuild; a crash loop on start is not.
    localStorage.removeItem(DRAFT_KEY);
  }
}

/** Part names become node names in the glTF, and duplicates would collide. */
function uniqueName(base: string): string {
  const clean = base.replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "part";
  if (!draft.some((p) => p.name === clean)) return clean;
  for (let n = 2; ; n++) if (!draft.some((p) => p.name === `${clean}_${n}`)) return `${clean}_${n}`;
}

function addDraftPart(job: api.Job) {
  draft.push({
    key: nextKey++,
    jobId: job.id,
    name: uniqueName((job.params.part_name as string) || job.id),
    position: [0, 0, 0],
    rotation: [0, 0, 0],
    scale: 1,
    useRaw: false,
  });
  renderDraft();
  saveDraft();
  void measure(job.id);
}

/**
 * Real bounds beat the job record's face count when placing a part by hand.
 * The result patches the affected rows in place — a full re-render would pull
 * focus out of the name field the user is most likely typing into right now.
 */
async function measure(id: string) {
  if (measured.has(id)) return;
  try {
    measured.set(id, await api.describeJob(id));
    for (const part of draft) {
      if (part.jobId === id) {
        const el = metaEls.get(part.key);
        if (el) el.textContent = draftMeta(part);
      }
    }
  } catch {
    // Placement still works without it; the row just shows the face count.
  }
}

function numberInput(value: number, step: string, onChange: (n: number) => void) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = step;
  input.value = String(value);
  input.addEventListener("input", () => {
    const n = Number.parseFloat(input.value);
    onChange(Number.isFinite(n) ? n : 0);
    saveDraft();
  });
  return input;
}

function vectorRow(label: string, vec: api.Vec3, step: string) {
  const row = document.createElement("div");
  row.className = "vec";
  const tag = document.createElement("span");
  tag.textContent = label;
  row.append(tag, ...([0, 1, 2] as const).map((i) => numberInput(vec[i], step, (n) => (vec[i] = n))));
  return row;
}

const metaEls = new Map<number, HTMLElement>();

function renderDraft() {
  metaEls.clear();
  els.draftEmpty.hidden = draft.length > 0;
  els.draftCount.textContent = draft.length ? `${draft.length}` : "";
  els.assemble.disabled = draft.length === 0;

  els.draftList.replaceChildren(
    ...draft.map((part) => {
      const card = document.createElement("div");
      card.className = "draft";

      const head = document.createElement("div");
      head.className = "draft-head";
      const name = document.createElement("input");
      name.type = "text";
      name.spellcheck = false;
      name.value = part.name;
      name.addEventListener("input", () => {
        part.name = name.value;
        saveDraft();
      });
      const remove = document.createElement("button");
      remove.className = "ghost";
      remove.type = "button";
      remove.textContent = "×";
      remove.title = "Remove";
      remove.addEventListener("click", () => {
        draft = draft.filter((p) => p.key !== part.key);
        renderDraft();
        saveDraft();
      });
      head.append(name, remove);

      const src = document.createElement("p");
      src.className = "detail";
      src.textContent = draftMeta(part);
      metaEls.set(part.key, src);

      const raw = document.createElement("label");
      raw.className = "check";
      const rawBox = document.createElement("input");
      rawBox.type = "checkbox";
      rawBox.checked = part.useRaw;
      rawBox.addEventListener("change", () => {
        part.useRaw = rawBox.checked;
        saveDraft();
      });
      raw.append(rawBox, document.createTextNode("raw mesh"));

      const scale = document.createElement("div");
      scale.className = "vec";
      const scaleTag = document.createElement("span");
      scaleTag.textContent = "scale";
      scale.append(
        scaleTag,
        numberInput(part.scale, "0.05", (n) => (part.scale = n || 1)),
        raw,
      );

      card.append(
        head,
        src,
        vectorRow("pos", part.position, "0.05"),
        vectorRow("rot°", part.rotation, "5"),
        scale,
      );
      return card;
    }),
  );
}

function draftMeta(part: DraftPart): string {
  const job = jobs.find((j) => j.id === part.jobId);
  const faces = part.useRaw
    ? (job?.result?.decimated_from ?? job?.result?.faces)
    : job?.result?.faces;
  const d = measured.get(part.jobId);
  const size = d ? ` · ${d.size.map((n) => n.toFixed(2)).join(" × ")}` : "";
  return `${part.jobId}${faces ? ` · ${faces.toLocaleString()} faces` : ""}${size}`;
}

els.sceneName.addEventListener("input", saveDraft);

els.clearScene.addEventListener("click", () => {
  draft = [];
  els.sceneName.value = "";
  els.assembleStatus.textContent = "";
  renderDraft();
  saveDraft();
});

els.assemble.addEventListener("click", () => void doAssemble());

const nonZero = (v: api.Vec3) => (v.some((n) => n !== 0) ? v : undefined);

async function doAssemble() {
  if (!draft.length) return;
  els.assemble.disabled = true;
  els.assembleStatus.className = "detail";
  els.assembleStatus.textContent = "assembling…";
  try {
    const scene = await api.assemble(
      draft.map((p) => ({
        job_id: p.jobId,
        name: p.name.trim() || p.jobId,
        position: nonZero(p.position),
        rotation: nonZero(p.rotation),
        scale: p.scale === 1 ? undefined : p.scale,
        use_raw: p.useRaw || undefined,
      })),
      els.sceneName.value.trim() || undefined,
    );
    const [x, y, z] = scene.size;
    els.assembleStatus.textContent =
      `${scene.scene_id} · ${scene.part_count} parts · ` +
      `${scene.total_faces.toLocaleString()} faces · ` +
      `${x.toFixed(2)} × ${y.toFixed(2)} × ${z.toFixed(2)}`;
    clearExport();
    await showScene(scene);
  } catch (e) {
    els.assembleStatus.className = "detail error";
    els.assembleStatus.textContent = errText(e);
  } finally {
    els.assemble.disabled = draft.length === 0;
  }
}

/* ---------- export ---------- */

function clearExport() {
  els.exportStatus.textContent = "";
  els.exportStatus.className = "detail";
  els.exportWarnings.hidden = true;
  els.exportWarnings.replaceChildren();
  els.exportFiles.hidden = true;
  els.exportFiles.replaceChildren();
}

els.export.addEventListener("click", () => void doExport());

async function doExport() {
  if (!viewing) return;
  els.export.disabled = true;
  clearExport();
  els.exportStatus.textContent = `exporting ${viewing.label}…`;

  const height = Number.parseFloat(els.heightStuds.value);
  try {
    const res = await api.exportScene({
      [viewing.kind === "scene" ? "scene_id" : "job_id"]: viewing.id,
      target: els.exportTarget.value as "roblox" | "dcc",
      height_studs: Number.isFinite(height) ? height : undefined,
    });
    const [x, y, z] = res.size;
    els.exportStatus.textContent =
      `${res.part_count} parts · ${res.total_faces.toLocaleString()} tris` +
      `${res.source_faces !== res.total_faces ? ` (from ${res.source_faces.toLocaleString()})` : ""} · ` +
      `${x.toFixed(2)} × ${y.toFixed(2)} × ${z.toFixed(2)} studs · pivot ${res.pivot}`;
    renderWarnings(res.warnings);
    renderExportFiles(res);
  } catch (e) {
    els.exportStatus.className = "detail error";
    els.exportStatus.textContent = errText(e);
  } finally {
    els.export.disabled = false;
  }
}

/**
 * Warnings are the whole reason the export result is worth reading — they are
 * how the user finds out the mesh was decimated to fit or will import grey.
 */
function renderWarnings(warnings: string[]) {
  if (!warnings.length) return;
  const title = document.createElement("strong");
  title.textContent = `${warnings.length} warning${warnings.length > 1 ? "s" : ""}`;
  const list = document.createElement("ul");
  list.append(
    ...warnings.map((w) => {
      const li = document.createElement("li");
      li.textContent = w;
      return li;
    }),
  );
  els.exportWarnings.replaceChildren(title, list);
  els.exportWarnings.hidden = false;
  // The export panel sits at the bottom of a column that can overflow; a
  // warning the user has to scroll to find is a warning they will not read.
  els.exportWarnings.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

const basename = (p: string) => p.split(/[\\/]/).pop() ?? p;

/** Flattens `files` — `obj_sidecars` is a list, the rest are single paths. */
function exportPaths(res: api.ExportResult): string[] {
  return Object.values(res.files).flatMap((v) => (Array.isArray(v) ? v : [v]));
}

function renderExportFiles(res: api.ExportResult) {
  els.exportFiles.replaceChildren(
    ...exportPaths(res).map((path) => {
      const li = document.createElement("li");
      li.className = "export-file";
      const name = document.createElement("span");
      name.textContent = basename(path);
      if (path === res.primary) name.className = "primary";
      const bytes = res.file_bytes[basename(path).split(".").pop() ?? ""];
      const size = document.createElement("span");
      size.className = "detail";
      size.textContent = bytes ? `${(bytes / 1024).toFixed(0)} KB` : "";
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.type = "button";
      btn.textContent = "Save";
      btn.addEventListener("click", () => void saveExported(path, btn));
      li.append(name, size, btn);
      return li;
    }),
  );
  els.exportFiles.hidden = false;
}

async function saveExported(path: string, btn: HTMLButtonElement) {
  const name = basename(path);
  const ext = name.split(".").pop();
  const dest = await save({
    defaultPath: name,
    filters: ext ? [{ name: ext.toUpperCase(), extensions: [ext] }] : undefined,
  });
  if (!dest) return;
  btn.disabled = true;
  btn.textContent = "…";
  try {
    // The server holds the file; Rust streams it to disk without it ever
    // passing through the webview.
    const written = await api.downloadExportedFile(path, dest);
    els.exportStatus.className = "detail";
    els.exportStatus.textContent = `saved ${written}`;
    btn.textContent = "Saved";
  } catch (e) {
    els.exportStatus.className = "detail error";
    els.exportStatus.textContent = errText(e);
    btn.textContent = "Save";
  } finally {
    btn.disabled = false;
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
  loadDraft();
  await Promise.all([pollHealth(), refreshJobs()]);
  // Face counts on the draft rows come from the job list, so draw it after.
  renderDraft();
  draft.forEach((p) => void measure(p.jobId));
  // Opening onto the last finished part beats opening onto an empty void.
  const latest = jobs.find((j) => j.status === "done");
  if (latest) await selectJob(latest.id);
}

void start();
