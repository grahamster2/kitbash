import { save } from "@tauri-apps/plugin-dialog";

import * as api from "./api";
import { PartInfo, Viewport } from "./viewport";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  baseUrl: $<HTMLInputElement>("base-url"),
  serverStatus: $<HTMLSpanElement>("server-status"),
  serverDetail: $<HTMLParagraphElement>("server-detail"),
  prompt: $<HTMLTextAreaElement>("prompt"),
  generateIdeas: $<HTMLButtonElement>("generate-ideas"),
  ideasHead: $<HTMLDivElement>("ideas-head"),
  ideasStatus: $<HTMLParagraphElement>("ideas-status"),
  regenerate: $<HTMLButtonElement>("regenerate"),
  candidates: $<HTMLDivElement>("candidates"),
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
const IDEA_COUNT = 4;
const DROP_LABEL = "…or drop a reference image";

/**
 * What the next mesh will be generated from. A dropped file travels as base64;
 * a picked candidate is already on the server and travels as an id, so the
 * bytes never make the round trip back up.
 */
type Reference =
  | { kind: "file"; b64: string }
  | { kind: "candidate"; imageId: string; prompt: string };

let jobs: api.Job[] = [];
let selectedId: string | null = null;
let reference: Reference | null = null;
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
// Same guard for image batches: a regenerate must not be overwritten by the
// thumbnails of the batch it replaced.
let batchToken = 0;
let batch: api.CandidateBatch | null = null;
let picked: string | null = null;
let ideasBusy = false;
const tiles = new Map<string, HTMLButtonElement>();
const thumbUrls = new Map<string, string>();

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
  // Job, scene and image ids belong to one server; nothing carries across a switch.
  viewing = null;
  measured.clear();
  batchToken++;
  clearCandidates();
  els.ideasHead.hidden = true;
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

/* ---------- reference candidates ---------- */

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function setIdeasStatus(text: string, error = false) {
  els.ideasHead.hidden = false;
  els.ideasStatus.className = error ? "detail error" : "detail";
  els.ideasStatus.textContent = text;
}

/**
 * A batch, from whichever endpoint this server has.
 *
 * `/images/candidates` is the batch endpoint; a server that predates it only
 * has the single-image POST. Four of those fanned out produce the same shape,
 * so the picker works against either build without the user having to know
 * which one is deployed — the fallback simply goes dormant once it is.
 */
async function fetchBatch(prompt: string, seed: number): Promise<api.CandidateBatch> {
  try {
    return await api.createCandidates({
      prompt,
      count: IDEA_COUNT,
      remove_background: true,
      seed,
    });
  } catch (e) {
    // 405, not just 404: on a server without the batch route, `/images/candidates`
    // still matches `GET /images/{image_id}`, so FastAPI rejects the method
    // rather than the path.
    if (!/\b40[45]\b/.test(errText(e))) throw e;
    const singles = await Promise.all(
      Array.from({ length: IDEA_COUNT }, (_, i) =>
        api.createImage({ prompt, seed: seed + i, remove_background: true }),
      ),
    );
    return {
      batch_id: `local-${seed}`,
      prompt,
      candidates: singles.map((s, i) => ({
        image_id: s.image_id,
        prompt,
        variant: null,
        seed: seed + i,
        bytes: s.bytes,
        path: s.path,
      })),
    };
  }
}

/**
 * The POST is specified to return the whole batch. A server that answered early
 * with a partial one would otherwise leave permanent placeholders on screen, so
 * re-read the batch record until it is full or the wait stops being plausible.
 */
async function topUp(first: api.CandidateBatch, token: number): Promise<api.CandidateBatch> {
  const deadline = Date.now() + 60_000;
  let cur = first;
  while (cur.candidates.length < IDEA_COUNT && Date.now() < deadline) {
    await sleep(1000);
    if (token !== batchToken) return cur;
    try {
      cur = await api.getBatch(cur.batch_id);
    } catch {
      return cur; // No batch record to poll; what arrived is what there is.
    }
    if (token !== batchToken) return cur;
    showBatch(cur, token);
  }
  return cur;
}

async function generateIdeas() {
  const prompt = els.prompt.value.trim();
  if (!prompt || ideasBusy) return;

  const token = ++batchToken;
  ideasBusy = true;
  els.generateIdeas.disabled = true;
  els.regenerate.disabled = true;
  clearCandidates();
  // Placeholders before the request, not after it: the wait is exactly when the
  // user needs to see that something is happening.
  showPlaceholders();

  const started = Date.now();
  const elapsed = () => (Date.now() - started) / 1000;
  setIdeasStatus(`generating ${IDEA_COUNT} ideas… 0s`);
  const tick = setInterval(
    () => setIdeasStatus(`generating ${IDEA_COUNT} ideas… ${elapsed().toFixed(0)}s`),
    250,
  );

  try {
    // An explicit random base seed rather than none, so a regenerate is
    // guaranteed to differ instead of depending on how the provider seeds.
    const seed = Math.floor(Math.random() * 2 ** 31);
    const first = await fetchBatch(prompt, seed);
    if (token !== batchToken) return;
    showBatch(first, token);
    const full = await topUp(first, token);
    if (token !== batchToken) return;
    setIdeasStatus(
      `${full.candidates.length} ideas in ${elapsed().toFixed(1)}s — pick one`,
    );
  } catch (e) {
    if (token !== batchToken) return;
    clearCandidates();
    setIdeasStatus(errText(e), true);
  } finally {
    clearInterval(tick);
    if (token === batchToken) {
      ideasBusy = false;
      els.generateIdeas.disabled = !els.prompt.value.trim();
      els.regenerate.disabled = false;
    }
  }
}

function showPlaceholders() {
  els.candidates.hidden = false;
  els.candidates.replaceChildren(
    ...Array.from({ length: IDEA_COUNT }, () => placeholder()),
  );
}

function placeholder(): HTMLDivElement {
  const div = document.createElement("div");
  div.className = "candidate loading";
  return div;
}

function showBatch(next: api.CandidateBatch, token: number) {
  batch = next;
  els.candidates.hidden = false;
  const cells: HTMLElement[] = next.candidates.map((c, i) => tileFor(c, i, token));
  while (cells.length < IDEA_COUNT) cells.push(placeholder());
  els.candidates.replaceChildren(...cells);
  markPicked();
}

/** Tiles are cached by image id so a top-up poll does not refetch a thumbnail. */
function tileFor(c: api.Candidate, index: number, token: number): HTMLButtonElement {
  const cached = tiles.get(c.image_id);
  if (cached) return cached;

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "candidate loading";
  btn.setAttribute("role", "radio");
  btn.setAttribute("aria-checked", "false");
  btn.setAttribute("aria-label", c.variant ?? `idea ${index + 1}`);
  btn.title = `${c.variant ? `${c.variant} · ` : ""}seed ${c.seed} · press ${index + 1}`;
  btn.tabIndex = index === 0 ? 0 : -1;
  if (c.variant) {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = c.variant;
    btn.append(tag);
  }
  btn.addEventListener("click", () => pick(c.image_id));
  tiles.set(c.image_id, btn);
  void loadThumb(c, btn, token);
  return btn;
}

async function loadThumb(c: api.Candidate, btn: HTMLButtonElement, token: number) {
  try {
    const png = await api.fetchImage(c.image_id);
    if (token !== batchToken) return;
    // A blob URL keeps a megapixel of base64 out of the DOM, and the app's CSP
    // already allows blob: for images.
    const url = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    thumbUrls.set(c.image_id, url);
    const img = document.createElement("img");
    img.alt = c.variant ?? c.prompt;
    img.src = url;
    // Prepend so the variant tag and tick stay painted over the image.
    btn.prepend(img);
    btn.classList.remove("loading");
  } catch (e) {
    if (token !== batchToken) return;
    btn.classList.remove("loading");
    btn.classList.add("failed");
    btn.textContent = "failed";
    btn.title = errText(e);
  }
}

function clearCandidates() {
  for (const url of thumbUrls.values()) URL.revokeObjectURL(url);
  thumbUrls.clear();
  tiles.clear();
  batch = null;
  picked = null;
  els.candidates.replaceChildren();
  els.candidates.hidden = true;
  els.candidates.classList.remove("picked");
  // A stale pick must not survive into the next batch.
  if (reference?.kind === "candidate") reference = null;
  updateGenerate();
}

function pick(imageId: string) {
  const c = batch?.candidates.find((x) => x.image_id === imageId);
  if (!c) return;
  picked = imageId;
  reference = { kind: "candidate", imageId, prompt: c.prompt };
  // One reference at a time — picking replaces whatever file was dropped.
  els.imagePreview.hidden = true;
  els.imagePreview.removeAttribute("src");
  els.dropLabel.textContent = DROP_LABEL;
  markPicked();
  updateGenerate();
  // The grid is tall enough to push the next step off the bottom of the column
  // on a short window; having picked, the user is looking for that button.
  els.generate.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function markPicked() {
  els.candidates.classList.toggle("picked", picked !== null);
  let index = 0;
  for (const [id, btn] of tiles) {
    const on = id === picked;
    btn.classList.toggle("on", on);
    btn.setAttribute("aria-checked", String(on));
    // Roving tabindex: one stop for the whole group, landing on the pick.
    btn.tabIndex = on || (picked === null && index === 0) ? 0 : -1;
    btn.querySelector(".tick")?.remove();
    if (on) {
      const tick = document.createElement("span");
      tick.className = "tick";
      tick.textContent = "✓";
      btn.append(tick);
    }
    index++;
  }
}

const ARROW_STEP: Record<string, number> = {
  ArrowRight: 1,
  ArrowLeft: -1,
  ArrowDown: 2,
  ArrowUp: -2,
};

els.candidates.addEventListener("keydown", (e) => {
  const step = ARROW_STEP[e.key];
  if (step === undefined) return;
  const ids = batch?.candidates.map((c) => c.image_id) ?? [];
  if (!ids.length) return;
  e.preventDefault();
  const from = picked ? ids.indexOf(picked) : 0;
  focusPick(ids[Math.min(ids.length - 1, Math.max(0, from + step))]);
});

// 1-4 picks without reaching for the mouse, but only when the keystroke is not
// already meant for a field.
document.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLElement && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  const n = Number(e.key);
  if (!Number.isInteger(n) || n < 1 || n > IDEA_COUNT) return;
  const c = batch?.candidates[n - 1];
  if (c) focusPick(c.image_id);
});

function focusPick(imageId: string) {
  pick(imageId);
  tiles.get(imageId)?.focus();
}

els.prompt.addEventListener("input", () => {
  els.generateIdeas.disabled = ideasBusy || !els.prompt.value.trim();
});

// Enter submits the prompt; Shift+Enter is still a newline.
els.prompt.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    void generateIdeas();
  }
});

els.generateIdeas.addEventListener("click", () => void generateIdeas());
els.regenerate.addEventListener("click", () => void generateIdeas());

/* ---------- submission ---------- */

function updateGenerate() {
  els.generate.disabled = reference === null || watching !== null;
}

function useImage(file: File) {
  const reader = new FileReader();
  reader.onload = () => {
    const dataUrl = reader.result as string;
    reference = { kind: "file", b64: dataUrl.slice(dataUrl.indexOf(",") + 1) };
    els.imagePreview.src = dataUrl;
    els.imagePreview.hidden = false;
    els.dropLabel.textContent = file.name;
    // A dropped file and a picked candidate are the same slot.
    picked = null;
    markPicked();
    updateGenerate();
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

/** Part names become glTF node names, so a prompt has to survive as an identifier. */
function slug(text: string): string | undefined {
  return (
    text
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 40) || undefined
  );
}

async function submit() {
  if (!reference) return;
  els.generate.disabled = true;
  els.submitStatus.className = "detail";
  els.submitStatus.textContent = "submitting…";

  const seed = Number.parseInt(els.seed.value, 10);
  const faces = Number.parseInt(els.targetFaces.value, 10);
  try {
    const job = await api.submitJob({
      // A picked candidate already lives on the server, so it goes by id — the
      // same reference can then drive several parts of one object.
      ...(reference.kind === "candidate"
        ? { image_id: reference.imageId }
        : { image_b64: reference.b64 }),
      part_name:
        els.partName.value.trim() ||
        (reference.kind === "candidate" ? slug(reference.prompt) : undefined),
      seed: Number.isFinite(seed) ? seed : undefined,
      target_faces: Number.isFinite(faces) ? faces : undefined,
    });
    watching = job.id;
    els.submitStatus.textContent = `queued as ${job.id}`;
    await refreshJobs();
  } catch (e) {
    els.submitStatus.className = "detail error";
    els.submitStatus.textContent = errText(e);
    updateGenerate();
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
      updateGenerate();
      await refreshJobs();
      await selectJob(id);
    } else if (job.status === "error") {
      els.submitStatus.className = "detail error";
      els.submitStatus.textContent = job.error ?? "generation failed";
      watching = null;
      updateGenerate();
    } else {
      const since = job.started_at ?? job.created_at;
      const secs = Math.max(0, Math.round(Date.now() / 1000 - since));
      // Say what the wait is worth up front — a mesh is 30-60s of GPU time and
      // a silent counter at 25s reads like a hang. Past a minute the honest
      // answer is different: the first job of a session also loads the weights,
      // measured at ~85s wall against 40s of actual generation.
      const hint =
        job.status !== "running"
          ? ""
          : secs < 60
            ? " · usually 30–60s"
            : " · the first job also loads the model";
      els.submitStatus.textContent = `${job.id} ${job.status} · ${secs}s${hint}`;
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
