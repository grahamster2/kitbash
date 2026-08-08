/**
 * The Kitbash desktop app.
 *
 * The shape of this file follows the pipeline it drives, and the order matters:
 *
 *   say what you want  ->  POST /strategy   ->  see the plan  ->  build it
 *
 * `/strategy` is the decision layer. It reads a plain subject and loose prose
 * about what the thing is *for*, and answers with one of three strategies, the
 * measured evidence for it, what it will cost before a second of GPU is spent,
 * and a draft plan in `/decompose` format that validates and can be run
 * unchanged. There is no LLM behind it and no GPU: it answers in about a
 * millisecond.
 *
 * That is why this app asks two questions and offers no technical knobs. The
 * triangle budget is not a setting here — 20,000 is Roblox's per-MeshPart
 * import cap and nothing else's, and `/strategy` derives it from the intent.
 * Nobody asking for a guard tower knows how many triangles a guard tower
 * should be, and being asked is the tell that the decision layer was skipped.
 */
import { save } from "@tauri-apps/plugin-dialog";

import * as api from "./api";
import { PartInfo, PartModes, Viewport } from "./viewport";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const els = {
  baseUrl: $<HTMLInputElement>("base-url"),
  serverStatus: $<HTMLSpanElement>("server-status"),
  serverDetail: $<HTMLParagraphElement>("server-detail"),
  subject: $<HTMLTextAreaElement>("subject"),
  intent: $<HTMLTextAreaElement>("intent"),
  plan: $<HTMLButtonElement>("plan"),
  planStatus: $<HTMLParagraphElement>("plan-status"),
  planPanel: $<HTMLElement>("plan-panel"),
  planHeadline: $<HTMLParagraphElement>("plan-headline"),
  planSub: $<HTMLParagraphElement>("plan-sub"),
  planWhy: $<HTMLDivElement>("plan-why"),
  planWarnings: $<HTMLDivElement>("plan-warnings"),
  planParts: $<HTMLUListElement>("plan-parts"),
  replan: $<HTMLButtonElement>("replan"),
  build: $<HTMLButtonElement>("build"),
  buildStatus: $<HTMLParagraphElement>("build-status"),
  refPanel: $<HTMLElement>("ref-panel"),
  prompt: $<HTMLTextAreaElement>("prompt"),
  generateIdeas: $<HTMLButtonElement>("generate-ideas"),
  ideasHead: $<HTMLDivElement>("ideas-head"),
  ideasStatus: $<HTMLParagraphElement>("ideas-status"),
  regenerate: $<HTMLButtonElement>("regenerate"),
  candidates: $<HTMLDivElement>("candidates"),
  imageInput: $<HTMLInputElement>("image-input"),
  imagePreview: $<HTMLImageElement>("image-preview"),
  dropLabel: $<HTMLSpanElement>("drop-label"),
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
/** The current recommendation, and the draft plan it came with. */
let rec: api.Recommendation | null = null;
let building = false;
/** How each part in view was built — the viewport needs it to light them. */
let partModes: PartModes = new Map();
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
  clearPlan();
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

/**
 * How a job's mesh should be lit, from what built it. A generated part carries
 * a back-projected photograph and must be shown unlit; a scripted one carries a
 * PBR material and has to be lit to read at all. See viewport.ts.
 */
const modeOfJob = (job?: api.Job): api.PartMode =>
  job?.type === "primitive" ? "script" : "generate";

async function selectJob(id: string) {
  const job = jobs.find((j) => j.id === id);
  selectedId = id;
  renderJobs();
  const label = job ? jobLabel(job) : id;
  await show({ kind: "job", id, label }, () => api.fetchMesh(id), new Map([[label, modeOfJob(job)]]));
}

async function showScene(scene: api.Scene, modes: PartModes, lightModes = modes) {
  selectedId = null;
  renderJobs();
  const label = scene.scene_path.split(/[\\/]/).pop() ?? scene.scene_id;
  await show(
    { kind: "scene", id: scene.scene_id, label },
    () => api.fetchScene(scene.scene_id),
    modes,
    lightModes,
  );
}

/**
 * `modes` is what each part *is* and goes on screen; `lightModes` is how to
 * light it. They differ on exactly one case: a mirrored part is its own thing
 * to the reader and its source's material to the renderer.
 */
async function show(
  next: NonNullable<typeof viewing>,
  fetch: () => Promise<ArrayBuffer>,
  modes: PartModes,
  lightModes: PartModes = modes,
) {
  const token = ++loadToken;
  showError(null);
  els.hudTitle.textContent = next.label;
  els.hudStats.textContent = "loading mesh…";

  try {
    const glb = await fetch();
    if (token !== loadToken) return;
    partModes = modes;
    const stats = await viewport.load(glb, lightModes);
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
      name.className = "lp-name";
      name.textContent = part.name;
      const tris = document.createElement("span");
      tris.className = "detail";
      tris.textContent = `${part.triangles.toLocaleString()} tris`;
      // Which parts cost a GPU minute and which cost a millisecond is the whole
      // argument of this project, and it belongs on screen next to the parts.
      const mode = partModes.get(part.name);
      li.append(...(mode ? [name, badge(mode), tris] : [name, tris]));
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

/* ---------- the plan ---------- */

const MODE_WORD: Record<api.PartMode, string> = {
  generate: "generated",
  script: "scripted",
  mirror: "mirrored",
};

function badge(mode: api.PartMode): HTMLSpanElement {
  const el = document.createElement("span");
  el.className = `badge ${mode}`;
  el.textContent = MODE_WORD[mode];
  el.title =
    mode === "generate"
      ? "a GPU generation from its own reference image — tens of seconds"
      : mode === "script"
        ? "built from primitives.py out of stated dimensions — about a millisecond, no GPU"
        : "the source part's mesh, reflected — free";
  return el;
}

function clearPlan() {
  rec = null;
  partModes = new Map();
  els.planPanel.hidden = true;
  els.refPanel.hidden = true;
  els.planStatus.textContent = "";
  els.planStatus.className = "detail";
  els.buildStatus.textContent = "";
  els.buildStatus.className = "detail";
}

function updatePlanButton() {
  els.plan.disabled = building || !els.subject.value.trim();
}

els.subject.addEventListener("input", updatePlanButton);
els.plan.addEventListener("click", () => void makePlan());
els.replan.addEventListener("click", () => void makePlan());
els.build.addEventListener("click", () => void build());

// Enter asks for the plan; Shift+Enter is still a newline.
for (const field of [els.subject, els.intent]) {
  field.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void makePlan();
    }
  });
}

/**
 * Ask the decision layer what this should be. Free, instant, and no GPU — so
 * this runs before anything is committed to, which is the entire point: the
 * criticism this answers is that a build took forty minutes and nobody saw the
 * price until it had been paid.
 */
async function makePlan() {
  const subject = els.subject.value.trim();
  if (!subject || building) return;
  els.plan.disabled = true;
  els.planStatus.className = "detail";
  els.planStatus.textContent = "thinking…";
  try {
    const next = await api.strategy({ subject, intent: els.intent.value.trim() || undefined });
    rec = next;
    els.planStatus.textContent = "";
    renderPlan(next);
  } catch (e) {
    els.planStatus.className = "detail error";
    els.planStatus.textContent = errText(e);
  } finally {
    updatePlanButton();
  }
}

/** "8 parts — 1 generated, 6 scripted, 1 mirrored", and never a zero. */
function partsSentence(c: api.Cost["parts"]): string {
  if (c.total === 1) return "One part — this is one sculptural whole";
  const bits = [
    c.generated ? `${c.generated} generated` : "",
    c.scripted ? `${c.scripted} scripted` : "",
    c.mirrored ? `${c.mirrored} mirrored` : "",
  ].filter(Boolean);
  return `${c.total} parts — ${bits.join(", ")}`;
}

function renderPlan(r: api.Recommendation) {
  els.planPanel.hidden = false;
  els.planHeadline.textContent = partsSentence(r.cost.parts);

  const budget = r.budget.target_assumed
    ? `${r.budget.target} assumed`
    : `for ${r.budget.target}`;
  els.planSub.textContent =
    `about ${r.cost.wall_human} · ${r.cost.triangles.total.toLocaleString()} triangles · ` +
    `${r.cost.estimated_size} · ${budget}`;

  // Why, in the recommender's own words. It cites a measurement for every
  // claim it makes, and those citations are the reason to trust the answer.
  const why: HTMLElement[] = [];
  for (const reason of r.reasoning.slice(0, 2)) {
    const p = document.createElement("p");
    p.className = "claim";
    p.textContent = reason.claim;
    p.title = `${reason.evidence}\n\n— ${reason.source}`;
    why.push(p);
  }
  // What the scripted and mirrored parts did NOT cost. The most persuasive
  // line the cost model produces, and it is already written in plain English.
  for (const saving of r.cost.savings) {
    const p = document.createElement("p");
    p.className = "detail saving";
    p.textContent = saving;
    why.push(p);
  }
  els.planWhy.replaceChildren(...why);
  els.planWhy.hidden = !why.length;

  renderCeilings(r.warnings);
  renderPlanParts(r.plan.parts);

  // `single` still needs a reference image, and choosing it is a judgement the
  // user makes better than any batch score would.
  const single = r.strategy === "single";
  els.refPanel.hidden = !single;
  if (single) {
    const drafted = r.plan.parts[0]?.prompt;
    // Only overwrite a prompt the user has not touched — a replan should not
    // silently discard their wording.
    if (drafted && (!els.prompt.value.trim() || els.prompt.dataset.drafted === "1")) {
      els.prompt.value = drafted;
      els.prompt.dataset.drafted = "1";
    }
    els.generateIdeas.disabled = ideasBusy || !els.prompt.value.trim();
  }
  updateBuild();
}

/**
 * The measured ceilings this plan is about to walk into. Every one cost real
 * GPU time to discover and every one is invisible until after the money is
 * spent, so they belong in front of the Build button rather than behind it.
 */
function renderCeilings(warnings: api.Ceiling[]) {
  const shown = warnings.filter((w) => w.severity !== "note");
  els.planWarnings.hidden = !shown.length;
  if (!shown.length) return;
  const title = document.createElement("strong");
  const blockers = shown.filter((w) => w.severity === "blocker").length;
  title.textContent = blockers
    ? `${blockers} known limit${blockers > 1 ? "s" : ""} this walks into`
    : `${shown.length} thing${shown.length > 1 ? "s" : ""} to know`;
  const list = document.createElement("ul");
  list.append(
    ...shown.map((w) => {
      const li = document.createElement("li");
      li.textContent = w.part ? `${w.part}: ${w.message}` : w.message;
      li.title = `${w.evidence}\n\n— ${w.source}`;
      return li;
    }),
  );
  els.planWarnings.replaceChildren(title, list);
}

function renderPlanParts(parts: api.PlanPart[]) {
  els.planParts.replaceChildren(
    ...parts.map((part) => {
      const li = document.createElement("li");
      li.className = "plan-part";
      li.id = `plan-part-${part.name}`;
      const name = document.createElement("span");
      name.className = "pp-name";
      name.textContent = part.name.replace(/_/g, " ");
      const status = document.createElement("span");
      status.className = "detail pp-status";
      status.textContent = part.kind ?? "";
      li.append(name, badge(part.mode), status);
      if (part.note) li.title = part.note;
      return li;
    }),
  );
}

function setPartStatus(name: string, text: string, cls = "") {
  const el = document.getElementById(`plan-part-${name}`)?.querySelector(".pp-status");
  if (!el) return;
  el.textContent = text;
  el.className = `detail pp-status ${cls}`;
}

function updateBuild() {
  const needsReference = rec?.strategy === "single" && reference === null;
  els.build.disabled = !rec || building || needsReference;
  els.build.textContent = needsReference ? "Pick a reference first" : "Build it";
}

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
  updateBuild();
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
  updateBuild();
  // The grid is tall enough to push the next step off the bottom of the column
  // on a short window; having picked, the user is looking for that button.
  els.build.scrollIntoView({ block: "nearest", behavior: "smooth" });
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

/* ---------- building ---------- */

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
    updateBuild();
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

function setBuildStatus(text: string, error = false) {
  els.buildStatus.className = error ? "detail error" : "detail";
  els.buildStatus.textContent = text;
}

/** Run the plan the user just looked at — one part or thirty, same button. */
async function build() {
  if (!rec || building) return;
  building = true;
  updateBuild();
  updatePlanButton();
  els.replan.disabled = true;
  try {
    if (rec.strategy === "single") await buildSingle(rec);
    else await buildMulti(rec);
  } catch (e) {
    setBuildStatus(errText(e), true);
  } finally {
    building = false;
    els.replan.disabled = false;
    updateBuild();
    updatePlanButton();
  }
}

/**
 * One sculptural whole, one generation. Not a fallback — for a skull it is the
 * right answer, and splitting one would invent seams that are not there.
 *
 * The budget comes off the plan, never off a field: the part's own
 * `target_faces` if the recommender set one, otherwise the plan's.
 */
async function buildSingle(r: api.Recommendation) {
  if (!reference) return;
  const part = r.plan.parts[0];
  const name =
    slug(part?.name ?? "") ??
    slug(r.plan.name ?? "") ??
    (reference.kind === "candidate" ? slug(reference.prompt) : undefined);
  setBuildStatus("submitting…");
  setPartStatus(part?.name ?? "", "submitting…");

  const job = await api.submitJob({
    // A picked candidate already lives on the server, so it goes by id — the
    // same reference can then drive several parts of one object.
    ...(reference.kind === "candidate"
      ? { image_id: reference.imageId }
      : { image_b64: reference.b64 }),
    part_name: name,
    seed: r.plan.seed,
    target_faces: part?.target_faces ?? r.plan.target_faces,
    generator: r.plan.generator,
    textured: r.plan.textured,
  });
  watching = job.id;
  setBuildStatus(`queued as ${job.id}`);
  await refreshJobs();
  await waitForJobs([job.id], (state) => {
    const j = state.get(job.id);
    if (!j) return;
    setPartStatus(part?.name ?? "", jobPhrase(j), j.status === "error" ? "error" : "");
    setBuildStatus(`${jobPhrase(j)}`, j.status === "error");
  });
  watching = null;
  const done = await api.getJob(job.id);
  if (done.status !== "done") throw new Error(done.error ?? "generation failed");
  setBuildStatus(`done in ${done.result?.generation_seconds.toFixed(0)}s`);
  await refreshJobs();
  await selectJob(job.id);
}

/**
 * Many parts. `/decompose` draws every reference image and builds every
 * scripted part before it answers, then hands back job ids and an `/assemble`
 * request already written — so the wait after it returns is GPU time on the
 * generated parts only, and the scripted ones are already finished.
 */
async function buildMulti(r: api.Recommendation) {
  setBuildStatus("drawing references and building the scripted parts…");
  const result = await api.decompose(r.plan);
  await refreshJobs();

  for (const p of result.parts) {
    setPartStatus(
      p.name,
      p.status === "error" ? (p.error ?? "failed") : p.status === "queued" ? "queued" : p.status,
      p.status === "error" ? "error" : "",
    );
  }

  const generated = result.parts.filter((p) => p.mode === "generate" && p.job_id);
  const ids = [...new Set(generated.map((p) => p.job_id!))];
  if (ids.length) {
    await waitForJobs(ids, (state) => {
      let finished = 0;
      for (const p of generated) {
        const j = state.get(p.job_id!);
        if (!j) continue;
        if (j.status === "done" || j.status === "error") finished++;
        setPartStatus(p.name, jobPhrase(j), j.status === "error" ? "error" : "");
      }
      setBuildStatus(`${finished}/${ids.length} generated part(s) finished`);
    });
  }

  // A part that failed takes its mesh out of the scene rather than the whole
  // build: seven good parts and a named failure is something you can finish by
  // rerolling one prompt.
  const state = await jobStates(ids);
  const dead = new Set(ids.filter((id) => state.get(id)?.status !== "done"));
  const parts = result.assemble_request.filter((p) => !dead.has(p.job_id));
  // Named, not counted. `result.failed` only knows about parts that fell over
  // while the plan was being submitted; a generation that failed afterwards
  // shows up here and nowhere else, and a mirror of it goes with it.
  const lost = [
    ...new Set([
      ...result.failed,
      ...result.assemble_request.filter((p) => dead.has(p.job_id)).map((p) => p.name),
    ]),
  ];
  if (!parts.length) throw new Error("every part failed; nothing to assemble");

  setBuildStatus(`assembling ${parts.length} part(s)…`);
  const built = result.parts.filter((p) => p.job_id && !dead.has(p.job_id));
  const declared: PartModes = new Map(built.map((p) => [p.name, p.mode]));
  const scene = await api.assemble(parts, r.plan.name ?? slug(r.subject));
  const [x, y, z] = scene.size;
  setBuildStatus(
    // Not metres: a plan with a `scale_reference` makes one unit worth that
    // part's real length, so the guard tower assembles 1.0 units across and is
    // eight metres wide. Export is where studs and metres get decided.
    `${scene.part_count} parts · ${scene.total_faces.toLocaleString()} tris · ` +
      `${x.toFixed(2)} × ${y.toFixed(2)} × ${z.toFixed(2)}` +
      (lost.length ? ` · ${lost.length} part(s) dropped: ${lost.join(", ")}` : ""),
    lost.length > 0,
  );
  clearExport();
  await showScene(scene, declared, modesFor(result));
}

/**
 * Which parts of a finished run were generated and which were scripted, keyed
 * by the node name they will carry in the scene. A mirror is neither: it is the
 * source part's mesh reflected, so it is lit however its source is.
 */
function modesFor(result: api.DecomposeResult): PartModes {
  const byJob = new Map<string, api.PartMode>();
  for (const p of result.parts) {
    if (p.job_id && p.mode !== "mirror") byJob.set(p.job_id, p.mode);
  }
  const modes: PartModes = new Map();
  for (const p of result.parts) {
    if (!p.job_id) continue;
    modes.set(p.name, p.mode === "mirror" ? (byJob.get(p.job_id) ?? "script") : p.mode);
  }
  return modes;
}

const jobStates = async (ids: string[]) =>
  new Map((await Promise.all(ids.map((id) => api.getJob(id)))).map((j) => [j.id, j]));

/**
 * Waits for a set of jobs, reporting after every sweep.
 *
 * The queue is single-worker, so eight generated parts are eight generations
 * end to end and the honest thing to do is say which one is building rather
 * than show one bar for all of them.
 */
async function waitForJobs(ids: string[], onTick: (state: Map<string, api.Job>) => void) {
  for (;;) {
    let state: Map<string, api.Job>;
    try {
      state = await jobStates(ids);
    } catch (e) {
      setBuildStatus(errText(e), true);
      await sleep(5000);
      continue;
    }
    onTick(state);
    if ([...state.values()].every((j) => j.status === "done" || j.status === "error")) return;
    await refreshJobs();
    await sleep(2500);
  }
}

/** What to say about one job while it is in flight. */
function jobPhrase(job: api.Job): string {
  if (job.status === "done") {
    const r = job.result;
    return r ? `${r.faces.toLocaleString()} tris · ${r.generation_seconds.toFixed(0)}s` : "done";
  }
  if (job.status === "error") return job.error ?? "failed";
  const since = job.started_at ?? job.created_at;
  const secs = Math.max(0, Math.round(Date.now() / 1000 - since));
  // Say what the wait is worth up front — a mesh is 30-60s of GPU time and a
  // silent counter at 25s reads like a hang. Past a minute the honest answer is
  // different: the first job of a session also loads the weights, measured at
  // ~85s wall against 40s of actual generation.
  const hint =
    job.status !== "running"
      ? ""
      : secs < 60
        ? " · usually 30–60s"
        : " · the first job also loads the model";
  return `${job.status} · ${secs}s${hint}`;
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
    // A hand-built scene has no plan behind it, so what built each part comes
    // off the job record instead — same answer, different source.
    await showScene(
      scene,
      new Map(
        draft.map((p) => [
          p.name.trim() || p.jobId,
          modeOfJob(jobs.find((j) => j.id === p.jobId)),
        ]),
      ),
    );
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
  // A build drives its own polling, so this only keeps the job list honest
  // about work started elsewhere — an agent through MCP, or another window.
  const busy = watching !== null || jobs.some(isActive);
  if (busy) {
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
