"use strict";

// VolleyIQ front end: upload a video, follow the job, review the result.
// Everything shown comes from the local server's API. User-provided text is
// always inserted with textContent, never as HTML.

const $ = (id) => document.getElementById(id);
const POLL_MS = 1500;
const HEALTH_MS = 20000;
const SECTIONS = ["analyze", "how", "recent"];
const VIEWS = ["home", "processing", "results", "failed", "stopped"];
const STAGES = ["extracting", "loading_model", "detecting", "tracking", "rendering"];
const STAGE_TEXT = {
  starting: "Starting…",
  extracting: "Extracting frames",
  loading_model: "Loading the AI model",
  detecting: "Detecting the ball",
  tracking: "Tracking the ball",
  rendering: "Rendering the annotated video",
};
const STAGE_LABEL = {
  extracting: "Extract", loading_model: "Load model", detecting: "Detect", tracking: "Track", rendering: "Render",
};
const STAGE_COLOUR = {
  extracting: "#3b82f6", loading_model: "#8b5cf6", detecting: "#c5f135", tracking: "#22d3ee", rendering: "#f59e0b",
};
const STATUS_INFO = {
  good: ["Good", "Position confirmed by smooth movement over at least 3 frames."],
  uncertain: ["Uncertain", "A position was chosen but not confirmed. Check it before relying on it."],
  lost: ["Lost", "No ball found in this frame (or only a static look-alike)."],
  bad: ["Bad", "Detections were rejected as impossible jumps, usually false positives."],
};
const STATUS_ORDER = ["good", "uncertain", "lost", "bad"];
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

let config = {
  max_upload_bytes: 500 * 1024 * 1024, allowed_extensions: [".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"],
  max_duration_s: 300, samples_per_second: 6, retention_hours: 24, max_queued: 3,
};
let selectedFile = null;
let previewUrl = null; // object URL of the chosen file, for the preview (revoked when it changes)
let uploadRequest = null; // the XMLHttpRequest while an upload is running
let pollTimer = null;
let toastTimer = null;
let elapsedTimer = null;
let healthTimer = null;
let stopped = false;
let olderServer = false; // a VolleyIQ server started before this page's features existed
let results = null; // {job, frames, fps, index}
let currentJob = null; // the job shown in the processing, results or failed view

// ---------- helpers ----------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function formatBytes(bytes) {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 ** 2) {
    const mb = bytes / 1024 ** 2;
    return `${Number.isInteger(mb) ? mb : mb.toFixed(1)} MB`;
  }
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "–";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function formatClock(seconds) {
  const s = Math.max(0, seconds);
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

function formatDate(epochSeconds) {
  return new Date(epochSeconds * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function setServerStatus(state, text) {
  const box = $("server-status");
  box.className = `server-status ${state}`;
  $("server-status-text").textContent = text;
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { cache: "no-store", ...options });
  } catch {
    if (!stopped) setServerStatus("offline", "Not reachable");
    const error = new Error("VolleyIQ is not reachable. Is it still running?");
    error.network = true;
    throw error;
  }
  if (!stopped) setServerStatus("online", olderServer ? "Running · restart to update" : "Running locally");
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    const error = new Error((data && data.error) || `Request failed (${response.status}).`);
    error.status = response.status;
    throw error;
  }
  return data;
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { box.hidden = true; }, 4500);
}

// Resolves to true only if the user confirms in the dialog.
function confirmAction(title, text, confirmLabel) {
  const dialog = $("confirm-dialog");
  if (typeof dialog.showModal !== "function") return Promise.resolve(window.confirm(`${title}\n\n${text}`));
  $("confirm-title").textContent = title;
  $("confirm-text").textContent = text;
  $("confirm-ok").textContent = confirmLabel;
  dialog.returnValue = "";
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true });
    dialog.showModal();
  });
}

function releasePlayer() {
  // Stop streaming the result video first: on Windows a file being read can't be deleted.
  const player = $("player");
  player.pause();
  player.removeAttribute("src");
  player.load();
}

// Deletes one job (cancelling it if it is running). Returns true if the server deleted it.
async function deleteJob(job, { running = false } = {}) {
  const name = job.filename || "this video";
  const confirmed = running
    ? await confirmAction("Cancel this analysis?",
      `The analysis of "${name}" stops and its uploaded video and partial results are deleted from this computer.`,
      "Cancel analysis")
    : await confirmAction("Delete this analysis?",
      `"${name}", its annotated video and its tracking CSV are permanently deleted from this computer.`, "Delete");
  if (!confirmed) return false;
  if (results && results.job.id === job.id) releasePlayer();
  try {
    const answer = await api(`/api/jobs/${job.id}`, { method: "DELETE" });
    toast(answer.result === "cancelling" ? `Cancelled "${name}". Its files are being deleted.` : `Deleted "${name}".`);
    return true;
  } catch (error) {
    if (error.status === 404) {
      toast(`"${name}" was already deleted.`);
      return true;
    }
    toast(`Couldn't delete "${name}": ${error.message}`);
    return false;
  }
}

async function clearHistory() {
  const confirmed = await confirmAction("Clear the whole history?",
    "Every analysis stored by VolleyIQ is permanently deleted from this computer, with its annotated video and " +
    "CSV. An analysis that is still running is cancelled.", "Clear history");
  if (!confirmed) return;
  try {
    const { deleted, cancelling } = await api("/api/jobs", { method: "DELETE" });
    const total = deleted + cancelling;
    toast(total ? `Deleted ${total} analys${total === 1 ? "is" : "es"}.` : "The history was already empty.");
  } catch (error) {
    toast(`Couldn't clear the history: ${error.message}`);
  }
  loadRecent();
}

function showView(name) {
  for (const view of VIEWS) $(`view-${view}`).hidden = view !== name;
  document.body.dataset.view = name;
}

function scrollToId(id) {
  const target = $(id);
  if (!target) return;
  target.scrollIntoView({ behavior: reducedMotion.matches ? "auto" : "smooth", block: "start" });
}

function applyFacts() {
  for (const node of document.querySelectorAll('[data-fact="samples"]')) node.textContent = String(config.samples_per_second);
  for (const node of document.querySelectorAll('[data-fact="retention"]')) node.textContent = String(config.retention_hours);
  const types = config.allowed_extensions.map((e) => e.slice(1).toUpperCase()).join(", ");
  const minutes = Math.round(config.max_duration_s / 60);
  $("limits-hint").textContent = `${types} · up to ${formatBytes(config.max_upload_bytes)} · up to ${minutes} minutes`;
  $("limits-text").textContent = `Supported: ${types}. Up to ${formatBytes(config.max_upload_bytes)} and ${minutes} minutes per clip.`;
  if (config.model) $("footer-model").textContent = config.model;
  if ("gpu" in config) {
    $("dash-device").textContent = config.gpu ? `GPU · ${config.gpu}` : "CPU (no CUDA GPU found)";
    $("analyze-note").textContent = config.gpu
      ? `The AI runs on this computer's ${config.gpu}. The page updates by itself.`
      : "The AI runs on this computer's CPU, which is slower than a GPU. The page updates by itself.";
  }
}

// ---------- upload -----------------------------------------------------

function showUploadError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = !message;
}

function chooseFile(file) {
  if (uploadRequest) return; // never swap the file during an upload
  showUploadError("");
  if (!file) return;
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!config.allowed_extensions.includes(ext)) {
    showUploadError(`Unsupported file type ${ext || "(none)"}. Use ${config.allowed_extensions.join(", ")}.`);
    return;
  }
  if (file.size === 0) {
    showUploadError("This file is empty.");
    return;
  }
  if (file.size > config.max_upload_bytes) {
    showUploadError(`This file is ${formatBytes(file.size)}. The limit is ${formatBytes(config.max_upload_bytes)}.`);
    return;
  }
  selectedFile = file;
  $("file-name").textContent = file.name;
  $("file-size").textContent = formatBytes(file.size);
  $("file-duration").textContent = "Reading…";
  $("file-resolution").textContent = "Reading…";
  $("selected").hidden = false;
  $("analyze-button").disabled = false;
  showPreview(file);
}

// Shows the chosen video in the page (it never leaves the browser for this) and checks its length.
function showPreview(file) {
  const preview = $("preview");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $("preview-note").hidden = true;
  preview.hidden = false;
  preview.onloadedmetadata = () => {
    if (selectedFile !== file) return;
    const duration = preview.duration;
    $("file-duration").textContent = isFinite(duration) ? formatDuration(duration) : "Checked on upload";
    $("file-resolution").textContent = preview.videoWidth ? `${preview.videoWidth} × ${preview.videoHeight}` : "Checked on upload";
    if (isFinite(duration) && duration > config.max_duration_s + 0.5) {
      showUploadError(`This video is ${formatDuration(duration)} long. The limit is ` +
        `${formatDuration(config.max_duration_s)}: trim it first.`);
      $("analyze-button").disabled = true;
    }
  };
  preview.onerror = () => {
    if (selectedFile !== file) return;
    // Some formats (e.g. AVI, MKV) can't play in a browser but can still be analysed; the server checks them.
    preview.hidden = true;
    $("preview-note").hidden = false;
    $("file-duration").textContent = "Checked on upload";
    $("file-resolution").textContent = "Checked on upload";
  };
  preview.src = previewUrl;
}

function clearFile() {
  selectedFile = null;
  $("file-input").value = "";
  $("selected").hidden = true;
  $("analyze-button").disabled = true;
  $("upload-progress").hidden = true;
  const preview = $("preview");
  preview.onloadedmetadata = preview.onerror = null;
  preview.removeAttribute("src");
  preview.load();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
}

function setUploading(active) {
  for (const id of ["analyze-button", "clear-file", "replace-file", "file-input"]) $(id).disabled = active;
  $("dropzone").classList.toggle("disabled", active);
  $("upload-progress").hidden = !active;
}

function setUploadProgress(percent, text) {
  $("upload-fill").style.width = `${percent}%`;
  $("upload-bar").setAttribute("aria-valuenow", String(percent));
  $("upload-text").textContent = text;
}

function upload() {
  if (!selectedFile || uploadRequest) return; // one upload at a time: no accidental duplicates
  showUploadError("");
  setUploading(true);
  setUploadProgress(0, "Uploading… 0%");

  const xhr = new XMLHttpRequest();
  uploadRequest = xhr;
  xhr.open("POST", `/api/jobs?filename=${encodeURIComponent(selectedFile.name)}`);
  xhr.setRequestHeader("Content-Type", "application/octet-stream");
  xhr.upload.onprogress = (event) => {
    if (!event.lengthComputable) return;
    const percent = Math.round((100 * event.loaded) / event.total);
    setUploadProgress(percent, percent < 100 ? `Uploading… ${percent}%` : "Checking the video…");
  };
  const finish = (message) => {
    uploadRequest = null;
    setUploading(false);
    showUploadError(message);
  };
  xhr.onload = () => {
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { /* not JSON */ }
    if (xhr.status === 202 && data && data.id) {
      finish("");
      clearFile();
      location.hash = `#job/${data.id}`;
    } else {
      finish((data && data.error) || `Upload failed (${xhr.status}).`);
    }
  };
  xhr.onerror = () => finish("The upload failed. Is VolleyIQ still running?");
  // The server deletes a partly received upload straight away.
  xhr.onabort = () => { finish(""); toast("Upload cancelled. Nothing was kept."); };
  xhr.send(selectedFile);
}

// ---------- recent analyses ---------------------------------------------

function recentItem(job) {
  const link = el("a", "recent-item");
  link.href = `#job/${job.id}`;
  const badge = el("span", `badge ${job.status}`, job.status);
  const text = el("div");
  text.style.minWidth = "0";
  const parts = [formatDate(job.created_at)];
  if (job.video) parts.push(formatDuration(job.video.duration_s), `${job.video.width}×${job.video.height}`);
  text.append(el("div", "recent-name", job.filename), el("div", "recent-sub", parts.join(" · ")));
  const metric = el("div", "recent-metric");
  if (job.status === "completed" && job.stats) {
    metric.append(el("strong", null, `${job.stats.ball_found_percent}%`), el("span", null, "ball located"));
  } else if (job.status === "failed") {
    metric.append(el("span", null, "see error"));
  } else {
    metric.append(el("span", null, "in progress"));
  }
  link.append(badge, text, metric);
  const running = job.status === "queued" || job.status === "processing";
  const remove = el("button", "icon-button");
  remove.type = "button";
  remove.title = running ? "Cancel and delete" : "Delete";
  remove.setAttribute("aria-label", `${running ? "Cancel and delete" : "Delete"} ${job.filename}`);
  remove.append(trashIcon());
  remove.addEventListener("click", async () => {
    remove.disabled = true;
    if (await deleteJob(job, { running })) loadRecent();
    else remove.disabled = false;
  });
  const item = el("li", "recent-row");
  item.append(link, remove);
  return item;
}

function trashIcon() {
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS(ns, "path");
  path.setAttribute("d", "M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3");
  svg.append(path);
  return svg;
}

// Dashboard totals, computed only from the analyses the server reports.
function renderDashboard(jobs) {
  const done = jobs.filter((job) => job.status === "completed" && job.stats);
  const frames = done.reduce((sum, job) => sum + (job.stats.frames_analysed || 0), 0);
  $("dash-completed").textContent = String(done.length);
  $("dash-frames").textContent = done.length ? frames.toLocaleString() : "–";
  $("dash-found").textContent = done.length
    ? `${(done.reduce((sum, job) => sum + job.stats.ball_found_percent, 0) / done.length).toFixed(1)}%`
    : "–";
}

async function loadRecent() {
  $("recent-loading").hidden = false;
  $("recent-empty").hidden = true;
  $("recent-error").hidden = true;
  $("recent-list").hidden = true;
  try {
    const { jobs } = await api("/api/jobs");
    $("recent-list").replaceChildren(...jobs.map(recentItem));
    $("recent-list").hidden = jobs.length === 0;
    $("recent-empty").hidden = jobs.length > 0;
    $("clear-history").hidden = jobs.length === 0;
    renderDashboard(jobs);
  } catch (error) {
    $("recent-error-text").textContent = error.message;
    $("recent-error").hidden = false;
  } finally {
    $("recent-loading").hidden = true;
  }
}

// ---------- processing -------------------------------------------------

function updateElapsed(job) {
  clearInterval(elapsedTimer);
  const since = job.started_at || job.created_at;
  const label = job.started_at ? "Running for" : "Waiting for";
  const tick = () => { $("proc-elapsed").textContent = `${label} ${formatDuration(Date.now() / 1000 - since)}`; };
  tick();
  elapsedTimer = setInterval(tick, 1000);
}

function renderProcessing(job) {
  showView("processing");
  const badge = $("proc-badge");
  badge.className = `badge ${job.status}`;
  badge.textContent = job.status === "queued" ? "Queued" : "Processing";
  $("proc-file").textContent = job.filename;
  updateElapsed(job);

  const fill = $("proc-fill");
  const bar = $("proc-bar");
  const detail = $("proc-detail");
  if (job.status === "queued") {
    $("proc-title").textContent = "Waiting to start";
    detail.textContent = job.queue_position > 1
      ? `Position ${job.queue_position} in the queue. Videos are analysed one at a time.`
      : "Starting shortly…";
    fill.className = "fill indeterminate";
    fill.style.width = "";
    bar.removeAttribute("aria-valuenow");
  } else {
    $("proc-title").textContent = STAGE_TEXT[job.stage] || "Processing…";
    const progress = job.progress;
    if (job.stage === "detecting" && progress && progress.total) {
      const percent = Math.round((100 * progress.done) / progress.total);
      fill.className = "fill";
      fill.style.width = `${percent}%`;
      bar.setAttribute("aria-valuenow", String(percent));
      bar.setAttribute("aria-valuemin", "0");
      bar.setAttribute("aria-valuemax", "100");
      detail.textContent = `${progress.done} of ${progress.total} frames analysed`;
    } else {
      fill.className = "fill indeterminate";
      fill.style.width = "";
      bar.removeAttribute("aria-valuenow");
      detail.textContent = job.stage === "loading_model"
        ? "The first analysis after starting VolleyIQ takes a little longer while the model loads."
        : " ";
    }
  }
  const current = STAGES.indexOf(job.stage);
  for (const item of $("stepper").children) {
    const index = STAGES.indexOf(item.dataset.stage);
    item.className = job.status === "processing" && index < current ? "done" : index === current ? "active" : "";
    if (index === current) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
}

// ---------- results ----------------------------------------------------

function kpi(label, value, sub, className = "") {
  const box = el("div", `kpi ${className}`.trim());
  box.append(el("div", "kpi-label", label), el("div", "kpi-value", value));
  if (sub) box.append(el("div", "kpi-sub", sub));
  return box;
}

function percentOf(count, total) {
  return total ? `${Math.round((100 * count) / total)}% of frames` : "";
}

function renderResults(job) {
  showView("results");
  clearInterval(elapsedTimer);
  const stats = job.stats;
  const video = stats.video;
  const counts = stats.status_counts;
  const total = stats.frames_analysed;
  $("res-title").textContent = job.filename;
  $("res-meta").textContent = `Analysed ${formatDate(job.finished_at || job.created_at)} · ${formatDuration(video.duration_s)} of video`;

  const player = $("player");
  const source = `/api/jobs/${job.id}/video`;
  if (player.getAttribute("src") !== source) {
    player.src = source;
    $("player-error").hidden = true;
  }
  $("download-video").href = source;
  $("download-csv").href = `/api/jobs/${job.id}/track.csv`;

  $("kpis").replaceChildren(
    kpi("Ball located", `${stats.ball_found_percent}%`, `in ${stats.frames_with_ball} of ${total} analysed frames (good + uncertain)`, "hero-kpi"),
    kpi("Good", String(counts.good), percentOf(counts.good, total)),
    kpi("Uncertain", String(counts.uncertain), percentOf(counts.uncertain, total)),
    kpi("Lost", String(counts.lost), percentOf(counts.lost, total)),
    kpi("Bad", String(counts.bad), percentOf(counts.bad, total)),
  );

  const bar = $("status-bar");
  const legend = $("legend");
  bar.replaceChildren();
  legend.replaceChildren();
  for (const key of STATUS_ORDER) {
    const [name, description] = STATUS_INFO[key];
    const part = el("span", key);
    part.style.width = `${total ? (100 * (counts[key] || 0)) / total : 0}%`;
    bar.append(part);
    const item = el("li");
    item.append(el("span", `swatch ${key}`), el("span", "name", name), el("span", "count", String(counts[key] || 0)),
      el("span", "desc", description));
    legend.append(item);
  }

  const device = String(stats.device).toLowerCase() === "cuda" ? "GPU (CUDA)" : String(stats.device).toUpperCase();
  const facts = [
    ["Duration", formatDuration(video.duration_s)],
    ["Resolution", `${video.width} × ${video.height}`],
    ["Source frame rate", `${Number(video.fps).toFixed(2)} fps`],
    ["Analysed", `every ${stats.sample_interval} frame(s), ${stats.output_fps} per second`],
    ["Frames analysed", String(total)],
    ["Ball boxes from detector", String(stats.ball_boxes)],
    ["Static look-alikes ignored", String(stats.static_objects_ignored)],
    ["Max move between frames", `${stats.max_displacement_px} px`],
    ["Processing time", formatDuration(stats.processing_s)],
    ["Ran on", device],
  ];
  $("facts").replaceChildren(...facts.flatMap(([term, value]) => [el("dt", null, term), el("dd", null, value)]));

  const timings = stats.timings_s || {};
  const timingTotal = STAGES.reduce((sum, stage) => sum + (timings[stage] || 0), 0) || 1;
  $("timing").replaceChildren(...STAGES.map((stage) => {
    const part = el("span");
    part.style.width = `${(100 * (timings[stage] || 0)) / timingTotal}%`;
    part.style.background = STAGE_COLOUR[stage];
    return part;
  }));
  $("timing-legend").replaceChildren(...STAGES.filter((s) => timings[s] != null).map((stage) => {
    const item = el("li");
    const swatch = el("i");
    swatch.style.background = STAGE_COLOUR[stage];
    item.append(swatch, `${STAGE_LABEL[stage]} ${formatDuration(timings[stage])}`);
    return item;
  }));

  $("model-line").textContent =
    `Detector ${stats.model} (revision ${String(stats.model_revision).slice(0, 8)}), ${stats.tracking_strategy} tracker. ` +
    `Video ${stats.output_codec}, ${stats.output_fps} frames per second. Positions are automatic and can be wrong.`;

  if (!results || results.job.id !== job.id) loadTracking(job);
}

async function loadTracking(job) {
  results = { job, frames: [], fps: job.stats.output_fps, index: 0 };
  $("timeline-loading").hidden = false;
  $("timeline-loading").textContent = "Loading per-frame tracking…";
  $("readout").hidden = true;
  try {
    const data = await api(`/api/jobs/${job.id}/tracking`);
    if (!results || results.job.id !== job.id) return;
    results.frames = data.frames;
    results.fps = data.output_fps || job.stats.output_fps;
    const timeline = $("timeline");
    timeline.setAttribute("aria-valuemax", String(data.frames.length));
    $("timeline-loading").hidden = true;
    $("readout").hidden = false;
    drawTimeline();
    showFrame(0);
  } catch (error) {
    $("timeline-loading").textContent = olderServer
      ? "The per-frame timeline needs the latest VolleyIQ. Restart VolleyIQ to see it."
      : `Couldn't load the per-frame tracking: ${error.message}`;
  }
}

async function checkServerVersion() {
  // Servers from before /api/health existed can still analyse videos, but can't be stopped from the
  // page or serve the timeline. Say so instead of showing controls that would fail.
  try {
    await api("/api/health");
  } catch (error) {
    if (error.status !== 404) return;
    olderServer = true;
    $("stop-button").hidden = true;
    setServerStatus("online", "Running · restart to update");
  }
}

function statusColours() {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(STATUS_ORDER.map((key) => [key, style.getPropertyValue(`--${key}`).trim()]));
}

function drawTimeline() {
  if (!results || !results.frames.length) return;
  const canvas = $("timeline-canvas");
  const box = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(box.width * ratio));
  canvas.height = Math.max(1, Math.round(box.height * ratio));
  const context = canvas.getContext("2d");
  const colours = statusColours();
  const count = results.frames.length;
  const width = canvas.width / count;
  context.clearRect(0, 0, canvas.width, canvas.height);
  results.frames.forEach((frame, index) => {
    const tall = frame.status === "good" || frame.status === "uncertain";
    const height = tall ? canvas.height : canvas.height * 0.45;
    context.fillStyle = colours[frame.status] || colours.lost;
    context.globalAlpha = tall ? 1 : 0.7;
    context.fillRect(Math.floor(index * width), canvas.height - height, Math.max(1, Math.ceil(width)), height);
  });
  context.globalAlpha = 1;
  positionPlayhead();
}

function positionPlayhead() {
  if (!results || !results.frames.length) return;
  const percent = ((results.index + 0.5) / results.frames.length) * 100;
  $("playhead").style.left = `calc(${percent}% - 1px)`;
}

function showFrame(index) {
  if (!results || !results.frames.length) return;
  const frames = results.frames;
  index = Math.max(0, Math.min(frames.length - 1, index));
  results.index = index;
  const frame = frames[index];
  const [name] = STATUS_INFO[frame.status] || [frame.status];
  const chip = $("readout-status");
  chip.textContent = name;
  chip.className = `readout-status ${frame.status}`;
  $("readout-frame").textContent =
    `Frame ${index + 1} of ${frames.length} · ${formatClock(index / results.fps)} · source frame ${frame.frame_number}`;
  $("readout-position").textContent = frame.x == null
    ? "No ball position"
    : `Ball at (${Math.round(frame.x)}, ${Math.round(frame.y)}) · model score ${frame.score.toFixed(2)}`;
  $("readout-reason").textContent = frame.reason || "";
  const timeline = $("timeline");
  timeline.setAttribute("aria-valuenow", String(index + 1));
  timeline.setAttribute("aria-valuetext", `Frame ${index + 1} of ${frames.length}: ${name}`);
  positionPlayhead();
}

function seekToFrame(index) {
  if (!results || !results.frames.length) return;
  index = Math.max(0, Math.min(results.frames.length - 1, index));
  const player = $("player");
  if (isFinite(results.fps) && results.fps > 0 && player.readyState > 0) {
    player.currentTime = (index + 0.5) / results.fps;
  }
  showFrame(index);
}

function frameAtTime(seconds) {
  return Math.floor(seconds * results.fps + 1e-3);
}

function syncFromVideo() {
  if (!results || !results.frames.length) return;
  const index = frameAtTime($("player").currentTime);
  if (index !== results.index) showFrame(index);
}

function setupTimeline() {
  const timeline = $("timeline");
  const indexFromPointer = (event) => {
    const box = timeline.getBoundingClientRect();
    return Math.floor(((event.clientX - box.left) / box.width) * results.frames.length);
  };
  let dragging = false;
  timeline.addEventListener("pointerdown", (event) => {
    if (!results || !results.frames.length) return;
    dragging = true;
    timeline.setPointerCapture(event.pointerId);
    $("player").pause();
    seekToFrame(indexFromPointer(event));
  });
  timeline.addEventListener("pointermove", (event) => { if (dragging) seekToFrame(indexFromPointer(event)); });
  const endDrag = () => { dragging = false; };
  timeline.addEventListener("pointerup", endDrag);
  timeline.addEventListener("pointercancel", endDrag);
  timeline.addEventListener("keydown", (event) => {
    if (!results || !results.frames.length) return;
    const steps = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10 };
    let target = null;
    if (event.key in steps) target = results.index + steps[event.key];
    else if (event.key === "Home") target = 0;
    else if (event.key === "End") target = results.frames.length - 1;
    if (target === null) return;
    event.preventDefault();
    $("player").pause();
    seekToFrame(target);
  });

  const player = $("player");
  player.addEventListener("timeupdate", syncFromVideo);
  player.addEventListener("seeked", syncFromVideo);
  player.addEventListener("error", () => { if (player.getAttribute("src")) $("player-error").hidden = false; });
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    const onFrame = () => { syncFromVideo(); player.requestVideoFrameCallback(onFrame); };
    player.requestVideoFrameCallback(onFrame);
  }
  if ("ResizeObserver" in window) new ResizeObserver(() => drawTimeline()).observe(timeline);
  else window.addEventListener("resize", drawTimeline);
}

// ---------- failed / stopped --------------------------------------------

function renderFailed(job) {
  showView("failed");
  clearInterval(elapsedTimer);
  $("fail-file").textContent = job.filename || "";
  $("delete-failed").hidden = !job.id;
  $("fail-message").textContent = job.error || "Something went wrong while processing this video.";
  $("fail-limit").textContent = `${Math.round(config.max_duration_s / 60)} minutes`;
}

async function openStopDialog() {
  const warning = $("stop-warning");
  warning.hidden = true;
  try {
    const { jobs } = await api("/api/jobs");
    const active = jobs.filter((job) => job.status === "queued" || job.status === "processing").length;
    if (active) {
      warning.textContent = `${active} analysis${active > 1 ? "es are" : " is"} still waiting or running and will be ` +
        "interrupted. You would need to upload the video again.";
      warning.hidden = false;
    }
  } catch { /* the dialog still works */ }
  const dialog = $("stop-dialog");
  dialog.returnValue = "";
  if (typeof dialog.showModal === "function") dialog.showModal();
  else if (window.confirm("Stop VolleyIQ on this computer?")) stopServer();
}

async function stopServer() {
  try {
    await api("/api/shutdown", { method: "POST", headers: { "X-VolleyIQ-Action": "shutdown" } });
  } catch (error) {
    if (!error.network) {
      window.alert(`VolleyIQ could not be stopped: ${error.message}`);
      return;
    }
  }
  stopped = true;
  clearTimeout(pollTimer);
  clearInterval(elapsedTimer);
  clearInterval(healthTimer);
  $("player").pause();
  $("player").removeAttribute("src");
  setServerStatus("offline", "Stopped");
  $("stop-button").disabled = true;
  showView("stopped");
}

// ---------- routing ----------------------------------------------------

async function pollJob(jobId) {
  clearTimeout(pollTimer);
  if (location.hash !== `#job/${jobId}` || stopped) return;
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (error) {
    if (error.network) {
      pollTimer = setTimeout(() => pollJob(jobId), POLL_MS * 2);
      return;
    }
    renderFailed({ error: `${error.message} Finished analyses are deleted after ${config.retention_hours} hours.` });
    return;
  }
  if (location.hash !== `#job/${jobId}`) return;
  currentJob = job;
  if (job.status === "completed") renderResults(job);
  else if (job.status === "failed") renderFailed(job);
  else {
    renderProcessing(job);
    pollTimer = setTimeout(() => pollJob(jobId), POLL_MS);
  }
}

function route() {
  if (stopped) return;
  clearTimeout(pollTimer);
  const hash = location.hash;
  const match = hash.match(/^#job\/([0-9a-f]{16})$/);
  if (match) {
    if (!results || results.job.id !== match[1]) results = null;
    if (!currentJob || currentJob.id !== match[1]) currentJob = null;
    window.scrollTo(0, 0);
    pollJob(match[1]);
    return;
  }
  clearInterval(elapsedTimer);
  results = null;
  currentJob = null;
  releasePlayer();
  const wasHome = document.body.dataset.view === "home";
  showView("home");
  loadRecent();
  const section = hash.slice(1);
  if (SECTIONS.includes(section)) {
    requestAnimationFrame(() => {
      scrollToId(section);
      if (section === "analyze") $("dropzone").focus({ preventScroll: true });
    });
  } else if (!wasHome || hash === "") {
    window.scrollTo(0, 0);
  }
}

function onLinkClick(event) {
  const link = event.target.closest('a[href^="#"]');
  if (!link) return;
  const hash = link.getAttribute("href");
  if (hash === location.hash || (hash === "#" && location.hash === "")) {
    // Same hash: no hashchange event, so route by hand (e.g. "Analyze a match" twice).
    event.preventDefault();
    route();
  } else if (hash === "#") {
    event.preventDefault();
    history.pushState(null, "", location.pathname);
    route();
  }
}

// ---------- start ------------------------------------------------------

async function init() {
  try {
    config = { ...config, ...(await api("/api/config")) };
  } catch { /* keep defaults; the server checks again */ }
  applyFacts();
  await checkServerVersion();

  const dropzone = $("dropzone");
  $("file-input").addEventListener("change", (event) => chooseFile(event.target.files[0]));
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); $("file-input").click(); }
  });
  for (const name of ["dragenter", "dragover"]) {
    dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
  }
  for (const name of ["dragleave", "drop"]) dropzone.addEventListener(name, () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    chooseFile(event.dataTransfer.files[0]);
  });
  // A file dropped anywhere else must not make the browser leave the app.
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  $("clear-file").addEventListener("click", clearFile);
  $("replace-file").addEventListener("click", () => $("file-input").click());
  $("cancel-upload").addEventListener("click", () => { if (uploadRequest) uploadRequest.abort(); });
  $("clear-history").addEventListener("click", clearHistory);
  const deleteCurrent = async () => {
    const job = currentJob;
    if (!job) return;
    if (await deleteJob(job, { running: job.status === "queued" || job.status === "processing" })) location.hash = "#recent";
  };
  for (const id of ["cancel-job", "delete-results", "delete-failed"]) $(id).addEventListener("click", deleteCurrent);
  $("analyze-button").addEventListener("click", upload);
  $("recent-retry").addEventListener("click", loadRecent);
  $("new-from-results").addEventListener("click", () => { location.hash = "#analyze"; });
  $("new-from-failed").addEventListener("click", () => { location.hash = "#analyze"; });
  $("stop-button").addEventListener("click", openStopDialog);
  $("stop-dialog").addEventListener("close", () => {
    if ($("stop-dialog").returnValue === "stop") stopServer();
  });
  document.addEventListener("click", onLinkClick);
  window.addEventListener("hashchange", route);
  setupTimeline();

  // Keeps the header's "Running locally" / "Not reachable" indicator honest.
  healthTimer = setInterval(() => { if (!stopped) api(olderServer ? "/api/config" : "/api/health").catch(() => {}); }, HEALTH_MS);
  route();
}

init();
