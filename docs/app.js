// VolleyIQ website: send a video to the VolleyIQ analysis server (a Hugging Face Space)
// and show the real result. Everything displayed comes from the server's response;
// user-provided text is always inserted with textContent, never as HTML.

import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client@2.7.2/dist/browser.js";
import { ALLOWED_EXTENSIONS, BACKEND, MAX_DURATION_S, MAX_UPLOAD_MB } from "./config.js";

const $ = (id) => document.getElementById(id);
const STEPS = ["upload", "progress", "error", "results"];
const STATUS_INFO = {
  good: ["Good", "Position confirmed by smooth movement over at least 3 frames."],
  uncertain: ["Uncertain", "A position was chosen but not confirmed. Check it before relying on it."],
  lost: ["Lost", "No ball found in this frame (or only a static look-alike)."],
  bad: ["Bad", "Detections were rejected as impossible jumps, usually false positives."],
};
const STATUS_ORDER = ["good", "uncertain", "lost", "bad"];
const MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024;

let backend = BACKEND;
let client = null;
let selectedFile = null;
let busy = false;
let result = null; // {frames, fps, index, videoUrl, csvUrl}

// ---------- helpers ----------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

function formatBytes(bytes) {
  if (bytes >= 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function formatDuration(seconds) {
  if (seconds == null || !isFinite(seconds)) return "–";
  const s = Math.max(0, Math.round(seconds));
  return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`;
}

function formatClock(seconds) {
  const s = Math.max(0, seconds);
  return `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

function showStep(name) {
  for (const step of STEPS) $(`step-${step}`).hidden = step !== name;
}

function setServer(state, text) {
  $("server").className = `server ${state}`;
  $("server-text").textContent = text;
}

function errorText(error) {
  // Gradio errors carry the server's message; anything else gets a plain explanation.
  const message = (error && (error.message || error.toString())) || "";
  return message && message !== "[object Object]" ? message : "The analysis server returned an error.";
}

// A local development backend can be chosen with ?backend=…, but only when this page is itself served locally.
function chooseBackend() {
  const local = ["localhost", "127.0.0.1"].includes(location.hostname);
  const override = new URLSearchParams(location.search).get("backend");
  if (local && override) backend = override;
}

// ---------- connecting --------------------------------------------------

async function connect() {
  if (!backend) return null;
  if (!client) {
    setServer("", "Connecting to the analysis server… it may take a minute to wake up.");
    client = await Client.connect(backend, { events: ["data", "status"] });
  }
  setServer("online", "Analysis server online");
  return client;
}

async function checkServer() {
  if (!backend) {
    setServer("offline", "Online analysis not available yet");
    $("unavailable").hidden = false;
    $("dropzone").classList.add("disabled");
    $("file-input").disabled = true;
    return;
  }
  try {
    await connect();
  } catch (error) {
    client = null;
    setServer("offline", "The analysis server isn't responding right now. Please try again in a few minutes.");
  }
}

// ---------- choosing a file ---------------------------------------------

function showUploadError(message) {
  $("upload-error").textContent = message;
  $("upload-error").hidden = !message;
}

function chooseFile(file) {
  showUploadError("");
  if (!file || !backend || busy) return;
  const dot = file.name.lastIndexOf(".");
  const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : "";
  if (!ALLOWED_EXTENSIONS.includes(ext)) {
    showUploadError(`Unsupported file type ${ext || "(none)"}. Use ${ALLOWED_EXTENSIONS.join(", ")}.`);
    return;
  }
  if (file.size === 0) return showUploadError("This file is empty.");
  if (file.size > MAX_UPLOAD_BYTES) {
    return showUploadError(`This file is ${formatBytes(file.size)}. The limit is ${MAX_UPLOAD_MB} MB.`);
  }
  selectedFile = file;
  $("file-name").textContent = file.name;
  $("file-size").textContent = `${formatBytes(file.size)} · ready to analyse`;
  $("selected").hidden = false;
  $("analyze-button").disabled = false;
}

function clearFile() {
  selectedFile = null;
  $("file-input").value = "";
  $("selected").hidden = true;
  $("analyze-button").disabled = true;
}

// ---------- analysing ---------------------------------------------------

function setProgress(title, detail, fraction) {
  $("progress-title").textContent = title;
  $("progress-detail").textContent = detail || " ";
  const fill = $("progress-fill");
  const bar = $("progress-bar");
  if (fraction == null) {
    fill.className = "fill indeterminate";
    fill.style.width = "";
    bar.removeAttribute("aria-valuenow");
  } else {
    const percent = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
    fill.className = "fill";
    fill.style.width = `${percent}%`;
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.setAttribute("aria-valuenow", String(percent));
  }
}

function onStatus(status) {
  if (status.stage === "pending" && status.queue && status.position != null) {
    const ahead = status.position;
    setProgress("Waiting in the queue", ahead > 0 ? `${ahead} video${ahead > 1 ? "s" : ""} ahead of yours.`
      : "Your video is next.", null);
  } else if (status.stage === "generating" || (status.progress_data && status.progress_data.length)) {
    const step = status.progress_data && status.progress_data[status.progress_data.length - 1];
    if (step && step.desc) setProgress(step.desc, "Real analysis in progress on the server.", step.progress);
    else setProgress("Analysing your video", "Real analysis in progress on the server.", null);
  }
}

async function analyze() {
  if (!selectedFile || busy) return;
  busy = true;
  const file = selectedFile;
  showStep("progress");
  setProgress("Uploading your video…", `${file.name} · ${formatBytes(file.size)}`, null);
  try {
    const app = await connect();
    let data = null;
    const job = app.submit("/analyze", [handle_file(file)]);
    for await (const message of job) {
      if (message.type === "status") {
        if (message.stage === "error") throw new Error(message.message || "The analysis failed on the server.");
        onStatus(message);
      } else if (message.type === "data") {
        data = message.data;
      }
    }
    if (!data) throw new Error("The server finished without returning a result.");
    showResults(file.name, data);
  } catch (error) {
    $("error-text").textContent = errorText(error);
    showStep("error");
  } finally {
    busy = false;
  }
}

// ---------- results -----------------------------------------------------

function fileUrl(value) {
  if (!value) return null;
  const file = value.video || value; // gr.Video may wrap the file
  return file.url || null;
}

function kpi(label, value, sub, wide) {
  const box = el("div", wide ? "kpi wide" : "kpi");
  box.append(el("div", "kpi-label", label), el("div", "kpi-value", value));
  if (sub) box.append(el("div", "kpi-sub", sub));
  return box;
}

function showResults(name, data) {
  const [videoOut, csvOut, stats, tracking] = data;
  const videoUrl = fileUrl(videoOut);
  const csvUrl = fileUrl(csvOut);
  if (!videoUrl || !stats || !tracking) throw new Error("The server's result was incomplete.");

  result = { frames: tracking.frames, fps: tracking.output_fps || stats.output_fps, index: 0, videoUrl, csvUrl,
    name: name.replace(/\.[^.]+$/, "") };
  $("result-title").textContent = name;
  const player = $("player");
  $("player-error").hidden = true;
  player.src = videoUrl;

  const counts = stats.status_counts;
  const total = stats.frames_analysed;
  const share = (count) => (total ? `${Math.round((100 * count) / total)}% of frames` : "");
  $("kpis").replaceChildren(
    kpi("Ball located", `${stats.ball_found_percent}%`, `in ${stats.frames_with_ball} of ${total} analysed frames`, true),
    kpi("Good", String(counts.good), share(counts.good)),
    kpi("Uncertain", String(counts.uncertain), share(counts.uncertain)),
    kpi("Lost", String(counts.lost), share(counts.lost)),
    kpi("Bad", String(counts.bad), share(counts.bad)),
  );

  const video = stats.video;
  const facts = [
    ["Duration", formatDuration(video.duration_s)],
    ["Resolution", `${video.width} × ${video.height}`],
    ["Frame rate", `${Number(video.fps).toFixed(2)} fps`],
    ["Analysed", `every ${stats.sample_interval} frame(s)`],
    ["Frames analysed", String(total)],
    ["Ball boxes from detector", String(stats.ball_boxes)],
    ["Processing time", formatDuration(stats.processing_s)],
    ["Ran on", String(stats.device).toLowerCase() === "cuda" ? "GPU" : String(stats.device).toUpperCase()],
  ];
  $("facts").replaceChildren(...facts.flatMap(([term, value]) => [el("dt", null, term), el("dd", null, value)]));

  $("timeline").setAttribute("aria-valuemax", String(result.frames.length));
  showStep("results");
  drawTimeline();
  showFrame(0);
}

async function download(url, filename) {
  // Cross-site links ignore the download attribute, so fetch the file and save it under a proper name.
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(String(response.status));
    const blob = await response.blob();
    const link = el("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  } catch {
    window.open(url, "_blank", "noopener");
  }
}

function statusColours() {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(STATUS_ORDER.map((key) => [key, style.getPropertyValue(`--${key}`).trim()]));
}

function drawTimeline() {
  if (!result || !result.frames.length) return;
  const canvas = $("timeline-canvas");
  const box = canvas.getBoundingClientRect();
  if (!box.width) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(box.width * ratio);
  canvas.height = Math.round(box.height * ratio);
  const context = canvas.getContext("2d");
  const colours = statusColours();
  const width = canvas.width / result.frames.length;
  context.clearRect(0, 0, canvas.width, canvas.height);
  result.frames.forEach((frame, index) => {
    const tall = frame.status === "good" || frame.status === "uncertain";
    const height = tall ? canvas.height : canvas.height * 0.45;
    context.globalAlpha = tall ? 1 : 0.7;
    context.fillStyle = colours[frame.status] || colours.lost;
    context.fillRect(Math.floor(index * width), canvas.height - height, Math.max(1, Math.ceil(width)), height);
  });
  context.globalAlpha = 1;
  placePlayhead();
}

function placePlayhead() {
  if (!result || !result.frames.length) return;
  $("playhead").style.left = `calc(${((result.index + 0.5) / result.frames.length) * 100}% - 1px)`;
}

function showFrame(index) {
  if (!result || !result.frames.length) return;
  index = Math.max(0, Math.min(result.frames.length - 1, index));
  result.index = index;
  const frame = result.frames[index];
  const [name] = STATUS_INFO[frame.status] || [frame.status];
  const chip = $("readout-status");
  chip.textContent = name;
  chip.className = `chip ${frame.status}`;
  $("readout-frame").textContent = `Frame ${index + 1} of ${result.frames.length} · ${formatClock(index / result.fps)}`;
  $("readout-position").textContent = frame.x == null
    ? "No ball position"
    : `Ball at (${Math.round(frame.x)}, ${Math.round(frame.y)}) · model score ${Number(frame.score).toFixed(2)}`;
  $("readout-reason").textContent = frame.reason || "";
  $("timeline").setAttribute("aria-valuenow", String(index + 1));
  $("timeline").setAttribute("aria-valuetext", `Frame ${index + 1} of ${result.frames.length}: ${name}`);
  placePlayhead();
}

function seekToFrame(index) {
  if (!result || !result.frames.length) return;
  index = Math.max(0, Math.min(result.frames.length - 1, index));
  const player = $("player");
  if (result.fps > 0 && player.readyState > 0) player.currentTime = (index + 0.5) / result.fps;
  showFrame(index);
}

function syncFromVideo() {
  if (!result || !result.frames.length || !(result.fps > 0)) return;
  const index = Math.floor($("player").currentTime * result.fps + 1e-3);
  if (index !== result.index) showFrame(index);
}

function setupTimeline() {
  const timeline = $("timeline");
  const fromPointer = (event) => {
    const box = timeline.getBoundingClientRect();
    return Math.floor(((event.clientX - box.left) / box.width) * result.frames.length);
  };
  let dragging = false;
  timeline.addEventListener("pointerdown", (event) => {
    if (!result) return;
    dragging = true;
    timeline.setPointerCapture(event.pointerId);
    $("player").pause();
    seekToFrame(fromPointer(event));
  });
  timeline.addEventListener("pointermove", (event) => { if (dragging) seekToFrame(fromPointer(event)); });
  timeline.addEventListener("pointerup", () => { dragging = false; });
  timeline.addEventListener("pointercancel", () => { dragging = false; });
  timeline.addEventListener("keydown", (event) => {
    if (!result) return;
    const steps = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 10, PageDown: -10 };
    let target = null;
    if (event.key in steps) target = result.index + steps[event.key];
    else if (event.key === "Home") target = 0;
    else if (event.key === "End") target = result.frames.length - 1;
    if (target === null) return;
    event.preventDefault();
    $("player").pause();
    seekToFrame(target);
  });
  const player = $("player");
  player.addEventListener("timeupdate", syncFromVideo);
  player.addEventListener("seeked", syncFromVideo);
  player.addEventListener("error", () => { if (player.getAttribute("src")) $("player-error").hidden = false; });
  if ("ResizeObserver" in window) new ResizeObserver(() => drawTimeline()).observe(timeline);
  else window.addEventListener("resize", drawTimeline);
}

function startOver() {
  result = null;
  const player = $("player");
  player.pause();
  player.removeAttribute("src");
  player.load();
  clearFile();
  showStep("upload");
  $("dropzone").focus();
}

// ---------- start -------------------------------------------------------

function init() {
  chooseBackend();
  const types = ALLOWED_EXTENSIONS.map((e) => e.slice(1).toUpperCase()).join(", ");
  $("limits").textContent = `${types} · up to ${MAX_UPLOAD_MB} MB · up to ${MAX_DURATION_S} seconds`;

  const dropzone = $("dropzone");
  $("file-input").addEventListener("change", (event) => chooseFile(event.target.files[0]));
  dropzone.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && backend) { event.preventDefault(); $("file-input").click(); }
  });
  for (const name of ["dragenter", "dragover"]) {
    dropzone.addEventListener(name, (event) => { event.preventDefault(); if (backend) dropzone.classList.add("dragging"); });
  }
  for (const name of ["dragleave", "drop"]) dropzone.addEventListener(name, () => dropzone.classList.remove("dragging"));
  dropzone.addEventListener("drop", (event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());

  $("clear-file").addEventListener("click", clearFile);
  $("analyze-button").addEventListener("click", analyze);
  $("retry-button").addEventListener("click", startOver);
  $("new-button").addEventListener("click", startOver);
  $("download-video").addEventListener("click", () => {
    if (result) download(result.videoUrl, `${result.name}-volleyiq${result.videoUrl.match(/\.(mp4|webm)(\?|$)/)?.[0].replace("?", "") || ".mp4"}`);
  });
  $("download-csv").addEventListener("click", () => { if (result) download(result.csvUrl, `${result.name}-volleyiq-tracking.csv`); });
  setupTimeline();
  showStep("upload");
  checkServer();
}

init();
