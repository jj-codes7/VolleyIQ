"""VolleyIQ public demo: a Hugging Face Space (Gradio + ZeroGPU).

It runs the same pipeline as the local app (volleyiq.webapp.pipeline):
extract about 6 frames per second, detect the ball with the pinned pretrained
model, track it with the baseline rules, and render an annotated video. Only
the detection step borrows a GPU (ZeroGPU); the rest runs on the Space's CPU.

The public website (GitHub Pages) calls the `/analyze` endpoint with the
official Gradio JavaScript client. Opening the Space directly shows a simple
Gradio page for the same endpoint.

Public-service limits: one analysis at a time (others wait in a short queue),
clips up to MAX_DURATION_S seconds and MAX_UPLOAD_MB, and a capped GPU time per
clip. Uploads and results live only in temporary folders: frames and detections
are deleted as soon as a request finishes, and results after RESULT_TTL_S.

Run locally (CPU) for testing:
    python space/app.py          (open http://127.0.0.1:7860)
"""

import logging
import os
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

# In the Space, the volleyiq package sits next to this file; in the repository it is one level up.
HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parent):
    if (candidate / "volleyiq" / "__init__.py").is_file():
        sys.path.insert(0, str(candidate))
        break

import gradio as gr  # noqa: E402
import spaces  # noqa: E402  (a no-op outside ZeroGPU)
import torch  # noqa: E402

from volleyiq.detect_ball import MODEL_ID, BallDetector  # noqa: E402
from volleyiq.webapp.pipeline import ALLOWED_EXTENSIONS, VideoError, detect_frames, process_video  # noqa: E402
from volleyiq.webapp.server import read_tracking  # noqa: E402

MAX_DURATION_S = 60
MAX_UPLOAD_MB = 100
MAX_GPU_SECONDS = 120  # hard cap per clip; ZeroGPU stops the step if it runs longer
GPU_SECONDS_BASE, GPU_SECONDS_PER_FRAME = 20, 0.25  # requested budget; only real use counts against quota
RESULT_TTL_S = 3600
QUEUE_SIZE = 8
WORK_ROOT = Path(tempfile.gettempdir()) / "volleyiq"
STAGE_PROGRESS = {
    "extracting": (0.05, "Extracting frames"),
    "loading_model": (0.15, "Preparing the AI model"),
    "detecting": (0.20, "Detecting the ball"),
    "tracking": (0.85, "Tracking the ball"),
    "rendering": (0.90, "Rendering the annotated video"),
}

log = logging.getLogger("volleyiq.space")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def pick_device():
    """CUDA on ZeroGPU (emulated outside @spaces.GPU, real inside it) or a local GPU; otherwise CPU."""
    forced = os.environ.get("VOLLEYIQ_DEVICE")
    if forced:
        return forced
    if os.environ.get("SPACES_ZERO_GPU") or torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ZeroGPU wants models placed on the GPU when the app starts, not inside the GPU function.
DETECTOR = BallDetector(device=pick_device())


def gpu_budget(frames_dir, detections_dir):
    """Seconds of GPU time to request for one clip, from its number of extracted frames."""
    frames = len(list(Path(frames_dir).glob("*.jpg")))
    return int(min(MAX_GPU_SECONDS, GPU_SECONDS_BASE + frames * GPU_SECONDS_PER_FRAME))


@spaces.GPU(duration=gpu_budget)
def detect_on_gpu(frames_dir, detections_dir):
    detect_frames(Path(frames_dir), Path(detections_dir), DETECTOR)


def sweep_old_results(root=WORK_ROOT, max_age_s=RESULT_TTL_S, now=None):
    """Delete request folders older than `max_age_s`. Returns how many were removed."""
    now = time.time() if now is None else now
    removed = 0
    if not root.is_dir():
        return removed
    for folder in root.iterdir():
        if folder.is_dir() and folder.name.startswith("job-") and now - folder.stat().st_mtime > max_age_s:
            shutil.rmtree(folder, ignore_errors=True)
            removed += 1
    return removed


def check_upload(path):
    """Reject anything that isn't a reasonably sized video file, before any work is done."""
    if not path:
        raise gr.Error("Choose a video first.")
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise gr.Error(f"Unsupported file type {path.suffix or '(none)'}. Use {', '.join(ALLOWED_EXTENSIONS)}.")
    if not path.is_file() or path.stat().st_size == 0:
        raise gr.Error("The uploaded file is empty.")
    if path.stat().st_size > MAX_UPLOAD_MB * 1024 * 1024:
        raise gr.Error(f"The file is larger than the {MAX_UPLOAD_MB} MB limit.")
    return path


def analyze(video, progress=gr.Progress()):
    """Analyse one uploaded video with the real VolleyIQ pipeline.

    Returns the annotated video, the per-frame tracking CSV, the statistics and
    the per-frame tracking data used by the website's timeline.
    """
    source = check_upload(video)
    sweep_old_results()
    job_dir = WORK_ROOT / f"job-{uuid.uuid4().hex}"
    job_dir.mkdir(parents=True)
    work_dir = job_dir / "work"
    work_dir.mkdir()

    def on_stage(stage):
        if stage in STAGE_PROGRESS:
            fraction, text = STAGE_PROGRESS[stage]
            if stage == "detecting" and DETECTOR.device == "cuda":
                text += " on a GPU"
            progress(fraction, desc=text)

    try:
        output, stats = process_video(
            source, work_dir, lambda: DETECTOR, on_stage,
            detect=lambda frames_dir, detections_dir, _detector: detect_on_gpu(str(frames_dir), str(detections_dir)),
            max_duration_s=MAX_DURATION_S,
        )
        # Keep only the results; frames and detection files are deleted straight away.
        video_out = job_dir / f"volleyiq-annotated{output.suffix}"
        csv_out = job_dir / "volleyiq-tracking.csv"
        shutil.move(str(output), video_out)
        shutil.move(str(work_dir / "track.csv"), csv_out)
        tracking = {"frames": read_tracking(csv_out), "output_fps": stats["output_fps"]}
        stats = {key: value for key, value in stats.items() if key != "model_revision"}
        stats["max_duration_s"] = MAX_DURATION_S
        progress(1.0, desc="Done")
        return str(video_out), str(csv_out), stats, tracking
    except VideoError as error:
        raise gr.Error(str(error)) from None
    except gr.Error:
        raise
    except Exception:
        log.exception("Analysis failed")
        raise gr.Error("The analysis failed on the server. Please try again, or try a different video.") from None
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


ABOUT = f"""
# VolleyIQ · volleyball ball tracking

Upload a volleyball clip (up to **{MAX_DURATION_S} seconds** and **{MAX_UPLOAD_MB} MB**; MP4, MOV, M4V, AVI, MKV or WEBM).
VolleyIQ samples about 6 frames per second, detects the ball with the pretrained
[{MODEL_ID}](https://huggingface.co/{MODEL_ID}) model, tracks it with transparent rules, and returns an
annotated video, statistics and a per-frame CSV.

It tracks the **ball only**: no player tracking, rally detection or tactical statistics, and the per-frame
statuses are rules, not a measured accuracy.

**Privacy:** your video is processed on this Hugging Face Space. It is not shared with other visitors and is not
kept: working files are deleted when the analysis ends and results within about an hour. Only upload videos you
have the right to use.
"""

with gr.Blocks(title="VolleyIQ", delete_cache=(RESULT_TTL_S, RESULT_TTL_S)) as demo:
    gr.Markdown(ABOUT)
    with gr.Row():
        with gr.Column():
            video_in = gr.File(label="Volleyball video", file_types=list(ALLOWED_EXTENSIONS), type="filepath")
            run = gr.Button("Analyze Video", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Annotated video")
            csv_out = gr.File(label="Tracking CSV")
    stats_out = gr.JSON(label="Statistics")
    tracking_out = gr.JSON(label="Per-frame tracking", visible=False)
    run.click(analyze, inputs=video_in, outputs=[video_out, csv_out, stats_out, tracking_out], api_name="analyze")

demo.queue(default_concurrency_limit=1, max_size=QUEUE_SIZE)

if __name__ == "__main__":
    demo.launch(max_file_size=f"{MAX_UPLOAD_MB}mb", show_error=True,
                server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
