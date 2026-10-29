"""Process one uploaded video with the existing VolleyIQ modules.

Steps, all reusing the command-line tools' code:

  1. probe_video        check OpenCV can read it, and that it isn't too long
  2. extract_frames     save about 6 frames per second (volleyiq.extract_frames)
  3. run_detection      pretrained ball detector on each frame (volleyiq.detect_ball)
  4. track              baseline rule-based tracker (volleyiq.track_ball)
  5. render_video       draw the tracking on each sampled frame (track_ball.draw_tracking)
                        into a video the browser can play (H.264 MP4, or VP8 WebM)

The output shows only the sampled frames, played at real-time speed. Ball
positions are never invented for the frames in between.
"""

import time
from pathlib import Path

import cv2

from volleyiq.annotate_ball import list_frames, load_image
from volleyiq.detect_ball import DEFAULT_THRESHOLD, DEFAULT_TILE, MODEL_ID, MODEL_REVISION, run_detection
from volleyiq.extract_frames import extract_frames
from volleyiq.track_ball import (
    BASELINE,
    DEFAULT_MAX_DISPLACEMENT,
    STATUSES,
    draw_tracking,
    find_static_positions,
    load_detections,
    track,
    write_csv,
)

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
MAX_DURATION_S = 300  # 5 minutes: about 1,800 sampled frames, roughly 7 minutes on the RTX 3050
MIN_SIZE = 64
SAMPLES_PER_SECOND = 6  # matches extract_frames --every 5 on ~30 FPS video
REFERENCE_WIDTH = 1920  # the 120 px default displacement was measured on 1920-wide footage
MAX_OUTPUT_WIDTH = 1920

# (OpenCV fourcc, file suffix, codecs that read back as browser-playable)
BROWSER_FORMATS = (("avc1", ".mp4", {"h264", "avc1"}), ("VP80", ".webm", {"VP80"}))
MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm"}


class VideoError(ValueError):
    """A problem with the user's video, worded so it can be shown to them."""


def _length_text(seconds):
    if seconds >= 120:
        minutes = round(seconds / 60, 1)
        return f"{minutes:g} minutes"
    return f"{round(seconds)} seconds"


def probe_video(path, max_duration_s=None):
    """Check that OpenCV can decode `path`. Returns its width, height, fps, frame count and duration.

    `max_duration_s` defaults to MAX_DURATION_S; the hosted demo passes a shorter limit.
    """
    limit = MAX_DURATION_S if max_duration_s is None else max_duration_s
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError("This file could not be opened as a video. It may be corrupted or use an unsupported codec.")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise VideoError("No frames could be decoded from this video. It may be corrupted or use an unsupported codec.")
        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    if not 1 <= fps <= 240:
        raise VideoError("The video's frame rate could not be read, so it can't be analysed.")
    if width < MIN_SIZE or height < MIN_SIZE:
        raise VideoError(f"The video is too small ({width} x {height}). It must be at least {MIN_SIZE} x {MIN_SIZE} pixels.")
    if frame_count <= 0:
        raise VideoError("The video's length could not be read, so it can't be analysed.")
    duration = frame_count / fps
    if duration > limit:
        raise VideoError(f"The video is {_length_text(duration)} long. The limit is {_length_text(limit)}; "
                         "trim it to a shorter clip first.")
    return {"width": width, "height": height, "fps": round(fps, 3), "frame_count": frame_count,
            "duration_s": round(duration, 2)}


def sample_interval(fps):
    """Keep every Nth frame so about SAMPLES_PER_SECOND frames per second are analysed."""
    return max(1, round(fps / SAMPLES_PER_SECOND))


def scaled_max_displacement(width):
    """The tracker's movement limit, scaled from the 1920-pixel-wide footage it was tuned on."""
    return round(max(20.0, DEFAULT_MAX_DISPLACEMENT * width / REFERENCE_WIDTH), 1)


def output_size(width, height):
    """Output frame size: at most MAX_OUTPUT_WIDTH wide, and even (H.264 needs even sizes)."""
    scale = min(1.0, MAX_OUTPUT_WIDTH / width)
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def _read_codec(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None, 0
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1")
        return fourcc, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()


def render_video(rows, frames_dir, output_base, fps):
    """Draw tracking on every sampled frame and save a browser-playable video.

    Tries H.264 MP4 first, then VP8 WebM, and checks the written file reads
    back with that codec. Returns (path, codec name).
    """
    frames = {path.name: path for path in list_frames(frames_dir)}
    first = load_image(frames[rows[0]["frame_name"]])
    width, height = first.shape[1], first.shape[0]
    size = output_size(width, height)

    for fourcc, suffix, accepted in BROWSER_FORMATS:
        path = Path(output_base).with_suffix(suffix)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        try:
            if not writer.isOpened():
                continue
            for index, row in enumerate(rows):
                image = draw_tracking(load_image(frames[row["frame_name"]]), rows, index)
                if (image.shape[1], image.shape[0]) != size:
                    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                writer.write(image)
        finally:
            writer.release()
        codec, count = _read_codec(path)
        if codec in accepted and count > 0:
            return path, codec
        path.unlink(missing_ok=True)
    raise RuntimeError("OpenCV could not write a browser-playable video (tried H.264 MP4 and VP8 WebM).")


def detect_frames(frames_dir, detections_dir, detector):
    """The detection step: the pretrained model on every extracted frame (volleyiq.detect_ball)."""
    run_detection(frames_dir, detections_dir, tile=DEFAULT_TILE, threshold=DEFAULT_THRESHOLD, detector=detector)


def process_video(input_path, work_dir, get_detector, on_stage=lambda stage: None, detect=detect_frames,
                  max_duration_s=None):
    """Run the whole analysis for one video. Returns (output video path, statistics dict).

    `get_detector()` returns the loaded model (or a stand-in in tests); it is
    called only when detection starts. `on_stage(name)` reports progress.
    `detect(frames_dir, detections_dir, detector)` runs the detection step; the
    hosted demo passes a version that runs on a borrowed GPU. `max_duration_s`
    overrides the video length limit (see probe_video).
    """
    work_dir = Path(work_dir)
    timings, started = {}, time.perf_counter()

    def stage(name):
        now = time.perf_counter()
        if stage.current:
            timings[stage.current] = round(now - stage.last, 2)
        stage.current, stage.last = name, now
        on_stage(name)
    stage.current, stage.last = None, started

    info = probe_video(input_path, max_duration_s)
    every = sample_interval(info["fps"])
    frames_dir, detections_dir = work_dir / "frames", work_dir / "detections"

    stage("extracting")
    sampled = extract_frames(input_path, frames_dir, every)

    stage("loading_model")
    detector = get_detector()

    stage("detecting")
    detect(frames_dir, detections_dir, detector)

    stage("tracking")
    frames = load_detections(detections_dir)
    static = find_static_positions(frames)
    max_displacement = scaled_max_displacement(info["width"])
    rows = track(frames, max_displacement, static_positions=static, strategy=BASELINE)
    write_csv(rows, work_dir / "track.csv")

    stage("rendering")
    output_fps = info["fps"] / every
    output_path, codec = render_video(rows, frames_dir, work_dir / "output", output_fps)
    stage(None)

    counts = {status: sum(1 for r in rows if r["status"] == status) for status in STATUSES}
    with_position = counts["good"] + counts["uncertain"]
    stats = {
        "video": info,
        "sample_interval": every,
        "frames_analysed": sampled,
        "output_fps": round(output_fps, 3),
        "output_codec": codec,
        "ball_boxes": sum(len(f["candidates"]) for f in frames),
        "status_counts": counts,
        "frames_with_ball": with_position,
        "ball_found_percent": round(100 * with_position / len(rows), 1) if rows else 0.0,
        "static_objects_ignored": len(static),
        "max_displacement_px": max_displacement,
        "tracking_strategy": BASELINE,
        "device": getattr(detector, "device", "unknown"),
        "model": MODEL_ID,
        "model_revision": getattr(detector, "revision", MODEL_REVISION),
        "timings_s": timings,
        "processing_s": round(time.perf_counter() - started, 1),
    }
    return output_path, stats
