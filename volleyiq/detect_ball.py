"""Automatically detect the volleyball (and people) in extracted frames.

Uses the pretrained detector grdesignbuild/volleyball-person-ball-detector
(a D-FINE-S fine-tune, Apache-2.0), pinned to one exact model revision. Every
box this tool writes is model output, not a human label: files are named
*.auto.*, every record says "source": "automatic", and results go to
outputs/detections/, never to data/annotations/.

This is per-frame object detection only. It does not track the ball between
frames and does not change or fill in the manual annotation CSV.

Usage:
    python -m volleyiq.detect_ball FRAMES_DIR [--frames NAME ...] [--limit N]
        [--threshold 0.4] [--tile 864] [--manual-csv CSV] [--out DIR]

Tiling: the ball is small, so large frames are cut into overlapping square
tiles plus one whole-frame pass, and the results are merged. --tile 0 runs
the whole frame only. The tiling and merging rules follow predict.py from the
model repository (Apache-2.0).

Outputs, in the --out folder (which must be new or empty):
    <frame>.auto.json         detections for one frame
    <frame>.auto.jpg          the frame with AUTO boxes (and MANUAL marks, if a CSV is given)
    ball_detections_auto.csv  one row per detected ball, plus a blank row for frames with none
    run_info.json             model, revision, settings and timings
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from volleyiq.annotate_ball import list_frames, load_image
from volleyiq.visualize_trajectory import read_annotations, save_image

MODEL_ID = "grdesignbuild/volleyball-person-ball-detector"
# The exact commit inspected and tested; change only after re-checking the model.
MODEL_REVISION = "92693c99a63841b1b121df86607bb4f839bbe107"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Weights are cached inside the project (git-ignored), not in the user's home folder.
DEFAULT_CACHE_DIR = PROJECT_ROOT / "models" / "hf_cache"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "detections"

DEFAULT_THRESHOLD = 0.4
DEFAULT_TILE = 864
TILE_OVERLAP = 0.25
NMS_IOU = 0.5
CONTAIN = 0.85  # drop a box this much inside a higher-scoring one
CUT_COVER = 0.4  # drop a tile-edge box this much covered by a kept one
EDGE_MARGIN = 3  # pixels: a box this close to an inner tile edge is "cut"

SOURCE = "automatic"
BALL, PERSON = "ball", "person"
CSV_COLUMNS = ["frame_name", "x", "y", "score", "x1", "y1", "x2", "y2", "source"]

# Colours are BGR (blue, green, red), which is OpenCV's order.
BALL_COLOUR = (255, 0, 255)  # magenta
PERSON_COLOUR = (0, 200, 0)  # green
MANUAL_COLOUR = (255, 255, 0)  # cyan
LABEL_COLOUR = (255, 255, 255)
AUTO_LABEL_TEXT = "AUTO boxes = automatic model detection, not manual annotation"
MANUAL_LABEL_TEXT = "MANUAL cross = hand label from CSV, shown for comparison"


def tiles_for(width, height, tile, overlap=TILE_OVERLAP):
    """Return the (x1, y1, x2, y2) windows to run the model on.

    A frame not much bigger than one tile (or tile=0) is run whole. Otherwise
    it is cut into overlapping tile x tile windows covering every pixel, plus
    one whole-frame pass for large, near objects.
    """
    if tile == 0 or max(width, height) <= tile * 1.15:
        return [(0, 0, width, height)]
    stride = int(tile * (1 - overlap))

    def starts(length):
        last = max(0, length - tile)
        values = list(range(0, last + 1, stride))
        if values[-1] != last:
            values.append(last)
        return values

    windows = [(x, y, min(width, x + tile), min(height, y + tile)) for y in starts(height) for x in starts(width)]
    windows.append((0, 0, width, height))
    return windows


def _cover(box, other):
    """Fraction of `box`'s area that lies inside `other`."""
    ix = max(0, min(box[2], other[2]) - max(box[0], other[0]))
    iy = max(0, min(box[3], other[3]) - max(box[1], other[1]))
    return ix * iy / max(1e-6, (box[2] - box[0]) * (box[3] - box[1]))


def merge(raw, id2label, nms_iou=NMS_IOU):
    """Merge detections from all windows into one list for the frame.

    `raw` is a list of (x1, y1, x2, y2, score, label_id, cut) in frame
    pixels, where `cut` means the box touched an inner tile edge. Per class:
    non-maximum suppression over uncut boxes, then drop boxes mostly inside a
    higher-scoring one, then keep a cut box only if nothing already covers it.
    Returns dicts sorted by score, highest first.
    """
    final = []
    for label_id in sorted({d[5] for d in raw}):
        same = [d for d in raw if d[5] == label_id]
        uncut = sorted((d for d in same if not d[6]), key=lambda d: -d[4])
        cut = sorted((d for d in same if d[6]), key=lambda d: -d[4])
        kept = []
        if uncut:
            boxes = torch.tensor([d[:4] for d in uncut], dtype=torch.float32)
            scores = torch.tensor([d[4] for d in uncut], dtype=torch.float32)
            for index in torchvision.ops.nms(boxes, scores, nms_iou).tolist():
                if not any(_cover(uncut[index], k) > CONTAIN for k in kept):
                    kept.append(uncut[index])
        for d in cut:
            if not any(_cover(d, k) > CUT_COVER for k in kept):
                kept.append(d)
        for x1, y1, x2, y2, score, _, _ in kept:
            final.append(make_detection(id2label[label_id], score, (x1, y1, x2, y2)))
    return sorted(final, key=lambda d: -d["score"])


def make_detection(label, score, box):
    x1, y1, x2, y2 = (round(float(v), 1) for v in box)
    return {
        "label": label,
        "score": round(float(score), 3),
        "box": [x1, y1, x2, y2],
        "centre": [round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)],
        "source": SOURCE,
    }


def detect(image, predict_crops, id2label, tile=DEFAULT_TILE, threshold=DEFAULT_THRESHOLD, batch_size=8):
    """Detect objects in one RGB image (H x W x 3 array).

    `predict_crops(crops, threshold)` runs the model on a list of image
    crops and returns, for each crop, (scores, label_ids, boxes) with boxes
    in that crop's pixels. Passing it in keeps this logic testable without
    the model weights.
    """
    height, width = image.shape[:2]
    windows = tiles_for(width, height, tile)
    raw = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start:start + batch_size]
        crops = [np.ascontiguousarray(image[y1:y2, x1:x2]) for x1, y1, x2, y2 in chunk]
        for (x1, y1, x2, y2), (scores, labels, boxes) in zip(chunk, predict_crops(crops, threshold)):
            whole = (x1, y1, x2, y2) == (0, 0, width, height)
            tile_w, tile_h = x2 - x1, y2 - y1
            for score, label_id, box in zip(scores, labels, boxes):
                if score < threshold:
                    continue
                cut = not whole and (
                    (box[0] < EDGE_MARGIN and x1 > 0)
                    or (box[1] < EDGE_MARGIN and y1 > 0)
                    or (box[2] > tile_w - EDGE_MARGIN and x2 < width)
                    or (box[3] > tile_h - EDGE_MARGIN and y2 < height)
                )
                raw.append((box[0] + x1, box[1] + y1, box[2] + x1, box[3] + y1, score, int(label_id), cut))
    return merge(raw, id2label)


class BallDetector:
    """The pretrained model, loaded once and pinned to MODEL_REVISION."""

    def __init__(self, revision=MODEL_REVISION, cache_dir=DEFAULT_CACHE_DIR, device="auto"):
        # Imported here so the rest of this module (and its tests) loads quickly.
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.revision = revision
        options = {"revision": revision, "cache_dir": str(cache_dir)}
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, backend="torchvision", **options)
        self.model = AutoModelForObjectDetection.from_pretrained(MODEL_ID, **options).to(device).eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict_crops(self, crops, threshold):
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        sizes = torch.tensor([crop.shape[:2] for crop in crops])
        results = self.processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=sizes)
        return [(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()) for r in results]


def _put_label(image, text, origin, colour, scale=0.5):
    """Write `text` with a dark background so it stays readable on any footage."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, 1)
    x = min(max(0, origin[0]), max(0, image.shape[1] - text_w))
    y = max(text_h + 2, origin[1])
    cv2.rectangle(image, (x, y - text_h - 2), (x + text_w, y + baseline), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, colour, 1, cv2.LINE_AA)


def draw_detections(image, detections, manual_point=None):
    """Draw AUTO boxes (and an optional MANUAL cross) on `image` in place."""
    for d in sorted(detections, key=lambda d: d["label"] == BALL):  # balls on top
        x1, y1, x2, y2 = (int(round(v)) for v in d["box"])
        colour = BALL_COLOUR if d["label"] == BALL else PERSON_COLOUR
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2 if d["label"] == BALL else 1)
        _put_label(image, f"AUTO {d['label']} {d['score']:.2f}", (x1, y1 - 4), colour, 0.5 if d["label"] == BALL else 0.4)

    lines = [AUTO_LABEL_TEXT]
    if manual_point is not None:
        cv2.drawMarker(image, manual_point, MANUAL_COLOUR, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        _put_label(image, "MANUAL", (manual_point[0] + 10, manual_point[1] + 22), MANUAL_COLOUR)
        lines.append(MANUAL_LABEL_TEXT)

    # Key in the bottom-right corner, like the manual trajectory images.
    height, width = image.shape[:2]
    for row, text in enumerate(reversed(lines)):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _put_label(image, text, (width - text_w - 8, height - 8 - row * 20), LABEL_COLOUR, 0.45)
    return image


def nearest_ball_distance(detections, point):
    """Pixel distance from `point` to the closest AUTO ball centre, or None."""
    balls = [d for d in detections if d["label"] == BALL]
    if point is None or not balls:
        return None
    return round(min(math.dist(point, d["centre"]) for d in balls), 1)


def default_output_dir(frames_dir, tile, threshold):
    return DEFAULT_OUTPUT_ROOT / f"{Path(frames_dir).name}_tile{tile}_t{threshold:.2f}"


def check_output_dir(out_dir):
    """Refuse to write into a folder that already has files, so nothing is overwritten."""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"Output folder is not empty: {out_dir}\nChoose a new folder with --out.")


def select_frames(frames_dir, frame_names=None, limit=None):
    frames = list_frames(frames_dir)
    if frame_names:
        by_name = {path.name: path for path in frames}
        missing = [name for name in frame_names if name not in by_name]
        if missing:
            raise ValueError(f"Frame(s) not found in {frames_dir}: {', '.join(missing)}")
        frames = [by_name[name] for name in frame_names]
    if limit is not None:
        if limit < 1:
            raise ValueError(f"--limit must be at least 1, got {limit}")
        frames = frames[:limit]
    return frames


def run_detection(frames_dir, out_dir=None, frame_names=None, limit=None, tile=DEFAULT_TILE,
                  threshold=DEFAULT_THRESHOLD, manual_csv=None, keep_person=False, detector=None):
    """Detect on the chosen frames and write the outputs described at the top.

    `detector` is anything with predict_crops(), id2label, device and
    revision; by default the pretrained BallDetector is loaded.
    Returns the list of per-frame result dicts.
    """
    if not 0 < threshold < 1:
        raise ValueError(f"--threshold must be between 0 and 1, got {threshold}")
    if tile != 0 and tile < 64:
        raise ValueError(f"--tile must be 0 (whole frame) or at least 64 pixels, got {tile}")
    out_dir = Path(out_dir) if out_dir else default_output_dir(frames_dir, tile, threshold)
    check_output_dir(out_dir)
    frames = select_frames(frames_dir, frame_names, limit)
    manual = {}
    if manual_csv:
        manual = {row["frame_name"]: row["point"] for row in read_annotations(manual_csv)}

    if detector is None:
        detector = BallDetector()
    out_dir.mkdir(parents=True, exist_ok=True)

    results, csv_rows = [], []
    for path in frames:
        image = load_image(path)  # BGR, for drawing
        started = time.perf_counter()
        detections = detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detector.predict_crops,
                            detector.id2label, tile, threshold)
        seconds = round(time.perf_counter() - started, 3)
        if not keep_person:
            detections = [d for d in detections if d["label"] == BALL]

        manual_point = manual.get(path.name)
        result = {
            "frame_name": path.name,
            "source": SOURCE,
            "image_size": [image.shape[1], image.shape[0]],
            "model": MODEL_ID,
            "revision": detector.revision,
            "threshold": threshold,
            "tile": tile,
            "seconds": seconds,
            "detections": detections,
            # The hand label is reference data only, kept apart from the detections.
            "manual_reference": None if manual_point is None else {"x": manual_point[0], "y": manual_point[1]},
            "nearest_ball_distance_px": nearest_ball_distance(detections, manual_point),
        }
        results.append(result)

        stem = path.stem
        (out_dir / f"{stem}.auto.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        save_image(draw_detections(image, detections, manual_point), out_dir / f"{stem}.auto.jpg")

        balls = [d for d in detections if d["label"] == BALL]
        for d in balls:
            csv_rows.append([path.name, *d["centre"], d["score"], *d["box"], SOURCE])
        if not balls:
            csv_rows.append([path.name, "", "", "", "", "", "", "", SOURCE])

    with open(out_dir / "ball_detections_auto.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(csv_rows)

    run_info = {
        "source": SOURCE,
        "note": "Automatic per-frame detections from a pretrained model. Not manual annotations, not tracking.",
        "model": MODEL_ID,
        "revision": detector.revision,
        "licence": "Apache-2.0",
        "device": detector.device,
        "threshold": threshold,
        "tile": tile,
        "keep_person": keep_person,
        "frames_dir": str(frames_dir),
        "manual_csv": str(manual_csv) if manual_csv else None,
        "frames": [r["frame_name"] for r in results],
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, indent=1), encoding="utf-8")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Automatic volleyball detection with a pretrained model (not manual annotation, not tracking)."
    )
    parser.add_argument("frames_dir", help="folder of extracted frames")
    parser.add_argument("--frames", nargs="+", metavar="NAME", help="frame file names to run on (default: all)")
    parser.add_argument("--limit", type=int, help="run on at most this many frames")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"minimum model confidence to keep a box (default {DEFAULT_THRESHOLD})")
    parser.add_argument("--tile", type=int, default=DEFAULT_TILE,
                        help=f"tile size in pixels, 0 = whole frame only (default {DEFAULT_TILE})")
    parser.add_argument("--manual-csv", help="manual annotation CSV, drawn as MANUAL marks for comparison")
    parser.add_argument("--keep-person", action="store_true", help="also keep person detections")
    parser.add_argument("--out", help="new or empty output folder (default: outputs/detections/<frames>_tile<N>_t<threshold>)")
    args = parser.parse_args(argv)

    try:
        results = run_detection(args.frames_dir, args.out, args.frames, args.limit, args.tile,
                                args.threshold, args.manual_csv, args.keep_person)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    for r in results:
        scores = [d["score"] for d in r["detections"] if d["label"] == BALL]
        extra = "" if r["nearest_ball_distance_px"] is None else f"  {r['nearest_ball_distance_px']} px from manual"
        print(f"{r['frame_name']}: {r['seconds']:.2f}s  AUTO ball scores {scores}{extra}")
    out_dir = args.out or default_output_dir(args.frames_dir, args.tile, args.threshold)
    print(f"Saved automatic detections for {len(results)} frame(s) in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
