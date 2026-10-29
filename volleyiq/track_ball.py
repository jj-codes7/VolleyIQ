"""Pick one ball per frame from automatic detections, with conservative rules.

Reads the <frame>.auto.json files written by volleyiq.detect_ball (the
detector is not run again) and, for every frame, either selects one ball
candidate or says why it can't. Nothing is ever guessed: a frame without an
acceptable candidate gets no position, not an interpolated one.

Usage:
    python -m volleyiq.track_ball DETECTIONS_DIR [--out DIR] [--max-displacement 120]
        [--strategy baseline|motion]

There are two strategies. "baseline" (the default) is described first;
"motion" changes only how candidates in range are ranked (see below).

The rules, in order, for each frame (frames are processed in time order):

  1. Static objects are ignored. A position where a "ball" is detected in at
     least --static-fraction of all frames (default 20%) is background, such
     as a wall fixture: the ball in play never stays put that long.
  2. With no active track, the highest-scoring remaining candidate starts a
     new track.
  3. With an active track, only candidates within --max-displacement pixels
     of the last selected position (times the number of sampled steps since
     then) are allowed. Among those, the one with the lowest
     distance / allowed + (1 - score) is selected, so closeness and
     confidence both count.
  4. After --max-lost consecutive frames without a selection the track is
     dropped, and the next candidate starts a new track (rule 2).

Each frame then gets a status:

  good       selected, AND part of an unbroken chain of at least --min-chain
             consecutive sampled frames, each step within the allowed
             displacement, AND it was the only candidate in range, AND its
             score is at least --min-good-score. High confidence alone is
             never enough.
  uncertain  selected, but at least one "good" condition fails. The reason
             column says which.
  lost       no position: the detector found no ball, or only static objects.
  bad        no position: the detector found ball candidate(s), but every one
             was too far from the track to be the same ball. These are
             usually false positives (arms, heads), but can be a real ball
             that moved faster than --max-displacement.

The "motion" strategy (experimental) keeps rules 1, 2 and 4, the same gate
in rule 3 and the same statuses, but ranks the candidates in range by
distance to a predicted position instead of the last position:

  - Prediction: constant velocity from the last two RELIABLE positions of
    the current track, in pixels per video frame, so gaps in frame numbers
    are handled. With fewer than two, or if they are more than
    (--max-lost + 1) sampled steps apart, there is no prediction and the last
    position is used, as in the baseline.
  - Reliable: selected, not ambiguous, score >= --min-good-score, and (when
    it starts a track) the only candidate. Other selections still move the
    gate but never feed the velocity, so one wrong pick can't bend the
    prediction. Lost and bad frames add nothing. Dropping a track (rule 4)
    forgets its velocity; a new track has to earn two reliable positions.
  - Ambiguity: with two or more candidates in range, the frame is ambiguous
    only if the second-best cost is within AMBIGUITY_MARGIN (0.25) of the
    best. The baseline calls every frame with two candidates in range
    ambiguous.
  - The gate is not centred on the prediction, on purpose: when the ball is
    hit and changes direction, the prediction can be far off, and the
    physical limit from the last position is the safer check.

Outputs, in the --out folder (which must be new or empty):
    ball_track.csv     one row per frame: status, position, score, diagnostics
                       (motion adds predicted position, margin and reliability)
    track_preview.mp4  the frames with the selected ball, its trail and status
    track_info.json    settings, static objects found, status counts
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import cv2

from volleyiq.annotate_ball import list_frames, load_image
from volleyiq.visualize_trajectory import frame_number_from_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "tracking"

DEFAULT_MAX_DISPLACEMENT = 120.0  # pixels per sampled step
DEFAULT_MAX_LOST = 2
DEFAULT_MIN_CHAIN = 3
DEFAULT_MIN_GOOD_SCORE = 0.5
DEFAULT_STATIC_FRACTION = 0.2
STATIC_RADIUS = 12.0  # pixels: detections this close count as the same spot
DEFAULT_FPS = 6.0  # every 5th frame of a ~30 FPS video
TRAIL_LENGTH = 12  # frames of trajectory drawn behind the ball

BASELINE, MOTION = "baseline", "motion"
STRATEGIES = (BASELINE, MOTION)
DEFAULT_STRATEGY = BASELINE
AMBIGUITY_MARGIN = 0.25  # motion: cost gap below which two candidates are a toss-up

GOOD, UNCERTAIN, LOST, BAD = "good", "uncertain", "lost", "bad"
STATUSES = (GOOD, UNCERTAIN, LOST, BAD)
CSV_COLUMNS = [
    "frame_name", "frame_number", "status", "reason", "x", "y", "score", "track_id",
    "chain_length", "n_candidates", "n_static", "n_in_range", "displacement_px",
    "steps_since_selected", "allowed_px", "nearest_rejected_px",
]
MOTION_COLUMNS = CSV_COLUMNS + [
    "predicted_x", "predicted_y", "prediction_error_px", "velocity_x", "velocity_y", "cost_margin", "reliable",
]


def columns_for(strategy):
    return MOTION_COLUMNS if strategy == MOTION else CSV_COLUMNS

# Colours are BGR (blue, green, red), which is OpenCV's order.
STATUS_COLOURS = {
    GOOD: (0, 200, 0),  # green
    UNCERTAIN: (0, 165, 255),  # orange
    LOST: (160, 160, 160),  # grey
    BAD: (0, 0, 255),  # red
}
LABEL_COLOUR = (255, 255, 255)
LABEL_TEXT = "AUTO tracking (model + rules) - not manual annotation"
LEGEND_TEXT = "green good | orange uncertain | grey lost | red bad"


def load_detections(detections_dir):
    """Read every <frame>.auto.json in `detections_dir`, sorted by frame number.

    Returns a list of dicts:
        {"frame_name", "frame_number", "candidates", "manual_reference"}
    where each candidate is {"centre": (x, y), "score", "box"} for a "ball"
    detection. Raises ValueError with a readable message on problems.
    """
    detections_dir = Path(detections_dir)
    if not detections_dir.is_dir():
        raise ValueError(f"Detections folder not found: {detections_dir}")
    paths = sorted(detections_dir.glob("*.auto.json"))
    if not paths:
        raise ValueError(f"No *.auto.json files in {detections_dir}. Run volleyiq.detect_ball first.")

    frames = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("source") != "automatic":
                raise ValueError("not an automatic detection file")
            candidates = [
                {"centre": tuple(d["centre"]), "score": float(d["score"]), "box": d["box"]}
                for d in data["detections"] if d["label"] == "ball"
            ]
            frames.append({
                "frame_name": data["frame_name"],
                "frame_number": frame_number_from_name(data["frame_name"]),
                "candidates": candidates,
                "manual_reference": data.get("manual_reference"),
            })
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"Cannot read detections from {path}: {error}") from None
    return sorted(frames, key=lambda f: f["frame_number"])


def find_static_positions(frames, fraction=DEFAULT_STATIC_FRACTION, radius=STATIC_RADIUS):
    """Return [{"centre", "frames"}] for spots detected in >= fraction of frames.

    fraction=0 turns the rule off.
    """
    if fraction <= 0 or not frames:
        return []
    needed = max(2, math.ceil(fraction * len(frames)))
    centres = [c["centre"] for f in frames for c in f["candidates"]]
    counted = []
    for centre in centres:
        hits = sum(1 for f in frames if any(math.dist(centre, c["centre"]) <= radius for c in f["candidates"]))
        if hits >= needed:
            counted.append((hits, centre))
    # One representative per spot: the most often seen centre.
    static = []
    for hits, centre in sorted(counted, key=lambda item: -item[0]):
        if not any(math.dist(centre, s["centre"]) <= radius for s in static):
            static.append({"centre": [round(v, 1) for v in centre], "frames": hits})
    return static


def is_static(candidate, static_positions, radius=STATIC_RADIUS):
    return any(math.dist(candidate["centre"], s["centre"]) <= radius for s in static_positions)


def _round(value):
    return None if value is None else round(value, 1)


def predict_position(anchors, frame_number, max_gap):
    """Constant-velocity prediction from the last two reliable positions.

    `anchors` are {"centre", "frame_number"} dicts in time order. Velocity is
    in pixels per video frame, so it works across missing frames. Returns
    (predicted (x, y), velocity (vx, vy)), or (None, None) if there aren't
    two anchors at most `max_gap` video frames apart.
    """
    if len(anchors) < 2:
        return None, None
    first, second = anchors[-2], anchors[-1]
    gap = second["frame_number"] - first["frame_number"]
    if not 0 < gap <= max_gap:
        return None, None
    velocity = tuple((b - a) / gap for a, b in zip(first["centre"], second["centre"]))
    ahead = frame_number - second["frame_number"]
    return tuple(p + v * ahead for p, v in zip(second["centre"], velocity)), velocity


def track(frames, max_displacement=DEFAULT_MAX_DISPLACEMENT, max_lost=DEFAULT_MAX_LOST,
          min_chain=DEFAULT_MIN_CHAIN, min_good_score=DEFAULT_MIN_GOOD_SCORE, static_positions=(),
          strategy=DEFAULT_STRATEGY):
    """Apply the rules in the module docstring. Returns one row dict per frame."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")
    numbers = [f["frame_number"] for f in frames]
    gaps = [b - a for a, b in zip(numbers, numbers[1:]) if b > a]
    sample_step = statistics.median(gaps) if gaps else 1
    max_anchor_gap = (max_lost + 1) * sample_step

    rows, last, missed, track_id = [], None, 0, 0
    anchors = []  # motion: reliable positions of the current track
    for frame in frames:
        usable = [c for c in frame["candidates"] if not is_static(c, static_positions)]
        row = {
            "frame_name": frame["frame_name"], "frame_number": frame["frame_number"],
            "status": None, "reason": "", "x": None, "y": None, "score": None, "track_id": None,
            "chain_length": None, "n_candidates": len(frame["candidates"]),
            "n_static": len(frame["candidates"]) - len(usable), "n_in_range": 0,
            "displacement_px": None, "steps_since_selected": None, "allowed_px": None,
            "nearest_rejected_px": None, "ambiguous": False,
            "manual_reference": frame.get("manual_reference"),
            "predicted_x": None, "predicted_y": None, "prediction_error_px": None,
            "velocity_x": None, "velocity_y": None, "cost_margin": None, "reliable": None,
        }
        if last is not None and missed >= max_lost:
            last, anchors = None, []  # rule 4: drop the track and its velocity

        chosen = None
        if last is None:
            if usable:  # rule 2: start a new track
                chosen = max(usable, key=lambda c: c["score"])
                track_id += 1
                row["n_in_range"] = len(usable)
                row["ambiguous"] = len(usable) > 1
        else:  # rule 3: stay close to the last selected position
            steps = max(1, round((frame["frame_number"] - last["frame_number"]) / sample_step))
            allowed = max_displacement * steps
            in_range = [c for c in usable if math.dist(c["centre"], last["centre"]) <= allowed]
            row.update(steps_since_selected=steps, allowed_px=_round(allowed), n_in_range=len(in_range))
            reference = last["centre"]
            if strategy == MOTION:
                predicted, velocity = predict_position(anchors, frame["frame_number"], max_anchor_gap)
                if predicted is not None:
                    reference = predicted
                    row.update(predicted_x=_round(predicted[0]), predicted_y=_round(predicted[1]),
                               velocity_x=round(velocity[0], 2), velocity_y=round(velocity[1], 2))
            if in_range:
                ranked = sorted(in_range, key=lambda c: math.dist(c["centre"], reference) / allowed + 1 - c["score"])
                chosen = ranked[0]
                row["displacement_px"] = _round(math.dist(chosen["centre"], last["centre"]))
                row["ambiguous"] = len(in_range) > 1
                if strategy == MOTION and len(ranked) > 1:
                    costs = [math.dist(c["centre"], reference) / allowed + 1 - c["score"] for c in ranked[:2]]
                    row["cost_margin"] = round(costs[1] - costs[0], 3)
                    row["ambiguous"] = row["cost_margin"] < AMBIGUITY_MARGIN
                if row["predicted_x"] is not None:
                    row["prediction_error_px"] = _round(math.dist(chosen["centre"], reference))
            elif usable:
                nearest = min(math.dist(c["centre"], last["centre"]) for c in usable)
                row.update(status=BAD, nearest_rejected_px=_round(nearest), reason=(
                    f"{len(usable)} candidate(s) rejected: nearest {nearest:.0f} px away, "
                    f"more than {allowed:.0f} px allowed"))

        if chosen is not None:
            row.update(x=_round(chosen["centre"][0]), y=_round(chosen["centre"][1]),
                       score=chosen["score"], track_id=track_id)
            last, missed = {"centre": chosen["centre"], "frame_number": frame["frame_number"]}, 0
            if strategy == MOTION:
                if row["steps_since_selected"] is None:
                    anchors = []  # a new track never inherits velocity
                row["reliable"] = not row["ambiguous"] and chosen["score"] >= min_good_score
                if row["reliable"]:
                    anchors.append(last)
        else:
            missed += 1
            if row["status"] is None:
                row["status"] = LOST
                row["reason"] = "only static-object detections" if frame["candidates"] else "no ball detected"
        rows.append(row)

    _assign_chains(rows)
    for row in rows:
        if row["track_id"] is not None:
            _classify_selected(row, min_chain, min_good_score)
    return rows


def _assign_chains(rows):
    """Give each selected row the length of its run of consecutive selected frames in one track."""
    start = 0
    for index in range(len(rows) + 1):
        linked = (
            index < len(rows) and index > 0
            and rows[index]["track_id"] is not None
            and rows[index]["track_id"] == rows[index - 1]["track_id"]
            and rows[index]["steps_since_selected"] == 1
        )
        if not linked:
            for row in rows[start:index]:
                if row["track_id"] is not None:
                    row["chain_length"] = index - start
            start = index


def _classify_selected(row, min_chain, min_good_score):
    problems = []
    if row["chain_length"] < min_chain:
        problems.append(f"chain of {row['chain_length']} frame(s), need {min_chain}")
    if row["ambiguous"]:
        margin = "" if row["cost_margin"] is None else f" (cost margin {row['cost_margin']:.2f} < {AMBIGUITY_MARGIN})"
        problems.append(f"{row['n_in_range']} candidates in range{margin}")
    if row["score"] < min_good_score:
        problems.append(f"score below {min_good_score}")
    row["status"] = UNCERTAIN if problems else GOOD
    row["reason"] = "; ".join(problems) if problems else f"continuous chain of {row['chain_length']} frames"


def summarise(rows):
    """Status counts, plus agreement with manual labels where the JSON carries them."""
    counts = {status: sum(1 for r in rows if r["status"] == status) for status in STATUSES}
    compared = []
    for row in rows:
        manual = row["manual_reference"]
        if manual:
            distance = None if row["x"] is None else round(math.dist((row["x"], row["y"]), (manual["x"], manual["y"])), 1)
            compared.append({"frame_name": row["frame_name"], "status": row["status"], "distance_px": distance})
    return {"frames": len(rows), "status_counts": counts, "manual_comparison": compared}


def write_csv(rows, path, columns=CSV_COLUMNS):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(["" if row[column] is None else row[column] for column in columns])


def _label(image, text, origin, colour, scale=0.5, thickness=1):
    """Write `text` on a dark box so it stays readable on any footage."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = min(max(0, origin[0]), max(0, image.shape[1] - text_w))
    y = max(text_h + 2, origin[1])
    cv2.rectangle(image, (x - 3, y - text_h - 4), (x + text_w + 3, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, colour, thickness, cv2.LINE_AA)


def draw_tracking(image, rows, index):
    """Draw frame `index`'s status, its selected ball and the recent trail on `image` in place."""
    row = rows[index]
    trail = rows[max(0, index - TRAIL_LENGTH + 1):index + 1]
    for previous, current in zip(trail, trail[1:]):
        # Join two points only when they are consecutive frames of one chain.
        if current["x"] is not None and previous["x"] is not None and current["track_id"] == previous["track_id"] \
                and current["steps_since_selected"] == 1:
            cv2.line(image, (int(previous["x"]), int(previous["y"])), (int(current["x"]), int(current["y"])),
                     STATUS_COLOURS[current["status"]], 2, cv2.LINE_AA)
    for past in trail[:-1]:
        if past["x"] is not None:
            filled = -1 if past["status"] == GOOD else 2
            cv2.circle(image, (int(past["x"]), int(past["y"])), 4, STATUS_COLOURS[past["status"]], filled, cv2.LINE_AA)

    colour = STATUS_COLOURS[row["status"]]
    if row.get("predicted_x") is not None:  # motion strategy only
        predicted = (int(row["predicted_x"]), int(row["predicted_y"]))
        cv2.drawMarker(image, predicted, LABEL_COLOUR, cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
    if row["x"] is not None:
        centre = (int(row["x"]), int(row["y"]))
        cv2.circle(image, centre, 18, colour, 3 if row["status"] == GOOD else 2, cv2.LINE_AA)
        _label(image, f"{row['status'].upper()} {row['score']:.2f}", (centre[0] + 22, centre[1] - 10), colour)

    _label(image, f"{row['status'].upper()}  {row['frame_name']}", (12, 34), colour, 0.9, 2)
    if row["reason"]:
        _label(image, row["reason"], (12, 66), LABEL_COLOUR, 0.55)

    height, width = image.shape[:2]
    for line, text in enumerate((LEGEND_TEXT, LABEL_TEXT)):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        _label(image, text, (width - text_w - 10, height - 10 - line * 22), LABEL_COLOUR, 0.45)
    return image


def frames_dir_from_run_info(detections_dir):
    """The frames folder recorded by detect_ball, or None."""
    info_path = Path(detections_dir) / "run_info.json"
    if not info_path.is_file():
        return None
    frames_dir = Path(json.loads(info_path.read_text(encoding="utf-8")).get("frames_dir") or "")
    for candidate in (frames_dir, PROJECT_ROOT / frames_dir):
        if str(frames_dir) and candidate.is_dir():
            return candidate
    return None


def write_preview(rows, frames_dir, output_path, fps=DEFAULT_FPS):
    """Write an MP4 of the tracked frames. Returns the number of frames written."""
    frames = {path.name: path for path in list_frames(frames_dir)}
    missing = [row["frame_name"] for row in rows if row["frame_name"] not in frames]
    if missing:
        raise ValueError(f"{len(missing)} frame(s) not found in {frames_dir}, e.g. {missing[0]}")

    writer, size = None, None
    try:
        for index, row in enumerate(rows):
            image = load_image(frames[row["frame_name"]])
            if writer is None:
                size = (image.shape[1], image.shape[0])
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
                if not writer.isOpened():
                    raise ValueError(f"Could not open a video writer for: {output_path}")
            if (image.shape[1], image.shape[0]) != size:
                raise ValueError(f"{row['frame_name']} is {image.shape[1]} x {image.shape[0]}, expected {size[0]} x {size[1]}")
            writer.write(draw_tracking(image, rows, index))
    finally:
        if writer is not None:
            writer.release()
    return len(rows)


def default_output_dir(detections_dir, max_displacement, strategy=DEFAULT_STRATEGY):
    # The baseline keeps its original folder name so earlier runs stay comparable.
    name = Path(detections_dir).name + ("" if strategy == BASELINE else f"_{strategy}")
    return DEFAULT_OUTPUT_ROOT / f"{name}_maxd{max_displacement:g}"


def check_output_dir(out_dir):
    """Refuse to write into a folder that already has files, so nothing is overwritten."""
    out_dir = Path(out_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"Output folder is not empty: {out_dir}\nChoose a new folder with --out.")


def check_settings(max_displacement, max_lost, min_chain, min_good_score, static_fraction, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"--strategy must be one of {', '.join(STRATEGIES)}, got {strategy!r}")
    if max_displacement <= 0:
        raise ValueError(f"--max-displacement must be positive, got {max_displacement}")
    if max_lost < 1 or min_chain < 1:
        raise ValueError("--max-lost and --min-chain must be at least 1")
    if not 0 <= static_fraction <= 1:
        raise ValueError(f"--static-fraction must be between 0 and 1, got {static_fraction}")
    if not 0 <= min_good_score <= 1:
        raise ValueError(f"--min-good-score must be between 0 and 1, got {min_good_score}")


def run_tracking(detections_dir, out_dir=None, max_displacement=DEFAULT_MAX_DISPLACEMENT,
                 max_lost=DEFAULT_MAX_LOST, min_chain=DEFAULT_MIN_CHAIN, min_good_score=DEFAULT_MIN_GOOD_SCORE,
                 static_fraction=DEFAULT_STATIC_FRACTION, frames_dir=None, fps=DEFAULT_FPS, video=True,
                 strategy=DEFAULT_STRATEGY):
    """Track, then write the CSV, the preview video (unless video=False) and track_info.json.

    Returns (rows, summary).
    """
    check_settings(max_displacement, max_lost, min_chain, min_good_score, static_fraction, strategy)
    if fps <= 0:
        raise ValueError(f"--fps must be positive, got {fps}")

    out_dir = Path(out_dir) if out_dir else default_output_dir(detections_dir, max_displacement, strategy)
    check_output_dir(out_dir)
    frames = load_detections(detections_dir)
    if video:
        frames_dir = frames_dir or frames_dir_from_run_info(detections_dir)
        if frames_dir is None:
            raise ValueError("Cannot find the frames folder for the preview. Pass --frames-dir (or --no-video).")

    static = find_static_positions(frames, static_fraction)
    rows = track(frames, max_displacement, max_lost, min_chain, min_good_score, static, strategy)
    summary = summarise(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "ball_track.csv", columns_for(strategy))
    if video:
        write_preview(rows, frames_dir, out_dir / "track_preview.mp4", fps)
    info = {
        "source": "automatic",
        "note": "Rule-based selection from automatic detections. Not manual annotations.",
        "detections_dir": str(detections_dir),
        "frames_dir": str(frames_dir) if video else None,
        "settings": {
            "strategy": strategy,
            "max_displacement": max_displacement, "max_lost": max_lost, "min_chain": min_chain,
            "min_good_score": min_good_score, "static_fraction": static_fraction,
            "static_radius": STATIC_RADIUS, "fps": fps,
        },
        "static_positions": static,
        **summary,
    }
    (out_dir / "track_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return rows, summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Select one ball per frame from automatic detections (rule-based, not manual annotation)."
    )
    parser.add_argument("detections_dir", help="folder of <frame>.auto.json files from volleyiq.detect_ball")
    parser.add_argument("--out", help="new or empty output folder (default: outputs/tracking/<detections>_maxd<N>)")
    parser.add_argument("--max-displacement", type=float, default=DEFAULT_MAX_DISPLACEMENT,
                        help=f"largest ball move in pixels between sampled frames (default {DEFAULT_MAX_DISPLACEMENT:g})")
    parser.add_argument("--max-lost", type=int, default=DEFAULT_MAX_LOST,
                        help=f"frames without a ball before the track is dropped (default {DEFAULT_MAX_LOST})")
    parser.add_argument("--min-chain", type=int, default=DEFAULT_MIN_CHAIN,
                        help=f"consecutive linked frames needed for 'good' (default {DEFAULT_MIN_CHAIN})")
    parser.add_argument("--min-good-score", type=float, default=DEFAULT_MIN_GOOD_SCORE,
                        help=f"lowest model score allowed for 'good' (default {DEFAULT_MIN_GOOD_SCORE})")
    parser.add_argument("--static-fraction", type=float, default=DEFAULT_STATIC_FRACTION,
                        help=f"ignore spots detected in at least this share of frames, 0 = off (default {DEFAULT_STATIC_FRACTION})")
    parser.add_argument("--frames-dir", help="extracted frames for the preview (default: from run_info.json)")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"preview video speed (default {DEFAULT_FPS:g})")
    parser.add_argument("--no-video", action="store_true", help="skip the preview video")
    parser.add_argument("--strategy", choices=STRATEGIES, default=DEFAULT_STRATEGY,
                        help=f"how to rank candidates: baseline = nearest to the last position, motion = nearest "
                             f"to a constant-velocity prediction (experimental). Default {DEFAULT_STRATEGY}")
    args = parser.parse_args(argv)

    out_dir = args.out or default_output_dir(args.detections_dir, args.max_displacement, args.strategy)
    try:
        _, summary = run_tracking(args.detections_dir, out_dir, args.max_displacement, args.max_lost, args.min_chain,
                                  args.min_good_score, args.static_fraction, args.frames_dir, args.fps,
                                  not args.no_video, args.strategy)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    counts = summary["status_counts"]
    print(f"Tracked {summary['frames']} frames: " + ", ".join(f"{counts[s]} {s}" for s in STATUSES))
    print(f"Saved {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
