"""Compare the ball-tracking strategies with the manual ball labels.

Runs every strategy in volleyiq.track_ball on the same automatic detections
(the detector is not run again) and scores each one against the hand labels
from volleyiq.annotate_ball.

Usage:
    python -m volleyiq.evaluate_tracking DETECTIONS_DIR MANUAL_CSV [--out DIR] [--max-displacement 120]

What counts as ground truth: only rows with status "labelled" and both x
and y. Frames marked "skipped" are NOT scored: a person may skip a frame
where the ball is only partly visible, so a tracker position there is not
necessarily wrong. They are reported separately. Rows that were never
annotated are ignored completely.

Definitions (per strategy):
    labelled_frames  labelled frames that also have a detection file
    matched          labelled frames where the tracker gave a position (any status)
    missed           labelled frames where it gave none (lost or bad)
    error_px         distance between the tracker's position and the hand label
    within_Npx       matched frames with error <= N pixels

Outputs, in the --out folder (which must be new or empty):
    baseline_track.csv, motion_track.csv  one per strategy, same columns as track_ball
    labelled_frames.csv                   each labelled frame: hand label and every strategy's answer
    comparison.json                       settings, metrics, and frames where the strategies differ
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

from volleyiq.annotate_ball import LABELLED, SKIPPED
from volleyiq.track_ball import (
    DEFAULT_MAX_DISPLACEMENT,
    DEFAULT_MAX_LOST,
    DEFAULT_MIN_CHAIN,
    DEFAULT_MIN_GOOD_SCORE,
    DEFAULT_STATIC_FRACTION,
    PROJECT_ROOT,
    STATUSES,
    STRATEGIES,
    check_output_dir,
    check_settings,
    columns_for,
    find_static_positions,
    load_detections,
    track,
    write_csv,
)

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
TOLERANCES = (5, 10, 15)


def read_manual_labels(csv_path):
    """Return ({frame_name: (x, y)} for labelled frames, {frame_name} for skipped frames)."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"Manual annotation CSV not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"frame_name", "x", "y", "status"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is not a VolleyIQ annotation CSV (missing columns: {', '.join(sorted(missing))}).")
        labelled, skipped = {}, set()
        for line_number, row in enumerate(reader, start=2):
            x, y = (row["x"] or "").strip(), (row["y"] or "").strip()
            status = (row["status"] or "").strip()
            if status == SKIPPED:
                skipped.add(row["frame_name"])
            elif x and y and status in ("", LABELLED):
                try:
                    labelled[row["frame_name"]] = (float(x), float(y))
                except ValueError:
                    raise ValueError(f"Bad coordinates on line {line_number} of {csv_path}: x={x!r}, y={y!r}") from None
    return labelled, skipped


def _stat(values, function):
    return round(function(values), 1) if values else None


def evaluate(rows, labelled, skipped):
    """Score one strategy's rows against the hand labels. Returns (metrics, per-frame errors)."""
    by_name = {row["frame_name"]: row for row in rows}
    scored = {}
    for name, point in labelled.items():
        row = by_name.get(name)
        if row is None:
            continue  # labelled, but no detection file for it
        error = None if row["x"] is None else round(math.dist((row["x"], row["y"]), point), 1)
        scored[name] = {"status": row["status"], "x": row["x"], "y": row["y"], "error_px": error}

    errors = [s["error_px"] for s in scored.values() if s["error_px"] is not None]
    good_errors = [s["error_px"] for s in scored.values() if s["status"] == "good"]
    on_skipped = [by_name[name]["status"] for name in skipped if name in by_name and by_name[name]["x"] is not None]
    metrics = {
        "labelled_frames": len(scored),
        "labelled_without_detections": len(labelled) - len(scored),
        "matched": len(errors),
        "missed": len(scored) - len(errors),
        "mean_error_px": _stat(errors, statistics.mean),
        "median_error_px": _stat(errors, statistics.median),
        "max_error_px": _stat(errors, max),
        **{f"within_{n}px": sum(1 for e in errors if e <= n) for n in TOLERANCES},
        "status_on_labelled": {status: sum(1 for s in scored.values() if s["status"] == status) for status in STATUSES},
        "good_on_labelled": len(good_errors),
        "good_mean_error_px": _stat(good_errors, statistics.mean),
        "skipped_frames": sum(1 for name in skipped if name in by_name),
        "positions_on_skipped_frames": {status: on_skipped.count(status) for status in STATUSES if on_skipped.count(status)},
        "status_counts_all_frames": {status: sum(1 for r in rows if r["status"] == status) for status in STATUSES},
    }
    return metrics, scored


def prediction_diagnostics(rows):
    """For the motion strategy: how close the chosen ball was to the prediction vs. the last position."""
    pairs = [(r["prediction_error_px"], r["displacement_px"]) for r in rows if r.get("prediction_error_px") is not None]
    if not pairs:
        return None
    predicted, last = [p for p, _ in pairs], [d for _, d in pairs]
    return {
        "frames_with_prediction": len(pairs),
        "median_px_from_prediction": _stat(predicted, statistics.median),
        "median_px_from_last_position": _stat(last, statistics.median),
        "max_px_from_prediction": _stat(predicted, max),
        "reliable_selections": sum(1 for r in rows if r.get("reliable") is True),
        "unreliable_selections": sum(1 for r in rows if r.get("reliable") is False),
    }


def differences(rows_by_strategy):
    """Frames where the strategies chose a different status or position."""
    names = list(rows_by_strategy)
    found = []
    for rows in zip(*rows_by_strategy.values()):
        answers = [(r["status"], r["x"], r["y"]) for r in rows]
        if len(set(answers)) > 1:
            found.append({"frame_name": rows[0]["frame_name"],
                          **{n: {"status": a[0], "x": a[1], "y": a[2]} for n, a in zip(names, answers)}})
    return found


def default_output_dir(detections_dir, max_displacement):
    return DEFAULT_OUTPUT_ROOT / f"{Path(detections_dir).name}_maxd{max_displacement:g}"


def run_evaluation(detections_dir, manual_csv, out_dir=None, max_displacement=DEFAULT_MAX_DISPLACEMENT,
                   max_lost=DEFAULT_MAX_LOST, min_chain=DEFAULT_MIN_CHAIN, min_good_score=DEFAULT_MIN_GOOD_SCORE,
                   static_fraction=DEFAULT_STATIC_FRACTION):
    """Run every strategy, score them and write the outputs. Returns the comparison dict."""
    for strategy in STRATEGIES:
        check_settings(max_displacement, max_lost, min_chain, min_good_score, static_fraction, strategy)
    out_dir = Path(out_dir) if out_dir else default_output_dir(detections_dir, max_displacement)
    check_output_dir(out_dir)
    labelled, skipped = read_manual_labels(manual_csv)
    if not labelled:
        raise ValueError(f"No labelled ball positions in {manual_csv}. Annotate some frames first.")
    frames = load_detections(detections_dir)
    static = find_static_positions(frames, static_fraction)

    rows_by_strategy, metrics, scored = {}, {}, {}
    for strategy in STRATEGIES:
        rows = track(frames, max_displacement, max_lost, min_chain, min_good_score, static, strategy)
        rows_by_strategy[strategy] = rows
        metrics[strategy], scored[strategy] = evaluate(rows, labelled, skipped)

    out_dir.mkdir(parents=True, exist_ok=True)
    for strategy, rows in rows_by_strategy.items():
        write_csv(rows, out_dir / f"{strategy}_track.csv", columns_for(strategy))

    with open(out_dir / "labelled_frames.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_name", "manual_x", "manual_y"]
                        + [f"{s}_{field}" for s in STRATEGIES for field in ("status", "x", "y", "error_px")])
        for name in sorted(scored[STRATEGIES[0]], key=lambda n: next(f["frame_number"] for f in frames if f["frame_name"] == n)):
            answers = [scored[s][name] for s in STRATEGIES]
            writer.writerow([name, *labelled[name]]
                            + ["" if a[field] is None else a[field] for a in answers for field in ("status", "x", "y", "error_px")])

    comparison = {
        "note": ("Scored only against frames a person labelled. Unlabelled frames are not ground truth; "
                 "skipped frames are reported but not scored. A small sample, not a general accuracy."),
        "detections_dir": str(detections_dir),
        "manual_csv": str(manual_csv),
        "frames": len(frames),
        "settings": {"max_displacement": max_displacement, "max_lost": max_lost, "min_chain": min_chain,
                     "min_good_score": min_good_score, "static_fraction": static_fraction},
        "static_positions": static,
        "metrics": metrics,
        "motion_prediction": prediction_diagnostics(rows_by_strategy.get("motion", [])),
        "differences": differences(rows_by_strategy),
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=1), encoding="utf-8")
    return comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare tracking strategies with manual ball labels.")
    parser.add_argument("detections_dir", help="folder of <frame>.auto.json files from volleyiq.detect_ball")
    parser.add_argument("manual_csv", help="manual annotation CSV from volleyiq.annotate_ball (only read)")
    parser.add_argument("--out", help="new or empty output folder (default: outputs/evaluation/<detections>_maxd<N>)")
    parser.add_argument("--max-displacement", type=float, default=DEFAULT_MAX_DISPLACEMENT,
                        help=f"largest ball move in pixels between sampled frames (default {DEFAULT_MAX_DISPLACEMENT:g})")
    args = parser.parse_args(argv)

    out_dir = args.out or default_output_dir(args.detections_dir, args.max_displacement)
    try:
        comparison = run_evaluation(args.detections_dir, args.manual_csv, out_dir, args.max_displacement)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"{'strategy':<10}{'labelled':>9}{'matched':>8}{'missed':>7}{'mean px':>9}{'median':>8}"
          f"{'<=5':>5}{'<=10':>6}{'<=15':>6}{'good':>6}")
    for strategy, m in comparison["metrics"].items():
        mean, median = ("-" if v is None else v for v in (m["mean_error_px"], m["median_error_px"]))
        print(f"{strategy:<10}{m['labelled_frames']:>9}{m['matched']:>8}{m['missed']:>7}{mean:>9}"
              f"{median:>8}{m['within_5px']:>5}{m['within_10px']:>6}{m['within_15px']:>6}"
              f"{m['good_on_labelled']:>6}")
    print(f"Frames where the strategies differ: {len(comparison['differences'])}")
    print(f"Saved {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
