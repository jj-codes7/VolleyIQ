"""Checks for the experimental motion-aware tracking strategy, using made-up detections.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

No model, weights or network are needed. The baseline strategy's own checks
are in test_track_ball.py; these compare it with the motion strategy.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from volleyiq.track_ball import (
    BAD,
    BASELINE,
    CSV_COLUMNS,
    GOOD,
    LOST,
    MOTION,
    MOTION_COLUMNS,
    UNCERTAIN,
    default_output_dir,
    predict_position,
    run_tracking,
    track,
)

STEP = 5  # frame numbers go 0, 5, 10, ... like extract_frames --every 5
MAX_D = 50.0


def make_frames(per_frame):
    """Frames for track(), from a list of candidate lists; a candidate is (x, y) or (x, y, score)."""
    return [
        {"frame_name": f"frame_{i * STEP:06d}.png", "frame_number": i * STEP,
         "candidates": [{"centre": (c[0], c[1]), "score": c[2] if len(c) > 2 else 0.9, "box": None} for c in cands]}
        for i, cands in enumerate(per_frame)
    ]


def moving_ball(count, start=(20, 100), step=(20, 0)):
    return [[(start[0] + i * step[0], start[1] + i * step[1])] for i in range(count)]


def run_both(per_frame, **options):
    options.setdefault("max_displacement", MAX_D)
    frames = make_frames(per_frame)
    return track(frames, strategy=BASELINE, **options), track(frames, strategy=MOTION, **options)


def position(row):
    return (row["x"], row["y"])


class PredictPositionTests(unittest.TestCase):
    def test_velocity_uses_frame_numbers(self):
        anchors = [{"centre": (0, 0), "frame_number": 0}, {"centre": (20, 10), "frame_number": 10}]
        predicted, velocity = predict_position(anchors, 15, max_gap=15)
        self.assertEqual(velocity, (2.0, 1.0))  # pixels per video frame
        self.assertEqual(predicted, (30.0, 15.0))

    def test_no_prediction_without_two_recent_anchors(self):
        self.assertEqual(predict_position([], 5, 15), (None, None))
        self.assertEqual(predict_position([{"centre": (0, 0), "frame_number": 0}], 5, 15), (None, None))
        far_apart = [{"centre": (0, 0), "frame_number": 0}, {"centre": (20, 0), "frame_number": 20}]
        self.assertEqual(predict_position(far_apart, 25, 15), (None, None))


class MotionStrategyTests(unittest.TestCase):
    def test_steady_movement(self):
        baseline, motion = run_both(moving_ball(6))
        self.assertEqual([position(r) for r in motion], [position(r) for r in baseline])
        self.assertEqual([r["status"] for r in motion], [GOOD] * 6)
        self.assertIsNone(motion[1]["predicted_x"])  # only one reliable position so far
        self.assertEqual((motion[2]["predicted_x"], motion[2]["predicted_y"]), (60, 100))
        self.assertEqual((motion[2]["velocity_x"], motion[2]["velocity_y"]), (4.0, 0.0))  # 20 px per 5 frames
        self.assertEqual(motion[5]["prediction_error_px"], 0)
        self.assertTrue(all(r["reliable"] for r in motion))

    def test_changing_direction_is_not_rejected(self):
        # Right for 4 frames, then back left. The reversal lands 40 px from the
        # prediction (more than MAX_D=30 would allow around it) but only 20 px
        # from the last position, so the gate around the last position keeps it.
        per_frame = moving_ball(4) + [[(60, 100)], [(40, 100)], [(20, 100)]]
        _, motion = run_both(per_frame, max_displacement=30)
        self.assertEqual(motion[4]["prediction_error_px"], 40.0)
        self.assertEqual([position(r) for r in motion[4:]], [(60, 100), (40, 100), (20, 100)])
        self.assertEqual({r["track_id"] for r in motion}, {1})
        self.assertEqual([r["status"] for r in motion], [GOOD] * 7)

    def test_missing_frame_prediction_spans_the_gap(self):
        per_frame = moving_ball(6)
        per_frame[3] = []
        _, motion = run_both(per_frame)
        self.assertEqual(motion[3]["status"], LOST)
        self.assertIsNone(motion[3]["x"])  # still no invented position
        self.assertEqual((motion[4]["predicted_x"], motion[4]["steps_since_selected"]), (100, 2))
        self.assertEqual(motion[4]["prediction_error_px"], 0)

    def test_recovery_starts_without_old_velocity(self):
        per_frame = moving_ball(3) + [[], []] + moving_ball(4, start=(250, 30), step=(0, 20))
        _, motion = run_both(per_frame, max_lost=2)
        self.assertEqual(motion[5]["track_id"], 2)
        self.assertIsNone(motion[5]["predicted_x"])
        self.assertIsNone(motion[6]["predicted_x"])  # new track: only one reliable position yet
        self.assertEqual((motion[7]["predicted_x"], motion[7]["predicted_y"]), (250, 70))  # new track's own motion
        self.assertEqual([r["status"] for r in motion[5:]], [GOOD] * 4)

    def test_competing_candidates_prefer_the_predicted_one(self):
        # Fast ball, 40 px per step. In frame 3 a distractor sits 10 px from the
        # last position; the real ball is 40 px on, exactly where predicted.
        per_frame = moving_ball(3, start=(40, 100), step=(40, 0)) + [[(130, 100), (160, 100)]] + [[(200, 100)]]
        baseline, motion = run_both(per_frame)
        self.assertEqual(position(baseline[3]), (130, 100))
        self.assertEqual(position(motion[3]), (160, 100))
        self.assertEqual(motion[3]["cost_margin"], 0.6)
        self.assertEqual(motion[3]["status"], GOOD)  # clear margin, so not ambiguous
        self.assertEqual(baseline[3]["status"], UNCERTAIN)

    def test_close_competitors_stay_uncertain(self):
        per_frame = moving_ball(3) + [[(80, 95), (80, 105)]] + [[(100, 100)]]
        _, motion = run_both(per_frame)
        self.assertLess(motion[3]["cost_margin"], 0.25)
        self.assertEqual(motion[3]["status"], UNCERTAIN)
        self.assertIn("cost margin", motion[3]["reason"])
        self.assertFalse(motion[3]["reliable"])

    def test_sudden_jump_is_bad_and_ignored(self):
        per_frame = moving_ball(3) + [[(300, 220)]] + [[(100, 100)]]
        _, motion = run_both(per_frame)
        self.assertEqual(motion[3]["status"], BAD)
        self.assertIsNone(motion[3]["reliable"])
        self.assertEqual((motion[4]["predicted_x"], motion[4]["predicted_y"]), (100, 100))
        self.assertEqual(position(motion[4]), (100, 100))

    def test_unreliable_pick_does_not_bend_velocity(self):
        # Frame 3: the only candidate is a low-score detection off the path.
        # It is selected (uncertain) but must not change the velocity. In frame
        # 4 a decoy sits where a contaminated velocity would point; the real
        # ball is where the clean velocity points.
        per_frame = moving_ball(3) + [[(80, 130, 0.45)]] + [[(100, 100), (100, 150)]]
        baseline, motion = run_both(per_frame)
        self.assertEqual(motion[3]["status"], UNCERTAIN)
        self.assertFalse(motion[3]["reliable"])
        self.assertEqual((motion[4]["predicted_x"], motion[4]["predicted_y"]), (100, 100))
        self.assertEqual(position(motion[4]), (100, 100))
        self.assertEqual(position(baseline[4]), (100, 150))  # nearest to the last (bad) pick

    def test_unknown_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
            track(make_frames(moving_ball(2)), strategy="kalman")


class MotionOutputTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.detections = self.tmp / "detections"
        self.detections.mkdir()
        for frame in make_frames(moving_ball(4)):
            data = {"frame_name": frame["frame_name"], "source": "automatic", "detections": [
                {"label": "ball", "score": 0.9, "box": None, "centre": list(c["centre"]), "source": "automatic"}
                for c in frame["candidates"]]}
            (self.detections / frame["frame_name"].replace(".png", ".auto.json")).write_text(json.dumps(data), encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def columns(self, out):
        with open(out / "ball_track.csv", newline="", encoding="utf-8") as handle:
            return next(csv.reader(handle))

    def test_each_strategy_keeps_its_own_columns(self):
        run_tracking(self.detections, self.tmp / "base", video=False)
        run_tracking(self.detections, self.tmp / "motion", video=False, strategy=MOTION)
        self.assertEqual(self.columns(self.tmp / "base"), CSV_COLUMNS)
        self.assertEqual(self.columns(self.tmp / "motion"), MOTION_COLUMNS)
        info = json.loads((self.tmp / "motion" / "track_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["settings"]["strategy"], MOTION)

    def test_default_folders_differ(self):
        self.assertEqual(default_output_dir(self.detections, 120).name, "detections_maxd120")
        self.assertEqual(default_output_dir(self.detections, 120, MOTION).name, "detections_motion_maxd120")

    def test_bad_strategy_rejected_before_writing(self):
        with self.assertRaisesRegex(ValueError, "--strategy"):
            run_tracking(self.detections, self.tmp / "out", video=False, strategy="kalman")
        self.assertFalse((self.tmp / "out").exists())


if __name__ == "__main__":
    unittest.main()
