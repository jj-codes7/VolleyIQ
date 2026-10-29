"""Checks for rule-based ball tracking, using made-up detections.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

No model, weights or network are needed: the detection JSON files are written
by the tests in the same format volleyiq.detect_ball uses. Everything is
written to a temporary folder that is deleted afterwards.
"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from volleyiq.track_ball import (
    BAD,
    CSV_COLUMNS,
    GOOD,
    LOST,
    STATUS_COLOURS,
    UNCERTAIN,
    draw_tracking,
    find_static_positions,
    load_detections,
    run_tracking,
    track,
)

STEP = 5  # frame numbers go 0, 5, 10, ... like extract_frames --every 5
MAX_D = 50.0


def detection(x, y, score=0.9, label="ball"):
    return {"label": label, "score": score, "box": [x - 10, y - 10, x + 10, y + 10],
            "centre": [x, y], "source": "automatic"}


def frame_json(index, candidates, manual=None):
    """One frame in detect_ball's JSON format; candidates are (x, y) or (x, y, score)."""
    return {
        "frame_name": f"frame_{index * STEP:06d}.png",
        "source": "automatic",
        "image_size": [320, 240],
        "detections": [detection(*c) for c in candidates],
        "manual_reference": manual,
    }


def make_frames(per_frame):
    """In-memory frames for track(), from a list of candidate lists."""
    return [
        {"frame_name": f"frame_{i * STEP:06d}.png", "frame_number": i * STEP,
         "candidates": [{"centre": (c[0], c[1]), "score": c[2] if len(c) > 2 else 0.9, "box": None} for c in cands]}
        for i, cands in enumerate(per_frame)
    ]


def statuses(rows):
    return [r["status"] for r in rows]


def moving_ball(count, start=(20, 100), step=(20, 0)):
    return [[(start[0] + i * step[0], start[1] + i * step[1])] for i in range(count)]


class TrackRulesTests(unittest.TestCase):
    def run_track(self, per_frame, **options):
        options.setdefault("max_displacement", MAX_D)
        return track(make_frames(per_frame), **options)

    def test_steady_ball_is_good(self):
        rows = self.run_track(moving_ball(5))
        self.assertEqual(statuses(rows), [GOOD] * 5)
        self.assertEqual([r["x"] for r in rows], [20, 40, 60, 80, 100])
        self.assertEqual(rows[1]["displacement_px"], 20.0)
        self.assertEqual({r["track_id"] for r in rows}, {1})

    def test_high_confidence_alone_is_not_good(self):
        rows = self.run_track([[], [(100, 100, 0.99)], []])
        self.assertEqual(statuses(rows), [LOST, UNCERTAIN, LOST])
        self.assertIn("chain of 1", rows[1]["reason"])

    def test_two_frame_chain_is_uncertain(self):
        rows = self.run_track(moving_ball(2))
        self.assertEqual(statuses(rows), [UNCERTAIN, UNCERTAIN])

    def test_low_score_is_uncertain(self):
        per_frame = moving_ball(4)
        per_frame[2] = [(60, 100, 0.45)]
        rows = self.run_track(per_frame)
        self.assertEqual(statuses(rows), [GOOD, GOOD, UNCERTAIN, GOOD])
        self.assertIn("score below", rows[2]["reason"])

    def test_missing_detection_gives_no_position(self):
        per_frame = moving_ball(6)
        per_frame[3] = []
        rows = self.run_track(per_frame)
        self.assertEqual(rows[3]["status"], LOST)
        self.assertEqual(rows[3]["reason"], "no ball detected")
        self.assertIsNone(rows[3]["x"])  # nothing interpolated
        # The next frame is 2 steps on, so twice the displacement is allowed,
        # but the chain is broken: 3 frames before the gap, 2 after.
        self.assertEqual(rows[4]["steps_since_selected"], 2)
        self.assertEqual(rows[4]["allowed_px"], 2 * MAX_D)
        self.assertEqual([r["chain_length"] for r in rows], [3, 3, 3, None, 2, 2])
        self.assertEqual(statuses(rows), [GOOD, GOOD, GOOD, LOST, UNCERTAIN, UNCERTAIN])

    def test_competing_candidate_far_away_is_ignored(self):
        per_frame = moving_ball(4)
        per_frame[2] = [(60, 100, 0.6), (300, 20, 0.99)]  # far, higher score
        rows = self.run_track(per_frame)
        self.assertEqual((rows[2]["x"], rows[2]["y"]), (60, 100))
        self.assertEqual(rows[2]["n_in_range"], 1)
        self.assertEqual(rows[2]["status"], GOOD)

    def test_competing_candidates_in_range_are_uncertain(self):
        per_frame = moving_ball(4)
        per_frame[2] = [(60, 100, 0.8), (75, 120, 0.85)]  # both in range; the first is closer
        rows = self.run_track(per_frame)
        self.assertEqual((rows[2]["x"], rows[2]["y"]), (60, 100))
        self.assertEqual(rows[2]["n_in_range"], 2)
        self.assertEqual(rows[2]["status"], UNCERTAIN)
        self.assertIn("2 candidates in range", rows[2]["reason"])

    def test_confidence_breaks_ties_in_range(self):
        per_frame = moving_ball(3) + [[(80, 100, 0.41), (80, 110, 0.95)]]
        rows = self.run_track(per_frame)
        self.assertEqual(rows[3]["y"], 110)

    def test_large_jump_is_bad(self):
        per_frame = moving_ball(3) + [[(250, 200)]] + [[(80, 100)], [(100, 100)]]
        rows = self.run_track(per_frame)
        self.assertEqual(rows[3]["status"], BAD)
        self.assertIsNone(rows[3]["x"])
        self.assertEqual(rows[3]["nearest_rejected_px"], 214.7)
        self.assertIn("rejected", rows[3]["reason"])
        # The track survives one bad frame and continues from (60, 100).
        self.assertEqual((rows[4]["x"], rows[4]["track_id"]), (80, 1))

    def test_recovery_after_lost_frames_starts_new_track(self):
        per_frame = moving_ball(3) + [[], []] + moving_ball(4, start=(250, 30))
        rows = self.run_track(per_frame, max_lost=2)
        self.assertEqual(statuses(rows[3:5]), [LOST, LOST])
        self.assertEqual(rows[5]["track_id"], 2)  # far away, but the old track was dropped
        self.assertIsNone(rows[5]["steps_since_selected"])
        self.assertEqual((rows[5]["x"], rows[5]["y"]), (250, 30))
        self.assertEqual(statuses(rows[5:]), [GOOD] * 4)  # confirmed by the chain that follows

    def test_far_candidate_before_track_is_dropped_is_bad(self):
        per_frame = moving_ball(3) + [[], [(250, 30)]]
        rows = self.run_track(per_frame, max_lost=2)
        self.assertEqual(statuses(rows[3:]), [LOST, BAD])

    def test_static_object_ignored(self):
        per_frame = [cands + [(300, 200, 0.95)] for cands in moving_ball(10)]
        per_frame[4] = [(300, 200, 0.95)]  # only the static object
        frames = make_frames(per_frame)
        static = find_static_positions(frames, fraction=0.2)
        self.assertEqual(static, [{"centre": [300, 200], "frames": 10}])
        rows = track(frames, MAX_D, static_positions=static)
        self.assertEqual(rows[4]["status"], LOST)
        self.assertEqual(rows[4]["reason"], "only static-object detections")
        self.assertEqual(rows[0]["n_static"], 1)
        self.assertTrue(all(r["x"] != 300 for r in rows))

    def test_static_rule_needs_enough_frames(self):
        per_frame = [[(300, 200)] if i < 2 else [] for i in range(10)]
        self.assertEqual(find_static_positions(make_frames(per_frame), fraction=0.5), [])
        self.assertEqual(find_static_positions(make_frames(per_frame), fraction=0), [])


class DrawTests(unittest.TestCase):
    def test_selected_ball_ring_uses_status_colour(self):
        rows = track(make_frames(moving_ball(3, start=(100, 100))), MAX_D)
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        draw_tracking(image, rows, 2)
        self.assertTrue(np.all(image[100, 140 - 18] == STATUS_COLOURS[GOOD]))  # left of the ring


class RunTrackingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.detections = self.tmp / "detections"
        self.frames = self.tmp / "frames"
        self.detections.mkdir()
        self.frames.mkdir()
        per_frame = moving_ball(4, start=(40, 120)) + [[]] + [[(300, 20, 0.99)]]
        for i, cands in enumerate(per_frame):
            data = frame_json(i, cands, manual={"x": 41, "y": 120} if i == 0 else None)
            data["detections"].append(detection(10, 10, label="person"))  # people are ignored
            (self.detections / f"frame_{i * STEP:06d}.auto.json").write_text(json.dumps(data), encoding="utf-8")
            cv2.imwrite(str(self.frames / f"frame_{i * STEP:06d}.png"), np.full((240, 320, 3), 40, dtype=np.uint8))
        (self.detections / "run_info.json").write_text(json.dumps({"frames_dir": str(self.frames)}), encoding="utf-8")
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def hashes(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.detections.iterdir()}

    def test_writes_csv_video_and_info_without_touching_detections(self):
        before = self.hashes()
        rows, summary = run_tracking(self.detections, self.out, max_displacement=MAX_D)
        self.assertEqual(self.hashes(), before)

        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["ball_track.csv", "track_info.json", "track_preview.mp4"])
        with open(self.out / "ball_track.csv", newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(list(csv_rows[0].keys()), CSV_COLUMNS)
        self.assertEqual([r["status"] for r in csv_rows], [GOOD] * 4 + [LOST, BAD])
        self.assertEqual((csv_rows[0]["x"], csv_rows[0]["y"], csv_rows[0]["score"]), ("40", "120", "0.9"))
        self.assertEqual((csv_rows[4]["x"], csv_rows[5]["x"]), ("", ""))

        capture = cv2.VideoCapture(str(self.out / "track_preview.mp4"))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 6)
        capture.release()

        info = json.loads((self.out / "track_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["status_counts"], {GOOD: 4, UNCERTAIN: 0, LOST: 1, BAD: 1})
        self.assertEqual(info["manual_comparison"], [{"frame_name": "frame_000000.png", "status": GOOD, "distance_px": 1.0}])
        self.assertEqual(info["settings"]["max_displacement"], MAX_D)

    def test_no_video(self):
        run_tracking(self.detections, self.out, video=False)
        self.assertFalse((self.out / "track_preview.mp4").exists())

    def test_refuses_non_empty_output_folder(self):
        self.out.mkdir()
        (self.out / "keep_me.txt").write_text("existing output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not empty"):
            run_tracking(self.detections, self.out)
        self.assertEqual((self.out / "keep_me.txt").read_text(encoding="utf-8"), "existing output")

    def test_missing_frames_folder(self):
        (self.detections / "run_info.json").write_text(json.dumps({"frames_dir": "nowhere"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frames folder"):
            run_tracking(self.detections, self.out)
        self.assertFalse(self.out.exists())  # nothing written on error

    def test_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            load_detections(self.tmp / "missing")
        with self.assertRaisesRegex(ValueError, "No \\*.auto.json"):
            load_detections(self.frames)
        with self.assertRaisesRegex(ValueError, "max-displacement"):
            run_tracking(self.detections, self.out, max_displacement=0)
        not_auto = self.tmp / "manual_like"
        not_auto.mkdir()
        data = frame_json(0, [(1, 1)])
        data["source"] = "manual"
        (not_auto / "frame_000000.auto.json").write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not an automatic detection file"):
            load_detections(not_auto)


if __name__ == "__main__":
    unittest.main()
