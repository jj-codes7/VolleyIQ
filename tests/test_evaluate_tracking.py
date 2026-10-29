"""Checks for comparing tracking strategies with manual labels, using made-up data.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

No model, weights or network are needed. Everything is written to a
temporary folder that is deleted afterwards.
"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from volleyiq.evaluate_tracking import evaluate, read_manual_labels, run_evaluation

STEP = 5


def name(index):
    return f"frame_{index * STEP:06d}.png"


def row(index, x=None, y=None, status="good"):
    return {"frame_name": name(index), "status": status if x is not None else "lost", "x": x, "y": y}


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_manual(self, text):
        path = self.tmp / "manual.csv"
        path.write_text(text, encoding="utf-8")
        return path


class ReadManualLabelsTests(TempDirTestCase):
    def test_only_labelled_rows_are_ground_truth(self):
        path = self.write_manual(
            "frame_name,x,y,status\n"
            "frame_000000.png,10,20,labelled\n"
            "frame_000005.png,,,skipped\n"
            "frame_000010.png,,,\n"
            "frame_000015.png,30.5,40,\n"
        )
        labelled, skipped = read_manual_labels(path)
        self.assertEqual(labelled, {"frame_000000.png": (10.0, 20.0), "frame_000015.png": (30.5, 40.0)})
        self.assertEqual(skipped, {"frame_000005.png"})

    def test_bad_files(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            read_manual_labels(self.tmp / "missing.csv")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            read_manual_labels(self.write_manual("frame_name,x,y\n"))
        with self.assertRaisesRegex(ValueError, "Bad coordinates"):
            read_manual_labels(self.write_manual("frame_name,x,y,status\nf.png,a,1,labelled\n"))


class EvaluateTests(unittest.TestCase):
    def test_metrics(self):
        rows = [row(0, 100, 100), row(1, 103, 104), row(2, 112, 100, "uncertain"), row(3), row(4, 300, 300), row(5, 50, 50)]
        labelled = {
            name(0): (100, 100),  # error 0
            name(1): (100, 100),  # error 5
            name(2): (100, 100),  # error 12
            name(3): (100, 100),  # missed: tracker gave no position
            name(9): (1, 1),  # labelled, but no detection file
        }
        skipped = {name(4), name(8)}  # frame 4 got a position; frame 8 has no detections
        metrics, scored = evaluate(rows, labelled, skipped)
        self.assertEqual(metrics["labelled_frames"], 4)
        self.assertEqual(metrics["labelled_without_detections"], 1)
        self.assertEqual((metrics["matched"], metrics["missed"]), (3, 1))
        self.assertEqual((metrics["mean_error_px"], metrics["median_error_px"], metrics["max_error_px"]), (5.7, 5.0, 12.0))
        self.assertEqual((metrics["within_5px"], metrics["within_10px"], metrics["within_15px"]), (2, 2, 3))
        self.assertEqual(metrics["status_on_labelled"], {"good": 2, "uncertain": 1, "lost": 1, "bad": 0})
        self.assertEqual((metrics["good_on_labelled"], metrics["good_mean_error_px"]), (2, 2.5))
        self.assertEqual(metrics["skipped_frames"], 1)
        self.assertEqual(metrics["positions_on_skipped_frames"], {"good": 1})
        self.assertNotIn(name(5), scored)  # unlabelled frames are never scored
        self.assertIsNone(scored[name(3)]["error_px"])

    def test_nothing_matched(self):
        metrics, _ = evaluate([row(0)], {name(0): (1, 1)}, set())
        self.assertEqual((metrics["matched"], metrics["missed"], metrics["mean_error_px"]), (0, 1, None))


class RunEvaluationTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.detections = self.tmp / "detections"
        self.detections.mkdir()
        # A ball moving right; frame 3 is an off-path low-score pick, and frame 4
        # has a decoy where only the baseline goes (see test_track_motion.py).
        per_frame = [[(20, 100)], [(40, 100)], [(60, 100)], [(80, 130, 0.45)], [(100, 100), (100, 150)], []]
        for i, cands in enumerate(per_frame):
            data = {"frame_name": name(i), "source": "automatic", "detections": [
                {"label": "ball", "score": c[2] if len(c) > 2 else 0.9, "box": None, "centre": [c[0], c[1]],
                 "source": "automatic"} for c in cands]}
            (self.detections / f"frame_{i * STEP:06d}.auto.json").write_text(json.dumps(data), encoding="utf-8")
        self.manual = self.write_manual(
            "frame_name,x,y,status\n"
            f"{name(0)},21,100,labelled\n"
            f"{name(4)},100,100,labelled\n"
            f"{name(5)},,,skipped\n"
        )
        self.out = self.tmp / "out"

    def test_writes_comparison(self):
        manual_before = hashlib.sha256(self.manual.read_bytes()).hexdigest()
        comparison = run_evaluation(self.detections, self.manual, self.out, max_displacement=50, static_fraction=0)
        self.assertEqual(hashlib.sha256(self.manual.read_bytes()).hexdigest(), manual_before)

        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["baseline_track.csv", "comparison.json", "labelled_frames.csv", "motion_track.csv"])
        baseline, motion = comparison["metrics"]["baseline"], comparison["metrics"]["motion"]
        self.assertEqual((baseline["labelled_frames"], baseline["matched"]), (2, 2))
        self.assertEqual((baseline["mean_error_px"], baseline["max_error_px"]), (25.5, 50.0))
        self.assertEqual((motion["mean_error_px"], motion["max_error_px"]), (0.5, 1.0))
        self.assertEqual((baseline["within_5px"], motion["within_5px"]), (1, 2))
        self.assertEqual([d["frame_name"] for d in comparison["differences"]], [name(4)])
        self.assertEqual(comparison["differences"][0]["motion"]["y"], 100)
        self.assertEqual(comparison["differences"][0]["baseline"]["y"], 150)
        self.assertEqual(json.loads((self.out / "comparison.json").read_text(encoding="utf-8"))["frames"], 6)

        with open(self.out / "labelled_frames.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r["frame_name"] for r in rows], [name(0), name(4)])
        self.assertEqual((rows[1]["baseline_error_px"], rows[1]["motion_error_px"]), ("50.0", "0.0"))

    def test_refuses_non_empty_output_folder(self):
        self.out.mkdir()
        (self.out / "keep_me.txt").write_text("existing output", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not empty"):
            run_evaluation(self.detections, self.manual, self.out)
        self.assertEqual([p.name for p in self.out.iterdir()], ["keep_me.txt"])

    def test_needs_labelled_frames(self):
        with self.assertRaisesRegex(ValueError, "No labelled ball positions"):
            run_evaluation(self.detections, self.write_manual("frame_name,x,y,status\nf.png,,,skipped\n"), self.out)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
