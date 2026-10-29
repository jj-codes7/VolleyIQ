"""Checks for automatic ball detection, without downloading the model.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

The pretrained model is replaced by FakeDetector, which "detects" pure-white
squares as balls and mid-grey squares as people. That exercises the real
tiling, merging and output code. Frames are PNG so colours stay exact.
Everything is written to a temporary folder that is deleted afterwards.
"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from volleyiq.detect_ball import (
    BALL_COLOUR,
    CSV_COLUMNS,
    DEFAULT_OUTPUT_ROOT,
    MODEL_REVISION,
    SOURCE,
    default_output_dir,
    detect,
    draw_detections,
    make_detection,
    merge,
    nearest_ball_distance,
    run_detection,
    tiles_for,
)

ID2LABEL = {0: "person", 1: "ball"}
WHITE, GREY = 255, 128


class FakeDetector:
    """Stands in for BallDetector: finds solid squares of known colours."""

    id2label = ID2LABEL
    device = "cpu"
    revision = MODEL_REVISION

    def __init__(self, score=0.9):
        self.score = score
        self.calls = 0

    def predict_crops(self, crops, threshold):
        self.calls += 1
        results = []
        for crop in crops:
            scores, labels, boxes = [], [], []
            for value, label_id in ((WHITE, 1), (GREY, 0)):
                ys, xs = np.nonzero(np.all(crop == value, axis=2))
                if len(xs) and self.score >= threshold:
                    scores.append(self.score)
                    labels.append(label_id)
                    boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
            results.append((scores, labels, boxes))
        return results


def make_frame(width=1920, height=1080, ball=None, person=None):
    """A dark frame with an optional white ball square and grey person square (x1, y1, x2, y2)."""
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    if person:
        image[person[1]:person[3], person[0]:person[2]] = GREY
    if ball:
        image[ball[1]:ball[3], ball[0]:ball[2]] = WHITE
    return image


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TilesTests(unittest.TestCase):
    def assert_covers(self, windows, width, height):
        covered = np.zeros((height, width), dtype=bool)
        for x1, y1, x2, y2 in windows:
            self.assertTrue(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)
            covered[y1:y2, x1:x2] = True
        self.assertTrue(covered.all())

    def test_small_frame_and_tile_zero_run_whole(self):
        self.assertEqual(tiles_for(640, 480, 864), [(0, 0, 640, 480)])
        self.assertEqual(tiles_for(1920, 1080, 0), [(0, 0, 1920, 1080)])

    def test_1080p_tiles(self):
        for tile, expected in ((864, 6 + 1), (640, 8 + 1)):
            windows = tiles_for(1920, 1080, tile)
            self.assertEqual(len(windows), expected)
            self.assertEqual(windows[-1], (0, 0, 1920, 1080))  # whole-frame pass
            self.assert_covers(windows, 1920, 1080)
            for x1, y1, x2, y2 in windows[:-1]:
                self.assertEqual((x2 - x1, y2 - y1), (tile, tile))

    def test_no_duplicate_windows_when_frame_shorter_than_tile(self):
        windows = tiles_for(3000, 900, 1200)
        self.assertEqual(len(windows), len(set(windows)))
        self.assert_covers(windows, 3000, 900)


class MergeTests(unittest.TestCase):
    def test_duplicates_suppressed_classes_kept_apart(self):
        raw = [
            (100, 100, 130, 130, 0.9, 1, False),
            (101, 101, 131, 131, 0.7, 1, False),  # same ball seen by another tile
            (100, 100, 130, 130, 0.8, 0, False),  # a person in the same place is a different class
        ]
        merged = merge(raw, ID2LABEL)
        self.assertEqual([(d["label"], d["score"]) for d in merged], [("ball", 0.9), ("person", 0.8)])

    def test_box_inside_higher_scoring_box_dropped(self):
        raw = [(0, 0, 100, 100, 0.9, 0, False), (10, 10, 50, 50, 0.6, 0, False)]
        self.assertEqual(len(merge(raw, ID2LABEL)), 1)

    def test_cut_box_only_kept_when_nothing_covers_it(self):
        whole = (100, 100, 140, 140, 0.8, 1, False)
        cut = (100, 100, 120, 140, 0.95, 1, True)
        self.assertEqual(merge([whole, cut], ID2LABEL)[0]["score"], 0.8)
        self.assertEqual(merge([cut], ID2LABEL)[0]["score"], 0.95)

    def test_detection_record(self):
        d = make_detection("ball", 0.81234, (10, 20, 30, 60))
        self.assertEqual(d, {"label": "ball", "score": 0.812, "box": [10.0, 20.0, 30.0, 60.0],
                             "centre": [20.0, 40.0], "source": SOURCE})


class DetectTests(unittest.TestCase):
    def test_ball_found_once_in_frame_coordinates(self):
        # x 640-670 lies in two overlapping 864 tiles and the whole-frame pass.
        image = make_frame(ball=(640, 300, 670, 330))
        for tile in (0, 640, 864):
            detections = detect(image, FakeDetector().predict_crops, ID2LABEL, tile=tile)
            self.assertEqual(len(detections), 1, f"tile {tile}")
            self.assertEqual(detections[0]["box"], [640.0, 300.0, 670.0, 330.0])
            self.assertEqual(detections[0]["centre"], [655.0, 315.0])

    def test_ball_split_by_tile_edge_is_merged(self):
        # 860-890 crosses the right edge (x=864) of the first 864 tile.
        image = make_frame(ball=(860, 100, 890, 130))
        detections = detect(image, FakeDetector().predict_crops, ID2LABEL, tile=864)
        self.assertEqual([d["box"] for d in detections], [[860.0, 100.0, 890.0, 130.0]])

    def test_threshold_filters_low_scores(self):
        image = make_frame(ball=(100, 100, 130, 130))
        self.assertEqual(detect(image, FakeDetector(score=0.3).predict_crops, ID2LABEL, threshold=0.4), [])
        self.assertEqual(len(detect(image, FakeDetector(score=0.3).predict_crops, ID2LABEL, threshold=0.2)), 1)

    def test_nearest_ball_distance(self):
        detections = [make_detection("ball", 0.9, (0, 0, 10, 10)), make_detection("person", 0.9, (95, 95, 105, 105))]
        self.assertEqual(nearest_ball_distance(detections, (8, 5)), 3.0)
        self.assertIsNone(nearest_ball_distance(detections, None))
        self.assertIsNone(nearest_ball_distance(detections[1:], (8, 5)))

    def test_draw_marks_ball(self):
        image = make_frame(400, 300)
        draw_detections(image, [make_detection("ball", 0.9, (100, 100, 140, 140))], manual_point=(120, 120))
        self.assertTrue(np.all(image[120, 100] == BALL_COLOUR))  # left edge of the box


class RunDetectionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.frames = self.tmp / "frames"
        self.frames.mkdir()
        frames = {
            "frame_000000.png": make_frame(ball=(100, 200, 130, 230), person=(500, 400, 600, 700)),
            "frame_000005.png": make_frame(person=(500, 400, 600, 700)),  # no ball
            "frame_000010.png": make_frame(ball=(900, 500, 930, 530)),
        }
        for name, image in frames.items():
            cv2.imwrite(str(self.frames / name), image)
        self.manual_csv = self.tmp / "manual.csv"
        self.manual_csv.write_text(
            "frame_name,x,y,status\n"
            "frame_000000.png,118,211,labelled\n"
            "frame_000005.png,,,skipped\n",
            encoding="utf-8",
        )
        self.out = self.tmp / "out"

    def tearDown(self):
        self._tmp.cleanup()

    def run_fake(self, **options):
        options.setdefault("detector", FakeDetector())
        return run_detection(self.frames, self.out, **options)

    def test_writes_clearly_marked_automatic_outputs(self):
        manual_before = file_hash(self.manual_csv)
        results = self.run_fake(manual_csv=self.manual_csv, tile=640)

        self.assertEqual(file_hash(self.manual_csv), manual_before)  # manual labels untouched
        names = sorted(p.name for p in self.out.iterdir())
        self.assertEqual(names, [
            "ball_detections_auto.csv", "frame_000000.auto.jpg", "frame_000000.auto.json",
            "frame_000005.auto.jpg", "frame_000005.auto.json", "frame_000010.auto.jpg",
            "frame_000010.auto.json", "run_info.json",
        ])

        first = json.loads((self.out / "frame_000000.auto.json").read_text(encoding="utf-8"))
        self.assertEqual(first["source"], SOURCE)
        self.assertEqual(first["revision"], MODEL_REVISION)
        self.assertEqual(first["tile"], 640)
        self.assertEqual([d["label"] for d in first["detections"]], ["ball"])  # people dropped by default
        self.assertTrue(all(d["source"] == SOURCE for d in first["detections"]))
        self.assertEqual(first["manual_reference"], {"x": 118, "y": 211})
        self.assertEqual(first["nearest_ball_distance_px"], 5.0)
        self.assertIsNone(results[1]["manual_reference"])

        with open(self.out / "ball_detections_auto.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(list(rows[0].keys()), CSV_COLUMNS)
        self.assertEqual([(r["frame_name"], r["x"], r["y"]) for r in rows], [
            ("frame_000000.png", "115.0", "215.0"),
            ("frame_000005.png", "", ""),  # every processed frame is listed
            ("frame_000010.png", "915.0", "515.0"),
        ])
        self.assertTrue(all(r["source"] == SOURCE for r in rows))

        info = json.loads((self.out / "run_info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["frames"], ["frame_000000.png", "frame_000005.png", "frame_000010.png"])

    def test_keep_person(self):
        results = self.run_fake(keep_person=True, frame_names=["frame_000005.png"])
        self.assertEqual([d["label"] for d in results[0]["detections"]], ["person"])

    def test_frame_selection(self):
        self.assertEqual(len(self.run_fake(limit=2)), 2)

    def test_refuses_non_empty_output_folder(self):
        self.out.mkdir()
        existing = self.out / "keep_me.txt"
        existing.write_text("existing output", encoding="utf-8")
        detector = FakeDetector()
        with self.assertRaisesRegex(ValueError, "not empty"):
            self.run_fake(detector=detector)
        self.assertEqual(existing.read_text(encoding="utf-8"), "existing output")
        self.assertEqual(detector.calls, 0)

    def test_bad_arguments(self):
        for options, message in (
            ({"threshold": 0}, "threshold"),
            ({"threshold": 1.5}, "threshold"),
            ({"tile": 10}, "tile"),
            ({"limit": 0}, "limit"),
            ({"frame_names": ["missing.png"]}, "not found"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                self.run_fake(**options)

    def test_default_output_dir_is_under_outputs_detections(self):
        path = default_output_dir(self.frames, 864, 0.4)
        self.assertEqual(path.parent, DEFAULT_OUTPUT_ROOT)
        self.assertEqual(path.name, "frames_tile864_t0.40")


if __name__ == "__main__":
    unittest.main()
