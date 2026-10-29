"""Checks for the manual ball annotation tool, without opening any window.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

Everything is written to a temporary folder that is deleted afterwards.
"""

import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

from volleyiq.annotate_ball import (
    HELP_BAR,
    LABELLED,
    SKIPPED,
    TOP_PANEL,
    AnnotationStore,
    Annotator,
    display_scale,
    display_to_image,
    draw_overlay,
    list_frames,
    load_image,
)
from volleyiq.extract_frames import extract_frames
from volleyiq.make_test_video import ball_position, make_test_video


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make_frames(self, names, size=(64, 48)):
        frames_dir = self.tmp / "frames"
        frames_dir.mkdir(exist_ok=True)
        for name in names:
            cv2.imwrite(str(frames_dir / name), np.zeros((size[1], size[0], 3), dtype=np.uint8))
        return frames_dir


class ListFramesTests(TempDirTestCase):
    def test_missing_folder(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            list_frames(self.tmp / "nope")

    def test_path_is_a_file(self):
        file_path = self.tmp / "file.txt"
        file_path.write_text("hi")
        with self.assertRaisesRegex(ValueError, "Not a folder"):
            list_frames(file_path)

    def test_empty_folder(self):
        with self.assertRaisesRegex(ValueError, "No images"):
            list_frames(self.tmp)

    def test_ignores_non_images_and_sorts_by_number(self):
        frames_dir = self.make_frames(["frame_10.jpg", "frame_2.jpg", "frame_1.png"])
        (frames_dir / "notes.txt").write_text("not an image")
        names = [p.name for p in list_frames(frames_dir)]
        self.assertEqual(names, ["frame_1.png", "frame_2.jpg", "frame_10.jpg"])


class LoadImageTests(TempDirTestCase):
    def test_unreadable_image(self):
        bad = self.tmp / "broken.jpg"
        bad.write_bytes(b"this is not a jpeg")
        with self.assertRaisesRegex(ValueError, "Could not decode"):
            load_image(bad)

    def test_empty_image_file(self):
        empty = self.tmp / "empty.jpg"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Could not decode"):
            load_image(empty)

    def test_valid_image(self):
        frames_dir = self.make_frames(["a.png"], size=(64, 48))
        self.assertEqual(load_image(frames_dir / "a.png").shape, (48, 64, 3))


class AnnotationStoreTests(TempDirTestCase):
    def test_save_writes_every_frame_with_blanks(self):
        csv_path = self.tmp / "out" / "ball.csv"
        store = AnnotationStore(["f0.jpg", "f1.jpg", "f2.jpg"], csv_path)
        store.set_point("f0.jpg", 10, 20)
        store.skip("f1.jpg")
        store.save()

        rows = read_rows(csv_path)
        self.assertEqual(list(rows[0].keys()), ["frame_name", "x", "y", "status"])
        self.assertEqual(
            [(r["frame_name"], r["x"], r["y"], r["status"]) for r in rows],
            [("f0.jpg", "10", "20", LABELLED), ("f1.jpg", "", "", SKIPPED), ("f2.jpg", "", "", "")],
        )
        self.assertFalse(csv_path.with_name("ball.csv.tmp").exists())

    def test_round_trip(self):
        csv_path = self.tmp / "ball.csv"
        store = AnnotationStore(["f0.jpg", "f1.jpg", "f2.jpg"], csv_path)
        store.set_point("f2.jpg", 5, 6)
        store.skip("f0.jpg")
        store.save()

        reloaded = AnnotationStore(["f0.jpg", "f1.jpg", "f2.jpg"], csv_path)
        reloaded.load()
        self.assertEqual(reloaded.get("f0.jpg"), (SKIPPED, None, None))
        self.assertEqual(reloaded.get("f1.jpg"), (None, None, None))
        self.assertEqual(reloaded.get("f2.jpg"), (LABELLED, 5, 6))
        self.assertEqual(reloaded.counts(), (1, 1, 1))

    def test_clear_and_correct(self):
        store = AnnotationStore(["f0.jpg"], self.tmp / "ball.csv")
        store.set_point("f0.jpg", 1, 1)
        store.set_point("f0.jpg", 7, 8)
        self.assertEqual(store.get("f0.jpg"), (LABELLED, 7, 8))
        store.clear("f0.jpg")
        self.assertEqual(store.get("f0.jpg"), (None, None, None))

    def test_refuses_unrelated_csv(self):
        csv_path = self.tmp / "other.csv"
        csv_path.write_text("name,score\nalice,3\n", encoding="utf-8")
        store = AnnotationStore(["f0.jpg"], csv_path)
        with self.assertRaisesRegex(ValueError, "not a VolleyIQ annotation CSV"):
            store.load()
        self.assertEqual(csv_path.read_text(encoding="utf-8"), "name,score\nalice,3\n")

    def test_bad_coordinates(self):
        csv_path = self.tmp / "ball.csv"
        csv_path.write_text("frame_name,x,y\nf0.jpg,abc,4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 2"):
            AnnotationStore(["f0.jpg"], csv_path).load()

    def test_accepts_csv_resaved_by_excel(self):
        csv_path = self.tmp / "ball.csv"
        csv_path.write_text("﻿frame_name,x,y,status\nf0.jpg,1,2,labelled\n", encoding="utf-8")
        store = AnnotationStore(["f0.jpg"], csv_path)
        store.load()
        self.assertEqual(store.get("f0.jpg"), (LABELLED, 1, 2))

    def test_keeps_rows_for_other_frames(self):
        csv_path = self.tmp / "ball.csv"
        csv_path.write_text("frame_name,x,y,status\nold.jpg,3,4,labelled\nf0.jpg,1,2,labelled\n", encoding="utf-8")
        store = AnnotationStore(["f0.jpg"], csv_path)
        self.assertEqual(store.load(), ["old.jpg"])
        store.save()
        names = [r["frame_name"] for r in read_rows(csv_path)]
        self.assertEqual(names, ["f0.jpg", "old.jpg"])


class AnnotatorTests(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.frames_dir = self.make_frames(["frame_000.jpg", "frame_030.jpg", "frame_060.jpg"])
        self.csv_path = self.tmp / "ball.csv"
        self.annotator = Annotator(list_frames(self.frames_dir), self.csv_path)

    def test_click_saves_immediately(self):
        self.assertTrue(self.annotator.click(12, 34, 64, 48))
        rows = read_rows(self.csv_path)
        self.assertEqual((rows[0]["x"], rows[0]["y"], rows[0]["status"]), ("12", "34", LABELLED))

    def test_click_outside_image_is_ignored(self):
        self.assertFalse(self.annotator.click(64, 10, 64, 48))
        self.assertFalse(self.annotator.click(-1, 10, 64, 48))
        self.assertEqual(self.annotator.store.get("frame_000.jpg")[0], None)

    def test_navigation_stays_in_range(self):
        self.annotator.previous()
        self.assertEqual(self.annotator.index, 0)
        for _ in range(5):
            self.annotator.next()
        self.assertEqual(self.annotator.index, 2)

    def test_skip_marks_and_advances(self):
        self.annotator.skip()
        self.assertEqual(self.annotator.index, 1)
        self.assertEqual(read_rows(self.csv_path)[0]["status"], SKIPPED)

    def test_clear_saves(self):
        self.annotator.click(5, 5, 64, 48)
        self.annotator.clear()
        self.assertEqual(read_rows(self.csv_path)[0]["x"], "")

    def test_resumes_at_first_unannotated_frame(self):
        self.annotator.click(5, 5, 64, 48)
        self.annotator.next()
        self.annotator.skip()
        resumed = Annotator(list_frames(self.frames_dir), self.csv_path)
        self.assertEqual(resumed.index, 2)
        self.assertEqual(resumed.store.counts(), (1, 1, 1))


class OverlayTests(TempDirTestCase):
    def test_overlay_adds_panels_without_covering_the_frame(self):
        frames_dir = self.make_frames(["f0.png"], size=(64, 48))
        annotator = Annotator(list_frames(frames_dir), self.tmp / "ball.csv")
        image = np.full((48, 64, 3), 200, dtype=np.uint8)
        shown = draw_overlay(image, annotator, 1.0)
        self.assertEqual(shown.shape, (TOP_PANEL + 48 + HELP_BAR, 64, 3))
        self.assertTrue((shown[TOP_PANEL:TOP_PANEL + 48] == 200).all())


class DisplayScalingTests(unittest.TestCase):
    def test_small_image_is_not_enlarged(self):
        self.assertEqual(display_scale(640, 360, 1280, 720), 1.0)

    def test_large_image_is_shrunk_to_fit(self):
        self.assertAlmostEqual(display_scale(1920, 1080, 1280, 720), 2 / 3)

    def test_click_maps_back_to_original_pixels(self):
        scale = display_scale(1920, 1080, 1280, 720)
        self.assertEqual(display_to_image(640, 360, scale), (960, 540))


class PipelineTests(TempDirTestCase):
    def test_synthetic_video_to_annotation_csv(self):
        """Generate a video, extract frames, and annotate them with the known
        synthetic ball positions, as a stand-in for a person clicking."""
        video = self.tmp / "synthetic.mp4"
        frames_dir = self.tmp / "frames"
        with redirect_stdout(StringIO()):
            make_test_video(video, duration=2.0)
            extract_frames(video, frames_dir, every=15)

        frame_paths = list_frames(frames_dir)
        annotator = Annotator(frame_paths, self.tmp / "ball.csv")
        for path in frame_paths:
            frame_number = int(path.stem.split("_")[1])
            x, y = ball_position(frame_number / 30, 640, 360)
            annotator.click(x, y, 640, 360)
            annotator.next()

        rows = read_rows(self.tmp / "ball.csv")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["status"] == LABELLED for r in rows))
        self.assertEqual(rows[0]["frame_name"], "frame_000000.jpg")


if __name__ == "__main__":
    unittest.main()
