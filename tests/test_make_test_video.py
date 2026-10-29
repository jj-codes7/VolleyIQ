"""Checks for the synthetic test video generator.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

Everything is written to a temporary folder that is deleted afterwards,
so these checks never leave files in the project.
"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2

from volleyiq.extract_frames import extract_frames
from volleyiq.make_test_video import ball_position, make_test_video


class MakeTestVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, name="test.mp4", **settings):
        with redirect_stdout(StringIO()):
            return make_test_video(self.tmp / name, **settings)

    def test_video_reopens_and_all_frames_read(self):
        path = self.tmp / "test.mp4"
        written = self.make(duration=2.0, width=640, height=360, fps=30)
        self.assertEqual(written, 60)

        capture = cv2.VideoCapture(str(path))
        try:
            self.assertTrue(capture.isOpened(), "cv2.VideoCapture could not open the video")
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 640)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 360)
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 30, delta=0.5)

            frames_read = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.assertEqual(frame.shape, (360, 640, 3))
                frames_read += 1
        finally:
            capture.release()

        self.assertEqual(frames_read, written)

    def test_ball_path_is_smooth_and_inside_frame(self):
        width, height, fps = 640, 360, 30
        previous = None
        for frame_index in range(fps * 6):
            x, y = ball_position(frame_index / fps, width, height)
            self.assertTrue(0 <= x < width and 0 <= y < height)
            if previous is not None:
                step = ((x - previous[0]) ** 2 + (y - previous[1]) ** 2) ** 0.5
                self.assertLess(step, 25, f"ball jumped {step:.1f}px at frame {frame_index}")
            previous = (x, y)

    def test_existing_extractor_reads_generated_video(self):
        path = self.tmp / "test.mp4"
        self.make(duration=2.0)
        frames_dir = self.tmp / "frames"
        with redirect_stdout(StringIO()):
            saved = extract_frames(path, frames_dir, every=30)
        self.assertEqual(saved, 2)
        self.assertEqual(
            sorted(p.name for p in frames_dir.iterdir()),
            ["frame_000000.jpg", "frame_000030.jpg"],
        )

    def test_rejects_bad_settings(self):
        bad_settings = [
            {"name": "test.avi"},
            {"duration": 0},
            {"fps": 0},
            {"width": 641},
            {"width": 100, "height": 50},
        ]
        for settings in bad_settings:
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.make(**settings)


if __name__ == "__main__":
    unittest.main()
