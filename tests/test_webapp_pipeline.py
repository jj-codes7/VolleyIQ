"""Checks for the web app's processing pipeline, on a synthetic video.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

The real frame extraction, detection loop, tracker and video rendering run,
but the AI model is replaced by FakeDetector, which finds the synthetic
video's yellow ball by its colour. No weights or network are needed.
Everything is written to a temporary folder that is deleted afterwards.
"""

import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from volleyiq.make_test_video import BALL, make_test_video
from volleyiq.webapp.pipeline import (
    BROWSER_FORMATS,
    VideoError,
    output_size,
    probe_video,
    process_video,
    sample_interval,
    scaled_max_displacement,
)

BALL_RGB = np.array(BALL[::-1], dtype=int)  # make_test_video's colours are BGR


class FakeDetector:
    """Finds the synthetic ball by colour, standing in for the pretrained model."""

    id2label = {0: "person", 1: "ball"}
    device = "cpu"
    revision = "fake"

    def predict_crops(self, crops, threshold):
        results = []
        for crop in crops:
            mask = np.all(np.abs(crop.astype(int) - BALL_RGB) < 50, axis=2)
            ys, xs = np.nonzero(mask)
            if len(xs) >= 10:
                results.append(([0.9], [1], [[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]]))
            else:
                results.append(([], [], []))
        return results


def make_video(path, **options):
    with redirect_stdout(StringIO()):
        make_test_video(path, **options)
    return path


class PipelineHelperTests(unittest.TestCase):
    def test_sample_interval(self):
        self.assertEqual([sample_interval(f) for f in (30, 29.97, 60, 24, 5)], [5, 5, 10, 4, 1])

    def test_scaled_max_displacement(self):
        self.assertEqual(scaled_max_displacement(1920), 120.0)
        self.assertEqual(scaled_max_displacement(640), 40.0)
        self.assertEqual(scaled_max_displacement(100), 20.0)  # never below 20 px

    def test_output_size_is_even_and_capped(self):
        self.assertEqual(output_size(3840, 2160), (1920, 1080))
        self.assertEqual(output_size(641, 361), (640, 360))


class ProbeVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_a_real_video(self):
        info = probe_video(make_video(self.tmp / "clip.mp4", duration=1, width=320, height=180, fps=30))
        self.assertEqual((info["width"], info["height"], info["fps"], info["frame_count"]), (320, 180, 30, 30))
        self.assertEqual(info["duration_s"], 1.0)

    def test_rejects_files_that_are_not_videos(self):
        fake = self.tmp / "fake.mp4"
        fake.write_bytes(b"this is not a video" * 100)
        with self.assertRaisesRegex(VideoError, "could not be opened|No frames"):
            probe_video(fake)

    def test_rejects_long_videos(self):
        path = make_video(self.tmp / "clip.mp4", duration=2, width=160, height=120, fps=10)
        with mock.patch("volleyiq.webapp.pipeline.MAX_DURATION_S", 1):
            with self.assertRaisesRegex(VideoError, "limit is"):
                probe_video(path)


class ProcessVideoTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_full_pipeline_with_fake_detector(self):
        video = make_video(self.tmp / "clip.mp4", duration=2, width=640, height=360, fps=30)
        work = self.tmp / "job"
        work.mkdir()
        stages, loads = [], []

        def get_detector():
            loads.append(1)
            return FakeDetector()

        with redirect_stdout(StringIO()):  # extract_frames prints a summary
            output, stats = process_video(video, work, get_detector, stages.append)

        self.assertEqual(stages, ["extracting", "loading_model", "detecting", "tracking", "rendering", None])
        self.assertEqual(len(loads), 1)
        self.assertEqual((stats["sample_interval"], stats["frames_analysed"]), (5, 12))
        self.assertEqual(stats["ball_boxes"], 12)  # the fake found the ball in every analysed frame
        self.assertEqual(sum(stats["status_counts"].values()), 12)
        self.assertEqual(stats["max_displacement_px"], 40.0)
        self.assertEqual(stats["output_fps"], 6.0)
        self.assertEqual(stats["tracking_strategy"], "baseline")
        self.assertEqual(set(stats["timings_s"]), {"extracting", "loading_model", "detecting", "tracking", "rendering"})

        # The output is a browser-playable video with one frame per analysed frame.
        self.assertEqual(output.parent, work)
        capture = cv2.VideoCapture(str(output))
        codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        accepted = next(a for _, suffix, a in BROWSER_FORMATS if suffix == output.suffix)
        self.assertIn(codec, accepted)
        self.assertEqual(stats["output_codec"], codec)
        self.assertEqual(frames, 12)

        with open(work / "track.csv", newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 12)

    def test_unreadable_video_fails_before_loading_the_model(self):
        fake = self.tmp / "fake.mp4"
        fake.write_bytes(b"\x00" * 2048)
        loads = []
        with self.assertRaises(VideoError):
            process_video(fake, self.tmp, lambda: loads.append(1))
        self.assertEqual(loads, [])


if __name__ == "__main__":
    unittest.main()
