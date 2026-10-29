"""Checks for the trajectory visualizer.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

Uses the synthetic video -> extract frames -> annotate pipeline, with the
known synthetic ball positions standing in for a person's clicks. Everything
is written to a temporary folder that is deleted afterwards.
"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

from volleyiq.annotate_ball import Annotator, list_frames, load_image
from volleyiq.extract_frames import extract_frames
from volleyiq.make_test_video import ball_position, make_test_video
from volleyiq.visualize_trajectory import (
    POINT_COLOUR,
    draw_trajectory,
    frame_number_from_name,
    make_trajectory_image,
    make_trajectory_video,
    read_annotations,
    split_into_segments,
)

FPS = 30


class TempDirTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write_csv(self, text):
        path = self.tmp / "ball.csv"
        path.write_text(text, encoding="utf-8")
        return path


class ReadAnnotationsTests(TempDirTestCase):
    def test_labelled_skipped_and_blank_rows(self):
        path = self.write_csv(
            "frame_name,x,y,status\n"
            "frame_000030.jpg,30,40,labelled\n"
            "frame_000000.jpg,10,20,labelled\n"
            "frame_000060.jpg,,,skipped\n"
            "frame_000090.jpg,,,\n"
        )
        rows = read_annotations(path)
        self.assertEqual([r["frame_number"] for r in rows], [0, 30, 60, 90])
        self.assertEqual([r["point"] for r in rows], [(10, 20), (30, 40), None, None])

    def test_skipped_row_with_coordinates_is_not_a_point(self):
        path = self.write_csv("frame_name,x,y,status\nframe_0.jpg,1,1,labelled\nframe_1.jpg,5,5,skipped\n")
        self.assertEqual([r["point"] for r in read_annotations(path)], [(1, 1), None])

    def test_missing_file(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            read_annotations(self.tmp / "nope.csv")

    def test_not_an_annotation_csv(self):
        with self.assertRaisesRegex(ValueError, "not a VolleyIQ annotation CSV"):
            read_annotations(self.write_csv("name,score\nalice,3\n"))

    def test_bad_coordinates(self):
        with self.assertRaisesRegex(ValueError, "line 2"):
            read_annotations(self.write_csv("frame_name,x,y\nframe_0.jpg,abc,4\n"))

    def test_no_labelled_points(self):
        with self.assertRaisesRegex(ValueError, "No labelled"):
            read_annotations(self.write_csv("frame_name,x,y,status\nframe_0.jpg,,,skipped\n"))

    def test_frame_number_from_name(self):
        self.assertEqual(frame_number_from_name("frame_000120.jpg"), 120)
        with self.assertRaisesRegex(ValueError, "frame number"):
            frame_number_from_name("ball.jpg")


class SegmentTests(unittest.TestCase):
    def row(self, number, point):
        return {"frame_name": f"frame_{number}.jpg", "frame_number": number, "point": point}

    def test_line_breaks_at_skipped_or_blank_frames(self):
        rows = [
            self.row(0, (1, 1)), self.row(5, (2, 2)),
            self.row(10, None),
            self.row(15, (3, 3)),
            self.row(20, None), self.row(25, None),
            self.row(30, (4, 4)), self.row(35, (5, 5)),
        ]
        segments = split_into_segments(rows)
        self.assertEqual(
            [[r["frame_number"] for r in s] for s in segments],
            [[0, 5], [15], [30, 35]],
        )

    def test_no_line_drawn_across_a_gap(self):
        rows = [self.row(0, (10, 50)), self.row(1, None), self.row(2, (90, 50))]
        image = draw_trajectory(np.zeros((100, 100, 3), dtype=np.uint8), rows)
        # Halfway between the two points: would be orange if a line crossed the gap.
        self.assertTrue((image[50, 50] == 0).all())

    def test_up_to_frame_hides_later_points(self):
        rows = [self.row(0, (10, 10)), self.row(30, (80, 10))]
        image = draw_trajectory(np.zeros((100, 100, 3), dtype=np.uint8), rows, up_to_frame=29)
        self.assertTrue((image[10, 80] == 0).all())


class PipelineTests(TempDirTestCase):
    """Synthetic video -> extract every 15th frame -> annotate -> visualize."""

    def setUp(self):
        super().setUp()
        self.video = self.tmp / "synthetic.mp4"
        self.frames_dir = self.tmp / "frames"
        self.csv = self.tmp / "ball.csv"
        with redirect_stdout(StringIO()):
            make_test_video(self.video, duration=2.0, fps=FPS)  # 60 frames
            extract_frames(self.video, self.frames_dir, every=15)  # 0, 15, 30, 45

        annotator = Annotator(list_frames(self.frames_dir), self.csv)
        self.points = {}
        for path in list_frames(self.frames_dir):
            number = frame_number_from_name(path.name)
            if number == 30:
                annotator.skip()  # pretend the ball was hidden in frame 30
                continue
            x, y = ball_position(number / FPS, 640, 360)
            annotator.click(x, y, 640, 360)
            annotator.next()
            self.points[number] = (x, y)

    def test_image_output(self):
        output = self.tmp / "out" / "trajectory.png"
        count = make_trajectory_image(self.csv, self.frames_dir, output)
        self.assertEqual(count, 3)

        image = load_image(output)
        self.assertEqual(image.shape, (360, 640, 3))
        x, y = self.points[45]
        self.assertEqual(tuple(int(v) for v in image[y, x]), POINT_COLOUR)

    def test_image_on_chosen_frame_and_bad_frame(self):
        output = self.tmp / "trajectory.png"
        make_trajectory_image(self.csv, self.frames_dir, output, frame_name="frame_000045.jpg")
        self.assertTrue(output.exists())
        with self.assertRaisesRegex(ValueError, "not found"):
            make_trajectory_image(self.csv, self.frames_dir, output, frame_name="frame_999999.jpg")

    def test_video_output_matches_original_and_path_grows(self):
        output = self.tmp / "trajectory.mp4"
        written = make_trajectory_video(self.csv, self.video, output)
        self.assertEqual(written, 60)

        capture = cv2.VideoCapture(str(output))
        try:
            self.assertTrue(capture.isOpened())
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 640)
            self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)), 360)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        self.assertEqual(len(frames), 60)

        # The point for frame 45 is red from frame 45 on, but not before.
        # Video compression changes colours slightly, so compare loosely.
        x, y = self.points[45]
        blue, green, red = (int(v) for v in frames[50][y, x])
        self.assertTrue(red > 180 and green < 90 and blue < 90, (blue, green, red))
        blue, green, red = (int(v) for v in frames[40][y, x])
        self.assertFalse(red > 180 and green < 90 and blue < 90, (blue, green, red))

    def test_video_rejects_csv_from_a_longer_video(self):
        csv_text = self.csv.read_text(encoding="utf-8") + "frame_000900.jpg,10,10,labelled\n"
        self.csv.write_text(csv_text, encoding="utf-8")
        output = self.tmp / "trajectory.mp4"
        with self.assertRaisesRegex(ValueError, "has only 60 frames"):
            make_trajectory_video(self.csv, self.video, output)
        self.assertFalse(output.exists())

    def test_points_outside_frame_are_rejected(self):
        self.csv.write_text("frame_name,x,y,status\nframe_000000.jpg,5000,10,labelled\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "outside"):
            make_trajectory_image(self.csv, self.frames_dir, self.tmp / "t.png")

    def test_bad_output_extensions(self):
        with self.assertRaisesRegex(ValueError, ".png or .jpg"):
            make_trajectory_image(self.csv, self.frames_dir, self.tmp / "t.gif")
        with self.assertRaisesRegex(ValueError, ".mp4"):
            make_trajectory_video(self.csv, self.video, self.tmp / "t.avi")


if __name__ == "__main__":
    unittest.main()
