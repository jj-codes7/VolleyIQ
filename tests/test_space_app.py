"""Checks for the Hugging Face Space app (space/app.py) and its build script.

The app needs Gradio and the `spaces` package, which the local app's .venv
doesn't have, so the app checks are skipped there. Run them in an environment
built from space/requirements.txt plus gradio and spaces, e.g.:
    python -B -m unittest tests.test_space_app -v

The pretrained model is loaded when the app is imported (as on the Space), so
the model files must be available; the analysis itself uses a stand-in
detector, so no GPU is needed.
"""

import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
HAS_GRADIO = all(importlib.util.find_spec(name) for name in ("gradio", "spaces"))


def load_space_app():
    os.environ.setdefault("VOLLEYIQ_DEVICE", "cpu")
    spec = importlib.util.spec_from_file_location("volleyiq_space_app", ROOT / "space" / "app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildSpaceTests(unittest.TestCase):
    def test_build_contains_only_what_the_space_needs(self):
        sys.path.insert(0, str(ROOT / "space"))
        try:
            import build_space
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as tmp:
            files = build_space.build(Path(tmp) / "space")
            self.assertTrue({"app.py", "README.md", "requirements.txt", "volleyiq/webapp/pipeline.py",
                             "volleyiq/detect_ball.py", "volleyiq/track_ball.py"} <= set(files))
            for name in files:
                self.assertNotRegex(name, r"__pycache__|\.pyc$|static/|launcher\.py|\.mp4$|safetensors|app_data|models/")
            with self.assertRaises(SystemExit):  # never replaces a folder it didn't build
                (Path(tmp) / "other").mkdir()
                (Path(tmp) / "other" / "keep.txt").write_text("x")
                build_space.build(Path(tmp) / "other")
            self.assertTrue((Path(tmp) / "other" / "keep.txt").exists())

    def test_space_metadata(self):
        readme = (ROOT / "space" / "README.md").read_text(encoding="utf-8")
        front = readme.split("---")[1]
        for line in ("sdk: gradio", "sdk_version: 6.31.0", 'python_version: "3.10"', "app_file: app.py",
                     "grdesignbuild/volleyball-person-ball-detector"):
            self.assertIn(line, front)
        requirements = (ROOT / "space" / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.13.0", requirements)
        self.assertNotIn("cu130", requirements)  # the Space uses PyPI's Linux wheels


@unittest.skipUnless(HAS_GRADIO, "needs gradio and spaces (installed on the Space, not in the local .venv)")
class SpaceAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = load_space_app()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_upload_checks(self):
        gr = self.app.gr
        with self.assertRaisesRegex(gr.Error, "Choose a video"):
            self.app.check_upload(None)
        notes = self.tmp / "notes.txt"
        notes.write_text("x")
        with self.assertRaisesRegex(gr.Error, "Unsupported file type"):
            self.app.check_upload(str(notes))
        empty = self.tmp / "empty.mp4"
        empty.write_bytes(b"")
        with self.assertRaisesRegex(gr.Error, "empty"):
            self.app.check_upload(str(empty))
        big = self.tmp / "big.mp4"
        with open(big, "wb") as handle:
            handle.truncate(self.app.MAX_UPLOAD_MB * 1024 * 1024 + 1)
        with self.assertRaisesRegex(gr.Error, "MB limit"):
            self.app.check_upload(str(big))

    def test_gpu_budget_grows_with_frames_and_is_capped(self):
        frames = self.tmp / "frames"
        frames.mkdir()
        self.assertEqual(self.app.gpu_budget(frames, None), self.app.GPU_SECONDS_BASE)
        for i in range(40):
            (frames / f"frame_{i:06d}.jpg").write_bytes(b"x")
        self.assertEqual(self.app.gpu_budget(frames, None), int(self.app.GPU_SECONDS_BASE + 40 * self.app.GPU_SECONDS_PER_FRAME))
        for i in range(40, 2000):
            (frames / f"frame_{i:06d}.jpg").write_bytes(b"x")
        self.assertEqual(self.app.gpu_budget(frames, None), self.app.MAX_GPU_SECONDS)

    def test_sweep_removes_only_old_job_folders(self):
        old, new, other = self.tmp / "job-old", self.tmp / "job-new", self.tmp / "not-a-job"
        for folder in (old, new, other):
            folder.mkdir()
        past = time.time() - 2 * self.app.RESULT_TTL_S
        os.utime(old, (past, past))
        os.utime(other, (past, past))
        self.assertEqual(self.app.sweep_old_results(self.tmp), 1)
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["job-new", "not-a-job"])

    def test_analyze_runs_the_real_pipeline_and_cleans_up(self):
        sys.path.insert(0, str(ROOT / "tests"))
        try:
            from test_webapp_pipeline import FakeDetector, make_video
        finally:
            sys.path.pop(0)
        video = make_video(self.tmp / "clip.mp4", duration=2, width=640, height=360, fps=30)
        work_root = self.tmp / "work"
        fake = FakeDetector()

        def fake_gpu(frames_dir, detections_dir):
            self.app.detect_frames(Path(frames_dir), Path(detections_dir), fake)

        with mock.patch.object(self.app, "WORK_ROOT", work_root), \
                mock.patch.object(self.app, "detect_on_gpu", fake_gpu), \
                mock.patch.object(self.app, "DETECTOR", fake), redirect_stdout(StringIO()):
            video_out, csv_out, stats, tracking = self.app.analyze(str(video), progress=lambda *a, **k: None)
        self.assertTrue(Path(video_out).is_file() and Path(csv_out).is_file())
        self.assertEqual(stats["frames_analysed"], 12)
        self.assertEqual(len(tracking["frames"]), 12)
        self.assertNotIn("model_revision", stats)
        job_dir = Path(video_out).parent
        self.assertEqual(sorted(p.name for p in job_dir.iterdir()), sorted([Path(video_out).name, "volleyiq-tracking.csv"]))

    def test_analyze_reports_bad_videos_without_details(self):
        fake_video = self.tmp / "fake.mp4"
        fake_video.write_bytes(b"not a video" * 100)
        with mock.patch.object(self.app, "WORK_ROOT", self.tmp / "work"):
            with self.assertRaisesRegex(self.app.gr.Error, "could not be opened"):
                self.app.analyze(str(fake_video), progress=lambda *a, **k: None)
        leftovers = list((self.tmp / "work").glob("job-*/work"))
        self.assertEqual(leftovers, [])  # working files removed even on failure


if __name__ == "__main__":
    unittest.main()
