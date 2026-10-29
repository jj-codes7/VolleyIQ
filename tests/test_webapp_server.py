"""Checks for the web app's HTTP API and background jobs.

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

A real server runs on a free local port, but the analysis is replaced by a
fake processor that the tests control, so no model is loaded. Uploads are
small synthetic videos. Everything is written to a temporary folder that is
deleted afterwards.
"""

import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from volleyiq.make_test_video import make_test_video
from volleyiq.webapp.jobs import COMPLETED, FAILED, PROCESSING, QUEUED, JobManager, display_name
from volleyiq.webapp.server import LOG_LIMIT_BYTES, VolleyIQServer, parse_range, redirect_output

MAX_UPLOAD = 2 * 1024 * 1024
# The same columns track_ball writes; the second reason contains a comma, so csv quotes it.
TRACK_CSV = (
    "frame_name,frame_number,status,reason,x,y,score,track_id,chain_length,n_candidates,n_static,n_in_range,"
    "displacement_px,steps_since_selected,allowed_px,nearest_rejected_px\n"
    "frame_000000.jpg,0,good,continuous chain of 3 frames,691.2,405.1,0.906,1,3,1,0,1,,,,\n"
    'frame_000005.jpg,5,bad,"1 candidate(s) rejected: nearest 1236 px away, more than 240 px allowed",,,,,,1,0,0,,2,240.0,1236.0\n'
)


def write_video(path, frames=5, size=(160, 120), fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), 40 * i % 255, dtype=np.uint8))
    writer.release()
    return path


class FakeProcessor:
    """Stands in for the real pipeline. Waits for `release` so tests can see each state."""

    def __init__(self, fail=None, output_name="output.mp4", write_output=True):
        self.release = threading.Event()
        self.fail = fail
        self.output_name = output_name
        self.write_output = write_output
        self.calls = []

    def __call__(self, input_path, work_dir, on_stage):
        self.calls.append(Path(input_path))
        (Path(work_dir) / "frames").mkdir()
        on_stage("detecting")
        self.release.wait(10)
        if self.fail:
            raise self.fail
        output = Path(work_dir) / self.output_name
        if self.write_output:
            write_video(output)
        else:
            output.write_bytes(b"not a video")
        (Path(work_dir) / "track.csv").write_text(TRACK_CSV, encoding="utf-8")
        return output, {"frames_analysed": 2, "output_fps": 6.0}


class ServerTestCase(unittest.TestCase):
    max_queued = 3

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.processor = FakeProcessor()
        self.manager = JobManager(self.tmp / "app_data", self.processor, MAX_UPLOAD, max_queued=self.max_queued)
        self.manager.start()
        self.server = VolleyIQServer(("127.0.0.1", 0), self.manager, MAX_UPLOAD)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        with redirect_stdout(StringIO()):
            make_test_video(self.tmp / "clip.mp4", duration=1, width=160, height=120, fps=10)
        self.video_bytes = (self.tmp / "clip.mp4").read_bytes()

    def tearDown(self):
        self.processor.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.manager.stop(timeout=10)
        self._tmp.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            data = response.read()
            return response.status, dict(response.getheaders()), data
        finally:
            connection.close()

    def get_json(self, path):
        status, _, data = self.request("GET", path)
        return status, json.loads(data)

    def upload(self, body=None, filename="rally.mp4", content_type="application/octet-stream"):
        status, _, data = self.request("POST", f"/api/jobs?filename={quote(filename)}", body=self.video_bytes if body is None else body,
                                       headers={"Content-Type": content_type})
        return status, json.loads(data)

    def wait_for(self, job_id, statuses, stage=None, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, job = self.get_json(f"/api/jobs/{job_id}")
            if job["status"] in statuses and (stage is None or job["stage"] == stage):
                return job
            time.sleep(0.05)
        self.fail(f"job never reached {statuses}")

    def job_dirs(self):
        return sorted(p.name for p in (self.tmp / "app_data" / "jobs").iterdir())


class StaticAndConfigTests(ServerTestCase):
    def test_page_and_assets(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"VolleyIQ", body)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        for path in ("/app.js", "/app.css"):
            self.assertEqual(self.request("GET", path)[0], 200)

    def test_config(self):
        status, config = self.get_json("/api/config")
        self.assertEqual(status, 200)
        self.assertEqual(config["max_upload_bytes"], MAX_UPLOAD)
        self.assertIn(".mp4", config["allowed_extensions"])
        self.assertEqual((config["samples_per_second"], config["retention_hours"], config["max_queued"]), (6, 24, 3))
        self.assertEqual(config["model"], "grdesignbuild/volleyball-person-ball-detector")

    def test_health(self):
        self.assertEqual(self.get_json("/api/health"), (200, {"app": "VolleyIQ", "status": "ok"}))

    def test_no_files_outside_the_whitelist(self):
        for path in ("/index.html/../app.js", "/../README.md", "/README.md", "/models/hf_cache", "/app_data/jobs",
                     "/static/app.js", "/volleyiq/webapp/server.py", "/api/jobs/../../README.md", "/%2e%2e/README.md"):
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path)[0], 404)

    def test_rejects_other_hosts_and_origins(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)
        status, _, _ = self.request("POST", "/api/jobs?filename=a.mp4", body=self.video_bytes,
                                    headers={"Content-Type": "application/octet-stream", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.job_dirs(), [])


class JobFlowTests(ServerTestCase):
    def test_upload_to_completed_video(self):
        status, job = self.upload(filename="C:\\Users\\me\\Rally 1.mp4")
        self.assertEqual(status, 202)
        self.assertEqual(job["status"], QUEUED)
        self.assertEqual(job["filename"], "Rally 1.mp4")  # display only
        self.assertRegex(job["id"], r"^[0-9a-f]{16}$")
        self.assertEqual(self.job_dirs(), [job["id"]])

        job = self.wait_for(job["id"], {PROCESSING}, stage="detecting")
        self.assertEqual(job["progress"], {"done": 0, "total": 0})
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/video")[0], 409)  # not ready yet
        self.assertEqual(self.processor.calls[0].name, "input.mp4")  # server-side name, not the upload's

        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/tracking")[0], 409)

        self.processor.release.set()
        job = self.wait_for(job["id"], {COMPLETED, FAILED})
        self.assertEqual(job["status"], COMPLETED)
        self.assertEqual(job["stats"], {"frames_analysed": 2, "output_fps": 6.0})

        status, headers, video = self.request("GET", f"/api/jobs/{job['id']}/video")
        self.assertEqual((status, headers["Content-Type"]), (200, "video/mp4"))
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        status, headers, part = self.request("GET", f"/api/jobs/{job['id']}/video", headers={"Range": "bytes=0-9"})
        self.assertEqual((status, part, headers["Content-Range"]), (206, video[:10], f"bytes 0-9/{len(video)}"))

        status, tracking = self.get_json(f"/api/jobs/{job['id']}/tracking")
        self.assertEqual(status, 200)
        self.assertEqual(tracking["output_fps"], 6.0)
        self.assertEqual(tracking["frames"], [
            {"frame_number": 0, "status": "good", "reason": "continuous chain of 3 frames",
             "x": 691.2, "y": 405.1, "score": 0.906},
            {"frame_number": 5, "status": "bad",
             "reason": "1 candidate(s) rejected: nearest 1236 px away, more than 240 px allowed",
             "x": None, "y": None, "score": None},
        ])

        # Temporary data is gone; the result stays.
        job_dir = self.tmp / "app_data" / "jobs" / job["id"]
        self.assertEqual(sorted(p.name for p in job_dir.iterdir()), ["job.json", "output.mp4", "track.csv"])

        _, listing = self.get_json("/api/jobs")
        self.assertEqual([j["id"] for j in listing["jobs"]], [job["id"]])

    def test_failed_processing_is_reported_and_never_completed(self):
        self.processor.fail = RuntimeError("detector exploded")
        _, job = self.upload()
        self.processor.release.set()
        job = self.wait_for(job["id"], {COMPLETED, FAILED})
        self.assertEqual(job["status"], FAILED)
        self.assertIn("detector exploded", job["error"])
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/video")[0], 409)
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/track.csv")[0], 409)
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/tracking")[0], 409)
        job_dir = self.tmp / "app_data" / "jobs" / job["id"]
        self.assertEqual(sorted(p.name for p in job_dir.iterdir()), ["job.json"])

    def test_unreadable_output_is_a_failure(self):
        self.processor.write_output = False
        _, job = self.upload()
        self.processor.release.set()
        job = self.wait_for(job["id"], {COMPLETED, FAILED})
        self.assertEqual(job["status"], FAILED)

    def test_output_outside_the_job_folder_is_a_failure(self):
        self.processor.output_name = "../elsewhere.mp4"
        _, job = self.upload()
        self.processor.release.set()
        self.assertEqual(self.wait_for(job["id"], {COMPLETED, FAILED})["status"], FAILED)

    def test_unknown_and_malformed_job_ids(self):
        for job_id in ("0123456789abcdef", "nothex", "0123456789ABCDEF", "0123456789abcdef0"):
            with self.subTest(job_id=job_id):
                self.assertEqual(self.request("GET", f"/api/jobs/{job_id}")[0], 404)
                self.assertEqual(self.request("GET", f"/api/jobs/{job_id}/video")[0], 404)


class InvalidUploadTests(ServerTestCase):
    def test_wrong_extension(self):
        status, body = self.upload(filename="notes.txt")
        self.assertEqual(status, 400)
        self.assertIn("Unsupported file type", body["error"])

    def test_not_a_video(self):
        status, body = self.upload(body=b"definitely not a video" * 50)
        self.assertEqual(status, 400)
        self.assertIn("could not be opened", body["error"])
        self.assertEqual(self.job_dirs(), [])  # nothing left behind

    def test_empty(self):
        self.assertEqual(self.upload(body=b"")[0], 400)

    def test_too_large(self):
        # Refused from the headers alone, before any of the body is read.
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.putrequest("POST", "/api/jobs?filename=a.mp4")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(MAX_UPLOAD + 1))
        connection.endheaders()
        response = connection.getresponse()
        self.assertEqual(response.status, 413)
        self.assertIn("MB limit", json.loads(response.read())["error"])
        connection.close()
        self.assertEqual(self.job_dirs(), [])

    def test_wrong_content_type(self):
        self.assertEqual(self.upload(content_type="multipart/form-data")[0], 415)

    def test_missing_length(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.putrequest("POST", "/api/jobs?filename=a.mp4")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 411)
        connection.close()

    def test_truncated_upload(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        connection.putrequest("POST", "/api/jobs?filename=a.mp4")
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("Content-Length", str(len(self.video_bytes)))
        connection.endheaders()
        connection.send(self.video_bytes[:100])
        connection.sock.shutdown(1)  # stop sending: the rest never arrives
        self.assertEqual(connection.getresponse().status, 400)
        connection.close()
        self.assertEqual(self.job_dirs(), [])


class ShutdownTests(ServerTestCase):
    def shutdown(self, headers):
        status, _, data = self.request("POST", "/api/shutdown", body=b"", headers=headers)
        return status, json.loads(data)

    def test_needs_the_custom_header(self):
        status, body = self.shutdown({})
        self.assertEqual(status, 400)
        self.assertIn("X-VolleyIQ-Action", body["error"])
        self.assertEqual(self.get_json("/api/health")[0], 200)  # still running

    def test_refused_from_other_websites(self):
        status, _ = self.shutdown({"X-VolleyIQ-Action": "shutdown", "Origin": "http://evil.example"})
        self.assertEqual(status, 403)
        self.assertEqual(self.get_json("/api/health")[0], 200)

    def test_stops_the_server_and_reports_active_jobs(self):
        self.assertEqual(self.upload()[0], 202)  # held in "processing" by the fake processor
        status, body = self.shutdown({"X-VolleyIQ-Action": "shutdown", "Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual((status, body), (202, {"stopping": True, "interrupted_jobs": 1}))
        self.thread.join(timeout=5)
        self.assertFalse(self.thread.is_alive())  # serve_forever returned


class RedirectOutputTests(unittest.TestCase):
    def test_writes_and_restarts_large_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "logs" / "server.log"
            log.parent.mkdir()
            log.write_text("x" * (LOG_LIMIT_BYTES + 1), encoding="utf-8")
            saved = sys.stdout, sys.stderr
            try:
                stream = redirect_output(log)
                print("hello from print")
                sys.stderr.write("hello from stderr\n")
            finally:
                sys.stdout, sys.stderr = saved
                stream.close()
            self.assertEqual(log.read_text(encoding="utf-8"), "hello from print\nhello from stderr\n")


class QueueLimitTests(ServerTestCase):
    max_queued = 1

    def test_too_many_waiting_jobs(self):
        self.assertEqual(self.upload()[0], 202)
        status, body = self.upload()
        self.assertEqual(status, 429)
        self.assertIn("waiting", body["error"])


class JobManagerStorageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.data = Path(self._tmp.name) / "app_data"
        self.jobs = self.data / "jobs"
        self.jobs.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def make_job(self, job_id, status, age_hours=0, extra_dirs=()):
        job_dir = self.jobs / job_id
        job_dir.mkdir()
        for name in extra_dirs:
            (job_dir / name).mkdir()
        created = time.time() - age_hours * 3600
        job = {"id": job_id, "status": status, "stage": "detecting" if status == PROCESSING else None,
               "filename": "a.mp4", "input": "input.mp4", "created_at": created,
               "finished_at": created if status in (COMPLETED, FAILED) else None, "error": None, "stats": None,
               "output": None}
        (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (job_dir / "input.mp4").write_bytes(b"x")
        return job_dir

    def test_interrupted_jobs_become_failed(self):
        job_dir = self.make_job("aaaaaaaaaaaaaaaa", PROCESSING, extra_dirs=("frames", "detections"))
        self.make_job("bbbbbbbbbbbbbbbb", QUEUED)
        manager = JobManager(self.data, FakeProcessor(), MAX_UPLOAD)
        for job_id in ("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb"):
            job = manager.get(job_id)
            self.assertEqual(job["status"], FAILED)
            self.assertIn("Interrupted", job["error"])
        self.assertEqual(sorted(p.name for p in job_dir.iterdir()), ["job.json"])
        saved = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], FAILED)

    def test_cleanup_removes_only_expired_job_folders(self):
        self.make_job("1111111111111111", COMPLETED, age_hours=30)
        self.make_job("2222222222222222", FAILED, age_hours=1)
        # Interrupted at load, so it "finishes" now and gets a full retention period to be seen.
        self.make_job("3333333333333333", PROCESSING, age_hours=50)
        other = self.jobs / "not-a-job"
        other.mkdir()
        (self.data / "keep.txt").write_text("not ours", encoding="utf-8")
        manager = JobManager(self.data, FakeProcessor(), MAX_UPLOAD, retention_hours=24)
        removed = manager.cleanup()
        self.assertEqual(removed, ["1111111111111111"])
        self.assertFalse((self.jobs / "1111111111111111").exists())
        self.assertTrue((self.jobs / "2222222222222222").exists())
        self.assertTrue((self.jobs / "3333333333333333").exists())
        self.assertTrue(other.exists())
        self.assertTrue((self.data / "keep.txt").exists())

    def test_cleanup_keeps_newest_max_jobs(self):
        for i, job_id in enumerate(("aaaaaaaaaaaaaaa1", "aaaaaaaaaaaaaaa2", "aaaaaaaaaaaaaaa3")):
            self.make_job(job_id, COMPLETED, age_hours=3 - i)
        manager = JobManager(self.data, FakeProcessor(), MAX_UPLOAD, max_jobs=2)
        self.assertEqual(manager.cleanup(), ["aaaaaaaaaaaaaaa1"])

    def test_unknown_ids_are_rejected(self):
        manager = JobManager(self.data, FakeProcessor(), MAX_UPLOAD)
        for job_id in ("../jobs", "", None, "g" * 16):
            self.assertIsNone(manager.get(job_id))
            self.assertIsNone(manager.output_file(job_id))


class HelperTests(unittest.TestCase):
    def test_parse_range(self):
        self.assertEqual(parse_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(parse_range("bytes=90-", 100), (90, 99))
        self.assertEqual(parse_range("bytes=-10", 100), (90, 99))
        self.assertEqual(parse_range("bytes=50-500", 100), (50, 99))
        for bad in ("bytes=100-", "bytes=9-0", "bytes=-", "items=0-1", "bytes=0-1,5-6", None):
            self.assertIsNone(parse_range(bad, 100))

    def test_display_name(self):
        self.assertEqual(display_name("C:\\Users\\x\\..\\clip.mp4"), "clip.mp4")
        self.assertEqual(display_name("../../etc/passwd"), "passwd")
        self.assertEqual(display_name("a\x00b.mp4"), "ab.mp4")
        self.assertEqual(display_name(""), "video")


if __name__ == "__main__":
    unittest.main()
