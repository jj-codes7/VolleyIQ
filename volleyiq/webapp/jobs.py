"""Background analysis jobs for the VolleyIQ web app.

One worker thread processes uploaded videos one at a time, in upload order.
Each job lives in its own folder, app_data/jobs/<job id>/, named with a
random id (never the uploaded file name):

    job.json        status, stage, statistics or error (rewritten on every change)
    input.<ext>     the uploaded video                 deleted when the job finishes
    frames/         extracted frames                   deleted when the job finishes
    detections/     detector output                    deleted when the job finishes
    output.mp4      the analysed video (or .webm)      kept until the job expires
    track.csv       tracking result per frame          kept until the job expires

Status moves queued -> processing -> completed or failed, never backwards.
A job is only "completed" after its output video has been read back. Jobs
that were queued or processing when the app stopped are marked failed the
next time it starts. Finished jobs are deleted after `retention_hours`, and
only the newest `max_jobs` are kept.
"""

import json
import logging
import os
import queue
import re
import secrets
import shutil
import threading
import time
from pathlib import Path

from volleyiq.webapp.pipeline import ALLOWED_EXTENSIONS, MEDIA_TYPES, VideoError, probe_video

QUEUED, PROCESSING, COMPLETED, FAILED = "queued", "processing", "completed", "failed"
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
TEMPORARY = ("frames", "detections")
CHUNK = 1024 * 1024
MAX_NAME_LENGTH = 120

log = logging.getLogger("volleyiq.webapp")


class UploadError(ValueError):
    """An upload that can't be accepted. `status` is the HTTP status code to send."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def display_name(filename):
    """The uploaded file's base name, for showing to the user only. Never used as a path."""
    name = re.split(r"[\\/]", str(filename or ""))[-1]
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    return name[:MAX_NAME_LENGTH] or "video"


class JobManager:
    def __init__(self, data_dir, processor, max_upload_bytes, retention_hours=24, max_jobs=20, max_queued=3):
        """`processor(input_path, work_dir, on_stage)` returns (output path, stats) or raises."""
        self.jobs_dir = Path(data_dir).resolve() / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.processor = processor
        self.max_upload_bytes = max_upload_bytes
        self.retention_s = retention_hours * 3600
        self.max_jobs = max_jobs
        self.max_queued = max_queued
        self._jobs = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._worker = None
        self._load_existing()

    # --- lifecycle -----------------------------------------------------

    def start(self):
        self.cleanup()
        self._worker = threading.Thread(target=self._work, name="volleyiq-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout=None):
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join(timeout)

    def _load_existing(self):
        for job_dir in self.jobs_dir.iterdir():
            path = job_dir / "job.json"
            if not (JOB_ID_PATTERN.match(job_dir.name) and path.is_file()):
                continue
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("Ignoring unreadable job file %s", path)
                continue
            if job.get("id") != job_dir.name:
                continue
            if job.get("status") in (QUEUED, PROCESSING):
                job.update(status=FAILED, stage=None, finished_at=time.time(),
                           error="Interrupted: VolleyIQ was closed before this video finished processing. "
                                 "Please upload it again.")
                self._clean_temporary(job_dir, job)
                self._save(job)
            self._jobs[job["id"]] = job

    # --- creating jobs -------------------------------------------------

    def create(self, stream, length, filename):
        """Save an upload of `length` bytes from `stream`, check it, and queue it. Returns the job."""
        suffix = Path(display_name(filename)).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise UploadError(f"Unsupported file type {suffix or '(none)'}. Use one of: {', '.join(ALLOWED_EXTENSIONS)}.")
        if length <= 0:
            raise UploadError("The uploaded file is empty.")
        if length > self.max_upload_bytes:
            raise UploadError(f"The file is larger than the {self.max_upload_bytes // (1024 * 1024)} MB limit.", 413)
        with self._lock:
            waiting = sum(1 for j in self._jobs.values() if j["status"] in (QUEUED, PROCESSING))
        if waiting >= self.max_queued:
            raise UploadError("Too many videos are already waiting. Try again when the current analysis finishes.", 429)

        job_id = secrets.token_hex(8)
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir()
        input_path = job_dir / f"input{suffix}"
        try:
            received = 0
            with open(input_path, "wb") as handle:
                while received < length:
                    chunk = stream.read(min(CHUNK, length - received))
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
            if received != length:
                raise UploadError("The upload was interrupted before the whole file arrived. Please try again.")
            try:
                video = probe_video(input_path)
            except VideoError as error:
                raise UploadError(str(error)) from None
        except BaseException:
            self._remove_job_dir(job_dir)
            raise

        job = {
            "id": job_id, "status": QUEUED, "stage": None, "filename": display_name(filename),
            "size_bytes": length, "input": input_path.name, "video": video, "created_at": time.time(),
            "started_at": None, "finished_at": None, "error": None, "stats": None, "output": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            self._save(job)
        self._queue.put(job_id)
        return self.get(job_id)

    # --- reading jobs --------------------------------------------------

    def get(self, job_id):
        """A copy of the job (with live detection progress), or None for unknown or malformed ids."""
        if not isinstance(job_id, str) or not JOB_ID_PATTERN.match(job_id):
            return None
        with self._lock:
            job = self._jobs.get(job_id)
            job = dict(job) if job else None
        if job and job["status"] == PROCESSING and job["stage"] == "detecting":
            detections = self.jobs_dir / job_id / "detections"
            done = len(list(detections.glob("*.auto.json"))) if detections.is_dir() else 0
            job["progress"] = {"done": done, "total": job.get("frames_total") or 0}
        position = None
        if job and job["status"] == QUEUED:
            with self._lock:
                waiting = sorted((j for j in self._jobs.values() if j["status"] == QUEUED), key=lambda j: j["created_at"])
            position = next((i + 1 for i, j in enumerate(waiting) if j["id"] == job_id), None)
            job["queue_position"] = position
        return job

    def list(self):
        with self._lock:
            ids = sorted(self._jobs, key=lambda i: -self._jobs[i]["created_at"])
        return [self.get(i) for i in ids]

    def output_file(self, job_id):
        """(path, media type) of a completed job's video, or None."""
        job = self.get(job_id)
        if not job or job["status"] != COMPLETED or not job["output"]:
            return None
        path = self.jobs_dir / job_id / job["output"]
        return (path, MEDIA_TYPES[path.suffix]) if path.is_file() else None

    def track_file(self, job_id):
        job = self.get(job_id)
        path = self.jobs_dir / job_id / "track.csv"
        return path if job and job["status"] == COMPLETED and path.is_file() else None

    # --- processing ----------------------------------------------------

    def _update(self, job_id, **changes):
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes)
            self._save(job)

    def _work(self):
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._run(job_id)
            except Exception:  # never let one job stop the worker
                log.exception("Unexpected error while running job %s", job_id)
            finally:
                self.cleanup()

    def _run(self, job_id):
        job_dir = self.jobs_dir / job_id
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["status"] != QUEUED:
                return
        self._update(job_id, status=PROCESSING, started_at=time.time(), stage="starting")

        def on_stage(stage):
            changes = {"stage": stage}
            if stage == "detecting":
                frames = job_dir / "frames"
                changes["frames_total"] = len(list(frames.glob("*.jpg"))) if frames.is_dir() else 0
            self._update(job_id, **changes)

        try:
            output, stats = self.processor(job_dir / job["input"], job_dir, on_stage)
            output = Path(output)
            # Only a video that exists inside this job's folder and can be read back counts.
            if output.resolve().parent != job_dir.resolve() or output.suffix not in MEDIA_TYPES:
                raise RuntimeError("The processor returned an unexpected output file.")
            probe_video(output)
            self._update(job_id, status=COMPLETED, stage=None, finished_at=time.time(), stats=stats, output=output.name)
        except VideoError as error:
            self._update(job_id, status=FAILED, stage=None, finished_at=time.time(), error=str(error))
        except Exception as error:
            log.exception("Job %s failed", job_id)
            self._update(job_id, status=FAILED, stage=None, finished_at=time.time(),
                         error=f"Processing failed: {type(error).__name__}: {error}")
        finally:
            with self._lock:
                self._clean_temporary(job_dir, self._jobs[job_id])

    # --- storage -------------------------------------------------------

    def _save(self, job):
        """Write job.json atomically (call with the lock held)."""
        path = self.jobs_dir / job["id"] / "job.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=1), encoding="utf-8")
        os.replace(temporary, path)

    def _clean_temporary(self, job_dir, job):
        for name in TEMPORARY:
            if (job_dir / name).is_dir():
                shutil.rmtree(job_dir / name, ignore_errors=True)
        if job.get("input"):
            (job_dir / job["input"]).unlink(missing_ok=True)

    def _remove_job_dir(self, job_dir):
        """Delete one job folder, but only a real job folder directly inside jobs_dir."""
        job_dir = Path(job_dir).resolve()
        if job_dir.parent == self.jobs_dir and JOB_ID_PATTERN.match(job_dir.name):
            shutil.rmtree(job_dir, ignore_errors=True)

    def cleanup(self, now=None):
        """Delete finished jobs older than the retention period, and all but the newest max_jobs."""
        now = time.time() if now is None else now
        with self._lock:
            finished = sorted((j for j in self._jobs.values() if j["status"] in (COMPLETED, FAILED)),
                              key=lambda j: -j["created_at"])
            expired = [j for i, j in enumerate(finished)
                       if i >= self.max_jobs or now - (j["finished_at"] or j["created_at"]) > self.retention_s]
            for job in expired:
                del self._jobs[job["id"]]
        for job in expired:
            self._remove_job_dir(self.jobs_dir / job["id"])
        return [job["id"] for job in expired]
