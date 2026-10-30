"""The VolleyIQ web app: a small local HTTP server around the analysis jobs.

Usage:
    python -m volleyiq.webapp [--port 8765] [--no-browser] [--max-upload-mb 500] [--log-file PATH]

Or double-click VolleyIQ.vbs (no console window) or VolleyIQ.bat (with a
console window) in the project folder.

It uses only Python's built-in http.server, listens on 127.0.0.1 (this
computer only) and is meant for one person at a time.

Routes:
    GET  /                       the web page (index.html, app.css, app.js only)
    GET  /api/health             {"app": "VolleyIQ"}: lets the launcher recognise a running app
    GET  /api/config             upload limits, accepted file types and pipeline settings
    GET  /api/jobs               recent jobs
    POST /api/jobs?filename=N    upload a video as the raw request body; starts a job
    GET  /api/jobs/<id>          one job's status
    GET  /api/jobs/<id>/video    the analysed video (supports seeking)
    GET  /api/jobs/<id>/track.csv  the tracking result
    GET  /api/jobs/<id>/tracking   the same tracking result as JSON, for the results page
    DELETE /api/jobs/<id>        delete a job and all its files (a processing job is cancelled first)
    DELETE /api/jobs             delete every job: clears the history
    POST /api/shutdown           stop the app (needs the X-VolleyIQ-Action: shutdown header)

Nothing else on disk is reachable: static files come from a fixed list, and
job ids must be 16 hex characters. Deleting only ever removes folders named
with a job id inside app_data/jobs/ (see JobManager._remove_job_dir).
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from volleyiq.webapp.jobs import JOB_ID_PATTERN, PROCESSING, QUEUED, JobManager, UploadError
from volleyiq.webapp.pipeline import ALLOWED_EXTENSIONS, MAX_DURATION_S, SAMPLES_PER_SECOND

APP_NAME = "VolleyIQ"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "app_data"
DEFAULT_PORT = 8765
LOG_LIMIT_BYTES = 5 * 1024 * 1024  # a longer log file is started afresh
SHUTDOWN_HEADER = ("X-VolleyIQ-Action", "shutdown")
TRACKING_FIELDS = ("frame_number", "status", "reason", "x", "y", "score")
DEFAULT_MAX_UPLOAD_MB = 500
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
JOB_ROUTE = re.compile(r"^/api/jobs/([^/]+)(/video|/track\.csv|/tracking)?$")
RANGE_PATTERN = re.compile(r"^bytes=(\d*)-(\d*)$")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'",
}
STREAM_CHUNK = 256 * 1024
DRAIN_LIMIT = 1024 * 1024  # refused bodies up to this size are read and dropped

log = logging.getLogger("volleyiq.webapp")


def parse_range(header, size):
    """(start, end) inclusive for a single "bytes=a-b" range, None for no/invalid range."""
    match = RANGE_PATTERN.match(header or "")
    if not match or size == 0:
        return None
    first, last = match.groups()
    if first == "" and last == "":
        return None
    if first == "":  # the last N bytes
        start, end = max(0, size - int(last)), size - 1
    else:
        start, end = int(first), min(size - 1, int(last)) if last else size - 1
    return (start, end) if start <= end and start < size else None


class VolleyIQServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows, SO_REUSEADDR lets a second server bind a port that is already in use.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address, manager, max_upload_bytes):
        super().__init__(address, Handler)
        self.manager = manager
        self.max_upload_bytes = max_upload_bytes

    def allowed_hosts(self):
        port = self.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}


class Handler(BaseHTTPRequestHandler):
    server_version = "VolleyIQ"
    sys_version = ""

    def log_message(self, format, *args):
        log.info("%s %s", self.address_string(), format % args)

    # --- responses -----------------------------------------------------

    def _headers(self, status, content_type, length, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        for name, value in {**SECURITY_HEADERS, **(extra or {})}.items():
            self.send_header(name, value)
        self.end_headers()

    def send_json(self, status, data):
        body = json.dumps(data).encode("utf-8")
        self._headers(status, "application/json", len(body), {"Cache-Control": "no-store"})
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error_json(self, status, message):
        self.send_json(status, {"error": message})

    def send_file(self, path, content_type, download_name=None):
        size = path.stat().st_size
        extra = {"Accept-Ranges": "bytes", "Cache-Control": "no-store"}
        if download_name:
            extra["Content-Disposition"] = f'attachment; filename="{download_name}"'
        byte_range = parse_range(self.headers.get("Range"), size)
        if self.headers.get("Range") and byte_range is None:
            self._headers(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, content_type, 0, {"Content-Range": f"bytes */{size}"})
            return
        start, end = byte_range or (0, size - 1)
        if byte_range:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        status = HTTPStatus.PARTIAL_CONTENT if byte_range else HTTPStatus.OK
        self._headers(status, content_type, end - start + 1, extra)
        if self.command == "HEAD":
            return
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(STREAM_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # the browser stopped reading, e.g. while seeking

    # --- request checks ------------------------------------------------

    def _untrusted_reason(self):
        """Why the request is refused, or None. Only this computer's browser may talk to the app
        (this blocks other websites and DNS rebinding)."""
        if self.headers.get("Host", "") not in self.server.allowed_hosts():
            return "Unexpected Host header."
        origin = self.headers.get("Origin")
        if origin is not None and origin.removeprefix("http://") not in self.server.allowed_hosts():
            return "Requests from other websites are not allowed."
        return None

    def _drain(self, remaining):
        """Read and drop a small unread request body, so the client receives the error reply
        (closing a socket with unread data can reset the connection on Windows)."""
        if 0 < remaining <= DRAIN_LIMIT:
            try:
                self.rfile.read(remaining)
            except OSError:
                pass
        if remaining > 0:
            self.close_connection = True

    # --- routes --------------------------------------------------------

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        reason = self._untrusted_reason()
        if reason:
            self.send_error_json(HTTPStatus.FORBIDDEN, reason)
            return
        path = urlsplit(self.path).path
        manager = self.server.manager
        if path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            self.send_file(STATIC_DIR / name, content_type)
        elif path == "/api/health":
            self.send_json(HTTPStatus.OK, {"app": APP_NAME, "status": "ok"})
        elif path == "/api/config":
            from volleyiq.detect_ball import MODEL_ID

            self.send_json(HTTPStatus.OK, {
                "max_upload_bytes": self.server.max_upload_bytes,
                "allowed_extensions": list(ALLOWED_EXTENSIONS),
                "max_duration_s": MAX_DURATION_S,
                "samples_per_second": SAMPLES_PER_SECOND,
                "retention_hours": round(manager.retention_s / 3600, 1),
                "max_queued": manager.max_queued,
                "model": MODEL_ID,
                "gpu": gpu_name(),
            })
        elif path == "/api/jobs":
            self.send_json(HTTPStatus.OK, {"jobs": manager.list()})
        elif match := JOB_ROUTE.match(path):
            self._job_route(*match.groups())
        else:
            self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")

    def _job_route(self, job_id, part):
        manager = self.server.manager
        job = manager.get(job_id) if JOB_ID_PATTERN.match(job_id) else None
        if job is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "No such job. It may have expired.")
        elif part is None:
            self.send_json(HTTPStatus.OK, job)
        elif part == "/video":
            found = manager.output_file(job_id)
            if found is None:
                self.send_error_json(HTTPStatus.CONFLICT, f"The video is not available: the job is {job['status']}.")
            else:
                self.send_file(*found)
        else:
            found = manager.track_file(job_id)
            if found is None:
                self.send_error_json(HTTPStatus.CONFLICT, f"The tracking file is not available: the job is {job['status']}.")
            elif part == "/tracking":
                self.send_json(HTTPStatus.OK, {"frames": read_tracking(found),
                                               "output_fps": (job.get("stats") or {}).get("output_fps")})
            else:
                self.send_file(found, "text/csv; charset=utf-8", download_name=f"volleyiq_{job_id}_track.csv")

    def do_DELETE(self):
        # DELETE is never sent cross-site without a CORS preflight (OPTIONS), which this server refuses.
        reason = self._untrusted_reason()
        if reason:
            return self.send_error_json(HTTPStatus.FORBIDDEN, reason)
        path = urlsplit(self.path).path
        manager = self.server.manager
        if path == "/api/jobs":
            return self.send_json(HTTPStatus.OK, manager.delete_all())
        match = JOB_ROUTE.match(path)
        if not match or match.group(2):
            return self.send_error_json(HTTPStatus.NOT_FOUND, "Not found.")
        result = manager.delete(match.group(1))
        if result is None:
            return self.send_error_json(HTTPStatus.NOT_FOUND, "No such job. It may already have been deleted.")
        self.send_json(HTTPStatus.OK, {"id": match.group(1), "result": result})

    def do_POST(self):
        try:
            length = max(0, int(self.headers.get("Content-Length", "")))
        except ValueError:
            length = None

        def reject(status, message, consumed=0):
            self._drain((length or 0) - consumed)
            self.send_error_json(status, message)

        url = urlsplit(self.path)
        reason = self._untrusted_reason()
        if reason:
            return reject(HTTPStatus.FORBIDDEN, reason)
        if url.path == "/api/shutdown":
            return self._shutdown(reject)
        if url.path != "/api/jobs":
            return reject(HTTPStatus.NOT_FOUND, "Not found.")
        # A raw body (not a form) forces browsers to ask permission for cross-site uploads.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/octet-stream":
            return reject(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Upload the video as application/octet-stream.")
        if length is None:
            self.close_connection = True
            return self.send_error_json(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required.")
        if length > self.server.max_upload_bytes:
            self.close_connection = True  # don't read a body we are refusing
            limit = self.server.max_upload_bytes // (1024 * 1024)
            return self.send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"The file is larger than the {limit} MB limit.")
        filename = parse_qs(url.query).get("filename", [""])[0]
        body = CountingReader(self.rfile)
        try:
            job = self.server.manager.create(body, length, filename)
        except UploadError as error:
            return reject(error.status, str(error), consumed=body.count)
        self.send_json(HTTPStatus.ACCEPTED, job)

    def _shutdown(self, reject):
        """Stop the server, so an app started without a console window can be closed from the page.

        The custom header can't be sent by another website without the browser asking this
        server first (a CORS preflight), which it never allows.
        """
        name, value = SHUTDOWN_HEADER
        if self.headers.get(name) != value:
            return reject(HTTPStatus.BAD_REQUEST, f"Missing the {name} header.")
        active = sum(1 for job in self.server.manager.list() if job["status"] in (QUEUED, PROCESSING))
        self.send_json(HTTPStatus.ACCEPTED, {"stopping": True, "interrupted_jobs": active})
        log.info("Shutdown requested from the web page")
        threading.Thread(target=self.server.shutdown, daemon=True).start()


def read_tracking(path):
    """The per-frame tracking rows from track.csv, with only the fields the page needs."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            item = {field: row.get(field, "") for field in TRACKING_FIELDS}
            item["frame_number"] = int(item["frame_number"])
            for field in ("x", "y", "score"):
                item[field] = float(item[field]) if item[field] else None
            rows.append(item)
    return rows


class CountingReader:
    """Wraps the request body to count how much of it has been read."""

    def __init__(self, raw):
        self.raw, self.count = raw, 0

    def read(self, size):
        data = self.raw.read(size)
        self.count += len(data)
        return data


def gpu_name():
    """The CUDA GPU the model runs on, or None (then it runs on the CPU)."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # no usable CUDA: the app still works on the CPU
        return None


class CancellableDetector:
    """The shared detector, checking before each batch of tiles whether the user cancelled this job."""

    def __init__(self, detector, check_cancelled):
        self._detector, self._check = detector, check_cancelled

    def predict_crops(self, *args, **kwargs):
        self._check()
        return self._detector.predict_crops(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._detector, name)


def make_real_processor():
    """The processor used by the app: the real pipeline, with the model loaded once and reused."""
    from volleyiq.webapp.pipeline import process_video

    lock, cache = threading.Lock(), {}

    def get_detector():
        with lock:
            if "detector" not in cache:
                from volleyiq.detect_ball import BallDetector

                try:
                    cache["detector"] = BallDetector()
                except OSError as error:
                    raise RuntimeError("The ball-detection model could not be loaded from models/hf_cache. "
                                       "See Troubleshooting in README.md.") from error
            return cache["detector"]

    def processor(input_path, work_dir, on_stage):
        check = getattr(on_stage, "check_cancelled", None)
        detector = (lambda: CancellableDetector(get_detector(), check)) if check else get_detector
        return process_video(input_path, work_dir, detector, on_stage)

    return processor


def main(argv=None):
    parser = argparse.ArgumentParser(description="Start the VolleyIQ web app on this computer.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="don't open the web page automatically")
    parser.add_argument("--max-upload-mb", type=int, default=DEFAULT_MAX_UPLOAD_MB,
                        help=f"largest accepted upload in MB (default {DEFAULT_MAX_UPLOAD_MB})")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="where jobs are stored (default app_data)")
    parser.add_argument("--log-file", help="write all output here instead of the console (used by VolleyIQ.vbs)")
    args = parser.parse_args(argv)

    # Without a console (pythonw), sys.stderr is None and progress bars would crash: use a log file.
    log_file = args.log_file or (None if sys.stderr else str(Path(args.data_dir) / "logs" / "server.log"))
    if log_file:
        redirect_output(log_file)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr)
    # Use only the model already in models/hf_cache; never download anything.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    url = f"http://127.0.0.1:{args.port}/"
    from volleyiq.launcher import is_running

    if is_running(url):  # e.g. VolleyIQ.bat double-clicked twice: just show the running app
        print(f"VolleyIQ is already running at {url}. Opening it in the browser.")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    max_upload = args.max_upload_mb * 1024 * 1024
    manager = JobManager(args.data_dir, make_real_processor(), max_upload)
    try:
        server = VolleyIQServer(("127.0.0.1", args.port), manager, max_upload)
    except OSError as error:
        print(f"Could not start on port {args.port}: {error}\n"
              f"VolleyIQ may already be running: open http://127.0.0.1:{args.port}/ "
              "or start it with another --port.", file=sys.stderr)
        return 1

    manager.start()
    print(f"VolleyIQ is running at {url}\nKeep this window open while you use it. Press Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, (url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VolleyIQ...")
    finally:
        server.server_close()
    print("VolleyIQ has stopped.")
    return 0


def redirect_output(path):
    """Send print() output, logging and errors to `path` (appending; restarted when large)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > LOG_LIMIT_BYTES:
        path.unlink()
    stream = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = stream
    return stream


if __name__ == "__main__":
    sys.exit(main())
