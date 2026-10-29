"""Checks for the no-console launcher (VolleyIQ.vbs / VolleyIQ.pyw -> volleyiq.launcher).

Run from the VolleyIQ folder:
    .\\.venv\\Scripts\\python.exe -B -m unittest discover -s tests -v

The VolleyIQ web app is never really started: process creation, the browser
and message boxes are replaced with stand-ins, and a small real server
answers the health check where one is needed. VolleyIQ.vbs itself is run
with cscript against throwaway copies of the project folder, with a stand-in
launcher, so its error handling can be checked without any message box.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from volleyiq import launcher


def serve(payload):
    """A local server whose /api/health returns `payload`. Returns (server, base url)."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/"


class FakeProcess:
    def __init__(self, exit_after=None):
        self.polls, self.exit_after, self.returncode = 0, exit_after, None

    def poll(self):
        self.polls += 1
        if self.exit_after is not None and self.polls >= self.exit_after:
            self.returncode = 1
        return self.returncode


class IsRunningTests(unittest.TestCase):
    def test_recognises_volleyiq(self):
        server, url = serve({"app": "VolleyIQ", "status": "ok"})
        try:
            self.assertTrue(launcher.is_running(url))
        finally:
            server.shutdown()
            server.server_close()

    def test_ignores_other_programs_on_the_port(self):
        server, url = serve({"app": "something else"})
        try:
            self.assertFalse(launcher.is_running(url))
        finally:
            server.shutdown()
            server.server_close()

    def test_nothing_listening(self):
        server, url = serve({})
        server.shutdown()
        server.server_close()
        self.assertFalse(launcher.is_running(url, timeout=0.5))
        self.assertIsNone(launcher.running_version(url, timeout=0.5))

    def test_running_version(self):
        for payload, expected in (
            ({"app": "VolleyIQ"}, "current"),
            # An older VolleyIQ has no /api/health but answers /api/config (this stand-in answers every path).
            ({"allowed_extensions": [".mp4"], "max_upload_bytes": 1}, "older"),
            ({"hello": "world"}, None),
        ):
            server, url = serve(payload)
            try:
                self.assertEqual(launcher.running_version(url), expected, payload)
            finally:
                server.shutdown()
                server.server_close()


class WaitUntilReadyTests(unittest.TestCase):
    def test_ready(self):
        server, url = serve({"app": "VolleyIQ"})
        try:
            self.assertEqual(launcher.wait_until_ready(FakeProcess(), url, timeout=5, poll_s=0.01), "ready")
        finally:
            server.shutdown()
            server.server_close()

    def test_process_exited(self):
        with mock.patch.object(launcher, "is_running", return_value=False):
            self.assertEqual(launcher.wait_until_ready(FakeProcess(exit_after=3), timeout=5, poll_s=0.01), "exited")

    def test_timeout(self):
        with mock.patch.object(launcher, "is_running", return_value=False):
            self.assertEqual(launcher.wait_until_ready(FakeProcess(), timeout=0.05, poll_s=0.01), "timeout")


class StartServerTests(unittest.TestCase):
    def test_starts_the_venv_pythonw_without_a_window(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(subprocess, "Popen") as popen:
            log = Path(tmp) / "logs" / "server.log"
            launcher.start_server(Path("C:/x/.venv/Scripts/pythonw.exe"), log, 8765)
            self.assertTrue(log.parent.is_dir())
        args, kwargs = popen.call_args
        self.assertEqual(args[0], [str(Path("C:/x/.venv/Scripts/pythonw.exe")), "-m", "volleyiq.webapp",
                                   "--no-browser", "--port", "8765", "--log-file", str(log)])
        self.assertTrue(kwargs["creationflags"] & getattr(subprocess, "CREATE_NO_WINDOW", 0) or not hasattr(subprocess, "CREATE_NO_WINDOW"))
        self.assertEqual(kwargs["cwd"], str(launcher.PROJECT_ROOT))

    def test_optional_data_dir(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(subprocess, "Popen") as popen:
            launcher.start_server(Path("p.exe"), Path(tmp) / "s.log", 9000, data_dir=Path(tmp) / "data")
        self.assertEqual(popen.call_args[0][0][-2:], ["--data-dir", str(Path(tmp) / "data")])

    def test_venv_path(self):
        self.assertEqual(launcher.venv_pythonw(Path("C:/p")), Path("C:/p/.venv/Scripts/pythonw.exe"))


class MainTests(unittest.TestCase):
    def setUp(self):
        patches = {
            "open": mock.patch.object(launcher.webbrowser, "open"),
            "message": mock.patch.object(launcher, "show_message"),
            "start": mock.patch.object(launcher, "start_server"),
        }
        self.mocks = {name: p.start() for name, p in patches.items()}
        for p in patches.values():
            self.addCleanup(p.stop)

    def test_already_running_just_opens_the_browser(self):
        with mock.patch.object(launcher, "running_version", return_value="current"):
            self.assertEqual(launcher.main([]), launcher.EXIT_OK)
        self.mocks["open"].assert_called_once_with(launcher.URL)
        self.mocks["start"].assert_not_called()
        self.mocks["message"].assert_not_called()

    def test_older_version_running_is_opened_with_a_note(self):
        with mock.patch.object(launcher, "running_version", return_value="older"):
            self.assertEqual(launcher.main([]), launcher.EXIT_OK)
        self.mocks["open"].assert_called_once_with(launcher.URL)
        self.mocks["start"].assert_not_called()
        text, = self.mocks["message"].call_args[0]
        self.assertIn("older version", text)
        self.assertEqual(self.mocks["message"].call_args[1], {"error": False})

    def test_starts_then_opens_the_browser(self):
        with mock.patch.object(launcher, "running_version", return_value=None), \
                mock.patch.object(launcher, "venv_pythonw", return_value=Path(__file__)), \
                mock.patch.object(launcher, "wait_until_ready", return_value="ready"):
            self.assertEqual(launcher.main([]), launcher.EXIT_OK)
        self.mocks["start"].assert_called_once()
        self.mocks["open"].assert_called_once_with(launcher.URL)
        self.mocks["message"].assert_not_called()

    def test_startup_failure_is_reported_with_the_log(self):
        process = FakeProcess()
        process.returncode = 1
        self.mocks["start"].return_value = process
        with mock.patch.object(launcher, "running_version", return_value=None), \
                mock.patch.object(launcher, "venv_pythonw", return_value=Path(__file__)), \
                mock.patch.object(launcher, "wait_until_ready", return_value="exited"), \
                mock.patch.object(launcher, "log_tail", return_value="OSError: port in use"):
            self.assertEqual(launcher.main([]), launcher.EXIT_REPORTED)
        text = self.mocks["message"].call_args[0][0]
        self.assertIn("could not start (exit code 1)", text)
        self.assertIn("OSError: port in use", text)
        self.assertIn("VolleyIQ.bat", text)
        self.mocks["open"].assert_not_called()

    def test_missing_environment(self):
        with mock.patch.object(launcher, "running_version", return_value=None), \
                mock.patch.object(launcher, "venv_pythonw", return_value=Path("C:/nowhere/pythonw.exe")):
            self.assertEqual(launcher.main([]), launcher.EXIT_REPORTED)
        self.assertIn("Could not find", self.mocks["message"].call_args[0][0])
        self.mocks["start"].assert_not_called()


class LogTailTests(unittest.TestCase):
    def test_tail_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "server.log"
            log.write_text("\n".join(f"line {i}" for i in range(30)), encoding="utf-8")
            self.assertEqual(launcher.log_tail(log, lines=2), "line 28\nline 29")
            self.assertEqual(launcher.log_tail(Path(tmp) / "none.log"), "(no log file was written)")


class OptionsTests(unittest.TestCase):
    def test_port_data_dir_and_no_browser(self):
        process = FakeProcess()
        with mock.patch.object(launcher, "running_version", return_value=None) as version, \
                mock.patch.object(launcher, "venv_pythonw", return_value=Path(__file__)), \
                mock.patch.object(launcher, "start_server", return_value=process) as start, \
                mock.patch.object(launcher, "wait_until_ready", return_value="ready") as wait, \
                mock.patch.object(launcher.webbrowser, "open") as browser:
            self.assertEqual(launcher.main(["--port", "8799", "--data-dir", "D:/x", "--no-browser"]), launcher.EXIT_OK)
        version.assert_called_once_with("http://127.0.0.1:8799/")
        start.assert_called_once_with(Path(__file__), Path("D:/x") / "logs" / "server.log", 8799, "D:/x")
        self.assertEqual(wait.call_args[0], (process, "http://127.0.0.1:8799/"))
        browser.assert_not_called()


class RunWrapperTests(unittest.TestCase):
    def test_unexpected_errors_are_shown_not_lost(self):
        with mock.patch.object(launcher, "main", side_effect=RuntimeError("boom")), \
                mock.patch.object(launcher, "show_message") as message:
            self.assertEqual(launcher.run([]), launcher.EXIT_REPORTED)
        self.assertIn("RuntimeError: boom", message.call_args[0][0])
        self.assertIn("VolleyIQ.bat", message.call_args[0][0])

    def test_bad_options_are_reported(self):
        with mock.patch.object(launcher, "show_message") as message, mock.patch("sys.stderr"):
            self.assertEqual(launcher.run(["--nonsense"]), launcher.EXIT_REPORTED)
        message.assert_called_once()

    def test_success_passes_through(self):
        with mock.patch.object(launcher, "main", return_value=launcher.EXIT_OK):
            self.assertEqual(launcher.run([]), launcher.EXIT_OK)


VBS = launcher.PROJECT_ROOT / "VolleyIQ.vbs"
CSCRIPT = shutil.which("cscript")


@unittest.skipUnless(sys.platform == "win32" and CSCRIPT, "VolleyIQ.vbs needs Windows Script Host")
class VbsTests(unittest.TestCase):
    """Run VolleyIQ.vbs from a throwaway folder laid out like the project."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "Volley IQ copy"  # a space, like real Windows paths
        self.root.mkdir()
        shutil.copy(VBS, self.root / "VolleyIQ.vbs")

    def tearDown(self):
        self._tmp.cleanup()

    def add_python(self, launcher_code):
        """A working copy of the venv's pythonw, plus a stand-in volleyiq/launcher.py."""
        venv = launcher.PROJECT_ROOT / ".venv"
        scripts = self.root / ".venv" / "Scripts"
        scripts.mkdir(parents=True)
        shutil.copy(venv / "Scripts" / "pythonw.exe", scripts / "pythonw.exe")
        shutil.copy(venv / "pyvenv.cfg", self.root / ".venv" / "pyvenv.cfg")
        (self.root / "volleyiq").mkdir()
        (self.root / "volleyiq" / "launcher.py").write_text(launcher_code, encoding="utf-8")

    def run_vbs(self, *args):
        # Started from another folder, to show the script doesn't rely on the working directory.
        result = subprocess.run([CSCRIPT, "//nologo", str(self.root / "VolleyIQ.vbs"), *args],
                                capture_output=True, text=True, timeout=60, cwd=tempfile.gettempdir())
        return result.returncode, result.stdout.strip()

    def test_missing_environment_is_explained(self):
        code, output = self.run_vbs()
        self.assertEqual(code, 1)
        self.assertIn("Could not find the project's Python environment", output)
        self.assertIn(str(self.root / ".venv" / "Scripts" / "pythonw.exe"), output)

    def test_runs_the_launcher_with_absolute_paths_and_options(self):
        record = self.root / "seen.json"
        self.add_python(
            "import json, os, sys\n"
            f"open({str(record)!r}, 'w').write(json.dumps({{'argv': sys.argv, 'cwd': os.getcwd(), "
            "'stdout': sys.stdout is None}))\n"
            "sys.exit(0)\n"
        )
        code, output = self.run_vbs("--port", "8799", "--no-browser")
        self.assertEqual((code, output), (0, ""))
        seen = json.loads(record.read_text(encoding="utf-8"))
        self.assertEqual(Path(seen["argv"][0]), self.root / "volleyiq" / "launcher.py")
        self.assertEqual(seen["argv"][1:], ["--port", "8799", "--no-browser"])
        self.assertEqual(Path(seen["cwd"]), self.root)
        self.assertTrue(seen["stdout"])  # pythonw: no console attached

    def test_reported_failure_stays_quiet(self):
        self.add_python("import sys\nsys.exit(2)\n")  # the launcher already showed a message box
        self.assertEqual(self.run_vbs(), (2, ""))

    def test_python_crash_is_explained(self):
        self.add_python("import sys\nsys.exit(7)\n")
        code, output = self.run_vbs()
        self.assertEqual(code, 1)
        self.assertIn("exit code 7", output)
        self.assertIn("VolleyIQ.bat", output)


if __name__ == "__main__":
    unittest.main()
