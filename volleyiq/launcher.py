"""Start the VolleyIQ web app without a console window, then open it in the browser.

Used by VolleyIQ.vbs (double-click it), which runs this file with the
project's .venv\\Scripts\\pythonw.exe and no window. VolleyIQ.pyw runs it too,
on computers where .pyw files open with Python. This file uses only Python's
standard library, so either Python can run it.

Usage (normally without options):
    pythonw volleyiq\\launcher.py [--port 8765] [--data-dir DIR] [--no-browser]

Exit codes: 0 = VolleyIQ is running; 2 = it isn't, and a message box has
already told the user why. Anything else means Python itself failed, and
VolleyIQ.vbs shows its own message.

What it does:
  1. If VolleyIQ already answers on http://127.0.0.1:8765, just open the browser
     (with a note if it is an older version that should be restarted).
  2. Otherwise start ".venv\\Scripts\\pythonw.exe -m volleyiq.webapp" in the
     background (no window), with its output in app_data\\logs\\server.log.
  3. Wait until /api/health answers, then open the browser.
  4. If the server exits or doesn't answer in time, show a Windows message box
     with the end of the log, and suggest VolleyIQ.bat.

To stop the background server, use "Stop VolleyIQ" in the web page.
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path

APP_NAME = "VolleyIQ"
EXIT_OK, EXIT_REPORTED = 0, 2  # VolleyIQ.vbs relies on these values
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORT = 8765
URL = f"http://127.0.0.1:{PORT}/"
LOG_FILE = PROJECT_ROOT / "app_data" / "logs" / "server.log"
STARTUP_TIMEOUT_S = 120  # the first start imports PyTorch, which can be slow
MB_ICON_ERROR, MB_ICON_INFO = 0x10, 0x40


def venv_pythonw(root=PROJECT_ROOT):
    return Path(root) / ".venv" / "Scripts" / "pythonw.exe"


def _get_json(url, timeout):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        return None


def is_running(url=URL, timeout=1.0):
    """True if a current VolleyIQ server answers at `url` (not just anything on that port)."""
    data = _get_json(url + "api/health", timeout)
    return isinstance(data, dict) and data.get("app") == APP_NAME


def running_version(url=URL, timeout=1.0):
    """"current", "older" (a VolleyIQ from before /api/health existed), or None."""
    if is_running(url, timeout):
        return "current"
    data = _get_json(url + "api/config", timeout)
    if isinstance(data, dict) and "allowed_extensions" in data and "max_upload_bytes" in data:
        return "older"
    return None


def log_tail(path=LOG_FILE, lines=12):
    """The last few lines of the server log, or a note that there is none."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no log file was written)"
    return "\n".join(text.strip().splitlines()[-lines:]) or "(the log file is empty)"


def show_message(text, error=True):
    """A Windows message box: the only way to report anything without a console."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, APP_NAME, MB_ICON_ERROR if error else MB_ICON_INFO)
    except (ImportError, AttributeError, OSError):
        print(text, file=sys.stderr)


def start_server(pythonw, log_file=LOG_FILE, port=PORT, data_dir=None):
    """Start the server in the background, with no window. Returns the process."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    command = [str(pythonw), "-m", "volleyiq.webapp", "--no-browser", "--port", str(port), "--log-file", str(log_file)]
    if data_dir:
        command += ["--data-dir", str(data_dir)]
    return subprocess.Popen(
        command, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )


def wait_until_ready(process, url=URL, timeout=STARTUP_TIMEOUT_S, poll_s=0.5):
    """Return "ready", "exited" (the process stopped) or "timeout"."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_running(url):
            return "ready"
        if process.poll() is not None:
            return "exited"
        time.sleep(poll_s)
    return "timeout"


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Start VolleyIQ in the background and open it in the browser.")
    parser.add_argument("--port", type=int, default=PORT, help=f"port to use (default {PORT})")
    parser.add_argument("--data-dir", help="where the app keeps jobs and logs (default app_data)")
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    url = f"http://127.0.0.1:{args.port}/"
    log_file = Path(args.data_dir) / "logs" / "server.log" if args.data_dir else LOG_FILE

    def open_browser():
        if not args.no_browser:
            webbrowser.open(url)

    version = running_version(url)
    if version == "current":
        open_browser()
        return EXIT_OK
    if version == "older":
        show_message("An older version of VolleyIQ is already running (probably started with VolleyIQ.bat). "
                     "It will open now.\n\nTo use the new version, stop it (close its console window or press "
                     "Ctrl+C in it), then start VolleyIQ again.", error=False)
        open_browser()
        return EXIT_OK

    pythonw = venv_pythonw()
    if not pythonw.is_file():
        show_message(f"Could not find the project's Python environment:\n{pythonw}\n\n"
                     "See \"Setup\" in README.md, or use VolleyIQ.bat.")
        return EXIT_REPORTED

    process = start_server(pythonw, log_file, args.port, args.data_dir)
    result = wait_until_ready(process, url)
    if result == "ready":
        open_browser()
        return EXIT_OK
    if result == "exited":
        show_message(f"VolleyIQ could not start (exit code {process.returncode}).\n\n"
                     f"Last lines of {log_file}:\n\n{log_tail(log_file)}\n\n"
                     "Double-click VolleyIQ.bat to see the full output.")
    else:
        show_message(f"VolleyIQ did not answer within {STARTUP_TIMEOUT_S} seconds. It may still be starting; "
                     f"try opening {url} in a moment.\n\nLast lines of {log_file}:\n\n{log_tail(log_file)}")
    return EXIT_REPORTED


def run(argv=None):
    """main(), but any unexpected error is shown in a message box: with no console, it would vanish."""
    try:
        return main(argv)
    except SystemExit as stop:  # bad command-line options
        if stop.code not in (0, None):
            show_message("VolleyIQ's launcher was given options it doesn't understand.")
            return EXIT_REPORTED
        return EXIT_OK
    except Exception:
        show_message("VolleyIQ's launcher hit an unexpected error:\n\n" + traceback.format_exc(limit=5)
                     + "\nDouble-click VolleyIQ.bat to start VolleyIQ with a console window instead.")
        return EXIT_REPORTED


if __name__ == "__main__":
    sys.exit(run())
