"""Manually mark the volleyball's centre in extracted frames.

A person clicks on the ball in each frame and the pixel coordinates are saved
to a CSV file. This is manual ground-truth labelling: no AI detection is
involved. The saved labels can later be used to check or train a detector.

Every change is saved to the CSV immediately, so closing the window or
quitting never loses work. Re-running the tool on the same CSV resumes
where you left off.

Usage:
    python -m volleyiq.annotate_ball FRAMES_DIR OUTPUT_CSV

Controls:
    Left click        mark the ball centre (click again to correct it)
    D or Right arrow  next frame
    A or Left arrow   previous frame
    S                 skip: ball not visible in this frame, then go to next
    C                 clear this frame's mark (back to unannotated)
    Q or Esc          save and quit
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CSV_COLUMNS = ["frame_name", "x", "y", "status"]

# Status values written to the CSV. Unannotated frames have a blank status.
LABELLED = "labelled"
SKIPPED = "skipped"

WINDOW_NAME = "VolleyIQ - manual ball annotation"


def natural_key(path):
    """Sort key so frame_2.jpg comes before frame_10.jpg."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def list_frames(frames_dir):
    """Return the image files in `frames_dir`, sorted by frame number.

    Raises ValueError with a readable message if the folder is missing,
    is not a folder, or contains no images.
    """
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise ValueError(f"Frames folder not found: {frames_dir}")
    if not frames_dir.is_dir():
        raise ValueError(f"Not a folder: {frames_dir}")

    frames = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    if not frames:
        raise ValueError(
            f"No images found in: {frames_dir}\n"
            f"Expected files ending in {', '.join(sorted(IMAGE_EXTENSIONS))}. "
            "Run volleyiq.extract_frames first."
        )
    return sorted(frames, key=natural_key)


def load_image(path):
    """Read an image file, raising ValueError if it cannot be decoded.

    Uses np.fromfile + cv2.imdecode instead of cv2.imread so that paths with
    non-English characters also work on Windows.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError as error:
        raise ValueError(f"Could not read image file: {path} ({error})") from error
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise ValueError(f"Could not decode image (corrupted or unsupported): {path}")
    return image


class AnnotationStore:
    """Ball positions for a list of frames, saved to and loaded from a CSV.

    Each frame is either unannotated (no entry), labelled with (x, y), or
    skipped (ball not visible).
    """

    def __init__(self, frame_names, csv_path):
        self.frame_names = list(frame_names)
        self.csv_path = Path(csv_path)
        self._entries = {}  # frame_name -> (status, x, y)
        self._other_rows = []  # rows for frames not in this folder, kept as-is

    def get(self, frame_name):
        """Return (status, x, y); status is None for unannotated frames."""
        return self._entries.get(frame_name, (None, None, None))

    def set_point(self, frame_name, x, y):
        self._entries[frame_name] = (LABELLED, int(x), int(y))

    def skip(self, frame_name):
        self._entries[frame_name] = (SKIPPED, None, None)

    def clear(self, frame_name):
        self._entries.pop(frame_name, None)

    def counts(self):
        """Return (labelled, skipped, unannotated) frame counts."""
        statuses = [self.get(name)[0] for name in self.frame_names]
        return statuses.count(LABELLED), statuses.count(SKIPPED), statuses.count(None)

    def first_unannotated_index(self):
        for index, name in enumerate(self.frame_names):
            if self.get(name)[0] is None:
                return index
        return 0

    def load(self):
        """Load an existing CSV, if there is one.

        Raises ValueError if the file exists but is not a valid annotation
        CSV, so a mistyped path never overwrites an unrelated file.
        Returns the names of rows that do not match any frame. Those rows
        are not shown, but are kept and written back unchanged on save.
        """
        if not self.csv_path.exists():
            return []
        # utf-8-sig also accepts the hidden marker Excel adds when re-saving a CSV.
        with open(self.csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = {"frame_name", "x", "y"} - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{self.csv_path} is not a VolleyIQ annotation CSV "
                    f"(missing columns: {', '.join(sorted(missing))}). "
                    "Choose a different output path."
                )
            known = set(self.frame_names)
            unknown = []
            for line_number, row in enumerate(reader, start=2):
                name = row["frame_name"]
                if name not in known:
                    unknown.append(name)
                    self._other_rows.append([row.get(column) or "" for column in CSV_COLUMNS])
                    continue
                x, y = (row["x"] or "").strip(), (row["y"] or "").strip()
                status = (row.get("status") or "").strip()
                if x and y:
                    try:
                        self.set_point(name, int(float(x)), int(float(y)))
                    except ValueError:
                        raise ValueError(
                            f"Bad coordinates on line {line_number} of {self.csv_path}: x={x!r}, y={y!r}"
                        ) from None
                elif status == SKIPPED:
                    self.skip(name)
        return unknown

    def save(self):
        """Write one row per frame, in frame order.

        Writes to a temporary file first and then swaps it in, so a crash
        part-way through never leaves a half-written CSV.
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.csv_path.with_name(self.csv_path.name + ".tmp")
        with open(temp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for name in self.frame_names:
                status, x, y = self.get(name)
                writer.writerow([name, "" if x is None else x, "" if y is None else y, status or ""])
            writer.writerows(self._other_rows)
        os.replace(temp_path, self.csv_path)


class Annotator:
    """The annotation session without any window: which frame is shown,
    what happens on a click or key, and saving after every change.

    Kept separate from the OpenCV window so it can be tested automatically.
    """

    def __init__(self, frame_paths, csv_path):
        self.frame_paths = list(frame_paths)
        self.store = AnnotationStore([p.name for p in self.frame_paths], csv_path)
        self.unknown_rows = self.store.load()
        self.index = self.store.first_unannotated_index()

    @property
    def frame_name(self):
        return self.frame_paths[self.index].name

    @property
    def frame_path(self):
        return self.frame_paths[self.index]

    def click(self, x, y, image_width, image_height):
        """Record a click at image pixel (x, y). Clicks outside the image are ignored."""
        if not (0 <= x < image_width and 0 <= y < image_height):
            return False
        self.store.set_point(self.frame_name, x, y)
        self.store.save()
        return True

    def next(self):
        self.index = min(self.index + 1, len(self.frame_paths) - 1)

    def previous(self):
        self.index = max(self.index - 1, 0)

    def skip(self):
        self.store.skip(self.frame_name)
        self.store.save()
        self.next()

    def clear(self):
        self.store.clear(self.frame_name)
        self.store.save()


def display_scale(image_width, image_height, max_width, max_height):
    """Scale factor (at most 1.0) that fits the image inside the window limits."""
    return min(1.0, max_width / image_width, max_height / image_height)


def display_to_image(x, y, scale):
    """Convert a click on the (possibly shrunk) window back to original image pixels."""
    return round(x / scale), round(y / scale)


# --- Everything below needs a screen and a person ---------------------------

# cv2.waitKeyEx codes for arrow keys on Windows.
KEY_LEFT = 2424832
KEY_RIGHT = 2555904
KEY_ESC = 27


# Info panel above the image and help bar below it, so they never hide the ball.
TOP_PANEL = 74
HELP_BAR = 26


def draw_overlay(image, annotator, scale):
    """Return a display image: info panel, the frame with its mark, help bar.

    The frame itself starts TOP_PANEL pixels down from the top of the window.
    """
    height, width = image.shape[:2]
    shown = cv2.resize(image, (round(width * scale), round(height * scale))) if scale < 1 else image.copy()

    status, x, y = annotator.store.get(annotator.frame_name)
    if status == LABELLED:
        cx, cy = round(x * scale), round(y * scale)
        cv2.circle(shown, (cx, cy), 10, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(shown, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        status_text = f"ball at x={x}, y={y}"
    elif status == SKIPPED:
        status_text = "SKIPPED (ball not visible)"
    else:
        status_text = "not annotated - click the ball centre"

    labelled, skipped, todo = annotator.store.counts()
    lines_top = [
        f"Frame {annotator.index + 1}/{len(annotator.frame_paths)}: {annotator.frame_name}",
        status_text,
        f"labelled {labelled} | skipped {skipped} | to do {todo}",
    ]
    help_text = "Click: mark  D/Right: next  A/Left: prev  S: skip  C: clear  Q/Esc: quit"

    font = cv2.FONT_HERSHEY_SIMPLEX
    canvas = cv2.copyMakeBorder(shown, TOP_PANEL, HELP_BAR, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for i, text in enumerate(lines_top):
        cv2.putText(canvas, text, (8, 22 * (i + 1)), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, help_text, (8, canvas.shape[0] - 8), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def error_image(message, width=640, height=360):
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.putText(image, "Could not read this image:", (20, height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(image, message[:70], (20, height // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def run_window(annotator, max_width, max_height):
    """Show frames in an OpenCV window until the user quits or closes it."""
    current = {"scale": 1.0, "size": (0, 0), "readable": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and current["readable"]:
            # Clicks on the info panel or help bar fall outside the image and are ignored.
            image_x, image_y = display_to_image(x, y - TOP_PANEL, current["scale"])
            annotator.click(image_x, image_y, *current["size"])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        loaded_index, image = None, None
        while True:
            if loaded_index != annotator.index:
                try:
                    image = load_image(annotator.frame_path)
                    current["readable"] = True
                except ValueError as error:
                    print(f"Warning: {error}", file=sys.stderr)
                    image = error_image(annotator.frame_name)
                    current["readable"] = False
                height, width = image.shape[:2]
                current["size"] = (width, height)
                current["scale"] = display_scale(width, height, max_width, max_height)
                loaded_index = annotator.index

            cv2.imshow(WINDOW_NAME, draw_overlay(image, annotator, current["scale"]))
            key = cv2.waitKeyEx(30)

            # Closing the window with its X button ends the session.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == -1:
                continue
            char = chr(key & 0xFF).lower() if key < 256 else ""
            if key == KEY_ESC or char == "q":
                break
            if key == KEY_RIGHT or char == "d":
                annotator.next()
            elif key == KEY_LEFT or char == "a":
                annotator.previous()
            elif char == "s":
                annotator.skip()
            elif char == "c":
                annotator.clear()
    finally:
        cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Manually mark the volleyball centre in extracted frames and save to CSV."
    )
    parser.add_argument("frames_dir", help="folder of extracted frames, e.g. data/frames/rally1")
    parser.add_argument("output_csv", help="CSV file to save to, e.g. data/annotations/rally1_ball.csv")
    parser.add_argument("--max-width", type=int, default=1280, help="largest window width in pixels (default: 1280)")
    parser.add_argument("--max-height", type=int, default=720, help="largest window height in pixels (default: 720)")
    args = parser.parse_args(argv)

    try:
        if Path(args.output_csv).suffix.lower() != ".csv":
            raise ValueError(f"Output must end in .csv, got: {args.output_csv}")
        frame_paths = list_frames(args.frames_dir)
        annotator = Annotator(frame_paths, args.output_csv)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if annotator.unknown_rows:
        print(
            f"Note: {len(annotator.unknown_rows)} CSV row(s) are for frames not in {args.frames_dir}. "
            "They will be kept unchanged."
        )
    # Write the CSV straight away so it lists every frame, even before any clicks.
    annotator.store.save()

    labelled, skipped, todo = annotator.store.counts()
    print(f"{len(frame_paths)} frames: {labelled} labelled, {skipped} skipped, {todo} to do.")
    print(f"Saving to {args.output_csv} after every change. Press Q or Esc to quit.")

    run_window(annotator, args.max_width, args.max_height)

    labelled, skipped, todo = annotator.store.counts()
    print(f"Saved {args.output_csv}: {labelled} labelled, {skipped} skipped, {todo} to do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
