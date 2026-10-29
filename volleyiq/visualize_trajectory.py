"""Draw the ball's path from a manual annotation CSV.

The positions come from volleyiq.annotate_ball, where a person clicked on the
ball. This tool only draws those clicks: it does not detect the ball, and it
never fills in positions for skipped or unannotated frames.

Two outputs:

  image  The whole path drawn on one extracted frame, saved as a picture.
  video  The original video with the path growing as the video plays.

Usage:
    python -m volleyiq.visualize_trajectory image CSV FRAMES_DIR OUTPUT_PNG [--frame NAME]
    python -m volleyiq.visualize_trajectory video CSV VIDEO OUTPUT_MP4

How lines are drawn: points from consecutive annotation rows are joined by a
line. If a frame between them was skipped (ball not visible) or not annotated
yet, the line is broken there, because we don't know where the ball went.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2

from volleyiq.annotate_ball import LABELLED, list_frames, load_image

# Colours are BGR (blue, green, red), which is OpenCV's order.
LINE_COLOUR = (0, 165, 255)  # orange
POINT_COLOUR = (0, 0, 255)  # red
START_COLOUR = (0, 200, 0)  # green
LABEL_COLOUR = (255, 255, 255)
LABEL_TEXT = "Manual annotations - not automatic detection"


def frame_number_from_name(frame_name):
    """Return the original video frame number from a name like frame_000120.jpg.

    extract_frames.py names every image after its frame number in the video,
    which is what lets us put each point back on the right video frame.
    """
    numbers = re.findall(r"\d+", Path(frame_name).stem)
    if not numbers:
        raise ValueError(
            f"Cannot find a frame number in {frame_name!r}. "
            "Expected names like frame_000120.jpg from volleyiq.extract_frames."
        )
    return int(numbers[-1])


def read_annotations(csv_path):
    """Read an annotation CSV.

    Returns a list of rows sorted by frame number. Each row is a dict:
        {"frame_name", "frame_number", "point"}
    where "point" is (x, y) for labelled frames and None for skipped or
    unannotated ones. Raises ValueError with a readable message on problems.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise ValueError(f"Annotation CSV not found: {csv_path}")

    # utf-8-sig also accepts the hidden marker Excel adds when re-saving a CSV.
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = {"frame_name", "x", "y"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{csv_path} is not a VolleyIQ annotation CSV "
                f"(missing columns: {', '.join(sorted(missing))})."
            )

        rows = []
        for line_number, row in enumerate(reader, start=2):
            x, y = (row["x"] or "").strip(), (row["y"] or "").strip()
            status = (row.get("status") or "").strip()
            point = None
            # Only a row with both coordinates (and not marked skipped) is a point.
            if x and y and status in ("", LABELLED):
                try:
                    point = (int(float(x)), int(float(y)))
                except ValueError:
                    raise ValueError(
                        f"Bad coordinates on line {line_number} of {csv_path}: x={x!r}, y={y!r}"
                    ) from None
            rows.append({
                "frame_name": row["frame_name"],
                "frame_number": frame_number_from_name(row["frame_name"]),
                "point": point,
            })

    if not any(row["point"] for row in rows):
        raise ValueError(f"No labelled ball positions in {csv_path}. Annotate some frames first.")
    return sorted(rows, key=lambda row: row["frame_number"])


def split_into_segments(rows):
    """Group labelled rows into unbroken runs.

    A skipped or unannotated row ends the current run, so no line is ever
    drawn across a frame where the ball's position is unknown.
    Returns a list of segments; each segment is a list of rows.
    """
    segments, current = [], []
    for row in rows:
        if row["point"] is None:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(row)
    if current:
        segments.append(current)
    return segments


def draw_trajectory(image, rows, up_to_frame=None):
    """Draw the path on `image` (changed in place) and return it.

    If `up_to_frame` is given, only points at or before that video frame are
    drawn, which makes the path grow during a video.
    """
    if up_to_frame is not None:
        rows = [row for row in rows if row["frame_number"] <= up_to_frame]

    for segment in split_into_segments(rows):
        points = [row["point"] for row in segment]
        for start, end in zip(points, points[1:]):
            cv2.line(image, start, end, LINE_COLOUR, 2, cv2.LINE_AA)

    labelled = [row for row in rows if row["point"]]
    for row in labelled:
        cv2.circle(image, row["point"], 4, POINT_COLOUR, -1, cv2.LINE_AA)
    # Draw the first point last so a later point in the same place can't hide it.
    if labelled:
        cv2.circle(image, labelled[0]["point"], 4, START_COLOUR, -1, cv2.LINE_AA)

    # Label in the bottom-right corner on a dark box, readable on any footage.
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.45
    (text_w, text_h), baseline = cv2.getTextSize(LABEL_TEXT, font, scale, 1)
    height, width = image.shape[:2]
    left, top = width - text_w - 12, height - text_h - baseline - 10
    cv2.rectangle(image, (left - 4, top - 4), (width, height), (0, 0, 0), -1)
    cv2.putText(image, LABEL_TEXT, (left, height - baseline - 6), font, scale, LABEL_COLOUR, 1, cv2.LINE_AA)
    return image


def check_points_fit(rows, width, height):
    """Raise ValueError if any point lies outside a width x height image."""
    for row in rows:
        if row["point"]:
            x, y = row["point"]
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    f"Point ({x}, {y}) for {row['frame_name']} is outside the "
                    f"{width} x {height} image. Is this the right CSV?"
                )


def save_image(image, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(output_path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image as {output_path.suffix}: {output_path}")
    data.tofile(str(output_path))


def make_trajectory_image(csv_path, frames_dir, output_path, frame_name=None):
    """Draw the full path on one extracted frame and save it.

    By default the first labelled frame is used as the background.
    Returns the number of labelled points drawn.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        raise ValueError(f"Output image must end in .png or .jpg, got: {output_path}")

    rows = read_annotations(csv_path)
    frames = {path.name: path for path in list_frames(frames_dir)}
    if frame_name is None:
        frame_name = next(row["frame_name"] for row in rows if row["point"])
    if frame_name not in frames:
        raise ValueError(f"Frame {frame_name!r} not found in {frames_dir}")

    image = load_image(frames[frame_name])
    check_points_fit(rows, image.shape[1], image.shape[0])
    draw_trajectory(image, rows)
    save_image(image, output_path)
    return sum(1 for row in rows if row["point"])


def make_trajectory_video(csv_path, video_path, output_path):
    """Copy the video with the path drawn on every frame, growing over time.

    Each point appears on the video frame it was annotated on, using the
    frame number in its file name. Returns the number of frames written.
    """
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output video must end in .mp4, got: {output_path}")
    video_path = Path(video_path)
    if not video_path.is_file():
        raise ValueError(f"Video not found: {video_path}")

    rows = read_annotations(csv_path)
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        check_points_fit(rows, width, height)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open a video writer for: {output_path}")

        # Read frames in order so frame numbers match extract_frames exactly.
        frame_number = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(draw_trajectory(frame, rows, up_to_frame=frame_number))
            frame_number += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    last_annotated = max(row["frame_number"] for row in rows)
    if last_annotated >= frame_number:
        output_path.unlink(missing_ok=True)
        raise ValueError(
            f"The CSV mentions frame {last_annotated}, but {video_path} has only "
            f"{frame_number} frames. Is this the video the frames were extracted from?"
        )
    return frame_number


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Draw the ball path from a manual annotation CSV (not automatic detection)."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    image_cmd = commands.add_parser("image", help="draw the whole path on one frame")
    image_cmd.add_argument("csv", help="annotation CSV from volleyiq.annotate_ball")
    image_cmd.add_argument("frames_dir", help="folder of extracted frames")
    image_cmd.add_argument("output", help="image to save, e.g. outputs/rally1_trajectory.png")
    image_cmd.add_argument("--frame", help="frame to draw on (default: first labelled frame)")

    video_cmd = commands.add_parser("video", help="draw the growing path on the original video")
    video_cmd.add_argument("csv", help="annotation CSV from volleyiq.annotate_ball")
    video_cmd.add_argument("video", help="the original video the frames were extracted from")
    video_cmd.add_argument("output", help="video to save, e.g. outputs/rally1_trajectory.mp4")

    args = parser.parse_args(argv)
    try:
        if args.command == "image":
            count = make_trajectory_image(args.csv, args.frames_dir, args.output, args.frame)
            print(f"Drew {count} manually labelled point(s). Saved {args.output}")
        else:
            frames = make_trajectory_video(args.csv, args.video, args.output)
            print(f"Wrote {frames} frames with the manual trajectory. Saved {args.output}")
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
