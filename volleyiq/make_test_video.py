"""Generate a short synthetic volleyball scene for testing the pipeline.

The video is drawn entirely with OpenCV shapes: a court floor, a net and a
ball flying back and forth over the net in smooth arcs. Every frame is
labelled "SYNTHETIC TEST VIDEO" and shows a frame counter.

This is NOT real match footage. It exists only to check that tools such as
extract_frames.py can read, process and write video. It says nothing about
how accurately any AI model tracks a real ball.

Usage:
    python -m volleyiq.make_test_video [--output PATH] [--duration SECONDS]
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

DEFAULT_OUTPUT = Path("data") / "videos" / "synthetic_test.mp4"

# Colours are BGR (blue, green, red), which is OpenCV's order.
SKY = (235, 206, 135)
FLOOR = (90, 140, 210)
LINE = (255, 255, 255)
NET_POST = (60, 60, 60)
NET_MESH = (40, 40, 40)
NET_TAPE = (245, 245, 245)
BALL = (40, 215, 245)
BALL_EDGE = (20, 70, 150)
BANNER = (0, 0, 160)
TEXT = (255, 255, 255)

# One flight of the ball from one side of the net to the other.
SECONDS_PER_ARC = 1.5


def court_geometry(width, height):
    """Return the key positions of the scene, scaled to the frame size."""
    floor_y = int(height * 0.80)
    net_x = width // 2
    net_top_y = int(height * 0.45)
    ball_radius = max(4, round(height * 0.025))
    return floor_y, net_x, net_top_y, ball_radius


def ball_position(time_s, width, height):
    """Return the (x, y) pixel centre of the ball at `time_s` seconds.

    The ball flies in a parabola between two hitting points, one on each side
    of the net, then flies back. Sine easing on x makes it slow near the
    hitting points and fast over the net, so the motion looks smooth.
    """
    floor_y, net_x, net_top_y, _ = court_geometry(width, height)
    left_x = int(width * 0.15)
    right_x = int(width * 0.85)
    contact_y = int(height * 0.62)
    peak_y = int(height * 0.15)

    arc_number = int(time_s // SECONDS_PER_ARC)
    progress = (time_s % SECONDS_PER_ARC) / SECONDS_PER_ARC  # 0.0 -> 1.0

    eased = (1 - math.cos(math.pi * progress)) / 2
    start_x, end_x = (left_x, right_x) if arc_number % 2 == 0 else (right_x, left_x)
    x = start_x + (end_x - start_x) * eased

    # Parabola: contact height at both ends, highest point in the middle.
    y = contact_y - (contact_y - peak_y) * 4 * progress * (1 - progress)
    return round(x), round(y)


def draw_frame(frame_index, total_frames, width, height, fps):
    """Draw one frame of the synthetic scene and return it as an image."""
    floor_y, net_x, net_top_y, ball_radius = court_geometry(width, height)
    image = np.full((height, width, 3), SKY, dtype=np.uint8)

    # Court floor and its lines.
    cv2.rectangle(image, (0, floor_y), (width, height), FLOOR, thickness=-1)
    cv2.line(image, (0, floor_y), (width, floor_y), LINE, 2)
    for x in (int(width * 0.05), int(width * 0.95)):
        cv2.line(image, (x, floor_y - 4), (x, floor_y + 4), LINE, 2)

    # Net seen from the side: a thin post, a mesh band and a white top tape.
    net_half = max(6, width // 64)
    cv2.line(image, (net_x, net_top_y - 6), (net_x, floor_y), NET_POST, 3)
    net_bottom_y = net_top_y + (floor_y - net_top_y) // 2
    for y in range(net_top_y, net_bottom_y + 1, 6):
        cv2.line(image, (net_x - net_half, y), (net_x + net_half, y), NET_MESH, 1)
    for x in range(net_x - net_half, net_x + net_half + 1, 4):
        cv2.line(image, (x, net_top_y), (x, net_bottom_y), NET_MESH, 1)
    cv2.rectangle(
        image,
        (net_x - net_half - 1, net_top_y - 4),
        (net_x + net_half + 1, net_top_y),
        NET_TAPE,
        thickness=-1,
    )

    # Ball with a simple shadow on the floor.
    time_s = frame_index / fps
    ball_x, ball_y = ball_position(time_s, width, height)
    cv2.ellipse(
        image, (ball_x, floor_y + 3), (ball_radius, max(2, ball_radius // 3)),
        0, 0, 360, (60, 90, 140), thickness=-1, lineType=cv2.LINE_AA,
    )
    cv2.circle(image, (ball_x, ball_y), ball_radius, BALL, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image, (ball_x, ball_y), ball_radius, BALL_EDGE, thickness=1, lineType=cv2.LINE_AA)

    # Labels: what this is, and which frame we are on.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = height / 720
    cv2.rectangle(image, (0, 0), (width, int(height * 0.08)), BANNER, thickness=-1)
    cv2.putText(
        image, "SYNTHETIC TEST VIDEO - not real footage",
        (8, int(height * 0.06)), font, scale * 1.1, TEXT, 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, f"frame {frame_index:04d} / {total_frames - 1:04d}   t={time_s:5.2f}s",
        (8, height - 10), font, scale * 1.1, TEXT, 1, cv2.LINE_AA,
    )
    return image


def make_test_video(output_path, duration=4.0, width=640, height=360, fps=30):
    """Write the synthetic scene to `output_path` as an MP4.

    Returns the number of frames written. Raises ValueError with a readable
    message for bad settings or if the video file cannot be written.
    """
    output_path = Path(output_path)

    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output must end in .mp4, got: {output_path}")
    if duration <= 0:
        raise ValueError(f"Duration must be more than 0 seconds, got {duration}.")
    if fps <= 0:
        raise ValueError(f"FPS must be more than 0, got {fps}.")
    if width < 160 or height < 90:
        raise ValueError(f"Size must be at least 160 x 90, got {width} x {height}.")
    if width % 2 or height % 2:
        raise ValueError(f"Width and height must be even numbers, got {width} x {height}.")

    total_frames = max(1, round(duration * fps))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # "mp4v" is the MPEG-4 codec built into OpenCV's Windows FFmpeg.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    try:
        if not writer.isOpened():
            raise ValueError(
                f"Could not open a video writer for: {output_path}\n"
                "Check that the folder is writable and the file is not open elsewhere."
            )

        print(f"Writing synthetic test video: {output_path}")
        print(f"Size {width} x {height}, {fps} FPS, {total_frames} frames ({total_frames / fps:.2f} s)")
        for frame_index in range(total_frames):
            writer.write(draw_frame(frame_index, total_frames, width, height, fps))
    finally:
        writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError(f"The video file was not written: {output_path}")

    print(f"Done. Wrote {total_frames} frames.")
    return total_frames


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a synthetic volleyball test video (not real footage)."
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"where to save the MP4 (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="length of the video in seconds (default: 4)",
    )
    parser.add_argument("--width", type=int, default=640, help="frame width in pixels (default: 640)")
    parser.add_argument("--height", type=int, default=360, help="frame height in pixels (default: 360)")
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default: 30)")
    args = parser.parse_args(argv)

    try:
        make_test_video(args.output, args.duration, args.width, args.height, args.fps)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
