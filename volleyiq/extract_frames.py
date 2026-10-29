"""Extract every Nth frame from a video and save it as a JPEG.

Frames are named after their original frame number in the video, e.g.
frame_000120.jpg is frame 120, so you can always find the exact moment again.

Usage:
    python -m volleyiq.extract_frames VIDEO OUTPUT_DIR [--every N]
"""

import argparse
import sys
from pathlib import Path

import cv2


def describe_video(capture):
    """Return (width, height, fps, total_frames) as reported by the video file."""
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, fps, total_frames


def extract_frames(video_path, output_dir, every=30):
    """Save every `every`-th frame of `video_path` into `output_dir`.

    Returns the number of frames saved. Raises ValueError with a readable
    message if the video cannot be used.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if every < 1:
        raise ValueError(f"Frame interval must be 1 or more, got {every}.")
    if not video_path.exists():
        raise ValueError(f"Video not found: {video_path}")
    if not video_path.is_file():
        raise ValueError(f"Not a file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(
                f"Could not open video: {video_path}\n"
                "The file may be corrupted or in a format OpenCV cannot read."
            )

        width, height, fps, total_frames = describe_video(capture)
        print(f"Video:        {video_path}")
        print(f"Dimensions:   {width} x {height} pixels")
        if fps > 0:
            print(f"FPS:          {fps:.2f}")
        else:
            print("FPS:          unknown")
        print(f"Total frames: {total_frames} (as reported by the file)")
        if fps > 0 and total_frames > 0:
            print(f"Duration:     {total_frames / fps:.2f} seconds")
        else:
            print("Duration:     unknown")
        print(f"Saving every {every} frame(s) to: {output_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Read frames in order. grab() skips cheaply; retrieve() decodes only
        # the frames we keep. This keeps frame numbers exact, unlike seeking.
        frame_index = 0
        saved = 0
        while capture.grab():
            if frame_index % every == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f"Could not decode frame {frame_index}.")
                out_path = output_dir / f"frame_{frame_index:06d}.jpg"
                if not cv2.imwrite(str(out_path), frame):
                    raise ValueError(f"Could not write image: {out_path}")
                saved += 1
            frame_index += 1

        if frame_index == 0:
            raise ValueError(
                f"No frames could be read from: {video_path}\n"
                "The file may be empty, corrupted, or in an unsupported format."
            )

        print(f"Frames read:  {frame_index}")
        print(f"Frames saved: {saved}")
        return saved
    finally:
        capture.release()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract every Nth frame from a video as JPEG images."
    )
    parser.add_argument("video", help="path to the input video, e.g. data/videos/rally1.mp4")
    parser.add_argument("output_dir", help="folder to save frames in, e.g. data/frames/rally1")
    parser.add_argument(
        "--every",
        type=int,
        default=30,
        help="save one frame every N frames (default: 30)",
    )
    args = parser.parse_args(argv)

    try:
        extract_frames(args.video, args.output_dir, args.every)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
