"""Capture frames or a video clip from a public traffic camera stream.

Supports HLS (.m3u8) and RTSP URLs. Uses ffmpeg under the hood.

Usage:
    # Extract 60 frames spaced 10 seconds apart (= 10 minutes of wall time)
    python scripts/capture_traffic_cam.py frames \
        --url "https://wzmedia.dot.ca.gov/D4/XXX.stream/playlist.m3u8" \
        --output-dir ~/hw02-dataset/raw-frames \
        --num-frames 60 \
        --interval 10

    # Record a 60-second clip for the frontend demo
    python scripts/capture_traffic_cam.py video \
        --url "https://wzmedia.dot.ca.gov/D4/XXX.stream/playlist.m3u8" \
        --output ~/hw02-dataset/video/clip.mp4 \
        --duration 60
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found.", file=sys.stderr)
        print("  macOS:  brew install ffmpeg", file=sys.stderr)
        print("  Ubuntu: sudo apt install ffmpeg", file=sys.stderr)
        sys.exit(1)


def capture_frames(url: str, output_dir: Path, num_frames: int, interval: float, prefix: str) -> None:
    """Extract num_frames frames from a stream, one every `interval` seconds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ffmpeg fps filter: emit 1 frame every `interval` seconds
    fps = f"1/{interval}"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        "-i", url,
        "-vf", f"fps={fps}",
        "-frames:v", str(num_frames),
        "-q:v", "2",  # high JPEG quality
        "-y",
        str(output_dir / f"{prefix}_%04d.jpg"),
    ]

    est_minutes = int((num_frames * interval) / 60)
    print(f"Capturing {num_frames} frames at 1 per {interval}s from:")
    print(f"  {url}")
    print(f"Expected wall time: ~{est_minutes} minute(s).\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: ffmpeg exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    captured = sorted(output_dir.glob(f"{prefix}_*.jpg"))
    print(f"\nCaptured {len(captured)} frames to {output_dir}")


def capture_video(url: str, output: Path, duration: int) -> None:
    """Record a continuous video clip of `duration` seconds from a stream."""
    output.parent.mkdir(parents=True, exist_ok=True)

    # Try stream copy first (fast, no re-encode)
    cmd_copy = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        "-i", url,
        "-t", str(duration),
        "-c", "copy",
        "-y",
        str(output),
    ]

    print(f"Recording {duration}s clip from:")
    print(f"  {url}\n")

    result = subprocess.run(cmd_copy)
    if result.returncode == 0:
        print(f"\nRecorded {duration}s clip to {output}")
        return

    # Fallback: re-encode with libx264 (some HLS sources have bad stream params)
    print("\nStream copy failed, retrying with libx264 re-encode...\n")
    cmd_reencode = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-stats",
        "-i", url,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-an",
        "-y",
        str(output),
    ]
    result = subprocess.run(cmd_reencode)
    if result.returncode != 0:
        print(f"\nERROR: ffmpeg exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"\nRecorded {duration}s clip to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture frames or a video clip from a public camera stream")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_frames = sub.add_parser("frames", help="Extract diverse frames for annotation")
    p_frames.add_argument("--url", required=True, help="HLS (.m3u8) or RTSP stream URL")
    p_frames.add_argument("--output-dir", type=Path, required=True)
    p_frames.add_argument("--num-frames", type=int, default=60)
    p_frames.add_argument("--interval", type=float, default=10.0, help="Seconds between frames")
    p_frames.add_argument("--prefix", default="frame")

    p_video = sub.add_parser("video", help="Record a short video clip for the demo")
    p_video.add_argument("--url", required=True, help="HLS (.m3u8) or RTSP stream URL")
    p_video.add_argument("--output", type=Path, required=True)
    p_video.add_argument("--duration", type=int, default=60)

    args = parser.parse_args()
    check_ffmpeg()

    if args.mode == "frames":
        capture_frames(args.url, args.output_dir, args.num_frames, args.interval, args.prefix)
    else:
        capture_video(args.url, args.output, args.duration)


if __name__ == "__main__":
    main()
