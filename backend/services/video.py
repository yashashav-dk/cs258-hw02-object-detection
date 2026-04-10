"""Video processing service: frame-by-frame detection with bbox rendering."""

import tempfile
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from backend.schemas.response import Detection, FrameResult

# Persistent directory for rendered video results
VIDEO_RESULTS_DIR = Path(tempfile.gettempdir()) / "detection_videos"
VIDEO_RESULTS_DIR.mkdir(exist_ok=True)

# Color palette for class labels (BGR for OpenCV)
PALETTE = [
    (68, 68, 239), (246, 130, 59), (37, 197, 34), (7, 179, 234),
    (168, 85, 168), (153, 102, 255), (180, 119, 31), (247, 127, 0),
]


def _get_color(cls_id: int) -> tuple[int, int, int]:
    return PALETTE[cls_id % len(PALETTE)]


def process_video(
    model: YOLO,
    video_bytes: bytes,
    video_store: dict[str, str],
) -> tuple[str, list[FrameResult], float, float, int]:
    """Process video frame-by-frame, return frame results + rendered video path.

    Returns:
        (result_id, frame_results, total_latency_ms, avg_latency_ms, total_frames)
    """
    result_id = uuid.uuid4().hex[:12]

    # Save input video to temp file
    input_path = VIDEO_RESULTS_DIR / f"{result_id}_input.mp4"
    with open(input_path, "wb") as f:
        f.write(video_bytes)

    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        raise ValueError("Could not open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Prepare output video (MP4 with H.264-compatible codec)
    output_path = VIDEO_RESULTS_DIR / f"{result_id}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_results: list[FrameResult] = []
    latencies: list[float] = []
    frame_idx = 0
    total_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inf_start = time.perf_counter()
        results = model(frame, verbose=False)
        latency_ms = (time.perf_counter() - inf_start) * 1000
        latencies.append(latency_ms)

        detections_list: list[Detection] = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            names = results[0].names
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = names[cls_id]

                detections_list.append(Detection(**{
                    "bbox": xyxy,
                    "class": cls_name,
                    "confidence": round(conf, 4),
                }))

                # Draw bbox on frame
                x1, y1, x2, y2 = map(int, xyxy)
                color = _get_color(cls_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    frame, label, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

        frame_results.append(FrameResult(frame=frame_idx, detections=detections_list))
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    input_path.unlink(missing_ok=True)

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0.0

    video_store[result_id] = str(output_path)

    return result_id, frame_results, total_latency_ms, avg_latency_ms, frame_idx
