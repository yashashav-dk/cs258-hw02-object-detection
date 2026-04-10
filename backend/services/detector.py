"""Detection service: runs inference on a single image."""

import time

import numpy as np
from ultralytics import YOLO

from backend.schemas.response import Detection, FrameResult


def detect_image(model: YOLO, image: np.ndarray) -> tuple[FrameResult, float]:
    """Run detection on a single image.

    Returns:
        Tuple of (FrameResult, latency_ms)
    """
    start = time.perf_counter()
    results = model(image, verbose=False)
    latency_ms = (time.perf_counter() - start) * 1000

    detections = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            cls_name = results[0].names[cls_id]
            detections.append(Detection(**{
                "bbox": xyxy,
                "class": cls_name,
                "confidence": round(conf, 4),
            }))

    frame_result = FrameResult(frame=0, detections=detections)
    return frame_result, latency_ms
