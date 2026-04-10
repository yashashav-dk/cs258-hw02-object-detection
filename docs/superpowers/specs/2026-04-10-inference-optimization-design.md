# HW02 Object Detection — Inference Optimization Design

**Date**: 2026-04-10
**Due**: 2026-04-17
**Option**: Option 2 — Inference Optimization
**Approach**: Ultralytics YOLO + TensorRT/ONNX

## Summary

Build a full-stack object detection inference application with two
models (YOLOv8, YOLOv11) served via FastAPI, displayed via a Next.js
frontend, and accelerated with TensorRT and ONNX Runtime. Evaluate
accuracy (COCO mAP) and speed (latency) across all model+runtime
combinations using a custom-annotated dataset.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Next.js Frontend                │
│  - Upload image/video                           │
│  - Select model + runtime                       │
│  - Display results (bboxes, latency, mAP)       │
└──────────────────────┬──────────────────────────┘
                       │ HTTP (multipart upload)
┌──────────────────────▼──────────────────────────┐
│               FastAPI Backend                    │
│  POST /detect/image   — single image inference  │
│  POST /detect/video   — video inference          │
│  GET  /models         — list available models    │
│  GET  /health         — health check             │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐               │
│  │ YOLOv8m     │  │ YOLOv11m    │               │
│  │ ├─ PyTorch  │  │ ├─ PyTorch  │               │
│  │ ├─ ONNX     │  │ ├─ ONNX     │               │
│  │ └─ TensorRT │  │ └─ TensorRT │               │
│  └─────────────┘  └─────────────┘               │
└─────────────────────────────────────────────────┘
```

Each model available in 3 runtimes: native PyTorch (baseline), ONNX
Runtime, and TensorRT. The client selects model + runtime per request.

## Backend (FastAPI — OpenAI-Compatible API)

The API follows an OpenAI-compatible request/response structure to
satisfy the assignment requirement of "similar to openai-compatible API."

### Endpoints

- `POST /v1/detect` — primary detection endpoint. Accepts JSON body:
  ```json
  {
    "model": "yolov8m" | "yolov11m",
    "runtime": "pytorch" | "onnx" | "tensorrt",
    "input": "<base64-encoded image or video>",
    "input_type": "image" | "video"
  }
  ```
  Returns structured JSON response:
  ```json
  {
    "id": "det-xxxxxxxx",
    "object": "detection.result",
    "model": "yolov8m",
    "runtime": "tensorrt",
    "created": 1712700000,
    "usage": {
      "inference_time_ms": 12.3
    },
    "results": [
      {
        "frame": 0,
        "detections": [
          {
            "bbox": [x1, y1, x2, y2],
            "class": "person",
            "confidence": 0.92
          }
        ]
      }
    ]
  }
  ```

- `POST /v1/detect/upload` — multipart file upload variant for the
  frontend. Same response format as above.

- `GET /v1/models` — returns available model/runtime combinations and
  their loaded status (mirrors OpenAI's `/v1/models` endpoint).

- `GET /health` — health check.

### Model Management

- Only one model+runtime loaded at a time (T4 has 16GB VRAM;
  loading all 6 concurrently risks OOM). Previous model unloaded
  before loading the next.
- Upload limits: 10MB images, 100MB video.
- Ultralytics API handles loading:
  - `YOLO("yolov8m.pt")` for PyTorch
  - `YOLO("yolov8m.onnx")` for ONNX
  - `YOLO("yolov8m.engine")` for TensorRT
- Export done on GCP VM via setup script: `model.export(format="onnx")`
  and `model.export(format="engine")`.

## Frontend (Next.js)

### Detection View (main page)

- File upload (drag-and-drop or click) for image or video.
- Model selector dropdown: YOLOv8 / YOLOv11.
- Runtime selector dropdown: PyTorch / ONNX / TensorRT.
- "Detect" button triggers inference.
- Results: image with bounding box overlays (colored by class, labels +
  confidence), latency badge, detection count.

### Video Results View

- Video player with bounding box overlays per frame.
- Per-frame latency bar chart.
- Summary stats: avg latency, total frames, total detections.

### Comparison / Benchmark View

- Run same input through all 6 model+runtime combos.
- Table showing latency, detection count, mAP, speedup factor relative
  to PyTorch baseline.

### Styling

Tailwind CSS. Minimal, clean, functional. No auth, no database, no
deployment — runs locally on `localhost:3000` calling `localhost:8000`.

## Evaluation

### Custom Dataset

- Minimum 50 images and one video clip (30-60 seconds) with common
  COCO-class objects (cars, people, chairs, etc.).
- Ground truth bounding boxes annotated in COCO JSON format using
  Label Studio or CVAT.
- This dataset used for both mAP evaluation and demo.

### Benchmark Script (standalone Python)

- Runs each model+runtime combo against the annotated dataset.
- Warm-up runs excluded, latency averaged over N iterations.
- Computes COCO mAP via `pycocotools`.
- Outputs results table:

```
Model     | Runtime   | mAP@0.5 | mAP@0.5:0.95 | Avg Latency (ms) | Speedup
----------|-----------|---------|---------------|-------------------|--------
YOLOv8m   | PyTorch   |         |               |                   | 1.0x
YOLOv8m   | ONNX      |         |               |                   |
YOLOv8m   | TensorRT  |         |               |                   |
YOLOv11m  | PyTorch   |         |               |                   | 1.0x
YOLOv11m  | ONNX      |         |               |                   |
YOLOv11m  | TensorRT  |         |               |                   |
```

### Acceleration Methods

1. **ONNX Runtime**: Converts PyTorch model to ONNX graph format.
   Enables graph optimizations (operator fusion, constant folding) and
   hardware-specific execution providers (CUDA EP for GPU). Expected
   speedup: 1.5-2x over native PyTorch.

2. **TensorRT**: NVIDIA's inference optimizer. Performs layer fusion,
   kernel auto-tuning, precision calibration (FP16), and memory
   optimization for the specific GPU. Expected speedup: 2-4x over
   native PyTorch.

## Infrastructure

- **Local machine**: Next.js frontend development, backend code editing.
- **Google Cloud VM (T4 GPU)**: Primary compute — runs FastAPI backend,
  TensorRT/ONNX model export, benchmark script, inference serving.
  T4 has native TensorRT support and FP16 acceleration, eliminating
  any compatibility risk.
- The backend runs on the GCP VM; the frontend connects to it via
  the VM's external IP or SSH tunnel.

## Deliverables

1. Source code: backend (FastAPI) + frontend (Next.js) + benchmark script
2. README with setup instructions and reproduction steps
3. Benchmark results table (mAP + latency for all 6 combos)
4. Description of acceleration methods and their impact
5. Custom annotated dataset (images + COCO JSON annotations)

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, Ultralytics, onnxruntime-gpu,
  tensorrt, pycocotools, opencv-python
- **Frontend**: Next.js 14+, React, Tailwind CSS
- **Annotation**: Label Studio or CVAT
- **Compute**: Google Cloud VM with NVIDIA T4 GPU
