# Feature Specification: Inference Optimization Pipeline

**Feature Branch**: `001-inference-optimization`
**Created**: 2026-04-10
**Status**: Draft
**Input**: User description: "Build a full-stack object detection inference optimization application with YOLOv8m/YOLOv11m, FastAPI backend (OpenAI-compatible API), Next.js frontend, TensorRT + ONNX Runtime acceleration, custom-annotated dataset evaluation"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Single Image Detection (Priority: P1)

A user uploads an image through the frontend, selects a model (YOLOv8m or YOLOv11m) and a runtime (PyTorch, ONNX, or TensorRT), and receives back the image annotated with bounding boxes, class labels, confidence scores, and the inference latency.

**Why this priority**: This is the core detection flow — everything else builds on it. Without working single-image inference, nothing else functions.

**Independent Test**: Upload a test image, select YOLOv8m + PyTorch, verify bounding boxes appear with correct labels and latency is displayed.

**Acceptance Scenarios**:

1. **Given** the backend is running with models loaded, **When** a user uploads a JPEG image and selects YOLOv8m + TensorRT, **Then** the frontend displays the image with bounding box overlays, class labels, confidence scores, and inference latency in milliseconds.
2. **Given** an image with no detectable objects, **When** the user runs detection, **Then** the frontend displays the original image with zero detections and the inference latency.
3. **Given** the user selects a model+runtime combination, **When** the backend receives the request, **Then** it returns an OpenAI-compatible JSON response with `id`, `object`, `model`, `runtime`, `usage.inference_time_ms`, and `results` fields.

---

### User Story 2 - Video Detection (Priority: P2)

A user uploads a video file, selects model and runtime, and receives an annotated video with per-frame bounding boxes plus summary statistics (average latency, total detections per frame).

**Why this priority**: The assignment explicitly requires video inference. This extends the core image flow to sequential frames.

**Independent Test**: Upload a 10-second video clip, run detection with YOLOv11m + ONNX, verify annotated video is returned with per-frame stats.

**Acceptance Scenarios**:

1. **Given** a user uploads an MP4 video (30-60 seconds), **When** detection runs with YOLOv11m + ONNX, **Then** the frontend shows a progress bar with frame count during processing, and upon completion displays the annotated video with bounding boxes on each frame and a per-frame latency chart.
2. **Given** a video upload, **When** processing completes, **Then** summary stats show average latency per frame, total frames processed, and total detections.
3. **Given** a video upload exceeding 100MB, **When** the user submits, **Then** the backend returns an error and the frontend displays a file-too-large message.

---

### User Story 3 - Benchmark Comparison (Priority: P2)

A user runs the same image or video through all 6 model+runtime combinations and sees a side-by-side comparison table showing latency, detection count, and speedup relative to the PyTorch baseline.

**Why this priority**: The assignment requires evaluating both accuracy/mAP and speed/latency. This view makes comparison straightforward for the report.

**Independent Test**: Upload a test image, trigger "Compare All," verify a table with 6 rows (2 models x 3 runtimes) showing latency and detection count for each.

**Acceptance Scenarios**:

1. **Given** a user uploads an image and clicks "Compare All," **When** all 6 inference runs complete, **Then** a table displays model, runtime, latency (ms), detection count, and speedup factor for each combination.

---

### User Story 4 - mAP Evaluation on Custom Dataset (Priority: P1)

A standalone benchmark script evaluates all 6 model+runtime combinations against the custom-annotated dataset, computing COCO mAP@0.5 and mAP@0.5:0.95 alongside average inference latency, and outputs a results table.

**Why this priority**: This is a core assignment requirement — accuracy and speed evaluation using custom annotations. Required for the submission report.

**Independent Test**: Run the benchmark script on the annotated dataset, verify it produces a table with mAP and latency for all 6 combinations.

**Acceptance Scenarios**:

1. **Given** a dataset of minimum 50 annotated images in COCO JSON format, **When** the benchmark script runs, **Then** it outputs mAP@0.5, mAP@0.5:0.95, average latency (ms), and speedup factor for each of the 6 model+runtime combinations.
2. **Given** the benchmark script, **When** run twice on the same dataset, **Then** results are reproducible (identical mAP, latency within acceptable variance).

---

### Edge Cases

- What happens when a user uploads a non-image/non-video file (e.g., PDF)?
  The backend MUST return a clear error; the frontend MUST display it.
- What happens when a model+runtime combination is not yet exported?
  The `/v1/models` endpoint reports it as unavailable; the frontend disables the option.
- What happens with very large images or long videos?
  The backend MUST enforce upload limits: 10MB for images, 100MB for video. Oversized uploads MUST be rejected with a clear error.
- What happens when the GPU runs out of memory during inference?
  The backend MUST catch OOM errors and return a descriptive error response.
- What happens when switching between model+runtime combinations?
  The backend MUST unload the current model before loading the requested one (only one model+runtime loaded at a time to stay within T4 16GB VRAM).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST serve object detection inference via a FastAPI backend with OpenAI-compatible API structure (`/v1/detect`, `/v1/detect/upload`, `/v1/models`, `/health`).
- **FR-002**: System MUST support two detection models: YOLOv8m and YOLOv11m.
- **FR-003**: System MUST support three inference runtimes per model: native PyTorch (baseline), ONNX Runtime, and TensorRT.
- **FR-004**: API responses MUST follow OpenAI-compatible JSON structure with `id`, `object`, `model`, `created`, `usage`, and `results` fields.
- **FR-005**: System MUST process both single images and video files frame-by-frame. Video processing MUST show a progress bar with frame count; results displayed after full completion.
- **FR-011**: Backend MUST enforce upload size limits: 10MB for images, 100MB for video.
- **FR-012**: Backend MUST load only one model+runtime combination at a time, unloading the previous before loading the next (single-model memory strategy for T4 16GB VRAM).
- **FR-006**: Frontend MUST allow users to upload images/videos, select model and runtime, and visualize detection results with bounding boxes, class labels, confidence scores, and latency.
- **FR-007**: Frontend MUST provide a comparison view showing all 6 model+runtime combinations side-by-side.
- **FR-008**: A standalone benchmark script MUST compute COCO mAP@0.5 and mAP@0.5:0.95 for each model+runtime combo using `pycocotools`.
- **FR-009**: Benchmark script MUST measure and report average inference latency with warm-up runs excluded.
- **FR-010**: System MUST use a custom-annotated dataset (minimum 50 images + 1 video clip) with COCO JSON ground truth annotations created by the student.

### Key Entities

- **DetectionRequest**: Model selection, runtime selection, input media (image or video), input type.
- **DetectionResult**: Request ID, model used, runtime used, timestamp, inference latency, list of per-frame detections.
- **Detection**: Bounding box coordinates (x1, y1, x2, y2), class label, confidence score.
- **BenchmarkResult**: Model, runtime, mAP@0.5, mAP@0.5:0.95, average latency, speedup factor.
- **ModelInfo**: Model name, available runtimes, loaded status.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All 6 model+runtime combinations produce valid detection results on the custom dataset without errors.
- **SC-002**: TensorRT inference is at least 1.5x faster than native PyTorch baseline for both models.
- **SC-003**: ONNX Runtime inference is at least 1.2x faster than native PyTorch baseline for both models.
- **SC-004**: mAP difference between PyTorch baseline and accelerated runtimes (ONNX, TensorRT) is less than 2 percentage points (demonstrating accuracy preservation).
- **SC-005**: Users can upload an image and see detection results (with bounding boxes and latency) within 5 seconds of clicking "Detect" (excluding model cold-start).
- **SC-006**: Benchmark script produces a complete results table for all 6 combinations in a single run.
- **SC-007**: All benchmark results are reproducible — running twice yields identical mAP values and latency within 10% variance.

## Clarifications

### Session 2026-04-10

- Q: How should the frontend handle video processing UX? → A: Wait for full processing with progress bar showing frame count (no streaming/websocket).
- Q: What maximum upload file sizes should the backend enforce? → A: 10MB for images, 100MB for video.
- Q: How should the backend manage GPU memory with 6 model+runtime combos on T4 (16GB)? → A: Load one model+runtime at a time; unload current before loading next.

## Assumptions

- Pre-trained YOLOv8m and YOLOv11m weights from Ultralytics are permitted (Option 2 focuses on inference, not training).
- Google Cloud VM with NVIDIA T4 GPU is available for backend serving, model export, and benchmarking.
- Frontend runs locally on the developer's machine, connecting to the GCP VM backend via external IP or SSH tunnel.
- The custom dataset uses objects from COCO classes (cars, people, chairs, etc.) so pre-trained model weights are applicable.
- Annotations are created using Label Studio or CVAT in COCO JSON format.
- TensorRT is available on the GCP VM via NVIDIA's container or pip package.
- Video input is limited to common formats (MP4, AVI) and reasonable duration (under 5 minutes).
