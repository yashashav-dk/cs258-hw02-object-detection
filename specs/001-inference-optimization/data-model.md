# Data Model: Inference Optimization Pipeline

**Date**: 2026-04-10

## Entities

### DetectionRequest

Represents an incoming inference request from the frontend.

| Field | Type | Description |
|-------|------|-------------|
| model | string enum | `"yolov8m"` or `"yolov11m"` |
| runtime | string enum | `"pytorch"`, `"onnx"`, or `"tensorrt"` |
| input | file or base64 | Image or video media |
| input_type | string enum | `"image"` or `"video"` |

**Validation**:
- model MUST be one of the two supported values
- runtime MUST be one of the three supported values
- input file size: max 10MB for images, 100MB for video
- Accepted formats: JPEG, PNG for images; MP4, AVI for video

### DetectionResponse

OpenAI-compatible response returned by the backend.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique request ID, format `"det-{uuid8}"` |
| object | string | Always `"detection.result"` |
| model | string | Model used (e.g., `"yolov8m"`) |
| runtime | string | Runtime used (e.g., `"tensorrt"`) |
| created | integer | Unix timestamp |
| usage.inference_time_ms | float | Total inference latency in ms |
| results | array | List of FrameResult objects |

### FrameResult

Per-frame detection output (single element for images, N for video).

| Field | Type | Description |
|-------|------|-------------|
| frame | integer | Frame index (0 for single image) |
| detections | array | List of Detection objects |

### Detection

A single detected object.

| Field | Type | Description |
|-------|------|-------------|
| bbox | array[4] | `[x1, y1, x2, y2]` pixel coordinates |
| class | string | COCO class label (e.g., `"person"`) |
| confidence | float | Detection confidence score (0.0–1.0) |

### BenchmarkResult

Output row from the benchmark evaluation script.

| Field | Type | Description |
|-------|------|-------------|
| model | string | Model name |
| runtime | string | Runtime name |
| map_50 | float | COCO mAP@0.5 |
| map_50_95 | float | COCO mAP@0.5:0.95 |
| avg_latency_ms | float | Average inference latency (ms) |
| speedup | float | Speedup factor vs PyTorch baseline |

### ModelInfo

Model availability information returned by `/v1/models`.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Model identifier (e.g., `"yolov8m"`) |
| runtimes | array | Available runtimes with loaded status |

### COCO Annotation (external format)

Ground truth annotations in standard COCO JSON format.

| Field | Type | Description |
|-------|------|-------------|
| image_id | integer | Unique image identifier |
| category_id | integer | COCO category ID |
| bbox | array[4] | `[x, y, width, height]` (COCO format) |
| area | float | Bounding box area |
| iscrowd | integer | 0 for individual instances |

## Relationships

- A DetectionRequest produces one DetectionResponse
- A DetectionResponse contains 1+ FrameResults (1 for image, N for video)
- Each FrameResult contains 0+ Detections
- BenchmarkResult is computed by comparing Detections against COCO Annotations
- ModelInfo tracks which model+runtime combos are available

## State Transitions

### Model Lifecycle (single-model strategy)

```
[Idle] → load_model(model, runtime) → [Loading] → ready → [Loaded]
[Loaded] → new request same model → [Loaded] (reuse)
[Loaded] → new request different model → unload → [Idle] → load_model → [Loading] → [Loaded]
```

Only one model+runtime combination is in [Loaded] state at any time.
