# Research: Inference Optimization Pipeline

**Date**: 2026-04-10

## R1: TensorRT Export on T4 GPU

**Decision**: Use Ultralytics `model.export(format="engine")` with
TensorRT pre-installed on the GCP VM.

**Rationale**: Ultralytics wraps TensorRT export natively but requires
TensorRT to be installed separately (not bundled). On T4, TensorRT 8.6+
with CUDA 11.8 is the recommended configuration. The `tensorrt` pip
package or full toolkit must be installed before calling export.

**Alternatives considered**:
- Manual ONNX → TensorRT conversion via `trtexec` CLI: more control
  but unnecessary complexity for this use case.
- NVIDIA NGC Docker container with TensorRT pre-installed: viable but
  adds Docker dependency. May use if pip install is problematic.

## R2: ONNX Runtime GPU Setup

**Decision**: Use `pip install onnxruntime-gpu` with CUDA execution
provider. Load exported ONNX models via
`onnxruntime.InferenceSession(path, providers=["CUDAExecutionProvider"])`.

**Rationale**: `onnxruntime-gpu` provides CUDA EP out of the box on T4
with no extra configuration beyond CUDA/cuDNN. Ultralytics'
`model.export(format="onnx")` produces fully compatible ONNX files.

**Alternatives considered**:
- `onnxruntime` (CPU only): no GPU acceleration, defeats purpose.
- ONNX Runtime with TensorRT EP: possible but overlaps with our
  dedicated TensorRT acceleration method.

## R3: FastAPI Upload Limits

**Decision**: Per-endpoint file size validation with `HTTPException(413)`
for oversized files. Defense-in-depth via Uvicorn `--limit-request-body`.

**Rationale**: Per-endpoint validation gives clearest error messages and
most control (10MB images, 100MB video). Server-level limit catches
anything that slips through.

**Alternatives considered**:
- Starlette `ContentSizeLimitMiddleware`: works but less granular
  (single limit for all endpoints).
- Client-side only validation: insufficient; server must enforce.

## R4: Video Bounding Box Display

**Decision**: Backend renders bounding boxes directly into video frames
using OpenCV, returns pre-annotated MP4. Frontend plays via standard
`<video>` element.

**Rationale**: Simpler and more reliable than canvas overlay. No
frame-synced JSON coordinates needed. OpenCV `cv2.rectangle` +
`cv2.putText` on each frame, then re-encode with `cv2.VideoWriter`.

**Alternatives considered**:
- Canvas overlay on HTML5 video with per-frame JSON: significantly
  more complex, frame sync issues, minimal benefit for homework.
- Return individual annotated frames as image sequence: large payload,
  poor UX.

## R5: COCO mAP Evaluation

**Decision**: Convert Ultralytics predictions to COCO results format
manually, then evaluate with `pycocotools.cocoeval.COCOeval`.

**Rationale**: Ultralytics predictions use `xyxy` format per-image;
pycocotools expects a JSON list of `{"image_id", "category_id",
"bbox" (xywh), "score"}` dicts. Conversion is straightforward. Using
pycocotools directly (rather than Ultralytics' built-in `model.val()`)
ensures standard COCO evaluation and satisfies the assignment
requirement for COCO-style metrics.

**Alternatives considered**:
- Ultralytics `model.val()` built-in mAP: uses internal calculation,
  not pycocotools. May not satisfy "COCO-style evaluation" requirement.
- `torchmetrics.detection.MeanAveragePrecision`: adds another
  dependency; pycocotools is the gold standard.
