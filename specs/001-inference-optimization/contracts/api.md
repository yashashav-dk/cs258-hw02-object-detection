# API Contract: Object Detection Service

**Base URL**: `http://<gcp-vm-ip>:8000`

## POST /v1/detect

Primary detection endpoint (base64 input).

**Request**:
```json
{
  "model": "yolov8m",
  "runtime": "tensorrt",
  "input": "<base64-encoded image or video>",
  "input_type": "image"
}
```

**Response** (200):
```json
{
  "id": "det-a1b2c3d4",
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
          "bbox": [100, 200, 300, 400],
          "class": "person",
          "confidence": 0.92
        }
      ]
    }
  ]
}
```

**Errors**:
- 400: Invalid model/runtime, unsupported file format
- 413: File exceeds size limit (10MB image / 100MB video)
- 503: Model not available (not yet exported)

## POST /v1/detect/upload

Multipart file upload variant for frontend.

**Request**: `multipart/form-data`
- `file`: image or video file
- `model`: `"yolov8m"` or `"yolov11m"`
- `runtime`: `"pytorch"`, `"onnx"`, or `"tensorrt"`

**Response**: Same as `/v1/detect`

For video input, response includes multiple frame results and
additional usage fields:
```json
{
  "usage": {
    "inference_time_ms": 1523.7,
    "avg_latency_per_frame_ms": 15.2,
    "total_frames": 100
  }
}
```

Video response also includes `annotated_video_url` pointing to
the rendered video with bounding boxes:
```json
{
  "annotated_video_url": "/v1/results/<result-id>/video"
}
```

## GET /v1/models

List available models and runtimes.

**Response** (200):
```json
{
  "object": "list",
  "data": [
    {
      "id": "yolov8m",
      "object": "model",
      "runtimes": [
        {"runtime": "pytorch", "available": true, "loaded": true},
        {"runtime": "onnx", "available": true, "loaded": false},
        {"runtime": "tensorrt", "available": true, "loaded": false}
      ]
    },
    {
      "id": "yolov11m",
      "object": "model",
      "runtimes": [
        {"runtime": "pytorch", "available": true, "loaded": false},
        {"runtime": "onnx", "available": true, "loaded": false},
        {"runtime": "tensorrt", "available": false, "loaded": false}
      ]
    }
  ]
}
```

## GET /health

Health check.

**Response** (200):
```json
{
  "status": "ok",
  "gpu": "NVIDIA Tesla T4",
  "gpu_memory_used_mb": 2048,
  "gpu_memory_total_mb": 15360
}
```

## GET /v1/results/{result_id}/video

Serve pre-rendered annotated video.

**Response**: `video/mp4` binary stream
