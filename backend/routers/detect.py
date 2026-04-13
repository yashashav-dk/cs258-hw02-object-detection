"""Detection endpoints — /v1/detect, /v1/detect/upload, /v1/detect/compare."""

import base64
import io
import tempfile

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from backend.schemas.request import (
    ALLOWED_IMAGE_TYPES,
    ALLOWED_VIDEO_TYPES,
    MAX_IMAGE_SIZE,
    MAX_VIDEO_SIZE,
    DetectionRequest,
    InputType,
    ModelName,
    RuntimeName,
)
from backend.schemas.response import DetectionResponse, FrameResult, UsageInfo
from backend.services.detector import detect_image

router = APIRouter(prefix="/v1")

# Store rendered video paths for serving
_video_results: dict[str, str] = {}


def _decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@router.post("/detect/upload", response_model=DetectionResponse)
async def detect_upload(
    request: Request,
    file: UploadFile = File(...),
    model: ModelName = Form(...),
    runtime: RuntimeName = Form(...),
) -> DetectionResponse:
    """Multipart file upload detection endpoint."""
    content_type = file.content_type or ""
    data = await file.read()

    # Determine input type and validate
    if content_type in ALLOWED_IMAGE_TYPES:
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)}MB limit")
        return await _detect_image_bytes(request, data, model, runtime)
    elif content_type in ALLOWED_VIDEO_TYPES:
        if len(data) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=413, detail=f"Video exceeds {MAX_VIDEO_SIZE // (1024*1024)}MB limit")
        return await _detect_video_bytes(request, data, model, runtime)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}. Use JPEG/PNG for images or MP4/AVI for video.")


@router.post("/detect", response_model=DetectionResponse)
async def detect_base64(
    request: Request,
    body: DetectionRequest,
) -> DetectionResponse:
    """Base64 input detection endpoint (OpenAI-compatible)."""
    try:
        data = base64.b64decode(body.input)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 input")

    if body.input_type == InputType.IMAGE:
        if len(data) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)}MB limit")
        return await _detect_image_bytes(request, data, body.model, body.runtime)
    else:
        if len(data) > MAX_VIDEO_SIZE:
            raise HTTPException(status_code=413, detail=f"Video exceeds {MAX_VIDEO_SIZE // (1024*1024)}MB limit")
        return await _detect_video_bytes(request, data, body.model, body.runtime)


async def _detect_image_bytes(
    request: Request,
    data: bytes,
    model_name: ModelName,
    runtime: RuntimeName,
) -> DetectionResponse:
    """Run image detection on raw bytes."""
    manager = request.app.state.model_manager
    model = manager.load(model_name, runtime)
    img = _decode_image(data)
    frame_result, latency_ms = detect_image(model, img)

    return DetectionResponse(
        model=model_name.value,
        runtime=runtime.value,
        usage=UsageInfo(inference_time_ms=round(latency_ms, 2)),
        results=[frame_result],
    )


async def _detect_video_bytes(
    request: Request,
    data: bytes,
    model_name: ModelName,
    runtime: RuntimeName,
) -> DetectionResponse:
    """Run video detection on raw bytes."""
    # Lazy import to avoid circular dependency before video.py exists
    from backend.services.video import process_video

    manager = request.app.state.model_manager
    model = manager.load(model_name, runtime)

    result_id, frame_results, total_latency_ms, avg_latency_ms, total_frames = process_video(
        model, data, _video_results
    )

    return DetectionResponse(
        model=model_name.value,
        runtime=runtime.value,
        usage=UsageInfo(
            inference_time_ms=round(total_latency_ms, 2),
            avg_latency_per_frame_ms=round(avg_latency_ms, 2),
            total_frames=total_frames,
        ),
        results=frame_results,
        annotated_video_url=f"/v1/results/{result_id}/video",
    )


@router.get("/results/{result_id}/video")
async def get_video_result(result_id: str):
    """Serve pre-rendered annotated video."""
    if result_id not in _video_results:
        raise HTTPException(status_code=404, detail="Video result not found")
    return FileResponse(_video_results[result_id], media_type="video/mp4")


@router.post("/detect/compare", response_model=list[DetectionResponse])
async def detect_compare(
    request: Request,
    file: UploadFile = File(...),
) -> list[DetectionResponse]:
    """Run detection with all 6 model+runtime combos on the same image."""
    content_type = file.content_type or ""
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Compare only supports images")

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_IMAGE_SIZE // (1024*1024)}MB limit")

    img = _decode_image(data)
    manager = request.app.state.model_manager
    responses = []

    for model_name in ModelName:
        for runtime in RuntimeName:
            if not manager.is_available(model_name, runtime):
                continue
            model = manager.load(model_name, runtime)
            frame_result, latency_ms = detect_image(model, img)
            responses.append(DetectionResponse(
                model=model_name.value,
                runtime=runtime.value,
                usage=UsageInfo(inference_time_ms=round(latency_ms, 2)),
                results=[frame_result],
            ))

    return responses
