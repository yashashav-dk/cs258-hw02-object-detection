"""Response schemas for the detection API (OpenAI-compatible)."""

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field


class Detection(BaseModel):
    """A single detected object."""
    bbox: list[float] = Field(description="[x1, y1, x2, y2] pixel coordinates")
    class_name: str = Field(alias="class", serialization_alias="class")
    confidence: float


class FrameResult(BaseModel):
    """Per-frame detection output."""
    frame: int
    detections: list[Detection]


class UsageInfo(BaseModel):
    """Inference timing information."""
    inference_time_ms: float
    avg_latency_per_frame_ms: Optional[float] = None
    total_frames: Optional[int] = None


class DetectionResponse(BaseModel):
    """OpenAI-compatible detection response."""
    id: str = Field(default_factory=lambda: f"det-{uuid.uuid4().hex[:8]}")
    object: str = "detection.result"
    model: str
    runtime: str
    created: int = Field(default_factory=lambda: int(time.time()))
    usage: UsageInfo
    results: list[FrameResult]
    annotated_video_url: Optional[str] = None

    model_config = {"populate_by_name": True}


class ModelRuntimeInfo(BaseModel):
    """Runtime availability for a model."""
    runtime: str
    available: bool
    loaded: bool


class ModelInfo(BaseModel):
    """Model information for /v1/models."""
    id: str
    object: str = "model"
    runtimes: list[ModelRuntimeInfo]


class ModelsListResponse(BaseModel):
    """Response for GET /v1/models."""
    object: str = "list"
    data: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str = "ok"
    gpu: Optional[str] = None
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
