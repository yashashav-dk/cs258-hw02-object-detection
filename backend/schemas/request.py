"""Request schemas for the detection API."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel

# Upload size limits (bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB


class ModelName(str, Enum):
    YOLOV8M = "yolov8m"
    YOLOV11M = "yolov11m"


class RuntimeName(str, Enum):
    PYTORCH = "pytorch"
    ONNX = "onnx"
    TENSORRT = "tensorrt"


class InputType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DetectionRequest(BaseModel):
    """OpenAI-compatible detection request (base64 input)."""
    model: ModelName
    runtime: RuntimeName
    input: str  # base64-encoded image or video
    input_type: InputType


ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/x-msvideo"}
