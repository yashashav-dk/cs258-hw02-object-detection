"""Model manager: loads/unloads a single model+runtime at a time."""

from pathlib import Path
from typing import Optional

from ultralytics import YOLO

from backend.schemas.request import ModelName, RuntimeName

# Map model names to weight file stems
MODEL_WEIGHTS = {
    ModelName.YOLOV8M: "yolov8m",
    ModelName.YOLOV11M: "yolo11m",
}

# Map runtimes to file extensions
RUNTIME_EXTENSIONS = {
    RuntimeName.PYTORCH: ".pt",
    RuntimeName.ONNX: ".onnx",
    RuntimeName.TORCHSCRIPT: ".torchscript",
}


class ModelManager:
    """Manages loading/unloading of YOLO models.

    Only one model+runtime is loaded at a time to fit within 16GB VRAM.
    """

    def __init__(self) -> None:
        self._loaded_model: Optional[YOLO] = None
        self._loaded_key: Optional[tuple[ModelName, RuntimeName]] = None

    @property
    def loaded_key(self) -> Optional[tuple[ModelName, RuntimeName]]:
        return self._loaded_key

    @property
    def model(self) -> Optional[YOLO]:
        return self._loaded_model

    def _weight_path(self, model: ModelName, runtime: RuntimeName) -> str:
        stem = MODEL_WEIGHTS[model]
        ext = RUNTIME_EXTENSIONS[runtime]
        return f"{stem}{ext}"

    def is_available(self, model: ModelName, runtime: RuntimeName) -> bool:
        """Check if the model weight file exists for the given runtime."""
        path = self._weight_path(model, runtime)
        return Path(path).exists()

    def is_loaded(self, model: ModelName, runtime: RuntimeName) -> bool:
        return self._loaded_key == (model, runtime)

    def load(self, model: ModelName, runtime: RuntimeName) -> YOLO:
        """Load a model+runtime, unloading the current one if different."""
        if self._loaded_key == (model, runtime) and self._loaded_model is not None:
            return self._loaded_model

        self.unload()

        path = self._weight_path(model, runtime)
        self._loaded_model = YOLO(path)
        self._loaded_key = (model, runtime)
        return self._loaded_model

    def unload(self) -> None:
        """Unload the current model to free GPU memory."""
        if self._loaded_model is not None:
            del self._loaded_model
            self._loaded_model = None
            self._loaded_key = None
            # Attempt to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def list_models(self) -> list[dict]:
        """List all models with their runtime availability and loaded status."""
        result = []
        for model in ModelName:
            runtimes = []
            for runtime in RuntimeName:
                runtimes.append({
                    "runtime": runtime.value,
                    "available": self.is_available(model, runtime),
                    "loaded": self.is_loaded(model, runtime),
                })
            result.append({
                "id": model.value,
                "object": "model",
                "runtimes": runtimes,
            })
        return result
