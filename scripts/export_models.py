"""Export YOLOv8m and YOLOv11m models to ONNX and TorchScript formats.

Run on GCP VM with GPU:
    python scripts/export_models.py
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

MODELS = {
    "yolov8m": "yolov8m.pt",
    "yolov11m": "yolo11m.pt",
}

EXPORT_FORMATS = ["onnx", "torchscript"]


def export_model(model_name: str, weight_file: str, output_dir: Path) -> None:
    """Download (if needed) and export a model to ONNX and TorchScript."""
    print(f"\n{'='*60}")
    print(f"Exporting {model_name}")
    print(f"{'='*60}")

    model = YOLO(weight_file)

    for fmt in EXPORT_FORMATS:
        print(f"\n  -> Exporting to {fmt.upper()}...")
        try:
            exported_path = model.export(format=fmt)
            print(f"     Saved: {exported_path}")
        except Exception as e:
            print(f"     FAILED: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX and TorchScript")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory for exported models")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, weight_file in MODELS.items():
        export_model(model_name, weight_file, args.output_dir)

    print("\n" + "="*60)
    print("Export complete. Available model files:")
    for ext in ["*.pt", "*.onnx", "*.torchscript"]:
        for f in Path(".").glob(ext):
            print(f"  {f}")
    print("="*60)


if __name__ == "__main__":
    main()
