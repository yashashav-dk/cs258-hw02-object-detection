"""Benchmark script: evaluate all model+runtime combos on custom dataset.

Usage:
    python scripts/benchmark.py --images data/images/ --annotations data/annotations.json
    python scripts/benchmark.py --print-table results/benchmark.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO

MODELS = {
    "yolov8m": "yolov8m.pt",
    "yolov11m": "yolo11m.pt",
}

RUNTIME_FILES = {
    "pytorch": ".pt",
    "onnx": ".onnx",
    "tensorrt": ".engine",
}

WARMUP_RUNS = 3
NUM_RUNS = 1  # Number of inference passes per image for latency averaging


def convert_to_coco_results(
    predictions: list[dict], image_id: int, model_names: dict[int, str]
) -> list[dict]:
    """Convert Ultralytics predictions to COCO results format.

    COCO expects: [{"image_id", "category_id", "bbox"(xywh), "score"}]
    """
    results = []
    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox"]
        w = x2 - x1
        h = y2 - y1
        results.append({
            "image_id": image_id,
            "category_id": pred["category_id"],
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(pred["score"], 4),
        })
    return results


def run_inference(model: YOLO, image_path: str) -> tuple[list[dict], float]:
    """Run inference on a single image, return predictions and latency."""
    img = cv2.imread(image_path)
    if img is None:
        return [], 0.0

    start = time.perf_counter()
    results = model(img, verbose=False)
    latency_ms = (time.perf_counter() - start) * 1000

    predictions = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            predictions.append({
                "bbox": xyxy,
                "category_id": cls_id,
                "score": conf,
            })

    return predictions, latency_ms


def evaluate_model(
    model_name: str,
    runtime: str,
    image_dir: Path,
    coco_gt: COCO,
) -> dict:
    """Evaluate a single model+runtime combo."""
    stem = MODELS[model_name].replace(".pt", "")
    ext = RUNTIME_FILES[runtime]
    weight_path = f"{stem}{ext}"

    if not Path(weight_path).exists():
        print(f"  SKIP: {weight_path} not found")
        return {
            "model": model_name,
            "runtime": runtime,
            "map_50": None,
            "map_50_95": None,
            "avg_latency_ms": None,
            "speedup": None,
        }

    print(f"  Loading {weight_path}...")
    model = YOLO(weight_path)

    image_ids = coco_gt.getImgIds()
    all_coco_results = []
    latencies = []

    # Warmup
    if image_ids:
        first_img = coco_gt.loadImgs(image_ids[0])[0]
        warmup_path = str(image_dir / first_img["file_name"])
        for _ in range(WARMUP_RUNS):
            run_inference(model, warmup_path)

    # Inference
    for img_id in image_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = str(image_dir / img_info["file_name"])

        preds, latency_ms = run_inference(model, img_path)
        latencies.append(latency_ms)

        coco_results = convert_to_coco_results(preds, img_id, model.names)
        all_coco_results.extend(coco_results)

    # Compute mAP
    map_50 = 0.0
    map_50_95 = 0.0

    if all_coco_results:
        coco_dt = coco_gt.loadRes(all_coco_results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map_50_95 = float(coco_eval.stats[0])  # AP @ IoU=0.50:0.95
        map_50 = float(coco_eval.stats[1])  # AP @ IoU=0.50

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Free GPU memory
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return {
        "model": model_name,
        "runtime": runtime,
        "map_50": round(map_50, 4),
        "map_50_95": round(map_50_95, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "speedup": None,  # Computed after all runtimes for a model
    }


def compute_speedups(results: list[dict]) -> list[dict]:
    """Compute speedup relative to PyTorch baseline for each model."""
    baselines = {}
    for r in results:
        if r["runtime"] == "pytorch" and r["avg_latency_ms"] is not None:
            baselines[r["model"]] = r["avg_latency_ms"]

    for r in results:
        baseline = baselines.get(r["model"])
        if baseline and r["avg_latency_ms"] is not None and r["avg_latency_ms"] > 0:
            r["speedup"] = round(baseline / r["avg_latency_ms"], 2)
        elif r["runtime"] == "pytorch" and r["avg_latency_ms"] is not None:
            r["speedup"] = 1.0

    return results


def print_results_table(results: list[dict]) -> None:
    """Print formatted results table."""
    header = f"{'Model':<12} {'Runtime':<12} {'mAP@0.5':>8} {'mAP@0.5:0.95':>14} {'Avg Latency (ms)':>18} {'Speedup':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        map50 = f"{r['map_50']:.4f}" if r["map_50"] is not None else "N/A"
        map5095 = f"{r['map_50_95']:.4f}" if r["map_50_95"] is not None else "N/A"
        latency = f"{r['avg_latency_ms']:.2f}" if r["avg_latency_ms"] is not None else "N/A"
        speedup = f"{r['speedup']:.2f}x" if r["speedup"] is not None else "N/A"
        print(f"{r['model']:<12} {r['runtime']:<12} {map50:>8} {map5095:>14} {latency:>18} {speedup:>8}")

    print("=" * len(header))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark object detection models")
    parser.add_argument("--images", type=Path, help="Path to images directory")
    parser.add_argument("--annotations", type=Path, help="Path to COCO JSON annotations")
    parser.add_argument("--output", type=Path, help="Save results JSON to this path")
    parser.add_argument("--print-table", type=Path, help="Print table from saved results JSON")
    args = parser.parse_args()

    if args.print_table:
        with open(args.print_table) as f:
            results = json.load(f)
        print_results_table(results)
        return

    if not args.images or not args.annotations:
        parser.error("--images and --annotations are required for evaluation")

    print(f"Loading annotations from {args.annotations}...")
    coco_gt = COCO(str(args.annotations))

    results = []
    for model_name in MODELS:
        for runtime in RUNTIME_FILES:
            print(f"\nEvaluating {model_name} + {runtime}...")
            result = evaluate_model(model_name, runtime, args.images, coco_gt)
            results.append(result)

    results = compute_speedups(results)
    print_results_table(results)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
