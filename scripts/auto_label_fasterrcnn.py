"""Auto-label captured frames using torchvision's Faster R-CNN as a
reference labeler.

Generates ground truth bounding boxes in COCO JSON format compatible
with the benchmark script (uses Ultralytics / YOLO COCO 80-class IDs).

Uses a different architecture (two-stage Faster R-CNN ResNet-50 FPN)
than the models under test (one-stage YOLO), so the benchmark
measures YOLO's agreement with an independent reference — not YOLO
against itself (which would be circular).

Also produces visualization samples for spot-checking.

Usage:
    python3 scripts/auto_label_fasterrcnn.py \\
        --frames-dir ~/hw02-dataset/raw-frames \\
        --output-dir ~/hw02-dataset \\
        --confidence 0.5
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

# Import COCO name → YOLO 80 ID mapping from the merge script to keep
# category IDs consistent with the benchmark expectations.
sys.path.insert(0, str(Path(__file__).parent))
from merge_roboflow_export import COCO_NAME_TO_ID  # noqa: E402

import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.models.detection import (  # noqa: E402
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import to_tensor  # noqa: E402


# Torchvision Faster R-CNN uses COCO 91-class IDs; this is the full
# category name list in order of their id.
TORCHVISION_COCO_NAMES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "N/A",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
    "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "N/A", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

# Keep only these classes for an urban dashcam scene
ALLOWED_CLASSES = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "traffic light", "stop sign",
}


def detect_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("mps", "auto") and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer in ("cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def visualize_samples(
    coco: dict,
    images_dir: Path,
    viz_dir: Path,
    num_samples: int,
    id_to_name: dict[int, str],
) -> None:
    """Draw bboxes on a few sample images for visual spot-check."""
    viz_dir.mkdir(parents=True, exist_ok=True)

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    id_to_image = {img["id"]: img for img in coco["images"]}
    rng = random.Random(42)
    sample_ids = rng.sample(list(id_to_image.keys()), min(num_samples, len(id_to_image)))

    # Distinct BGR colors for cv2
    palette = [
        (68, 68, 239), (246, 130, 59), (94, 197, 34), (8, 179, 234),
        (168, 85, 168), (153, 72, 236), (166, 184, 20), (22, 115, 249),
    ]

    for img_id in sample_ids:
        img_info = id_to_image[img_id]
        src = images_dir / img_info["file_name"]
        img = cv2.imread(str(src))
        if img is None:
            continue

        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cid = ann["category_id"]
            color = palette[cid % len(palette)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{id_to_name.get(cid, '?')} {ann.get('score', 1.0):.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 2, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

        dst = viz_dir / f"viz_{img_info['file_name']}"
        cv2.imwrite(str(dst), img)

    print(f"Visualization samples saved to: {viz_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-label frames with Faster R-CNN")
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--num-viz", type=int, default=8, help="Number of visualization samples to produce")
    args = parser.parse_args()

    images_out = args.output_dir / "images"
    viz_out = args.output_dir / "viz"
    images_out.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(
        list(args.frames_dir.glob("*.jpg"))
        + list(args.frames_dir.glob("*.jpeg"))
        + list(args.frames_dir.glob("*.png"))
    )
    if not frame_files:
        print(f"ERROR: no frames found in {args.frames_dir}", file=sys.stderr)
        sys.exit(1)

    device = detect_device(args.device)
    print(f"Device: {device}")
    print(f"Frames: {len(frame_files)}")
    print(f"Loading Faster R-CNN (ResNet-50 FPN)...")

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    model.to(device)

    # Build COCO JSON structure
    coco = {
        "info": {"description": "HW02 auto-labeled dataset via torchvision Faster R-CNN ResNet-50 FPN"},
        "licenses": [{"id": 1, "name": "Student Work", "url": ""}],
        "categories": [],
        "images": [],
        "annotations": [],
    }

    # Categories with YOLO 80-class IDs (only the ones we allow)
    used_ids = sorted({COCO_NAME_TO_ID[name] for name in ALLOWED_CLASSES if name in COCO_NAME_TO_ID})
    id_to_name = {v: k for k, v in COCO_NAME_TO_ID.items()}
    coco["categories"] = [
        {"id": cid, "name": id_to_name[cid], "supercategory": "object"}
        for cid in used_ids
    ]

    next_image_id = 1
    next_annotation_id = 1

    t0 = time.perf_counter()
    with torch.no_grad():
        for i, frame_path in enumerate(frame_files, start=1):
            img_pil = Image.open(frame_path).convert("RGB")
            width, height = img_pil.size
            img_tensor = to_tensor(img_pil).to(device)

            dst = images_out / frame_path.name
            if not dst.exists():
                shutil.copy2(frame_path, dst)

            predictions = model([img_tensor])[0]

            image_id = next_image_id
            next_image_id += 1

            coco["images"].append({
                "id": image_id,
                "file_name": frame_path.name,
                "width": width,
                "height": height,
                "license": 1,
            })

            boxes = predictions["boxes"].cpu().tolist()
            scores = predictions["scores"].cpu().tolist()
            labels = predictions["labels"].cpu().tolist()

            kept = 0
            for box, score, label_id in zip(boxes, scores, labels):
                if score < args.confidence:
                    continue
                if label_id < 0 or label_id >= len(TORCHVISION_COCO_NAMES):
                    continue
                class_name = TORCHVISION_COCO_NAMES[label_id]
                if class_name not in ALLOWED_CLASSES:
                    continue
                if class_name not in COCO_NAME_TO_ID:
                    continue

                yolo_id = COCO_NAME_TO_ID[class_name]
                x1, y1, x2, y2 = box
                # Clip to image bounds
                x1 = max(0.0, min(width - 1.0, x1))
                y1 = max(0.0, min(height - 1.0, y1))
                x2 = max(0.0, min(width - 1.0, x2))
                y2 = max(0.0, min(height - 1.0, y2))
                w = x2 - x1
                h = y2 - y1
                if w < 2 or h < 2:
                    continue

                coco["annotations"].append({
                    "id": next_annotation_id,
                    "image_id": image_id,
                    "category_id": yolo_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "area": round(w * h, 2),
                    "iscrowd": 0,
                    "segmentation": [],
                    "score": round(score, 4),  # extra field; pycocotools ignores
                })
                next_annotation_id += 1
                kept += 1

            print(f"  [{i}/{len(frame_files)}] {frame_path.name}: {kept} detections")

    elapsed = time.perf_counter() - t0
    print(f"\nInference took {elapsed:.1f}s")

    output_annotations = args.output_dir / "annotations.json"
    with open(output_annotations, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"\nAuto-labeling complete:")
    print(f"  Images:              {len(coco['images'])}")
    print(f"  Annotations:         {len(coco['annotations'])}")
    print(f"  Categories used:     {len(coco['categories'])}")
    print(f"  Output JSON:         {output_annotations}")
    print(f"  Images dir:          {images_out}")

    # Class distribution
    class_counts: dict[str, int] = {}
    for ann in coco["annotations"]:
        name = id_to_name.get(ann["category_id"], f"id_{ann['category_id']}")
        class_counts[name] = class_counts.get(name, 0) + 1

    print("\nClass distribution:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:15s} {count}")

    if args.num_viz > 0:
        visualize_samples(coco, images_out, viz_out, args.num_viz, id_to_name)


if __name__ == "__main__":
    main()
