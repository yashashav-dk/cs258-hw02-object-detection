"""Render all images with bounding boxes drawn from an existing COCO
annotations JSON.

This is a pure visualization step — it does NOT re-run any model. It
reads the existing annotations file and draws the boxes onto copies
of each image so the user can visually review annotation quality.

Usage:
    python3 scripts/render_viz.py \\
        --annotations ~/hw02-dataset/annotations.json \\
        --images-dir ~/hw02-dataset/images \\
        --output-dir ~/hw02-dataset/viz_all
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

# Distinct BGR colors for cv2 drawing, keyed by COCO category id
PALETTE = {
    0:  (68, 68, 239),    # person - red
    1:  (34, 197, 94),    # bicycle - green
    2:  (246, 130, 59),   # car - orange
    3:  (234, 179, 8),    # motorcycle - yellow
    5:  (168, 85, 247),   # bus - purple
    7:  (59, 130, 246),   # truck - blue
    9:  (20, 184, 166),   # traffic light - teal
    11: (236, 72, 153),   # stop sign - pink
}
DEFAULT_COLOR = (128, 128, 128)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render bboxes from COCO annotations onto images")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.annotations) as f:
        coco = json.load(f)

    id_to_image = {img["id"]: img for img in coco["images"]}
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    total = len(id_to_image)
    rendered = 0
    total_boxes = 0

    for img_id, img_info in sorted(id_to_image.items()):
        src = args.images_dir / img_info["file_name"]
        if not src.exists():
            print(f"  skip (missing file): {src}")
            continue

        img = cv2.imread(str(src))
        if img is None:
            print(f"  skip (unreadable): {src}")
            continue

        anns = anns_by_image.get(img_id, [])
        total_boxes += len(anns)

        # Draw all bboxes
        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            cid = ann["category_id"]
            color = PALETTE.get(cid, DEFAULT_COLOR)
            thickness = max(2, min(img.shape[0], img.shape[1]) // 400)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = id_to_name.get(cid, f"id_{cid}")
            score = ann.get("score")
            if score is not None:
                label = f"{label} {score:.2f}"

            fontScale = max(0.5, min(img.shape[0], img.shape[1]) / 1200)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fontScale, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                img, label, (x1 + 3, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 255, 255), 1, cv2.LINE_AA,
            )

        # Add frame index + detection count in top-left corner
        header = f"{img_info['file_name']}  |  {len(anns)} detections"
        cv2.rectangle(img, (0, 0), (min(img.shape[1], 500), 24), (0, 0, 0), -1)
        cv2.putText(
            img, header, (6, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        dst = args.output_dir / f"viz_{img_info['file_name']}"
        cv2.imwrite(str(dst), img)
        rendered += 1

    print(f"\nRendered {rendered}/{total} images to {args.output_dir}")
    print(f"Total boxes drawn: {total_boxes}")
    print(f"Avg boxes per image: {total_boxes / rendered:.1f}" if rendered else "")


if __name__ == "__main__":
    main()
