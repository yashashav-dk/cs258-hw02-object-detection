"""Merge Roboflow COCO export (train/valid/test splits) into a single
flat dataset for benchmark evaluation.

Roboflow exports COCO datasets with separate train/, valid/, test/
subdirectories, each with its own _annotations.coco.json. This script
flattens all three into a single images/ directory + a single
annotations.json file with consistent image_id and annotation_id
numbering.

Usage:
    python scripts/merge_roboflow_export.py \
        --roboflow-dir ~/Downloads/hw02-object-detection.v1i.coco \
        --output-dir ~/hw02-dataset
"""

import argparse
import json
import shutil
from pathlib import Path


def load_split(split_dir: Path) -> tuple[dict, list[Path]]:
    """Load a single split's annotations and image paths."""
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return None, []

    with open(ann_file) as f:
        data = json.load(f)

    image_files = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.jpeg")) + sorted(split_dir.glob("*.png"))
    return data, image_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Roboflow COCO export into a flat dataset")
    parser.add_argument("--roboflow-dir", type=Path, required=True, help="Unzipped Roboflow export root (contains train/valid/test)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (will create images/ subdir + annotations.json)")
    args = parser.parse_args()

    output_images = args.output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    merged = {
        "info": {"description": "HW02 Custom Object Detection Dataset (merged from Roboflow export)"},
        "licenses": [],
        "categories": None,
        "images": [],
        "annotations": [],
    }

    next_image_id = 1
    next_annotation_id = 1
    image_id_remap: dict[tuple[str, int], int] = {}

    for split_name in ("train", "valid", "test"):
        split_dir = args.roboflow_dir / split_name
        if not split_dir.exists():
            continue

        data, _ = load_split(split_dir)
        if data is None:
            continue

        # Categories: take from the first split that has them; verify others match
        if merged["categories"] is None:
            merged["categories"] = data.get("categories", [])
            print(f"Loaded {len(merged['categories'])} categories from {split_name}")

        # Build local image_id -> new image_id mapping for this split
        split_id_map: dict[int, int] = {}
        for img in data.get("images", []):
            old_id = img["id"]
            new_id = next_image_id
            next_image_id += 1
            split_id_map[old_id] = new_id

            # Copy image file to output_images
            src = split_dir / img["file_name"]
            if not src.exists():
                print(f"  WARN: missing image {src}")
                continue
            dst = output_images / img["file_name"]
            if not dst.exists():
                shutil.copy2(src, dst)

            merged["images"].append({
                "id": new_id,
                "file_name": img["file_name"],
                "width": img.get("width"),
                "height": img.get("height"),
                "license": img.get("license", 1),
            })

        # Remap annotations
        for ann in data.get("annotations", []):
            old_image_id = ann["image_id"]
            if old_image_id not in split_id_map:
                continue
            merged["annotations"].append({
                "id": next_annotation_id,
                "image_id": split_id_map[old_image_id],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })
            next_annotation_id += 1

        print(f"  {split_name}: merged {len(split_id_map)} images, {len([a for a in data.get('annotations', []) if a['image_id'] in split_id_map])} annotations")

    output_annotations = args.output_dir / "annotations.json"
    with open(output_annotations, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerge complete:")
    print(f"  Images:      {len(merged['images'])} (in {output_images})")
    print(f"  Annotations: {len(merged['annotations'])}")
    print(f"  Categories:  {len(merged['categories'])}")
    print(f"  Output JSON: {output_annotations}")

    if len(merged['images']) < 50:
        print(f"\nWARNING: Only {len(merged['images'])} images. Assignment requires at least 50.")


if __name__ == "__main__":
    main()
