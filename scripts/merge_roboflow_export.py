"""Merge Roboflow COCO export (train/valid/test splits) into a single
flat dataset for benchmark evaluation, remapping category IDs to the
COCO 80-class standard so YOLO predictions align with ground truth.

Roboflow exports COCO datasets with separate train/, valid/, test/
subdirectories, each with its own _annotations.coco.json, and assigns
its own category IDs (not COCO standard). This script:

1. Flattens train/valid/test into a single images/ directory
2. Merges all annotations with consistent new image_id / annotation_id
3. Remaps Roboflow category IDs to COCO 80 standard IDs by class name
4. Writes a single annotations.json with COCO-standard category IDs

Usage:
    python scripts/merge_roboflow_export.py \\
        --roboflow-dir ~/Downloads/hw02-object-detection.v1i.coco \\
        --output-dir ~/hw02-dataset
"""

import argparse
import json
import shutil
from pathlib import Path

# Standard COCO 80-class name → ID (same as Ultralytics YOLO)
COCO_NAME_TO_ID = {
    "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "airplane": 4,
    "bus": 5, "train": 6, "truck": 7, "boat": 8, "traffic light": 9,
    "fire hydrant": 10, "stop sign": 11, "parking meter": 12, "bench": 13,
    "bird": 14, "cat": 15, "dog": 16, "horse": 17, "sheep": 18, "cow": 19,
    "elephant": 20, "bear": 21, "zebra": 22, "giraffe": 23, "backpack": 24,
    "umbrella": 25, "handbag": 26, "tie": 27, "suitcase": 28, "frisbee": 29,
    "skis": 30, "snowboard": 31, "sports ball": 32, "kite": 33,
    "baseball bat": 34, "baseball glove": 35, "skateboard": 36,
    "surfboard": 37, "tennis racket": 38, "bottle": 39, "wine glass": 40,
    "cup": 41, "fork": 42, "knife": 43, "spoon": 44, "bowl": 45,
    "banana": 46, "apple": 47, "sandwich": 48, "orange": 49, "broccoli": 50,
    "carrot": 51, "hot dog": 52, "pizza": 53, "donut": 54, "cake": 55,
    "chair": 56, "couch": 57, "potted plant": 58, "bed": 59,
    "dining table": 60, "toilet": 61, "tv": 62, "laptop": 63, "mouse": 64,
    "remote": 65, "keyboard": 66, "cell phone": 67, "microwave": 68,
    "oven": 69, "toaster": 70, "sink": 71, "refrigerator": 72, "book": 73,
    "clock": 74, "vase": 75, "scissors": 76, "teddy bear": 77,
    "toothbrush": 78,
}

# Common Roboflow/colloquial aliases → canonical COCO name
ALIASES = {
    "people": "person",
    "pedestrian": "person",
    "auto": "car",
    "automobile": "car",
    "bike": "bicycle",
    "cyclist": "bicycle",
    "motorbike": "motorcycle",
    "lorry": "truck",
    "phone": "cell phone",
    "cellphone": "cell phone",
    "mobile": "cell phone",
    "laptop computer": "laptop",
    "tv monitor": "tv",
    "television": "tv",
    "couch sofa": "couch",
    "sofa": "couch",
}


def canonicalize(name: str) -> str:
    """Normalize a class name for COCO lookup."""
    n = name.strip().lower()
    n = n.replace("_", " ").replace("-", " ")
    n = " ".join(n.split())  # collapse whitespace
    if n in ALIASES:
        n = ALIASES[n]
    return n


def build_category_mapping(roboflow_categories: list[dict]) -> tuple[dict[int, int], list[dict], list[str]]:
    """Map Roboflow category IDs → COCO standard IDs by name.

    Returns:
        (roboflow_id → coco_id map, coco-format category list for output, unmapped_names)
    """
    id_map: dict[int, int] = {}
    used_coco_ids: set[int] = set()
    unmapped: list[str] = []

    for cat in roboflow_categories:
        rb_id = cat["id"]
        rb_name = cat.get("name", "")
        canonical = canonicalize(rb_name)

        if canonical in COCO_NAME_TO_ID:
            coco_id = COCO_NAME_TO_ID[canonical]
            id_map[rb_id] = coco_id
            used_coco_ids.add(coco_id)
        else:
            unmapped.append(rb_name)

    # Build output categories list using only the COCO classes actually present
    # (pycocotools accepts any subset of COCO 80 as long as GT + predictions agree)
    id_to_name = {v: k for k, v in COCO_NAME_TO_ID.items()}
    output_categories = [
        {"id": coco_id, "name": id_to_name[coco_id], "supercategory": "object"}
        for coco_id in sorted(used_coco_ids)
    ]

    return id_map, output_categories, unmapped


def load_split(split_dir: Path) -> tuple[dict | None, list[Path]]:
    """Load a single split's annotations and image paths."""
    ann_file = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        return None, []
    with open(ann_file) as f:
        data = json.load(f)
    image_files = sorted(split_dir.glob("*.jpg")) + sorted(split_dir.glob("*.jpeg")) + sorted(split_dir.glob("*.png"))
    return data, image_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Roboflow COCO export into a flat dataset with COCO-standard category IDs")
    parser.add_argument("--roboflow-dir", type=Path, required=True, help="Unzipped Roboflow export root (contains train/valid/test)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (will create images/ subdir + annotations.json)")
    args = parser.parse_args()

    output_images = args.output_dir / "images"
    output_images.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect all Roboflow categories across splits so the ID map is global
    all_rb_categories: dict[int, dict] = {}
    for split_name in ("train", "valid", "test"):
        split_dir = args.roboflow_dir / split_name
        if not split_dir.exists():
            continue
        data, _ = load_split(split_dir)
        if data is None:
            continue
        for cat in data.get("categories", []):
            all_rb_categories[cat["id"]] = cat

    if not all_rb_categories:
        print("ERROR: no categories found in any split. Is the Roboflow export path correct?")
        return

    rb_cats_list = list(all_rb_categories.values())
    id_map, output_categories, unmapped = build_category_mapping(rb_cats_list)

    print("Category mapping:")
    id_to_name = {v: k for k, v in COCO_NAME_TO_ID.items()}
    for cat in rb_cats_list:
        rb_id = cat["id"]
        rb_name = cat.get("name", "")
        if rb_id in id_map:
            coco_id = id_map[rb_id]
            coco_name = id_to_name[coco_id]
            print(f"  {rb_name!r} (roboflow id {rb_id}) -> COCO id {coco_id} ({coco_name!r})")
        else:
            print(f"  {rb_name!r} (roboflow id {rb_id}) -> UNMAPPED (no COCO equivalent)")

    if unmapped:
        print("\nWARNING: some Roboflow classes could not be mapped to COCO 80:")
        for name in unmapped:
            print(f"  - {name!r}")
        print("Annotations for these classes will be dropped.")
        print("Fix: rename the class in Roboflow to a COCO-standard name and re-export.")

    # Pass 2: merge splits with image/annotation remapping
    merged = {
        "info": {"description": "HW02 Custom Object Detection Dataset (merged from Roboflow export, COCO 80 IDs)"},
        "licenses": [],
        "categories": output_categories,
        "images": [],
        "annotations": [],
    }

    next_image_id = 1
    next_annotation_id = 1
    dropped_annotations = 0

    for split_name in ("train", "valid", "test"):
        split_dir = args.roboflow_dir / split_name
        if not split_dir.exists():
            continue
        data, _ = load_split(split_dir)
        if data is None:
            continue

        split_id_map: dict[int, int] = {}

        for img in data.get("images", []):
            old_id = img["id"]
            new_id = next_image_id
            next_image_id += 1
            split_id_map[old_id] = new_id

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

        split_ann_count = 0
        for ann in data.get("annotations", []):
            if ann["image_id"] not in split_id_map:
                continue
            rb_cat_id = ann["category_id"]
            if rb_cat_id not in id_map:
                dropped_annotations += 1
                continue
            merged["annotations"].append({
                "id": next_annotation_id,
                "image_id": split_id_map[ann["image_id"]],
                "category_id": id_map[rb_cat_id],  # remapped to COCO
                "bbox": ann["bbox"],
                "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd": ann.get("iscrowd", 0),
                "segmentation": ann.get("segmentation", []),
            })
            next_annotation_id += 1
            split_ann_count += 1

        print(f"  {split_name}: merged {len(split_id_map)} images, {split_ann_count} annotations")

    output_annotations = args.output_dir / "annotations.json"
    with open(output_annotations, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerge complete:")
    print(f"  Images:              {len(merged['images'])} (in {output_images})")
    print(f"  Annotations:         {len(merged['annotations'])}")
    print(f"  Categories:          {len(merged['categories'])} COCO classes")
    if dropped_annotations:
        print(f"  Dropped annotations: {dropped_annotations} (from unmapped classes)")
    print(f"  Output JSON:         {output_annotations}")

    if len(merged["images"]) < 50:
        print(f"\nWARNING: Only {len(merged['images'])} images. Assignment requires at least 50.")


if __name__ == "__main__":
    main()
