# Data Model: Deploy and Benchmark

**Date**: 2026-04-10

## Entities

### CloudVM

Represents the provisioned GCP compute instance.

| Field | Type | Description |
|-------|------|-------------|
| name | string | e.g. `yolo-v100` |
| project | string | `cudabenchmarking` |
| zone | string | `us-west1-b` |
| machine_type | string | `n1-standard-8` |
| accelerator | string | `nvidia-tesla-v100` (count=1) |
| image_family | string | `common-cu118` |
| image_project | string | `deeplearning-platform-release` |
| boot_disk_size_gb | int | 50 |
| external_ip | string | Ephemeral IPv4 (assigned on boot) |

### BootstrapState

Tracks progress of the VM bootstrap process. Each step is
idempotent and verifiable.

| Field | Type | Description |
|-------|------|-------------|
| nvidia_driver_installed | bool | `nvidia-smi` works |
| repo_cloned | bool | `~/homework` directory exists |
| venv_created | bool | `~/homework/venv` exists |
| backend_deps_installed | bool | `pip list` includes fastapi, ultralytics |
| tensorrt_installed | bool | `python -c "import tensorrt"` succeeds |
| models_exported | bool | 6 model files present |

### Dataset

Custom-annotated evaluation dataset.

| Field | Type | Description |
|-------|------|-------------|
| image_count | int | Minimum 50 |
| video_count | int | Minimum 1 (30-60 seconds) |
| annotations_path | string | `data/annotations.json` |
| images_path | string | `data/images/` |
| video_path | string | `data/video/` |
| categories | list[string] | COCO class names used |

### BenchmarkRun

A single execution of `scripts/benchmark.py`.

| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | Start time of the run |
| gpu_name | string | e.g. `Tesla V100` |
| cuda_version | string | e.g. `11.8` |
| driver_version | string | e.g. `525.105.17` |
| results | list[BenchmarkResult] | 6 rows (2 models × 3 runtimes) |
| total_runtime_seconds | float | Wall-clock duration |
| output_path | string | `results/benchmark.json` |

### BenchmarkResult

Per-combo output row (same schema as feature 001 data-model).

| Field | Type | Description |
|-------|------|-------------|
| model | string | `yolov8m` or `yolov11m` |
| runtime | string | `pytorch`, `onnx`, or `tensorrt` |
| map_50 | float | COCO mAP@0.5 |
| map_50_95 | float | COCO mAP@0.5:0.95 |
| avg_latency_ms | float | Average latency with warmup excluded |
| speedup | float | Speedup vs PyTorch baseline |

## Lifecycle

```
[none] → gcloud create → [VM provisioned] → ssh + bootstrap
  → [VM ready] → annotate + upload dataset → [dataset loaded]
  → run benchmark → [results captured] → git commit results
  → gcloud delete → [none]
```

The VM lifecycle is explicitly bounded: the instance MUST be deleted
after results are captured to avoid ongoing GPU billing.
