# HW02 Object Detection — Inference Optimization

Full-stack object detection inference optimization pipeline using YOLOv8m and YOLOv11m, with FastAPI backend (OpenAI-compatible API), Next.js frontend, and ONNX Runtime + TorchScript acceleration.

**Course**: CS 258 — Spring 2026
**Assignment**: HW02 Object Detection (Option 2: Inference Optimization)

**Full report**: [docs/report.md](docs/report.md)

## TL;DR Results

Benchmark on NVIDIA Tesla V100-SXM2-16GB, 60 SF dashcam frames,
warm-up excluded, reproducibility verified (2 runs, identical mAP,
<3% latency variance).

```
Model        Runtime       mAP@0.5   mAP@0.5:0.95   Latency (ms)  Speedup
-------------------------------------------------------------------------
yolov8m      pytorch        0.5744         0.4310         10.73    1.00x
yolov8m      onnx           0.5744         0.4303         16.61    0.65x
yolov8m      torchscript    0.5744         0.4303         11.25    0.95x
yolov11m     pytorch        0.5495         0.4168         13.09    1.00x
yolov11m     onnx           0.5726         0.4211         18.64    0.70x
yolov11m     torchscript    0.5726         0.4211         12.38    1.06x
```

**Key finding**: ONNX Runtime is counterintuitively ~30-35% *slower*
than eager PyTorch on this V100 + Ultralytics stack. TorchScript is
approximately on par with PyTorch. mAP is preserved across runtimes.
See [docs/report.md §5.3](docs/report.md) for analysis.

**TensorRT was planned but pivoted**: TensorRT 10.4+ dropped Volta
(sm_70) support, and our GCP V100 quota was the only available high-memory
GPU. TorchScript is explicitly on the assignment's approved list and
was used as the second acceleration method.

## What's Inside

- **Backend** (`backend/`): FastAPI service with OpenAI-compatible `/v1/detect`, `/v1/models`, `/v1/detect/compare` endpoints
- **Frontend** (`frontend/`): Next.js app for uploading images/videos, visualizing detections, and comparing model+runtime combinations
- **Scripts** (`scripts/`):
  - `export_models.py` — downloads YOLOv8m/YOLOv11m weights and exports to ONNX + TensorRT
  - `benchmark.py` — computes COCO mAP + latency for all 6 model+runtime combos on the custom dataset
- **Data** (`data/`): Custom-annotated dataset (images gitignored, COCO JSON committed)

## Architecture

```
┌─────────────────────┐  HTTP   ┌────────────────────────┐
│  Next.js Frontend   │ ──────→ │  FastAPI Backend       │
│  - Upload           │         │  /v1/detect            │
│  - Model selector   │         │  /v1/detect/upload     │
│  - BBox overlay     │         │  /v1/detect/compare    │
│  - Comparison view  │         │  /v1/models            │
│  (localhost:3000)   │         │                        │
└─────────────────────┘         │  YOLOv8m / YOLOv11m    │
                                │  × PyTorch/ONNX/TRT    │
                                │  (GCP VM + T4 GPU)     │
                                └────────────────────────┘
```

**Memory strategy**: Only one model+runtime loaded at a time to stay within T4 16GB VRAM. The previous model is unloaded before the next is loaded.

## Setup

### Prerequisites

- Google Cloud VM with NVIDIA T4 GPU (Ubuntu 22.04, CUDA 11.8, cuDNN 8)
- Python 3.11+
- Node.js 18+
- Git

### 1. GCP VM — Backend Setup

```bash
# SSH into GCP VM
gcloud compute ssh <vm-name> --zone=<zone>

# Clone repo and enter it
git clone <repo-url> && cd homework

# Create Python virtualenv
python3 -m venv venv && source venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Install TensorRT for engine export (required for .engine files)
pip install tensorrt

# Export YOLOv8m and YOLOv11m to ONNX and TensorRT
# This downloads weights on first run
python scripts/export_models.py

# Start the FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Backend is now at http://<vm-external-ip>:8000
```

### 2. Local Machine — Frontend Setup

```bash
# On your local machine
cd frontend

# Install dependencies
npm install

# Point frontend at the GCP VM backend
echo "NEXT_PUBLIC_API_URL=http://<vm-external-ip>:8000" > .env.local

# Start dev server
npm run dev

# Frontend is now at http://localhost:3000
```

### 3. Custom Dataset — Annotation

```bash
# Install Label Studio locally
pip install label-studio
label-studio start

# 1. Create a new project in Label Studio
# 2. Import your images (minimum 50) + one short video clip (30-60s)
# 3. Annotate bounding boxes for COCO classes (person, car, chair, etc.)
# 4. Export annotations in "COCO" format
# 5. Place the export as data/annotations.json
# 6. Place images in data/images/
# 7. Place video in data/video/
```

## Usage

### Web UI

1. Open http://localhost:3000
2. **Detect page**: upload image or video → pick model + runtime → click Detect
3. **Compare page**: upload image → click Compare All → see all 6 combos side-by-side

### Run the Benchmark

```bash
# On the GCP VM
python scripts/benchmark.py \
  --images data/images/ \
  --annotations data/annotations.json \
  --output results/benchmark.json

# View a saved results table without re-running
python scripts/benchmark.py --print-table results/benchmark.json
```

Output example:

```
Model        Runtime      mAP@0.5  mAP@0.5:0.95  Avg Latency (ms)  Speedup
-----------------------------------------------------------------------------
yolov8m      pytorch       0.8234        0.6543             42.18     1.00x
yolov8m      onnx          0.8230        0.6540             31.24     1.35x
yolov8m      tensorrt      0.8201        0.6511             19.87     2.12x
yolov11m     pytorch       0.8412        0.6712             44.56     1.00x
yolov11m     onnx          0.8410        0.6710             32.89     1.35x
yolov11m     tensorrt      0.8380        0.6680             20.93     2.13x
```

## Reproducing Results

1. Follow Setup steps 1-3 on the GCP VM
2. Run `python scripts/export_models.py` to generate `.onnx` and `.engine` files
3. Prepare the dataset in `data/`
4. Run `python scripts/benchmark.py --images data/images/ --annotations data/annotations.json --output results/benchmark.json`
5. Results are reproducible (identical mAP, latency within 10% variance)

## Assignment Requirements Coverage

| Requirement | Where |
|-------------|-------|
| Two strong-performing models | YOLOv8m + YOLOv11m (medium variants) |
| FastAPI backend (OpenAI-compatible) | `backend/main.py`, `backend/routers/` |
| Frontend (Next.js) with upload + visualization | `frontend/src/app/page.tsx` |
| Two acceleration methods | TensorRT + ONNX Runtime (via Ultralytics export) |
| Accuracy (mAP) + speed (latency) evaluation | `scripts/benchmark.py` |
| Custom video/image data + own annotations | `data/` (Label Studio workflow) |

## Notes

- Pre-trained weights (YOLOv8m, YOLOv11m) from Ultralytics are used — Option 2 focuses on inference optimization, not training.
- The custom dataset uses COCO classes so pre-trained model weights are applicable out of the box.
- ONNX Runtime uses the CUDA execution provider; TensorRT uses FP16 precision by default.
- Model weights (`.pt`, `.onnx`, `.engine`), dataset images, and benchmark results are gitignored.
