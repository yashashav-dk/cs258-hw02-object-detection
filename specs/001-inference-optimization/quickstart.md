# Quickstart: Inference Optimization Pipeline

## Prerequisites

- Google Cloud VM with NVIDIA T4 GPU (Ubuntu 22.04)
- CUDA 11.8 + cuDNN 8.x installed on VM
- Python 3.11+ on VM
- Node.js 18+ on local machine
- Git

## 1. GCP VM Setup (backend + inference)

```bash
# SSH into your GCP VM
gcloud compute ssh <vm-name> --zone=<zone>

# Clone the repo
git clone <repo-url> && cd homework

# Create Python virtual environment
python3 -m venv venv && source venv/bin/activate

# Install backend dependencies
pip install -r requirements.txt

# Install TensorRT (required for TensorRT export)
pip install tensorrt

# Export models to ONNX and TensorRT
python scripts/export_models.py

# Start the FastAPI backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## 2. Local Setup (frontend)

```bash
# In a new terminal on your local machine
cd frontend

# Install dependencies
npm install

# Set backend URL (your GCP VM external IP)
echo "NEXT_PUBLIC_API_URL=http://<vm-external-ip>:8000" > .env.local

# Start dev server
npm run dev
# Frontend at http://localhost:3000
```

## 3. Prepare Custom Dataset

```bash
# On your local machine, install Label Studio
pip install label-studio
label-studio start

# Import your images, annotate bounding boxes
# Export annotations in COCO JSON format
# Place in: data/annotations.json and data/images/
```

## 4. Run Benchmark

```bash
# On the GCP VM
python scripts/benchmark.py \
  --images data/images/ \
  --annotations data/annotations.json \
  --output results/benchmark.json

# View results table
python scripts/benchmark.py --print-table results/benchmark.json
```

## 5. Verify

1. Open http://localhost:3000 in browser
2. Upload a test image → select YOLOv8m + PyTorch → click Detect
3. Verify bounding boxes and latency appear
4. Try "Compare All" to see all 6 model+runtime results
5. Run benchmark script and verify mAP + latency table
