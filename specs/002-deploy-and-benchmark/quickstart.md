# Quickstart: Deploy and Benchmark

This is the hands-on runbook. Each step corresponds to a command
contract in `contracts/commands.md`.

## Prerequisites

- gcloud CLI authenticated, project `cudabenchmarking`, zone `us-west1-b`
- Public repo at `https://github.com/yashashav-dk/cs258-hw02-object-detection`
- Label Studio available locally (`pip install label-studio`)

## Phase A: Annotate Dataset (Local, ~2-3 hours)

1. Capture 50+ images (phone camera, screen captures) featuring COCO
   classes: person, car, chair, laptop, bottle, dog, etc.
2. Record one 30-60 second video clip featuring similar objects.
3. Start Label Studio: `label-studio start`
4. Create project, import images, draw bounding boxes with COCO labels.
5. Export annotations in "COCO" format.
6. Organize as:
   ```
   ~/hw02-dataset/
   ├── images/
   │   ├── img_001.jpg
   │   └── ... (50+)
   ├── video/
   │   └── clip.mp4
   └── annotations.json
   ```

## Phase B: Provision VM (Local, ~3 minutes)

```bash
# Provision V100 VM
gcloud compute instances create yolo-v100 \
  --project=cudabenchmarking \
  --zone=us-west1-b \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=common-cu118 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-balanced \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform
```

## Phase C: SSH and Bootstrap (~15-25 minutes)

```bash
# SSH with port forwarding for backend access
gcloud compute ssh yolo-v100 \
  --project=cudabenchmarking \
  --zone=us-west1-b \
  -- -L 8000:localhost:8000
```

On the VM (wait for NVIDIA driver install prompt on first boot if any):

```bash
# Verify GPU
nvidia-smi

# Clone and bootstrap
cd ~ && git clone https://github.com/yashashav-dk/cs258-hw02-object-detection.git homework
cd ~/homework
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
pip install tensorrt

# Export models
python scripts/export_models.py

# Verify 6 model files
ls *.pt *.onnx *.engine
```

## Phase D: Upload Dataset (Local, ~5 minutes)

In a new local terminal:

```bash
gcloud compute scp --recurse ~/hw02-dataset/images yolo-v100:~/homework/data/ \
  --project=cudabenchmarking --zone=us-west1-b

gcloud compute scp --recurse ~/hw02-dataset/video yolo-v100:~/homework/data/ \
  --project=cudabenchmarking --zone=us-west1-b

gcloud compute scp ~/hw02-dataset/annotations.json yolo-v100:~/homework/data/annotations.json \
  --project=cudabenchmarking --zone=us-west1-b
```

## Phase E: Run Benchmark (~10-20 minutes on VM)

Back in the SSH session:

```bash
cd ~/homework && source venv/bin/activate
python scripts/benchmark.py \
  --images data/images/ \
  --annotations data/annotations.json \
  --output results/benchmark.json
```

The script prints a formatted table and saves the JSON.

## Phase F: Optional — Manual Test via Backend (~5 minutes)

On the VM:

```bash
cd ~/homework && source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Locally (SSH tunnel is already active, local port 8000 → VM port 8000):

```bash
cd frontend
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm install
npm run dev
```

Open http://localhost:3000 and upload an image.

## Phase G: Retrieve and Commit Results (Local, ~2 minutes)

```bash
# Pull results back to local
gcloud compute scp yolo-v100:~/homework/results/benchmark.json ./results/benchmark.json \
  --project=cudabenchmarking --zone=us-west1-b

# Commit results (force add since results/ is gitignored)
git add -f results/benchmark.json
git commit -m "results: benchmark output from V100 VM"
git push
```

## Phase H: Tear Down VM (Local, ~1 minute)

**IMPORTANT**: Do not skip this step. V100 instances cost ~$2.86/hr.

```bash
gcloud compute instances delete yolo-v100 \
  --project=cudabenchmarking \
  --zone=us-west1-b \
  --quiet
```

## Total Time & Cost Estimate

- **Time**: 3-5 hours (dataset annotation is the dominant cost)
- **GCP billing**: ~$1.50-$3.00 for 2-4 hours of VM runtime
- **Deliverable**: `results/benchmark.json` + printed table for the report
