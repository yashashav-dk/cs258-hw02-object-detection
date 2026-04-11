# Command Contracts: Deploy and Benchmark

All commands are executed in order. Each section is idempotent or
guarded against re-execution. Paths are absolute where needed.

## C1: Provision VM (local machine)

```bash
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

**Post-condition**: VM `yolo-v100` exists in project `cudabenchmarking`,
zone `us-west1-b`, with V100 GPU attached.

## C2: SSH into VM with Port Forwarding (local machine)

```bash
gcloud compute ssh yolo-v100 \
  --project=cudabenchmarking \
  --zone=us-west1-b \
  -- -L 8000:localhost:8000
```

**Post-condition**: Interactive shell on VM. Local port 8000 is
forwarded to VM port 8000 for the FastAPI backend.

On first SSH, accept the NVIDIA driver installation prompt if it
appears. Verify GPU with `nvidia-smi`.

## C3: Bootstrap (on VM)

```bash
# Clone repo
cd ~ && git clone https://github.com/yashashav-dk/cs258-hw02-object-detection.git homework
cd ~/homework

# Python venv
python3 -m venv venv
source venv/bin/activate

# Install backend deps
pip install --upgrade pip
pip install -r backend/requirements.txt

# Install TensorRT
pip install tensorrt

# Export models (downloads weights on first run)
python scripts/export_models.py
```

**Post-condition**: 6 model files exist in `~/homework/`:
`yolov8m.pt`, `yolov8m.onnx`, `yolov8m.engine`,
`yolo11m.pt`, `yolo11m.onnx`, `yolo11m.engine`.

Verify with: `ls *.pt *.onnx *.engine`

## C4: Upload Annotated Dataset (local machine)

Assumes dataset was annotated locally with Label Studio and exported
to `~/hw02-dataset/` with:
- `~/hw02-dataset/images/*.jpg` (50+ images)
- `~/hw02-dataset/video/clip.mp4`
- `~/hw02-dataset/annotations.json`

Transfer to VM:

```bash
gcloud compute scp --recurse ~/hw02-dataset/images \
  yolo-v100:~/homework/data/ \
  --project=cudabenchmarking --zone=us-west1-b

gcloud compute scp --recurse ~/hw02-dataset/video \
  yolo-v100:~/homework/data/ \
  --project=cudabenchmarking --zone=us-west1-b

gcloud compute scp ~/hw02-dataset/annotations.json \
  yolo-v100:~/homework/data/annotations.json \
  --project=cudabenchmarking --zone=us-west1-b
```

**Post-condition**: `~/homework/data/images/`,
`~/homework/data/video/`, and `~/homework/data/annotations.json`
exist on the VM.

## C5: Start Backend (on VM, optional)

```bash
cd ~/homework && source venv/bin/activate
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Post-condition**: Backend listening on port 8000. Accessible
locally via the SSH tunnel at `http://localhost:8000`.

Verify: in a second terminal on local machine, run
`curl http://localhost:8000/health`.

## C6: Run Benchmark (on VM)

```bash
cd ~/homework && source venv/bin/activate
python scripts/benchmark.py \
  --images data/images/ \
  --annotations data/annotations.json \
  --output results/benchmark.json
```

**Post-condition**: `results/benchmark.json` contains 6 entries with
mAP@0.5, mAP@0.5:0.95, avg_latency_ms, and speedup for each
model+runtime combo. Printed table shown on stdout.

## C7: Retrieve Results (local machine)

```bash
gcloud compute scp yolo-v100:~/homework/results/benchmark.json \
  ./results/benchmark.json \
  --project=cudabenchmarking --zone=us-west1-b

# Commit and push results
git add results/benchmark.json
git commit -m "results: add benchmark output from V100 VM"
git push
```

**Post-condition**: `results/benchmark.json` exists on local machine
and is committed to the repo (even though `results/` is gitignored
by default — benchmark.json will be force-added via `git add -f` or
by temporarily adjusting .gitignore for this file only).

**Note**: Update `.gitignore` to exclude `results/*` but keep
`!results/benchmark.json` committed.

## C8: Tear Down VM (local machine)

```bash
gcloud compute instances delete yolo-v100 \
  --project=cudabenchmarking \
  --zone=us-west1-b \
  --quiet
```

**Post-condition**: VM deleted. No further GPU billing.
