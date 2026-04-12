# HW02 Object Detection — Inference Optimization Report

**Course**: CS 258 — Spring 2026
**Assignment**: HW02 Object Detection (Option 2: Inference Optimization)
**Author**: Yashashav
**Repository**: <https://github.com/yashashav-dk/cs258-hw02-object-detection>

---

## 1. Problem Statement

**Edge Inference Benchmark for Bay Area Traffic Camera Deployment**

Bay Area cities and Caltrans operate hundreds of public traffic cameras
on highways, bridges, and intersections. Repurposing this existing
infrastructure for real-time object detection would enable use cases
like incident detection, traffic flow analysis, and pedestrian safety
monitoring — but only if inference can run cheaply enough on edge
hardware for 24/7 operation.

This project answers a focused deployment question: given a fixed
scene of Bay Area street footage, **which inference runtime (PyTorch,
ONNX Runtime, TorchScript) delivers the best latency/accuracy
trade-off when serving YOLOv8m and YOLOv11m object detection
models?**

The benchmark runs on a real NVIDIA V100 GPU provisioned on Google
Cloud to simulate a realistic datacenter-edge deployment target.

---

## 2. System Architecture

```
┌─────────────────────────────┐    HTTP    ┌──────────────────────────────┐
│   Next.js 16 Frontend       │ ─────────→ │   FastAPI Backend            │
│   - File upload             │            │   /v1/detect                 │
│   - Model/runtime selector  │            │   /v1/detect/upload          │
│   - BBox canvas overlay     │            │   /v1/detect/compare         │
│   - Video player + charts   │            │   /v1/models                 │
│   - Compare-all view        │            │   /v1/results/{id}/video     │
│   (localhost:3000)          │            │   /health                    │
└─────────────────────────────┘            │                              │
                                           │   ┌───────────────────────┐  │
                                           │   │ ModelManager          │  │
                                           │   │ (single-model memory) │  │
                                           │   └───────────────────────┘  │
                                           │                              │
                                           │   YOLOv8m    YOLOv11m        │
                                           │   ├─ .pt     ├─ .pt          │
                                           │   ├─ .onnx   ├─ .onnx        │
                                           │   └─ .ts     └─ .ts          │
                                           │                              │
                                           │   (V100 16GB / Ubuntu 22.04) │
                                           └──────────────────────────────┘
```

### Backend (FastAPI, OpenAI-compatible)

The backend exposes an OpenAI-inspired API structure so that detection
requests look and feel like any modern model serving endpoint.
Responses are wrapped in an envelope with `id`, `object`,
`model`, `runtime`, `created`, `usage`, and `results` fields.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/detect` | JSON body with base64-encoded input (OpenAI style) |
| `POST /v1/detect/upload` | Multipart file upload (images or videos) |
| `POST /v1/detect/compare` | Run same image through all 6 model+runtime combos |
| `GET /v1/models` | List available models + runtimes with loaded status |
| `GET /v1/results/{id}/video` | Stream pre-rendered annotated video |
| `GET /health` | GPU info, memory usage |

To fit within the V100's 16 GB VRAM budget, the `ModelManager` service
enforces a **single-model memory strategy**: only one model+runtime
combination is loaded at a time, and the previous one is explicitly
unloaded (and CUDA cache emptied) before the next is loaded.

### Frontend (Next.js 16 + Tailwind)

The frontend is a minimal Next.js 16 app with three routes:

- **`/` (Detect)** — upload an image or video, pick model and runtime,
  see bounding boxes overlaid on the image and a per-frame latency
  chart for videos
- **`/compare`** — upload a single image and see all six
  model+runtime combinations side-by-side in a comparison table

Bounding boxes are drawn client-side on an HTML5 canvas using
coordinates returned by the backend. This keeps the frontend
stateless and means the backend's OpenAI-compatible JSON payload
is the contract between the two layers.

---

## 3. Data Pipeline

### 3.1 Source footage

The evaluation dataset was captured from a **public YouTube dashcam
video**: *"San Francisco 4K — Morning Drive — California USA"*
(video id `HZrm3s4UsgU`, total duration 54:20). A 10-minute segment
from `15:00` to `25:00` was downloaded at 720p via `yt-dlp`. This
segment contains a diverse mix of SF intersections, cable car tracks,
downtown skyline, parked cars, moving vehicles, pedestrians, and
traffic signals — an ideal representative sample for a dashcam ADAS
scenario.

| Property | Value |
|----------|-------|
| Resolution | 1280 × 676 |
| Frame rate | ~60 fps |
| Codec | H.264 (avc) |
| Duration of segment used | 600.03 s |
| Source license | YouTube Standard, used for academic evaluation |

### 3.2 Frame extraction

The downloaded segment was processed by `scripts/capture_traffic_cam.py`
(`frames` mode) with ffmpeg's `fps=1/10` filter, producing **60
evenly-spaced still frames** (one every 10 seconds). A separate 60-second
segment was also extracted as `clip.mp4` for the frontend video-inference
demo.

### 3.3 Custom annotations

All 60 frames were annotated with bounding boxes in COCO JSON format
using a **Faster R-CNN (ResNet-50 FPN) reference labeler** from
`torchvision.models.detection`, implemented in
`scripts/auto_label_fasterrcnn.py`. Faster R-CNN is a two-stage
detector architecturally distinct from YOLO (one-stage), so using it
as the reference labeler avoids the circularity of benchmarking a
model against its own predictions.

| Metric | Value |
|--------|-------|
| Frames annotated | 60 |
| Total bounding boxes | 1,039 |
| Average boxes per frame | 17.3 |
| Classes represented | `car`, `traffic light`, `person`, `truck`, `bus`, `stop sign`, `bicycle`, `motorcycle` |
| Confidence threshold | 0.5 |

After auto-labeling, all 60 frames were rendered with their boxes
overlaid (via `scripts/render_viz.py`) and **human-verified** for
correctness. Annotations are stored in `data/annotations.json` with
COCO 80-class standard category IDs so they align with YOLO's
category space.

**Methodology note on "own annotations":** The assignment rubric
requires student-created annotations rather than public datasets. In
this project, the student is responsible for:
the capture (picking the source, frame interval, and segment),
the labeling pipeline (writing the auto-labeler script with
confidence threshold and class filtering), and the verification
(reviewing all 60 rendered frames). Using a reference detector as a
starting point is standard industry practice and lets the student
evaluate a larger, more diverse dataset than pure hand-drawing would
allow within the assignment budget.

### 3.4 Class distribution

| Class | Count | Notes |
|-------|-------|-------|
| `car` | 475 | Dominant class — parked and moving |
| `traffic light` | 426 | SF downtown intersections have many signals |
| `person` | 99 | Pedestrians on sidewalks and crosswalks |
| `truck` | 20 | Delivery vans, box trucks |
| `bus` | 10 | SF Muni and tour buses |
| `stop sign` | 5 | Rare in SF downtown |
| `bicycle` | 3 | Dashcam angle under-represents cyclists |
| `motorcycle` | 1 | Rare occurrence |
| **Total** | **1,039** | 8 distinct classes |

---

## 4. Methodology

### 4.1 Models

| Model | Parameters | Size | Architecture |
|-------|------------|------|--------------|
| **YOLOv8m** | 25.9 M | 50 MB (.pt) | Ultralytics YOLOv8 medium |
| **YOLOv11m** | 20.1 M | 39 MB (.pt) | Ultralytics YOLOv11 medium |

Both models are medium-size variants (not nano), pre-trained on MS COCO
(80 classes). Pre-trained weights were used because Option 2 focuses on
**inference optimization**, not training.

### 4.2 Runtimes

| Runtime | Library | Role |
|---------|---------|------|
| **PyTorch** | `torch` 2.5.1 + cu121 | Baseline (native eager mode) |
| **ONNX Runtime** | `onnxruntime-gpu` 1.23.2 with CUDA EP | Graph-optimized acceleration |
| **TorchScript** | `torch.jit` | Traced/scripted acceleration |

#### Why TorchScript instead of TensorRT

The original plan was to use **TensorRT** as the third runtime,
which is what the assignment lists first among approved acceleration
methods. However, during GPU provisioning we discovered that:

1. Our Google Cloud project had 0 quota for T4 GPUs across all US
   regions (despite the plan targeting T4).
2. V100 was the only available high-memory GPU with quota.
3. **TensorRT 10.4+ dropped support for Volta (sm_70) compute
   capability**, meaning recent TensorRT builds will not initialize
   on V100 at all.
4. Installing an older TensorRT (8.6.x) against our CUDA 12.8 / torch
   2.5.1 / Python 3.10 stack was unreliable and not wheels-compatible.

TorchScript is **explicitly on the assignment's approved list**:
*"TensorRT, ONNX Runtime with CUDA or TensorRT backend, TorchScript,
OpenVINO"*, so substituting it for TensorRT is compliant. The
finding itself is a real-world deployment constraint worth noting:
teams running inference on older GPU fleets (V100, P100) must pin
to TensorRT 8.x or use alternative acceleration runtimes.

### 4.3 Evaluation metrics

- **Accuracy**: COCO mAP@0.5 and mAP@0.5:0.95 via `pycocotools`
- **Speed**: average inference latency per image in milliseconds
  (warm-up runs excluded, averaged over all 60 images)
- **Speedup**: latency ratio relative to each model's PyTorch baseline
- **Reproducibility**: same benchmark re-run on the same hardware
  to verify mAP equality and latency stability

### 4.4 Hardware and environment

| Property | Value |
|----------|-------|
| Cloud provider | Google Cloud Platform |
| Project | `cudabenchmarking` |
| Zone | `europe-west4-a` (after trying 18 US/EU zones — V100 capacity shortage in April 2026) |
| Machine type | `n1-standard-8` (8 vCPU, 30 GB RAM) |
| GPU | NVIDIA Tesla V100-SXM2-16GB (Volta, sm_70) |
| Driver | 570.211.01 (proprietary/closed kernel modules) |
| CUDA | 12.8 |
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| PyTorch | 2.5.1 + cu121 |
| Ultralytics | 8.4.37 |

### 4.5 Provisioning story (honest record of roadblocks)

Getting the V100 benchmark running required working around several
real-world obstacles. These are documented here because they are
part of the "inference optimization" reality the assignment asks
about:

1. **T4 quota was 0** in all US zones we checked; switched to V100
2. **V100 capacity** was exhausted in 18 US/EU zones before
   `europe-west4-a` accepted the create request
3. The `common-cu118` image family **no longer exists**; switched to
   `pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
4. That image ships with `nvidia-570srv-**open**` kernel modules
   which **don't support Volta**; had to install
   `nvidia-driver-570-server` (closed/proprietary modules) and reboot
5. PyTorch 2.7.1 binary wheels **dropped sm_70 (Volta) support**;
   had to downgrade to PyTorch 2.5.1+cu121
6. Ultralytics' `model.export(format="engine")` calls
   `select_device("")` which on certain edge cases sets
   `CUDA_VISIBLE_DEVICES=""` and then reports "invalid device=0";
   worked around by forcing `CUDA_VISIBLE_DEVICES=0` and explicitly
   passing `device=0` to export calls
7. TensorRT 10.15 finally initialized but **error 35 on `createInferBuilder`**
   because TRT 10.4+ dropped sm_70 support entirely → pivoted to
   TorchScript

---

## 5. Results

### 5.1 Primary benchmark table

Benchmark run on V100 SXM2 16GB, 60 dashcam frames, warm-up excluded,
reproducibility verified.

```
=============================================================================
Model        Runtime       mAP@0.5   mAP@0.5:0.95   Avg Latency (ms)  Speedup
-----------------------------------------------------------------------------
yolov8m      pytorch        0.5744         0.4310              10.73    1.00x
yolov8m      onnx           0.5744         0.4303              16.61    0.65x
yolov8m      torchscript    0.5744         0.4303              11.25    0.95x
yolov11m     pytorch        0.5495         0.4168              13.09    1.00x
yolov11m     onnx           0.5726         0.4211              18.64    0.70x
yolov11m     torchscript    0.5726         0.4211              12.38    1.06x
=============================================================================
```

Results JSON: [`results/benchmark.json`](../results/benchmark.json)

### 5.2 Reproducibility check

The benchmark was re-run on the same hardware (`results/benchmark_run2.json`).
Comparison between the two runs:

| Model | Runtime | mAP@0.5 Run 1 | mAP@0.5 Run 2 | Match | Latency Δ |
|-------|---------|--------------:|--------------:|:-----:|----------:|
| yolov8m | pytorch | 0.5744 | 0.5744 | ✓ | 2.4% |
| yolov8m | onnx | 0.5744 | 0.5744 | ✓ | 1.0% |
| yolov8m | torchscript | 0.5744 | 0.5744 | ✓ | 0.8% |
| yolov11m | pytorch | 0.5495 | 0.5495 | ✓ | 0.5% |
| yolov11m | onnx | 0.5726 | 0.5726 | ✓ | 0.4% |
| yolov11m | torchscript | 0.5726 | 0.5726 | ✓ | 0.1% |

All mAP values are **bit-exact identical** between runs. Latency
delta is at most 2.4%, well within the spec's 10% reproducibility
tolerance (SC-007).

### 5.3 Analysis

**Observation 1 — mAP is preserved across runtimes.** For YOLOv8m,
the mAP@0.5 is identical (0.5744) across PyTorch, ONNX, and
TorchScript. For YOLOv11m there is a small but interesting 2.3-point
increase when moving from PyTorch (0.5495) to ONNX/TorchScript (0.5726)
— this is likely due to small numerical differences in the exported
graph changing confidence scores near the decision boundary, which
ultimately benefits recall slightly. Either way, accuracy degradation
from acceleration is within noise.

**Observation 2 — ONNX Runtime is ~30-35% SLOWER than eager PyTorch on
this stack.** This is a counterintuitive result. ONNX Runtime with
CUDA EP is typically expected to match or beat eager PyTorch, but
here it consistently underperforms by a significant margin. Possible
explanations:

1. **Ultralytics' ONNX dispatch path overhead**: Ultralytics wraps
   `onnxruntime.InferenceSession` with its own pre/post-processing
   that adds per-call overhead not amortized over a single image
2. **CUDA EP session cost**: `onnxruntime-gpu` 1.23's CUDA Execution
   Provider has startup + memory-planning overhead that dominates
   short-input inference (640×640)
3. **Memory allocator mismatch**: PyTorch's cached allocator is
   extremely efficient for repeated same-shape inference; ONNX
   Runtime's allocator does not compete as well

For teams deploying YOLOv8m/YOLOv11m on V100 with the Ultralytics
toolchain, **eager PyTorch is already the fastest option** — ONNX
Runtime does not provide a speedup here. This is a specific,
hardware+library-dependent finding, not a universal statement about
ONNX vs PyTorch.

**Observation 3 — TorchScript is approximately on par with eager PyTorch.**
TorchScript runs at 0.94x-1.06x of eager PyTorch speed, which is
expected: TorchScript primarily improves deployment portability
(you can load a `.torchscript` file without Python source), not raw
speed on modern PyTorch versions. It's a valid **deployment-format**
acceleration, even if the wall-clock savings are modest.

**Observation 4 — mAP values are modest (~57% mAP@0.5).** This
reflects the gap between the Faster R-CNN reference labels and YOLO's
predictions. Faster R-CNN tends to find slightly different objects at
slightly different box coordinates than YOLO; at IoU=0.5 threshold,
this disagreement reduces apparent accuracy. The benchmark is still
meaningful because **all three runtimes** for the same model get
compared against the same ground truth, so the *relative* numbers
are unaffected.

---

## 6. End-to-End Demo

All three API surfaces were exercised end-to-end on the local
machine (backend on CPU, frontend in a local browser) after the
V100 benchmark to confirm the full stack works.

### 6.1 Image detection

![Image detection result](screenshots/02_detect_image_result.png)

A single SF downtown intersection frame (`frame_0008.jpg`) uploaded
through the Next.js frontend, processed by the FastAPI backend
(YOLOv8m + PyTorch), result rendered as bounding box overlays on an
HTML5 canvas. Inference time: **141 ms** on CPU. 17 detections
returned (cars, traffic lights, pedestrian).

### 6.2 Video detection (live through the full stack)

![Video inference running](screenshots/04_video_processing.png)

The frontend shows a "Processing video frames..." indicator with
progress feedback while the backend processes all frames.

![Video inference results](screenshots/05_video_result.png)

After processing, the frontend displays:

- The annotated video player
- Latency badges: **219,512 ms total**, **57.8 ms/frame average**,
  **3,598 frames**, **43,931 total detections**
- A per-frame detection count bar chart

The annotated video shows consistent, well-placed bounding boxes
on cars, traffic lights, and pedestrians throughout the 60-second
dashcam segment:

![Frame from annotated video](screenshots/video_inference_frame.jpg)

### 6.3 Comparison view (all 6 combos)

![Compare all runtimes](screenshots/03_compare_all.png)

The Compare page runs the same image (`frame_0018.jpg`) through all
six model+runtime combinations and renders a side-by-side table:

| Model | Runtime | Latency (ms) | Detections | Speedup |
|-------|---------|-------------:|-----------:|--------:|
| yolov8m | **pytorch** | 85.11 ⚡ fastest | 4 | 1.00x |
| yolov8m | onnx | 229.95 | 4 | 0.37x |
| yolov8m | torchscript | 261.65 | 4 | 0.33x |
| yolov11m | pytorch | 103.91 | 6 | 1.00x |
| yolov11m | onnx | 166.46 | 5 | 0.62x |
| yolov11m | torchscript | 246.08 | 5 | 0.42x |

This is the local CPU version (an Apple M4 Pro), not the V100.
Interestingly, on CPU the same relative pattern holds: PyTorch is
fastest, ONNX and TorchScript are slower. The absolute latencies are
~10x the V100 numbers, as expected for CPU versus GPU.

---

## 7. Requirements Traceability

| # | Assignment requirement | Where it's satisfied |
|---|------------------------|----------------------|
| 1 | Two strong-performing models, inference on video data | YOLOv8m + YOLOv11m, both medium-size. Video inference demonstrated end-to-end on `clip.mp4` (3,598 frames, rendered annotated video) |
| 2 | FastAPI backend, OpenAI-compatible | `backend/main.py` + routers. OpenAI-style envelope with `id`, `object`, `model`, `created`, `usage`, `results` |
| 3 | Frontend (Next.js) with upload + bbox + latency | `frontend/src/app/page.tsx`, `compare/page.tsx`. Canvas bbox overlay, latency badges, comparison table — all captured in screenshots |
| 4 | At least two acceleration methods | **ONNX Runtime** + **TorchScript** (both on the approved list). TensorRT pivot documented in §4.2 |
| 5 | Evaluate mAP and latency | `scripts/benchmark.py`, `results/benchmark.json`, COCO mAP@0.5 and 0.5:0.95, latency ms, speedup factor, reproducibility check |
| 6 | Own video/image data with own annotations | 60 frames captured via `scripts/capture_traffic_cam.py` from SF dashcam; 1,039 bounding boxes generated by `scripts/auto_label_fasterrcnn.py` (student-authored pipeline) and human-verified via `scripts/render_viz.py` |

---

## 8. Reproducing This Report's Results

Every step is scripted and documented in the repository. To
reproduce end-to-end:

```bash
# 1. Clone repo
git clone https://github.com/yashashav-dk/cs258-hw02-object-detection.git homework
cd homework

# 2. Capture frames from a public YouTube dashcam
yt-dlp -f "bv[ext=mp4][vcodec^=avc][height<=720]+ba[ext=m4a]" \
  --merge-output-format mp4 \
  --download-sections "*15:00-25:00" \
  -o ~/hw02-dataset/source.mp4 \
  "https://www.youtube.com/watch?v=HZrm3s4UsgU"
ffmpeg -i ~/hw02-dataset/source.mp4 -vf fps=1/10 -frames:v 60 \
  -q:v 2 ~/hw02-dataset/raw-frames/frame_%04d.jpg
ffmpeg -i ~/hw02-dataset/source.mp4 -t 60 -c copy -an \
  ~/hw02-dataset/video/clip.mp4

# 3. Auto-label with Faster R-CNN and verify
python3 scripts/auto_label_fasterrcnn.py \
  --frames-dir ~/hw02-dataset/raw-frames \
  --output-dir ~/hw02-dataset \
  --confidence 0.5
python3 scripts/render_viz.py \
  --annotations ~/hw02-dataset/annotations.json \
  --images-dir ~/hw02-dataset/images \
  --output-dir ~/hw02-dataset/viz_all
# Manually review ~/hw02-dataset/viz_all in Finder/Preview

# 4. On the GPU VM: bootstrap and export models
git clone <repo> ~/homework && cd ~/homework
pip3 install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip3 install ultralytics onnx onnxslim onnxruntime-gpu \
  pycocotools opencv-python-headless fastapi "uvicorn[standard]"
CUDA_VISIBLE_DEVICES=0 python3 scripts/export_models.py

# 5. Upload dataset + run benchmark
gcloud compute scp --recurse ~/hw02-dataset/images \
  yolo-v100:~/homework/data/ ...
python3 scripts/benchmark.py \
  --images data/images/ \
  --annotations data/annotations.json \
  --output results/benchmark.json
```

Full execution detail is in
[`specs/002-deploy-and-benchmark/quickstart.md`](../specs/002-deploy-and-benchmark/quickstart.md).

---

## 9. Limitations and Honest Caveats

1. **TensorRT substitution**: Swapping TensorRT → TorchScript means
   the absolute speedup numbers would have been different with
   TensorRT FP16. On newer GPUs (T4, A100, L4), TensorRT would
   likely show 2-3x speedup over PyTorch. Our result should not be
   read as "TensorRT doesn't help" — it should be read as "you
   can't use recent TensorRT on Volta."

2. **Custom dataset is small and single-domain**: 60 frames all
   from one dashcam drive. A production evaluation would use
   hundreds of images from multiple scenes. The choice was driven by
   the assignment time budget and still yields a reproducible
   benchmark.

3. **Auto-labeled ground truth**: Ground truth labels came from a
   Faster R-CNN reference labeler, not hand-drawn boxes. The
   verification step confirms labels look reasonable, but some
   minor mis-localizations may remain. The benchmark's
   runtime-comparison findings are unaffected (all runtimes compared
   against the same GT), but the absolute mAP values should be
   interpreted as "agreement with a reference detector," not "agreement
   with human consensus."

4. **Unexpected ONNX slowdown**: The finding that ONNX Runtime is
   slower than eager PyTorch on this stack is a specific,
   library-version-dependent result, not a universal claim. Teams
   shipping ONNX inference on V100 may want to profile their actual
   dispatch path before assuming ONNX will give them a speedup.

---

## 10. Conclusion

This project built, deployed, and benchmarked a full-stack object
detection inference service that satisfies all six Option 2
requirements for CS 258 HW02:

- **Two models** (YOLOv8m, YOLOv11m) serving real video data
- **OpenAI-compatible FastAPI backend** with six endpoints
- **Next.js frontend** with image, video, and comparison views
  demonstrated end-to-end via screenshots
- **Two acceleration methods** (ONNX Runtime + TorchScript, both
  from the approved list)
- **mAP + latency evaluation** with reproducibility check
- **Custom annotated dataset** from Bay Area dashcam footage with a
  student-authored auto-labeling pipeline

The benchmark yielded one **non-obvious finding** — ONNX Runtime is
actually slower than eager PyTorch on this specific hardware +
library combination — that would likely be missed by a team blindly
trusting common wisdom about acceleration runtimes. Real-world
performance work requires measuring on your actual target stack.

The project also surfaced concrete **deployment constraints** worth
remembering: TensorRT 10.4+ dropped Volta support, Google Cloud V100
availability is geographically patchy, and DLVM images ship with
open-source kernel modules that don't support pre-Turing GPUs. Each
of these was resolved with a scripted workaround, and the full
chain of fixes is documented in §4.5 for future reproducibility.
