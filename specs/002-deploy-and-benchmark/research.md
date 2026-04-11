# Research: Deploy and Benchmark

**Date**: 2026-04-10

## R1: GCP Deep Learning VM Image

**Decision**: Use `common-cu118` image family from
`deeplearning-platform-release` project.

**Rationale**: Current CUDA 11.8 base DLVM image with NVIDIA driver
R525+ pre-installed (activates on first boot via
`install-nvidia-driver=True` metadata). Compatible with the
`nvidia-tensorrt` pip wheel (TRT 8.6) and Ultralytics out of the box.
PyTorch is not pre-installed but `pip install` works immediately.

**Alternatives considered**:
- `pytorch-1-13-cu113`: has PyTorch pre-installed but CUDA 11.3 is
  too old for current TensorRT.
- Manual Ubuntu 22.04 + custom CUDA install: much more setup work.

## R2: Machine Type

**Decision**: `n1-standard-8` (8 vCPU, 30 GB RAM) with 1x V100.

**Rationale**: `n1-standard-4` is the minimum but TensorRT engine
builds for YOLOv8m/YOLOv11m are memory-hungry and OpenCV video decode
is CPU-bound. n1-standard-8 is only ~$0.10/hr more and avoids OOM
during engine export. V100 is only attachable to N1 family.

**Cost**: Approximately $2.86/hr on-demand ($2.48 V100 + $0.38 n1-standard-8). For a 2-3 hour benchmark run, total cost is ~$6-9.

**GPU choice note**: The original plan targeted NVIDIA T4, but T4 quota is 0 in the `cudabenchmarking` project across all US regions. V100 quota is 1 in `us-west1`, so V100 was selected as a strictly better available alternative (faster Tensor Cores, 16GB HBM2 vs 16GB GDDR6, better FP16 performance for TensorRT). The single-model memory strategy still applies (16GB VRAM).

**Alternatives considered**:
- `n1-standard-4`: risk of OOM during TensorRT export.
- `n1-highmem-4`: more RAM but fewer cores hurts video decode.

## R3: Provisioning Command

**Decision**: Single `gcloud compute instances create` with explicit
V100 accelerator, `common-cu118` image, 50GB pd-balanced boot disk,
TERMINATE maintenance policy (required for GPU), and
`install-nvidia-driver=True` metadata.

**Rationale**: TERMINATE is mandatory for GPU instances (no live
migration). External ephemeral IP is attached by default. 50GB is
enough for CUDA + PyTorch + Ultralytics + 6 exported models + dataset
with headroom.

## R4: Firewall for Port 8000

**Decision**: Use **SSH port forwarding** instead of opening port 8000
to the internet.

**Rationale**: The default GCP VPC only opens 22/3389/ICMP. Port 8000
is blocked inbound. For a short-lived student instance, SSH tunneling
is safer and simpler than managing a firewall rule.

The tunnel command: `gcloud compute ssh yolo-v100 --zone=us-west1-b -- -L 8000:localhost:8000`

After this, the local Next.js frontend points at `http://localhost:8000`
(no external IP needed).

**Alternatives considered**:
- `gcloud compute firewall-rules create allow-fastapi-8000 ...`:
  works but exposes the backend publicly. Only acceptable with
  source-ranges restricted to the student's home IP.

## R5: Annotation Tool Choice

**Decision**: Use **Label Studio** locally on the student's machine,
then transfer the annotated export to the VM.

**Rationale**: Label Studio has a mature COCO export and runs
standalone via `pip install label-studio && label-studio start`.
Running it locally keeps the GPU VM focused on inference, avoids
installing UI tooling on the server, and the export is a single JSON
file easy to transfer.

**Alternatives considered**:
- CVAT: more features but heavier setup (Docker Compose).
- Running Label Studio on the VM: wastes GPU time and adds port
  forwarding complexity.

## R6: Dataset Size and Content

**Decision**: Minimum 50 images + 1 video clip (30-60s), capturing
common COCO classes (person, car, chair, bottle, laptop, etc.).

**Rationale**: 50 images is the assignment-required minimum. Using
COCO classes means the pre-trained weights apply directly without
fine-tuning. Student's phone camera is sufficient.

**Alternatives considered**:
- 100+ images: more robust mAP but exceeds homework scope.
- Custom classes requiring fine-tuning: out of Option 2 scope
  (Option 2 is inference optimization, not training).
