# Feature Specification: Deploy and Benchmark

**Feature Branch**: `002-deploy-and-benchmark`
**Created**: 2026-04-10
**Status**: Draft
**Input**: User description: "Provision GCP V100 VM, bootstrap environment, clone and deploy backend, annotate custom dataset, export models, run benchmark, capture results for submission"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Provision GPU Compute Environment (Priority: P1)

The student provisions a cloud VM with an NVIDIA V100 GPU, CUDA, and PyTorch pre-installed, so backend inference and model export can run on GPU-accelerated hardware.

**Why this priority**: Every subsequent step (model export, backend serving, benchmarking) depends on having a working GPU environment. Without this, nothing else can execute.

**Independent Test**: SSH into the provisioned VM, run `nvidia-smi`, confirm V100 is visible and CUDA is available.

**Acceptance Scenarios**:

1. **Given** the student has gcloud CLI configured with an authenticated account and project, **When** the provisioning command is executed, **Then** a VM with V100 GPU becomes reachable via SSH and `nvidia-smi` reports the V100 with CUDA drivers.
2. **Given** a provisioned VM, **When** the student disconnects and reconnects via SSH, **Then** GPU state and installed software persist across sessions.

---

### User Story 2 - Bootstrap Backend Environment (Priority: P1)

The student clones the project repository onto the VM, installs Python dependencies, installs TensorRT, and exports YOLOv8m and YOLOv11m to ONNX and TensorRT formats so the backend can serve all 6 model+runtime combinations.

**Why this priority**: The FastAPI backend and benchmark script both require the exported model files (`.onnx`, `.engine`) to function. Without the bootstrap, no inference can run.

**Independent Test**: After bootstrap, run `ls *.pt *.onnx *.engine` on the VM and confirm 6 model files exist (2 models × 3 formats).

**Acceptance Scenarios**:

1. **Given** a fresh VM with CUDA installed, **When** the student runs the documented bootstrap steps, **Then** all Python dependencies install without error and the model export script completes successfully, producing 6 model files.
2. **Given** the exported model files exist, **When** the student starts the FastAPI backend, **Then** the `/v1/models` endpoint reports all 6 model+runtime combinations as available.
3. **Given** a running backend, **When** the student calls `/health` from the VM or from the local frontend, **Then** it returns GPU details including V100 identification and memory usage.

---

### User Story 3 - Annotate Custom Dataset (Priority: P1)

The student collects images and a short video clip, annotates ground truth bounding boxes in COCO JSON format, and places the annotated dataset at the expected paths so the benchmark script can evaluate mAP against it.

**Why this priority**: The assignment explicitly requires custom annotations, not downloaded datasets. This is a core rubric requirement.

**Independent Test**: Open `data/annotations.json`, verify it contains at least 50 annotated images with bounding boxes in COCO format, and that `pycocotools.coco.COCO` can load it without error.

**Acceptance Scenarios**:

1. **Given** 50+ captured images and 1 short video clip (30-60s), **When** the student annotates them using Label Studio or CVAT, **Then** the export is a valid COCO JSON file loadable by pycocotools.
2. **Given** a valid annotations file, **When** the benchmark script loads it, **Then** it enumerates image IDs and reads corresponding image files from `data/images/` without errors.

---

### User Story 4 - Execute Benchmark and Capture Results (Priority: P1)

The student runs the benchmark script against the annotated dataset on the V100 VM, producing a complete results table showing COCO mAP and latency for all 6 model+runtime combinations, and saves the output for inclusion in the submission report.

**Why this priority**: This produces the final deliverable data for the assignment — the table comparing accuracy and speed across runtimes is the core outcome of Option 2.

**Independent Test**: After running `python scripts/benchmark.py --images data/images/ --annotations data/annotations.json --output results/benchmark.json`, verify `results/benchmark.json` exists with 6 entries, each containing mAP@0.5, mAP@0.5:0.95, avg_latency_ms, and speedup.

**Acceptance Scenarios**:

1. **Given** all 6 model files exist and the dataset is annotated, **When** the benchmark script runs, **Then** it outputs a complete table (printed and saved as JSON) with all 6 model+runtime combinations populated.
2. **Given** the benchmark results, **When** the student re-runs the script, **Then** mAP values are identical and latency is within 10% of the first run.
3. **Given** the saved `results/benchmark.json`, **When** `python scripts/benchmark.py --print-table results/benchmark.json` is executed, **Then** the same formatted table is reprinted without re-running inference.

---

### Edge Cases

- What happens if GPU quota on the GCP project is insufficient for a V100?
  The student MUST request quota increase or choose an alternative zone; provisioning MUST fail clearly if no V100 is available.
- What happens if TensorRT installation fails on the VM?
  The bootstrap MUST report a clear error and the student MUST be able to fall back to exporting only ONNX (with 4 combos instead of 6) and continue benchmarking.
- What happens if a COCO annotation file references an image that does not exist in `data/images/`?
  The benchmark script MUST skip missing images and report them as warnings without crashing.
- What happens when running the benchmark twice without restarting (model already loaded)?
  The benchmark script MUST always reload models cleanly to ensure reproducibility (no stale state).
- What happens when the VM runs out of GPU memory during benchmarking?
  The benchmark script MUST catch OOM errors, mark that combination as failed, and continue with the remaining combinations.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The student MUST be able to provision a GPU-enabled VM with a single documented command using gcloud CLI, targeting an NVIDIA V100 GPU.
- **FR-002**: The VM MUST have CUDA 11.8+, cuDNN 8+, and Python 3.11+ available (either pre-installed via a Deep Learning VM image, or installed via bootstrap).
- **FR-003**: Bootstrap MUST include: git clone of the public repo, Python venv creation, pip install of backend requirements, pip install of tensorrt, and execution of `scripts/export_models.py`.
- **FR-004**: Bootstrap MUST produce 6 model files: `yolov8m.{pt,onnx,engine}` and `yolo11m.{pt,onnx,engine}`.
- **FR-005**: The student MUST capture and annotate at least 50 images and at least 1 video clip (30-60 seconds) using Label Studio or CVAT, exporting in COCO JSON format.
- **FR-006**: The annotated dataset MUST be placed at `data/images/`, `data/video/`, and `data/annotations.json` on the VM.
- **FR-007**: The benchmark script MUST execute end-to-end on the VM and produce a complete 6-row results table.
- **FR-008**: Results MUST be saved to `results/benchmark.json` for later retrieval and print-only replay.
- **FR-009**: The VM MUST be reachable for SSH and the backend MUST be startable on port 8000 with the external IP accessible to the local frontend (or explicitly tunneled via SSH).
- **FR-010**: All steps MUST be documented in the repository so the entire pipeline can be reproduced by another user given the same GCP project access.

### Key Entities

- **CloudVM**: Name, zone, machine type, GPU type, boot disk image, external IP, SSH keys.
- **BootstrapState**: Repo cloned, dependencies installed, TensorRT installed, models exported.
- **Dataset**: Image count, video count, annotation file path, COCO category list.
- **BenchmarkRun**: Timestamp, environment details (GPU name, driver, CUDA version), per-combo results, total runtime.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: GPU-enabled VM is provisioned and SSH-accessible within 10 minutes of running the provisioning command.
- **SC-002**: Bootstrap completes end-to-end (clone → install → export) in under 30 minutes on a fresh VM.
- **SC-003**: All 6 model files are successfully produced by the export step; if TensorRT fails, at least the 4 PyTorch+ONNX combinations succeed.
- **SC-004**: Custom dataset contains at least 50 annotated images and 1 annotated video clip, validated by pycocotools loading without errors.
- **SC-005**: Benchmark script produces a complete results table with mAP@0.5, mAP@0.5:0.95, avg latency, and speedup for all 6 combos in a single run.
- **SC-006**: TensorRT latency is at least 1.5x faster than PyTorch baseline; ONNX Runtime is at least 1.2x faster (matching HW02 spec SC-002/SC-003 from feature 001).
- **SC-007**: Benchmark is reproducible — re-running produces identical mAP values and latency within 10% variance.
- **SC-008**: All commands required to reproduce the full deployment and benchmark are documented in a runbook (`docs/runbook.md` or README section).

## Assumptions

- The student has an active Google Cloud project (`cudabenchmarking`) with billing enabled and sufficient GPU quota for a V100 instance in `us-west1-b` (1 V100 quota available; T4 quota was 0, so V100 was selected as a strictly better available alternative).
- gcloud CLI is installed and authenticated on the student's local machine with the correct project and zone configured.
- The project repository (`yashashav-dk/cs258-hw02-object-detection`) is public and accessible via HTTPS clone.
- Label Studio will be used for annotation on the student's local machine (not the VM), and the annotated export will be transferred to the VM via `gcloud compute scp` or committed to git.
- Images/videos for annotation will be captured by the student (phone photos, screen recordings, or public-domain footage) and will contain common COCO classes.
- The V100 has sufficient VRAM (16GB HBM2) to load any single YOLOv8m or YOLOv11m model+runtime, with the single-model memory strategy enforced by the backend.
- Network connectivity between the local frontend and the GCP VM is available either via the VM's external IP or via an SSH tunnel.
