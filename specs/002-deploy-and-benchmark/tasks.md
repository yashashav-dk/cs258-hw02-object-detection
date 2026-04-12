# Tasks: Deploy and Benchmark

**Input**: Design documents from `/specs/002-deploy-and-benchmark/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/commands.md, quickstart.md

**Tests**: Not applicable — this is a deployment runbook, not source code. Verification happens at checkpoint steps within each phase.

**Organization**: Tasks grouped by user story. Each story is a distinct phase of the deployment pipeline (provision → bootstrap → annotate → benchmark).

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files/processes, no dependencies)
- **[Story]**: US1 (Provision VM), US2 (Bootstrap), US3 (Annotate), US4 (Benchmark)

---

## Phase 1: Setup

**Purpose**: Verify local prerequisites before touching cloud infrastructure

- [x] T001 Verify gcloud CLI is authenticated and active project is `cudabenchmarking` by running `gcloud config list` on local machine
- [x] T002 Verify GPU quota for NVIDIA V100 in `us-west1-b` by running `gcloud compute regions describe us-west1 --project=cudabenchmarking` and checking `NVIDIA_V100_GPUS` quota (limit ≥1). Note: original plan targeted T4 but project has 0 T4 quota; V100 was selected as a strictly better available alternative.
- [x] T003 [P] Verify the public repo is reachable at `https://github.com/yashashav-dk/cs258-hw02-object-detection.git`
- [ ] T004 [P] Install Label Studio locally for annotation: `pip install label-studio` on local machine

---

## Phase 2: Foundational

**Purpose**: None — all infrastructure is created within user story phases. Phase 2 is skipped because there are no blocking prerequisites shared across all user stories (US1 must run first, but that's a story ordering concern, not a foundational concern).

---

## Phase 3: User Story 1 — Provision GPU Compute Environment (Priority: P1)

**Goal**: Create a running V100 GPU VM accessible via SSH
**Independent Test**: SSH into VM, run `nvidia-smi`, confirm V100 is visible with CUDA drivers active

### Implementation for User Story 1

- [ ] T005 [US1] Provision V100 VM by running the `gcloud compute instances create yolo-v100` command from `specs/002-deploy-and-benchmark/contracts/commands.md` C1 section on local machine
- [ ] T006 [US1] Wait 60-90 seconds for VM boot to complete, then SSH with port forwarding using `gcloud compute ssh yolo-v100 --project=cudabenchmarking --zone=us-west1-b -- -L 8000:localhost:8000` on local machine
- [ ] T007 [US1] On first SSH, accept NVIDIA driver installation prompt if shown (driver installs on first boot via `install-nvidia-driver=True` metadata)
- [ ] T008 [US1] Verify GPU is active on VM: run `nvidia-smi` and confirm output shows `Tesla V100` with CUDA version 11.8+
- [ ] T009 [US1] Verify Python 3.11+ is available on VM: run `python3 --version`

**Checkpoint**: VM provisioned, SSH working, V100 + CUDA verified. Ready for bootstrap.

---

## Phase 4: User Story 2 — Bootstrap Backend Environment (Priority: P1)

**Goal**: Clone repo, install deps, install TensorRT, export all 6 model files
**Independent Test**: Run `ls *.pt *.onnx *.engine` on VM, confirm 6 model files exist

### Implementation for User Story 2

- [ ] T010 [US2] Clone repo on VM: `cd ~ && git clone https://github.com/yashashav-dk/cs258-hw02-object-detection.git homework && cd ~/homework`
- [ ] T011 [US2] Create Python virtualenv on VM: `python3 -m venv venv && source venv/bin/activate`
- [ ] T012 [US2] Upgrade pip and install backend dependencies on VM: `pip install --upgrade pip && pip install -r backend/requirements.txt`
- [ ] T013 [US2] Install TensorRT on VM: `pip install tensorrt` (required for `.engine` export)
- [ ] T014 [US2] Run model export script on VM: `python scripts/export_models.py` (downloads YOLOv8m and YOLOv11m weights and exports each to ONNX and TensorRT)
- [ ] T015 [US2] Verify all 6 model files exist on VM: `ls yolov8m.pt yolov8m.onnx yolov8m.engine yolo11m.pt yolo11m.onnx yolo11m.engine` — expect all 6 to be listed
- [ ] T016 [US2] (Optional smoke test) Start backend briefly: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`, then in a second terminal on local machine run `curl http://localhost:8000/health` (SSH tunnel is active) — verify GPU info is returned. Stop the backend with Ctrl+C afterwards.

**Checkpoint**: Backend environment fully provisioned. All 6 model files exist. Backend starts successfully. Ready for dataset preparation.

---

## Phase 5: User Story 3 — Annotate Custom Dataset (Priority: P1)

**Goal**: Produce a COCO JSON annotation file for 50+ images and 1 video clip, then upload to VM
**Independent Test**: Load `data/annotations.json` on VM with `python -c "from pycocotools.coco import COCO; c = COCO('data/annotations.json'); print(len(c.getImgIds()))"` — expect ≥50

### Implementation for User Story 3

- [x] T017 [US3] Capture 60 frames from Caltrans D4 traffic cam (SR-123 @ 40th St, Emeryville) using `python3 scripts/capture_traffic_cam.py frames --url <m3u8> --output-dir ~/hw02-dataset/raw-frames --num-frames 60 --interval 10`
- [x] T018 [US3] Record 60-second video clip from the same stream using `python3 scripts/capture_traffic_cam.py video --url <m3u8> --output ~/hw02-dataset/video/clip.mp4 --duration 60`
- [ ] T019 [US3] Upload captured frames to Roboflow: project `hw02-object-detection`, drag-and-drop from `~/hw02-dataset/raw-frames/`
- [ ] T020 [US3] In Roboflow, create bounding-box annotations using these exact COCO-standard class names: `person`, `car`, `bicycle`, `motorcycle`, `bus`, `truck`, `traffic light`
- [ ] T021 [US3] Annotate all 50+ frames with bounding boxes (3-8 boxes per frame typical; skip tiny/occluded objects)
- [ ] T022 [US3] Generate dataset version in Roboflow (preprocessing defaults, augmentations OFF) and export as COCO JSON → `~/Downloads/hw02-object-detection.v1i.coco.zip`; unzip to `~/Downloads/hw02-object-detection.v1i.coco/`
- [ ] T022a [US3] Merge Roboflow export into flat dataset with COCO-standard category IDs: `python3 scripts/merge_roboflow_export.py --roboflow-dir ~/Downloads/hw02-object-detection.v1i.coco --output-dir ~/hw02-dataset`
- [ ] T023 [US3] Upload images to VM: `gcloud compute scp --recurse ~/hw02-dataset/images yolo-v100:~/homework/data/ --project=cudabenchmarking --zone=us-west1-b` on local machine (the merge script writes annotated frames to `~/hw02-dataset/images/`)
- [ ] T024 [US3] [P] Upload video to VM: `gcloud compute scp --recurse ~/hw02-dataset/video yolo-v100:~/homework/data/ --project=cudabenchmarking --zone=us-west1-b` on local machine
- [ ] T025 [US3] [P] Upload annotations to VM: `gcloud compute scp ~/hw02-dataset/annotations.json yolo-v100:~/homework/data/annotations.json --project=cudabenchmarking --zone=us-west1-b` on local machine
- [ ] T026 [US3] Validate annotations on VM: `cd ~/homework && source venv/bin/activate && python -c "from pycocotools.coco import COCO; c = COCO('data/annotations.json'); print(f'{len(c.getImgIds())} images, {len(c.getAnnIds())} annotations')"` — expect ≥50 images

**Checkpoint**: Dataset ready on VM. Ready to run benchmark.

---

## Phase 6: User Story 4 — Execute Benchmark and Capture Results (Priority: P1)

**Goal**: Run `scripts/benchmark.py`, produce `results/benchmark.json` with 6 rows, retrieve locally, commit to repo
**Independent Test**: Open `results/benchmark.json` locally, verify 6 entries each with `map_50`, `map_50_95`, `avg_latency_ms`, and `speedup`

### Implementation for User Story 4

- [ ] T027 [US4] Run benchmark on VM: `cd ~/homework && source venv/bin/activate && python scripts/benchmark.py --images data/images/ --annotations data/annotations.json --output results/benchmark.json`
- [ ] T028 [US4] Capture stdout of benchmark run for the submission report (the formatted results table)
- [ ] T029 [US4] Verify `results/benchmark.json` exists on VM and contains 6 entries: `cat results/benchmark.json | python -m json.tool | head -40`
- [ ] T030 [US4] (Reproducibility check) Re-run benchmark a second time and verify mAP values are identical and latency is within 10% of the first run (addresses SC-007)
- [ ] T031 [US4] Retrieve results to local machine: `gcloud compute scp yolo-v100:~/homework/results/benchmark.json ./results/benchmark.json --project=cudabenchmarking --zone=us-west1-b`
- [ ] T032 [US4] Update `.gitignore` locally to exclude `results/*` but keep `!results/benchmark.json` committable
- [ ] T033 [US4] Commit results locally: `git add -f results/benchmark.json && git commit -m "results: benchmark output from V100 VM"` (no Claude co-author)
- [ ] T034 [US4] Push to GitHub: `git push`

**Checkpoint**: Benchmark complete. Results committed to public repo. Ready for tear down.

---

## Phase 7: Polish & Tear Down

**Purpose**: Commit annotated dataset annotations, document results, delete VM to stop billing

- [ ] T035 [P] Commit `data/annotations.json` to the repo (overwriting placeholder): `git add data/annotations.json && git commit -m "data: add custom COCO annotations for benchmark dataset"` on local machine
- [ ] T036 [P] Add a results section to `README.md` referencing `results/benchmark.json` and summarizing the observed speedups and mAP values for the submission
- [ ] T037 Commit README update: `git add README.md && git commit -m "docs: add benchmark results summary to README"` on local machine
- [ ] T038 Push all commits: `git push`
- [ ] T039 Tear down VM to stop billing: `gcloud compute instances delete yolo-v100 --project=cudabenchmarking --zone=us-west1-b --quiet` on local machine
- [ ] T040 Verify VM is deleted: `gcloud compute instances list --project=cudabenchmarking --zones=us-west1-b` should not show `yolo-v100`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **US1 (Phase 3)**: Depends on Phase 1 (quota verified)
- **US2 (Phase 4)**: Depends on US1 (needs SSH access to VM)
- **US3 (Phase 5)**: Can start in parallel with US1/US2 locally — annotation is independent of VM state. Upload tasks (T023-T025) depend on US2 (VM has `~/homework/data/` directory)
- **US4 (Phase 6)**: Depends on US2 (model files) and US3 (dataset on VM)
- **Polish (Phase 7)**: Depends on US4 (results retrieved)

### Within Each User Story

- Sequential by default — each command's post-condition is the next command's precondition
- Parallel opportunities limited to independent scp uploads (T023/T024/T025) and final polish tasks (T035/T036)

### Parallel Opportunities

- **Time-saving parallel path**: While the VM is provisioning (T005-T009) and bootstrapping (T010-T015) — ~30-40 minutes total — the student can simultaneously annotate the dataset (T017-T022) on their local machine. This cuts total wall-clock time significantly.
- T023 + T024 + T025: scp uploads can run in parallel (different terminals)
- T035 + T036: git commits for annotations and README can be prepared in parallel

---

## Implementation Strategy

### Critical Path (Fastest Time to Results)

1. **Start annotation (T017-T022) in background locally** — this is the longest step (~2-3 hours)
2. In parallel, provision VM (T005-T009) and bootstrap (T010-T015) — ~30-40 minutes
3. Once both complete: upload dataset (T023-T026)
4. Run benchmark (T027-T030)
5. Retrieve and commit (T031-T034)
6. Tear down (T039-T040)

### MVP Path (Minimum to Submit)

If time-constrained, the absolute minimum submission is:
1. Complete Phase 1 (Setup)
2. Complete Phase 3 (US1 Provision)
3. Complete Phase 4 (US2 Bootstrap)
4. Complete Phase 5 (US3 Annotate) — even with fewer than 50 images if absolutely necessary, then note the limitation in the report
5. Complete Phase 6 (US4 Benchmark) — even if only 4 combos succeed (PyTorch + ONNX for both models), partial results are better than none
6. Tear down immediately (T039)

### Cost Management

- Total expected VM runtime: 1-3 hours
- Cost estimate: $0.75 - $2.50
- **CRITICAL**: T039 (VM deletion) MUST run before you walk away. Set a phone alarm if needed.

---

## Notes

- No new source code is written in this feature — feature 001 supplies everything. These tasks are pure deployment/operations.
- Each task in Phase 3-6 maps directly to a command in `contracts/commands.md` — refer to that file for exact copy-paste commands.
- `quickstart.md` is the hands-on runbook version of these tasks.
- Task T032 is a deliberate `.gitignore` override because `results/` is gitignored by default, but we want `results/benchmark.json` committed as the assignment deliverable.
- No Claude co-author in any commit message (user explicitly requested this).
