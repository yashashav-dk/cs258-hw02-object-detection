# Tasks: Inference Optimization Pipeline

**Input**: Design documents from `/specs/001-inference-optimization/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/api.md

**Tests**: Not explicitly requested in the spec. Test tasks are omitted.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story (US1, US2, US3, US4)
- Exact file paths included in descriptions

## Phase 1: Setup

**Purpose**: Project initialization and structure

- [x] T001 Create project directory structure: `backend/`, `backend/routers/`, `backend/services/`, `backend/schemas/`, `frontend/`, `scripts/`, `data/images/`, `data/video/`
- [x] T002 Initialize Python backend with `backend/requirements.txt` (fastapi, uvicorn, ultralytics, onnxruntime-gpu, tensorrt, pycocotools, opencv-python, python-multipart)
- [x] T003 [P] Initialize Next.js frontend in `frontend/` with TypeScript, Tailwind CSS
- [x] T004 [P] Create `.gitignore` excluding `data/images/`, `data/video/`, `*.pt`, `*.onnx`, `*.engine`, `__pycache__/`, `node_modules/`, `.next/`
- [x] T005 [P] Create `scripts/export_models.py` skeleton (downloads YOLOv8m + YOLOv11m weights, exports to ONNX and TensorRT)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core backend infrastructure that MUST be complete before ANY user story

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create Pydantic request schemas in `backend/schemas/request.py` (DetectionRequest with model enum, runtime enum, input_type enum; file size validation constants)
- [x] T007 [P] Create Pydantic response schemas in `backend/schemas/response.py` (DetectionResponse, FrameResult, Detection, matching OpenAI-compatible contract from contracts/api.md)
- [x] T008 Implement ModelManager service in `backend/services/model_manager.py` (load/unload single model+runtime at a time, track loaded state, list available models with runtime status)
- [x] T009 Create FastAPI app with CORS and lifespan in `backend/main.py` (include routers, configure CORS for localhost:3000, initialize ModelManager)
- [x] T010 [P] Implement health endpoint in `backend/routers/health.py` (GPU info via torch.cuda, memory usage)
- [x] T011 [P] Implement models listing endpoint in `backend/routers/models.py` (GET /v1/models returning available model+runtime combos with loaded status)
- [x] T012 Implement `scripts/export_models.py` fully: download YOLOv8m + YOLOv11m, export each to ONNX (`model.export(format="onnx")`) and TensorRT (`model.export(format="engine")`)

**Checkpoint**: Backend skeleton running, models exportable, health + models endpoints working

---

## Phase 3: User Story 1 — Single Image Detection (Priority: P1) MVP

**Goal**: Upload image, select model+runtime, see bounding boxes + latency
**Independent Test**: Upload JPEG → select YOLOv8m+PyTorch → verify bboxes + latency displayed

### Implementation for User Story 1

- [x] T013 [US1] Implement image detection logic in `backend/services/detector.py` (run single image through loaded model, return list of Detection objects with bbox/class/confidence, measure latency)
- [x] T014 [US1] Implement `/v1/detect/upload` endpoint for image in `backend/routers/detect.py` (accept multipart file + model + runtime params, validate file type + size ≤10MB, call ModelManager to load, call detector, return OpenAI-compatible DetectionResponse)
- [x] T015 [US1] Implement `/v1/detect` base64 endpoint in `backend/routers/detect.py` (accept JSON body with base64 input, decode, reuse same detection logic)
- [x] T016 [P] [US1] Create FileUpload component in `frontend/src/components/FileUpload.tsx` (drag-and-drop + click, accept image files, preview selected file)
- [x] T017 [P] [US1] Create ModelSelector component in `frontend/src/components/ModelSelector.tsx` (dropdowns for model and runtime selection, fetch available options from /v1/models)
- [x] T018 [US1] Create API client in `frontend/src/lib/api.ts` (functions: detectImage, detectVideo, getModels, compareAll — all calling backend endpoints)
- [x] T019 [US1] Create BBoxOverlay component in `frontend/src/components/BBoxOverlay.tsx` (draw bounding boxes on image using canvas overlay, colored by class, show label + confidence)
- [x] T020 [US1] Create DetectionResult component in `frontend/src/components/DetectionResult.tsx` (display annotated image via BBoxOverlay, latency badge, detection count)
- [x] T021 [US1] Build main detection page in `frontend/src/app/page.tsx` (compose FileUpload + ModelSelector + Detect button + DetectionResult, wire up API calls, show loading spinner during inference)

**Checkpoint**: Single image detection fully functional end-to-end. Upload image → pick model+runtime → see bboxes + latency.

---

## Phase 4: User Story 4 — mAP Evaluation on Custom Dataset (Priority: P1)

**Goal**: Benchmark script computes COCO mAP + latency for all 6 combos
**Independent Test**: Run `python scripts/benchmark.py --images data/images/ --annotations data/annotations.json` → verify results table

### Implementation for User Story 4

- [x] T022 [US4] Create COCO prediction converter in `scripts/benchmark.py` (convert Ultralytics xyxy predictions to COCO `{"image_id", "category_id", "bbox"(xywh), "score"}` format)
- [x] T023 [US4] Implement benchmark evaluation loop in `scripts/benchmark.py` (for each of 6 model+runtime combos: load model, run inference on all images with warm-up, compute avg latency, collect predictions)
- [x] T024 [US4] Implement COCO mAP computation in `scripts/benchmark.py` (load ground truth annotations, run COCOeval for mAP@0.5 and mAP@0.5:0.95, compute speedup relative to PyTorch baseline)
- [x] T025 [US4] Implement results output in `scripts/benchmark.py` (print formatted table to stdout, save JSON to `results/benchmark.json` with --output flag)

**Checkpoint**: Benchmark script produces complete results table for all 6 combos with mAP + latency + speedup.

---

## Phase 5: User Story 2 — Video Detection (Priority: P2)

**Goal**: Upload video, get annotated video with per-frame bboxes + stats
**Independent Test**: Upload 10s MP4 → YOLOv11m+ONNX → verify annotated video + per-frame latency chart

### Implementation for User Story 2

- [x] T026 [US2] Implement video processing service in `backend/services/video.py` (read video frames with OpenCV, run detection per frame, draw bboxes on frames with cv2.rectangle + cv2.putText, write annotated video with cv2.VideoWriter, track per-frame latency)
- [x] T027 [US2] Add video detection to `/v1/detect/upload` endpoint in `backend/routers/detect.py` (detect video input_type, validate size ≤100MB, call video service, return DetectionResponse with per-frame results + annotated_video_url)
- [x] T028 [US2] Add `/v1/results/{result_id}/video` endpoint in `backend/routers/detect.py` (serve pre-rendered annotated video as video/mp4 stream)
- [x] T029 [P] [US2] Create VideoResult component in `frontend/src/components/VideoResult.tsx` (video player for annotated video, per-frame latency bar chart, summary stats: avg latency, total frames, total detections)
- [x] T030 [US2] Update main page in `frontend/src/app/page.tsx` (detect video uploads, show progress bar during processing, display VideoResult component on completion)

**Checkpoint**: Video detection end-to-end working. Upload video → annotated video + stats displayed.

---

## Phase 6: User Story 3 — Benchmark Comparison (Priority: P2)

**Goal**: Compare all 6 model+runtime combos on same input, show table
**Independent Test**: Upload image → click "Compare All" → verify 6-row table with latency + detection count + speedup

### Implementation for User Story 3

- [x] T031 [US3] Add `/v1/detect/compare` endpoint in `backend/routers/detect.py` (run same image through all 6 model+runtime combos sequentially, loading/unloading each, return array of 6 DetectionResponses)
- [x] T032 [P] [US3] Create ComparisonTable component in `frontend/src/components/ComparisonTable.tsx` (table with columns: model, runtime, latency, detection count, speedup factor; highlight fastest runtime)
- [x] T033 [US3] Build comparison page in `frontend/src/app/compare/page.tsx` (FileUpload + "Compare All" button, call compare endpoint, display ComparisonTable)
- [x] T034 [US3] Add navigation between detection and comparison views in `frontend/src/app/page.tsx` and `frontend/src/app/compare/page.tsx` (nav links or tabs)

**Checkpoint**: Comparison view working. Upload → Compare All → see 6-row results table.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final documentation, cleanup, and submission prep

- [x] T035 [P] Create README.md at repository root (setup instructions for GCP VM + local frontend, how to export models, how to run benchmark, how to use the app)
- [x] T036 [P] Create `data/annotations.json` placeholder with COCO JSON schema example and instructions for annotation workflow (Label Studio)
- [x] T037 Run full pipeline validation: export models → start backend → start frontend → test image detection → test video detection → test comparison → run benchmark script
- [ ] T038 Generate final benchmark results table and save to `results/benchmark.json`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational — MVP
- **US4 (Phase 4)**: Depends on Foundational — can run in parallel with US1 (separate scripts/ directory)
- **US2 (Phase 5)**: Depends on US1 (extends image detection to video)
- **US3 (Phase 6)**: Depends on US1 (reuses detection infrastructure)
- **Polish (Phase 7)**: Depends on all user stories

### Within Each User Story

- Backend services before endpoints
- Endpoints before frontend components
- Frontend components before page composition
- Core logic before integration

### Parallel Opportunities

- T003 + T004 + T005: all independent setup tasks
- T006 + T007: request and response schemas (different files)
- T010 + T011: health and models endpoints (different files)
- T016 + T017: FileUpload and ModelSelector components (different files)
- US1 (Phase 3) + US4 (Phase 4): can proceed in parallel after Foundational
- T029: VideoResult component can start while T026-T028 (backend video) are in progress
- T032: ComparisonTable can start while T031 (compare endpoint) is in progress
- T035 + T036: README and annotation placeholder (different files)

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL — blocks all stories)
3. Complete Phase 3: User Story 1 (Single Image Detection)
4. **STOP and VALIDATE**: Upload image, verify bboxes + latency
5. This alone demonstrates core functionality

### Incremental Delivery

1. Setup + Foundational → Backend skeleton running
2. US1 → Single image detection working → Demo-ready MVP
3. US4 → Benchmark script produces results table → Evaluation ready
4. US2 → Video detection working → Full media support
5. US3 → Comparison view → Side-by-side benchmarking
6. Polish → README, final validation, submission prep

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story
- Each user story is independently testable after its phase completes
- US4 (benchmark script) is prioritized as P1 alongside US1 because mAP evaluation is a core assignment requirement
- Single-model memory strategy: T008 (ModelManager) handles load/unload — all downstream tasks inherit this behavior
- Commit after each completed phase or logical group
