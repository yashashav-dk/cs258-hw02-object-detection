# Implementation Plan: Inference Optimization Pipeline

**Branch**: `001-inference-optimization` | **Date**: 2026-04-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-inference-optimization/spec.md`

## Summary

Build a full-stack object detection inference application serving
YOLOv8m and YOLOv11m via a FastAPI backend (OpenAI-compatible API),
with a Next.js frontend for visualization, accelerated by TensorRT
and ONNX Runtime. Evaluate accuracy (COCO mAP) and speed (latency)
across all 6 model+runtime combinations using a custom-annotated
dataset of 50+ images. Backend runs on GCP VM with T4 GPU.

## Technical Context

**Language/Version**: Python 3.11+ (backend, benchmark), TypeScript (frontend)
**Primary Dependencies**: FastAPI, Ultralytics, onnxruntime-gpu, tensorrt, pycocotools, opencv-python (backend); Next.js 14+, React, Tailwind CSS (frontend)
**Storage**: Filesystem (uploaded files, exported models, annotated dataset, rendered videos)
**Testing**: pytest (backend), manual verification (frontend)
**Target Platform**: Linux (GCP VM with T4 GPU for backend), macOS/Linux (local for frontend)
**Project Type**: web-service + web-frontend + benchmark-script
**Performance Goals**: TensorRT ≥1.5x speedup, ONNX ≥1.2x speedup over PyTorch baseline; image detection response <5s excluding cold-start
**Constraints**: T4 16GB VRAM — single model+runtime loaded at a time; 10MB image / 100MB video upload limits
**Scale/Scope**: Single user (homework demo), 50+ annotated images, 1 video clip

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Reproducibility | PASS | Pinned deps (requirements.txt, package-lock.json), fixed random seeds where applicable, scripted export + benchmark pipeline |
| II. Faithful to Assignment Scope | PASS | All 6 Option 2 requirements addressed: 2 models (YOLOv8m, YOLOv11m), FastAPI+OpenAI-compatible API, Next.js frontend, TensorRT+ONNX acceleration, mAP+latency evaluation, custom annotations |
| III. Clean & Readable Code | PASS | Separated into backend/, frontend/, scripts/ modules with clear responsibilities |
| IV. Proper Documentation | PASS | README, quickstart.md, API contracts, inline headers planned |
| V. Honest Reporting | PASS | pycocotools for standard COCO mAP, warm-up excluded, averaged latency, all 6 combos reported |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-inference-optimization/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── api.md           # API contract
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
backend/
├── main.py              # FastAPI app, CORS, lifespan
├── routers/
│   ├── detect.py        # /v1/detect, /v1/detect/upload
│   ├── models.py        # /v1/models
│   └── health.py        # /health
├── services/
│   ├── model_manager.py # Load/unload single model+runtime
│   ├── detector.py      # Run inference, format results
│   └── video.py         # Frame-by-frame video processing + rendering
├── schemas/
│   ├── request.py       # DetectionRequest pydantic models
│   └── response.py      # DetectionResponse, FrameResult, Detection
└── requirements.txt

frontend/
├── src/
│   ├── app/
│   │   ├── page.tsx     # Detection view (main)
│   │   └── compare/
│   │       └── page.tsx # Benchmark comparison view
│   ├── components/
│   │   ├── FileUpload.tsx
│   │   ├── ModelSelector.tsx
│   │   ├── DetectionResult.tsx
│   │   ├── VideoResult.tsx
│   │   ├── ComparisonTable.tsx
│   │   └── BBoxOverlay.tsx
│   └── lib/
│       └── api.ts       # API client for backend
├── package.json
├── tailwind.config.ts
└── next.config.ts

scripts/
├── export_models.py     # Export YOLOv8m/YOLOv11m to ONNX + TensorRT
└── benchmark.py         # COCO mAP + latency evaluation

data/
├── images/              # Custom dataset images (gitignored)
├── video/               # Custom dataset video (gitignored)
└── annotations.json     # COCO JSON ground truth (committed)
```

**Structure Decision**: Web application structure (backend + frontend)
with additional scripts/ for offline benchmark and export tooling.
data/ holds the custom dataset (images gitignored, annotations committed).

## Complexity Tracking

No Constitution Check violations. No complexity justifications needed.
