<!--
Sync Impact Report
- Version change: 1.0.0 → 1.1.0
- Modified principles:
  - II. Correct Implementation → II. Faithful to Assignment Scope
    (updated to reflect inference optimization, not model training)
  - III. Clean & Readable Code (updated module examples to match
    full-stack project: backend, frontend, benchmark)
- Added sections: None
- Removed sections: None
- Modified sections:
  - Academic Integrity & Constraints: updated pre-trained model
    clause to reflect Option 2 permits pre-trained weights
  - Development Workflow: added GCP VM split-infrastructure guidance
- Templates requiring updates:
  - .specify/templates/plan-template.md ✅ no changes needed (generic)
  - .specify/templates/spec-template.md ✅ no changes needed (generic)
  - .specify/templates/tasks-template.md ✅ no changes needed (generic)
  - .specify/templates/commands/*.md — no command templates exist
- Follow-up TODOs: None
-->

# HW02 Object Detection Constitution

## Core Principles

### I. Reproducibility

All experiments, results, and outputs MUST be reproducible from
source. Random seeds MUST be fixed and documented where applicable.
Environment dependencies MUST be pinned (`requirements.txt` for
backend, `package.json` lock file for frontend). Data preprocessing
and annotation pipelines MUST be deterministic and scripted — no
manual transformations. Any benchmark result reported MUST be
regenerable by running the documented pipeline end-to-end on the
specified GCP VM configuration.

### II. Faithful to Assignment Scope

The implementation MUST satisfy all six requirements of Option 2
(Inference Optimization):
- At least two strong-performing detection models (YOLOv8m, YOLOv11m)
- FastAPI backend with OpenAI-compatible API structure
- Frontend (Next.js) for upload, inference, and visualization
- At least two acceleration methods (TensorRT, ONNX Runtime)
- Evaluation of both accuracy (COCO mAP) and speed (latency)
- Custom-annotated dataset for evaluation

Deviations from these requirements MUST be explicitly documented
with justification. Scope creep beyond assignment requirements
MUST be avoided — focus on satisfying rubric criteria.

### III. Clean & Readable Code

Code MUST be well-organized with clear separation of concerns:
backend (API routes, model management, inference logic), frontend
(upload, visualization, comparison views), and benchmark scripts
in distinct modules. Variable and function names MUST be
descriptive. Magic numbers MUST be extracted into named constants
or configuration. Code MUST run without modification given the
documented setup steps for both local (frontend) and GCP VM
(backend + inference) environments.

### IV. Proper Documentation

A README MUST explain how to set up both environments (local
frontend, GCP VM backend), run inference, and reproduce all
benchmark results. Each module MUST include a header describing
its purpose. Key design decisions (model selection, acceleration
method choices, API design) MUST be documented. Benchmark outputs
(mAP tables, latency comparisons, speedup factors) MUST be
clearly labeled and reproducible.

### V. Honest Reporting of Results

All evaluation metrics MUST be computed on the custom-annotated
dataset using COCO evaluation protocol. Results MUST NOT be
cherry-picked — report representative performance with warm-up
runs excluded and latency averaged over multiple iterations.
Accuracy differences between runtimes (PyTorch vs ONNX vs
TensorRT) MUST be reported honestly, including any mAP
degradation from format conversion. Failure modes and limitations
of each acceleration method MUST be acknowledged.

## Academic Integrity & Constraints

- All submitted work MUST be original and comply with the
  course's academic integrity policy.
- Collaboration boundaries defined by the course syllabus MUST
  be respected.
- External code or resources used MUST be cited with source
  and license.
- Pre-trained model weights (YOLOv8m, YOLOv11m) are permitted
  as Option 2 focuses on inference optimization, not training.
  The Ultralytics library and its weights MUST be cited.
- Custom annotations MUST be the student's own work — downloading
  pre-existing annotation datasets is not sufficient.
- AI-assisted code generation MUST be disclosed per course
  policy.

## Development Workflow

- Work incrementally: implement one component, verify it works,
  then move to the next (backend → export → benchmark → frontend).
- Validate intermediate outputs visually (e.g., bounding box
  overlays, latency measurements) before proceeding downstream.
- Use version control commits at meaningful checkpoints (e.g.,
  API serving, model export working, frontend upload functional).
- Backend and inference run on GCP VM (T4 GPU); frontend
  development runs locally. Code MUST be kept in sync via git.
- Before submission, run the full pipeline from a clean state
  to confirm reproducibility on both environments.
- Keep the repository clean: no model weight files, no large
  data files, no compiled TensorRT engines, no IDE-specific
  configuration committed. Use `.gitignore` appropriately.

## Governance

This constitution governs all development on the HW02 Object
Detection homework (Option 2: Inference Optimization). Amendments
require updating this document, incrementing the version per
semantic versioning, and verifying consistency with dependent
templates.

- **Amendment procedure**: Edit this file, update version and
  date, run consistency propagation checklist.
- **Versioning policy**: MAJOR for principle removal/redefinition,
  MINOR for new principles or material expansion, PATCH for
  clarifications and typo fixes.
- **Compliance**: All code, documentation, and reports MUST align
  with these principles before submission.

**Version**: 1.1.0 | **Ratified**: 2026-04-10 | **Last Amended**: 2026-04-10
