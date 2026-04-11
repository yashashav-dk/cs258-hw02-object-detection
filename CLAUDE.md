# homework Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-10

## Active Technologies
- Bash (command execution); Python 3.11 on VM (runtime only) + gcloud CLI (local), NVIDIA driver + CUDA 11.8 (VM), tensorrt pip package (VM), Label Studio (local for annotation) (002-deploy-and-benchmark)
- GCP persistent disk (50GB boot) on VM, local filesystem for dataset, git for results (002-deploy-and-benchmark)

- Python 3.11+ (backend, benchmark), TypeScript (frontend) + FastAPI, Ultralytics, onnxruntime-gpu, tensorrt, pycocotools, opencv-python (backend); Next.js 14+, React, Tailwind CSS (frontend) (001-inference-optimization)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

cd src && pytest && ruff check .

## Code Style

Python 3.11+ (backend, benchmark), TypeScript (frontend): Follow standard conventions

## Recent Changes
- 002-deploy-and-benchmark: Added Bash (command execution); Python 3.11 on VM (runtime only) + gcloud CLI (local), NVIDIA driver + CUDA 11.8 (VM), tensorrt pip package (VM), Label Studio (local for annotation)

- 001-inference-optimization: Added Python 3.11+ (backend, benchmark), TypeScript (frontend) + FastAPI, Ultralytics, onnxruntime-gpu, tensorrt, pycocotools, opencv-python (backend); Next.js 14+, React, Tailwind CSS (frontend)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
