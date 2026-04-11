# Implementation Plan: Deploy and Benchmark

**Branch**: `002-deploy-and-benchmark` | **Date**: 2026-04-10 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-deploy-and-benchmark/spec.md`

## Summary

Provision a GCP V100 VM, bootstrap the backend environment (clone,
install deps, install TensorRT, export models), annotate a custom
dataset locally via Label Studio and upload it, run the benchmark
script on the VM, retrieve and commit results, then tear down the
VM to stop billing.

This is a deployment/operations feature — the "implementation" is
a sequence of reproducible commands rather than new source code.

## Technical Context

**Language/Version**: Bash (command execution); Python 3.11 on VM (runtime only)
**Primary Dependencies**: gcloud CLI (local), NVIDIA driver + CUDA 11.8 (VM), tensorrt pip package (VM), Label Studio (local for annotation)
**Storage**: GCP persistent disk (50GB boot) on VM, local filesystem for dataset, git for results
**Testing**: Manual verification at each checkpoint (nvidia-smi, file listing, benchmark output)
**Target Platform**: Local machine (macOS/Linux with gcloud) + GCP VM (Debian-based Deep Learning VM)
**Project Type**: deployment-runbook
**Performance Goals**: Provisioning <10min, bootstrap <30min, benchmark <20min, end-to-end <5hrs including annotation
**Constraints**: V100 16GB HBM2 (single-model memory strategy inherited from feature 001); GCP on-demand billing (~$2.86/hr — tear down promptly); assignment due 2026-04-17. Note: original target was T4 but project has 0 T4 quota; V100 is a strictly better available alternative.
**Scale/Scope**: Single VM, single user, single benchmark run, 50+ images + 1 video

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Reproducibility | PASS | Every command is documented in `contracts/commands.md` and `quickstart.md`; any user with project access can repeat the process end-to-end |
| II. Faithful to Assignment Scope | PASS | All commands directly produce the Option 2 deliverables (benchmark with custom annotations); no out-of-scope work |
| III. Clean & Readable Code | PASS | No new source code — this is a runbook. Commands are organized by phase with clear post-conditions |
| IV. Proper Documentation | PASS | `quickstart.md` is the runbook, `contracts/commands.md` has exact commands with post-conditions |
| V. Honest Reporting | PASS | Benchmark results come from a single script run on real hardware with warm-up excluded; reproducibility check built in |

No violations. No complexity tracking needed.

## Project Structure

### Documentation (this feature)

```text
specs/002-deploy-and-benchmark/
├── plan.md                 # This file
├── research.md             # Phase 0: GCP image/machine type/firewall decisions
├── data-model.md           # Phase 1: CloudVM, BootstrapState, Dataset, BenchmarkRun entities
├── quickstart.md           # Phase 1: Hands-on runbook (A → H phases)
├── contracts/
│   └── commands.md         # Phase 1: Exact command contracts with post-conditions
└── tasks.md                # Phase 2: Task breakdown (/speckit.tasks)
```

### Artifacts Produced (repository root)

```text
data/
├── images/                 # Custom images (50+) — gitignored
├── video/                  # Custom video clip(s) — gitignored
└── annotations.json        # COCO JSON (committed, overwriting placeholder)

results/
└── benchmark.json          # Final benchmark output (force-committed)

# No new backend/ or frontend/ code — feature 001 supplies all source.
```

**Structure Decision**: No new source code. This feature produces
documentation (`specs/002-deploy-and-benchmark/*`) and data artifacts
(`data/annotations.json`, `results/benchmark.json`).

## Complexity Tracking

No Constitution Check violations. No complexity justifications needed.
