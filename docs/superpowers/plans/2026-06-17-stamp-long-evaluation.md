# Stamp Long Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the target-star-only long-exposure stamp benchmark tooling for H100 evaluation.

**Architecture:** Add a focused `stamp_long` toolset with one shared core module, four benchmark entrypoints, and one Slurm wrapper. The core owns exposure scaling, deterministic seeding, worker sharding, stamp rendering, npy/manifest output, metrics, and benchmark matrix definitions; runner scripts only select cases and execution mode.

**Tech Stack:** Python, NumPy, optional PyTorch CUDA for vectorized stamp generation, optional Photsim7 cosmic-ray event library conversion, Slurm on the H100 cluster.

---

### Task 1: Core API And Tests

**Files:**
- Create: `stamp_long/test_stamp_long_core.py`
- Create: `stamp_long/stamp_long_core.py`

- [x] Write failing tests for exposure scaling, deterministic seeds, worker sharding, npy output, manifest rows, benchmark case generation, IO worker semantics, and Photsim7 cosmic-ray ADU-to-electrons conversion.
- [x] Run `conda run -n etbase python -m pytest stamp_long/test_stamp_long_core.py -q` and verify failures are due to missing `stamp_long_core`.
- [x] Implement the core API and rerun the focused tests.

### Task 2: Benchmark Runners

**Files:**
- Create: `stamp_long/run_stamp_long_smoke.py`
- Create: `stamp_long/run_stamp_long_compute_benchmark.py`
- Create: `stamp_long/run_stamp_long_io_benchmark.py`
- Create: `stamp_long/run_stamp_long_physics_benchmark.py`

- [x] Add thin CLI wrappers around `stamp_long_core.run_stage`.
- [x] Each runner exposes `--output-root`, `--workers-per-gpu`, `--gpus`, `--dry-run`, `--write-mode`, `--max-cases`, seed/case filters, and render/noise parameter overrides.
- [x] Verify each runner with `--dry-run --max-cases 1`.

### Task 3: H100 Slurm Wrapper

**Files:**
- Create: `stamp_long/submit_stamp_long_h100.sh`
- Create: `stamp_long/sync_stamp_long_h100.sh`

- [x] Add a Slurm script requesting 3 H100 GPUs, activating `etbase-clu`, exporting ET/Photsim7 data paths, recording environment metadata, and dispatching one benchmark stage.
- [x] Ensure the script defaults to `physics`, `workers-per-gpu=10`, and writes under `/cluster/home/cxgao/sshfs-share/ET-mainsim/stamp_long`.
- [x] Add a local sync helper for ET-mainsim, Photsim7, and Photsim7-data before cluster submission.
- [x] Do not run a GPU-heavy job locally; verify shell syntax only.

### Task 4: Verification

**Files:**
- Modify as needed under `stamp_long/`

- [x] Run focused unit tests.
- [x] Run local dry-run commands for all four Python runner scripts.
- [x] Run `bash -n stamp_long/submit_stamp_long_h100.sh` and `bash -n stamp_long/sync_stamp_long_h100.sh`.
- [x] Run a local CPU sample write and inspect the saved npy/manifest.
- [x] Inspect the implementation for accidental old raw/ADC output assumptions.

### Task 5: Reviewed Parameter Set Integration

**Files:**
- Modify: `stamp_long/test_stamp_long_core.py`
- Modify: `stamp_long/stamp_long_core.py`
- Modify: `stamp_long/submit_stamp_long_h100.sh`
- Modify: `stamp_long/ET 多星 Stamp 长时间仿真资源评估实施方案.md`

- [x] Write failing tests for 5 e/pix read noise, default Photsim7-data asset paths, 12 deg PSF stamp loading with subpixels=7, and reviewed jitter settings.
- [x] Run `conda run -n etbase python -m pytest stamp_long/test_stamp_long_core.py -q` and verify failures are due to old defaults or missing functions.
- [x] Update `stamp_long_core.py` defaults, render options, path resolution, PSF bundle loading, background/scattered/dark means, and metadata.
- [x] Update `submit_stamp_long_h100.sh` defaults and CLI argument forwarding for the reviewed parameter set.
- [x] Run focused unit tests, Python compilation, shell syntax checks, runner dry-runs, and a local CPU sample write.
