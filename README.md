# ET-mainsim

ET-mainsim is the reference application for end-to-end Earth 2.0 Telescope
simulations. It owns presets, command-line workflows, output directories,
resume policy, local worker launch, Slurm templates, and user-facing examples.
Photsim7 remains authoritative for catalogs, photometry, PSFs, dynamic effects,
rendering, detector electronics, RNG, and data-product schemas.

The maintained workflow available today is one physical ET `main_rd` full
frame. Stamp and legacy full-effect workflows are tracked in the
[four-PR maintenance plan](docs/devs/et_mainsim_four_pr_maintenance_plan.md).

## Install

Python must match Photsim7: `>=3.12,<3.14`. In the ET workspace, use the
existing `etbase` environment and install the sibling Photsim7 checkout first:

```bash
conda activate etbase
python -m pip install -e /home/cxgao/ET/Photsim7
python -m pip install -e /home/cxgao/ET/ET-mainsim
```

The package root is lightweight. `import et_mainsim` does not initialize
Torch, Ray, CUDA, catalogs, PSFs, or other external assets.

## Quick Start

Inspect the shipped, validated presets:

```bash
et-mainsim presets
et-mainsim show et-full-frame-smoke
et-mainsim show et-full-frame-production --format json
```

`--dry-run` resolves overrides and prints the canonical plan without creating
an output directory, querying a catalog, loading a PSF, or initializing CUDA:

```bash
et-mainsim run et-full-frame --preset smoke --dry-run
```

Run the one-cadence CPU smoke:

```bash
export ET_DATA_DIR=/home/cxgao/ET/Photsim7-data
et-mainsim run et-full-frame --preset smoke
```

The production preset requires the real ET data, Gaia catalog, and focal-plane
registry:

```bash
export ET_DATA_DIR=/home/cxgao/ET/Photsim7-data
export GAIA_CATALOG_DIR=/home/cxgao/gaia_dr3_19mag
export ET_FOCALPLANE_ROOT=/home/cxgao/ET/et_focalplane
export RESULTS_ROOT=/home/cxgao/Results/ET-mainsim

et-mainsim run et-full-frame \
  --preset production \
  --gpus 0,1 \
  --workers-per-device 1
```

The production scientific contract is one `9120 x 8900` physical `main_rd`,
180 ten-second cadences, Gaia G input converted by the documented G2V
approximation to ET AB magnitude, one 28 cm telescope, 58% optical efficiency,
80% QE, and catalog epoch J2000. Override the epoch explicitly when needed:

```bash
et-mainsim run et-full-frame \
  --preset production \
  --target-epoch-jyear 2028.5
```

## Run Contract

Scientific configuration is canonical Photsim7 `SimulationSpec` JSON.
Execution policy is ET-mainsim TOML. Machine paths, GPU assignment, resume,
overwrite, and preview policy do not belong in the scientific preset.

Every run writes `run_manifest.json` with the resolved spec, execution plan,
catalog request/cache metadata, ET-mainsim and Photsim7 Git provenance, frame
plan, attempt history, artifacts, completion summary, or failure details.
Writes are atomic and the parent launcher is the only run-level manifest writer.

Resume is enabled by default. A frame is skipped only when its NPY payload,
summary, versioned Photsim7 schema, shape, and dtype all pass readback. Use
`--overwrite` to rerender an identity-matching run. A different scientific spec,
catalog request, frame plan, or execution identity fails closed.

See [Full-frame workflow](docs/full_frame_workflow.md) for configuration,
outputs, recovery behavior, and worker details. Acceptance evidence is recorded
in [Full-frame PR 1 validation](docs/full_frame_validation.md).

## Slurm

[`slurm/et_full_frame.sbatch`](slurm/et_full_frame.sbatch) runs the same CLI on
the H100 cluster with `etbase-clu`. Set cluster asset paths in the environment
before submission; the template writes through the SSHFS result mount.

## Compatibility Surface

Historical `scripts/`, `stamp_long/`, `main_rd_grb/`, and
`main_rd_g18_parallel/` entrypoints remain available during PR 1. The active
application does not import their physics builders. `MainRdRunSpec` is a
temporary compatibility adapter and now exposes `target_epoch_jyear`; removal
and migration are reserved for PR 4.

The existing main-RD benchmark evaluator remains at
`main_rd_g18_parallel/evaluate_main_rd_benchmark.py` for compatibility and can
read both historical `run_config.json` outputs and the new unified manifest.
