# ET-mainsim

ET-mainsim is the reference application for Earth 2.0 Telescope simulations.
It owns presets, CLI orchestration, manifests, local workers, Slurm templates,
resume policy, and examples. Photsim7 owns all catalog, photometry, PSF,
dynamic-effect, detector, RNG, and product-schema behavior.

Three end-to-end workflows are maintained:

```text
et-mainsim run et-full-frame --preset smoke|production
et-mainsim run et-stamp --preset smoke|production
et-mainsim run legacy-sim --preset full-effects-smoke|full-effects-production
```

## Install

Python must match Photsim7: `>=3.12,<3.14`. In the ET workspace use the
existing `etbase` environment:

```bash
conda activate etbase
python -m pip install -e /home/cxgao/ET/Photsim7
python -m pip install -e /home/cxgao/ET/ET-mainsim
```

The package import is lightweight and does not initialize Torch, Ray, CUDA,
catalogs, or external assets.

## Quick Start

```bash
et-mainsim presets
et-mainsim show et-stamp-production --format json
et-mainsim run et-full-frame --preset smoke --dry-run
```

Local smoke runs need only the Photsim7 asset root:

```bash
export ET_DATA_DIR=/home/cxgao/ET/Photsim7-data
et-mainsim run et-full-frame --preset smoke
et-mainsim run et-stamp --preset smoke
et-mainsim run legacy-sim --preset full-effects-smoke
```

Production physical-catalog runs also require:

```bash
export GAIA_CATALOG_DIR=/home/cxgao/gaia_dr3_19mag
export ET_FOCALPLANE_ROOT=/home/cxgao/ET/et_focalplane
export RESULTS_ROOT=/home/cxgao/Results/ET-mainsim
```

### Stamp Table Input

Stamp simulation also accepts a query-independent table. Every row is one
independent target-only scene and therefore does not initialize or query a
full-frame catalog:

```bash
et-mainsim run et-stamp \
  --preset smoke \
  --input-table targets.csv
```

Required columns are `gaia_g_mag` (Gaia G, Vega) and `psf_id`. Optional
`source_id`, `detector_xpix`, and `detector_ypix` columns are supported; both
coordinates must be supplied together. Missing coordinates default to the
physical detector center. The current documented conversion is
`et_mag (AB) = gaia_g_mag (Vega)` for G2V-like sources. See the packaged
`et_stamp_table_example.csv` and [stamp workflow](docs/stamp_workflow.md).

## Run Contract

Scientific configuration is a canonical Photsim7 `SimulationSpec`. Execution
policy and typed `[workload]` configuration are ET-mainsim TOML. Machine paths,
GPU assignment, Ray resources, resume, overwrite, and benchmark controls do not
belong in the scientific spec.

Every run writes an atomic `run_manifest.json` containing the resolved spec,
workload and execution identity, paths, attempt history, provenance, product
locations, completion summary, or failure. Identity drift fails closed.

- Full frame resumes only validated NPY + summary + schema items.
- Stamp resumes HDF5 shard items and skips only a fully validated target.
  Direct-table identity includes the resolved path, byte size, and nanosecond
  modification time; changing the table requires a new run ID or overwrite.
- Legacy skips only an entirely complete workload; partial pickle/OA output is
  rejected and requires `--overwrite` or a new run ID.
- `--dry-run` creates no output and initializes no catalog, PSF, CUDA, or Ray.

## Slurm And Tools

Maintained H100 templates are under `slurm/`. Full-frame performance tools and
the 600 W thermal-load reproducer are under `benchmarks/`. Historical last90
artifacts can be read without rerunning via `tools/artifact_readback/`.

The removed script layout is preserved by Git tag `legacy-scripts-final`.
See [migration](docs/main_rd_photsim7_migration.md) for command and artifact
mapping. Current details are in [full frame](docs/full_frame_workflow.md),
[stamp](docs/stamp_workflow.md), and [legacy](docs/legacy_workflow.md).
