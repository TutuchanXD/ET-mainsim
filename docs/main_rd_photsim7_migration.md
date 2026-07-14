# ET-mainsim Migration

## Boundary

ET-mainsim now contains application orchestration only. All maintained science
paths call public Photsim7 APIs. The deleted implementation is preserved at the
annotated Git tag `legacy-scripts-final`; no duplicate `archive/` copy exists on
the main branch.

## Command Mapping

| Removed path or command | Maintained replacement |
| --- | --- |
| `main_rd_g18_parallel/simulate_main_rd_8900x9120_g18.py` | `et-mainsim run et-full-frame --preset production` |
| `main_rd_grb/simulate_main_rd_full_10s_smoke.py` | `et-mainsim run et-full-frame --preset smoke` |
| `main_rd_grb/*extend360.py` | full-frame production with `--frames 360` |
| `stamp_long/run_stamp_long_smoke.py` | `et-mainsim run et-stamp --preset smoke` |
| `stamp_long/*physics_benchmark.py` | `et-mainsim run et-stamp --preset production` plus `slurm/et_stamp.sbatch` |
| `scripts/et_sim_100_det.py` full profile | `et-mainsim run legacy-sim --preset full-effects-production` |
| versioned `et_sim_100_det_v*.py` | Photsim7 `legacy-sim/full-effects` inventory and component evidence |
| `main_rd_1000_eval/*` | full-frame preset/CLI plus `benchmarks/evaluate_main_rd_benchmark.py` |
| old thermal clock script | `benchmarks/run_photsim_600w_thermal_benchmark.sbatch` |

The old `config/et_100_det_inputs_1h.xlsx` was deleted. Current defaults are
typed Photsim7 specs; ET-mainsim run TOML contains only workload and execution
policy.

## Artifact Policy

New runs use `run_manifest.json` and versioned Photsim7 product schemas. Old
run directories are read-only and are never resumed or appended by the new
application.

- Historical full-frame NPY products remain directly readable with NumPy.
- Current full-frame NPY products require their summary/schema sidecars for
  resume validation.
- Current stamp raw and coadd products are separate HDF5 shards with per-item
  schema, truth, and RNG sidecars.
- Legacy pickle/OA output is retained by `legacy-sim`, with a full 23-enabled /
  4-disabled effect manifest in every run directory.
- `tools/artifact_readback/export_last90_truth_tables.py` is retained only for
  read-only reconstruction of completed last90 runs.

Cross-version resume is intentionally unsupported. Use a new run ID when the
scientific spec, workload, catalog identity, or execution identity changes.

## Coordinates And Magnitudes

Frame arrays use NumPy `[y, x]`; renderer `x` is the column axis and `y` is the
row axis. `frame_xpix/frame_ypix` are local frame coordinates.
`detector_xpix/detector_ypix` are absolute physical detector coordinates. Stamp
windows are centered in the latter coordinate system.

Gaia input is Gaia G Vega. ET photon-rate input is ET AB. The current explicit
G2V approximation is `et_mag (AB) = gaia_g_mag (Vega)`. One ET telescope uses
a 28 cm aperture, 58% optical efficiency, and 80% QE under the Photsim7 ET
calibration.
