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

Production and full-effects defaults now use 100 jitter-integrated PSF models
with 300 samples per model. The authoritative native ET input has shape
`[model, spacecraft_xyz, sample] = [100, 3, 300]`; older larger-bank
descriptions are migration history, not runnable defaults.

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

## Geometry, PSF, And Jitter Selection Truth

Stage 2 removes implicit geometry inference from the maintained science paths:

- coordinate mode is ICRS at epoch J2000.0, binds the focal-plane registry by
  content hash, projects through that registry, and uses the nearest radial
  PSF-node angle;
- no-coordinate mode requires an explicit PSF ID and a version-2
  `reference_field_nonphysical` declaration with the reference angle,
  orientation, 4.83 arcsec/pix scale, and axis signs;
- ET stamp table input records these choices and asset identities in
  `et_mainsim.stamp_source_input_truth.v2`;
- the accepted PSF bundle and native jitter-bank array/manifest are verified
  against owner-side hashes before deserialization/load;
- full-frame and stamp use the same Photsim7 `simulation_context.v2` and jitter
  selector scope, so the same logical cadence is independent of worker/GPU,
  output path, request order, and stamp window.

The native bank is
`jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.npy`, with
array SHA-256
`696a986c82902ad18f136f284a30b2ce506998d3e900ea2601a3e6af001cc4d0`.
Its manifest SHA-256 is
`267453c0cc5355f7edfaff76164c56ea38052a866bb967bb124c920394bf7274`.

The ordered migration PR stack includes deterministic pixel goldens,
standalone durable selection sidecars, and the shared-exposure crop/shard
workflow. Historical selection-evidence v1 remains the pre-delivery snapshot
that it records; the implementation stack does not turn it into a full-stage
runtime claim. These are engineering acceptance gates, not a claim of complete
legacy science alignment. Cross-backend/ensemble tolerances, AT-SD25, and the
final science-metric gates remain open. See
[Stage 2 geometry, PSF, and jitter selection truth](stage2_selection_truth.md).

## Coordinates And Magnitudes

Frame arrays use NumPy `[y, x]`; renderer `x` is the column axis and `y` is the
row axis. `frame_xpix/frame_ypix` are local frame coordinates.
`detector_xpix/detector_ypix` are absolute physical detector coordinates. Stamp
windows are centered in the latter coordinate system.

Gaia input is Gaia G Vega. ET photon-rate input is ET AB. The current explicit
G2V approximation is `et_mag (AB) = gaia_g_mag (Vega)`. One ET telescope uses
a 28 cm aperture, 58% optical efficiency, and 80% QE under the Photsim7 ET
calibration.
