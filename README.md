# ET-mainsim

Photsim7 0.5.6 的配套支持包括 `SimulationSpec.geometry`、逐帧
`physical_sky_projection`、绑定源/相机/观测时间/积分时长的预计算 DVA，以及动态
geometry/PSF truth。full-frame 的 shared crop 保持父运行首帧的 detector 窗口，stamp 保持几何声明首个
pose 的窗口；stamp coadd 与严格 sidecar 完成校验支持跨指向切换。运行身份包含完整配置，
修改几何或观测起点后须使用新 run。坐标与边界见上游
[观测几何文档](https://github.com/TutuchanXD/Photsim7/blob/main/docs/observation_geometry.md)。

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
python -m pip install -e '/home/cxgao/ET/Photsim7[gpu]'
python -m pip install -e /home/cxgao/ET/et_focalplane
python -m pip install -e /home/cxgao/ET/ET-mainsim
```

Coordinate-table stamp inputs require `et-coord>=0.1.2`, the first release
containing the semantic focal-plane registry owner-attestation API used by the
maintained Galaxy producer.

The current full-test and release-engineering contracts are validated against
ET-coordinate commit `f9cec8038b021c9540a026b94e876dc3240071d1`
(version 0.1.2) and Photsim7 commit
`7611ef0857d27847b03306d9d7472b0d5c028a93` (version 0.5.6). Install those exact
dependency snapshots before installing the ET-mainsim wheel with `--no-deps`,
then run `python -m pip check`.

The maintained runtime requires `photsim7[gpu]>=0.5.6,<0.6`. Its imports use
canonical or retained public APIs and do not require the 16 top-level facades
removed in Photsim7 0.5.0. The GPU extra supplies the Torch/Kornia backend for
both CPU and CUDA execution; CUDA availability is checked separately at runtime.

The package import is lightweight and does not initialize Torch, Ray, CUDA,
catalogs, or external assets.

### CI and local verification

With `etbase` activated, `./scripts/ci/smoke.sh` and `./scripts/ci/full.sh`
invoke the same isolated CI bootstrap as GitHub Actions. Use `--python
python3.13` for the other supported interpreter. Full verification installs
the frozen dependency snapshots and runs the complete receipt-checked suite;
it does not modify the developer environment.

ET-mainsim is public; persistent self-hosted runners are intentionally not attached to the repository.
See [CI operations](docs/devs/ci_operations.md) for the hosted security model,
private-dependency review boundary, local commands, and unchanged required checks.

### Developer: continuous time-shard planning

The generic `et-stamp` CLI still uses its historical target-worker model, but
the Galaxy formal producer now implements globally contiguous raw-frame shards,
atomic raw/coadd delivery and a frozen time-plan contract. See
[Galaxy raw-coverage science delivery](docs/galaxy_raw_coverage_science_delivery_zh.md)
for the operational product path, and
[continuous time-shard planning](docs/continuous_time_shards.md) for the
generic API and future scheduler hook.

## Configurable inputs and reproducibility

`--spec` accepts canonical JSON and parameter workbooks. Scientific values come
from the ET preset, explicit workbook rows, then explicit CLI overrides.
Per-stream RNG seeds pass through unchanged. Each run records its effective
configuration and actual asset contents, and rejects incompatible resumes.
See [configuration, cache and resume rules](docs/configurable_inputs_zh.md).

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

Production physical-catalog cache generation also requires:

```bash
export GAIA_CATALOG_DIR=/home/cxgao/gaia_dr3_19mag
export ET_FOCALPLANE_ROOT=/home/cxgao/ET/et_focalplane
export RESULTS_ROOT=/home/cxgao/Results/ET-mainsim
```

Once a canonical cache exists, full-frame and catalog-stamp rendering can run
without the Gaia shard directory. The ET focal-plane registry remains a
rendering dependency, and generation-time path identities must still match the
resolved spec because cache request metadata is validated.

### Stamp Table Input

Stamp simulation also accepts a query-independent table. Every row is one
independent target-only scene and therefore does not initialize or query a
full-frame catalog:

```bash
et-mainsim run et-stamp \
  --preset smoke \
  --input-table targets.csv
```

`gaia_g_mag` (Gaia G, Vega) is required. Location is mutually exclusive:
provide ICRS/J2000 `ra_deg` plus `dec_deg` for focal-plane mapping and nearest
radial PSF selection, or omit sky coordinates and provide an explicit
`psf_id`. In explicit-PSF mode, optional `detector_xpix` plus
`detector_ypix` default together to the physical detector center. The input
semantic is Gaia G Vega; an AB number must not be silently treated as a Gaia G
Vega magnitude in a formal science campaign.

Frame-aligned intrinsic variability is an optional second long-format table:

```bash
et-mainsim run et-stamp \
  --preset production \
  --input-table targets.ecsv \
  --variability-table variability.ecsv
```

Targets link to curves with optional `curve_id`; each variability curve must
contain exactly one finite, non-negative `relative_flux` for every raw
`frame_index`. Input time columns are not interpreted. See the packaged
`et_stamp_variability_target_example.csv` plus
`et_stamp_variability_example.csv`, [stamp workflow](docs/stamp_workflow.md),
and the [Chinese science-team input
contract](docs/source_variability_inputs_zh.md).

## Run Contract

Scientific configuration is a canonical Photsim7 `SimulationSpec`. Execution
policy and typed `[workload]` configuration are ET-mainsim TOML. Machine paths,
GPU assignment, Ray resources, resume, overwrite, and benchmark controls do not
belong in the scientific spec.

Every run writes an atomic `run_manifest.json` containing the resolved spec,
workload and execution identity, paths, attempt history, provenance, product
locations, completion summary, or failure. Identity drift fails closed.

- Full frame resumes only validated NPY + summary + schema items.
  An optional `[workload.shared_exposure_stamps]` section can persist ordered
  100 x 300 target crops from the same parent exposure. It renders each
  cadence once per fresh render or deterministic reconstruction, regardless of
  target/product count. It adds no stamp-specific RNG draws and treats the
  parent, batched product shards, and per-frame completion marker as one
  fail-closed resume bundle.
- Stamp resumes HDF5 shard items and skips only a fully validated target.
  Direct-table identity includes the resolved path, byte size, and SHA-256;
  variability truth is also content-validated before a target is skipped.
  Changing an input requires a new run ID or overwrite.
- Legacy skips only an entirely complete workload; partial pickle/OA output is
  rejected and requires `--overwrite` or a new run ID.
- `--dry-run` creates no output and initializes no catalog, PSF, CUDA, or Ray.

Shared-exposure full-frame stamps are configured by signed 64-bit source IDs,
shape, and product keys. Their immutable target plan comes from authoritative
Photsim7 full-frame pixel geometry; edge windows are zero padded. The HDF5
products are deterministic crops, not independent stamp simulations and not a
claim of transferred source truth. See [full frame shared-exposure
products](docs/full_frame_workflow.md#shared-exposure-stamp-products).

## Slurm And Tools

Maintained H100 templates are under `slurm/`. Full-frame performance tools and
the 600 W thermal-load reproducer are under `benchmarks/`. Historical last90
artifacts can be read without rerunning via `tools/artifact_readback/`.

The removed script layout is preserved by Git tag `legacy-scripts-final`.
See [migration](docs/main_rd_photsim7_migration.md) for command and artifact
mapping. Current details are in [full frame](docs/full_frame_workflow.md),
[stamp](docs/stamp_workflow.md), [stamp science delivery
bundles](docs/stamp_science_delivery_zh.md), and
[legacy](docs/legacy_workflow.md).

S4b telescope layouts require Photsim7 0.5.6. Scientific JSON accepts an explicit
`instrument.telescopes` list with stable nonnegative IDs, a homogeneous detector
association, and either `coincident` or `reference_translation` placement. The
optional workbook row `Telescope Layout` accepts the same list as JSON. Count
and list length must agree. List order is canonicalized before run identity is
computed. Legacy scope zero keeps its root directory; other configurations use
`scope_<id>/`, including a single nonzero ID. Reference translations require a
nonphysical reference catalog and explicit PSF field selection; they do not
claim calibrated ET pointing geometry.

Full-frame shared-exposure stamps use each scope's parent and projected target
windows. Translated layouts store each plan at
`shared_exposure/scope_<id>/target_plan.json`; manifests and completion markers
reference that plan. Reordering scopes preserves resume identity; changing
members, detector association or layout requires a new run. Independent target
MC remains single-scope. Real CPU/CUDA acceptance is available through
`validation/test_s4b_telescope_layouts.py` with the same asset environment as S3.
