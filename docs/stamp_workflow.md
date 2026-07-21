# ET Stamp Workflow

## Presets

`et-stamp-smoke` renders two 10 s CPU cadences and one coadd from a packaged
target-plus-neighbor field. `et-stamp-production` uses one physical ET target,
its local neighbors, a 15 x 15 stamp, seven subpixels, 360 raw 10 s cadences,
and twelve 300 s coadds. Its Gaia query is bounded to 0.07 degrees before the
target detector is selected. The production profile inherits the frozen native
ET jitter bank with 100 models, three spacecraft axes, and 300 samples per
model (`[100, 3, 300]`); the smoke profile disables jitter integration rather
than substituting a smaller science bank.

The physical-catalog target defaults to J2000 RA `304.41406499712303` deg,
Dec `51.81987707392268` deg, G < 18. Photsim7 orders the nearest catalog source
first and preserves absolute detector coordinates.

## Input Modes

Catalog mode builds one local scene and may include PSF-support-overlapping
neighbors. Cache generation needs Gaia and ET focal-plane paths; a subsequent
run may omit the Gaia shards but still needs the focal-plane registry for
physical projection. Generation-time path identities remain part of strict
cache request validation.

Table mode accepts CSV or ECSV. `gaia_g_mag` is required and is interpreted as
Gaia G Vega. Optional `curve_id` links a target to a frame-aligned intrinsic
variability curve. Each row is rendered as a separate one-source scene and
never calls `build_catalog_from_spec`.

Each row uses exactly one location mode:

- ICRS/J2000 `ra_deg` plus `dec_deg`: the fixed-transit focal-plane registry
  resolves detector pixels and field angle at epoch `2000.0`. The registry
  directory's content identity is frozen into the input truth and rechecked by
  Photsim7; a changed registry, different detector, or out-of-field coordinate
  fails. The PSF bundle node with the nearest radial field angle is selected,
  with the lower field ID as the deterministic tie break. This route requires
  `et-coord` and the registry but not Gaia catalog shards.
- Explicit `psf_id` without RA/Dec: optional `detector_xpix` and
  `detector_ypix` must be supplied together. Omitting both uses:

```text
detector_xpix = (detector_cols - 1) / 2
detector_ypix = (detector_rows - 1) / 2
```

This is the physical detector center, not a local `frame_xpix` origin. The row
is carried to Photsim7 with a version-2 `reference_field_nonphysical`
declaration containing the selected PSF-node angle, reference polar angle,
4.83 arcsec/pix scale, and axis signs. It is a deterministic reference-field
approximation and must not be interpreted as a physical sky-to-detector
solution.

The optional variability CSV/ECSV is long-format with required `curve_id`,
`frame_index`, and `relative_flux` columns. Every curve must contain each raw
frame index exactly once from zero through `N - 1`; factors must be finite and
non-negative. Targets without a curve remain static. Time-like input columns
are deliberately ignored: science providers must align or resample physical
time to simulation raw frames before delivery. Curves that no selected target
references are still validated and recorded in provenance.

The factor multiplies each source's baseline photon/electron count before PSF
scene summation and stellar Poisson sampling. A coadd therefore simulates every
variable raw cadence first and sums detector-domain products afterward. See
the [Chinese science-team input contract](source_variability_inputs_zh.md) for
schemas, examples, physical semantics, and the assessment of current team
data.

## Stage 2 Selection Identity

Table mode records `et_mainsim.stamp_source_input_truth.v2` for every target.
It binds the target-table identity, optional variability-table identity,
geometry mode, registry identity when applicable, accepted PSF-bundle identity,
selection policy, selected PSF ID, node angle, and angular residual. The same
payload is copied into raw/coadd product schemas, the variability-truth ECSV
metadata, and `target_artifacts.json`. The worker independently verifies that
the PSF bundle it reads has the SHA-256 accepted by the `SimulationSpec` before
Photsim7 deserializes it.

Full-frame and stamp rendering use the same Photsim7
`simulation_context.v2`. Jitter-model selection is scoped by `run_seed`,
`science_realization_id`, `spacecraft_id`, and absolute raw-frame index, not by
worker, GPU, output path, target request, or stamp window. Consequently, the
same logical cadence and physical realization select the same native jitter
model and emit the same selection RNG trace in both chains.

The current Stage 2 gate verifies geometry, PSF-node selection, bank identity,
and jitter-model selection identity. It does **not** yet prove complete
full-frame/stamp image equivalence: PSF normalization/crop/captured-flux
goldens, persisted standalone selection sidecars, shared-exposure crop, and
ensemble/statistical tolerances remain follow-up work. See
[Stage 2 geometry, PSF, and jitter selection truth](stage2_selection_truth.md).

## Products

Each target directory contains:

```text
stamps/target_<source_id>/
  raw.h5
  coadd.h5
  source_variability_truth.ecsv
  target_artifacts.json
  schemas/raw/frame_NNNNNN.json
  schemas/coadd/coadd_NNNNNN.json
  electron_components/frame_NNNNNN.npz  # opt-in
```

`raw.h5` stores detector-domain canonical 10 s products. `coadd.h5` stores a
`uint64` or `float64` sum of contiguous raw detector-domain products. Readout
noise and cosmic rays are independently sampled per raw cadence; they are not
scaled as one long exposure. Sidecars retain window coordinates, scene IDs,
truth, events, product domain/unit, and full RNG trace. PSF selection currently
appears in runtime provenance and `target_artifacts.json`; a separate durable
PSF-selection sidecar is still part of the remaining Stage 2 work.

`save_raw` and `save_coadd` control both the HDF5 shard and its schema
directory. Electron-domain component NPZ files remain an independent opt-in.

`write_batch_size` controls the bounded raw/coadd HDF5 write buffer and
defaults to 32. The last, shorter batch is flushed before shard finalization.
This is an I/O and memory tuning value, not part of the scientific/product
identity, so it may be changed when resuming a partial run; each attempt records
the value it actually used.

Resume validates the final HDF5 shard, logical IDs, frame IDs, completion
status, all expected schema sidecars, and the variability truth table. Truth
validation includes exact frame/source coverage, finite non-negative factors,
`effective = baseline * relative_flux`, and content digest. A complete target
is skipped as one unit; incompatible shard identity fails closed. Direct
target and variability tables use resolved path, byte size, and SHA-256.
