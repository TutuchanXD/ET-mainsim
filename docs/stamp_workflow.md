# ET Stamp Workflow

## Presets

`et-stamp-smoke` renders two 10 s CPU cadences and one coadd from a packaged
target-plus-neighbor field. `et-stamp-production` uses one physical ET target,
its local neighbors, a 15 x 15 stamp, seven subpixels, 360 raw 10 s cadences,
and twelve 300 s coadds. Its Gaia query is bounded to 0.07 degrees before the
target detector is selected.

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
  resolves detector pixels and field angle. A different detector or
  out-of-field coordinate fails. The PSF bundle node with the nearest radial
  field angle is selected. This route requires `et-coord` and the registry but
  not Gaia catalog shards.
- Explicit `psf_id` without RA/Dec: optional `detector_xpix` and
  `detector_ypix` must be supplied together. Omitting both uses:

```text
detector_xpix = (detector_cols - 1) / 2
detector_ypix = (detector_rows - 1) / 2
```

This is the physical detector center, not a local `frame_xpix` origin.

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
truth, events, product domain/unit, and full RNG trace.

`save_raw` and `save_coadd` control both the HDF5 shard and its schema
directory. Electron-domain component NPZ files remain an independent opt-in.

Resume validates the final HDF5 shard, logical IDs, frame IDs, completion
status, all expected schema sidecars, and the variability truth table. Truth
validation includes exact frame/source coverage, finite non-negative factors,
`effective = baseline * relative_flux`, and content digest. A complete target
is skipped as one unit; incompatible shard identity fails closed. Direct
target and variability tables use resolved path, byte size, and SHA-256.
