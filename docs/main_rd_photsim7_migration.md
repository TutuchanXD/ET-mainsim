# Main-RD Photsim7 Package Migration

## Scope

The active ET-mainsim main-detector paths now use Photsim7 package services:

- `main_rd_g18_parallel/main_rd_parallel_core.py`
- `main_rd_grb/simulate_main_rd_full_10s_smoke.py`
- `main_rd_grb/simulate_main_rd_full_10s_g17.py`
- `main_rd_grb/simulate_main_rd_full_10s_g17_extend360.py`

Historical `main_rd_grb/et_sim_10*.py`, `photsim`, `photsim3`, and stamp-long
physics are not migrated by this change.

## Active Call Chain

1. A wrapper creates the temporary compatibility adapter `MainRdRunSpec`.
2. CLI overrides are resolved by `build_main_rd_simulation_spec(...)`.
3. `n_frames` is authoritative and `observing_duration` is derived as
   `n_frames * exposure_s`; both are serialized in `run_config.json`.
4. A package `PreparedStarCatalog` is read or created through
   `StarCatalogCache`.
5. `build_full_frame_services(...)` constructs and reuses the catalog table,
   physical/reference projector, compact effect timeline, PSF manager,
   detector response, cosmic injector, and `SeedTree`.
6. Each assigned cadence calls `run_single_cadence_full_frame(...)` with the
   same service bundle and a frame-specific scattered-light override.
7. `FullFrameArtifactWriter` preserves NPY/event/bias/summary paths and writes
   a versioned frame-product schema sidecar.

The launcher remains responsible for GPU assignment, process lifetime, frame
selection, resume policy, run labels, and output roots.

## Scientific Contracts

### Magnitude and throughput

- Gaia input is `gaia_g_mag`/`g_mean_mag` in the Vega system.
- ET photon-rate input is ET AB magnitude.
- The current documented approximation is
  `et_mag (AB) = gaia_g_mag (Vega)` for G2V-like sources.
- One ET telescope uses a `28 cm` aperture, `58%` optical efficiency, and `80%`
  QE with the Photsim7 ET calibration.
- Values above `100%`, including the removed `101%` legacy setting, are
  rejected by the typed spec.

The previous worker had two conflicting values: `101%` in its config mapping
and `1.0` passed directly to catalog photometry. Consequently old frames cannot
be corrected by one universal multiplier. Comparison must state which old path
actually produced the photon table.

### Coordinates and dynamic effects

Frame arrays use NumPy `[y, x]`. Renderer `x` increases with columns and `y`
with rows; the image/detector origin is lower-left. Catalog `x0/y0` are centered
frame offsets. `frame_xpix/frame_ypix` are frame coordinates. Absolute
`detector_xpix/detector_ypix` identify locations on the physical ET detector.

Package dynamic components retain native coordinate frames in
`effects_timeseries.npz`:

- PSD low-frequency motion: spacecraft attitude `(x, y, z)` in arcsec.
- DVA and thermal drift: radial focal-plane arcsec versus field angle.
- Momentum dump and scripted motion: renderer pixel `(x, y)`.
- PSF breathing: dimensionless scale.

They are projected per source at each cadence in a fixed order. The historical
truth exporter now reconstructs this package timeline and no longer assumes
that every source shares one global ET focal-plane displacement.

## Outputs and Compatibility

NPY remains the active default. Existing frame, cosmic-event, bias-vector,
optional mask, optional stellar-mean, preview, worker-summary, and run-config
paths are preserved. New files are:

- `frame_summaries/frame_NNNNNN_schema.json`
- `effects_timeseries.metadata.json`

`run_config.json` and worker summaries contain the canonical
`simulation_spec`, `compatibility_adapter=MainRdRunSpec`, service provenance,
catalog provenance, and effect schema. No SHA-256 is required by this migration.

## Temporary Compatibility Surface

The following names are no longer used by active main-rd workers:

- `build_star_catalog(...)`
- `build_psf_manager(...)`
- `build_detector_response_sampler(...)`
- `make_renderer(...)`
- `apply_detector_chain(...)`

They can be removed after one compatibility cycle and an `rg` audit.
`build_full_effect_timeseries(...)` and `jitter_integrated_psf_offsets(...)`
must remain until stamp-long is separately migrated because stamp-long imports
them today.

## Verification

Required evidence before merging:

- hermetic package contract with deterministic tiny frames;
- real legacy 17,779-source cache read through `StarCatalogCache`;
- worker delegation/output tests that fail if a legacy physics builder runs;
- active-wrapper capture tests;
- old and package-timeline truth-export tests;
- full ET-mainsim local test selection;
- full Photsim7 regression suite;
- one isolated H100 `8900 x 9120`, one-cadence, physical Gaia `G<17` run with
  exact Photsim7 and ET-mainsim Git provenance.
