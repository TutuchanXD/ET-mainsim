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
neighbors. It needs Gaia and ET focal-plane paths.

Table mode accepts CSV or ECSV. Required aliases normalize to `gaia_g_mag` and
`psf_id`; optional aliases normalize to `source_id`, `detector_xpix`, and
`detector_ypix`. Each row is rendered as a separate one-source scene with its
explicit PSF ID. It never calls `build_catalog_from_spec` and needs no Gaia or
focal-plane registry. Coordinates omitted from both columns default to:

```text
detector_xpix = (detector_cols - 1) / 2
detector_ypix = (detector_rows - 1) / 2
```

This is the physical detector center, not a local `frame_xpix` origin.

## Products

Each target directory contains:

```text
stamps/target_<source_id>/
  raw.h5
  coadd.h5
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
status, and all expected schema sidecars. A complete target is skipped as one
unit; incompatible shard identity fails closed. For direct tables, the
resolved path, byte size, and nanosecond modification time are part of the run
identity. The contract intentionally does not calculate a content hash.
