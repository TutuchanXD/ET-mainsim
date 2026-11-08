# Legacy expectation workflow

`et-mainsim run et-legacy-expectation --config run.json` consumes the public
Photsim7 `legacy_expectation_electrons_v1` profile. Use the upstream revision
pinned in `ci/full_pytest_contract.toml`; the version number alone does not
identify this source snapshot. Existing `et-full-frame`, `et-stamp`,
`et-equivalent-coadd` and `legacy-sim` commands retain their contracts.

## Inputs

Paths in JSON or TOML resolve relative to the configuration file. A minimal
catalog-stamp configuration is:

```json
{
  "schema_id": "et_mainsim.legacy_expectation_config.v1",
  "spec_path": "spec.json",
  "catalog_path": "catalog.json",
  "variability_path": "variability.json",
  "aperture_path": "apertures.npz",
  "data_root": "/data/Photsim7-data",
  "output_root": "results",
  "run_id": "six-scope-120s",
  "kind": "catalog-stamp",
  "target_ids": [101],
  "stamp_shape": [21, 21],
  "output_kind": "pixels",
  "cpu_threads": 2
}
```

`spec.json` is `SimulationSpec.to_json_dict()`. Explicitly select the legacy
expectation profile, telescope count/layout, raw integration and sampling,
complete raw groups (`n_raw_frames_per_coadd`), observing origin, RNG seed and
science realization. `psf.compute_device` selects CPU or CUDA. The profile
requires unsaturated raw expectations; accumulated electrons may exceed a raw
full well. Cosmic rays are unsupported. This workflow requires
`sky.subtract_nonstellar_mean=true`; it never fits a background or normalization
to the measured sequence. Upstream rejects unsupported effects and stage claims.

The prepared catalog JSON contains exactly `data`, `metadata` and
`raw_source_arrays`, corresponding to `PreparedStarCatalog`. Preserve its
photometric and geometry declarations. `variability_path` is optional; when
present it contains `source_ids` and `relative_flux`, with factors covering the
**entire** local raw grid, including groups omitted by a worker. The upstream
public catalog builder can prepare a catalog before this workflow; the
application does not implement a second catalog calibration or renderer.

`apertures.npz` contains one 2D array per requested source, keyed by its decimal
source ID. Weights are finite and between 0 and 1. These are fixed apertures;
training, if desired, uses the public `train_legacy_aperture` API on explicitly
chosen expectation/variance maps before execution. Optional
`normalization_electron_rate` is a fixed positive rate applied to each scope.
Without it normalized fields are NaN, and physical electrons/rates remain valid.

Set `kind="full-frame"` to render one common full image per scope/group and
measure all requested targets with full-image aperture arrays. Set
`kind="table-stamp"` and replace `catalog_path`/`variability_path` with
`input_table_path`/optional `variability_table_path` to reuse the established
[table input schema](source_variability_inputs_zh.md). Table rows require
explicit stable `source_id` values. An optional `focalplane_registry` supplies
coordinate-mode geometry. Each row is an independent target scene without
neighbors; the realization derives from the base realization and source ID.
Catalog stamps retain the complete common catalog and its neighbors.

## Products and execution

Default `output_kind="pixels"` retains coadd electrons, expectation/noise/valid
pixel companions, original executed-stage/RNG/geometry truth, per-scope public
reference products and combined public reference products. The public
`read_reference_product` supports later measurement with another aperture.
`save_raw_expectations=true` additionally retains per-raw expectation maps;
these are **not raw noisy images**. The default is false.

`output_kind="lightcurve"` retains compact measured arrays and complete
reduction/source/RNG manifests. It does not retain pixels and cannot support
remeasurement. A group still uses temporary coadd files while being measured;
those are removed before publication. Both modes provide `light_curves.npz`
with `target_<id>/scope_<id>__<field>` and `target_<id>/combined__<field>` keys.
Use `read_legacy_expectation_group` to verify all companions before accessing
these arrays. Scope IDs come from the actual upstream instrument contract.

Per-scope flux is integrated electrons and rate is electrons/s. Combined flux
is the unweighted sum of used scopes; `effective_integration_s` sums their
integration times, not wall-clock duration. The combined product records
per-scope flags, presence, weights and actual contributors; it does not assume
independence or promise a square-root-of-six noise improvement. The workflow
requires every configured scope's product to be present before publishing a
group, while upstream quality flags can still exclude a scope from a sum.

```sh
et-mainsim run et-legacy-expectation --config run.json --dry-run
et-mainsim run et-legacy-expectation --config run.json --shard-count 4 --shard-index 0
et-mainsim run et-legacy-expectation --config run.json --shard-count 4 --shard-index 1
et-mainsim run et-legacy-expectation --config run.json --shard-count 4 --shard-index 2
et-mainsim run et-legacy-expectation --config run.json --shard-count 4 --shard-index 3
et-mainsim run et-legacy-expectation --config run.json --verify-only
```

Independent processes may execute disjoint shards concurrently. Each computes
the full observation's services/timeline and chooses groups afterward. Optional
repeated `--group-index` selects/reorders groups. Neither selection, worker rank
nor run directory enters scientific RNG identity. Duplicate active work is
rejected by a process lock. Completed groups are immutable and checksum verified
on resume. Atomic directory publication precedes a durable progress checkpoint;
a restart can recover a complete published group if the process died between
those operations. Deleting an already checkpointed group is an error.

The immutable run manifest binds actual configuration, effective specs, input
file and asset hashes, source code hashes and runtime/numeric environment.
Resume across input/code/environment changes is rejected. `--verify-only`
requires every planned group; shard execution reports `partial` until all are
present. A missing pixel/companion is an error, not a reason to silently redraw
noise. Process-killed temporary directories under `.staging` are unpublished;
they may be removed after confirming no worker owns the group lock.

`absolute_raw_frame_start_index` defaults to zero. Nonzero offsets affect both
raw identity and upstream absolute times; output seconds are relative to the
declared observing origin, not the first selected worker group. Different coadd
lengths are independent realizations under the upstream contract. Run them as
separate configured products; this command does not infer nested cadences.

## Resource boundaries

Execution and readback operate on one complete time group. The common scene and
effects are shared across scopes; table service graphs are released between
targets. Full frames are rendered once per scope and cached temporarily for
measurement of multiple apertures. Scope combination retains one target's
per-scope image inputs, so memory scales with image size and scope count rather
than the number of time groups. Raw expectation saving adds memory proportional
to raw group length. Parallel shard processes each have their own service graph.

`max_group_pixels` (default 200,000,000) bounds
`image_rows * image_cols * scope_count * target_count` per group. It is a work
budget, **not a peak RAM guarantee**. Public reduction uses float64 images and
multiple copies/masks; reserve at least 200 bytes per image pixel per scope,
plus PSF/response/timeline state, and measure representative jobs before scaling.
Pixel mode stores both coadds and public remeasurement products, including
combined members; reserve several copies on disk and temporary space for one
group per concurrent worker. Light mode reduces delivered disk use but still
needs temporary pixel space and the same scientific rendering resources.

This is M6 workflow readiness. Acceptance of scientific residuals and campaign
promotion remains the separate M7 review; no #43 campaign is started here.
