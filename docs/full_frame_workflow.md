# ET Full-Frame Workflow

## Ownership Boundary

ET-mainsim owns application policy only:

- preset and CLI selection;
- local paths and environment expansion;
- frame selection and worker assignment;
- catalog-cache preparation policy;
- resume, overwrite, logs, Slurm, and run status.

The active worker delegates scientific construction and rendering to these
Photsim7 package APIs:

1. `SimulationSpec.from_json(...)`
2. `build_catalog_from_spec(...)` and `StarCatalogCache`
3. `build_full_frame_services(...)`
4. `run_single_cadence_full_frame(...)`
5. `FullFrameArtifactWriter`

No ET-mainsim compatibility renderer, detector chain, PSF builder, dynamic
effect builder, or catalog query is called by this workflow.

## Preset Files

Each shipped profile contains two independent files under
`src/et_mainsim/presets/`:

| File | Authority |
| --- | --- |
| `*.spec.json` | Scientific and reproducibility semantics |
| `*.run.toml` | Execution defaults and path policy |

Use `--spec PATH` and `--config PATH` to supply validated replacements. CLI
flags override only the explicitly named values. The fully resolved scientific
spec and execution configuration are written to `run_manifest.json` before
catalog preparation or rendering.

## Paths

Empty path values in shipped TOML files resolve from the environment:

| Value | Environment | Required |
| --- | --- | --- |
| output root | `RESULTS_ROOT` | No; defaults to `./results/et-mainsim` |
| Photsim7 data | `ET_DATA_DIR` | Yes for execution |
| Gaia source catalog | `GAIA_CATALOG_DIR` | Production only |
| focal-plane data | `ET_FOCALPLANE_ROOT/data` | Production only |

There is no hostname-based path fallback. Relative configured paths resolve
against the invocation working directory; `~`, `$NAME`, and `${NAME}` are
expanded at the application boundary.

## Catalog Epoch

`catalog.target_epoch_jyear` is a decimal Julian year and defaults to `2000.0`
(J2000.0). It is independent of `observation.observing_start_date`. Photsim7
uses it only for linear proper-motion propagation from each source `ref_epoch`;
this contract does not add parallax or radial-velocity propagation. Changing
the target epoch changes the catalog request/cache identity and the run's
scientific identity.

## Stage 2 Geometry And Selection Identity

The production catalog uses an explicit `physical_et_focalplane` geometry
declaration. It fixes ICRS coordinates to epoch J2000.0, records the focal-plane
registry directory with a content hash, projects each source through that exact
registry, and selects the PSF node nearest in radial field angle. An asset that
has moved without preserving its content identity, a registry-content change,
or an unverifiable declaration fails closed.

Packaged detector-coordinate inputs use a version-2
`reference_field_nonphysical` declaration only when their reference angle,
polar angle, 4.83 arcsec/pix scale, and axis signs are explicitly present. This
mode supports deterministic smoke/reference work but carries no physical
sky-to-detector claim. Coordinate-derived and explicit-reference sources are
therefore never inferred from a mutable `source_type` label alone.

Production accepts the ET PSF bundle
`psf/et/241006/D280mm-focus` with its frozen SHA-256 in the scientific spec.
Photsim7 verifies that identity before deserialization and emits geometry-bound
PSF selection truth. A custom bundle may be byte-integrity verified, but it does
not receive the owner-accepted science-conformance claim merely by supplying
its own hash.

The full-effects production profile uses the native ET attitude bank at
`jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.npy`, with
shape `[100, 3, 300]`. Its array and strict manifest are both hash-verified
before use. Per-cadence model selection is derived from the shared Photsim7
`simulation_context.v2`: `run_seed`, `science_realization_id`,
`spacecraft_id`, and absolute raw-frame index. The stamp chain consumes the
same context contract, so execution topology and crop/window identity do not
change the selected model.

The Stage 2 selection-identity sub-gate is complete; the complete science
alignment gate is not. Deterministic PSF crop/captured-flux goldens,
shared-exposure full-frame/stamp image parity, standalone selection sidecars,
and preregistered ensemble tolerances remain open. See
[Stage 2 geometry, PSF, and jitter selection truth](stage2_selection_truth.md).

## Commands

```bash
# Read-only plan
et-mainsim run et-full-frame --preset smoke --dry-run

# One installed-package CPU cadence
et-mainsim run et-full-frame --preset smoke --device cpu

# Selected production frames on two local GPUs
et-mainsim run et-full-frame \
  --preset production \
  --frame-indices 0,90,179 \
  --gpus 0,1 \
  --workers-per-device 1

# Build or validate only the exact catalog request/cache
et-mainsim run et-full-frame \
  --preset production \
  --prepare-catalog-only
```

`local-subprocess` assigns frame indices by rank stride. For requested frames
`0,1,2,3` and two workers, rank 0 receives `0,2` and rank 1 receives `1,3`.
Each CUDA subprocess sees one physical assignment through
`CUDA_VISIBLE_DEVICES`; inside the worker, Photsim7 uses device `cuda`.

## Output Layout

```text
<output-root>/<run-id>/
  run_manifest.json
  cache/stars.npz
  frames/frame_NNNNNN.npy
  frame_summaries/frame_NNNNNN.json
  frame_summaries/frame_NNNNNN_schema.json
  cosmic_events/
  bias/
  preview/
  effects_timeseries.npz
  effects_timeseries.metadata.json
  worker_NN_start.json
  worker_NN_done.json
  worker_requests/
  logs/
```

Effect files exist only when dynamic effects are enabled; worker request and log
directories exist only for the subprocess backend. Frame summaries preserve
Photsim7 fields and add an `et_mainsim` namespace for rank, source count,
elapsed time, device, and peak CUDA memory. The canonical frame-product sidecar
remains `photsim7.single_cadence_frame_products.v1`. Geometry/PSF selection is
currently retained through service/frame provenance and jitter selection
through provenance plus the merged RNG trace; dedicated durable selection
sidecars are not yet the completed product contract.

## Resume and Failure Semantics

- `resume=true` skips only a payload with a valid summary and frame-product
  schema whose frame index, shape, and dtype match the resolved spec.
- Partial or malformed artifacts are rerendered under resume.
- `resume=false` refuses existing frame artifacts unless `overwrite=true`.
- Resume and overwrite are mutually exclusive in one execution config.
- An existing manifest must match workflow, run id, scientific spec, and stable
  execution identity. Control flags such as resume, overwrite, progress, and
  cache refresh do not change identity.
- Each retry creates a numbered attempt. Completed and failed runs can start a
  new identity-matching attempt; a manifest already marked running rejects a
  second launcher.
- Worker exceptions mark the run failed with exception type and message.

Historical run directories without `run_manifest.json` are read-only. This
workflow does not claim cross-version resume compatibility with them.
