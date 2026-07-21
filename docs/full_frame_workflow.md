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

The ordered Stage 2 delivery stack implements the pixel golden and
selection-identity engineering gates. When jitter-integrated selection truth
is enabled and available, the full-frame workflow requires its durable
sidecars before accepting a parent frame as resumable; the explicit no-JI
path instead persists strict `verification_status="unavailable"` metadata.
The historical selection-evidence v1 file remains a pre-delivery snapshot and
is not silently rewritten as a current full-stage claim. The complete legacy
science-alignment gate is still open: ensemble tolerances and the final
AT-SD25 comparison remain separate acceptance work. See
[Stage 2 geometry, PSF, and jitter selection truth](stage2_selection_truth.md).

## Shared-Exposure Stamp Products

The optional `workload.shared_exposure_stamps` contract extracts many target
stamps from each already-rendered full-frame exposure. It does not run an
independent stamp simulation. A fresh render or deterministic recovery makes
exactly one `run_single_cadence_full_frame(...)` call for that worker-owned
frame; adding targets or products does not add calls. Every configured crop is
derived from the same in-memory result before it is released.

Use a custom run TOML to enable it:

```toml
[workload]
kind = "full-frame"

[workload.shared_exposure_stamps]
enabled = true
target_source_ids = [5853498713190525696, 5853498713190525824]
stamp_rows = 100
stamp_cols = 300
frames_per_shard = 32
product_keys = ["final_stamp", "electron_stamp"]
```

The repository includes a runnable CPU-smoke example at
[`docs/examples/et_full_frame_shared_exposure.run.toml`](examples/et_full_frame_shared_exposure.run.toml):

```bash
et-mainsim run et-full-frame \
  --preset smoke \
  --config docs/examples/et_full_frame_shared_exposure.run.toml
```

The shipped presets keep this feature disabled. `target_source_ids` must be a
non-empty, unique sequence of signed 64-bit IDs present in the authoritative
full-frame source geometry. Their configured order is preserved. The target
plan freezes each source's static base renderer position and one detector
window before the first crop. A window crossing a detector edge is retained
at the requested shape and padded with exact zeros outside the detector.

`product_keys` must include `final_stamp`. Supported values are:

- `final_stamp`, `electron_stamp`, `adu_stamp_pre_adc`, and `dn_stamp`;
- `cosmic_events.mask`;
- one-level electron components such as
  `electron_components.background_noise` (the renderer's zero-mean Poisson
  background-noise realization, not a background mean image).

A requested optional product must actually exist in the Photsim7 cadence
result; otherwise the worker fails closed. Product order is part of the run
identity. The default shape is 100 x 300 pixels, matching the frozen Stage 1
decision.

The crop/shard layer performs no stamp-specific random draw beyond those
already made by the parent exposure. Parent rendering and deterministic
reconstruction still execute the parent's RNG streams. The shards and
completion markers therefore state `zero_new_rng_draws=true` relative to the
parent and
`independent_stamp_simulation=false`. They deliberately do not claim source
truth transfer, verified target-to-parent truth association, or a scientific
parent lineage hash. File SHA-256 values in completion markers are storage and
resume guards only.

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
  shared_exposure/
    target_plan.json
    completion/frame_NNNNNNNNN.json
    shards/worker_NNNN/
      batch_NNNNNN_<batch-sha256>/
        shared-exposure-products/<plan-sha256>/<product-sha256>.h5
```

Effect files exist only when dynamic effects are enabled; worker request and log
directories exist only for the subprocess backend. Frame summaries preserve
Photsim7 fields and add an `et_mainsim` namespace for rank, source count,
elapsed time, device, and peak CUDA memory. The canonical frame-product sidecar
remains `photsim7.single_cadence_frame_products.v1`. Geometry, PSF, and
per-cadence jitter selection are persisted and strictly read back through the
Stage 2 selection sidecars when JI selection truth is active; no-JI runs carry
the explicit unavailable declaration instead.

When shared-exposure stamps are enabled, `run_manifest.json` records the root,
target plan, completion-marker directory, worker-shard directory, schemas,
target order, stamp shape, product order, and `frames_per_shard`. Each HDF5
shard spans one deterministic worker-local batch for exactly one product.
The default batch contains 32 worker-owned cadences; the last batch may be
shorter. Batching is storage/recovery topology only and does not enter RNG,
PSF, effect, or cadence-selection scope.

## Resume and Failure Semantics

- `resume=true` skips only a payload with a valid summary and frame-product
  schema whose frame index, shape, and dtype match the resolved spec.
- Partial or malformed artifacts are rerendered under resume.
- `resume=false` refuses existing frame artifacts unless `overwrite=true`.
- Resume and overwrite are mutually exclusive in one execution config.
- For a coordinator-launched shared-exposure overwrite, the coordinator removes
  exactly `run_dir/shared_exposure/` before starting either in-process or local
  subprocess workers. Parent frames, logs, manifests, catalog caches, and other
  run-directory paths are not part of that cleanup. A direct `run_worker(...)`
  overwrite refuses an existing shared-exposure bundle instead of letting
  independently launched workers race to delete or replace it.
- An existing manifest must match workflow, run id, scientific spec, and stable
  execution identity. Control flags such as resume, overwrite, progress, and
  cache refresh do not change identity.
- Each retry creates a numbered attempt. Completed and failed runs can start a
  new identity-matching attempt; a manifest already marked running rejects a
  second launcher.
- Worker exceptions mark the run failed with exception type and message.

With shared-exposure stamps enabled, parent and crop artifacts form one
completion bundle:

| Resume state | Action |
| --- | --- |
| Valid parent, final batch shards, and marker | Validate all references and let the worker skip without reading the catalog cache or building services; the public launcher still performs its catalog preparation step |
| Valid parent and final batch shards, marker missing | Deterministically reconstruct only the missing-marker frame and require exact parent and crop bytes before replacing the commit witness |
| Valid parent and a missing/partial crop | Deterministically reconstruct only the affected worker-local batch with `artifact_writer=None`, require exact parent shape/dtype/C-order bytes, then fill only missing items |
| Every parent valid and every assigned item in a partial batch COMPLETE | Reconstruct and exactly compare that batch before publishing the closed shard and markers |
| Incomplete parent with any complete crop | Fail before catalog loading or rendering |
| Existing marker, plan, shard, or storage-guard drift | Fail closed; never silently replace the artifact |

Reconstruction does not rewrite the parent NPY, summaries, selection
sidecars, cosmic mask, stellar-mean diagnostic, frame metrics, or effect
timeseries. Existing `COMPLETE` crop items are fingerprinted before opening a
writer and must exactly match the deterministic reconstruction before any
missing sibling is written. After `finalize()`, every processed item is read
back from the closed final HDF5 and compared with its in-memory fingerprint
before any marker is published. A verified final/partial hard-link crash state
is completed by removing only the stale partial alias. Writers are always
closed on exceptions.

Each batch is finalized, read back, and receives its frame markers before the
worker enters the next batch. With the default 32-cadence batch, a normal
single active-batch interruption replays at most 32 cadences for that worker,
instead of all cadences assigned to the worker.

Resume discovery and the coordinator's final completeness check also inspect
one batch at a time. Shard status snapshots, `COMPLETE`-item fingerprints,
pending crops, shard paths, and storage-guard cache entries are released when
that batch is finished; their resident count is therefore bounded by the
active batch rather than the total cadence count. Marker validation for all
assigned batches still completes before any verified linked-partial alias is
cleaned up.

Completion-marker validation reuses a caller-owned guard cache whose identity
includes device, inode, size, nanosecond modification time, and nanosecond
change time. Thus one worker batch hashes each immutable large shard once,
instead of once per frame, while any observed file-identity change forces a
rehash and drift check.

The current full-frame workflow does not yet inject source-intrinsic light
curves. Shared-exposure products inherit whatever source behavior was present
in the parent full-frame cadence; they do not add variability themselves.

Historical run directories without `run_manifest.json` are read-only. This
workflow does not claim cross-version resume compatibility with them.
