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
remains `photsim7.single_cadence_frame_products.v1`.

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
