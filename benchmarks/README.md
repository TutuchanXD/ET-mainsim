# Benchmarks

Benchmark workers use the production `SimulationSpec` and the same strict
catalog request validation as normal runs. Legacy `MainRdRunSpec` NPZ caches
do not contain a canonical request and are intentionally rejected.

Prepare one reusable cache before submitting a load test:

```bash
python -m benchmarks.run_full_frame_thermal_load \
  --output-root /cluster/home/cxgao/sshfs-share/slurm_validation/cache-builds \
  --run-id catalog-g17 \
  --catalog-cache /cluster/home/cxgao/sshfs-share/slurm_validation/et-mainsim-final-canonical-caches/g_lt_17/stars.npz \
  --mag-limit 17 \
  --frames 1 \
  --jitter-models 100 \
  --seed 20260714 \
  --prepare-catalog-only
```

Cache preparation requires `GAIA_CATALOG_DIR`, `ET_FOCALPLANE_ROOT`, and
`ET_DATA_DIR`. Rendering can then run on a node without the Gaia shards. The
performance pilot consumes G < 16, 16.5, 17, 17.5, and 18 caches under the same
root. The 600 W reproducer consumes the G < 17 cache and refuses to start when
the requested cache or 600 W GPU power limit is absent.
