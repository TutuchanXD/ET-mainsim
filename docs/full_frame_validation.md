# Full-Frame PR 1 Validation

This file records the acceptance evidence for the installable ET full-frame
application. It distinguishes workflow/infrastructure checks from production
science evidence.

## Local Gates

Environment: `etbase`, Python 3.12, `ET_DATA_DIR` set to the maintained
Photsim7 data bundle.

| Gate | Result |
| --- | --- |
| ET-mainsim full test suite | 121 passed; four pre-existing multiprocessing fork deprecation warnings |
| Ruff check for new package/tests/evaluator | passed |
| Ruff format check for new package/tests/evaluator | passed |
| ShellCheck for maintained and benchmark Slurm scripts | passed |
| `compileall` | passed |
| PEP 517 wheel build | `et_mainsim-0.1.0-py3-none-any.whl` built |
| Wheel content audit | package modules plus both JSON/TOML presets and smoke CSV present |
| Clean wheel-target import and `presets` | passed outside the repository working directory |
| Installed-wheel CPU smoke | completed; frame/summary/schema read back |
| In-process resume | completed with one validated skip |
| In-process overwrite | completed with one rerender |
| Local subprocess CPU smoke | completed with worker request/log/result artifacts |

## Production Catalog Contract

The production `--prepare-catalog-only` command ran against the real local Gaia
catalog and ET focal-plane registry, then ran a second time against the exact
cache:

- status: `completed`;
- source count: `1,830,321` for G < 18;
- detector: `main_rd`, shape `[9120, 8900]`;
- request schema: `photsim7.catalog_request.v1`;
- target epoch: J2000 (`2000.0`);
- magnitude contract: `Gaia_G` input, `ET` photon system,
  `gaia_g_vega_equals_et_ab_g2v_approx` conversion;
- changing the same run id to epoch `2028.5` was rejected as a scientific-spec
  identity conflict before cache reuse.

## H100 Gates

Both jobs used `etbase-clu`, the current local Photsim7 source snapshot, Slurm,
and SSHFS outputs. Job 202664 used an exact hash-matched copy of the final PR 1
ET-mainsim source; job 202662 used the same unchanged worker science path for
the full-shape gate. Both passed independent artifact readback and left no
worker, resource-tracker, compute process, or GPU memory allocation behind.

### Job 202664: final-source CLI CUDA smoke

- Slurm: `COMPLETED`, exit `0:0`, elapsed `00:00:26`;
- allocation: one H100, 8 CPUs, 64 GiB;
- backend: `local-subprocess`, device `cuda`;
- frame: `(64, 64)`, `uint16`;
- schema: `photsim7.single_cadence_frame_products.v1`;
- Photsim7 pipeline: `0.366 s`;
- peak CUDA allocated/reserved: `0.205/2.0 MiB`.

### Job 202662: current dependency full-shape gate

- Slurm: `COMPLETED`, exit `0:0`, elapsed `00:00:18`;
- allocation: one H100, 16 CPUs, 128 GiB;
- frame: `(9120, 8900)`, `uint16`, `162,336,000` bytes;
- Photsim7 pipeline: `1.223 s`;
- peak CUDA allocated/reserved: `4030.04/4032.0 MiB`;
- independent manifest/frame/schema readback: passed.

This full-shape gate deliberately uses the smoke scene: one packaged star,
fixed PSF field, and disabled dynamic effects, cosmic rays, and detector
response. It validates the new application's full-detector scheduling, memory,
artifact, and readback path; it is not presented as a production Gaia science
simulation. Historical realistic full-frame job 202658 remains useful context,
but is not counted as current-dependency acceptance evidence.

The remote Photsim7 import emitted existing invalid-escape `SyntaxWarning`
messages from `plot.py`; these did not affect either job and are outside this
ET-mainsim PR.
