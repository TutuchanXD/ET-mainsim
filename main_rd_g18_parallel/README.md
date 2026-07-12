# main_rd Parallel Full-Effect Simulations

This directory contains parallel ET `main_rd` orchestration entry points backed
by `main_rd_parallel_core.py`. Active workers convert `MainRdRunSpec` once to a
Photsim7 `SimulationSpec`, build reusable `FullFrameServices`, and render each
cadence through `run_single_cadence_full_frame(...)`. The scripts only select
the source field, image shape, sky background, subpixel grid, frame count, and
project-specific output schedule.

Outputs default to:

```text
/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel/
```

Use the `etbase` environment:

```bash
conda activate etbase
cd /home/cxgao/ET/ET-mainsim/main_rd_g18_parallel
```

## Common Full-Effect Settings

Unless a script says otherwise, the shared settings are:

| Parameter | Value |
| --- | --- |
| Detector | `main_rd` |
| Exposure | `10 s` |
| Pixel size | `10 um` |
| Pixel scale | `4.83 arcsec/pix` |
| Optical efficiency | `58%` |
| ET quantum efficiency | `80%` |
| Dark current | `1 e-/s/pix` |
| Inter-pixel response RMS | `1%` |
| Intra-pixel response RMS | `1%` |
| Flat-field correction | disabled |
| Full well | `90680 e-` |
| Gain | `1.4 e-/ADU` |
| ADC | `16 bit`, rounded to integer DN |
| Bias | `3500 ADU` |
| Readout noise | `6 e-/pix`, applied after full-well clipping and before gain |
| Cosmic rays | enabled, `5 events cm^-2 s^-1`, 10 um event library |
| PSD motion | enabled, `ET_DATA_DIR/pds/ET_psd3-2.pkl` |
| Jitter-integrated PSF | enabled |
| DVA drift | enabled, field angle/theta `12 deg` |
| Thermal drift | enabled |
| Momentum dumps | enabled |
| WEED PSF breathing | enabled |

The PSD motion is split at the single-frame cadence. Motion slower than the
`10 s` cadence is projected from its native spacecraft-attitude frame into a
renderer `(x, y)` offset for each source; faster motion is integrated into PSF
models. DVA and thermal terms remain radial focal-plane components until the
same per-cadence projection. The saved `effects_timeseries.npz` therefore holds
compact native components, with its schema in
`effects_timeseries.metadata.json`.

ET photon rates use the accepted approximation
`et_mag (AB) = gaia_g_mag (Vega)` and the ET report calibration with explicit
`58%` optical efficiency and `80%` QE. The old main-rd path advertised an
invalid `101%` efficiency while separately passing `1.0` to its star builder,
so old products do not have one trustworthy global correction ratio. Relative
to the advertised `101%` scalar, the new throughput factor is
`0.58 * 0.80 / 1.01 = 0.4594`; relative to a path that already applied `80%` QE
but used unity optical efficiency, it is `0.58`.

## Script Index

| Script | Image | Source field | Frames | Sky | Subpixels | Column noise | Scattered light |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `simulate_main_rd_1000x1000_g18.py` | `1000 x 1000` | Gaia around `main_rd`, `G<18` | `180` | `21 mag/arcsec^2` | `3 x 3` | `5 ADU` | disabled |
| `simulate_main_rd_8900x9120_g18.py` | `8900 x 9120` | Gaia around `main_rd`, `G<18` | `180` | `21 mag/arcsec^2` | `3 x 3` | `5 ADU` | disabled |
| `simulate_main_rd_500x500_magdist_g23_colnoise0.py` | `500 x 500` | random positions from `310-50-2420.csv`, `G<=23` | `180` | `21 mag/arcsec^2` | `3 x 3` | `0 ADU` | disabled |
| `simulate_main_rd_500x500_magdist_g24_sky23p2_colnoise0.py` | `500 x 500` | random positions from `310-50-2420.csv`, `G<=24` | `180` | `23.2 mag/arcsec^2` | `3 x 3` | `0 ADU` | disabled |
| `simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py` | `500 x 500` | detector-xy CSV, `G<=24` | `180` | `23.2 mag/arcsec^2` | `3 x 3` | `0 ADU` | disabled |
| `simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_straylight10_last90.py` (legacy filename) | `500 x 500` | detector-xy CSV, `G<=24` | `270` | `22 mag/arcsec^2` | `1 x 1` | `0 ADU` | frames `180..269`: `10 e-/pix/frame` |
| `simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py` | `500 x 500` | detector-xy CSV, `G<=24` | `270` | `23.2 mag/arcsec^2` | `5 x 5` | `0 ADU` | frames `180..269`: `10 e-/pix/frame` |
| `simulate_main_rd_700x700_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py` | `700 x 700` | detector-xy 700 CSV, `G<=24` | `270` | `23.2 mag/arcsec^2` | `5 x 5` | `0 ADU` | frames `180..269`: `10 e-/pix/frame` |

Detector-xy CSV conventions:

| Script family | CSV | Rows | Coordinate/magnitude handling |
| --- | --- | --- | --- |
| `500x500_detectorxy` | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv` | `17779` | `gmag` is Gaia G and enters ET photometry through the current numeric-equality approximation; `x0,y0` are detector-centered frame offsets and are not rounded, randomized, or reprojected. |
| `700x700_detectorxy` | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square700pix_glt24_detector_xy.csv` | `36459` | Same convention. |

The detailed parameter checklist for the three scattered-light branches is kept
in `DETECTORXY_STRAYLIGHT_BRANCHES_PARAMETER_CHECKLIST.md`.

## Output Contract

Legacy NPY output remains the default:

```text
frames/frame_000000.npy
cosmic_events/frame_000000_events.npy
bias/frame_000000_column_noise_adu.npy
frame_summaries/frame_000000.json
preview/frame_000000.png
```

Every new frame also has
`frame_summaries/frame_000000_schema.json` with schema id
`photsim7.single_cadence_frame_products.v1`, units, domains, shape, coordinate
convention, RNG trace, and package provenance. Current main-rd wrappers do not
select packed HDF5 output; NPY compatibility remains authoritative for these
runs. `export_last90_truth_tables.py` reads both the historical global-offset
format and the package native-component format. New-format truth is projected
per source and uses typed ET photometry.

## Ownership Boundary

Photsim7 owns catalog normalization/cache schema, magnitude-to-electron-rate
conversion, projector selection, dynamic effects, PSF construction, detector
response, detector electronics, cosmic injection, SeedTree streams, and frame
product schemas. ET-mainsim retains CLI parsing, run labels, worker/GPU process
scheduling, resume selection, scattered-light experiment schedules, and legacy
path organization.

The local PSF, renderer, and electronics helpers remain only as one-cycle
compatibility code. `build_full_effect_timeseries(...)` and
`jitter_integrated_psf_offsets(...)` cannot yet be removed because the
out-of-scope `stamp_long` workflow still imports them; active `main_rd` workers
do not call any of these helpers.

## Scattered-Light Branch Definition

For the three `straylight10_last90` scripts:

| Frame range | Added scattered light |
| --- | --- |
| `0..179` | `0 e-/pix/frame` |
| `180..269` | `10 e-/pix/frame` |

Internally this is passed to Photsim7 as `1.0 e-/s/pix`, because the cadence is
`10 s`.

## Smoke Test Commands

The `--frame-indices` option renders only selected frame numbers while keeping
the script's configured total frame count. It is useful for validating frame 0
and frame 180 without rendering all 270 frames.

```bash
SMOKE_ROOT=/home/cxgao/Results/ET-mainsim/main_rd_detectorxy_straylight_smoke

python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_straylight10_last90.py \
  --output-root "$SMOKE_ROOT" \
  --frames 270 \
  --frame-indices 0,180 \
  --preview-count 181 \
  --gpus 0,1 \
  --workers-per-gpu 1

python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py \
  --output-root "$SMOKE_ROOT" \
  --frames 270 \
  --frame-indices 0,180 \
  --preview-count 181 \
  --gpus 0,1 \
  --workers-per-gpu 1

python simulate_main_rd_700x700_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py \
  --output-root "$SMOKE_ROOT" \
  --frames 270 \
  --frame-indices 0,180 \
  --preview-count 181 \
  --gpus 0,1 \
  --workers-per-gpu 1
```

## Production Commands

Use six workers across two local GPUs:

```bash
python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_straylight10_last90.py \
  --gpus 0,1 \
  --workers-per-gpu 3

python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py \
  --gpus 0,1 \
  --workers-per-gpu 3

python simulate_main_rd_700x700_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py \
  --gpus 0,1 \
  --workers-per-gpu 3
```

`--frames 270` is optional for these three scripts because it is already their
default.

## Useful Options

Preview star cache and frame assignment without rendering:

```bash
python simulate_main_rd_700x700_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py \
  --dry-run
```

Run a small control on one GPU:

```bash
python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py \
  --frames 2 \
  --gpus 0
```

Disable selected effects for control runs:

```bash
python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py \
  --no-psd-motion \
  --no-dva-drift \
  --no-thermal-drift \
  --no-momentum-dump \
  --no-psf-breathing
```
