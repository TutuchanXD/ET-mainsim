# main_rd Parallel Full-Effect Simulations

This directory contains parallel ET `main_rd` simulation entry points backed by
`main_rd_parallel_core.py`. The scripts share the same detector/electronics and
motion-effect implementation; each entry point only overrides the source field,
image shape, sky background, subpixel grid, frame count, or scattered-light
schedule stated below.

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
| PSD motion | enabled, `/home/cxgao/ET/photsim6_cache/ET_psd3-2.pkl` |
| Jitter-integrated PSF | enabled |
| DVA drift | enabled, field angle/theta `12 deg` |
| Thermal drift | enabled |
| Momentum dumps | enabled |
| WEED PSF breathing | enabled |

The PSD motion is split at the single-frame cadence. Motion slower than the
`10 s` cadence is applied as frame-to-frame centroid drift; faster motion is
integrated into PSF models.

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
| `500x500_detectorxy` | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv` | `17779` | `gmag` is used directly as ET magnitude; `x0,y0` are detector-centered pixel coordinates and are not rounded, randomized, or reprojected. |
| `700x700_detectorxy` | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square700pix_glt24_detector_xy.csv` | `36459` | Same convention. |

The detailed parameter checklist for the three scattered-light branches is kept
in `DETECTORXY_STRAYLIGHT_BRANCHES_PARAMETER_CHECKLIST.md`.

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
