# Detector-XY Straylight Branches Parameter Checklist

These scripts branch from:

- `simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py`

The shared definition of the added straylight is:

- Frames `0..179`: no added scattered light.
- Frames `180..269`: add `10 e-/pix/frame`.
- Because the cadence is `10 s`, this is implemented as `1.0 e-/s/pix`.

All branches retain:

- Sky background: `23.2 ET mag/arcsec^2`
- Column noise: `0 ADU`
- Pixel scale: `4.83 arcsec/pix`
- Pixel width: `10 um`
- PSF bundle: `241006/D280mm-focus`
- PSF field angle: fixed `12 deg`
- Jitter-integrated PSF: enabled
- PSD drift, DVA drift, thermal drift, momentum dumps, WEED PSF breathing: enabled
- Cosmic rays: enabled at `5 events cm^-2 s^-1`
- Pixel-to-pixel response variation: `1%`
- Intra-pixel response variation: `1%`
- Flat-field correction: disabled
- Scattered light baseline: `0 e-/s/pix`

## Branch 1

Script:

```text
simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_straylight10_last90.py  # legacy filename; configured sky=22, subpix=1
```

Run label:

```text
main_rd_500x500_detectorxy_310-50-2420_sky22_colnoise0_subpix1_straylight10_last90
```

| Parameter | Value |
| --- | --- |
| Frame size | `500 x 500` |
| Frames | `270` |
| Exposure | `10 s` |
| Source CSV | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv` |
| Source rows | `17779` |
| Magnitude column | `gmag`, used directly as ET magnitude |
| Coordinate columns | `x0`, `y0`, used directly as detector-centered pixel coordinates |
| Subpixel grid | `3 x 3` |
| Added scattered light | frames `180..269`: `10 e-/pix/frame` |

## Branch 2

Script:

```text
simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py
```

Run label:

```text
main_rd_500x500_detectorxy_310-50-2420_sky23p2_colnoise0_subpix5_straylight10_last90
```

| Parameter | Value |
| --- | --- |
| Frame size | `500 x 500` |
| Frames | `270` |
| Exposure | `10 s` |
| Source CSV | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv` |
| Source rows | `17779` |
| Magnitude column | `gmag`, used directly as ET magnitude |
| Coordinate columns | `x0`, `y0`, used directly as detector-centered pixel coordinates |
| Subpixel grid | `5 x 5` |
| Added scattered light | frames `180..269`: `10 e-/pix/frame` |

## Branch 3

Script:

```text
simulate_main_rd_700x700_detectorxy_sky23p2_colnoise0_subpix5_straylight10_last90.py
```

Run label:

```text
main_rd_700x700_detectorxy_310-50-2420_sky23p2_colnoise0_subpix5_straylight10_last90
```

| Parameter | Value |
| --- | --- |
| Frame size | `700 x 700` |
| Frames | `270` |
| Exposure | `10 s` |
| Source CSV | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square700pix_glt24_detector_xy.csv` |
| Source rows | `36459` |
| Magnitude column | `gmag`, used directly as ET magnitude |
| Coordinate columns | `x0`, `y0`, used directly as detector-centered pixel coordinates |
| Subpixel grid | `5 x 5` |
| Added scattered light | frames `180..269`: `10 e-/pix/frame` |

## Parallel Run Template

Use six workers across the two local GPUs:

```bash
python <script>.py \
  --gpus 0,1 \
  --workers-per-gpu 3
```

The scripts default to `270` frames, so `--frames 270` is optional.
