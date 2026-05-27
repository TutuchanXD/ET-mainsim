# Detector-XY 500x500 Sky 23.2 Parameter Checklist

This checklist records the enabled parameters for:

- `simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py`

This script is a branch of the current `500 x 500 / sky=23.2 / column-noise=0`
simulation. The only intended science change is the source table:

- It uses `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv`.
- It renders every row in that CSV.
- CSV `gmag` values are used directly as ET magnitudes.
- CSV `x0` and `y0` are used directly as detector-centered pixel coordinates.

No rounding, integer casting, reprojection, or random placement is applied to
the input coordinates.

## Entry Point

```bash
conda activate etbase
cd /home/cxgao/ET/ET-mainsim/main_rd_g18_parallel

python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py
```

For the 3-day run, use six workers across the two local GPUs:

```bash
python simulate_main_rd_500x500_detectorxy_sky23p2_colnoise0.py \
  --frames 25920 \
  --gpus 0,1 \
  --workers-per-gpu 3
```

Default output directory:

```text
/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel/main_rd_500x500_detectorxy_310-50-2420_sky23p2_colnoise0
```

## Common Run Settings

| Parameter | Value |
| --- | --- |
| Detector label | `main_rd` |
| Frame size | `500 x 500` pixels |
| Number of frames | `180` |
| Exposure time | `10 s` |
| Total simulated duration | `1800 s` |
| Default GPUs | `0,1` |
| Default workers per GPU | `1` |
| Recommended long-run workers per GPU | `3` |
| Random seed | `20260516` |
| Preview frames | `2` |
| Run label | `main_rd_500x500_detectorxy_310-50-2420_sky23p2_colnoise0` |

## Source Catalog

| Parameter | Value |
| --- | --- |
| Star source | Detector-centered CSV |
| CSV path | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420_square_detector_xy.csv` |
| Source ID column | `source_id` |
| Magnitude column | `gmag` |
| Magnitude interpretation | `gmag` is used directly as ET magnitude |
| x coordinate column | `x0` |
| y coordinate column | `y0` |
| Coordinate interpretation | Detector-centered pixel coordinates, image center is `(0, 0)` |
| Rendered rows | All rows |
| Stars in current CSV | `17779` |
| gmag range in current CSV | `5.8161435` to `23.999893` |
| x0 range in current CSV | `-249.97214748526392` to `249.95073493403945` |
| y0 range in current CSV | `-249.9070033361152` to `249.9937805584423` |
| Rounding / integer cast | Disabled |
| Reprojection | Disabled |
| Random placement | Disabled |
| Edge truncation | Allowed |

## Optical And PSF Settings

| Parameter | Value |
| --- | --- |
| Pixel scale | `4.83 arcsec/pix` |
| Pixel width | `10 um` |
| PSF bundle | `241006/D280mm-focus` |
| PSF field angle | Fixed `12 deg` |
| Field-dependent PSF | Enabled, selected from fixed `12 deg` field angle |
| Subpixel sampling | `3` |

## Jitter And Dynamic Effects

These settings are unchanged from the current 500x500 full-effect script.

| Parameter | Value |
| --- | --- |
| PSD input | `/home/cxgao/ET/photsim6_cache/ET_psd3-2.pkl` |
| Split frequency | `0.1 Hz` |
| Low-frequency motion | `<= 0.1 Hz`, applied as frame-to-frame centroid drift |
| High-frequency motion | `> 0.1 Hz`, integrated into PSF models |
| Jitter-integrated PSF | Enabled |
| Number of jitter PSF models | `300` |
| Samples per jitter PSF model | `600` |
| DVA drift | Enabled |
| Thermal drift | Enabled |
| Momentum dumps | Enabled |
| WEED PSF breathing | Enabled |

## Detector And Electronics

| Parameter | Value |
| --- | --- |
| Sky surface brightness | `23.2 ET mag/arcsec^2` |
| Sky background rate | about `13.59 e-/s/pix` |
| Sky background per 10s frame | about `135.9 e-/pix` |
| Dark current | `1.0 e-/s/pix` |
| Scattered light | `0.0 e-/s/pix` |
| Full well | `90680 e-` |
| Gain | `1.4 e-/ADU` |
| Readout noise | `6 e-/pix` |
| Bias | `3500 ADU` |
| Column noise sigma | `0 ADU` |
| ADC bit depth | `16` |
| ADC clip range | `0..65535` |
| Output type | `round -> uint16` |

## Cosmic Rays

| Parameter | Value |
| --- | --- |
| Cosmic rays | Enabled |
| Event rate | `5 events cm^-2 s^-1` |
| Event library | `Photsim7-data/cosmic_ray/dark_test_10um/event_library_10um.npz` |
| Event pixel size | `10 um` |
| Expected events, 500x500, 10 s | about `12.5 events/frame` |

## Pixel Response

| Parameter | Value |
| --- | --- |
| Pixel-to-pixel response variation | Enabled, `1%` |
| Intra-pixel response variation | Enabled, `1%` |
| Flat-field correction | Disabled |
