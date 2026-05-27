# Synthetic 500x500 ET-Mag Distribution Simulation Parameter Checklist

This checklist records the enabled parameters for:

- `simulate_main_rd_500x500_magdist_g23_colnoise0.py`

Do not treat this file as a new parameter source. The executable defaults are
defined in `main_rd_parallel_core.py` and the entrypoint override block in
`simulate_main_rd_500x500_magdist_g23_colnoise0.py`.

## Entry Point

```bash
conda activate etbase
cd /home/cxgao/ET/ET-mainsim/main_rd_g18_parallel

python simulate_main_rd_500x500_magdist_g23_colnoise0.py
```

For the 3-day run, use six workers across the two local GPUs:

```bash
python simulate_main_rd_500x500_magdist_g23_colnoise0.py \
  --frames 25920 \
  --gpus 0,1 \
  --workers-per-gpu 3 \
  --preview-count 2 \
  --overwrite
```

Default output directory:

```text
/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel/main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0
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
| Run label | `main_rd_500x500_magdist_310-50-2420_g_lt_23_colnoise0` |

## Synthetic Star Catalog

| Parameter | Value |
| --- | --- |
| Star source | Synthetic magnitude distribution |
| Magnitude asset | `/home/cxgao/ET/Photsim7-data/ET_mag/310-50-2420.csv` |
| Magnitude column | `mwmsc_gmag` |
| Magnitude interpretation | Gaia G is treated as ET magnitude |
| Magnitude cut | `mwmsc_gmag <= 23.0` |
| Boundary inclusion | Includes exactly `23.0` |
| Stars selected from current asset | `37685` |
| Position model | Uniform independent random x/y centers within `500 x 500` |
| Position reuse | Cached once; all frames use the same synthetic star field |
| Star overlap | Allowed |
| Edge truncation | Allowed |
| Synthetic source IDs | Sequential `0..N-1` |

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

These settings match the existing `1000 x 1000` full-effect script.

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
| Sky surface brightness | `21 mag/arcsec^2` |
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

## Saved Products

| Product | Status |
| --- | --- |
| Synthetic star cache | Saved under the run directory |
| Frame image arrays | Saved as `.npy` |
| Frame summaries | Saved as JSON |
| Column noise vectors | Saved by default and should be all zeros |
| Cosmic ray event payloads | Saved |
| Cosmic ray masks | Disabled by default, can be enabled with `--save-cosmic-mask` |
| Stellar mean images | Disabled by default, can be enabled with `--save-stellar-mean` |
| Effect time series | Saved as `effects_timeseries.npz` |
| Preview PNGs | First `2` frames by default |
