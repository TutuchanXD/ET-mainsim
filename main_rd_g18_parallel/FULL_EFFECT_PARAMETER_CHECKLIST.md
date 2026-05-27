# Main RD G<18 Full-Effect Simulation Parameter Checklist

This checklist records the enabled parameters for the two main detector
simulation entrypoints in this directory:

- `simulate_main_rd_1000x1000_g18.py`
- `simulate_main_rd_8900x9120_g18.py`

Do not treat this file as a new parameter source. The executable defaults are
defined in `main_rd_parallel_core.py`; this document is for manual review before
running production simulations.

## Entry Points

```bash
conda activate etbase
cd /home/cxgao/ET/ET-mainsim/main_rd_g18_parallel

python simulate_main_rd_1000x1000_g18.py
python simulate_main_rd_8900x9120_g18.py
```

Default output root:

```text
/home/cxgao/Results/ET-mainsim/main_rd_g18_parallel
```

## Common Run Settings

| Parameter | Value |
| --- | --- |
| Detector | `main_rd` |
| Magnitude cut | `G < 18` |
| Number of frames | `180` |
| Exposure time | `10 s` |
| Total simulated duration | `1800 s` |
| Default GPUs | `0,1` |
| Default workers per GPU | `1` |
| Random seed | `20260516` |
| Preview frames | `2` |
| Star catalog directory | `/home/cxgao/gaia_dr3_19mag` |

## Image Sizes

| Script | Frame rows | Frame columns |
| --- | ---: | ---: |
| `simulate_main_rd_1000x1000_g18.py` | `1000` | `1000` |
| `simulate_main_rd_8900x9120_g18.py` | `9120` | `8900` |

## Field Center

| Parameter | Value |
| --- | --- |
| RA center | `304.41406499712303 deg` |
| Dec center | `51.81987707392268 deg` |
| Field x | `-6.10175 deg` |
| Field y | `-6.23275 deg` |
| Detector x center | `4450 pix` |
| Detector y center | `4560 pix` |

## Optical And PSF Settings

| Parameter | Value |
| --- | --- |
| Pixel scale | `4.83 arcsec/pix` |
| Pixel width | `10 um` |
| PSF bundle | `241006/D280mm-focus` |
| Field-dependent PSF | Enabled |
| Subpixel sampling | `3` |

## Jitter And Frame Motion

| Parameter | Value |
| --- | --- |
| PSD input | `/home/cxgao/ET/photsim6_cache/ET_psd3-2.pkl` |
| Split frequency | `0.1 Hz` |
| Low-frequency motion | `<= 0.1 Hz`, applied as frame-to-frame centroid drift |
| High-frequency motion | `> 0.1 Hz`, integrated into PSF models |
| Jitter-integrated PSF | Enabled |
| Number of jitter PSF models | `300` |
| Samples per jitter PSF model | `600` |
| Attitude axes used | PSD `x`, `y`, and `z` |
| Projection field angle | `10 deg` |
| Projection x-axis angle | `45 deg` |

## Dynamic Effects

| Effect | Status | Parameters |
| --- | --- | --- |
| PSD drift | Enabled | low-frequency x/y/z attitude PSD projected to focal-plane x/y |
| DVA drift | Enabled | `ET_DVA_effect_models_slim_v231117.pkl`, `psf_field_angle=12 deg`, `theta=12 deg`, `t0=0 day` |
| Thermal drift | Enabled | amplitude `0.022 arcsec`, baseline step `0.03 arcsec / 3 day`, `4 cycles / 3 day`, theta `12 deg` |
| Momentum dumps | Enabled | `random_walk_within_circle`, cycle `3 day`, R68 `0.15 arcsec` |
| WEED PSF breathing | Enabled | period `3 day`, PSF scale `0.99 -> 1.01` |

For this 180-frame, 10-second cadence run, the 3-day momentum-dump period means
the jump component is expected to be effectively zero over the 30-minute
duration, while the effect path remains enabled.

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
| Column noise sigma | `5 ADU` |
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
| Expected events, 1000x1000, 10 s | about `50 events/frame` |
| Expected events, 8900x9120, 10 s | about `4058 events/frame` |

## Pixel Response

| Parameter | Value |
| --- | --- |
| Pixel-to-pixel response variation | Enabled, `1%` |
| Intra-pixel response variation | Enabled, `1%` |
| Flat-field correction | Disabled |

## Saved Products

| Product | Status |
| --- | --- |
| Frame image arrays | Saved as `.npy` |
| Frame summaries | Saved as JSON |
| Column noise vectors | Saved by default |
| Cosmic ray event payloads | Saved |
| Cosmic ray masks | Disabled by default, can be enabled with `--save-cosmic-mask` |
| Stellar mean images | Disabled by default, can be enabled with `--save-stellar-mean` |
| Effect time series | Saved as `effects_timeseries.npz` |
| Preview PNGs | First `2` frames by default |

## Explicitly Disabled By Current Requirement

| Effect | Reason |
| --- | --- |
| Scattered light | User requested current version not to add scattered light |
| Flat-field correction | User requested no flat-field / flat correction for this run |
| Compression/transit experiment logic | Not part of the current image simulation script |

## Useful Overrides

```bash
# Check configuration and star cache path without rendering frames.
python simulate_main_rd_1000x1000_g18.py --dry-run

# Use only one GPU.
python simulate_main_rd_1000x1000_g18.py --gpus 0

# Regenerate the star cache.
python simulate_main_rd_1000x1000_g18.py --force-star-cache

# Rerun into an existing output directory.
python simulate_main_rd_1000x1000_g18.py --overwrite
```
