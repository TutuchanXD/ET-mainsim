# ET-mainsim

ET-mainsim preserves and maintains the ET transit-telescope main-detector
simulation scripts that were previously kept in the legacy `Photosim6ft`
checkout. The scripts generate cropped ET main-detector image simulations for
transit-telescope detector fields, with selectable effect profiles for noise,
pointing motion, DVA, thermal drift, and subpixel response terms.

Source recovery:

- Original repository: <https://github.com/TutuchanXD/Photosim6ft>
- Recovered source commit: `1007481`
- Recovered script family: `et_sim_100*.py`

The legacy `Photosim6ft` repository can be archived after this repository is in
place. Runtime code here now defaults to the local Photosim7 checkout instead of
the removed local `Photosim6ft` directory.

## Contents

| Path | Purpose |
| --- | --- |
| `scripts/et_sim_100_det.py` | Main ET transit main-detector simulation driver. |
| `scripts/et_sim_100_det_v1_noise_psf.py` | Wrapper for the `v1_noise_psf` profile. |
| `scripts/et_sim_100_det_v2_point_drift_jitter.py` | Wrapper for the `v2_point_drift_jitter` profile. |
| `scripts/et_sim_100_det_v3_dva.py` | Wrapper for the `v3_dva` profile. |
| `scripts/et_sim_100_det_v4_thermal.py` | Wrapper for the `v4_thermal` profile. |
| `scripts/et_sim_100_det_v5_prv_subpixel.py` | Wrapper for the `v5_prv_subpixel` profile. |
| `config/et_100_det_inputs_1h.xlsx` | Baseline ET main-detector runtime spreadsheet recovered with the scripts. |
| `main_rd_g18_parallel/` | Current parallel Photsim7-based `main_rd` full-effect simulation scripts, including detector-coordinate 500x500/700x700 sky23.2 scattered-light branches. |

## Runtime Inputs

The scripts expect the Photosim7 source checkout and the Photosim6 data bundle to
exist locally:

| Variable | Purpose | Default |
| --- | --- | --- |
| `PHOTSIM7_ROOT` | Local Photosim7 checkout root | `/home/cxgao/ET/Photosim7` |
| `ET_DATA_DIR` | Photosim6 data root | `/home/cxgao/ET/Photosim6/data` |
| `ET_CONFIG_XLSX` | Main-detector configuration spreadsheet | `config/et_100_det_inputs_1h.xlsx` |
| `ET_EFFECT_PROFILE` | Effect profile selected by `et_sim_100_det.py` | `full` |
| `ET_PROFILE_TARGET_FRAMES` | Optional frame-count override for quick runs | unset |
| `ET_RUN_ALL_BATCHES` | Run all configured sky-center batches | script default |
| `ET_FIELD_CENTER_INDEX` | Select one configured field center | script default |
| `ET_OUTPUT_RUN_NAME_OVERRIDE` | Override the output run directory name | profile dependent |

Outputs default to `/home/cxgao/ET/FSG_images_sims/<run-name>/...`.

## Effect Profiles

The main driver defines one full profile plus five focused profiles. The wrapper
scripts set `ET_EFFECT_PROFILE`, reduce the run to 20 target frames where
appropriate, and set a deterministic output run name.

| Profile | Wrapper | Enabled effects beyond the static image model |
| --- | --- | --- |
| `full` | Run `et_sim_100_det.py` directly | Package-default variant settings, pointing drift, DVA, thermal drift, momentum dump, and jitter-integrated PSF. |
| `v1_noise_psf` | `et_sim_100_det_v1_noise_psf.py` | Background light, scattered light, dark current, readout noise, target star, background stars, and static PSF. Stellar photon noise and gain are disabled in the script baseline. |
| `v2_point_drift_jitter` | `et_sim_100_det_v2_point_drift_jitter.py` | The v1 baseline plus pointing drift and jitter-integrated PSF. Slow motion is kept as frame-to-frame drift and high-frequency motion is used for JI-PSF blurring. |
| `v3_dva` | `et_sim_100_det_v3_dva.py` | The v1 baseline plus DVA drift only. |
| `v4_thermal` | `et_sim_100_det_v4_thermal.py` | The v1 baseline plus the ET/TESS thermal drift model. |
| `v5_prv_subpixel` | `et_sim_100_det_v5_prv_subpixel.py` | The v1 baseline plus inter-pixel response, intra-pixel response, and pixel-phase response terms. |

## Typical Usage

Run a focused smoke-scale profile:

```bash
cd /home/cxgao/ET/ET-mainsim
python scripts/et_sim_100_det_v1_noise_psf.py
```

Run the full profile:

```bash
cd /home/cxgao/ET/ET-mainsim
ET_RUN_ALL_BATCHES=1 python scripts/et_sim_100_det.py
```

Override the Photosim7 checkout or use a different spreadsheet:

```bash
PHOTSIM7_ROOT=/home/cxgao/ET/Photosim7 \
ET_CONFIG_XLSX=/home/cxgao/ET/ET-mainsim/config/et_100_det_inputs_1h.xlsx \
python scripts/et_sim_100_det_v3_dva.py
```

## Notes

- The driver still imports Photosim internals through the `photsim6` compatibility
  package name because the current Photosim7 source tree keeps that API surface.
- The recovered scripts are intentionally kept close to their original
  experiment form. Future cleanup should prefer small PRs that preserve the
  numerical behavior of each effect profile.
