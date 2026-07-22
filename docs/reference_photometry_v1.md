# Reference photometry v1

`et_mainsim.reference_photometry` supplies the first standard reduction for
formal stamp-delivery HDF5 bundles.  It is deliberately conservative:
the only detector observation is `final_dn`; every electron-domain quantity
emitted by this reducer is a calibration-derived analysis product.

## Required bundle contract

The selected HDF5 group (the root by default) must contain these datasets.

| Dataset | Shape | Unit / meaning |
| --- | --- | --- |
| `final_dn` | `(n_cadence, ny, nx)` | Physical observed detector product in DN. |
| `background_expectation_e` | `(n_cadence, ny, nx)` | Expected background in electrons; not a noise realization. |
| `bias_level_sum_dn` | scalar, `(n_cadence,)`, or `(n_cadence, 1, 1)` | Bias added to the corresponding stored cadence. |
| `column_noise_sum_dn_by_x` | scalar, `(nx,)`, `(n_cadence, nx)`, or `(n_cadence, 1, nx)` | Column component added to the corresponding stored cadence. |
| `valid_mask` | `(n_cadence, ny, nx)` | Valid-pixel mask. |
| `saturated_mask` | `(n_cadence, ny, nx)` | Saturation mask. |
| `cosmic_mask` | `(n_cadence, ny, nx)` | Cosmic-affected-pixel mask. |
| `time_index` | `(n_cadence,)` | Absolute cadence start coordinate. |

The group or root attributes must also include `gain_e_per_dn` and
`time_index_unit`.  The accepted time units are `frame_index` and `seconds`.
For `frame_index`, `raw_frame_seconds` is mandatory.  An optional
`exposure_seconds` or `coadd_exposure_seconds` scalar/dataset provides the
actual accumulated exposure in each delivered cadence; it is required for a
single-cadence CDPP calculation and strongly recommended for all formal
deliveries.

`time_index` represents the **start** of each integrated interval, not a
display-only index or an unqualified midpoint.  It is absolute from the run
origin, so time-sharded products retain common 30/90/390-minute CDPP bin
boundaries.

正式 `et_mainsim.stamp_delivery_bundle.v1` 不使用历史的 `time_index` 字段，
而是使用 `time_start_seconds` 与 `exposure_seconds`。请调用
`reduce_stamp_delivery_bundle_v1(path)`，它会严格读取该正式 schema 并经由
已验证的 adapter 建立本模块输入；不要把正式 bundle 传给
`reduce_reference_photometry_bundle_v1()`，后者只服务于旧式 composite HDF5
格式。

对于一个目标跨多个连续 time shard 的正式生产，请调用
`reduce_stamp_delivery_series_v1(bundle_paths)`。它只流式读取中心 `13×13`
aperture crop，先验证每个 formal bundle 的 header、run identity、gain、全局
raw-frame 区间和时间连续性，再一次性计算整条曲线的 CDPP；不能逐 shard 计算
CDPP 后做平均。

## Derived light curve

For every 13x13 central fixed-aperture pixel, the reducer calculates

```text
calibrated_e = (final_dn - bias_level_sum_dn - column_noise_sum_dn_by_x)
               * gain_e_per_dn
               - background_expectation_e
```

and sums `calibrated_e` over the aperture.  It never reads or subtracts a
background-realization image.  Removing a realization would remove real
Poisson noise from the observation and would make the result physically
misleading.

The v1 quality policy is strict: if any aperture pixel is invalid, saturated,
or cosmic-affected in a cadence, that cadence's `flux_e` is `NaN`.  The
reducer does not estimate a missing pixel or rescale a partial aperture.
`aperture_valid` and `aperture_usable_pixel_count` make the decision explicit.

## CDPP

`compute_cadence_aware_cdpp` reports 30-, 90-, and 390-minute values.  It
uses the legacy mean-absolute-deviation normalization
`1.4826 * mean(abs(flux - mean(flux))) / mean(flux) * 1e6`, but implements
its own CPU time binning instead of calling legacy `bin_lcs`.

Each output-bin flux is the sum of all valid derived electrons whose physical
exposure intervals fully cover that bin.  Leading/trailing partial windows are
not counted; internal gaps, invalid samples, saturation, and cosmic masks
create rejected windows.  Consequently the reported CDPP is cadence-aware and
does not assume that a 30-second, 1-minute, 2-minute, or 5-minute delivery is
a synthetic uniformly sampled 10-second curve.

This v1 metric does not detrend the light curve.  A separately approved
science-analysis layer may apply a trend treatment only when its treatment of
real astrophysical variation is explicit; the raw fixed-aperture curve and
complete-window counts always remain available.

## Variable-source residual CDPP

For an injected variable source, the ordinary fixed-aperture CDPP necessarily
contains real intrinsic variability and must be labelled as an
``astrophysical-plus-instrument`` statistic.  It is not an instrumental-noise
claim.

`compute_injected_model_residual_v1(result, raw_frame_factors=...)` provides
the companion diagnostic for a formal delivery sequence.  It uses the frozen
10 s exposure-averaged injection factors and each output cadence's absolute
raw-frame interval to form the corresponding factor sum, fits the physically
required through-origin relation

```text
derived_aperture_flux_e = scale * sum(raw_frame_factor)
```

and reports residual ppm plus 30/90/390-minute complete-window CDPP.  The fit
has no free intercept because background expectation has already been removed
from the derived electron image; adding one is ill-conditioned for small
variability and can absorb real detector residuals.  This residual CDPP is the
standard instrument-like metric for the Galaxy injected production.
