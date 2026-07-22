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
| `valid_mask` | `(n_cadence, ny, nx)` | Valid-pixel binary mask; every stored value is exactly `0` or `1`. |
| `saturated_mask` | `(n_cadence, ny, nx)` | Saturation binary mask; every stored value is exactly `0` or `1`. |
| `cosmic_mask` | `(n_cadence, ny, nx)` | Cosmic-affected binary mask; every stored value is exactly `0` or `1`. |
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

### Formal 流式读取的严格边界

`reduce_stamp_delivery_series_v1()` 不是宽松的“尽量读出曲线”工具。它在一个
shard 内和相邻 shard 之间都要求曝光区间首尾连续：对每个后续 cadence，
`time_start_seconds[i]` 必须等于
`time_start_seconds[i-1] + exposure_seconds[i-1]`，采用固定的绝对容差
`1e-8 s`。只满足时间单调递增而中间存在 gap 或 overlap 的 bundle 会被拒绝，
而不是被当作普通缺失点继续计算 CDPP。`raw_frame_start_index`/`stop` 也必须
以相同的无缝方式相接。

三个 mask 都是 wire-format 的二元量：接受布尔值或整数 dtype，但每个值必须
严格为 `0` 或 `1`。读取器不会把 `2`、`-1` 或其他“非零即真”的值悄悄转换成
`True`。这避免损坏的质量平面在固定 aperture 的有效性判断中被误解释。

正式 `gain_e_per_dn` 可以是标量、静态 `(ny, nx)` gain map，或逐 cadence 的
`(n_cadence, ny, nx)` gain cube。对于前两者，同一流式序列的各 shard 必须具有
完全相同的静态 gain；对于 3-D gain，读取器仅在当前 batch 读取并验证中心
`13×13` crop，因此不会将整段生产的 gain cube 读入内存，且允许其随 cadence
变化。3-D gain 在正式 HDF5 中必须作为 dataset（非标量 gain 的标准写法）储存；
它不能与标量或静态-map shard 混用。

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

This v1 metric is **undetrended** and only legacy-MAD-compatible: it uses the
legacy MAD normalization after complete physical-time binning, but it is not
the legacy PCA/SG (principal-component / Savitzky-Golay) processing chain, and
does not call or reproduce legacy `bin_lcs`.  A separately approved
science-analysis layer may apply PCA, SG, or another trend treatment only when
its treatment of real astrophysical variation is explicit; the raw
fixed-aperture curve and complete-window counts always remain available.

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
variability and can absorb real detector residuals.  This residual CDPP is an
undetrended instrument-like residual diagnostic for the Galaxy injected
production.  它不是 legacy PCA/SG/detrend 标准流程的 CDPP；若将来需要该另一套
科学指标，必须先冻结趋势模型、训练/拟合样本与不确定度口径，不能把它与本模块的
reference/residual MAD 指标混称。

## Formal standard-analysis CLI/API

`et_mainsim.standard_stamp_analysis` packages the above primitive functions
into the maintained post-processing entry point for a completed formal Galaxy
production.  It is intentionally scoped to
`et_mainsim.galaxy_stamp_production.v1` manifest schema version 2: the caller
provides a production manifest, target, case, and delivery cadence; it does
not hard-code a run-root path or enumerate shards by directory globbing.

```bash
python -m et_mainsim.standard_stamp_analysis \
  --production-manifest /path/to/production_manifest.json \
  --source-id 2119800835728275584 \
  --case injected \
  --cadence-seconds 60 \
  --output-dir /path/to/standard_analysis/source_2119800835728275584/injected/coadd_60s
```

Its Python API is equivalent:

```python
from et_mainsim.standard_stamp_analysis import (
    StandardStampAnalysisRequest,
    run_standard_stamp_analysis_v1,
)

result = run_standard_stamp_analysis_v1(
    StandardStampAnalysisRequest(
        production_manifest_path="/path/to/production_manifest.json",
        source_id=2119800835728275584,
        case="injected",
        cadence_seconds=60,
        output_dir="/path/to/analysis-output",
    )
)
print(result.analysis_manifest_path)
```

公开 API 如下；内部 JSON/path helpers 不属于稳定接口。

| API | 用途 | 关键输入/输出 |
| --- | --- | --- |
| `StandardStampAnalysisRequest(...)` | 冻结一次分析请求。 | `production_manifest_path`、signed-int64 `source_id`、`case`、整数倍 raw exposure 的 `cadence_seconds`、`output_dir`；默认 CDPP window 为 30/90/390 min。 |
| `discover_standard_stamp_analysis_input(request)` | 只做 manifest、time plan、发布完整性和 injected snapshot 的验证。 | 返回 manifest-relative HDF5 path 列表；若尚未完整发布则抛出 `StandardStampAnalysisNotReadyError`。 |
| `run_standard_stamp_analysis_v1(request)` | 流式产生 CSV 与 JSON analysis manifest。 | 返回 `StandardStampAnalysisResult`，其中有两个最终文件路径和 cadence 计数。 |
| `StandardStampAnalysisError` | 正式输入或语义合同不成立。 | 包含 schema、identity、cadence、delivery caller provenance 等失败。 |
| `StandardStampAnalysisNotReadyError` | 正式 run 仍缺预期 final HDF5 shard。 | 不读取 partial；等待生产补齐后以同一 request 重试。 |

`--batch-frames` 只限制 HDF5 中心 aperture 的流式读取 batch；它不改变仿真、
coadd、RNG 或 CDPP 时间定义。分析输出是一个不可变的目录级交付：已有
`complete: true` 的标准分析即使指定 `--overwrite` 也绝不覆盖。若路径残留的是不完整
目录，`--overwrite` 只会将该目录原样归档为同级
`.output-name.incomplete-<uuid>`，并写入 `INCOMPLETE_ARCHIVE.json`
（`complete: false`、`discovery_policy: not_a_standard_analysis_product`），随后发布
新的完整目录；它不会修改任何 HDF5 delivery，也不会删除旧派生文件。

发现阶段会读取 manifest-relative 的 time-shard plan，验证 plan 的 SHA-256
identity，并要求该 `target × case × cadence` 的**每一个**预期 HDF5 final member
已经发布。缺任何 shard 时抛出 `StandardStampAnalysisNotReadyError`，不会读取
partial/staging 文件、不会输出半条 light curve，也不会把一日快看误称为完整正式
分析。对 `injected`，它还验证 target 的 frozen factor snapshot 本身以及每个
delivery bundle 写入的 snapshot identity；因此结果不能把别的输入曲线残差误标为
该目标的注入恢复。

开始 reduction 前，CLI 对所选 cadence 的每一个 final HDF5 调用
`validate_stamp_delivery_bundle()`，再流式计算整个文件的 SHA-256。最终
`analysis_manifest.json` 的 `delivery.bundle_receipts[]` 记录相对路径、完整文件
`size_bytes`、`sha256` 和 schema readback 摘要；分析完成后还会比较文件的
device/inode/size/mtime 状态，若输入在验证或 reduction 期间被替换/改写则失败，不
发布分析文件。全文件 hash 是有意的可复现性成本，特别是 raw 10 s 交付会明显增加
后处理 I/O；它不改变任何 HDF5 或仿真随机数。

`analysis_manifest.json` 同时保留执行端的绝对
`production_manifest_path`，以及稳定的
`production_manifest_relative_to_run_root`（当前为
`production_manifest.json`）。前者用于审计实际执行位置，因 H100 与工作站的
sshfs 挂载根不同而允许变化；跨挂载读回必须以 `run_id`、relative path 和
`production_manifest_identity` 联合验证，不能把两个等价挂载路径的字符串差异误判为
输入漂移。

成功时 `output_dir` 只产生两个小型派生分析文件：

| 文件 | 内容 |
| --- | --- |
| `reference_lightcurve.csv` | 中心 `13×13` fixed aperture 的每 cadence 派生电子 flux、绝对时间/原始帧半开区间、`aperture_valid`、可用像元数和不可用像元数。injected case 另外有 frozen-factor sum、through-origin model flux、residual e-/ppm 与 residual valid flag。空的 `flux_derived_e` 或 residual 值表示该 cadence 因质量口径无效，不是零 flux。 |
| `analysis_manifest.json` | 输入 production/交付 identity receipt、固定 aperture 与质量汇总、顶层 `observed_cdpp`、injected-model residual CDPP、输出字段与统计口径。任何无法估计的 CDPP 以 JSON `null` 表示，绝不写非标准 `NaN`。 |

这两个文件不会分别暴露给读者：它们先在 `output_dir` 同级的私有
`.output-name.staging-*` 目录中完成单文件原子写入与 `fsync`，再以同一文件系统上的
目录 rename 一次性发布。若 CSV 后的 manifest 生成/写入失败，staging 目录会清理，最终
`output_dir` 不存在；因此普通消费者看到的已发布目录总是同时包含 CSV 和
`complete: true` manifest。

`ordinary_cdpp_label` 在 variable injected case 固定为
`undetrended_astrophysical_plus_instrument_legacy_compatible_diagnostic`；它包括真实
的内禀光变，不能作为纯仪器指标。`injected_model_residual_cdpp_label` 标记对应冻结
`q(t)` 的未趋势去除残差诊断。两者都采用完整物理曝光窗口和 legacy-compatible
mean-MAD normalization；这只复用统计口径，正式链路明确不调用 legacy pickle、
PCA、Savitzky--Golay 或 `bin_lcs`，也绝不将该输出称为 legacy-standard CDPP。
