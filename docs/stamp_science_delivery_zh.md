# 独立 Stamp 科学交付包（v1）

`et_mainsim.stamp_delivery_bundle.v1` 是独立目标、单一时间 shard 的正式
Stamp 科学交付基础格式。它解决的是“什么才是观测、如何保留可校准信息、
如何避免半成品被误用”三个问题；它不改变 Photsim7 的物理渲染或随机数定义。

实现位于 `et_mainsim.stamp_delivery`，依赖 `h5py>=3.10`。当前 writer/read
API 为：

```python
from et_mainsim.stamp_delivery import (
    StampDeliveryBundle,
    read_stamp_delivery_bundle,
    write_stamp_delivery_bundle,
)

bundle = StampDeliveryBundle.from_arrays(...)
write_stamp_delivery_bundle("target_42/shard_0003_raw.h5", bundle)
checked = read_stamp_delivery_bundle("target_42/shard_0003_raw.h5")
```

## 核心语义

`final_dn` 是本合同中唯一的**真实探测器观测量**。它是原始 10 s 帧或由连续
原始帧相加得到的 coadd DN；不能把背景、bias、列噪声、mask 或真值平面当作与
它并列的第二幅“观测图像”。

其余平面仅用于下列用途：

1. 在电子域校准 `final_dn`；
2. 标记不能用于测光的像素/曝光；
3. 保留绝对时间和原始帧范围；
4. 追溯该文件来自哪个 run、目标、输入和科学配置。

对每个输出平面，推荐的电子域校准为：

```text
E_cal = (final_dn - bias_level_sum_dn - column_noise_sum_dn_by_x)
        * gain_e_per_dn
E_bgsub = E_cal - background_expectation_e
```

其中 `background_expectation_e` 是背景的**期望值**，不是本次随机实现的背景
realization。不得从 `final_dn` 中扣除任何 background-realization image；那会把
真实观测中应保留的泊松涨落错误删除。v1 bundle 不写出这种 realization 平面，
且 `provenance.background_realization_used` 必须为 `false`。

## 文件原子性与完成状态

writer 在目标文件同一目录中依次执行：

```text
.<final-name>.<uuid>.partial 写入
        -> complete=true
        -> 完整 HDF5 回读与 schema 校验
        -> os.replace(partial, final) 原子重命名
```

同时使用同目录 `.lock` 防止两个 producer 写同一最终文件。读者只接受根属性
`complete=true` 的最终文件；缺少字段、半成品、错误 schema、被篡改的 quality
mask 或不符合物理语义的 provenance 都会失败。正式生产不从 partial 文件续跑，
而应让调度器重做该 target-time shard。

## HDF5 schema

根属性如下。

| 属性 | 类型 | 含义 |
| --- | --- | --- |
| `schema_id` | string | 固定为 `et_mainsim.stamp_delivery_bundle.v1` |
| `schema_version` | int | 当前为 `1` |
| `complete` | bool | 只有 `true` 才可交付 |
| `product_kind` | string | `raw` 或 `coadd` |
| `coadd_factor` | int | raw 为 `1`；coadd 为连续相加的原始 10 s 帧数，且大于 `1` |
| `observation_product` | string | 固定为 `final_dn` |
| `background_realization_used` | bool | 固定为 `false` |
| `gain_e_per_dn` | float，可选 | 标量 gain；非标量 gain 改以同名 dataset 写入 |

所有 image plane 的形状均为 `(n_frames, ny, nx)`；第 0 维是该 target-time
shard 内按时间递增的输出帧。`n_frames` 不得为 0。

| Dataset | 单位 / dtype | 形状 | 含义 |
| --- | --- | --- | --- |
| `final_dn` | 无符号 integer DN；coadd 强制 `uint64` | `(n, ny, nx)` | 唯一 detector observation。raw 保留 producer 的无符号 DN dtype；coadd 以 `uint64` 防溢出。 |
| `background_expectation_e` | `float64`, e- | `(n, ny, nx)` | 该输出帧累积背景的期望电子数，非负。 |
| `bias_level_sum_dn` | `float64`, DN | `(n,)` | 该输出帧累积 bias level。 |
| `column_noise_sum_dn_by_x` | `float64`, DN | `(n, nx)` | 每个输出帧、每一 stamp 列的累计有符号列噪声。按 x 广播至 y。 |
| `gain_e_per_dn` | `float64`, e-/DN | scalar、`(ny,nx)` 或 `(n,ny,nx)` | 电子域校准所需 gain。标量写根属性，其他形状写 dataset。 |
| `valid_mask` | `uint8`（0/1） | `(n, ny, nx)` | 像素是否有效。 |
| `fullwell_count` | `uint16` | `(n, ny, nx)` | 该输出像素在 `coadd_factor` 个原始帧中触及 full well 的次数。 |
| `adc_low_count` | `uint16` | `(n, ny, nx)` | ADC 下端裁剪次数。 |
| `adc_high_count` | `uint16` | `(n, ny, nx)` | ADC 上端裁剪次数。 |
| `cosmic_count` | `uint16` | `(n, ny, nx)` | 受 cosmic-ray 影响的原始帧次数。 |
| `saturated_mask` | `uint8`（0/1） | `(n, ny, nx)` | 便利字段，严格等于 `fullwell_count>0 OR adc_low_count>0 OR adc_high_count>0`。 |
| `cosmic_mask` | `uint8`（0/1） | `(n, ny, nx)` | 便利字段，严格等于 `cosmic_count>0`。 |
| `time_start_seconds` | `float64`, s | `(n,)` | 仿真绝对起始时间；严格递增。 |
| `exposure_seconds` | `float64`, s | `(n,)` | 每个输出帧的积分时间，必须为正。raw 正常为 10 s，coadd 为 `coadd_factor * 10 s`。 |
| `raw_frame_start_index` | `int64` | `(n,)` | 输出帧覆盖的绝对 raw-frame 起始 index。 |
| `raw_frame_stop_index_exclusive` | `int64` | `(n,)` | 半开区间终点；raw 的宽度严格为 1，coadd 的宽度严格为 `coadd_factor`。 |
| `manifest_json` | UTF-8 JSON object | scalar | 目标、run、时间 shard、配置/输入 identity 等交付级清单。 |
| `provenance_json` | UTF-8 JSON object | scalar | 代码、随机数、效果 receipt、科学配置等来源信息；其中必须声明 `observation_product="final_dn"` 和 `background_realization_used=false`。 |

质量 count 必须在 `0..coadd_factor` 内。这让 raw 的 count 自然为 0 或 1，而
coadd 保留“一个像素在多少个 raw exposure 中异常”的信息，不把它降成不可逆的
单一 bool。

## Raw、coadd 与时间

raw bundle 的 `coadd_factor=1`，每个输出平面覆盖恰好一个绝对 raw-frame index。
coadd bundle 的 `coadd_factor>1`，每个输出平面覆盖一个半开、宽度恰为
`coadd_factor` 的连续 raw-frame index 区间。连续 target-time shard 由上层
time-shard planner 决定；本格式保留绝对 index，因此不同 shard 可以无歧义拼接，
而不会把每个 worker 的局部 index 当成物理时间。

`final_dn` coadd 只能是 raw `final_dn` 的 detector-domain 求和，故写入类型固定
为 `uint64`。不能先把 read noise/cosmic/noise 过程折算成一张长曝光图再相加。
同样，背景期望、bias、列噪声和各 quality count 都是对应 raw 过程的逐帧累积量。

## 与标准测光/CDPP 的衔接

`StampDeliveryBundle.to_reference_photometry_payload()` 返回的字典可直接传给
`reference_photometry_v1` 的
`ReferencePhotometryInput.from_arrays(**payload)`：

```python
from et_mainsim.reference_photometry import (
    ReferencePhotometryInput,
    reduce_reference_photometry_v1,
)
from et_mainsim.stamp_delivery import read_stamp_delivery_bundle

bundle = read_stamp_delivery_bundle("target_42/shard_0003_raw.h5")
delivery = ReferencePhotometryInput.from_arrays(
    **bundle.to_reference_photometry_payload()
)
result = reduce_reference_photometry_v1(delivery)
```

adapter 会将 quality count 转为保守的 `saturated_mask` 与 `cosmic_mask`，并以
`time_start_seconds` 作为 `time_index_unit="seconds"`。它不会暴露背景
realization。后续 reference photometry 的固定口径、完整窗口 CDPP、以及任何
科学团队自定义口径都应从 `final_dn` 和本合同中明确的校准/质量字段开始。

## 当前集成边界

本模块仅定义并验证交付 wire format。把 Photsim7 单帧输出映射为这些字段、把
多个 raw 帧流式 coadd、由 target-time shard worker 填充 `manifest_json`/`provenance_json`，
属于下一层 production workflow 的工作。该 workflow 必须在写入前确认：

1. `final_dn` 是已完成 legacy-aligned detector electronics 后的产品；
2. 质量 count、bias、column noise、背景期望来自同一组 raw frames；
3. manifest 绑定 run ID、target ID、全局时间 shard、输入光变曲线 identity、
   science configuration 和相关 Photsim7 product/provenance；
4. 没有把 optional diagnostic/background image 标为第二个观测产品，或用于默认
   背景扣除。

这些约束是正式科学数据生产的前提；v1 writer 会对格式和已定义的语义失败关闭，
但不会替 producer 猜测未写入的科学配置。
