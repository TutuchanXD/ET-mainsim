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

对于 time-shard worker，不需要把整个 shard 累积在内存中。可使用
`StampDeliveryBundleAppender`：每次 `append()` 一个已由
`StampDeliveryBundle.from_arrays()` 验证过的有限 batch，最后只调用一次
`complete()`。每个 batch 必须具有相同的 `product_kind`、`coadd_factor`、
stamp shape、静态 `gain_e_per_dn`、`manifest` 和 `provenance`；batch 之间不能
在时间或 raw-frame 区间重叠。streaming appender 的 gain 仅支持标量或固定的
`(ny,nx)` map，避免未显式设计的逐帧 gain 混合。

```python
from et_mainsim.stamp_delivery import StampDeliveryBundleAppender

with StampDeliveryBundleAppender(
    final_path,
    product_kind="raw",
    coadd_factor=1,
    stamp_shape=(ny, nx),
    gain_e_per_dn=gain_e_per_dn,
    manifest=manifest,
    provenance=provenance,
) as writer:
    for batch in bounded_batches:
        writer.append(StampDeliveryBundle.from_arrays(**batch))
    report = writer.complete()
```

离开 context 前未执行 `complete()`，或任一 `append()`/readback 出错时，partial
文件会被删除；因此它同样没有 append-after-final 或从 partial 续跑的语义。

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
        -> 分块 HDF5 回读与完整 schema/语义校验
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

对 Galaxy raw 10 s formal campaign，HDF5 delivery bundle 本身不等于已经放行的
science light curve：还必须经 campaign QC、每源 `raw_10s_strict`、冻结 policy 的
`raw_10s_coverage_v2` 与十源 summary。完整的用户交付和 CDPP 解释见
[galaxy_raw_coverage_science_delivery_zh.md](galaxy_raw_coverage_science_delivery_zh.md)。

## 已接入的正式 production 层

当前 `et_mainsim.independent_stamp_production` 已把 Photsim7 的单帧 stamp 输出
映射到本合同，并由
`et_mainsim.galaxy_stamp_production` 提供 Galaxy 独立目标的 prepare/worker 入口。
该 worker 的固定约束是：

1. `final_dn` 来自已完成 legacy-aligned detector electronics 的同一 raw frame；
2. 质量 count、bias、column noise 和背景期望来自完全相同的一组 raw frames；
3. `manifest_json` 绑定 run ID、target ID、全局 raw-frame shard、输入 `q(t)`
   snapshot、科学配置、PSF/坐标解析与软件 provenance；
4. 生产 manifest 使用相对资源路径和内容 SHA-256，使共享盘在工作站/H100 的
   不同挂载根目录下仍可安全运行；路径差异本身不被误判为输入漂移；
5. target-time shard 不支持 resume。一个 shard 的 raw 与全部 coadd 成员必须完成
   合同验证后才可作为交付集合出现；partial 或成员不全的目录不可交付；
6. 不会把 optional diagnostic 或 background image 标作第二个观测产品，或用它做
   默认背景扣除。

Galaxy 的 paired `static`/`injected` control 必须审计真实物理 RNG，而不是
`rng_trace_scope`。后者只是 `execution_context.labels`；它可包含 `case`，但不会进入
detector seed。新 Galaxy worker 在 raw/coadd 的 caller manifest 和 caller provenance
均写入 `physical_rng_pairing`，schema 为
`et_mainsim.galaxy_physical_rng_pairing.v1` / version 1。读取端应确认同一 source 和
同一 shard 的两个 case 该记录完全相同：其中的 SeedTree `run_seed`、canonical
`SimulationContext.detector_rng_scope`（`science_realization_id`、`spacecraft_id`、
`detector_id`、`scope_id`）、绝对 raw-frame 公式/半开区间以及 `target_spec_sha256` 都是
一致的；同时必须看到 `source_id_in_physical_rng_identity=false`、
`case_not_in_physical_rng_identity=true` 和
`rng_trace_scope_role=execution_label_only`。source ID 只作为 comparison label，不能被
误读为 per-target stochastic seed。

这也限定了 independent stamp 的语义：它是 target-only/no-neighbors 的场景选择、任务
调度与原子交付，不是 full-frame 的严格 crop，且不制造每个目标独立的随机 detector
field。同 detector、同一绝对 frame 的目标使用共同的按绝对像元/块/列寻址的随机场；
非重叠位置有不同坐标地址，重叠位置应相同。不同 detector 的 `detector_id` 属于物理
scope，因而使用不同随机场；所有目标仍共享同一 campaign dynamics 和全局时间轴。

历史 v1/v2 预检 bundle 未写入上述直接审计记录，但同源 static/injected 仍可依据
相同的真实 context scope、run seed、detector 与绝对 frame interval 确认物理配对；不能
仅因旧 execution label 含 `case` 而将它们判为 unpaired noise。它们是否可用于科学交付
仍由其它 production gate 决定。

生产 writer 对格式、时间范围、质量 mask、完成状态与已声明语义失败关闭；它不会
替科学团队猜测未冻结的波段转换、物理相位、最优 aperture 或场景邻星。
