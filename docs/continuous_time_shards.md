# 连续全局 raw-frame 时间分片（开发者 API）

`et_mainsim.time_shards` 是正式长时标 stamp 生产的**纯规划层**。它不
导入 Photsim7、不创建渲染服务、不写 HDF5 图像，也不改变现有
`et-mainsim run et-stamp` 行为；当前用途是先冻结可复现的时间分片和
coadd 时间语义，供后续调度/执行层消费。

现有 `StampWorkerRequest` 只按 `target_ids[rank::world_size]` 分配目标，
而 `workflows/stamp.py::_render_target()` 会从完整 `SimulationSpec` 的
coadd 0 开始运行。因此它不能直接用于按时间分片的正式生产。

## 时间约定

- `raw_start_index` 为包含端点，`raw_stop_index` 为排除端点；二者均为从
  全局原始帧 0 开始的绝对索引，而不是分片内索引。
- 一个 raw frame 的曝光区间是
  `[raw_index * raw_exposure_seconds, (raw_index + 1) * raw_exposure_seconds)`。
- `coadd_size=N` 表示连续的 `N` 个 raw frame。coadd 的时间戳是整个曝光
  区间的中点，而不是首帧或末帧的时间戳。
- 要同时产出 30 s、1 min、2 min、5 min，10 s raw exposure 对应
  `coadd_sizes=(3, 6, 12, 30)`；共同边界是 `lcm(3, 6, 12, 30) = 60` 个
  raw frame（600 s）。每个时间分片的起止都必须落在这个共同边界上。
- 请求区间的起点必须已在共同边界上，不能通过丢弃开头帧来重新锚定。
  终点若留有不足 60 个 raw frame 的尾部，规划器只拒绝该尾部，并在
  manifest 中显式记录；不会丢弃任何完整 coadd。

这使每个分片可独立运行，同时合并后仍是无重叠、无缺口的全局时间轴，
并可从同一批 raw realization 派生四种 cadence，而无需为每种 cadence
重新渲染。

## 最小 API 示例

```python
from et_mainsim.time_shards import (
    coadd_sizes_for_cadences,
    plan_continuous_time_shards,
)

coadd_sizes = coadd_sizes_for_cadences(
    raw_exposure_seconds=10.0,
    cadence_seconds=(30.0, 60.0, 120.0, 300.0),
)
plan = plan_continuous_time_shards(
    raw_start_index=0,
    raw_stop_index=777_617,       # 90 days + 17 raw frames: 17-frame tail is rejected
    coadd_sizes=coadd_sizes,      # (3, 6, 12, 30)
    raw_exposure_seconds=10.0,
    max_raw_frames_per_shard=21_600,
)

assert plan.alignment_raw_frames == 60
assert plan.rejected_tail_raw_interval == (777_600, 777_617)
plan.write_manifest("run/time_shards.json")

for shard in plan.shards:
    # 5-minute products are globally indexed even when this is not shard 0.
    for coadd in shard.iter_coadd_windows(30):
        print(
            shard.shard_id,
            coadd.coadd_index,
            coadd.raw_start_index,
            coadd.raw_stop_index,
            coadd.midpoint_time_seconds,
        )
```

`ContinuousTimeShardPlan.to_manifest_dict()` 返回可嵌入 run manifest 的
JSON-safe 对象；`from_manifest_dict()` 会重新检查全局 raw 原点、所有边界、
coadd 计数、cadence、tail policy 和连续覆盖。`write_manifest()` 使用原子
替换写入独立 JSON 文件。

## manifest 关键字段

```text
time_axis.kind = global_raw_frame_index
time_axis.origin_raw_frame_index = 0
time_axis.coadd_timestamp = exposure_interval_midpoint
raw_frame_interval = requested [start, stop)
accepted_raw_frame_interval = complete global-coadd [start, stop)
rejected_tail_raw_frame_interval = null | rejected [start, stop)
coadd_sizes = [3, 6, 12, 30]
global_alignment_raw_frames = 60
shards[] = ordered, non-overlapping [start, stop) intervals
```

每个 `shards[]` 元素带 `shard_id`、全局 raw 区间、raw frame 数和每种
coadd size 的完整 coadd 数。`validate_time_shard_coverage()` 显式拒绝
gap、overlap、错误顺序、非连续 shard ID 或不同的 coadd 定义。

## 与现有调度入口的最小接入点（尚未实现）

后续执行层应采用下面的边界；本模块刻意不在本提交中越过这些边界。

1. 在 `workflows/stamp.py::run_stamp()` 完成输入身份验证后、启动 worker 前，
   用正式的全局 raw 时间范围创建 `ContinuousTimeShardPlan`。把
   `plan.to_manifest_dict()` 写为 `run_dir/time_shards.json`，并将其摘要/路径
   作为 run identity 的一部分写入 `run_manifest.json`。时间范围、raw exposure、
   coadd sizes、shard 上限的变化必须使 resume 失败关闭。
2. 新建版本化的**时间 worker request**，其一项工作为
   `(target_id, shard_id, raw_start_index, raw_stop_index)`。不要给现有
   `StampWorkerRequest` 偷加可选字段：它的语义是按 target 分配，当前
   `run_stamp_coadd()` 也只接收相对于整个 `SimulationSpec` 的 `coadd_index`。
3. 新 worker 必须把 `raw_start_index` 传入/映射到 Photsim7 的绝对 raw-frame
   索引，使 jitter、temperature/DVA、source variability 和 RNG 仍按同一全局
   realization 取值。它应逐个处理 `ContinuousTimeShard` 的连续 raw frames，
   再用 `iter_coadd_windows()` 产出 30 s/1 min/2 min/5 min 成品；不得使用旧的
   交错 coadd-shard 方式，也不得在不同 cadence 之间重新抽样。
4. 每个 target×time-shard 产物应写入独立目录/文件，并带本 manifest 的内容
   hash、目标输入身份、绝对 raw 区间和产品清单。最终汇总器必须先调用
   `validate_time_shard_coverage()`，再允许将分片暴露为完整时间序列。

## 当前限制

本模块只解决调度前的确定性时间规划。它尚未：

- 改造 `StampWorkerRequest`、CLI、Slurm 模板或 resume 语义；
- 将 source-variability provider 改为文件分块读取；
- 写入 `final_dn`、背景、mask 或电子域辅助产品；
- 实现绝对探测器位置的 Stage 9 electronics，或宣称独立 stamp 与全幅共享
  曝光一致。

因此该规划 manifest 可以作为正式生产链路的身份边界，但在新的时间 worker
和产品汇总器落地前，不能单独被视为可执行的科学数据生产任务。
