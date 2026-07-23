# 银河系内禀光变正式 Independent Stamp 生产说明

本文说明当前可以进入正式科学数据生产的第一条源光变链路：银河系团队提交的
`mock_lightcurves_sourceid.fits`。它不是全幅或共享曝光模拟，而是**每个目标独立
渲染、调度和原子交付的 stamp**。所有目标仍使用同一 campaign 的绝对 raw-frame
时间轴和动态效应（温度、指向、legacy-aligned PSF breathing 等）；"independent" 绝不
表示每个目标拥有彼此独立的物理随机场。

正式执行入口是：

```text
scripts/run_galaxy_independent_stamp_production.py
```

它有两个子命令：`prepare` 冻结输入、时间分片和配置；`run-target` 只渲染一个
目标的一个或多个明确分片。正式 Slurm 作业必须显式传入 `--shard-id`，不能省略
该参数而意外把 90 天全部交给一个任务。

## 1. 已冻结的科学范围

| 项目 | 正式 v1 取值 / 规则 |
| --- | --- |
| 场景 | 每个 target 单独调度、单独原子交付，且 catalog 只保留该目标、无邻星；不是 full-frame 的严格 crop，也不宣称等价于 full-frame/shared-exposure 场景。 |
| 目标数 | 10 个 Galaxy 源；默认 Gaia ID 在代码中的 `DEFAULT_GALAXY_PRODUCTION_SOURCE_IDS`。 |
| 时间长度 | 90 天；原始曝光 10 s。 |
| 交付 cadence | raw 10 s，以及由同一 raw realization 求和得到的 30 s、1 min、2 min、5 min。 |
| stamp | `100 × 300` pixel；像元尺度统一为 `4.83 arcsec/pixel`。 |
| 星等语义 | 只接受 Gaia G Vega 基准星等。 |
| 背景 | 写出背景**期望**，不写出背景随机 realization。 |
| 响应效应 | SD-20 探测器响应效应在正式生产中 fail-closed 关闭；temperature-driven legacy PSF breathing、光子/背景/暗电流/读出/数字化等由已对齐的 Photsim7 链路处理。 |
| 物理随机场 | 同一 detector、同一绝对 raw frame 的目标共享按绝对探测器坐标/块/列寻址的物理随机场；空间不重叠位置取得不同坐标地址，重叠像元必须复用同一 realization。不同 detector 的 `detector_id` 进入物理 scope，使用不同随机场。 |

当前 10 个目标的 Gaia G 都约为 11.33--11.65 mag。其中两个 rotation 曲线的振幅
明显大于其余 subgiant 曲线，适合作为注入恢复和残差 CDPP 的代表目标。Galaxy
FITS 共含 74 个源，所有曲线的可用相对时间覆盖约 1461 天，足以覆盖 90 天正式
生产。

## 2. Galaxy FITS 输入如何被解释

正式 profile 读取下列列：

| FITS 列 | 用途 |
| --- | --- |
| `Source` | Gaia source ID，必须在 signed int64 范围内且唯一。 |
| `Gmag` | Gaia G Vega 基准星等。 |
| `RAJ2000`, `DEJ2000` | ICRS/J2000 坐标；用于 ET focal-plane 映射和最近视场角 PSF 选择。 |
| `class` | 输入源类别，只作 provenance。 |
| `time` | 曲线节点间的相对间隔；绝对 epoch 不进入仿真。 |
| `relative_flux` | `Delta F / F_ref` 的 clean 内禀相对流量。 |

注入因子定义为：

```text
q(t) = 1 + Delta F(t) / F_ref
```

FITS 中的绝对日期、MJD 或物理观测起点不会作为 ET 时间原点使用。第一条有限
节点被重置为仿真 raw frame 0；后续只保留节点间的相对间隔来描述曲线形状。对每
一个 10 s 原始曝光，程序在 clean、分段线性的 `q(t)` 上计算该曝光区间的**精确
平均值**，再把它交给 Photsim7。因此不会把数天采样点复制成许多相同的 10 s 值，
也不会将数据文件的绝对日期误当成 ET 仿真日期。

坐标路径是：

```text
RA/Dec -> frozen ET focal-plane registry -> detector/pixel/field angle
       -> nearest available PSF node -> stamp render
```

准备阶段会冻结 registry 的 semantic identity。运行阶段会重新计算 H100 上的
identity，允许 registry 根目录因挂载不同而变化，但要求 CSV 内容、坐标算法、
owner-frozen 状态和 owner attestation 全部一致；任一不一致都会在渲染前失败。

## 3. 光变进入探测器前的物理位置

对每个 raw frame，先由 Gaia G Vega 基准亮度得到目标的期望源电子数，再乘以
`q(t)`，随后才进行 PSF 投影和恒星泊松抽样：

```text
Gaia G Vega baseline source expectation
        × exposure-averaged q(t)
        -> PSF rendering / scene sum
        -> stellar Poisson sampling
        -> sky + scattered light + dark current
        -> full-well/readout/gain/ADC/cosmic/bias/column chain
        -> final_dn
```

因此变亮/变暗会同时改变源的期望光子数和对应的散粒噪声；这不是对已完成图像做
后处理缩放。coadd 也不平均 `q(t)` 后重渲染，而是先逐 10 s raw frame 注入和模拟，
再在 detector DN 域对连续帧求和。

## 4. 交付给科学团队的内容

每个 `target × case × time-shard` 有一个 raw HDF5 和四个 coadd HDF5。`case` 为
`static` 或 `injected`；正式科学产出使用 `injected`，而 `static` 仅作配对验证，
不需要全量复制一遍 90 天数据。

配对验证使用 Photsim7 的**真实物理** RNG identity，而不是执行标签。对同一 Gaia
source、同一绝对 raw-frame shard，`static` 与 `injected` 的
`SimulationContext.detector_rng_scope` 必须逐 frame 相同；其 canonical 字段为
`science_realization_id`、`spacecraft_id`、`absolute_raw_frame_index`、`detector_id` 和
`scope_id`，并由同一个 SeedTree `run_seed` 派生。`case` 只保留在 HDF5 的
`manifest_json.caller_manifest.case`、输入 truth 和 execution labels 中；它**不进入**
物理 seed。`rng_trace_scope` 也是 execution/provenance label，Photsim7 在 detector
seed 归一化前会丢弃它，不能据此判断像素是否配对。

新 worker 在每个 raw/coadd bundle 的
`manifest_json.caller_manifest.physical_rng_pairing` 与
`provenance_json.caller_provenance.physical_rng_pairing` 同时写入
`et_mainsim.galaxy_physical_rng_pairing.v1` / version 1。该只读审计记录包含 SeedTree
`run_seed`、不含 frame 项的 canonical context scope、
`absolute_raw_frame_start_index + local_frame_index` 公式与该 shard 的绝对半开区间、
`target_spec_sha256`，以及明确的
`source_id_in_physical_rng_identity=false`、
`case_not_in_physical_rng_identity=true` 和
`rng_trace_scope_role=execution_label_only`。`source_id` 只作为 comparison label；不把
它写入物理 identity，才能维持与 full-frame 物理坐标场一致的重叠像元语义。

因此，**同 detector 的不同 source 并非彼此独立的 stochastic realization**：它们在
同一 campaign/frame 共享同一个坐标可寻址的 detector random field，非重叠位置因坐标
不同而有不同 draw，重叠位置则必须一致。不同 detector 因 `detector_id` 不同而使用
不同物理随机场。这里的 independent 仅指 target-only/no-neighbors 场景、任务调度和
原子交付；它既不改变 campaign dynamics，也不承诺是 full-frame 图像的严格裁剪。

此前 v1/v2 预检产物没有这条紧凑的直接审计记录，但其同源 `static`/`injected` 是由
同一真实 context scope、run seed、detector 和绝对 frame interval 产生的物理配对；
不能因旧的 `rng_trace_scope` 含有 `case` 而将其判为未配对噪声。它们仍是预检/未完成
科学交付的产物，应按其余 production gate 判断是否可用，而不是以 RNG 不配对为由否定。

`final_dn` 是唯一真实的探测器观测量。其余平面不是第二份图像，而是使团队可以
从 `final_dn` 自行得到电子域图像、选取自己的最优孔径并完成测光的校准和质量信息：

```text
E_cal   = (final_dn - bias_level_sum_dn - column_noise_sum_dn_by_x)
          * gain_e_per_dn
E_bgsub = E_cal - background_expectation_e
```

其中 `background_expectation_e` 是期望值，绝不能用不存在的 background
realization 替代它；否则会错误删除真实的泊松涨落。交付还包括有效像元、full
well/ADC/cosmic count、绝对 raw-frame 区间、时间轴、输入曲线 identity、PSF/坐标
解析、软件版本和完整 manifest。详见
[stamp_science_delivery_zh.md](stamp_science_delivery_zh.md)。

一个 shard 的 raw 与四种 coadd 作为一个集合完成验证后才会对外发布；partial
目录、单个文件或不完整集合都不是可交付科学产品，且不支持续跑。

## 5. 标准快看测光和 CDPP

ET 提供的标准产物不是替代科学团队测光，而是用于回归、快看和可比较的性能检查：

1. 从全部连续 shard 流式读取中心 `13 × 13` 固定孔径，而不是把 90 天图像载入内存；
2. 严格按上式从 `final_dn` 派生电子数；任何 aperture pixel 无效、饱和或 cosmic
   时，该 cadence 记为无效；
3. 用全局时间原点 `0 s` 聚合完整 30、90、390 min 窗口；不逐 shard 计算 CDPP 后
   平均；
4. 采用 legacy-compatible 的 mean-MAD 统计口径
   `1.4826 * mean(abs(x - mean(x))) / mean(x) * 1e6`。

对变量源必须同时给出两类指标：未经模型去除的观测光变 CDPP（包含真实天体变化，
不能解释为仪器噪声）和已知注入 `q(t)` 模型残差 CDPP（以无截距的物理比例拟合
为基准）。后者才是变量注入条件下的仪器型残差指标。

## 6. 可搬迁 manifest 与 H100 执行

`production_manifest.json` 使用 schema v2。所有运行时资源均记录为相对 manifest
根目录的路径，同时保留准备机绝对路径仅作 provenance。这样同一 run root 可以从
`/home/cxgao/Results-sshfs/...` 映射为 H100 的
`/cluster/home/cxgao/sshfs-share/...`，无需修改 manifest，也不会把路径差异误判为
输入篡改。

H100 作业必须显式覆盖本机路径：

```bash
python scripts/run_galaxy_independent_stamp_production.py run-target \
  --manifest /cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
  --source-id <Gaia-ID> --case injected --shard-id <day-index> \
  --data-root /cluster/home/cxgao/ET/Photsim7-data \
  --focalplane-registry /cluster/home/cxgao/ET/et_focalplane-stamp-production/data \
  --device cuda --batch-size 64
```

作业数组的粒度是 `target × 连续日 shard`，初始并发不超过空闲 GPU 数。先完成一
目标一日基准，记录 wall time、RSS、写入量和完整读回时间，再决定每个任务合并多少
个连续日 shard。

全部渲染 array 成功后，使用
`scripts/galaxy_standard_stamp_analysis_array_slurm.sh` 提交一个
`0-9%1` 的 CPU array，并对渲染 array 施加 `afterok` 依赖。它从冻结 manifest 的
target 顺序解析 source ID，严格要求恰好 10 个目标；每个成员固定分析 injected 的
60 s coadd，且仍由 `standard_stamp_analysis` 再次确认该 source 的 90 个 final HDF5、
time-plan、factor snapshot、SHA-256 和 schema。它不会读取 partial/staging 文件，
也不会替换已完成的 analysis 目录。`%1` 是有意的：全文件 validation/hash 会访问
共享存储，串行执行避免在渲染结束后制造不受控的 I/O 峰值。

在提交标准测光前，还必须对**全量** formal delivery 运行一次 metadata-level
completion gate：

```bash
python scripts/audit_galaxy_campaign_delivery.py \
  --production-manifest /cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/production_manifest.json \
  --case injected \
  --output-json /cluster/home/cxgao/sshfs-share/ET_stamp_science/<run>/quality_control/injected_campaign_delivery_qc.json \
  --require-complete
```

它从冻结 manifest 推导完整的 `target × shard × {raw,30,60,120,300 s}` 矩阵，并逐一
检查 final HDF5 的 schema、`complete=true`、`final_dn` 语义、产品/coadd identity、
stamp shape、caller 的 run/case/target provenance，以及每一个 raw-frame 半开区间、时间
起点和曝光长度是否与冻结 time plan 精确一致。任何缺失、错误成员、未知 final HDF5
或残留 `.partial` 都会使 `--require-complete` 返回非零，不能继续声明完整交付。

该 gate **不**重复读取约 9 TB 的 image payload：每个 HDF5 在原子发布前已经执行了
分块的全 payload 合同验证；随后的 60 s 标准分析还会再次完整验证并 SHA-256 绑定它所
实际使用的 90 个输入。三者分别覆盖单文件 payload、全 campaign 时间/产品覆盖和测光
输入身份，不能相互替代。省略 `--require-complete` 时命令可安全用于运行中的进度收据，
但 `ready=false` 的收据绝不是科学交付。

在 H100 上应通过
`scripts/galaxy_campaign_delivery_qc_slurm.sh` 以 render array 的 `afterok` 依赖提交该
gate；随后标准分析 array 必须依赖 QC job 的 `afterok`，而不是直接依赖 render array。
这样 render 的 Slurm 成功、所有 final 成员完整且时间/身份正确，是标准光变与 CDPP 开始
前不可跳过的两个独立条件。

## 7. 其他科学团队输入的当前状态

### SN 团队

`SN_gaiaG_redshift_grid.zip` 中的曲线可用于工程注入检验，但不能进入 Gaia G Vega
正式生产：

- `zpsys=ab`，而正式输入要求 Gaia G Vega；
- 使用裁剪的 `gaia_g_3260_9290` 通带，和完整 Gaia G 的零点/颜色项并不等价；
- 时间采样约为天级，需要团队确认物理相位、重采样规则和曝光内平均定义；
- 文件缺少 RA/Dec 或明确 PSF ID，无法冻结 detector/PSF；
- 需要明确基准星等、宿主与瞬变的分解、clean 曲线选择及机器可读版本/checksum。

此前把 AB 数值暂时当作 Vega 数值的短模拟仅是链路 smoke test，不能作为科学结论或
正式数据产品。

### Aster 团队

Aster 输入可保留为独立的亮星/饱和验证，不进入当前 Galaxy 正式样本。按照已批准
的快看约定，可使用一颗 `G=6` 星和指定的 12 度 PSF 来验证 full-well/ADC 质量
mask；该场景应明确标为饱和测试，不能报告伪造的 CDPP。若未来要进入正式生产，仍
需提供 Gaia G Vega 基准、稳定源 ID、ICRS 坐标或明确 PSF ID、clean 相对光变及
版本 identity。

## 8. 当前不作的科学声明

- 不宣称 independent stamp 的噪声相关性等同 full-frame 或共享曝光模拟；
- 不把 AB 数值当作 Gaia G Vega 的长期转换规则；
- 不替团队自动猜测物理零点、相位、宿主分量或 SED 颜色项；
- 不把中心固定 13×13 aperture 作为团队的最优 aperture；
- 不把 injected 原始光变的 CDPP 误称为仪器噪声。

这些边界会随正式 run manifest 和每个 HDF5 bundle 一起保留，保证数据使用者能
判断哪些量是观察、哪些量是校准派生、哪些结论在当前场景下成立。
