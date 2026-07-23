# Galaxy 独立 Stamp 正式科学交付与 raw-coverage 分析说明

本文定义 Galaxy 团队内禀光变独立 stamp 正式数据的稳定交付合同，以及从原始
`final_dn` 到标准参考光变和 coverage-aware CDPP 的唯一正式分析链。它面向数据
使用者、科学团队和运行维护者，不以某次 Slurm job ID、某个挂载路径或运行中进度作为
依据。

当前正式 Galaxy 生产的输入解释、几何映射和物理效果配置见
[galaxy_independent_stamp_production_zh.md](galaxy_independent_stamp_production_zh.md)；
每个 HDF5 的完整 schema 见
[stamp_science_delivery_zh.md](stamp_science_delivery_zh.md)。本文不替代这两份合同，
而是明确它们如何组成可交付的科学数据流程。

## 1. 适用范围与产品状态

本合同仅适用于满足下列条件的 Galaxy formal campaign：

- 输入为 Galaxy FITS 中的 Gaia G Vega 基准星等、ICRS 坐标和 clean
  `Delta F / F_ref`；
- 每个目标是独立、target-only/no-neighbors 的 `100 x 300` pixel stamp；
- raw exposure 为 10 s，像元尺度为 4.83 arcsec/pixel；
- 生产 manifest、factor snapshot、focal-plane registry、PSF bundle、时间计划和软件
  provenance 均已冻结并经内容 identity 复核；
- 交付只在 campaign QC、单源严格测光、coverage-aware 分析和 campaign summary 都
  成功后才被声明为 ready。

`independent stamp` 是一种场景与交付粒度，不是 full-frame 的严格裁剪：它没有邻星
场，也不能据此声称全幅/共享曝光的混合、拥挤或共同系统误差已经被重现。另一方面，
同一 detector、同一绝对 raw-frame 的随机物理场仍按绝对像元/块/列地址派生；不同
source 不是人为重置的独立 detector random seed。`static`/`injected` pair 用相同的
物理 RNG identity，`case` 仅是执行和真值标签，不进入物理随机数 identity。

Galaxy 正式产品与以下资产必须严格区分：

| 资产 | 状态 | 可否作为本合同的正式 CDPP 依据 |
| --- | --- | --- |
| 90 天 Galaxy raw `final_dn`、QC、raw 10 s strict、coverage v2、十源 summary | 正式候选；必须通过全部 gate | 可以 |
| 30 天 Galaxy/SN/Aster static-vs-injected notebook | 工程快看/链路验证 | 不可以 |
| Aster `G=6` 产物 | full-well/ADC 饱和响应验证 | 不可以 |
| 旧 60 s coadd standard-analysis receipt | 历史部署证据 | 不可以 |
| Photsim7 legacy pickle/PCA/OA workflow | 历史分析基线 | 不直接适用于 independent stamp |

## 2. 唯一观测量与可校准电子域图像

每个 HDF5 bundle 中，**`final_dn` 是唯一真实的探测器观测量**。它已经包含本次 raw
exposure 的源、背景、暗电流、光子散粒噪声、读出、bias、列噪声、full-well/ADC 和
cosmic-ray 等在正式配置中启用的链路效果。`background_expectation_e`、bias、列噪声、
质量 mask 和 provenance 不是另一幅观测图像，也不能与 `final_dn` 并列交付为可随意
替换的 science image。

科学团队若需在电子域做自定义 aperture photometry，应按每个输出帧和像元计算：

```text
E_cal = (final_dn - bias_level_sum_dn - column_noise_sum_dn_by_x)
        * gain_e_per_dn

E_bgsub = E_cal - background_expectation_e
```

其中 `column_noise_sum_dn_by_x` 按 stamp 的 x 列广播至 y；`gain_e_per_dn` 可能是根
属性的标量，也可能是同名 dataset。`background_expectation_e` 是背景的**期望电子数**，
不是本次随机背景 realization。正式 bundle 明确声明
`background_realization_used=false`，因此严禁构造或假定一个不存在的 background image
再从 `final_dn` 中扣除；那会错误抹去真实背景泊松涨落。

建议的自定义测光步骤是：

1. 选择 science aperture 与可选的背景估计区域；
2. 对每个 aperture pixel 检查 `valid_mask`、`saturated_mask` 和 `cosmic_mask`，并根据
   研究目的决定是剔除 exposure、掩蔽像元还是采用自己的质量模型；
3. 从 `final_dn` 和上述 companions 计算 `E_bgsub`，再做 aperture sum；
4. 保留所用像元、质量判断、背景估计、曝光时间和输出单位的 manifest，以便可复现。

ET 提供的中心 `13 x 13` fixed aperture reference curve 只是统一 QA/比较的参考口径，
不是对科学团队“最优 aperture”的规定。

## 3. 输入曲线、时间对齐和光变注入

Galaxy FITS 的正式解释如下：

| 输入字段 | 正式含义 |
| --- | --- |
| `Source` | Gaia source ID；以 int64 保存和冻结 |
| `Gmag` | Gaia G Vega 基准星等 |
| `RAJ2000`, `DEJ2000` | ICRS/J2000 坐标；映射 detector、物理像元和 field angle |
| `class` | 科学类别；保留 provenance，不决定光子数公式 |
| `relative_flux` | `Delta F / F_ref` 的 clean 内禀相对流量 |
| `time` | 只保留有限节点之间的相对间隔；绝对 epoch 不作为 ET 仿真时间 |

注入因子定义为：

```text
q(t) = 1 + Delta F(t) / F_ref
```

第一条有限节点对应 simulation raw frame 0。程序构造 clean、分段线性的 `q(t)`，并为
每个 10 s exposure 计算区间精确平均值，冻结为每源的 factor snapshot。它不会把三天
或数天采样点直接复制为大量相同的 10 s 点，也不会读取源文件的绝对 MJD/epoch 作为
仿真时间。coadd 始终是先逐 raw frame 注入、渲染、数字化，再在 detector DN 域相加。

位置和 PSF 的正式路径是：

```text
RA/Dec
  -> frozen ET focal-plane registry
  -> detector、physical pixel、field angle
  -> nearest available PSF node
  -> 100 x 300 target-only stamp
```

光变发生在 PSF 投影和恒星泊松抽样**之前**：Gaia G Vega 基准源电子期望先乘以 `q(t)`，
随后经过 PSF、恒星泊松、背景/暗电流、读出与数字化。因而源亮度变化同时改变期望光子
数和其散粒噪声；它不是对完成的 `final_dn` 图像作后处理缩放。

## 4. 物理与交付配置边界

正式 Galaxy profile 使用温度表驱动的 legacy-aligned PSF breathing：TESS 温度数据给出
热状态，PSF scale 按已冻结的 reference temperature 和 scale-per-degree 配置计算。
它不是早期试验性“每 3 天线性 sawtooth”实现。动态温度、指向、PSF、源/背景光子、暗
电流、读出、数字化和 cosmic-ray 的实际开关都应以 `production_manifest.json` 和每个
HDF5 的 `provenance_json` 为准。

正式基线中 SD-20 detector-response 项（inter-pixel、intra-pixel、pixel phase、scripted
pixel/whole-pixel/flat gain）按已审议配置 fail-closed 关闭。此处的“关闭”不是可以由
下游数据使用者隐式补回的校正；若研究需要这些效果，必须准备新的、完整冻结且验证的
科学配置。

每个 target-time shard 的交付集合为：

```text
raw.h5                  # 10 s, coadd_factor=1
coadd_30s.h5            # 连续 3 raw frames 的 DN 求和
coadd_60s.h5            # 连续 6 raw frames 的 DN 求和
coadd_120s.h5           # 连续 12 raw frames 的 DN 求和
coadd_300s.h5           # 连续 30 raw frames 的 DN 求和
```

writer 先在同目录写入 partial，完成 HDF5 readback/contract 验证后才原子 rename 到 final。
一个 shard 的 raw 和四个 coadd 成员必须完整；`.partial`、`.lock`、`.incoming`、scratch
残留、未知 HDF5 或不完整集合均不可交付，且正式生产不从 partial 续跑。

### 4.1 本次 v3 与后续 staged 生产的写入模式

`galaxy_independent_90d_v3` 的 production manifest 在 staged writer-mode 冻结功能合入
之前准备，因此其历史语义是 `direct_shared_filesystem`：worker 直接在最终共享文件系统
根写入上述 partial，再完成 readback 后原子 rename。最终报告必须如实保留这一事实，不能
将 v3 事后描述为 node-local staged production。

后续新 campaign 则必须在 prepare 时显式冻结 `staged_local_scratch_v1`：每个 worker 先在
node-local scratch 写完整 raw/coadd 集合，校验 production manifest、target、time shard 和
HDF contract 后再原子 publish 到共享根。staged publisher 对 manifest/time-plan 漂移、错误
target/shard、incoming/lock/partial 残留均失败关闭；它改善共享盘写入鲁棒性，但不会改变
`final_dn` 的物理定义或本节的科学输入/输出语义。

## 5. 正式分析链与放行 gate

正式分析严格按下列顺序执行：

```text
delivery raw/coadd HDF5
        |
        |  campaign delivery QC: 完整矩阵、schema、时间、identity、残留
        v
raw_10s_strict
        |
        |  central 13 x 13 reference aperture + cadence-level quality gate
        v
raw_10s_coverage_v2
        |
        |  global time bins + coverage accounting + no imputation + CDPP
        v
raw_10s_coverage_v2_summary
```

正式交付只有在四个 gate 都成功时才可声明 ready：

1. **Campaign delivery QC**：从冻结 manifest 派生全部
   `target x time-shard x {raw,30,60,120,300 s}` 成员，检查 HDF5
   `complete=true`、`final_dn` 语义、schema、target/case/run provenance、绝对 raw-frame
   区间、时间轴和所有未完成残留。它不以文件数相同代替 identity 校验。
2. **raw 10 s strict**：从连续 raw HDF5 提取固定中心 `13 x 13` aperture。该 aperture
   中任一 pixel 的 `valid_mask=false`、饱和/full-well/ADC 或 cosmic flag 命中时，整条
   10 s cadence 失效；不做像元修补、flux 补点或以邻近帧替代。
3. **raw 10 s coverage v2**：只在完整、干净的 10 s cadences 上聚合全局时间窗口；输出
   coverage、有效曝光、accepted/rejected 原因和统计量。它不会把短曝光伪装成完整曝光，
   也不会插值补齐坏 cadence。
4. **campaign summary**：逐 target 汇总 v2 单源结果，并再次绑定 policy、production
   manifest、输入 HDF5 与 QC identity。它不是把十个源做物理意义上的“平均光变曲线”。

本次 formal 路径的典型输出树为：

```text
analysis/
  source_<Gaia-ID>/injected/
    raw_10s_strict/
      reference_lightcurve.csv
      analysis_manifest.json
    raw_10s_coverage_v2/
      coverage_aware_binned_lightcurve.csv
      coverage_aware_analysis_manifest.json
  campaign/injected/
    raw_10s_coverage_v2_summary/
```

实际 run root 还会写出 campaign QC receipt、冻结 coverage policy 和最终交付说明；使用
者应读取 manifest/receipt 内的 content identity，而不是仅依赖本说明中的示例路径。

## 6. coverage policy 与 CDPP 的精确定义

正式 policy 必须作为不可变 JSON 与分析输出一起交付。当前 raw 10 s coverage v2 的
语义是：

| 项目 | 规则 |
| --- | --- |
| 观测产品 | `final_dn`；不使用 background realization |
| case | `injected` |
| 输入 cadence | 10 s raw HDF5 |
| aperture QA | fixed central `13 x 13`；坏像元使整个 cadence 无效 |
| bin 原点 | 全局 `t=0 s`，不在 shard 边界重置 |
| 窗口 | 30、90、390 min |
| minimum coverage | 0.95 |
| minimum accepted bins | 10 |
| 有效 bin 归一化 | 只按实际有效 exposure 归一化 |
| 坏 cadence | 整条 omission；不补像元、不补 flux、不插值 |

对于每个窗口，统计只使用 policy 指定的 accepted bins。CDPP 的 legacy-compatible
robust statistic 为：

```text
MAD_sigma = 1.4826 * mean(abs(x - median(x)))
```

若 `x` 是归一化 flux/rate，CDPP 以 ppm 表示；若 `x` 已是相对模型残差，直接以相对
量换算 ppm。严格使用相同的有效 bins、window origin 和 policy identity；不可把各个
shard 分别求 CDPP 再平均。

每源必须同时区分：

- **observed light-curve CDPP**：包含已注入的真实天体内禀变化和仪器噪声，不能解释为
  “纯仪器 CDPP”；
- **known-q(t) model residual CDPP**：从已冻结输入因子得到的残差诊断，才可作为变量
  源条件下的仪器型性能指标。

若某窗口可用 bins 少于 `minimum_accepted_bins`，CDPP 的 `null` 是正确且应保留的科学
结果，不能写成 0、NaN 后再平均或由其他窗口插补。summary 中的跨源数值也不能被称为
一个单一的 campaign-average instrument CDPP；应逐 source、逐 window、逐 coverage
状态解释。

## 7. 与 legacy 分析的关系

本链只复用了 legacy 已验证的 **mean-MAD robust statistic 形式**。它没有将
independent stamp HDF5 伪装成历史 legacy pickle，也没有运行 PCA、Savitzky--Golay、
历史 `bin_lcs`、OA 或“最优 aperture”流程。

原因是完整 legacy analysis 的数学和数据契约需要同场多星、方形/全场数据布局、ensemble
PCA/OA 辅助量与明确的 target-exclusion 规则；对单目标、非方形 `100 x 300` independent
stamp 强行运行会错误地移除天体信号或制造没有物理解释的 correction。因此本产品可以称
为 **legacy-MAD-compatible**，不能称为“完整 legacy 分析链已对齐”。未来如需 full
legacy parity，必须另行生产符合 legacy 入口 contract 的同场数据及相应无噪声/OA 辅助
产品。

## 8. 使用、追溯与已知限制

每次交付应至少让数据使用者获得：

- 所有 final raw/coadd HDF5，以及完整的 production manifest、time plan、input/factor
  snapshot identity 和 HDF5 manifest/provenance；
- campaign QC receipt、冻结 coverage policy、每源 strict/v2 manifest 与 campaign
  summary；
- 本文、HDF5 schema 说明、run-specific `FINAL_DELIVERY_zh.md`；
- 可只读复核的 notebook（它们是消费者，不是正式 producer）。

在实际分析前，使用者应逐项确认：production manifest 的 run ID/sha256、policy sha256、
campaign QC ready 状态、目标 ID、case、HDF5 `complete`、input snapshot identity 以及
所用 HDF5 清单。任一项缺失或不匹配时，不应自行宣布数据“可交付”。

目前只有 Galaxy FITS profile 具备此处定义的正式输入语义。SN 输入仍需 Gaia G
AB-to-Vega/通带、90 天曲线、真实或明确的 detector/PSF 与 science input contract；
Aster 的原始 G=6 数据保留为饱和验证，非 precision/CDPP sample。它们的 30 天工程快看
结果可用于链路诊断，但不能与本合同的 Galaxy formal 产品混为一谈。

## 9. run-specific 最终说明的最低内容

每个完成 campaign 的 run root 必须另写 `FINAL_DELIVERY_zh.md`，并在完成后冻结以下
实际值：

1. production manifest、time plan、frozen coverage policy、输入 FITS/factor snapshot、
   QC receipt、单源分析和 campaign summary 的 identity；
2. target 清单、duration、raw/coadd cadence、stamp shape、pixel scale、正式软件 commit
   和 clean/dirty provenance；
3. campaign QC 的完整覆盖状态、无残留声明及任何明确的排除项；
4. 每源每窗口的 observed CDPP、model-residual CDPP、coverage、accepted/rejected bins，
   并保留 `null`；
5. 已知边界、legacy compatibility boundary、SN/Aster 的状态及数据使用注意事项。

历史 receipt 不得被覆盖或编辑成新状态。若 run root 同时保存旧 coadd 分析 receipt 和
本合同的新 raw-coverage 结果，应新增 disposition sidecar 明确标明旧产物是 historical、
held、superseded 或其他**真实**状态；最终权威永远是冻结的 QC/policy/analysis manifests
和本次 `FINAL_DELIVERY_zh.md`。
