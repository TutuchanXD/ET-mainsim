# 可配置输入与 RNG：ET-mainsim 配套维护

## S3 实施记录（2026-09-20，进行中）

实现分支：两仓 `feat/configurable-workflow-inputs`；基线 Photsim7
`17894e6`、ET-mainsim `a605d08`。本阶段用户已授权完整实施及必要仿真，
可使用 H100 Slurm；生产改动按 Red → Green → Refactor 验证。

- [x] 新增 Photsim7 `load_simulation_spec`，显式工作簿行覆盖应用预设，完整 JSON 可重放。
- [x] CLI 接入工作簿、部分执行 TOML；用户科学设备设置与显式执行设备覆盖顺序。
- [x] 常用全幅/stamp 预设移除历史 PSF/bank 默认摘要与固定 bank；cosmic 默认继承 RNG。
- [x] 保留未显式覆盖的完整观测时间轴。
- [x] 自动资产内容身份、有效配置文件、manifest 输入一致性与星表内容检查。
- [x] 共同曝光存储按固定全局帧组定位，新增 2→3 worker 中断续跑回归。
- [ ] 完成 worker 输入复核、缓存重建/离线读取、失败恢复和边界检查。
- [ ] 补齐源代码/API/package/测试清单契约及依赖候选版本。
- [ ] 完成全套检查、真实工作簿与资产联合验证、CPU/CUDA 验证及 producer 回归。
- [ ] 更新使用说明、issue/PR 和最终验收记录。

当前局部工作流回归 109 passed；全套检查仍在进行，尚不可验收或合并。
真实 default.xlsx 已成功与全幅和 stamp 生产预设组合解析，未改动工作簿或数据资产。

- 日期：2026-09-17。
- 状态：已确认方向，待实施；本次只发布计划和 issues，不运行仓库测试。
- 主计划：同工作区 `Photsim7/docs/devs/configurable_inputs_rng_maintenance_plan_zh.md`。
- 公开总跟踪：[Photsim7 #164](https://github.com/TutuchanXD/Photsim7/issues/164)（含完整计划）。
- 本仓任务：[ET-mainsim #51](https://github.com/TutuchanXD/ET-mainsim/issues/51)。

## 1. 兼容范围与职责

用户确认 ET-mainsim 是 Photsim7 唯一必须维护的外部兼容对象，用于方便运行 ET 全幅和 stamp。维护 full-frame、catalog/table stamp，以及这些入口依赖的生产工作流。允许修正错误设计导致的旧数值变化，无需保持额外的历史结果。

Photsim7 负责参数模型、科学计算、RNG 和科学产物规则；ET-mainsim 负责 ET 预设、CLI、调度、manifest、缓存、输出和续跑。ET-mainsim 不再保存第二套科学默认值规则或随机算法。

## 2. 近期 S3 任务

- [ ] 更新实际使用的 smoke/production 等预设，移除默认资产摘要和固定历史 jitter bank。
- [ ] 支持工作簿/JSON 的统一科学配置入口，经 Photsim7 loader 解析。
- [ ] 明确 ET 默认值 → 用户配置 → 显式 CLI 的优先级；保留 `--seed` 总种子入口，分项种子经配置完整传递。
- [ ] 导出最终有效配置；记录实际资产内容、RNG 输入/派生值、科学身份、软件环境、后端和精度。
- [ ] 用户不填写 SHA256；内容摘要用于缓存失效和同一 run 的一致性，不能变成历史资产准入条件。
- [ ] 修改完成判定、selection 记录和续跑逻辑，使自定义合法资产可正常完成；不同输入不能混入同一次续跑。
- [ ] 全幅、catalog/table stamp 及相关 producer 均验证；区分共同曝光裁剪和独立随机场景。
- [ ] 同输入重复运行、不同调度及中断续跑满足 Photsim7 声明的确定性保证。
- [ ] 与 Photsim7 联合发布真实 `default.xlsx`，保留用户参数、格式、其他 sheet、符号链接和备份；五类效应显式开启；数据目录不初始化 Git。
- [ ] 更新依赖约束、CLI/config 示例、运行与迁移说明。

主要触点：`src/et_mainsim/cli.py`、`presets/`、`workflows/full_frame.py`、`workflows/stamp.py`、manifest/provenance/cache/resume 和受维护 producer。

## 3. 依赖与验收

适配工作可提前开展，但 S3 关闭前必须完成 Photsim7 [S1 输入/资产修复 #162](https://github.com/TutuchanXD/Photsim7/issues/162) 与 [S2 RNG 参数化 #165](https://github.com/TutuchanXD/Photsim7/issues/165) 的实现与候选联合验收。保持每个合并点两仓可用，必要时分适配与切换两步交付。

| 场景 | 验收 |
| --- | --- |
| 配置等价 | 工作簿与等价 JSON 解析为相同有效配置和科学结果 |
| 自定义资产 | 合法 PSF/PSD 等替换后可启动、完成和读回，无默认历史摘要门槛 |
| 种子控制 | 总种子和分项覆盖从用户输入传递到实际消费者 |
| 调度/续跑 | 同物理实验不受 worker/分片/输出目录影响；续跑不重复或替换随机实现 |
| 输入变化 | 内容或科学参数变化正确失效缓存，拒绝混合续跑，允许建立新实验 |
| 工作流 | full-frame、catalog stamp、table stamp 和依赖的 producer 回归通过 |
| 交付 | 小规模真实资产验收、适用全套检查和所声明后端保证完成 |

未来生产代码修改按 Red → Green → Refactor 实施。本次文档交付不运行仓库测试，不启动 CPU/GPU 仿真或正式 campaign。

## 4. 与既有计划的关系

- `et_mainsim_four_pr_maintenance_plan.md` 是历史阶段记录；其中固定 10 s、xlsx 仅兼容入口等假设将由新维护阶段逐项更新，不能据此阻止本次已确认的方向。
- [ET-mainsim #43](https://github.com/TutuchanXD/ET-mainsim/issues/43) 的正式 campaign 独立跟踪。本任务保护其依赖入口，既不启动生产，也不自动宣称该任务完成。
- 后续 Photsim7 时间参数化、仪器布局、几何及标定任务只在能力实际交付后同步接入。本任务不实施 Kepler 仿真或 equivalent-coadd 近似。
