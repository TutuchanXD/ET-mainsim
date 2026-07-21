# ET-mainsim 四 PR 完整维护计划

> 状态：四个实施阶段已完成。主分支目标结构由 PR 4 收口；删除前脚本树
> 固定在 annotated tag `legacy-scripts-final`。本文件继续作为设计决策和
> 验收矩阵记录，不再作为待办清单。

## 1. 状态与目标

本文档是 ET-mainsim 从实验脚本集合收敛为 Photsim7 参考工作流应用的
权威实施计划。计划覆盖两个仓库：

- `ET-mainsim`：维护 CLI、run preset、运行目录、进程/GPU/Slurm 调度、
  resume、日志和用户教程。
- `Photsim7`：维护 scientific model、typed spec、catalog、PSF、dynamic
  effects、renderer、detector chain、RNG 和标准 data products。

最终交付三个可直接运行的端到端工作流：

```text
et-mainsim run et-full-frame --preset smoke|production
et-mainsim run et-stamp      --preset smoke|production
et-mainsim run legacy-sim   --preset full-effects-smoke|full-effects-production
```

四个 PR 完成后，ET-mainsim 不再拥有任何独立的物理模型或 detector
实现，但仍保留必要的应用层模块。这里的“thin”指不重复 scientific
domain logic，不是指仓库只能包含无法测试的单文件脚本。

## 2. 已确认决策

1. 仓库继续命名为 `ET-mainsim`，定位为 Photsim7 的 ET 参考工作流。
2. ET full-frame v1 表示单个 physical `main_rd` detector；不扩展到
   100-detector 或 multi-telescope focal-plane batch。
3. ET stamp 使用真实 target、邻近星和 absolute detector coordinates；
   标准 final product 位于 detector domain，electron-domain components
   可选保存。
4. Canonical stamp cadence 为 10 s raw cadence；长曝光通过显式 coadd
   构建，不沿用直接放大读出噪声的近似。
5. Legacy workflow 必须复刻 `scripts/et_sim_100_det.py` 默认
   `ET_EFFECT_PROFILE=full` 中全部启用的效果。
6. Scientific configuration 使用 Photsim7 `SimulationSpec` JSON；
   ET-mainsim execution policy 使用 TOML。
7. xlsx 仅保留为 Photsim7 compatibility input；ET-mainsim 中过期的
   42-row workbook 最终删除。
8. 新运行写统一 `run_manifest.json`；旧输出保证只读，不承诺跨版本
   resume 或继续写入。
9. 第一阶段面向内部 `etbase`/`etbase-clu` 用户，Python 版本与
   Photsim7 对齐为 `>=3.12,<3.14`。
10. 支持 local CPU smoke、local CUDA multiprocess 和 H100 Slurm；Ray
    仅属于 legacy workflow。
11. 历史实现通过 Git tag 和迁移表追溯，不在主分支保留新的
    `archive/` 代码副本。

## 3. 目标架构

```text
ET-mainsim/
  pyproject.toml
  src/et_mainsim/
    __init__.py
    __main__.py
    cli.py
    config.py
    manifest.py
    provenance.py
    presets/
      et_full_frame_smoke.spec.json
      et_full_frame_smoke.run.toml
      et_full_frame_production.spec.json
      et_full_frame_production.run.toml
    workflows/
      __init__.py
      full_frame.py
      stamp.py                 # PR 4
      legacy.py                # PR 4
  slurm/
    et_full_frame.sbatch
    et_stamp.sbatch            # PR 4
    legacy_sim.sbatch          # PR 4
  benchmarks/
  tests/
  docs/
```

### 3.1 Scientific/runtime boundary

`SimulationSpec` JSON 只表达可复现的仿真语义。运行节点相关内容不得
写入 scientific spec，包括：

- output root；
- local GPU ids、workers per GPU；
- Slurm partition/resources；
- resume/overwrite；
- preview count 和 benchmark report path；
- 本机 checkout path。

这些字段由 run TOML 或 CLI 提供。环境路径在 ET-mainsim 边界展开后，
resolved `SimulationSpec` 必须完整写入 manifest。

### 3.2 Run manifest

`run_manifest.json` 使用 `et_mainsim.run_manifest` schema，至少记录：

- schema id/version；
- workflow、preset、run id、状态和时间戳；
- resolved `SimulationSpec`；
- resolved execution configuration；
- frame plan 和 completion summary；
- catalog cache identity/path；
- ET-mainsim 与 Photsim7 Git provenance；
- output product schema 和关键 artifact paths；
- failure type/message（失败时）。

Manifest 采用 atomic replace 写入。父 launcher 是唯一 run-level writer；
worker 只写 worker summary 和 frame artifacts。

### 3.3 Resume contract

- `resume=true`：仅跳过同时通过 data product 与 summary/schema 检查的
  complete item。
- 只有 payload 文件而没有 summary/schema 时视为 incomplete。
- manifest 中 scientific spec、catalog request 或 execution identity
  冲突时 fail closed。
- `overwrite=true` 与 resume 互斥。
- dry-run 不创建目录、不查询 catalog、不加载 PSF、不初始化 CUDA。

## 4. PR 总览与依赖

```text
PR 1: ET-mainsim full-frame application ────────────────┐
PR 2: Photsim7 package-level stamp pipeline ── parallel ├─> PR 4
PR 3: Photsim7 legacy full-effect contract ─── parallel ┘
PR 4: ET-mainsim integration, deletion and release
```

PR 1 可独立提供可用的 full-frame CLI。PR 2 与 PR 3 在 Photsim7 中互不
依赖，可以并行开发。PR 4 只在两个 package API 都稳定后合并。

Photsim7 package-topology #84-#92 仍是长期 namespace 迁移路线；四个
交付 PR 只依赖 documented public facade。#92 负责把 maintained downstream
迁往 canonical namespace，不阻塞 PR 1 的用户可用性。

## 5. PR 1 - ET full-frame 应用与运行契约

### 5.1 范围

- 新增 `pyproject.toml` 和 `et-mainsim` console entry point。
- 新增 lightweight package，顶层 import 不加载 Torch、Ray 或外部资产。
- 新增 canonical spec JSON、run TOML、preset discovery 和 validation。
- 新增 `et-mainsim presets`、`et-mainsim show`、
  `et-mainsim run et-full-frame`。
- 使用 `SimulationSpec`、`StarCatalogCache`、`build_catalog_from_spec`、
  `build_full_frame_services`、`run_single_cadence_full_frame` 和
  `FullFrameArtifactWriter`；不得调用 ET-mainsim legacy physics builders。
- 支持 CPU in-process、local subprocess/CUDA worker assignment 和 Slurm
  template。
- 支持 frame selection、resume、overwrite、cache-only、dry-run、preview、
  optional cosmic mask 和 stellar mean。
- 实现 run manifest、worker summary、failure status 和 Git provenance。
- 在 compatibility `MainRdRunSpec` 中补齐 `target_epoch_jyear`，关闭 #6；
  epoch 必须进入 canonical spec、catalog request/cache identity 和 provenance。
- 将当前未提交的 main-rd benchmark evaluator、测试和 Slurm pilot 收入
  `benchmarks/` 或保留为有明确 deprecation plan 的兼容工具。
- 更新 README/getting-started/validation 文档。

### 5.2 非目标

- 不实现 stamp 或 legacy workflow。
- 不改变 Photsim7 scientific formula、effect profile 或 artifact schema。
- 不删除仍由 historical wrappers 和 stamp-long 使用的 compatibility code；
  删除工作集中在 PR 4。
- 不承诺从旧 run directory 继续写入。

### 5.3 必须测试

- RED/GREEN：CLI help/list/show/dry-run、invalid config、environment expansion。
- RED/GREEN：spec/run config separation、epoch override、manifest transition。
- RED/GREEN：worker assignment、resume validation、failure propagation。
- Contract：active path 只能调用 Photsim7 package services/pipeline。
- Hermetic tiny frame：真实 Photsim7 renderer + detector chain + artifact readback。
- Installed wheel/editable install：clean subprocess 中 CLI 可导入。
- Existing ET-mainsim full test suite。
- H100：small real-asset smoke；full detector one-cadence readback 在 package
  dependency 未改变时允许复用 #5 的 9120x8900 证据，否则必须重跑。

### 5.4 完成标准

```text
et-mainsim presets
et-mainsim show et-full-frame-smoke
et-mainsim run et-full-frame --preset smoke --dry-run
et-mainsim run et-full-frame --preset smoke --device cpu
```

以上命令可从安装后的环境运行；manifest、frame、summary 和 schema 可由
独立 readback 测试读取；GitHub PR 为 ready 状态。

## 6. PR 2 - Photsim7 package-level stamp pipeline

### 6.1 范围

- 新增 `StampServices` 与 `build_stamp_services(...)`。
- 新增 `run_single_cadence_stamp(...)` 和 standard stamp product schema。
- 输入支持 prepared/queried catalog、target selection 和邻近星裁剪。
- 保留 absolute detector coordinates，并从 physical position 选择 PSF field。
- 使用 `EffectTimeseries.for_cadence(...)` 和 per-source projection。
- 统一 stellar/background/scattered/dark/readout、detector response、full well、
  gain、ADC、bias/column noise 和 2-D cosmic-ray ordering。
- 新增 10 s raw cadence 和显式 coadd contract。
- 扩展 `StampShardWriter/Reader` provenance，不破坏已有 schema reader。
- 提供 SeedTree/rng trace、truth、events 和 component metadata。

### 6.2 完成标准

- 同一 spec/seed/device 产生 deterministic-equivalent output。
- 单 target、target+neighbor、edge clipping、rectangular stamp、coadd、cosmic
  和 detector-response tests 全部通过。
- full-frame 与 stamp 对同一局部 scene 的 component-level contract 一致。
- 禁用组件时不解析对应外部资产。
- Photsim7 full suite、installed-wheel 和 ET-mainsim pinned contract 通过。

## 7. PR 3 - Photsim7 legacy full-effect workflow

### 7.1 权威效果集合

`legacy-sim/full-effects` 必须显式固定以下 enabled effects，禁止依赖
`Variant.accepted_settings` 的隐式默认值：

| 类别 | 必须启用 |
| --- | --- |
| Scene | target star, background stars |
| Photon/noise | stellar photon noise, background, scattered light, dark current, readout noise |
| Gain | scripted gain, whole-pixel normal gain, whole-pixel sinusoidal gain |
| Motion | ET PSD low-frequency drift, TESS roll drift, DVA, thermal drift, momentum dump |
| PSF | frozen native ET 100 x 3 x 300 jitter bank; TESS-temperature-driven PSF breathing |
| Pixel response | inter-pixel, intra-pixel, pixel-phase response |
| Reduction | coadding, Kepler optimal aperture and OA helper variants |

以下保持 disabled：flat-field correction、pixel-flux filtering、transit
injection。Cosmic rays 不在原 `et_sim_100_det.py` full profile 的 active
legacy path 中，不得虚构为 legacy parity；用户可通过独立 opt-in preset
启用 package cosmic pipeline。

### 7.2 范围

- 新增显式 typed legacy ET preset/factory 和 effect inventory metadata。
- `Simulator` 提供可预测的 local Ray mode；不使用硬编码远程地址。
- `start_ray=False` 的语义明确为 setup-only，或实现可运行的 local
  execution path；不得在 `run()` 中因缺失 `data_manager` 才失败。
- 保留 legacy pickle/NPZ readers 和 OA/variant outputs。
- 将旧 global RNG 使用收敛到可记录的 seed contract；不能改变 legacy
  必须依赖的序列顺序而无测试证据。
- 以组件级差分测试证明每个 enabled effect 被实际消费，不只检查布尔值。

### 7.3 完成标准

- 一个 CPU/local-Ray smoke 和一个 CUDA smoke 可端到端生成 legacy outputs。
- manifest/provenance 列出每个 enabled effect、数值参数和 runtime consumer。
- 对可隔离组件提供 seeded parity 或有界数值差异证据。
- 不延续旧 workbook 的 101% optical efficiency 或过期 magnitude formula。
- Photsim7 Simulator/variant/Ray/full suite 通过。

## 8. PR 4 - 三工作流集成、历史删除与发布

### 8.1 范围

- 接入 `et-stamp` 和 `legacy-sim` CLI/presets/Slurm templates。
- 三工作流共用 run config、manifest、provenance 和错误模型。
- 迁移或删除：
  - `scripts/et_sim_100_det*.py`；
  - `stamp_long/stamp_long_core.py` 及 benchmark-only wrappers；
  - `main_rd_g18_parallel/main_rd_parallel_core.py` 中不再可达的 physics
    compatibility builders；
  - historical `main_rd_grb/et_sim_10*.py`；
  - duplicate `main_rd_1000_eval` rendering/config implementation；
  - `config/et_100_det_inputs_1h.xlsx`；
  - stale README/checklists。
- GRB-specific continuation/last90 tools 迁往 approved downstream location，
  或明确保留为 `tools/` 中只读 artifact postprocessing。
- 建立旧命令到新命令、旧输出到 reader 的 migration table。
- 完成 CPU/CUDA/H100 production validation 和首次 ET-mainsim application
  version/tag。

### 8.2 完成标准

- 三套 smoke commands 从 clean install 运行成功。
- 三套 production preset 通过 config/asset preflight。
- ET full-frame H100 one-cadence、stamp shard、legacy full-effects output 均有
  独立 readback evidence。
- `rg` 证明 maintained runtime 不再 import legacy physics builders、
  `photsim6` alias 或 repository `sys.path` injection。
- 工作区干净，文档、CLI help 和 shipped preset 完全一致。

## 9. Verification matrix

| Gate | PR 1 | PR 2 | PR 3 | PR 4 |
| --- | --- | --- | --- | --- |
| Unit/contract tests | required | required | required | required |
| Clean-process import | required | required | required | required |
| Installed package CLI | required | n/a | n/a | required |
| Hermetic CPU smoke | full-frame | stamp | legacy setup/local | all |
| Local CUDA smoke | full-frame | stamp | legacy | all |
| H100 small smoke | required | required | required | all |
| H100 full-size evidence | reuse/rerun by dependency | representative stamp | representative legacy | final audit |
| Artifact readback | full-frame | stamp | pickle/NPZ | all |

H100 作业必须使用 `etbase-clu`，写入 `/home/cxgao/Results-sshfs` 对应挂载，
使用 Slurm `sbatch`，并在结束后确认无 orphan multiprocessing workers。

## 10. Completion audit

四 PR 目标只有在以下证据同时成立时才算完成：

1. 三条 CLI workflow 均存在 smoke 与 production preset。
2. shipped spec/run config 均可解析，resolved spec 写入 manifest。
3. full-frame、stamp 和 legacy 每条 active call chain 都只调用 Photsim7
   scientific APIs。
4. legacy full-effect inventory 中每个 enabled effect 都有 runtime-consumer
   test 或 component-level evidence。
5. old workbook、source-path alias、duplicate physics implementation 已删除。
6. old artifacts 有清晰 reader/migration 说明；不声称支持未经验证的 resume。
7. local tests、installed-package tests、H100 checks 和 artifact readback 全部
   有精确 commit/job/output evidence。
8. 所有相关 PR ready/merged，GitHub issues 状态和文档同步，最终 worktree
   clean。
