# ET 多星 Stamp 长时间仿真资源评估实施方案

## 0. 修订口径

本方案用于评估 ET 多星、target-star-only stamp 长时间仿真的资源需求。上一版方案中的曝光组存在错误，本版采用正确曝光组：

```text
30 s / 60 s / 180 s / 300 s
```

本版不再采用单个全年 benchmark 再外推的口径。正式评估矩阵直接采用用户指定的两类规模：

| 规模组 | stamp | 曝光 | 星数 | 时长 | 主要用途 |
| --- | --- | --- | ---: | ---: | --- |
| 短时高星数 | 11x11 / 15x15 | 30 / 60 / 180 / 300 s | 1680 | 1 day | 8 种组合的吞吐、I/O、文件数、GPU 并行效率 |
| 长时低星数 | 11x11 / 15x15 | 30 / 60 / 180 / 300 s | 240 | 7 days | 8 种组合的长时间动态项、CR、drift、manifest、续跑稳定性 |

两个规模组的总 star-day 相同：

```text
1680 stars x 1 day = 240 stars x 7 days = 1680 star-days
```

因此同一 stamp 和曝光下，两组输出文件数和像素载荷相同；差异主要体现在 worker 组织、frame loop 长度、动态项时间演化、续跑稳定性和 manifest 压力。

---

## 1. 输出定义

| 项目 | 设定 |
| --- | --- |
| 输出产品 | one target star x one output frame -> one 2-D stamp |
| stamp 内容 | 只包含目标星，不包含背景星和邻星 |
| 输出单位 | electrons |
| 输出 dtype | `float32` |
| 输出格式 | `npy` 用于 smoke/debug；production 使用 sharded HDF5 |
| 文件粒度 | NPY 为单个 star-frame；HDF5 为单个 worker/case shard |
| 输出 cadence | 等于输出曝光时间 |
| stamp 尺寸 | 11x11 和 15x15 都作为正式评估组合 |
| duty cycle | 100% |
| quarter/roll | 本轮不评估，固定焦面条件 |
| 目标星亮度 | 暂不接入外部星表；每颗星按 ET mag 12.5-14.5 均匀随机采样 |

本轮仍采用直接长曝光近似：不显式生成 10 s 子曝光并离线叠加，而是直接按目标曝光时间生成最终 stamp。10 s 只作为参数缩放参考。
ET mag 到 detector electron/s 的转换调用 Photsim7 canonical photometry service，不再维护 ET-mainsim 本地零点公式。

---

## 2. 曝光派生参数

读出噪声按等效 10 s coadd 缩放：

```text
n_coadd_equiv = exposure_s / 10
read_noise_e_pix = 5.0 * sqrt(n_coadd_equiv)
```

这里 10 s 参考读出噪声采用 `Photsim7-data/config/default.xlsx` 当前显式值：`5.0 e/pix`。

| 输出曝光 | `n_coadd_equiv` | frames/day | frames/7 days | read noise |
| ---: | ---: | ---: | ---: | ---: |
| 30 s | 3 | 2880 | 20160 | 8.660 e/pix |
| 60 s | 6 | 1440 | 10080 | 12.247 e/pix |
| 180 s | 18 | 480 | 3360 | 21.213 e/pix |
| 300 s | 30 | 288 | 2016 | 27.386 e/pix |

缩放规则：

* 目标星期望电子数由 ET mag 转为 `star_flux_e_s` 后再按 `exposure_s` 线性增加；
* sky background 期望电子数按 `exposure_s` 线性增加；
* scattered light 期望电子数按 `exposure_s` 线性增加；
* dark current 期望电子数按 `exposure_s` 线性增加；
* cosmic ray event expectation 按 `exposure_s` 线性增加；
* read noise 标准差按 `sqrt(exposure_s / 10)` 缩放；
* bias、ADC、full-well clipping 不进入本轮 electron-output 链路。

---

## 3. 物理和噪声链路

本轮目标是资源评估，不是完整科学产品验证，但正式 case 应尽量保留最终生产相关成本。

| 模块 | 拟采用口径 |
| --- | --- |
| PSF | Photsim7 `SingleCadenceStampRenderer`，使用 12 deg local-grid PSF |
| PSF bundle | `psf/et/241006/D280mm-focus` |
| PSF field id | `6`，对应 12 deg |
| Subpixels Per Pixel Dim | `7` |
| Jitter-integrated PSF | enabled，按每个曝光时间重新定义积分窗口和拆频边界 |
| Jitter bank 默认规模 | `300 models x 600 frames/model`，正式运行高精度默认值 |
| Motion split | `f > 1/exposure_s` 进入 jitter-integrated PSF；`f <= 1/exposure_s` 表现为 frame-to-frame drift |
| Star flux | `star_flux_mode=random_et_mag`，每颗星稳定采样 ET mag 12.5-14.5 |
| ET mag zero point | `0.91526 * 615.75 * 1_961_225 e/s` |
| PSD drift | enabled，复用 main_rd full-effect motion timeseries 逻辑 |
| DVA drift | enabled，固定 12 deg 几何 |
| Thermal drift | enabled |
| Momentum dump | enabled |
| PSF breathing | enabled |
| Detector response | stamp-local inter/intra-PRV 和 pixel response profile |
| Whole-pixel gain modulation | 本轮不进入实跑链路 |
| Sky background | `26 e/s/pix` |
| Scattered light | `5 e/s/pix` |
| Dark current | `1 e/s/pix` |
| Subtract Nonstellar Mean | `False`，保留 DC 背景、散射光和暗电流均值 |
| Cosmic ray library | `cosmic_ray/dark_test_10um/event_library_10um.npz` |
| Cosmic ray unit | 若 event stamp 为 ADU，注入前按 `gain_e_per_adu=1.4` 转为 electrons |

随机数种子仍采用分层派生：

```text
seed = f(global_seed, exposure_s, frame_id, star_id, effect_type)
```

该口径保证失败重跑时，单个 star/frame/effect 可复现。

---

## 4. 正式评估矩阵

正式评估共 16 个 case：

```text
2 duration/star-count groups x 2 stamp sizes x 4 exposure times = 16 cases
```

### 4.1 短时高星数组

| case id | stamp | 曝光 | 星数 | 时长 | frames/star | 文件数 | 目的 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S1D11E030` | 11x11 | 30 s | 1680 | 1 day | 2880 | 4,838,400 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D11E060` | 11x11 | 60 s | 1680 | 1 day | 1440 | 2,419,200 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D11E180` | 11x11 | 180 s | 1680 | 1 day | 480 | 806,400 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D11E300` | 11x11 | 300 s | 1680 | 1 day | 288 | 483,840 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D15E030` | 15x15 | 30 s | 1680 | 1 day | 2880 | 4,838,400 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D15E060` | 15x15 | 60 s | 1680 | 1 day | 1440 | 2,419,200 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D15E180` | 15x15 | 180 s | 1680 | 1 day | 480 | 806,400 | 吞吐、I/O、文件数、GPU 并行效率 |
| `S1D15E300` | 15x15 | 300 s | 1680 | 1 day | 288 | 483,840 | 吞吐、I/O、文件数、GPU 并行效率 |

### 4.2 长时低星数组

| case id | stamp | 曝光 | 星数 | 时长 | frames/star | 文件数 | 目的 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `L7D11E030` | 11x11 | 30 s | 240 | 7 days | 20160 | 4,838,400 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D11E060` | 11x11 | 60 s | 240 | 7 days | 10080 | 2,419,200 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D11E180` | 11x11 | 180 s | 240 | 7 days | 3360 | 806,400 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D11E300` | 11x11 | 300 s | 240 | 7 days | 2016 | 483,840 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D15E030` | 15x15 | 30 s | 240 | 7 days | 20160 | 4,838,400 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D15E060` | 15x15 | 60 s | 240 | 7 days | 10080 | 2,419,200 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D15E180` | 15x15 | 180 s | 240 | 7 days | 3360 | 806,400 | 长时间动态项、CR、drift、manifest、续跑稳定性 |
| `L7D15E300` | 15x15 | 300 s | 240 | 7 days | 2016 | 483,840 | 长时间动态项、CR、drift、manifest、续跑稳定性 |

推荐执行顺序：

1. 先跑 `S1D11E300`，确认最小文件量完整链路；
2. 再跑 `S1D15E030`，确认最大 I/O 和最大像素量；
3. 再跑短时高星数组剩余 case；
4. 最后跑长时低星数组，用于验证长时间动态项、CR、manifest 和续跑。

---

## 5. 存储和文件数估算

`.npy` 文件包含 128 bytes header。实际单文件大小为：

| stamp | 像素数 | payload bytes | `.npy` bytes |
| ---: | ---: | ---: | ---: |
| 11x11 | 121 | 484 | 612 |
| 15x15 | 225 | 900 | 1028 |

同一 stamp 和曝光下，两组规模的文件数相同，因此存储表对短时高星数组和长时低星数组都适用。

| stamp | 曝光 | 文件数 | payload GiB | `.npy` GiB |
| ---: | ---: | ---: | ---: | ---: |
| 11x11 | 30 s | 4,838,400 | 2.181 | 2.758 |
| 11x11 | 60 s | 2,419,200 | 1.090 | 1.379 |
| 11x11 | 180 s | 806,400 | 0.363 | 0.460 |
| 11x11 | 300 s | 483,840 | 0.218 | 0.276 |
| 15x15 | 30 s | 4,838,400 | 4.056 | 4.632 |
| 15x15 | 60 s | 2,419,200 | 2.028 | 2.316 |
| 15x15 | 180 s | 806,400 | 0.676 | 0.772 |
| 15x15 | 300 s | 483,840 | 0.406 | 0.463 |

单个规模组 8 个 case 合计：

| 指标 | 数值 |
| --- | ---: |
| 文件数 | 17,095,680 |
| payload | 11.018 GiB |
| `.npy` 实际数组文件 | 13.056 GiB |

两个规模组 16 个 case 合计：

| 指标 | 数值 |
| --- | ---: |
| 文件数 | 34,191,360 |
| payload | 22.035 GiB |
| `.npy` 实际数组文件 | 26.111 GiB |

以上不包含 filesystem metadata、目录项、manifest、summary、日志和 Slurm 输出。容量本身不是主要风险；主要风险是大量小文件写盘、manifest 写入方式、进程间返回记录和续跑一致性。

---

## 6. I/O 和 manifest 设计要求

当前实现提供两个明确模式：

| 模式 | 用途 | item index |
| --- | --- | --- |
| `--output-format npy` | smoke/debug/legacy comparison | 每个 worker 的 `manifest.workerNN.csv` |
| `--output-format hdf5` | production | shard 内 `star_ids`、`frame_ids`、`seeds`、`status` datasets |

HDF5 模式只允许 `write_mode=all`，一个 worker/case 只打开一个 writer。父进程只写 `photsim7.image_shard_index.v1` 相对路径索引，不再产生 per-stamp CSV。这样最大 case 的文件数从 O(stars x frames) 降为 O(workers)。

NPY compatibility mode 继续使用 worker 分片 manifest：

正式执行前应改为以下之一：

| 方案 | 描述 | 推荐度 |
| --- | --- | --- |
| worker 分片 manifest | 每个 worker 写自己的 `manifest.workerNN.csv`，主进程只写 `manifest_index.json` 和 summary | 推荐 |
| day/frame 分片 manifest | 按 day 或 frame block 写多个 manifest，适合长时低星数组续跑 | 可选 |
| 主进程 streaming writer | worker 通过 queue 发送记录，主进程边收边写 | 可选，复杂度更高 |

NPY manifest 保留以下字段：

| 字段 | 说明 |
| --- | --- |
| `case_id` | 16 个正式 case id 之一 |
| `scale_group` | `short_high_star` 或 `long_low_star` |
| `stamp_size` | 11 或 15 |
| `exposure_time_s` | 30 / 60 / 180 / 300 |
| `n_coadd_equiv` | 3 / 6 / 18 / 30 |
| `star_id` | 目标星编号 |
| `frame_id` | 输出帧编号 |
| `time_s` | `frame_id * exposure_time_s` |
| `seed` | 分层随机种子 |
| `dtype` | `float32` |
| `unit` | `electrons` |
| `file_path` | stamp npy 路径 |
| `file_size_bytes` | 实际写出字节数 |
| `status` | `completed` / `skipped` / `skipped_existing` / `failed` |

续跑要求：

* 单个 `.npy` 采用临时文件写入后 atomic rename；
* 已存在且 shape/dtype 正确的 stamp 可跳过，manifest 标记为 `skipped_existing`；
* 每个 worker 写独立 manifest shard，主进程只写 `manifest_index.json`；
* HDF5 live file 使用 `.partial.h5`，status 依次为 unwritten/writing/complete/failed；
* HDF5 重跑跳过 complete item，重试 writing/failed/unwritten item，全部完成后 atomic rename 为 `.h5`；
* 已完成 HDF5 shard 必须经过 run/case/ids/shape/dtype/unit/domain/provenance 校验后才能整 shard 跳过；
* summary 中记录 `expected_files`、`n_written`、`n_skipped`、`n_failed`、`files_per_s`；
* 失败重跑不能改变同一 star/frame/effect 的随机结果。

---

## 7. 输出目录建议

```text
stamp_long/
  <case_id>/
    environment.json
    case_config.json
    summary.json
    manifest_index.json
    manifests/
      manifest.worker000.csv
      manifest.worker001.csv
      ...
    exp030/
      star_000000/
        frame_000000.npy
        frame_000001.npy
        ...
    exp060/
    exp180/
    exp300/
```

Production HDF5 layout:

```text
stamp_long/
  <case_id>/
    environment.json
    case_config.json
    summary.json
    shard_index.json
    shards/
      stamps.worker000.h5
      stamps.worker001.h5
      ...
```

说明：

* 每个 case 只包含一个曝光值，因此 `expXXX` 目录用于路径一致性和后续合并分析；
* 若文件系统对单目录文件数敏感，可在 `star_XXXXXX` 之下增加 frame block 子目录；
* `summary.json` 不应嵌入全量 manifest 记录，只保存聚合统计。

---

## 8. H100 执行输入

| 参数 | 默认值 |
| --- | --- |
| Slurm partition | `gpu` |
| GPU 申请 | `--gres=gpu:3` |
| CPU 申请 | `--cpus-per-task=72` |
| 内存申请 | `--mem=256G` |
| wall time | 初始建议 `2-00:00:00`，根据首批 case 实测调整 |
| conda env | `etbase-clu` |
| `ET_ROOT` | `/cluster/home/cxgao/ET` |
| `PHOTSIM7_ROOT` | `/cluster/home/cxgao/ET/Photsim7` |
| `ET_DATA_DIR` | `/cluster/home/cxgao/ET/Photsim7-data` |
| `OUTPUT_ROOT` | `/cluster/home/cxgao/sshfs-share/ET-mainsim/stamp_long` |
| `WORKERS_PER_GPU` | 初始建议 10 |
| `GPUS` | `0,1,2` |
| `DEVICE` | `cuda` |
| `WRITE_MODE` | `all` |
| `OUTPUT_FORMAT` | physics/io 为 `hdf5`；smoke/compute 为 `npy` |
| `STAR_FLUX_MODE` | `random_et_mag` |
| `ET_MAG_MIN` / `ET_MAG_MAX` | `12.5` / `14.5` |
| `JITTER_PSF_MODELS` / `JITTER_FRAMES_PER_MODEL` | `300` / `600` |

正式 benchmark 要求 CUDA 路径可用。若显式设置 `DEVICE=cuda` 且 CUDA 不可用，应直接失败，不允许静默 fallback 到 CPU。

---

## 9. 执行流程

### 9.1 同步和环境检查

正式运行前同步：

```bash
REMOTE=<cluster-host> ./stamp_long/sync_stamp_long_h100.sh
```

在 H100 上做 Slurm 预检：

```bash
ssh 119.78.226.37
cd /cluster/home/cxgao/ET/ET-mainsim
sbatch --test-only stamp_long/submit_stamp_long_h100.sh
```

环境记录必须包含：

* ET-mainsim git commit；
* Photsim7 git commit；
* Photsim7-data 同步时间或 dry-run 差异；
* Python、NumPy、PyTorch、CUDA、driver；
* GPU 型号、数量、显存；
* 输出文件系统路径和可用空间；
* `ET_DATA_DIR`、PSF bundle、PSD、DVA、cosmic ray event library 存在性。

### 9.2 Preflight

正式矩阵前先做极小 sample：

| 项目 | 建议 |
| --- | --- |
| stamp | 11x11 和 15x15 都覆盖 |
| exposure | 30 / 60 / 180 / 300 都覆盖 |
| stars | 1-2 |
| frames | 2-5 |
| write mode | sample |
| GPU | 1 |

Preflight 验收：

* 所有资源路径存在；
* 输出 shape、dtype、unit 正确；
* 无 NaN / inf；
* metadata 记录 `motion_split_hz = 1 / exposure_s`；
* metadata 记录 `jitter_model_index`；
* detector response 和 CR 统计存在；
* sample npy 可读；
* summary 和 manifest 分片可读。

### 9.3 正式矩阵

正式矩阵建议分批执行：

```text
batch 1: S1D11E300
batch 2: S1D15E030
batch 3: S1D short_high_star remaining cases
batch 4: L7D long_low_star cases
```

每个 case 结束后检查：

* `summary.json` 中 `n_stamps == expected_files`；
* `n_written + n_skipped == expected_files`；
* `n_failed == 0`；
* files/s、star-stamp/s、pixel/s；
* worker 负载是否均衡；
* Slurm 日志是否有 CUDA OOM、worker crash 或文件系统错误；
* 长时低星数组中 drift、DVA、thermal、momentum dump、PSF breathing metadata 是否跨 7 天变化。

---

## 10. 代码改造清单

本节是执行前需要落地的代码改造项，不阻塞本文档修订。

| 项目 | 修改方向 |
| --- | --- |
| 曝光组 | 将正式矩阵统一为 `30,60,180,300` |
| case 生成 | 增加 16 个正式 case，或增加 `--matrix-preset stamp_scale_v2` 自动生成 |
| CLI | 增加 `--n-stars`、`--duration-days`、`--stamp-sizes`、`--exposures` 的矩阵生成能力 |
| Slurm wrapper | 增加 `MATRIX_PRESET`、`SCALE_GROUP`、`DURATION_DAYS`、`N_STARS` 等环境覆盖 |
| manifest | NPY 保留 worker 分片；HDF5 使用 shard 内 ids/status/seeds 和相对 JSON index |
| resume | 支持已写文件校验和跳过 |
| summary | 增加 `expected_files`、`n_written`、`n_skipped`、`n_failed`、`files_per_s`、`output_bytes` |
| 测试 | 覆盖 180 s 参数、16-case 矩阵、dry-run 计数、manifest 分片和 resume |

当前运行前准备已落地：16-case 矩阵、NPY worker manifest、HDF5 one-shard-per-worker、partial resume/atomic finalize、canonical ET mag 12.5-14.5 electron-rate conversion、`300 x 600` Jitter-integrated PSF 默认值和 Slurm wrapper 参数透传。

目标命令形态示例：

```bash
MATRIX_PRESET=stamp_scale_v2 \
SCALE_GROUP=short_high_star \
WRITE_MODE=all \
OUTPUT_FORMAT=hdf5 \
GPUS=0,1,2 \
WORKERS_PER_GPU=10 \
sbatch stamp_long/submit_stamp_long_h100.sh

MATRIX_PRESET=stamp_scale_v2 \
SCALE_GROUP=long_low_star \
WRITE_MODE=all \
OUTPUT_FORMAT=hdf5 \
GPUS=0,1,2 \
WORKERS_PER_GPU=10 \
sbatch stamp_long/submit_stamp_long_h100.sh
```

如果按单 case 执行：

```bash
CASE_IDS=S1D11E300 sbatch stamp_long/submit_stamp_long_h100.sh
CASE_IDS=S1D15E030 sbatch stamp_long/submit_stamp_long_h100.sh
CASE_IDS=L7D15E030 sbatch stamp_long/submit_stamp_long_h100.sh
```

---

## 11. 结果分析表

每个 case 汇总为：

| case id | stamp | exposure | stars | days | files | npy GiB | wall time | files/s | star-stamp/s | pixel/s | GPU util | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `S1D11E030` | 11 | 30 | 1680 | 1 | 4,838,400 | 2.758 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D11E060` | 11 | 60 | 1680 | 1 | 2,419,200 | 1.379 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D11E180` | 11 | 180 | 1680 | 1 | 806,400 | 0.460 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D11E300` | 11 | 300 | 1680 | 1 | 483,840 | 0.276 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D15E030` | 15 | 30 | 1680 | 1 | 4,838,400 | 4.632 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D15E060` | 15 | 60 | 1680 | 1 | 2,419,200 | 2.316 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D15E180` | 15 | 180 | 1680 | 1 | 806,400 | 0.772 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `S1D15E300` | 15 | 300 | 1680 | 1 | 483,840 | 0.463 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D11E030` | 11 | 30 | 240 | 7 | 4,838,400 | 2.758 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D11E060` | 11 | 60 | 240 | 7 | 2,419,200 | 1.379 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D11E180` | 11 | 180 | 240 | 7 | 806,400 | 0.460 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D11E300` | 11 | 300 | 240 | 7 | 483,840 | 0.276 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D15E030` | 15 | 30 | 240 | 7 | 4,838,400 | 4.632 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D15E060` | 15 | 60 | 240 | 7 | 2,419,200 | 2.316 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D15E180` | 15 | 180 | 240 | 7 | 806,400 | 0.772 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |
| `L7D15E300` | 15 | 300 | 240 | 7 | 483,840 | 0.463 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 |

最终报告应回答：

1. 11x11 与 15x15 的像素量变化是否线性反映到 wall time；
2. 30/60/180/300 s 的成本是否主要随输出帧数下降；
3. 最大小文件 case 的 files/s 是否稳定；
4. 3 张 H100 的并行效率是否受 Python worker、I/O 或 GPU 计算限制；
5. 长时低星数组是否暴露动态项、CR、manifest 或续跑稳定性问题；
6. 当前 `.npy` 单文件布局是否可用于后续生产，或必须改为 chunk/container 格式。

---

## 12. 风险和处置

| 风险 | 影响 | 处置 |
| --- | --- | --- |
| 大量小 `.npy` 文件 | 文件系统 metadata 压力、files/s 降低 | 正式评估记录 files/s；生产阶段考虑 chunk/container |
| manifest 全量聚合 | 主进程内存和序列化压力 | worker 分片 manifest |
| CUDA fallback | CPU 结果误当 GPU benchmark | `DEVICE=cuda` 下 CUDA 不可用直接失败 |
| JI-PSF bank 构建成本高 | 首帧或每 worker 初始化慢 | summary 分离初始化耗时和逐 stamp 耗时 |
| 长时间动态项未真正变化 | 7 day case 失去验证意义 | case_config/summary 记录动态项采样范围 |
| cosmic ray 单位错误 | 事件能量错误 | 记录 gain，抽样检查 event electrons |
| seed 不可复现 | 续跑结果不一致 | 使用 exposure/frame/star/effect 分层 seed |
| 代码和数据未同步到 H100 | benchmark 不可复现 | 运行前记录 cluster git commit 和 `ET_DATA_DIR` 状态 |

---

## 13. 当前适用范围

本方案只评估固定焦面、target-star-only stamp 的资源需求。它不覆盖：

* quarter/roll 导致的焦面位置变化；
* 背景星和邻星混入；
* full-well、saturation、nonlinearity；
* ADC、bias、DN 输出；
* bad pixels 和 flat-field correction；
* 真实生产星表曝光分配策略。

因此，本轮结果可用于判断 stamp 长时间仿真的工程可行性、吞吐、I/O、GPU 并行效率和续跑稳定性，但不能直接作为完整 ET 多 quarter 科学产品真实性评估。
