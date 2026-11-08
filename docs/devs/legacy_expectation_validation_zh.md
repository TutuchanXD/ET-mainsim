# M6 通用工作流验证记录

2026-11-09，ET-mainsim #59 / PR #60。科学运行代码为 `94a2dac`，上游
Photsim7 为已合并的 `429b7687895ea97eca2f23d5c09b9f823bed3bee`。
后续交付提交只增加验证驱动、记录和打包清单，不修改该次运行的应用科学路径。

本地完整测试：**1,144 passed，2 skipped**，165.54 s。跳过项为仓库既定的
两个未随源码分发的 notebook 检查。新增 37 项回归覆盖真实公共 API、三类输入、
六镜、原始期望保存、复测、缺失/损坏/冲突拒绝、单组中断、进程分片和表格重排。
验证驱动加入分发清单后，打包与 CI 合同相关测试另有 **107 passed**。

## 实际仿真

使用真实 ET PSF `psf/et/241006/D280mm-focus`、像素相位资产、像素/子像素响应、
ET PSD、JI、温度驱动的热漂移和 PSF 呼吸。原始曝光为 10 s，共 24 帧，
每 6 raw 合成一个 60 s coadd，共 4 组、6 台望远镜。full-frame 为 65×65，
stamp 为 21×21；catalog 含目标和邻星，table 含两个独立目标。
使用参考场几何，不声称是物理天空 DVA 案例；momentum dump 已配置，但该短时间
窗口不保证发生 dump。完整资产、参数、消费阶段、RNG 与时间真值在运行产物中。

| 后端 / 输入 | 执行、校验和已完成续跑总耗时 | 像素与轻量反序分片比较 |
| --- | ---: | ---: |
| CPU / catalog stamp | 2.97 s | 340 个数组逐位一致 |
| CPU / full-frame | 2.05 s | 340 个数组逐位一致 |
| CPU / table stamp | 7.05 s | 680 个数组逐位一致 |
| CUDA / catalog stamp | 2.99 s | 340 个数组逐位一致 |
| CUDA / full-frame | 2.57 s | 340 个数组逐位一致 |
| CUDA / table stamp | 7.69 s | 680 个数组逐位一致 |

总计 **2,720 个测量数组逐位一致**。各后端的 full-frame 与 stamp 公共像素
另各比较 288 个 coadd/伴随数组，合计 **576 个数组逐位一致**。
这里比较的是同一后端、同一物理随机实现下的编排不变性，不给出新的科学残差阈值。

CUDA 设备为本机 RTX 5090，未提交 H100 作业。进程累计峰值 RSS 为
2,221,544 KiB，CUDA 累计峰值分配为 64,273,920 bytes。这些是小案例的进程
高水位，不能外推为完整探测器或大规模生产的单组内存。测试期间也有其他验证进程，
耗时用于复现记录，不作为吞吐基准。

CPU 与 CUDA 使用各自 Torch Poisson 后端，不要求含噪图像逐位一致；跨设备差异
按字段保留为诊断。续跑身份绑定设备类型、数值库与代码，不能把 CPU 和 CUDA
当成同一个可任意切换的缓存执行环境。应用不会据此宣称六镜噪声缩小 √6。

## 重放与证据

```sh
conda run --no-capture-output -n etbase python scripts/validate_legacy_expectation_workflow.py \
  --data-root /home/cxgao/ET/Photsim7-data \
  --output /home/cxgao/ET/outputs/m6-replay-new
```

输出目录必须是新目录。脚本生成三类配置，执行像素/轻量产品、分片/反序/续跑，
验证完整回读并生成字段级比较。固定科学资产保留在外部；每次执行自动记录其哈希。

原始记录位于 `/home/cxgao/ET/outputs/m6-legacy-expectation-20261109/`，
归档为相邻的 `m6-legacy-expectation-20261109.tar.gz`：

- 大小：48,062,383 bytes。
- SHA256：`796ac5fc97decb10041cd27e77c71d00dc8487c2245ce90b5fedfb5fe2b6d061`。
- `SHA256SUMS.json` 覆盖 1,034 个文件，已逐文件核对。
- 提交中的 [机器可读摘要](evidence/legacy_expectation_20261109.json) 保留每组产物
  清单、运行身份摘要、资源、比较结果与跨设备诊断。

M6 完成的是可回读、可复现的工程接入。M7 科学负责人验收以及 #43 生产 campaign
保持各自独立的范围。
