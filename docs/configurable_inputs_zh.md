# 工作簿、RNG、缓存与续跑

本阶段要求 Photsim7 **0.5.1** 的加载及内容身份接口。ET-mainsim 负责 ET 预设、
执行调度与运行记录；科学参数解析和 RNG 规则由 Photsim7 提供。

## 输入优先级

```bash
et-mainsim run et-full-frame --preset production \
  --spec /home/cxgao/ET/Photsim7-data/default.xlsx \
  --config run.toml --seed 17 --run-id experiment-17
```

工作簿显式行覆盖 ET 科学预设；canonical JSON 是完整科学配置。
`--config` 是执行 TOML，可以只填写需要覆盖的部分；显式 CLI 参数最后生效。
科学 JSON 中的设备设置覆盖预设设备，显式 TOML/CLI 设备设置优先于科学 JSON。
执行 backend 与设备必须兼容，例如 CUDA 使用 `local-subprocess`。

`--seed` 修改总种子，保留分项种子。工作簿的 `RNG Seed <stream>` 或 JSON 的
`rng.stream_seeds` 控制分项，`inherit` 表示继承总种子。常用预设不再额外填写旧
`cosmic_rays.seed`，因此 cosmic 默认也受总种子控制；用户已有的显式分项值仍保留。

未提供 `--spec` 时继续使用选择的内置预设，不会隐式读取机器上的可变工作簿。
smoke 是小规模安装检查；真实 default.xlsx 的五类效应均保持显式开启。
生产预设使用当前 PSD 合成 jitter，PSF/bank 不再绑定历史摘要。
显式选择的 `legacy-sim-full-effects-*` 仍表示已有的历史科学参考契约。

## 时间、天区和科学结果

没有 `--frames` 时保持用户的总观测定义和显式时间表。`--frame-indices` 只选择
执行帧；它不会缩短总观测时间。`--frames` 可改变规则观测长度；与显式时间表冲突
时必须修改科学配置或改用帧选择，不能静默抹掉时间表。

DVA 需要真实的 ET 天区及 focal-plane 几何。仅有星等、PSF ID 的参考场景不能
表达其物理方向；使用天球坐标表及 focal-plane registry，或明确配置适合该场景的
效应策略。启用 DVA 的验收使用具有真实 ET 几何的目标。

## 运行记录与缓存

运行目录写入 `effective_spec.json` 以及 `run_manifest.json`。记录包含最终有效
RNG、实际资产内容、星表科学数组身份、代码内容、运行库、设备与精度。
用户不用维护摘要。参数或资产内容改变后，可以建立新的 run；新实验遇到不匹配
的派生星表缓存会重建，并在成功后原子替换，失败时保留旧缓存。

同一 run 拒绝混入不同科学输入或数值环境；旧 manifest 缺少输入证据时需要新
run ID。显式 `--overwrite` 也不会绕过科学输入检查。离线源只能使用已有、请求
匹配的缓存，其生成来源与本次实际数组分别记录。

## 调度与恢复

worker 数量、GPU 编号、同设备的执行 backend、预览数量以及纯批处理大小不属于
科学身份。设备类型、数值精度、科学参数及产物规则仍参与一致性检查。
共同曝光分片使用固定全局帧组，worker 数变化不会重新定义已完成分片。
分片大小属于存储布局，修改它需要新 run。

同一 run 同时只允许一个协调进程。操作系统文件锁在进程退出时释放；持锁的新
协调进程可以将遗留的 `running` attempt 标为 `interrupted` 并恢复。worker 启动
时复核输入，完成判定继续校验原始科学产物及 selection sidecar。

可复现保证针对科学数组；日志时间、进程编号及容器字节不要求相同。CPU 与 CUDA、
不同 GPU 架构或运行库版本之间不声明逐位一致。

## 验收入口

常规回归运行 `python -m pytest`。外部资产验收独立运行：

```bash
ET_DATA_DIR=/path/to/Photsim7-data \
ET_FOCALPLANE_ROOT=/path/to/et_focalplane \
ET_S3_WORKBOOK=/path/to/default.xlsx ET_S3_DEVICE=cpu \
python -m pytest validation/test_s3_real_assets.py -q
```

将 `ET_S3_DEVICE` 改为 `cuda` 可验证 CUDA 子进程入口。该验收只缩小场景和观测
规模，保持五类效应开启，检查全幅、catalog/table stamp 的重复运行、续跑、
输出目录独立性与分项种子控制。正式 campaign #43 不由本阶段自动启动。
