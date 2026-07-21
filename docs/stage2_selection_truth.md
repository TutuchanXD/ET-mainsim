# Stage 2 geometry、PSF 与 JI selection truth

## 范围与状态

Stage 2 冻结并验证的是“同一物理 realization 如何选择 geometry、PSF node 和
jitter-integrated（JI）model”的离散身份合同。它保证 full-frame 与 stamp 在同一
logical cadence 下使用同一选择依据，并让资产或声明漂移时 fail closed。

这不是完整科学链路验收。按顺序合并的 Stage 2 交付栈实现 deterministic pixel
golden、standalone selection sidecar 和 shared-exposure crop/artifact 工程合同；历史
selection-evidence v1 仍保留其生成时的 pre-delivery 边界，不被改写成当前 full-stage
声明。尚未关闭的门包括跨 backend/ensemble tolerance、AT-SD25，以及最终 RMS/CDPP
产品比较。因此不得把“Stage 2 selection identity 工程实现已交付”简写成
“full-frame、stamp 与 legacy 已完整科学对齐”。

## 两种 geometry authority

| 输入模式 | geometry authority | PSF 策略 | 可以声称什么 |
| --- | --- | --- | --- |
| 有 `ra_deg`、`dec_deg` | `physical_et_focalplane`：ICRS、epoch 2000.0、focal-plane registry resolved path 与 content hash | 按源的径向视场角选择最近 node；平局取较小 field ID | registry 身份验证通过后，可声明 physical sky-to-detector geometry |
| 无坐标、有 `psf_id` | version-2 `reference_field_nonphysical`：reference field angle、polar angle、4.83 arcsec/pix、x/y axis sign | 只接受并验证显式 field ID | 只能声明 deterministic reference approximation，不能声明 physical sky mapping |

ET stamp table mode 要求每一行严格选择其中一种模式。其
`et_mainsim.stamp_source_input_truth.v2` 记录：源表/光变表身份、geometry 模式、
registry 身份（若适用）、detector/field 解、PSF bundle 身份、选择策略、node 与角度
残差。该 payload 会进入 raw/coadd schema、`source_variability_truth.ecsv` metadata
和 `target_artifacts.json`。

## PSF bundle 与 node truth

ET production 接受的 bundle 是：

```text
psf/et/241006/D280mm-focus
SHA-256 9b01d7d890c1a92829ea7ab52fab349a019417dcff7754888620bd6c99c7f1ec
```

ET-mainsim 先从输入侧记录实际 bundle identity，再把独立的 accepted hash 放入
`SimulationSpec`。Photsim7 在反序列化前比较实际字节 SHA-256；随后才用已验证的
geometry truth 生成 node-selection truth。仅由调用者给自定义 bundle 配一个相同的
“expected hash”，最多证明 byte integrity，不会自动获得 owner-accepted science
conformance claim。

PSF selection truth 绑定 geometry-truth content hash、bundle identity、有序 node
表、每个 source 的 field-angle authority、选择结果与角差。相同 source/geometry/
bundle 在 full-frame 与 stamp 中必须产生相同的 canonical truth hash。

## Native ET jitter bank

| 字段 | 冻结值 |
| --- | --- |
| logical ID | `legacy_science_v1_et_attitude_xyz_100x3x300_v1` |
| 路径 | `jitter/et/native/legacy_science_v1_et_attitude_xyz_100x3x300_v1.npy` |
| shape | `[100, 3, 300]`，轴依次为 model、spacecraft XYZ、sample |
| array SHA-256 | `696a986c82902ad18f136f284a30b2ce506998d3e900ea2601a3e6af001cc4d0` |
| manifest SHA-256 | `267453c0cc5355f7edfaff76164c56ea38052a866bb967bb124c920394bf7274` |
| exposure / sample spacing | 10 s / 1/30 s |

loader 在 `numpy.load` 前校验 array hash，并严格校验 manifest schema、源 PSD、
生成参数和 manifest hash。native bank 是 spacecraft XYZ；每个 source/scope 的 field
XY 是后续 geometry projection 的派生量，不能把派生 XY bank 伪装成 native 资产。

## 每 cadence 的 model selection

Photsim7 `simulation_context.v2` 明确携带：

- `run_seed`；
- `science_realization_id`；
- `spacecraft_id`；
- `absolute_raw_frame_start_index`。

每 cadence 使用 absolute raw-frame index，并在
`effects.jitter_model_selection` SeedTree stream 上选择 `[0, 100)` 的 model。
selection truth 同时保存 bank 的 actual/expected array 与 manifest hashes、所选 model、
物理 scope 和 RNG trace。pipeline 在合并 trace 时会用当前 active SeedTree 重新派生并
核对该选择，不能注入来自另一个 run seed 或 spacecraft 的 selector。

以下字段被明确排除在科学选择 scope 外：worker/rank/GPU、shard/batch、执行顺序、
输出目录、run label、target request、stamp origin 和 stamp shape。这样，同一 logical
cadence 在 full-frame 与 stamp 中的 geometry hash、PSF selection hash、JI model 和
RNG trace 可以逐项一致。

预注册的 noise-off single-source golden fixture 是一个显式例外：它以独立
`preregistered_golden_fixture` authority 固定 model 37，只用于后续 crop/golden
验收；正常 science cadence 仍使用 SeedTree stochastic selector，不能为命中 37
而修改 RNG 算法。

## Fail-closed 条件

以下任一情况都不能降级成“继续运行但仍声称 conformant”：

- coordinate row 的 epoch 不是 2000.0、registry identity 缺失或 content hash 漂移；
- 无坐标且没有显式 PSF ID，或 reference declaration 参数不完整；
- PSF bundle 的实际 hash 与 `SimulationSpec` accepted hash 不同；
- JI array/manifest 路径、hash、shape、logical ID 或 loader contract 不一致；
- selector 的 SeedTree、science realization、spacecraft 或 absolute raw-frame origin
  与 active `simulation_context.v2` 不一致。

自定义但完整可验证的资产可以输出 `science_conformance_claim=false` 的 provenance；
它们不能冒充 owner 接受的 ET reference calibration。

## 当前交付边界

已具备：

- table stamp 的 `stamp_source_input_truth.v2` 与 per-product 引用；
- geometry 与 PSF selection 的 canonical runtime truth/provenance；
- frozen native bank 的 owner hash 与 strict loader；
- per-cadence JI selection truth、active-SeedTree 验证，以及 full/stamp 同 cadence
  selection identity 测试；
- JI truth 启用/可用时独立持久化并严格 readback 的 geometry、PSF 和 per-cadence
  selection sidecars；no-JI 路径使用严格的 unavailable declaration；
- deterministic pixel golden 与 shared-exposure parent crop/shard/marker 合同。

仍需后续 Stage 2/3/4 工作：

- 补齐尚未纳入当前 golden 的 absolute-detector RNG 与 pixel-first component 覆盖；
- 在查看结果前预注册 statistical、跨 backend 与最终科学指标容差；
- 完成 AT-SD25 与最终 RMS/CDPP 科学判据，不以工程合同替代科学验收。
