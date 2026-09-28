# 受限静态 equivalent coadd 工作流

```bash
et-mainsim run et-equivalent-coadd --config equivalent.toml --dry-run
et-mainsim run et-equivalent-coadd --config equivalent.toml
et-mainsim run et-equivalent-coadd --config equivalent.toml --verify-only
```

该入口消费 Photsim7 的 `EquivalentCoaddRequest`，支持 30/60/120/300 s 的
stamp 或全幅正式产品。科学配置、实际输入身份、组内恒定 variability、采样前 clipping
预算及产品 schema 均由 Photsim7 验证。ET-mainsim 负责文件输入、组选择、CPU 线程、
运行锁、应用 manifest 和续跑。普通 raw/coadd 工作流保持原有语义。

包依赖最低为 `photsim7[gpu]>=0.5.11,<0.6`；上游必须包含 `run_equivalent_coadd_product` 和 `read_equivalent_coadd_product`；
版本号本身不能证明接口存在。CI 固定上游的实际提交，安装步骤见 README。
科学与 API 边界见 [Photsim7 equivalent API](https://github.com/TutuchanXD/Photsim7/blob/main/docs/api/equivalent_coadd.md)。

## 配置与输入

```toml
schema_id = "et_mainsim.equivalent_coadd_run.v1"
run_id = "static-detector-120s"
request_path = "request.json"
catalog_path = "catalog.json"
variability_path = "variability.json"
data_root = "/path/to/Photsim7-data"
output_root = "results"
cpu_threads = 2
block_shape = [512, 512]
resume = true
# 可选：省略时执行请求声明范围内的全部组。
# coadd_indices = [0, 1]
```

JSON 配置同样受支持；所有相对路径以配置文件所在目录为基准。未知字段会被拒绝。
`run_id` 为单个非空路径分量。组索引为相对于原始请求的局部 coadd index；不接受
空列表、重复索引、布尔值或范围之外的索引。执行顺序可以改变，绝对 raw 身份保持不变。
选择一部分组时仍使用原请求的完整组范围和 clipping 风险分配，不能缩小分母来扩大预算。

三个科学输入文件分别为：

- `request.json`：`EquivalentCoaddRequest.to_json_dict()` 的完整输出，包含内容 SHA-256。
- `catalog.json`：`{"data": {...}, "metadata": {...}, "raw_source_arrays": null}`。
  `data` 为 prepared catalog 的数值／字符串列，数组以 JSON 列表保存；几何等必要声明
  原样放在 `metadata`。这里的目录必须与构造请求时的实际目录一致。
- `variability.json`：`SourceVariabilityDelivery.to_json_dict()` 的输出，包含完整 raw 轴、
  实际源 ID、10 s sampling/averaging window、epoch 和绝对起点。

请求通过上游 `EquivalentCoaddRequest.from_source` 构造。指定 target 和 stamp shape
得到 stamp 请求；两者同时不设置得到全幅请求。ET-mainsim 不修改请求中的 seed、
cadence、配置、科学资产或精度声明。文件身份冲突会在执行前失败。

`--dry-run` 检查请求文件和组选择，不加载 PSF、不创建输出目录。实际运行还检查
当前科学资产。CPU 线程数是显式执行策略，作用于 Torch、Numba 和 native pools，
结束或失败后恢复调用方设置；不支持隐含的 CUDA、Ray 或普通 raw-frame 覆盖参数。

## 产物、续跑与独立检查

运行目录为 `output_root/run_id`，包含 `run_manifest.json`、保存的请求／应用配置，
以及 `products/group_00000000/` 等逐组产品。每组使用上游正式
`photsim7.equivalent_coadd_product.v1`，由上游在完整读回后原子发布。

应用 manifest 记录实际 DN/电子数/分量期望/mask 的数组身份、raw indices 和时间窗、
clipping 证书及每个 product manifest 的 SHA-256。DN 为 `uint32`，不能改名或强制转换后
冒充现有 `uint64` raw-DN sum stamp delivery。所有输入的精度资格保持 `unqualified`：
固定矩阵的 10 ppm／1% 方法资格不自动扩展到本次源场。

`resume=true` 保留完整请求和应用身份，逐组重新验证实际文件后复用。后续组失败时，
应用状态记录为 failed；同一配置再次执行可复用已验证的完整前缀。输入／实现／数值环境、
保存的请求／配置或完成记录发生冲突时失败，不覆盖旧运行。数值身份在第一组执行前绑定
Python、CPU/平台、科学包与 native library 版本、Torch build 和线程策略；没有产物的
失败任务也不能跨环境续跑。初始化 sidecar 写入中断时，仅在 manifest 仍为 planned、
无 attempt/产物且身份完全匹配时补齐缺失文件，已有冲突文件不覆盖。没有自动删除／覆盖模式。

正常执行直接消费上游发布前完整读回的结果。`--verify-only` 是独立消费入口：不渲染，
核对应用完成记录并调用上游 reader，重新检查真实文件和数组、参数、原始时间窗、
RNG roots、风险证书和保存电子数的读出重放。即使有人更新了产品内的自校验 hash，
产品 manifest 仍须匹配应用层保存的 hash。hash 用于完整性检查，不是数字签名。

内存与耗时随产品和执行布局而变；正式产品包含诊断参数及完整验证，不能把纯渲染时间
当作总交付时间。运行与独立检查结束前均重新计算科学输入和资产身份；
发现中途变化时不能宣告完成或验证成功。本工作流不执行新的统计资格判定，也不触发普通 stamp 光变分析。
