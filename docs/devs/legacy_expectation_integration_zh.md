# M6：legacy expectation 通用工作流

对应 ET-mainsim #59；上游基线为 Photsim7 PR #200 合并提交
`429b7687895ea97eca2f23d5c09b9f823bed3bee`。

## 实施范围

新增显式的 `et-legacy-expectation` 应用入口，支持 full-frame、catalog-stamp、
table-stamp。现有 final-DN 和 legacy 入口继续按原合同运行。应用只编排输入、
分片、持久化和回读；每组图像由 Photsim7 legacy expectation 公共入口生成，
孔径测光与六镜合成调用公共 reference photometry。

配置明确给出 SimulationSpec、目标及固定孔径、输出种类、raw 保存策略与科学
realization。catalog/full-frame 属于同一曝光；table 中每行是独立 Monte Carlo
场景，realization 从稳定 source ID 派生，不使用任务 rank、顺序或输出目录。
不同 coadd 长度依照上游合同是独立 realization，不声明严格嵌套。

以一个完整 coadd（所有配置望远镜和目标）为发布单元。先写临时目录，完成校验
后原子发布；续跑必须验证已完成组及科学输入身份。分片只改变执行的组索引，
不改变完整观察的时间轴、动态效应、RNG 身份或输入光变。缺组保留 partial 状态，
不能当作完整六镜产品发布。像素模式保留 coadd、伴随数组和可复测产品；轻量模式
保留测量、质量、输入与执行真值，明确不支持更换孔径后的复测。

## 验证与交付

- CPU 小例覆盖三类输入、电子单位、raw/coadd 时间、六镜单镜与合成产品。
- 顺序变化、分片和中断续跑逐数组比较；输入漂移、缺失、损坏与冲突必须拒绝。
- 支持的 CUDA 小例和真实资产六镜多组案例，记录实际资源、耗时及证据。
- 运行受影响测试及仓库 CI，更新安装依赖身份和测试清单，最终提交 Ready PR。

M6 的完成表示通用工作流可重复执行和验收，不替代 M7 的科学负责人认可，
也不启动 #43 campaign 或迁移既有产物。
