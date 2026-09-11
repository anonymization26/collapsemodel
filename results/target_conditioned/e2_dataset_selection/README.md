# E2 真实冻结表示实验

> 状态：真实数据、冻结特征、首轮预注册方法比较和事后 oracle headroom 诊断均已完成；H2 与 H2b 均未通过。

## 已验收产物

PACS 与 Office-Home 均来自官方发布包。发布包收据、逐样本内容哈希、无标签划分、
候选块、重复内容审计和四份冻结特征缓存已经通过完整校验。版本库保存小体积
manifest 与 metadata；原始图像和 `features.npz` 不进入 Git。

| 数据集 | 样本数 | Manifest ID | ResNet-50 | DINOv2 ViT-B/14 |
| --- | ---: | --- | --- | --- |
| PACS | 9,991 | `4113ee4b81f3558c983d0d2a64b90f72d2669663b5644f7fc6aea7a23d3ca79b` | `[9991, 2048]`, 480.57 图/秒 | `[9991, 768]`, 523.29 图/秒 |
| Office-Home | 15,588 | `78c3ddd0f2743b74e377d3f15e46d781648a9480acd536461d984e4bdb6a55fd` | `[15588, 2048]`, 190.88 图/秒 | `[15588, 768]`, 172.29 图/秒 |

所有特征由同一代码版本 `1fec645f5adb924e6fb5c452a71bd00439e18ef7` 在单张 NPU 上抽取。
缓存内容 SHA-256、模型状态、检查点、预处理、代码和 manifest 绑定关系见各
`metadata.json`；汇总索引见 `artifact_index.json`。

## 重复内容处理

PACS 的 9,991 个样本内容均唯一。Office-Home 有 398 个重复内容组、412 个额外副本，
其中 9 组跨域、87 组跨类别。对于每个目标域，源域中与目标域内容完全相同的副本被标为
`excluded_target_duplicate`：Art 目标任务排除 9 个，Real World 目标任务排除 12 个。
同域重复内容共享划分；跨类别重复保留官方标签并在 manifest 中报告。

## 方法比较结果

确认性配置冻结在 `code/configs/target_conditioned_e2/experiment_v1.json`，运行器版本为
`1d3ce3885c6f7da2e165f6c7635adb654e54be78`。PACS 和 Office-Home 的两个编码器、全部目标域、
`K in {1, 3, 5}` 以及 Random 的 20 次重复均已完成，共产生 1,536 行原始指标和 8 个主统计单位。

主预算 `K=3` 下，Target A-opt 相对最强目标无关基线 DPP Subspace 的平均相对 Brier 改善为
0.351%，`delta` 的 95% 置信区间跨过零；相对同信息基线 Second-moment MMD 的平均相对改善为
0.273%，区间虽排除零，但未达到 2% 的实质改善门槛。两项门控均失败。完整结果、运行时间和
结论边界见 [`method_v1/README.md`](method_v1/README.md)，机器可读结论见
[`method_v1/summary/summary.json`](method_v1/summary/summary.json)。

## 事后 Oracle 诊断

在不改变父选择和确认性门控的前提下，后续对每个任务的 12 个候选块在 `K in {1, 3, 5}` 下
穷举了全部 16,384 个组合。主预算 `K=3` 中，真实 oracle 相对最强目标无关基线 DPP Subspace
的平均 Brier headroom 为 0.781%，95% bootstrap 区间为 `[0.162%, 1.617%]`，仍低于 2% 实质
门槛。Target A-opt 相对 oracle 的归一化遗憾为 0.436%，真实 top-10 组合命中率为 62.5%，代理
与真实 Brier 的组合级 Spearman 为 0.826。

该结果表明当前候选池整体可利用 headroom 有限；同时，Target A-opt 只捕获平均 31.6% 的 oracle
改善，且在 PACS 各目标域上存在明显异质性。完整结果和限制见
[`oracle_headroom_v1/README.md`](oracle_headroom_v1/README.md)。这是使用 `target-test` 定义 oracle
的事后探索，不更新 H2/H2b，也不等同于 H3 的 shortlist 召回实验。

## 结论边界

数据准备产物证明 E2 输入可追溯、划分可复算、特征可核验；`method_v1` 进一步给出真实冻结
表示上的确认性负结果；事后穷举进一步表明整体 oracle headroom 未达到 2% 门槛。当前只能声称
无标签目标二阶几何含有一定组合排序信号，且 Target A-opt 在该网格上基本不劣并呈现很小的 Brier
方向性收益，不能声称其相对经典覆盖或简单目标匹配具有实质优势，也不能声称已实现有效 shortlist
预筛选。

旧版单域 DomainNet 特征缺少完整多域任务、统一预处理和逐样本 ID，不计入 E2 证据。
协议与复核命令见 `docs/E2_DATA_PIPELINE.md`。
