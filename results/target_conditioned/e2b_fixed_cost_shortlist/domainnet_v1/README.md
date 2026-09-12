# E2b：DomainNet 固定成本 shortlist 实验

## 结论

E2b 已完整执行，并按冻结协议通过 H3。主方法 Target A-opt 在 `S=227` 时从 455 个等成本组合中
排除 228 个，平均保留 `98.333%` 的独立测试 top-10，验证选中组合的平均测试归一化 Brier 遗憾为
`0.001311%`。

这个通过结果不是 Target A-opt 的相对优势证据。同信息 Second-moment MMD 在主设置上得到相同的
recall 和遗憾，并在全部 12 个编码器-目标域任务中选出同一个最终组合。当前最稳妥的结论是：
**目标条件二阶几何可以在本协议下用于等成本组合预筛选，但尚未证明 Target A-opt 优于更简单的
二阶矩匹配。**

## 冻结设计

- 数据：DomainNet 2019 cleaned，六个域，按类名 SHA-256 固定选择 100 类；
- 样本：每域 1,536 个 target-selection、1,024 个 target-validation、3,072 个 anchor-pool 和
  2,048 个独立 target-test，共 46,080 个样本；
- 候选：CLIP ViT-B/32 域内第一主轴的 low/middle/high 三层，每块 512 个样本，共 18 块；
- 组合：目标域自身三块排除后，从其余 15 块中穷举 `C(15,3)=455` 个组合，每个组合固定使用
  1,536 个训练样本；
- 评价：ResNet-50 和 DINOv2-B/14 冻结特征，固定 Rademacher JL 投影到 128 维；
- 流程：U0 screen、train 内独立 validation、选择冻结后的官方 test 穷举审计；
- 统计：先在同一目标域内平均两个编码器，再以六个目标域为统计单位；20 次随机排序不视为独立样本。

完整预注册协议见 `../../../../docs/E2B_FIXED_COST_SHORTLIST.md`，机器配置见
`../../../../code/configs/target_conditioned_e2b/domainnet_v1.json`。

## 主结果

| 方法 | `S=227` top-10 recall | 验证选择测试遗憾 | Screen Spearman | H3 |
|---|---:|---:|---:|---:|
| Target A-opt | 98.333% | 0.001311% | 0.7443 | PASS |
| Second-moment MMD | 98.333% | 0.001311% | 0.6902 | PASS |
| Bayesian D-opt | 82.500% | 0.002658% | 0.4996 | FAIL |
| Merged effective rank | 81.667% | 0.002909% | 0.4951 | FAIL |
| Random，20 次重复 | 48.958% | 0.003899% | 0.0016 | FAIL |
| Target energy | 46.667% | 0.009347% | -0.0399 | FAIL |
| DPP subspace | 27.500% | 0.009067% | -0.0633 | FAIL |
| Domain balance | 25.000% | 0.012083% | 0.0681 | FAIL |

Target A-opt 的六域 bootstrap 95% recall 区间为 `[96.667%, 100%]`，遗憾区间为
`[0.000228%, 0.003031%]`，组合缩减为 `50.110%`。逐方法、逐编码器和逐目标域原始汇总分别位于
`summary/method_summary.csv`、`summary/per_encoder.csv` 和 `summary/per_unit.csv`。

## Shortlist 尺度

| 方法 | S=10 | S=25 | S=50 | S=100 | S=227 |
|---|---:|---:|---:|---:|---:|
| Target A-opt recall | 24.167% | 45.000% | 63.333% | 80.000% | 98.333% |
| Second-moment MMD recall | 27.500% | 47.500% | 70.833% | 86.667% | 98.333% |

两者 `S=227` shortlist 的平均 Jaccard 为 `0.8307`，但验证阶段在 12/12 个任务中选择了相同组合。
这表明主设置的有效候选区域相近，同时不能用最终选择结果区分两种方法。

## 结果边界

1. H3 是 Target A-opt 的绝对可用性门槛，不是相对方法检验。MMD 同样通过，因此不能写成
   “Target A-opt 优于同信息基线”。
2. 事后审计中，单任务 455 个组合的完整相对 Brier 跨度仅为 `0.0280%` 至 `0.4242%`，远低于
   预注册的 `5%` 遗憾阈值；所有方法都轻易通过遗憾子门槛。H3 的主要有效证据来自 top-10 recall，
   而不是该遗憾阈值。
3. 不同组合的 accuracy 范围仍较大，说明候选并非完全等价；但 Brier oracle 与 accuracy oracle
   不总一致，论文必须把 Brier 明确写成冻结主指标，并把 accuracy 作为支持性指标单独报告。
4. 只有一个独立数据集和六个目标域单位，不能推出跨数据集普适性。E2b 也不能追溯性改变 PACS 和
   Office-Home 上 H2/H2b 的失败结论。

## 完整性与产物

- Manifest ID：`1ba25059df70be4d9d22ec1c3c25bdd03e6a83795fbb44a45b3a8e0bb6b20839`；
- Candidate set ID：`545fe86c8c9e830448f14f0ce0a300a33389f3eb9787e118572e7e78da87e9a8`；
- Summary ID：`2d473cb9eb977e859b2f08e8193d03bcac1b8924aac1403bc3ae2e6ea3a12fe9`；
- 两个 screen、validation 和 test-audit 流程均通过配置哈希、上游产物哈希、行数、组合覆盖和访问权限校验；
- 验证阶段每个方法或随机重复均物理评估自己的 227 个组合，未共享跨方法结果；
- 官方 test 只在 screen 和 validation 选择文件冻结后读取，test-audit 对每个编码器和目标域穷举 455 个组合。

仓库保留逐样本 manifest、候选、screen 排名、validation 指标、test-audit 全组合指标和自动汇总。
约 620 MB 的三份特征矩阵未重复提交；`features/*/metadata.json` 保存其形状、精确 SHA-256、模型权重
SHA-256、预处理、代码版本和运行环境，可用于核对外部冻结缓存。
