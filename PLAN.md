# 两阶段无标签数据池压缩：完整实验计划

## 1. 实验目标

本计划不再以“Collapse 是否优于随机”为唯一问题，而是回答五个相互独立的问题：

1. 冻结表示的谱摘要能否在不访问标签的情况下保留高价值候选池？
2. Collapse 中的 alignment 信息是否稳定优于单池有效秩等简单摘要？
3. 紧凑摘要相对经典全特征子集选择方法损失多少选择质量？
4. 第一阶段谱目标能否降低第二阶段任务验证成本，同时保持最终下游效用？
5. 方法在不同数据域、编码器、选择预算和通信预算下是否稳定？

主要贡献的验收对象是“两阶段预筛选框架”，而不是预设 Collapse 必须胜出。所有负面结果均保留。

## 2. 当前计算基准

服务器为 8 张 Ascend 910B4，每张约 32 GB HBM。2026-09-02 复查时 NPU 0–1 被其他任务占用，
NPU 2–7 空闲，因此当前特征提取按 **4 张 NPU 并行**执行；后续排期仍按实时空闲卡数计算，
不能默认长期独占 4 张卡。

已完成实验的实测吞吐：

| 项目 | 实际规模 | 实际时间 | 折算吞吐 |
|---|---:|---:|---:|
| Stage-1 谱筛选 | 24 配置、216 方法结果 | 约 6 分钟 | 约 15 秒/配置，CPU |
| Stage-2 adapter | 648 次训练、5,184 目标结果 | 19 分 34 秒 | 约 1.81 秒/训练 |
| Stage-2 NPU 消耗 | 648 次训练 | 0.326 NPU-hours | 约 1,988 次训练/NPU-hour |

上述 adapter 基准只适用于预计算 ResNet-50 特征上的 256 维瓶颈网络。ViT/CLIP/DINO 特征维度、
LoRA 和端到端训练必须单独 benchmark，不能直接套用该吞吐。

## 3. 公平性原则

### 3.1 数据公平

- 使用官方 train/validation/test 划分，不再从同一 1,000 样本缓存同时构造训练和测试。
- 源数据池和目标数据集在实验前固定，不根据结果调整。
- 明确标记共享原始图像的数据集，例如 CIFAR-100 与 CIFAR-100-Coarse，不能当作独立域。
- 将 MNIST/USPS/Semeion、OrganMNIST 系列等相关数据集归入同一 source family。
- 人工重叠实验与自然多源实验分开报告，不用 clone 压力测试替代自然场景证据。
- 每个自然数据集合至少包含 20 个候选池；目标数据与候选池保持数据集级隔离。

### 3.2 模型公平

至少使用四类视觉编码器：

| 编码器 | 作用 |
|---|---|
| ResNet-50 | 监督 CNN 基线，与已有实验衔接 |
| ViT-B/16 | 监督 Transformer |
| DINOv2 ViT-S/B | 自监督表示 |
| CLIP ViT-B/32 | 视觉语言表示 |

可选增加 SBERT 文本实验，但文本与视觉分开校准和报告。所有方法在同一编码器、同一特征归一化、
同一候选池上比较。校准参数不能跨 test pool 拟合。

### 3.3 信息和通信预算公平

设置两条独立赛道：

1. **全信息赛道**：所有方法均可访问完整冻结特征，评价选择质量上限。
2. **摘要赛道**：方法必须满足相同传输字节预算，评价准确率—通信 Pareto 前沿。

必须记录每个方法的输入：标量数、向量维度、top-k 子空间大小、总字节数、CPU/NPU 时间。
不能将一个标量的 `r_sum`、top-20 子空间的 Collapse 和使用完整特征的方法直接称为等成本比较。

### 3.4 训练预算公平

每个下游实验同时报告三种协议：

- 固定总训练样本数与总步数。
- 固定每个被选数据池的样本数与步数。
- 固定实际 NPU-hours。

预算 3 和预算 5 不得在“总步数同为 600、每池步数不同”的条件下直接比较优劣。

## 4. 数据设计

### 4.1 受控重叠集合

- 22 个候选源数据集，先按 family 去重后报告有效独立域数量。
- 重叠率：0%、10%、25%、50%、75%、100%。
- 变体类型：精确样本副本、同分布不重叠采样、数据增强变体、标签无关噪声变体。
- 候选池构造种子：至少 10 个，目标为 20 个。
- 选择预算：3、5、8、10 个池。
- 每个重叠率同时包含自然池和受控变体，避免所有高秩池都被人工复制。

### 4.2 自然多源集合

建立至少四组候选集合：

1. 通用视觉：自然图像、纹理、物体、场景和卫星数据。
2. 数字与字符：MNIST、USPS、Semeion、SVHN 等。
3. 医学图像：MedMNIST 各子域，按器官和成像类型标记 family。
4. 跨域混合集合：从上述各域抽取相同数量的数据池。

每组使用独立目标数据集，执行 leave-one-domain-out。自然集合是主结果，受控重叠只作为机制实验。

### 4.3 样本量

- Stage-1 特征摘要：每池 250、500、1,000 三档样本量。
- Stage-2 adapter：固定总样本与固定每池样本两套协议。
- 类别数较多的目标必须使用官方测试集；每类测试样本不足 10 时不作单独强结论。

## 5. 算法矩阵

### 5.1 最低成本与消融基线

- Random-100：每个配置至少 100 个随机子集，报告分布和分位数。
- Pool size / energy ranking。
- 单池 effective-rank ranking，即当前 `r_sum` 实现，重命名避免误解。
- `max(r_A,r_B)` 与真正的 pairwise `r_A+r_B` greedy。
- Collapse 去除 alignment、去除 energy ratio、不同 top-k 的消融。

### 5.2 经典全特征基线

- Full merged-effective-rank greedy。
- Facility Location greedy。
- k-Center Greedy / farthest-first traversal。
- k-Medoids/PAM。
- Agglomerative clustering 后选择 medoid。
- DPP 或 greedy log-determinant。
- Vendi-Greedy。
- PCA leverage-score sampling 或 column/row subset selection。

这些方法统一使用 pool-level similarity。至少比较三种 pool 表示：均值、top-k 子空间投影核、
以及池间平均/最邻近样本相似度。

### 5.3 摘要方法

- 四摘要 Collapse。
- 谱加权 alignment。
- 4-bucket 和 8-bucket 谱摘要。
- top-k 子空间，k = 10、20、50。
- randomized low-rank sketch，维度 = 16、32、64、128。
- confidence-triggered fallback：紧凑摘要低置信度时升级为 richer sketch。

### 5.4 任务感知参考上界

- CRAIG / GradMatch：监督梯度 coreset，仅作为上界。
- ELFS：使用聚类与代理训练动态的无标签样本级方法。
- 全候选 Stage-2 穷举：小规模集合上的最终 utility oracle。

任务感知方法与摘要方法的计算和标签权限不同，必须单独成表，不能作为同成本主比较。

## 6. 实验阶段

### E0. 实现正确性与预算审计

- 对每个算法建立小矩阵单元测试和穷举验证。
- 验证选择数量、去重、固定随机种子和通信字节计算。
- 检查方法是否意外访问标签、目标验证集或 family metadata。
- 输出 `method_resource_inventory.csv`。

验收：所有方法输入权限清晰，Full-rank/DPP/Facility Location 在小集合上与直接计算一致。

### E1. 特征提取与编码器稳定性

- 对四个编码器按官方 split 提取全部候选和目标特征；主实验每个 split 分层采样上限为
  5,000，另对小型数据集保留全部样本。
- 每个编码器抽取 3 个数据集重复运行，确认特征缓存确定性。
- 运行 centering、L2 normalization 和不归一化消融。

验收：缓存具有数据集、split、encoder、checkpoint 和预处理版本哈希。

### E2. 受控机制实验

- 扫描重叠类型、重叠率、预算、样本量和构造种子。
- 指标：重复检测 AUROC/AUPRC、真实合并有效秩、Oracle regret、选择分歧率。
- 分析 alignment 是否在高重叠下相对 rank-only 产生稳定增益。

验收：结论必须在至少 3 个编码器和 10 个构造种子上方向一致，否则报告为负面结果。

### E3. 自然数据池预筛选

- 在四组自然候选集合上运行全部经典基线和摘要方法。
- 主要指标：Stage-2 最优候选 Recall@L、top-k recall、最终 utility regret。
- 次要指标：有效秩、Vendi、重复族数量和运行时间。

验收：在移除至少 50% 候选时，最佳候选召回率达到 90%，并在 leave-one-domain-out 中保持。

### E4. 通信—精度—计算前沿

- 对 scalar、bucket、top-k subspace、sketch 和 full feature 扫描通信预算。
- 横轴分别报告 bytes/pool、CPU seconds、NPU-hours。
- 纵轴报告 merged-rank error、shortlist recall 和最终 regret。

验收：至少存在一个摘要配置，在 shortlist recall 上优于等字节的简单摘要，并显著低于完整特征成本。

### E5. 固定特征 adapter 验证

- 选择通过 E3 的方法，不对所有弱方法重复消耗 NPU。
- 使用 4 个编码器、8 个目标、3 个预算、5 个 adapter seeds。
- 同时运行固定总步数和固定每池步数协议。
- adapter 宽度：128、256、512；训练步数：300、600、1,200，主结果只使用预注册配置。

验收：两阶段方法在相同 NPU-hours 下接近全候选验证，或在相同 utility 下减少至少 50% Stage-2 评估。

### E6. 轻量真实数据消费验证

- 只对 E5 中最稳定的 4 至 6 个方法运行 LoRA/轻量微调。
- 选择 3 个代表性目标、2 个编码器、2 个预算、3 个 seeds。
- 固定样本数、token/image 数、优化步数和 NPU-hours。

验收：若谱选择收益不能在真实数据消费中复现，则论文只保留 frozen-representation pre-screening 定位。

### E7. 统计与失效分析

- target-level 和 source-family-level cluster bootstrap。
- leave-one-dataset-out、leave-one-domain-out、leave-one-encoder-out。
- 层次混合效应模型：方法为固定效应，target/source family/encoder 为随机效应。
- 主要比较预注册为 Collapse 或 richer summary vs rank-only；其余进行多重比较校正。
- 报告效应量、95% CI、配置胜率和最差域结果，不只报告 p 值。

## 7. NPU 与时间预算

### 7.1 必须先执行的 benchmark

在正式提取特征或运行 LoRA 前，分别执行 10 分钟 benchmark：

- 每个编码器提取 10,000 个样本的吞吐和峰值 HBM。
- 每个 LoRA 配置运行 100 步，记录 step/s 和峰值 HBM。
- adapter 在 4 种特征维度下各运行 100 次。

估算公式：

`NPU-hours = runs * seconds_per_run / 3600`

`wall-clock = NPU-hours / usable_NPUs / utilization`

共享服务器 utilization 按 0.7 估算；不得按 8 张卡全部可用计算承诺时间。

### 7.2 核心实验预算

| 阶段 | 规模假设 | NPU-hours | 单 NPU wall-clock | 4 NPU wall-clock |
|---|---|---:|---:|---:|
| E0 单元测试 | 小规模 | <0.2 | <1 小时 | <1 小时 |
| E1 特征提取 | 33 数据集 × 4 编码器 × 官方 split | 6–12 | 9–18 小时 | 2.5–5 小时 |
| E2/E3 Stage-1 | 经典方法与摘要方法 | 0–1 | CPU 12–36 小时 | CPU 12–36 小时 |
| E5 adapter 主实验 | 约 10,000–15,000 次训练 | 5–8（按当前吞吐） | 7–12 小时 | 2–3 小时 |
| E5 adapter 消融 | 宽度/步数子集 | 2–4 | 3–6 小时 | 1–2 小时 |
| E6 LoRA | 4–6 方法 × 3 目标 × 2 编码器 × 2 预算 × 3 seeds | 30–60 | 2–4 天 | 11–22 小时 |
| 复跑与失败余量 | 20% | 9–17 | 13–24 小时 | 3–6 小时 |
| **核心合计，不含 LoRA** | E0–E5 | **13–25** | **约 1–2 天** | **约 6–12 小时** |
| **完整合计，含 LoRA** | E0–E7 | **50–85** | **约 3–5 天** | **约 1–2 天** |

四卡时间仅在明确获得 4 张空闲 NPU 后成立。当前服务器状态下应采用单卡保守估计，并让
CPU Stage-1 与共享 NPU adapter 并行。特征提取和 LoRA 不应与占用 14–28 GB HBM 的其他任务同卡运行。

### 7.3 CPU、内存与存储

- Full merged-rank、Vendi、DPP 和 Facility Location 优先在 CPU 多核运行。
- 预计 CPU wall-clock 24–72 小时，可按候选集合和 encoder 并行。
- 在每个 split 上限 5,000 样本时，特征缓存约 4–10 GB；若保存完整官方 split，按编码器
  维度和数据集规模预留 20–60 GB。
- 中间相似度矩阵和结果预留 20 GB；总磁盘预算 80 GB。
- 每个作业必须输出 PID、配置快照、开始/结束时间、git commit 和失败状态。

## 8. 执行顺序与停止规则

- [x] P0：实现 Random-100、Full-rank greedy、Facility Location、k-Center、k-Medoids、DPP、Vendi。
- [x] P1a：建立方法权限、通信字节和运行时间资源清单。
- [ ] P1b：完成 scalar、bucket、subspace、sketch、full feature 的等字节完整扫描。
- [x] P2：完成四编码器特征 benchmark，再决定实际并行数。
- [x] P3：完成精确样本重叠的 E2 停止扫描；停止扩大四摘要/Collapse，保留 DPP-subspace。
- [ ] P4：运行 E3 自然池实验和 shortlist recall（混合自然池、严格 repeated-CV utility 与
  source-family-aware LODO 已完成；四组各至少 20 个候选的独立集合和第 8 个目标尚未完成）。
- [ ] P5：只将 E3 中有竞争力的方法送入 E5 adapter（pilot 未通过验收，暂缓扩大）。
- [ ] P6：若 E5 达到成本—效用验收标准，再运行 E6 LoRA。
- [ ] P7：完成 cluster bootstrap、LODO、跨编码器分析和最终表格（target/source-family cluster
  bootstrap、leave-one-source/domain/encoder-out 与跨编码器表已完成；层次混合效应模型因
  DPP/rank 配对 588 项中 587 项为零差异而近退化，尚未强行拟合）。

### 2026-09-02 执行状态

- P0 已完成 24 个 ResNet-50 受控配置、21 类方法和每配置 Random-100，共 2,880 行结果；
  当前全部单元测试 10/10 通过，三个 shard 的脚本哈希一致。
- P0 公平性复核发现原 `Collapse` 在选中池后读取完整特征重新计算累计 SVD，不能只按
  163,848 bytes/pool 计费。修正后其 28 池、平均预算 4 的 selector traffic 为 12,779,744
  bytes；DPP-subspace 无完整特征反馈，总量为 4,587,632 bytes。
- 原 full-feedback `Collapse` 与 `rank-only` 的 Full-rank 相对 gap 分别为 18.586% 和
  18.581%，没有显示增量价值；DPP-subspace 的 gap 为 0.569%。
- P2 已在每个编码器 10,000 图像上完成：ResNet-50 789.40、ViT-B/16 32.83、CLIP
  ViT-B/32 181.23、DINOv2 ViT-B/14 660.68 img/s。ViT 和 CLIP 存在部分算子 CPU fallback，
  只能用于实际排期，不能作为纯 NPU 性能结论。
- E1 的五数据集公共子集已在 NPU 2–5 完成：四编码器各 21,880 个向量，共 87,520 个；
  每个缓存保存 checkpoint、preprocess、脚本和文件哈希。完整 33 数据集 E1 和归一化消融未完成。
- P3 使用 5 数据集 × 4 个互斥基础池 + 5 个受控别名池、10 个构造种子、6 个重叠率、
  预算 3/5/8/10 和 4 个编码器，共 4,800 行严格 summary-only 结果。当前仅覆盖精确样本重叠，
  增强变体和特征噪声仍待补充。

严格 summary-only 相对 `rank-only` 的总体结果：

| 编码器 | Collapse-sketch 相对增益 | 胜/平/负 | DPP-subspace 相对增益 | 胜/负 |
|---|---:|---:|---:|---:|
| CLIP ViT-B/32 | -0.03% | 107/60/73 | +3.31% | 235/5 |
| DINOv2 ViT-B/14 | +0.62% | 120/102/18 | +3.71% | 240/0 |
| ResNet-50 | +0.56% | 80/136/24 | +10.95% | 231/9 |
| ViT-B/16 | -2.17% | 121/13/106 | +3.81% | 227/13 |

- `Collapse-sketch` 在四个编码器上方向不一致，并在预算 3 上四个编码器均为负；不满足 E2
  “至少 3 个编码器方向一致”的验收条件。原四摘要方法又没有定义无完整特征反馈的多步状态更新，
  因此停止扩大并降级为理论启发消融。
- DPP-subspace 只访问 top-k 子空间和单池质量标量，在四个编码器上均稳定优于 rank-only；
  旧 P4 自然池实验以它、Facility Location、rank-only 和 full-information greedy 参考为主。

### 2026-09-05 自然池 pilot 状态

- 已完成 21 个源池、7 个独立目标和 4 个编码器的混合自然池 pilot。源特征缓存为 84 个，目标
  官方 train/test 缓存为 56 个；每个编码器完成 735 行穷举 Stage-2 效用测量。
- Stage-2 预注册配置为每源池 500 个分层样本、宽度 256、600 步、5 个 seeds；Stage-1 使用
  1,000 个无标签样本、`top-k=20` 和预算 3/5/10。
- 在移除 52.4% 候选的 L=10 设置下，DPP-subspace 与 rank-only 的跨编码器平均最佳候选召回
  均为 82.14%，平均验证 regret 均为 0.001519，均仅在 2/4 个编码器上通过验收。
- L=5 时 DPP-subspace 的平均召回为 57.14%，低于 rank-only 的 75.00%；L=3 时分别为
  50.00% 和 64.29%。当前没有 alignment 相对简单秩摘要的稳定增量价值。
- DPP-subspace 在 28 个 encoder-target 配对上的召回胜/平/负：L=3 为 1/22/5，L=5 为
  1/21/6，L=10 为 0/28/0。因此触发“Recall 低于 90% 不得声称安全预筛选”的停止规则，P5
  主消融和 P6 LoRA 暂不启动。
- 四编码器在 CIFAR-10、PathMNIST、Rendered-SST2 上的独立特征复跑为 12/12 逐字节一致。
  ViT-B/16 注意力算子存在 CPU fallback，报告时间时必须标为混合 CPU/NPU 墙钟。
- post-hoc 目标划分敏感性诊断又完成 8 个 encoder-split 组合、5,880 行效用结果。三个划分的
  DPP Recall@10 为 82.14%/82.14%/75.00%，rank-only 为 82.14%/82.14%/78.57%，均未达到
  跨编码器 90% 门槛。
- 只有 5/28 个 encoder-target 在三个划分上保持共同的 tie-aware 最优源池；源池效用排序两两
  Spearman 平均为 0.544，且单源跨划分标准差 0.00778 大于平均 top-1 margin 0.00444。后续
  必须先用交叉验证或 bootstrap 置信集合稳定 Stage-2 utility oracle，不能把单次 top-1 当作精确真值。
- 完整结果、执行记录和限制见 `results/two_stage_natural_shortlist/README.md`。P4 仍缺四组各至少
  20 个候选池、第 8 个目标和 source-family-aware LODO，不能因本 pilot 完成而勾选。

### 2026-09-07 严格 repeated-CV 与 LODO 状态

- 为消除单 holdout oracle 噪声，已实现同一 adapter 内的 3 个独立分层 5-fold 分区。四个编码器
  共训练 420 个源池 adapter，记录 2,940 个 adapter-target utility；每条记录含 15 个 fold 和
  3 个 partition accuracy。官方 test 每个组合只评估一次，未参与选择。
- 三次“独立启动的 CV 运行”不能直接合并：2,940 个对齐键中有 132 个 test accuracy 发生变化，
  51 个变化超过 0.001，最大变化 0.02344，说明 NPU adapter 重训引入非确定性。正式结果改用
  单次训练、联合评估三个 CV 分区的协议。
- 严格联合 CV 在 `L=10/21`（删除 52.38%）下，DPP-subspace 和 rank-only 的精确召回均为
  85.71%，family recall 也均为 85.71%；DPP 未达到 90% 验收线。`L=5` 时 DPP 为 75.00%，
  rank-only 为 82.14%。
- 同一 adapter 的三个 CV 分区中，19/28 个 encoder-target 保持同一 primary oracle，20/28 至少
  共享一个 tie-aware oracle；family 口径没有增加。两两 Spearman 均值为 0.8943，但仍有
  67/84 个 top-1 margin 不超过 0.005。
- 四组 source-domain LODO 已完成。约 50% 删除下，DPP 精确召回依次为 82.14%（medical）、
  85.71%（digits/characters）、53.57%（general vision）和 85.71%（rendered text）；四组均未
  通过 90% 验收线。四组汇总 DPP 为 76.79%，rank-only 为 79.46%。
- Target-cluster bootstrap 的 LODO 总体 DPP recall 为 76.79%，95% CI `[65.18%, 86.61%]`；
  DPP 减 rank-only 为 -2.68 个百分点，95% CI `[-7.14, 0.89]`。多样性项仍无稳定增量证据。
- P4 的第 8 个目标和四组各至少 20 候选的独立集合没有可审计的一致缓存，不能用旧版特征或
  macOS `._*` 元数据文件拼接。P4 保持未完成；该缺口不推翻当前失败结论，但限制外推范围。
- 结果与审计见 `results/two_stage_repeated_cv_utility/README.md` 和
  `results/two_stage_repeated_cv_lodo/README.md`。停止规则继续生效：不扩大 E5，不启动 E6。
- 进一步完成 21 个 leave-one-source folds x 4 编码器，共 588 个 `L=10/20` 主分析单元。
  DPP-subspace recall 为 85.71%，rank-only 为 85.54%；配对结果为 1 胜、587 平、0 负，平均差
  仅 +0.17 个百分点。DPP validation regret 还略高 0.0000084，无法构成实际优势。
- Leave-one-source 的 DPP 最差召回为 82.14%（删除 STL-10 或 Tiny-ImageNet），最好为 89.29%
  （删除 DermaMNIST 或 PathMNIST），21 折均未通过 90%。Target-cluster 95% CI 为
  `[71.09%, 99.66%]`；结果见 `results/two_stage_repeated_cv_source_lodo/README.md`。

停止规则：

- Collapse 在 3 个编码器上均不优于 rank-only：降级为理论启发的消融，不再作为主算法。
- richer summary 不优于 Vendi/Facility Location 且通信成本更高：不主张实用优势。
- Stage-1 shortlist 的最佳候选召回率低于 90%：不得声称可安全预筛选。
- adapter 收益不能在 LoRA 中复现：应用范围限定为冻结表示诊断。

## 9. 最终报告表格

1. 数据集、family、官方 split、样本数和编码器表。
2. 方法输入权限、通信字节、CPU/NPU 时间表。
3. 受控重叠 AUROC/AUPRC 与 merged-rank regret 表。
4. 自然池 shortlist Recall@L 与最终 utility regret 主表。
5. 通信—召回率—NPU-hours Pareto 图。
6. 编码器、预算、域和样本量分层表。
7. Adapter 与 LoRA 数据消费结果表。
8. LODO、leave-one-encoder-out、cluster bootstrap 和最差域结果表。
9. 失败诊断和负面结果表。

## 10. 当前判断

受控实验表明 top-k 子空间 alignment 能检测同分布或重叠池，且 DPP-subspace 能改善受控集合上的
合并有效秩；但这一优势没有转化为自然池 Stage-2 shortlist recall。更严格的同一 adapter、三分区
5-fold 协议中，DPP-subspace 在移除 52.4% 候选时召回 85.71%，与 rank-only 相同；四组 LODO
汇总进一步降到 76.79%，低于 rank-only 的 79.46%，且 general-vision LODO 只有 53.57%。现有
证据不支持通用、domain-robust 或“安全”的无标签预筛选主张。P5/P6 按停止规则不执行；剩余工作
仅用于界定负面结果和通信成本，不应继续扩张应用性主张。

## 11. 2026-09-08 审计修复 Checklist

详细证据和数值见 `AUDIT_2026-09-08.md`。以下“完成”只表示代码或文档修复已经实现；凡是改变
抽样、算法或指标的项目，必须等待新实验后才能标记为“经验验证完成”。

### P0：理论与主算法对应

- [x] 实现可组合 rank-$L$ Gram PSD 因子，不再把累计状态重新压成四个标量。
- [x] 实现截断 Gram 有效秩直接 greedy、尾部核质量/平方质量/尾秩界。
- [x] 实现 Theorem 8 的候选 log-rank 区间与逐步 exact-greedy 分离证书。
- [x] 统一 Legacy Collapse 的单池与累计统计层级，均使用同一 top-$L$ sketch。
- [x] 保留 Legacy Collapse 为消融，不再将它列为理论主算法。
- [ ] 在四编码器真实缓存上扫描 $L=10,20,50$，报告区间覆盖率、宽度、证书率和回退率。

### P1：无标签边界与来源权重

- [x] Stage-1 改用只读取 `H` 的 loader，可直接处理不含 `y` 的 NPZ。
- [x] 来源特征抽样默认改为 deterministic unlabeled random sampling。
- [x] 默认拒绝标签分层或来源不明的旧 Stage-1 缓存。
- [x] 自然池默认使用 equal-source 权重，使每个来源总平方能量约为 1。
- [x] 缓存清单记录抽样方式、输入 SHA-256、总能量和尾部界。
- [ ] 重新提取四编码器标签无关来源缓存并验证逐字节确定性。

### P2：公平指标与经典基线

- [x] 将 global-lineage AUROC 与 same-dataset matched AUROC/AUPRC、Top-1、MRR 分开。
- [x] 0% overlap 的 parent ground truth 标为未定义，只保留 designated-parent 负对照。
- [x] 将 `full_merged_rank` 重命名为 `exact_merged_rank_greedy`，明确它不是组合上界。
- [x] 加入有组合数门限的 exhaustive merged-rank oracle，并固定 greedy 反例测试。
- [x] 经典重复来源默认使用固定种子随机抽取；最高秩复制仅作为显式压力测试。
- [x] 加入 agglomerative-medoid 与 centroid leverage-score 基线。
- [x] 保留精确 recall，同时输出固定容差曲线和 paired-bootstrap 候选置信集召回。
- [ ] 用修正指标重跑 controlled overlap，并按 encoder/seed 报告 matched 指标置信区间。
- [ ] 运行 scalar、rank-$L$、subspace 和 full-feature 的等字节 Pareto 扫描。

### P3：Stage-2 与可复现性

- [x] `torch_npu` 延迟到 `adapt` 阶段导入，CPU 可独立运行筛选和汇总。
- [x] 适配 run fingerprint 绑定代码、筛选 manifest、参数和全部输入缓存哈希。
- [x] 按完整 source/seed 组原子写入；发现半组结果时拒绝不安全恢复。
- [x] 默认启用 PyTorch deterministic algorithms，并显式设置 NPU seed。
- [x] 固定 Stage-2 来源样本数；不足预算默认报错而不是静默缩短。
- [x] 加入 `frozen_identity` 与 `random_adapter` 两个来源无关基线。
- [x] 高维 ridge 在样本少于维度时使用数学等价的对偶求解。
- [ ] 在服务器验证 Ascend 确定性开关是否被所有使用算子支持。
- [ ] 重跑 repeated 5-fold CV，并报告 adapter 相对 identity/random-adapter 的增量。
- [ ] 在 frozen-feature gate 通过后再执行 LoRA 与多池混合验证。

### P4：数值、数据覆盖与发布

- [x] Gram/Scatter 在乘法前转为 `float64`，使用相对谱阈值排除伪小特征值。
- [x] Arrow shard 按内容哈希去重；来源和目标缓存元数据绑定原始文件哈希。
- [x] 修正版缓存与结果默认写入独立的 `unlabeled`/`corrected_v2` 路径。
- [x] 单元测试扩展到 rank-$L$ 可组合性、理论区间、matched 指标、label-free loader、
  resume 门禁、数值秩和 greedy 反例。
- [ ] 建立彼此独立的医学、字符、通用视觉和混合候选集合，每组至少 20 个池。
- [ ] 增加相应的非通用视觉目标和第 8 个目标；重新评估 domain robustness。
- [ ] 新结果完成前，不把历史 pilot 表格改写为修正版算法结果。
- [ ] 将通过验证的新结果同步进论文正文、附录、README 和匿名发布包。
- [ ] 使用有效的匿名 GitHub 凭据推送当前分支；现有 token 于 2026-09-08 返回 401。
