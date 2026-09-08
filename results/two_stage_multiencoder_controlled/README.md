# 四编码器受控重叠停止扫描

## 协议

每个编码器使用 CIFAR-10、CIFAR-100、DTD、EuroSAT、SVHN 的正式预处理特征。每个数据集按
构造种子切成 4 个互斥基础池，并增加 1 个别名池，共 25 个候选池；每池 250 个样本。别名池与
对应基础池共享 0%、10%、25%、50%、75% 或 100% 样本，其余样本来自同数据集未使用部分。

- 编码器：ResNet-50、ViT-B/16、CLIP ViT-B/32、DINOv2 ViT-B/14。
- 构造种子：10 个。
- 选择预算：3、5、8、10。
- 方法：rank-only、Collapse、DPP-subspace、Facility-subspace、lineage oracle。
- 规模：每个编码器 60 个候选集合 × 4 个预算 × 5 个方法 = 1,200 行，共 4,800 行。
- Stage-1 方法不读取特征文件中的 `y`；完整特征只用于事后计算真实合并有效秩。

胜/平/负使用相对数值容差：
`abs(delta) <= 1e-6 * max(abs(rank_reff), 1)`。

## 两条信息路径

`hybrid_full_feedback/` 保留旧实现的诊断结果。该实现每选中一池，就读取其完整特征并重新计算
累计 SVD，因而不是纯摘要方法。

`summary_only/` 是严格路径。每池只传 top-20 奇异值、右子空间、单池有效秩和核范数；已选集合
由加权 low-rank sketch 更新，不读取选中池完整特征。其每池通信量为：

| 编码器 | Collapse-sketch | DPP-subspace |
|---|---:|---:|
| ResNet-50 | 163,928 bytes | 163,844 bytes |
| ViT-B/16 | 61,528 bytes | 61,444 bytes |
| DINOv2 ViT-B/14 | 61,528 bytes | 61,444 bytes |
| CLIP ViT-B/32 | 41,048 bytes | 40,964 bytes |

## 严格结果

| 编码器 | Collapse-sketch 相对增益 | 胜/平/负 | DPP-subspace 相对增益 | 胜/负 |
|---|---:|---:|---:|---:|
| CLIP ViT-B/32 | -0.03% | 107/60/73 | +3.31% | 235/5 |
| DINOv2 ViT-B/14 | +0.62% | 120/102/18 | +3.71% | 240/0 |
| ResNet-50 | +0.56% | 80/136/24 | +10.95% | 231/9 |
| ViT-B/16 | -2.17% | 121/13/106 | +3.81% | 227/13 |

Collapse-sketch 的问题集中在小预算：预算 3 时四编码器的平均有效秩差均为负；预算 5 时只有
ViT 显著为负，其余近似持平；预算 8/10 才普遍转正。这说明 top-k sketch 的累计状态更新不能
稳定替代真实合并谱，尤其在前几步选择最重要时。

DPP-subspace 在四个编码器上总体均为正，且 960 个配置中 933 胜、27 负。它与
Collapse-sketch 的通信量几乎相同，但不需要定义递归标量谱预测。因此 P3 停止规则的决定是：

1. 停止扩大原四摘要/Collapse 多步选择器，将其降级为理论启发消融。
2. 保留 pairwise subspace alignment 作为冗余信号。
3. P4 自然池实验以 DPP-subspace、Facility、rank-only 和 full-information 上界为主。

## Alignment 检测

top-20 子空间相似度对别名 lineage 的 AUROC 在 0% 样本重叠时约为 0.922–0.924，在 50%
重叠时为 0.985–0.996，在 75% 和 100% 时为 1.0。alignment 本身能识别同分布或重叠池；
负面结果针对的是 Collapse 的多步状态更新，而不是 alignment 信号不存在。

## Hybrid 诊断

full-feedback Collapse 在四编码器上的平均相对增益分别为 +3.46%、+3.40%、+8.46%、+3.56%。
它明显优于严格 sketch 的结果，但这种改善使用了选中池完整特征，不能放入同通信预算主表。
该对照直接量化了“精确累计谱反馈”与“可部署摘要状态”之间的缺口。

## 文件与复现

- `summary_only/summary_stopping_results.csv`：严格路径 4,800 行结果。
- `summary_only/summary_stopping_decision.csv`：按编码器的停止规则统计。
- `summary_only/summary_stopping_pairwise.csv`：按重叠率和预算分层的成对统计。
- `summary_only/summary_stopping_alignment.csv`：alignment AUROC/AUPRC。
- `summary_only/shards/`：5 个原始 shard、manifest 和文件哈希。
- `hybrid_full_feedback/`：同结构的完整特征反馈诊断结果。

严格 v2 脚本 SHA-256 为
`bcf1f48f6293de01931658f4c1f848a908c4f25db54b1344a99a202e7e88b3d2`。
本地重新运行合并器可逐文件验证 checksum、脚本哈希、唯一键和 4,800 行计数。

## 结论边界

本实验只覆盖五个数据集内部切片和精确样本重叠，不能替代自然多源池。尚未完成增强变体、特征
噪声、bucket/sketch 维度扫描、shortlist recall、下游 utility 和 leave-one-domain-out；因此不能
据此声称 DPP-subspace 已实现通用无标签预筛选。
