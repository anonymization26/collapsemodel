# 联合重复交叉验证 Stage-2 效用实验

## 目的

本实验替代单次 train/validation holdout。它固定一次源池 adapter 训练，在同一组目标特征上评估
3 个独立分层 5-fold 分区，从而把“目标划分变化”与“adapter 重训的 NPU 非确定性”分开。

## 协议

- 编码器：ResNet-50、ViT-B/16、CLIP ViT-B/32、DINOv2 ViT-B/14。
- 候选源池：21；独立目标：7；每个源池 5 个 adapter seeds。
- 共训练 `4 x 21 x 5 = 420` 个源监督 adapter，得到 `4 x 21 x 5 x 7 = 2,940`
  条 target utility 记录。
- 每个目标使用 CV seeds `20260905/20260906/20260907`，每个 seed 为严格分层 5-fold；每条记录
  含 15 个 fold accuracy 和 3 个 partition accuracy。
- Stage-1 只读取无标签冻结特征；Stage-2 adapter 使用候选源池标签训练。官方 test split 不参与
  shortlist 或超参数选择，每个 adapter-target 组合只评估一次 test。
- 主预算 `L=10/21`，删除 52.38% 的候选。验收线为精确最佳候选召回率至少 90%。

协议完整性审计：四个编码器各 735 条数据行；2,940/2,940 条均为 3 partitions、15 folds，且
`validation_examples = 3 x unique_validation_examples`。

## 主结果

跨编码器共 28 个 encoder-target 单元：

| 方法 | Recall@3 | Recall@5 | Recall@10 | Family Recall@10 | Validation regret@10 | 通过验收 |
|---|---:|---:|---:|---:|---:|---|
| DPP-subspace | 67.86% | 75.00% | 85.71% | 85.71% | 0.000647 | 否 |
| rank-only | 67.86% | 82.14% | 85.71% | 85.71% | 0.000642 | 否 |
| full-merged-rank | 71.43% | 82.14% | 85.71% | 85.71% | 0.000564 | 否 |
| random-100 | 13.86% | 24.79% | 46.71% | 50.21% | 0.002692 | 否 |

`L=10` 时 DPP 和 rank-only 在 28/28 个配对上召回相同；`L=5` 时 DPP 相对 rank-only 的
平均差为 -7.14 个百分点。DPP 没有显示稳定的多样性增益。

Target-cluster bootstrap（7 个 target clusters，10,000 次）在 `L=10` 得到：

- DPP Recall：85.71%，95% CI `[71.43%, 100.00%]`；bootstrap 中达到 90% 的比例为 36.34%。
- rank-only Recall：85.71%，95% CI `[71.43%, 100.00%]`；达到 90% 的比例为 36.33%。
- DPP 减 rank-only Recall：0，95% CI `[0, 0]`。
- DPP 减 rank-only validation regret：0.0000048，95% CI `[-0.000027, 0.000041]`。

## Oracle 稳定性

联合运行内部的三个 CV 分区显示：

- 19/28 个 encoder-target 单元保持同一 primary oracle；20/28 至少有共同的 tie-aware oracle。
- source-family 口径仍为 19/28 和 20/28，没有消除不稳定性。
- 源池 utility 排序的两两 Spearman：均值 0.8943，中位数 0.9429，最小值 0.0688。
- 84 个 partition-level top-1 margin 中，19 个不超过 0.001，67 个不超过 0.005。
- 单源跨分区标准差均值为 0.001304，平均 top-1 margin 为 0.003340。

因此 CV 明显改善了单 holdout 的噪声，但仍有 9/28 个单元的 primary oracle 随分区变化。

## 运行成本

NPU4-7 并行运行，最早开始到最晚结束为 5 分 54 秒；四个作业墙钟合计约 1,054 秒，约
0.293 NPU-hours。逐作业时间见 `logs/`。

## 文件

- `folds_5_seeds_20260905_20260906_20260907/*/adaptation_results.csv`：完整 utility 数据。
- `folds_5_seeds_20260905_20260906_20260907/cross_encoder_summary.csv`：跨编码器主表。
- `folds_5_seeds_20260905_20260906_20260907/cluster_bootstrap_summary.csv`：target/source-family
  cluster bootstrap。
- `folds_5_seeds_20260905_20260906_20260907/leave_one_encoder_out.csv`：留一编码器分析。
- `folds_5_seeds_20260905_20260906_20260907/target_summary.csv`：逐目标结果。
- `folds_5_seeds_20260905_20260906_20260907/partition_stability/`：同一 adapter 内的分区稳定性。

## 结论

在更严格且去除重训混杂的 utility 协议下，DPP-subspace 仍未达到 90% shortlist recall，也未优于
rank-only。按照预注册停止规则，不进入扩大 adapter 消融和 LoRA 阶段。
