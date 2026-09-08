# Leave-one-source-out 实验

## 目的与协议

该实验逐一删除 21 个候选源池中的一个，在剩余 20 个候选上重新执行 Stage-1，并用严格联合重复
5-fold utility 表重新定义 oracle。主预算为 `L=10/20`，恰好删除 50% 候选。

为避免 84 次重复 SVD，每个编码器只计算一次 21 池谱统计和 subspace similarity，再对各留一子矩阵
运行 DPP、rank-only、Facility Location、k-Center、pool-Vendi 和 Random-100。四个编码器共得到：

- 21 个被删除源池 x 4 个编码器 x 7 个目标 = 588 个主分析单元；
- 84 个完整 folds，84/84 个 shortlist 文件均为 2,205 数据行加表头；
- 所有 fold 使用同一份已完成的 Stage-2 utility，不重新训练 adapter。

## 汇总结果

| 方法 | Exact Recall@10 | Family Recall@10 | Validation regret | 通过 90% 验收 |
|---|---:|---:|---:|---|
| DPP-subspace | 85.71% | 85.71% | 0.000626 | 否 |
| rank-only | 85.54% | 85.54% | 0.000618 | 否 |
| pool-Vendi-subspace | 76.02% | 80.95% | 0.000475 | 否 |
| k-Center-subspace | 61.73% | 67.18% | 0.000864 | 否 |
| Facility Location | 29.25% | 33.84% | 0.002665 | 否 |
| Random-100 | 49.55% | 53.72% | 0.002412 | 否 |

DPP 相对 rank-only 在 588 个配对上为 1 胜、587 平、0 负，平均召回差仅 +0.17 个百分点。其
validation regret 反而平均高 0.0000084（10 胜、559 平、19 负）。这不是稳定或有实际意义的优势。

## 稳定性

Target-cluster bootstrap（7 clusters，10,000 次）：

- DPP Recall：85.71%，95% CI `[71.09%, 99.66%]`；达到 90% 的 bootstrap 比例为 36.66%。
- rank-only Recall：85.54%，95% CI `[71.09%, 99.66%]`；达到 90% 的比例为 35.40%。
- DPP 减 rank-only Recall：+0.17 个百分点，95% CI `[0, 0.51]`。
- DPP 减 rank-only validation regret：+0.0000084，95% CI `[-0.000013, 0.000038]`。

按编码器分层，CLIP 和 ViT-B/16 的 DPP Recall 为 100%，DINOv2 与 ResNet-50 都为 71.43%。
按目标分层，Beans 为 50%，EuroSAT 为 52.38%，其余目标为 97.62%–100%。因此总体数字主要由
编码器和目标组成驱动，不能解释为普适稳定性。

逐删除源池时，DPP 的最差 Recall 为 82.14%（删除 STL-10 或 Tiny-ImageNet），最好为 89.29%
（删除 DermaMNIST 或 PathMNIST）；21 个 leave-one-source folds 无一达到 90%。

## 运行成本与文件

共享预处理后的墙钟：CLIP 18 秒、DINOv2 25 秒、ViT 25 秒、ResNet-50 44 秒；四作业并行墙钟
约 44 秒，无额外 NPU adapter 成本。

- `exclude_*/`：每个删除源池、每个编码器的 manifest、明细和汇总。
- `source_lodo_summary.csv`：逐删除源池和全局汇总。
- `source_lodo_dpp_vs_rank_paired.csv`：588 个主配对。
- `source_lodo_cluster_bootstrap_summary.csv`：target、source-family、excluded-source bootstrap。
- `source_lodo_leave_one_encoder_out.csv`：留一编码器汇总。
- `*_batch_manifest.json` 和 `logs/`：输入哈希、脚本哈希及运行时间。

## 结论

Leave-one-source-out 没有改变主判断。DPP 与 rank-only 几乎完全重合，且二者都未达到安全预筛选
要求。该结果补齐 E7 的 source-dataset exclusion 诊断，但不支持恢复 E5/E6。
