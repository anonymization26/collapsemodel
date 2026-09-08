# Source-domain LODO 实验

## 协议

在联合重复 5-fold utility 表上，分别从 Stage-1 候选池中完全删除一个源域，并在剩余候选中重新
计算 oracle 和 shortlist。每个域选择“删除比例不低于 50% 且最接近 50%”的预注册主预算。

| 留出的源域 | 剩余候选 | 主预算 | 删除比例 |
|---|---:|---:|---:|
| medical | 10 | 5 | 50.00% |
| digits/characters | 17 | 8 | 52.94% |
| general vision | 16 | 8 | 50.00% |
| rendered text | 20 | 10 | 50.00% |

每个设置覆盖 4 个编码器和 7 个目标，共 28 个 encoder-target 单元。LODO 只重跑 CPU Stage-1
和汇总，不重复训练 adapter。

## 主结果

| 留出的源域 | DPP Recall | DPP Family Recall | rank-only Recall | full-rank Recall | DPP validation regret |
|---|---:|---:|---:|---:|---:|
| medical | 82.14% | 92.86% | 89.29% | 89.29% | 0.000785 |
| digits/characters | 85.71% | 89.29% | 89.29% | 89.29% | 0.000612 |
| general vision | 53.57% | 64.29% | 53.57% | 64.29% | 0.001466 |
| rendered text | 85.71% | 85.71% | 85.71% | 89.29% | 0.000156 |
| 四组汇总 | 76.79% | 83.04% | 79.46% | 83.04% | 0.000755 |

四个域的精确召回均未达到 90%。唯一超过 90% 的局部数字是留出 medical 后的 DPP family
recall（92.86%），但其精确源池 recall 只有 82.14%，不满足主验收标准。

## 统计诊断

四组主预算合并后，target-cluster bootstrap（7 clusters，10,000 次）为：

- DPP Recall：76.79%，95% CI `[65.18%, 86.61%]`。
- rank-only Recall：79.46%，95% CI `[66.07%, 91.07%]`。
- DPP 减 rank-only Recall：-2.68 个百分点，95% CI `[-7.14, 0.89]`。
- DPP 减 rank-only validation regret：-0.000111，95% CI `[-0.000215, 0.000003]`。

按 primary-oracle source family 等权重的 14-cluster bootstrap，DPP 与 rank-only 的召回估计分别为
52.50% 和 53.27%，区间很宽；结果不支持跨 source family 的稳定泛化。留一编码器结果也不稳定：
正式池中去掉 DINOv2 或 ResNet 后可达到 90.48%，去掉 CLIP 或 ViT 后只有 80.95%。

最明显的失效是留出 general vision：DPP 和 rank-only 都降到 53.57%。这说明当前候选排序主要依赖
general-vision 源池，不能视为 domain-robust pre-screening。

## 运行记录与文件

- DINOv2、CLIP、ViT 的四折 LODO 墙钟约 1.5–2.3 分钟；ResNet-50 为 8 分 48 秒。
- `exclude_*/`：各域、各编码器的 manifest、shortlist 明细和汇总。
- `lodo_primary_summary.csv`：统一主预算表。
- `lodo_dpp_vs_rank_paired.csv`：112 个 encoder-target-domain 配对。
- `lodo_cluster_bootstrap_summary.csv`：target、source-family 和 excluded-domain bootstrap。
- `lodo_leave_one_encoder_out.csv`：汇总 LODO 的留一编码器分析。
- `logs/`：四个 runner 的开始、结束时间及脚本哈希。

## 结论

LODO 直接否定“在移除至少 50% 候选时保持 90% 最佳候选召回”的验收条件。DPP 的多样性项没有
稳定优于 rank-only；因此不扩大 E5，也不启动 E6 LoRA。
