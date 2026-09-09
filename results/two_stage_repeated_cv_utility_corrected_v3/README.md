# corrected_v3 严格 held-out Stage-2 审计

本目录保存 21 个来源、7 个目标、4 个编码器和 5 个 adapter seeds 的修正版自然池审计结果。
每个来源监督 bottleneck adapter 使用来源标签训练；目标 train 内部的 3 个独立分层 5-fold 划分
只用于 ridge probe 验证和来源选择。选择 manifest 及 checkpoint 哈希冻结后，独立命令才读取目标
test。四个编码器各有 735 个 `source x seed x target` 适配结果。

## Shortlist 结果

在 `L=10/21`（移除 52.38% 来源）时：

| 方法 | 严格 top-1 recall | Held-out test regret（相对 adapter oracle） |
|---|---:|---:|
| rank-L Gram | 75.00% | 0.000221 |
| rank-only | 85.71% | 0.000820 |
| DPP-subspace | 85.71% | 0.000730 |
| Full-Gram greedy | 89.29% | 0.000765 |

所有方法均未通过“至少移除一半候选且 recall 不低于 90%”的预注册门槛。DPP 与 rank-only 的
差异很小；对应 cluster bootstrap 结果见 `cluster_bootstrap_summary.csv`。

## 来源无关对照

| 表示 | 平均 held-out 准确率 |
|---|---:|
| frozen identity | 0.83654 |
| untrained random adapter | 0.77144 |
| validation-oracle source-supervised adapter | 0.76951 |

oracle 相对 identity 为 -0.06703，target-cluster 95% CI 为 `[-0.09771, -0.04400]`；相对 random
adapter 为 -0.00193，95% CI 为 `[-0.00409, 0.00022]`。每个确定性方法在每个 shortlist 大小下
都在 28/28 个 encoder-target 单元中低于 frozen identity。因此，小 regret 只说明 selector 接近
一个整体有害的 adapter 族内 oracle，不能解释为下游效用或数据消费收益。

## 可复现范围

`aggregation_manifest.json` 绑定所有 shortlist、held-out、原始 test evaluation、运行 sidecar 和
对照文件。每个编码器子目录保留复核聚合所需的 CSV 与 manifest，但不分发特征缓存和 adapter
checkpoint。当前审计为了估计穷举 regret 而离线训练了全部候选，所以总墙钟不能作为部署节省。
LoRA 与多池混合未运行，因为 frozen-feature gate 已经失败。

在具有匹配特征缓存和 Ascend 环境的机器上，可按每张 NPU 分配编码器运行：

```bash
PROJECT_ROOT="$PWD" \
SOURCE_DIR=/path/to/source/features \
TARGET_DIR=/path/to/target/features \
BASE_RESULTS=/path/to/screening/results \
RESULT_ROOT=/path/to/output \
bash scripts/run_two_stage_repeated_cv_utility_npu.sh 4 resnet50

python3 scripts/summarize_two_stage_cross_encoder.py \
  --results-dir /path/to/output \
  --expected-encoders resnet50 vit_b16 clip_b32 dinov2_b14
```
