# E2 探索性 Oracle Headroom 协议

> 状态：在运行前冻结的事后诊断协议。它使用已经打开过的 E2 `target-test`，不属于新的确认性检验。

## 问题

首轮真实 E2 中，Target A-opt 相对最强基线只有约 `0.3%` 的 Brier 改善。该结果至少有三种解释：

1. 当前候选池本身几乎没有可利用的组合差异；
2. 候选池存在明显 headroom，但 A-opt 代理或 greedy 搜索没有找到；
3. A-opt 排序合理，但冻结 ridge 与 Brier 的效应尺度很小。

本实验只区分这些解释，不重新计算 H2/H2b，也不据此修改方法或阈值。

## 穷举范围

每个数据集-编码器-目标域任务有 12 个候选块。对 `K in {1, 3, 5}` 穷举：

| 预算 | 组合数 |
| ---: | ---: |
| 1 | 12 |
| 3 | 220 |
| 5 | 792 |
| 合计 | 1,024 |

四个数据集-编码器运行、每个运行四个目标域，共评价 16,384 个唯一组合。每个组合使用与父 E2
完全相同的行 L2 归一化、ridge 正则、类别词表、`target-test` 和指标实现。

## 输出量

- 事后真实 Brier oracle 及最差组合；
- Target A-opt 和所有父实验方法的真实组合 rank、相对 oracle regret 与 top-10 recall；
- 父实验已评价选择中的最佳事后 envelope，以及 oracle 相对该 envelope 的剩余 headroom；
- Target A-opt 代理与真实 Brier 的组合级 Spearman 相关；
- 两个编码器对同一组合的真实 Brier 排序稳定性。

归一化 regret 固定为

```text
(selected_brier - oracle_brier) / abs(oracle_brier)
```

组合分数相同时使用候选 ID 的字典序确定唯一 rank；同时报告最优容差集合大小，避免把数值平局包装成
稳定 top-1。`Q=min(10, number_of_combinations)`。

## A-opt 排序估计

对所有 16,384 个组合逐一精确计算 full-dimensional A-opt trace 会额外重复昂贵矩阵分解。本协议使用
64 个固定 Rademacher 探针，以共同随机数估计
`tr(C_T (lambda I + H_S^T H_S)^-1)`。探针右端与 ridge 的 one-hot 右端在同一次线性求解中处理，
因此不改变 ridge 预测。每个组合同时报告前后 32 个探针的估计及相对差异；相关性只能在探针稳定性
可接受时解释为 A-opt 排序诊断，不能当作确定性证书。

## 解释规则

- oracle 相对强基线也小于 2%：候选任务缺少预注册意义上的实质 headroom；优先重构候选池。
- oracle headroom 明显大于 2%，但 Target A-opt regret 较大：代理目标或 greedy 搜索是主要瓶颈。
- Target A-opt 接近 oracle，但绝对改善仍小：下游读出、校准或任务尺度限制了实际效应。
- 结论在数据集或编码器间反向：必须报告异质性，不能只报告总均值。

这些规则是探索性诊断，不是新的通过门槛。任何方法修改都必须在新的目标测试集或新数据集上重新冻结
确认性协议。

## 运行

单个数据集-编码器运行：

```bash
bash scripts/run_target_conditioned_e2_oracle_server.sh \
  DATASET ENCODER MANIFEST_ROOT FEATURE_ROOT PARENT_E2_ROOT OUTPUT_ROOT
```

四个运行完成后统一汇总：

```bash
python3 scripts/summarize_target_conditioned_e2_oracle.py \
  --run-dir OUTPUT_ROOT/pacs/resnet50 \
  --run-dir OUTPUT_ROOT/pacs/dinov2_b14 \
  --run-dir OUTPUT_ROOT/office_home/resnet50 \
  --run-dir OUTPUT_ROOT/office_home/dinov2_b14 \
  --out-dir OUTPUT_ROOT/summary
```
