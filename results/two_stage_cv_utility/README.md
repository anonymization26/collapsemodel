# 独立重复 CV 诊断（非正式主结果）

本目录保留三个分别启动、分别训练 adapter 的 5-fold CV 运行。它们最初用于目标划分敏感性分析，
后来发现同时改变了两个因素：CV partition seed 和 NPU adapter 重训结果。因此只能作为系统
非确定性诊断，不能用于估计纯粹的 CV partition 效应。

## 发现

按 `(encoder, source, adapter_seed, target)` 对齐三次运行，共有 2,940 个完整键：

- 132 个键的官方 test accuracy 差异超过 `1e-8`；
- 51 个键的差异超过 0.001；9 个超过 0.01；
- test accuracy range 的均值为 0.000106，中位数为 0，最大值为 0.0234375。

目标 CV seed 不应影响官方 test 预测；这些变化证明重训本身存在 NPU 非确定性。脚本
`scripts/merge_two_stage_cv_repeats.py` 会在 test accuracy 变化超过 `1e-8` 时拒绝合并，防止把
独立重训误写为 repeated-CV。

三个运行在 `L=10` 的精确召回为：

| CV seed | DPP-subspace | rank-only |
|---|---:|---:|
| 20260905 | 96.43% | 96.43% |
| 20260906 | 85.71% | 89.29% |
| 20260907 | 85.71% | 85.71% |

正式结论应使用 `results/two_stage_repeated_cv_utility/`：该运行对每个 adapter 只训练一次，再在
三个目标 CV 分区上评估。
