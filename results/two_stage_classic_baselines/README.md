# 经典基线受控筛选实验

> **历史 pilot，非修正版结果。** 本实验复制最高有效秩来源，且把 exact-score greedy 误称为
> 参考上界。修正版默认随机选择重复来源，并单独提供有规模门限的 exhaustive oracle；见
> `AUDIT_2026-09-08.md`。下表只用于审计旧协议。

## 实验范围

本目录比较两阶段 Stage-1 方法与经典子集选择方法。实验使用 ResNet-50 冻结特征、22 个候选源、
6 个受控重叠别名、3 个构造种子、4 个重叠率（25%、50%、75%、100%）、选择预算 3/5，
共 24 个配置。每池采样 250 个样本；随机基线在每个配置重复 100 次。

所有方法均不访问标签或目标数据。`family_rank` 使用真实 family 元数据，因此只作为去重 oracle，
不属于可部署方法。`full_merged_rank` 使用完整特征逐步精确评分，只是贪心参考，不是组合上界。

## 主要结果

| 方法 | 合并有效秩均值 | 相对 exact greedy gap | 平均重复池数 | 胜过 rank-only 的配置比例 |
|---|---:|---:|---:|---:|
| Full merged-rank | 674.190 | 0.000% | 0.000 | 100.0% |
| Family oracle | 674.021 | 0.021% | 0.000 | 100.0% |
| DPP, sample nearest-neighbor | 672.699 | 0.188% | 0.125 | 100.0% |
| DPP, top-20 subspace | 669.729 | 0.569% | 0.167 | 100.0% |
| k-Medoids, top-20 subspace | 608.594 | 9.829% | 0.000 | 75.0% |
| Facility Location, top-20 subspace | 596.424 | 11.340% | 0.000 | 70.8% |
| Merged Vendi | 584.356 | 12.056% | 1.250 | 45.8% |
| Rank-only | 548.671 | 18.581% | 1.500 | 0.0% |
| Collapse | 548.633 | 18.586% | 1.500 | 58.3% |
| Random-100 | 475.633 | 29.538% | 0.087 | 16.7% |

`Collapse` 与 `rank-only` 的均值和重叠退化曲线几乎相同，当前单编码器实验不支持 alignment
带来稳定增量价值。相反，top-20 subspace DPP 在 24/24 个配置中都优于 rank-only，且接近
旧 exact-score merged-rank greedy 参考。

## 名义摘要与实际 selector traffic

在当前 ResNet-50、每池 250 样本、top-k=20 的设定下：

| 方法 | 摘要字节/池 | 选中池完整特征反馈 | 平均 selector 总字节 | 相对 Full-rank gap |
|---|---:|---:|---:|---:|
| Rank-only | 4 | 0 | 112 | 18.581% |
| Collapse, full-feedback | 163,848 | 2,048,000/选中池 | 12,779,744 | 18.586% |
| DPP, top-20 subspace | 163,844 | 0 | 4,587,632 | 0.569% |
| DPP, full sample representation | 2,048,000 | 0 | 57,344,000 | 0.188% |

这里撤回“Collapse 与 DPP 严格同字节”的旧表述。现有 Collapse 实现在每次选中池后，用该池的
完整特征重新计算累计 SVD；预算 3/5 分别增加 6.144/10.240 MB 反馈，表中的平均总字节按平均
预算 4 计算。DPP-subspace 只使用一次性摘要，选择阶段不读取完整特征，因此它在本实验中既更
准确也使用更少 selector traffic。严格 summary-only Collapse 必须用加权低秩 sketch 更新累计
状态，并作为单独实验报告。

完整候选集合的一次性传输量约为 4.59 MB（top-k subspace）和 57.34 MB（完整样本特征）。

`selection_seconds` 不包含 representation/similarity 构建时间。资源表中的
`shared_all_representation_preprocessing_seconds_mean` 同时构建了三类表示，因此
`end_to_end_upper_seconds_mean` 是保守上界，不是每个单独方法的精确端到端耗时。

## 文件

- `classic_screening_results.csv`：2,880 行逐次结果。
- `classic_screening_summary.csv`：按方法汇总。
- `method_resource_inventory.csv`：权限、通信量与运行时间审计。
- `classic_screening_manifest.json`：配置、选择集合、进程与脚本哈希。
- `classic_screening_shard*.log`：三个 CPU shard 的完整日志。

## 结论边界

这仍是人工精确重叠、单编码器、单一池规模的机制实验。它不能替代自然候选池上的 shortlist
recall、下游 utility 或跨编码器结论；这些结论必须等待 E2/E3 后续实验。
