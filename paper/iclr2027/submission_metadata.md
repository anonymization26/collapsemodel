# ICLR 2027 投稿信息

由 `build_assets.py` 从唯一题目、关键词和摘要文本源生成，与论文输入保持一致。
状态：作者审阅稿，尚未提交 OpenReview；不表示已完成全部补实验或作者科学核验。

## Title

Feature Geometry for Data-Pool Screening: Utility, Stability, and Limits

## Abstract

Selecting source-data pools for a target task requires deciding which combinations are worth training and validating. Frozen representations allow comparisons before supervised evaluation, but geometric diversity alone does not indicate whether the represented directions matter to the target. We study a two-stage approach that uses unlabeled features to form a shortlist and labeled validation to make the final choice. Starting from effective rank, constructed by exponentiating the entropy of normalized feature singular values, we examine how pooled features distribute variation across directions. We then compare target-weighted A-optimal design, motivated by a shared Bayesian linear model, with second-moment matching and target-blind diversity criteria. On PACS and Office-Home, direct selection produces small gains, and even the best evaluated combinations offer limited improvement over the baselines. In an equal-sample-cost DomainNet study, retaining roughly half the combinations preserves 98.3% of the test Brier top ten on average. Exploratory changes to projection seeds and the classifier readout preserve high average recall, including when candidate losses are more widely separated. Second-moment matching is competitive with A-optimal screening. These findings support using target-aware feature geometry to narrow the set of source combinations that receive supervised evaluation.

## Keywords

data selection, data-pool screening, frozen representations, optimal experimental design, distribution shift, empirical evaluation

## 中文题目

特征几何用于数据池预筛选：效用、稳定性与局限

## 中文摘要

为目标任务选择源数据池，需要先判断哪些组合值得训练和验证。冻结表示允许在监督评价之前比较候选，但几何多样性本身并不说明所覆盖的方向是否与目标有关。本文研究两阶段方案，先用无标签特征构建短名单，再通过带标签的验证数据作最终选择。我们从有效秩出发，将归一化特征奇异值的熵取指数，考察合并特征的变化如何分布在不同方向上；随后比较由共享贝叶斯线性模型引出的目标加权 A-optimal 设计、二阶矩匹配和目标无关多样性准则。在 PACS 和 Office-Home 上，直接选择的收益较小，即使已评价的最佳组合相对基线也只有有限改善。在固定样本成本的 DomainNet 实验中，保留约一半组合，平均可召回测试 Brier 前十名中的 98.3%。探索性实验改变投影种子和分类器读出后，仍获得较高平均召回，包括候选损失差异更明显的情形。二阶矩匹配与 A-optimal 筛选具有竞争力。这些发现支持利用目标感知特征几何，缩小需要接受监督评价的源组合范围。

## 中文关键词

数据选择、数据池预筛选、冻结表示、最优实验设计、分布偏移、实证评价

英文摘要词数（按空白分词）：189。
