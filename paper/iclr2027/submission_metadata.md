# ICLR 2027 投稿信息

由 `build_assets.py` 从唯一题目、关键词和摘要文本源生成，与论文输入保持一致。
状态：作者审阅稿，尚未提交 OpenReview；不表示已完成全部补实验或作者科学核验。

## Title

Feature Geometry for Data-Pool Screening: Utility, Stability, and Limits

## Abstract

Selecting source-data pools for a target task can require training and validating a predictor for many combinations of pools. Frozen features allow an earlier comparison, but retaining promising candidates is useful only if the final choice and the work saved justify screening. We evaluate a two-stage protocol that uses unlabeled source and target features to form a shortlist, then trains predictors on source labels and selects one using target validation. We compare target-weighted experimental design (A-opt) with second-moment matching (MMD) and target-blind diversity. The analysis explains the design score under a shared linear model and separates candidate omission from validation error. On equal-sample-cost DomainNet combinations, evaluating 50 A-opt or MMD candidates gives lower mean test Brier than evaluating 227 random candidates. Moment matching remains competitive across exploratory projection and readout changes. Domain-level matching also performs well after fixed-parameter repartitioning, while within-domain gains depend on construction and loss metric. Direct-selection gains on PACS and Office-Home are small, with limited improvement headroom. Independent timing shows savings with cached features. Adding measured feature-extraction costs preserves the logistic-regression saving but reverses it for ridge regression. These results assess screening through final selection quality and computation cost, rather than candidate retention alone.

## Keywords

data selection, data-pool screening, frozen representations, optimal experimental design, distribution shift, empirical evaluation

## 中文题目

特征几何用于数据池预筛选：效用、稳定性与局限

## 中文摘要

为目标任务选择源数据池，可能需要针对许多数据池组合分别训练和验证预测器。冻结特征允许更早比较候选，但保留好候选本身还不够：最终选择的质量与节省的工作必须能够说明预筛选的价值。本文评价一个两阶段协议，先用无标签源特征和目标特征形成短名单，再用源标签训练预测器，通过目标验证选出一个。我们比较目标加权实验设计（A-opt）、二阶矩匹配（MMD）和目标无关多样性。分析在共享线性模型下解释实验设计评分，并区分候选遗漏与验证误差。在固定样本成本的 DomainNet 组合上，评价50个 A-opt 或 MMD 候选的平均测试 Brier 低于评价227个随机候选。探索性地改变投影和读出器后，矩匹配仍有竞争力。固定参数重新分块后，域级匹配同样表现良好，而域内收益取决于构造和损失指标。PACS 和 Office-Home 的直接选择收益较小，可用改善空间也有限。独立计时表明，已有特征缓存时可以节省计算。加入实测特征提取成本后，逻辑回归仍有节省，岭回归的收益则发生反转。这些结果从最终选择质量和计算成本评价预筛选，而不只看候选保留率。

## 中文关键词

数据选择、数据池预筛选、冻结表示、最优实验设计、分布偏移、实证评价

英文摘要词数（按空白分词）：197。
