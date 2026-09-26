# ICLR 2027 投稿信息

由 `build_assets.py` 从唯一题目、关键词和摘要文本源生成，与论文输入保持一致。
状态：作者审阅稿，尚未提交 OpenReview；不表示已完成全部补实验或作者科学核验。

## Title

Feature Geometry for Data-Pool Screening: Utility, Stability, and Limits

## Abstract

Selecting source-data pools for a target task can require training and validating a predictor for many combinations of pools. We propose a geometry-guided framework that shortlists combinations using unlabeled source and target features from frozen encoders, then trains predictors on source labels and selects one by target validation. Its core screening method applies target-weighted A-optimal design to pooled Gram matrices and the target second moment, favoring source coverage that reduces uncertainty along target-relevant feature directions. A shared linear model explains the score, while an error decomposition separates candidate omission from validation error. On equal-sample-cost DomainNet combinations, an exploratory budget analysis with logistic regression finds lower mean test Brier from evaluating 50 A-opt candidates than 227 random candidates. In the original primary study, second-moment matching (MMD) matches A-opt on recall and final loss; it has higher mean recall at smaller budgets. Independent timing shows savings with cached features; adding measured feature-extraction costs preserves the logistic-regression saving but reverses it for ridge. These results support using target-aware geometry to reduce supervised data-pool evaluations, with savings depending on readout and feature preparation costs.

## Keywords

data selection, data-pool screening, frozen representations, optimal experimental design, distribution shift, empirical evaluation

## 中文题目

特征几何用于数据池预筛选：效用、稳定性与局限

## 中文摘要

为目标任务选择源数据池，可能需要针对许多数据池组合分别训练和验证预测器。本文提出一个几何驱动的预筛选框架，先利用冻结编码器提取的无标签源特征和目标特征筛出候选组合，再用源标签训练预测器，通过目标验证完成最终选择。框架的核心筛选方法将目标加权 A-optimal 设计用于合并 Gram 矩阵与目标二阶矩，优先保留能够降低目标相关特征方向上不确定性的源数据组合。共享线性模型解释这一评分，误差分解则区分候选遗漏与验证误差。在固定样本成本的 DomainNet 组合上，使用逻辑回归的探索性预算分析表明，评价50个 A-opt 候选的平均测试 Brier 低于评价227个随机候选。在原主实验中，二阶矩匹配（MMD）的召回率和最终损失与 A-opt 持平；在较小预算下，MMD 具有更高的平均召回率。独立计时显示，已有特征缓存时可以节省计算；加入实测特征提取成本后，逻辑回归仍有节省，岭回归的收益则反转。这些结果支持利用目标感知几何减少数据池的监督评价，实际节省则取决于读出器和特征准备成本。

## 中文关键词

数据选择、数据池预筛选、冻结表示、最优实验设计、分布偏移、实证评价

英文摘要词数（按空白分词）：180。
