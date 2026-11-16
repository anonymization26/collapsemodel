# 论文修订建议与证据缺口

核对版本为 Paper07 的 `rebuttal` 分支本地提交 `8e377c1`。下列英文是供审阅的替换稿。当前 TeX 和匿名 PDF 均保留原样。

## 主张与现有证据

**给定一组固定的训练数据组合，可以利用组合之间共享的数据池，把有限的拟合预算分配给更值得试跑的候选。** 以 DomainNet 为例，每个目标域有 15 个源池，从中选 3 个共有 455 个等样本成本的候选。两个组合若共享两个池，已观测验证损失之差在加性近似下提供了替换第三个池的信息。逐次查询先试 15 个线性独立的组合，再根据已观测分数调整后 10 次试跑，最后从已试候选中选出验证损失最低的模型。分两批版本在前 15 次完成后一次选出剩余 10 个候选，便于并行。

理论上，若整个候选族的验证损失精确可加且观测精确，15 个满秩查询可以确定全部验证损失。实际流程把加性当作预测近似。前 15 次单独作为选型预算时，四种子结果按域平均有 4/6 个域比全量参考差超过 +0.005。论文应把 25 次表述为当前实验有证据支持的预算。

| 策略 | 拟合次数 | 四投影种子归档回放 Brier ↓ | 原种子实测单线程 | 原种子实测四线程 |
|---|---:|---:|---:|---:|
| 逐次查询 | 25 | 0.864174 | 1.384 秒 | 0.793 秒 |
| 分两批查询 | 25 | 0.864428 | 1.390 秒 | 0.453 秒 |
| 固定二阶矩 MMD | 25 | 0.865156 | 待测 | 待测 |
| 固定 Target A-opt | 25 | 0.865851 | 待测 | 待测 |
| 随机 | 25 | 0.876178 | 待测 | 待测 |
| 固定二阶矩 MMD | 50 | 0.864365 | 2.666 秒 | 0.824 秒 |
| 全量验证 | 455 | 0.864118 | 25.702 秒 | 7.554 秒 |

Brier 衡量预测概率的误差，越低越好。四个额外投影种子使用同一批数据和候选池，回放只读取试过组合的验证损失，确认赢家后才审计测试分数。逐次 25 相对 MMD 50 和全量验证的平均差分别为 −0.000191 和 +0.000056，两个参照下均为 0/24 个种子与域平均超过 +0.005。分两批 25 比逐次 25 平均高 0.000254，出现 1/48 个编码器、种子与域单元高于 +0.005 的情况。

服务器实测在原投影种子上重新拟合，覆盖六域双编码器，从驻留投影特征计到选出验证赢家。单线程逐次 25 相对 MMD 50 的配对中位加速为 1.92 倍。四线程时逐次为 1.04 倍，分两批为 1.86 倍。原种子实测与四种子回放各自有独立口径。旧稿静态筛选的原图成本为分项测量后相加，新查询策略的原图全流程耗时仍待测。

岭回归属于补充机制证据。固定旧赢家时，目标验证集温度校准使 Brier 从 0.986695 降到 0.743338，准确率保持 36.00%。用分开的验证子集拟合温度和重新选型时，校准 A-opt 50 与全量 455 的平均 Brier 为 0.725347 和 0.725498，耗时为 1.114 和 10.047 秒。该计时从驻留特征和预计算排名开始，包含温度搜索。A-opt 去逆加权消融用于解释该分数的作用，Wasserstein 对照用于比较同一固定候选族上的筛选方式。二者保持辅助结果的定位。

来源为[四种子回放](runs/domainnet-pool-query-multiseed/Results.md)、[分批回放](runs/domainnet-pool-query-batch/Results.md)、[服务器实测](runs/domainnet-pool-query-batch/server/Logistic_Extended_Results.md)、[岭回归校准](runs/domainnet-ridge-temperature/Results.md)和[校准计时](runs/domainnet-pool-query-batch/server/Results.md)。

## 哪些已改，哪些建议再改

| 位置 | 当前版本 | 拟调整 |
|---|---|---|
| 摘要与引言 | 已把主贡献从静态几何筛选转向逐次查询，摘要已有 15 次的理论条件和 25 对 50 的回放数据。摘要和引言仍主要介绍原有静态筛选的计时。 | 增补一句新策略的服务器实测及其缓存口径。引言用两个组合共享两个池的例子解释验证反馈，明确一次查询包含拟合和验证。 |
| §3.2 与附录 F | 已给加性模型、满秩初始化、逐次更新、探索项和证明。 | 在主方法末尾加一小段说明如何从逐次更新得到分两批版本。保留 15 次识别命题的假设和实际选型规则。 |
| §5.2 与附录 H | 表 1 已报告四种子 25 对 50，正文与附录已有两档线程的实测。已有的同预算 MMD 25、A-opt 25、随机 25 未出现在表 1。 | 表 1 保留逐种子主结果，正文加一句同预算结果。把分批的单元级退化放在其耗时收益旁边，避免只读域平均。 |
| §5.3 与附录 E、G、H | 已纳入二维反例、校准、逆加权和传输对照。原静态筛选的原图成本表仍在正文。 | 分别注明三种成本口径。新逻辑回归从驻留特征开始并计入 MMD 排名，校准岭回归还要求排名已准备好，原图分项只来自原静态实验。正文维持 10 页，细表放附录。 |

## 建议替换的英文文字

### 摘要候选稿

> Selecting among fixed combinations of source-data pools can require fitting and validating hundreds of candidate models. We propose sequential pool-only querying, which uses the overlap among candidates to estimate additive pool contributions from observed validation losses and decide which candidate to fit next. For all triples from 15 pools, 15 linearly independent queries identify the full validation-loss surface under exact additivity and exact observations. In archived replays on four additional DomainNet projections, sequential querying with 25 fits yields mean logistic test Brier of 0.864174, compared with 0.864365 for fixed second-moment matching at 50 fits and 0.864118 for full validation at 455 fits. In a separate original-projection server study with resident features and four fitting threads, a two-batch 25-fit variant takes 0.453 s to select a model, compared with 0.824 s for fixed second-moment matching at 50 fits. We also characterize when target-aware geometry, probability calibration, and feature preparation affect the quality and cost of static screening.

### §3.2 直观例子与分批实现

> Consider two candidate triples that share pools A and B and differ in their third pool. Under the additive approximation, the difference in their validation losses informs the relative contributions of the two alternatives. We use this overlap throughout the candidate family. In the two-batch variant, we fit the contribution model after the first 15 queries, rank the unqueried candidates once, and dispatch the next ten fits together. Both variants select the evaluated candidate with the lowest observed validation Brier.

### §5.2 同预算对照与证据范围

> At the same budget of 25 fits, the four-projection replay gives mean test Brier of 0.864174 for sequential pool-only querying, 0.865156 for fixed MMD, 0.865851 for fixed Target A-opt, and 0.876178 for random ordering. With the same 25-fit budget, two-batch querying increases mean Brier by 0.000254 relative to sequential querying, with one of 48 encoder–projection–domain units exceeding +0.005. Separate server runs at the original projection seed measure wall time from resident projected features through validation selection. At four fitting threads the median paired speedups over fixed MMD 50 are 1.04× for sequential querying and 1.86× for two-batch querying.

## 还缺哪些实验数据

| 优先级 | 数据缺口 | 当前有的证据与建议的最小补充 |
|---|---|---|
| 1 | 逐次更新的收益来自哪里 | 同预算静态排序与随机已有。两批版本已和逐次版本共享前 15 次，能比较“更新一次”和“继续更新”。进一步共用前 15 次后，对比后 10 次随机抽取、按固定 MMD 顺序补齐，以及去掉探索项只用预计损失挑选。各策略均从全部 25 个已评估候选中选赢家。可用冻结归档回放，无需重训。报告逐单元 Brier、全量参考差值和赢家一致率。 |
| 2 | 独立目标域验证 | 四个投影种子共享 DomainNet 数据。Office-Home 和 BloodMNIST 已有静态筛选补充，尚无按相同预算实测新查询策略的质量与成本。现有 Optdigits 方向仅完成缓存预检。先冻结候选、测试划分、25/50/全量预算与 +0.005 阈值，再在独立目标上验证。 |
| 3 | 岭回归的正则化与特征维度 | 已有验证集校准与耗时，实测使用 128 维和固定 λ=1。优先复用缓存，在既有的两半验证划分上比较固定 λ=1 与验证选 λ∈{0.1, 1, 10}，记录每个候选三次拟合的真实成本。64/96/128 维嵌套压缩能检查低维敏感性，超过 128 维要有投影前特征。 |
| 4 | 新策略从原图起算的耗时 | §5.3 有旧静态筛选的分项成本，附录 H 有新策略的缓存实测。若主张扩展到首次处理新目标域，应在同一机器和设置下测准备特征、排名或查询、拟合和选择的总时间，单列缓存加载与提取。 |
| 后续 | 候选规模变化与预算规则 | 构造的 4,060 候选检验测了精确 A-opt 排名扩展。留一域回放中的 MMD 停止规则平均只省 2.08 次。新策略尚缺更大候选族上的质量–成本曲线和用验证反馈决定何时停止的独立检验。各候选复用目标验证集。若要主张能减少目标标签，还需单独改变验证集大小。 |

可同时检查加性近似的预测能力。查询路径冻结后，使用未查询候选的归档验证损失计算预测误差和排序质量，比较第 15 次与第 25 次的估计。该诊断可以检验池间交互是否影响预测，所用分数继续与选型时可见的信息隔离。

若这轮只补一项来支持新方法的机制，先做共用前 15 次的对照。如果审稿回复优先回应 HauM 对校准结论的稳健性，再做缓存上的 λ 选择。现有结果已足以陈述 DomainNet 的拟合次数与缓存耗时，跨数据集和原图起算的结论仍需上述实测。

## 定稿前的复现核对

当前工作目录的分批 `Protocol.md` 把后续计时协议加到了冻结文件内，与原 `freeze.sha256` 不符。公开发布用快照已恢复四种子回放的原冻结协议，并将服务器计时说明放到单独的 `TimingProtocol.md`，三项查询回放均通过。正式定稿时可同步这一拆分，沿用已有冻结轨迹与数值。
