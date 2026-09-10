# 论文整体改造 Checklist

## 总体目标

将论文从“具有普适误差界的闭式理论”调整为：

> 精确的局部 Gram 谱几何分析 + 理论启发的低维摘要预测器 + 冻结表示数据池的预筛选与冗余诊断。

全文必须严格区分：

- [ ] 无条件精确结论。
- [ ] 强假设下精确结论。
- [ ] 一般数据上的经验近似。
- [ ] 尚未验证的未来应用。

状态说明：`[x]` 已完成；`[ ]` 未完成；标题后的“进行中”表示已修改一部分，但尚未达到验收标准。

## 第一阶段：重写理论主线

### 1. 增加一般精确起点

- [x] 给出无条件恒等式 $G_{A\cup B}=H_A^\top H_A+H_B^\top H_B$。
- [x] 给出换基表达 $Q_A^\top G_{A\cup B}Q_A=\widetilde\Sigma_A^2+C\Sigma_B^2C^\top$。
- [x] 明确一般困难来自完整谱形状、谱权重与方向对应、稠密交叉耦合矩阵 $C$。
- [x] 明确平均 principal-angle alignment 不是充分统计量。
- [x] 在 `paper/main.tex` 中增加 Proposition `prop:gram_general` 作为无条件理论起点。

### 2. 修正 Gram-block Lemma

- [x] 将共享全局 $\gamma$ 改为逐方向 $\gamma_i=\sigma_{B,i}/\sigma_{A,i}$。
- [x] 使用逐方向 $\alpha_i=\cos^2\theta_i$。
- [x] 在 pairing/decoupling 条件下给出精确 $2\times2$ 块特征值。
- [x] 说明该结论不要求全局 proportional spectra。
- [x] 在正文和附录中给出证明。
- [ ] 再检查 mode-wise pairing、orthogonal decoupling 和未配对方向的形式化定义是否足够独立、完整。

### 3. 单独提出比例谱闭式命题

- [x] 增加 proportional-spectrum、uniform-alignment 条件下的窄命题。
- [x] 给出 $r_{\mathrm{merged}}=r_A\exp(H_b(q))$ 且 $r_A=r_B$。
- [x] 明确该命题不能推出允许 $r_A\neq r_B$ 的一般预测公式。

### 4. 重命名实用公式

- [x] 将实用公式命名为 **Collapse Summary Predictor**。
- [x] 将其定位为对 $\{\gamma_i,\alpha_i\}$ 分布进行全局压缩后的结构化近似。
- [x] 明确该公式不是一般恒等式，也不是普适有界估计器。

### 5. 撤回错误结论

- [x] 从论文中删除原 Theorem 1。
- [x] 删除原 Equation (9) 的普适误差界。
- [x] 删除 multiplicative guarantee。
- [x] 删除 mathematical correctness across general spectra。
- [x] 删除依赖错误定理的 sample-size sensitivity 理论解释。
- [x] 删除普适 safety certificate 表述。
- [ ] 全文再次搜索 `Theorem 1`、`guarantee`、`certificate` 和旧公式编号，确认无派生残留。

## 第二阶段：修正机制与阈值表述

### 6. 重写 superadditivity threshold（已完成）

- [x] 将一般数据阈值降级为 model-predicted diagnostic threshold。
- [x] 将严格结论限定在 proportional-spectrum、uniform-alignment 理想模型中。
- [x] 统一为相对高核范数来源 $r_{\mathrm{dom}}$ 的增益，并加入 $10^{-10}$ 相对数值容差。
- [x] 将 Collapse-4S 公式、超加性判定和数值秩截断集中到 `code/metrics/collapse_core.py`。
- [x] 按统一口径重算 E1a，并提交 280 行原始结果与机器可读摘要。
- [ ] 审计摘要、图注、正文和附录中的 `if and only if`、`guarantee`、`exact threshold`，逐项补充适用假设。

### 7. 将 collapse 改为局部谱机制（进行中）

- [x] 保留 directional collapse。
- [x] 保留 energy collapse。
- [x] 保留 balanced complementarity / superadditive regime。
- [ ] 增加 heterogeneous/mixed collapse。
- [ ] 增加 cross-coupling collapse。
- [ ] 明确真实数据的不同方向可同时处于不同 regime，不能总被划入唯一全局区域。

### 8. 增加四标量不充分反例

- [x] 构造具有相同 $r_A,r_B,\gamma,\alpha$ 但 merged effective rank 不同的显式反例。
- [x] 在正文中增加 Proposition `prop:insufficient`。
- [x] 给出反例数值：两组 merged effective rank 分别约为 2.958 和 3.661。
- [x] 在附录和 `theory.md` 中同步反例及解释。

## 第三阶段：强化经验验证

### 9. 重命名实验章节

- [x] 将实验章节改为 **Evaluation of the Summary Predictor**。
- [x] 不再声称实验“证明理论正确”。
- [ ] 将实验明确拆成：理想假设一致性、偏离假设后的退化、真实数据排序与校准、冗余检测与预筛选价值。

### 10. 增加依赖性稳健分析

- [ ] dataset-level cluster bootstrap。
- [ ] leave-one-dataset-out（LODO）。
- [ ] leave-one-domain-out。
- [ ] 按 encoder family 分别报告。
- [ ] 报告 Pearson、Spearman、校准 $R^2$ 的范围和最差值。
- [ ] 用上述结果替换 pair-level bootstrap 的泛化主张。

### 11. 测量模型失效指标

- [ ] 计算加权 $\operatorname{Var}_w(\gamma_i)$。
- [ ] 计算加权 $\operatorname{Var}_w(\alpha_i)$。
- [ ] 计算两侧归一化谱距离。
- [ ] 定义并计算 pairing residual 或 cross-coupling residual $\|E\|$。
- [ ] 分析失效指标与预测误差的相关性。
- [ ] 将这些指标定位为低置信度诊断，而不是新误差界。

### 12. 重新分析 alignment 增量

- [ ] 按低、中、高 alignment 分层比较 Collapse 与 $r_A+r_B$。
- [ ] 按平衡/失衡 energy ratio 分层。
- [ ] 区分 clean pool 与自然冗余 pool。
- [ ] 报告 selection disagreement、retention regret 和 downstream LP 差异。
- [ ] 明确 clean low-alignment pools 上两者基本等价。
- [ ] 明确 alignment 的价值主要体现在高重合或冗余候选。
- [ ] 将 clone attack 定位为压力测试，而非自然池平均收益证据。

### 13. 增加更强的摘要模型基线

- [ ] 四标量 Collapse。
- [ ] 谱加权 alignment。
- [ ] 4--8 bucket spectral summary。
- [ ] low-rank sketch。
- [ ] full merged SVD 上界。
- [ ] 绘制“通信量/存储量—预测精度—选择质量”曲线。

## 第四阶段：收缩并加强应用定位

### 14. 改名为冻结表示预筛选（进行中）

- [x] 正文已使用 frozen-encoder / frozen representation 语境。
- [ ] 全文统一应用名称为 **label-free frozen-representation pool pre-screening**。
- [ ] 明确 pipeline：候选数据源 → 冻结编码器摘要 → Collapse 预筛选 → 缩小候选池 → 实际训练或适配。
- [ ] 删除或收缩预训练收益、continual learning 收益和端到端数据整理收益主张。

### 15. 澄清 downstream evaluation

- [ ] 明确 encoder 始终冻结。
- [ ] 说明 selected datasets 如何决定 projection。
- [ ] 说明 linear probe 的训练数据。
- [ ] 说明 test target。
- [ ] 说明被选池是否直接参与训练。
- [ ] 将现有 LP 定位为表示空间验证，而不是端到端训练验证。

### 16. 可选的轻量数据消费实验

- [ ] 决定是否增加 linear adapter 或 LoRA 实验。
- [ ] 让 selected pools 实际参与训练。
- [ ] 固定样本量、步数和超参数。
- [ ] 比较 Collapse、$r_A+r_B$、Centroid-FF 和 Random。
- [ ] 至少运行 3 seeds 和多个 targets。

## 第五阶段：全文声明同步

### 17. 重写摘要（进行中）

- [x] 删除普适 correctness guarantee 和错误误差界。
- [x] 摘要已将公式定位为 structured summary predictor。
- [ ] 首句改为 frozen-representation pool pre-screening 问题。
- [ ] 按“一般 Gram 恒等式 → paired block → summary predictor → 稳健经验结果 → 自然池有限增益 → 高冗余优势 → 应用边界”重排。
- [ ] 删除或改写 closed-form bridge。
- [ ] 在完成 cluster bootstrap/LODO 后更新经验数字。

### 18. 重写 Contributions（进行中）

- [x] 理论贡献已改为一般 Gram 表达、paired-direction 分解和比例谱命题。
- [x] 已写明四标量不充分。
- [ ] 将贡献最终收敛为四项：问题、精确局部理论、summary predictor 与失效诊断、依赖性稳健验证与压缩实验。

### 19. 修正图 1（进行中）

- [x] 图注已区分 exact mode-wise structure 与 approximate predictor。
- [ ] 图中明确标出 Exact operator identity。
- [ ] 图中明确标出 Exact paired blocks。
- [ ] 图中明确标出 Approximate global summaries。
- [ ] 增加 Confidence/fallback。
- [ ] 将 Greedy compression 改为 Greedy pre-screening。
- [ ] 删除暗示三通道共同导出精确预测的视觉表达。

### 20. 重写 Conclusion 与 Limitations（进行中）

- [x] Conclusion 已删除错误普适定理主张。
- [x] Limitations 已写明四标量不充分和 paired-direction 假设边界。
- [ ] Conclusion 仅保留精确局部谱机制、经验摘要预测、冻结表示预筛选、高 alignment 冗余诊断及通信/计算优势。
- [ ] Limitations 明确当前没有一般误差界。
- [ ] Limitations 明确 clean natural pools 上相对简单基线的差异有限。
- [ ] Limitations 明确 calibration 依赖 encoder。
- [ ] Limitations 明确尚未证明端到端训练收益。
- [ ] Limitations 明确稠密 cross-coupling 会破坏逐方向模型。

## 第六阶段：附录与代码同步

### 21. 重构附录（进行中）

- [x] 删除 Theorem 1 proof。
- [x] 删除 Theorem 1 empirical verification。
- [x] 增加一般 Gram 换基证明。
- [x] 增加 paired-direction 局部分解证明。
- [x] 增加 proportional-spectrum 命题证明。
- [x] 增加四标量不充分反例。
- [ ] 增加 cluster bootstrap/LODO 方法。
- [ ] 增加 alignment 分层分析。
- [ ] 增加 failure-diagnostic 指标定义和结果。

### 22. 更新代码与复现说明

- [ ] 新增 `cluster_bootstrap.py`。
- [ ] 新增 `leave_one_dataset_out.py`。
- [ ] 新增 `alignment_stratified_ablation.py`。
- [ ] 新增 `summary_fidelity.py`。
- [ ] 整理或新增 `counterexamples.py`。
- [ ] 更新 README。
- [ ] 更新结果表。
- [ ] 更新匿名仓库复现命令。

## 推荐执行顺序

- [x] 1. 完成严格数学改写和反例。
- [x] 2. 删除所有错误定理及其派生主张。
- [ ] 3. 跑 cluster bootstrap、LODO 和 alignment 分层分析。
- [ ] 4. 决定是否实现 bucket/sketch predictor。
- [ ] 5. 决定是否补 lightweight adaptation。
- [ ] 6. 重写摘要、引言、贡献、结论和限制。
- [ ] 7. 更新图 1、附录、代码与复现文档。
- [ ] 8. 全文搜索 `exact`、`guarantee`、`certificate`、`closed-form`、`theorem` 并逐项审计。
- [ ] 9. 编译并检查主文页数、引用和匿名性。

## 完成标准

- [ ] 每个“精确”主张都明确给出适用假设和证明位置。
- [ ] 每个一般数据主张都有相应实验支持。
- [ ] 每个端到端应用主张都不超过实际验证范围。
- [ ] 四标量 summary predictor 的能力边界和失效条件清晰可见。
- [ ] 自然池上相对简单基线的有限增益被如实报告。
- [ ] 全文、附录、代码、README、匿名仓库和中文版本保持同步。
