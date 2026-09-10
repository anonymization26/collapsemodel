# Target-Conditioned Data Selection

## 从谱多样性预测转向目标风险最小化

> 文档状态：研究提案，尚未构成已证明或已验证的论文结论。
>
> 本文档描述一条新的理论主线。现有 Collapse-4S 只保留为局部谱机制诊断，
> Collapse-Gram-$L$ 只保留为可组合 PSD 压缩后端。新的主选择目标不再是合并有效秩，
> 而是目标分布上的预测风险或参数不确定性。

## 1. 为什么必须改变目标

现有理论能够严格处理以下问题：

1. 固定预处理下，多来源 Gram 算子可以相加；
2. 完整 Gram 可以精确恢复合并谱；
3. rank-$L$ PSD sketch 可以给出合并有效秩的确定性误差区间；
4. 四摘要 Collapse-4S 可以解释比例谱、成对方向模型中的局部行为。

但最终希望解决的是：

> 在有限采集、标注、存储或训练预算下，应该选择哪些数据集，以及每个数据集中应该选择哪些样本，
> 才能最大程度改善一个已声明目标任务或目标任务族？

合并有效秩并不直接回答这个问题。它会奖励目标任务完全不使用的新方向，也无法识别标签翻转、
条件分布偏移、标签噪声和训练算法失配。现有无标签不可区分反例进一步说明：若不增加连接任务与
几何的结构假设，仅凭无标签表示不可能对任意监督任务给出普适最优保证。

现有实验也表明继续收紧有效秩预测并不足够：

- rank-$L$ 有效秩区间覆盖正确，但在当前 $L\leq 50$ 扫描中没有产生可用的 greedy 决策证书；
- 自然池中的谱方法没有稳定达到预注册 shortlist recall 门槛；
- 当前来源监督 bottleneck adapter 族整体弱于 frozen identity，因此“小的 selector regret”
  不能解释为数据消费收益。

新的数学目标必须同时满足三点：

1. 分数直接对应一个明确的下游风险，而不是只对应表示多样性；
2. 数据集和样本都能表示成可加贡献；
3. 低秩近似、搜索和有限样本误差可以逐项进入同一条 regret 界。

## 2. 任务与权限定义

给定固定编码器

$$
\phi:\mathcal X\to\mathbb R^d,
\qquad h=\phi(x),
$$

以及候选数据池

$$
\mathcal D_1,\ldots,\mathcal D_M.
$$

若下游线性读出包含截距，则统一把常数 `1` 追加到 $h$ 中；理论和实现不得分别使用中心化特征与
含截距特征而不作说明。

每个候选池具有预先声明的成本 $c_i>0$，包括采集、传输、标注、存储或训练成本。选择器输出

$$
S\subseteq[M],
\qquad \sum_{i\in S}c_i\le B.
$$

选中数据集后，还可以在其并集中选择样本集合 $Q$，满足样本级成本约束。

必须区分三种信息权限：

| 权限 | 选择阶段可见信息 | 可支持的主张 |
|---|---|---|
| U0 | 候选无标签特征、目标无标签特征、固定编码器 | 在显式共享任务模型下控制预测方差；不能识别任意条件偏移 |
| U1 | U0 + 候选来源标签 | 可计算来源交叉矩、梯度或训练损失；仍不能直接观察目标任务偏差 |
| U2 | U1 + 独立的少量目标训练/验证标签 | 可直接计算任务条件化风险并执行严格的两阶段选择 |

不同权限下的方法必须分表比较。U1/U2 方法不能作为 U0 方法的同权限基线。

## 3. 核心正模型：目标条件化贝叶斯线性设计

### 3.1 候选信息矩阵

对候选池 $i$，固定其实际采样与加权协议，并定义

$$
G_i
=\sum_{x\in\mathcal D_i}a_x h_xh_x^\top
\succeq0.
$$

权重 $a_x$ 必须在查看选择结果前确定。不能对不同候选集合重新中心化或重新缩放。对于单个候选
样本 $x$，其贡献就是秩一矩阵

$$
G_x=a_xh_xh_x^\top.
$$

因此，数据集级和样本级选择共享同一种 PSD 加法结构。

### 3.2 明确的统计假设

首先研究冻结表示上的贝叶斯线性模型：

$$
y=h^\top\theta+\varepsilon,
\qquad
\theta\sim\mathcal N(0,P_0),
\qquad
\varepsilon\sim\mathcal N(0,\sigma^2).
$$

所有候选来源与目标任务共享同一个条件标签参数 $\theta$。来源可以具有不同的边际特征分布，
但暂不允许任意的 $P_i(y\mid h)$ 漂移。这个条件是正结果成立的核心，不应隐藏在实验描述中。

目标域只需提供无标签特征二阶矩

$$
C_T=\mathbb E_{x\sim T}[h_xh_x^\top].
$$

选中集合 $S$ 后，参数后验协方差为

$$
P_S
=\left(P_0^{-1}+\sigma^{-2}\sum_{i\in S}G_i\right)^{-1}.
$$

### 3.3 拟证明的核心风险恒等式

在上述模型下，使用后验均值预测目标样本时，贝叶斯平方预测风险为

$$
\boxed{
R_T(S)=\sigma^2+\operatorname{tr}(C_TP_S).}
$$

不可约噪声 $\sigma^2$ 与选择无关，所以主选择目标为 target-weighted A-optimal design：

$$
\boxed{
J_T(S)=\operatorname{tr}(C_TP_S),
\qquad
\min_{S:\,c(S)\le B}J_T(S).}
$$

这个目标只奖励降低目标域实际占用方向上的参数不确定性。与有效秩不同，在与目标子空间正交的方向
加入大量新谱质量不会改善 $J_T$。

### 3.4 样本级精确边际收益

对当前集合 $S$ 和一个权重为 $a$ 的候选样本 $h$，Sherman--Morrison 公式给出

$$
P_{S\cup\{h\}}
=P_S-
\frac{aP_Shh^\top P_S}{\sigma^2+ah^\top P_Sh}.
$$

因此，目标风险的精确边际下降为

$$
\boxed{
\Delta_T(h\mid S)
=\frac{a h^\top P_SC_TP_Sh}
{\sigma^2+a h^\top P_Sh}.}
$$

样本级选择可以直接按 $\Delta_T(h\mid S)/c_h$ 评分。分子衡量该样本是否覆盖目标相关且当前不确定
的方向，分母抑制已经被当前设计充分解释的冗余样本。

### 3.5 数据集级精确边际收益

若候选池 $i$ 的加权特征矩阵为 $H_i$，Woodbury 公式给出

$$
P_S-P_{S\cup\{i\}}
=P_SH_i^\top
\left(\sigma^2I+H_iP_SH_i^\top\right)^{-1}
H_iP_S.
$$

数据集级边际收益为

$$
\boxed{
\Delta_T(i\mid S)
=\operatorname{tr}\!\left[
C_TP_SH_i^\top
(\sigma^2I+H_iP_SH_i^\top)^{-1}
H_iP_S
\right].}
$$

实际实现应通过候选低秩因子或小核心计算该式，不能形成 $N_i\times N_i$ 稠密逆矩阵。

## 4. 优化结构

### 4.1 A-optimal 主目标

$J_T(S)$ 直接对应目标预测风险，是论文应优先优化和评估的目标。它关于已选信息矩阵单调下降，
但集合函数通常不具有标准次模性。因此不能直接套用普通 greedy 的 $(1-1/e)$ 保证。

可先求解连续松弛：

$$
\begin{aligned}
\min_{w\in[0,1]^M}\quad
&\operatorname{tr}\!\left[
C_T\left(P_0^{-1}+\sigma^{-2}\sum_iw_iG_i\right)^{-1}
\right],\\
\text{s.t.}\quad&\sum_i c_iw_i\le B.
\end{aligned}
$$

该目标关于 $w$ 是凸的。离散化阶段可研究 dependent rounding、swap rounding 或 leverage-aware
sampling，但在完成证明前只报告相对连续下界和小规模穷举 oracle 的 empirical regret。

### 4.2 D-optimal 信息增益辅助目标

贝叶斯参数信息增益为

$$
F_D(S)
=\frac12\log\det P_0
-\frac12\log\det P_S.
$$

在固定 PSD 候选增量和基数约束下，该目标单调且具有次模结构，可以支持标准 greedy 近似保证。
它直接对应参数互信息，但不等于 target-weighted 预测风险。因此它应作为：

- 有优化保证的辅助选择器；
- A-optimal 方法的强基线；
- 当 A-optimal 搜索成本过高时的可扩展候选生成器。

现有 DPP-subspace 使用的是数据池间相似度核。它与这里对可加信息矩阵计算的 D-optimal 目标不同，
两者必须在代码和实验命名中明确区分。

### 4.3 多目标和未知目标

若目标任务从已声明分布 $\Pi$ 中抽取，则平均目标风险仍可写成同一形式：

$$
\mathbb E_{T\sim\Pi}J_T(S)
=\operatorname{tr}(\overline C_\Pi P_S),
\qquad
\overline C_\Pi=\mathbb E_{T\sim\Pi}C_T.
$$

若希望控制最差目标，可研究

$$
\min_S\sup_{C\in\mathcal U_T}\operatorname{tr}(CP_S),
$$

其中 $\mathcal U_T$ 是由多个目标域或目标协方差置信集形成的集合。这条 distributionally robust
路线比完全 target-blind 的“通用多样性”主张更诚实，也更适合解释 LODO 失效。

## 5. 少量目标标签下的精确 ridge 风险

如果允许 U2 权限，可以避免未知的几何到 utility 单调函数。对具有共同输出空间的来源标签矩阵
$Y_i$，保存可加充分统计量

$$
G_i=H_i^\top H_i,
\qquad
B_i=H_i^\top Y_i.
$$

对来源集合 $S$，ridge 解为

$$
W_S=(G_S+\lambda I)^{-1}B_S,
\qquad
G_S=\sum_{i\in S}G_i,
\qquad
B_S=\sum_{i\in S}B_i.
$$

对独立的目标训练或验证划分，定义

$$
G_T=H_T^\top H_T,
\qquad
B_T=H_T^\top Y_T,
\qquad
Q_T=Y_T^\top Y_T.
$$

则目标平方损失可以不执行迭代训练而精确计算：

$$
\boxed{
\widehat R_T(S)
=\frac1{n_T}\left[
\operatorname{tr}(W_S^\top G_TW_S)
-2\operatorname{tr}(W_S^\top B_T)
+\operatorname{tr}(Q_T)
\right].}
$$

这给出一个可穷举、低噪声的任务效用 oracle，适合首先建立完整的数学与数据闭环。分类实验应把
平方损失或 Brier score 作为与定理直接对应的主指标，把准确率作为次指标。若要从平方风险推出
分类错误率，需要额外给出目标 margin 条件和低 margin 质量项，不能直接等同二者。

## 6. 数据集内样本 coreset

选中来源后，还需要减少实际使用的样本。对 frozen ridge 场景，建议同时研究两条样本路线：

1. 直接按第 3.4 节的 target-weighted A-optimal 边际收益选择；
2. 使用 ridge leverage score 或 sensitivity sampling 构造加权 coreset。

第二条路线的目标是证明，对参数域中的所有 $W$，加权样本子集 $Q$ 同时满足

$$
(1-\epsilon)L_D(W)
\le L_Q(W)
\le(1+\epsilon)L_D(W).
$$

从统一目标近似可以推出 coreset ERM 相对完整数据 ERM 的训练损失界；再结合共享条件模型、
目标协方差覆盖或 margin 条件，才能进一步推出目标风险界。

数据集级和样本级误差需要组合，而不是分别报告：若数据集选择产生设计 regret
$\epsilon_{\mathrm{group}}$，组内 coreset 产生谱近似误差 $\epsilon_{\mathrm{row}}$，最终定理应显式
包含两项以及它们通过逆矩阵扰动产生的放大系数。

## 7. rank-$L$ sketch 的新角色

现有 rank-$L$ sketch 可以继续使用，但它近似的对象改为目标风险，而不是有效秩。设

$$
G_S=\widetilde G_S+E_S,
\qquad
0\preceq E_S\preceq\tau_S I.
$$

令

$$
A_S=P_0^{-1}+\sigma^{-2}\widetilde G_S.
$$

由 Loewner 单调性，真实目标风险具有可计算区间

$$
\operatorname{tr}\!\left[
C_T(A_S+\sigma^{-2}\tau_SI)^{-1}
\right]
\le J_T(S)
\le
\operatorname{tr}(C_TA_S^{-1}).
$$

若 $\lambda_{\min}(A_S)\ge\lambda>0$，还可得到粗略宽度界

$$
0\le J_T(\widetilde G_S)-J_T(G_S)
\le
\frac{\sigma^{-2}\operatorname{tr}(C_T)}{\lambda^2}
\|E_S\|_{\mathrm{op}}.
$$

相较谱熵在零特征值附近的高敏感性，正则化逆矩阵目标是平滑的，并且截断产生单侧保守风险。
这有希望让自适应 rank 真正产生决策证书：在最小化问题中，若一个候选的风险区间上界严格低于
所有竞争候选的下界，则该轮选择与 Full-Gram 评分一致；否则只提高竞争候选的 sketch rank。

目标协方差本身也可以用 rank-$L_T$ sketch 表示。来源尾部误差、目标协方差误差和数值求逆误差
必须分别记录，不能合并成一个无法解释的经验校准常数。

## 8. 有限目标样本与总体风险

实际只观察

$$
\widehat C_T=\frac1{n_T}\sum_{j=1}^{n_T}h_jh_j^\top.
$$

在特征范数有界或次高斯条件下，可用矩阵 Bernstein 控制

$$
\|\widehat C_T-C_T\|_{\mathrm{op}}.
$$

由

$$
|\operatorname{tr}[(\widehat C_T-C_T)P_S]|
\le
\|\widehat C_T-C_T\|_{\mathrm{op}}\operatorname{tr}(P_S),
$$

可得到目标无标签样本量对风险排序稳定性的显式影响。实验必须报告：

- $n_T$ 增长时的风险估计误差；
- 候选排名翻转率；
- shortlist Jaccard 和 oracle recall；
- 经验 Bernstein 界或 bootstrap 区间的覆盖率与宽度。

## 9. 条件偏移与不可识别项

共享线性参数不是所有真实任务都满足。更一般地写成

$$
y=h^\top\theta+\delta_i(h)+\varepsilon,
$$

其中 $\delta_i$ 表示来源 $i$ 相对目标条件标签函数的偏差。仅用 U0 信息无法识别 $\delta_i$；
保持所有 $H_i$ 不变而交换标签即可反转最优来源。

因此正结果必须采用以下两种处理之一：

1. 将共享条件模型明确限定为论文定理和主要可控实验的适用范围；
2. 使用少量目标标签、来源标签交叉矩或目标梯度估计条件偏移，并把估计误差加入风险界。

标签置换、类别条件旋转和 source-specific teacher 应作为必做反例。方法在这些设置下失败不是实现
缺陷，而是检验理论边界是否与不可能性结论一致。

## 10. 非线性模型的后续扩展

只有 frozen linear/ridge 主线得到正结果后，才进入神经网络适配。在线性化点 $\theta_0$ 附近，

$$
f_\theta(x)
\approx f_{\theta_0}(x)
+J_{\theta_0}(x)(\theta-\theta_0).
$$

用 Jacobian、last-layer gradient、empirical Fisher 或 NTK 特征替代 $h$，即可复用信息矩阵、
A-optimal 风险和样本边际收益。新的总界必须加入：

- 参数移动半径；
- Hessian 或 Jacobian Lipschitz 常数；
- 一阶近似余项；
- 优化没有达到线性化解的误差。

如果实测线性化误差大于 selector 之间的效用差异，则不得把局部理论外推到 LoRA 或全参数微调。

## 11. 与现有 Collapse 工作的关系

| 组件 | 新定位 |
|---|---|
| Collapse-4S | 比例谱、成对方向条件下的局部机制诊断与消融 |
| Full-Gram effective rank | 表示多样性分析指标，不再作为任务效用主目标 |
| Collapse-Gram-$L$ | 压缩并组合候选 PSD 信息矩阵的计算后端 |
| 现有 DPP-subspace | 无标签几何覆盖基线 |
| Target A-optimal | 新的主要风险目标 |
| D-optimal information gain | 具有次模优化结构的辅助方法 |
| Ridge sufficient statistics | 少量目标标签下的精确任务风险路径 |
| Jacobian/Fisher sketch | 条件通过后的非线性扩展 |

不建议在验证前把新方法继续命名为 Collapse 的“加强版”。更准确的暂定描述是：

> Target-conditioned regularized Gram design for dataset and sample selection.

## 12. 预期可形成的贡献

若理论和实验均通过预注册门槛，可以形成以下贡献：

1. 一个在显式贝叶斯线性或 frozen-ridge 条件下，直接等于目标预测风险的数据选择目标；
2. 一个统一数据集 PSD 块与样本秩一贡献的层级选择框架；
3. 正则化风险目标下的低秩 sketch 区间、自适应 rank 和决策证书；
4. A-optimal 风险、D-optimal 信息增益、coreset 和任务感知 oracle 的公平比较；
5. 对共享条件、目标样本量和线性化假设失效区域的系统反例与负面边界。

最重要的成功标准不是得到更高的相关系数，而是：在独立目标上，使用显著少于完整候选的数据和
训练成本，仍能以高概率保留目标风险最优或近优集合，并且观测 regret 被声明的误差链实际覆盖。
