# Collapse Model 与两阶段数据池压缩：完整数学基座

> 状态：理论统一稿。本文只把已经能够严格成立的数学结果写成定理；需要真实数据支持的部分统一标记为经验假设。旧四标量 Collapse 公式仅保留为历史基线，不再承担一般正确性主张。

> 文档关系：本文件统一当前仓库的 `theory.md`、Paper07 的 rank-$L$ Gram 理论和 `IDEA.md` 的 two-stage 决策框架。三者若在表述强度上冲突，以本文件第 10--12 节的主张分级为准。

## 0. 逻辑总览

整套理论必须区分五类逻辑地位，不能把它们混写：

1. **无条件精确代数**：多个特征池纵向合并后，右 Gram 矩阵可加；完整 Gram、完整低秩因子和完整方向耦合矩阵给出完全相同的非零谱。
2. **无条件的信息极限**：边际谱、principal angles 乃至更低维的四标量摘要，都不能在一般情形下唯一决定合并谱。
3. **有条件的机制解释**：只有当两侧奇异方向满足 block-compatible 配对条件时，合并谱才分解为独立的 $2\times2$ 块，并产生逐方向的能量平衡与方向互补解释。
4. **无条件但可能保守的计算近似保证**：精确 top-$L$ PSD 截断可以组合成 rank-$L$ Gram sketch，并通过尾部核质量给出 log-effective-rank 误差界和排序证书。
5. **条件性的统计与任务桥接**：有限样本 Gram 到总体几何、有效秩到下游效用、有限验证集到真实效用都需要额外条件；这些条件必须由独立抽样、真实训练和统计实验检验。

因此，完整逻辑链是：

```text
统一冻结表示与来源权重
    -> Gram 可加性
    -> 完整耦合谱的精确表示
    -> 证明低维边际摘要存在信息极限
    -> 在 block-compatible 条件下解释局部互补机制
    -> 用 rank-L PSD sketch 建立可组合近似和失效证书
    -> 在非单调组合目标上执行穷举、Beam 或启发式搜索
    -> 区分经验 Gram 与总体 Gram 的抽样误差
    -> 只把谱方法用于无标签预筛选
    -> 用任务感知 Stage-2 处理标签、目标相关性与优化因素
    -> 用独立验证控制 shortlist 内的最终选择误差
```

三份现有思想在这条链中的位置是：

| 组成 | 数学身份 | 在最终方法中的作用 |
|---|---|---|
| `theory.md` 的完整 Gram 与耦合核心 | 无条件精确代数 | 定义真正需要近似的目标 |
| `theory.md` 的逐方向 Collapse 机制 | block-compatible 下的条件性定理 | 提供可解释性，不承担一般计算 |
| Paper07 的 rank-$L$ Gram sketch | 带 PSD 尾部条件的确定性近似 | 低通信 Stage-1 主算法 |
| `IDEA.md` 的 two-stage 流程 | 决策架构 | 用任务验证补足无标签不可识别性 |
| 本文件的总误差链 | 条件性端到端定理 | 明确抽样、搜索、sketch、代理和验证五项误差 |

## 1. 问题、预处理与记号

### 1.1 候选数据源和冻结表示

给定 $M$ 个候选数据源

$$
\mathcal D_1,\ldots,\mathcal D_M,
$$

所有样本必须经过同一个冻结编码器 $f$、同一层输出、同一中心化规则和同一归一化规则。第 $j$ 个来源包含 $n_j$ 个样本，得到

$$
H_j=
\begin{bmatrix}
z_{j,1}^{\top}\\
\vdots\\
z_{j,n_j}^{\top}
\end{bmatrix}
\in\mathbb R^{n_j\times d}.
$$

主设定采用逐样本 L2 归一化：

$$
\|z_{j,i}\|_2=1.
$$

中心化与否会改变 Gram 谱，因此不是无关实现细节，必须在实验协议中预先固定。

### 1.2 来源权重

给每个来源指定权重 $\beta_j>0$，定义

$$
X_j=\sqrt{\frac{\beta_j}{n_j}}H_j,
\qquad
G_j=X_j^{\top}X_j\succeq0.
$$

若各行已经 L2 归一化，则

$$
\operatorname{tr}(G_j)=\|X_j\|_F^2=\beta_j.
$$

这使两个常用目标具有明确含义：

- $\beta_j=1$：每个来源贡献相同总二阶质量，即来源等权；
- $\beta_j=n_j$：每个样本贡献相同质量，即样本等权。

两种权重定义对应不同问题。候选池样本数不等时，不能在观察结果后切换口径。

### 1.2.1 中心化约定与可组合性

Gram 可加性总是对“实际送入纵向堆叠的矩阵”成立，但三种中心化协议对应不同目标：

1. **不中心化或使用固定外部均值**：每个 $X_j$ 可独立构造，后续直接相加。
2. **各来源分别中心化**：各来源内部均值被删除，Gram 仍可相加，但来源均值之间的差异也被删除。
3. **对合并集合使用全局均值**：全局均值依赖所选集合 $S$，不能只把各来源独立中心化后的 Gram 相加。

第三种协议仍然可以由可加充分统计量精确计算。定义来源均值

$$
\mu_j=\frac1{n_j}\sum_{i=1}^{n_j}z_{j,i},
$$

集合的加权总质量和全局均值为

$$
W_S=\sum_{j\in S}\beta_j,
\qquad
\mu_S=\frac1{W_S}\sum_{j\in S}\beta_j\mu_j.
$$

则全局中心化后的加权 Gram 为

$$
\boxed{
G_S^{\mathrm{centered}}
=\sum_{j\in S}\frac{\beta_j}{n_j}H_j^{\top}H_j
-W_S\mu_S\mu_S^{\top}.}
$$

因此，每个来源只要额外提供 $(\beta_j,\mu_j)$，全局中心化目标仍可组合。证明由加权恒等式

$$
\sum_{j\in S}\frac{\beta_j}{n_j}
\sum_{i=1}^{n_j}(z_{j,i}-\mu_S)(z_{j,i}-\mu_S)^{\top}
=\sum_{j\in S}\frac{\beta_j}{n_j}H_j^{\top}H_j
-W_S\mu_S\mu_S^{\top}
$$

直接得到。

这个公式说明全局中心化可以由可加统计量精确恢复，但它不再是固定 PSD 矩阵
$G_j$ 的简单求和，因为秩一修正项依赖 $S$。若中心化前各行单位范数，则

$$
\operatorname{tr}(G_S^{\mathrm{centered}})
=W_S-W_S\|\mu_S\|_2^2,
$$

所以 $\beta_j$ 只表示中心化前质量，不能再解释为中心化后的固定来源质量。更重要的是，
第 5 节的 PSD 下近似及其单边尾界不能原样套用到这个集合依赖的秩一修正上。

逐行归一化与中心化不交换，所以论文必须固定顺序：

- 若全局中心化发生在最终逐行归一化之后，上述均值和二阶矩足以精确组合，但结果行不再单位范数；
- 若先按候选集合全局中心化、再逐行归一化，则每一行的分母也依赖 $S$，仅凭
  $(G_j,\mu_j)$ 不能恢复结果；
- 若中心化和归一化均在来源内完成且不依赖候选集合，则可以得到固定的 $X_j$，但它描述的是来源内几何，而不是合并集合的全局协方差。

本文后续主理论默认：$H_j$ 已经完成所有**与候选集合无关**的预处理，最终各行单位范数，且选择过程中不再重新做全局中心化。这样既保留
$\operatorname{tr}(G_j)=\beta_j$ 的权重解释，也满足第 5 节所需的固定 PSD 可加结构。若实验改用集合依赖的全局中心化，必须单独标注，并重新推导 sketch 误差。

### 1.3 合并矩阵和有效秩

对非空来源集合 $S\subseteq[M]$，定义纵向合并矩阵

$$
X_S=\operatorname{stack}_{j\in S}(X_j).
$$

全文的 $\log$ 均为自然对数。有效秩只对非零矩阵定义；主设定中
$\beta_j>0$ 且每个来源至少有一个单位范数表示，因此任何非空集合都满足 $X_S\ne0$。
若采用中心化后可能得到零矩阵的替代协议，必须单独处理该退化情形，不能在
$\operatorname{tr}\sqrt G=0$ 时继续做归一化。

设非零奇异值为

$$
s_1(X_S)\ge\cdots\ge s_r(X_S)>0,
$$

核质量、归一化幅度谱、谱熵和有效秩分别定义为

$$
\nu(X_S)=\|X_S\|_*=\sum_{i=1}^r s_i(X_S),
$$

$$
p_i(X_S)=\frac{s_i(X_S)}{\nu(X_S)},
$$

$$
\mathcal H(X_S)=-\sum_{i=1}^r p_i(X_S)\log p_i(X_S),
$$

其中约定 $0\log0=0$，以便后续把不同秩的谱补零到同一维数。

$$
R(X_S)=\exp\bigl(\mathcal H(X_S)\bigr).
$$

这里采用的是**奇异值幅度**的 Shannon 熵，而不是平方奇异值的熵。对 Gram 矩阵 $G\succeq0$，相应定义为

$$
q_i(G)=\frac{\sqrt{\lambda_i(G)}}{\operatorname{tr}\sqrt G},
\qquad
\mathcal H_G(G)=-\sum_iq_i(G)\log q_i(G),
$$

$$
R_G(G)=\exp\bigl(\mathcal H_G(G)\bigr).
$$

于是

$$
R(X_S)=R_G(X_S^{\top}X_S).
$$

有效秩满足

$$
1\le R(X_S)\le\operatorname{rank}(X_S),
$$

并且对任意 $c>0$ 有尺度不变性

$$
R(cX_S)=R(X_S).
$$

但各来源之间的相对缩放会改变合并谱，因此 $\beta_j$ 不能被整体尺度不变性消去。

### 1.3.1 有效秩的谱均匀性解释

若两个归一化幅度谱 $p,q$ 满足 $p$ majorize $q$，即 $p$ 比 $q$ 更集中，则 Shannon 熵的 Schur 凹性给出

$$
\mathcal H(p)\le\mathcal H(q),
\qquad
R(p)\le R(q).
$$

因此，有效秩严格衡量的是**归一化谱是否均匀**，而不是总能量大小。秩为 $r$ 时：

- 只有一个非零奇异值时 $R=1$；
- $r$ 个非零奇异值完全相等时 $R=r$；
- 向头部方向继续增加质量可能使谱更集中并降低 $R$。

最后一点正是后文“增加来源仍可能降低有效秩”的根本原因。

### 1.4 无标签谱选择目标

固定预算 $K$，定义

$$
F(S)=\log R_G(G_S),
\qquad
G_S=\sum_{j\in S}G_j,
$$

以及组合问题

$$
\max_{S\subseteq[M],\ |S|=K}F(S).
$$

记任一谱最优解为

$$
S_F^*\in\arg\max_{S\in\mathcal A_K}F(S),
\qquad
\mathcal A_K=\{S\subseteq[M]:|S|=K\}.
$$

使用 $\log R$ 只是为了直接处理熵；由于对数单调，它与最大化 $R$ 有相同最优集合。

这个目标只表示冻结特征空间中的谱多样性，不等同于下游准确率、损失、校准误差或端到端训练收益。

### 1.5 核心符号与误差来源

| 符号 | 含义 |
|---|---|
| $M,n_j,d$ | 来源数、来源样本数、表示维数 |
| $\beta_j$ | 预先固定的来源总权重 |
| $K$ | 必须选择的来源数 |
| $L_j$ | 来源 $j$ 的 Gram sketch 截断秩 |
| $G_j,G_S$ | 来源 Gram 与集合合并 Gram |
| $F(S)$ | log-effective-rank 谱代理目标 |
| $\mathcal C$ | Stage-1 实际枚举或保留的组合搜索族 |
| $\mathcal L$ | 交给 Stage-2 的候选组合 shortlist |
| $U_T(S)$ | 固定任务与训练协议下的真实期望效用 |
| $\chi$ | 经验 Gram 到总体 Gram 的谱分数误差 |
| $e_S$ | rank-$L$ sketch 对组合 $S$ 的谱分数误差 |
| $\eta_{\mathrm{search}}$ | 搜索族遗漏全局谱最优解造成的误差 |
| $\xi_T$ | 谱代理无法解释任务效用的结构性残差 |
| $\zeta_T$ | 有限训练种子与有限验证分数相对真实期望效用的误差 |

前四节主要回答“$G_S$ 的谱由什么决定”；第 5--6 节回答“如何近似计算并搜索”；
第 7--8 节才回答“这些谱结论在什么附加条件下能够影响任务选择”。

## 2. 数学层级 I：无条件精确谱代数

### 定理 1：Gram 可加性

对任意非空 $S$，有

$$
\boxed{G_S=X_S^{\top}X_S=\sum_{j\in S}X_j^{\top}X_j
=\sum_{j\in S}G_j.}
$$

#### 证明

$X_S$ 由各个 $X_j$ 沿样本维纵向堆叠，因此矩阵乘法中不存在不同来源之间的样本行交叉项：

$$
X_S^{\top}X_S
=
\begin{bmatrix}
X_{j_1}^{\top}&\cdots&X_{j_{|S|}}^{\top}
\end{bmatrix}
\begin{bmatrix}
X_{j_1}\\ \vdots\\ X_{j_{|S|}}
\end{bmatrix}
=\sum_{j\in S}X_j^{\top}X_j.
\qquad\square
$$

这个恒等式意味着：只要完整 $G_j$ 可用，计算任意集合的合并有效秩不需要重新访问原始样本。

### 定理 2：完整因子与完整耦合矩阵的等谱表示

令每个来源采用紧致 SVD：

$$
X_j=U_j\Sigma_jV_j^{\top},
\qquad
V_j\in\mathbb R^{d\times r_j}.
$$

定义完整右谱因子

$$
B_j=V_j\Sigma_j\in\mathbb R^{d\times r_j},
$$

以及集合拼接因子

$$
B_S=[B_j]_{j\in S}.
$$

则

$$
G_S=B_SB_S^{\top}.
$$

进一步定义完整耦合核心

$$
K_S=B_S^{\top}B_S.
$$

$G_S$ 与 $K_S$ 具有完全相同的非零特征值，包括代数重数。

对两个来源 $A,B$，耦合核心显式写为

$$
\boxed{
K_{AB}=
\begin{bmatrix}
\Sigma_A^2&\Sigma_AM\Sigma_B\\
\Sigma_BM^{\top}\Sigma_A&\Sigma_B^2
\end{bmatrix},
\qquad
M=V_A^{\top}V_B.}
$$

#### 证明

由 SVD 直接得到

$$
G_j=X_j^{\top}X_j=V_j\Sigma_j^2V_j^{\top}=B_jB_j^{\top}.
$$

因此

$$
G_S=\sum_{j\in S}B_jB_j^{\top}=B_SB_S^{\top}.
$$

任意矩阵 $B_S$ 的 $B_SB_S^{\top}$ 与 $B_S^{\top}B_S$ 的非零特征值都等于 $B_S$ 的非零奇异值平方，所以二者非零谱相同。双来源分块式由

$$
B_A^{\top}B_B
=\Sigma_AV_A^{\top}V_B\Sigma_B
=\Sigma_AM\Sigma_B
$$

立即得到。$\square$

### 2.1 该定理准确说明了什么

1. 完整 Gram、完整因子和完整耦合矩阵是同一精确代数的不同表示。
2. 左奇异向量 $U_j$ 不出现在合并右 Gram 谱中；旧“三通道”叙述中的独立 $U$ 通道不是决定合并谱所需的信息。这个结论只针对谱目标，不表示样本侧几何或 $U_j$ 对标签结构和下游任务无关。
3. 真正决定交互作用的是加权耦合块 $\Sigma_AV_A^{\top}V_B\Sigma_B$，而不只是单一 alignment 标量。
4. 这一定理解决的是“给定完整信息如何准确计算”，不是“如何用极少信息预测”。

### 2.2 精确表示不自动等于低成本表示

每个来源的典型 float32 存储量为

$$
\text{raw feature}:4n_jd,
\qquad
\text{full Gram}:4d^2,
\qquad
\text{full factor}:4dr_j\ \text{bytes}.
$$

当 $n_j<d$ 时，完整 Gram 甚至可能比原始特征更大。完整 Gram 也会泄露二阶统计结构，不能自动宣称隐私保护。

## 3. 数学层级 II：低维摘要的信息极限

### 3.1 Principal angles 丢失了什么

对两个右奇异子空间，交叉矩阵为

$$
M=V_A^{\top}V_B.
$$

$M$ 的奇异值是 principal-angle 余弦：

$$
\sigma_i(M)=\cos\theta_i.
$$

但是，完整耦合核心需要 $\Sigma_AM\Sigma_B$。只保留 $M$ 的奇异值会丢失 $M$ 的左右奇异向量，而这些向量决定：

- 哪个大奇异值方向与对侧哪个方向重合；
- 高 alignment 位于谱头部还是谱尾部；
- 一个方向是否同时耦合到多个对侧方向；
- 谱质量与几何重叠之间的相关结构。

因此，“子空间角度相同”不意味着“带权方向耦合相同”。

### 定理 3：边际谱与完整 principal angles 仍不足以确定合并有效秩

不存在一个对所有矩阵都正确的函数，只输入 $X_A,X_B$ 的边际奇异值和两个右奇异子空间的完整 principal-angle 集合，就能唯一确定 $R([X_A;X_B])$。

#### 证明：逐行归一化反例

取标准基 $e_1,e_2,e_3$，令

$$
H_A=[e_1,e_1,e_1,e_1,e_2]^{\top},
$$

$$
H_B^{(1)}=[e_1,e_1,e_1,e_1,e_3]^{\top},
$$

$$
H_B^{(2)}=[e_1,e_3,e_3,e_3,e_3]^{\top}.
$$

每一行的 L2 范数都为 1，三者的非零奇异值均为 $(2,1)$。$A$ 的支撑为 $\operatorname{span}(e_1,e_2)$，两个 $B$ 的支撑均为 $\operatorname{span}(e_1,e_3)$，所以两组 principal angles 都是

$$
\{0,\pi/2\}.
$$

忽略不影响有效秩的公共来源缩放，两个合并 Gram 分别为

$$
G_{A\cup B^{(1)}}=\operatorname{diag}(8,1,1),
$$

$$
G_{A\cup B^{(2)}}=\operatorname{diag}(5,1,4).
$$

相应有效秩为

$$
R_{A\cup B^{(1)}}\approx2.626012,
\qquad
R_{A\cup B^{(2)}}\approx2.849536.
$$

两组输入具有相同边际谱和相同 principal angles，却得到不同输出，故所述函数不存在。$\square$

### 推论 3.1：旧四标量不是充分统计量

四标量

$$
(R_A,R_B,\gamma,\bar\alpha)
$$

中，旧方法取全局核质量比

$$
\gamma=\frac{\nu(X_B)}{\nu(X_A)}
$$

以及前 $k$ 个 principal angles 的平均平方余弦

$$
\bar\alpha=\frac1k\sum_{i=1}^{k}\cos^2\theta_i.
$$

这里 $k\le\min(r_A,r_B)$ 是预先固定的保留方向数。

这组摘要比“完整边际谱加完整 principal angles”包含的信息更少，因此不可能成为一般合并有效秩的充分统计量。

旧闭式公式可以作为经验基线，但不能称为：

- 无条件精确公式；
- 充分统计量；
- 普适误差保证；
- 由平均 alignment 唯一决定的合并定理。

这个信息极限不是证明技术不够强，而是输入摘要本身不可识别。

## 4. 数学层级 III：block-compatible 条件下的精确局部机制

### 假设 A：block-compatible singular directions

经过两侧奇异方向的某种重排后，存在 $m$ 对匹配方向，使

$$
v_{A,i}^{\top}v_{B,j}=0,\qquad i\ne j,
$$

$$
v_{A,i}^{\top}v_{B,i}=c_i\in[0,1].
$$

所有未匹配方向还必须与对侧全部方向正交。定义

$$
\alpha_i=c_i^2.
$$

该条件比“principal angles 存在”强得多。一般真实耦合矩阵 $V_A^{\top}V_B$ 是稠密的，因此本节是条件性精确理论，不是默认适用于所有数据的分解。

### 4.1 假设偏离如何量化

令

$$
M=V_A^{\top}V_B,
\qquad
C=\Sigma_A M\Sigma_B.
$$

对一个预先规定的一对一匹配 $\pi$，令 $\mathcal P_\pi(C)$ 只保留匹配位置
$(i,\pi(i))$ 上的元素，其余元素置零，并定义

$$
\delta_{\mathrm{off}}(\pi)
=\frac{\|C-\mathcal P_\pi(C)\|_F}{\|C\|_F}.
$$

若 $C=0$，约定该比值为 0。再令

$$
\delta_{\mathrm{off}}^*
=\min_{\pi\ \text{为一对一部分匹配}}
\delta_{\mathrm{off}}(\pi).
$$

在固定 SVD 基下，该最小化等价于对边权
$|C_{ij}|^2$ 做最大权二分匹配，因为

$$
\|C-\mathcal P_\pi(C)\|_F^2
=\|C\|_F^2-\sum_{i:(i,\pi(i))\in\pi}|C_{i,\pi(i)}|^2.
$$

当所有所考虑奇异值严格为正时，$\delta_{\mathrm{off}}^*=0$ 等价于加权交叉耦合可以被重排成互不耦合的匹配块，也就是本节假设在这些方向上成立。

这个残差还给出 Gram 核的直接扰动量。定义

$$
K=
\begin{bmatrix}
\Sigma_A^2&C\\
C^{\top}&\Sigma_B^2
\end{bmatrix},
\qquad
K_\pi=
\begin{bmatrix}
\Sigma_A^2&\mathcal P_\pi(C)\\
\mathcal P_\pi(C)^{\top}&\Sigma_B^2
\end{bmatrix}.
$$

由于每个保留项满足
$|C_{ij}|\le\sigma_{A,i}\sigma_{B,j}$，$K_\pi$ 可重排为若干 PSD
$2\times2$ 块和未匹配的一维块，因此仍为 PSD。

由对称矩阵的 Hoffman--Wielandt 不等式，按降序排列特征值后有

$$
\boxed{
\|\lambda(K)-\lambda(K_\pi)\|_2
\le \|K-K_\pi\|_F
=\sqrt2\,\|C-\mathcal P_\pi(C)\|_F.}
$$

因此，$\delta_{\mathrm{off}}^*$ 是“局部块模型是否接近真实 Gram 谱”的可测诊断，而不是定理成立的替代条件。把 Gram 特征值再变成归一化奇异值熵时，仍需处理平方根在零附近的敏感性以及归一化误差，不能仅凭一个较小的平均 alignment 宣称有效秩误差较小。

若存在重复奇异值，$V_A,V_B$ 在对应奇异子空间内并不唯一，所以固定 SVD 基下的 $\delta_{\mathrm{off}}^*$ 不是内禀量。严格比较时，应再对**精确重根子空间内允许的正交旋转**取最小值；对近重根进行聚类则必须预先固定容差，并明确它只是数值诊断。

### 定理 4：逐方向 $2\times2$ 精确谱

设第 $i$ 对方向的奇异值幅度为 $a_i,b_i$。其耦合核心为

$$
K_i=
\begin{bmatrix}
a_i^2&a_ib_ic_i\\
a_ib_ic_i&b_i^2
\end{bmatrix}.
$$

两个 Gram 特征值为

$$
\boxed{
\lambda_{\pm,i}
=\frac{a_i^2+b_i^2\pm
\sqrt{(a_i^2-b_i^2)^2+4a_i^2b_i^2\alpha_i}}{2}.}
$$

对应合并奇异值为

$$
s_{\pm,i}=\sqrt{\lambda_{\pm,i}}.
$$

#### 证明

$K_i$ 的迹和行列式分别为

$$
\operatorname{tr}(K_i)=a_i^2+b_i^2,
$$

$$
\det(K_i)=a_i^2b_i^2(1-c_i^2)
=a_i^2b_i^2(1-\alpha_i).
$$

代入二阶特征方程

$$
\lambda^2-\operatorname{tr}(K_i)\lambda+\det(K_i)=0
$$

即可得到所述两根。block-compatible 条件保证不同 $i$ 的块互不耦合，所以所有块特征值与未匹配方向共同组成完整非零谱。$\square$

### 4.2 数值稳定形式

记

$$
\Delta_i
=\sqrt{(a_i^2-b_i^2)^2+4a_i^2b_i^2\alpha_i}.
$$

当 $\alpha_i\approx1$ 时，直接计算

$$
\lambda_{-,i}=\frac{\operatorname{tr}(K_i)-\Delta_i}{2}
$$

会发生灾难性消减。应先计算 $\lambda_{+,i}$，再使用

$$
\boxed{
\lambda_{-,i}
=\frac{a_i^2b_i^2(1-\alpha_i)}{\lambda_{+,i}}.}
$$

### 4.3 三种局部机制

1. **方向塌陷**：若 $\alpha_i=1$，则

   $$
   \lambda_{+,i}=a_i^2+b_i^2,
   \qquad
   \lambda_{-,i}=0.
   $$

   两侧质量合并到一个方向。

2. **方向互补**：若 $\alpha_i=0$，则

   $$
   \{\lambda_{+,i},\lambda_{-,i}\}
   =\{a_i^2,b_i^2\}.
   $$

   两个方向都被保留。

3. **能量塌陷**：即使 $\alpha_i=0$，当 $b_i/a_i\to0$ 或 $\infty$ 时，弱分支在归一化谱中的质量仍趋近于零。

所以局部互补不是只由 alignment 决定，而是方向不重合和两侧质量平衡的共同结果。

### 定理 5：局部互补参数与分支平衡

定义

$$
\eta_i=\frac{2a_ib_i}{a_i^2+b_i^2}\in[0,1],
$$

$$
z_i=\eta_i\sqrt{1-\alpha_i}\in[0,1],
$$

以及较大分支在该块核质量中的比例

$$
q_i=\frac{s_{+,i}}{s_{+,i}+s_{-,i}}\in[1/2,1].
$$

记二元熵为

$$
h_b(q)=-q\log q-(1-q)\log(1-q).
$$

则

$$
\boxed{
q_i=\frac12\left(1+
\sqrt{\frac{1-z_i}{1+z_i}}
\right).}
$$

二元熵 $h_b(q_i)$ 随 $z_i$ 单调增加，在 $z_i=0$ 时为 0，在 $z_i=1$ 时达到 $\log2$。
因此该二维块自身的有效秩为

$$
R_i=\exp(h_b(q_i))\in[1,2].
$$

#### 证明

记 $S_i=a_i^2+b_i^2$。由定理 4 的行列式可得

$$
s_{+,i}s_{-,i}
=a_ib_i\sqrt{1-\alpha_i}
=\frac{S_iz_i}{2}.
$$

又因为 $s_{+,i}^2+s_{-,i}^2=S_i$，所以

$$
\frac{(s_{+,i}-s_{-,i})^2}
{(s_{+,i}+s_{-,i})^2}
=\frac{1-z_i}{1+z_i}.
$$

而

$$
2q_i-1
=\frac{s_{+,i}-s_{-,i}}
{s_{+,i}+s_{-,i}},
$$

得到闭式表达。$q_i$ 随 $z_i$ 从 1 单调下降到 $1/2$，而 $h_b(q)$ 在 $[1/2,1]$ 上随 $q$ 下降而增加，故结论成立。$\square$

该定理精确刻画局部互补，但不能推出全局有效秩随 $z_i$ 单调增加，因为局部块在全局核质量中的权重也会变化。

### 定理 6：精确块熵链式分解

把未匹配方向写成 $(s_{+,i},s_{-,i})=(a_i,0)$ 或 $(b_i,0)$。对每个块定义

$$
t_i=s_{+,i}+s_{-,i},
\qquad
Z=\sum_i t_i,
$$

$$
w_i=\frac{t_i}{Z},
\qquad
q_i=\frac{s_{+,i}}{t_i}.
$$

则完整合并谱熵满足

$$
\boxed{
\mathcal H(X_{A\cup B})
=\mathcal H(w)+\sum_iw_i h_b(q_i).}
$$

因此

$$
\boxed{
e^{\mathcal H(w)}
\le R(X_{A\cup B})
\le2e^{\mathcal H(w)}.}
$$

#### 证明

第 $i$ 个块的两个全局概率正好为

$$
p_{+,i}=w_iq_i,
\qquad
p_{-,i}=w_i(1-q_i).
$$

代入 Shannon 熵并展开：

$$
-\sum_i\left[
w_iq_i\log(w_iq_i)
+w_i(1-q_i)\log(w_i(1-q_i))
\right]
$$

$$
=-\sum_iw_i\log w_i
+\sum_iw_i h_b(q_i).
$$

再由 $0\le h_b(q_i)\le\log2$ 得到有效秩上下界。$\square$

这里的 $\mathcal H(w)$ 和块内奖励可能依赖合法 matching；只有总熵是内禀不变量。因此，不能把二者分别包装成不依赖配对规则的全局指标。

### 推论 6.1：完全正交支撑

若

$$
V_A^{\top}V_B=0,
$$

则合并奇异值是两侧奇异值的并集。令

$$
\omega=\frac{\nu_A}{\nu_A+\nu_B},
$$

则

$$
\mathcal H(A\cup B)
=h_b(\omega)
+\omega\mathcal H(A)
+(1-\omega)\mathcal H(B),
$$

$$
\boxed{
R(A\cup B)
=e^{h_b(\omega)}R_A^{\omega}R_B^{1-\omega}.}
$$

### 推论 6.2：比例谱与统一 alignment

若两侧秩相同、所有非零方向均被匹配、block-compatible，并且对所有方向都有

$$
b_i=\gamma a_i,
\qquad
\alpha_i=\alpha,
$$

则

$$
s_{\pm,i}=c_{\pm}a_i,
$$

其中

$$
c_{\pm}
=\sqrt{
\frac{1+\gamma^2\pm
\sqrt{(1-\gamma^2)^2+4\gamma^2\alpha}}{2}
}.
$$

令

$$
q=\frac{c_+}{c_++c_-},
$$

则

$$
\boxed{
\mathcal H(A\cup B)=\mathcal H(A)+h_b(q),
\qquad
R(A\cup B)=R_Ae^{h_b(q)}.}
$$

比例谱意味着两侧归一化谱完全相同，所以必然有 $R_A=R_B$。因此，允许 $R_A\ne R_B$ 的旧四标量插值式不是该推论的精确推广。

## 5. 数学层级 IV：rank-$L$ Gram sketch 的无条件近似保证

block-compatible 在一般数据上不成立。要获得无需该条件的可组合近似，应直接近似每个 PSD Gram 算子，而不是继续压缩成平均 alignment。

### 5.1 精确 PSD 截断

对来源 $j$ 的紧致 SVD，取前 $L_j$ 个方向：

$$
B_j^{(L_j)}=V_{j,L_j}\Sigma_{j,L_j},
$$

$$
\widetilde G_j
=B_j^{(L_j)}B_j^{(L_j)\top}.
$$

残差为

$$
E_j=G_j-\widetilde G_j\succeq0.
$$

定义尾部核质量、尾部平方质量和尾秩：

$$
\tau_j=\operatorname{tr}\sqrt{E_j}
=\sum_{i>L_j}\sigma_{j,i},
$$

$$
\rho_j=\operatorname{tr}(E_j)
=\sum_{i>L_j}\sigma_{j,i}^2,
$$

$$
u_j=\operatorname{rank}(E_j).
$$

这些等式用于理论记号；实现只需提交可认证上界

$$
\overline\tau_j\ge\tau_j,
\qquad
\overline\rho_j\ge\rho_j,
\qquad
\overline u_j\ge u_j,
$$

并在后续公式中替换相应真值。主设定下

$$
\rho_j
=\operatorname{tr}(G_j)-\sum_{i\le L_j}\sigma_{j,i}^2
=\beta_j-\sum_{i\le L_j}\sigma_{j,i}^2,
$$

而且总可以取

$$
\overline u_j=\min(n_j,d)-L_j.
$$

所以第二个尾界只需 top-$L_j$ 奇异值和总平方质量，不要求计算完整尾部 SVD。若
$\tau_j$ 没有可靠上界，可以放弃第一个尾界，只使用 $(\overline\rho_j,\overline u_j)$。

集合 $S$ 的近似和残差为

$$
\widetilde G_S=\sum_{j\in S}\widetilde G_j,
\qquad
E_S=\sum_{j\in S}E_j,
$$

并可拼接低秩因子

$$
\widetilde B_S=[B_j^{(L_j)}]_{j\in S},
\qquad
\widetilde G_S=\widetilde B_S\widetilde B_S^{\top}.
$$

因此无需显式形成 $d\times d$ 矩阵。只需对小核心

$$
\widetilde K_S
=\widetilde B_S^{\top}\widetilde B_S
\in\mathbb R^{L_S\times L_S},
\qquad
L_S=\sum_{j\in S}L_j,
$$

做特征分解；它与 $\widetilde G_S$ 的非零谱完全相同。

所以

$$
G_S=\widetilde G_S+E_S,
\qquad E_S\succeq0.
$$

这正是该 sketch 能够递归组合的原因：中心节点只需累加低秩因子对应的 Gram 贡献，不需要把已选集合再次压成四标量。

### 引理 7：缺失核质量的两个上界

定义

$$
T_S^{(1)}=\sum_{j\in S}\tau_j.
$$

则

$$
\operatorname{tr}\sqrt{E_S}\le T_S^{(1)}.
$$

进一步令

$$
u_S=\min\left(d,\sum_{j\in S}u_j\right),
$$

则

$$
\operatorname{tr}\sqrt{E_S}
\le
T_S^{(2)}
:=\sqrt{u_S\sum_{j\in S}\rho_j}.
$$

因此可采用更紧的合法尾界

$$
\boxed{
T_S=\min\left(T_S^{(1)},T_S^{(2)}\right).}
$$

#### 证明

对 PSD 矩阵，平方根迹满足次可加性：

$$
\operatorname{tr}\sqrt{A+B}
\le\operatorname{tr}\sqrt A+\operatorname{tr}\sqrt B.
$$

一个直接证明是令
$C=[A^{1/2},B^{1/2}]$，则 $CC^{\top}=A+B$，从而

$$
\operatorname{tr}\sqrt{A+B}
=\|C\|_*
=\|[A^{1/2},0]+[0,B^{1/2}]\|_*
\le\operatorname{tr}\sqrt A+\operatorname{tr}\sqrt B,
$$

其中最后一步是核范数三角不等式。

递归应用即可得到第一个上界。对 $E_S$ 的非零特征值 $\mu_1,\ldots,\mu_u$ 使用 Cauchy--Schwarz：

$$
\operatorname{tr}\sqrt{E_S}
=\sum_{i=1}^u\sqrt{\mu_i}
\le\sqrt{u\sum_{i=1}^u\mu_i}
=\sqrt{\operatorname{rank}(E_S)\operatorname{tr}(E_S)}.
$$

再使用

$$
\operatorname{rank}(E_S)\le u_S,
\qquad
\operatorname{tr}(E_S)=\sum_{j\in S}\rho_j
$$

即可得到第二个上界。$\square$

### 定理 8：确定性的 log-effective-rank 误差界

假设 $\widetilde G_S\ne0$。定义

$$
\widetilde\nu_S=\operatorname{tr}\sqrt{\widetilde G_S},
$$

$$
\varepsilon_S
=\frac{T_S}{\widetilde\nu_S+T_S}.
$$

令 $D_S\ge2$ 是同时容纳 $G_S$ 与 $\widetilde G_S$ 非零谱的维数上界。总可以取 $D_S=d$；更紧时可取

$$
D_S=\min\left(d,\sum_{j\in S}\operatorname{rank}(G_j)\right).
$$

定义

$$
\Phi_D(\varepsilon)=
\begin{cases}
h_b(\varepsilon)+\varepsilon\log(D-1),
&0\le\varepsilon\le1-1/D,\\
\log D,&1-1/D<\varepsilon\le1.
\end{cases}
$$

则

$$
\boxed{
\left|
\log R_G(G_S)-\log R_G(\widetilde G_S)
\right|
\le\Phi_{D_S}(\varepsilon_S).}
$$

#### 证明

把 $\widetilde G_S$ 和 $G_S$ 的特征值按降序排列，并定义幅度谱

$$
a_i=\sqrt{\lambda_i(\widetilde G_S)},
\qquad
b_i=\sqrt{\lambda_i(G_S)}.
$$

由于

$$
G_S=\widetilde G_S+E_S\succeq\widetilde G_S,
$$

特征值单调性给出 $b_i\ge a_i$。令

$$
\delta_S=\sum_i(b_i-a_i)
=\operatorname{tr}\sqrt{G_S}
-\operatorname{tr}\sqrt{\widetilde G_S}.
$$

平方根迹次可加性和引理 7 给出

$$
0\le\delta_S
\le\operatorname{tr}\sqrt{E_S}
\le T_S.
$$

记真实和近似归一化幅度谱为

$$
p_i=\frac{b_i}{\widetilde\nu_S+\delta_S},
\qquad
\widetilde p_i=\frac{a_i}{\widetilde\nu_S}.
$$

若 $\delta_S>0$，令

$$
r_i=\frac{b_i-a_i}{\delta_S}.
$$

则 $r$ 是概率分布，并且

$$
p
=\frac{\widetilde\nu_S}
{\widetilde\nu_S+\delta_S}\widetilde p
+\frac{\delta_S}
{\widetilde\nu_S+\delta_S}r.
$$

因此总变差距离满足

$$
\operatorname{TV}(p,\widetilde p)
\le
\frac{\delta_S}
{\widetilde\nu_S+\delta_S}
\le
\frac{T_S}{\widetilde\nu_S+T_S}
=\varepsilon_S.
$$

当 $\delta_S=0$ 时该距离为零。最后对至多 $D_S$ 维的两个概率分布应用 Fannes--Audenaert 熵连续性界，得到

$$
|\mathcal H_G(G_S)-\mathcal H_G(\widetilde G_S)|
\le\Phi_{D_S}(\varepsilon_S).
$$

利用 $\log R_G=\mathcal H_G$ 即得结论。$\square$

若 $D_S=1$，两个非零谱都只有一个方向，有效秩恒为 1，误差直接为零。

### 推论 8.1：候选的确定性区间

对当前已选集合 $S$ 和候选 $j\notin S$，定义

$$
\widehat F_j
=\log R_G(\widetilde G_{S\cup\{j\}}),
$$

$$
e_j=\Phi_{D_{S\cup\{j\}}}
(\varepsilon_{S\cup\{j\}}).
$$

则真实分数必定位于

$$
\boxed{
F(S\cup\{j\})
\in[\widehat F_j-e_j,\widehat F_j+e_j].}
$$

注意：$e_j$ 必须累计**当前已选池和候选池的全部尾部**。只计算候选 $j$ 自身的尾部会产生错误证书。

### 推论 8.2：单步排序证书

若存在候选 $j^*$ 满足

$$
\boxed{
\widehat F_{j^*}-e_{j^*}
>
\max_{j\ne j^*}(\widehat F_j+e_j),}
$$

则 $j^*$ 必然是该轮 Full-Gram 分数唯一最高的候选。

#### 证明

对任意 $j\ne j^*$，由区间界有

$$
F(S\cup\{j^*\})
\ge\widehat F_{j^*}-e_{j^*}
>
\widehat F_j+e_j
\ge F(S\cup\{j\}).
\qquad\square
$$

这个证书只说明“近似计算复现了本轮 exact greedy 的选择”，不说明 greedy 的最终集合是组合全局最优。

### 推论 8.3：没有证书时的单步分数遗憾

若近似方法选择

$$
\widehat j=\arg\max_j\widehat F_j,
$$

而真实最优候选为 $j^{\star}$，则

$$
F(S\cup\{j^{\star}\})
-F(S\cup\{\widehat j\})
\le e_{j^{\star}}+e_{\widehat j}
\le2\max_je_j.
$$

这给出当前一步的分数损失上界，但由于 $F$ 没有已证明的单调次模结构，不能把每步上界直接相加并声称最终集合具有全局近似比。

### 推论 8.4：有限搜索族上的全局证书与谱遗憾

令 $\mathcal C\subseteq\mathcal A_K$ 是实际枚举并比较的有限组合族。对每个
$S\in\mathcal C$，假设已经得到

$$
F(S)\in[\widehat F(S)-e_S,\widehat F(S)+e_S].
$$

若

$$
\widehat S=\arg\max_{S\in\mathcal C}\widehat F(S),
\qquad
S_{\mathcal C}^*=\arg\max_{S\in\mathcal C}F(S),
$$

则

$$
\boxed{
F(S_{\mathcal C}^*)-F(\widehat S)
\le e_{S_{\mathcal C}^*}+e_{\widehat S}
\le2\max_{S\in\mathcal C}e_S.}
$$

证明与推论 8.3 相同：真实最优项先用上界替换，利用
$\widehat F(S_{\mathcal C}^*)\le\widehat F(\widehat S)$，再对所选项使用下界。

若某个 $S^\dagger\in\mathcal C$ 满足

$$
\boxed{
\widehat F(S^\dagger)-e_{S^\dagger}
>
\max_{S\in\mathcal C\setminus\{S^\dagger\}}
(\widehat F(S)+e_S),}
$$

则 $S^\dagger$ 是 $\mathcal C$ 内唯一的 Full-Gram 最优集合。如果
$\mathcal C=\mathcal A_K$，这就是大小为 $K$ 的全局谱最优证书；如果
$\mathcal C$ 只是 Beam Search 保留下来的节点，则结论只在该保留族内成立。

更一般地，令搜索遗漏为

$$
\eta_{\mathrm{search}}
=F(S_F^*)-\max_{S\in\mathcal C}F(S),
$$

则近似分数在 $\mathcal C$ 中选出的 $\widehat S$ 满足

$$
\boxed{
F(S_F^*)-F(\widehat S)
\le
\eta_{\mathrm{search}}+2\max_{S\in\mathcal C}e_S.}
$$

这条式子把后文桥接定理中的 $\eta$ 明确拆成了**组合搜索遗漏**与
**rank-$L$ 谱近似误差**。前一项必须通过穷举对照或搜索消融估计，不能由
Gram sketch 定理自动控制。

### 5.2 自适应精化算法的数学原则

一个严格的自适应流程应当：

1. 所有来源先发送较小 rank 的精确 PSD 截断因子及 $(\tau_j,\rho_j,u_j)$；
2. 为每个候选计算区间；
3. 若存在分离证书，接受该轮选择；
4. 若区间重叠，只提高竞争候选及已选来源的 rank；
5. 若仍不能认证，回退到更高 rank 或 Full-Gram。

该流程保证“认证即正确”，但不保证低 rank 时一定能够认证。慢速谱衰减、候选分数间隔小或池规模大时，区间可能直到接近 full rank 才分离。

### 5.3 保证成立所需的实现条件

严格定理要求：

- $\widetilde G_j$ 是 $G_j$ 的真实 PSD 下近似；
- 尾部质量是核尾上界，不能用 Frobenius 尾范数直接替代；
- 已选集合的尾部不能被遗忘；
- 数值零应在 Gram 特征值尺度判断后再开平方；
- float16 量化和一般 randomized SVD 不会自动保持 $G_j-\widetilde G_j\succeq0$。

量化或随机近似若要进入严格版本，必须额外控制

$$
\|\widehat G_j-\widetilde G_j\|
$$

并把相应扰动项传递到谱分布和熵误差中。否则只能作为无证书经验版本。

### 5.4 通信量

rank-$L_j$ 因子需要传输

$$
dL_j
$$

个浮点数，外加常数个尾部标量。因此 float32 主体通信量约为

$$
4dL_j\ \text{bytes/source}.
$$

这通常低于 $4d^2$ 的 Full-Gram，但是否低于 $4n_jd$ 的原始特征取决于 $L_j<n_j$ 是否成立。通信优势必须按真实 $n_j,d,L_j$ 报告，不能只比较渐近复杂度。

对一个集合 $S$，直接形成小核心的时间和内存上界分别为

$$
O(dL_S^2+L_S^3),
\qquad
O(L_S^2),
$$

其中第一项包括 Gram 核构造，第二项包括特征分解。批量比较许多集合时，可预计算并缓存
来源对交叉块

$$
B_i^{(L_i)\top}B_j^{(L_j)},
$$

再拼装候选核心；这减少重复乘法，但缓存量随候选来源对数增长。论文应分别报告特征提取、
sketch 构造、通信、核心拼装和 Stage-2 训练成本。

## 6. 组合优化边界

### 命题 9：有效秩目标不保证单调

取

$$
G_A=\operatorname{diag}(1/2,1/2),
\qquad
G_B=\operatorname{diag}(1,0).
$$

两者迹都为 1，因此这个例子符合 $\beta_A=\beta_B=1$ 的来源等权主设定，并可分别由
在 $e_1,e_2$ 上均匀取样、以及只在 $e_1$ 上取样的单位范数表示得到。此时

$$
R_G(G_A)=2,
$$

而

$$
R_G(G_A+G_B)
=R_G(\operatorname{diag}(3/2,1/2))
\approx1.928623<2.
$$

所以增加一个来源可能降低有效秩。$\square$

### 命题 9.1：有效秩目标一般不具有次模性

即使每个来源的迹都等于 1，$F(S)=\log R_G(G_S)$ 也可能违反 diminishing returns。

#### 证明

取三个来源

$$
G_0=\operatorname{diag}(1/5,4/5),
\qquad
G_1=\operatorname{diag}(0,1),
\qquad
G_2=\operatorname{diag}(1,0).
$$

三者均可由单位范数标准基样本的经验 Gram 得到。令

$$
A=\{0\},
\qquad
B=\{0,1\},
\qquad
x=2,
$$

所以 $A\subset B$ 且 $x\notin B$。直接计算归一化幅度谱的熵：

$$
F(A)=h_b(1/3)\approx0.636514,
$$

$$
F(A\cup\{x\})\approx0.688036,
$$

$$
F(B)=h_b(1/4)\approx0.562335,
$$

$$
F(B\cup\{x\})\approx0.688036.
$$

因此

$$
F(A\cup\{x\})-F(A)
\approx0.051522
<
0.125701
\approx F(B\cup\{x\})-F(B),
$$

与次模函数要求的边际收益递减方向相反。$\square$

由此必须区分三个问题：

1. **分数计算**：给定 $S$，计算 $F(S)$；Full-Gram 可以精确解决。
2. **分数近似**：用 rank-$L$ sketch 近似 $F(S)$；定理 8 可以提供区间。
3. **组合搜索**：在 $\binom MK$ 个集合中找最大值；除小规模穷举外，Greedy 和 Beam 当前仍是启发式搜索。

命题 9 排除了单调性，命题 9.1 进一步排除一般次模性。因此，标准单调次模 greedy 的
$(1-1/e)$ 保证不适用；若要获得组合近似比，必须利用额外的数据结构假设或改用具有已知结构的目标函数。

### 6.1 合理的搜索层级

- 小规模：穷举所有 $K$ 子集，得到精确谱目标最优集合；
- 中等规模：Beam Search，并报告相对小规模穷举的搜索 regret；
- 大规模：Greedy 或候选剪枝，但只能称启发式；
- 任何规模：rank-$L$ 证书只认证被比较分数，不自动认证搜索空间中未保留的集合。

## 7. 从谱目标到下游效用：为什么必须有第二阶段

### 7.1 下游效用不是 Gram 的函数

令目标任务为 $T$，固定模型、优化器、样本预算、训练步数和评估指标。用来源集合 $S$
训练后，在训练随机性与目标数据分布上的真实期望效用记为

$$
U_T(S).
$$

若指标原本是越小越好的损失，则可取其相反数，使全文统一为“效用越大越好”。单次训练或
有限验证集上观测到的是后文的 $\widehat U_T(S)$，不应与 $U_T(S)$ 混为一谈。

$U_T(S)$ 还依赖：

- 标签语义和类别比例；
- 标签噪声；
- 来源与目标的条件分布关系；
- 样本难度；
- 训练算法、模型容量、正则化和计算预算；
- 目标验证集的定义。

这些信息都不由无标签 Gram 唯一确定。

### 命题 10：无标签不可区分性

不存在一个仅观察 $(H_1,\ldots,H_M)$ 的选择器，能够对所有可能的监督标签分配和目标任务都保证选出下游最优来源集合。

#### 证明

考虑 $M=2,K=1$ 的二分类问题。两个来源都含有相同数量的标量表示
$x=+1$ 和 $x=-1$，所以 $H_1,H_2$ 以及全部无标签 Gram 输入完全相同。目标分布均匀取
$x\in\{-1,+1\}$，真实标签为 $y=x$。

固定下游训练协议为无截距最小二乘线性探针：

$$
\widehat w
\in\arg\min_w\sum_i(wx_i-y_i)^2,
$$

预测符号为 $\operatorname{sign}(\widehat wx)$。在世界一中，令来源 1 的隐藏标签为
$y=x$、来源 2 为 $y=-x$，则两个来源分别训练出 $\widehat w=+1$ 和
$\widehat w=-1$，目标准确率分别为 1 和 0。在世界二中交换两个来源的隐藏标签，最优来源也随之交换。

两个世界的所有无标签表示完全相同，所以任何只观察 $(H_1,H_2)$ 的确定性选择器必须在两个世界输出同一个来源；它至少在一个世界失败。随机选择器也不可能在两个世界都以概率 1 选对。故若不增加连接标签、任务和几何的结构性假设，就不存在普适下游最优保证。$\square$

该命题说明，Stage-2 不是工程上的补丁，而是从信息论角度处理无标签几何无法识别因素的必要组成。

### 7.2 一个明确的条件性桥接定理

虽然不存在普适桥接，但可以把经验假设写成可检验的条件。

令 $\mathcal A_K$ 表示全部大小为 $K$ 的来源集合。假设对固定任务 $T$，存在单调非减且 $L_g$-Lipschitz 的函数 $g_T$，满足

$$
\boxed{
|U_T(S)-g_T(F(S))|\le\xi_T,
\qquad\forall S\in\mathcal A_K.}
$$

$\xi_T$ 表示谱目标不能解释的最大任务残差。令

$$
S_F^*=\arg\max_{S\in\mathcal A_K}F(S),
\qquad
S_U^*=\arg\max_{S\in\mathcal A_K}U_T(S).
$$

若 Stage-1 产生的候选集合族 $\mathcal L\subseteq\mathcal A_K$ 中存在 $\widehat S$，满足

$$
F(S_F^*)-F(\widehat S)\le\eta,
$$

而 Stage-2 选择

$$
S_{\mathcal L}^*=\arg\max_{S\in\mathcal L}U_T(S),
$$

则

$$
\boxed{
U_T(S_U^*)-U_T(S_{\mathcal L}^*)
\le L_g\eta+2\xi_T.}
$$

#### 证明

由于 $S_F^*$ 最大化 $F$，且 $g_T$ 单调，

$$
U_T(S_U^*)
\le g_T(F(S_U^*))+\xi_T
\le g_T(F(S_F^*))+\xi_T.
$$

由 Lipschitz 性和 $F(S_F^*)-F(\widehat S)\le\eta$，

$$
g_T(F(S_F^*))
\le g_T(F(\widehat S))+L_g\eta.
$$

再由统一残差条件，

$$
g_T(F(\widehat S))
\le U_T(\widehat S)+\xi_T
\le U_T(S_{\mathcal L}^*)+\xi_T.
$$

合并三式即得结论。$\square$

这个定理的意义不是宣称 $\xi_T$ 很小，而是明确区分：

- $\eta$：sketch、搜索和 shortlist 对谱目标造成的误差；
- $\xi_T$：谱目标与真实任务效用之间的结构性缺口。

Paper07 的 rank-$L$ 理论主要控制前者；真实 adapter/LoRA 与独立目标实验才可能估计后者。若实验证明 $\xi_T$ 很大，两阶段方法必须扩大 shortlist，而不能通过加强线性代数主张来回避。

对给定 $L_g$，这一假设所需的最小统一残差可以写成

$$
\xi_T^*(L_g)
=
\inf_{\substack{g\ \text{单调非减}\\
\operatorname{Lip}(g)\le L_g}}
\sup_{S\in\mathcal A_K}
|U_T(S)-g(F(S))|.
$$

小规模穷举时，可用带 Lipschitz 约束的 isotonic regression 估计这条误差曲线，并在独立任务或独立来源划分上评估。只报告 Pearson/Spearman 相关系数不够：高平均相关不控制最坏候选残差，也不推出 top-$Q$ recall 或 utility regret。若只观察 shortlist 内的组合，则最多估计局部残差，不能声称对全部 $\mathcal A_K$ 的统一 $\xi_T$ 界。

### 条件性桥接定理的有限验证集版本

上一定理假设 Stage-2 能观察真实效用 $U_T$。实际系统只能得到有限验证集上的估计
$\widehat U_T$。假设在某个事件 $\mathcal E$ 上，对 shortlist 中所有组合同时成立

$$
\boxed{
\sup_{S\in\mathcal L}
|\widehat U_T(S)-U_T(S)|\le\zeta_T.}
$$

实际 Stage-2 输出

$$
\widehat S_{\mathcal L}
=\arg\max_{S\in\mathcal L}\widehat U_T(S).
$$

则在事件 $\mathcal E$ 上，

$$
\boxed{
U_T(S_U^*)-U_T(\widehat S_{\mathcal L})
\le L_g\eta+2\xi_T+2\zeta_T.}
$$

#### 证明

令 $S_{\mathcal L}^*=\arg\max_{S\in\mathcal L}U_T(S)$。桥接定理已经给出

$$
U_T(S_U^*)-U_T(S_{\mathcal L}^*)
\le L_g\eta+2\xi_T.
$$

另一方面，由统一验证误差和 $\widehat S_{\mathcal L}$ 的定义，

$$
U_T(S_{\mathcal L}^*)
\le\widehat U_T(S_{\mathcal L}^*)+\zeta_T
\le\widehat U_T(\widehat S_{\mathcal L})+\zeta_T
\le U_T(\widehat S_{\mathcal L})+2\zeta_T.
$$

两部分相加即得结论。$\square$

为了同时覆盖训练随机性与有限验证样本，令 $\omega_1,\ldots,\omega_R$ 是预先抽取的
独立训练随机种子，$U_T(S,\omega_r)\in[0,1]$ 是第 $r$ 个训练结果在目标分布上的
population utility，并令

$$
U_T(S)=\mathbb E_{\omega}[U_T(S,\omega)].
$$

每个训练结果再用 $n_{\mathrm{val}}$ 个与训练独立的目标样本估计，记分数为
$\widehat U_T(S,\omega_r)$，最终使用

$$
\widehat U_T(S)
=\frac1R\sum_{r=1}^{R}\widehat U_T(S,\omega_r).
$$

若 $Q=|\mathcal L|$ 个候选在查看验证结果前已经固定，则两次 Hoeffding 加 union bound
给出：以至少 $1-\delta$ 的概率，

$$
\boxed{
\zeta_T
\le
\underbrace{
\sqrt{\frac{\log(4Q/\delta)}{2R}}
}_{\zeta_{\mathrm{seed}}}
+
\underbrace{
\sqrt{\frac{\log(4QR/\delta)}{2n_{\mathrm{val}}}}
}_{\zeta_{\mathrm{val}}}.}
$$

第一项控制有限随机种子均值相对训练随机性期望的偏差；第二项先同时控制所有
$(S,r)$ 的验证误差，再对 $R$ 次结果取平均。候选之间共用同一验证集并不妨碍 union
bound，但验证样本必须与训练独立，且每个模型内部的验证样本应满足所用浓缩不等式的条件。

若论文把效用明确定义为一组预先固定种子的条件均值，而不是训练随机性期望，则可去掉
$\zeta_{\mathrm{seed}}$。反之，只运行一个随机种子时，不能仅用
$n_{\mathrm{val}}$ 很大来声称 $\zeta_T$ 很小。重复调参、事后选择随机种子或多轮查看同一验证集还会引入自适应选择偏差；此时应使用独立最终测试集、嵌套验证或显式的自适应数据分析控制。
上述闭式 Hoeffding 项只直接适用于逐样本有界均值；F1、AUROC 等非可加指标需要对应的统计界或预先规定的重采样区间。

因此，端到端误差有三个来源：

$$
\boxed{
\underbrace{L_g\eta}_{\text{谱近似与搜索}}
+\underbrace{2\xi_T}_{\text{几何代理缺口}}
+\underbrace{2\zeta_T}_{\text{有限验证与选择偏差}}.}
$$

只有第一项属于无条件线性代数或组合搜索分析；后两项必须由任务实验与验证协议支撑。

## 8. 两阶段方法的正式数学定义

### 8.1 Stage-1：无标签谱预筛选

Stage-1 的输入权限仅包括

$$
\{B_j^{(L_j)},\tau_j,\rho_j,u_j,\beta_j,\text{metadata}_j\}_{j=1}^M,
$$

或作为全信息参考的 $G_j$。它不访问标签和目标验证结果。

Stage-1 可以输出两种对象：

1. 个体来源短名单 $C\subseteq[M]$，$|C|=m$ 且 $K\le m\le M$；
2. 直接输出 $Q$ 个大小为 $K$ 的候选组合 $\mathcal L_Q$。

若输出个体短名单，其诱导组合空间为

$$
\mathcal L(C)=\{S\subseteq C:|S|=K\}.
$$

### 8.2 Stage-2：任务感知验证

Stage-2 在固定样本数、训练步数、初始化和计算预算下，对 $\mathcal L(C)$ 或 $\mathcal L_Q$ 中的组合执行真实数据消费，并选择验证效用最高者。

令全部任务最优集合组成

$$
\mathcal S_U^*
=\arg\max_{S\in\mathcal A_K}U_T(S).
$$

关键指标为：

$$
\operatorname{Recall}(C)
=\mathbf1\{\mathcal S_U^*\cap\mathcal L(C)\ne\varnothing\},
$$

以及

$$
\operatorname{Regret}_U(C)
=\max_{S\in\mathcal A_K}U_T(S)
-\max_{S\in\mathcal L(C)}U_T(S).
$$

如果输出候选组合列表，则相应定义为

$$
\operatorname{Recall}(\mathcal L_Q)
=\mathbf1\{\mathcal S_U^*\cap\mathcal L_Q\ne\varnothing\},
$$

$$
\operatorname{Regret}_U(\mathcal L_Q)
=\max_{S\in\mathcal A_K}U_T(S)-
\max_{S\in\mathcal L_Q}U_T(S).
$$

两阶段方法的真正价值应通过以下 Pareto 前沿衡量：

$$
(\text{通信字节},\ \text{Stage-1 时间},\ \text{Stage-2 次数},\
\ \operatorname{Recall},\ \operatorname{Regret}_U).
$$

只报告所选集合的有效秩，不能证明两阶段系统节省了任务验证成本。

### 8.3 从观测 Gram 到总体几何

前述确定性定理都以当前抽取到的表示矩阵 $H_j$ 为条件。若论文还要声称排序可以推广到
“来源分布本身”，就必须再加入一层抽样误差分析。

先考虑不中心化或使用固定外部中心的情形。令来源 $j$ 的表示向量
$z_{j,1},\ldots,z_{j,n_j}$ 独立同分布，且 $\|z_{j,i}\|_2\le1$。定义总体与经验二阶矩

$$
\overline G_j
=\beta_j\,\mathbb E[z_jz_j^{\top}],
\qquad
G_j
=\frac{\beta_j}{n_j}
\sum_{i=1}^{n_j}z_{j,i}z_{j,i}^{\top}.
$$

矩阵 Bernstein 不等式给出，对任意 $t>0$，

$$
\boxed{
\Pr\!\left(
\|G_j-\overline G_j\|_{\mathrm{op}}
\ge\beta_j t
\right)
\le
2d\exp\!\left(
-\frac{n_jt^2}{2+2t/3}
\right).}
$$

具体地，令

$$
Y_i=z_{j,i}z_{j,i}^{\top}
-\mathbb E[z_jz_j^{\top}].
$$

则 $\mathbb E[Y_i]=0$。由于
$0\preceq z_{j,i}z_{j,i}^{\top}\preceq I$ 且
$0\preceq\mathbb E[z_jz_j^{\top}]\preceq I$，有
$\|Y_i\|_{\mathrm{op}}\le1$，并可保守地取

$$
\left\|\sum_{i=1}^{n_j}\mathbb E[Y_i^2]\right\|_{\mathrm{op}}
\le n_j.
$$

把自伴矩阵 Bernstein 不等式应用于 $\sum_iY_i$，再使用

$$
G_j-\overline G_j
=\frac{\beta_j}{n_j}\sum_{i=1}^{n_j}Y_i
$$

即可得到上式。对 $M$ 个来源同时成立时，可将单来源失败概率设为
$\delta/M$ 并使用 union bound。

对任意集合 $S$，三角不等式进一步给出

$$
\left\|
G_S-\overline G_S
\right\|_{\mathrm{op}}
\le
\sum_{j\in S}
\|G_j-\overline G_j\|_{\mathrm{op}},
\qquad
\overline G_S=\sum_{j\in S}\overline G_j.
$$

但是，算子范数集中**不能直接推出有效秩排序稳定**，因为平方根在零特征值附近只有
$1/2$-Hölder 连续性。具体地，若两个非零 PSD 矩阵 $A,B$ 的谱最多有
$D\ge2$ 个分量，且

$$
\|A-B\|_{\mathrm{op}}\le\epsilon,
$$

则 Weyl 不等式与 $|\sqrt x-\sqrt y|\le\sqrt{|x-y|}$ 给出

$$
\sum_{i=1}^{D}
\left|\sqrt{\lambda_i(A)}-\sqrt{\lambda_i(B)}\right|
\le D\sqrt\epsilon.
$$

记

$$
\nu_A=\operatorname{tr}\sqrt A,
\qquad
\nu_B=\operatorname{tr}\sqrt B,
$$

则两者归一化幅度谱的总变差满足

$$
\operatorname{TV}(p_A,p_B)
\le
\min\left(
1,
\frac{D\sqrt\epsilon}{\max(\nu_A,\nu_B)}
\right).
$$

为看到归一化步骤，令
$a_i=\sqrt{\lambda_i(A)}$、$b_i=\sqrt{\lambda_i(B)}$，并记
$\Delta=\|a-b\|_1\le D\sqrt\epsilon$。则

$$
2\operatorname{TV}(p_A,p_B)
=\left\|\frac a{\nu_A}-\frac b{\nu_B}\right\|_1
\le\frac{2\Delta}{\nu_A}.
$$

交换 $A,B$ 又得到不超过 $2\Delta/\nu_B$，所以
$\operatorname{TV}(p_A,p_B)\le\Delta/\max(\nu_A,\nu_B)$。

将右侧记为 $\varepsilon_{\mathrm{pop}}$，并像定理 8 一样在
$1-1/D$ 处截断 Fannes--Audenaert 界，便有

$$
\boxed{
|\log R_G(A)-\log R_G(B)|
\le\Phi_D(\varepsilon_{\mathrm{pop}}).}
$$

因此，若来源级浓缩事件给出

$$
\|G_j-\overline G_j\|_{\mathrm{op}}\le\delta_j,
$$

则对每个候选集合可取

$$
\epsilon_S^{\mathrm{op}}=\sum_{j\in S}\delta_j,
$$

并令 $D_S^{\mathrm{stat}}$ 是同时覆盖 $G_S$ 和 $\overline G_S$ 谱支撑的维数上界。
除非已有独立的总体低秩假设，必须取

$$
D_S^{\mathrm{stat}}=d;
$$

不能用经验样本秩替代，因为有限样本可能没有观察到总体中存在的方向。于是可定义

$$
\varepsilon_{\mathrm{pop},S}
=\min\left(
1,
\frac{D_S^{\mathrm{stat}}\sqrt{\epsilon_S^{\mathrm{op}}}}
{\nu_{n,S}}
\right),
\qquad
\nu_{n,S}=\operatorname{tr}\sqrt{G_S}.
$$

这里使用了
$\max(\nu_{n,S},\nu_{\mathrm{pop},S})\ge\nu_{n,S}$。若只保存经验 Gram 的精确
PSD 截断，则 $\nu_{n,S}\ge\widetilde\nu_S$，可再用 $\widetilde\nu_S$ 替换分母得到更保守但可计算的界。于是

$$
\chi
=\max_{S\in\mathcal A_K}
\Phi_{D_S^{\mathrm{stat}}}(\varepsilon_{\mathrm{pop},S})
$$

可以作为第 8.4 节所需的统一抽样误差；实际不必声称这个最坏情形界很紧。若无法穷举
$\mathcal A_K$，还可以用所有大小为 $K$ 的来源级误差和之最大值以及统一核质量下界形成更松的上界。

这个界在小特征值很多时可能非常松，且粗略速率会从二阶矩的 $n^{-1/2}$ 退化为幅度谱的
$n^{-1/4}$。它揭示了一个必须通过实验检查的风险：经验 Gram 的有效秩看似稳定，并不自动意味着总体排序稳定。

若使用样本均值进行中心化，还要把均值估计误差加入协方差扰动；若每个来源先独立中心化，则对应的是来源内协方差；若候选集合重新计算全局均值，则应使用第 1.2.1 节的可加均值统计量。三种总体对象不能混用。

因此建议把以下项目列为统计稳定性实验，而不是理论定理的“验证”：

- 对每个来源按样本量画 $F(S)$ 的收敛曲线；
- 对候选集合做分层 bootstrap，报告排序翻转率和 top-$Q$ recall 区间；
- 使用独立抽样重复构建 shortlist，报告 Jaccard 稳定性；
- 将表示抽样波动与训练随机种子波动分开报告。

### 8.4 端到端总误差链

现在可以把各层拼成一条完整但明确有条件的保证。定义

$$
F_{\mathrm{pop}}(S)=\log R_G(\overline G_S),
\qquad
F_n(S)=\log R_G(G_S).
$$

假设在同一个高概率事件上满足：

1. **抽样误差**：对所有 $S\in\mathcal A_K$，

   $$
   |F_n(S)-F_{\mathrm{pop}}(S)|\le\chi;
   $$

2. **sketch 误差**：对实际搜索族 $\mathcal C\subseteq\mathcal A_K$，

   $$
   |\widehat F(S)-F_n(S)|\le e_S\le e_{\max};
   $$

3. **搜索遗漏**：若 $S_n^*\in\arg\max_{S\in\mathcal A_K}F_n(S)$，则

   $$
   \eta_{\mathrm{search},n}
   =F_n(S_n^*)-\max_{S\in\mathcal C}F_n(S);
   $$

4. **效用桥接**：存在单调非减的 $L_g$-Lipschitz 函数 $g_T$，使

   $$
   |U_T(S)-g_T(F_{\mathrm{pop}}(S))|\le\xi_T,
   \qquad \forall S\in\mathcal A_K;
   $$

5. **验证误差**：对 Stage-2 shortlist $\mathcal L$，

   $$
   \sup_{S\in\mathcal L}
   |\widehat U_T(S)-U_T(S)|\le\zeta_T.
   $$

令

$$
\widehat S\in\arg\max_{S\in\mathcal C}\widehat F(S),
$$

并要求 Stage-1 输出的 $\mathcal L$ 至少包含 $\widehat S$。再令实际 Stage-2 输出

$$
S_{\mathrm{out}}
\in\arg\max_{S\in\mathcal L}\widehat U_T(S).
$$

则 population spectral regret 满足

$$
\boxed{
F_{\mathrm{pop}}(S_{\mathrm{pop}}^*)
-F_{\mathrm{pop}}(\widehat S)
\le
2\chi+\eta_{\mathrm{search},n}+2e_{\max},}
$$

其中
$S_{\mathrm{pop}}^*\in\arg\max_{S\in\mathcal A_K}F_{\mathrm{pop}}(S)$。最终任务效用满足

$$
\boxed{
U_T(S_U^*)-U_T(S_{\mathrm{out}})
\le
L_g\!\left(
2\chi+\eta_{\mathrm{search},n}+2e_{\max}
\right)
+2\xi_T+2\zeta_T.}
$$

#### 证明链

由 $S_n^*$ 对经验谱目标最优以及抽样误差界，

$$
F_{\mathrm{pop}}(S_{\mathrm{pop}}^*)
\le F_n(S_{\mathrm{pop}}^*)+\chi
\le F_n(S_n^*)+\chi.
$$

同时

$$
F_n(\widehat S)
\ge F_{\mathrm{pop}}(\widehat S)-\chi.
$$

因此

$$
F_{\mathrm{pop}}(S_{\mathrm{pop}}^*)
-F_{\mathrm{pop}}(\widehat S)
\le
2\chi+F_n(S_n^*)-F_n(\widehat S).
$$

把最后一项拆成

$$
F_n(S_n^*)-F_n(\widehat S)
=\eta_{\mathrm{search},n}
+\left[
\max_{S\in\mathcal C}F_n(S)-F_n(\widehat S)
\right],
$$

并应用推论 8.4，方括号不超过 $2e_{\max}$，得到第一条界。然后在第 7.2 节的
条件性桥接定理中令 $F=F_{\mathrm{pop}}$、
$\eta=2\chi+\eta_{\mathrm{search},n}+2e_{\max}$，最后应用有限验证集版本，即得第二条界。$\square$

这个总界不是“无标签选择必然成功”的结论。它的作用是把可证明项和待验证项完全拆开：

$$
\boxed{
\text{总误差}
=\text{抽样}
+\text{搜索}
+\text{sketch}
+\text{代理失配}
+\text{验证选择}.}
$$

$e_{\max}$ 可由精确 PSD 尾部确定性控制；$\chi$ 需要抽样稳定性；
$\eta_{\mathrm{search},n}$ 需要穷举或搜索对照；$\xi_T$ 是最关键且无法由无标签谱理论消除的任务相关项；$\zeta_T$ 由验证集规模和选择协议控制。若桥接关系直接以经验分数 $F_n$ 定义，则公式中可以去掉 $2\chi$，但相应主张只针对当前观测数据，不再是总体分布保证。

### 8.5 假设台账

| 假设 | 用于哪一层 | 违反后的结论 |
|---|---|---|
| 编码器和预处理对所有来源一致 | 全部谱比较 | Gram 分数不再处于共同坐标与尺度 |
| 预处理不依赖候选集合 | 固定 $G_j$ 可加与 rank-$L$ sketch | 集合依赖的重新归一化会破坏预计算结构 |
| $\beta_j$ 在看结果前固定 | 来源权重解释 | 事后调权引入选择偏差并改变目标 |
| block-compatible | 逐方向 $2\times2$ 机制和块熵分解 | 只能回到完整耦合核心或扰动诊断 |
| 精确 top-$L$ PSD 下近似与合法尾界 | 定理 8 及排序证书 | 近似分数仍可使用，但没有该确定性证书 |
| 枚举族覆盖待声称的搜索空间 | 全局谱证书 | 结论只能限定在保留族内 |
| 有界独立抽样 | 经验到总体的 Bernstein 界 | 必须改用适合相关、分层或重尾数据的浓缩工具 |
| 单调 Lipschitz 代理且统一残差小 | 谱到任务 utility regret | 无标签谱不提供任务保证 |
| shortlist 预先固定且验证独立 | 有限验证选择界 | 需要嵌套验证或自适应选择校正 |

其中只有 block-compatible 是局部解释层的专用假设；Full-Gram 恒等式和 rank-$L$ Gram
路线本身不依赖它。最强、也最需要跨任务实验证据的假设是 utility bridge，而不是
线性代数部分。

## 9. 四种方法在统一框架中的位置

### 9.1 Full-Gram Effective Rank

- 保存完整 $G_j$；
- 对给定集合的谱分数精确；
- 可作为算法评价的 full-information 参考；
- 不等于低通信或隐私方法；
- 穷举、Beam 和 Greedy 的搜索性质仍需分别报告。

### 9.2 Truncated-Gram Effective Rank

- 保存 $B_j^{(L)}$ 和尾部元数据；
- 直接计算 $R_G(\sum_j\widetilde G_j)$；
- 可以给出误差区间和自适应回退；
- 这是当前最符合完整 Gram 代数的可部署 Collapse 候选。

### 9.3 DPP/Subspace Diversity

- 使用 top-$L$ 子空间或池间核直接奖励方向覆盖；
- 不需要把当前集合再次压成四标量；
- 是 rank-$L$ 方法必须在同字节预算下比较的强基线；
- 它优化的 log-determinant/子空间覆盖目标与有效秩目标不同，不能把两者结果互相替代。

### 9.4 Legacy Four-Summary Collapse

- 使用 $(R_A,R_B,\gamma,\bar\alpha)$；
- 一般不精确；
- 多步状态更新会再次丢失谱质量与方向对应关系；
- 只保留为理论启发消融，不再作为主算法。

### 9.5 当前代码对应关系（2026-09-08）

- `scripts/two_stage_classic_baselines.py::GramSketch` 保存转置约定下的
  $B_j=\Sigma_{j,L}V_{j,L}^{\top}\in\mathbb R^{L\times d}$；因此代码累加的是
  $B_j^{\top}B_j$，与本节使用 $V_{j,L}\Sigma_{j,L}$ 时的 $B_jB_j^{\top}$ 完全等价。
- `make_gram_sketch` 在离线实验中用完整 SVD 计算精确尾部核质量和平方质量；部署时可以替换为
  任何合法上界，但不能把 Frobenius 尾范数直接当核尾质量。
- `gram_sketch_statistics` 直接对拼接因子的小核心谱计算 $\widehat F$、$T_S$、$\varepsilon_S$
  和定理 8 区间。
- `rank_l_gram_greedy` 每一步都累计已选来源的全部 sketch 与尾部，输出是否满足推论 8.2 的
  严格分离条件。
- `scripts/two_stage_summary_only_controlled.py` 和
  `scripts/two_stage_natural_shortlist.py` 已将该方法注册为 `rank_l_gram`。
- `collapse_sketch` 只表示 Legacy Four-Summary 消融。历史 `results/` 在本次实现之前生成，不能
  视为 `rank_l_gram` 的实验结果；证据状态见 `AUDIT_2026-09-08.md`。

## 10. 数学主张与实验证据的对应关系

| 主张 | 身份 | 所需验证 |
|---|---|---|
| $G_S=\sum_jG_j$ | 无条件恒等式 | 单元测试只检查实现 |
| 全局中心化由二阶矩和均值恢复 | 无条件恒等式，但修正依赖 $S$ | 核对预处理顺序；不能直接沿用 PSD sketch 界 |
| $G_S$ 与 $K_S$ 非零等谱 | 无条件定理 | 随机数值实验只检查实现 |
| 边际谱和 principal angles 不充分 | 不可能性定理 | 显式反例即可证明 |
| block-compatible 的 $2\times2$ 谱 | 条件性精确定理 | 真实数据须报告条件偏离，不能用拟合相关性替代假设 |
| 非对角耦合残差控制 Gram 谱扰动 | 无条件矩阵扰动界 | 对重根子空间处理基不唯一性 |
| 块熵分解 | 条件性精确定理 | 数值实验只检查实现 |
| rank-$L$ 熵误差界 | 无条件近似定理，但要求 PSD 截断 | 报告覆盖、区间宽度、认证率和 full-rank 回退率 |
| exact/approx greedy 单步一致 | 条件性证书 | 必须有严格区间分离 |
| 有限枚举族内的谱 regret | 确定性误差界 | 明确枚举族；不能外推到被剪枝集合 |
| $F$ 一般非单调且非次模 | 显式反例证明 | 单元测试复算反例即可 |
| Greedy/Beam 全局最优 | 当前不成立的主张 | 小规模穷举只能测经验 regret |
| 经验 Gram 排序推广到总体 Gram | 条件性统计主张 | 样本量曲线、bootstrap、排序稳定性和质量下界 |
| 高有效秩通常有更高任务效用 | 经验假设 | 独立任务、编码器、来源族、置信区间和稳健统计 |
| 条件性 utility-regret 分解 | 条件性定理 | 报告或上界 $\chi,\eta_{\mathrm{search}},e,\xi,\zeta$ |
| two-stage 节省计算且保留效用 | 系统经验主张 | shortlist recall、utility regret 与 NPU-hours |
| sketch 保护隐私 | 当前无依据 | 需要独立威胁模型、攻击与形式化隐私机制 |

## 11. 正式论文中建议的定理顺序

正文理论可压缩为以下结构：

1. 定义候选集合、预处理协议、来源权重、加权 Gram 和有效秩目标；
2. 定理 1：固定来源 Gram 的可加性；
3. 定理 2：完整因子及完整耦合核心与合并 Gram 非零等谱；
4. 定理 3：边际谱加完整 principal angles 的不可识别反例；
5. 定理 4--6：block-compatible 条件下的逐方向谱、局部互补参数和块熵分解；
6. 引理 7 与定理 8：rank-$L$ PSD sketch 的尾界和熵误差界；
7. 推论 8.1--8.4：单步证书、有限搜索族证书、谱 regret 与自适应精化；
8. 命题 9 与 9.1：目标一般非单调、非次模，以及 Greedy/Beam 的保证边界；
9. 命题 10：无标签几何不能普适决定监督效用；
10. 条件性桥接定理：谱 shortlist 质量到任务 utility regret；
11. 有限样本补充：总体 Gram、有限验证误差与端到端五项误差链。

正文应突出“精确代数 -> 信息极限 -> 可解释特例 -> 可认证近似”。较长证明、数值稳定性、尾界改进和边界情况可放附录。

## 12. 最终可保留与必须撤回的表述

### 可以严格保留

- 多来源合并 Gram 的可加性；
- 固定顺序下全局中心化 Gram 的均值/二阶矩恢复公式；
- 完整 Gram、完整因子和完整耦合核心的非零等谱性；
- 边际谱与 principal angles 的信息不足；
- block-compatible 条件下的逐方向谱和熵链式分解；
- 非对角加权耦合残差对 Gram 特征值的扰动界；
- 精确 PSD top-$L$ sketch 的确定性熵区间；
- 区间分离时的单步 exact-greedy 排序证书；
- 有限枚举搜索族内的谱 regret 和严格分离证书；
- 有效秩目标一般既非单调也非次模；
- 无标签几何不能普适保证监督任务最优；
- 在显式抽样、代理和验证假设下的端到端误差分解。

### 只能作为经验主张

- 真实表示近似 block-compatible；
- 较小 $L$ 通常足以准确排序；
- 证书在真实谱上经常能够分离候选；
- 有效秩与下游效用存在稳定正相关；
- two-stage 能在固定效用下减少至少某一比例的训练成本。
- 经验 Gram 的候选排序在新抽样下稳定；
- 条件性桥接残差 $\xi_T$ 足够小，因而 shortlist 能覆盖任务最优组合。

### 必须撤回

- 四标量对一般矩阵唯一决定合并有效秩；
- 平均 alignment 是充分统计量；
- 旧误差界在未控制谱质量配对和非对角耦合时普适成立；
- alignment 降低必然提高全局有效秩；
- $F$ 已被证明为单调或次模；
- Greedy/Beam 已被证明全局最优；
- 最大有效秩必然最大化下游性能；
- 低秩摘要天然提供隐私保证。
- 集合依赖的全局中心化可以不经修正地沿用固定 PSD rank-$L$ 证书。

## 13. 一句话数学定位

> 对固定且集合无关的预处理，多来源冻结表示的合并谱由各来源 PSD Gram 算子的和精确决定；低维边际谱与 principal angles 一般不足以恢复该谱，但奇异方向可配对时存在精确局部互补分解，精确 PSD rank-$L$ 截断则提供无需配对假设、可组合且可诊断失效的近似路线。该理论严格控制的是观测表示上的谱计算误差；从样本到总体、从谱到任务效用、再从有限验证到最终选择，都必须通过显式条件和独立实验补齐。
