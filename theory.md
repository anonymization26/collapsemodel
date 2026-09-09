# Collapse Model：精确理论、近似边界与最终定位

> 2026-09-08 更新：旧四标量公式仅作为经验基线；一般情形下的主计算方法已升级为
> 可组合的 rank-$L$ PSD Gram sketch。完整证明、尾界和两阶段误差链见
> `MATHEMATICAL_FOUNDATION.md`。

## 1. 研究问题

给定两个由同一冻结编码器得到的特征矩阵

$$
H_A\in\mathbb{R}^{n_A\times d},\qquad
H_B\in\mathbb{R}^{n_B\times d},
$$

将它们按样本维纵向合并：

$$
H_{A\cup B}=
\begin{bmatrix}
H_A\\
H_B
\end{bmatrix}.
$$

Collapse Model 试图回答三个递进问题：

1. 合并矩阵的奇异值谱如何由两个输入矩阵决定？
2. 哪些谱能量与子空间几何因素导致合并后的多样性增加或塌陷？
3. 能否仅用少量数据集级摘要，近似预测合并有效秩并进行数据池选择？

这三个问题对应不同的理论难度。第一个问题存在一般精确表达；第二个问题在方向可配对时存在精确的局部分解；第三个问题通常只能得到近似预测器，而不是普适闭式定理。

---

## 2. 完全一般的精确起点

记

$$
H_A=U_A\Sigma_A V_A^\top,\qquad
H_B=U_B\Sigma_B V_B^\top.
$$

合并矩阵的右 Gram 矩阵满足无条件精确恒等式：

$$
G_{A\cup B}
=H_{A\cup B}^\top H_{A\cup B}
=H_A^\top H_A+H_B^\top H_B.
$$

代入 SVD 得到：

$$
G_{A\cup B}
=V_A\Sigma_A^2V_A^\top
+V_B\Sigma_B^2V_B^\top.
$$

若 $\lambda_j(G_{A\cup B})$ 是该矩阵的特征值，则合并矩阵的奇异值为

$$
s_j=\sqrt{\lambda_j(G_{A\cup B})}.
$$

采用奇异值归一化的谱熵有效秩时，定义

$$
p_j=\frac{s_j}{\sum_\ell s_\ell},
\qquad
r_{\mathrm{eff}}(H_{A\cup B})
=\exp\left(-\sum_jp_j\log p_j\right).
$$

因此，一般情况下的精确答案是：先计算两个加权子空间算子之和的完整谱，再计算谱熵。这个答案没有理论漏洞，但仍依赖完整矩阵或足以恢复完整 Gram 算子的高维信息，不能直接满足 summary-only selection 的目标。

### 2.1 一般问题的真正困难

困难不在于不知道两个输入矩阵各自的谱，而在于不知道它们的谱权重如何与相对方向对应。将 $V_A$ 的右奇异向量补全为整个特征空间的一组正交基 $Q_A$，并把 $\Sigma_A^2$ 用零特征值补成 $d\times d$ 的对角矩阵。令

$$
C=Q_A^\top V_B.
$$

则在 $Q_A$ 基底下，核心算子精确写为

$$
Q_A^\top G_{A\cup B}Q_A
=\widetilde\Sigma_A^2+C\Sigma_B^2C^\top,
$$

一般的 $C$ 是稠密矩阵。这意味着 $A$ 的一个奇异方向可能同时与 $B$ 的多个方向耦合。仅知道两个子空间的平均 principal-angle alignment，无法恢复这种权重与方向的对应关系。

### 2.2 该基底表达的逐步推导

为避免低秩矩阵带来的维度歧义，先写两个矩阵的紧致 SVD：

$$
H_A=U_A\Sigma_A V_A^\top,
\qquad
V_A\in\mathbb{R}^{d\times \rho_A},
$$

$$
H_B=U_B\Sigma_B V_B^\top,
\qquad
V_B\in\mathbb{R}^{d\times \rho_B},
$$

其中 $\rho_A,\rho_B$ 表示代数秩；后文的 $r_A,r_B$ 专门表示谱熵有效秩。对应 Gram 算子为

$$
G_A=V_A\Sigma_A^2V_A^\top,
\qquad
G_B=V_B\Sigma_B^2V_B^\top.
$$

将 $V_A$ 补全为一个 $d\times d$ 正交矩阵：

$$
Q_A=[V_A\;V_{A,\perp}],
\qquad
Q_A^\top Q_A=I_d.
$$

由于 $Q_A$ 的前 $\rho_A$ 列就是 $V_A$，所以

$$
Q_A^\top V_A=
\begin{bmatrix}
I_{\rho_A}\\
0
\end{bmatrix}.
$$

因此

$$
Q_A^\top G_AQ_A
=Q_A^\top V_A\Sigma_A^2V_A^\top Q_A
=\widetilde\Sigma_A^2,
$$

其中

$$
\widetilde\Sigma_A^2
=\operatorname{diag}
(\sigma_{A,1}^2,\ldots,\sigma_{A,\rho_A}^2,0,\ldots,0)
\in\mathbb{R}^{d\times d}.
$$

再定义 $B$ 的右奇异方向在 $Q_A$ 基底中的坐标：

$$
C=Q_A^\top V_B\in\mathbb{R}^{d\times \rho_B}.
$$

于是

$$
Q_A^\top G_BQ_A
=Q_A^\top V_B\Sigma_B^2V_B^\top Q_A
=C\Sigma_B^2C^\top.
$$

两式相加就得到

$$
Q_A^\top G_{A\cup B}Q_A
=\widetilde\Sigma_A^2+C\Sigma_B^2C^\top.
$$

这是精确等式，不是近似。因为 $Q_A$ 是正交矩阵，左右乘 $Q_A^\top$ 和 $Q_A$ 只是对同一线性算子进行正交换基。因此

$$
G_{A\cup B}
\quad\text{与}\quad
\widetilde\Sigma_A^2+C\Sigma_B^2C^\top
$$

具有完全相同的特征值，也就产生完全相同的合并奇异值和有效秩。

此外，由于 $V_B$ 的列正交，$C$ 的列也正交：

$$
C^\top C
=V_B^\top Q_AQ_A^\top V_B
=I_{\rho_B}.
$$

所以 $C$ 不是任意系数矩阵，而是 $B$ 的正交右奇异方向在 $A$ 的完整正交坐标系中的表示。

### 2.3 对称的完整基表达

也可以把 $V_B$ 补全为完整正交基 $Q_B$，并定义

$$
R=Q_A^\top Q_B.
$$

此时 $R\in\mathbb{R}^{d\times d}$ 是正交矩阵，满足

$$
R^\top R=RR^\top=I_d.
$$

令 $\widetilde\Sigma_B^2$ 为补零后的 $d\times d$ 对角谱矩阵，则

$$
Q_A^\top G_{A\cup B}Q_A
=\widetilde\Sigma_A^2
+R\widetilde\Sigma_B^2R^\top.
$$

这个表达把一般合并问题拆成三类信息：

1. $\widetilde\Sigma_A^2$：数据集 $A$ 在各方向上的谱能量；
2. $\widetilde\Sigma_B^2$：数据集 $B$ 在各方向上的谱能量；
3. $R$：两套完整右奇异坐标系之间的相对旋转。

因此，合并谱不仅取决于“两侧各有多少能量”，还取决于 $B$ 的每个谱权重经 $R$ 旋转后落到 $A$ 的哪些方向上。

### 2.4 稠密 $C$ 为什么阻止逐方向分解

展开 $B$ 项的矩阵元素：

$$
\left(C\Sigma_B^2C^\top\right)_{ij}
=\sum_{k=1}^{\rho_B}
\sigma_{B,k}^2C_{ik}C_{jk}.
$$

当 $i\neq j$ 时，只要存在某个 $B$ 方向同时在 $Q_A$ 的第 $i$ 和第 $j$ 个方向上有非零投影，就会产生非零交叉项。这些交叉项把不同的 $A$ 方向耦合起来，使整个特征值问题不能拆成独立标量或独立方向对。

只有当经过适当的方向排列后，$R$ 或 $C$ 具有相互独立的稀疏配对结构时，算子才会分解成若干互不耦合的低维 blocks。最理想的情形是每个 $A$ 方向只与一个 $B$ 方向耦合，此时每个联合子空间至多为二维，进而得到后文的 $2\times2$ 特征值公式。

### 2.5 与 principal angles 的关系

紧致交叉矩阵

$$
M=V_A^\top V_B
$$

是 $C$ 的前 $\rho_A$ 行。$M$ 的奇异值是两个右奇异子空间之间 principal angles 的余弦：

$$
\sigma_i(M)=\cos\theta_i.
$$

但是，principal angles 只保留 $M$ 的奇异值，不保留 $M$ 的左右奇异向量。这些奇异向量决定了 principal directions 如何由原始谱方向混合而成。若 $\Sigma_A^2$ 和 $\Sigma_B^2$ 不是各向同性的，这种混合会改变大谱权重与高 alignment 的配对方式，从而改变合并谱。

所以：

> $M$ 的完整结构包含谱方向之间的对应关系；principal-angle 集合只包含子空间几何的旋转不变量；平均 $\alpha$ 又进一步把整个角度集合压缩成一个标量。

信息在这两次压缩中逐步丢失。这正是一般情形无法仅由平均 alignment 精确预测合并有效秩的根本原因。

---

## 3. 为什么 principal angles 本身不够

Principal angles 描述两个子空间整体的相对位置，但它们不自动满足以下条件：

- principal vectors 同时也是 $H_A$ 和 $H_B$ 的奇异向量；
- 第 $i$ 个大奇异值方向恰好与第 $i$ 个 principal direction 对应；
- 不同方向对之间不存在交叉耦合；
- 相同 alignment 出现在谱头部与谱尾部时具有相同影响。

这一区分十分关键。两个矩阵可以具有：

- 相同的个体奇异值集合；
- 相同的子空间维数；
- 相同的 principal-angle 集合；

但如果奇异值在 principal-vector bases 上的排列不同，合并谱仍然可能不同。

因此，原理论中需要明确加入一个比“存在 principal angles”更强的兼容性条件：奇异向量基与成对 principal-vector bases 兼容。

---

## 4. 成对、解耦方向下的精确局部模型

假设存在方向对

$$
v_{A,i},\qquad v_{B,i},
$$

满足

$$
v_{A,i}^\top v_{B,j}=0,\qquad i\neq j,
$$

以及

$$
v_{A,i}^\top v_{B,i}=\cos\theta_i.
$$

令

$$
\alpha_i=\cos^2\theta_i.
$$

此时不同方向对相互解耦，第 $i$ 对方向对应一个精确二维谱问题。其两个 Gram 特征值为

$$
\lambda_{\pm,i}
=\frac{
\sigma_{A,i}^2+\sigma_{B,i}^2
\pm
\sqrt{
(\sigma_{A,i}^2-\sigma_{B,i}^2)^2
+4\sigma_{A,i}^2\sigma_{B,i}^2\alpha_i
}
}{2}.
$$

合并矩阵对应的两个奇异值为

$$
s_{+,i}=\sqrt{\lambda_{+,i}},\qquad
s_{-,i}=\sqrt{\lambda_{-,i}}.
$$

如果所有方向均满足该解耦结构，完整合并谱就是

$$
\{s_{+,1},s_{-,1},s_{+,2},s_{-,2},\ldots\}.
$$

这应当成为 Collapse Model 修订后最重要的精确理论结果。

### 4.1 逐方向能量比例

定义

$$
\gamma_i=\frac{\sigma_{B,i}}{\sigma_{A,i}}.
$$

则

$$
\lambda_{\pm,i}
=\sigma_{A,i}^2
\frac{
1+\gamma_i^2
\pm\sqrt{(1-\gamma_i^2)^2+4\gamma_i^2\alpha_i}
}{2}.
$$

定义分支缩放因子

$$
c_{\pm,i}
=\sqrt{
\frac{
1+\gamma_i^2
\pm\sqrt{(1-\gamma_i^2)^2+4\gamma_i^2\alpha_i}
}{2}
},
$$

则

$$
s_{\pm,i}=\sigma_{A,i}c_{\pm,i}.
$$

对应的局部分支权重为

$$
q_i=\frac{c_{+,i}}{c_{+,i}+c_{-,i}}.
$$

一般情况下，每个方向都有自己的 $\gamma_i$、$\alpha_i$ 与 $q_i$。因此，真正精确的模型通常不存在一个对所有方向通用的单一 $r^*$。

### 4.2 精确的逐方向有效秩

令总核质量为

$$
Z=\sum_j\sigma_{A,j}(c_{+,j}+c_{-,j}).
$$

则合并谱的归一化权重为

$$
p_{\pm,i}=\frac{\sigma_{A,i}c_{\pm,i}}{Z}.
$$

所以

$$
r_{\mathrm{eff}}(H_{A\cup B})
=\exp\left[
-\sum_i
\left(
p_{+,i}\log p_{+,i}
+p_{-,i}\log p_{-,i}
\right)
\right].
$$

这个表达在成对、解耦条件下是精确的，但它需要完整的逐方向信息，因而属于解释性理论，而不是通信成本很低的选择算法。

---

## 5. 比例谱与统一 alignment 的闭式模型

若进一步假设

$$
\sigma_{B,i}=\gamma\sigma_{A,i},\qquad \forall i,
$$

以及

$$
\alpha_i=\alpha,\qquad \forall i,
$$

则

$$
c_{\pm,i}=c_\pm,qquad q_i=q.
$$

整个合并谱变为同一基础谱的两个缩放副本：

$$
\{c_+\sigma_{A,i}\}_i
\cup
\{c_-\sigma_{A,i}\}_i.
$$

其中

$$
q=\frac{c_+}{c_++c_-}.
$$

此时谱熵具有精确分解：

$$
\mathcal H_{\mathrm{merged}}
=\mathcal H_A+H_b(q),
$$

因而

$$
r_{\mathrm{merged}}
=r_A\exp(H_b(q)).
$$

这里

$$
H_b(q)=-q\log q-(1-q)\log(1-q).
$$

由于 proportional spectra 意味着两个归一化奇异值谱相同，所以必然有

$$
r_A=r_B.
$$

因此，允许 $r_A\neq r_B$ 的当前预测公式不能作为比例谱模型的精确推论。

对一般摘要预测器，若 $q<1$，其预测满足

$$
\widehat r_{\mathrm{merged}}>r_{\mathrm{dom}}
$$

当且仅当

$$
\frac{r_{\mathrm{dom}}}{r_{\mathrm{sub}}}
<\rho_c(\alpha,\gamma)
=\exp\left(\frac{H_b(q)}{1-q}\right).
$$

该阈值只是在摘要模型内部的精确判别边界。特别地，
$q\to1^{-}$ 时 $\rho_c\to\infty$，但预测增益同时趋于零；在退化点
$q=1$，上述比值形式无定义，而预测公式直接给出
$\widehat r_{\mathrm{merged}}=r_{\mathrm{dom}}$。因此，二元
“superadditive”判定必须与增益幅度同时报告。

---

## 6. 当前摘要预测器的正确身份

当前实用预测器可写成

$$
\log\widehat r_{\mathrm{merged}}
=H_b(q)+q\log r_{\mathrm{dom}}
+(1-q)\log r_{\mathrm{sub}}.
$$

它可以理解为：

- 用 $H_b(q)$ 表示两个合并谱分支的混合熵；
- 用 $q$ 和 $1-q$ 对两个输入谱熵进行几何插值；
- 用全局 $\gamma$ 和平均 $\alpha$ 近似所有逐方向的 $\gamma_i$ 和 $\alpha_i$。

这一公式具有明确的理论动机，但在一般谱上不是精确恒等式。它更准确的定位是：

> 由精确双分支 Gram 谱结构启发的低维摘要近似。

这种定位介于纯经验回归与普适定理之间。模型形式来自精确谱分析，但其一般化效果必须依靠真实数据实验、校准和依赖性稳健统计来验证。

---

## 7. 现有理论的主要缺陷

### 7.1 全局核范数比不是逐方向比例

论文定义

$$
\gamma=\frac{\sum_i\sigma_{B,i}}{\sum_i\sigma_{A,i}}.
$$

它只约束奇异值之和，不能推出

$$
\sigma_{B,i}=\gamma\sigma_{A,i}.
$$

因此，把全局 $\gamma$ 直接代入每个局部 block，是一个近似步骤，而不是代数恒等变换。

### 7.2 平均 alignment 不是充分统计量

例如

$$
(\alpha_1,\alpha_2)=(1,0)
$$

与

$$
(\alpha_1,\alpha_2)=(0.5,0.5)
$$

具有相同均值，但一般会产生不同的分支谱。平均 alignment 丢失了方向间异质性。

### 7.3 alignment 与谱权重存在耦合

头部奇异方向上的高 alignment 通常比谱尾部的高 alignment 影响更大。因此，一个更合理的摘要应考虑谱加权 alignment，例如

$$
\bar\alpha_w
=\frac{\sum_iw_i\alpha_i}{\sum_iw_i},
$$

其中 $w_i$ 可以依赖 $\sigma_{A,i}\sigma_{B,i}$ 或该方向对的核质量。

### 7.4 四个标量不是充分统计量

一般情况下，$r_A$、$r_B$、全局 $\gamma$ 和平均 $\alpha$ 不能唯一确定合并有效秩。它们没有保留：

- 两侧完整谱形状；
- $\gamma_i$ 的分布；
- $\alpha_i$ 的分布；
- alignment 与谱权重的相关性；
- 非配对方向的交叉耦合。

因此，不存在一个仅依赖这四个标量、对所有矩阵均精确的普适预测公式。

一个显式反例如下。令两侧奇异值均为 $(1,1)$，取

$$
V_A=(e_1,e_2)\subset\mathbb R^4.
$$

对矩阵对 X，令

$$
V_B^{(X)}=(e_1,e_3),
$$

其逐方向 alignment 为 $(1,0)$。对矩阵对 Y，令

$$
V_B^{(Y)}=\left(
\frac{e_1+e_3}{\sqrt2},
\frac{e_2+e_4}{\sqrt2}
\right),
$$

其逐方向 alignment 为 $(1/2,1/2)$。两个例子均满足

$$
r_A=r_B=2,\qquad \gamma=1,\qquad \alpha=1/2.
$$

但 X 的合并奇异值为

$$
(\sqrt2,1,1),
$$

对应有效秩约为 $2.958$；Y 的合并奇异值为两份

$$
\sqrt{1+1/\sqrt2}
\quad\text{和}\quad
\sqrt{1-1/\sqrt2},
$$

对应有效秩约为 $3.661$。因此，相同的四个全局摘要可以产生不同的
合并有效秩。平均 alignment 丢失的方向异质性不是证明细节，而是
不可识别性的根本来源。

### 7.5 原误差界缺失必要控制量

合理的误差界至少需要控制：

- $\gamma_i$ 围绕全局 $\gamma$ 的离散程度；
- $\alpha_i$ 的谱加权方差；
- 两个归一化谱之间的距离；
- 非配对耦合矩阵的范数；
- 谱尾部对熵的敏感性。

原 Theorem 1 未控制这些因素，因而可能在上界右侧为零时仍产生非零误差。当前形式应撤回，而不是仅修改证明文字。

---

## 8. 不同塌陷情景的精确解释

### 8.1 完全方向塌陷

若

$$
\alpha_i=1,
$$

则

$$
\lambda_{+,i}=\sigma_{A,i}^2+\sigma_{B,i}^2,
\qquad
\lambda_{-,i}=0.
$$

两个输入方向完全合并，只剩一个非零分支。这是严格意义上的 directional collapse。

### 8.2 完全方向互补

若

$$
\alpha_i=0,
$$

则

$$
\lambda_{+,i}=\max(\sigma_{A,i}^2,\sigma_{B,i}^2),
$$

$$
\lambda_{-,i}=\min(\sigma_{A,i}^2,\sigma_{B,i}^2).
$$

两个方向均被保留。若能量也较平衡，谱熵通常显著增加。

### 8.3 能量塌陷

若

$$
\gamma_i\to0,
$$

则

$$
\lambda_{+,i}\to\sigma_{A,i}^2,
\qquad
\lambda_{-,i}\to0.
$$

即使两个方向正交，弱侧也难以在归一化谱中形成有影响力的分支。$\gamma_i\to\infty$ 时同理，只是由 $B$ 主导。

### 8.4 平衡且互补

若

$$
\gamma_i\approx1,
\qquad
\alpha_i\approx0,
$$

则两个分支接近等权，每个方向对近似产生两个独立谱方向。该区域通常具有最大的谱熵增益。

### 8.5 平衡但高度重合

若

$$
\gamma_i\approx1,
\qquad
\alpha_i\approx1,
$$

则一个分支吸收几乎全部能量，另一个分支接近零。能量增加但独立方向不增加。这是 clone attack 所暴露的典型情景。

### 8.6 非均匀混合塌陷

真实数据最可能同时包含：

- 头部方向的 directional collapse；
- 尾部方向的互补增益；
- 部分方向的 energy collapse；
- 部分方向的近似平衡。

因此，完整数据集对不一定属于唯一 regime。更精确的表述是：不同方向对处于不同状态，整体有效秩是这些局部状态经过谱权重归一化后的综合结果。

### 8.7 交叉耦合塌陷

当 $C=V_A^\top V_B$ 稠密时，一个强方向可能把能量分散到多个对侧方向。此时逐方向 $2\times2$ 解释也只是近似。可以将 $C$ 分为近似配对部分与残差：

$$
C=C_{\mathrm{pair}}+E.
$$

$E$ 的谱范数或 Frobenius 范数可作为非配对耦合强度，并可能成为未来误差界的重要控制量。

---

## 9. 可以进一步发展的精确与近似理论

### 9.1 从标量摘要升级为分桶摘要

不必在“完整谱”与“四个标量”之间二选一。可以按谱质量把方向划分为若干桶，每个桶传输：

- 核质量；
- 局部有效秩；
- 加权平均 $\gamma_i$；
- 加权平均与方差 $\alpha_i$；
- 跨桶耦合摘要。

这会形成多分辨率 Collapse Model：通信量仍远低于完整特征，但比单一 $\gamma$ 和 $\alpha$ 更忠实。

### 9.2 基于扰动理论的新误差路线

可以定义一个理想的 paired block 算子 $G_0$ 和真实算子 $G$：

$$
G=G_0+\Delta G.
$$

先用 Weyl、Hoffman--Wielandt 或相关矩阵扰动工具控制特征值变化，再把奇异值变化传递到归一化谱，最后使用带有最小质量或截断条件的熵连续性界。

一个可能的新误差结构是

$$
|\log r_{\mathrm{eff}}(G)-\log\widehat r|
\leq
C_1\|E\|
+C_2\operatorname{Var}_w(\gamma_i)
+C_3\operatorname{Var}_w(\alpha_i)
+C_4D_{\mathrm{spec}}(A,B)
+\varepsilon_{\mathrm{tail}}.
$$

这里的常数必须依赖谱间隙、截断阈值或最小归一化谱质量。该形式目前只是研究方向，不能在没有完整证明时写成定理。

### 9.3 从绝对误差保证转向排序保证

数据池选择真正需要的往往不是精确预测 $r_{\mathrm{merged}}$，而是正确比较两个候选数据集的边际增益。因此可以研究：

$$
\Pr\bigl(
\widehat\Delta(A,B_1)>\widehat\Delta(A,B_2)
\Rightarrow
\Delta(A,B_1)>\Delta(A,B_2)
\bigr).
$$

如果两个候选的真实增益间隔大于近似误差之和，就可以得到局部排序稳定性。相比普适的有效秩绝对误差界，这种结果更贴近 greedy selection 的实际用途，也更可能在合理假设下成立。

### 9.4 从普适证书转向可诊断的置信度

可以用以下信号判断摘要近似何时可信：

- $\gamma_i$ 的离散程度较小；
- $\alpha_i$ 的加权方差较小；
- 非配对残差 $\|E\|$ 较小；
- 两侧谱形状距离较小；
- 预测候选之间的 margin 足够大。

当这些量较差时，系统不必继续给出强预测，而可以回退到：

- 传输更多谱分桶摘要；
- 请求低秩 sketch；
- 对少量候选执行真实合并计算；
- 将结果标记为低置信度。

这种“预测器加失效诊断”比宣称一个实际上不成立的普适证书更可靠。

### 9.5 已成立的 rank-$L$ PSD Gram 误差界

对固定预处理后的来源 $j$，取 top-$L$ PSD 截断

$$
\widetilde G_j=B_j^\top B_j,\qquad E_j=G_j-\widetilde G_j\succeq0.
$$

令 $\tau_j=\operatorname{tr}\sqrt{E_j}$、
$\omega_j=\operatorname{tr}(E_j)$、$u_j=\operatorname{rank}(E_j)$。对集合 $S$ 定义

$$
T_S=\min\left\{\sum_{j\in S}\tau_j,
\sqrt{\min(d,\sum_{j\in S}u_j)\sum_{j\in S}\omega_j}\right\},
$$

$$
\varepsilon_S=\frac{T_S}{\operatorname{tr}\sqrt{\widetilde G_S}+T_S}.
$$

则 Fannes--Audenaert 连续性界给出确定性结论

$$
\left|\log R_G(G_S)-\log R_G(\widetilde G_S)\right|
\leq \Phi_{D_S}(\varepsilon_S).
$$

该结论不需要方向配对假设，并直接导出候选分数区间。若某候选下界严格高于所有竞争者上界，
则可认证该轮选择与 exact Full-Gram greedy 一致。证书只覆盖单个 greedy 步骤；有效秩目标一般
非单调、非次模，因此不能推出最终子集全局最优。

---

## 10. 与数据池压缩应用的关系

Collapse Model 的直接用途不是替代端到端训练，而是在训练前进行数据集级预筛选。典型流程是：

```text
候选数据源
→ 冻结编码器特征
→ 谱与子空间摘要
→ 估计边际表示覆盖
→ 排除高度冗余的数据源
→ 对缩小后的候选池进行真实训练或适配
```

适用场景包括：

- 只能从 $M$ 个数据源中采购或保留 $k\ll M$ 个；
- 原始数据或完整特征不便跨机构共享；
- 数据以流式方式持续到达；
- 存储、授权或适配预算有限；
- 穷举所有数据集组合并实际训练成本过高。

它直接回答的是：

> 哪些数据源在冻结表示空间中提供互补覆盖，哪些数据源主要重复已有方向？

它不能单独回答：

> 哪个数据组合一定取得最高的端到端下游准确率？

因此，Collapse 应当被视为低成本筛选器和冗余诊断器，而不是训练效果的替代评价器。

---

## 11. 论文最终应保留与撤回的内容

| 内容 | 最终定位 |
|---|---|
| $G_{A\cup B}=G_A+G_B$ | 无条件精确恒等式 |
| 成对方向的 $\lambda_{\pm,i}$ | 在 pairing/decoupling 条件下精确 |
| energy/directional collapse 机制 | 局部谱层面成立 |
| proportional-spectrum 闭式分解 | 在强假设下精确 |
| 允许 $r_A\neq r_B$ 的摘要公式 | 理论启发的经验近似 |
| superadditivity threshold | 理想模型下精确，一般数据上是诊断量 |
| 原 Theorem 1 | 当前结论无效，应撤回 |
| rank-$L$ PSD Gram sketch | 固定预处理和合法尾界下具有确定性 log-effective-rank 区间 |
| 自然池上的 alignment 增量 | 当前证据有限，不应宣称普遍优势 |
| clone attack | 人工高秩重复压力测试，只说明 alignment 在该构造中有用 |
| 冻结特征压缩 | 当前主要实证应用 |
| 来源监督的 lightweight adapter | 当前审计中整体不优于冻结 identity，不能作为数据消费收益证据 |
| 端到端预训练收益 | 尚未建立 |

---

## 12. 最终理论定位

### 12.1 一句话版本

> 合并特征矩阵的谱由个体谱能量与右奇异子空间几何共同决定；在成对、解耦方向条件下，这种相互作用具有精确的 $2\times2$ Gram 特征值表达，而实用 Collapse 公式是将逐方向能量比例和 alignment 压缩为全局摘要后的经验近似。

### 12.2 更完整的论文定位

Collapse Model 不应再被定位为“一般谱条件下具有普适误差证书的闭式理论”。更准确的定位是：

1. **解释性理论**：揭示合并谱中能量不平衡与方向重叠的不同作用，并在方向配对条件下给出精确局部分解。
2. **理想化解析模型**：在 proportional spectra 与 uniform alignment 下得到可解释的双分支熵公式。
3. **可认证计算近似**：一般谱上以 rank-$L$ PSD Gram sketch 为主方法，用尾部界控制谱熵误差并在区间分离时认证单步排序。
4. **经验基线与应用层筛选**：旧四标量公式仅作为机制消融；最终候选必须由独立的任务感知第二阶段验证。当前实现是“来源监督适配器 + 目标训练集交叉验证选择”，不是目标训练的适配器。

### 12.3 这一定位为何仍有价值

撤回原误差界会降低理论声明的强度，但不会使工作退化为没有结构的经验指标：

- 预测器不是任意回归式，而是从 Gram 谱相互作用推导出的结构化近似；
- 精确 block 分解给出了 alignment 必须进入合并谱分析的数学理由；
- energy collapse 与 directional collapse 对应不同的可解释失效机制；
- clone attack 表明单体 rank 无法在所有情形下替代相对几何；
- 427 个真实样本对上的相关性说明该近似可作为诊断量，但相对 $r_A+r_B$ 的排序增量很小，不能据此主张选择优势；
- summary-only 设计仍对应通信受限、流式和预算受限的数据源筛选需求，但不自动提供隐私保证。
- 三种子精确参照中，rank-$L$ 区间覆盖 288/288 个前缀，却得到 0/720 步证书和 0/288
  Full-Gram 有序前缀完全一致；这验证了“误差界”与“可用选择保证”必须分开报告。

因此，最终最稳健的贡献声明不是“我们证明了一般数据集合并有效秩的普适公式”，而是：

> 我们给出精确的局部谱几何与信息不足反例，并构造带确定性尾部界的 rank-$L$ Gram 近似；其通信收益需由等字节 Pareto 实验验证，任务效用则需由独立的两阶段实验检验，而不是由谱理论直接推出。当前来源监督适配器实验未通过相对冻结 identity 的效用门槛。

---

## 13. 后续理论工作的优先级

1. 形式化 pairing/decoupling 条件，并证明逐方向 block 定理。
2. 将 proportional-spectrum 闭式结果改写为独立、无矛盾的命题，并明确 $r_A=r_B$。
3. 删除原 Theorem 1，避免在 rebuttal 阶段提出未经证明的新普适界。
4. 保留具有相同 $r_A,r_B,\gamma,\alpha$ 但不同合并有效秩的显式反例，明确四标量不是充分统计量。
5. 测量真实数据中的 $\operatorname{Var}_w(\gamma_i)$、$\operatorname{Var}_w(\alpha_i)$ 与非配对残差 $\|E\|$，分析它们与预测误差的关系。
6. 研究分桶谱摘要或低秩 sketch，在通信成本与预测精度之间建立可验证的折中。
7. 优先研究选择排序稳定性与低置信度回退机制，而不是再次尝试过强的普适绝对误差界。
