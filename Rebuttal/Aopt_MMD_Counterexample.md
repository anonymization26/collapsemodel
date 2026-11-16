# Equal-cost moment matching can reverse the target-risk ranking

Let the frozen feature space be two-dimensional with unit basis vectors $e_1,e_2$. The target feature distribution assigns probabilities $3/4$ and $1/4$ to $e_1$ and $e_2$. Its second moment is $C_T=\operatorname{diag}(3/4,1/4)$, which is also the empirical moment of a target set with three copies of $e_1$ and one of $e_2$.

Consider source combinations $A$ and $B$ with eight unit-norm feature rows each. Combination $A$ has six rows equal to $e_1$ and two equal to $e_2$. Combination $B$ has five and three, respectively. Thus $G_A=\operatorname{diag}(6,2)$ and $G_B=\operatorname{diag}(5,3)$, both with trace eight. They can be realized as unions of two disjoint four-row blocks drawn from a common library. The shared block has three $e_1$ rows and one $e_2$ row. The two alternative blocks have counts $(3,1)$ and $(2,2)$. Repeated feature vectors can belong to distinct samples.

For the paper's squared second-moment MMD, $f_M(S)=\|G_S/8-C_T\|_F^2$. Combination $A$ matches the target moment exactly, giving $f_M(A)=0$. The difference for $B$ is $\operatorname{diag}(-1/8,1/8)$, so $f_M(B)=1/32$.

Under the shared Gaussian linear model with prior $\beta\sim\mathcal N(0,I)$ and unit independent observation noise, the posterior covariance after observing source labels is $(I+G_S)^{-1}$. For the posterior-mean predictor under squared loss, integrating over the coefficient posterior and an independent target observation gives $R_T(S)=1+\operatorname{tr}[C_T(I+G_S)^{-1}]$. Consequently,

$$
f_A(A)=\frac{3/4}{7}+\frac{1/4}{3}=\frac4{21},
\qquad
f_A(B)=\frac{3/4}{6}+\frac{1/4}{4}=\frac3{16}.
$$

Hence $R_T(A)-R_T(B)=1/336>0$, while $f_M(A)<f_M(B)$. The one-row transfer from direction 1 to direction 2 raises the direction-1 contribution by $1/56$ and reduces the direction-2 contribution by $1/48$. The latter dominates because the second direction starts with less source information. All matrices are full rank. This establishes a strict separation under the paper's shared linear model. The classification results are a separate empirical question.

| Combination | Samples | Gram trace | Squared MMD | A-opt variance | Predictive risk |
|---|---:|---:|---:|---:|---:|
| A | 8 | 8 | 0 | 4/21 ≈ 0.190476190 | 25/21 |
| B | 8 | 8 | 1/32 | 3/16 = 0.187500000 | 19/16 |

Run `python3 Rebuttal/verify_aopt_mmd_counterexample.py` from the package root. The script constructs the Gram matrices from the rows and verifies the strict inequalities using rational arithmetic. See `runs/aopt-mmd-counterexample/verification.log`.

## Rebuttal-ready wording

> Target A-opt and second-moment MMD can rank equal-cost pool combinations differently, even when one combination matches the target moment exactly. In two dimensions, let the target second moment be diag(3/4, 1/4), and consider eight unit-norm source features with Gram matrices diag(6, 2) and diag(5, 3). Both have trace eight. Their squared MMD scores are 0 and 1/32, respectively. Under our shared linear model with unit prior precision and noise variance, their target-weighted posterior variances are 4/21 and 3/16, respectively. The exact moment match therefore incurs higher target predictive risk by 1/336. This example isolates the effect of inverse weighting under the assumed model. Our classification experiments separately assess whether that distinction yields a practical advantage over MMD.
