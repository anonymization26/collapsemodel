# Response to Reviewer yTYk

Thank you for the review. We agree that the manuscript overstates its theory. We will withdraw the invalid bound and separate the local identity from the empirical approximation.

## What remains supported

Although the universal bound is invalid, the main empirical findings remain supported. The exact paired-direction decomposition identifies energy imbalance and subspace overlap as distinct redundancy mechanisms. In the new dependence-aware 435-pair analysis, the predictor retains Pearson 0.961 and Spearman 0.974, with cluster intervals [0.942, 0.965] and [0.915, 0.989], and LODO Pearson never below 0.959. Compression experiments show strong spectral-diversity retention from summary-only selection; the clone attack exposes rank-only scores' inability to detect duplicated subspaces; and fixed-budget adaptation gives +1.42 points over random selection, with 95% CI [0.66, 2.17]. The intended use is dataset-level pre-screening when only $k\ll M$ sources can be retained, acquired, or adapted, for example under storage, streaming, acquisition, or feature-sharing constraints. Collapse estimates marginal frozen-representation coverage from compact spectral and subspace summaries before expensive training. It does not replace downstream validation or establish universal pretraining gains, but can reject redundant sources when exhaustive subset training or full feature sharing is impractical.

## Q1. Corrected theory and remaining contribution

The reviewer is correct that the global ratio $\gamma=\lVert H_B\rVert_*/\lVert H_A\rVert_*$ does not imply $\sigma_{B,i}=\gamma\sigma_{A,i}$.

If right-singular directions admit paired principal-vector bases and different pairs are decoupled, with $\alpha_i=\cos^2\theta_i$, pair $i$ has exact Gram eigenvalues

$\lambda_{\pm,i}=\frac{\sigma_{A,i}^2+\sigma_{B,i}^2\pm\sqrt{(\sigma_{A,i}^2-\sigma_{B,i}^2)^2+4\sigma_{A,i}^2\sigma_{B,i}^2\alpha_i}}{2}$.

High $\alpha_i$ suppresses the secondary branch (directional redundancy), while singular-value imbalance makes one branch negligible (energy dominance).

A common $r^*$ additionally requires $\sigma_{B,i}=\gamma\sigma_{A,i}$ and $\alpha_i=\alpha$. Proportional spectra imply $r_A=r_B$; hence the formula using global $\gamma$ while allowing $r_A\neq r_B$ is not exact. We will present it as a summary approximation replacing direction-dependent $\gamma_i$ and $\alpha_i$ by global values.

We confirm the counterexample to Theorem 1: the weighted perturbation does not reduce to an unweighted cancellation, and the entropy term is not generally controlled by $\log\kappa$. We will withdraw the theorem. The superadditivity threshold is exact only under proportional spectra and uniform alignment, and diagnostic otherwise. To avoid its singular form, we will directly compare the merged log-rank with the dominant dataset's log-rank.

The remaining theory is an exact local decomposition, an idealized closed form, and an empirically assessed approximation.

## Q2. Data-consuming validation and practical scope

In subspace-projection LP, the encoder is frozen; selected datasets determine a subspace and only the target readout is trained.

We added a fixed-budget diagnostic: each method selects three of nine candidates from label-free summaries. They directly train a shared supervised bottleneck adapter with dataset-specific heads for 600 steps, evaluated on five disjoint targets across three seeds.

Collapse obtains 63.82% mean accuracy versus 62.40% over five fixed random subsets: +1.42 points, with target-cluster 95% CI [0.66, 2.17]. This is feature-space adaptation.

Collapse and $r_A+r_B$ select the same subset here. Across 19 natural anchor tasks, they differ four times, with two wins and two losses. We therefore claim no stable average alignment advantage. Its supported role is a safeguard in the clone attack, where individual ranks barely change but duplicated subspaces defeat rank-only scores.

## Q3. Dependence-aware validation

We agree that ordinary pairwise bootstrap is inappropriate because pairs sharing endpoints are dependent.

On the retained 57 pairs, dataset-cluster bootstrap gives Pearson $r=0.928$, 95% CI [0.782, 0.966]; minimum leave-one-dataset-out (LODO) Pearson is 0.868.

We also built a new 435-pair set: 264 visual pairs from ResNet-50, ViT-B/16, CLIP ViT-L/14, and DINO ViT-S/16 plus 171 SBERT pairs, covering 31 datasets. This is additional validation, not an exact recomputation of the original 427 pairs.

Pearson is 0.961 with cluster 95% CI [0.942, 0.965]; Spearman is 0.974 with CI [0.915, 0.989]. LODO Pearson stays in [0.959, 0.962]. Within-probe Pearson is 0.996 (ResNet), 0.978 (ViT), 0.984 (CLIP), 0.960 (DINO), and 0.887 (SBERT). We will report both levels because aggregate metrics reflect cross-probe scales, and state weaker SBERT behavior as a limitation.

We will position Collapse as a summary-statistics method for frozen-representation compression, not a universal certificate.
