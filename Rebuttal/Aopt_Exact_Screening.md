# Exact top-k A-opt screening from additive pool bounds

This is a proposed extension for scaling fixed-pool screening. The guarantee concerns the A-opt shortlist under exact arithmetic. Real-data pruning rates and timing remain to be measured.

Let $P_S=\lambda I+\sum_{j\in S}G_j\succ0$ with $\lambda>0$, and let $C\succeq0$ be the target second moment. The A-opt score is $f(S)=\operatorname{tr}(CP_S^{-1})$. Choose a fixed positive definite anchor $A$ without using validation or test labels, and define $W=A^{-1}CA^{-1}$ and $D_S=P_S-A$.

The resolvent identity, applied twice, gives
$$
P_S^{-1}=A^{-1}-A^{-1}D_S A^{-1}+A^{-1}D_SP_S^{-1}D_S A^{-1}.
$$
Taking the trace against $C$ gives the exact decomposition
$$
f(S)=\underbrace{\operatorname{tr}(CA^{-1})-\operatorname{tr}(WD_S)}_{L(S)}
  +\underbrace{\operatorname{tr}(WD_SP_S^{-1}D_S)}_{R(S)}.
$$
The remainder is $\|P_S^{-1/2}D_SW^{1/2}\|_F^2\ge0$. Therefore $L(S)\le f(S)$, and $L(S)$ is a constant minus $\sum_{j\in S}\operatorname{tr}(WG_j)$. Compute the block terms once, then use scalar sums to screen candidate combinations.

Sort candidates by their lower bounds. Maintain the current best $k$ exactly scored candidates, ordered by score and candidate ID. Once $k$ exact scores exist, stop when the next lower bound is **strictly** greater than the current kth exact score. Every remaining true score is above the kth score, so the returned shortlist and its ordering equal exhaustive A-opt. Strict inequality retains ties for exact scoring. Floating-point implementations should handle near-boundary cases conservatively. A proof-level floating-point certificate requires validated bounds or error envelopes. Ordinary float64 agreement is an empirical check.

## How this relates to MMD

For an equal-cost inventory with $n$ rows per union, choose the reference $A=\lambda I+nC$ and write $D_S=G_S-nC$. Squared MMD is $\|D_S\|_F^2/n^2$, while
$$
f(S)-\operatorname{tr}(CA^{-1})
=-\operatorname{tr}(WD_S)+R(S),
\qquad
0\le R(S)\le \frac{\|W\|_{\mathrm{op}}}{\lambda_{\min}(P_S)}\|D_S\|_F^2.
$$
The upper bound follows from the squared Frobenius norm expression and submultiplicativity with operator norms. A candidate beats the exact moment match whenever its directional gain $\operatorname{tr}(WD_S)$ strictly exceeds this upper bound. MMD captures the magnitude of the deviation. The A-opt comparison also depends on its direction. A moment-matched reference need not occur in the candidate family for the lower-bound algorithm.

If $C=cI$ and the candidate Grams have equal trace $nc d$, take $A=\mu I$ with $\mu=\lambda+nc$ and $w=c/\mu^2$. The first-order term vanishes because $\operatorname{tr}(D_S)=0$. If $\|D_S\|_{\mathrm{op}}\le\delta\mu$ with $0\le\delta<1$, then
$$
\frac{wn^2}{\mu(1+\delta)} f_M(S)
\le f(S)-f(A)
\le \frac{wn^2}{\mu(1-\delta)} f_M(S),
$$
where $f(A)=\operatorname{tr}(CA^{-1})$. This follows by sandwiching $P_S^{-1}$ between the scalar inverse bounds. With exact target isotropy and a tight precision spectrum, A-opt excess risk is approximately proportional to squared MMD. The bounds certify agreement for sufficiently separated candidate distances. They permit rank changes at close margins. Numerical sanity check on 100 noncommuting positive-definite cases gave maximum identity error $5.68\times10^{-14}$.

## Acceptance criteria for a scaling claim

Check shortlist identity at the primary budgets, including an adversarial tie case. Compare total wall time and solve counts with a strong exact implementation at the same matrix dimension, threads, inventory and budgets. Charge anchor construction, scalar bounds, ordering, and exact evaluations. The sorted scan still constructs all cheap candidate bounds, and no worst-case reduction in exact solves is guaranteed. Report geometry compute separately from subsequent supervised fitting counts. Validation and test Brier are invariant if the exact shortlist and downstream procedure are preserved.
