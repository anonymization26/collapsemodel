# Abstract — OpenReview submission copy

> Paste the block below directly into OpenReview's *Abstract* field.
> Per OpenReview docs, math uses `$...$` for inline and `$$...$$` for display.
> Text matches the paper's `\begin{abstract}` block verbatim (inline math only,
> no display equations, exactly as in `main.tex`).
>
> If a particular OpenReview *preview* shows the raw `$...$` and chews up
> underscores into italics, that field is rendering as plain text rather than
> MathJax — the published abstract page renders MathJax and shows the formulas
> correctly. Verify by submitting (or by typing `$x^2$` into the preview to
> see whether it renders).

---

Given $M$ candidate datasets, which $k \ll M$ subset best preserves representational diversity when their features are merged — without access to labels or downstream tasks? We address this *label-free data-pool compression* problem by deriving the **Collapse Model**, a closed-form bridge between spectral geometry and compression decisions. The model predicts the merged effective rank ($r_{\mathrm{eff}}$) from a $2 \times 2$ Gram-block eigenstructure via two compression-relevant SVD channels (energy ratio $\gamma$, directional alignment $\alpha$), yielding an explicit superadditivity threshold $\rho_c(\alpha,\gamma)$ that drives a greedy compression algorithm using only *per-dataset summary statistics* — suitable for privacy-constrained or streaming settings where features cannot be shared. A third channel (intrinsic dimension $\mathrm{UID}$) characterizes individual dataset geometry and is used for encoder diagnostics but does not enter the merged-rank prediction formula. Across 427 real feature-matrix pairs from 5 encoder families and 2 modalities the formula attains Pearson $r = 0.97$. Collapse-guided compression is near-optimal ($\leq 1\%$ regret vs. exhaustive search), matches the closest published spectral-diversity baseline (Vendi-Greedy) on retention while requiring only summary statistics rather than full feature matrices, and strictly outperforms Facility Location / $k$-Medoids on downstream linear-probe accuracy (sign test $p < 10^{-3}$, Collapse exceeds Random's $95\%$ CI on the mean LP). The comparison reveals two distinct LP-relevant diversity axes (*spectral retention* captured by Collapse and *centroid coverage* captured by farthest-first traversal), clarifying when each signal dominates. The framework scales to $M = 43$ pools at up to $14\times$ compression and identifies a quantitative supervised / self-supervised spectral-geometry divide via a bimodal principal-angle distribution exposed in CLIP and DINO encoders. Code is available at https://anonymous.4open.science/r/collapsemodel-1DB6.

---

## Plain-text fallback (no math rendering required)

If you confirm the target field is plain-text only (the `$x^2$` smoke test stays
literal after submission), use the Unicode version below — every formula is
spelled out without `$...$`, so it survives any renderer.

Given M candidate datasets, which k ≪ M subset best preserves representational diversity when their features are merged — without access to labels or downstream tasks? We address this label-free data-pool compression problem by deriving the Collapse Model, a closed-form bridge between spectral geometry and compression decisions. The model predicts the merged effective rank (r_eff) from a 2×2 Gram-block eigenstructure via two compression-relevant SVD channels (energy ratio γ, directional alignment α), yielding an explicit superadditivity threshold ρ_c(α, γ) that drives a greedy compression algorithm using only per-dataset summary statistics — suitable for privacy-constrained or streaming settings where features cannot be shared. A third channel (intrinsic dimension UID) characterizes individual dataset geometry and is used for encoder diagnostics but does not enter the merged-rank prediction formula. Across 427 real feature-matrix pairs from 5 encoder families and 2 modalities the formula attains Pearson r = 0.97. Collapse-guided compression is near-optimal (≤1% regret vs. exhaustive search), matches the closest published spectral-diversity baseline (Vendi-Greedy) on retention while requiring only summary statistics rather than full feature matrices, and strictly outperforms Facility Location / k-Medoids on downstream linear-probe accuracy (sign test p < 10⁻³, Collapse exceeds Random's 95% CI on the mean LP). The comparison reveals two distinct LP-relevant diversity axes (spectral retention captured by Collapse and centroid coverage captured by farthest-first traversal), clarifying when each signal dominates. The framework scales to M = 43 pools at up to 14× compression and identifies a quantitative supervised / self-supervised spectral-geometry divide via a bimodal principal-angle distribution exposed in CLIP and DINO encoders. Code is available at https://anonymous.4open.science/r/collapsemodel-1DB6.
