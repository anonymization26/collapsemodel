# Two-Stage Label-Free Data-Pool Compression

## 1. Problem

Suppose there are many candidate data pools \(\mathcal D_1,\ldots,\mathcal D_M\), but the
budget allows only a small subset to be consumed by downstream training or adaptation.
Labels may be unavailable, and moving all raw samples or frozen features to a central
node may be infeasible.

The current Collapse theory characterizes the local Gram-spectrum interaction between
two frozen representation pools. Its strongest practical use is therefore not to replace
downstream evaluation, but to cheaply remove spectrally redundant candidates before a
task-aware selection stage.

## 2. Core Idea

Use a two-stage funnel:

1. **Stage 1: label-free spectral pre-screening.** Compute per-pool summaries under one
   frozen encoder and use the Collapse Summary Predictor to reject pools whose marginal
   representation-space contribution is small.
2. **Stage 2: task-aware validation.** Evaluate only the surviving shortlist with a small
   but real data-consumption procedure, such as a linear adapter or LoRA run under a fixed
   compute budget, and select the final subset by validation performance.

The division of labor is explicit:

- Stage 1 optimizes inexpensive redundancy removal and candidate recall.
- Stage 2 resolves information that spectral summaries cannot identify, including task
  relevance, class balance, label noise, sample difficulty, and optimization effects.

The proposed contribution is the **combination** of exact local Gram geometry and a
summary-based first-stage selector. It is not a claim that effective rank is a sufficient
statistic for downstream utility.

## 3. Stage 1: Spectral Pre-Screening

### 3.1 Inputs

For each pool \(\mathcal D_i\), extract a frozen feature matrix \(H_i\) using the same
encoder, feature dimension, preprocessing, centering rule, and normalization rule.

Each pool exports a summary package:

- sample count and feature energy;
- effective rank or the spectral entropy needed to compute it;
- the top-\(k\) right singular subspace;
- optionally, bucketed spectral mass or a randomized low-rank sketch;
- diagnostic metadata identifying encoder and normalization versions.

The top-\(k\) subspace is not a scalar summary. Communication cost and possible privacy
leakage must therefore be measured rather than described as negligible.

### 3.2 Marginal score

For a currently selected set \(S\), estimate the marginal spectral contribution of a
candidate pool \(i\):

\[
\widehat{\Delta}_{\mathrm{spec}}(i\mid S)
=\widehat r_{\mathrm{eff}}(H_{S\cup\{i\}})
-\widehat r_{\mathrm{eff}}(H_S).
\]

The first implementation may aggregate the selected set into one running summary and use
the pairwise Collapse Summary Predictor. Later variants can compare scalar summaries,
bucketed spectra, and low-rank sketches under the same communication budget.

### 3.3 Selection rule

Run greedy pre-screening until one of the following holds:

- the shortlist reaches a fixed size \(L\);
- the cumulative sample or storage budget is reached;
- every remaining marginal score is below a threshold \(\tau\).

To preserve potentially useful outliers, use a recall-oriented policy: retain the top
\(L\) candidates and all candidates whose score lies within a tolerance of the boundary.
Stage 1 should be evaluated primarily by whether it discards the eventual best candidates,
not only by its own effective-rank objective.

### 3.4 Confidence and fallback

Attach a low-confidence flag when one or more assumptions are strongly violated:

- high variation among mode-wise energy ratios \(\gamma_i\);
- high variation among mode-wise alignments \(\alpha_i\);
- large normalized spectral-shape distance;
- large pairing or cross-coupling residual;
- incompatible encoder, modality, or feature normalization.

Low-confidence candidates should pass through to Stage 2 or be scored with a richer sketch.
This flag is an empirical diagnostic, not a certified error bound.

## 4. Stage 2: Task-Aware Validation

Stage 2 consumes actual samples from each shortlisted pool under a fixed per-candidate
budget. The default protocol is:

1. initialize the same pretrained backbone for every candidate;
2. train a linear adapter or LoRA module for a fixed number of samples, steps, and seeds;
3. evaluate on a held-out target validation set;
4. select the final pool subset by mean validation performance, with compute and sample
   usage included in the objective or reported separately.

If labels are unavailable during selection, Stage 2 can use a fixed self-supervised proxy
loss and reserve labels strictly for final evaluation. The paper must distinguish this
setting from supervised validation.

## 5. Intended Deployment Scenarios

- federated or multi-institutional data acquisition, where each site exports summaries;
- selecting among many public corpora before expensive pretraining or adaptation;
- continual-learning memory management at the granularity of tasks or data sources;
- detecting copied, augmented, or heavily overlapping data releases;
- reducing the number of candidate mixtures entering costly downstream evaluation.

The method is less suitable when candidate pools use incompatible encoders, sample-level
selection is required, task relevance dominates redundancy, or full features and training
labels are already cheap to access.

## 6. Claims Supported by the Design

If validated, the paper may claim that:

- spectral pre-screening reduces the number of expensive Stage-2 evaluations;
- the Collapse interaction score improves redundancy detection over additive rank-only
  summaries in high-overlap regimes;
- a two-stage selector can approach exhaustive task-aware selection at lower compute;
- confidence diagnostics identify cases where the compact summary should not be trusted.

The paper should not claim:

- a universal guarantee on merged effective rank from four summaries;
- that spectral diversity alone determines downstream accuracy;
- end-to-end training gains without experiments that actually consume selected data;
- privacy protection without a formal or empirical leakage analysis.

## 7. Primary Research Questions

1. How much Stage-2 compute can spectral pre-screening save at fixed final utility?
2. Does alignment information outperform rank-only summaries on natural redundancy, not
   only constructed clone attacks?
3. What shortlist size preserves the best task-aware candidates with high probability?
4. When do the predictor diagnostics reliably identify summary-model failure?
5. What accuracy is gained by transmitting richer summaries, and at what communication
   and privacy cost?
