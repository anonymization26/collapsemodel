# Controlled-Overlap Two-Stage Experiment

## Setup

- Frozen features: ResNet-50, 1,000 cached samples per dataset.
- Candidate pool size: 500 samples.
- Candidate sources: 9 source datasets plus variants of the three highest-rank sources.
- Controlled sample overlap: 0%, 25%, 50%, 75%, and 100%.
- Stage-1 budget: select 3 pools.
- Stage-1 methods: Collapse, rank-only (`r_sum`), Centroid-FF, full-feature merged-rank
  Oracle, and 5 random selections.
- Stage 2: 256-dimensional shared adapter, 600 steps, 3 seeds, evaluated by ridge probes
  on DTD, EuroSAT, Flowers102, Oxford Pets, and Food101.
- Server: Ascend 910B4, NPU 7.

## Stage-1 Results

At overlap levels from 25% through 100%, the full-feature Oracle selected three distinct
families: STL-10, CIFAR-100, and CIFAR-10. Collapse and rank-only selection both retained
a duplicated family.

| Overlap | Collapse merged rank | Rank-only | Oracle | Collapse regret |
|---:|---:|---:|---:|---:|
| 0% | 884.34 | 886.57 | 886.57 | 2.23 |
| 25% | 832.73 | 832.18 | 876.82 | 44.09 |
| 50% | 776.01 | 773.18 | 876.69 | 100.67 |
| 75% | 718.51 | 713.31 | 876.69 | 158.17 |
| 100% | 657.82 | 650.49 | 876.69 | 218.87 |

Alignment made Collapse slightly less sensitive to increasing overlap than rank-only
selection, but the compact predictor did not prevent duplicate-family selection.

## Stage-2 Results

Mean accuracy over 3 seeds and 5 target datasets:

| Overlap | Collapse | Rank-only | Centroid-FF | Oracle | Random mean |
|---:|---:|---:|---:|---:|---:|
| 0% | 0.6305 | 0.6363 | 0.6272 | 0.6292 | 0.6337 |
| 25% | 0.6420 | 0.6379 | 0.6261 | 0.6470 | 0.6355 |
| 50% | 0.6341 | 0.6429 | 0.6257 | 0.6325 | 0.6298 |
| 75% | 0.6353 | 0.6428 | 0.6257 | 0.6325 | 0.6300 |
| 100% | 0.6362 | 0.6438 | 0.6257 | 0.6325 | 0.6279 |

Exploratory paired comparisons over overlap, seed, and target units:

- Collapse minus rank-only: -0.0051, paired t-test p = 0.077.
- Collapse minus Centroid-FF: +0.0095, p = 0.00059.
- Collapse minus full-rank Oracle: +0.0009, p = 0.742.
- Collapse minus the mean random selection: +0.0042, p = 0.070.

Across all method-overlap cells, merged effective rank had only moderate association with
Stage-2 accuracy: Spearman rho = 0.438 and Pearson r = 0.480.

These p-values are exploratory. The 75 paired units reuse targets and overlap-derived pools,
so they are not independent evidence of cross-dataset generalization.

## Interpretation

The result supports the two-stage architecture but not a strong claim for the compact
Collapse selector. Spectral pre-screening and downstream validation optimize different
quantities. In this controlled collection, Collapse did not beat rank-only selection and
did not reproduce the full-feature Oracle's redundancy avoidance. Stage 2 was necessary
because maximizing merged effective rank did not consistently maximize adapter accuracy.

The next experiment should test a richer bucketed spectrum or low-rank sketch as the
Stage-1 representation. Further expansion of the four-summary selector is not justified
unless it improves shortlist recall on held-out natural pools.

## Files

- `screening_manifest.json`: exact candidate families and selections.
- `screening_results.csv`: Stage-1 metrics for every overlap and method.
- `adaptation_results.csv`: per-overlap, method, seed, and target accuracy.
- `screen.log` and `adapt.log`: server execution logs.
