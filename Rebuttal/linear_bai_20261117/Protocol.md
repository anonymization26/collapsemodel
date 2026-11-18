# Linear BAI / experimental-design comparison, 2026-11-17

Frozen before any new baseline outcome is inspected. This is a follow-up on a
previously inspected test collection, NOT a new confirmatory held-out study.
No outcome-dependent hyperparameter choice or omission of failed cells is allowed.

## Shared test bed

Reuse only the freshly generated 2026-11-17 campaign: 48 DomainNet logistic units
(6 targets x 2 pool constructions x 4 seeds, d=128) and 8 Optdigits logistic units.
Candidate set: all 455 triples of 15 pools. Features for the *selection model* are
the 15-dimensional pool-incidence rows (distinct from classifier features).
Identical fitted classifier, fixed validation Brier, target labels, test split,
pool memberships and full-validation-selected reference for every method.
The same data are reused across seeds: do not treat 48 cells as 48 independent
datasets. Primary reporting splits DomainNet/Optdigits and hash/PC1 pools.

Quality uses a reveal-only replay of archived fresh validation losses. Full model
and data hashes are checked. Selection trajectories are locked before new test
predictions. Quality replay is NOT a fresh fit or a runtime benchmark.
Algorithms receive no full loss vector, test values, MMD score or target features
unless explicitly named MMD/Target-A-opt. All ties: rounded 12-decimal criterion,
then original candidate index. Final observed-loss winner uses unrounded losses.

## Equal-fit-budget suite

Budgets 25 (primary), 50 and 100 (sensitivity), no repeated candidate fits.
All incidence-model methods share the original rank-revealing 15-query basis.
OLS has no intercept/regularizer, exactly as the proposed method. No noise is
injected: repeated queries of our deterministic objective provide no new evidence.

- Proposed sequential LCB, coefficient .001; proposed two-batch; pure exploitation.
- Greedy discrete D-design: maximize x' V^-1 x (log determinant increment).
- Greedy discrete A-design: maximize x' V^-2 x/(1+x' V^-1 x).
  This estimates pool coefficients and is NOT the manuscript's Target-A-opt.
- Greedy discrete G-design: minimize max_z z'(V+xx')^-1 z over all 455 rows.
- Greedy discrete XY-design: minimize max_(z,z') (z-z')'(V+xx')^-1(z-z').
  These are exact one-step criteria, not globally optimal size-B subsets.
- LinGapE-greedy-FB-NR: predicted incumbent i; challenger j maximizes
  mu_i-mu_j+c*sqrt((x_i-x_j)'V^-1(x_i-x_j)); choose unqueried x minimizing
  this gap variance after adding xx'. Constants c=.001,.01,.1,1 are ALL reported,
  with .001 the scale-matched primary. No best-c selection on test outcomes.
  This uses Xu et al. (2018), Algorithm 2 and Eq.9 sampling, but changes their
  all-K initialization, ridge, confidence bound, repeated sampling and stopping.
  It is explicitly an adaptation, not a reproduction of their fixed-confidence
  guarantee. With K=455 the literal original initialization already exceeds B=25.
- Original MMD and Target-A-opt rankings; original random25 reference available
  from prior results. These do not get the 15-query incidence initialization.

For each selection policy, report three distinct terminal conventions:
1. observed: B queries using that policy; choose best observed validation loss.
2. recommend: B-1 policy queries, predict all candidate losses by OLS, fit the
   predicted minimum if unqueried (otherwise the next predicted unqueried row),
   then choose best observed of all B. This prevents penalizing pure exploration
   for failing to deploy its predicted winner. Apply to proposed policies too.
3. native_prediction: the OLS-predicted minimum after B-1 observations, irrespective
   of its final observed score; its model is guaranteed available within B fits.
   This isolates linear BAI's native recommendation from best-observed fallback.
MMD/Target-A-opt only use observed termination; neither assumes additive losses.

## Original OD-LinBAI at an admissible budget

Yang & Tan (2022), Algorithm 1, original Eq.1, T=200, d=15, K=455:
m=(200-120-8-4-2)/4=16.5; four independent-estimation phases retain 8,4,2,1.
Phase 1 uses a Steiner triple system on GF(2)^4 without zero, with 35 triples.
Uniform mass on these triples has the same information matrix as uniform mass
on all triples, hence is an exact G-optimal design, support35<=120.
Later phases use an independently checked G-optimal continuous design in their
actual span (uniform when independent; multiplicative D-opt iteration otherwise,
max leverage <=(1+1e-6)*rank, fail rather than silently accept nonconvergence).
Allocate ceil(m*pi_i), use only current-phase observations in weighted OLS,
retain lowest predicted losses. Track original pull count, distinct fits, native
winner and best-observed fallback separately. Memoized repeated deterministic
measurements do not create independent samples or restore the stochastic theorem.
Compare proposed sequential at both cap200 and EXACT matched distinct-fit count.
Do not claim an original OD-LinBAI result at B25: its Eq.1 gives negative m.
Do not rename a truncated RAGE-inspired heuristic as original RAGE; RAGE's
fixed-confidence rounding/noise assumptions are discussed as related work only.

## Validation and runtime

Before real outcomes: synthetic exact-additive recovery, brute-force matrix
checks for G/XY/A/D/LinGapE rules, coefficient-scale checks, unique-fit budget,
deployment availability, zero hidden-loss access, deterministic tie handling,
Steiner information-matrix equality and OD rank/allocation checks.

Report validation simple regret and exact full-validation winner match as direct
selection objectives; test Brier, accuracy, NLL and >+.005 Brier gaps as downstream
descriptive outcomes. No equivalence/significance claim based on that threshold.
Report cell-wise wins/ties/losses and domain-averaged paired differences; no
hyperparameter oracle masquerading as a method. No formal p-values over repeats.

Independent resident-feature wall time: 7 preselected hash/first-seed units (six
DomainNet, Optdigits), one fitting worker, BLAS1, three rotated repetitions.
Methods at B25: sequential-observed, batch-observed, exploit-observed,
D-recommend, A-recommend, G-recommend, XY-recommend, LinGapE(.001)-recommend,
LinGapE(.1)-recommend. Include basis/design computation, allocation, actual new
classifier fitting and validation. No shared fitted models or precomputed design
sequences in these independent runs; check selected IDs against replay.
Data load and input-hash time recorded separately. Common raw-image preprocessing
is unchanged from the prior campaign; do not reuse its timing as a new measurement.
All numerical selection/readout work is CPU-based; CUDA_VISIBLE_DEVICES=0 for
every server command, and any required GPU work is confined to physical GPU0.

## Sources

- Soare, Lazaric & Munos (2014), Best-Arm Identification in Linear Bandits:
  https://papers.neurips.cc/paper_files/paper/2014/hash/f8d84caae48546d0934d637ab54f7086-Abstract.html
- Xu, Honda & Sugiyama (2018), A fully adaptive algorithm for pure exploration
  in linear bandits, PMLR84:843-851: https://proceedings.mlr.press/v84/xu18d.html
- Fiez, Jain, Jamieson & Ratliff (2019), Sequential Experimental Design for
  Transductive Linear Bandits:
  https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html
- Yang & Tan (2022), Minimax Optimal Fixed-Budget Best Arm Identification in
  Linear Bandits, DOI10.52202/068431-0890:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/4f9342b74c3bb63f6e030d8263082ab6-Abstract-Conference.html
  Metadata cross-checked with OpenAlex. Algorithm1/Eq1 checked against publisher PDF.

Citation workflow used citation-management, Scientific Agent Skills (Kassis et
al., 2026), https://doi.org/10.48550/arXiv.2609.00065 (metadata checked 2026-11-17).
This is a tooling acknowledgement, not evidence for a scientific claim.
