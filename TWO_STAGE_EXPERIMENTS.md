# Two-Stage Compression Experiment Plan

## 1. Experimental Objective

Test the operational claim that a label-free spectral first stage can reduce expensive
task-aware evaluations while preserving the final quality of selected data pools.

The main endpoint is not prediction correlation. It is the **compute-utility frontier**:
downstream utility achieved versus the number and cost of Stage-2 evaluations.

## 2. Experimental Units and Splits

Treat a whole dataset, domain, source, or controlled shard as one candidate pool. Build
candidate collections with at least 20 pools so that pre-screening is meaningful.

Use three collection types:

1. **Controlled synthetic pools:** known spectra, alignment, tail perturbations, and dense
   cross-coupling. These test mechanisms and failure diagnostics.
2. **Controlled real-feature pools:** natural datasets plus duplicates, augmentations,
   class-overlap variants, and sample-overlap variants at known overlap rates.
3. **Natural multi-source pools:** independently collected domains or corpora with no
   artificial copies. These determine whether the method has practical value beyond a
   stress test.

Split at the pool or domain level. Pair-level random splits are not sufficient because
pairs sharing one dataset are dependent. Report leave-one-dataset-out and
leave-one-domain-out results.

## 3. Encoders and Modalities

Run at least:

- vision: one supervised encoder, one self-supervised encoder, and one vision-language
  encoder;
- text: one sentence encoder and, if compute permits, one decoder-model embedding;
- optional cross-encoder transfer as an explicit out-of-distribution test.

Calibrate only on training pools. Never fit calibration parameters on test pools or on
different pairs containing a held-out test dataset.

## 4. Methods

### 4.1 Stage-1 selectors

- Random shortlist.
- Pool size or sample-count ranking.
- Effective-rank-only greedy.
- Additive \(r_A+r_B\) or Max-rank baseline.
- Centroid farthest-first.
- Facility Location or \(k\)-Medoids using full pool-level distances.
- Vendi-based greedy selection using full similarities when feasible.
- Collapse four-summary predictor.
- Collapse with spectrum-weighted alignment.
- Bucketed spectral summary with 4 and 8 buckets.
- Randomized low-rank sketch.
- Oracle full merged SVD, used as an upper reference rather than a deployable baseline.

Compare methods at matched communication cost where possible, and separately show the
full-information baselines.

### 4.2 Stage-2 evaluators

Use two levels:

- **Cheap proxy:** frozen backbone plus linear adapter or linear probe.
- **Data-consuming validation:** LoRA or another lightweight adapter trained on samples
  from the selected pools.

Fix the number of examples, optimization steps, data order policy, hyperparameters, and
seeds across selectors. Use at least three seeds for all headline results.

### 4.3 Oracles

- Exhaustive Stage-2 evaluation of every candidate when the collection is small.
- Beam search or a high-budget search when exhaustive subset enumeration is infeasible.
- True merged effective rank from full features for evaluating Stage-1 prediction error.

The downstream oracle, rather than the full-SVD oracle, defines the best final candidate.

## 5. Core Experiments

### E1. Mechanism and counterexample validation

Vary spectrum shape, energy ratio, principal-angle alignment, spectral-tail mismatch, and
cross-coupling independently.

Report:

- exact paired-block eigenvalue agreement under the stated assumptions;
- degradation after each assumption is violated;
- examples with identical compact summaries but different merged effective ranks;
- correlation between diagnostic residuals and absolute prediction error.

Success criterion: exact claims hold numerically to tolerance, and diagnostic error curves
increase monotonically enough to support a fallback policy. This experiment cannot validate
a universal error bound.

### E2. Natural redundancy detection

Construct real-feature pools with 0%, 10%, 25%, 50%, 75%, and 100% sample overlap, plus
independent augmentation and near-duplicate variants. Include naturally related sources
whose overlap is not constructed.

Report AUROC/AUPRC for redundant-pool detection, marginal effective-rank regret, selection
disagreement, and performance stratified by alignment and energy balance.

Success criterion: Collapse improves materially over rank-only baselines in natural or
moderately controlled high-overlap settings, not only at 100% cloning.

### E3. Shortlist recall

Evaluate shortlist sizes \(L\in\{3,5,10,20\}\) or equivalent fractions of the candidate
collection. Exhaustively run the Stage-2 evaluator for all candidates to identify the true
top candidates, then measure whether each Stage-1 method retains them.

Primary metrics:

- Recall@\(L\) of the best Stage-2 candidate;
- top-\(k\) candidate recall;
- normalized downstream regret after Stage 2;
- fraction of candidates removed before Stage 2.

Primary success criterion: at least 90% recall of the best candidate while eliminating at
least 50% of Stage-2 evaluations on multiple held-out collections. Report the full curve
rather than relying only on this threshold.

### E4. End-to-end two-stage selection

For each target task, run Stage 1, train adapters only for the shortlist, and choose the
final pool or pool mixture by target validation performance. Compare against evaluating all
candidates and against equal-cost baselines.

Report target test performance, Stage-2 GPU-hours, samples consumed, wall-clock time, and
total summary communication.

Success criterion: statistically indistinguishable downstream utility from exhaustive
evaluation with a substantial reduction in Stage-2 cost, or superior utility at matched
total compute.

### E5. Communication-accuracy frontier

Sweep summary fidelity:

- scalar rank and energy only;
- four-summary Collapse;
- top-\(k\) subspaces for multiple \(k\);
- 4/8-bucket spectra;
- low-rank sketches;
- full feature or kernel information.

Plot bytes transmitted against merged-rank prediction error, shortlist recall, and final
downstream regret. Include summary computation time.

Success criterion: identify a nontrivial operating point that dominates both scalar-only
summaries and full-feature transfer in the target deployment regime.

### E6. Confidence-triggered fallback

Define confidence using held-out calibration of spectral-shape distance, mode-wise
variation, and cross-coupling residual. Compare:

- always use compact Collapse;
- always use the richer sketch;
- use the richer sketch only for low-confidence cases.

Report coverage-risk curves, fallback rate, communication, prediction error, shortlist
recall, and downstream regret.

Success criterion: adaptive fallback approaches rich-sketch quality with materially lower
average communication.

## 6. Ablations

- Remove alignment and retain only rank/energy.
- Replace spectrum-weighted alignment with unweighted principal-angle alignment.
- Vary top-subspace dimension \(k\).
- Remove calibration or transfer calibration across encoders.
- Remove the confidence fallback.
- Change greedy aggregation order.
- Vary shortlist size and Stage-2 budget jointly.
- Compare supervised validation with self-supervised proxy selection.
- Separate clean, naturally redundant, and constructed-clone collections.

## 7. Statistical Protocol

- Use dataset-level cluster bootstrap confidence intervals.
- Use leave-one-dataset-out and leave-one-domain-out evaluation.
- Report mean, standard deviation, and 95% confidence intervals over at least three seeds.
- For paired selector comparisons, use paired bootstrap or a permutation test at the target
  collection level.
- Correct for multiple headline comparisons or designate one primary Collapse-versus-rank
  comparison before running the final evaluation.
- Report effect sizes and confidence intervals, not only p-values.

Do not use pair-level bootstrap intervals as evidence of cross-dataset generalization.

## 8. Resource-Normalized Reporting

For every method, report:

- raw samples or features accessed centrally;
- bytes transmitted per pool;
- summary extraction time;
- Stage-1 CPU/GPU time;
- number of Stage-2 candidates evaluated;
- Stage-2 GPU-hours and samples consumed;
- final downstream utility and regret.

This prevents a full-information baseline from appearing operationally equivalent to a
summary-only method, while still preserving it as a quality reference.

## 9. Minimum Viable Experiment Sequence

- [ ] M1. Implement pool-level splits and dataset-level cluster bootstrap.
- [ ] M2. Reproduce current predictor results without pair leakage.
- [ ] M3. Add rank-only, Vendi, centroid, and full-SVD baselines.
- [ ] M4. Run controlled overlap and natural redundancy experiments.
- [ ] M5. Measure shortlist recall using exhaustive cheap Stage-2 evaluation.
- [ ] M6. Run fixed-budget LoRA/adapter experiments on at least three targets.
- [ ] M7. Add bucketed/sketch summaries and communication accounting.
- [ ] M8. Implement confidence-triggered fallback.
- [ ] M9. Run LODO/domain holdout tests and final statistical comparisons.
- [ ] M10. Update the paper only after the end-to-end claims pass their criteria.

## 10. Decision Rules

Proceed with the two-stage method as a main paper contribution if E3 and E4 show that it
preserves downstream utility while reducing Stage-2 cost across natural held-out pools.

Retain it only as a diagnostic or appendix result if gains occur only on constructed clones,
if shortlist recall is unstable across encoders, or if rank-only/Vendi baselines match it at
equal cost.

Reject the compact predictor as an operational selector if its false-negative rate removes
the best downstream candidates more often than equal-cost random or rank-only screening.
