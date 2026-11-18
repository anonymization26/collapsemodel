# OpenReview rebuttal

## Summary of revisions

**Scope of the revision.** We have narrowed the paper's claim to low-budget model selection among fixed source-pool combinations. This extends the original fixed-pool screening task by using early validation scores to allocate later fits.

**Main result.** On the freshly fitted DomainNet candidate set, sequential 25 and full validation at 455 fits have mean test Brier of 0.916072 and 0.916078. Sequential querying selects the same validation winner in 39/48 configurations, and 0/48 have a test Brier gap above +0.005. The configurations share data. Figure 1 explains the static and feedback-based selection routes. Figure 2 and Table 1 report the new results, including the Optdigits stress test and same-budget comparisons. Appendix E gives the protocols. The recorded results and verification scripts are available in the [reproducibility repository](https://github.com/anonymization26/collapsemodel/blob/a4f317a0b50fed229019e4559e88530d3ebc11c8/Rebuttal/README.md).

## Reply to HauM

**Scope.** We have narrowed the empirical claim to budgeted selection among fixed source-pool combinations, with Target A-opt evaluated as a geometric screen whose relative performance depends on the setting. We use Target A-opt to assess target-weighted prediction uncertainty under a shared linear model. The Bayesian interpretation and validation-error decomposition serve as diagnostics.

**Inverse weighting.** Holding candidates, features, and the 50-fit budget fixed, Target A-opt gives mean logistic test Brier 0.864104 versus 0.896467 for linear target energy and 0.863951 for second-moment MMD (Appendix D). This isolates a benefit of inverse weighting relative to target energy.

**Ridge calibration.** Validation-only temperature scaling reduces test Brier from 0.986695 to 0.743338 for the same ridge winners across 12 encoder–target configurations, with accuracy unchanged at 36.00%. The uniform 100-class baseline is 0.99. In split-validation reselection at 50 fits, the mean gaps to calibrated full validation are −0.000150 for Target A-opt, +0.002461 for MMD, and +0.016541 for random. The counts above +0.005 are 1/6, 2/6, and 5/6 domain averages. This favors A-opt in that calibrated setting. A pre-fit rule for choosing between the screens remains open.

**Ridge sensitivity and cost.** With resident features and precomputed rankings, calibrated Target A-opt 50 and MMD 50 each take 1.114 s versus 10.047 s for 455 (Appendix D). In the new six-domain ridge study, after calibration at fixed regularization, sequential 25 exceeds +0.005 relative to full validation in 0/6 domains at 128 dimensions and 3/6 at 256 dimensions. Validation tuning over three regularization values costs 75 fits for 25 candidates. The image-cost reversal in Section 5.3 concerns uncalibrated ridge.

**Independent data.** With a fixed 25-fit budget on Optdigits, sequential querying matches the full-validation Brier under hash partitioning (0.299730). One independently timed hash configuration takes 0.552 s versus 9.632 s for full validation from resident features. Under first-principal-component (PC1) partitioning, Brier is 0.406669 versus 0.339867 for full validation. The budget and protocol were fixed before the readout outcomes. Both constructions use artificial pools. Per seed, 169–267 of the 455 PC1 candidates omit a training class, compared with zero under hash partitioning. Class coverage may contribute to the performance difference. [Label-coverage audit](https://github.com/anonymization26/collapsemodel/blob/a4f317a0b50fed229019e4559e88530d3ebc11c8/Rebuttal/runs/fresh-20261117/posthoc_source_label_coverage.json)

**Candidate scale and stopping.** A static A-opt stress test preserves the exact top-50 at 4,060 constructed candidates and reduces ranking time from 2.35 s to 1.23 s. The retrospective leave-one-domain MMD stopping rule saves 2.08 fits on average versus fixed 50. Appendix E reports both checks. The new sequential budget evidence covers the 455-candidate family.

## Reply to wsqf

**Scope and contribution.** We have narrowed the contribution to using shared pool structure and validation feedback for budgeted selection over fixed source-pool combinations. Target A-opt applies the established transductive experimental-design criterion to fixed pool unions. The added comparisons examine when the additive loss approximation supports low-budget selection and when it breaks down.

**Same-budget baselines.** We replayed validation scores from the new fits to compare our policy with greedy experimental designs and a finite-budget, no-repeat adaptation of LinGapE. These methods use the same pool indicators and initial 15 fits. Baselines may spend the 25th fit on a predicted recommendation. Mean DomainNet test Brier is 0.916072 for sequential querying, 0.916716 for D-design, and 0.916678 for adapted LinGapE with the matched coefficient of 0.001. On Optdigits with PC1 pools, XY-design gives 0.345173 versus 0.406669 for sequential querying. Appendix E reports the prespecified LinGapE coefficients and the alternative termination rules.

**Seed variation and presentation.** Across four seeds, the mean paired Brier gap of sequential 25 to full validation ranges from −0.000077 to +0.000057. The seed repeats share data. Table 1 shows between-seed standard deviations and highlights the best mean at the shared 25-fit budget. The main text distinguishes the archived replay, fresh evaluations, and timing studies by name.

**Transport comparison.** We also compare raw and whitened sliced Wasserstein against MMD and A-opt on the same fixed pool unions (Appendix D). This comparison uses frozen embeddings. Direct comparisons with the named sample-level methods remain open.

## Reply to hbbQ

**Scope.** We have narrowed the claim to choosing a fitted model under a limited fit-and-validation budget when the training choices are predefined source-pool combinations, such as combinations of previously prepared batches.

**Role of geometry.** The static screens use two geometric quantities. The source Gram matrix measures a candidate’s feature coverage by direction. The target second moment measures target feature energy in those directions. Appendix A gives an equal-size, equal-trace example in which exact moment matching has higher target prediction risk under the shared linear model.

**Validation feedback.** The search policy estimates pool contributions from observed validation losses. When two candidate triples share two pools, their loss difference informs the estimated effect of replacing the third. Fifteen linearly independent queries identify the validation-loss surface under exact additivity and exact observations. An unqueried predicted winner still requires a fit.

**Quality at 25 fits.** With identical first 15 fits, using validation feedback for the final ten improves mean test Brier by 0.003599 over filling those slots by fixed MMD rank. In a replay on freshly fitted DomainNet candidates spanning six targets, four projection seeds, and two pool constructions, sequential 25 recovers the 455-fit validation winner in 39/48 configurations, and 0/48 have a test Brier gap above +0.005. Repeated configurations share data.

**Failure case.** On the Optdigits PC1 construction, fixed MMD at 25 fits matches full validation, while the pool-query Brier is 0.066802 higher. Per seed, 169–267 of 455 PC1 candidates omit a training class, compared with zero under hash partitioning. The comparison reflects changes in class coverage as well as potential pool interactions.

**Measured runtime.** From resident features, the median times for sequential 25, two-batch 25, and MMD 50 are 3.337, 3.284, and 6.228 s with one fitting worker, and 1.922, 1.154, and 2.069 s with four workers. Batching lets the ten final fits run in parallel. The two-batch policy increases mean test Brier by 0.000015, and removing the leverage bonus preserves 47 of 48 sequential winners. Timing runs refit candidates independently. In a separate single Clipart run per policy starting from local images, sequential 25, MMD 50, and full validation take 19.60, 23.89, and 76.48 s, respectively.
