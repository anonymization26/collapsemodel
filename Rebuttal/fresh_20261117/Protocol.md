# Fresh experiments for the pool-query narrative

Frozen design date: 2026-11-17. New data preparation, feature computation, supervised
fits, validation choices and timing on the user-authorized V100 server. CUDA_VISIBLE_DEVICES=0.
No prior feature cache or old validation/test outcome is an input. Existing method code
and public benchmark definitions may be reused. Previous published results informed
this design; this is a new replication, not a historically blinded method discovery.

## Data and fixed design

1. DomainNet cleaned official split files and image archives from https://ai.bu.edu/M3SDA/.
   Select 100 of 345 classes by SHA-256 order under seed 20261117, before seeing any
   prediction. In every domain select 1536 source rows, 512 unlabeled target-screen rows
   and 1024 target-validation rows from the official train split; 2048 test rows from
   the official test split. Use deterministic ID-hash ordering. Roles are disjoint within
   a domain. All six targets are reported. Every target has 15 eligible blocks of 512
   source images; every candidate contains three blocks (455 unions, 1536 training rows).
   Use new ImageNet-pretrained ResNet-50 frozen features; cache full 2048-dimensional
   representations so larger projected dimensions can be tested. Primary projection
   dimension 128, seeds 20261117 through 20261120, with pre/post row L2 normalization.
   Use two source constructions: hash blocks and blocks stratified by the first raw
   feature principal component computed from source features only. Their image inventory
   is identical. Pool membership is frozen before any readout result.
2. UCI Optdigits, freshly downloaded from its official repository. Official train rows
   supply 3000 source examples (15 pools of 200). Official test rows, collected from
   different writers, are deterministically split into 256 unlabeled target-screen,
   512 validation and the remaining 1029 final-test rows. Native 64-dimensional count
   features, divided by 16 then L2-normalized. Four seeds 20261117–20261120 and hash / source
   feature-PC1 pool constructions. Pools are artificial; this is not evidence for
   naturally defined source domains. Disclose this limitation even if results improve.

## Methods and checkpoints

Primary readout: multinomial logistic regression, C=1, no intercept, LBFGS, tol=1e-6,
max_iter=1000; convergence failure aborts rather than silently changes a setting.
Use fixed class vocabulary and aligned zero-filled absent-class probabilities.
Primary endpoint: test Brier of the validation-Brier winner. Report accuracy/NLL of
that same winner, without metric-specific reselection.

Use the unchanged full-rank 15-query initialization and lexical ties. Policies at 25:
sequential (exploration coefficient 0.001), two-batch, sequential with no exploration,
same initial 15 then fixed MMD additions, same initial 15 then random additions (20
predeclared permutations). Also initial15, static MMD25/50, A-opt25/50, random25 (20
orders), and full455. A method sees only queried validation scores. Save query trajectories
and selection SHA-256 before evaluating test predictions. Keep the full455 winner
validation-selected; a test oracle is never a deployed selector.

After selection is locked, diagnose additive prediction at 15 vs 25 on the same
unqueried-at-25 candidates. Report held-out RMSE, Spearman, R2, top-25 validation recall.
An all-candidate additive fit is only an explanatory diagnostic.

## Timing and robustness

Quality audits may reuse a fitted candidate across policies; this duration is NOT a
deployment speed claim. Time fresh independent executions separately for sequential25,
batch25, MMD25, MMD50, A-opt25 and full455, with 1 and 4 fitting workers, BLAS=1 per
worker, three repetitions, rotated policy order. Include ranking and acquisition costs;
report worker-sum time separately from elapsed wall time. Seed 20261117 and hash pools
are the timing setting. Keep test scoring outside decision time.

Ridge robustness: use lambda in {0.1,1,10}, temperatures exp(linspace(log(.01),log(100),81)),
first validation half for lambda/temperature selection, second half for candidate choice.
All methods use identical splits and pay three fits for tuned lambda. For DomainNet use
64/96/128/256/512 projected dimensions at seed 20261117, reporting each dimension rather
than selecting it on test outcomes. Uncalibrated, calibrated fixed-lambda and calibrated
tuned-lambda arms share the candidate-selection half. Report real computation costs.

## Analysis, failure reporting and scope

Report every target / construction / projection separately and aggregate repeats within
target. Differences > +0.005 to full-validation quality are descriptive exceedances, not
equivalence or significance tests. Seeds and shared pools are not independent datasets.
No dropping unfavorable targets, changing exploration or choosing budgets after test.
Keep downloads, feature extraction, cache loading, ranking, fitting, validation, tuning
and test auditing separated. Cold-image cost of adaptive queries requires an actual
pipeline execution; adding old feature times to new results is not allowed.

Data availability and integrity checks may adjust preparation before readouts, with an
amendment written before outcome access. No performance-based amendments are allowed.
