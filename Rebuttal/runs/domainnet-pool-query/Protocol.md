# DomainNet sequential pool-contribution replay

## Inputs and outcome access

Use the archived logistic readout at frozen projection seed `20260911`. The six DomainNet target domains and both encoders each have the same 455 feasible three-pool combinations for all methods within that unit. A query reveals one candidate's validation Brier and counts as one fit and validation evaluation in the simulated deployment. The archived models are reused in the replay.

The validation oracle checks archived file hashes and candidate IDs and releases one score per requested ID. Each policy has its own query ledger and can use only scores revealed on that ledger. The source JSON files contain all validation scores, so the restriction applies to policy access through the oracle. Geometry and pool membership are available for all candidates before any query. No validation outcomes from other domains are used to fit the geometry offset. Run selection and test audit in separate invocations. Save and hash the query traces and winners before audit. In audit, select the exhaustive reference by minimizing validation Brier over all 455 candidates, then read the archived test scores. This archive was previously used for other analyses, and the general pool-additivity design was informed by earlier validation diagnostics.

## Fixed budgets and policies

Use nested prefixes of 15, 25 and 50 queries for each policy. The winner at each budget is the queried candidate with lowest observed validation Brier, breaking ties by candidate ID.

- **Geometry + pool** and **pool only** share the first 15 queries within a unit. Sort candidates and pools by ID. Form the binary pool incidence matrix with 15 columns. Choose 15 rows by greedy pivoted Gram–Schmidt. At each step choose the unselected row with the largest squared residual norm, rounded to 12 decimals, breaking ties by ID. Check that the resulting design has rank 15.
- For **geometry + pool**, form a fixed offset `g = 0.005 * (rank_Aopt + rank_MMD) / (2 * 454)` using zero-based positions in the two archived rankings. The coefficient is set to the practical Brier tolerance and fixes the scale without label fitting. For **pool only**, set `g = 0`.
- After the common start, for each policy and query step fit ordinary least squares on revealed `validation Brier - g` using the queried rows. For unqueried candidate `x`, compute `g(x) + xᵀβ − 0.001 * sqrt(xᵀ(X_QᵀX_Q)⁻¹x)`. The final term favors less well-covered pool combinations. Its fixed coefficient is one fifth of the practical tolerance. Query the minimum, breaking ties after rounding the acquisition score to 12 decimals by candidate ID. Refit after every query. The leverage term is a search heuristic.
- **Fixed MMD** and **fixed A-opt** use the corresponding archived ranking prefixes. **Random** uses the same 20 archived permutations as the previous query-efficiency replay.

## Audit and primary contrast

Report test Brier and signed test Brier gap to the exhaustive-validation winner. Average random repeats within each unit, then average the two encoders within each target, then weight the six targets equally. Count targets with signed mean gap strictly above `+0.005`. The primary contrast is **geometry + pool at 25 vs fixed MMD at 50**. Report the signed gap between these two policies by target, its equal-target mean, and the number of targets with gap strictly above `+0.005`. Use this margin to describe practical closeness in the replay. Also report pool only at 25 against the same comparator to isolate the value of the geometry offset. The six targets share source pools, and the results cover one projection seed.

Check the 264 fixed-50 baseline winners against the previous frozen query-efficiency selections before audit. Then check their exhaustive reference IDs and per-domain test summaries against the previous replay. Record query sequences, scores, winners, and archive and freeze checks. Query counts measure avoided fits in replay. Wall-clock training savings require separate measurements.

From the submission-package root, execute the following in order.

```sh
shasum -a 256 Rebuttal/runs/domainnet-pool-query/Protocol.md Rebuttal/run_pool_query.py > Rebuttal/runs/domainnet-pool-query/freeze.sha256
python3 Rebuttal/run_pool_query.py select
python3 Rebuttal/run_pool_query.py audit
```
