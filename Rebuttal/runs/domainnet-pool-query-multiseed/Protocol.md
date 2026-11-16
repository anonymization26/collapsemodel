# DomainNet pool-query projection-seed replay

Replay every remaining archived projection seed, `20260917` through `20260920`, for the same six targets and two encoders. Each seed uses its own archived rankings and logistic validation/test losses. Use `20260911` to check that the driver reproduces every original query, revealed score and winner. Summaries for the additional seeds exclude that check seed.

Import the frozen search functions from `Rebuttal/run_pool_query.py`. Use the original 15/25/50 nested budgets, full-rank start, fixed 0.005 rank-based geometry offset, 0.001 leverage coefficient, per-query least-squares update and tie rules. The only difference between the adaptive policies is the offset. Use each seed's fixed A-opt and MMD prefixes and the 20 archived random orders. Check candidate identity and paired random orders against the original seed. Each policy receives only the validation score for its current query from the oracle. Run selection and audit separately. Hash all trajectories before opening the test archives. Use the validation-selected winner across all 455 candidates as the full reference.

The main comparison is pool only at 25 queries versus fixed MMD at 50. Also report geometry plus pool at 25 versus both comparators. Average random repeats within a unit, both encoders within a seed and domain, then the four seeds within each domain. Report the six equal-weight domain means, each of the 24 seed-domain contrasts, and counts with signed gaps strictly above +0.005 at both resolutions. Record the number of domains with any seed above the threshold. Compare geometry plus pool directly with pool only at 25, using a negative difference to indicate improvement. The archived projections use the same underlying data and candidate pools, so this audit measures sensitivity to projection, readout refitting and ranking changes on the benchmark.

In the test audit, cross-check fixed MMD, fixed A-opt and each random repeat at budgets 25 and 50 against their archived winner IDs and test Brier. Cross-check the mean full-validation Brier against the previously reproduced four-seed budget audit. Record per-trial, per-domain, overall and primary-contrast CSVs, the compressed query traces and logs. The replay counts supervised evaluations using existing archived fits.

Run from the submission-package root.

```sh
out=Rebuttal/runs/domainnet-pool-query-multiseed
shasum -a 256 "$out/Protocol.md" Rebuttal/run_pool_query_multiseed.py Rebuttal/run_pool_query.py > "$out/freeze.sha256"
python3 -B Rebuttal/run_pool_query_multiseed.py select
python3 -B Rebuttal/run_pool_query_multiseed.py audit
```
