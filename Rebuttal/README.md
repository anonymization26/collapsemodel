# ICLR 2027 rebuttal supplement

The revised anonymous manuscript and LaTeX sources are under [submission/paper/iclr2027](submission/paper/iclr2027/). The original anonymized artifact remains at the repository root. This supplement adds the pool-query method, numerical checks, result tables and timing drivers.

- [Four additional DomainNet projection seeds](runs/domainnet-pool-query-multiseed/Results.md) and [two-batch comparison](runs/domainnet-pool-query-batch/Results.md) use archived validation and test scores.
- [Logistic timing](runs/domainnet-pool-query-batch/server/Logistic_Extended_Results.md) uses fresh fits on the original projection seed, from resident projected features through validation selection.
- [Ridge calibration](runs/domainnet-ridge-temperature/Results.md) and [calibrated ridge timing](runs/domainnet-pool-query-batch/server/ridge_timing_summary.csv) use disjoint validation halves for temperature fitting and candidate selection. The timing starts from resident features and precomputed ranks.
- [Same-candidate transport](runs/domainnet-transport-screening/full/Results.md) measures ranking from resident features and evaluates choices using archived logistic scores.

Run `python3 reproduce.py verify` to verify the original archive or `python3 reproduce.py unpack` to extract its recorded results. Replay scripts that accept `--archive` can point to the extracted `results/target_conditioned/revision_20260917/revision_results/e2b_readout_projection_v1` directory. Fresh timing requires matching projected feature caches and target splits, as specified by the timing drivers and Appendix H.

To audit the frozen query choices in a disposable checkout after unpacking:

    mkdir -p Rebuttal/runs/replay-001/package
    ln -s ../../../../results Rebuttal/runs/replay-001/package/results
    python3 Rebuttal/run_pool_query.py audit
    python3 Rebuttal/run_pool_query_multiseed.py audit
    python3 Rebuttal/run_pool_query_batch.py audit

The timing CSVs are reported results. The live timing drivers require the frozen
feature inputs identified in the manuscript protocol.
