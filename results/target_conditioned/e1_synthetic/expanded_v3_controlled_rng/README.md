# E1 Target-Conditioned Synthetic Selection

Status: completed merge of 20 independently generated seed shards.

- Seeds: 20
- Configurations: 4320
- Shared-model rows: 168480
- Conditional-shift rows: 155520
- H2 synthetic gate: passed
- H2b same-information gate: passed
- Legacy dense-byte H5 pilot gate: failed
- Controlled target-sample RNG audit: passed

The merge rejects overlapping seeds, mismatched configurations, duplicate raw
rows, and incomplete Cartesian grids. H5 communication claims must use the
dedicated E5/H5 report, which also compares against packed symmetric Grams.
