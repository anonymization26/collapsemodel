# DomainNet fixed-pool transport replay

Scope: 12 encoder-domain units at projection seed 20260911. The final candidate minimizes archived logistic validation Brier within its shortlist. Test Brier gaps use the winner selected by validation over all 455 candidates. Random averages 20 archived orders. Domain means weight encoders equally, and the overall mean weights included domains equally.

| Budget | Screening method | Test Brier | Gap to full | Domains over +0.005 |
|---:|---|---:|---:|---:|
| 50 | sliced_w2 | 0.866406 | +0.002649 | 1/6 |
| 50 | whitened_sliced_w2 | 0.865342 | +0.001585 | 0/6 |
| 50 | target_a | 0.864104 | +0.000347 | 0/6 |
| 50 | second_moment_mmd | 0.863951 | +0.000194 | 0/6 |
| 50 | random | 0.871557 | +0.007800 | 4/6 |
| 227 | sliced_w2 | 0.864193 | +0.000436 | 0/6 |
| 227 | whitened_sliced_w2 | 0.863674 | -0.000083 | 0/6 |
| 227 | target_a | 0.863757 | +0.000000 | 0/6 |
| 227 | second_moment_mmd | 0.863757 | +0.000000 | 0/6 |
| 227 | random | 0.865482 | +0.001725 | 1/6 |
| 455 | full | 0.863757 | +0.000000 | 0/6 |

Transport is a fixed-pool adaptation. Screen-side timing and exact A-opt checks are in screen_metadata.json. The archived comparator losses do not constitute a new end-to-end timing measurement.
