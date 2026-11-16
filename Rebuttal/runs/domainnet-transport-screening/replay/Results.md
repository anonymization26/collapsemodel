# DomainNet fixed-pool transport replay

Scope: 1 encoder-domain units at projection seed 20260911. The final candidate minimizes archived logistic validation Brier within its shortlist. Test Brier gaps use the winner selected by validation over all 455 candidates. Random averages 20 archived orders. Domain means weight encoders equally, and the overall mean weights included domains equally.

| Budget | Screening method | Test Brier | Gap to full | Domains over +0.005 |
|---:|---|---:|---:|---:|
| 50 | sliced_w2 | 0.919607 | +0.008339 | 1/1 |
| 50 | whitened_sliced_w2 | 0.919607 | +0.008339 | 1/1 |
| 50 | target_a | 0.911268 | +0.000000 | 0/1 |
| 50 | second_moment_mmd | 0.911268 | +0.000000 | 0/1 |
| 50 | random | 0.921937 | +0.010668 | 1/1 |
| 227 | sliced_w2 | 0.911268 | +0.000000 | 0/1 |
| 227 | whitened_sliced_w2 | 0.911268 | +0.000000 | 0/1 |
| 227 | target_a | 0.911268 | +0.000000 | 0/1 |
| 227 | second_moment_mmd | 0.911268 | +0.000000 | 0/1 |
| 227 | random | 0.913317 | +0.002049 | 0/1 |
| 455 | full | 0.911268 | +0.000000 | 0/1 |

Transport is a fixed-pool adaptation. Screen-side timing and exact A-opt checks are in screen_metadata.json. The archived comparator losses do not constitute a new end-to-end timing measurement.
