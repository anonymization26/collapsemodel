# DomainNet query efficiency results

Δ = selected model test Brier minus test Brier of the 455-query validation winner (mean 0.863757). Domain averages weight both encoders equally, after averaging random repeats.

| Method | Policy | Mean queries | Mean Δ | Domains Δ > 0.005 |
|---|---|---|---|---|
| target_a | adaptive | 58.50 | +0.000742 | 0 |
| target_a | fixed50 | 50.00 | +0.000347 | 0 |
| second_moment_mmd | adaptive | 47.92 | -0.000083 | 0 |
| second_moment_mmd | fixed50 | 50.00 | +0.000194 | 0 |
| random | adaptive | 69.63 | +0.008149 | 4 |
| random | fixed50 | 50.00 | +0.007800 | 4 |

| Target | Method | Policy | Mean queries | Δ |
|---|---|---|---|---|
| clipart | target_a | adaptive | 62.50 | +0.001053 |
| infograph | target_a | adaptive | 25.00 | +0.002708 |
| painting | target_a | adaptive | 37.50 | +0.000000 |
| quickdraw | target_a | adaptive | 138.50 | +0.000689 |
| real | target_a | adaptive | 25.00 | +0.000000 |
| sketch | target_a | adaptive | 62.50 | +0.000000 |
| clipart | target_a | fixed50 | 50.00 | +0.000000 |
| infograph | target_a | fixed50 | 50.00 | +0.000000 |
| painting | target_a | fixed50 | 50.00 | +0.000000 |
| quickdraw | target_a | fixed50 | 50.00 | +0.002081 |
| real | target_a | fixed50 | 50.00 | +0.000000 |
| sketch | target_a | fixed50 | 50.00 | +0.000000 |
| clipart | second_moment_mmd | adaptive | 75.00 | +0.000000 |
| infograph | second_moment_mmd | adaptive | 75.00 | +0.000534 |
| painting | second_moment_mmd | adaptive | 37.50 | +0.000000 |
| quickdraw | second_moment_mmd | adaptive | 50.00 | +0.000000 |
| real | second_moment_mmd | adaptive | 25.00 | +0.000000 |
| sketch | second_moment_mmd | adaptive | 25.00 | -0.001033 |
| clipart | second_moment_mmd | fixed50 | 50.00 | +0.000000 |
| infograph | second_moment_mmd | fixed50 | 50.00 | +0.000534 |
| painting | second_moment_mmd | fixed50 | 50.00 | +0.000000 |
| quickdraw | second_moment_mmd | fixed50 | 50.00 | +0.000000 |
| real | second_moment_mmd | fixed50 | 50.00 | +0.000000 |
| sketch | second_moment_mmd | fixed50 | 50.00 | +0.000629 |
| clipart | random | adaptive | 75.97 | +0.011414 |
| infograph | random | adaptive | 79.10 | +0.003183 |
| painting | random | adaptive | 72.12 | +0.006491 |
| quickdraw | random | adaptive | 46.98 | +0.002887 |
| real | random | adaptive | 60.72 | +0.019262 |
| sketch | random | adaptive | 82.90 | +0.005658 |
| clipart | random | fixed50 | 50.00 | +0.011682 |
| infograph | random | fixed50 | 50.00 | +0.002969 |
| painting | random | fixed50 | 50.00 | +0.005341 |
| quickdraw | random | fixed50 | 50.00 | +0.002723 |
| real | random | fixed50 | 50.00 | +0.017952 |
| sketch | random | fixed50 | 50.00 | +0.006131 |

All folds selected threshold zero using the frozen calibration rule. Adaptive MMD saved 2.08 queries on average versus fixed 50. Adaptive A-opt and random both used more queries and had higher mean Brier than their fixed-50 counterparts. These are retrospective results at one projection seed.
