# DomainNet ridge temperature calibration

All 12 T=1 gates passed. Maximum absolute Brier errors were 1.95e-09 on validation and 1.86e-09 on test.

Brier is lower-is-better. Accuracy is top-1. Signed gaps are relative to the 455-fit validation winner under the same arm. Random averages 20 orders per unit. Domain averages weight the two encoders equally, and the overall average weights all six domains equally.

Original winners use full validation for temperature fitting. Reselection compares both arms on the same held-out selection half of validation. Uniform Brier is 0.99.

## Overall results

| Budget | Method | Arm | Brier | Accuracy | Brier gap | Accuracy gap (pp) |
|---|---|---|---|---|---|---|
| 50 | target_a | original | 0.986705 | 36.06% | +0.000010 | +0.06 |
| 50 | second_moment_mmd | original | 0.986695 | 35.87% | +0.000000 | -0.13 |
| 50 | random | original | 0.986814 | 35.15% | +0.000119 | -0.85 |
| 50 | target_a | original_calibrated | 0.742272 | 36.06% | -0.001067 | +0.06 |
| 50 | second_moment_mmd | original_calibrated | 0.743470 | 35.87% | +0.000132 | -0.13 |
| 50 | random | original_calibrated | 0.752662 | 35.15% | +0.009324 | -0.85 |
| 50 | target_a | split_uncalibrated | 0.986718 | 36.37% | +0.000020 | -0.03 |
| 50 | second_moment_mmd | split_uncalibrated | 0.986695 | 36.43% | -0.000003 | +0.02 |
| 50 | random | split_uncalibrated | 0.986826 | 35.00% | +0.000127 | -1.41 |
| 50 | target_a | split_calibrated | 0.725347 | 37.63% | -0.000150 | -0.14 |
| 50 | second_moment_mmd | split_calibrated | 0.727958 | 37.61% | +0.002461 | -0.17 |
| 50 | random | split_calibrated | 0.742038 | 36.35% | +0.016541 | -1.42 |
| 227 | target_a | original | 0.986695 | 36.00% | +0.000000 | +0.00 |
| 227 | second_moment_mmd | original | 0.986695 | 36.00% | +0.000000 | +0.00 |
| 227 | random | original | 0.986720 | 35.75% | +0.000026 | -0.25 |
| 227 | target_a | original_calibrated | 0.743338 | 36.00% | +0.000000 | +0.00 |
| 227 | second_moment_mmd | original_calibrated | 0.743338 | 36.00% | +0.000000 | +0.00 |
| 227 | random | original_calibrated | 0.745899 | 35.75% | +0.002560 | -0.25 |
| 227 | target_a | split_uncalibrated | 0.986699 | 36.40% | +0.000000 | +0.00 |
| 227 | second_moment_mmd | split_uncalibrated | 0.986699 | 36.40% | +0.000000 | +0.00 |
| 227 | random | split_uncalibrated | 0.986732 | 35.65% | +0.000033 | -0.75 |
| 227 | target_a | split_calibrated | 0.725498 | 37.78% | +0.000000 | +0.00 |
| 227 | second_moment_mmd | split_calibrated | 0.725498 | 37.78% | +0.000000 | +0.00 |
| 227 | random | split_calibrated | 0.730224 | 37.37% | +0.004726 | -0.40 |
| 455 | full | original | 0.986695 | 36.00% | +0.000000 | +0.00 |
| 455 | full | original_calibrated | 0.743338 | 36.00% | +0.000000 | +0.00 |
| 455 | full | split_uncalibrated | 0.986699 | 36.40% | +0.000000 | +0.00 |
| 455 | full | split_calibrated | 0.725498 | 37.78% | +0.000000 | +0.00 |

## Per-domain calibrated results

| Domain | Budget | Method | Arm | Brier | Accuracy | Brier gap | Accuracy gap (pp) |
|---|---|---|---|---|---|---|---|
| clipart | 50 | target_a | original_calibrated | 0.663927 | 44.60% | -0.009899 | +0.46 |
| clipart | 50 | second_moment_mmd | original_calibrated | 0.666954 | 44.48% | -0.006871 | +0.34 |
| clipart | 50 | random | original_calibrated | 0.684269 | 43.10% | +0.010444 | -1.04 |
| clipart | 50 | target_a | split_calibrated | 0.646309 | 45.97% | +0.000000 | +0.00 |
| clipart | 50 | second_moment_mmd | split_calibrated | 0.654629 | 45.65% | +0.008320 | -0.32 |
| clipart | 50 | random | split_calibrated | 0.677238 | 43.55% | +0.030929 | -2.42 |
| clipart | 227 | target_a | original_calibrated | 0.673825 | 44.14% | +0.000000 | +0.00 |
| clipart | 227 | second_moment_mmd | original_calibrated | 0.673825 | 44.14% | +0.000000 | +0.00 |
| clipart | 227 | random | original_calibrated | 0.672264 | 44.40% | -0.001562 | +0.26 |
| clipart | 227 | target_a | split_calibrated | 0.646309 | 45.97% | +0.000000 | +0.00 |
| clipart | 227 | second_moment_mmd | split_calibrated | 0.646309 | 45.97% | +0.000000 | +0.00 |
| clipart | 227 | random | split_calibrated | 0.657414 | 45.21% | +0.011105 | -0.76 |
| clipart | 455 | full | original_calibrated | 0.673825 | 44.14% | +0.000000 | +0.00 |
| clipart | 455 | full | split_calibrated | 0.646309 | 45.97% | +0.000000 | +0.00 |
| infograph | 50 | target_a | original_calibrated | 0.827903 | 27.15% | +0.000528 | +0.05 |
| infograph | 50 | second_moment_mmd | original_calibrated | 0.835038 | 26.00% | +0.007663 | -1.10 |
| infograph | 50 | random | original_calibrated | 0.837331 | 26.38% | +0.009956 | -0.72 |
| infograph | 50 | target_a | split_calibrated | 0.825416 | 27.00% | +0.005827 | -0.83 |
| infograph | 50 | second_moment_mmd | split_calibrated | 0.830296 | 26.90% | +0.010707 | -0.93 |
| infograph | 50 | random | split_calibrated | 0.830406 | 26.85% | +0.010817 | -0.98 |
| infograph | 227 | target_a | original_calibrated | 0.827375 | 27.10% | +0.000000 | +0.00 |
| infograph | 227 | second_moment_mmd | original_calibrated | 0.827375 | 27.10% | +0.000000 | +0.00 |
| infograph | 227 | random | original_calibrated | 0.831691 | 26.46% | +0.004316 | -0.64 |
| infograph | 227 | target_a | split_calibrated | 0.819589 | 27.83% | +0.000000 | +0.00 |
| infograph | 227 | second_moment_mmd | split_calibrated | 0.819589 | 27.83% | +0.000000 | +0.00 |
| infograph | 227 | random | split_calibrated | 0.820816 | 27.72% | +0.001227 | -0.11 |
| infograph | 455 | full | original_calibrated | 0.827375 | 27.10% | +0.000000 | +0.00 |
| infograph | 455 | full | split_calibrated | 0.819589 | 27.83% | +0.000000 | +0.00 |
| painting | 50 | target_a | original_calibrated | 0.658367 | 47.41% | +0.000000 | +0.00 |
| painting | 50 | second_moment_mmd | original_calibrated | 0.658367 | 47.41% | +0.000000 | +0.00 |
| painting | 50 | random | original_calibrated | 0.681857 | 44.22% | +0.023490 | -3.20 |
| painting | 50 | target_a | split_calibrated | 0.643222 | 47.58% | -0.007351 | +0.56 |
| painting | 50 | second_moment_mmd | split_calibrated | 0.643222 | 47.58% | -0.007351 | +0.56 |
| painting | 50 | random | split_calibrated | 0.661250 | 46.20% | +0.010677 | -0.82 |
| painting | 227 | target_a | original_calibrated | 0.658367 | 47.41% | +0.000000 | +0.00 |
| painting | 227 | second_moment_mmd | original_calibrated | 0.658367 | 47.41% | +0.000000 | +0.00 |
| painting | 227 | random | original_calibrated | 0.676185 | 45.28% | +0.017818 | -2.13 |
| painting | 227 | target_a | split_calibrated | 0.650573 | 47.02% | +0.000000 | +0.00 |
| painting | 227 | second_moment_mmd | split_calibrated | 0.650573 | 47.02% | +0.000000 | +0.00 |
| painting | 227 | random | split_calibrated | 0.651223 | 47.15% | +0.000650 | +0.13 |
| painting | 455 | full | original_calibrated | 0.658367 | 47.41% | +0.000000 | +0.00 |
| painting | 455 | full | split_calibrated | 0.650573 | 47.02% | +0.000000 | +0.00 |
| quickdraw | 50 | target_a | original_calibrated | 0.958663 | 9.77% | +0.002970 | -0.15 |
| quickdraw | 50 | second_moment_mmd | original_calibrated | 0.955693 | 9.91% | +0.000000 | +0.00 |
| quickdraw | 50 | random | original_calibrated | 0.963116 | 9.09% | +0.007423 | -0.83 |
| quickdraw | 50 | target_a | split_calibrated | 0.959641 | 9.55% | +0.002641 | -0.95 |
| quickdraw | 50 | second_moment_mmd | split_calibrated | 0.957518 | 10.06% | +0.000518 | -0.44 |
| quickdraw | 50 | random | split_calibrated | 0.961950 | 9.40% | +0.004951 | -1.09 |
| quickdraw | 227 | target_a | original_calibrated | 0.955693 | 9.91% | +0.000000 | +0.00 |
| quickdraw | 227 | second_moment_mmd | original_calibrated | 0.955693 | 9.91% | +0.000000 | +0.00 |
| quickdraw | 227 | random | original_calibrated | 0.958277 | 9.47% | +0.002584 | -0.44 |
| quickdraw | 227 | target_a | split_calibrated | 0.957000 | 10.50% | +0.000000 | +0.00 |
| quickdraw | 227 | second_moment_mmd | split_calibrated | 0.957000 | 10.50% | +0.000000 | +0.00 |
| quickdraw | 227 | random | split_calibrated | 0.957780 | 10.03% | +0.000781 | -0.47 |
| quickdraw | 455 | full | original_calibrated | 0.955693 | 9.91% | +0.000000 | +0.00 |
| quickdraw | 455 | full | split_calibrated | 0.957000 | 10.50% | +0.000000 | +0.00 |
| real | 50 | target_a | original_calibrated | 0.629396 | 47.92% | +0.000000 | +0.00 |
| real | 50 | second_moment_mmd | original_calibrated | 0.629396 | 47.92% | +0.000000 | +0.00 |
| real | 50 | random | original_calibrated | 0.624428 | 49.70% | -0.004968 | +1.77 |
| real | 50 | target_a | split_calibrated | 0.582004 | 54.22% | +0.000000 | +0.00 |
| real | 50 | second_moment_mmd | split_calibrated | 0.584575 | 54.35% | +0.002571 | +0.12 |
| real | 50 | random | split_calibrated | 0.607938 | 52.06% | +0.025935 | -2.16 |
| real | 227 | target_a | original_calibrated | 0.629396 | 47.92% | +0.000000 | +0.00 |
| real | 227 | second_moment_mmd | original_calibrated | 0.629396 | 47.92% | +0.000000 | +0.00 |
| real | 227 | random | original_calibrated | 0.620134 | 49.67% | -0.009261 | +1.74 |
| real | 227 | target_a | split_calibrated | 0.582004 | 54.22% | +0.000000 | +0.00 |
| real | 227 | second_moment_mmd | split_calibrated | 0.582004 | 54.22% | +0.000000 | +0.00 |
| real | 227 | random | split_calibrated | 0.593611 | 53.15% | +0.011608 | -1.07 |
| real | 455 | full | original_calibrated | 0.629396 | 47.92% | +0.000000 | +0.00 |
| real | 455 | full | split_calibrated | 0.582004 | 54.22% | +0.000000 | +0.00 |
| sketch | 50 | target_a | original_calibrated | 0.715374 | 39.50% | +0.000000 | +0.00 |
| sketch | 50 | second_moment_mmd | original_calibrated | 0.715374 | 39.50% | +0.000000 | +0.00 |
| sketch | 50 | random | original_calibrated | 0.724971 | 38.39% | +0.009597 | -1.11 |
| sketch | 50 | target_a | split_calibrated | 0.695491 | 41.48% | -0.002019 | +0.37 |
| sketch | 50 | second_moment_mmd | split_calibrated | 0.697511 | 41.11% | +0.000000 | +0.00 |
| sketch | 50 | random | split_calibrated | 0.713447 | 40.04% | +0.015937 | -1.07 |
| sketch | 227 | target_a | original_calibrated | 0.715374 | 39.50% | +0.000000 | +0.00 |
| sketch | 227 | second_moment_mmd | original_calibrated | 0.715374 | 39.50% | +0.000000 | +0.00 |
| sketch | 227 | random | original_calibrated | 0.716840 | 39.25% | +0.001466 | -0.26 |
| sketch | 227 | target_a | split_calibrated | 0.697511 | 41.11% | +0.000000 | +0.00 |
| sketch | 227 | second_moment_mmd | split_calibrated | 0.697511 | 41.11% | +0.000000 | +0.00 |
| sketch | 227 | random | split_calibrated | 0.700496 | 40.97% | +0.002985 | -0.14 |
| sketch | 455 | full | original_calibrated | 0.715374 | 39.50% | +0.000000 | +0.00 |
| sketch | 455 | full | split_calibrated | 0.697511 | 41.11% | +0.000000 | +0.00 |

All arms are available in per_domain.csv and per_encoder_domain.csv.
