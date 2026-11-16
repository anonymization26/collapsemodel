# BloodMNIST held-out logistic results

Budget 50 of 455. Lower Brier is better.

| Method | Held-out Brier | Difference from A-opt |
|---|---:|---:|
| a_opt | 0.498567 | +0.000000 |
| target_energy | 0.503201 | +0.004634 |
| mmd | 0.497116 | -0.001451 |
| random_mean | 0.500335 | +0.001768 |

All gaps against A-opt are below the preset practical threshold of 0.005. The 23 shortlists shared 427 distinct fitted candidates. The held-out set consists of 570 rows from the cached validation split. The source pools were constructed from feature strata.
