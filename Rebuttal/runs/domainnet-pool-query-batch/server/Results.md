# Server replay and Clipart timing pilot

The four-seed batch replay reproduced the local frozen decisions and both result CSVs byte for byte. See the replay audit.

All times below are medians of three independent one-thread runs on Clipart / ResNet-50 at seed 20260911. The logistic timings start with resident projected features and include candidate screening and fitting through the validation winner. Feature extraction and test evaluation are outside the timed interval.

| Logistic policy | Fits | Seconds | Test Brier |
|---|---:|---:|---:|
| Sequential | 25 | 1.431 | 0.911268 |
| Fixed second batch | 25 | 1.443 | 0.911268 |
| MMD | 50 | 2.838 | 0.911268 |
| Full validation | 455 | 27.532 | 0.911268 |

All 12 runs match the archived query order and select the same winner. Maximum absolute validation and test Brier discrepancies are 1.58e-9 and 6.29e-10. The median sequential acquisition time is 3.25 ms. A single unit with serial execution does not establish a cross-domain or parallel speedup. Logistic timing and individual runs

The ridge comparison starts with resident projected features and precomputed rankings. It times block statistics, fits, temperature grid search when applicable, validation and winner selection. Selection uses the same 512 rows in both arms. The calibrated arm searches temperatures on the other 512 validation rows.

| Ridge policy | Fits | Uncalibrated seconds | Calibrated seconds | Search seconds | Calibrated test Brier |
|---|---:|---:|---:|---:|---:|
| A-opt | 50 | 0.065 | 1.038 | 0.973 | 0.791511 |
| MMD | 50 | 0.065 | 1.038 | 0.972 | 0.808526 |
| A-opt | 227 | 0.276 | 4.734 | 4.448 | 0.791511 |
| MMD | 227 | 0.277 | 4.737 | 4.459 | 0.791511 |
| Full validation | 455 | 0.542 | 9.509 | 8.966 | 0.791511 |

All 30 ridge runs reproduce the archived winners and test metrics. The temperature search accounts for roughly 94% of calibrated decision time. MMD 50 is 0.017015 worse in test Brier than calibrated full validation in this unit. Ridge timing and individual runs

## Ridge timing across six domains and two encoders

The full seed-20260911 run uses the same split-validation protocol and timing scope as the pilot. All 360 runs reproduce the archived winners, temperatures and test metrics. The table averages each unit's three-run median over the 12 encoder–domain units. Test Brier is the corresponding unit mean.

| Ridge policy | Fits | Uncalibrated seconds | Calibrated seconds | Search seconds | Calibrated test Brier |
|---|---:|---:|---:|---:|---:|
| A-opt | 50 | 0.070 | 1.114 | 1.039 | 0.725347 |
| MMD | 50 | 0.071 | 1.114 | 1.039 | 0.727958 |
| A-opt | 227 | 0.300 | 5.009 | 4.694 | 0.725498 |
| MMD | 227 | 0.296 | 5.064 | 4.747 | 0.725498 |
| Full validation | 455 | 0.586 | 10.047 | 9.430 | 0.725498 |

Running calibration and selection for the A-opt 50 shortlist takes roughly one ninth of the corresponding full procedure after rankings are available. Image feature extraction and ranking time remain outside the ridge measurements. [Summary CSV](ridge_timing_summary.csv) · [Per-unit medians](ridge_timing_per_unit.csv) · Run records · Server log

## Logistic timing across six domains and two encoders

At seed 20260911, all 288 timed runs matched the archived query order and winner. From resident projected features, the median across 12 encoder–domain units was 1.384 seconds for sequential 25, 1.390 seconds for two-batch 25, 2.666 seconds for MMD 50 and 25.702 seconds for full 455 at one fitting thread. At four fitting threads the medians were 0.793, 0.453, 0.824 and 7.554 seconds. The median paired speedup over MMD 50 was 1.04× for sequential and 1.86× for two-batch at four threads. The two-batch policy was faster in all 12 units. Sequential was faster in 11.

The two 25-fit policies selected the same model as each other in all 12 units at both thread settings. Their mean test Brier was 0.863951, with signed gaps of +0.00000043 to MMD 50 and +0.000194 to full validation. For both references, 0/6 domain averages and 0/12 encoder–domain units exceeded +0.005. In the four additional projection seeds, the archived replay found a +0.000254 mean Brier gap for two-batch 25 relative to sequential 25, with one encoder–domain–seed unit above +0.005. Live timing here covers one projection seed and excludes feature extraction.

[Verified report](Logistic_Extended_Results.md) · [Per domain](logistic_extended_per_domain.csv) · [Per encoder and domain](logistic_extended_per_unit.csv) · Server log · Verification log
