# DomainNet resident-feature logistic timing

Six target domains and two encoders at projection seed 20260911. Each unit has three independent
runs of each policy in rotated order at one and four workers. Within a policy run, each candidate
is fitted independently. Timings start at resident projected features and end at the validation winner.
Four-worker runs dispatch fitting tasks to four threads sharing resident data, with one BLAS thread
per fit. Sequential acquisition fits its
first 15 candidates in parallel and then fits each dependent candidate in order. Batch acquisition
fits its initial 15 and remaining 10 in two separate waves. MMD50 and Full455 each fit one wave.
Worker setup, feature extraction, and test scoring are outside the timed interval.

Environment: Python 3.11.11, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, Linux-5.10.134-19.3.1.al8.x86_64-x86_64-with-glibc2.32.

Medians are taken across repeats before aggregating across encoder and domain units. Brier scores
average repetitions and encoders within each domain, then weight the six domains equally.
Ratios are medians of paired unit ratios. Counts use a strictly greater than +0.005 signed gap.

| Workers | Policy | Fits | Median seconds [unit range] | Speedup vs full | Speedup vs MMD50 | Test Brier | Gap vs full | Gap vs MMD50 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | sequential25 | 25 | 1.384 [1.267, 1.524] | 18.95× | 1.92× | 0.863951 | +0.000194 | +0.000000 |
| 1 | batch25 | 25 | 1.390 [1.274, 1.517] | 18.88× | 1.91× | 0.863951 | +0.000194 | +0.000000 |
| 1 | mmd50 | 50 | 2.666 [2.432, 2.857] | 9.91× | 1.00× | 0.863951 | +0.000194 | +0.000000 |
| 1 | full455 | 455 | 25.702 [23.371, 28.027] | 1.00× | 0.10× | 0.863757 | +0.000000 | -0.000194 |
| 4 | sequential25 | 25 | 0.793 [0.727, 0.899] | 9.79× | 1.04× | 0.863951 | +0.000194 | +0.000000 |
| 4 | batch25 | 25 | 0.453 [0.403, 0.505] | 17.01× | 1.86× | 0.863951 | +0.000194 | +0.000000 |
| 4 | mmd50 | 50 | 0.824 [0.750, 0.901] | 9.33× | 1.00× | 0.863951 | +0.000194 | +0.000000 |
| 4 | full455 | 455 | 7.554 [6.841, 8.269] | 1.00× | 0.11× | 0.863757 | +0.000000 | -0.000194 |

| Workers | Policy | Gap vs full > +0.005 (domains, encoder–domains) | Gap vs MMD50 > +0.005 (domains, encoder–domains) |
|---:|---|---:|---:|
| 1 | sequential25 | 0/6, 0/12 | 0/6, 0/12 |
| 1 | batch25 | 0/6, 0/12 | 0/6, 0/12 |
| 1 | mmd50 | 0/6, 0/12 | 0/6, 0/12 |
| 1 | full455 | 0/6, 0/12 | 0/6, 0/12 |
| 4 | sequential25 | 0/6, 0/12 | 0/6, 0/12 |
| 4 | batch25 | 0/6, 0/12 | 0/6, 0/12 |
| 4 | mmd50 | 0/6, 0/12 | 0/6, 0/12 |
| 4 | full455 | 0/6, 0/12 | 0/6, 0/12 |

| Policy | Median four-worker speedup [unit range] | Same query order | Same winner | Maximum absolute Brier change |
|---|---:|---:|---:|---:|
| sequential25 | 1.74× [1.65, 1.80] | 36/36 | 36/36 | 0 |
| batch25 | 3.10× [3.01, 3.16] | 36/36 | 36/36 | 0 |
| mmd50 | 3.23× [3.15, 3.26] | 36/36 | 36/36 | 0 |
| full455 | 3.40× [3.36, 3.42] | 36/36 | 36/36 | 0 |

| Workers | Median batch speedup vs sequential [unit range] | Batch minus sequential Brier | Domains > +0.005 | Same winner |
|---:|---:|---:|---:|---:|
| 1 | 1.00× [0.98, 1.01] | +0.000000 | 0/6 | 36/36 |
| 4 | 1.77× [1.69, 1.85] | +0.000000 | 0/6 | 36/36 |

All 24 unit decision files passed identity and SHA-256 checks. All 288 timed runs passed the archive score gates. Maximum validation and test score errors were 9.9e-08 and 7.03e-08. Archived query orders matched 288/288 and archived winners matched 288/288. See the per-unit tables for any differences.

The per-unit table reports evaluation phase wall time separately from summed worker fit and validation
times. Summed worker times can overlap and do not add to the wall time.
