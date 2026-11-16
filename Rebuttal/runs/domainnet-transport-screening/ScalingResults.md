# Exact A-opt size stress test

Clipart / ResNet-50, 128-dimensional features. Each constructed candidate uses 768 rows. Compare timings within each inventory. All shortlists match exhaustive Cholesky exactly in float64. Times are medians of three runs.

| Candidates | Budget | Exhaustive time (s) | Mean anchor solves | Mean anchor time (s) | Moment anchor solves | Moment anchor time (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 455 | 50 | 0.2112 | 215 | 0.2050 | 455 | 0.3618 |
| 455 | 227 | 0.2112 | 401 | 0.3532 | 455 | 0.2868 |
| 1140 | 50 | 0.7414 | 368 | 0.1667 | 1140 | 1.1660 |
| 1140 | 227 | 0.7414 | 633 | 0.5850 | 1140 | 0.9056 |
| 2300 | 50 | 1.4238 | 910 | 0.5787 | 2300 | 1.3232 |
| 2300 | 227 | 1.4238 | 1099 | 0.6729 | 2300 | 1.3286 |
| 4060 | 50 | 2.3492 | 1904 | 1.2275 | 4060 | 2.2626 |
| 4060 | 227 | 2.3492 | 2044 | 1.1490 | 4060 | 2.4797 |
