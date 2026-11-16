# DomainNet pool-query results across additional projection seeds

Four additional seeds, six targets, two encoders and 455 candidates per unit. The original seed is used for regression verification. Test scores follow the frozen validation choices. All differences are signed, and a negative value favors the first policy.

## Per-domain test Brier averaged across four seeds

| Target | Full 455 | Pool only 25 | Pool − full | MMD 50 | Pool − MMD | Geometry + pool 25 | Geometry − pool |
|---|---:|---:|---:|---:|---:|---:|---:|
| clipart | 0.837925 | 0.837925 | +0.000000 | 0.837925 | +0.000000 | 0.837925 | +0.000000 |
| infograph | 0.909722 | 0.909722 | +0.000000 | 0.911228 | -0.001506 | 0.909722 | +0.000000 |
| painting | 0.815281 | 0.815281 | +0.000000 | 0.815281 | +0.000000 | 0.815281 | +0.000000 |
| quickdraw | 0.975994 | 0.976332 | +0.000339 | 0.975944 | +0.000388 | 0.976377 | +0.000045 |
| real | 0.799150 | 0.799150 | +0.000000 | 0.799150 | +0.000000 | 0.799150 | +0.000000 |
| sketch | 0.846634 | 0.846634 | +0.000000 | 0.846665 | -0.000031 | 0.846634 | +0.000000 |
| Mean | 0.864118 | 0.864174 | +0.000056 | 0.864365 | -0.000191 | 0.864182 | +0.000007 |

## Per-seed means and domain exceedances

Each gap cell shows the mean signed gap and the count of six domains exceeding +0.005.

| Seed | Pool 25 − MMD 50 | Pool 25 − full | Geometry 25 − MMD 50 | Geometry 25 − full | Geometry 25 − pool 25 |
|---|---:|---:|---:|---:|---:|
| 20260917 | -0.000058 (0/6) | +0.000060 (0/6) | -0.000118 (0/6) | +0.000000 (0/6) | -0.000060 (0/6) |
| 20260918 | -0.000356 (0/6) | +0.000047 (0/6) | -0.000356 (0/6) | +0.000047 (0/6) | +0.000000 (0/6) |
| 20260919 | +0.000118 (0/6) | +0.000118 (0/6) | +0.000118 (0/6) | +0.000118 (0/6) | +0.000000 (0/6) |
| 20260920 | -0.000470 (0/6) | +0.000000 (0/6) | -0.000380 (0/6) | +0.000090 (0/6) | +0.000090 (0/6) |
| mean | -0.000191 (0/6) | +0.000056 (0/6) | -0.000184 (0/6) | +0.000064 (0/6) | +0.000007 (0/6) |

| Contrast | Seed–domain gaps > +0.005 | Domains with any seed > +0.005 | Worst seed–domain gap |
|---|---:|---:|---:|
| pool_vs_mmd | 0/24 | 0/6 | +0.000708 (quickdraw, 20260919) |
| pool_vs_full | 0/24 | 0/6 | +0.000708 (quickdraw, 20260919) |
| geometry_vs_mmd | 0/24 | 0/6 | +0.000739 (quickdraw, 20260920) |
| geometry_vs_full | 0/24 | 0/6 | +0.000708 (quickdraw, 20260919) |
| geometry_vs_pool | 0/24 | 0/6 | +0.000541 (quickdraw, 20260920) |

## All policies averaged across four seeds

| Budget | Method | Test Brier | Gap to full | Domains with gap > +0.005 |
|---:|---|---:|---:|---:|
| 15 | Geometry + pool | 0.872948 | +0.008831 | 4/6 |
| 15 | Pool only | 0.872948 | +0.008831 | 4/6 |
| 15 | Fixed MMD | 0.865414 | +0.001296 | 0/6 |
| 15 | Fixed A-opt | 0.866959 | +0.002842 | 1/6 |
| 15 | Random | 0.879308 | +0.015191 | 5/6 |
| 25 | Geometry + pool | 0.864182 | +0.000064 | 0/6 |
| 25 | Pool only | 0.864174 | +0.000056 | 0/6 |
| 25 | Fixed MMD | 0.865156 | +0.001038 | 0/6 |
| 25 | Fixed A-opt | 0.865851 | +0.001734 | 0/6 |
| 25 | Random | 0.876178 | +0.012060 | 5/6 |
| 50 | Geometry + pool | 0.864147 | +0.000029 | 0/6 |
| 50 | Pool only | 0.864147 | +0.000029 | 0/6 |
| 50 | Fixed MMD | 0.864365 | +0.000248 | 0/6 |
| 50 | Fixed A-opt | 0.864511 | +0.000393 | 0/6 |
| 50 | Random | 0.871864 | +0.007747 | 4/6 |

The [primary contrasts](primary_contrast.csv) include test Brier and signed gaps for every seed and domain. The [per-domain CSV](per_domain.csv) includes all budgets and methods. The seeds use the same underlying data and candidate pools. Query counts describe the archived-fit replay.
