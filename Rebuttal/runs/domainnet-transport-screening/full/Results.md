# DomainNet six-domain fixed-pool transport comparison

Six target domains and two encoders at frozen projection seed 20260911. Each shortlist winner minimizes archived logistic validation Brier. The signed gap subtracts the test Brier of the winner selected by validation over all 455 candidates. Lower Brier is better. Random first averages 20 archived orders per encoder and domain.

Screening times are per-unit medians from the server runs, averaged over the two encoders within each domain and equally over domains. A ranking is computed once and serves both budgets, so its time is the same in both rows. Times use resident projected features and exclude image extraction, readout training, and validation. Missing timings are shown as em dashes. The transport variants use frozen features under the same pool constraints.

## Overall

| Budget | Screen | Test Brier | Gap to full | Screen seconds | Domains above +0.005 |
|---|---|---|---|---|---|
| 50 | Sliced Wasserstein | 0.866406 | +0.002649 | 0.2777 | 1/6 |
| 50 | Whitened sliced Wasserstein | 0.865342 | +0.001585 | 0.2948 | 0/6 |
| 50 | A-opt | 0.864104 | +0.000347 | 0.5482 | 0/6 |
| 50 | MMD | 0.863951 | +0.000194 | 0.0387 | 0/6 |
| 50 | Random (20 orders) | 0.871557 | +0.007800 | — | 4/6 |
| 227 | Sliced Wasserstein | 0.864193 | +0.000436 | 0.2777 | 0/6 |
| 227 | Whitened sliced Wasserstein | 0.863674 | -0.000083 | 0.2948 | 0/6 |
| 227 | A-opt | 0.863757 | +0.000000 | 0.5482 | 0/6 |
| 227 | MMD | 0.863757 | +0.000000 | 0.0387 | 0/6 |
| 227 | Random (20 orders) | 0.865482 | +0.001725 | — | 1/6 |
| 455 | Full validation | 0.863757 | +0.000000 | — | 0/6 |

## Per domain

| Domain | Budget | Screen | Test Brier | Gap to full | Screen seconds |
|---|---|---|---|---|---|
| clipart | 50 | Sliced Wasserstein | 0.839163 | +0.004170 | 0.3033 |
| clipart | 50 | Whitened sliced Wasserstein | 0.839163 | +0.004170 | 0.3134 |
| clipart | 50 | A-opt | 0.834993 | +0.000000 | 0.6417 |
| clipart | 50 | MMD | 0.834993 | +0.000000 | 0.0426 |
| clipart | 50 | Random (20 orders) | 0.846675 | +0.011682 | — |
| clipart | 227 | Sliced Wasserstein | 0.834993 | +0.000000 | 0.3033 |
| clipart | 227 | Whitened sliced Wasserstein | 0.834993 | +0.000000 | 0.3134 |
| clipart | 227 | A-opt | 0.834993 | +0.000000 | 0.6417 |
| clipart | 227 | MMD | 0.834993 | +0.000000 | 0.0426 |
| clipart | 227 | Random (20 orders) | 0.836883 | +0.001890 | — |
| clipart | 455 | Full validation | 0.834993 | +0.000000 | — |
| infograph | 50 | Sliced Wasserstein | 0.911583 | +0.001806 | 0.2649 |
| infograph | 50 | Whitened sliced Wasserstein | 0.912209 | +0.002432 | 0.2522 |
| infograph | 50 | A-opt | 0.909777 | +0.000000 | 0.4504 |
| infograph | 50 | MMD | 0.910311 | +0.000534 | 0.0358 |
| infograph | 50 | Random (20 orders) | 0.912746 | +0.002969 | — |
| infograph | 227 | Sliced Wasserstein | 0.910311 | +0.000534 | 0.2649 |
| infograph | 227 | Whitened sliced Wasserstein | 0.910311 | +0.000534 | 0.2522 |
| infograph | 227 | A-opt | 0.909777 | +0.000000 | 0.4504 |
| infograph | 227 | MMD | 0.909777 | +0.000000 | 0.0358 |
| infograph | 227 | Random (20 orders) | 0.909726 | -0.000051 | — |
| infograph | 455 | Full validation | 0.909777 | +0.000000 | — |
| painting | 50 | Sliced Wasserstein | 0.816649 | +0.000000 | 0.2637 |
| painting | 50 | Whitened sliced Wasserstein | 0.816649 | +0.000000 | 0.3385 |
| painting | 50 | A-opt | 0.816649 | +0.000000 | 0.6060 |
| painting | 50 | MMD | 0.816649 | +0.000000 | 0.0535 |
| painting | 50 | Random (20 orders) | 0.821990 | +0.005341 | — |
| painting | 227 | Sliced Wasserstein | 0.816649 | +0.000000 | 0.2637 |
| painting | 227 | Whitened sliced Wasserstein | 0.816649 | +0.000000 | 0.3385 |
| painting | 227 | A-opt | 0.816649 | +0.000000 | 0.6060 |
| painting | 227 | MMD | 0.816649 | +0.000000 | 0.0535 |
| painting | 227 | Random (20 orders) | 0.817228 | +0.000579 | — |
| painting | 455 | Full validation | 0.816649 | +0.000000 | — |
| quickdraw | 50 | Sliced Wasserstein | 0.975949 | +0.000000 | 0.2557 |
| quickdraw | 50 | Whitened sliced Wasserstein | 0.975949 | +0.000000 | 0.3146 |
| quickdraw | 50 | A-opt | 0.978031 | +0.002081 | 0.3908 |
| quickdraw | 50 | MMD | 0.975949 | +0.000000 | 0.0362 |
| quickdraw | 50 | Random (20 orders) | 0.978673 | +0.002723 | — |
| quickdraw | 227 | Sliced Wasserstein | 0.975949 | +0.000000 | 0.2557 |
| quickdraw | 227 | Whitened sliced Wasserstein | 0.975949 | +0.000000 | 0.3146 |
| quickdraw | 227 | A-opt | 0.975949 | +0.000000 | 0.3908 |
| quickdraw | 227 | MMD | 0.975949 | +0.000000 | 0.0362 |
| quickdraw | 227 | Random (20 orders) | 0.977326 | +0.001377 | — |
| quickdraw | 455 | Full validation | 0.975949 | +0.000000 | — |
| real | 50 | Sliced Wasserstein | 0.795367 | +0.000000 | 0.2855 |
| real | 50 | Whitened sliced Wasserstein | 0.795367 | +0.000000 | 0.2482 |
| real | 50 | A-opt | 0.795367 | +0.000000 | 0.5705 |
| real | 50 | MMD | 0.795367 | +0.000000 | 0.0231 |
| real | 50 | Random (20 orders) | 0.813319 | +0.017952 | — |
| real | 227 | Sliced Wasserstein | 0.795367 | +0.000000 | 0.2855 |
| real | 227 | Whitened sliced Wasserstein | 0.795367 | +0.000000 | 0.2482 |
| real | 227 | A-opt | 0.795367 | +0.000000 | 0.5705 |
| real | 227 | MMD | 0.795367 | +0.000000 | 0.0231 |
| real | 227 | Random (20 orders) | 0.802312 | +0.006945 | — |
| real | 455 | Full validation | 0.795367 | +0.000000 | — |
| sketch | 50 | Sliced Wasserstein | 0.859723 | +0.009915 | 0.2929 |
| sketch | 50 | Whitened sliced Wasserstein | 0.852713 | +0.002906 | 0.3022 |
| sketch | 50 | A-opt | 0.849807 | +0.000000 | 0.6297 |
| sketch | 50 | MMD | 0.850436 | +0.000629 | 0.0413 |
| sketch | 50 | Random (20 orders) | 0.855938 | +0.006131 | — |
| sketch | 227 | Sliced Wasserstein | 0.851888 | +0.002081 | 0.2929 |
| sketch | 227 | Whitened sliced Wasserstein | 0.848774 | -0.001033 | 0.3022 |
| sketch | 227 | A-opt | 0.849807 | +0.000000 | 0.6297 |
| sketch | 227 | MMD | 0.849807 | +0.000000 | 0.0413 |
| sketch | 227 | Random (20 orders) | 0.849418 | -0.000389 | — |
| sketch | 455 | Full validation | 0.849807 | +0.000000 | — |

Per-encoder medians are in timing_per_encoder_domain.csv. Run-level screen timings and archive provenance are recorded in replay/screen_metadata.json.

## Ranking verification and interpretation

All 455 candidate IDs matched for every unit. Both recalculated baselines matched all 48 shortlist sets at budgets 50 and 227 in the paired readout archive. Relative to the earlier primary archive, four of 48 shortlist comparisons differed by one member. Outcomes use the paired readout archive consistently. See ranking-gate.log.

At budget 50, whitening reduced mean Brier relative to raw sliced Wasserstein, but both transport variants had higher mean Brier than MMD and A-opt. Whitened transport had no domain with a signed gap above +0.005, despite the clipart/ResNet-50 single-unit pilot exceeding the threshold. That pilot is a single encoder and domain. The reported six-domain exceedances average both encoders first. At 227, all geometric methods were close to full validation. MMD had the lowest mean screening time among the four geometric rankings. This is a frozen-feature transport comparison under fixed pool membership, and the timings cover screening only.
