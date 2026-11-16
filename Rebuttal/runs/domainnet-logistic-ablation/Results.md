# DomainNet logistic inverse weighting ablation

Frozen projection seed 20260911. Selection uses validation Brier, with candidate ID as the tie breaker. Each target averages ResNet-50 and DINOv2 B/14. The overall mean weights six targets equally. Random averages 20 archived repeats per encoder and target.

All 36 ranking comparisons matched the 455 candidate IDs. Between the submission primary and readout archives, 168 of 180 shortlist comparisons matched exactly, with one candidate swapped in each remaining comparison. Results use rankings and losses from the same readout lineage. Only the earlier clipart / ResNet-50 pilot was rebuilt from images.

## Overall test Brier

| Budget | A-opt | Target energy | MMD | Random |
|---:|---:|---:|---:|---:|
| 10 | 0.867049 | 0.922707 | 0.867382 | 0.883293 |
| 25 | 0.865084 | 0.913167 | 0.864716 | 0.875995 |
| 50 | 0.864104 | 0.896467 | 0.863951 | 0.871557 |
| 100 | 0.864076 | 0.884744 | 0.863846 | 0.868382 |
| 227 | 0.863757 | 0.873177 | 0.863757 | 0.865482 |

## Per-target test Brier at budgets 50 and 227

| Target | Budget | A-opt | Target energy | MMD | Random |
|---|---:|---:|---:|---:|---:|
| clipart | 50 | 0.834993 | 0.872625 | 0.834993 | 0.846675 |
| clipart | 227 | 0.834993 | 0.850896 | 0.834993 | 0.836883 |
| infograph | 50 | 0.909777 | 0.924915 | 0.910311 | 0.912746 |
| infograph | 227 | 0.909777 | 0.910937 | 0.909777 | 0.909726 |
| painting | 50 | 0.816649 | 0.865037 | 0.816649 | 0.821990 |
| painting | 227 | 0.816649 | 0.819136 | 0.816649 | 0.817228 |
| quickdraw | 50 | 0.978031 | 0.975949 | 0.975949 | 0.978673 |
| quickdraw | 227 | 0.975949 | 0.975949 | 0.975949 | 0.977326 |
| real | 50 | 0.795367 | 0.845228 | 0.795367 | 0.813319 |
| real | 227 | 0.795367 | 0.805623 | 0.795367 | 0.802312 |
| sketch | 50 | 0.849807 | 0.895047 | 0.850436 | 0.855938 |
| sketch | 227 | 0.849807 | 0.876520 | 0.849807 | 0.849418 |
