# DomainNet sequential pool-contribution results

Archived logistic readout at projection seed 20260911, six targets and two encoders. Each method uses only its own queried validation scores. Test scores were read after decisions were locked. Δ is the selected model's test Brier minus the test Brier of the 455-fit validation winner. Lower is better. Random averages 20 orders per unit before averaging encoders and targets.

## Primary contrast against fixed MMD at 50

| Target | Geometry + pool 25 | Pool only 25 | MMD 50 | Geometry gap | Pool-only gap |
|---|---:|---:|---:|---:|---:|
| clipart | 0.834993 | 0.834993 | 0.834993 | +0.000000 | +0.000000 |
| infograph | 0.909777 | 0.909777 | 0.910311 | -0.000534 | -0.000534 |
| painting | 0.816649 | 0.816649 | 0.816649 | +0.000000 | +0.000000 |
| quickdraw | 0.977361 | 0.977114 | 0.975949 | +0.001412 | +0.001165 |
| real | 0.795367 | 0.795367 | 0.795367 | +0.000000 | +0.000000 |
| sketch | 0.849807 | 0.849807 | 0.850436 | -0.000629 | -0.000629 |
| Mean | 0.863992 | 0.863951 | 0.863951 | +0.000042 | +0.000000 |

The signed gap versus MMD 50 exceeds +0.005 in 0/6 domains for geometry + pool and 0/6 for pool only.

## Overall against exhaustive validation

Mean test Brier of the exhaustive-validation reference: 0.863757.

| Budget | Method | Test Brier | Δ | Domains Δ > 0.005 |
|---:|---|---:|---:|---:|
| 15 | Geometry + pool | 0.872911 | +0.009154 | 4/6 |
| 15 | Pool only | 0.872911 | +0.009154 | 4/6 |
| 15 | Fixed MMD | 0.866182 | +0.002424 | 1/6 |
| 15 | Fixed A-opt | 0.866400 | +0.002642 | 1/6 |
| 15 | Random | 0.878968 | +0.015211 | 5/6 |
| 25 | Geometry + pool | 0.863992 | +0.000235 | 0/6 |
| 25 | Pool only | 0.863951 | +0.000194 | 0/6 |
| 25 | Fixed MMD | 0.864716 | +0.000959 | 0/6 |
| 25 | Fixed A-opt | 0.865084 | +0.001327 | 0/6 |
| 25 | Random | 0.875995 | +0.012238 | 5/6 |
| 50 | Geometry + pool | 0.863889 | +0.000132 | 0/6 |
| 50 | Pool only | 0.863889 | +0.000132 | 0/6 |
| 50 | Fixed MMD | 0.863951 | +0.000194 | 0/6 |
| 50 | Fixed A-opt | 0.864104 | +0.000347 | 0/6 |
| 50 | Random | 0.871557 | +0.007800 | 4/6 |

## Per-domain results at 15 queries

Each method cell is test Brier (Δ to exhaustive validation).

| Target | Full | Geometry + pool | Pool only | Fixed MMD | Fixed A-opt | Random |
|---|---:|---:|---:|---:|---:|---:|
| clipart | 0.834993 | 0.852035 (+0.017042) | 0.852035 (+0.017042) | 0.846479 (+0.011486) | 0.842887 (+0.007894) | 0.856192 (+0.021199) |
| infograph | 0.909777 | 0.909005 (-0.000772) | 0.909005 (-0.000772) | 0.912395 (+0.002618) | 0.912485 (+0.002708) | 0.918975 (+0.009198) |
| painting | 0.816649 | 0.824667 (+0.008018) | 0.824667 (+0.008018) | 0.816649 (+0.000000) | 0.818463 (+0.001814) | 0.833408 (+0.016758) |
| quickdraw | 0.975949 | 0.978692 (+0.002742) | 0.978692 (+0.002742) | 0.977424 (+0.001475) | 0.978389 (+0.002440) | 0.979041 (+0.003092) |
| real | 0.795367 | 0.808774 (+0.013407) | 0.808774 (+0.013407) | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.825594 (+0.030227) |
| sketch | 0.849807 | 0.864294 (+0.014487) | 0.864294 (+0.014487) | 0.848774 (-0.001033) | 0.850807 (+0.001000) | 0.860596 (+0.010788) |

## Per-domain results at 25 queries

Each method cell is test Brier (Δ to exhaustive validation).

| Target | Full | Geometry + pool | Pool only | Fixed MMD | Fixed A-opt | Random |
|---|---:|---:|---:|---:|---:|---:|
| clipart | 0.834993 | 0.834993 (+0.000000) | 0.834993 (+0.000000) | 0.839163 (+0.004170) | 0.836769 (+0.001776) | 0.852681 (+0.017688) |
| infograph | 0.909777 | 0.909777 (+0.000000) | 0.909777 (+0.000000) | 0.912395 (+0.002618) | 0.912485 (+0.002708) | 0.915861 (+0.006084) |
| painting | 0.816649 | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.828297 (+0.011647) |
| quickdraw | 0.975949 | 0.977361 (+0.001412) | 0.977114 (+0.001165) | 0.975949 (+0.000000) | 0.978113 (+0.002164) | 0.978872 (+0.002922) |
| real | 0.795367 | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.821058 (+0.025691) |
| sketch | 0.849807 | 0.849807 (+0.000000) | 0.849807 (+0.000000) | 0.848774 (-0.001033) | 0.851122 (+0.001314) | 0.859201 (+0.009394) |

## Per-domain results at 50 queries

Each method cell is test Brier (Δ to exhaustive validation).

| Target | Full | Geometry + pool | Pool only | Fixed MMD | Fixed A-opt | Random |
|---|---:|---:|---:|---:|---:|---:|
| clipart | 0.834993 | 0.834993 (+0.000000) | 0.834993 (+0.000000) | 0.834993 (+0.000000) | 0.834993 (+0.000000) | 0.846675 (+0.011682) |
| infograph | 0.909777 | 0.909777 (+0.000000) | 0.909777 (+0.000000) | 0.910311 (+0.000534) | 0.909777 (+0.000000) | 0.912746 (+0.002969) |
| painting | 0.816649 | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.816649 (+0.000000) | 0.821990 (+0.005341) |
| quickdraw | 0.975949 | 0.976743 (+0.000794) | 0.976743 (+0.000794) | 0.975949 (+0.000000) | 0.978031 (+0.002081) | 0.978673 (+0.002723) |
| real | 0.795367 | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.795367 (+0.000000) | 0.813319 (+0.017952) |
| sketch | 0.849807 | 0.849807 (+0.000000) | 0.849807 (+0.000000) | 0.850436 (+0.000629) | 0.849807 (+0.000000) | 0.855938 (+0.006131) |

## Scope

The adaptive methods have identical choices at budget 15 because their exploration prefix is shared. The full reference is a validation winner, so a negative test gap can occur. The +0.005 counts use domain means and a signed tolerance. The targets share source pools. The existing archive was used in earlier analyses. Query counts describe simulated fit budgets.
