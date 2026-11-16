# Office-Home logistic results

Budget 50 of 220. Lower test Brier is better.

| Target | A-opt | Target energy | MMD | Random mean |
|---|---:|---:|---:|---:|
| Art | 0.847702 | 0.884889 | 0.847702 | 0.851215 |
| Clipart | 0.905802 | 0.903112 | 0.903112 | 0.904134 |
| Product | 0.784255 | 0.840521 | 0.784255 | 0.793564 |
| Real-World | 0.790619 | 0.820720 | 0.790619 | 0.791642 |
| overall | 0.832095 | 0.862310 | 0.831422 | 0.835139 |

The four domains are equally weighted. A-opt clears the preset 0.005 mean Brier threshold against target energy. The mean improvement over random falls below that threshold. MMD ties A-opt in three domains and is better on Clipart.
