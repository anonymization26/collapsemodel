# DomainNet sequential query versus a fixed second batch

Local replay on the four additional projection seeds, six domains and two encoders. Both policies use the same 15 full-rank queries and acquisition score. The batch policy freezes its order after the first fit to those 15 scores. All 48 traces were locked before test audit.

| Queries | Fixed second batch | Sequential | Fixed MMD 50 | Full validation 455 | Batch minus sequential | Same winner |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.864428 | 0.864174 | 0.864365 | 0.864118 | +0.000254 | 44/48 |
| 50 | 0.864185 | 0.864147 | 0.864365 | 0.864118 | +0.000038 | 45/48 |

At 25 queries the batch policy's mean gaps to MMD and full validation are +0.000063 and +0.000311. Against each reference and sequential querying, 0/24 seed–domain gaps exceed +0.005 after averaging encoders. The largest gap is +0.004536 in Sketch at seed 20260919. DINOv2 B/14 accounts for the entire gap with +0.009073, so the count before averaging encoders is 1/48. At 50 queries both counts are zero.

Sequential updates slightly improve mean test Brier in this replay. The fixed second batch achieves similar average quality at equal fit count. The one- and four-thread timings use fresh logistic fits at the original projection seed 20260911. [Server report](server/Logistic_Extended_Results.md)

[Per-encoder results](per_unit.csv) · [Seed–domain results](per_seed_domain.csv) · Selection log · Audit log
