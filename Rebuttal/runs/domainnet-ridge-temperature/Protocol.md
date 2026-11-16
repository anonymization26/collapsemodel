# DomainNet ridge temperature extension

Apply the clipart pilot unchanged to six domains and ResNet-50 / DINOv2 B/14, at projection seed 20260911. Use the paired readout archive, frozen samples, candidate IDs and rankings. Verify T=1 validation and test Brier for every candidate, with a maximum absolute discrepancy of 0.0001 in each unit. All twelve gates must pass before the calibration phase.

Use ridge lambda 1. Minimize Brier over the pilot's 81 log-spaced temperatures from 0.01 to 100, including 1. Calibrate fixed archived winners on all target validation rows. For reselection, use the original pilot ID hash to split validation 512/512 for temperature fitting and candidate selection. Use the same selection rows for the T=1 comparison. Freeze winners and temperatures before final evaluation.

Compare A-opt, second-moment MMD and 20 random orders at the original budgets, highlighting 50 and 227, with a 455-fit validation reference for each arm. Average random repeats within encoder and domain, then average encoders within domain and weight domains equally. Report test Brier, top-1 accuracy and signed gaps to the corresponding full-validation arm.
