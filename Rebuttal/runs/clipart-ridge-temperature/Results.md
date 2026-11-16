# Clipart / ResNet-50 ridge temperature pilot

The maximum absolute T=1 Brier discrepancy was 7.20e-11 on validation and 3.60e-11 on test. Uniform 100-class prediction has Brier 0.99. Random averages 20 orders. Accuracy is top-1. Each gap uses its corresponding full-validation arm.

| Budget | Screen | Arm | Temperature | Test Brier | Accuracy | Gap to full |
|---:|---|---|---:|---:|---:|---:|
| 50 | target_a | original | 1.0000 | 0.987591 | 31.93% | -0.000024 |
| 50 | target_a | original_calibrated | 0.0355 | 0.791511 | 31.93% | -0.019797 |
| 50 | target_a | split_uncalibrated | 1.0000 | 0.987606 | 31.69% | -0.000010 |
| 50 | target_a | split_calibrated | 0.0355 | 0.791511 | 31.93% | +0.000000 |
| 50 | second_moment_mmd | original | 1.0000 | 0.987606 | 31.69% | -0.000010 |
| 50 | second_moment_mmd | original_calibrated | 0.0398 | 0.797566 | 31.69% | -0.013742 |
| 50 | second_moment_mmd | split_uncalibrated | 1.0000 | 0.987606 | 31.69% | -0.000010 |
| 50 | second_moment_mmd | split_calibrated | 0.0398 | 0.808526 | 30.81% | +0.017015 |
| 50 | random | original | 1.0000 | 0.987725 | 28.93% | +0.000110 |
| 50 | random | original_calibrated | 0.0420 | 0.823017 | 28.93% | +0.011709 |
| 50 | random | split_uncalibrated | 1.0000 | 0.987732 | 29.12% | +0.000116 |
| 50 | random | split_calibrated | 0.0392 | 0.825684 | 28.92% | +0.034173 |
| 227 | target_a | original | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 227 | target_a | original_calibrated | 0.0447 | 0.811308 | 31.01% | +0.000000 |
| 227 | target_a | split_uncalibrated | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 227 | target_a | split_calibrated | 0.0355 | 0.791511 | 31.93% | +0.000000 |
| 227 | second_moment_mmd | original | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 227 | second_moment_mmd | original_calibrated | 0.0447 | 0.811308 | 31.01% | +0.000000 |
| 227 | second_moment_mmd | split_uncalibrated | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 227 | second_moment_mmd | split_calibrated | 0.0355 | 0.791511 | 31.93% | +0.000000 |
| 227 | random | original | 1.0000 | 0.987619 | 31.16% | +0.000003 |
| 227 | random | original_calibrated | 0.0416 | 0.807001 | 31.16% | -0.004307 |
| 227 | random | split_uncalibrated | 1.0000 | 0.987622 | 31.11% | +0.000006 |
| 227 | random | split_calibrated | 0.0368 | 0.803917 | 31.13% | +0.012406 |
| 455 | full | original | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 455 | full | original_calibrated | 0.0447 | 0.811308 | 31.01% | +0.000000 |
| 455 | full | split_uncalibrated | 1.0000 | 0.987616 | 31.01% | +0.000000 |
| 455 | full | split_calibrated | 0.0355 | 0.791511 | 31.93% | +0.000000 |

Complete budget results are in summary.csv. Per-repeat winners and scores are in results.csv. Fixed winners preserve accuracy exactly across all methods and budgets. Original winners use full validation for temperature fitting. The reselection comparison uses a shared 512-row selection half for both arms.
