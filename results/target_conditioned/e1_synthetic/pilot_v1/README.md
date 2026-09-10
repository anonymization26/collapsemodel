# E1 Target-Conditioned Synthetic Selection

Status: completed experimental run; results are synthetic evidence only.

- Configurations: 80
- Shared-model rows: 2480
- Conditional-shift rows: 1200
- Elapsed seconds: 93.628
- Git revision: `unknown`

`shared_model_results.csv` evaluates the theorem-aligned Bayes target risk.
`conditional_shift_results.csv` violates the shared conditional model and is a
required failure-boundary diagnostic. `summary.json` is generated only from the
two raw tables.
