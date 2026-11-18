# Pre-DomainNet-readout execution details

Recorded after the eight Optdigits logistic units completed, but before DomainNet
feature extraction or any DomainNet supervised fit. This clarifies implementation
details without changing the frozen primary comparisons or using DomainNet outcomes.

- Hash source blocks use SHA-256 of `seed:image_path` within each source domain.
  PC1 blocks use centered raw 2048D source features, with eigenvector sign fixed by
  its largest absolute loading. The same image inventory is retained in both arms.
- Rademacher matrices have 512 columns; the first d columns define nested
  projections for d=64/96/128/256/512. Scaling by 1/sqrt(d) cancels under post-L2.
- Primary logistic: all six targets, both pool constructions, all four seeds, d=128.
- Ridge dimension/lambda sensitivity: all six targets, hash pools, seed 20261117,
  all five dimensions. Optdigits ridge covers both constructions and all four seeds.
- Independent resident-feature logistic timing: Optdigits hash seed 20261117;
  DomainNet hash d=128 seed 20261117, all six targets. Three repetitions at both
  one and four fitting workers. Timing runs occur after other campaign compute jobs.
- No-feature-cache image timing: preselected first alphabetical target, clipart,
  hash pools, d=128, seed 20261117, one fitting worker. Run sequential25, batch25,
  MMD50 and full455 separately, one repetition each. Include model loading,
  decoding/transforms, GPU extraction, projection and selection. Raw files and
  the pretrained checkpoint already exist; download/Python import time and
  controlled OS-cold-cache behavior are outside this claim. No uncertainty
  estimate is implied by one repetition.
- Primary quality fits can run in four independent CPU processes, each with BLAS=1.
  GPU feature extraction has its own lock and exposes physical GPU 0 only.
- Source archive download may resume locally verified files from this same fresh
  campaign. These are new raw downloads, not historical experimental caches.
- A failure aborts the relevant stage and is reported. No result-based retries,
  hyperparameter changes, selective exclusions or replacement of negative results.
