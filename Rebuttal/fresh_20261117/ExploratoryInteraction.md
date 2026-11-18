# Exploratory interaction diagnostic

This diagnostic was added after seeing the Optdigits pool-construction failure
and the first DomainNet clipart units. It is explicitly post hoc, not a
pre-registered success criterion, a deployed selector or a 25-query algorithm.

For every completed primary logistic unit, take the newly computed 455 validation
losses. Use the same fixed five folds of candidate IDs (RNG seed 20261117) in
every unit. Compare ordinary least squares on 15 pool indicators with ordinary
least squares on 105 pair-presence indicators. The latter span the linear terms
for fixed triples. Predict each held-out fold using the other four folds and
report out-of-fold R2, RMSE and Spearman. No ridge parameter is selected.

No test labels or test metrics are read. This can reveal whether pair terms help
predict unobserved combinations within the same candidate family. It does not
establish causality, independent-sample significance, transfer to new pools or
that 25 queries suffice to fit the larger model. Even favorable results remain
mechanism diagnostics rather than evidence for a new acquisition policy.
