"""Correctness checks that protect the experimental interpretation."""
from itertools import combinations
import numpy as np
from sklearn.linear_model import LogisticRegression
from policies import design_basis,run_policy,method_specs
from run_fresh import model_fit,probability

def main():
    design,basis=design_basis(list(combinations(range(15),3)),15)
    beta=np.random.default_rng(9).normal(size=15)
    truth=design@beta
    recovered=np.linalg.lstsq(design[basis],truth[basis],rcond=None)[0]
    np.testing.assert_allclose(design@recovered,truth,atol=1e-12)
    for name,mode,budget,repeat in method_specs(9):
        calls=[]
        def reveal(i):calls.append(i);return truth[i]
        d=run_policy(mode,design,basis,reveal,np.arange(455),budget)
        assert calls==d['queries'] and len(set(calls))==budget
        assert d['winner']==min(calls,key=lambda i:(truth[i],i))
        if mode not in ('mmd','aopt','random','full'):assert calls[:15]==basis
        if mode in ('exploit','sequential','batch'):assert d['winner']==int(np.argmin(truth))
    rng=np.random.default_rng(11);x=rng.normal(size=(80,8))
    for vocabulary in ([1,4],[0,2,4]):
        y=rng.choice(vocabulary,80)
        w,mask=model_fit(x,y,5)
        ref=LogisticRegression(C=1,fit_intercept=False,solver='lbfgs',tol=1e-6,max_iter=1000).fit(x,y)
        p=probability(x,w,mask)
        np.testing.assert_allclose(p[:,ref.classes_],ref.predict_proba(x),atol=1e-12)
        assert np.all(p[:,~mask]==0)
    print('PASS: exact additive recovery, query budgets/access, shared initialization, class probability alignment')

if __name__=='__main__':main()
