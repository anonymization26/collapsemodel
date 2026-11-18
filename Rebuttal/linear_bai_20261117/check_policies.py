import json
from pathlib import Path
import sys
import numpy as np
from linear_policies import X,basis,geometry_pick,adaptive_pick,run,steiner_support,od_linbai

def main():
    q=basis();v=X[q].T@X[q];inv=np.linalg.inv(v)
    rng=np.random.default_rng(17112026);theta=rng.normal(size=15);loss=X@theta
    assert np.allclose(X@np.linalg.lstsq(X[q],loss[q],rcond=None)[0],loss,atol=1e-12)
    scores={k:[] for k in ('a','d','g','xy')}
    for j in range(455):
        updated=v+np.outer(X[j],X[j]);iv=np.linalg.inv(updated);cross=X@iv@X.T;diag=np.diag(cross)
        scores['a'].append(np.trace(iv));scores['d'].append(-np.linalg.slogdet(updated)[1])
        scores['g'].append(diag.max());scores['xy'].append((diag[:,None]+diag[None,:]-2*cross).max())
    for k in scores:
        s=np.array(scores[k]);s[q]=np.inf;j=geometry_pick(v,q,k)
        assert abs(s[j]-s.min())<1e-10,(k,s[j],s.min())
    # Independently compute incumbent/challenger and all updated gap variances.
    for c in (.001,.01,.1,1):
        beta=np.linalg.solve(v,X[q].T@loss[q]);mu=X@beta;i=int(np.argmin(mu))
        d=X-X[i];upper=mu[i]-mu+c*np.sqrt(np.einsum('ij,jk,ik->i',d,inv,d));upper[i]=-np.inf
        j=int(np.argmax(upper));delta=X[i]-X[j]
        brute=np.array([delta@np.linalg.inv(v+np.outer(x,x))@delta for x in X]);brute[q]=np.inf
        pick=adaptive_pick(q,loss[q],f'lingape_{c}')
        assert abs(brute[pick]-brute.min())<1e-10
    tested=0
    for method in ('seq','batch','exploit','a','d','g','lingape_0.001','lingape_0.1'):
        for finish in ('observed','recommend','native_prediction'):
            seen=[]
            def reveal(i):assert i not in seen;seen.append(i);return loss[i]
            r=run(method,25,reveal,finish)
            assert seen==r['queries'] and len(seen)==25
            if finish!='observed' or method in ('seq','batch','exploit'):
                assert r['winner']==int(np.argmin(loss)),(method,finish)
            tested+=1
    s=steiner_support();assert np.allclose(np.einsum('ij,jk,ik->i',X,np.linalg.inv(X[s].T@X[s]/35),X),15)
    seen=[]
    def reveal_od(i):assert i not in seen;seen.append(i);return loss[i]
    od=od_linbai(reveal_od);assert od['winner']==int(np.argmin(loss))
    assert len(od['pulls'])<=200 and len(od['queries'])==len(seen)
    assert [len(p['active']) for p in od['phases']]==[8,4,2,1]
    result=dict(status='passed',exact_criteria_checked=['a','d','g','xy','lingape'],
                exact_additive_policy_checks=tested,steiner_support=35,od_pulls=len(od['pulls']),od_unique=len(seen))
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(result,indent=2)+'\n')
    print(result)

if __name__=='__main__':main()
