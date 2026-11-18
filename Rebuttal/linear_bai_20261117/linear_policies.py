"""Finite deterministic fit-budget adaptations; see Protocol.md for departures."""
from itertools import combinations
import numpy as np

MEMBERS=list(combinations(range(15),3))
X=np.zeros((455,15),dtype=float)
for j,c in enumerate(MEMBERS): X[j,list(c)]=1.

def argmin(v,excluded=()):
    z=np.round(v,12).copy();z[list(excluded)]=np.inf
    return int(np.argmin(z))

def basis():
    residual=X.copy();q=[]
    for _ in range(15):
        norms=np.einsum('ij,ij->i',residual,residual)
        j=argmin(-norms,q);q.append(j)
        direction=residual[j]/np.sqrt(norms[j]);residual-=np.outer(residual@direction,direction)
    assert np.linalg.matrix_rank(X[q])==15
    return q

def geometry_pick(v,q,kind):
    inv=np.linalg.inv(v);cross=X@inv@X.T;diag=np.diag(cross)
    if kind=='d':score=-diag
    elif kind=='a':score=-np.sum((X@inv)**2,axis=1)/(1+diag)
    elif kind=='g':score=np.max(diag[:,None]-cross**2/(1+diag)[None,:],axis=0)
    elif kind=='xy':
        distance=np.maximum(diag[:,None]+diag[None,:]-2*cross,0)
        score=np.empty(len(X))
        # Exact all-pair criterion, one candidate at a time to bound memory.
        for j in range(len(X)):
            if j in q:score[j]=np.inf;continue
            diff=cross[:,j,None]-cross[None,:,j]
            score[j]=np.max(distance-diff**2/(1+diag[j]))
    else:raise ValueError(kind)
    return argmin(score,q)

def static_sequence(kind,n=100):
    q=basis();v=X[q].T@X[q]
    while len(q)<n:
        j=geometry_pick(v,q,kind);q.append(j);v+=np.outer(X[j],X[j])
    return q

def adaptive_pick(q,y,method):
    z=X[q];v=z.T@z;beta=np.linalg.lstsq(z,y,rcond=None)[0]
    mu=X@beta;inv=np.linalg.inv(v)
    if method in ('seq','batch','exploit'):
        c=0. if method=='exploit' else .001
        lev=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',X,inv,X),0))
        return argmin(mu-c*lev,q)
    assert method.startswith('lingape_')
    c=float(method.split('_',1)[1]);i=argmin(mu)
    directions=X-X[i];var=np.maximum(np.einsum('ij,jk,ik->i',directions,inv,directions),0)
    challenger=argmin(-(mu[i]-mu+c*np.sqrt(var)),[i])
    delta=X[i]-X[challenger];variance=float(delta@inv@delta)
    gain=(X@inv@delta)**2/(1+np.einsum('ij,jk,ik->i',X,inv,X))
    return argmin(variance-gain,q)

def run(method,budget,reveal,finish='observed',orders=None,static=None):
    q=[];y=[]
    def query(j):
        assert j not in q
        y.append(float(reveal(j)));q.append(int(j))
    target=budget-(finish!='observed')
    if method in ('mmd','target_aopt'):
        assert finish=='observed';order=orders[method]
        for j in order[:target]:query(j)
    else:
        for j in basis():query(j)
        if method in ('d','a','g','xy'):
            if static is not None:
                for j in static[method][15:target]:query(j)
            else:
                while len(q)<target:query(geometry_pick(X[q].T@X[q],q,method))
        elif method=='batch':
            z=X[q];beta=np.linalg.lstsq(z,y,rcond=None)[0]
            lev=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(z.T@z),X.T),axis=0)
            score=np.round(X@beta-.001*lev,12);score[q]=np.inf
            for j in np.argsort(score,kind='stable')[:target-15]:query(int(j))
        else:
            while len(q)<target:query(adaptive_pick(q,y,method))
    prediction=None
    if finish!='observed':
        mu=X@np.linalg.lstsq(X[q],y,rcond=None)[0];prediction=argmin(mu)
        query(prediction if prediction not in q else argmin(mu,q))
    best=q[min(range(len(q)),key=lambda i:(y[i],q[i]))]
    winner=prediction if finish=='native_prediction' else best
    assert len(q)==budget and len(set(q))==budget and winner in q
    return dict(method=method,budget=budget,finish=finish,queries=q,observed=y,
                winner=winner,prediction=prediction,best_observed=best)

def steiner_support():
    triples={tuple(sorted((a-1,b-1,(a^b)-1))) for a in range(1,16) for b in range(a+1,16)}
    ids=[i for i,t in enumerate(MEMBERS) if t in triples]
    assert len(ids)==35
    assert np.allclose(X[ids].T@X[ids]/35,X.T@X/455,atol=1e-14)
    return ids

def g_weights(rows):
    u,s,vt=np.linalg.svd(rows,full_matrices=False);r=int(np.sum(s>s[0]*1e-12))
    z=rows@vt[:r].T
    p=np.ones(len(z))/len(z)
    for it in range(20000):
        v=z.T@(p[:,None]*z);leverage=np.einsum('ij,jk,ik->i',z,np.linalg.inv(v),z)
        if leverage.max()<=r*(1+1e-6):return p,z,r,float(leverage.max()),it
        p*=leverage/r;p/=p.sum()
    raise RuntimeError('G-optimal design did not converge')

def od_linbai(reveal,budget=200):
    assert budget==200
    m=(budget-120-8-4-2)/4
    active=list(range(455));values={};queries=[];pulls=[];phases=[]
    for phase in range(1,5):
        ids=steiner_support() if phase==1 else active
        p,z,rank,lev,it=g_weights(X[ids]);counts=np.ceil(m*p).astype(int)
        for j in ids:
            if j not in values:values[j]=float(reveal(j));queries.append(j)
        pulls.extend(j for j,n in zip(ids,counts) for _ in range(int(n)))
        w=np.sqrt(counts);beta=np.linalg.lstsq(X[ids]*w[:,None],np.array([values[j] for j in ids])*w,rcond=None)[0]
        prediction=X[active]@beta
        active=sorted(active,key=lambda j:(round(float(prediction[active.index(j)]),12),j))[:int(np.ceil(15/2**phase))]
        phases.append(dict(phase=phase,measurement_ids=ids,counts=counts.tolist(),rank=rank,
                           maximum_leverage=lev,g_iterations=it,active=active.copy()))
    assert len(pulls)<=budget and active[0] in values
    best=min(queries,key=lambda i:(values[i],i))
    return dict(method='od_linbai_original',budget=budget,finish='native_prediction',
                queries=queries,observed=[values[j] for j in queries],winner=active[0],
                best_observed=best,pulls=pulls,phases=phases)
