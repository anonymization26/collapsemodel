"""Pool-only policies. A policy can access losses only through reveal(index)."""
import numpy as np


def design_basis(members, pool_count):
    design=np.zeros((len(members),pool_count),dtype=np.float64)
    for i,c in enumerate(members):design[i,list(c)]=1
    residual=design.copy(); basis=[]
    for _ in range(pool_count):
        norms=np.einsum('ij,ij->i',residual,residual);norms[basis]=-1
        pick=int(np.argmax(np.round(norms,12)))
        assert norms[pick]>1e-10
        basis.append(pick);direction=residual[pick]/np.sqrt(norms[pick])
        residual-=np.outer(residual@direction,direction)
    assert np.linalg.matrix_rank(design[basis])==pool_count
    return design,basis


def run_policy(method,design,basis,reveal,order=None,budget=25):
    q=[];v=[]
    def query(indices):
        for i in indices:
            assert i not in q
            q.append(int(i));v.append(float(reveal(int(i))))
    if method in ('mmd','aopt','random','full'):
        query(order[:budget])
    else:
        query(basis)
        if method=='initial':pass
        elif method in ('init_mmd','init_random'):
            query([int(i) for i in order if i not in q][:budget-len(q)])
        else:
            assert method in ('sequential','batch','exploit')
            while len(q)<budget:
                x=design[q]
                beta=np.linalg.lstsq(x,np.asarray(v),rcond=None)[0]
                leverage=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(x.T@x),design.T),axis=0)
                score=np.round(design@beta-(0 if method=='exploit' else .001)*leverage,12)
                score[q]=np.inf
                remaining=sorted(set(range(len(design)))-set(q),key=lambda i:(score[i],i))
                query(remaining[:budget-len(q)] if method=='batch' else remaining[:1])
    assert len(q)==budget and len(set(q))==budget
    return dict(queries=q,observed=v,winner=q[min(range(len(q)),key=lambda i:(v[i],q[i]))])


def method_specs(seed):
    specs=[('sequential25','sequential',25,0),('batch25','batch',25,0),
           ('exploit25','exploit',25,0),('init_mmd25','init_mmd',25,0),('initial15','initial',15,0),
           ('mmd25','mmd',25,0),('mmd50','mmd',50,0),('aopt25','aopt',25,0),('aopt50','aopt',50,0),('full455','full',455,0)]
    specs += [(name,mode,25,r) for name,mode in [('init_random25','init_random'),('random25','random')] for r in range(20)]
    return specs
