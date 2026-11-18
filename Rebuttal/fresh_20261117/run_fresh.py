"""Fresh logistic fits, locked selections, test audit, and independent timing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from policies import design_basis,run_policy,method_specs


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(p,o):
    Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')


def table(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def normalized(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def model_fit(x,y,c):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model=LogisticRegression(C=1,fit_intercept=False,solver='lbfgs',tol=1e-6,max_iter=1000).fit(x,y)
    w=np.zeros((x.shape[1],c));mask=np.zeros(c,dtype=bool);mask[model.classes_]=True
    if len(model.classes_)==2:
        w[:,model.classes_[0]]=-model.coef_[0]/2;w[:,model.classes_[1]]=model.coef_[0]/2
    else:w[:,model.classes_]=model.coef_.T
    return w,mask


def probability(x,w,mask):
    z=x@w;z[:,~mask]=-np.inf
    return softmax(z,axis=1)


def metrics(p,y):
    b=p.copy();b[np.arange(len(y)),y]-=1
    return dict(brier=float(np.mean(np.sum(b*b,axis=1))),accuracy=float(np.mean(p.argmax(axis=1)==y)),
                nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-15)).mean()))


def load_selection(unit):
    meta=json.loads((unit/'manifest.json').read_text())
    assert sha(unit/'selection.npz')==meta['selection_sha256']
    z=np.load(unit/'selection.npz',allow_pickle=False)
    return meta,{k:np.asarray(z[k]) for k in z.files}


def rankings(data,members):
    x=data['source_x'];blocks=data['blocks'];t=data['screen_x'];ct=t.T@t/len(t)
    gs=[x[b].T@x[b] for b in blocks];mmd=[];aopt=[]
    n=3*len(blocks[0]);eye=np.eye(x.shape[1])
    start=perf_counter()
    for combo in members:
        g=sum(gs[i] for i in combo);mmd.append(float(np.sum((g/n-ct)**2)))
    mt=perf_counter()-start
    start=perf_counter()
    for combo in members:
        g=sum(gs[i] for i in combo);aopt.append(float(np.trace(np.linalg.solve(eye+g,ct))))
    at=perf_counter()-start
    return dict(mmd=np.argsort(mmd,kind='stable').tolist(),aopt=np.argsort(aopt,kind='stable').tolist()),dict(mmd=mt,aopt=at)


def select(unit,out):
    out.mkdir(parents=True,exist_ok=False)
    meta,data=load_selection(unit);c=meta['classes'];members=list(combinations(range(15),3))
    design,basis=design_basis(members,15)
    save(out/'freeze.json',dict(manifest_sha256=sha(unit/'manifest.json'),
        sources={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('policies.py'),Path(__file__).with_name('Protocol.md')]},
        python=platform.python_version(),numpy=np.__version__,test_opened=False))
    with threadpool_limits(limits=1):
        rank,timing=rankings(data,members)
        weights=[];masks=[];loss=[];tick=perf_counter()
        for j,combo in enumerate(members):
            ix=np.concatenate([data['blocks'][i] for i in combo])
            w,m=model_fit(data['source_x'][ix],data['source_y'][ix],c)
            value=metrics(probability(data['validation_x'],w,m),data['validation_y'])['brier']
            weights.append(w);masks.append(m);loss.append(value)
            if j%100==0:print(unit.name,'fits',j,flush=True)
        audit_fit_seconds=perf_counter()-tick
        decisions=[]
        for name,mode,budget,repeat in method_specs(meta['seed']):
            if mode in ('random','init_random'):order=np.random.default_rng(meta['seed']+repeat).permutation(455).tolist()
            elif mode in ('mmd','init_mmd'):order=rank['mmd']
            elif mode=='aopt':order=rank['aopt']
            else:order=list(range(455))
            observed=[]
            def reveal(i):observed.append(i);return loss[i]
            d=run_policy(mode,design,basis,reveal,order,budget)
            assert observed==d['queries'] and len(observed)==budget
            decisions.append(dict(method=name,repeat=repeat,budget=budget,**d))
    np.savez_compressed(out/'models.npz',weights=np.stack(weights),masks=np.stack(masks),validation_brier=np.array(loss))
    save(out/'decisions.json',dict(meta=meta,decisions=decisions,rankings=rank,ranking_seconds=timing,
        audit_full_fit_seconds=audit_fit_seconds,warning='Full fits reused only for quality audit, not deployment timing'))
    save(out/'selection_lock.json',dict(decisions=sha(out/'decisions.json'),models=sha(out/'models.npz'),freeze=sha(out/'freeze.json')))
    print('LOCKED',unit.name,len(decisions),'selection paths before any test prediction',flush=True)


def audit(unit,out):
    lock=json.loads((out/'selection_lock.json').read_text())
    assert lock==dict(decisions=sha(out/'decisions.json'),models=sha(out/'models.npz'),freeze=sha(out/'freeze.json'))
    d=json.loads((out/'decisions.json').read_text());meta=d['meta']
    assert sha(unit/'manifest.json')==json.loads((out/'freeze.json').read_text())['manifest_sha256']
    assert sha(unit/'test.npz')==meta['test_sha256']
    test=np.load(unit/'test.npz',allow_pickle=False);model=np.load(out/'models.npz',allow_pickle=False)
    decisions=d['decisions']; losses=model['validation_brier'];w=model['weights'];mask=model['masks']
    selected=sorted({r['winner'] for r in decisions})
    with threadpool_limits(limits=1):
        evaluated={i:metrics(probability(test['x'],w[i],mask[i]),test['y']) for i in selected}
    full=next(r['winner'] for r in decisions if r['method']=='full455')
    seq=next(r['winner'] for r in decisions if r['method']=='sequential25')
    mmd=next(r['winner'] for r in decisions if r['method']=='mmd50')
    rows=[]
    for r in decisions:
        assert r['observed']==[float(losses[i]) for i in r['queries']]
        win=min(r['queries'],key=lambda i:(losses[i],i));assert win==r['winner']
        value=evaluated[win]
        rows.append(dict(unit=unit.name,method=r['method'],repeat=r['repeat'],budget=r['budget'],winner=win,
            **value,full_brier=evaluated[full]['brier'],gap_full=value['brier']-evaluated[full]['brier'],
            gap_seq=value['brier']-evaluated[seq]['brier'],gap_mmd50=value['brier']-evaluated[mmd]['brier'],same_seq=int(win==seq)))
    table(out/'outcomes.csv',rows)
    design,basis=design_basis(list(combinations(range(15),3)),15);diag=[]
    with threadpool_limits(limits=1):
        for r in decisions:
            if r['method'] not in ('sequential25','batch25','exploit25','init_mmd25'):continue
            held=sorted(set(range(455))-set(r['queries']));actual=losses[held]
            for n in (15,25):
                beta=np.linalg.lstsq(design[r['queries'][:n]],r['observed'][:n],rcond=None)[0];pred=design[held]@beta
                sse=float(np.sum((actual-pred)**2));sst=float(np.sum((actual-actual.mean())**2))
                diag.append(dict(unit=unit.name,method=r['method'],n=n,heldout=len(held),
                    rmse=float(np.sqrt(sse/len(held))),r2=1-sse/sst if sst else None,
                    spearman=float(spearmanr(actual,pred).statistic),
                    top25_recall=len(set(np.argsort(actual,kind='stable')[:25])&set(np.argsort(pred,kind='stable')[:25]))/25))
        pred=design@np.linalg.lstsq(design,losses,rcond=None)[0]
    table(out/'diagnostics.csv',diag)
    save(out/'audit.json',dict(status='complete',selection_lock=sha(out/'selection_lock.json'),
        full_additive_r2=float(1-np.sum((losses-pred)**2)/np.sum((losses-losses.mean())**2)),
        validation_span=float(losses.max()-losses.min()),outcomes_sha256=sha(out/'outcomes.csv')))
    print('AUDITED',unit.name,{r['method']:r['brier'] for r in rows if r['repeat']==0},flush=True)


def time_policy(data,c,method,workers):
    members=list(combinations(range(15),3));memo={};records=[];start=perf_counter()
    design,basis=design_basis(members,15)
    rank=None
    if method in ('mmd25','mmd50','aopt25'):
        # Ranking includes the moments and only the requested score, not both algorithms.
        x=data['source_x'];t=data['screen_x'];ct=t.T@t/len(t)
        gs=[x[b].T@x[b] for b in data['blocks']];score=[];n=3*len(data['blocks'][0])
        for combo in members:
            g=sum(gs[i] for i in combo)
            score.append(float(np.trace(np.linalg.solve(np.eye(x.shape[1])+g,ct))) if method=='aopt25' else float(np.sum((g/n-ct)**2)))
        rank=np.argsort(score,kind='stable').tolist()
    def evaluate(i):
        tick=perf_counter();ix=np.concatenate([data['blocks'][b] for b in members[i]])
        w,mask=model_fit(data['source_x'][ix],data['source_y'][ix],c)
        value=metrics(probability(data['validation_x'],w,mask),data['validation_y'])['brier']
        return i,value,w,mask,perf_counter()-tick
    with ThreadPoolExecutor(max_workers=workers) as executor:
        def batch(indices):
            for i,v,w,m,t in executor.map(evaluate,indices):memo[i]=(v,w,m,t);records.append(i)
        if method in ('mmd25','mmd50','aopt25','full455'):
            k=455 if method=='full455' else (50 if method=='mmd50' else 25)
            batch(list(range(455)) if method=='full455' else rank[:k])
        else:
            batch(basis)
            while len(records)<25:
                x=design[records];beta=np.linalg.lstsq(x,[memo[i][0] for i in records],rcond=None)[0]
                leverage=np.linalg.norm(np.linalg.solve(np.linalg.cholesky(x.T@x),design.T),axis=0)
                score=np.round(design@beta-.001*leverage,12);score[records]=np.inf
                order=np.argsort(score,kind='stable')
                batch(order[:25-len(records)].tolist() if method=='batch25' else [int(order[0])])
    best=min(records,key=lambda i:(memo[i][0],i));elapsed=perf_counter()-start
    return dict(method=method,workers=workers,fits=len(records),seconds=elapsed,
                worker_sum_seconds=sum(memo[i][3] for i in records),winner=best,validation_brier=memo[best][0])


def timing(unit,out):
    meta,data=load_selection(unit)
    decisions=json.loads((out/'decisions.json').read_text())['decisions']
    reference={r['method']:r for r in decisions if r['repeat']==0};rows=[]
    methods=['sequential25','batch25','mmd25','mmd50','aopt25','full455']
    with threadpool_limits(limits=1):
        for workers in (1,4):
            for repeat in range(3):
                for method in methods[repeat:]+methods[:repeat]:
                    r=time_policy(data,meta['classes'],method,workers)
                    assert r['winner']==reference[method]['winner']
                    assert abs(r['validation_brier']-min(reference[method]['observed']))<1e-10
                    rows.append(dict(unit=unit.name,repeat=repeat,**r));print('TIMING',rows[-1],flush=True)
    table(out/'timing.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['select','audit','timing']);p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.phase](a.unit,a.out)
