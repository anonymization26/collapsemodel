"""Ridge sensitivity: disjoint calibration/selection; locked test audit."""
import argparse
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from policies import design_basis,run_policy,method_specs
from run_fresh import load_selection,rankings,metrics,save,sha,table

LAMBDAS=[.1,1.,10.]
TEMPERATURES=np.exp(np.linspace(np.log(.01),np.log(100),81))
ARMS=['uncalibrated_fixed','calibrated_fixed','calibrated_tuned']

def choose_temperature(logits,y):
    p=softmax(logits[None,:,:]/TEMPERATURES[:,None,None],axis=2)
    p[:,np.arange(len(y)),y]-=1
    scores=np.mean(np.sum(p*p,axis=2),axis=1)
    k=int(np.argmin(scores));return float(TEMPERATURES[k]),float(scores[k])

def main(unit,out):
    out.mkdir(parents=True,exist_ok=False);meta,d=load_selection(unit)
    save(out/'freeze.json',dict(manifest_sha256=sha(unit/'manifest.json'),code_sha256=sha(__file__),
        policies_sha256=sha(Path(__file__).with_name('policies.py')),protocol_sha256=sha(Path(__file__).with_name('Protocol.md')),
        validation_rule='first half calibration only, second half candidate selection only',lambdas=LAMBDAS,
        temperatures=TEMPERATURES.tolist(),test_opened=False))
    members=list(combinations(range(15),3));design,basis=design_basis(members,15)
    split=len(d['validation_y'])//2;cx=d['validation_x'][:split];cy=d['validation_y'][:split]
    vx=d['validation_x'][split:];vy=d['validation_y'][split:];c=meta['classes']
    weights=[];temps=[];chosen_lambda=[];val=[];cost=[];tick=perf_counter()
    with threadpool_limits(limits=1):
        ranking,_=rankings(d,members)
        for j,combo in enumerate(members):
            ix=np.concatenate([d['blocks'][b] for b in combo]);x=d['source_x'][ix];y=d['source_y'][ix]
            gram=x.T@x;rhs=x.T@np.eye(c)[y];ws=[];ts=[];scores=[];fittimes=[];caltimes=[]
            for lam in LAMBDAS:
                begin=perf_counter();w=np.linalg.solve(gram+lam*np.eye(x.shape[1]),rhs);fittimes.append(perf_counter()-begin)
                begin=perf_counter();t,score=choose_temperature(cx@w,cy);caltimes.append(perf_counter()-begin)
                ws.append(w);ts.append(t);scores.append(score)
            best=int(np.argmin(scores));arm_w=np.stack([ws[1],ws[1],ws[best]]);arm_t=np.array([1.,ts[1],ts[best]])
            weights.append(arm_w);temps.append(arm_t);chosen_lambda.append([1.,1.,LAMBDAS[best]])
            val.append([metrics(softmax(vx@w/t,axis=1),vy)['brier'] for w,t in zip(arm_w,arm_t)])
            cost.append(dict(candidate=j,fit_seconds=fittimes,calibration_seconds=caltimes))
        weights=np.stack(weights);temps=np.stack(temps);val=np.asarray(val);decisions=[]
        for arm_idx,arm in enumerate(ARMS):
            for name,mode,budget,repeat in method_specs(meta['seed']):
                order=np.random.default_rng(meta['seed']+repeat).permutation(455).tolist() if mode in ('random','init_random') else ranking['mmd'] if mode in ('mmd','init_mmd') else ranking['aopt'] if mode=='aopt' else list(range(455))
                r=run_policy(mode,design,basis,lambda i:val[i,arm_idx],order,budget)
                decisions.append(dict(arm=arm,arm_index=arm_idx,method=name,repeat=repeat,budget=budget,
                    supervised_fits=budget*(3 if arm=='calibrated_tuned' else 1),**r))
        np.savez_compressed(out/'models.npz',weights=weights,temperatures=temps,validation_brier=val,lambdas=chosen_lambda)
        save(out/'decisions.json',dict(decisions=decisions,costs=cost,audit_seconds=perf_counter()-tick,
            cost_note='Shared full-family audit; measured component times are NOT independent deployment wall times'))
        save(out/'selection_lock.json',dict(decisions=sha(out/'decisions.json'),models=sha(out/'models.npz'),freeze=sha(out/'freeze.json')))
        assert sha(unit/'test.npz')==meta['test_sha256'];test=np.load(unit/'test.npz',allow_pickle=False)
        cache={};rows=[]
        for r in decisions:
            i,a=r['winner'],r['arm_index'];key=(i,a)
            if key not in cache:cache[key]=metrics(softmax(test['x']@weights[i,a]/temps[i,a],axis=1),test['y'])
            rows.append(dict(unit=unit.name,arm=r['arm'],method=r['method'],repeat=r['repeat'],budget=r['budget'],
                fits=r['supervised_fits'],winner=i,temperature=float(temps[i,a]),lambda_value=chosen_lambda[i][a],**cache[key]))
        table(out/'outcomes.csv',rows)
        save(out/'audit.json',dict(status='complete',selection_lock=sha(out/'selection_lock.json'),outcomes_sha256=sha(out/'outcomes.csv')))
    print('RIDGE COMPLETE',unit.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.unit,a.out)
