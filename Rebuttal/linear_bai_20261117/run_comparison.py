"""Replay selection in a sealed phase, then audit test predictions separately."""
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'fresh_20261117'))
from run_fresh import sha,save,table,metrics,probability,load_selection,model_fit
from linear_policies import X,run,static_sequence,od_linbai

METHODS=['seq','batch','exploit','d','a','g','xy','lingape_0.001','lingape_0.01','lingape_0.1','lingape_1']
TIMED=[('seq','observed'),('batch','observed'),('exploit','observed'),('d','recommend'),
       ('a','recommend'),('g','recommend'),('xy','recommend'),('lingape_0.001','recommend'),('lingape_0.1','recommend')]

def inputs(old):
    lock=json.loads((old/'selection_lock.json').read_text())
    for name in ('decisions','models','freeze'):
        assert lock[name]==sha(old/(name+('.npz' if name=='models' else '.json')))
    d=json.loads((old/'decisions.json').read_text())
    losses=np.load(old/'models.npz')['validation_brier']
    return d,losses

def select(old,out,design_file):
    out.mkdir(exist_ok=False,parents=True);d,losses=inputs(old)
    static=json.loads(design_file.read_text())['sequences'];orders={'mmd':d['rankings']['mmd'],'target_aopt':d['rankings']['aopt']}
    save(out/'freeze.json',dict(old_lock=sha(old/'selection_lock.json'),design=sha(design_file),
        sources={p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.md')},test_opened=False))
    decisions=[]
    def execute(method,b,finish):
        visible=[]
        def reveal(j):visible.append(j);return float(losses[j])
        tick=perf_counter();r=run(method,b,reveal,finish,orders,static)
        assert visible==r['queries'];r['replay_seconds']=perf_counter()-tick
        return r
    with threadpool_limits(limits=1):
        for budget in (25,50,100):
            for method in METHODS:
                for finish in ('observed','recommend','native_prediction'):
                    decisions.append(execute(method,budget,finish))
            for method in ('mmd','target_aopt'):decisions.append(execute(method,budget,'observed'))
        # Regression against the previously frozen primary algorithm at B25.
        names={'seq':'sequential25','batch':'batch25','exploit':'exploit25','mmd':'mmd25','target_aopt':'aopt25'}
        for method,name in names.items():
            now=next(r for r in decisions if r['method']==method and r['budget']==25 and r['finish']=='observed')
            previous=next(r for r in d['decisions'] if r['method']==name)
            assert now['queries']==previous['queries'] and now['winner']==previous['winner'],(old.name,method)
        visible=[]
        def reveal(j):visible.append(j);return float(losses[j])
        od=od_linbai(reveal);assert visible==od['queries'];decisions.append(od)
        fallback={**od,'finish':'observed','winner':od['best_observed']};decisions.append(fallback)
        for b,label in [(len(od['queries']),'matched_od_unique'),(200,'matched_od_cap')]:
            r=execute('seq',b,'observed');r['comparison']=label;decisions.append(r)
    save(out/'decisions.json',dict(unit=old.name,meta=d['meta'],decisions=decisions))
    save(out/'selection_lock.json',dict(decisions=sha(out/'decisions.json'),freeze=sha(out/'freeze.json')))
    print('LOCKED',old.name,len(decisions),flush=True)

def audit(old,unit,out):
    lock=json.loads((out/'selection_lock.json').read_text())
    assert lock==dict(decisions=sha(out/'decisions.json'),freeze=sha(out/'freeze.json'))
    d=json.loads((out/'decisions.json').read_text());previous,loss=inputs(old);meta=d['meta']
    assert sha(unit/'manifest.json')==json.loads((old/'freeze.json').read_text())['manifest_sha256']
    assert sha(unit/'test.npz')==meta['test_sha256']
    full=next(r['winner'] for r in previous['decisions'] if r['method']=='full455')
    seq=next(r['winner'] for r in previous['decisions'] if r['method']=='sequential25')
    test=np.load(unit/'test.npz');models=np.load(old/'models.npz');weights=models['weights'];masks=models['masks']
    ids={full,seq}|{r['winner'] for r in d['decisions']};evaluated={}
    with threadpool_limits(limits=1):
        for j in sorted(ids):evaluated[j]=metrics(probability(test['x'],weights[j],masks[j]),test['y'])
    rows=[]
    for r in d['decisions']:
        q=r['queries'];assert r['observed']==loss[q].tolist() and r['winner'] in q
        if r['method']!='od_linbai_original':assert len(q)==r['budget']==len(set(q))
        best=min(q,key=lambda j:(loss[j],j));assert best==r['best_observed']
        w=r['winner'];val=evaluated[w]
        rows.append(dict(unit=old.name,dataset='domainnet' if old.name.startswith('domainnet') else 'optdigits',
            construction='pc1' if '_pc1_' in old.name else 'hash',method=r['method'],finish=r['finish'],
            budget=r['budget'],comparison=r.get('comparison','standard'),unique_fits=len(q),pulls=len(r.get('pulls',q)),
            winner=w,same_full=int(w==full),validation_brier=float(loss[w]),validation_regret=float(loss[w]-loss[full]),
            **val,gap_full=val['brier']-evaluated[full]['brier'],gap_seq25=val['brier']-evaluated[seq]['brier'],
            full_brier=evaluated[full]['brier']))
    table(out/'outcomes.csv',rows)
    save(out/'audit.json',dict(status='complete',selection_lock=sha(out/'selection_lock.json'),
        old_lock=sha(old/'selection_lock.json'),test_sha256=sha(unit/'test.npz'),outcomes=sha(out/'outcomes.csv'),
        decision_count=len(rows),evaluated_test_models=len(ids)))
    print('AUDITED',old.name,len(rows),flush=True)

def time_one(data,classes,method,finish):
    from linear_policies import MEMBERS
    evaluations=[];tick=perf_counter()
    def reveal(j):
        start=perf_counter();ix=np.concatenate([data['blocks'][i] for i in MEMBERS[j]])
        w,m=model_fit(data['source_x'][ix],data['source_y'][ix],classes)
        fit=perf_counter()-start;start=perf_counter()
        val=metrics(probability(data['validation_x'],w,m),data['validation_y'])['brier']
        evaluations.append(dict(j=j,fit_seconds=fit,validation_seconds=perf_counter()-start))
        return val
    r=run(method,25,reveal,finish)
    seconds=perf_counter()-tick
    return dict(method=method,finish=finish,budget=25,seconds=seconds,
                fit_seconds=sum(x['fit_seconds'] for x in evaluations),
                validation_seconds=sum(x['validation_seconds'] for x in evaluations),
                allocation_seconds=seconds-sum(x['fit_seconds']+x['validation_seconds'] for x in evaluations),
                winner=r['winner'],queries=r['queries'],validation_brier=min(r['observed']))

def timing(unit,out):
    tick=perf_counter();meta,data=load_selection(unit);load=perf_counter()-tick
    d=json.loads((out/'decisions.json').read_text())['decisions'];rows=[]
    with threadpool_limits(limits=1):
        for repeat in range(3):
            for method,finish in TIMED[repeat:]+TIMED[:repeat]:
                r=time_one(data,meta['classes'],method,finish)
                ref=next(x for x in d if x['method']==method and x['finish']==finish and x['budget']==25)
                assert r['winner']==ref['winner'] and r['queries']==ref['queries']
                assert abs(r['validation_brier']-min(ref['observed']))<1e-10
                del r['queries'];rows.append(dict(unit=unit.name,repeat=repeat,workers=1,load_seconds=load,**r))
                table(out/'timing.csv',rows);print('TIMED',unit.name,repeat,method,round(r['seconds'],3),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['design','select','audit','timing'])
    p.add_argument('--old',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--unit',type=Path);p.add_argument('--design',type=Path)
    a=p.parse_args()
    if a.phase=='design':
        sequences={};times={}
        with threadpool_limits(limits=1):
            for kind in ('d','a','g','xy'):
                tick=perf_counter();sequences[kind]=static_sequence(kind);times[kind]=perf_counter()-tick
                print('DESIGN',kind,times[kind],flush=True)
        save(a.out,dict(sequences=sequences,seconds=times))
    elif a.phase=='select':select(a.old,a.out,a.design)
    elif a.phase=='audit':audit(a.old,a.unit,a.out)
    else:timing(a.unit,a.out)
