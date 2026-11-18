"""Independent artifact, budget, deployment and timing checks."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent.parent;ROOT=BASE/'linear_runs'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())

def main():
    assert load(ROOT/'status.json')['status']=='complete'
    freeze=load(ROOT/'source_freeze.json')
    for name,digest in freeze.items():assert sha(BASE/name)==digest,name
    locks=load(ROOT/'all_selection_locks.json');assert len(locks)==56
    decisions=timings=0
    for name,lockhash in locks.items():
        p=ROOT/name;old=BASE/'runs'/name
        assert sha(p/'selection_lock.json')==lockhash
        for key,digest in load(p/'selection_lock.json').items():assert sha(p/(key+'.json'))==digest
        audit=load(p/'audit.json');assert audit['status']=='complete'
        assert sha(p/'outcomes.csv')==audit['outcomes']
        assert sha(old/'selection_lock.json')==audit['old_lock']
        # Reference model hash is independently recomputed, not just trusted.
        oldlock=load(old/'selection_lock.json');assert sha(old/'models.npz')==oldlock['models']
        loss=np.load(old/'models.npz')['validation_brier'];full=int(np.argmin(loss))
        dd=load(p/'decisions.json')['decisions'];rr=list(csv.DictReader((p/'outcomes.csv').open()))
        assert len(dd)==len(rr)==109
        for d,r in zip(dd,rr):
            q=d['queries'];assert len(set(q))==len(q);assert np.array_equal(loss[q],d['observed'])
            assert d['winner'] in q and int(r['winner'])==d['winner']
            assert abs(float(r['validation_regret'])-(loss[d['winner']]-loss[full]))<1e-12
            assert float(r['validation_regret'])>=-1e-14
            best=min(q,key=lambda j:(loss[j],j));assert best==d['best_observed']
            if d['finish']=='observed':assert d['winner']==best
            if d['method']=='od_linbai_original':
                assert len(d['pulls'])<=200
                assert set(d['pulls'])==set(q)
                assert [len(x['active']) for x in d['phases']]==[8,4,2,1]
                assert all(x['maximum_leverage']<=x['rank']*(1+1e-6) for x in d['phases'])
            else:assert len(q)==d['budget']
            decisions+=1
        if (p/'timing.csv').exists():
            tt=list(csv.DictReader((p/'timing.csv').open()));assert len(tt)==27
            for t in tt:
                d=next(d for d in dd if d['method']==t['method'] and d['finish']==t['finish'] and d['budget']==25)
                assert int(t['winner'])==d['winner']
                assert abs(float(t['validation_brier'])-min(d['observed']))<1e-10
                assert abs(float(t['seconds'])-sum(float(t[k]) for k in ('fit_seconds','validation_seconds','allocation_seconds')))<1e-9
                timings+=1
    assert decisions==6104 and timings==189
    result=dict(status='passed',units=56,decisions=decisions,independent_timings=timings,
                frozen_sources=len(freeze),model_hashes_checked=56,all_selection_locks=sha(ROOT/'all_selection_locks.json'))
    (ROOT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':main()
