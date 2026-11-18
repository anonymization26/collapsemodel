"""One additional instrumented execution per method; not a speedup estimate.

Keep the frozen primary time_policy implementation unchanged. Wrap its existing
fit/probability/metric calls for a serial decomposition. This diagnostic must run
after the campaign's other compute tasks.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
import run_fresh as rf

def main(unit,out):
    assert not out.exists();meta,data=rf.load_selection(unit);originals={k:getattr(rf,k) for k in ['model_fit','probability','metrics']};rows=[]
    reference={r['method']:r for r in json.loads((out.parent/unit.name/'decisions.json').read_text())['decisions'] if r['repeat']==0}
    with threadpool_limits(limits=1):
        for method in ['sequential25','batch25','mmd25','mmd50','aopt25','full455']:
            durations={k:[] for k in originals}
            def wrap(name):
                def call(*args,**kwargs):
                    start=perf_counter();result=originals[name](*args,**kwargs);durations[name].append(perf_counter()-start);return result
                return call
            for name in originals:setattr(rf,name,wrap(name))
            try:r=rf.time_policy(data,meta['classes'],method,1)
            finally:
                for name,original in originals.items():setattr(rf,name,original)
            assert len(durations['model_fit'])==r['fits']
            assert r['winner']==reference[method]['winner']
            assert abs(r['validation_brier']-min(reference[method]['observed']))<1e-10
            fit=sum(durations['model_fit']);validation=sum(durations['probability'])+sum(durations['metrics'])
            residual=r['seconds']-fit-validation;assert residual>=0
            rows.append(dict(**r,fit_seconds=fit,validation_seconds=validation,ranking_acquisition_dispatch_seconds=residual))
        loading=[]
        for repeat in range(3):
            start=perf_counter();check=rf.sha(unit/'selection.npz');hash_seconds=perf_counter()-start;assert check==meta['selection_sha256']
            start=perf_counter()
            with np.load(unit/'selection.npz',allow_pickle=False) as z:cache={k:z[k] for k in z.files}
            loading.append(dict(repeat=repeat,cache_decode_seconds=perf_counter()-start,integrity_hash_seconds=hash_seconds))
    rf.save(out,dict(unit=unit.name,code_sha256=rf.sha(__file__),timed_implementation_sha256=rf.sha(rf.__file__),rows=rows,loading=loading,
        note='One extra serial execution per method; component measurement only. OS file cache uncontrolled. Not primary 3-repeat speedup estimates.'))
    print('COST BREAKDOWN',unit.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.unit,a.out)
