"""Run selection for all cells before test auditing; time actual fits last."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import json,os,subprocess,sys,time,hashlib

BASE=Path(__file__).resolve().parent.parent
CODE=Path(__file__).resolve().parent
OUT=BASE/'linear_runs'
DATA=Path('/mnt/datadisk0/datasets/collapsemodel_20261117/fresh')
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(stage,status='running',**extra):
    (OUT/'status.json').write_text(json.dumps(dict(stage=stage,status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra),indent=2)+'\n')
    print(stage,status,flush=True)
def call(args,log):
    with (OUT/log).open('a') as f:subprocess.run([sys.executable,'-u',str(CODE/'run_comparison.py'),*map(str,args)],env=ENV,check=True,stdout=f,stderr=subprocess.STDOUT)

def main():
    OUT.mkdir(exist_ok=True)
    assert json.loads((OUT/'implementation_checks.json').read_text())['status']=='passed'
    sources={str(p.relative_to(BASE)):sha(p) for root in (CODE,BASE/'fresh_20261117') for p in root.glob('*') if p.suffix in ('.py','.md')}
    (OUT/'source_freeze.json').write_text(json.dumps(sources,indent=2)+'\n')
    units=sorted(p for p in DATA.iterdir() if p.is_dir() and
                 (p.name.startswith('optdigits_') or (p.name.startswith('domainnet_') and '_d128_' in p.name)))
    assert len(units)==56,len(units)
    old=BASE/'runs'
    state('geometry_designs')
    if not (OUT/'designs.json').exists():call(['design','--out',OUT/'designs.json'],'design.log')
    state('select_all_without_test',units=len(units))
    def select(u):
        o=OUT/u.name
        if (o/'selection_lock.json').exists():return
        assert not o.exists(),f'Partial output requires inspection: {o}'
        call(['select','--old',old/u.name,'--out',o,'--design',OUT/'designs.json'],u.name+'.log')
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(select,u) for u in units]):f.result()
    all_locks={u.name:sha(OUT/u.name/'selection_lock.json') for u in units}
    (OUT/'all_selection_locks.json').write_text(json.dumps(all_locks,indent=2)+'\n')
    state('test_audit_after_all_locks')
    def audit(u):
        o=OUT/u.name
        if (o/'audit.json').exists():return
        call(['audit','--old',old/u.name,'--out',o,'--unit',u],u.name+'.log')
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(audit,u) for u in units]):f.result()
    state('independent_fit_timing')
    timed=[u for u in units if '_hash_' in u.name and u.name.endswith('_20261117')]
    assert len(timed)==7
    for u in timed:
        call(['timing','--unit',u,'--out',OUT/u.name],'timing_'+u.name+'.log')
    for name,digest in sources.items():assert sha(BASE/name)==digest,name
    state('all_experiments','complete',units=len(units),timing_runs=7*9*3)

if __name__=='__main__':
    try:main()
    except Exception as e:state('failure','failed',error=repr(e));raise
