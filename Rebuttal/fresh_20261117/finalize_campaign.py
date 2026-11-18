"""Finish diagnostics and artifact verification after the compute campaign."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_fresh import save

def main(base,data):
    run=base/'runs';code=Path(__file__).resolve().parent
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'0'}
    def status(stage,state='running',**extra):save(run/'finalization_status.json',dict(stage=stage,state=state,**extra))
    def call(script,args):subprocess.run([sys.executable,'-u',str(code/script),*map(str,args)],check=True,env=env)
    try:
        status('waiting_campaign')
        while True:
            s=json.loads((run/'campaign_status.json').read_text())
            if s['state']=='failed':raise RuntimeError(s)
            if s['state']=='complete':break
            time.sleep(15)
        status('cost_decomposition')
        out=run/'cost_breakdown_clipart.json'
        if not out.exists():call('cost_breakdown.py',['--unit',data/'fresh/domainnet_clipart_hash_d128_20261117','--out',out])
        status('verification')
        call('check_core.py',[])
        call('check_units.py',['--root',data/'fresh','--out',run/'data_verification.json'])
        call('audit_outputs.py',['--root',run])
        call('interaction_diagnostic.py',['--root',run])
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in code.iterdir() if p.is_file() and p.name!='source_manifest.json'}
        save(run/'final_source_manifest.json',source)
        status('report')
        call('summarize.py',['--root',run])
        status('all_artifacts','complete')
    except Exception as exc:
        status('failure','failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True);a=p.parse_args();main(a.base,a.data)
