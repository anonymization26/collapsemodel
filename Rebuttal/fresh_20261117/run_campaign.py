"""Resumable campaign driver. A failed scientific fit is never silently retried."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from run_fresh import save,sha

def main(base,data,download_pid):
    code=Path(__file__).resolve().parent;run=base/'runs';run.mkdir(exist_ok=True)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'0','TORCH_HOME':str(base/'torch_home')}
    status=run/'campaign_status.json'
    def stage(name,state='running',**extra):
        save(status,dict(stage=name,state=state,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra));print(name,state,flush=True)
    def call(script,args,log,gpu=False):
        cmd=[sys.executable,'-u',str(code/script),*map(str,args)]
        if gpu:cmd=['flock','/home/ubuntu/caozheng/gpu0.lock',*cmd]
        with (run/log).open('a') as f:subprocess.run(cmd,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
    try:
        stage('waiting_fresh_download')
        if download_pid:
            while Path(f'/proc/{download_pid}').exists():
                text=Path(f'/proc/{download_pid}/stat').read_text()
                if text.split()[2]=='Z':break
                time.sleep(10)
        stage('data_integrity')
        call('prepare_domainnet.py',['--root',data/'domainnet','--verify'],'domainnet_integrity.log')
        stage('gpu0_feature_extraction')
        call('domainnet_features.py',['extract','--root',data/'domainnet'],'domainnet_features.log',gpu=True)
        stage('prepare_projected_units')
        call('domainnet_features.py',['units','--root',data/'domainnet','--out',data/'fresh'],'domainnet_units.log')
        all_units=sorted((data/'fresh').glob('domainnet_*'))
        primary=[p for p in all_units if '_d128_' in p.name]
        assert len(primary)==48
        stage('logistic_quality',units=len(primary))
        def quality(unit):
            out=run/unit.name
            if (out/'audit.json').exists():return
            if not out.exists():call('run_fresh.py',['select','--unit',unit,'--out',out],unit.name+'.log')
            assert (out/'selection_lock.json').exists(),f'Incomplete fit requires investigation: {out}'
            call('run_fresh.py',['audit','--unit',unit,'--out',out],unit.name+'.log')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(quality,u) for u in primary]):f.result()
        ridge=[p for p in all_units if '_hash_' in p.name and p.name.endswith('_20261117')]
        assert len(ridge)==30
        stage('ridge_dimensions',units=len(ridge))
        def ridge_run(unit):
            out=run/('ridge_'+unit.name)
            if (out/'audit.json').exists():return
            assert not out.exists(),f'Incomplete ridge fit requires investigation: {out}'
            call('run_ridge.py',['--unit',unit,'--out',out],'ridge_'+unit.name+'.log')
        with ThreadPoolExecutor(max_workers=4) as pool:
            for f in as_completed([pool.submit(ridge_run,u) for u in ridge]):f.result()
        stage('independent_resident_timing')
        for unit in primary:
            if '_hash_' not in unit.name or not unit.name.endswith('_20261117'):continue
            out=run/unit.name
            if not (out/'timing.csv').exists():call('run_fresh.py',['timing','--unit',unit,'--out',out],'timing_'+unit.name+'.log')
        stage('cold_image_pipeline')
        unit=data/'fresh/domainnet_clipart_hash_d128_20261117'
        for method in ['sequential25','batch25','mmd50','full455']:
            out=run/f'cold_{method}.json'
            if not out.exists():call('cold_images.py',['--root',data/'domainnet','--unit',unit,'--out',out,'--method',method,'--reference',run/unit.name],'cold_images.log',gpu=True)
        stage('all_experiments','complete')
    except Exception as exc:
        stage('failure','failed',error=str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--download-pid',type=int);a=p.parse_args();main(a.base,a.data,a.download_pid)
