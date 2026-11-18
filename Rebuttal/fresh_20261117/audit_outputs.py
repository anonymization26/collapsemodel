"""Independent artifact checks; audit frozen selections without refitting models."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run_fresh import sha,save

def main(root):
    checked=[]
    for unit in sorted(root.iterdir()):
        if not unit.is_dir() or not (unit/'audit.json').exists():continue
        lock=json.loads((unit/'selection_lock.json').read_text())
        for key,value in lock.items():assert sha(unit/(key+('.npz' if key=='models' else '.json')))==value,(unit,key)
        audit=json.loads((unit/'audit.json').read_text())
        assert audit['selection_lock']==sha(unit/'selection_lock.json')
        assert audit['outcomes_sha256']==sha(unit/'outcomes.csv')
        decisions=json.loads((unit/'decisions.json').read_text())['decisions']
        models=np.load(unit/'models.npz',allow_pickle=False);loss=models['validation_brier']
        for d in decisions:
            q=d['queries'];assert len(q)==len(set(q))==d['budget']
            expected=loss[:,d['arm_index']] if 'arm_index' in d else loss
            np.testing.assert_array_equal(d['observed'],expected[q])
            assert d['winner']==min(q,key=lambda i:(expected[i],i))
        actual=list(csv.DictReader((unit/'outcomes.csv').open()));assert len(actual)==len(decisions)
        for r,d in zip(actual,decisions):
            assert r['method']==d['method'] and int(r['winner'])==d['winner']
            assert 0<=float(r['brier'])<=2 and 0<=float(r['accuracy'])<=1 and float(r['nll'])>=0
        checked.append(dict(unit=unit.name,decisions=len(decisions),audit_sha256=sha(unit/'audit.json')))
    save(root/'verification.json',dict(status='passed',checked=checked,checker_sha256=sha(__file__)))
    print('VERIFIED',len(checked),'completed units; locks, query budgets, selections, metrics',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
