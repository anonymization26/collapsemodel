import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
from policies import design_basis
from run_fresh import table,save,sha

def main(root):
    candidates=list(combinations(range(15),3));pairs=list(combinations(range(15),2))
    linear,_=design_basis(candidates,15)
    pair=np.array([[int(set(p)<=set(c)) for p in pairs] for c in candidates],dtype=float)
    folds=np.array_split(np.random.default_rng(20261117).permutation(455),5);rows=[]
    with threadpool_limits(limits=1):
        for unit in sorted(root.iterdir()):
            if not unit.is_dir() or not unit.name.startswith(('optdigits_','domainnet_')) or not (unit/'audit.json').exists():continue
            y=np.load(unit/'models.npz',allow_pickle=False)['validation_brier']
            for name,x in [('additive15',linear),('pair105',pair)]:
                pred=np.empty(455)
                for test in folds:
                    train=np.setdiff1d(np.arange(455),test)
                    beta=np.linalg.lstsq(x[train],y[train],rcond=None)[0];pred[test]=x[test]@beta
                rows.append(dict(unit=unit.name,model=name,parameters=x.shape[1],oof_r2=float(1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)),
                    oof_rmse=float(np.sqrt(np.mean((y-pred)**2))),oof_spearman=float(spearmanr(y,pred).statistic)))
    table(root/'exploratory_interactions.csv',rows)
    save(root/'exploratory_interactions_provenance.json',dict(code_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('ExploratoryInteraction.md')),test_data_used=False,post_hoc=True))
    print('INTERACTION DIAGNOSTIC',len(rows)//2,'units',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
