"""Download official Optdigits; freeze new pool constructions and target roles."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import requests
from run_fresh import normalized,save,sha

URL='https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip'


def order(n,tag):return sorted(range(n),key=lambda i:hashlib.sha256(f'{tag}:{i}'.encode()).digest())


def main(root):
    root.mkdir(parents=True,exist_ok=True);archive=root/'optdigits.zip'
    if not archive.exists():
        r=requests.get(URL,timeout=90);r.raise_for_status();archive.write_bytes(r.content)
    with zipfile.ZipFile(archive) as z:
        def read(name):
            match=next(n for n in z.namelist() if n.endswith(name));return np.loadtxt(io.BytesIO(z.read(match)),delimiter=',')
        train,test=read('optdigits.tra'),read('optdigits.tes')
    assert train.shape==(3823,65) and test.shape==(1797,65)
    x,y=normalized(train[:,:64]/16),train[:,-1].astype(int)
    t,ty=normalized(test[:,:64]/16),test[:,-1].astype(int)
    # Writer partitions are official; byte-identical feature rows are separately reported.
    duplicates=len(set(map(tuple,train[:,:64]))&set(map(tuple,test[:,:64])))
    for seed in range(20261117,20261121):
        src=order(len(x),f'source:{seed}')[:3000];target=order(len(t),f'target:{seed}')
        screen,val,hold=target[:256],target[256:768],target[768:]
        sx=x[src];sy=y[src]
        u,s,v=np.linalg.svd(sx-sx.mean(axis=0),full_matrices=False);direction=v[0]
        if direction[np.argmax(abs(direction))]<0:direction=-direction
        pc=sx@direction
        for construction in ('hash','pc1'):
            unit=root/f'optdigits_{construction}_{seed}';unit.mkdir(exist_ok=False)
            block_order=np.arange(3000) if construction=='hash' else np.argsort(pc,kind='stable')
            np.savez_compressed(unit/'selection.npz',source_x=sx,source_y=sy,blocks=block_order.reshape(15,200),
                screen_x=t[screen],validation_x=t[val],validation_y=ty[val])
            np.savez_compressed(unit/'test.npz',x=t[hold],y=ty[hold])
            save(unit/'manifest.json',dict(dataset='optdigits',seed=seed,construction=construction,classes=10,
                source_url=URL,archive_sha256=sha(archive),selection_sha256=sha(unit/'selection.npz'),test_sha256=sha(unit/'test.npz'),
                source_rows=src,target_screen_rows=screen,target_validation_rows=val,target_test_rows=hold,
                cross_official_split_identical_features=duplicates,features='native 64D counts /16, row L2 normalized',
                pool_note='artificial equal-size source pools; not natural source domains'))
            print('PREPARED',unit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
