"""Check pool coverage and matched construction inputs; never inspect test labels."""
import argparse
from pathlib import Path
import numpy as np
from run_fresh import load_selection,save,sha

def main(root,out):
    checked=[]
    for unit in sorted(root.glob('*_hash_*')):
        if not unit.is_dir() or not (unit/'manifest.json').exists():continue
        other=unit.with_name(unit.name.replace('_hash_','_pc1_'))
        if not (other/'manifest.json').exists():continue
        meta,d=load_selection(unit);meta2,e=load_selection(other)
        for key in ['source_x','source_y','screen_x','validation_x','validation_y']:
            np.testing.assert_array_equal(d[key],e[key])
        assert meta['test_sha256']==meta2['test_sha256']
        for data in [d,e]:
            blocks=data['blocks'];assert blocks.shape[0]==15
            np.testing.assert_array_equal(np.sort(blocks.flatten()),np.arange(len(data['source_x'])))
            for key in ['source_x','screen_x','validation_x']:
                assert np.isfinite(data[key]).all()
                norm=np.linalg.norm(data[key],axis=1)
                assert np.all((np.abs(norm-1)<1e-10)|(norm==0))
        target_sets=[set(meta[k]) for k in ['target_screen_rows','target_validation_rows','target_test_rows']]
        assert all(not (target_sets[i]&target_sets[j]) for i in range(3) for j in range(i+1,3))
        if 'source_global_rows' in meta:
            assert all(not (set(meta['source_global_rows'])&target) for target in target_sets)
        checked.append(dict(hash_unit=unit.name,pc1_unit=other.name,source_rows=len(d['source_x']),pool_size=d['blocks'].shape[1]))
    save(out,dict(status='passed',note='No test predictions or labels opened',pairs=checked,checker_sha256=sha(__file__)))
    print('VERIFIED',len(checked),'matched construction pairs: identical data, disjoint target roles, complete disjoint pools',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
