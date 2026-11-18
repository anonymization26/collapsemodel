"""Actual no-feature-cache pipeline, one predeclared clipart case per method.

OS page cache is uncontrolled; Python imports are excluded. Model loading, image
decoding/transforms, GPU extraction, projection, ranking, fitting and selection
are measured in one execution. No old per-stage times are added.
"""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet50,ResNet50_Weights
from PIL import Image
from threadpoolctl import threadpool_limits
from run_fresh import normalized,time_policy,save,sha

def main(root,unit,out,method,reference):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    assert not out.exists();torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    total=perf_counter();meta=json.loads((unit/'manifest.json').read_text());inv=json.loads((root/'inventory.json').read_text())
    assert meta['target']=='clipart' and meta['construction']=='hash' and meta['seed']==20261117 and meta['dimension']==128
    groups=[meta['source_global_rows'],meta['target_screen_rows'] if method=='mmd50' else [],meta['target_validation_rows']];rows=sum(groups,[])
    # Load pool IDs only; never read source_x / validation_x from the feature cache.
    z=np.load(unit/'selection.npz',allow_pickle=False);blocks=z['blocks'];z.close()
    transform=ResNet50_Weights.IMAGENET1K_V2.transforms();start=perf_counter()
    model=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2);model.fc=torch.nn.Identity();model.eval().cuda()
    torch.cuda.synchronize();setup=perf_counter()-start
    class Images(Dataset):
        def __len__(self):return len(rows)
        def __getitem__(self,i):
            with Image.open(root/'images'/inv['records'][rows[i]]['path']) as image:return transform(image.convert('RGB'))
    loader=DataLoader(Images(),batch_size=128,num_workers=4,pin_memory=True,shuffle=False)
    start=perf_counter();features=[]
    with torch.inference_mode():
        for image in loader:features.append(model(image.cuda(non_blocking=True)).cpu().numpy())
    torch.cuda.synchronize();extraction=perf_counter()-start
    start=perf_counter()
    with threadpool_limits(limits=1):
        raw=np.concatenate(features).astype(np.float64);projection=np.random.default_rng(meta['seed']).choice(np.array([-1.,1.]),size=(2048,512))
        x=normalized(normalized(raw)@projection[:,:128]);projection_seconds=perf_counter()-start
        n1,n2,n3=map(len,groups);y=np.array([inv['records'][i]['label'] for i in rows])
        data=dict(source_x=x[:n1],source_y=y[:n1],blocks=blocks,screen_x=x[n1:n1+n2],validation_x=x[n1+n2:],validation_y=y[n1+n2:])
        result=time_policy(data,100,method,1)
    wall=perf_counter()-total
    expected=next(r for r in json.loads((reference/'decisions.json').read_text())['decisions'] if r['method']==method and r['repeat']==0)
    assert result['winner']==expected['winner'],('Cold/resident winner mismatch',result['winner'],expected['winner'])
    assert abs(result['validation_brier']-min(expected['observed']))<1e-6
    save(out,dict(**result,model_setup_seconds=setup,image_feature_seconds=extraction,projection_seconds=projection_seconds,total_seconds=wall,
        images=len(rows),physical_gpu=0,code_sha256=sha(__file__),unit_manifest_sha256=sha(unit/'manifest.json'),
        scenario='No feature cache; existing raw files and pretrained checkpoint; OS page cache uncontrolled; Python imports excluded; one repeat'))
    print('COLD COMPLETE',method,wall,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--unit',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=['sequential25','batch25','mmd50','full455'],required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();main(a.root,a.unit,a.out,a.method,a.reference)
