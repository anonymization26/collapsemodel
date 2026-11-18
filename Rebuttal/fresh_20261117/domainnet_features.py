"""Fresh ResNet50 image extraction on physical GPU 0 and frozen projected units."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits
from run_fresh import normalized,save,sha
from prepare_domainnet import DOMAINS,verify

def extract(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0','Only physical GPU 0 is authorized'
    import torch
    from torch.utils.data import Dataset,DataLoader
    from torchvision.models import resnet50,ResNet50_Weights
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    inv=json.loads((root/'inventory.json').read_text());verify(root,inv)
    if (root/'features_manifest.json').exists():
        m=json.loads((root/'features_manifest.json').read_text())
        assert sha(root/'raw_features.npy')==m['features_sha256'];print('FEATURES ALREADY VERIFIED');return
    weights=ResNet50_Weights.IMAGENET1K_V2;transform=weights.transforms()
    tick=perf_counter();model=resnet50(weights=weights);model.fc=torch.nn.Identity();model.eval().cuda()
    setup_seconds=perf_counter()-tick
    class Images(Dataset):
        def __len__(self):return len(inv['records'])
        def __getitem__(self,i):
            with Image.open(root/'images'/inv['records'][i]['path']) as image:return transform(image.convert('RGB'))
    loader=DataLoader(Images(),batch_size=128,num_workers=4,shuffle=False,pin_memory=True)
    feat=np.lib.format.open_memmap(root/'raw_features.partial.npy',mode='w+',dtype=np.float32,shape=(len(inv['records']),2048))
    tick=perf_counter();pos=0
    with torch.inference_mode():
        for images in loader:
            value=model(images.cuda(non_blocking=True)).cpu().numpy();feat[pos:pos+len(value)]=value;pos+=len(value)
            if pos%2048==0:print('FEATURES',pos,'/',len(feat),flush=True)
    torch.cuda.synchronize();elapsed=perf_counter()-tick;feat.flush();del feat
    (root/'raw_features.partial.npy').replace(root/'raw_features.npy')
    checkpoint=Path(torch.hub.get_dir())/'checkpoints'/weights.url.split('/')[-1]
    save(root/'features_manifest.json',dict(encoder='torchvision ResNet50 IMAGENET1K_V2',torch=torch.__version__,
        checkpoint_url=weights.url,checkpoint_sha256=sha(checkpoint),transform=str(transform),
        inventory_sha256=sha(root/'inventory.json'),integrity_sha256=sha(root/'integrity.json'),
        features_sha256=sha(root/'raw_features.npy'),model_setup_download_seconds=setup_seconds,
        extraction_seconds=elapsed,physical_gpu=0,device=torch.cuda.get_device_name(0),
        rows=pos,dimensions=2048,source_code_sha256=sha(__file__)))
    print('EXTRACTED',pos,'seconds',elapsed,flush=True)

def units(root,out):
    inv=json.loads((root/'inventory.json').read_text());fm=json.loads((root/'features_manifest.json').read_text())
    assert sha(root/'raw_features.npy')==fm['features_sha256']
    raw=np.load(root/'raw_features.npy',mmap_mode='r');records=inv['records'];out.mkdir(parents=True,exist_ok=True)
    labels=np.array([r['label'] for r in records]);indices={(d,role):np.array([i for i,r in enumerate(records) if r['domain']==d and r['role']==role]) for d in DOMAINS for role in ['source','screen','validation','test']}
    pcs={}
    with threadpool_limits(limits=1):
        for domain in DOMAINS:
            ix=indices[domain,'source'];x=np.asarray(raw[ix],dtype=np.float64);x-=x.mean(axis=0)
            # Source-only eigendecomposition; sign fixed by largest loading.
            vals,vec=np.linalg.eigh(x.T@x);v=vec[:,-1]
            if v[np.argmax(np.abs(v))]<0:v=-v
            pcs[domain]=np.argsort(x@v,kind='stable')
        for seed in range(20261117,20261121):
            # First d columns form nested Rademacher projections within this seed.
            projection=np.random.default_rng(seed).choice(np.array([-1.,1.]),size=(2048,512))
            base=normalized(np.asarray(raw,dtype=np.float64));projected=base@projection
            for dimension in ([64,96,128,256,512] if seed==20261117 else [128]):
                x=normalized(projected[:,:dimension])
                for target in DOMAINS:
                    sources=[d for d in DOMAINS if d!=target];src=np.concatenate([indices[d,'source'] for d in sources])
                    for construction in ['hash','pc1']:
                        name=f'domainnet_{target}_{construction}_d{dimension}_{seed}';unit=out/name
                        if unit.exists():
                            assert (unit/'manifest.json').exists();continue
                        unit.mkdir();blocks=[]
                        for n,domain in enumerate(sources):
                            blockorder=np.array(sorted(range(1536),key=lambda i:hashlib.sha256(f'{seed}:{records[indices[domain,"source"][i]]["path"]}'.encode()).digest())) if construction=='hash' else pcs[domain]
                            blocks.extend((blockorder+n*1536).reshape(3,512))
                        screen=indices[target,'screen'];val=indices[target,'validation'];test=indices[target,'test']
                        np.savez_compressed(unit/'selection.npz',source_x=x[src],source_y=labels[src],blocks=np.stack(blocks),screen_x=x[screen],validation_x=x[val],validation_y=labels[val])
                        np.savez_compressed(unit/'test.npz',x=x[test],y=labels[test])
                        save(unit/'manifest.json',dict(dataset='DomainNet_cleaned_100class',target=target,seed=seed,construction=construction,classes=100,dimension=dimension,
                            selection_sha256=sha(unit/'selection.npz'),test_sha256=sha(unit/'test.npz'),
                            inventory_sha256=sha(root/'inventory.json'),features_manifest_sha256=sha(root/'features_manifest.json'),
                            source_domains=sources,source_global_rows=src.tolist(),target_screen_rows=screen.tolist(),target_validation_rows=val.tolist(),target_test_rows=test.tolist(),
                            features='Fresh ResNet50 V2 2048D; pre/post L2; nested Rademacher projection',preparation_code_sha256=sha(__file__)))
                        print('UNIT',name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','units']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.phase=='extract':extract(a.root)
    else:units(a.root,a.out)
