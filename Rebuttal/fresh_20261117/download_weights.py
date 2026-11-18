"""Resumable official weight download with the torchvision hash check."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
from prepare_domainnet import fetch

def main(root):
    root.mkdir(parents=True,exist_ok=True)
    url='https://download.pytorch.org/models/resnet50-11ad3fa6.pth'
    target=root/'resnet50-11ad3fa6.pth'
    if target.exists():
        assert hashlib.sha256(target.read_bytes()).hexdigest().startswith('11ad3fa6');return
    _,headers=fetch(url,0,0);size=int(headers['Content-Range'].split('/')[-1]);parts=root/'resnet50_parts';parts.mkdir(exist_ok=True)
    chunks=[(i,min(i+(1<<20)-1,size-1)) for i in range(0,size,1<<20)]
    def one(pair):
        lo,hi=pair;p=parts/str(lo)
        if not p.exists() or p.stat().st_size!=hi-lo+1:p.write_bytes(fetch(url,lo,hi)[0])
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(one,chunks))
    temp=target.with_suffix('.partial')
    with temp.open('wb') as f:
        for lo,hi in chunks:f.write((parts/str(lo)).read_bytes())
    digest=hashlib.sha256(temp.read_bytes()).hexdigest();assert digest.startswith('11ad3fa6')
    temp.replace(target)
    for p in parts.iterdir():p.unlink()
    parts.rmdir();print('WEIGHTS VERIFIED',digest,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
