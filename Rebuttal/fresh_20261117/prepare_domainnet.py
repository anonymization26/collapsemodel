"""Prepare a frozen DomainNet subset directly from official ZIP byte ranges.

No historical cache is an input. ZIP CRC and decoded-image checks protect partial
downloads. Source URLs, selected IDs, archive metadata and image hashes are saved.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import struct
import threading
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
import requests
from run_fresh import save,sha

BASE='https://csr.bu.edu/ftp/visda/2019/multi-source/'
DOMAINS=['clipart','infograph','painting','quickdraw','real','sketch']
HTTP=threading.local()

def fetch(url,start=None,end=None):
    headers={} if start is None else {'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'}
    for attempt in range(6):
        try:
            if not hasattr(HTTP,'session'):HTTP.session=requests.Session()
            r=HTTP.session.get(url,headers=headers,timeout=(20,90));r.raise_for_status()
            if start is not None:
                assert r.status_code==206,(r.status_code,url)
                assert len(r.content)==end-start+1,(len(r.content),end-start+1)
                assert r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
            return r.content,dict(r.headers)
        except Exception:
            if attempt==5:raise
            time.sleep(1+attempt)

class RemoteZip(io.RawIOBase):
    def __init__(self,url):
        self.url=url;self.pos=0
        _,h=fetch(url,0,0);self.size=int(h['Content-Range'].split('/')[-1]);self.headers=h
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos if whence==1 else self.size)+offset
        return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else min(n,self.size-self.pos)
        if n<=0:return b''
        begin=self.pos;end=begin+n-1
        chunks=[(s,min(s+262143,end)) for s in range(begin,end+1,262144)]
        with ThreadPoolExecutor(max_workers=12) as pool:
            parts=list(pool.map(lambda p:fetch(self.url,*p)[0],chunks))
        self.pos+=n;return b''.join(parts)

def inventory(root):
    manifest=root/'inventory.json'
    if manifest.exists():return json.loads(manifest.read_text())
    labels=sorted(range(345),key=lambda c:hashlib.sha256(f'20261117:class:{c}'.encode()).digest())[:100]
    mapping={c:i for i,c in enumerate(sorted(labels))};records=[];files={}
    def get_split(pair):
        domain,split=pair;path=root/f'{domain}_{split}.txt'
        if not path.exists():path.write_bytes(fetch(BASE+f'domainnet/txt/{domain}_{split}.txt')[0])
        print('SPLIT READY',path.name,flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(get_split,[(d,s) for d in DOMAINS for s in ('train','test')]))
    for domain in DOMAINS:
        for split in ('train','test'):
            url=BASE+f'domainnet/txt/{domain}_{split}.txt';path=root/f'{domain}_{split}.txt'
            if not path.exists():path.write_bytes(fetch(url)[0])
            rows=[(n,int(c)) for line in path.read_text().splitlines() for n,c in [line.rsplit(' ',1)] if int(c) in mapping]
            rows.sort(key=lambda row:hashlib.sha256(f'20261117:{domain}:{split}:{row[0]}'.encode()).digest())
            roles=[('source',0,1536),('screen',1536,2048),('validation',2048,3072)] if split=='train' else [('test',0,2048)]
            assert len(rows)>=roles[-1][2],(domain,split,len(rows))
            for role,lo,hi in roles:
                records.extend(dict(path=n,label=mapping[c],original_label=c,domain=domain,role=role) for n,c in rows[lo:hi])
            files[path.name]=dict(url=url,sha256=sha(path),eligible_rows=len(rows))
    assert len({r['path'] for r in records})==len(records)==30720
    result=dict(seed=20261117,classes=sorted(labels),splits=files,records=records)
    save(manifest,result);return result

def download_domain(root,inv,domain):
    url=BASE+('groundtruth/' if domain in ('clipart','painting') else '')+domain+'.zip'
    catalog=root/f'{domain}_zip_catalog.json'
    if catalog.exists():data=json.loads(catalog.read_text())
    else:
        f=RemoteZip(url)
        with zipfile.ZipFile(f) as z:
            data=dict(url=url,size=f.size,headers=f.headers,entries={i.filename:dict(offset=i.header_offset,size=i.compress_size,raw=i.file_size,crc=i.CRC,compression=i.compress_type) for i in z.infolist()})
        save(catalog,data)
    wanted=[r for r in inv['records'] if r['domain']==domain]
    tasks=[]
    for row in wanted:
        target=root/'images'/row['path']
        if not target.exists():tasks.append((row,data['entries'][row['path']]))
    print('DOWNLOAD',domain,len(tasks),'remaining',flush=True)
    # Coalesce nearby selected entries to amortize intercontinental HTTP latency.
    groups=[]
    for row,entry in sorted(tasks,key=lambda item:item[1]['offset']):
        start=entry['offset'];end=min(data['size']-1,start+30+len(row['path'].encode())+4096+entry['size']-1)
        if groups and start-groups[-1][1]<=32768 and end-groups[-1][0]<1048576:
            groups[-1][1]=max(end,groups[-1][1]);groups[-1][2].append((row,entry))
        else:groups.append([start,end,[(row,entry)]])
    def one(group):
        lo,hi,items=group;blob,_=fetch(url,lo,hi)
        for row,entry in items:
            offset=entry['offset']-lo
            sig,version,flag,method,mt,md,crc,cs,us,fn,ex=struct.unpack('<IHHHHHIIIHH',blob[offset:offset+30])
            assert sig==0x04034b50 and method==entry['compression']
            start=offset+30+fn+ex
            if start+entry['size']>len(blob):
                compressed,_=fetch(url,lo+start,lo+start+entry['size']-1)
            else:compressed=blob[start:start+entry['size']]
            raw=zlib.decompress(compressed,-15) if method==8 else compressed
            assert method in (0,8) and len(raw)==entry['raw'] and zlib.crc32(raw)==entry['crc']
            with Image.open(io.BytesIO(raw)) as im:im.verify()
            target=root/'images'/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix(target.suffix+'.partial');temp.write_bytes(raw);temp.replace(target)
        return len(items)
    done=0
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i,future in enumerate(as_completed([pool.submit(one,item) for item in groups]),1):
            done+=future.result()
            if i%20==0:print('DOWNLOADED',domain,done,'/',len(tasks),flush=True)
    print('COMPLETE',domain,flush=True)

def verify(root,inv):
    records=[];groups={}
    for r in inv['records']:
        p=root/'images'/r['path'];h=sha(p);groups.setdefault(h,[]).append(r['path'])
        records.append(dict(**r,sha256=h,bytes=p.stat().st_size))
    duplicates=[v for v in groups.values() if len(v)>1]
    save(root/'integrity.json',dict(records=records,duplicate_groups=duplicates,inventory_sha256=sha(root/'inventory.json')))
    # Do not train across exact duplicate images; amend preparation explicitly if needed.
    assert not duplicates,('Exact duplicates require a pre-outcome preparation amendment',len(duplicates))
    print('VERIFIED',len(records),'images; no exact duplicates',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--domain',choices=DOMAINS);p.add_argument('--inventory-only',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True);inv=inventory(a.root)
    if a.verify:verify(a.root,inv)
    elif not a.inventory_only:
        for d in ([a.domain] if a.domain else DOMAINS):download_domain(a.root,inv,d)
