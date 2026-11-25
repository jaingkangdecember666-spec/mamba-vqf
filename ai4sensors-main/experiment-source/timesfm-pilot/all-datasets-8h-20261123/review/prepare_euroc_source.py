"""Stage a small public source for external-only transfer training; verify the authors' LFS digests."""
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'C001/.deps'))
import h5py

REV='435274141c08ab318c01be904de9754943b4feed'
REPO='daniel-om-weber/riann_dev'
ROOM1=('V1_01_easy','V1_02_medium','V1_03_difficult')
OUT=ROOT/'external-sources/euroc-vicon-room1'


def fetch(url):
    req=Request(url,headers={'User-Agent':'IMU-research-data-audit'})
    with urlopen(req,timeout=25) as response:return response.read()


def acquire(name):
    remote=quote(f'data/EuRoC-MAV/EurocMAV::{name}.hdf5',safe='/')
    pointer_url=f'https://raw.githubusercontent.com/{REPO}/{REV}/{remote}'
    media_url=f'https://media.githubusercontent.com/media/{REPO}/{REV}/{remote}'
    pointer=fetch(pointer_url).decode('ascii')
    checksum=re.search(r'^oid sha256:([a-f0-9]{64})$',pointer,re.M).group(1)
    n=int(re.search(r'^size (\d+)$',pointer,re.M).group(1))
    if n<=0 or n>20_000_000:raise ValueError('Unexpected size for compact, camera-free source')
    destination=OUT/(name+'.hdf5')
    if destination.is_file():payload=destination.read_bytes()
    else:payload=fetch(media_url)
    if len(payload)!=n or hashlib.sha256(payload).hexdigest()!=checksum or payload[:8]!=b'\x89HDF\r\n\x1a\n':
        raise ValueError(f'Integrity failed for {name}')
    if not destination.is_file():
        temp=destination.with_suffix('.partial');temp.write_bytes(payload);temp.replace(destination)
    with h5py.File(destination,'r') as f:
        fields={key:list(value.shape) for key,value in f.items() if hasattr(value,'shape')}
        attrs={k:str(v) for k,v in f.attrs.items()}
        print(name,fields,attrs,flush=True)
    return {'record':name,'partition':'external_training_source','room':'V1','file':destination.name,
            'bytes':n,'sha256':checksum,'source_commit':REV,'verified_pointer':pointer_url,'data_url':media_url,
            'fields':fields,'attributes':attrs}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(acquire,ROOM1))
    manifest={'purpose':'Potential external training only; never part of the eight target datasets. No sealed target labels or scores. No Vicon room 2 is downloaded.',
              'original_source':'https://doi.org/10.3929/ethz-b-000690084','conversion':'https://github.com/'+REPO+'/blob/'+REV+'/riann/prep/euroc.py',
              'reference_provenance':'Prepared from published EuRoC state ground-truth estimates aligned with IMU measurements; fused/reference imperfections are possible.',
              'records':results}
    (OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()

