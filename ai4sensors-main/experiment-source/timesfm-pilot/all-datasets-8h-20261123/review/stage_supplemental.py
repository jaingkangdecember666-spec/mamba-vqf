"""Cache the bytes for two prospective transfer checks without opening labels or running inference."""
import hashlib
import json
import re
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path

from prepare_ridi_source import URL as RIDI_URL, BYTES as RIDI_SIZE
from restore_euroc_inputs import fetch
from verify_euroc_acquisition import RangeReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'external-sources/sealed-transfer-tests'


def persist(path,value):
    staging=path.with_suffix('.partial')
    staging.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    staging.replace(path)


def copy_checked(archive,entry,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        data=destination.read_bytes()
    else:
        data=archive.read(entry)
        partial=destination.with_suffix('.partial')
        partial.write_bytes(data);partial.replace(destination)
    if len(data)!=entry.file_size or zlib.crc32(data)&0xffffffff != entry.CRC:
        raise ValueError(f'ZIP checksum mismatch: {entry.filename}')
    return {'path':str(destination.relative_to(OUT)),'zip_member':entry.filename,'source_crc32':f'{entry.CRC:08x}',
            'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}


def ridi(name):
    reader=RangeReader(RIDI_URL,RIDI_SIZE)
    with zipfile.ZipFile(reader) as z:
        docs={}
        for channel in ('acce','gyro','pose'):
            key=f'{name}/{channel}.txt'
            docs[channel]=copy_checked(z,z.getinfo('data_publish_v2/'+key),OUT/'RIDI'/key)
    return {'dataset':'RIDI','record':name,'transferred_bytes':reader.transferred,'source_files':docs}


def euroc(name,folder,revision):
    record=next(r for r in folder if r['path']==name+'.zip')
    url='https://huggingface.co/datasets/pepijn223/euroc-mirror/resolve/'+revision+'/'+record['path']
    reader=RangeReader(url,record['size'])
    with zipfile.ZipFile(reader) as z:
        docs={}
        for channel in ('imu0','state_groundtruth_estimate0'):
            entry,=(e for e in z.infolist() if e.filename.endswith('mav0/'+channel+'/data.csv'))
            docs[channel]=copy_checked(z,entry,OUT/'EuRoC'/name/(channel+'.csv'))
    return {'dataset':'EuRoC','record':name,'archive_url':url,'archive_bytes':record['size'],
            'transferred_bytes':reader.transferred,'source_files':docs}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    split=ROOT/'external-sources/ridi-phone-train/original/list_test_publish_v2.txt'
    rid=[s.strip().split(',') for s in split.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    assert len(rid)==25 and len({r[0] for r in rid})==25
    assert all(re.fullmatch(r'[A-Za-z0-9_]+',s) and s.split('_')[0]!='hao' for s,_ in rid)
    rev='d05bd6a92a5f456562153f83d65805fea27cb57d'
    rooms=['V2_01_easy','V2_02_medium','V2_03_difficult']
    protocol={'role':'Secondary external-transfer holdouts for two imported training sources. All eight target-domain '
              'denominators and the selection criterion stay as previously frozen. No secondary holdout score may select a model.',
              'RIDI':{'origin':'https://yanhangpublic.github.io/ridi/index.html','archive':RIDI_URL,
              'selection':'Every entry in the official 25-record public test list, all subjects distinct from hao in training.',
              'test_list_sha256':hashlib.sha256(split.read_bytes()).hexdigest(),'records':rid,
              'input':'Same 200 Hz causal left-hold raw accelerometer and gyro, maximum 20ms packet age as for training.',
              'reference':'Tango same-phone VIO. Interpolate valid pose brackets no longer than 25ms, remove first/last 800 poses, skip first 10s; all retained native rows shared by both methods.'},
              'EuRoC':{'origin':'https://doi.org/10.3929/ethz-b-000690084','mirror_revision':rev,'records':rooms,
              'input':'Original 200Hz IMU CSV, as acquired, with original packet timing. Room 1 was an external training source.',
              'reference':'Use Room 2 supplied state ground truth, wxyz, world +Z. SLERP only within <=25ms brackets; '
              'skip first 10s and nonfinite raw input. All three Room 2 sequences included, no label or score-based exclusions.'}}
    target=OUT/'protocol.json'
    if target.exists():
        original=json.loads(target.read_text(encoding='utf8'))
        assert {k:v for k,v in original.items() if k!='frozen_at'}==protocol
    else:
        persist(target,{'frozen_at':datetime.now().astimezone().isoformat(timespec='seconds'),**protocol})
    catalog=json.loads(fetch('https://huggingface.co/api/datasets/pepijn223/euroc-mirror/tree/'+rev))
    outputs=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as executor:
        tasks={executor.submit(euroc,n,catalog,rev):('EuRoC',n) for n in rooms}
        tasks.update({executor.submit(ridi,n):('RIDI',n) for n,_ in rid})
        for done in as_completed(tasks):
            d,n=tasks[done]
            try:
                outputs.append(done.result())
                print('cached',d,n,flush=True)
            except Exception as err:
                errors.append({'dataset':d,'record':n,'exception':repr(err)})
                print('fetch error',d,n,repr(err),flush=True)
    persist(OUT/'source-files.json',{'opened_numerical_data':False,'selection_lock_required_for_scoring':True,
            'records':sorted(outputs,key=lambda r:(r['dataset'],r['record'])),'errors':errors})
    print(len(outputs),'of',len(tasks),'byte manifests')
    return not errors


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
