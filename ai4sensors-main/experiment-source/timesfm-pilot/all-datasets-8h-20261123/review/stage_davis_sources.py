"""Predeclare DAVIS handheld scenes and stage only the two published sensor tables from each original ZIP."""
import hashlib
import json
import re
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from pathlib import Path
from urllib.request import Request,urlopen

from verify_euroc_acquisition import RangeReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'external-sources/davis-handheld'
ORIGIN='https://rpg.ifi.uzh.ch/davis_data.html'
BASE='https://download.ifi.uzh.ch/rpg/web/datasets/davis/'
SCENES=('shapes','boxes','poster')
MOTIONS=('rotation','translation','6dof')


def persist(p,d):
    temp=p.with_suffix('.stage')
    temp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
    temp.replace(p)


def fetch(name,role):
    url=BASE+name+'.zip'
    request=Request(url,headers={'Range':'bytes=0-0','User-Agent':'IMU-sensor-source-audit','Accept-Encoding':'identity'})
    with urlopen(request,timeout=25) as res:
        matched=re.fullmatch(r'bytes 0-0/(\d+)',res.headers.get('Content-Range',''))
        assert res.status==206 and matched and res.read(2)==b'P'
        total=int(matched.group(1))
        etag=res.headers.get('ETag')
    reader=RangeReader(url,total)
    fields={}
    folder=OUT/role/name
    folder.mkdir(exist_ok=True,parents=True)
    with zipfile.ZipFile(reader) as archive:
        for filename in ('imu.txt','groundtruth.txt'):
            member,=(e for e in archive.infolist() if e.filename.split('/')[-1]==filename and not e.filename.startswith('__MACOSX/'))
            target=folder/filename
            data=target.read_bytes() if target.is_file() else archive.read(member)
            if len(data)!=member.file_size or (zlib.crc32(data)&0xffffffff)!=member.CRC:
                raise ValueError(f'Source CRC or size mismatch for {name} {filename}')
            if not target.exists():
                staging=target.with_suffix('.partial');staging.write_bytes(data);staging.replace(target)
            fields[filename]={'path':target.relative_to(OUT).as_posix(),'zip_member':member.filename,'bytes':len(data),
                             'sha256':hashlib.sha256(data).hexdigest(),'source_crc32':f'{member.CRC:08x}'}
    return {'record':name,'role':role,'origin_zip':url,'zip_size_bytes':total,'etag':etag,'transferred':reader.transferred,'members':fields}


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    rows=[{'record':s+'_'+m,'scene':s,'motion':m,'role':'sealed_poster' if s=='poster' else 'external_train'}for s in SCENES for m in MOTIONS]
    body={'dataset':'RPG DAVIS240C handheld inertial and optical sequences','origin':ORIGIN,'paper':'https://arxiv.org/abs/1610.08336',
          'authors':'Elias Mueggler, Henri Rebecq, Guillermo Gallego, Tobi Delbruck, Davide Scaramuzza',
          'role':'Optional additional external mobile optical source. Two complete three-motion scenes are available for external fitting. Reserve the complete third scene for the single chosen predictor, after overall selection. This same instrument and lab give a scene/sequence split, not device independence. Keep the original eight primary datasets and their selection criterion.',
          'records':rows,
          'geometry':'Use the authors IMU columns (t,ax,ay,az,gx,gy,gz) and optical camera-to-world ground truth (t,px,py,pz,qx,qy,qz,qw). Check gyro scale against finite optical rotations; compare the published body axes using relative rotations, and the shared world up against quasi-static specific force. Decide any raw-unit conversion and global vertical convention exclusively from these six training sequences, with no estimator or candidate scores. Document or exclude unsuitable motion/clock references rather than reorienting a test recording to benefit a method.',
          'mask':'Use native IMU rows, common timestamp brackets up to 25 ms with finite optical quaternions, and skip ten initial seconds. The same native indices apply to every predictor. Pose interpolation defines labels; network inputs include only arrived accelerometer and gyro.',
          'sealed_access':'Stage only original member bytes and ZIP sizes/CRC/SHA for poster. Numerical reading or scoring requires the genuine global model lock.'}
    protocol=OUT/'protocol.json'
    if protocol.exists():
        prev=json.loads(protocol.read_bytes());assert {k:v for k,v in prev.items() if k!='frozen_at'}==body
    else:
        persist(protocol,{'frozen_at':datetime.now().astimezone().isoformat(),**body})
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks={pool.submit(fetch,r['record'],r['role']):r for r in rows}
        for task in as_completed(tasks):
            row=tasks[task]
            try:
                records.append(task.result());print('verified ZIP member bytes',row['record'],flush=True)
            except Exception as exc:
                errors.append({**row,'error':repr(exc)});print('failed',row['record'],repr(exc),flush=True)
    persist(OUT/'source-files.json',{'records':sorted(records,key=lambda x:x['record']),'errors':errors,
            'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'numerical_data_opened_by_staging':False})
    print(len(records),'of',len(rows))
    return not errors


if __name__=='__main__':
    raise SystemExit(0 if main() else 1)

