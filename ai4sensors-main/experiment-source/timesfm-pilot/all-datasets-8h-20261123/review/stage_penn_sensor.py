"""Stage only the four VI IMU texts and external marker trajectory files from the publishers."""
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'external-sources/penn-handheld'
URL='https://visiondata.cis.upenn.edu/penncosyvio/tarfiles/visensor.tar'
TOTAL=10507970560
COMMIT='b178edf7b103bbec6dfb34b494df8bdff7681eeb'
RAW='https://raw.githubusercontent.com/daniilidis-group/penncosyvio/'+COMMIT+'/'
ROLES={'af':'external_training','as':'external_training','bf':'sealed_after_model_lock','bs':'sealed_after_model_lock'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def store(path,data):
    if path.is_file():
        assert sha(path.read_bytes())==sha(data),path
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as f:f.write(data)
    return {'path':path.relative_to(BASE).as_posix(),'bytes':len(data),'sha256':sha(data)}


def curl(*args):
    return subprocess.run(['curl.exe','--fail','--location','--silent','--show-error','--max-time','30','--retry','2',*args],
                          capture_output=True,check=True,timeout=105).stdout


def fetch_range(start,end):
    size=end-start+1
    data=curl('--include','--max-filesize',str(size),'--range',f'{start}-{end}',URL+f'?bytes={start}-{end}')
    headers=data[:-size]
    match=re.search(rb'(?im)^Content-Range:\s*bytes\s+(\d+)-(\d+)/(\d+)[ \t\r]*$',headers)
    if not match or tuple(map(int,match.groups()))!=(start,end,TOTAL):raise RuntimeError('Archive returned a different range')
    assert headers.endswith(b'\r\n\r\n')
    return data[-size:]


def image_tail(cursor,prefix):
    """Locate a late valid TAR header in this directory, then let conventional traversal recover its exact boundary."""
    low=cursor;high=TOTAL;count=0
    while high-low>2*1024*1024:
        mid=((low+high)//1024)*512
        end=min(mid+1024*1024-1,TOTAL-1)
        window=fetch_range(mid,end);count+=len(window)
        found=None
        for i in range(0,len(window)-511,512):
            block=window[i:i+512]
            if not (block.startswith(b'visensor/') or block.startswith(b'./visensor/')):continue
            try:info=tarfile.TarInfo.frombuf(block,'utf8','strict')
            except tarfile.HeaderError:continue
            if info.name.startswith('visensor') or info.name.startswith('./visensor'):
                found=(mid+i,info.name);break
        if found and found[1].startswith(prefix):low=found[0]
        else:high=mid
    return low,count


def main():
    BASE.mkdir(parents=True,exist_ok=True)
    protocol=BASE/'protocol.json'
    if not protocol.exists():
        plan={'registered_at':datetime.now().astimezone().isoformat(),'source':URL,'bytes':TOTAL,'github_commit':COMMIT,
              'role_by_published_sequence':ROLES,'instrument':'Skybotix VI-Sensor ADIS16488 only. Single handheld rigid rig, four traversals of approximately the same route in the Singh Center.',
              'input':'Publisher imu.txt in m/s² and rad/s, native common timestamps. No camera data or Tango odometry enters the predictor.',
              'reference':'Publisher AprilTag marker optical poses, body-to-world rotation, world elevation +Z and published extrinsic body-to-VI rotation. Interpolate only between adjacent available marker timestamps no more than 75 ms apart; exclude first ten seconds and any unobserved IMU rows. Both estimators share that mask.',
              'quality':'Use training AF and AS to check native sample cadence, raw gyro vs relative optical rotations, quasi-static gravity, and published mount matrix without fitting target rotations or offsets.',
              'formal':'BF and BS use the same frozen transformation and predeclared angle/mask. Before overall lock only bytes, TAR header checksums and file hashes may be taken. Separate supplemental scope if usable. Implementation reads numeric held-out rows only with a verified overall selection.',
              'documentation':['https://daniilidis-group.github.io/penncosyvio/','https://daniilidis-group.github.io/penncosyvio/file_format/','https://daniilidis-group.github.io/penncosyvio/extrinsic_calib/','https://daniilidis-group.github.io/penncosyvio/docs/penncosyvio.pdf']}
        store(protocol,(json.dumps(plan,indent=2)+'\n').encode())
    assert json.loads(protocol.read_bytes())['role_by_published_sequence']==ROLES
    rows={n:{'record':n,'role':role}for n,role in ROLES.items()}
    for name in ROLES:
        key='data/ground_truth/'+name+'/pose.txt'
        rows[name]['pose']={**store(BASE/name/'pose.txt',curl(RAW+key)),'url':RAW+key}
    calibration=store(BASE/'extrinsic_calibration.txt',curl(RAW+'tools/misc/extrinsic_calibration.txt'))
    cursor=0;files=0;cached=b'';base=-1;requested=0;members=[];skipped={}
    while cursor+512<=TOTAL and len([r for r in rows.values() if 'imu'in r])<len(ROLES):
        if not(base<=cursor and cursor+512<=base+len(cached)):
            base=cursor;end=min(base+32767,TOTAL-1);cached=fetch_range(base,end);requested+=len(cached)
        header=cached[cursor-base:cursor-base+512]
        if not any(header):break
        info=tarfile.TarInfo.frombuf(header,'utf8','strict');files+=1
        bits=Path(info.name).as_posix().split('/')
        if files<=12:print('header',cursor,info.name,info.size,flush=True)
        if len(bits)>2 and bits[-2].endswith('_cam_frames') and bits[-1].endswith('.png'):
            prefix=info.name.rsplit('/',1)[0]+'/'
            if prefix not in skipped:
                new_cursor,read_bytes=image_tail(cursor,prefix);requested+=read_bytes
                skipped[prefix]={'first_seen':cursor,'last_confirmed_image_header':new_cursor}
                print('skip image directory',prefix,cursor,new_cursor,flush=True)
                if new_cursor>cursor:
                    cursor=new_cursor;continue
        if bits[-1]=='imu.txt' and len(bits)>=2 and bits[-2] in rows and info.isfile():
            name=bits[-2];assert 10000<info.size<30000000
            content=fetch_range(cursor+512,cursor+511+info.size);requested+=len(content)
            rows[name]['imu']={**store(BASE/name/'imu.txt',content),'member':info.name,'offset':cursor+512,
                               'tar_header_sha256':sha(header),'tar_checksum':info.chksum}
            print('saved',name,info.size,flush=True)
        if files<=60:members.append([cursor,info.name,info.size])
        cursor+=512+((info.size+511)//512)*512
        if files>12000:raise RuntimeError('Archive member layout requires a more efficient index')
    assert all('imu'in r for r in rows.values()),[(n,list(r))for n,r in rows.items()]
    catalog={'registered_sha256':sha(protocol.read_bytes()),'archive_url':URL,'archive_bytes':TOTAL,'inspected_member_count':files,
             'range_payload_bytes_read':requested,'initial_member_headers':members,'image_directory_ranges':skipped,
             'rig_to_sensor_calibration':calibration,'records':list(rows.values())}
    store(BASE/'source-files.json',(json.dumps(catalog,indent=2)+'\n').encode())
    print([(n,r['imu']['bytes'])for n,r in rows.items()])


if __name__=='__main__':main()
