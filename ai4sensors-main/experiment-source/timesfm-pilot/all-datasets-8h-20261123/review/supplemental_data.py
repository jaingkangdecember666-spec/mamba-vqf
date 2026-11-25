"""Raw adapters for the two preregistered transfer checks, available only after the campaign selection lock."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from check_sources import ROOT
from prepare_ridi_source import load_sensor, align_sensors

OUT=ROOT/'external-sources/sealed-transfer-tests'


def text_json(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def bytes_verified(entry):
    path=OUT/entry['path']
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=entry['sha256'] or len(data)!=entry['bytes']:
        raise ValueError(f'Invalid sealed-test source: {path.name}')
    return data


def timed_directions(t,tp,quat,skip_pose_edges=0):
    ok=np.isfinite(quat).all(1) & (np.linalg.norm(quat,axis=1)>.8) & (np.linalg.norm(quat,axis=1)<1.2)
    tp,quat=tp[ok],quat[ok]
    if len(tp)<=2*skip_pose_edges+1 or np.any(np.diff(tp)<=0):
        raise ValueError('Insufficient monotone reference poses')
    p=np.searchsorted(tp,t,side='right')-1
    safe=p.clip(0,len(tp)-2)
    valid=(p>=0)&(p+1<len(tp))&((tp[safe+1]-tp[safe])<=25_000_000)
    valid&=(t>=tp[skip_pose_edges]) & (t<=tp[-skip_pose_edges-1])
    rot=Slerp((tp-tp[0])*1e-9,Rotation.from_quat(quat,scalar_first=True))
    truth=np.zeros((len(t),3))
    truth[valid]=rot((t[valid]-tp[0])*1e-9).inv().apply([0.,0.,1.])
    return truth,valid


def ridi_native(row):
    files=row['source_files']
    ta,a,_=load_sensor(bytes_verified(files['acce']),3)
    tg,g,_=load_sensor(bytes_verified(files['gyro']),3)
    tp,pose,duplicates=load_sensor(bytes_verified(files['pose']),7)
    t=np.arange(max(ta[0],tg[0]),min(ta[-1],tg[-1])+1,5_000_000,dtype=np.int64)
    signal,obs,_,_=align_sensors(ta,a,tg,g,t)
    truth,valid=timed_directions(t,tp,np.roll(pose[:,3:7],1,axis=1),800)
    mask=obs & valid & (t-t[0]>=10_000_000_000)
    quality={'source':'same-phone Tango VIO','reference_rows_deduplicated':duplicates,'grid_rows':len(t),
             'observed_rows':int(obs.sum()),'eligible_reference_rows':int(valid.sum())}
    return (signal,200.,truth,mask),quality


def euroc_native(row):
    from io import BytesIO
    files=row['source_files']
    def parsed(source,size):
        a=np.loadtxt(BytesIO(bytes_verified(source)),delimiter=',',comments='#',
                     dtype=[('t','<i8'),('v','<f8',(size,))])
        if not np.all(np.diff(a['t'])>0):raise ValueError('Nonmonotone acquisition clock')
        return a['t'],a['v']
    ta,imu=parsed(files['imu0'],6);tp,state=parsed(files['state_groundtruth_estimate0'],16)
    gaps=np.diff(ta)
    steps=np.rint(gaps/5_000_000).astype(int)
    if steps.min()<1 or np.max(abs(gaps-steps*5_000_000))>500_000:
        raise ValueError('Native acquisition intervals cannot be represented by nominal 200 Hz to within 0.5 ms')
    idx=np.r_[0,np.cumsum(steps)]
    t=np.full(idx[-1]+1,ta[0],dtype=np.int64)
    t[idx]=ta
    measured=np.isfinite(imu).all(1)
    signal=np.full((len(t),6),np.nan)
    signal[idx[measured]]=np.c_[imu[measured,3:],imu[measured,:3]]
    truth,valid=timed_directions(t,tp,state[:,3:7])
    mask=valid & np.isfinite(signal).all(1) & (t-t[0]>=10_000_000_000)
    quality={'source':'published Vicon-room fused reference state; original gyro without bias subtraction',
             'raw_imu_packets':len(ta),'nominal_grid_rows':len(t),'missing_native_intervals':len(t)-len(ta),
             'max_interval_residual_ns':float(np.max(abs(gaps-steps*5_000_000))),
             'released_state_rows':len(tp)}
    return (signal,200.,truth,mask),quality


class Supplemental:
    def __init__(self,lock):
        p=Path(lock);selected=text_json(p)
        inv=ROOT/'C001/inventory.json'
        if selected.get('inventory_sha256')!=hashlib.sha256(inv.read_bytes()).hexdigest():
            raise ValueError('A valid selection lock for this eight-domain campaign is required')
        self.protocol=text_json(OUT/'protocol.json')
        manifest=text_json(OUT/'source-files.json')
        self.rows={(r['dataset'],r['record']):r for r in manifest['records']}
        self.missing={(r['dataset'],r['record']):r for r in manifest['errors']}

    def cases(self):
        return [('RIDI',n,n.split('_')[0],g) for n,g in self.protocol['RIDI']['records']]+[
            ('EuRoC',n,'Vicon room 2',n.split('_')[-1]) for n in self.protocol['EuRoC']['records']]

    def native(self,dataset,name):
        key=dataset,name
        if key not in self.rows:raise FileNotFoundError(self.missing.get(key,{}).get('exception',name))
        return (ridi_native if dataset=='RIDI' else euroc_native)(self.rows[key])
