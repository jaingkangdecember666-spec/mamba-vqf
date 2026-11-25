"""Independent raw TUM-VI and YT recordings for paired source and mask checks."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parent.parent
PILOT = ROOT.parent
TRANSFER = PILOT / 'transfer-mamba-20261123'
DATA = Path('E:/数据集')
sys.path[:0] = [str(TRANSFER/'review'), str(ROOT/'C001/.deps')]
from replay_native import score, weights


def tum(room):
    prefix = DATA/'TUM-VI'/f'dataset-room{room}_512_16'
    imu_path, = prefix.rglob('imu0/data.csv')
    pose_path, = prefix.rglob('mocap0/data.csv')

    def read(path, indices):
        a = np.loadtxt(path, delimiter=',', skiprows=1, usecols=(0,*indices),
                       dtype=[('t','<i8'),('v','<f8',len(indices))])
        return a['t'],a['v']

    ti,raw = read(imu_path,range(1,7))
    tp,q = read(pose_path,range(4,8))
    assert np.all(np.diff(ti)>0) and np.all(np.diff(tp)>0)
    j = np.searchsorted(tp,ti,side='right')-1
    inside = (j>=0)&(j+1<len(tp))
    j = j.clip(0,len(tp)-2)
    goodq = np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>0)
    eligible = inside & goodq[j] & goodq[j+1] & ((tp[j+1]-tp[j])<=25_000_000) & ((ti-ti[0])>=10_000_000_000)
    zero = min(tp[0],ti[0])
    interpolator = Slerp((tp[goodq]-zero)*1e-9,Rotation.from_quat(q[goodq],scalar_first=True))
    up = np.zeros((len(ti),3))
    up[eligible] = interpolator((ti[eligible]-zero)*1e-9).inv().apply([0,0,1])
    return np.c_[raw[:,3:6],raw[:,:3]],1e9/float(np.median(np.diff(ti))),up,eligible


def yt(record, exclude_seconds=10., clock='nominal'):
    import h5py
    with h5py.File(DATA/'YT_server/YT_server_200Hz.h5','r') as f:
        g = f['valid'][record]
        t = g['ts'][...].reshape(-1).astype('float64')
        a, w, q = [g[k][...].astype('float64') for k in ('acc','gyr','qua')]
    good = np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>0)
    gravity = np.zeros_like(a)
    gravity[good] = Rotation.from_quat(q[good],scalar_first=True).inv().apply([0.,0.,1.])
    hz = 200. if clock=='nominal' else 1/float(np.median(np.diff(t)))
    return np.c_[a,w],hz,gravity,good & ((t-t[0])>=exclude_seconds)


def measured(raw,model):
    imu,hz,up,allowed = raw
    observed = np.isfinite(imu).all(1)
    prev = np.maximum.accumulate(np.where(observed,np.arange(len(imu)),-1))
    filled = imu[np.maximum(prev,0)].copy()
    filled[prev<0]=0
    mask = allowed&observed
    row=score(filled,hz,up,mask,{'candidate':model},{'candidate':{'candidate':1.}},1.,{'candidate'})
    assert row['indices_sha256']==hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()
    return row


if __name__=='__main__':
    import torch
    torch.set_num_threads(2)
    head=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    results={f'room{n}':measured(tum(n),head) for n in (1,2)}
    print(json.dumps(results,indent=2))
    (ROOT/'review/tum_development_replay.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
