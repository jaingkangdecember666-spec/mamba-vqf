"""Measure how long the default filter's tilt error persists in development recordings."""
import csv
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from native_catalog import Catalog, ROOT


def characterize(record, expected):
    signal,hz,ref,mask=record
    finite=np.isfinite(signal).all(1)
    previous=np.maximum.accumulate(np.where(finite,np.arange(len(finite)),-1))
    filled=signal[previous.clip(min=0)].copy();filled[previous<0]=0.
    q=VQF(1/hz).updateBatch(np.ascontiguousarray(filled[:,3:]),np.ascontiguousarray(filled[:,:3]))['quat6D']
    where=np.flatnonzero(mask&finite)
    assert hashlib.sha256(where.astype('<i8').tobytes()).hexdigest()==expected['scoring_indices_sha256']
    observed=Rotation.from_quat(q[where],scalar_first=True).apply(ref[where])
    horiz=observed[:,:2];size=np.linalg.norm(horiz,axis=1)
    degrees=np.rad2deg(np.arctan2(size,observed[:,2]))
    native_rmse=float(np.sqrt(np.mean(degrees**2)))
    assert abs(native_rmse-float(expected['full_vqf_deg']))<1e-4, (expected['record'],native_rmse,expected['full_vqf_deg'])
    errors=np.zeros((len(filled),2))
    errors[where]=horiz*(degrees/np.maximum(size,1e-15))[:,None]
    include=mask&finite;bad=np.r_[0,np.cumsum(~include)]
    times=[]
    for lag in (.10,.64,2.,5.,10.):
        n=max(1,round(lag*hz))
        if n>=len(errors):continue
        # Both endpoints and every intervening frame must refer to the same continuous scoring segment.
        left=np.flatnonzero((bad[n+1:]-bad[:-n-1])==0)
        if len(left)<32:continue
        x=errors[left];y=errors[left+n]
        x=x-x.mean(axis=0);y=y-y.mean(axis=0)
        scale=np.sqrt((x*x).sum()*(y*y).sum())
        times.append({'lag_s':lag,'native_pairs':len(left),
                      'centered_vector_correlation':float((x*y).sum()/scale) if scale else None})
    return {'scored_native_frames':len(where),'baseline_rmse_deg':native_rmse,'lags':times}


def main():
    dev=Catalog()
    with (ROOT/'C006/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
        controls={(r['dataset'],r['record']):r for r in csv.DictReader(f)}
    # Selection is by acquisition/design name only. Keep support for controls with distinct dynamics.
    chosen=[]
    groups={'BROAD':['undisturbed_slow_rotation_with_breaks','undisturbed_fast_combined'],
            'SHICE':['R1','R4'],'TUM-VI':['room1','room2'],'IMU-Array':['calibration_trial'],
            'OxIOD':['handbag','handheld','multi devices','running'],'RepoIMU':['Pendulum/Test02','TStick/Test08'],
            'Sassari':['medium']}
    for d,names in groups.items():
        for group in names:
            chosen.append(min((k for k,r in controls.items() if k[0]==d and r['group']==group)))
    chosen+=sorted(k for k in controls if k[0]=='YT_Server')[:2]
    answer=[]
    for d,name in chosen:
        raw,_=dev.native(d,name)
        item={'dataset':d,'record':name,'group':controls[d,name]['group'],**characterize(raw,controls[d,name])}
        answer.append(item)
        print(d,name,[(x['lag_s'],round(x['centered_vector_correlation'],3)) for x in item['lags']],flush=True)
    path=ROOT/'review/development_residual_memory.json'
    path.write_text(json.dumps({'method':'Default VQF only. Correlation of centered two-dimensional signed tilt-error vectors '
              'in its common world frame on fully valid native intervals. Past true errors are available to this diagnostic only. '
              'This characterizes error persistence and does not measure predictability from raw IMU.',
              'development_anchors':answer},indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
