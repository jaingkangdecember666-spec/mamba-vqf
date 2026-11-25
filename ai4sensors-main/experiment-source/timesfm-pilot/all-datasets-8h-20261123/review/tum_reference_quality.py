"""Compare finite differences of the calibrated optical rotations to raw IMU gyro on each TUM room."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from check_sources import DATA,ROOT,tum
from lock_from_frontier import read,digest
from locked_replay import SelectedPredictor


def read_channels(path,columns):
    table=np.loadtxt(path,delimiter=',',skiprows=1,usecols=(0,*columns),
         dtype=[('tick','<i8'),('values','<f8',len(columns))])
    return table['tick'],table['values']


def directions(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b)))


def optical_gyro(name):
    directory=DATA/'TUM-VI'/('dataset-'+name+'_512_16')
    sensor,=directory.rglob('imu0/data.csv')
    optical,=directory.rglob('mocap0/data.csv')
    ti,w=read_channels(sensor,(1,2,3))
    tp,quats=read_channels(optical,(4,5,6,7))
    assert np.all(np.diff(ti)>0) and np.all(np.diff(tp)>0)
    dt=(tp[2:]-tp[:-2])*1e-9
    qok=np.isfinite(quats).all(1)&(np.linalg.norm(quats,axis=1)>.95)&(np.linalg.norm(quats,axis=1)<1.05)
    ok=qok[:-2]&qok[2:]&(tp[:-2]>=ti[0]+10_000_000_000)&(tp[2:]<=ti[-1])&(dt>=.012)&(dt<=.025)
    ids=np.flatnonzero(ok)
    begin=tp[ids]-ti[0]
    span=tp[ids+2]-tp[ids]
    ref=(Rotation.from_quat(quats[ids],scalar_first=True).inv()*Rotation.from_quat(quats[ids+2],scalar_first=True)).as_rotvec()
    # Integrate four positive rotations expressed in the moving sensor frame. No time shift or fitted rotation.
    grid=ti-ti[0]
    gyro=Rotation.identity(len(ids))
    for frac in (.125,.375,.625,.875):
        at=begin+span*frac
        value=np.stack([np.interp(at,grid,w[:,j])for j in range(3)],axis=1)
        gyro=gyro*Rotation.from_rotvec(value*(span*2.5e-10)[:,None])
    raw=gyro.as_rotvec()
    refspeed=np.linalg.norm(ref,axis=1)
    rawspeed=np.linalg.norm(raw,axis=1)
    active=(refspeed/dt[ids]>.4)&(rawspeed/dt[ids]>.4)&np.isfinite(raw).all(1)
    assert active.any()
    angles=directions(ref[active],raw[active])
    ratios=refspeed[active]/rawspeed[active]
    x,hz,truth,mask=tum(int(name.removeprefix('room')))
    quiet=mask & (np.linalg.norm(x[:,3:],axis=1)<.05) & (np.abs(np.linalg.norm(x[:,:3],axis=1)-9.80665)<.3)
    quiet_angles=directions(x[quiet,:3],truth[quiet])
    return {'native_rows':len(ti),'mocap_rows':len(tp),'scoring_mask_rows':int(mask.sum()),'native_hz':hz,
            'mocap_median_hz':1.e9/np.median(np.diff(tp)),'max_mocap_gap_seconds':float(np.diff(tp).max()*1.e-9),
            'dynamic_optical_pairs':int(active.sum()),'optical_vs_gyro_direction_median_deg':float(np.median(angles)),
            'optical_vs_gyro_direction_p90_deg':float(np.quantile(angles,.9)),
            'optical_to_gyro_displacement_median_ratio':float(np.median(ratios)),
            'quiet_rows':int(quiet.sum()),'quiet_acc_vs_optical_vertical_median_deg':float(np.median(quiet_angles)) if len(quiet_angles) else None}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--development',action='store_true')
    ap.add_argument('--selection-lock',type=Path)
    ap.add_argument('--main-csv',type=Path)
    args=ap.parse_args()
    if args.development==bool(args.selection_lock) or bool(args.selection_lock)!=bool(args.main_csv):
        ap.error('Use --development, or supply both --selection-lock and --main-csv')
    formal=bool(args.selection_lock)
    rooms=['room'+str(i) for i in ((3,4,5,6) if formal else (1,2))]
    audit={'scope':'Physical plausibility of calibrated optical motion and measured angular velocity over the same two camera-tracker intervals. Four advancing-body increments using the measured gyro, zero fitted time shift, zero fitted rotation. Direction statistics use angular speed above 0.4 rad/s in both instruments. Quiet frames use angular speed below 0.05 rad/s and specific-force norm within 0.3 m/s² of standard gravity. No tuning or changes to the original scoring mask.',
           'source':'https://cvg.cit.tum.de/_media/spezial/bib/schubert2018vidataset.pdf','rows':[]}
    if formal:
        torch.set_num_threads(2)
        lock=read(args.selection_lock)
        assert {r['record']for r in lock['formal_cases'] if r['dataset']=='TUM-VI'}==set(rooms)
        model=SelectedPredictor(args.selection_lock)
        with args.main_csv.open(encoding='utf-8-sig',newline='') as f:
            source={r['record']:r for r in csv.DictReader(f) if r['dataset']=='TUM-VI'}
        assert set(source)==set(rooms)
        audit.update(lock_sha256=digest(args.selection_lock),main_csv_sha256=digest(args.main_csv))
    for room in rooms:
        r=optical_gyro(room)
        if formal:
            prior=source[room]
            if prior.get('candidate_deg'):
                prediction=model.score(tum(int(room.removeprefix('room'))))
                assert prediction['scoring_indices_sha256']==prior['scoring_indices_sha256']
                assert prediction['scored_frames']==int(prior['scored_frames'])
                gap=max(abs(float(prior[k])-prediction[k])for k in ('candidate_deg','full_vqf_deg','physical_deg'))
                assert gap<2e-4
                r.update(raw_paired_score=prediction,max_abs_published_error_deg=gap)
            r['formal_status']=prior['status']
        audit['rows'].append({'room':room,**r})
    suffix='final' if formal else 'development'
    target=ROOT/'review'/('tum_'+suffix+'_reference.json')
    target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print([(r['room'],r['dynamic_optical_pairs'],r['optical_vs_gyro_direction_median_deg'])for r in audit['rows']])


if __name__=='__main__':
    main()

