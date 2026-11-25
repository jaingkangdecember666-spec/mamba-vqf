"""Prepare genuine DAVIS six-axis and optical pairs, with the poster scene inaccessible to development evaluation."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation,Slerp

from check_sources import ROOT
from lock_from_frontier import digest,read

BASE=ROOT/'external-sources/davis-handheld'
UP=np.array([0.,0.,1.])


class Davis:
    def __init__(self,model_lock=None):
        self.protocol=read(BASE/'protocol.json')
        files=read(BASE/'source-files.json')
        self.entries={r['record']:r for r in files['records']}
        self.roles={r['record']:r['role'] for r in self.protocol['records']}
        assert set(self.entries)==set(self.roles) and len(self.entries)==9 and not files['errors']
        assert files['protocol_sha256']==digest(BASE/'protocol.json')
        self.formal=False
        if model_lock:
            expected=(ROOT/'selection-lock.json').resolve()
            if Path(model_lock).resolve()!=expected or not expected.is_file():
                raise ValueError('The poster scene requires the genuine overall model lock')
            lock=read(expected)
            relative=(BASE/'preprocessing-lock.json').relative_to(ROOT).as_posix()
            assert lock['evidence_sha256'][relative]==digest(BASE/'preprocessing-lock.json')
            self.formal=True

    def native(self,name):
        entry=self.entries[name]
        if self.roles[name]=='sealed_poster' and not self.formal:
            raise ValueError('Reserved DAVIS poster scene')
        assert entry['role']==self.roles[name]
        for member in entry['members'].values():
            assert digest(BASE/member['path'])==member['sha256']
        imu=np.loadtxt(BASE/entry['members']['imu.txt']['path'])
        optical=np.loadtxt(BASE/entry['members']['groundtruth.txt']['path'])
        assert imu.shape[1]==7 and optical.shape[1]==8 and np.isfinite(imu).all() and np.isfinite(optical).all()
        t=imu[:,0];tp=optical[:,0];x=np.ascontiguousarray(imu[:,1:])
        assert np.all(np.diff(t)>0.) and np.all(np.diff(tp)>0.)
        hz=1./np.median(np.diff(t))
        qraw=optical[:,4:8]
        assert np.all(np.abs(np.linalg.norm(qraw,axis=1)-1.)<1.e-4)
        q=Rotation.from_quat(qraw)
        bracket=np.searchsorted(tp,t,side='right')-1
        real=(bracket>=0)&(bracket+1<len(tp))
        j=bracket.clip(0,len(tp)-2)
        mask=real & (tp[j+1]-tp[j]<=.025) & (t-t[0]>=10.)
        all_valid=real & (tp[j+1]-tp[j]<=.025)
        interpolated=Slerp(tp,q)(t[all_valid])
        up=np.zeros((len(x),3));up[all_valid]=interpolated.inv().apply(UP)
        rates=imu[:,4:7]
        # Two measured optical intervals on the original clock. The gyro mean is integrated in its moving body frame.
        spans=tp[2:]-tp[:-2]
        chosen=np.flatnonzero((spans>=.0075)&(spans<=.015)&(tp[:-2]>=t[0])&(tp[2:]<=t[-1]))
        size=spans[chosen]
        measured=Rotation.identity(len(chosen))
        for frac in (.125,.375,.625,.875):
            z=tp[chosen]+frac*size
            rate=np.column_stack([np.interp(z,t,rates[:,axis]) for axis in range(3)])
            measured=measured*Rotation.from_rotvec(rate*(size*.25)[:,None])
        body=measured.as_rotvec()
        relative=(q[chosen].inv()*q[chosen+2]).as_rotvec()
        bmag=np.linalg.norm(body,axis=1);omag=np.linalg.norm(relative,axis=1)
        dynamic=(bmag/size>.3)&(omag/size>.3)
        theta=np.degrees(np.arctan2(np.linalg.norm(np.cross(body[dynamic],relative[dynamic]),axis=1),
                                        np.einsum('ij,ij->i',body[dynamic],relative[dynamic])))
        accel=x[all_valid,:3];w=x[all_valid,3:]
        length=np.linalg.norm(accel,axis=1)
        quiet=(np.linalg.norm(w,axis=1)<.12)&(np.abs(length-9.80665)<.5)
        world=interpolated.apply(accel/np.maximum(length[:,None],1e-12))
        quiet_err=np.degrees(np.arctan2(np.linalg.norm(world[quiet,:2],axis=1),world[quiet,2]))
        info={'native_rows':len(x),'native_hz':float(hz),'median_dt_ms':1000./float(hz),'duration_s':float(t[-1]-t[0]),
             'optical_rows':len(tp),'median_optical_hz':1./np.median(np.diff(tp)),'largest_optical_gap_ms':float(1000.*np.diff(tp).max()),
             'scored_rows':int(mask.sum()),'gyro_optical_dynamic_pairs':int(dynamic.sum()),
             'gyro_optical_median_angle_deg':float(np.median(theta)),
             'optical_over_gyro_median_angular_displacement':float(np.median(omag[dynamic]/bmag[dynamic])),
             'quiet_native_rows':int(quiet.sum()),'quiet_world_direction_median':np.median(world[quiet],axis=0).tolist() if quiet.any() else None,
             'quiet_reference_vs_force_median_deg':float(np.median(quiet_err)) if quiet.any() else None}
        return (x,float(hz),up,mask),info


def prepare():
    source=Davis()
    output=[]
    for name,role in source.roles.items():
        if role!='external_train':continue
        raw,info=source.native(name)
        assert info['gyro_optical_dynamic_pairs']>100 and info['quiet_native_rows']>100
        assert .75<info['optical_over_gyro_median_angular_displacement']<1.3
        assert info['gyro_optical_median_angle_deg']<20. and info['quiet_world_direction_median'][2]>.98
        folder=BASE/'prepared-train'
        folder.mkdir(exist_ok=True)
        p=folder/(name+'.npz')
        if p.exists():
            with np.load(p) as f:
                for k,v in zip(('six_axis','native_hz','optical_up','mask'),raw):
                    np.testing.assert_array_equal(f[k],v)
        else:
            np.savez_compressed(p,six_axis=raw[0],native_hz=raw[1],optical_up=raw[2],mask=raw[3])
        output.append({'record':name,'prepared':p.relative_to(ROOT).as_posix(),'sha256':digest(p),**info})
    assert len(output)==6
    for row in source.protocol['records']:
        if row['role']=='sealed_poster':
            try:source.native(row['record'])
            except ValueError as error:assert 'Reserved' in str(error)
            else:raise AssertionError('Poster scene is not sealed')
    spec={'protocol_sha256':digest(BASE/'protocol.json'),'source_bytes_manifest_sha256':digest(BASE/'source-files.json'),
          'loader_sha256':digest(Path(__file__)),'units':'Original m/s² and rad/s. Keep the published camera coordinate axes and synchronized timestamps. Optical quaternion xyzw maps body into world. Use world +Z.',
          'primary_mask':'First ten seconds omitted; use original IMU rows where a finite adjacent optical pair is separated by at most 25 ms.',
          'source_records':output,'sealed_records':['poster_rotation','poster_translation','poster_6dof'],
          'unseen_numerical_records':3}
    lock=BASE/'preprocessing-lock.json'
    if lock.exists():
        prev=read(lock);assert {k:v for k,v in prev.items() if k!='locked_at'}==spec
    else:
        with lock.open('x',encoding='utf8')as f:json.dump({'locked_at':datetime.now().astimezone().isoformat(),**spec},f,indent=2)
    print('six training captures, native/scoring rows',sum(r['native_rows'] for r in output),sum(r['scored_rows'] for r in output))
    print([(r['record'],round(r['gyro_optical_median_angle_deg'],2),round(r['quiet_reference_vs_force_median_deg'],2)) for r in output])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare-training',action='store_true',required=True)
    p.parse_args()
    prepare()

