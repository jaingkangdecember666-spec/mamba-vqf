"""Audit all eight development denominators and recreate both runtime gates on native recordings."""
import csv
import hashlib
import json
import numpy as np
import torch
from check_array import base_inputs
from check_sources import ROOT,DATA,tum,weights
from native_blocks import block_inputs
from optical_reference import extract
from optical_score import infer_canonical,estimate,metric


def load_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def raw_gates(raw,head):
    signal,hz,ref,mask=raw;obs=np.isfinite(signal).all(1)
    prev=np.maximum.accumulate(np.where(obs,np.arange(len(obs)),-1))
    signal=signal[prev.clip(min=0)].copy();signal[prev<0]=0.
    bias,on,_,_=estimate(signal,obs,1./hz)
    signal[:,:3]-=bias
    q,w,at,release=block_inputs(signal,hz)
    rr=np.sqrt(np.mean(np.sum((w[:,:,3:6].astype(float)-w[:,:,9:12])**2,axis=2),axis=1))
    soft=rr**2/(rr**2+.01);rest=w[:,-1,12].astype(bool)
    ids=np.flatnonzero(mask&obs);lookup=np.searchsorted(release,ids,side='right')
    target=q[ids].apply(ref[ids])
    delta=q[at].apply(infer_canonical(head,w))
    z=np.array([0,0,1.])
    result={}
    for k,factor in [('fixed',np.ones(len(at))),('soft',soft),('rest',~rest)]:
        up=z+.25*factor[:,None]*delta
        up/=np.linalg.norm(up,axis=1,keepdims=True)
        result[k]=metric(np.vstack((z,up))[lookup],target)
    return ids,np.column_stack((release,rr,soft,rest)),result


def main():
    torch.set_num_threads(2)
    inv=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    allowed={(r['dataset'],r['record']):r for r in inv['records'] if r['side'] in ('development','development_previously_used_for_selection')}
    first=load_csv(ROOT/'C005/sequences.csv');second=load_csv(ROOT/'C006/sequences.csv')
    assert set(allowed)=={(r['dataset'],r['record']) for r in first}=={(r['dataset'],r['record']) for r in second}
    assert len(first)==len(second)==len(allowed)==213
    v5={(r['dataset'],r['record']):r for r in first}
    prior={(r['dataset'],r['record']):r for r in load_csv(ROOT/'C004/sequences.csv')}
    report={'development_records':len(first),'C004_controls':len(prior),'native_replays':[],'domains':{}}
    table=np.memmap(DATA/'IMU-Array-main/Calibration/Group1_ref_data.bin',dtype='<f8',mode='r').reshape(-1,7)
    model=weights(ROOT/'C004/B-readout.pt','mamba_v1')
    anchors={('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('RepoIMU','Pendulum/Pendulum_Test02a_Trial1_Segment_1'),
             ('IMU-Array','calibration_trial/ImuArray_1_1_IMU')}
    with np.load(ROOT/'C005/scoring-masks.npz') as zz, np.load(ROOT/'C005/gate-traces.npz') as g5,np.load(ROOT/'C006/gate-traces.npz') as g6:
        for r in second:
            key=(r['dataset'],r['record']);a=v5[key];spec=allowed[key]
            assert r['scoring_indices_sha256']==a['scoring_indices_sha256']
            assert all(r[k]==a[k] for k in ['full_vqf_deg','c003_deg','B_fixed_deg','B_gated_deg'])
            if key in prior:
                old=prior[key]
                assert r['scoring_indices_sha256']==old['scoring_indices_sha256']
                assert all(float(r[x])==float(old[y]) for x,y in [('full_vqf_deg','full_vqf_deg'),('c003_deg','c003_deg'),('B_fixed_deg','B_deg')])
            k=r['mask_key'];mask=zz[k];ids=np.flatnonzero(mask)
            assert len(mask)==int(r['native_frames']) and len(ids)==int(r['scored_frames'])
            assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
            s=g5[k];t=g6[k];np.testing.assert_allclose(s,t[:,:3],atol=1e-12,rtol=1e-12)
            expected=s[:,1]**2/(.01+s[:,1]**2)
            np.testing.assert_allclose(s[:,2],expected,atol=1e-12,rtol=1e-12)
            rel=t[:,0].astype('int64')
            assert np.all(np.diff(rel)>0) and np.all(rel<len(mask))
            rest=np.r_[False,t[:,3].astype(bool)][np.searchsorted(rel,ids,side='right')]
            assert int(rest.sum())==int(r['rest_scored_frames'])
            if key in anchors:
                if key[0]=='TUM-VI':raw=tum(2)
                elif key[0]=='IMU-Array':raw,info=base_inputs(spec,table)
                else:raw=extract(spec)[:4]
                nr,physical,scores=raw_gates(raw,model);np.testing.assert_array_equal(nr,ids)
                np.testing.assert_array_equal(physical[:,0],t[:,0]);np.testing.assert_array_equal(physical[:,3],t[:,3])
                gate_error=np.max(np.abs(physical[:,1:3]-t[:,1:3]))
                assert gate_error<5e-5
                dif=max(abs(scores[x]-float(r[y])) for x,y in [('fixed','B_fixed_deg'),('soft','B_gated_deg'),('rest','B_rest_gated_deg')])
                assert dif<1e-5
                report['native_replays'].append({'dataset':key[0],'record':key[1],'rmse_deg':scores,
                                                'max_score_error_deg':float(dif),'max_gyro_or_gate_difference':float(gate_error)})
    for d in sorted(set(k[0] for k in allowed)):
        sub=[r for r in second if r['dataset']==d]
        report['domains'][d]={'records':len(sub),'scored_frames':sum(int(r['scored_frames']) for r in sub),
                              'means':{k:float(np.mean([float(r[k]) for r in sub])) for k in ['full_vqf_deg','c003_deg','B_fixed_deg','B_gated_deg','B_rest_gated_deg']}}
    path=ROOT/'review/C005_C006_audit.json'
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('records',len(second),'anchors',report['native_replays'])


if __name__=='__main__':main()

