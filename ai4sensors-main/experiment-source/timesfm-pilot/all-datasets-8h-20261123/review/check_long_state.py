"""Replay the locked two-second force correction and check the external minimax fit."""
import csv
import hashlib
import json
import numpy as np
import torch
from check_ema_fit import force_history
from check_sources import ROOT,weights
from group_uncertainty import paired_by_group
from native_blocks import block_inputs
from native_catalog import Catalog
from optical_score import infer_canonical,estimate,metric


def raw_directions(raw,model,k):
    signal,hz,_,_=raw
    observed=np.isfinite(signal).all(1)
    last=np.maximum.accumulate(np.where(observed,np.arange(len(observed)),-1))
    filled=signal[last.clip(min=0)].copy();filled[last<0]=0.
    default,window,at,release=block_inputs(filled,hz)
    q=default
    bias,enabled,_,_=estimate(filled,observed,1/hz)
    if enabled.any():
        filled[:,:3]-=bias
        q,window,at,release=block_inputs(filled,hz)
    m=force_history(filled,observed,hz,q)[at]
    rest=window[:,-1,12].astype(bool)
    m[rest]=0.
    cap=np.tan(np.deg2rad(6.))
    extra=m*np.minimum(k,cap/np.maximum(np.linalg.norm(m,axis=1),1e-15))[:,None]
    correction=.25*q[at].apply(infer_canonical(model,window))
    correction[rest]=0.
    vertical=np.array([0.,0.,1.])
    lookup=np.searchsorted(release,np.arange(len(signal)),side='right')
    def native(delta):
        u=vertical+delta;u/=np.linalg.norm(u,axis=1,keepdims=True)
        return q.inv().apply(np.vstack((vertical,u))[lookup])
    return {'full_vqf_deg':default.inv().apply(vertical),'C006_deg':native(correction),
            'C011_deg':native(correction+extra),'physical_only_deg':native(extra)}


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C011';lock=read(batch/'coefficient-lock.json')
    source=read(ROOT/'review/C011_source_audit.json')
    selection=read(batch/'external-selection.json')
    assert abs(source['fitted_k']-lock['k'])<5e-6
    old={(r['dataset'],r['record']):r for r in rows(ROOT/'C006/sequences.csv')}
    actual=rows(batch/'sequences.csv')
    assert set(old)=={(r['dataset'],r['record']) for r in actual} and len(actual)==213
    model=weights(ROOT/'C004/B-readout.pt','mamba_v1')
    catalog=Catalog()
    presets={'BROAD':'04_undisturbed_slow_rotation_with_breaks_A','Sassari':'medium/AP1','SHICE':'R4-1',
             'TUM-VI':'room1','IMU-Array':'calibration_trial/ImuArray_1_1_IMU','OxIOD':'running/data1/syn/imu1.csv',
             'RepoIMU':'Pendulum/Pendulum_Test02a_Trial1_Segment_1','YT_Server':min(n for d,n in old if d=='YT_Server')}
    anchors=[]
    for r in actual:
        key=r['dataset'],r['record']
        previous=old[key]
        assert r['status']=='scored' and r['scoring_indices_sha256']==previous['scoring_indices_sha256']
        assert abs(float(r['full_vqf_deg'])-float(previous['full_vqf_deg']))<1e-6
        assert abs(float(r['C006_deg'])-float(previous['B_rest_gated_deg']))<1e-6
        if presets[key[0]]!=key[1]:continue
        data,_=catalog.native(*key)
        _,_,target,mask=data
        indices=np.flatnonzero(mask)
        assert hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
        pred=raw_directions(data,model,lock['k'])
        scores={k:metric(v[mask],target[mask]) for k,v in pred.items()}
        error=max(abs(float(r[k])-v) for k,v in scores.items())
        assert error<1e-5,(key,scores,error)
        anchors.append({'dataset':key[0],'record':key[1],'scores_deg':scores,'max_replay_gap_deg':error})
    assert len(anchors)==8
    result={'locked_k':lock['k'],'independent_external_k':source['fitted_k'],
            'abs_k_difference':abs(source['fitted_k']-lock['k']),'training_samples':source['draws'],
            'external_loss_ratios':source['ratios_at_fit'],'all_native_rows':len(actual),'native_anchors':anchors,
            'results_sha256':hashlib.sha256((batch/'results.json').read_bytes()).hexdigest(),
            'domains':{d:paired_by_group([r for r in actual if r['dataset']==d],'full_vqf_deg','C011_deg') for d in presets}}
    result['external_selection_file_sha256']=hashlib.sha256((batch/'external-selection.json').read_bytes()).hexdigest()
    assert result['external_selection_file_sha256']==lock['external_selection_sha256'] and selection
    (ROOT/'review/C011_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('k',lock['k'],source['fitted_k'],'eight native domains maximum replay difference',max(a['max_replay_gap_deg'] for a in anchors))


if __name__=='__main__':
    main()
