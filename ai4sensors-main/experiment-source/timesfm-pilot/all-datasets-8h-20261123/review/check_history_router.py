"""Recompute all C023 gate gradients, AdamW steps, native probabilities and independent raw-record errors."""
import argparse
import json

import numpy as np
import torch
from scipy.special import softmax

from check_sources import ROOT
from check_training_batch import csv_rows
from lock_from_frontier import read,inference,digest
from locked_replay import Predictor
from native_catalog import Catalog


def prediction(z,p0,param,base,fast,slow,use):
    logits=np.broadcast_to(np.log(p0),(len(z),3)).copy()
    width=z.shape[1]
    logits[:,:2]+=z@param[:2*width].reshape(2,width).T+param[2*width:]
    prob=softmax(logits,axis=1)
    mixture=(prob[:,:1]*fast+prob[:,1:2]*slow)*use[:,None]
    b=base+mixture
    norm=np.linalg.norm(b,axis=1,keepdims=True)
    return b/norm,norm,mixture,prob


def gradient(z,p0,param,base,fast,slow,truth,use,weight):
    u,norm,mixture,p=prediction(z,p0,param,base,fast,slow,use)
    error=u-truth
    loss=(error*error).sum(1)
    g=2.*(error-u*(error*u).sum(1,keepdims=True))/norm
    first=p[:,0]*np.sum(g*(fast-mixture),axis=1)*use*weight
    second=p[:,1]*np.sum(g*(slow-mixture),axis=1)*use*weight
    both=np.c_[first,second]
    dw=both.T@z
    return weight@loss,np.r_[dw.ravel(),both.sum(0)]


def audit_training(batch='C023'):
    b=ROOT/batch
    k=read(ROOT/'C022/coefficient-lock.json')['k']
    prior=np.array([2*k/3,k/3,1-k])
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    np.testing.assert_allclose(cp['p0'].numpy(),prior,rtol=0,atol=1e-15)
    with np.load(ROOT/'C019/training-inputs.npz')as f:
        raw=f['pooled'][:,1,:].astype('f8')
    z=(raw-cp['feature_mean'].numpy())/cp['feature_std'].numpy()
    with np.load(ROOT/'C022/training-slots.npz') as f:
        a={k:f[k] for k in ('center','fast','slow','truth','enabled','record_equal_weights','domain','native_release')}
    u,*_=prediction(z,prior,np.zeros(50),a['center'],a['fast'],a['slow'],a['enabled'])
    initial=((u-a['truth'])**2).sum(1)
    w=a['record_equal_weights'].copy()
    for d in np.unique(a['domain']):
        sel=a['domain']==d
        w[sel]/=w[sel]@initial[sel]
    with np.load(b/'training-io.npz') as io:
        np.testing.assert_array_equal(io['normalized_input'],z)
        np.testing.assert_allclose(w,io['relative_loss_weight'],rtol=1e-10,atol=1e-11)
        np.testing.assert_array_equal(a['native_release'],io['native_release'])
    steps=[json.loads(line) for line in (b/'optimizer-steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in steps]==list(range(1,161))
    order=np.arange(160)
    if batch=='C027':
        order=np.random.Generator(np.random.PCG64(53129)).permutation(160)
        np.testing.assert_array_equal(order,read(b/'sampling-lock.json')['permutation'])
        np.testing.assert_array_equal(order,[r['source_batch_index'] for r in steps])
        with np.load(ROOT/'C023/training-io.npz')as parent,np.load(b/'training-io.npz')as child,np.load(ROOT/'C022/training-slots.npz')as sources:
            for field in ('normalized_input','relative_loss_weight','native_release'):
                np.testing.assert_array_equal(parent[field],child[field])
            np.testing.assert_array_equal(parent['record_ordinal'],child['source_record_ordinal'])
            np.testing.assert_array_equal(sources['native_input_last'],child['native_input_last'])
    current=np.zeros(50);first=np.zeros(50);second=np.zeros(50)
    loss_gap=gradient_gap=updated_gap=0.
    for row in steps:
        idx=row['step'];source_index=int(order[idx-1]);selection=slice(source_index*96,(source_index+1)*96)
        np.testing.assert_allclose(current,row['params_before'],rtol=1e-12,atol=1e-12)
        args=[a[key][selection] for key in ('center','fast','slow','truth','enabled')]
        value,g=gradient(z[selection],prior,current,*args,40.*w[selection])
        loss_gap=max(loss_gap,abs(value-row['objective']))
        gradient_gap=max(gradient_gap,np.abs(g-row['raw_gradient']).max())
        clipped=g*min(1.,2./(np.linalg.norm(g)+1e-6))
        np.testing.assert_allclose(clipped,row['clipped_gradient'],rtol=1e-9,atol=1e-12)
        first=.9*first+.1*clipped;second=.999*second+.001*clipped**2
        current=(1.-.01*.01)*current-.01*(first/(1.-.9**idx))/(np.sqrt(second/(1.-.999**idx))+1e-8)
        updated_gap=max(updated_gap,np.abs(current-row['params_after']).max())
    assert max(loss_gap,gradient_gap,updated_gap)<1e-10,(loss_gap,gradient_gap,updated_gap)
    np.testing.assert_allclose(current,np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()],atol=1e-12)
    _,_,_,probs=prediction(z,prior,current,a['center'],a['fast'],a['slow'],a['enabled'])
    with np.load(b/'training-io.npz') as io:
        gap=float(np.abs(probs-io['prob_trained']).max());assert gap<1e-10
    return {'steps':len(steps),'all_parameter_updates':50*len(steps),'max_loss_gap':loss_gap,'max_gradient_element_gap':gradient_gap,
            'max_parameter_gap':updated_gap,'probability_max_gap':gap,'frozen_prior':prior.tolist()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch',nargs='?',choices=('C023','C027'),default='C023')
    batch=parser.parse_args().batch
    torch.set_num_threads(2)
    b=ROOT/batch;report=audit_training(batch)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference((batch,batch+'_deg'),defs))
    raw=Catalog();previous={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C022/sequences.csv')}
    control={(r['dataset'],r['record']):r for r in read(ROOT/'review/vqf_response_controls_development.json')['records']}
    records=csv_rows(b/'sequences.csv')
    assert len(records)==len(previous)==213
    choose={('SHICE','R1-1'),('TUM-VI','room2'),('IMU-Array','calibration_trial/ImuArray_1_1_IMU'),
             ('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    proofs=[]
    for row in records:
        key=row['dataset'],row['record'];old=previous[key];ctrl=control[key]
        assert row['status']=='scored' and row['scoring_indices_sha256']==old['scoring_indices_sha256']==ctrl['scoring_indices_sha256']
        assert row['scored_frames']==old['scored_frames'] and row['group']==old['group']
        for k in ('full_vqf_deg','C020_deg','C022_deg'):
            assert abs(float(row[k])-float(old[k]))<1e-5
        for k,orig in [('fixed_tau1p5_deg','fast_deg'),('fixed_tau3_deg','default_deg'),('fixed_tau6_deg','slow_deg')]:
            assert abs(float(row[k])-ctrl[orig])<1e-5
        if key in choose:
            data,_=raw.native(*key)
            result=model.score(data)
            diff=max(abs(result['candidate_deg']-float(row[batch+'_deg'])),abs(result['physical_deg']-float(old['physical_only_deg'])),
                    abs(result['full_vqf_deg']-float(row['full_vqf_deg'])))
            assert diff<1e-5 and result['scoring_indices_sha256']==row['scoring_indices_sha256'],(key,diff)
            proofs.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':diff})
    assert len(proofs)==len(choose)
    report.update(matched_records=len(records),raw_replays=proofs,checkpoint_sha256=digest(b/'router.pt'))
    (ROOT/'review'/f'{batch}_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('router gradient gap',report['max_gradient_element_gap'],'native gap',max(r['max_abs_gap_deg']for r in proofs))


if __name__=='__main__':
    main()

