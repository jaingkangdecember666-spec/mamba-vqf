"""Reconstruct C028's reference labels, recording balance, physical predictions and all optimizer updates."""
import argparse
import json

import numpy as np
import torch
from scipy.special import logsumexp

from check_history_router import prediction
from check_sources import ROOT
from check_training_batch import csv_rows
from external_response_teacher import teacher
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch',nargs='?',choices=('C028','C029'),default='C028')
    batch=parser.parse_args().batch
    torch.set_num_threads(2)
    b=ROOT/batch
    keys=('center','fast','slow','truth','enabled','features','domain','condition','native_release','native_input_last','source_record_ordinal')
    with np.load(ROOT/'C026/training-inputs.npz')as f:
        a={k:f[k].reshape((-1,)+f[k].shape[2:]) for k in keys}
    cls,confidence,sq=teacher(*[a[k] for k in ('center','fast','slow','truth','enabled')])
    groups=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    weights=np.zeros_like(confidence)
    for group in sorted(set(groups)):
        mask=groups==group
        records=np.unique(a['source_record_ordinal'][mask])
        portion=1./(12. if group in ('raw','fixed_prefix_bias_removed') else 6.)
        for ordinal in records:
            sel=mask&(a['source_record_ordinal']==ordinal)
            assert confidence[sel].sum()>0.
            weights[sel]=portion*confidence[sel]/(len(records)*confidence[sel].sum())
    np.testing.assert_allclose(weights.sum(),1.,atol=1.e-13)
    with np.load(ROOT/'C028/teacher-slots.npz')as archived:
        np.testing.assert_array_equal(cls,archived['label'].ravel())
        np.testing.assert_allclose(confidence,archived['confidence_deg2'].ravel(),atol=1e-10)
        np.testing.assert_allclose(sq,archived['candidate_squared_angle_deg2'].reshape(-1,3),atol=1e-10)
        np.testing.assert_allclose(weights,archived['global_weight'].ravel(),atol=1e-13)
        for k in ('domain','condition','native_release','native_input_last','source_record_ordinal','enabled'):
            np.testing.assert_array_equal(a[k],archived[k].ravel())
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    old=torch.load(ROOT/'C026/router.pt',map_location='cpu',weights_only=True)
    for field in ('old_feature_mean','old_feature_std','context_mean','context_std','log_p0','p0'):
        torch.testing.assert_close(cp[field],old[field],atol=0.,rtol=0.)
    features=a['features']
    if batch=='C029':
        from causal_response_geometry import NAMES,measures
        assert tuple(cp['geometry_names'])==NAMES
        raw=measures(a['fast'],a['slow'],features[:,-1])
        mass=np.zeros(len(features))
        for group in sorted(set(groups)):
            sel=groups==group
            records=np.unique(a['source_record_ordinal'][sel])
            fraction=1./(12. if group in ('raw','fixed_prefix_bias_removed') else 6.)
            for record in records:
                ix=sel&(a['source_record_ordinal']==record)
                mass[ix]=fraction/(len(records)*ix.sum())
        mean=mass@raw;sd=np.sqrt(mass@(raw-mean)**2)
        np.testing.assert_allclose(mean,cp['geometry_mean'].numpy(),rtol=1e-12)
        np.testing.assert_allclose(sd,cp['geometry_std'].numpy(),rtol=1e-12)
        features=np.column_stack((features,(raw-mean)/sd))
        with np.load(b/'geometry-training.npz')as extra:
            np.testing.assert_allclose(features,extra['features'].reshape(features.shape),atol=1e-12)
            np.testing.assert_allclose(mass,extra['geometry_stat_weight'].ravel(),atol=1e-13)
    width=features.shape[1]
    prior=cp['log_p0'].numpy()
    state=np.zeros(2*width+2);m=np.zeros_like(state);v=m.copy()
    rows=[json.loads(s)for s in (b/'optimizer-steps.jsonl').read_text().splitlines() if s]
    gaps=dict(loss=0.,gradient=0.,per_group_gradient=0.,updated_parameter=0.)
    assert len(rows)==160
    for i,row in enumerate(rows):
        assert row['step']==i+1 and row['source_batch_index']==i
        sel=slice(i*110,(i+1)*110)
        x=features[sel];y=cls[sel];w=160*weights[sel]
        logits=np.tile(prior,(len(x),1))
        logits[:,:2]+=x@state[:2*width].reshape(2,width).T+state[2*width:]
        lp=logits-logsumexp(logits,axis=1,keepdims=True)
        ce=-lp[np.arange(len(x)),y]
        residual=np.exp(lp);residual[np.arange(len(x)),y]-=1.
        error=residual[:,:2]*w[:,None]
        def deriv(use):
            return np.r_[(error[use].T@x[use]).ravel(),error[use].sum(0)]
        g=deriv(np.ones(len(x),bool))
        gaps['gradient']=max(gaps['gradient'],float(np.max(np.abs(g-row['gradient_before_clip']))))
        gaps['loss']=max(gaps['loss'],float(abs(w@ce-row['objective'])))
        for domain,receipt in row['domain_or_EuRoC_condition_ledger'].items():
            local=groups[sel]==domain
            gaps['per_group_gradient']=max(gaps['per_group_gradient'],float(np.max(np.abs(deriv(local)-receipt['gradient']))))
        clipped=g*min(1.,2./(np.linalg.norm(g)+1e-6))
        m=.9*m+.1*clipped;v=.999*v+.001*clipped**2
        state=.9999*state-.01*(m/(1.-.9**(i+1)))/(np.sqrt(v/(1.-.999**(i+1)))+1e-8)
        gaps['updated_parameter']=max(gaps['updated_parameter'],float(np.max(np.abs(state-row['weights_after']))))
    assert max(gaps.values())<1e-10,gaps
    np.testing.assert_allclose(state,np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()],atol=1e-12)
    u,_,_,p=prediction(features,cp['p0'].numpy(),state,*[a[k] for k in ('center','fast','slow','enabled')])
    with np.load(b/'training-io.npz')as f:
        np.testing.assert_allclose(p,f[batch+'_p'].reshape(-1,3),atol=1e-12)
        np.testing.assert_allclose(u,f[batch+'_direction'].reshape(-1,3),atol=1e-12)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference((batch,batch+'_deg'),defs))
    parent_batch='C028' if batch=='C029' else 'C026'
    oldrows={(r['dataset'],r['record']):r for r in csv_rows(ROOT/parent_batch/'sequences.csv')}
    dev=csv_rows(b/'sequences.csv')
    assert len(dev)==len(oldrows)==213
    choose={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    proofs=[];catalog=Catalog()
    for row in dev:
        key=row['dataset'],row['record'];parent=oldrows[key]
        assert all(parent[k]==row[k] for k in ('group','scored_frames','scoring_indices_sha256','status'))
        for k in ('full_vqf_deg','C023_deg',parent_batch+'_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(row[k])-float(parent[k]))<1e-5
        if key in choose:
            raw,_=catalog.native(*key);score=model.score(raw)
            gap=max(abs(score['candidate_deg']-float(row[batch+'_deg'])),abs(score['full_vqf_deg']-float(row['full_vqf_deg'])))
            assert gap<1e-5 and row['scoring_indices_sha256']==score['scoring_indices_sha256']
            proofs.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':gap})
    assert len(proofs)==len(choose)
    report={'source_slots':len(weights),'source_acquisitions':len(np.unique(a['source_record_ordinal'])),'steps':len(rows),'scalar_updates':len(state)*len(rows),
            'independent_max_gap':gaps,'development_records':len(dev),'native_replays':proofs,
            'checkpoint_sha256':digest(b/'router.pt'),'source_sha256':digest(ROOT/'C026/training-inputs.npz')}
    (ROOT/'review'/f'{batch}_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(gaps,max(r['max_abs_gap_deg']for r in proofs))


if __name__=='__main__':main()
