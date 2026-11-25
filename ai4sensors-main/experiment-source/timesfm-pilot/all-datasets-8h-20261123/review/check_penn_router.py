"""Reproduce participation by both Penn A acquisitions, externally fitted parameters and paired development results."""
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_history_router import gradient,prediction
from check_training_batch import csv_rows
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from native_blocks import block_inputs
from native_catalog import Catalog
from optical_score import canonical_frame
from penn_handheld import Penn
from response_router import alternatives
from source_motion_router_diagnostic import native


def main():
    torch.set_num_threads(2)
    b=ROOT/'C033'
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    receipt=read(b/'router-lock.json');assert receipt['fit_count']==1 and receipt['checkpoint_sha256']==digest(b/'router.pt')
    with np.load(b/'training-inputs.npz')as f:a={k:f[k]for k in f}
    with np.load(ROOT/'C022/training-slots.npz')as old:
        for k in ('center','fast','slow','truth','enabled','domain','record_equal_weights','native_release','native_input_last','source_record_ordinal','rest'):
            np.testing.assert_array_equal(a[k][:,:96].reshape(old[k].shape),old[k])
    with np.load(ROOT/'C023/training-io.npz')as old:
        np.testing.assert_array_equal(a['z'][:,:96].reshape(-1,24),old['normalized_input'])
    flat=lambda k:a[k].reshape((-1,)+a[k].shape[2:])
    z,old,fast,slow,y,use=(flat(k)for k in ('z','center','fast','slow','truth','enabled'))
    assert z.shape==(15680,24) and len(np.unique(a['source_record_ordinal']))==141
    prior=cp['p0'].numpy()
    init,*_=prediction(z,prior,np.zeros(50),old,fast,slow,use)
    w=flat('record_equal_weights').copy();initial=((init-y)**2).sum(1)
    for d in np.unique(a['domain']):
        sel=flat('domain')==d;assert abs(w[sel].sum()-1.)<1e-10
        w[sel]/=w[sel]@initial[sel]
    np.testing.assert_allclose(w,flat('effective_loss_weights'),rtol=5e-11,atol=2e-10)
    logs=[json.loads(t)for t in (b/'optimizer-steps.jsonl').read_text().splitlines()]
    assert len(logs)==160
    par=np.zeros(50);m=np.zeros(50);v=np.zeros(50);gaps=[]
    contributing={'af':0,'as':0}
    for i,row in enumerate(logs,1):
        assert row['step']==i
        ix=np.arange(98*(i-1),98*i)
        def part(k):
            return gradient(z[k],prior,par,old[k],fast[k],slow[k],y[k],use[k],32.*w[k])
        loss,g=part(ix)
        gaps.extend([abs(loss-row['objective']),np.max(np.abs(g-row['gradient_before_clip']))])
        for name,component in row['source_and_penn_record_gradients'].items():
            key='source_record_name' if name in contributing else 'domain'
            take=ix[flat(key)[ix]==name]
            assert len(take)==component['slots']
            if len(take):
                L,grad=part(take)
            else:L,grad=0.,np.zeros(50)
            gaps.extend([abs(L-component['objective']),np.max(np.abs(grad-component['gradient']))])
            if name in contributing:contributing[name]+=int(np.linalg.norm(grad)>0.)
        g*=min(1.,2./(np.linalg.norm(g)+1e-6))
        m=.9*m+.1*g;v=.999*v+.001*g*g
        par=(1.-.01*.01)*par-.01*(m/(1.-.9**i))/(np.sqrt(v/(1.-.999**i))+1e-8)
        gaps.append(np.max(np.abs(par-row['weights_after'])))
    assert max(gaps)<1e-10 and min(contributing.values())>0
    np.testing.assert_allclose(par,np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()],atol=1e-12)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    base=Predictor(inference(('C020','C020_deg'),defs)); model=Predictor(inference(('C033','C033_deg'),defs))
    sources=[];ds=Penn()
    for j,name in enumerate(('af','as'),96):
        orig,info=ds.native(name);sig,hz=native(orig);q,hist,cut,release=block_inputs(sig,hz)
        ok=canonical_frame(hist)[2]&~hist[:,-1,12].astype(bool)&orig[3][release]
        available=release[ok].tolist()
        take=np.array(sorted(available,key=lambda x:(hashlib.sha256(f'C033|{name}|{x}'.encode()).digest(),x))[:160])
        np.testing.assert_array_equal(take,a['native_release'][:,j])
        stop=cut[np.searchsorted(release,take)]
        np.testing.assert_array_equal(stop,a['native_input_last'][:,j])
        np.testing.assert_array_equal(a['enabled'][:,j],True)
        def at(tau):
            qs=VQF(1./hz,tauAcc=tau).updateBatch(np.ascontiguousarray(sig[:,3:]),np.ascontiguousarray(sig[:,:3]))['quat6D']
            return Rotation.from_quat(qs[stop],scalar_first=True)
        f,s=alternatives(q[stop],at(1.5),at(6.))
        center=q[take].apply(base.predict(orig[0],hz,take)['candidate_deg']);center/=center[:,2:3]
        errors={k:float(np.max(np.abs(a[k][:,j]-v)))for k,v in [('fast',f),('slow',s),('center',center),('truth',q[take].apply(orig[2][take]))]}
        assert max(errors.values())<2e-7 and len(set(take))==160
        score=model.score(orig);official=next(r for r in read(b/'penn-training-source-scores.json')['records'] if r['record']==name)
        assert score['scoring_indices_sha256']==official['mask_sha256']==info['scoring_indices_sha256']
        e=abs(score['candidate_deg']-official['full_record_tilt_RMSE_deg']['C033']);assert e<1e-5
        sources.append({'name':name,'independently_reconstructed_eligible_releases':len(available),'selected_unique':len(take),
                        'input_max_abs_errors':errors,'full_record_RMSE_max_gap_deg':e})
    rows=csv_rows(b/'sequences.csv');previous={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C031/sequences.csv')}
    picks={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    matches=[];db=Catalog();assert len(rows)==len(previous)==213
    for r in rows:
        key=r['dataset'],r['record'];other=previous[key]
        for field in ('group','status','scoring_indices_sha256','scored_frames'):assert r[field]==other[field]
        for field in ('C023_deg','C031_deg','full_vqf_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(r[field])-float(other[field]))<1e-5
        if key in picks:
            raw,_=db.native(*key);s=model.score(raw)
            gap=max(abs(s['candidate_deg']-float(r['C033_deg'])),abs(s['full_vqf_deg']-float(r['full_vqf_deg'])))
            assert s['scoring_indices_sha256']==r['scoring_indices_sha256'] and gap<1e-5
            matches.append({'dataset':key[0],'record':key[1],'max_gap_deg':gap})
    assert len(matches)==len(picks)
    report={'complete_external_draws':15680,'legacy_draws_unchanged':15360,'distinct_external_acquisitions':141,
            'new_external_penn_draws':320,'optimizer_steps':160,'parameters_per_update':50,'max_optimizer_component_gap':max(gaps),
            'nonzero_steps_by_new_record':contributing,'original_source_checks':sources,'paired_development_records':213,
            'independent_development_replays':matches,'checkpoint_sha256':digest(b/'router.pt')}
    (ROOT/'review/C033_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(max(gaps),contributing,max(r['max_gap_deg']for r in matches))


if __name__=='__main__':main()
