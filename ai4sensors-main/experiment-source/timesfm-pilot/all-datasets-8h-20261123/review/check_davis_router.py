"""Independently verify the DAVIS rows, source participation, all optimizer steps, and native development scores."""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_history_router import gradient,prediction
from check_training_batch import csv_rows
from check_sources import ROOT
from davis_handheld import Davis
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from response_router import alternatives
from source_motion_router_diagnostic import native


def captures(a,base):
    index=a['source_record_ordinal'].ravel();rows=[]
    reader=Davis()
    details=read(ROOT/'C024/training-preparation.json')['rows']
    for item in details:
        n=item['record'];sel=index==item['ordinal']
        assert reader.roles[n]=='external_train'
        raw,_=reader.native(n);x,hz=native(raw)
        cut=a['native_input_last'].ravel()[sel];release=a['native_release'].ravel()[sel]
        assert len(np.unique(release))==len(release) and np.all(cut<=release)
        w=np.ascontiguousarray(x[:,3:]);acc=np.ascontiguousarray(x[:,:3])
        def orientation(t):
            q=VQF(1./hz,tauAcc=t).updateBatch(w,acc)['quat6D']
            return Rotation.from_quat(q,scalar_first=True)
        central=orientation(3.)
        f,s=alternatives(central[cut],orientation(1.5)[cut],orientation(6.)[cut])
        truth=central[release].apply(raw[2][release])
        assert np.all(raw[3][release])
        center=central[release].apply(base.predict(raw[0],hz,release)['candidate_deg'])
        center/=center[:,2:3]
        results={}
        for k,v in [('center',center),('fast',f),('slow',s),('truth',truth)]:
            gap=np.abs(a[k].reshape(-1,3)[sel]-v).max()
            assert gap<2e-7,(n,k,gap)
            results[k]=float(gap)
        rows.append({'record':n,'slots':int(sel.sum()),'unique_releases':len(np.unique(release)),
                     'source_native_mask_and_birth_time_ok':True,'max_component_gap':results})
    assert len(rows)==6 and sum(r['slots']for r in rows)==1280
    return rows


def optimizer(a,cp):
    features=a['z'].reshape(-1,24)
    flat=lambda k:a[k].reshape((-1,)+a[k].shape[2:])
    prior=cp['p0'].numpy()
    old,fast,slow,y,use=(flat(k)for k in ('center','fast','slow','truth','enabled'))
    init,*_=prediction(features,prior,np.zeros(50),old,fast,slow,use)
    w=flat('record_equal_weights').copy()
    loss0=((init-y)**2).sum(1)
    for d in np.unique(a['domain']):
        sel=flat('domain')==d
        assert abs(w[sel].sum()-1.)<1e-10
        w[sel]/=w[sel]@loss0[sel]
    with np.load(ROOT/'C024/training-io.npz')as io:
        np.testing.assert_allclose(w,io['effective_loss_weights'],rtol=5e-11,atol=2e-10)
    logs=[json.loads(t)for t in (ROOT/'C024/optimizer-steps.jsonl').read_text().splitlines()]
    assert len(logs)==160
    param=np.zeros(50);m=np.zeros(50);v=np.zeros(50)
    weight_gap=gradient_gap=per_capture_gap=loss_gap=0.;nonzero={}
    for i,row in enumerate(logs):
        assert row['step']==i+1
        ix=np.arange(104*i,104*(i+1))
        def grad(ix):
            return gradient(features[ix],prior,param,old[ix],fast[ix],slow[ix],y[ix],use[ix],32.*w[ix])
        loss,g=grad(ix)
        loss_gap=max(loss_gap,abs(loss-row['objective']))
        gradient_gap=max(gradient_gap,np.max(np.abs(g-row['gradient_before_clip'])))
        for j,(name,entry) in enumerate(row['DAVIS_per_capture'].items()):
            pick=ix[flat('source_record_ordinal')[ix]==139+j]
            assert len(pick) in (1,2)
            _,component=grad(pick)
            per_capture_gap=max(per_capture_gap,np.max(np.abs(component-entry['true_gradient'])))
            nonzero[name]=nonzero.get(name,0)+int(np.linalg.norm(component)>0.)
        np.testing.assert_allclose(param,row['weights_before'],rtol=1e-11,atol=1e-13)
        g*=min(1.,2./(np.linalg.norm(g)+1e-6))
        m=.9*m+.1*g;v=.999*v+.001*g*g
        param=(1.-.01*.01)*param-.01*(m/(1.-.9**(i+1)))/(np.sqrt(v/(1.-.999**(i+1)))+1e-8)
        weight_gap=max(weight_gap,np.max(np.abs(param-row['weights_after'])))
    assert max(gradient_gap,weight_gap,per_capture_gap,loss_gap)<1e-10
    np.testing.assert_allclose(param,np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()],atol=1e-12)
    return {'parameter_updates':50*len(logs),'actual_steps':len(logs),'lost_or_duplicate_fit':False,
            'gradient_max_abs_gap':gradient_gap,'per_capture_gradient_max_abs_gap':per_capture_gap,
            'weight_update_max_abs_gap':weight_gap,'objective_max_abs_gap':loss_gap,'nonzero_gradient_steps_by_capture':nonzero}


def main():
    torch.set_num_threads(2)
    b=ROOT/'C024'
    cp=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    lock=read(b/'router-lock.json')
    assert lock['fit_count']==1 and lock['checkpoint_sha256']==digest(b/'router.pt')
    with np.load(b/'training-inputs.npz')as npz:a={k:npz[k]for k in npz.files}
    with np.load(ROOT/'C022/training-slots.npz')as old:
        for k in ('center','fast','slow','truth','enabled','domain','record_equal_weights','native_release','native_input_last','source_record_ordinal'):
            np.testing.assert_array_equal(a[k][:,:96].reshape(old[k].shape),old[k])
    with np.load(ROOT/'C023/training-io.npz')as old:
        np.testing.assert_array_equal(a['z'][:,:96].reshape(-1,24),old['normalized_input'])
    report=optimizer(a,cp)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    base=Predictor(inference(('C020','C020_deg'),defs))
    report['native_training_captures']=captures(a,base)
    model=Predictor(inference(('C024','C024_deg'),defs));raw=Catalog()
    rows=csv_rows(b/'sequences.csv');old=csv_rows(ROOT/'C023/sequences.csv')
    assert len(rows)==len(old)==213
    picks={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    proofs=[]
    for current,previous in zip(rows,old,strict=True):
        for k in ('dataset','record','group','scoring_indices_sha256','scored_frames','status'):assert current[k]==previous[k]
        for k in ('C020_deg','C022_deg','C023_deg','full_vqf_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(current[k])-float(previous[k]))<1e-5
        key=current['dataset'],current['record']
        if key in picks:
            data,_=raw.native(*key);score=model.score(data)
            assert score['scoring_indices_sha256']==current['scoring_indices_sha256']
            gap=max(abs(score[k]-float(current[c])) for k,c in [('candidate_deg','C024_deg'),('full_vqf_deg','full_vqf_deg')])
            assert gap<1e-5
            proofs.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':gap})
    assert len(proofs)==len(picks)
    report.update(raw_development_replays=proofs,development_controls=len(rows),source_records=145,legacy_identical_slots=15360,
                  new_DAVIS_slots=1280,checkpoint_sha256=digest(b/'router.pt'))
    (ROOT/'review/C024_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(report['per_capture_gradient_max_abs_gap'],max(r['max_abs_gap_deg']for r in proofs))


if __name__=='__main__':main()
