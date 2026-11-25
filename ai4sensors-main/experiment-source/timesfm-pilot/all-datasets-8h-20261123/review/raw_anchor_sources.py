"""Reconstruct unmodified VQF on the original external C023 training instants, independently of any target panel."""
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from filter_response_audit import phones
from lock_from_frontier import digest,inference,read
from locked_replay import Predictor
from native_catalog import body_reference,load_external,yt_record
from penn_handheld import Penn

UP=np.array([0.,0.,1.])


def source_inputs(keys):
    inventory={(r['dataset'],r['record']):r for r in read(ROOT/'C001/inventory.json')['records']}
    for domain,name in sorted(keys):
        if domain=='RIDI':continue
        r=inventory[domain,name]
        if domain=='YT_Server':
            assert r['side']=='external_training_candidate' and r['official_split']=='train'
            yield domain,name,yt_record(r)
        else:
            assert domain in ('BROAD','Sassari') and r['side']=='historical_source_train'
            x,quat,mask,hz=load_external({'domain':domain,'name':name,'path':r['path']})
            ref,ok=body_reference(quat,UP)
            yield domain,name,(x,hz,ref,mask&ok)
    yield from phones()


def native(signal,hz,idx):
    observed=np.isfinite(signal).all(1)
    last=np.maximum.accumulate(np.where(observed,np.arange(len(signal)),-1))
    x=signal[last.clip(min=0)].copy()
    x[last<0]=0.
    a=np.ascontiguousarray(x[:,:3]);w=np.ascontiguousarray(x[:,3:])
    output=VQF(1./hz).updateBatch(w,a)
    q=Rotation.from_quat(output['quat6D'],scalar_first=True)
    # Smoothed specific force in the raw VQF inertial frame. All observations are causal.
    alpha=-np.expm1(-1./hz)
    unit=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
    smoothed=lfilter([alpha],[1.,alpha-1.],q.apply(unit),axis=0)
    smoothed/=np.maximum(np.linalg.norm(smoothed,axis=1,keepdims=True),1e-12)
    return q[idx].inv().apply(UP),q[idx].inv().apply(smoothed[idx]),np.linalg.norm(output['bias'][idx],axis=1)


def angles(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum('ij,ij->i',a,b)))


def diagnostics(base,model,ref,force,epsilon=1e-12):
    d=angles(base,model)
    # End-point derivatives are descriptive only. The derivative is of squared angular error w.r.t.
    # a normalized linear interpolation of the *two estimators* along a single observed direction.
    deriv=[]
    v=model-base
    for endpoint in (base,model):
        cos=(endpoint*ref).sum(1).clip(-1.,1.)
        theta=np.deg2rad(angles(endpoint,ref))
        tangent=v-endpoint*(v*endpoint).sum(1)[:,None]
        factor=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>epsilon)
        deriv.append(-2*factor*(tangent*ref).sum(1)*(180/np.pi)**2)
    mid=base+.5*v;mid/=np.linalg.norm(mid,axis=1,keepdims=True)
    observed=angles(base,force)-angles(model,force)
    return {'raw':float(np.sqrt(np.mean(angles(base,ref)**2))),'C023':float(np.sqrt(np.mean(angles(model,ref)**2))),
            'halfway':float(np.sqrt(np.mean(angles(mid,ref)**2))),'derivative_at_raw':float(np.mean(deriv[0])),
            'derivative_at_C023':float(np.mean(deriv[1])),'median_model_disagreement_deg':float(np.median(d)),
            'past_force_favors_model_fraction':float((observed>0.).mean()),
            'mean_candidate_minus_raw_chord_when_past_force_favors_model':float(np.mean((np.sum((model-ref)**2,axis=1)-np.sum((base-ref)**2,axis=1))[observed>0.])) if (observed>0).any() else None}


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/'C022/training-slots.npz')as f:
        domains,names,times,target,q=(f[k]for k in ('domain','source_record_name','native_release','truth','q_default_release'))
    with np.load(ROOT/'C023/training-io.npz')as f: prediction=f['C023_prediction']
    raw_dir=np.zeros_like(target);force=np.zeros_like(target);bias=np.zeros(len(target))
    keys=set(zip(domains,names));seen=set();rows=[]
    for ds,name,raw in source_inputs(keys):
        assert (ds,name) in keys and (ds,name) not in seen
        pick=np.flatnonzero((domains==ds)&(names==name))
        t=times[pick]
        s,hz,truth,mask=raw
        assert np.isfinite(s[t]).all() and mask[t].all(),(ds,name)
        rotation=Rotation.from_quat(q[pick],scalar_first=True)
        np.testing.assert_allclose(rotation.apply(truth[t]),target[pick],atol=2e-6)
        rb,fa,gb=native(s,hz,t)
        raw_dir[pick]=rotation.apply(rb);force[pick]=rotation.apply(fa);bias[pick]=gb
        rows.append({'domain':ds,'record':name,'slots':len(t),**diagnostics(raw_dir[pick],prediction[pick],target[pick],force[pick])})
        seen.add((ds,name))
    assert keys==seen and len(keys)==139
    payload=ROOT/'review/external-raw-anchor-139.npz'
    np.savez_compressed(payload,domain=domains,record=names,native_release=times,raw_default=raw_dir,
                        past_force=force,estimated_bias_rad_s=bias,C023=prediction,truth=target)
    recipes={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C023','C023_deg'),recipes))
    for name in ('af','as'):
        raw,quality=Penn().native(name)
        s,hz,truth,mask=raw;idx=np.flatnonzero(mask)
        predicted=model.predict(s,hz,idx);rb,fa,gb=native(s,hz,idx)
        np.testing.assert_allclose(rb,predicted['full_vqf_deg'],atol=1e-13)
        rows.append({'domain':'PennCOSYVIO','record':name,'slots':len(idx),**diagnostics(rb,predicted['candidate_deg'],truth[idx],fa)})
    summary={}
    for d in sorted({r['domain']for r in rows}):
        part=[r for r in rows if r['domain']==d]
        summary[d]={k:float(np.mean([r[k]for r in part]))for k in ('raw','C023','halfway','derivative_at_raw','derivative_at_C023','past_force_favors_model_fraction')}
    (ROOT/'review/raw_anchor_sources.json').write_text(json.dumps({'rows':rows,'source_macro':summary,'raw_bank_sha256':digest(payload),
                 'scope':'Only four original source groups on their archived 15,360 C023 training slots and two A training records from Penn on all admissible native rows. Per-record endpoints and derivatives. No target data, no parameter selection.'},indent=2)+'\n',encoding='utf8')
    print(summary)


if __name__=='__main__':main()
