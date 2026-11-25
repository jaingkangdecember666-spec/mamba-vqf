"""Rebuild the exact C023 physical-only ablation at each external C033 training instant."""
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from anchor_blend_features import included_degrees
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from penn_handheld import Penn
from raw_anchor_sources import source_inputs
from source_motion_router_diagnostic import native as corrected
from stable_force_source import force_state


def main():
    torch.set_num_threads(2)
    path=ROOT/'review/raw-anchor-training-141.npz'
    with np.load(path)as f:
        saved={k:f[k]for k in f}
    with np.load(ROOT/'C033/training-inputs.npz')as f:
        flat=lambda k:f[k].reshape((-1,)+f[k].shape[2:])
        at,t,rest,valid,fast,slow,truth=(flat(k)for k in ('native_input_last','native_release','rest','enabled','fast','slow','truth'))
        np.testing.assert_array_equal(t,saved['native_release'])
    kforce=read(ROOT/'C017/coefficient-lock.json')['k']
    kbank=read(ROOT/'C022/coefficient-lock.json')['k']
    dom,names=saved['domain'],saved['record']
    existing=set(zip(dom[dom!='PennCOSYVIO'],names[dom!='PennCOSYVIO']))
    def records():
        yield from source_inputs(existing)
        for name in ('af','as'):yield 'PennCOSYVIO',name,Penn().native(name)[0]
    physical=np.zeros_like(truth);details=[];seen=set()
    spec={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C023','C023_deg'),spec))
    for d,n,raw in records():
        assert (d,n) not in seen;seen.add((d,n))
        ix=np.flatnonzero((dom==d)&(names==n));ts=t[ix];cut=at[ix]
        clean,hz=corrected(raw)
        q=VQF(1./hz).updateBatch(np.ascontiguousarray(clean[:,3:]),np.ascontiguousarray(clean[:,:3]))['quat6D']
        rot=Rotation.from_quat(q,scalar_first=True)
        np.testing.assert_allclose(rot[ts].apply(raw[2][ts]),truth[ix],atol=2e-6)
        observed=np.isfinite(raw[0]).all(1)
        direction,spread=force_state(clean,observed,rot,hz)
        release_blocks=69+10*np.arange(max(0,(int(np.floor(len(clean)*100/hz+1e-9))-60)//10))
        all_t=np.ceil((release_blocks+1)*(hz/100.)-1e-9).astype(int)-1
        pos=np.searchsorted(all_t,ts);np.testing.assert_array_equal(ts,all_t[pos])
        first=np.floor((6+10*pos)*(hz/100.)+1e-9).astype(int)
        valid_acc=observed & (np.linalg.norm(clean[:,:3],axis=1)>1e-12)
        last=np.maximum.accumulate(np.where(valid_acc,np.arange(len(clean)),-1))
        force=direction[cut].copy()
        force[(spread[cut]>=3.)|rest[ix]|(last[cut]<first)]=0.
        gain=np.minimum(kforce,np.tan(np.deg2rad(6.))/np.maximum(np.linalg.norm(force,axis=1),1e-15))
        y=np.array([0.,0.,1.])+force*gain[:,None]+kbank*(2.*fast[ix]+slow[ix])/3.*valid[ix,None]
        physical[ix]=y/np.linalg.norm(y,axis=1,keepdims=True)
        metrics={key:float(np.sqrt(np.mean(included_degrees(x,truth[ix])**2)))for key,x in
                 [('physical',physical[ix]),('default',saved['raw_default'][ix]),('C023',saved['C023'][ix])]}
        extra={}
        if d=='PennCOSYVIO' or n in ('hao_handheld1','01_undisturbed_slow_rotation_A','fast/AP1'):
            independent=model.predict(raw[0],hz,ts)['physical_deg']
            error=float(np.max(included_degrees(independent,rot[ts].inv().apply(physical[ix]))));assert error<1e-5
            extra={'complete_model_max_prediction_gap_deg':error}
        details.append({'domain':d,'record':n,'slots':len(ts),**metrics,**extra})
    assert len(seen)==141 and len(t)==15680
    artifact=ROOT/'review/external-physical-anchor-141.npz'
    np.savez_compressed(artifact,physical=physical,domain=dom,record=names,native_release=t)
    groups={}
    for d in sorted(set(dom)):
        r=[v for v in details if v['domain']==d]
        groups[d]={k:float(np.mean([v[k]for v in r]))for k in ('default','physical','C023')}
    out={'scope':'No change to predictions. Rebuilt the physical ablation in the same frame on all 15,680 existing external training slots as a control. Five full-native cross-checks. No validation or formal input.',
         'rows':details,'group_record_means_deg':groups,'physical_bank_sha256':digest(artifact),'primary_external_bank_sha256':digest(path)}
    (ROOT/'review/external-physical-anchor-141.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(groups)


if __name__=='__main__':main()
