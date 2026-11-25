"""Supply a fully aligned raw-FullVQF alternative on the existing C033 external training instants."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from anchor_blend_features import design
from check_history_router import prediction
from lock_from_frontier import ROOT,read,digest
from penn_handheld import Penn
from raw_anchor_sources import native
from source_motion_router_diagnostic import native as corrected


def main():
    origin=ROOT/'review/external-raw-anchor-139.npz'
    ledger=ROOT/'C033/training-inputs.npz'
    anchor=torch.load(ROOT/'C023/router.pt',map_location='cpu',weights_only=True)
    p=np.r_[anchor['state_dict']['weight'].numpy().ravel(),anchor['state_dict']['bias'].numpy()]
    with np.load(ledger)as f:
        dims=f['enabled'].shape
        assert dims==(160,98)
        flat=lambda k:f[k].reshape((-1,)+f[k].shape[2:])
        domains,names,times,target,record_id=(flat(k)for k in ('domain','source_record_name','native_release','truth','source_record_ordinal'))
        pred,*_=prediction(flat('z'),anchor['p0'].numpy(),p,*[flat(k)for k in ('center','fast','slow','enabled')])
    raw_dir=np.empty_like(target);force=np.empty_like(target);bias=np.empty(len(target))
    original=(np.arange(len(domains))%98)<96
    with np.load(origin)as old:
        for k,v in [('domain',domains),('record',names),('native_release',times),('truth',target)]:
            np.testing.assert_array_equal(v[original],old[k])
        np.testing.assert_allclose(pred[original],old['C023'],rtol=1e-12,atol=1e-13)
        raw_dir[original]=old['raw_default'];force[original]=old['past_force'];bias[original]=old['estimated_bias_rad_s']
    ds=Penn()
    for n in ('af','as'):
        idx=np.flatnonzero(names==n);t=times[idx]
        assert len(idx)==len(np.unique(t))==160 and np.all(domains[idx]=='PennCOSYVIO')
        packet,qa=ds.native(n)
        x,hz,u,mask=packet
        assert mask[t].all()
        clean,_=corrected(packet)
        q=VQF(1./hz).updateBatch(np.ascontiguousarray(clean[:,3:]),np.ascontiguousarray(clean[:,:3]))['quat6D']
        rotation=Rotation.from_quat(q[t],scalar_first=True)
        np.testing.assert_allclose(rotation.apply(u[t]),target[idx],atol=1e-10)
        raw,sm,b=native(x,hz,t)
        raw_dir[idx]=rotation.apply(raw);force[idx]=rotation.apply(sm);bias[idx]=b
    z=design(raw_dir,pred,force,bias)
    weights=ROOT/'review/raw-anchor-training-141.npz'
    np.savez_compressed(weights,domain=domains,record=names,record_id=record_id,native_release=times,
                        raw_default=raw_dir,C023=pred,truth=target,past_force=force,raw_gyro_bias_rad_s=bias,features=z)
    result={'scope':'The exact 160 x 98 real external slots of C033, in original order, plus unaltered FullVQF and the native causal one-second specific-force observer. Rotations at the sensor origin preserve angles. C023 is frozen; no fit or target reads in this construction.',
            'slots':len(times),'record_counts':{d:len(set(names[domains==d])) for d in np.unique(domains)},
            'input_sha256':{str(p.relative_to(ROOT)):digest(p)for p in (origin,ledger,ROOT/'C023/router.pt',ROOT/'external-sources/penn-handheld/preprocessing-lock.json')},
            'output_sha256':digest(weights),'features_sha256':digest(ROOT/'review/anchor_blend_features.py')}
    (ROOT/'review/raw-anchor-training-141.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['slots'],result['record_counts'])


if __name__=='__main__':main()
