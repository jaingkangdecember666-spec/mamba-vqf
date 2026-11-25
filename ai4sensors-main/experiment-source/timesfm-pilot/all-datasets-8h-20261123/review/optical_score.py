"""Replay optical development anchors from immutable Mamba weights and a shared optical reference."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from check_sources import ROOT, TRANSFER, weights
from native_blocks import block_inputs
from optical_reference import extract
from replay_native import canonical_frame, metric

sys.path.insert(0,str(ROOT/'C003'))
from online_bias import estimate


@torch.inference_mode()
def infer_canonical(model, windows):
    network,avg,std,cap=model
    framed,axes,active=canonical_frame(windows)
    parts=[]
    for idx in range(0,len(framed),128):
        x=torch.from_numpy(framed[idx:idx+128]).to(avg.device)
        parts.append(network(((x-avg)/std).clamp(-12,12)).cpu().numpy())
    local=np.concatenate(parts).astype('float64')*.1
    body=np.einsum('nij,nj->ni',axes,local)
    body[~active]=0.
    up=windows[:,-1,6:9].astype('float64')
    diff=body-(body*up).sum(axis=1)[:,None]*up
    scale=np.minimum(1.,np.tan(np.deg2rad(cap))/np.maximum(1e-12,np.linalg.norm(diff,axis=1)))
    return scale[:,None]*diff


def comparison(signal,hz,ref,mask,models,online_bias=True):
    observed=np.isfinite(signal).all(1)
    valid=mask&observed
    if not valid.any():raise ValueError('No paired native scoring rows')
    previous=np.maximum.accumulate(np.where(observed,np.arange(len(signal)),-1))
    signal=signal[np.maximum(previous,0)].copy();signal[previous<0]=0.
    q,w,last,release=block_inputs(signal,hz)
    ids=np.flatnonzero(valid)
    truth=q[ids].apply(ref[ids]);z=np.array([0.,0.,1.])
    rmse={'full_vqf':metric(np.broadcast_to(z,truth.shape),truth)}
    active_count=0
    if online_bias:
        bias,on,info,_=estimate(signal,observed,1/hz);active_count=int((on&valid).sum())
        if on.any():
            cleaned=signal.copy();cleaned[:,:3]-=bias
            q,w,last,release=block_inputs(cleaned,hz)
            truth=q[ids].apply(ref[ids])
    rmse['front_end_vqf']=metric(np.broadcast_to(z,truth.shape),truth)
    lookup=np.searchsorted(release,ids,side='right')
    for name,(head,gain) in models.items():
        offset=q[last].apply(infer_canonical(head,w));up=z[None,:]+gain*offset
        up/=np.linalg.norm(up,axis=1,keepdims=True)
        rmse[name]=metric(np.vstack((z,up))[lookup],truth)
    return {'scored_native_frames':len(ids),'mask_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest(),
            'active_bias_scored_frames':active_count,'rmse_deg':rmse}


def main():
    torch.set_num_threads(2)
    data=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    anchors=json.loads((ROOT/'review/mount_development_audit.json').read_text('utf8'))
    head=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    results=[]
    for pick in anchors:
        row=next(r for r in data['records'] if r['side']=='development' and r['record']==pick['name'])
        signal,hz,truth,eligible,qa=extract(row)
        score=comparison(signal,hz,truth,eligible,{'C003':(head,.25)})
        results.append({'domain':row['dataset'],'group':row['group'],'record':row['record'],
                        'reference_quality_flags':qa['quality_flags'],'reference_geometry':qa,**score})
    (ROOT/'review/optical_anchor_replay.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')
    for r in results:print(r['record'],r['rmse_deg'],r['reference_quality_flags'])


if __name__=='__main__':main()

