"""Reconstruct C011's coefficient from native external training data and the archived draw schedule."""
import json
from itertools import islice
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter
from check_sources import ROOT, weights
from long_state_source_audit import training_records
from native_blocks import block_inputs
from optical_score import infer_canonical


def force_history(signal,observed,hz,rotation):
    force=rotation[observed].apply(signal[observed,:3])/9.80665
    force[:,2]=0.
    alpha=-np.expm1(-1./(2.*hz))
    run=lfilter([alpha],[1.,alpha-1.],force,axis=0)
    last=np.cumsum(observed)-1
    state=np.zeros_like(signal[:,:3])
    arrived=last>=0
    state[arrived]=run[last[arrived]]
    return state


def score_blocks(record,drawn,model):
    signal,hz,target,_=record
    observed=np.isfinite(signal).all(1)
    past=np.maximum.accumulate(np.where(observed,np.arange(len(signal)),-1))
    filled=signal[past.clip(min=0)].copy();filled[past<0]=0.
    q,windows,at,release=block_inputs(filled,hz)
    unique,inverse=np.unique(drawn,return_inverse=True)
    pos=np.searchsorted(release,unique)
    np.testing.assert_array_equal(release[pos],unique)
    w=windows[pos]
    rest=w[:,-1,12].astype(bool)
    delta=q[at[pos]].apply(infer_canonical(model,w))*.25
    delta[rest]=0.
    z=np.array([0.,0.,1.])
    base=z+delta
    ema=force_history(filled,observed,hz,q)[at[pos]]
    ema[rest]=0.
    truth=q[unique].apply(target[unique])
    return base[inverse],ema[inverse],truth[inverse],rest[inverse]


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/'C004/schedules.npz') as f:
        matrix=f['B'].ravel();off=f['record_offsets'];times=f['release_indices'];indices=np.searchsorted(off[1:],matrix,side='right')
    sources=json.loads((ROOT/'C004/training-preparation.json').read_text('utf8'))['records']
    model=weights(ROOT/'C004/B-readout.pt','mamba_v1')
    assert all(r['active_native_frames']==0 for r in sources)
    groups={};records=[]
    for i,(d,n,r) in enumerate(islice(training_records(),len(sources))):
        assert sources[i]['record']==n and sources[i]['domain']==d
        take=times[matrix[indices==i]]
        base,ema,truth,rest=score_blocks(r,take,model)
        assert len(take)==sources[i]['B_draws'] and len(take)>0
        group=groups.setdefault(d,[])
        group.append((base,ema,truth,np.full(len(take),1./len(take))))
        records.append({'domain':d,'record':n,'draws':len(take),'rest_draws':int(rest.sum())})
    info={}
    for d,entries in groups.items():
        b,m,y,w=(np.concatenate(parts) for parts in zip(*entries))
        w/=len(entries)
        norm=np.linalg.norm(m,axis=1)
        info[d]=(b,m,y,w,norm)
    def losses(k):
        result={}
        for d,(b,m,y,w,norm) in info.items():
            scale=np.minimum(k,np.tan(np.deg2rad(6.))/np.maximum(norm,1e-15))
            u=b+m*scale[:,None]
            u/=np.linalg.norm(u,axis=1,keepdims=True)
            result[d]=float(w@np.sum((u-y)**2,axis=1))
        return result
    original=losses(0.)
    def ratios(k):
        return {d:v/original[d] for d,v in losses(k).items()}
    def objective(k):return max(ratios(k).values())
    grid=np.linspace(0.,1.,101)
    values=[objective(k) for k in grid]
    trial=list(grid)
    for i in range(1,len(grid)-1):
        if values[i]<=min(values[i-1],values[i+1]):
            opt=minimize_scalar(objective,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-12})
            trial.append(float(opt.x))
    k=min(trial,key=lambda x:(objective(x),x))
    result={'source_records':len(records),'draws':len(matrix),'rest_draws':sum(x['rest_draws'] for x in records),
            'rest_draws_by_domain':{d:sum(x['rest_draws'] for x in records if x['domain']==d) for d in groups},
            'fitted_k':float(k),'base_loss_by_domain':original,'ratios_at_fit':ratios(k),'worst_ratio':objective(k),
            'native_reader_and_ema':'Independent scipy rotation, lfilter on finite observations, direct chord means. '
                'All 63 external training records have zero C003 bias activations. No target scores read.',
            'records':records}
    (ROOT/'review/C011_source_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(k,result['ratios_at_fit'],result['rest_draws_by_domain'],flush=True)


if __name__=='__main__':
    main()
