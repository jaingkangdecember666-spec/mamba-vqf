"""Verify the causal stable-force gate and tangent on the four actual external training draws."""
import json
from itertools import chain,islice

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

from check_sources import ROOT,TRANSFER,weights
from filter_response_audit import phones
from long_state_source_audit import training_records
from native_blocks import block_inputs
from optical_score import estimate,infer_canonical

UP=np.array([0.,0.,1.])


def force_state(x,obs,q,hz):
    use=obs & (np.linalg.norm(x[:,:3],axis=1)>1e-12)
    a=x[use,:3]
    world=q[use].apply(a/np.linalg.norm(a,axis=1)[:,None])
    alpha=-np.expm1(-1./(.25*hz))
    series=lfilter([alpha],[1.,alpha-1.],world,axis=0)
    before=np.cumsum(use)-1
    state=np.zeros((len(x),3))
    if len(series):state[before>=0]=series[before[before>=0]]
    length=np.linalg.norm(state,axis=1)
    spread=np.degrees(np.sqrt(2.*np.maximum(0.,1.-length)))
    f=state/np.maximum(length[:,None],1e-12);f[:,2]=0.
    return f,spread


def sample(raw,take,heads):
    x,hz,truth,mask=raw
    obs=np.isfinite(x).all(1)
    past=np.maximum.accumulate(np.where(obs,np.arange(len(x)),-1))
    clean=x[past.clip(min=0)].copy();clean[past<0]=0.
    offset,active,_,_=estimate(clean,obs,1./hz)
    if active.any():clean[:,:3]-=offset
    q,w,at,release=block_inputs(clean,hz)
    times,inv=np.unique(take,return_inverse=True)
    pos=np.searchsorted(release,times)
    np.testing.assert_array_equal(release[pos],times)
    rest=w[pos,-1,12].astype(bool)
    force,spread=force_state(clean,obs,q,hz)
    enable=(spread[at[pos]]<3.) & ~rest
    f=force[at[pos]].copy();f[~enable]=0.
    d1=q[at[pos]].apply(infer_canonical(heads[0],w[pos]))
    d2=q[at[pos]].apply(infer_canonical(heads[1],w[pos]));d2[rest]=0.
    base=UP+.125*(d1+d2)
    y=q[times].apply(truth[times])
    return base[inv],f[inv],y[inv],enable[inv],int(active.sum())


def loss(base,f,y,k):
    cap=np.tan(np.deg2rad(6.))
    u=base+f*np.minimum(k,cap/np.maximum(np.linalg.norm(f,axis=1),1e-15))[:,None]
    u=u/np.linalg.norm(u,axis=1,keepdims=True)
    return ((u-y)**2).sum(1)


def main():
    torch.set_num_threads(2)
    head=(weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),weights(ROOT/'C013/seed-C006-readout.pt','mamba_v1'))
    with np.load(ROOT/'C015/sampling.npz') as f:
        ords=f['C015_record_ordinal'].ravel();times=f['C015_native_release'].ravel()
    groups={};details=[]
    for i,(d,n,data) in enumerate(chain(islice(training_records(),63),phones())):
        take=times[ords==i];assert len(take)
        b,f,y,enabled,active=sample(data,take,head)
        entry=(b,f,y,np.full(len(take),1./len(take)))
        groups.setdefault(d,[]).append(entry)
        details.append({'source':d,'record':n,'sampled_slots':len(take),'stable_nonrest_slots':int(enabled.sum()),
                        'causal_accel_debias_active_native_rows':active})
    arrays={};summary={}
    for d,records in groups.items():
        b,f,y,w=(np.concatenate(s) for s in zip(*records))
        w/=len(records)
        zero=w@loss(b,f,y,0.)
        arrays[d]=(b,f,y,w,zero)
        norm=np.linalg.norm(b,axis=1,keepdims=True);u=b/norm
        derivative=((u-y)*(f-u*(u*f).sum(1,keepdims=True))).sum(1)*2./norm[:,0]
        summary[d]={'records':len(records),'draws':len(b),'enabled':int((np.linalg.norm(f,axis=1)>0.).sum()),
                    'relative_loss_derivative_at_zero':float(w@derivative/zero)}
    def ratios(k):
        return {d:float(w@loss(b,f,y,k)/zero) for d,(b,f,y,w,zero) in arrays.items()}
    def worst(k):return max(ratios(k).values())
    fit=minimize_scalar(worst,bounds=(0.,1.),method='bounded',options={'xatol':1e-12})
    guess=min([0.,1.,float(fit.x)],key=worst)
    out={'scope':'C014-B fixed, 15,360 original C015 four-source draw slots. Uses external references only. '
                 'An independent one-dimensional check is included for later comparison; no eight-domain scores or changes to deployed coefficients.',
         'domains':summary,'n_slots':len(ords),'records':details,'estimated_k':guess,'estimated_relative_losses':ratios(guess)}
    (ROOT/'review/stable_force_source_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(summary,guess,ratios(guess),flush=True)


if __name__=='__main__':
    main()
