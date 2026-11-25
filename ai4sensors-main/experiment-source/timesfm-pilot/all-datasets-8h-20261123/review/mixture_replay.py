"""Native tilt predictions for fixed mixes of causal Mamba tangent corrections."""
import numpy as np

from native_blocks import block_inputs
from optical_score import estimate, infer_canonical


UP=np.array([0.,0.,1.])


def replay(signal,hz,models,groups,indices,stable_force_gains=None,additional_heads=None,filter_bank_gains=None,filter_router=None):
    observed=np.isfinite(signal).all(axis=1)
    previous=np.maximum.accumulate(np.where(observed,np.arange(len(signal)),-1))
    x=signal[previous.clip(min=0)].copy()
    x[previous<0]=0.
    raw_q,windows,at,releases=block_inputs(x,hz)
    q=raw_q
    bias,active,_,_=estimate(x,observed,1/hz)
    if active.any():
        x[:,:3]-=bias
        q,windows,at,releases=block_inputs(x,hz)
    if not len(releases):
        out={name:q[indices].inv().apply(UP) for name in groups}
        out['full_vqf_deg']=raw_q[indices].inv().apply(UP)
        return out
    rest=windows[:,-1,12].astype(bool)
    bank=None
    routed=None
    if filter_router:
        from response_router import release_alternatives
        fast,slow=release_alternatives(x,hz,q,at)
        context=None
        if filter_router.context:
            from router_motion_context import series
            context=series(x,hz)[at]
        pp,valid=filter_router.probabilities(windows,context,(fast,slow))
        usable=valid & ~rest
        bank=(2.*fast+slow)/3.
        bank[~usable]=0.
        routed=pp[:,:1]*fast+pp[:,1:2]*slow
        routed[~usable]=0.
    elif filter_bank_gains and any(filter_bank_gains.values()):
        from filter_response_bank import response_series
        from optical_score import canonical_frame
        bank=response_series(x,hz,q,at)
        usable=canonical_frame(windows)[2] & ~rest
        bank[~usable]=0.
    force=None
    if stable_force_gains:
        from stable_force_source import force_state
        direction,spread=force_state(x,observed,q,hz)
        force=direction[at].copy()
        valid_input=observed & (np.linalg.norm(x[:,:3],axis=1)>1e-12)
        last_valid=np.maximum.accumulate(np.where(valid_input,np.arange(len(x)),-1))
        first_block=6+10*np.arange(len(at))
        first_native=np.floor(first_block*(hz/100.)+1e-9).astype('int64')
        force[(spread[at]>=3.) | rest | (last_valid[at]<first_native)]=0.
    lookup=np.searchsorted(releases,indices,side='right')
    output={'full_vqf_deg':raw_q[indices].inv().apply(UP)}
    pieces={}
    coarse=None
    for name,(model,gain,gate) in models.items():
        if getattr(model[0],'history_scale',1)==4:
            from multirate_blocks import long_windows
            if coarse is None:
                coarse,ready=long_windows(windows)
            delta=np.zeros((len(windows),3))
            if ready.any():
                delta[ready]=gain*q[at[ready]].apply(infer_canonical(model,coarse[ready]))
        else:
            delta=gain*q[at].apply(infer_canonical(model,windows))
        if gate=='rest_off':
            delta[rest]=0.
        elif gate!='always':
            raise ValueError(gate)
        pieces[name]=delta
    frame=q[indices].inv()
    for name,components in groups.items():
        if components and not np.isclose(sum(components.values()),1.):
            raise ValueError(f'Invalid Mamba mixing fractions for {name}')
        delta=sum((coeff*pieces[part] for part,coeff in components.items()),start=np.zeros((len(releases),3)))
        if additional_heads:
            for part in additional_heads.get(name,()):
                delta+=pieces[part]
        if stable_force_gains and name in stable_force_gains:
            k=stable_force_gains[name]
            if not 0. <= k <= 1.:
                raise ValueError(f'Invalid stable-force gain {k}')
            scale=np.minimum(k,np.tan(np.deg2rad(6.))/np.maximum(np.linalg.norm(force,axis=1),1e-15))
            delta+=scale[:,None]*force
        if bank is not None and filter_bank_gains and name in filter_bank_gains:
            delta+=filter_bank_gains[name]*bank
        if routed is not None and name=='candidate_deg':
            delta+=routed
        direction=UP+delta
        direction/=np.linalg.norm(direction,axis=1,keepdims=True)
        output[name]=frame.apply(np.vstack((UP,direction))[lookup])
    return output
