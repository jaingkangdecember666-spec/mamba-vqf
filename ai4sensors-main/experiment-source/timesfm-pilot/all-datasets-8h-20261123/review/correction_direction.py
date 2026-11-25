"""Measure whether B6's tangent update points toward ground truth at zero gain, on development data only."""
import json
from collections import defaultdict
import numpy as np
import torch
from check_sources import ROOT, TRANSFER, tum, yt
from replay_native import UP, canonical_frame, external_case, features, weights


@torch.inference_mode()
def characterize(source,model):
    signal,hz,ref,mask=source
    valid=np.isfinite(signal).all(axis=1)
    prev=np.maximum.accumulate(np.where(valid,np.arange(len(valid)),-1))
    x=signal[prev.clip(min=0)].copy()
    x[prev<0]=0.
    q,win,last,release=features(x,hz)
    body,bases,observable=canonical_frame(win)
    head,mean,std,cap=model
    pieces=[head(((torch.from_numpy(body[k:k+128])-mean)/std).clamp(-12,12)).numpy() for k in range(0,len(body),128)]
    delta=np.einsum('nij,nj->ni',bases,np.concatenate(pieces).astype(float))*.1
    delta[~observable]=0.
    u=q[last].inv().apply(UP)
    delta-=np.sum(delta*u,axis=1)[:,None]*u
    delta*=np.minimum(1,np.tan(np.deg2rad(cap))/np.maximum(1e-12,np.linalg.norm(delta,axis=1,keepdims=True)))
    ids=np.flatnonzero(mask&valid)
    true=q[ids].apply(ref[ids])
    index=np.searchsorted(release,ids,side='right')
    dw=np.vstack((np.zeros((1,3)),q[last].apply(delta)))[index]
    horizontal=true-true[:,2,None]*UP
    sin=np.linalg.norm(horizontal,axis=1)
    theta=np.arctan2(sin,true[:,2])
    residual=horizontal*(theta/np.maximum(sin,1e-12))[:,None]
    cov=np.sum(residual*dw,axis=1)
    power=np.einsum('ij,ij->i',dw,dw)
    rmse=np.sqrt(np.mean(theta**2))
    d=np.rad2deg(-cov.mean()/rmse)
    a=np.rad2deg(power.mean()/rmse)
    return {'frames':len(ids),'baseline_deg':float(np.rad2deg(rmse)),
            'derivative_deg_per_unit_gain_at_zero':float(d),'quadratic_curvature_scale':float(a),
            'tangent_update_rms_deg':float(np.rad2deg(np.sqrt(power.mean()))),
            'mean_projection_rad2':float(cov.mean()),'mean_update_power_rad2':float(power.mean())}


def main():
    torch.set_num_threads(2)
    model=weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1')
    sources=[(f'TUM-VI',f'room{n}',lambda n=n:tum(n)) for n in (1,2)]
    split=json.loads((TRANSFER/'external-split.json').read_text('utf8'))
    sources += [(r['domain'],r['name'],lambda r=r:external_case(r)) for r in split['records'] if r['split']=='validation']
    inv=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    sources += [('YT_Server',n,lambda n=n:yt(n)) for n in inv['yt']['selected_valid']]
    records=defaultdict(list)
    for domain,name,source in sources:
        records[domain].append({'name':name,**characterize(source(),model)})
    summary={}
    for d,items in records.items():
        slope=np.mean([r['derivative_deg_per_unit_gain_at_zero'] for r in items]); curve=np.mean([r['quadratic_curvature_scale'] for r in items]);
        summary[d]={'sequences':len(items),'macro_slope_at_zero_deg':float(slope),
                    'small_angle_minimum_gain_approx':float(-slope/curve) if curve else None,
                    'negative_directional_slopes':sum(r['derivative_deg_per_unit_gain_at_zero']<0 for r in items)}
    (ROOT/'review/development_direction.json').write_text(json.dumps({'summary':summary,'records':records},indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
