"""Test whether a stable observed specific-force direction identifies recoverable tilt error."""
import json
from collections import defaultdict
from itertools import chain

import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from filter_response_audit import phones
from long_state_source_audit import training_records
from native_catalog import Catalog


def diagnostic(raw):
    x,hz,reference,valid=raw
    observed=np.isfinite(x).all(1)
    ix=np.maximum.accumulate(np.where(observed,np.arange(len(x)),-1))
    fill=x[ix.clip(min=0)].copy();fill[ix<0]=0.
    a=np.ascontiguousarray(fill[:,:3]);g=np.ascontiguousarray(fill[:,3:])
    norm=np.linalg.norm(a,axis=1)
    state=VQF(1./hz).updateBatch(g,a)
    rot=Rotation.from_quat(state['quat6D'],scalar_first=True)
    direction=rot.apply(a/np.maximum(norm[:,None],1e-12))
    alpha=-np.expm1(-1./(hz*.25))
    known=np.flatnonzero(observed)
    causal=lfilter([alpha],[1.,alpha-1.],direction[known],axis=0)
    last=np.cumsum(observed)-1
    mean=causal[last.clip(min=0)].copy();mean[last<0]=0.
    length=np.linalg.norm(mean,axis=1)
    spread=np.rad2deg(np.sqrt(2.*np.maximum(0.,1.-length)))
    mean/=np.maximum(length[:,None],1e-12)
    truth=rot.apply(reference)
    target_angle=np.arctan2(np.linalg.norm(truth[:,:2],axis=1),truth[:,2])
    direct_angle=np.arctan2(np.linalg.norm(np.cross(mean,truth),axis=1),(mean*truth).sum(1))
    target=truth[:,:2]*(target_angle/np.maximum(np.linalg.norm(truth[:,:2],axis=1),1e-12))[:,None]
    correction=mean[:,:2]
    speed=np.linalg.norm(g,axis=1)
    base=valid & observed & (np.arange(len(x))/hz>=10.) & (length>.01)
    rotating=(speed>=.5) & (speed<2.)
    near=np.abs(norm/9.80665-1.)<.05
    tests={'all':base,'turntable_speed_near_g':base&rotating&near,
           'stable_force_direction':base&(spread<3.) & ~state['restDetected'],
           'near_g_stable_rotating':base & rotating & near & (spread<3.),
           'unstable_force_direction_motion':base & (spread>=3.) & ~state['restDetected']}
    result={}
    for key,m in tests.items():
        if m.sum()<100:continue
        src=correction[m];y=target[m]
        result[key]={'frames':int(m.sum()),'vqf_rmse_deg':float(np.degrees(np.sqrt(np.mean(target_angle[m]**2)))),
                     'force_direction_rmse_deg':float(np.degrees(np.sqrt(np.mean(direct_angle[m]**2)))),
                     'xx':float((src*src).sum(1).mean()),'xy':float((src*y).sum(1).mean()),'yy':float((y*y).sum(1).mean())}
    return result


def own_development():
    cat=Catalog()
    for (d,n),r in cat.rows.items():
        if d=='SHICE' and r['side']=='development':
            raw,_=cat.native(d,n)
            yield 'SHICE_development',n,raw


def main():
    stats=defaultdict(list)
    for d,n,raw in chain(training_records(),phones(),own_development()):
        stats[d].append({'record':n,'statistics':diagnostic(raw)})
    summary={}
    for d,rows in stats.items():
        groups={}
        for key in sorted(set().union(*(r['statistics'] for r in rows))):
            vals=[r['statistics'][key] for r in rows if key in r['statistics']]
            a={k:float(np.mean([r[k] for r in vals])) for k in ('vqf_rmse_deg','force_direction_rmse_deg','xx','xy','yy')}
            groups[key]={'records_with_100_frames':len(vals),'frames':sum(r['frames'] for r in vals),
                  'vqf_rmse_deg':a['vqf_rmse_deg'],'force_direction_rmse_deg':a['force_direction_rmse_deg'],
                  'signed_direction_cosine':a['xy']/max(1e-16,np.sqrt(a['xx']*a['yy'])),
                  'local_projection_scale':a['xy']/max(1e-16,a['xx']),
                  'positive_projection_records':sum(r['xy']>0 for r in vals)}
        summary[d]=groups
    out={'scope':'Descriptive tangent alignment with reference, no parameter fit. All 63 primary external sources, '
                 'three EuRoC Room 1 and four RIDI train recordings; 37 SHICE development records are identified separately. '
                 'No sealed recording opened. 0.25 s one-sided averaging in the VQF world frame, force spread under '
                 '3 degrees and |norm(a)/g-1| below .05 are defined independently of candidate scores. Means weight records equally.',
         'summary':summary,'records':stats}
    (ROOT/'review/gravity_direction_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d,v in summary.items():print(d,{k:round(z['signed_direction_cosine'],3) for k,z in v.items()},flush=True)


if __name__=='__main__':
    main()
