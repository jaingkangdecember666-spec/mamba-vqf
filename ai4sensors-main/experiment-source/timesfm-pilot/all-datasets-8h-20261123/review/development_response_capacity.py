"""Diagnose response-selection errors on a name-selected development panel. Hindsight is never deployable."""
import csv
import json
from collections import defaultdict
from math import sqrt

import numpy as np
import torch

from check_sources import ROOT
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_blocks import block_inputs
from native_catalog import Catalog
from optical_score import canonical_frame
from response_router import release_alternatives
from router_motion_context import NAMES,series
from source_motion_router_diagnostic import native


def picks(rows):
    groups=defaultdict(list)
    for r in rows:
        d=r['dataset']
        part=r['group'] if d in ('SHICE','OxIOD','TUM-VI') else r['record'].split('/')[0] if d=='RepoIMU' else 'first'
        groups[d,part].append(r)
    return [min(g,key=lambda r:r['record']) for _,g in sorted(groups.items())]


def diagnose(raw,row,center,router):
    signal,hz,truth,mask=raw
    ids=np.flatnonzero(mask&np.isfinite(signal).all(1))
    x,_=native(raw);q,win,cuts,release=block_inputs(x,hz)
    assert len(release)
    expected=center.predict(signal,hz,release)['candidate_deg']
    c=q[release].apply(expected)
    c/=c[:,2:3]
    f,s=release_alternatives(x,hz,q,cuts)
    p,valid=router.probabilities(win)
    active=valid&~win[:,-1,12].astype(bool)
    f[~active]=0.;s[~active]=0.
    actions=np.stack((np.zeros_like(c),f,s,p[:,:1]*f+p[:,1:2]*s))
    m=actions+c
    m/=np.linalg.norm(m,axis=-1,keepdims=True)
    before=np.tile([0.,0.,1.],(len(m),1,1))
    dirs=np.concatenate((before,m),axis=1)
    loc=np.searchsorted(release,ids,side='right')
    refworld=q[ids].apply(truth[ids])
    loss=[]
    for each in dirs:
        vec=each[loc]
        theta=np.degrees(np.arctan2(np.linalg.norm(np.cross(vec,refworld),axis=1),np.sum(vec*refworld,axis=1)))
        loss.append(np.bincount(loc,weights=theta*theta,minlength=len(release)+1))
    loss=np.stack(loss)
    counts=np.bincount(loc,minlength=len(release)+1)
    macro=np.sqrt(loss.sum(1)/len(ids))
    assert abs(macro[0]-float(row['C020_deg']))<1e-5 and abs(macro[3]-float(row['C023_deg']))<1e-5
    best=np.argmin(loss[:3],axis=0)
    names=('C020','full_fast_tangent','full_slow_tangent','C023')
    active_group=np.r_[False,active]&(counts>0)
    fctx=series(x,hz)[cuts]
    stats={}
    for i,name in enumerate(names[:3]):
        sub=(best==i)&active_group
        k=sub[1:];mass=counts[sub].sum()
        stats[name]={'release_intervals':int(sub.sum()),'native_frames':int(mass),
                     'C023_p_fast_median':float(np.median(p[k,0])) if np.any(k) else None,
                     'C023_p_slow_median':float(np.median(p[k,1])) if np.any(k) else None,
                     'physical_observer_medians':dict(zip(NAMES,np.median(fctx[k],axis=0).tolist())) if np.any(k) else {}}
    return {'dataset':row['dataset'],'group':row['group'],'record':row['record'],'frames':len(ids),
            'common_index_sha256':row['scoring_indices_sha256'],'scored_active_intervals':int(active_group.sum()),
            'RMSE_deg':dict(zip(names,macro.tolist())),'per_interval_hindsight_deg':sqrt(float(loss[:3].min(axis=0).sum()/len(ids))),
            'best_discrete_choice_on_scored_active_intervals':stats}


def main():
    torch.set_num_threads(2)
    spec={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    base=Predictor(inference(('C020','C020_deg'),spec))
    routed=Predictor(inference(('C023','C023_deg'),spec))
    src=ROOT/'C023/sequences.csv'
    with src.open(encoding='utf-8-sig',newline='')as f:rows=picks(list(csv.DictReader(f)))
    db=Catalog();results=[]
    for row in rows:
        raw,_=db.native(row['dataset'],row['record'])
        results.append(diagnose(raw,row,base,routed.router))
    out={'scope':'First development record by name for each SHICE device, Oxford carry group, TUM trajectory, Repo apparatus, and each other domain. Each of three fixed outputs preserves the full C020 physical and learned center and active gate. The full fast/slow choices add their complete 6-degree-capped tangent. Hindsight picks one of three discrete corrections using future reference from the evaluation interval and is a diagnostic only. It is not an independent test, not a deployed method, and not entered in the development candidate ranking.',
         'source_sha256':digest(src),'panel_records':len(rows),'records':results}
    (ROOT/'review/development_response_capacity.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for r in results:print(r['dataset'],r['record'],[round(v,4)for v in r['RMSE_deg'].values()],round(r['per_interval_hindsight_deg'],4))


if __name__=='__main__':main()
