"""Source-only test of whether disagreements among the three available causal estimates predict fast/slow preference."""
import json

import numpy as np

from check_sources import ROOT
from external_response_teacher import teacher
from lock_from_frontier import digest


def observations(fast,slow):
    # Norms and their included angle use only the capped, earth-horizontal vectors available to the deployed mixer.
    a=np.linalg.norm(fast,axis=1);b=np.linalg.norm(slow,axis=1)
    fa=np.degrees(np.arctan(a));sa=np.degrees(np.arctan(b))
    f=fast+[0.,0.,1.];s=slow+[0.,0.,1.]
    fs=np.degrees(np.arctan2(np.linalg.norm(np.cross(f,s),axis=1),np.einsum('ij,ij->i',f,s)))
    return {'fast_separation_deg':fa,'slow_separation_deg':sa,'slow_minus_fast_separation_deg':sa-fa,
            'fast_slow_disagreement_deg':fs,
            'signed_inner_product_deg2':np.sum(fast*slow,axis=1)*(180./np.pi)**2}


def auc(x,labels,weights):
    # Weighted pairwise ordering with exact ties given half credit.
    x=np.asarray(x)
    _,inverse=np.unique(x,return_inverse=True)
    plus=np.bincount(inverse,weights=weights*(labels==0),minlength=inverse.max()+1)
    minus=np.bincount(inverse,weights=weights*(labels==1),minlength=inverse.max()+1)
    if min(plus.sum(),minus.sum())<1e-12:return None
    return float(plus@(np.cumsum(minus)-minus/2.)/(plus.sum()*minus.sum()))


def main():
    path=ROOT/'C026/training-inputs.npz'
    with np.load(path)as inp:
        a={k:inp[k].reshape((-1,)+inp[k].shape[2:]) for k in ('center','fast','slow','truth','enabled','domain','condition','source_record_ordinal','features')}
    win,gap,_=teacher(*[a[k] for k in ('center','fast','slow','truth','enabled')])
    obs=observations(a['fast'],a['slow'])
    for i,n in enumerate(('current_gyro_rms','current_force_norm','current_force_spread','current_gyro_bias')):
        obs[n]=a['features'][:,24+i]
    source=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    record=a['source_record_ordinal']
    groups={};coverage={}
    for d in sorted(set(source)):
        sels=(source==d)&(win<2)&(gap>0)
        rids=np.unique(record[sels])
        weight=np.zeros_like(gap)
        for r in rids:
            ix=sels&(record==r)
            weight[ix]=gap[ix]/gap[ix].sum()/len(rids)
        coverage[d]={'median_active_bias_z':float(np.median(obs['current_gyro_bias'][sels])),
                     'median_active_separation_deg':float(np.median(obs['fast_slow_disagreement_deg'][sels])),
                     'median_bias_z_by_class':[float(np.median(obs['current_gyro_bias'][sels&(win==i)]))
                                              if np.any(sels&(win==i)) else None for i in (0,1)]}
        info={}
        for name,x in obs.items():
            within=[auc(x[sels&(record==r)],win[sels&(record==r)],weight[sels&(record==r)])for r in rids]
            valid=[v for v in within if v is not None]
            info[name]={'pooled_fast_above_slow_auc':auc(x[sels],win[sels],weight[sels]),
                        'mean_within_record_auc':float(np.mean(valid))if valid else None,
                        'records_with_both_choices':len(valid)}
        groups[d]=info
    out={'scope':'Only existing external training slots from 148 unique acquisitions. Fast/slow teacher preference is measured from the external reference. Each recording carries equal mass within a source or condition. Values above 0.5 mean a higher observed value orders a fast-favoring release above a slow-favoring one; this is a feature association, not prediction accuracy. No parameter search, target reference or fitted model.',
         'source_sha256':digest(path),'groups':groups,'active_record_conditions':coverage}
    (ROOT/'review/response_disagreement.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d,col in groups.items():
        print(d,{k:round(v['pooled_fast_above_slow_auc'],2) if v['pooled_fast_above_slow_auc'] is not None else None for k,v in col.items()})


if __name__=='__main__':main()
