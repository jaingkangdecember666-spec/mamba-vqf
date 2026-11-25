"""Quantify source routing and development changes caused solely by rescheduling the same 160 updates."""
import csv
import json

import numpy as np

from check_sources import ROOT
from lock_from_frontier import digest


def trajectory(batch):
    with (ROOT/batch/'optimizer-steps.jsonl').open(encoding='utf8')as f:
        return np.array([json.loads(s)['params_after']for s in f if s.strip()])


def main():
    pa,pb=[ROOT/b/'training-io.npz'for b in ('C023','C027')]
    with np.load(pa)as fa,np.load(pb)as fb,np.load(ROOT/'C022/training-slots.npz')as source:
        old,new=fa['prob_trained'],fb['prob_trained']
        domains,active,baseweight=source['domain'],source['enabled'].astype(bool),source['record_equal_weights']
        records=source['source_record_ordinal']
    by_source={}
    for d in sorted(set(domains)):
        sel=(domains==d)&active
        idx=np.flatnonzero(sel);w=baseweight[idx];w/=w.sum()
        a,b=old[idx],new[idx]
        by_source[d]={'active_slots':len(idx),'active_record_coverage':int(len(np.unique(records[idx]))),
                      'mean_absolute_probability_difference':(w@np.abs(a-b)).tolist(),
                      'different_most_likely_action_fraction':float(w@(np.argmax(a,axis=1)!=np.argmax(b,axis=1)))}
    ds=list(csv.DictReader((ROOT/'C027/sequences.csv').open(encoding='utf-8-sig',newline='')))
    targets={}
    for d in sorted({r['dataset']for r in ds}):
        r=[x for x in ds if x['dataset']==d]
        val=lambda field:np.array([float(x[field]) for x in r])
        a,b,base=map(val,('C023_deg','C027_deg','full_vqf_deg'))
        delta=b-a;large=np.argmax(np.abs(delta))
        targets[d]={'count':len(r),'better_with_permuted_order':int((delta<0).sum()),
                    'advantage_against_default_changes_sign':int(((a<base)!=(b<base)).sum()),
                    'mean_order_effect_deg':float(delta.mean()),'largest_abs_order_effect_deg':float(delta[large]),
                    'largest_effect_record':r[large]['record']}
    trajectories=[trajectory(n)for n in ('C023','C027')]
    assert all(s.shape==(160,50)for s in trajectories)
    stats={}
    for name,v in zip(('C023','C027'),trajectories):
        stats[name]={'last40_distance_end_to_mean':float(np.linalg.norm(v[-1]-v[-40:].mean(0))),
                     'last40_rms_distance_to_end':float(np.sqrt(((v[-40:]-v[-1])**2).sum(1).mean())),
                     'weight_l2_end':float(np.linalg.norm(v[-1]))}
    out={'scope':'Only the two completed, predeclared fits. Source weighted action probabilities refer to the same native slots. Development rows are descriptive and have influenced selection.',
         'C023_probability_sha256':digest(pa),'C027_probability_sha256':digest(pb),'same_213_panel_sha256':digest(ROOT/'C027/sequences.csv'),
         'final_weight_difference_l2':float(np.linalg.norm(trajectories[0][-1]-trajectories[1][-1])),
         'last_update_variability':stats,'external_active_slots':by_source,'development':targets}
    (ROOT/'review/order_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('parameter L2',out['final_weight_difference_l2'])
    for d,g in targets.items():print(d,g['advantage_against_default_changes_sign'],g['count'],g['largest_abs_order_effect_deg'])


if __name__=='__main__':main()
