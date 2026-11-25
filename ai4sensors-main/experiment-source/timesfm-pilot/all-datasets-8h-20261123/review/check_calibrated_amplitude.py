"""Rebuild C032 calibration from original C022/C023/C026 observations and verify its identity with C030."""
import json
import math
from collections import defaultdict

import numpy as np
import torch

from anchored_response_features import features
from check_sources import ROOT
from check_training_batch import csv_rows
from lock_from_frontier import digest, read


def main():
    b=ROOT/'C032'
    with np.load(ROOT/'C022/training-slots.npz') as f:
        ds,ids=f['domain'],f['source_record_name']
        eligible=f['enabled'] & f['intrinsic_valid'] & (np.abs(np.linalg.norm(f['truth'],axis=1)-1.)<=2e-12)
        fast,slow=f['fast'],f['slow']
        releases=f['native_release'];last=f['native_input_last']
    with np.load(ROOT/'C026/training-inputs.npz') as f:
        z=f['features'][:,:96,:].reshape(-1,28)
    with np.load(ROOT/'C023/training-io.npz') as f:
        np.testing.assert_array_equal(z[:,:24],f['normalized_input'])
    logits=[]
    for name in ('C023','C027'):
        cp=torch.load(ROOT/name/'router.pt',weights_only=True,map_location='cpu')
        w=cp['state_dict']['weight'].numpy()
        bias=cp['state_dict']['bias'].numpy()+cp['log_p0'].numpy()[:2]
        logits.append(z[:,:24] @ (w[0]-w[1]) + bias[0]-bias[1])
    order=logits[0]-logits[1]
    anchor=torch.load(ROOT/'C030/router.pt',weights_only=True,map_location='cpu')
    correction=2*features(fast,slow,z[:,24:28]) @ anchor['a'].numpy()
    values=defaultdict(list)
    for idx in np.flatnonzero(eligible):
        values[str(ds[idx]),str(ids[idx])].append(int(idx))
    source=defaultdict(list)
    for key,indices in values.items():
        source[key[0]].append(indices)
    weight=np.zeros(len(ds))
    for records in source.values():
        for indices in records:
            weight[indices]=1./(len(source)*len(records)*len(indices))
    def rms(x):
        return math.sqrt(math.fsum(float(w*v*v) for w,v in zip(weight,x)))
    o,p=rms(order),rms(correction)
    gamma=min(1.,o/p) if p else 1.
    declared=read(b/'calibration.json')
    assert max(abs(o-declared['RMS_order_log_odds']),abs(p-declared['RMS_unscaled_C030_correction_log_odds']))<1e-12
    assert gamma==declared['gamma']==1. and declared['equivalent_to_C030']
    with np.load(b/'calibration-slots.npz') as f:
        for k,v in [('eligible',eligible),('weight',weight),('domain',ds),('source_record_name',ids),
                    ('native_release',releases),('native_input_last',last)]:
            np.testing.assert_array_equal(f[k],v)
        np.testing.assert_allclose(f['C023_minus_C027_log_odds'],order,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(f['C030_minus_C023_log_odds'],correction,rtol=1e-12,atol=1e-12)
    own=torch.load(b/'router.pt',weights_only=True,map_location='cpu')
    for k in ('a','context_mean','context_std','old_feature_mean','old_feature_std','p0','log_p0'):
        np.testing.assert_array_equal(own[k].numpy(),anchor[k].numpy())
    for k,v in anchor['state_dict'].items():np.testing.assert_array_equal(v.numpy(),own['state_dict'][k].numpy())
    old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C030/sequences.csv')}
    rows=csv_rows(b/'sequences.csv');assert len(old)==len(rows)==213
    gaps=[]
    for r in rows:
        other=old[r['dataset'],r['record']]
        for k in ('group','status','scoring_indices_sha256','scored_frames'): assert r[k]==other[k]
        gaps.append(abs(float(r['C032_deg'])-float(other['C030_deg'])))
    assert max(gaps)<1e-5
    result={'gamma':gamma,'weighted_RMS_order':o,'weighted_RMS_physical':p,'admitted_slots':int(eligible.sum()),
            'eligible_record_counts':{k:len(v) for k,v in source.items()},'original_slots':len(ds),
            'all_packed_neural_parameters_and_four_physical_coefficients_exact':True,'paired_development_records':len(rows),
            'max_candidate_vs_C030_difference_deg':max(gaps),'new_effective_policies':0,'whole_replay_of':'C030',
            'new_neural_optimization_steps':0,'recomputed_from_original_artifacts':True,'deployed_sha256':digest(b/'router.pt')}
    (ROOT/'review/C032_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result)


if __name__=='__main__':main()
