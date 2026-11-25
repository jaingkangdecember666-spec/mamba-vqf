"""Verify the seed change at the two operating points with the same 213-record development panel."""
import hashlib
import json

import numpy as np
import torch

from check_gates import raw_gates
from check_sources import ROOT, TRANSFER, weights
from check_training_batch import csv_rows, digest, read
from group_uncertainty import paired_by_group
from native_catalog import Catalog


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C013'
    b6=TRANSFER/'batch6/mamba-intrinsic-selected.pt'
    b7=TRANSFER/'batch7/mamba-intrinsic-seed314159-selected.pt'
    b7fit=batch/'seed-C006-readout.pt'
    cps=[torch.load(p,map_location='cpu',weights_only=True) for p in [b6,b7,b7fit]]
    assert all(torch.equal(cps[0][k],cp[k]) for k in ('mean','std') for cp in cps)
    model0,model1,fitted=[x['state_dict'] for x in cps]
    changed=[k for k in model1 if not torch.equal(model1[k],model0[k])]
    assert 'sequence.A_log' in changed and 'input.weight' in changed
    assert {k for k in model1 if not torch.equal(model1[k],fitted[k])}=={'output.weight','output.bias'}
    hist6,hist7=[read(TRANSFER/n/'protocol.json') for n in ('batch6','batch7')]
    assert hist6['training_schedule_sha256']==hist7['batch_schedule_sha256']
    assert hist7['initialization']['batch6_reconstructed_sha256']!=hist7['initialization']['batch7_sha256']
    assert hist7['initialization_seed']==314159 and hist7['sampling_seed']==1729
    with np.load(batch/'training-windows.npz') as used,np.load(ROOT/'C004/schedules.npz') as src,np.load(ROOT/'C012/training-windows.npz') as previous:
        draw=src['B']
        np.testing.assert_array_equal(used['original_B_schedule'],draw)
        np.testing.assert_array_equal(used['native_release'],src['release_indices'][draw])
        np.testing.assert_array_equal(used['C004_point_targets'],previous['C004_points'][:,:,-3:])
        assert not np.array_equal(used['B7_latent_and_up'][:,:,:-3],previous['C004_points'][:,:,:-6])
    tr=read(batch/'training.json')
    for k,h in tr['artifacts_sha256'].items():assert digest(batch/k)==h
    assert tr['fit_count']==1 and tr['seed_C003_optimizer_steps']==0
    steps=csv_rows(batch/'training-steps.csv')
    assert [int(x['step']) for x in steps]==list(range(1,161))
    old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C006/sequences.csv')}
    rows=csv_rows(batch/'sequences.csv')
    assert len(rows)==213 and set(old)=={(r['dataset'],r['record']) for r in rows}
    for r in rows:
        initial=old[r['dataset'],r['record']]
        assert r['status']=='scored' and not r.get('error')
        assert r['scoring_indices_sha256']==initial['scoring_indices_sha256'] and r['scored_frames']==initial['scored_frames']
        for a,b in [('full_vqf_deg','full_vqf_deg'),('c003_deg','c003_deg'),('C006_deg','B_rest_gated_deg')]:
            assert abs(float(r[a])-float(initial[b]))<1e-6
    cat=Catalog()
    anchors={('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('YT_Server',min(n for d,n in old if d=='YT_Server'))}
    configs=[('seed_C003_deg',weights(b7,'mamba_v1'),'fixed'),('seed_C006_deg',weights(b7fit,'mamba_v1'),'rest')]
    checks=[]
    for r in rows:
        key=r['dataset'],r['record']
        if key not in anchors:continue
        raw,_=cat.native(*key)
        for col,model,policy in configs:
            ids,_,m=raw_gates(raw,model)
            assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
            diff=abs(m[policy]-float(r[col]))
            assert diff<1e-5
            checks.append({'dataset':key[0],'record':key[1],'column':col,'recalculated_deg':m[policy],'absolute_gap_deg':diff})
    out={'external_sampling_seed':1729,'network_initializations':[1729,314159],'historical_epochs':[6,5],
         'same_source_draws':int(draw.size),'shared_native_panel':len(rows),'two_new_seed_paths':[digest(b7),digest(b7fit)],
         'historical_schedule_sha256':hist7['batch_schedule_sha256'],'changed_tensors_between_trained_seeds':changed,
         'changed_tensors_on_new_fit':['output.weight','output.bias'],'native_replays':checks,
         'domains':{col:{d:paired_by_group([r for r in rows if r['dataset']==d],'full_vqf_deg',col)
                         for d in sorted({r['dataset'] for r in rows})} for col,_,_ in configs}}
    (ROOT/'review/C013_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(len(rows),len(checks),max(r['absolute_gap_deg'] for r in checks))


if __name__=='__main__':
    main()
