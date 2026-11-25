"""Verify C004's source identities, matched draws, checkpoint tensors, loss ledger and native scores."""
import csv
import hashlib
import json

import numpy as np
import torch
from check_sources import ROOT, TRANSFER, tum, weights, yt
from optical_score import comparison


def read(file):
    return json.loads(file.read_text('utf-8-sig'))


def rows(file):
    with file.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C004'
    inv=read(ROOT/'C001/inventory.json')
    proto=read(batch/'protocol.json');prep=read(batch/'training-preparation.json')['records']
    full=read(batch/'results.json')
    records=inv['records']
    locked={(x['dataset'],x['record']):x for x in records}
    assert len(prep)==63
    for ix,x in enumerate(prep):
        frozen=locked[(x['domain'],x['record'])]
        assert frozen['group']==x['group']
        assert frozen['side']==('external_training_candidate' if x['domain']=='YT_Server' else 'historical_source_train')
        assert (x['domain'],x['record'])==(proto['source_records'][ix]['domain'],proto['source_records'][ix]['record'])
    yt_fits={x['group'] for x in prep if x['domain']=='YT_Server'}
    assert len(yt_fits)==24 and all(x['domain'] in ('BROAD','Sassari','YT_Server') for x in prep)
    test={r['group'] for r in records if r['dataset']=='YT_Server' and r['official_split']=='test'}
    valid=set(inv['yt']['selected_groups']);assert not yt_fits&(test|valid)
    with np.load(batch/'schedules.npz') as draws:
        a,b,common,off=[draws[k] for k in ('A','B','shared_slots','record_offsets')]
        assert a.shape==b.shape==(160,96) and np.array_equal(a==b,common) and (common.sum(1)==64).all()
        for label,values in [('A',a),('B',b)]:
            ids=np.searchsorted(off[1:],values,side='right')
            count=np.bincount(ids.ravel(),minlength=len(prep))
            np.testing.assert_array_equal(count,[q[label+'_draws'] for q in prep])
            is_yt=np.array([x['domain']=='YT_Server' for x in prep])[ids]
            assert (is_yt.sum(1)==(0 if label=='A' else 32)).all()
    logs=rows(batch/'training-steps.csv')
    for label in ('A','B'):
        included=[x for x in logs if x['arm']==label]
        assert len(included)==160 and len(set(int(x['step']) for x in included))==160
        assert np.isfinite([[float(x['loss']),float(x['gradient_norm'])] for x in included]).all()
    base_file=TRANSFER/'batch6/mamba-intrinsic-selected.pt'
    base=torch.load(base_file,map_location='cpu',weights_only=True)
    verify={}
    model={'C003':(weights(base_file,'mamba_v1'),.25)}
    for label in ('A','B'):
        path=batch/(label+'-readout.pt')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        changed=[k for k,v in checkpoint['state_dict'].items() if not torch.equal(v,base['state_dict'][k])]
        assert set(changed)=={'output.bias','output.weight'}
        assert all(torch.equal(checkpoint[k],base[k]) for k in ('mean','std'))
        assert float(checkpoint['cap_deg'])==float(base['cap_deg'])
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        assert sha==full['weights'][label]['weights_sha256']
        verify[label]={'sha256':sha,'changed_layers':changed,'changed_parameters':sum(checkpoint['state_dict'][k].numel() for k in changed)}
        model[label]=(weights(path,'mamba_v1'),.25)
    prior={(r['dataset'],r['record']):r for r in rows(ROOT/'C003/sequences.csv')}
    observations=rows(batch/'sequences.csv')
    assert len(observations)==len(prior)==85
    anchors=[];domain={}
    for r in observations:
        d,name=r['dataset'],r['record'];previous=prior[d,name]
        assert r['scoring_indices_sha256']==previous['scoring_indices_sha256']
        assert float(r['c003_deg'])==float(previous['debiased_b6_gain_025_deg'])
        domain.setdefault(d,[]).append([float(r[k]) for k in ('full_vqf_deg','c003_deg','A_deg','B_deg')])
        raw=tum(int(name[-1])) if d=='TUM-VI' else yt(name) if d=='YT_Server' and name==inv['yt']['selected_valid'][0] else None
        if raw:
            actual=comparison(*raw,model)
            assert actual['mask_sha256']==r['scoring_indices_sha256']
            residual=max(abs(actual['rmse_deg'][k]-float(r[q])) for k,q in [('full_vqf','full_vqf_deg'),('C003','c003_deg'),('A','A_deg'),('B','B_deg')])
            assert residual<1e-5
            anchors.append({'dataset':d,'name':name,'rmse':actual['rmse_deg'],'max_native_replay_delta_deg':residual})
    out={'sha256':hashlib.sha256((batch/'results.json').read_bytes()).hexdigest(),'networks':verify,'training_records':len(prep),
         'yt_fit_groups':len(yt_fits),'test_groups':len(test),'reserved_validation_groups':len(valid),'group_overlap':0,
         'matching_draws_per_step':64,'steps_per_arm':160,'evaluated_records':len(observations),
         'mean_order':['FullVQF','C003','A','B'],'means':{k:np.mean(v,axis=0).tolist() for k,v in domain.items()},'raw_replay':anchors}
    (ROOT/'review/C004_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(verify,anchors)


if __name__=='__main__':main()

