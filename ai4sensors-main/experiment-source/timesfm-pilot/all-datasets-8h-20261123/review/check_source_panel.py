"""Verify exactly which external trials entered C019 and independently reconstruct their optimizer contribution."""
import hashlib
import json
from collections import Counter

import numpy as np
import torch

from check_sources import ROOT
from check_temporal_readout import loss
from check_training_batch import csv_rows,read,digest
from locked_replay import Predictor
from lock_from_frontier import inference
from native_catalog import Catalog


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C019'
    lock=read(batch/'readout-lock.json')
    prep=read(batch/'training-preparation.json')
    panel=read(ROOT/'review/yt_training_panel.json')
    assert prep['panel_sha256']==digest(ROOT/'review/yt_training_panel.json')
    assert lock['C019_weight_sha256']==digest(batch/'readout.pt')
    cp=torch.load(batch/'readout.pt',weights_only=True,map_location='cpu')
    hist=[json.loads(x) for x in (batch/'optimizer-steps.jsonl').read_text(encoding='utf8').splitlines() if x]
    assert [r['step'] for r in hist]==list(range(1,161))
    with np.load(batch/'training-inputs.npz') as new,np.load(ROOT/'C018/training-inputs.npz') as old:
        add=new['replacement_YT'].astype(bool)
        retained=new['retained_original_YT'].astype(bool)
        yt=new['domain']=='YT_Server'
        np.testing.assert_array_equal(add|retained,yt)
        assert not (add&retained).any()
        assert (int(yt.sum()),int(retained.sum()),int(add.sum()),int((~yt).sum()))==(4685,1180,3505,10675)
        for field in ('pooled','canonical_up','world_basis','center','truth','valid','rest','native_release','native_input_last'):
            np.testing.assert_array_equal(new[field][~add],old[field][~add])
        for ordinal in np.unique(old['source_record_ordinal'][yt]):
            ix=np.flatnonzero(old['source_record_ordinal']==ordinal)
            np.testing.assert_array_equal(np.flatnonzero(retained & (old['source_record_ordinal']==ordinal)),ix[::4])
        names=new['source_record_name']
        assert set(names[yt])=={r['record'] for r in panel['records']}
        extra_counts=Counter(names[add])
        assert len(extra_counts)==72 and set(extra_counts.values())=={48,49}
        releases=new['native_release']
        assert len({(n,int(t)) for n,t in zip(names[add],releases[add])})==3505
        seq=new['pooled'][:,1].astype('f8')
        mu=cp['feature_mean'].numpy();std=cp['feature_std'].numpy()
        np.testing.assert_array_equal(mu,seq.mean(axis=0).astype('f4'))
        np.testing.assert_array_equal(std,np.maximum(1e-4,seq.std(axis=0)).astype('f4'))
        args=((seq-mu)/std,new['canonical_up'],new['world_basis'],new['valid']&~new['rest'],new['center'],new['truth'])
        max_loss=0.;directions=[];prev=np.zeros(75);new_nonzero=0
        for j,r in enumerate(hist):
            before=np.asarray(r['weights_before'])
            after=np.asarray(r['weights_after'])
            np.testing.assert_array_equal(before,prev)
            ids=np.arange(96*j,96*j+96)
            max_loss=max(max_loss,abs(loss(ids,before,*args).mean()-r['loss']),
                         abs(loss(ids,after,*args).mean()-r['post_step_loss']))
            subgroup=r['YT_gradient_breakdown']['new_72']
            indices=ids[add[ids]]
            assert len(indices)==subgroup['slots']
            g=np.asarray(subgroup['grad_before_global_clip']);norm=np.linalg.norm(g)
            new_nonzero+=norm>0.
            if j in (0,79,159) and norm>0.:
                direction=g/norm;eps=1e-5
                hi=loss(indices,before+eps*direction,*args).sum()/96
                lo=loss(indices,before-eps*direction,*args).sum()/96
                numeric=(hi-lo)/(2.*eps)
                assert np.isclose(numeric,norm,rtol=2e-4,atol=1e-8),(j,numeric,norm)
                directions.append({'step':j+1,'new_72_slots':len(indices),'logged_grad_l2':float(norm),'finite_difference':float(numeric)})
            prev=after
        params=cp['state_dict']
        np.testing.assert_array_equal(prev,np.r_[params['weight'].numpy().ravel(),params['bias'].numpy()])
        assert max_loss<1e-7 and len(directions)==3
    oldrows={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C018/sequences.csv')}
    rows=csv_rows(batch/'sequences.csv')
    assert len(rows)==213 and set(oldrows)=={(r['dataset'],r['record']) for r in rows}
    index={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C019','C019_deg'),index))
    catalog=Catalog()
    rawchecks=[]
    wanted={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('YT_Server','hdf_F2_server_20_5')}
    for r in rows:
        key=r['dataset'],r['record'];old=oldrows[key]
        assert r['status']=='scored' and r['scored_frames']==old['scored_frames'] and r['scoring_indices_sha256']==old['scoring_indices_sha256']
        assert abs(float(r['full_vqf_deg'])-float(old['full_vqf_deg']))<1e-5
        if key in wanted:
            raw,_=catalog.native(*key);scores=model.score(raw)
            gap=max(abs(scores['full_vqf_deg']-float(r['full_vqf_deg'])),abs(scores['candidate_deg']-float(r['C019_deg'])))
            assert scores['scoring_indices_sha256']==r['scoring_indices_sha256'] and gap<1e-5
            rawchecks.append({'dataset':key[0],'record':key[1],'max_abs_gap_deg':gap})
    assert len(rawchecks)==len(wanted)
    audit={'same_non_YT_slots':10675,'same_YT_slots':1180,'new_YT_slots':3505,'new_trials':72,
           'new_trials_slot_range':[48,49],'new_unique_trial_releases':3505,'same_records_development':213,
           'true_new_source_gradient_steps':int(new_nonzero),'finite_difference_checks':directions,
           'all_320_step_loss_max_abs_gap':max_loss,'raw_replays':rawchecks,'results_sha256':digest(batch/'results.json')}
    (ROOT/'review/C019_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print('source-only checks',int(new_nonzero),'step loss mismatch',max_loss,'raw gaps',[v['max_abs_gap_deg'] for v in rawchecks])


if __name__=='__main__':
    main()
