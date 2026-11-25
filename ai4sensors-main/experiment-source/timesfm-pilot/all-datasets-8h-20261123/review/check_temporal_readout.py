"""Rebuild C018 training losses from its saved states and test both heads on raw development records."""
import hashlib
import json

import numpy as np
import torch

from check_sources import ROOT,TRANSFER,weights
from check_training_batch import csv_rows,read,digest
from group_uncertainty import paired_by_group
from mixture_replay import replay
from native_catalog import Catalog
from optical_score import metric
from temporal_readout import load_readout


def loss(chosen,params,z,up,basis,enable,center,y):
    zz=z[chosen]
    body_up=up[chosen]
    vec=.1*(zz@params[:72].reshape(3,24).T+params[72:])
    tangent=vec-np.sum(vec*body_up,axis=1)[:,None]*body_up
    tangent*=np.minimum(1.,np.tan(np.deg2rad(6.))/np.maximum(np.linalg.norm(tangent,axis=1),1e-8))[:,None]
    world=np.einsum('nij,nj->ni',basis[chosen],tangent)
    world[:,2]=0.
    estimated=center[chosen]+.125*world*enable[chosen,None]
    estimated/=np.linalg.norm(estimated,axis=1,keepdims=True)
    return np.sum((estimated-y[chosen])**2,axis=1)


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C018'
    lock=read(batch/'pair-lock.json')
    c17=read(ROOT/'C017/coefficient-lock.json')
    assert digest(ROOT/'C017/coefficient-lock.json')==lock['C017_coefficient_lock_sha256']
    heads={'B6':(weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),.25,'always'),
           'B7':(weights(ROOT/'C013/seed-C006-readout.pt','mamba_v1'),.25,'rest_off')}
    training={}
    with np.load(batch/'training-inputs.npz') as f, np.load(ROOT/'C015/sampling.npz') as old:
        for new,previous in [('native_release','C015_native_release'),('source_record_ordinal','C015_record_ordinal'),
                             ('global_window','C015_global_window')]:
            np.testing.assert_array_equal(f[new],old[previous].ravel())
        pooled,up,basis,center,y=(f[k] for k in ('pooled','canonical_up','world_basis','center','truth'))
        enabled=f['valid'] & ~f['rest']
        assert len(pooled)==15360 and len(np.unique(f['source_record_ordinal']))==67
        for arm_id,arm in enumerate(('A','B')):
            path=batch/f'{arm}-readout.pt'
            assert digest(path)==lock['readout_sha256'][arm]
            cp=torch.load(path,map_location='cpu',weights_only=True)
            raw=pooled[:,arm_id].astype('f8')
            mu=cp['feature_mean'].numpy();scale=cp['feature_std'].numpy()
            np.testing.assert_array_equal(mu,raw.mean(axis=0).astype('f4'))
            np.testing.assert_array_equal(scale,np.maximum(1e-4,raw.std(axis=0)).astype('f4'))
            z=(raw-mu)/scale
            h=[json.loads(r) for r in (batch/f'{arm}-steps.jsonl').read_text(encoding='utf8').splitlines() if r]
            assert [r['step'] for r in h]==list(range(1,161))
            previous=np.zeros(75)
            largest=0.
            for j,r in enumerate(h):
                before,after=(np.asarray(r[k]) for k in ('weights_before','weights_after'))
                np.testing.assert_array_equal(previous,before)
                assert np.linalg.norm(after-before)>0.
                subset=slice(j*96,(j+1)*96)
                for params,key in [(before,'mean_chord_loss'),(after,'mean_chord_loss_after')]:
                    calculated=loss(subset,params,z,up,basis,enabled,center,y).mean()
                    largest=max(largest,abs(calculated-r[key]))
                previous=after
            assert largest<1e-7,(arm,largest)
            state=cp['state_dict']
            np.testing.assert_array_equal(previous,np.r_[state['weight'].numpy().ravel(),state['bias'].numpy()])
            model,gain=load_readout(heads['B6'][0],path)
            heads[arm]=(model,gain,'rest_off')
            training[arm]={'steps':len(h),'paired_before_and_after_losses':2*len(h),
                           'max_abs_loss_error':float(largest),'checkpoint_sha256':digest(path)}
    mix={'B6':.5,'B7':.5}
    groups={k:mix for k in ('C017_deg','A_deg','B_deg')}
    force=dict.fromkeys(groups,c17['k'])
    extras={'A_deg':['A'],'B_deg':['B']}
    rows=csv_rows(batch/'sequences.csv')
    old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C017/sequences.csv')}
    assert set(old)=={(r['dataset'],r['record']) for r in rows} and len(rows)==213
    wanted={('SHICE','R1-1'),('TUM-VI','room2'),('IMU-Array','calibration_trial/ImuArray_1_1_IMU'),
            ('OxIOD','running/data1/syn/imu1.csv')}
    catalog=Catalog()
    raw_replays=[]
    for r in rows:
        key=r['dataset'],r['record']
        assert r['status']=='scored'
        for field in ('group','scoring_indices_sha256','scored_frames'):
            assert r[field]==old[key][field]
        for field in ('full_vqf_deg','C017_deg'):
            assert abs(float(r[field])-float(old[key][field]))<1e-5
        if key in wanted:
            (raw,hz,truth,mask),_=catalog.native(*key)
            idx=np.flatnonzero(mask & np.isfinite(raw).all(1))
            assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
            pred=replay(raw,hz,heads,groups,idx,force,extras)
            scores={k:metric(v,truth[idx]) for k,v in pred.items()}
            gap=max(abs(float(r[k])-v) for k,v in scores.items())
            assert gap<1e-5,(key,gap)
            raw_replays.append({'dataset':key[0],'record':key[1],'scores':scores,'max_abs_gap_deg':gap})
    assert len(raw_replays)==len(wanted)
    out={'pair_lock_sha256':digest(batch/'pair-lock.json'),'two_arms':training,'same_external_slots':15360,
         'same_development_rows':213,'raw_replays':raw_replays,'results_sha256':digest(batch/'results.json'),
         'groups':{a:{d:paired_by_group([r for r in rows if r['dataset']==d],'full_vqf_deg',a)
                       for d in sorted({r['dataset'] for r in rows})} for a in ('A_deg','B_deg')}}
    (ROOT/'review/C018_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('loss gaps',[r['max_abs_loss_error'] for r in training.values()],'max native deg',max(r['max_abs_gap_deg'] for r in raw_replays))


if __name__=='__main__':
    main()
