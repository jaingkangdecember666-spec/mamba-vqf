"""Check the single C022 scalar on independently reconstructed external slots and on raw development sensors."""
import csv
import json

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from check_sources import ROOT
from lock_from_frontier import read,inference,digest
from locked_replay import Predictor
from native_catalog import Catalog


def scores(b,t,v,w,domain,k):
    p=b+k*v
    p/=np.linalg.norm(p,axis=1,keepdims=True)
    loss=((p-t)**2).sum(1)
    return {d:float(w[sel]@loss[sel])for d,sel in domain.items()}


def main():
    torch.set_num_threads(2)
    b=ROOT/'C022';lock=read(b/'coefficient-lock.json')
    assert lock['fit_count']==1 and digest(b/'training-slots.npz')==lock['training_slots_sha256']
    authoritative=read(b/'external-selection.json')
    with np.load(ROOT/'review/auxiliary_filter_training_slots.npz') as ours,np.load(b/'training-slots.npz') as supplied:
        a={k:ours[k]for k in ours.files}
        for me,theirs in [('record_ordinal','source_record_ordinal'),('active','enabled'),('domain','domain'),
                            ('release','native_release'),('input_last','native_input_last'),('truth','truth'),('record_equal_weights','record_equal_weights')]:
            np.testing.assert_array_equal(a[me],supplied[theirs])
        gaps={k:float(np.abs(a[k]-supplied[k]).max())for k in ('center','fast','slow')}
        assert max(gaps.values())<2e-7,gaps
    mixture=(2.*a['fast']+a['slow'])/3.
    mixture[~a['active']]=0.
    domain={d:a['domain']==d for d in np.unique(a['domain'])}
    def direct(k):return scores(a['center'],a['truth'],mixture,a['record_equal_weights'],domain,k)
    zero=direct(0.)
    def ratio(k):return {d:v/zero[d]for d,v in direct(k).items()}
    def objective(k):return max(ratio(k).values())
    grid=np.linspace(0.,1.,129)
    i=int(np.argmin([objective(k)for k in grid]))
    approx=minimize_scalar(objective,bounds=(grid[max(0,i-1)],grid[min(128,i+1)]),method='bounded',options={'xatol':1e-12})
    chosen=min((0.,1.,float(approx.x)),key=objective)
    k=lock['k']
    assert abs(chosen-k)<3e-5
    ratios=ratio(k)
    for d in domain:
        np.testing.assert_allclose(direct(k)[d],authoritative['domains'][d]['C022_loss_at_lock'],rtol=5e-7,atol=1e-12)
    assert max(ratios.values())<1.
    recipes={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C022','C022_deg'),recipes))
    assert model.bank['candidate_deg']==k
    raw=Catalog()
    with (b/'sequences.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    with (ROOT/'C020/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
        before={(r['dataset'],r['record']):r for r in csv.DictReader(f)}
    assert len(rows)==len(before)==213
    samples={('SHICE','R1-1'),('TUM-VI','room2'),('IMU-Array','calibration_trial/ImuArray_1_1_IMU'),
             ('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    actual=[]
    for r in rows:
        key=r['dataset'],r['record']
        previous=before[key]
        assert r['status']=='scored' and r['group']==previous['group'] and r['scored_frames']==previous['scored_frames']
        assert r['scoring_indices_sha256']==previous['scoring_indices_sha256']
        assert max(abs(float(r[c])-float(previous[c]))for c in ('full_vqf_deg','C020_deg'))<1e-5
        if key not in samples:continue
        data,_=raw.native(*key)
        score=model.score(data)
        assert score['scoring_indices_sha256']==r['scoring_indices_sha256']
        err=max(abs(score[k]-float(r[col]))for k,col in [('full_vqf_deg','full_vqf_deg'),('candidate_deg','C022_deg'),('physical_deg','physical_only_deg')])
        assert err<1e-5,(key,err)
        actual.append({'dataset':key[0],'record':key[1],'candidate_deg':score['candidate_deg'],'physical_deg':score['physical_deg'],'max_abs_gap_deg':err})
    assert len(actual)==len(samples)
    report={'coefficient_lock_sha256':digest(b/'coefficient-lock.json'),'independent_external_slots':len(a['center']),
            'external_records':len(np.unique(a['record_ordinal'])),'source_array_max_abs_differences':gaps,
            'coefficient':k,'independent_coefficient':chosen,'coefficient_gap':abs(k-chosen),'locked_source_loss_ratios':ratios,
            'same_mask_development_records':len(rows),'raw_replays':actual}
    (ROOT/'review/C022_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('external k diff',abs(k-chosen),'raw maximum gap',max(r['max_abs_gap_deg']for r in actual),ratios)


if __name__=='__main__':
    main()

