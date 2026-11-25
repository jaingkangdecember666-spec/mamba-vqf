"""Verify C031 directly against both optimization trajectories and its fixed external input panel."""
import json
import math

import numpy as np
import torch

from check_history_router import prediction
from check_sources import ROOT
from check_training_batch import csv_rows
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog


def main():
    torch.set_num_threads(2);b=ROOT/'C031';rows=[]
    for parent in ('C023','C027'):
        history=[json.loads(line)for line in (ROOT/parent/'optimizer-steps.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==list(range(1,161))
        rows.extend(r['params_after'] for r in history[80:])
    expected=np.array([math.fsum(col)/160 for col in zip(*rows)])
    cp=torch.load(b/'router.pt',weights_only=True,map_location='cpu')
    w=np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()]
    diff=float(np.max(np.abs(w-expected)));assert diff<1e-14
    with np.load(b/'synthesis-vectors.npz')as f:
        np.testing.assert_array_equal(rows,f['params_after'].reshape(160,50))
    with np.load(ROOT/'C022/training-slots.npz')as f:
        a={k:f[k]for k in ('center','fast','slow','truth','enabled')}
    with np.load(ROOT/'C023/training-io.npz')as f:z=f['normalized_input']
    y,_,_,prob=prediction(z,cp['p0'].numpy(),w,*[a[k] for k in ('center','fast','slow','enabled')])
    with np.load(b/'source-replay.npz')as f:
        np.testing.assert_allclose(y,f['C031_direction'].reshape(-1,3),atol=1e-13)
        np.testing.assert_allclose(prob,f['C031_prob'].reshape(-1,3),atol=1e-13)
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C031','C031_deg'),defs));db=Catalog()
    parents={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C027/sequences.csv')}
    checks=[];panel=csv_rows(b/'sequences.csv')
    cases={('SHICE','R1-1'),('OxIOD','running/data1/syn/imu1.csv'),('TUM-VI','room1'),('TUM-VI','room2')}
    assert len(panel)==len(parents)==213
    for r in panel:
        key=r['dataset'],r['record'];old=parents[key]
        assert all(r[k]==old[k]for k in ('group','scored_frames','scoring_indices_sha256','status'))
        for field in ('C023_deg','C027_deg','full_vqf_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(r[field])-float(old[field]))<1e-5
        if key in cases:
            raw,_=db.native(*key);sc=model.score(raw)
            e=max(abs(sc[k]-float(r[c]))for k,c in [('full_vqf_deg','full_vqf_deg'),('candidate_deg','C031_deg')])
            assert e<1e-5 and sc['scoring_indices_sha256']==r['scoring_indices_sha256']
            checks.append({'dataset':key[0],'record':key[1],'raw_max_gap_deg':e})
    assert len(checks)==len(cases)
    result={'original_update_vectors':len(rows),'parameters_per_vector':50,'additional_training_steps':0,'max_parameter_gap':diff,
            'source_slots':len(z),'matched_development_records':len(panel),'independent_native_rows':checks,'deployed_sha256':digest(b/'router.pt')}
    (ROOT/'review/C031_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(diff,max(r['raw_max_gap_deg'] for r in checks))


if __name__=='__main__':main()
