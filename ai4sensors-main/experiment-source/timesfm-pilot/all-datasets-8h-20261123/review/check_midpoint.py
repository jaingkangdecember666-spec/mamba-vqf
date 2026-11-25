"""Verify the single midpoint in the shared C023/C035 parameterization and its native development controls."""
import json

import numpy as np
import torch

from check_training_batch import csv_rows
from check_history_router import prediction
from anchor_blend_features import included_degrees
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from source_router_trust import TrustRisk,weights


def main():
    torch.set_num_threads(2)
    paths=[ROOT/name/'router.pt'for name in ('C023','C035','C038')]
    parents=[torch.load(p,map_location='cpu',weights_only=True)for p in paths]
    for key in ('feature_mean','feature_std','log_p0','p0'):
        assert torch.equal(parents[0][key],parents[1][key]) and torch.equal(parents[0][key],parents[2][key])
    a,b,c=(weights(name)[0]for name in ('C023','C035','C038'))
    gap=float(np.max(abs((a+b)*.5-c)));assert gap<1e-14 and len(c)==50
    risk=TrustRisk();values,_=risk.relative(c)
    source_gaps={}
    with np.load(ROOT/'C038/source-replay.npz')as replay,np.load(ROOT/'C033/training-inputs.npz')as original:
        for k in ('domain','source_record_name','source_record_ordinal','native_release','native_input_last','enabled'):
            np.testing.assert_array_equal(replay[k].ravel(),original[k].ravel())
        for name,params in [('C023',a),('C035',b),('C038',c)]:
            u,_,_,p=prediction(risk.z,risk.prior,params,risk.center,risk.f,risk.s,risk.enabled)
            deg=float(included_degrees(u,replay[name+'_direction']).max());prob=float(abs(p-replay[name+'_prob']).max())
            assert deg<1.e-6 and prob<1.e-10
            source_gaps[name]={'max_direction_gap_deg':deg,'max_probability_gap':prob}
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C038','C038_deg'),defs));db=Catalog()
    table=csv_rows(ROOT/'C038/sequences.csv')
    old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C035/sequences.csv')}
    assert len(table)==len(old)==213
    native=[]
    for row in table:
        key=row['dataset'],row['record'];ref=old[key]
        for k in ('status','group','scored_frames','scoring_indices_sha256'):assert row[k]==ref[k]
        for k in ('C023_deg','C035_deg','full_vqf_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(row[k])-float(ref[k]))<1e-5
        if key in {('SHICE','R1-1'),('OxIOD','running/data1/syn/imu1.csv'),('TUM-VI','room1'),('TUM-VI','room2')}:
            raw,_=db.native(*key);v=model.score(raw)
            assert v['scoring_indices_sha256']==row['scoring_indices_sha256']
            delta=abs(v['candidate_deg']-float(row['C038_deg']));assert delta<1e-5
            native.append({'dataset':key[0],'record':key[1],'candidate_max_abs_error_deg':delta})
    assert len(native)==4
    receipt={'checkpoint_hashes':[digest(p)for p in paths],'parameter_max_abs_gap':gap,
             'source_ratios_to_C023':dict(zip(risk.domains,values.tolist())),'distinct_source_records':len(risk.n),
             'aligned_external_training_slots':len(risk.z),'source_prediction_checks':source_gaps,
             'native_development_rows':native,'paired_development_records':len(table)}
    (ROOT/'review/C038_audit.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    print(values,gap,native)


if __name__=='__main__':main()
