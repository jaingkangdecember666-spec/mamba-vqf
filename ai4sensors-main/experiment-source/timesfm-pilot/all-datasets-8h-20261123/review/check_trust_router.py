"""Audit the external minimax fit, all 50 weight updates per step, and independent native development anchors."""
import json

import numpy as np
import torch
from scipy.optimize import linprog

from check_training_batch import csv_rows
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from source_router_trust import TrustRisk,weights


def main():
    torch.set_num_threads(2)
    b=ROOT/'C035'; risk=TrustRisk(); receipt=read(b/'solver-receipt.json')
    training=read(b/'training.json'); assert receipt['starts_at_C023'] and receipt['neural_fits']==receipt['solve_count']==1
    opt,prior=weights('C035');np.testing.assert_array_equal(prior,risk.prior)
    records=lambda file:[json.loads(s)for s in (b/file).read_text().splitlines()]
    steps=records('neural-steps.jsonl')
    assert [r['step']for r in steps]==list(range(1,training['steps']+1))
    last=risk.start
    for r in steps:
        np.testing.assert_array_equal(last,r['params_before'])
        after=np.asarray(r['params_after']);np.testing.assert_allclose(after-last,r['actual_update'],rtol=0,atol=1.e-14)
        assert np.linalg.norm(after-last)>0
        last=after
    np.testing.assert_allclose(last,opt,atol=1.e-14,rtol=0)
    ev=records('solver-evaluations.jsonl'); assert len(ev)==receipt['distinct_solver_point_count']
    gap=jac_gap=0.
    for point in ev:
        val,deriv=risk.relative(np.asarray(point['x'][:50]))
        gap=max(gap,float(np.max(np.abs(val-point['source_ratios']))))
        jac_gap=max(jac_gap,float(np.max(np.abs(deriv-point['relative_risk_jacobian']))))
    assert max(gap,jac_gap)<1.e-8
    val,grad=risk.relative(opt)
    delta=opt-risk.start;distance=np.linalg.norm(delta);assert distance<=risk.radius+1.e-7
    active=val>=val.max()-1.e-7
    tangent=np.c_[grad[active],-np.ones(np.sum(active))]
    if distance>=risk.radius-1.e-6:tangent=np.vstack((tangent,np.r_[delta,0.]))
    lp=linprog(np.r_[np.zeros(50),1.],A_ub=tangent,b_ub=np.zeros(len(tangent)),bounds=[(-1.,1.)]*50+[(None,None)],method='highs')
    assert lp.success and np.max(val)<1. and abs(np.max(val)-receipt['result_objective'])<2.e-7
    before=torch.load(ROOT/'C023/router.pt',map_location='cpu',weights_only=True)
    after=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    for k in ('feature_mean','feature_std','p0','log_p0'): assert torch.equal(before[k],after[k])
    recipes={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    predictor=Predictor(inference(('C035','C035_deg'),recipes));db=Catalog()
    rows=csv_rows(b/'sequences.csv');previous={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C033/sequences.csv')}
    assert len(rows)==len(previous)==213
    anchors={('BROAD','04_undisturbed_slow_rotation_with_breaks_A'),('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv')}
    native=[]
    for r in rows:
        key=r['dataset'],r['record'];old=previous[key]
        for k in ('group','status','scored_frames','scoring_indices_sha256'):assert old[k]==r[k]
        for k in ('full_vqf_deg','C023_deg','C033_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):assert abs(float(r[k])-float(old[k]))<1.e-5
        if key in anchors:
            raw,_=db.native(*key); result=predictor.score(raw)
            assert result['scoring_indices_sha256']==r['scoring_indices_sha256']
            diff=abs(result['candidate_deg']-float(r['C035_deg']));assert diff<1.e-5
            native.append({'dataset':key[0],'record':key[1],'error_deg':diff})
    assert len(native)==len(anchors)
    result={'source_domains':dict(zip(risk.domains,[len(x)for x in risk.members])),'slots':len(risk.z),'radius':risk.radius,
            'distance':distance,'accepted_parameter_changes':len(steps),'checked_solver_points':len(ev),'max_risk_gap':gap,
            'max_Jacobian_gap':jac_gap,'relative_external_risks':dict(zip(risk.domains,val.tolist())),
            'smallest_active_max_derivative_in_unit_tangent_box':lp.fun,'consistent_development_records':len(rows),
            'raw_development_anchors':native,'selected_checkpoint_sha256':digest(b/'router.pt')}
    (ROOT/'review/C035_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(len(steps),len(ev),gap,jac_gap,lp.fun,[r['error_deg']for r in native])


if __name__=='__main__':main()
