"""Check every external solver evaluation, all development controls, four native recordings, and packet causality."""
import json

import numpy as np
import torch
from scipy.optimize import linprog

from anchor_blend_risk import AnchorRisk
from anchor_blend_features import included_degrees
from check_training_batch import csv_rows
from lock_from_frontier import ROOT,read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog
from penn_handheld import Penn
from stream_latency import Streaming


def main():
    torch.set_num_threads(2)
    b=ROOT/'C034';receipt=read(b/'solver-receipt.json')
    bank=ROOT/'review/raw-anchor-training-141.npz'
    assert receipt['fit_bank_sha256']==digest(bank) and receipt['solve_count']==1
    beta=np.array(receipt['locked_beta'])
    risk=AnchorRisk(bank);assert risk.domains==receipt['group_order']
    history=[json.loads(s)for s in (b/'solver-evaluations.jsonl').read_text().splitlines()]
    val_error=grad_error=0.
    for point in history:
        ratios,jac=risk.risk(point['x'][:4])
        val_error=max(val_error,np.max(np.abs(ratios-point['ratios'])))
        grad_error=max(grad_error,np.max(np.abs(jac-point['ratio_jacobian'])))
    assert max(val_error,grad_error)<2e-10 and len(history)==receipt['distinct_solver_point_count']
    ratios,jac=risk.risk(beta);fd=np.column_stack([(risk.risk(beta+e*3e-5)[0]-risk.risk(beta-e*3e-5)[0])/(6e-5)for e in np.eye(4)])
    grad_check=float(np.max(np.abs(fd-jac)));assert grad_check<1e-7
    assert abs(ratios.max()-receipt['terminal_epigraph_max_group_ratio'])<2e-9
    ids=np.flatnonzero(ratios>=ratios.max()-2e-7)
    bounds=[(0.,1.)if x<=-4+1e-7 else (-1.,0.)if x>=4-1e-7 else (-1.,1.)for x in beta]
    slope=linprog([0.,0.,0.,0.,1.],A_ub=np.c_[jac[ids],-np.ones(len(ids))],b_ub=np.zeros(len(ids)),bounds=[*bounds,(None,None)],method='highs')
    assert slope.success
    registered={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    selected=Predictor(inference(('C034','C034_deg'),registered));db=Catalog()
    earlier={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C033/sequences.csv')}
    rows=csv_rows(b/'sequences.csv');assert len(rows)==len(earlier)==213
    keys={('BROAD','04_undisturbed_slow_rotation_with_breaks_A'),('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv')}
    cases=[]
    for row in rows:
        pair=row['dataset'],row['record'];prior=earlier[pair]
        for col in ('group','scored_frames','scoring_indices_sha256','status'):assert row[col]==prior[col]
        for col in ('full_vqf_deg','C023_deg','C033_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(row[col])-float(prior[col]))<1e-5
        if pair in keys:
            raw,_=db.native(*pair);res=selected.score(raw)
            assert res['scoring_indices_sha256']==row['scoring_indices_sha256']
            e=max(abs(res['candidate_deg']-float(row['C034_deg'])),abs(res['full_vqf_deg']-float(row['full_vqf_deg'])))
            assert e<1e-5
            cases.append({'dataset':pair[0],'record':pair[1],'max_gap_deg':e})
    assert len(cases)==len(keys)
    packet_stream=Streaming(selected.stream_heads(),selected.force['candidate_deg'],0.,selected.router,beta)
    (signal,_,_,_),_=Penn().native('as')
    # Use exactly 200 Hz in both implementations to isolate native-packet causality and initialization.
    sig=np.ascontiguousarray(signal[:3000]);ids0=np.arange(len(sig))
    offline=selected.predict(sig,200.,ids0)['candidate_deg'];online=np.array([packet_stream.packet(x)[0]for x in sig])
    stream_gap=float(included_degrees(offline,online).max())
    # Deliberately change future IMU packets, keeping the same historical inputs.
    altered=sig.copy();altered[1750:,:3]*=-1.
    past=selected.predict(altered,200.,np.arange(1750))['candidate_deg']
    future_gap=float(included_degrees(past,offline[:1750]).max())
    assert max(stream_gap,future_gap)<1e-5
    result={'checked_distinct_solver_points':len(history),'source_max_ratio_discrepancy':val_error,
            'source_max_Jacobian_discrepancy':grad_error,'centered_finite_difference_discrepancy':grad_check,
            'locked_beta':beta.tolist(),'terminal_source_ratios':dict(zip(risk.domains,ratios.tolist())),
            'terminal_active_conditions':[risk.domains[i]for i in ids],'minimal_max_first_derivative_at_feasible_bounds':slope.fun,
            'independent_native_development_rows':cases,'development_records':len(rows),'packet_stream_rows':3000,
            'native_stream_vs_batch_max_deg':stream_gap,'future_perturbation_max_past_angle_deg':future_gap,
            'source_status':receipt['result_message'],'selection_receipt_sha256':digest(b/'solver-receipt.json')}
    (ROOT/'review/C034_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(len(history),val_error,grad_error,stream_gap,future_gap)


if __name__=='__main__':main()
