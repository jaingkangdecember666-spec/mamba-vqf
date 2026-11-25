"""Test on genuine external sources whether causal observables can reduce all source errors locally."""
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linprog

from anchor_blend_features import design,blend,included_degrees
from lock_from_frontier import ROOT,inference,read,digest
from locked_replay import Predictor
from penn_handheld import Penn
from raw_anchor_sources import native


def main():
    torch.set_num_threads(2)
    bank=ROOT/'review/external-raw-anchor-139.npz'
    with np.load(bank) as f:
        payload={k:[f[k]] for k in ('raw_default','C023','truth','past_force','estimated_bias_rad_s','domain','record')}
    recipes={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    estimator=Predictor(inference(('C023','C023_deg'),recipes))
    ds=Penn()
    for name in ('af','as'):
        (s,hz,truth,mask),_=ds.native(name)
        idx=np.flatnonzero(mask)
        rb,force,gyro=native(s,hz,idx)
        c=estimator.predict(s,hz,idx)['candidate_deg']
        for key,values in [('raw_default',rb),('C023',c),('truth',truth[idx]),('past_force',force),
                         ('estimated_bias_rad_s',gyro),('domain',np.full(len(c),'PennCOSYVIO')),('record',np.full(len(c),name))]:
            payload[key].append(values)
    p={k:np.concatenate(v) for k,v in payload.items()}
    z=design(p['raw_default'],p['C023'],p['past_force'],p['estimated_bias_rad_s'])
    d=p['domain'];r=p['record'];domains=sorted(set(d))
    # X and delta are causal. Only evaluation of actual tilt-angle losses uses source truth.
    def risk(param):
        prob=1./(1.+np.exp(-(z@np.asarray(param))))
        y=blend(p['raw_default'],p['C023'],prob)
        sq=included_degrees(y,p['truth'])**2
        base=included_degrees(p['raw_default'],p['truth'])**2
        values=[]
        for domain in domains:
            part=d==domain
            by_record=[np.flatnonzero(part & (r==name)) for name in np.unique(r[part])]
            values.append(np.mean([np.sqrt(np.mean(sq[j]))for j in by_record])/np.mean([np.sqrt(np.mean(base[j]))for j in by_record]))
        return np.array(values)
    null=np.zeros(4)
    ratio=risk(null)
    derivatives=np.column_stack([(risk(e*1e-4)-risk(-e*1e-4))/(2e-4) for e in np.eye(4)])
    # Existence of a *local* direction at equal mix. The LP direction is not a fitted predictor.
    lim=linprog([0.,0.,0.,0.,1.],A_ub=np.c_[derivatives,-np.ones(len(domains))],b_ub=np.zeros(len(domains)),
                bounds=[(-1.,1.)]*4+[(None,None)],method='highs')
    assert lim.success
    out={'domains':domains,'record_counts':{d0:len(set(r[d==d0])) for d0 in domains},
         'feature_order':['intercept','candidate/default angular separation divided by 6deg capped at 1',
                          'agreement with 1s causal force direction divided by separation','tanh(raw VQF estimated gyro bias in deg/s)'],
         'risk_at_50_percent_mix':ratio.tolist(),'per_group_gradient_at_50_percent_mix':derivatives.tolist(),
         'unit_box_minimax_direction':lim.x[:4].tolist(),'max_directional_derivative':float(lim.x[-1]),
         'half_step_directional_changes':(risk(.5*lim.x[:4])-ratio).tolist(),'raw_bank_sha256':digest(bank),
         'scope':'5 external sources only: 139 original recordings at historical training slots and 2 Penn A training records on eligible native rows. No model deployed, no target read. The derivative at the fixed equal blend uses exact angular RMSE, record-equal per source and relative to unmodified default FullVQF.'}
    (ROOT/'review/anchor_blend_evidence.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(out['domains'],out['risk_at_50_percent_mix'],out['max_directional_derivative'],out['half_step_directional_changes'])


if __name__=='__main__':main()
