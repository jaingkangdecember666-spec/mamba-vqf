"""Recompute all four-parameter solver evaluations, finite differences and shared native controls."""
import json

import numpy as np
import torch
from scipy.optimize import linprog
from scipy.special import softmax

from anchored_response_features import features
from check_sources import ROOT
from check_training_batch import csv_rows
from lock_from_frontier import read,digest,inference
from locked_replay import Predictor
from native_catalog import Catalog


class ExternalRisk:
    def __init__(self,arrays,checkpoint,names):
        self.a=arrays
        z=arrays['features']
        self.phi=features(arrays['fast'],arrays['slow'],z[:,24:28])
        self.logits=np.tile(checkpoint['log_p0'].numpy(),(len(z),1))
        self.logits[:,:2]+=z[:,:24]@checkpoint['state_dict']['weight'].numpy().T+checkpoint['state_dict']['bias'].numpy()
        units=np.where(arrays['domain']=='EuRoC',arrays['condition'],arrays['domain'])
        ids=[(g,int(r))for g,r in zip(units,arrays['source_record_ordinal'])]
        unique=sorted(set(ids))
        index={key:i for i,key in enumerate(unique)}
        self.number=np.array([index[k]for k in ids])
        self.sizes=np.bincount(self.number)
        self.groups=[np.array([i for i,(g,_)in enumerate(unique)if g==name])for name in names]
        assert all(len(v)>0 for v in self.groups)

    def value(self,coeff):
        a=self.a;logits=self.logits.copy()
        adjustment=self.phi@coeff
        logits[:,0]+=adjustment;logits[:,1]-=adjustment
        p=softmax(logits,axis=1)
        mix=(p[:,:1]*a['fast']+p[:,1:2]*a['slow'])*a['enabled'][:,None]
        b=a['center']+mix;n=np.linalg.norm(b,axis=1,keepdims=True);u=b/n
        theta=np.arctan2(np.linalg.norm(np.cross(u,a['truth']),axis=1),np.einsum('ij,ij->i',u,a['truth']))
        rms=np.sqrt(np.bincount(self.number,weights=theta**2)/self.sizes)
        group_rms=np.array([rms[i].mean()for i in self.groups])
        factor=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>1e-9)
        err=u-a['truth'];du=2.*(err-u*np.einsum('ij,ij->i',err,u)[:,None])/n
        dlog=np.column_stack([p[:,i]*np.einsum('ij,ij->i',du,a[k]-mix)*a['enabled']for i,k in enumerate(('fast','slow'))])
        drow=self.phi*((dlog[:,0]-dlog[:,1])*factor)[:,None]
        dr=np.column_stack([np.bincount(self.number,weights=c)/self.sizes/(2*rms) for c in drow.T])
        return group_rms,np.array([dr[i].mean(0)for i in self.groups]),u,p


def main():
    torch.set_num_threads(2);b=ROOT/'C030'
    cfg=read(b/'solver-receipt.json')
    with np.load(ROOT/'C026/training-inputs.npz')as f:
        keys=('center','fast','slow','truth','enabled','features','domain','condition','source_record_ordinal')
        arrays={k:f[k].reshape((-1,)+f[k].shape[2:])for k in keys}
    old=torch.load(ROOT/'C023/router.pt',map_location='cpu',weights_only=True)
    new=torch.load(b/'router.pt',map_location='cpu',weights_only=True)
    risk=ExternalRisk(arrays,old,cfg['group_order']);a=np.array(cfg['locked_a'])
    cp=new['state_dict']['weight'].numpy()
    np.testing.assert_array_equal(cp[:,:24],old['state_dict']['weight'].numpy())
    np.testing.assert_array_equal(cp[:,24:28],np.zeros((2,4)))
    np.testing.assert_array_equal(cp[:,28:],np.stack((a,-a)))
    np.testing.assert_array_equal(new['state_dict']['bias'].numpy(),old['state_dict']['bias'].numpy())
    base=risk.value(np.zeros(4))[0]
    evaluations=[json.loads(s)for s in (b/'solver-evaluations.jsonl').read_text().splitlines()if s]
    loss_gap=grad_gap=0.
    for record in evaluations:
        val,grad,_,_=risk.value(np.array(record['a']))
        loss_gap=max(loss_gap,float(np.max(np.abs(val/base-record['group_RMSE_ratio']))))
        grad_gap=max(grad_gap,float(np.max(np.abs(grad/base[:,None]-record['group_ratio_jacobian']))))
    val,grad,y,p=risk.value(a);ratio=val/base;ratio_jac=grad/base[:,None]
    delta=1e-5;diff=[]
    for v in np.eye(4):
        diff.append((risk.value(a+delta*v)[0]-risk.value(a-delta*v)[0])/base/(2*delta))
    fd_gap=float(np.max(np.abs(np.asarray(diff).T-ratio_jac)))
    assert max(loss_gap,grad_gap)<2e-11 and fd_gap<2e-8,(loss_gap,grad_gap,fd_gap)
    assert abs(ratio.max()-cfg['terminal_epigraph_max_group_ratio'])<1e-10 and ratio.max()<1.
    with np.load(b/'training-io.npz')as f:
        np.testing.assert_allclose(y,f['C030_direction'].reshape(-1,3),atol=1e-12)
        np.testing.assert_allclose(p,f['C030_prob'].reshape(-1,3),atol=1e-12)
    active=np.flatnonzero(ratio>=ratio.max()-1e-7)
    bounds=[(0,1)if x<=-1+1e-8 else (-1,0)if x>=1-1e-8 else (-1,1)for x in a]
    feasible=linprog([0,0,0,0,1],A_ub=np.c_[ratio_jac[active],-np.ones(len(active))],b_ub=np.zeros(len(active)),
                     bounds=[*bounds,(None,None)],method='highs')
    assert feasible.success and feasible.fun>-1e-7,feasible.message
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    model=Predictor(inference(('C030','C030_deg'),defs));catalog=Catalog()
    prior={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C029/sequences.csv')};rows=csv_rows(b/'sequences.csv')
    assert len(rows)==len(prior)==213
    selection={('SHICE','R1-1'),('TUM-VI','room2'),('OxIOD','running/data1/syn/imu1.csv'),('BROAD','04_undisturbed_slow_rotation_with_breaks_A')}
    proof=[]
    for row in rows:
        key=row['dataset'],row['record'];previous=prior[key]
        for k in ('status','scored_frames','scoring_indices_sha256','group'):assert row[k]==previous[k]
        for k in ('full_vqf_deg','C023_deg','C029_deg','fixed_tau1p5_deg','fixed_tau3_deg','fixed_tau6_deg'):
            assert abs(float(row[k])-float(previous[k]))<1e-5
        if key in selection:
            raw,_=catalog.native(*key);ans=model.score(raw)
            gap=max(abs(ans[k]-float(row[col]))for k,col in [('full_vqf_deg','full_vqf_deg'),('candidate_deg','C030_deg')])
            assert gap<1e-5 and ans['scoring_indices_sha256']==row['scoring_indices_sha256']
            proof.append({'dataset':key[0],'record':key[1],'max_gap_deg':gap})
    assert len(proof)==len(selection)
    report={'source_slots':len(y),'source_record_conditions':len(risk.sizes),'actual_coefficient_evaluations_checked':len(evaluations),
            'max_source_relative_objective_gap':loss_gap,'max_source_jacobian_gap':grad_gap,
            'centered_finite_difference_max_gap':fd_gap,'terminal_active_conditions':[cfg['group_order'][i]for i in active],
            'lowest_worst_first_derivative_within_active_bounds':float(feasible.fun),'terminal_group_ratios':dict(zip(cfg['group_order'],ratio.tolist())),
            'record_rows':len(rows),'native_checks':proof,'locked_score_sha256':digest(b/'sequences.csv'),'checkpoint_sha256':digest(b/'router.pt')}
    (ROOT/'review/C030_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(loss_gap,grad_gap,fd_gap,max(r['max_gap_deg']for r in proof))


if __name__=='__main__':main()
