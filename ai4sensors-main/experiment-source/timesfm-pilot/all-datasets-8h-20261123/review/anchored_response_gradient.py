"""Check external first derivatives of a four-coefficient correction about the previously verified C023."""
import json

import numpy as np
import torch
from scipy.optimize import linprog

from causal_response_geometry import measures
from check_history_router import prediction
from check_sources import ROOT
from lock_from_frontier import digest


def main():
    p=ROOT/'C026/training-inputs.npz'
    with np.load(p)as f:
        a={k:f[k].reshape((-1,)+f[k].shape[2:])for k in ('features','fast','slow','center','truth','enabled','domain','condition','source_record_ordinal')}
    cp=torch.load(ROOT/'C023/router.pt',weights_only=True,map_location='cpu')
    coeff=np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()]
    z=a['features']
    u,norm,mix,prob=prediction(z[:,:24],cp['p0'].numpy(),coeff,*[a[k]for k in ('center','fast','slow','enabled')])
    chord=np.sum((u-a['truth'])**2,axis=1)
    theta=2.*np.arcsin(np.sqrt(np.clip(chord,0.,4.))/2.)
    jac=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>1e-10)
    err=u-a['truth'];d=2.*(err-u*np.einsum('ij,ij->i',err,u)[:,None])/norm
    logit_grad=np.column_stack([prob[:,i]*np.einsum('ij,ij->i',d,a[k]-mix)*a['enabled'] for i,k in enumerate(('fast','slow'))])
    delta=(logit_grad[:,0]-logit_grad[:,1])*jac
    angle=measures(a['fast'],a['slow'],z[:,-1])[:,0]
    phi=angle[:,None]/6.*np.column_stack((np.ones(len(z)),z[:,27],z[:,25],z[:,26]))
    dcols=phi*delta[:,None]
    groups=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    summary={}
    for g in sorted(set(groups)):
        grads=[];base=[]
        for r in np.unique(a['source_record_ordinal'][groups==g]):
            ix=(groups==g)&(a['source_record_ordinal']==r)
            rms=np.sqrt(np.mean(theta[ix]**2));base.append(rms)
            grads.append(dcols[ix].mean(0)/(2*rms))
        avg=float(np.mean(base))
        summary[g]={'source_record_macro_C023_deg':float(np.degrees(avg)),
                    'fractional_RMSE_gradient':(np.mean(grads,axis=0)/avg).tolist(),
                    'record_conditions':len(grads)}
    mat=np.array([v['fractional_RMSE_gradient']for v in summary.values()])
    directions={}
    for width in (2,4):
        sub=mat[:,:width]
        sol=linprog(np.r_[np.zeros(width),1.],A_ub=np.c_[sub,-np.ones(len(sub))],b_ub=np.zeros(len(sub)),
                    bounds=[(-1.,1.)]*width+[(None,None)],method='highs')
        assert sol.success
        directions[str(width)]={'best_possible_worst_group_directional_derivative':float(sol.fun),
                                'linearized_direction':sol.x[:width].tolist()}
    out={'scope':'Derivatives only at the published C023, evaluated on the exact C026 external training slots. The two response logits receive opposite additions, the original logit stays fixed. D is the present angle in degrees between the two capped physical alternatives. Every feature vanishes when the physical alternatives coincide. Group objective is angular RMSE within each acquisition followed by equal average and division by that group at C023. The linear program checks whether a simultaneous strict first-order improvement is possible with bounded coefficients. No predictions from a new configuration were scored.',
         'geometry_basis':['D/6','D/6 times normalized gyro bias','D/6 times normalized force norm departure','D/6 times normalized world force spread'],
         'source_sha256':digest(p),'base_router_sha256':digest(ROOT/'C023/router.pt'),'groups':summary,
         'first_order_feasibility':directions}
    (ROOT/'review/anchored_response_gradient.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(directions)


if __name__=='__main__':main()
