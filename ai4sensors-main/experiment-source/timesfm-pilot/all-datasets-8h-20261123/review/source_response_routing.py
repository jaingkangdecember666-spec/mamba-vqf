"""First-order, source-only test of whether the frozen Mamba history can route fast and slow gravity estimates."""
import json

import numpy as np
import torch
from scipy.optimize import minimize

from check_sources import ROOT


def common_direction(matrix):
    norms=np.linalg.norm(matrix,axis=1)
    unit=matrix/norms[:,None]
    n=len(unit)
    gram=unit@unit.T
    fit=minimize(lambda w:float(w@gram@w),np.full(n,1./n),jac=lambda w:2.*gram@w,
            bounds=[(0.,1.)]*n,constraints={'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones_like(w)},
            method='SLSQP',options={'ftol':1e-13,'maxiter':200})
    assert fit.success
    direction=-fit.x@unit
    return direction/max(1e-12,np.linalg.norm(direction)),fit.x


def main():
    lock=json.loads((ROOT/'C022/coefficient-lock.json').read_bytes())
    k=lock['k'];p=np.array([2*k/3,k/3])
    with np.load(ROOT/'review/auxiliary_filter_training_slots.npz')as f:
        a={key:f[key]for key in f.files}
    prior=torch.load(ROOT/'C019/readout.pt',map_location='cpu',weights_only=True)
    with np.load(ROOT/'C019/training-inputs.npz')as source:
        np.testing.assert_array_equal(a['release'],source['native_release'])
        h=(source['pooled'][:,1,:].astype('f8')-prior['feature_mean'].numpy())/prior['feature_std'].numpy()
    features=np.c_[h,np.ones(len(h))]
    possible=np.stack((a['fast'],a['slow']),axis=1)
    possible[~a['active']]=0.
    mixture=np.einsum('b,nbk->nk',p,possible)
    est=a['center']+mixture
    magnitude=np.linalg.norm(est,axis=1,keepdims=True)
    y=est/magnitude
    error=y-a['truth']
    loss=(error**2).sum(1)
    # Two logits vary, leaving the default logit fixed. Differentiate the normalized physical direction.
    tangent=p[None,:,None]*(possible-mixture[:,None,:])
    tangent-=y[:,None,:]*np.einsum('nbk,nk->nb',tangent,y)[:,:,None]
    logit_derivative=2.*np.einsum('nbk,nk->nb',tangent,error)/magnitude
    domains=sorted(set(a['domain']))
    gradients=[];overview={}
    for source_name in domains:
        group=a['domain']==source_name
        w=a['record_equal_weights'][group]
        denominator=w@loss[group]
        gradient=np.einsum('n,nb,nd->bd',w,logit_derivative[group],features[group])/denominator
        gradients.append(gradient.ravel())
        overview[source_name]={'slots':int(group.sum()),'records':len(set(a['record_ordinal'][group])),
                 'two_constant_logit_gradients':gradient[:,-1].tolist(),'context_weight_gradient_norm':float(np.linalg.norm(gradient[:,:-1]))}
    gradients=np.array(gradients)
    joint,weights=common_direction(gradients)
    cosine=gradients@gradients.T/np.outer(np.linalg.norm(gradients,axis=1),np.linalg.norm(gradients,axis=1))
    for i,d in enumerate(domains):
        held,_=common_direction(np.delete(gradients,i,axis=0))
        overview[d].update(joint_relative_loss_slope=float(gradients[i]@joint),leave_one_source_out_slope=float(gradients[i]@held))
    output={'scope':'No network trained. Differentiate the source-record-balanced squared-chord loss at the single fixed C022 response mixture. Two hypothetical softmax logits can depend on 24 current B6 historical-state contrast features plus a bias. A common local direction exists when each normalized source gradient has a negative projection. Held-out-source slopes are source-proxy sensitivity only, not domain test results.',
        'coefficient':k,'domain_order':domains,'pairwise_gradient_cosine':cosine.tolist(),'common_direction_normalized_weights':weights.tolist(),'domains':overview}
    (ROOT/'review/source_response_routing.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    for name,data in overview.items():print(name,round(data['context_weight_gradient_norm'],5),round(data['joint_relative_loss_slope'],5),round(data['leave_one_source_out_slope'],5))


if __name__=='__main__':
    main()

