"""Measure the gradients of two source-only objectives at the frozen C022 response prior."""
import json

import numpy as np
import torch

from check_history_router import gradient,prediction
from check_sources import ROOT
from lock_from_frontier import digest


def main():
    archive=ROOT/'C026/training-inputs.npz'
    with np.load(archive)as f:
        names=('features','center','fast','slow','truth','enabled','effective_loss_weights','domain','condition','source_record_ordinal')
        a={k:f[k].reshape((-1,)+f[k].shape[2:]) for k in names}
    cp=torch.load(ROOT/'C026/router.pt',map_location='cpu',weights_only=True)
    features=a['features'];zero=np.zeros(58);p0=cp['p0'].numpy()
    center,fast,slow,truth,use=(a[k]for k in ('center','fast','slow','truth','enabled'))
    u,*_=prediction(features,p0,zero,center,fast,slow,use)
    chord=np.clip(np.sum((u-truth)**2,axis=1),0.,4.)
    theta=2.*np.arcsin(np.sqrt(chord)/2.)
    # Derivative of theta**2 with respect to squared chord, continuously extended at perfect agreement.
    factor=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>1.e-10)
    groups=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    adjusted=np.zeros_like(chord)
    overview={}
    for domain in sorted(set(groups)):
        sel=groups==domain
        original=np.unique(a['source_record_ordinal'][sel])
        groupspec=[]
        for record in original:
            idx=np.flatnonzero(sel&(a['source_record_ordinal']==record))
            rmse=float(np.sqrt(np.mean(theta[idx]**2)))
            assert rmse>1e-8
            groupspec.append((idx,rmse))
        rmse0=np.mean([r for ix,r in groupspec])
        for idx,rmse in groupspec:
            adjusted[idx]=factor[idx]/(2.*len(groupspec)*len(idx)*rmse*rmse0)
        if domain in ('raw','fixed_prefix_bias_removed'):adjusted[sel]*=.5
        args=[x[sel]for x in (center,fast,slow,truth,use)]
        def grad(weights):
            return gradient(features[sel],p0,zero,*args,weights[sel])[1]
        g=grad(a['effective_loss_weights']);new=grad(adjusted)
        overview[domain]={'record_conditions':len(original),'source_macro_rmse_deg':float(np.degrees(rmse0)),
                          'smallest_record_rmse_deg':float(np.degrees(min(x[1] for x in groupspec))),
                          'largest_record_rmse_deg':float(np.degrees(max(x[1] for x in groupspec))),
                          'gradient_cosine':float((g@new)/(np.linalg.norm(g)*np.linalg.norm(new))),
                          'reweighted_to_original_gradient_norm':float(np.linalg.norm(new)/np.linalg.norm(g))}
    g=gradient(features,p0,zero,center,fast,slow,truth,use,a['effective_loss_weights']/6)[1]
    h=gradient(features,p0,zero,center,fast,slow,truth,use,adjusted/6)[1]
    out={'scope':'Only the 17,600 archived training slots, sampled from 148 original acquisitions. Directional comparison at a common zero-step router. The previous objective takes recording-equal squared chord divided by its source mean. The alternative differentiates recording-equal angular RMSE divided by its source mean RMSE, giving both EuRoC conditions half mass. Uses no target reference, fitting or prediction change.',
         'source_sha256':digest(archive),'source_groups':overview,
         'overall_gradient_cosine':float((g@h)/(np.linalg.norm(g)*np.linalg.norm(h))),
         'overall_gradient_norm_ratio':float(np.linalg.norm(h)/np.linalg.norm(g))}
    (ROOT/'review/record_objective_alignment.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(out['overall_gradient_cosine'],[(d,round(v['gradient_cosine'],3)) for d,v in overview.items()])


if __name__=='__main__':main()
