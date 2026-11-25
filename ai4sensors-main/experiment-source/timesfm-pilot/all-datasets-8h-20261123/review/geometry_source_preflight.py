"""Check additional causal observables on the existing external acquisitions, with no target scores or candidate fitting."""
import json

import numpy as np
import torch

from causal_response_geometry import NAMES,measures
from check_history_router import prediction
from check_sources import ROOT
from lock_from_frontier import digest


def main():
    p=ROOT/'C026/training-inputs.npz'
    with np.load(p)as f:
        a={k:f[k].reshape((-1,)+f[k].shape[2:])for k in ('features','fast','slow','center','truth','enabled','domain','condition','source_record_ordinal','effective_loss_weights')}
    z=a['features'];groups=np.where(a['domain']=='EuRoC',a['condition'],a['domain'])
    raw=measures(a['fast'],a['slow'],z[:,-1])
    w=np.zeros(len(raw))
    for g in set(groups):
        gsel=groups==g
        records=np.unique(a['source_record_ordinal'][gsel])
        portion=1./(12. if g in ('raw','fixed_prefix_bias_removed') else 6.)
        for r in records:
            ix=gsel&(a['source_record_ordinal']==r)
            w[ix]=portion/(len(records)*ix.sum())
    np.testing.assert_allclose(w.sum(),1.,atol=1e-13)
    avg=w@raw;std=np.sqrt(w@(raw-avg)**2);obs=(raw-avg)/std
    design=np.column_stack((np.ones(len(z)),z));root=np.sqrt(w[:,None])
    projection=np.linalg.lstsq(design*root,obs*root,rcond=None)[0]
    residual=obs-design@projection
    output={'scope':'Description of the 148 C026 training acquisitions. Each original record within each of six sources has equal weight; both EuRoC conditions carry half their source mass. The projection is used only to quantify how much variation the old features can explain. No resulting coefficients are transferred to a candidate.',
            'source_sha256':digest(p),'names':list(NAMES),'source_mean':avg.tolist(),'source_std':std.tolist(),
            'fraction_of_feature_variance_remaining_after_projection_on_old28':(w@residual**2).tolist()}
    cp=torch.load(ROOT/'C026/router.pt',weights_only=True,map_location='cpu')
    coeff=np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()]
    u,norm,mix,prob=prediction(z,cp['p0'].numpy(),coeff,*[a[k]for k in ('center','fast','slow','enabled')])
    error=u-a['truth'];d=2.*(error-u*np.einsum('ij,ij->i',error,u)[:,None])/norm
    scale=a['enabled']*a['effective_loss_weights']/6.
    raw_delta=np.column_stack([prob[:,i]*np.einsum('ij,ij->i',d,a[k]-mix)*scale for i,k in enumerate(('fast','slow'))])
    output['gradient_at_C026_new_zero_coefficients']=(raw_delta.T@obs).tolist()
    output['gradient_after_projecting_out_old_features']=(raw_delta.T@residual).tolist()
    source={}
    for g in sorted(set(groups)):
        ix=groups==g
        source[g]=(raw_delta[ix].T@residual[ix]).tolist()
    output['per_source_gradient_after_projection']=source
    with np.load(ROOT/'C028/teacher-slots.npz')as teacher,np.load(ROOT/'C028/training-io.npz')as io:
        labels=teacher['label'].ravel();ww=teacher['global_weight'].ravel()
        ce=io['C028_p'].reshape(-1,3).copy()
        ce[np.arange(len(ce)),labels]-=1.
        ce=ce[:,:2]*ww[:,None]
        output['C028_teacher_gradient_after_projection']=(ce.T@residual).tolist()
    (ROOT/'review/geometry_source_preflight.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print('scales',avg,std,'remaining variance',w@residual**2,'partial gradients',(raw_delta.T@residual).tolist())


if __name__=='__main__':main()
