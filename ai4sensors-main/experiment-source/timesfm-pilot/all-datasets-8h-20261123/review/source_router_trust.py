"""Exact record-equal angular risks for small changes to the externally trained C023 Mamba readout."""
import json

import numpy as np
import torch
from scipy.optimize import linprog

from check_history_router import prediction
from lock_from_frontier import ROOT,read,digest


def weights(batch):
    ck=torch.load(ROOT/batch/'router.pt',weights_only=True,map_location='cpu')
    return np.r_[ck['state_dict']['weight'].numpy().ravel(),ck['state_dict']['bias'].numpy()],ck['p0'].numpy()


class TrustRisk:
    def __init__(self):
        self.start,self.prior=weights('C023')
        self.radius=float(np.linalg.norm(weights('C027')[0]-self.start))
        with np.load(ROOT/'C033/training-inputs.npz')as f:
            flat=lambda k:f[k].reshape((-1,)+f[k].shape[2:])
            self.z,self.center,self.f,self.s,self.truth,self.enabled=(flat(k)for k in ('z','center','fast','slow','truth','enabled'))
            dom=flat('domain');idx=flat('source_record_ordinal')
        self.n=np.bincount(idx);self.ids=idx
        assert len(self.n)==141 and min(self.n)>0
        self.domains=sorted(set(dom))
        self.members=[np.unique(idx[dom==d]) for d in self.domains]
        self.ref,_=self.absolute(self.start)

    def absolute(self,params):
        u,length,mix,p=prediction(self.z,self.prior,params,self.center,self.f,self.s,self.enabled)
        theta=np.arctan2(np.linalg.norm(np.cross(u,self.truth),axis=1),np.sum(u*self.truth,axis=1))
        rec=np.sqrt(np.bincount(self.ids,weights=theta**2)/self.n)
        means=np.array([rec[v].mean() for v in self.members])
        factor=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>1e-9)
        du=-2.*factor[:,None]*(self.truth-u*(u*self.truth).sum(1)[:,None])/length
        gate=np.column_stack([p[:,i]*np.sum(du*(alternative-mix),axis=1)*self.enabled for i,alternative in enumerate((self.f,self.s))])
        dz=np.column_stack((self.z,np.ones(len(self.z))))
        derivatives=[]
        for ids in self.members:
            ix=np.isin(self.ids,ids)
            factor=1./(2.*self.n[self.ids[ix]]*rec[self.ids[ix]]*len(ids))
            grad=(gate[ix]*factor[:,None]).T@dz[ix]
            derivatives.append(np.r_[grad[:,:-1].ravel(),grad[:,-1]])
        return means*180/np.pi,np.vstack(derivatives)*180/np.pi

    def relative(self,params):
        v,g=self.absolute(params)
        return v/self.ref,g/self.ref[:,None]


def main():
    risk=TrustRisk()
    ref,g=risk.relative(risk.start)
    assert np.allclose(ref,1.)
    new_c033=weights('C033')[0]
    displacement=new_c033-risk.start
    fd_direction=displacement/np.linalg.norm(displacement)
    step=1e-5*fd_direction
    observed=(risk.relative(risk.start+step)[0]-risk.relative(risk.start-step)[0])/(2e-5)
    discrepancy=float(np.max(np.abs(observed-g@fd_direction)))
    assert discrepancy<1e-7
    solve=linprog([*np.zeros(50),1.],A_ub=np.c_[g,-np.ones(len(g))],b_ub=np.zeros(len(g)),
                  bounds=[(-1.,1.)]*50+[(None,None)],method='highs')
    assert solve.success
    direction=solve.x[:50];direction/=max(1e-30,np.linalg.norm(direction))
    info={'source_recording_counts':dict(zip(risk.domains,[len(r)for r in risk.members])),'domains':risk.domains,
          'C023_macro_deg':risk.ref.tolist(),'C033_over_C023_ratios':risk.relative(new_c033)[0].tolist(),
          'C023_to_C033_parameter_distance':float(np.linalg.norm(displacement)),
          'C023_to_its_own_order_replicate_distance':risk.radius,'finite_difference_max_discrepancy':discrepancy,
          'maximum_domain_directional_derivative_at_C023_for_one_common_unit_direction':float(max(g@direction)),
          'recorded_source_panel_sha256':digest(ROOT/'C033/training-inputs.npz'),
          'scope':'Five genuine external source groups. Find only whether a simultaneous descent direction exists at C023 and compute the actual training-order radius; no new checkpoint, fit, or target score.'}
    (ROOT/'review/held_router_trust_preflight.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    print(info)


if __name__=='__main__':main()
