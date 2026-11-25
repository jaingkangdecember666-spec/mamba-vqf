"""Record-equal external angular RMSE ratios and derivatives for the raw-VQF interpolation."""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from anchor_blend_features import included_degrees
from lock_from_frontier import ROOT, digest


class AnchorRisk:
    def __init__(self,path):
        with np.load(path) as f:
            self.z=f['features'];self.raw=f['raw_default'];self.candidate=f['C023'];self.truth=f['truth']
            domain=f['domain'];name=f['record']
        ids=sorted(set(zip(domain,name)))
        idmap={k:i for i,k in enumerate(ids)}
        self.index=np.array([idmap[k]for k in zip(domain,name)])
        self.count=np.bincount(self.index)
        self.domains=sorted(set(domain))
        self.members=[np.array([i for i,k in enumerate(ids) if k[0]==d])for d in self.domains]
        self.diff=self.candidate-self.raw
        base=included_degrees(self.raw,self.truth)
        per_record=np.sqrt(np.bincount(self.index,weights=base**2)/self.count)
        self.den=np.array([per_record[m].mean() for m in self.members])

    def risk(self,beta):
        g=expit(self.z@np.asarray(beta))
        v=self.raw+g[:,None]*self.diff
        length=np.linalg.norm(v,axis=1)
        y=v/length[:,None]
        theta=np.deg2rad(included_degrees(y,self.truth))
        per_record=np.sqrt(np.bincount(self.index,weights=theta**2)/self.count)
        values=np.array([per_record[m].mean()for m in self.members])*180/np.pi/self.den
        # Differentiate row angle squared first, then root within each individual recording.
        q=self.diff-y*np.sum(self.diff*y,axis=1)[:,None]
        f=np.divide(theta,np.sin(theta),out=np.ones_like(theta),where=theta>1e-9)
        row=-2.*f*np.sum(q*self.truth,axis=1)/length*g*(1.-g)
        derivatives=np.array([np.bincount(self.index,weights=row*column)/self.count for column in self.z.T]).T
        derivatives=np.divide(derivatives,2*per_record[:,None],where=per_record[:,None]>1e-15,out=np.zeros_like(derivatives))
        jac=np.vstack([derivatives[m].mean(axis=0)for m in self.members])*180/np.pi/self.den[:,None]
        return values,jac


def main():
    path=ROOT/'review/raw-anchor-training-141.npz'
    risk=AnchorRisk(path)
    null=np.zeros(4)
    val,jac=risk.risk(null)
    fd=np.column_stack([(risk.risk(e*1e-4)[0]-risk.risk(-e*1e-4)[0])/(2e-4)for e in np.eye(4)])
    assert np.max(np.abs(fd-jac))<1e-7
    q={'domains':risk.domains,'raw_default_record_macro_deg':risk.den.tolist(),
       'initial_zero_coefficients_ratio':val.tolist(),'initial_jacobian':jac.tolist(),
       'finite_difference_max_discrepancy':float(np.max(np.abs(fd-jac))),'bank_sha256':digest(path),
       'scope':'Pre-fit gradient at fixed half correction on 15,680 exact source slots; no fitted policy, target references or comparisons.'}
    (ROOT/'review/raw-anchor-risk-preflight.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
    print(risk.domains,val,q['finite_difference_max_discrepancy'])


if __name__=='__main__':main()
