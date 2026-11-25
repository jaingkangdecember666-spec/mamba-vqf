"""Apply an externally calibrated interpolation between unmodified FullVQF and one frozen Mamba estimate."""
import numpy as np
from scipy.signal import lfilter
from scipy.special import expit
from scipy.spatial.transform import Rotation
from vqf import VQF

from anchor_blend_features import blend,design

UP=np.array([0.,0.,1.])


def weight_and_blend(raw,model,force,bias,coefficients):
    w=expit(design(raw,model,force,bias) @ np.asarray(coefficients))
    return blend(raw,model,w)


def apply_offline(signal,hz,indices,output,coefficients):
    obs=np.isfinite(signal).all(1)
    last=np.maximum.accumulate(np.where(obs,np.arange(len(obs)),-1))
    x=signal[last.clip(min=0)].copy();x[last<0]=0.
    a=np.ascontiguousarray(x[:,:3]);w=np.ascontiguousarray(x[:,3:])
    result=VQF(1./hz).updateBatch(w,a)
    q=Rotation.from_quat(result['quat6D'],scalar_first=True)
    length=np.linalg.norm(a,axis=1,keepdims=True)
    local=a/np.maximum(length,1e-12)
    alpha=-np.expm1(-1./hz)
    f=lfilter([alpha],[1.,alpha-1.],q.apply(local),axis=0)[indices]
    f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
    raw=output['full_vqf_deg']
    # The worker and this replay independently compute the native raw VQF.
    output['candidate_deg']=weight_and_blend(raw,output['candidate_deg'],q[indices].inv().apply(f),
        np.linalg.norm(result['bias'][indices],axis=1),coefficients)
    return output


class StreamingBlend:
    def __init__(self,hz,coefficients):
        self.filter=VQF(1./hz)
        self.world_force=np.zeros(3)
        self.alpha=-np.expm1(-1./hz)
        self.coefficients=np.asarray(coefficients,dtype=float)
        assert self.coefficients.shape==(4,) and np.isfinite(self.coefficients).all()

    def packet(self,row,model):
        self.filter.update(row[3:],row[:3])
        q=Rotation.from_quat(self.filter.getQuat6D(),scalar_first=True)
        u=row[:3]/max(np.linalg.norm(row[:3]),1e-12)
        self.world_force+=self.alpha*(q.apply(u)-self.world_force)
        norm=max(np.linalg.norm(self.world_force),1e-12)
        f=q.inv().apply(self.world_force/norm)
        bias=np.linalg.norm(self.filter.getBiasEstimate()[0])
        return weight_and_blend(q.inv().apply(UP)[None],model[None],f[None],np.array([bias]),self.coefficients)[0]
