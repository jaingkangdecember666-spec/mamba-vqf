"""Bound the learned change to the contemporaneous physical gravity direction on the unit sphere."""
import numpy as np


def cap_direction(physical, learned, cap_deg):
    if not np.isfinite(cap_deg) or cap_deg<0:
        raise ValueError('The physical displacement limit must be finite and nonnegative')
    a=np.asarray(physical,dtype=float)
    b=np.asarray(learned,dtype=float)
    if a.shape!=b.shape or a.shape[-1]!=3:
        raise ValueError('Expected paired 3D directions')
    a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    dot=np.clip(np.sum(a*b,axis=-1,keepdims=True),-1.,1.)
    tangent=b-dot*a
    length=np.linalg.norm(tangent,axis=-1,keepdims=True)
    angle=np.arctan2(length,dot)
    # In the antipodal case choose a deterministic orthogonal axis.
    ambiguous=(length<1.e-13)&(dot<0.)
    if np.any(ambiguous):
        axis=np.eye(3)[np.argmin(np.abs(a),axis=-1)]
        ortho=axis-a*np.sum(a*axis,axis=-1,keepdims=True)
        tangent=np.where(ambiguous,ortho,tangent)
        length=np.linalg.norm(tangent,axis=-1,keepdims=True)
    unit=np.divide(tangent,length,out=np.zeros_like(tangent),where=length>1.e-15)
    cap=np.radians(cap_deg)
    answer=np.where(angle<=cap,b,np.cos(cap)*a+np.sin(cap)*unit)
    return answer/np.linalg.norm(answer,axis=-1,keepdims=True)
