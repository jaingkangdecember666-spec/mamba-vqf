"""Rotation-invariant context from two seconds of causal inertial/VQF history at native rate."""
import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF


NAMES=('gyro_rms','force_norm_departure_rms','world_force_direction_spread','estimated_gyro_bias_norm')
TAU=2.
G=9.80665


def series(signal,hz):
    """Input is the same finite, forward-filled, causally recentered six-axis stream used by VQF."""
    data=np.ascontiguousarray(signal,dtype=np.float64)
    w=np.ascontiguousarray(data[:,3:]);a=np.ascontiguousarray(data[:,:3])
    base=VQF(1./hz).updateBatch(w,a)
    q=Rotation.from_quat(base['quat6D'],scalar_first=True)
    norm=np.linalg.norm(a,axis=1)
    good=norm>1e-12
    unit=np.zeros_like(a)
    unit[good]=q[good].apply(a[good]/norm[good,None])
    vector=np.column_stack(((w*w).sum(1),(norm/G-1.)**2,unit,good))
    alpha=-np.expm1(-1./(TAU*hz))
    smooth=lfilter([alpha],[1.,alpha-1.],vector,axis=0)
    mass=-np.expm1(-np.arange(1,len(data)+1)/(TAU*hz))
    deviation=np.zeros(len(data))
    measured=smooth[:,5]>1e-12
    coherence=np.linalg.norm(smooth[measured,2:5],axis=1)/smooth[measured,5]
    deviation[measured]=np.sqrt(2.*np.maximum(0.,1.-coherence))
    return np.column_stack((np.log1p(np.sqrt(np.maximum(0.,smooth[:,0]/mass))),
                            np.log1p(10.*np.sqrt(np.maximum(0.,smooth[:,1]/mass))),
                            np.log1p(10.*deviation),
                            np.log1p(np.degrees(np.linalg.norm(base['bias'],axis=1)))))
