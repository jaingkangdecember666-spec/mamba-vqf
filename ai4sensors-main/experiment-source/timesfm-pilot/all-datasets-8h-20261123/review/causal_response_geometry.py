"""Two per-release observations computed before the router emits any weights."""
import numpy as np


NAMES=('fast_slow_angle_deg','fast_slow_angle_deg_times_normalized_bias')


def measures(fast,slow,bias_z):
    f=np.atleast_2d(fast)+[0.,0.,1.]
    s=np.atleast_2d(slow)+[0.,0.,1.]
    bias=np.atleast_1d(bias_z)
    assert f.shape==s.shape==(len(bias),3)
    angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(f,s),axis=1),np.einsum('ij,ij->i',f,s)))
    return np.column_stack((angle,angle*bias))
