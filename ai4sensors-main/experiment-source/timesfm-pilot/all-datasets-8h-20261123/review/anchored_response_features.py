"""Observe how far to shift the two response logits around the frozen C023 estimator."""
import numpy as np

from causal_response_geometry import measures


NAMES=('D_over_6','D_over_6_times_bias_z','D_over_6_times_force_norm_z','D_over_6_times_force_spread_z')


def features(fast,slow,motion_z):
    states=np.atleast_2d(motion_z)
    d=measures(fast,slow,states[:,3])[:,0]/6.
    return d[:,None]*np.column_stack((np.ones(len(d)),states[:,3],states[:,1],states[:,2]))
