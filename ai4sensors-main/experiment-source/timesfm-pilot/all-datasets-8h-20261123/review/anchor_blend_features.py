"""Causal, rotation-invariant observables for how much of an observed Mamba correction to apply."""
import numpy as np


def included_degrees(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def design(raw,model,smoothed_specific_force,estimated_bias_rad_s):
    separation=included_degrees(raw,model)
    support=(included_degrees(raw,smoothed_specific_force)-included_degrees(model,smoothed_specific_force))
    agreement=np.divide(support,separation,out=np.zeros_like(support),where=separation>1e-7).clip(-1.,1.)
    return np.column_stack((np.ones(len(separation)),np.minimum(separation/6.,1.),agreement,
                            np.tanh(np.rad2deg(estimated_bias_rad_s))))


def blend(raw,model,weight):
    v=raw+(model-raw)*weight[:,None]
    return v/np.linalg.norm(v,axis=1,keepdims=True)
