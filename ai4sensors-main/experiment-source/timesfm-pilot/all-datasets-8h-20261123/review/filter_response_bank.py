"""Half- and double-response-time gravity estimates, compared in the same estimated world at input-last."""
import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

UP=np.array([0.,0.,1.])


def capped_horizontal(world):
    tangent=np.zeros_like(world)
    tangent[...,:2]=world[...,:2]/world[...,2:].clip(min=1e-6)
    norm=np.linalg.norm(tangent,axis=-1,keepdims=True)
    return tangent*np.minimum(1.,np.tan(np.deg2rad(6.))/np.maximum(1e-12,norm))


def response_delta(base,fast,slow):
    fast_world=base.apply(fast.inv().apply(UP))
    slow_world=base.apply(slow.inv().apply(UP))
    return (2.*capped_horizontal(fast_world)+capped_horizontal(slow_world))/3.


def response_series(signal,hz,central,cuts):
    acc=np.ascontiguousarray(signal[:,:3]);gyro=np.ascontiguousarray(signal[:,3:])
    def at(tau):
        state=VQF(1./hz,tauAcc=tau).updateBatch(gyro,acc)
        return Rotation.from_quat(state['quat6D'][cuts],scalar_first=True)
    return response_delta(central[cuts],at(1.5),at(6.))

