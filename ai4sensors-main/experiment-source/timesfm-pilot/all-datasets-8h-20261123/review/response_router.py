"""Native replay of external-only Mamba-conditioned weights over three causal gravity responses."""
import numpy as np
import torch
from scipy.special import softmax
from scipy.spatial.transform import Rotation
from vqf import VQF

from filter_response_bank import UP,capped_horizontal
from optical_score import canonical_frame


def alternatives(central,fast,slow):
    return (capped_horizontal(central.apply(fast.inv().apply(UP))),
            capped_horizontal(central.apply(slow.inv().apply(UP))))


def release_alternatives(signal,hz,central,indices):
    a=np.ascontiguousarray(signal[:,:3]);w=np.ascontiguousarray(signal[:,3:])
    def at(tau):
        data=VQF(1./hz,tauAcc=tau).updateBatch(w,a)
        return Rotation.from_quat(data['quat6D'][indices],scalar_first=True)
    return alternatives(central[indices],at(1.5),at(6.))


class ResponseRouter:
    def __init__(self,backbone,path,expected_k,context=None):
        self.core,self.mean,self.std,_=backbone
        cp=torch.load(path,map_location='cpu',weights_only=True)
        self.context=context
        self.geometry=context in ('motion_and_response_geometry','anchored_response_physics')
        if context=='native_motion_2s' or self.geometry:
            from router_motion_context import NAMES
            if tuple(cp['feature_names'])!=NAMES:
                raise ValueError('Four observer columns differ from the registered order')
            self.feature_mean=np.concatenate([cp['old_feature_mean'].numpy(),cp['context_mean'].numpy()])
            self.feature_std=np.concatenate([cp['old_feature_std'].numpy(),cp['context_std'].numpy()])
            if self.geometry:
                if context=='anchored_response_physics':
                    from anchored_response_features import NAMES as GEOMETRY
                else:
                    from causal_response_geometry import NAMES as GEOMETRY
                if tuple(cp['geometry_names'])!=GEOMETRY:raise ValueError('Response geometry columns differ')
                self.feature_mean=np.r_[self.feature_mean,cp['geometry_mean'].numpy()]
                self.feature_std=np.r_[self.feature_std,cp['geometry_std'].numpy()]
        elif context is None:
            self.feature_mean=cp['feature_mean'].numpy().astype('f8')
            self.feature_std=cp['feature_std'].numpy().astype('f8')
        else:
            raise ValueError(f'Unrecognized observer context {context}')
        self.weight=cp['state_dict']['weight'].detach().cpu().numpy()
        self.bias=cp['state_dict']['bias'].detach().cpu().numpy()
        self.log0=cp['log_p0'].detach().cpu().numpy()
        self.prior=cp['p0'].detach().cpu().numpy()
        dim={'anchored_response_physics':32,'motion_and_response_geometry':30,'native_motion_2s':28,None:24}[context]
        assert self.weight.shape==(2,dim) and self.bias.shape==(2,) and self.feature_mean.shape==self.feature_std.shape==(dim,)
        assert np.all(self.feature_std>0.) and np.isfinite(self.weight).all() and np.isfinite(self.bias).all()
        np.testing.assert_allclose(self.prior,[2*expected_k/3,expected_k/3,1-expected_k],atol=1e-15,rtol=0.)
        np.testing.assert_allclose(self.log0,np.log(self.prior),atol=1e-14,rtol=0.)

    @torch.inference_mode()
    def probabilities(self,windows,observed_context=None,responses=None):
        oriented,_,valid=canonical_frame(windows)
        states=[]
        for i in range(0,len(oriented),128):
            x=torch.from_numpy(oriented[i:i+128]).to(self.mean.device)
            x=((x-self.mean)/self.std).clamp(-12.,12.)
            h=self.core.input(x)
            h=h+self.core.sequence(self.core.norm(h))
            states.append((h[:,:56].mean(1)-h[:,-8:].mean(1)).cpu().numpy())
        x=np.concatenate(states).astype('f8')
        if self.context:
            c=np.asarray(observed_context)
            if c.shape!=(len(x),4) or not np.isfinite(c).all():
                raise ValueError('Expected four finite motion observers for every causal release')
            x=np.column_stack((x,c))
            if self.geometry:
                from causal_response_geometry import measures
                if responses is None:raise ValueError('Expected a pair of current, capped VQF responses')
                motion_z=(c-self.feature_mean[24:28])/self.feature_std[24:28]
                if self.context=='anchored_response_physics':
                    from anchored_response_features import features
                    extra=features(*responses,motion_z)
                else:
                    extra=measures(*responses,motion_z[:,3])
                x=np.column_stack((x,extra))
        x=(x-self.feature_mean)/self.feature_std
        logits=np.broadcast_to(self.log0,(len(x),3)).copy()
        logits[:,:2]+=x@self.weight.T+self.bias
        return softmax(logits,axis=1),valid

