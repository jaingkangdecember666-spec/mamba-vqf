"""Reconstruct the two C018 temporal representations from the original B6 selective-state layer."""
import torch
from torch import nn
from torch.nn import functional as F


class TemporalReadout(nn.Module):
    def __init__(self,core,saved):
        super().__init__()
        self.core=core
        self.feature=saved['feature']
        if self.feature not in ('H[-1]','mean(H[:56])-mean(H[-8:])'):
            raise ValueError(f'Unsupported temporal feature {self.feature}')
        self.register_buffer('center',saved['feature_mean'])
        self.register_buffer('scale',saved['feature_std'])
        self.register_buffer('weight',saved['state_dict']['weight'])
        self.register_buffer('bias',saved['state_dict']['bias'])
        if not (self.center.shape==self.scale.shape==(24,) and self.weight.shape==(3,24) and self.bias.shape==(3,)):
            raise ValueError('Invalid readout dimensions')
        if not all(torch.isfinite(x).all() for x in (self.center,self.scale,self.weight,self.bias)) or (self.scale<=0.).any():
            raise ValueError('Invalid normalization or readout weights')

    def forward(self,x):
        h=self.core.input(x)
        h=h+self.core.sequence(self.core.norm(h))
        code=h[:,-1] if self.feature=='H[-1]' else h[:,:56].mean(1)-h[:,-8:].mean(1)
        return F.linear((code-self.center)/self.scale,self.weight,self.bias)


def load_readout(backbone,checkpoint):
    network,imu_mean,imu_std,_=backbone
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    head=TemporalReadout(network,saved).to(imu_mean.device).eval()
    return (head,imu_mean,imu_std,6.),float(saved['gain'])


def trained_layout(cp):
    if 'architecture' in cp:
        arch=cp['architecture']
        shape=(arch['window'],arch['hidden_size'],arch['input_size'])
        feature,cap,ms=arch['pooling'],arch['tangent_cap_deg'],10
        mean_key,std_key='C019_feature_mean','C019_feature_std'
    else:
        arch=cp['configuration']
        shape=(arch['steps'],arch['hidden_size'],arch['input_features'])
        feature,cap,ms=arch['history_feature'],arch['cap_deg'],arch['block_ms']
        mean_key,std_key='feature_mean','feature_std'
    if cp['model']!='mamba_v1' or shape!=(64,24,13) or ms not in (10,40):
        raise ValueError('Invalid full temporal Mamba architecture')
    return feature,cap,arch['gain'],ms//10,mean_key,std_key


def load_trained_readout(checkpoint,device):
    from models import ResidualHead
    cp=torch.load(checkpoint,weights_only=True,map_location='cpu')
    feature,cap,gain,scale,mean_key,std_key=trained_layout(cp)
    core=ResidualHead(cp['model'])
    core.load_state_dict(cp['state_dict'])
    spec={'feature':feature,'feature_mean':cp[mean_key],'feature_std':cp[std_key],
          'state_dict':{'weight':cp['state_dict']['output.weight'],'bias':cp['state_dict']['output.bias']}}
    head=TemporalReadout(core,spec).to(device).eval()
    head.history_scale=scale
    return (head,cp['B6_input_mean'].to(device),cp['B6_input_std'].to(device),float(cap)),float(gain)
