"""Compare external source gradients on the same Mamba readout before changing the training mixture."""
import hashlib
import json

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from torch.nn import functional as F

from check_sources import ROOT, TRANSFER, weights
from native_blocks import block_inputs
from optical_score import canonical_frame


@torch.inference_mode()
def phone_points(row, head):
    path=ROOT/'external-sources/ridi-phone-train'/row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['file_sha256']
    with np.load(path) as f:
        assert f['imu_observed'].all()
        time=f['time_ns']
        x=np.c_[f['acc_m_s2'],f['gyr_rad_s']]
        truth=Rotation.from_quat(f['tango_wxyz'],scalar_first=True).inv().apply([0.,0.,1.])
        good=f['reference_valid'] & (time-time[0]>=10_000_000_000)
    q,win,last,rel=block_inputs(x,200.)
    ok=np.flatnonzero(good[rel])
    assert len(ok)>=256
    take=ok[np.linspace(0,len(ok)-1,256).round().astype(int)]
    hist=win[take]
    local,basis,active=canonical_frame(hist)
    assert active.all() and np.array_equal(rel[take],last[take])
    target=np.einsum('ni,nij->nj',truth[rel[take]],basis).astype('f4')
    network,mean,std,_=head
    latents=[]
    for i in range(0,len(take),128):
        inp=torch.from_numpy(local[i:i+128]).to(mean.device)
        first=network.input(((inp-mean)/std).clamp(-12,12))
        last_hidden=(first+network.sequence(network.norm(first)))[:,-1].cpu().numpy()
        latents.append(last_hidden)
    return np.c_[np.concatenate(latents),local[:,-1,6:9],target], hist[:,-1,12].astype(bool)


def derivative(checkpoint, points, rest):
    w=checkpoint['state_dict']['output.weight'].double().requires_grad_()
    b=checkpoint['state_dict']['output.bias'].double().requires_grad_()
    raw=torch.from_numpy(points.astype(float))
    latent,up,y=raw[:,:-6],raw[:,-6:-3],raw[:,-3:]
    z=.1*(latent@w.T+b)
    z=z-(up*z).sum(1,keepdim=True)*up
    cap=np.tan(np.deg2rad(float(checkpoint['cap_deg'])))
    z=z*(cap/z.norm(dim=1,keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    z=z*torch.from_numpy(~rest)[:,None]
    pred=F.normalize(up+.25*z,dim=1)
    loss=(pred-y).square().sum(1).mean()
    dw,db=torch.autograd.grad(loss,(w,b))
    g=torch.cat((dw.flatten(),db)).detach().numpy()
    return g,{'windows':len(points),'rest':int(rest.sum()),'fixed_gain_loss':float(loss.detach()),
              'default_direction_loss':float((up-y).square().sum(1).mean()),'gradient_l2':float(np.linalg.norm(g))}


def local_common_direction(gradients,stats):
    names=list(gradients)
    g=np.stack([gradients[n]*stats[n]['gradient_l2']/stats[n]['fixed_gain_loss'] for n in names])
    gram=g@g.T
    res=minimize(lambda w:.5*w@gram@w,np.ones(len(names))/len(names),jac=lambda w:gram@w,method='SLSQP',
                 bounds=[(0.,1.)]*len(names),constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,
                 'jac':lambda w:np.ones(len(w))}],options={'ftol':1e-13})
    assert res.success and np.all(gram@res.x >= res.x@gram@res.x-1e-6)
    sign=gram@res.x
    return {'mixture_weights':dict(zip(names,res.x.tolist())),
            'common_descent_norm':float(np.sqrt(res.x@gram@res.x)),
            'negative_directional_derivative_of_relative_loss':dict(zip(names,sign.tolist())),
            'interpretation':'Local first-order calculation at the saved readout, without taking an optimizer step.'}


def main():
    torch.set_num_threads(2)
    base=TRANSFER/'batch6/mamba-intrinsic-selected.pt'
    drawn={}
    with np.load(ROOT/'C012/training-windows.npz') as win:
        v=win['C004_points'].reshape(-1,30)
        rest=win['vqf_rest'].ravel()
        ids=win['record_ordinal'].ravel()
        original=json.loads((ROOT/'C004/protocol.json').read_text('utf8'))['source_records']
        domains=np.array([r['domain'] for r in original])[ids]
        for d in sorted(set(domains)):
            drawn[d]=(v[domains==d],rest[domains==d])
    head=weights(base,'mamba_v1')
    phone=json.loads((ROOT/'external-sources/ridi-phone-train/source-audit.json').read_text('utf8'))
    runs=[phone_points(r,head) for r in phone['rows']]
    drawn['RIDI']=tuple(np.concatenate(x) for x in zip(*runs))
    out={}
    for name,path in [('original_B6',base),('C006',ROOT/'C004/B-readout.pt'),('C012',ROOT/'C012/rest-trained-readout.pt')]:
        cp=torch.load(path,map_location='cpu',weights_only=True)
        grads={};evidence={}
        for d,(p,flags) in drawn.items():
            grad,stats=derivative(cp,p,flags)
            grads[d]=grad/np.linalg.norm(grad)
            evidence[d]=stats
        out[name]={'statistics':evidence,'gradient_cosines':{a:{b:float(e@f) for b,f in grads.items()} for a,e in grads.items()},
                   'minimum_norm_common_direction':local_common_direction(grads,evidence)}
    output={'scope':'Source-only diagnostic; no optimizer updates. Compares the gradient of deployed rest-off prediction '
                    'on three actual matched source draws and four RIDI phone recordings, 256 evenly spaced eligible releases each.',
            'checkpoints':out}
    (ROOT/'review/source_gradient_agreement.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    for name,x in out.items():
        print(name,[(d,round(row['fixed_gain_loss'],6),round(row['gradient_l2'],6)) for d,row in x['statistics'].items()],
              'RIDI cos', {d:round(v,3) for d,v in x['gradient_cosines']['RIDI'].items()})


if __name__=='__main__':
    main()
