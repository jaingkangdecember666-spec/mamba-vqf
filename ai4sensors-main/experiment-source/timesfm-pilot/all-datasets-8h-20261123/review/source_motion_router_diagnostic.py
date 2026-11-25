"""Check whether physical observer states add a common source-loss direction beyond the 24 Mamba features."""
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from auxiliary_filter_directions import sources
from check_history_router import prediction
from check_sources import ROOT
from davis_handheld import Davis
from lock_from_frontier import digest
from optical_score import estimate
from router_motion_context import NAMES,series
from source_response_routing import common_direction


def native(raw):
    xx,hz,_,_=raw
    keep=np.isfinite(xx).all(1)
    idx=np.maximum.accumulate(np.where(keep,np.arange(len(keep)),-1))
    x=xx[idx.clip(min=0)].copy();x[idx<0]=0.
    offset,on,_,_=estimate(x,keep,1./hz)
    if on.any():x[:,:3]-=offset
    return x,hz


def main():
    torch.set_num_threads(1)
    with np.load(ROOT/'C019/training-inputs.npz') as f:
        a={k:f[k]for k in ('source_record_ordinal','native_input_last','native_release','source_record_name','domain','pooled')}
    source_rows=[]
    features=np.zeros((len(a['native_release']),4))
    for i,(d,n,raw) in enumerate(sources()):
        take=a['source_record_ordinal']==i
        assert take.any() and np.all(a['source_record_name'][take]==n) and np.all(a['domain'][take]==d)
        x,hz=native(raw)
        state=series(x,hz)
        features[take]=state[a['native_input_last'][take]]
        if i in (0,35,73,138):
            stop=min(2500,len(x))
            short=series(x[:stop],hz)
            np.testing.assert_allclose(short,state[:stop],rtol=1e-12,atol=1e-10)
            flip=Rotation.from_rotvec([0.,0.,np.pi/2])
            rotated=np.column_stack((flip.apply(x[:stop,:3]),flip.apply(x[:stop,3:])))
            spin=series(rotated,hz)
            assert np.abs(short-spin).max()<2e-5,(d,n,np.abs(short-spin).max())
        source_rows.append({'domain':d,'record':n,'slots':int(take.sum())})
    assert len(source_rows)==139
    output=ROOT/'review/source_motion_router_features.npz'
    np.savez_compressed(output,features=features,native_input_last=a['native_input_last'],native_release=a['native_release'],
                        source_record_ordinal=a['source_record_ordinal'],feature_names=NAMES)
    cp=torch.load(ROOT/'C023/router.pt',map_location='cpu',weights_only=True)
    h=(a['pooled'][:,1,:].astype('f8')-cp['feature_mean'].numpy())/cp['feature_std'].numpy()
    pp=cp['p0'].numpy()
    params=np.r_[cp['state_dict']['weight'].numpy().ravel(),cp['state_dict']['bias'].numpy()]
    with np.load(ROOT/'C022/training-slots.npz')as f:
        dat={k:f[k]for k in ('center','fast','slow','truth','enabled','record_equal_weights','domain')}
    u,norm,mix,prob=prediction(h,pp,params,dat['center'],dat['fast'],dat['slow'],dat['enabled'])
    err=u-dat['truth'];loss=(err**2).sum(1)
    g=2*(err-u*(err*u).sum(1,keepdims=True))/norm
    grads=np.column_stack((prob[:,0]*(g*(dat['fast']-mix)).sum(1),prob[:,1]*(g*(dat['slow']-mix)).sum(1)))
    grads[~dat['enabled']]=0.
    w=dat['record_equal_weights'];sd=features.std(0);avg=features.mean(0)
    z=(features-avg)/sd
    # Only use physical variance unseen by a linear map of the existing features, with no labels in this projection.
    old=np.column_stack((h,np.ones(len(h))))
    rootw=np.sqrt(w)[:,None]
    coef=np.linalg.lstsq(old*rootw,z*rootw,rcond=None)[0]
    extra=z-old@coef
    groups=sorted(set(dat['domain']))
    derivs=[];summary={}
    for d in groups:
        sel=dat['domain']==d
        den=w[sel]@loss[sel]
        G=np.einsum('n,ni,nj->ij',w[sel],grads[sel],extra[sel])/den
        derivs.append(G.ravel())
        summary[d]={'records':len(np.unique(a['source_record_ordinal'][sel])),'slots':int(sel.sum()),
                    'relative_gradients_on_novel_motion_context':G.tolist()}
    derivs=np.asarray(derivs);common,_=common_direction(derivs)
    for i,d in enumerate(groups):
        held,_=common_direction(np.delete(derivs,i,axis=0))
        summary[d].update(common_first_order_slope=float(derivs[i]@common),omitted_source_slope=float(derivs[i]@held))
    extra_dir=ROOT/'review/davis_source_motion_context'
    extra_dir.mkdir(exist_ok=True)
    reader=Davis();davis={}
    for n,role in reader.roles.items():
        if role!='external_train':continue
        raw,_=reader.native(n);x,hz=native(raw);native_features=series(x,hz)
        path=extra_dir/(n+'.npy');np.save(path,native_features,allow_pickle=False)
        davis[n]={'native_rows':len(x),'file':str(path.relative_to(ROOT)), 'sha256':digest(path)}
    result={'scope':'Four prespecified scalar observers at two seconds. The 139-record check evaluates the normalized source loss gradient at the frozen C023 checkpoint, after removing the linear projection of observer inputs on the 24 old features. No parameter optimization, no development or test scores. For DAVIS store only the six external source files.','feature_names':NAMES,'native_features_sha256':digest(output),'source_slots':len(z),
            'source_records':len(source_rows),'original_feature_std':sd.tolist(),'fractional_variance_beyond_old_linear_features':np.average(extra**2,axis=0,weights=w).tolist(),'domain_gradients':summary,
            'davis':davis,'code_sha256':digest(Path(__file__).with_name('router_motion_context.py'))}
    (ROOT/'review/source_motion_router_diagnostic.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print({d:round(x['common_first_order_slope'],6) for d,x in summary.items()})


if __name__=='__main__':main()
