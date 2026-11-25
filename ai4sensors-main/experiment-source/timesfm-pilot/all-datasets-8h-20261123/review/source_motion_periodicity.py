"""Test whether repeating translational and angular motion explains conflicting external filter-response slopes."""
import json

import numpy as np

from auxiliary_filter_directions import sources
from check_sources import ROOT


def rhythm(signal,hz,cuts):
    offsets=np.ceil(np.arange(80,-1,-1)*hz/20.-1.e-9).astype('i8')
    idx=cuts[:,None]-offsets[None,:]
    present=idx[:,0]>=0
    native=signal[idx.clip(min=0)]
    present&=np.isfinite(native).all(axis=(1,2))
    mag=np.stack((np.linalg.norm(native[:,:,:3],axis=2)/9.80665,np.linalg.norm(native[:,:,3:],axis=2)),axis=2)
    mag[~present]=0.
    support=present & (mag[:,:,0].std(1)>=.05) & (mag[:,:,1].std(1)>=.1)
    out=np.zeros(len(cuts))
    for lag in range(8,21):
        left=mag[:,:-lag];right=mag[:,lag:]
        left=left-left.mean(1,keepdims=True);right=right-right.mean(1,keepdims=True)
        correlation=(left*right).sum(1)/np.maximum(1e-12,np.sqrt((left*left).sum(1)*(right*right).sum(1)))
        out=np.maximum(out,correlation.min(1))
    out[~support]=0.
    return out,present,support


def main():
    location=ROOT/'review/auxiliary_filter_training_slots.npz'
    with np.load(location) as bank:
        a={k:bank[k] for k in bank.files}
    order=a['record_ordinal']
    score=np.zeros(len(order));present=np.zeros_like(score,dtype=bool);support=present.copy()
    first=[]
    for i,(domain,name,(signal,hz,_,_)) in enumerate(sources()):
        sel=order==i
        cuts=a['input_last'][sel]
        score[sel],present[sel],support[sel]=rhythm(signal,hz,cuts)
        if domain not in first:
            cut=cuts[len(cuts)//2]
            if cut>5*hz and cut+1<len(signal):
                clone=signal.copy();clone[cut+1:]=50.
                p1=rhythm(signal,hz,np.array([cut]))
                p2=rhythm(clone,hz,np.array([cut]))
                assert all(np.array_equal(x,y) for x,y in zip(p1,p2))
                first.append(domain)
    assert len(first)==4
    size=np.linalg.norm(a['center'],axis=1,keepdims=True)
    u=a['center']/size
    error=u-a['truth']
    weight=a['record_equal_weights']
    output={}
    for d in dict.fromkeys(a['domain']):
        group=a['domain']==d
        denominator=weight[group]@(error[group]**2).sum(1)
        data={}
        for label,m in [('rhythmic',score>=.6),('other',score<.6)]:
            sel=group&m&a['active']
            row={'slots':int(sel.sum()),'sequences':int(len(np.unique(order[sel])))}
            for key in ('fast','slow'):
                t=a[key]-u*(a[key]*u).sum(1,keepdims=True)
                derivative=2.*(error*t).sum(1)/size[:,0]
                row[key+'_fractional_whole_source_slope']=float(weight[sel]@derivative[sel]/denominator)
            data[label]=row
        active=group & a['active']
        output[d]={'slots':int(group.sum()),'active_slots':int(active.sum()),'full_history_slots':int((group&present).sum()),
                'sufficient_motion_slots':int((group&support).sum()),'score_quantiles':np.quantile(score[active],[.1,.5,.9]).tolist(),'strata':data}
    out={'scope':'Training sources only. On an ordered 4 s causal window at 20 Hz, compute correlations at periods 0.4 to 1.0 s for both acceleration norm and angular-speed norm. The smaller of the two correlations is used at each lag. Strong repeated motion requires a common correlation at least 0.6, acceleration-norm standard deviation at least 0.05 g and angular-speed standard deviation at least 0.1 rad/s. This descriptor uses rotation-invariant norms and no source identity. Gradients stay normalized by the entire source. No coefficients fitted here.',
         'future_perturbation_domains':first,'domains':output}
    (ROOT/'review/source_motion_periodicity.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d,v in output.items():
        print(d,round(v['strata']['rhythmic']['slots']/v['active_slots'],3),[(k,round(x['fast_fractional_whole_source_slope'],4),round(x['slow_fractional_whole_source_slope'],4)) for k,x in v['strata'].items()])


if __name__=='__main__':
    main()

