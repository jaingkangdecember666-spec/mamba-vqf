"""Measure source-only directional derivatives for two causal VQF response times around the C020 Mamba."""
import json
from itertools import chain,islice

import numpy as np
import torch
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from filter_response_audit import phones
from long_state_source_audit import training_records
from native_catalog import yt_record
from optical_score import estimate
from temporal_readout import load_trained_readout

UP=np.array([0.,0.,1.])


def sources():
    yield from chain(islice(training_records(),63),phones())
    more=json.loads((ROOT/'C019/training-preparation.json').read_bytes())['new_YT_trial_order']
    inv=json.loads((ROOT/'C001/inventory.json').read_bytes())['records']
    indexed={r['record']:r for r in inv if r['dataset']=='YT_Server' and r['official_split']=='train'}
    for name in more:
        assert indexed[name]['side']=='external_training_candidate'
        yield 'YT_Server',name,yt_record(indexed[name])


def trained_center(a):
    bundle,gain=load_trained_readout(ROOT/'C020/mamba-end-to-end.pt','cpu')
    net,mean,std,cap=bundle
    canonical=np.load(ROOT/'C020/training-canonical-history.npy',mmap_mode='r')
    out=[]
    with torch.inference_mode():
        for i in range(0,len(canonical),128):
            x=torch.from_numpy(np.array(canonical[i:i+128],copy=True))
            out.append(net(((x-mean)/std).clamp(-12,12)).numpy())
    local=.1*np.concatenate(out).astype('f8')
    up=a['canonical_up']
    local-=np.sum(local*up,axis=1,keepdims=True)*up
    local*=np.minimum(1.,np.tan(np.deg2rad(cap))/np.maximum(1e-12,np.linalg.norm(local,axis=1,keepdims=True)))
    world=np.einsum('nij,nj->ni',a['world_basis'],local)
    world[:,2]=0.
    world[a['rest']|~a['valid']]=0.
    return a['center']+gain*world


def auxiliary_vectors(raw,cuts,release,y):
    signal,hz,truth,_=raw
    ob=np.isfinite(signal).all(1)
    tail=np.maximum.accumulate(np.where(ob,np.arange(len(ob)),-1))
    clean=signal[tail.clip(min=0)].copy();clean[tail<0]=0.
    offset,active,_,_=estimate(clean,ob,1./hz)
    if active.any():
        clean[:,:3]-=offset
    acc=np.ascontiguousarray(clean[:,:3]);gyro=np.ascontiguousarray(clean[:,3:])
    def run(tau):
        state=VQF(1./hz,tauAcc=tau).updateBatch(gyro,acc)
        return Rotation.from_quat(state['quat6D'],scalar_first=True)
    base=run(3.)
    comparison=base[release].apply(truth[release])
    np.testing.assert_allclose(comparison,y,rtol=0,atol=2e-6)
    vectors=[]
    for tau in (1.5,6.):
        world=base[cuts].apply(run(tau)[cuts].inv().apply(UP))
        tangent=world[:,:2]/world[:,2:].clip(min=1e-6)
        tangent*=np.minimum(1.,np.tan(np.deg2rad(6.))/np.maximum(1e-12,np.linalg.norm(tangent,axis=1,keepdims=True)))
        vectors.append(np.column_stack((tangent,np.zeros(len(tangent)))))
    alpha=-np.expm1(-1./hz)
    speed=np.sqrt(lfilter([alpha],[1.,alpha-1.],(gyro*gyro).sum(1)))[cuts]
    load=np.sqrt(lfilter([alpha],[1.,alpha-1.],(np.linalg.norm(acc,axis=1)/9.80665-1.)**2))[cuts]
    return (*vectors,speed,load)


def main():
    torch.set_num_threads(2)
    with np.load(ROOT/'C019/training-inputs.npz') as f:
        a={k:f[k] for k in f.files}
    center=trained_center(a)
    centersize=np.linalg.norm(center,axis=1,keepdims=True)
    direction=center/centersize
    error=direction-a['truth']
    loss=(error*error).sum(1)
    numbers=a['source_record_ordinal']
    n=len(numbers)
    fast=np.zeros((n,3));slow=np.zeros_like(fast);speed=np.zeros(n);load=np.zeros(n)
    records=0
    for i,(domain,name,raw) in enumerate(sources()):
        chosen=numbers==i
        assert chosen.any() and np.all(a['source_record_name'][chosen]==name) and np.all(a['domain'][chosen]==domain)
        fast[chosen],slow[chosen],speed[chosen],load[chosen]=auxiliary_vectors(raw,a['native_input_last'][chosen],a['native_release'][chosen],a['truth'][chosen])
        records+=1
    assert records==139 and n==15360
    active=~a['rest'] & a['valid']
    np.savez_compressed(ROOT/'review/auxiliary_filter_training_slots.npz',center=center,truth=a['truth'],
        fast=fast,slow=slow,active=active,record_ordinal=numbers,domain=a['domain'],
        record_equal_weights=a['record_equal_weights'],input_last=a['native_input_last'],release=a['native_release'],
        speed_rms_rad_s=speed,force_departure_rms_g=load)
    bins={'all':active,'load_below_0.05g':active&(load<.05),'load_at_least_0.05g':active&(load>=.05),
          'gyro_below_0.5':active&(speed<.5),'gyro_at_least_0.5':active&(speed>=.5)}
    summary={}
    for d in dict.fromkeys(a['domain']):
        domain=a['domain']==d
        weights=a['record_equal_weights']
        assert np.isclose(weights[domain].sum(),1.)
        denominator=weights[domain]@loss[domain]
        stats={}
        for key,cond in bins.items():
            sel=domain&cond
            item={'slots':int(sel.sum()),'records':len(np.unique(numbers[sel]))}
            for short,delta in [('fast',fast),('slow',slow)]:
                tangent=delta-direction*(delta*direction).sum(1,keepdims=True)
                derivative=2.*(error*tangent).sum(1)/centersize[:,0]
                item[short+'_fractional_loss_slope']=float(weights[sel]@derivative[sel]/denominator)
                by_record=[float(weights[sel&(numbers==i)]@derivative[sel&(numbers==i)])for i in np.unique(numbers[sel])]
                item[short+'_helped_source_records']=sum(v<0. for v in by_record)
            stats[key]=item
        summary[d]=stats
    evidence={'scope':'Analytic slope at zero added gain, normalized by whole-source C020 chord loss. Each source record has equal total weight. The two causal filters change only tauAcc, from 3.0 s to 1.5 s or 6.0 s. Their horizontal deviations are capped at 6 degrees and measured in the current VQF world at input-last. All 15360 supervised slots are external training rows; no optimization or evaluation set here.',
              'native_records':records,'slots':n,'summary':summary}
    (ROOT/'review/auxiliary_filter_directions.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
    for d,stats in summary.items():
        print(d,{k:(v['slots'],round(v['fast_fractional_loss_slope'],4),round(v['slow_fractional_loss_slope'],4)) for k,v in stats.items()})


if __name__=='__main__':
    main()

