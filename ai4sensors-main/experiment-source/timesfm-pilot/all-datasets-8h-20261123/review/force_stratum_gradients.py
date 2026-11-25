"""Measure the signed need for more stable-force correction after C017 on external training inputs."""
import json
from itertools import chain,islice

import numpy as np
from scipy.signal import lfilter

from check_sources import ROOT
from filter_response_audit import phones
from long_state_source_audit import training_records


def main():
    lock=json.loads((ROOT/'C017/coefficient-lock.json').read_bytes())
    with np.load(ROOT/'C017/training-slots.npz') as archive:
        a={k:archive[k] for k in archive.files}
    k=lock['k']
    f=a['force'];cap=np.tan(np.deg2rad(6.))
    fn=np.linalg.norm(f,axis=1)
    bounded=cap/np.maximum(fn,1e-15)
    base=a['C014_B']+np.minimum(k,bounded)[:,None]*f
    length=np.linalg.norm(base,axis=1)
    direction=base/length[:,None]
    truth=a['truth']
    discrepancy=direction-truth
    gradient=2.*np.sum(discrepancy*(f-direction*np.sum(f*direction,axis=1)[:,None]),axis=1)/length
    gradient[k>=bounded]=0.
    loss=np.sum(discrepancy**2,axis=1)
    enabled=a['support'].astype(bool)
    loads=np.zeros(len(f));speeds=np.zeros(len(f));names=[]
    for i,(domain,name,raw) in enumerate(chain(islice(training_records(),63),phones())):
        x,hz,*_=raw
        observed=np.isfinite(x).all(1)
        use=np.flatnonzero(observed)
        end=a['native_input_last'][a['source_record_ordinal']==i]
        previous=np.searchsorted(use,end,side='right')-1
        assert (previous>=0).all()
        alpha=-np.expm1(-1./(.25*hz))
        local=x[use]
        avg_load=np.sqrt(lfilter([alpha],[1.,alpha-1.],(np.linalg.norm(local[:,:3],axis=1)/9.80665-1.)**2))
        avg_speed=np.sqrt(lfilter([alpha],[1.,alpha-1.],np.linalg.norm(local[:,3:],axis=1)**2))
        match=a['source_record_ordinal']==i
        loads[match]=avg_load[previous]
        speeds[match]=avg_speed[previous]
        names.append(domain)
    row_domains=np.asarray(names)[a['source_record_ordinal']]
    strata={'all_stable':enabled,'force_rms_lt_0.02g':enabled&(loads<.02),'force_rms_ge_0.02g':enabled&(loads>=.02),
            'gyro_rms_lt_0.05':enabled&(speeds<.05),'gyro_rms_0.05_to_0.5':enabled&(speeds>=.05)&(speeds<.5),
            'gyro_rms_ge_0.5':enabled&(speeds>=.5)}
    results={}
    for d in dict.fromkeys(names):
        src=(row_domains==d)
        w=a['weight']
        assert abs(w[src].sum()-1.)<1e-9
        total=w[src]@loss[src]
        results[d]={}
        for label,keep in strata.items():
            sel=src&keep
            dloss=float(w[sel]@gradient[sel]/total)
            results[d][label]={'training_slots':int(sel.sum()),'records':len(np.unique(a['source_record_ordinal'][sel])),
                               'd_relative_whole_domain_loss_per_unit_gain':dloss}
    output={'scope':'External four-source C015 release ledger only. Exact C017 operating point. Conditions use past 0.25 s '
                 'EW RMS of measured angular speed and relative acceleration-magnitude departure. Cells report the derivative '
                 'of the entire record-balanced domain loss if only that stratum receives more correction. This diagnostic '
                 'does not alter any coefficient. Source sample counts indicate when a sign has little support.',
            'k':k,'domains':results}
    (ROOT/'review/force_stratum_gradients.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print({d:{k:(r['training_slots'],round(r['d_relative_whole_domain_loss_per_unit_gain'],5)) for k,r in cols.items()}
           for d,cols in results.items()})


if __name__=='__main__':
    main()
