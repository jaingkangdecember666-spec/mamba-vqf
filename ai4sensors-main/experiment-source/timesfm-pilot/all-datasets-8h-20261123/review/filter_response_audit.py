"""Measure two changes to the gravity filter response on training sources only."""
import json
from collections import defaultdict

import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from long_state_source_audit import training_records


def phones():
    d=ROOT/'external-sources/ridi-phone-train'
    for row in json.loads((d/'source-audit.json').read_text())['rows']:
        with np.load(d/row['file']) as f:
            signal=np.c_[f['acc_m_s2'],f['gyr_rad_s']]
            ref=Rotation.from_quat(f['tango_wxyz'],scalar_first=True).inv().apply([0.,0.,1.])
            valid=f['imu_observed'] & f['reference_valid']
        yield 'RIDI',row['record'],(signal,200.,ref,valid)


def record_values(payload):
    signal,hz,ref,valid=payload
    obs=np.isfinite(signal).all(1)
    prev=np.maximum.accumulate(np.where(obs,np.arange(len(obs)),-1))
    fill=signal[prev.clip(min=0)].copy()
    fill[prev<0]=0.
    acc=np.ascontiguousarray(fill[:,:3]);gyro=np.ascontiguousarray(fill[:,3:])
    alpha=-np.expm1(-1./hz)
    effective=np.sqrt(lfilter([alpha],[1.,alpha-1.],(np.linalg.norm(acc,axis=1)/9.80665-1.)**2))
    good=valid & obs & (np.arange(len(signal))/hz>=10.)
    sels={'all':good,'low_1s_force_rms_below_0.05g':good&(effective<.05),'middle':good&(effective>=.05)&(effective<.2),
          'high_1s_force_rms_at_least_0.2g':good&(effective>=.2)}
    out={};record_features={}
    for label,kwargs in [('default',{}),('fast',{'tauAcc':1.5}),('slow',{'tauAcc':6.})]:
        state=VQF(1./hz,**kwargs).updateBatch(gyro,acc)
        quat=state['quat6D']
        if label=='default':
            record_features={'specific_force_rms_g_median':float(np.median(effective[good])),
                             'estimated_gyro_bias_rad_s_median':float(np.median(np.linalg.norm(state['bias'][good],axis=1))),
                             'gyro_norm_rad_s_median':float(np.median(np.linalg.norm(gyro[good],axis=1))),
                             'rest_fraction':float(state['restDetected'][good].mean())}
        up=Rotation.from_quat(quat,scalar_first=True).inv().apply([0.,0.,1.])
        deg=np.degrees(np.arctan2(np.linalg.norm(np.cross(up,ref),axis=1),(up*ref).sum(1)))
        for kind,mask in sels.items():
            if mask.sum()<100:continue
            out.setdefault(kind,{'rows':int(mask.sum())})[label]=float(np.sqrt(np.mean(deg[mask]**2)))
    return out,record_features


def main():
    from itertools import chain
    reports=defaultdict(list)
    for domain,name,raw in chain(training_records(),phones()):
        masks,features=record_values(raw)
        reports[domain].append({'record':name,'masks':masks,'features':features})
    sets={}
    for d,records in reports.items():
        sets[d]={}
        for mask in sorted({k for r in records for k in r['masks']}):
            values=[r['masks'][mask] for r in records if mask in r['masks']]
            means={k:float(np.mean([r[k] for r in values])) for k in ('default','fast','slow')}
            sets[d][mask]={'records':len(values),'source_macro_deg':means,
                'wins_vs_default':{k:sum(r[k]<r['default'] for r in values) for k in ('fast','slow')}}
    feature_summary={d:{k:np.quantile([r['features'][k] for r in records],[.1,.5,.9]).tolist()
                       for k in records[0]['features']} for d,records in reports.items()}
    out={'scope':'External train sources, one vote per record within a source and stratum. Raw six-axis inputs; '
                 'compared two fixed factor-of-two changes to the default 3-second acceleration time constant. '
                 'Probes do not update parameters and use no eight-domain development or final scores.',
         'summary':sets,'record_feature_p10_median_p90':feature_summary,'records':reports}
    (ROOT/'review/filter_response_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for d,items in sets.items(): print(d,items['all'])


if __name__=='__main__':
    main()
