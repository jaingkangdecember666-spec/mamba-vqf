"""Measure rotation-invariant predictive directions in the external training bank only."""
import json
import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from native_catalog import body_reference, load_external, yt_record


def training_records():
    inventory=json.loads((ROOT/'C001/inventory.json').read_text('utf8'))['records']
    specs=json.loads((ROOT/'C004/protocol.json').read_text('utf8'))['source_records']
    ix={(r['dataset'],r['record']):r for r in inventory}
    for spec in specs:
        d,n=spec['domain'],spec['record']
        r=ix[d,n]
        if d=='YT_Server':
            assert r['side']=='external_training_candidate' and r['official_split']=='train'
            yield d,n,yt_record(r)
        else:
            assert d in ('BROAD','Sassari') and r['side']=='historical_source_train'
            imu,quats,mask,hz=load_external({'domain':d,'name':n,'path':r['path']})
            truth,valid=body_reference(quats,[0.,0.,1.])
            yield d,n,(imu,hz,truth,mask&valid)
    directory=ROOT/'external-sources/euroc-original-inputs'
    for track in json.loads((directory/'source-audit.json').read_text('utf8'))['records']:
        with np.load(directory/track['prepared_file']) as data:
            signal=np.c_[data['acc_m_s2'],data['gyr_rad_s']]
            truth,valid=body_reference(data['alternative_wxyz'],[0.,0.,1.])
            yield 'EuRoC',track['record'],(signal,200.,truth,valid&data['movement']&data['alternative_valid'])


def moments(payload):
    signal,hz,truth,valid=payload
    assert len(truth)==len(signal)==len(valid)
    finite=np.isfinite(signal).all(1)
    previous=np.maximum.accumulate(np.where(finite,np.arange(len(finite)),-1))
    cleaned=signal[previous.clip(min=0)].copy();cleaned[previous<0]=0.
    acc=np.ascontiguousarray(cleaned[:,:3],dtype=np.float64)
    gyro=np.ascontiguousarray(cleaned[:,3:],dtype=np.float64)
    default=VQF(1./hz).updateBatch(gyro,acc)
    rot=Rotation.from_quat(default['quat6D'],scalar_first=True)
    slow=VQF(1./hz,tauAcc=6.).updateBatch(gyro,acc)
    alt=Rotation.from_quat(slow['quat6D'],scalar_first=True).inv().apply([0.,0.,1.])
    alt_xy=rot.apply(alt)[:,:2]
    force=rot.apply(acc)[:,:2]/9.80665
    decay=np.exp(-1./(hz*2.))
    averaged=lfilter([1.-decay],[1.,-decay],force,axis=0)
    valid=valid&finite&(np.arange(len(signal))/hz>=10.)&(~default['restDetected'])
    if valid.sum()<25:
        return None
    target=rot[valid].apply(truth[valid])[:,:2]
    ans={}
    for k,feature in (('horizontal_force_now',force),('horizontal_force_ema_2s',averaged),
                      ('slower_filter_tilt_direction',alt_xy)):
        x=feature[valid]
        ans[k]={'xx':float((x*x).sum(1).mean()),'xy':float((x*target).sum(1).mean()),
                'yy':float((target*target).sum(1).mean())}
    return {'n':int(valid.sum()),'moments':ans}


def main():
    table=[]
    for d,n,source in training_records():
        row=moments(source)
        if row is not None:table.append({'domain':d,'record':n,**row})
    by_domain={}
    for d in sorted({r['domain'] for r in table}):
        group=[r for r in table if r['domain']==d]
        details={}
        for k in group[0]['moments']:
            xx,xy,yy=[np.mean([r['moments'][k][v] for r in group]) for v in ('xx','xy','yy')]
            details[k]={'origin_slope':float(xy/xx),'cosine':float(xy/np.sqrt(xx*yy)),
                        'negative_record_slopes':sum(r['moments'][k]['xy']<0. for r in group)}
        by_domain[d]={'records':len(group),'directional_statistics':details}
    output={'scope':'One vote per external training record. There are no target-panel scores, candidate fits or '
                    'holdout reads. Each direction uses only measured IMU and past filter states. Statistics measure '
                    'dot products with the training-only true horizontal displacement in the default VQF frame; a '
                    'shared scalar slope is only a diagnostic of predictive direction.',
            'domains':by_domain,'records':table}
    (ROOT/'review/long_state_source_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print(by_domain)


if __name__=='__main__':
    main()
