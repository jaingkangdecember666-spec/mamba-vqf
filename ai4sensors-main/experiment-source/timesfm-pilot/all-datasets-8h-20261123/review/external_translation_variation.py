"""Separate the measured low-frequency horizontal translation from three external IMU recordings."""
import json

import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.spatial.transform import Rotation

from check_sources import ROOT
from euroc_paired_sources import NAMES,paired
from lock_from_frontier import digest,read
from vqf_response_controls import fixed_scores


def motion(name):
    if name not in NAMES:raise ValueError('Expected one of three admitted V1 source recordings')
    directory=ROOT/'external-sources/euroc-original-inputs'
    source=next(r for r in read(directory/'source-audit.json')['records'] if r['record']==name)
    p=directory/source['prepared_file']
    assert digest(p)==source['prepared_sha256']
    states=directory/(name+'.csv')
    with np.load(p)as f:
        t=f['utc_ns'];q=f['alternative_wxyz'].copy();good=f['alternative_valid'].copy()
        q[~good]=f['original_wxyz'][~good]
    raw=np.loadtxt(states,delimiter=',',comments='#',dtype=[('time','i8'),('values','f8',(16,))])
    times=raw['time'];velocity=raw['values'][:,7:10]
    assert np.all(np.diff(times)>0) and t.dtype==np.int64
    x=(t-t[0])*1e-9
    state_times=(times-t[0])*1e-9
    aligned=np.column_stack([np.interp(x,state_times,velocity[:,axis])for axis in (0,1)])
    hz=200.;sos=butter(3,1.,fs=hz,output='sos')
    horizontal=np.gradient(sosfiltfilt(sos,aligned,axis=0),1./hz,axis=0)
    qa=Rotation.from_quat(q,scalar_first=True)
    delta=qa.inv().apply(np.c_[horizontal,np.zeros(len(t))])
    records=paired(name)
    world=qa.apply(records[0][0][:,:3])[:,:2]
    measured=sosfiltfilt(sos,world,axis=0)
    valid=records[0][3]&good&(x>=state_times[0]+2)&(x<=state_times[-1]-2)
    assert valid.sum()>1000 and np.all(np.isfinite(delta))
    a=horizontal[valid];b=measured[valid]
    correlation=[float(np.corrcoef(a[:,j],b[:,j])[0,1])for j in (0,1)]
    scale=float(np.einsum('ij,ij->',a,b)/np.einsum('ij,ij->',a,a))
    info={'source_state_sha256':digest(states),'source_stream_sha256':digest(p),'usable_rows':int(valid.sum()),
          'velocity_reference_median_hz':float(1e9/np.median(np.diff(times))),
          'horizontal_component_correlation_xy':correlation,'IMU_on_velocity_derivative_scale':scale,
          'horizontal_accel_rms_m_s2':float(np.sqrt(np.mean(np.sum(a*a,axis=1)))),
          'horizontal_accel_p95_m_s2':float(np.quantile(np.linalg.norm(a,axis=1),.95))}
    return records,delta,valid,info


def main():
    dataset=[]
    for name in NAMES:
        streams,delta,valid,info=motion(name)
        info['record']=name;info['conditions']={}
        for label,(xx,hz,ref,_) in zip(('raw_gyro','fixed_prefix_bias_removed'),streams):
            for tag,factor in [('unaltered',0.),('subtract_horizontal_translation',-1.),('double_horizontal_translation',1.)]:
                data=xx.copy();data[:,:3]+=factor*delta
                np.testing.assert_array_equal(data[:,3:],xx[:,3:])
                info['conditions'][label+'/'+tag]=fixed_scores((data,hz,ref,valid))
        assert len({v['scoring_indices_sha256']for v in info['conditions'].values()})==1
        dataset.append(info)
    out={'scope':'Offline source-side physical diagnostic only. Derivative of the published world-frame horizontal velocity after a 1 Hz zero-phase third-order Butterworth filter, expressed in the IMU frame via the source orientation. The original six-axis signals are the measured acquisitions; adding or subtracting this measured component constructs paired variations. Angular motion, vertical world acceleration, all gyro rows, optical gravity and common scoring mask are identical across each pair. An extra two-second boundary on reference velocity supports filtering. No target data, new recordings, model training or deployable reference inputs.',
         'records':dataset,'condition_means':{c:{k:float(np.mean([r['conditions'][c][k]for r in dataset]))for k in ('fast_deg','default_deg','slow_deg')}for c in dataset[0]['conditions']}}
    (ROOT/'review/external_translation_variation.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for r in dataset:print(r['record'],r['horizontal_component_correlation_xy'],r['IMU_on_velocity_derivative_scale'])
    print(out['condition_means'])


if __name__=='__main__':main()
