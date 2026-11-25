"""Evaluate one reference-assisted gyro-bias training augmentation on the six admitted DAVIS source captures."""
import json

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

from davis_handheld import Davis,BASE
from check_sources import ROOT
from lock_from_frontier import digest
from vqf_response_controls import fixed_scores


def estimate_bias(t,gyro,tp,optical):
    # Use only low-turn optical intervals, never a target-domain pose. This estimate is an external training variant.
    q=Rotation.from_quat(optical)
    dt=tp[1:]-tp[:-1]
    v=(q[:-1].inv()*q[1:]).as_rotvec()/dt[:,None]
    integral=cumulative_trapezoid(gyro,t,axis=0,initial=0.)
    at=np.column_stack([np.interp(tp,t,integral[:,j]) for j in range(3)])
    measured_turn=np.diff(at,axis=0)/dt[:,None]
    use=(tp[:-1]>=t[0])&(tp[1:]<=t[-1])&(dt>.002)&(dt<.012)&(np.linalg.norm(v,axis=1)<.12)
    if use.sum()<30:
        return None,{'quiet_optical_intervals':int(use.sum())}
    residual=measured_turn[use]-v[use]
    bias=np.median(residual,axis=0)
    # Independent intervals from each time half, plus an interleaving check.
    half=(tp[:-1][use]-tp[0])/(tp[-1]-tp[0])<.5
    splits=[np.median(part,axis=0) for part in (residual[half],residual[~half]) if len(part)>=30]
    return bias,{'quiet_optical_intervals':int(use.sum()),'fraction_in_first_half':float(half.mean()),
                 'median_abs_scatter_deg_s':np.degrees(np.median(np.abs(residual-bias),axis=0)).tolist(),
                 'time_half_bias_deg_s':[np.degrees(x).tolist()for x in splits],
                 'odd_even_estimate_gap_deg_s':float(np.degrees(np.linalg.norm(np.median(residual[::2],0)-np.median(residual[1::2],0))))}


def main():
    ds=Davis()
    out=[];acquisitions={}
    for name,role in ds.roles.items():
        if role!='external_train':continue
        raw,_=ds.native(name);files=ds.entries[name]['members']
        imu=np.loadtxt(BASE/files['imu.txt']['path']);opt=np.loadtxt(BASE/files['groundtruth.txt']['path'])
        np.testing.assert_array_equal(imu[:,1:],raw[0])
        bias,qa=estimate_bias(imu[:,0],imu[:,4:],opt[:,0],opt[:,4:])
        acquisitions[name]=(bias,qa,raw)
    for name,(local,qa,raw) in acquisitions.items():
        other=[b for n,(b,detail,_) in acquisitions.items() if n!=name and b is not None and detail['quiet_optical_intervals']>=150]
        if len(other)<2:
            out.append({'record':name,'status':'insufficient_other_captures','qualified_other_captures':len(other),'independent_intervals':qa})
            continue
        bias=np.median(other,axis=0)
        corrected=raw[0].copy();corrected[:,3:]-=bias
        before=fixed_scores(raw);after=fixed_scores((corrected,*raw[1:]))
        assert before['scoring_indices_sha256']==after['scoring_indices_sha256']
        out.append({'record':name,'status':'scored','bias_rad_s':bias.tolist(),'bias_deg_s':np.degrees(bias).tolist(),
                    'own_quiet_bias_deg_s':np.degrees(local).tolist() if local is not None else None,
                    'qualified_other_captures':len(other),
                    'independent_intervals':qa,'raw':before,'external_reference_centered':after})
    paired=[r for r in out if r['status']=='scored']
    means={condition:{k:float(np.mean([r[condition][k] for r in paired])) for k in ('fast_deg','default_deg','slow_deg')}
           for condition in ('raw','external_reference_centered')} if paired else {}
    dest=ROOT/'review/davis_bias_counterfactual.json'
    dest.write_text(json.dumps({'scope':'Same six source motions, original compared with removing the median quiet optical gyro bias from the other training acquisitions with at least 150 qualifying optical intervals (at least two other acquisitions required). Low support is reported rather than filled from the same recording. Six independent captures. No model or target-domain scores.','preprocessing_lock_sha256':digest(BASE/'preprocessing-lock.json'),'paired_captures':len(paired),'means':means,'source_rows':out},indent=2)+'\n',encoding='utf8')
    print(means)


if __name__=='__main__':main()
