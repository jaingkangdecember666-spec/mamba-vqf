"""Test physical consistency of the prospective 96 YT training references, without running a candidate."""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

from check_sources import DATA,ROOT
import h5py


def angles(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),(a*b).sum(1)))


def examine(g):
    t=g['ts'][:].ravel().astype('f8')
    a,w,q=(g[k][:].astype('f8') for k in ('acc','gyr','qua'))
    lag=4
    finite=np.isfinite(q).all(1)&(np.linalg.norm(q,axis=1)>.8)&(np.linalg.norm(q,axis=1)<1.2)
    assert np.isfinite(t).all() and np.all(np.diff(t)>0.) and finite.all()
    rot=Rotation.from_quat(q,scalar_first=True)
    dt=t[lag:]-t[:-lag]
    relative=rot[:-lag].inv()*rot[lag:]
    rate=relative.as_rotvec()/dt[:,None]
    # Centered gyro over the same 20 ms span, used only to validate the reference.
    smoothed=(.5*(w[:-lag]+w[lag:])+sum(w[j:j+len(dt)] for j in range(1,lag)))/lag
    gyro_size=np.linalg.norm(smoothed,axis=1)
    ref_size=np.linalg.norm(rate,axis=1)
    paired=(dt>.015)&(dt<.025)&(gyro_size>.2)&(ref_size>.2)&np.isfinite(smoothed).all(1)
    error=angles(smoothed[paired],rate[paired])
    ratios=ref_size[paired]/gyro_size[paired]
    quiet=(t-t[0]>=10.)&(np.linalg.norm(w,axis=1)<.1)&(np.abs(np.linalg.norm(a,axis=1)-9.80665)<.3)
    level=angles(a[quiet],rot[quiet].inv().apply([0.,0.,1.])) if quiet.any() else np.array([])
    return {'rows':len(t),'median_clock_hz':1./np.median(np.diff(t)),'dynamic_pairs':int(paired.sum()),
            'gyro_vs_reference_median_deg':float(np.median(error)) if len(error) else None,
            'ref_over_gyro_median_speed':float(np.median(ratios)) if len(ratios) else None,
            'quiet_frames':int(quiet.sum()),'quiet_specific_force_vs_reference_median_deg':float(np.median(level)) if len(level) else None}


def main():
    frozen=ROOT/'review/yt_training_panel.json'
    panel=json.loads(frozen.read_bytes())
    results=[]
    with h5py.File(DATA/'YT_server/YT_server_200Hz.h5','r') as f:
        source=f['train']
        for row in panel['records']:
            data=examine(source[row['record']])
            assert data['rows']==row['native_frames']
            results.append({**row,**data})
    summary={}
    for label,selected in [('retained_24',[s for s in results if s['in_original_24']]),
                           ('additional_72',[s for s in results if not s['in_original_24']])]:
        stats={}
        for key in ('median_clock_hz','gyro_vs_reference_median_deg','ref_over_gyro_median_speed',
                    'quiet_specific_force_vs_reference_median_deg'):
            vals=[s[key] for s in selected if s[key] is not None]
            stats[key]={'records':len(vals),'p10_median_p90':np.percentile(vals,[10,50,90]).tolist() if vals else []}
        summary[label]=stats
    result={'scope':'Prespecified 96-per-trial YT training tracks. Physical agreement with the same sensor cannot establish '
          'an independent reference. No candidate scores or fitted model, no development or test rows read.',
          'training_panel_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'summary':summary,'records':results}
    (ROOT/'review/yt_training_reference.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print({k:{a:(u['records'],[round(v,3) for v in u['p10_median_p90']]) for a,u in r.items()} for k,r in summary.items()})


if __name__=='__main__':
    main()
