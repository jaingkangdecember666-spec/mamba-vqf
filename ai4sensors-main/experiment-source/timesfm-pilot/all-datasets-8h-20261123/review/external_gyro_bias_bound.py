"""Determine on external real sensor records whether VQF's default gyro-bias bound is an active constraint."""
import json
from collections import defaultdict
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from auxiliary_filter_directions import sources
from check_sources import ROOT
from davis_handheld import Davis
from optical_score import metric

UP=[0.,0.,1.]


def inspect(raw):
    x,hz,y,mask=raw
    valid=np.isfinite(x).all(1)
    past=np.maximum.accumulate(np.where(valid,np.arange(len(x)),-1))
    clean=x[past.clip(min=0)].copy();clean[past<0]=0.
    use=mask&valid
    a=np.ascontiguousarray(clean[:,:3]);w=np.ascontiguousarray(clean[:,3:])
    out={}
    for label,options in [('default',{}),('expanded',{'biasClip':8.})]:
        filter_=VQF(1./hz,**options)
        cap=filter_.params['biasClip']
        states=filter_.updateBatch(w,a)
        pred=Rotation.from_quat(states['quat6D'][use],scalar_first=True).inv().apply(UP)
        absolute=np.abs(np.degrees(states['bias'][use])).max(1)
        out[label]={'bias_clip_deg_s':cap,'tilt_rmse_deg':metric(pred,y[use]),
                    'any_axis_at_99pct_clip_fraction':float((absolute>=.99*cap).mean()),
                    'max_abs_bias_axis_deg_s':float(absolute.max()),'rest_fraction':float(states['restDetected'][use].mean())}
    return out


def main():
    rows=[]
    for d,n,raw in sources():
        rows.append({'domain':d,'record':n,**inspect(raw)})
    david=Davis()
    for n,role in david.roles.items():
        if role=='external_train':
            raw,_=david.native(n)
            rows.append({'domain':'DAVIS','record':n,**inspect(raw)})
    groups=defaultdict(list)
    for r in rows:groups[r['domain']].append(r)
    summary={}
    for d,r in groups.items():
        summary[d]={'records':len(r),'default_deg':fmean(x['default']['tilt_rmse_deg']for x in r),
            'expanded_8_deg_s_deg':fmean(x['expanded']['tilt_rmse_deg']for x in r),
            'records_helped':sum(x['expanded']['tilt_rmse_deg']<x['default']['tilt_rmse_deg']-1e-8 for x in r),
            'records_with_default_clip':sum(x['default']['any_axis_at_99pct_clip_fraction']>.001 for x in r),
            'record_mean_default_clipped_frame_fraction':fmean(x['default']['any_axis_at_99pct_clip_fraction']for x in r)}
    (ROOT/'review/external_gyro_bias_bound.json').write_text(json.dumps({'scope':'External training recordings only. Keep the normal 3 s accelerometer response. Compare stock VQF against the same filter with only the gyro bias bound increased from 2 to 8 deg/s. Each source recording has one vote. No learned model or target-domain prediction here.','summary':summary,'records':rows},indent=2)+'\n',encoding='utf8')
    print(summary)


if __name__=='__main__':
    main()

