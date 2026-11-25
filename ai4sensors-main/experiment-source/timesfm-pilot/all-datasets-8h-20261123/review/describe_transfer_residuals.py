"""Describe the actual frozen neural correction versus its physical ablation on supplementary native rows."""
import csv
import hashlib
import json
from statistics import fmean

import numpy as np
import torch

from anchor_blend_features import included_degrees
from davis_handheld import Davis
from lock_from_frontier import ROOT,REVIEW,read,digest
from locked_replay import SelectedPredictor
from penn_handheld import Penn
from supplemental_data import Supplemental


def main():
    lock=ROOT/'selection-lock.json';torch.set_num_threads(1)
    model=SelectedPredictor(lock)
    extra=Supplemental(lock);davis=Davis(lock);penn=Penn(lock)
    paths=[REVIEW/n/'sequences.csv'for n in ('final-supplemental','final-davis','final-penn')]
    rows=[]
    for p in paths:
        with p.open(encoding='utf-8-sig',newline='')as f:rows.extend(csv.DictReader(f))
    assert len(rows)==33
    result=[]
    for row in rows:
        d,n=row['dataset'],row['record']
        if not row.get('candidate_deg'):continue
        if d=='DAVIS':raw,quality=davis.native(n)
        elif d=='PennCOSYVIO':raw,quality=penn.native(n)
        else:raw,quality=extra.native(d,n)
        x,hz,y,mask=raw;ix=np.flatnonzero(mask&np.isfinite(x).all(1))
        assert len(ix)==int(row['scored_frames']) and hashlib.sha256(ix.astype('<i8').tobytes()).hexdigest()==row['scoring_indices_sha256']
        v=model.predict(x,hz,ix);a,b=v['physical_deg'],v['candidate_deg'];truth=y[ix]
        p_error=included_degrees(a,truth);n_error=included_degrees(b,truth)
        rmse=lambda x:float(np.sqrt(np.mean(x*x)))
        assert abs(rmse(p_error)-float(row['physical_deg']))<1.e-5
        assert abs(rmse(n_error)-float(row['candidate_deg']))<1.e-5
        change=included_degrees(a,b);active=change>.01
        step=b-a*np.sum(a*b,axis=1)[:,None];target=truth-a*np.sum(a*truth,axis=1)[:,None]
        valid_direction=active&(np.linalg.norm(target,axis=1)>1.e-8)
        point_away=(np.sum(step*target,axis=1)<0)
        item={'dataset':d,'record':n,'scored_frames':len(ix),'physical_deg':rmse(p_error),'candidate_deg':rmse(n_error),
              'record_delta_deg':rmse(n_error)-rmse(p_error),'explicit_correction_fraction':float(active.mean()),
              'median_correction_deg':float(np.median(change)),'p95_correction_deg':float(np.quantile(change,.95)),
              'direction_eligible_native_rows':int(valid_direction.sum()),
              'away_fraction_given_active':float(point_away[valid_direction].mean())if valid_direction.any()else None,
              'lower_error_fraction_given_active':float((n_error[active]<p_error[active]).mean())if active.any()else None}
        result.append(item)
    domains={}
    for d in sorted({r['dataset']for r in result}):
        part=[r for r in result if r['dataset']==d]
        summary={k:fmean(r[k]for r in part if r[k] is not None)for k in ('physical_deg','candidate_deg','record_delta_deg','explicit_correction_fraction',
                        'median_correction_deg','p95_correction_deg','away_fraction_given_active','lower_error_fraction_given_active')}
        domains[d]={'records':len(part),'records_improved_over_physical':sum(r['record_delta_deg']<0 for r in part),**summary}
    audit={'selection_lock_sha256':digest(lock),'external_table_hashes':{p.relative_to(ROOT).as_posix():digest(p)for p in paths},
           'scope':'Descriptive after locking. Compare fixed predictions on exactly their scored native rows, then average each record equally. A correction over 0.01 degree counts as active. The direction test uses the tangent at the physical prediction and the tangent toward its reference. No parameters fitted or selected.', 'records':result,'domains':domains}
    (REVIEW/'transfer-correction-directions.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    for d,r in domains.items():print(d,r['records_improved_over_physical'],r['records'],round(r['away_fraction_given_active'],3),round(r['record_delta_deg'],5))


if __name__=='__main__':main()
