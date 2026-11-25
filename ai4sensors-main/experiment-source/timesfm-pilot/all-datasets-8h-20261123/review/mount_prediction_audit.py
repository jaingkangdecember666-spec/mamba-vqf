"""Rotate the same native development measurements and references, then replay the unchanged estimators."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from check_sources import ROOT,TRANSFER,weights
from mixture_replay import replay
from native_catalog import Catalog
from optical_score import metric


def speed_strata(signal,ids,truth,pred,hz):
    rates=np.linalg.norm(signal[ids,3:],axis=1)
    bins=np.searchsorted([.05,.5,2.],rates,side='right')
    rows={}
    for i,name in enumerate(('below_0p05','0p05_to_0p5','0p5_to_2','at_least_2')):
        take=bins==i
        n=int(take.sum())
        part={'frames':n,'minimum_frames':int(np.ceil(hz-1e-6)),'supported':bool(n>=np.ceil(hz-1e-6))}
        if part['supported']:
            part['rmse_deg']={k:metric(v[take],truth[take])for k,v in pred.items()}
        rows[name]=part
    assert sum(r['frames']for r in rows.values())==len(ids)
    return rows


def audit_case(catalog,row,predict,candidate_key):
    d,name=row['dataset'],row['record']
    (signal,hz,truth,keep),_=catalog.native(d,name)
    ids=np.flatnonzero(keep & np.isfinite(signal).all(axis=1))
    assert hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()==row['scoring_indices_sha256']
    rotations={'original':Rotation.identity(),
               'x90':Rotation.from_rotvec(np.pi/2*np.array([1.,0.,0.])),
               'cycle_xyz':Rotation.from_rotvec((2*np.pi/3)*np.ones(3)/np.sqrt(3)),
               'oblique_8deg':Rotation.from_rotvec(np.deg2rad(8)*np.array([1.,-2.,1.])/np.sqrt(6))}
    results={}
    for key,r in rotations.items():
        rotated=np.c_[r.apply(signal[:,:3]),r.apply(signal[:,3:])]
        target=r.apply(truth[ids])
        pred=predict(rotated,hz,ids)
        scores={k:metric(v,target) for k,v in pred.items()}
        if key=='original':
            original=pred
            for k in scores:
                assert abs(float(row[k])-scores[k])<1e-5,(d,name,k)
        motion={k:metric(r.inv().apply(v),original[k]) for k,v in pred.items()}
        results[key]={'paired_rmse_deg':scores,'candidate_minus_default_deg':scores[candidate_key]-scores['full_vqf_deg'],
                      'angle_rmse_vs_predictions_in_original_axes_deg':motion}
    record={'dataset':d,'record':name,'group':row['group'],'frames':len(ids),
            'scoring_indices_sha256':row['scoring_indices_sha256'],'rotations':results}
    if d=='SHICE' and candidate_key=='candidate_deg':
        record['native_gyro_speed_bins']=speed_strata(signal,ids,truth[ids],original,hz)
    return record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--selection-lock',type=Path)
    ap.add_argument('--main-csv',type=Path)
    args=ap.parse_args()
    if bool(args.selection_lock)!=bool(args.main_csv):
        ap.error('Use both --selection-lock and --main-csv for the R5/R6 final replay')
    torch.set_num_threads(2)
    p=args.main_csv if args.selection_lock else ROOT/'C014/sequences.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    catalog=Catalog(args.selection_lock)
    if args.selection_lock:
        from locked_replay import SelectedPredictor,sha256
        model=SelectedPredictor(args.selection_lock)
        predict=model.predict
        candidate='candidate_deg'
        candidates=[r for r in rows if r['dataset']=='SHICE']
        assert len(candidates)==20 and {r['group'] for r in candidates}=={'R5','R6'}
        chosen=[r for r in candidates if r.get('candidate_deg')]
        scope='Twenty R5/R6 records selected as a block before formal scoring. Same physical captures and single frozen predictor under three prespecified sensor coordinate rotations. '
        target=ROOT/'review/final_self_mount_predictions.json'
    else:
        heads={'B6':(weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),.25,'always'),
               'B7':(weights(ROOT/'C013/seed-C006-readout.pt','mamba_v1'),.25,'rest_off')}
        predict=lambda signal,hz,ids:replay(signal,hz,heads,{'B_deg':{'B6':.5,'B7':.5}},ids)
        candidate='B_deg'
        picks={}
        for row in sorted(rows,key=lambda a:(a['dataset'],a['record'])):
            key=(row['dataset'],row['group'] if row['dataset']=='SHICE' else '')
            picks.setdefault(key,row)
        assert len(picks)==11
        chosen=picks.values()
        scope='C014-B already fixed. Eleven development records chosen by first name within each domain, or within each SHICE device. '
        target=ROOT/'review/development_mount_predictions.json'
    out=[]
    for row in chosen:
        item=audit_case(catalog,row,predict,candidate)
        out.append(item)
        print(item['dataset'],item['record'],[round(v['candidate_minus_default_deg'],5) for v in item['rotations'].values()],flush=True)
    report={'scope':scope+
              'Use three rigid changes of sensor axes fixed before reading the results. Rotate all six sensor channels and the body-frame scoring '
              'reference by the same proper rotation. Run each estimator from its ordinary initial state. All scoring indices unchanged. '
              'This is a sensitivity analysis on existing physical motions.',
            'rows_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':out}
    if args.selection_lock:
        report['selection_lock_sha256']=sha256(args.selection_lock)
        report['missing']=[r['record'] for r in candidates if not r.get('candidate_deg')]
        report['by_device']={}
        report['motion_by_device']={}
        for group in ('R5','R6'):
            group_rows=[r for r in out if r['group']==group]
            report['by_device'][group]={}
            report['motion_by_device'][group]={}
            for name in ('below_0p05','0p05_to_0p5','0p5_to_2','at_least_2'):
                all_bins=[r['native_gyro_speed_bins'][name]for r in group_rows]
                supported=[s for s in all_bins if s['supported']]
                report['motion_by_device'][group][name]={'records':len(supported),
                   'observed_frames':sum(s['frames']for s in all_bins),'scored_frames':sum(s['frames']for s in supported),
                   'record_means_deg':{k:float(np.mean([s['rmse_deg'][k]for s in supported]))for k in ('full_vqf_deg','physical_deg','candidate_deg')} if supported else {},
                   'improved_records':sum(s['rmse_deg']['candidate_deg']<s['rmse_deg']['full_vqf_deg']for s in supported)}
            for case in ('original','x90','cycle_xyz','oblique_8deg'):
                delta=[r['rotations'][case]['candidate_minus_default_deg']for r in group_rows]
                report['by_device'][group][case]={'scored':len(delta),'improved_records':sum(v<0. for v in delta),
                    'candidate_minus_default_deg':float(np.mean(delta)) if delta else None}
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
