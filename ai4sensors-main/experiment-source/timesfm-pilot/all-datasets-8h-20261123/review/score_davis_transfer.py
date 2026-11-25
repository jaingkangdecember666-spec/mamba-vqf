"""Evaluate the prespecified poster scene using the sole frozen predictor and identical native inertial rows."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean

import torch

from check_sources import ROOT
from davis_handheld import Davis,BASE
from lock_from_frontier import digest,read
from locked_replay import SelectedPredictor
from vqf_response_controls import fixed_scores


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('selection_lock',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    ds=Davis(args.selection_lock)
    model=SelectedPredictor(args.selection_lock)
    requested=read(BASE/'preprocessing-lock.json')['sealed_records']
    assert set(requested)=={name for name,role in ds.roles.items() if role=='sealed_poster'}
    destination=ROOT/'review/final-davis'
    destination.mkdir(exist_ok=True)
    rows=[]
    for name in requested:
        row={'dataset':'DAVIS','record':name,'scene':'poster','motion':name.split('_')[1]}
        try:
            raw,quality=ds.native(name)
            score=model.score(raw)
            controls=fixed_scores(raw)
            assert controls['scoring_indices_sha256']==score['scoring_indices_sha256']
            assert abs(controls['default_deg']-score['full_vqf_deg'])<1e-5
            row.update(status='scored',**score,fixed_vqf_1p5_deg=controls['fast_deg'],fixed_vqf_6_deg=controls['slow_deg'],reference_checks=quality)
        except (AssertionError,OSError,RuntimeError,ValueError) as exc:
            row.update(status='evaluation_failed',detail=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    fields=('dataset','record','scene','motion','status','detail','scored_frames','native_frames','rate_hz','scoring_indices_sha256',
            'full_vqf_deg','candidate_deg','physical_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg')
    with (destination/'sequences.csv').open('w',encoding='utf-8-sig',newline='')as f:
        out=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        out.writeheader();out.writerows(rows)
    scored=[r for r in rows if r['status']=='scored']
    summary={'selection_lock_sha256':digest(args.selection_lock),'preprocessing_lock_sha256':digest(BASE/'preprocessing-lock.json'),
         'completed_at':datetime.now().astimezone().isoformat(),'records':len(scored),'independent_scenes':1,
         'native_records_predeclared':len(requested),'total_scoring_frames':sum(r['scored_frames']for r in scored),
         'means':{k:fmean(r[k] for r in scored)for k in ('full_vqf_deg','candidate_deg','physical_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg')} if scored else {},
         'records_improved':sum(r['candidate_deg']<r['full_vqf_deg']for r in scored),'rows':rows,
         'scope':'All three poster captures. Three distinct motions in one scene, same lab and instrument as the six other source captures. No model selection on the poster results.'}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('poster',len(rows),summary['means'])


if __name__=='__main__':
    main()

