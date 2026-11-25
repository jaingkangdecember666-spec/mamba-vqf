"""Run the preregistered EuRoC room-2 and RIDI test panel once with the selected predictor."""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

import torch

from locked_replay import SelectedPredictor,sha256
from supplemental_data import OUT,Supplemental
from check_sources import ROOT
from vqf_response_controls import fixed_scores


def write_json(path,value):
    stage=path.with_suffix('.tmp')
    stage.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    stage.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('selection_lock',type=Path)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    database=Supplemental(args.selection_lock)
    predictor=SelectedPredictor(args.selection_lock,args.device)
    cases=database.cases()
    assert Counter(d for d,*_ in cases)=={'RIDI':25,'EuRoC':3}
    destination=ROOT/'review/final-supplemental'
    destination.mkdir(exist_ok=True)
    evidence={'selection_lock':sha256(args.selection_lock),'protocol':sha256(OUT/'protocol.json'),
              'source_manifest':sha256(OUT/'source-files.json'),'reader':sha256(Path(__file__)),
              'adapter':sha256(Path(__file__).with_name('supplemental_data.py')),
              'fixed_vqf':sha256(Path(__file__).with_name('vqf_response_controls.py')),
              'replay':sha256(Path(__file__).with_name('locked_replay.py')),
              'mixture':sha256(Path(__file__).with_name('mixture_replay.py'))}
    manifest=destination/'run.json'
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf8'))
        if old['hashes']!=evidence or old['device']!=args.device:
            raise ValueError('A different frozen run already owns these results')
    else:
        old={'started':datetime.now().astimezone().isoformat(),'hashes':evidence,'device':args.device}
        write_json(manifest,old)
    rows=[]
    for i,(domain,name,group,setting) in enumerate(cases):
        path=destination/f'row-{i+1:02}.json'
        key=domain,name
        if path.exists():
            row=json.loads(path.read_text(encoding='utf8'))
            if (row['dataset'],row['record'])!=key or row['selection_sha256']!=evidence['selection_lock']:
                raise ValueError('A recovered row belongs to another source or model')
        else:
            tick=perf_counter()
            row={'dataset':domain,'record':name,'group':group,'setting':setting,'selection_sha256':evidence['selection_lock']}
            if key in database.missing:
                row.update(status='source_unavailable',detail=database.missing[key])
            else:
                try:
                    raw,quality=database.native(domain,name)
                    score=predictor.score(raw)
                    fixed=fixed_scores(raw)
                    assert score['scoring_indices_sha256']==fixed['scoring_indices_sha256'] and score['scored_frames']==fixed['scored_frames']
                    assert abs(score['full_vqf_deg']-fixed['default_deg'])<1e-5
                    row.update(score,quality=quality,status='scored',fixed_vqf_1p5_deg=fixed['fast_deg'],fixed_vqf_6_deg=fixed['slow_deg'])
                except (ValueError,OSError,RuntimeError,AssertionError) as exc:
                    row.update(status='evaluation_failed',detail=f'{type(exc).__name__}: {exc}')
            row.update(seconds=perf_counter()-tick,completed=datetime.now().astimezone().isoformat())
            write_json(path,row)
            print(domain,name,row['status'],round(row['seconds'],2),flush=True)
        rows.append(row)
    columns=('dataset','record','group','setting','status','scored_frames','scoring_indices_sha256',
             'full_vqf_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg','physical_deg','candidate_deg','native_frames','rate_hz','seconds','detail')
    with (destination/'sequences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    domains={}
    for domain in ('RIDI','EuRoC'):
        relevant=[r for r in rows if r['dataset']==domain]
        scored=[r for r in relevant if r['status']=='scored']
        summary={'requested':len(relevant),'statuses':dict(Counter(r['status'] for r in relevant))}
        if scored:
            summary.update(means={k:fmean(r[k] for r in scored) for k in ('full_vqf_deg','physical_deg','candidate_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg')},
                scored_frames=sum(r['scored_frames'] for r in scored),improving_records=sum(r['candidate_deg']<r['full_vqf_deg'] for r in scored))
        domains[domain]=summary
    write_json(destination/'summary.json',{**old,'last_record_completed':max(r['completed'] for r in rows),
          'domains':domains,'row_hashes':{f'{i+1:02}':sha256(destination/f'row-{i+1:02}.json') for i in range(len(rows))}})
    print(domains)


if __name__=='__main__':
    main()
