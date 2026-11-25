"""Check the sealed main panel, per-record results, coverage extensions, and 14 prespecified raw replays."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import fmean,median

import numpy as np
import torch

from check_sources import ROOT
from group_uncertainty import paired_by_group
from locked_replay import SelectedPredictor,sha256
from native_catalog import Catalog


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def identity(row):
    return row['dataset'],row['record']


def summarize(rows):
    def scored(r):
        return r.get('full_vqf_deg') not in (None,'') and r.get('candidate_deg') not in (None,'')
    successful=[r for r in rows if scored(r)]
    summary={'requested':len(rows),'scored':len(successful),'missing':[{'record':r['record'],'status':r.get('status'),
              'reason':r.get('error') or r.get('reason') or r.get('detail')} for r in rows if not scored(r)]}
    if not successful:
        return summary
    for r in successful:
        values=[float(r[k]) for k in ('full_vqf_deg','candidate_deg','physical_deg')]
        assert all(np.isfinite(a) and a>=0. for a in values)
        assert int(r['scored_frames'])>0 and len(r['scoring_indices_sha256'])==64
    means={k:fmean(float(r[k]) for r in successful) for k in ('full_vqf_deg','candidate_deg','physical_deg')}
    diffs=[float(r['candidate_deg'])-float(r['full_vqf_deg']) for r in successful]
    worst=max(successful,key=lambda r:float(r['candidate_deg'])-float(r['full_vqf_deg']))
    summary.update(means=means,improvement_percent=100*(1.-means['candidate_deg']/means['full_vqf_deg']),
                   paired_delta_median_deg=median(diffs),ties=sum(d==0. for d in diffs),
                   worse_records=sum(d>0. for d in diffs),scored_frames=sum(int(r['scored_frames']) for r in successful),
                   largest_regression_record=worst['record'],largest_candidate_minus_default_deg=max(diffs),
                   groups=paired_by_group(successful,'full_vqf_deg','candidate_deg'))
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('selection_lock',type=Path)
    parser.add_argument('main_csv',type=Path)
    parser.add_argument('additional_csv',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    predictor=SelectedPredictor(args.selection_lock)
    lock=predictor.lock
    for relative,expected in lock['evidence_sha256'].items():
        assert sha256(ROOT/relative)==expected,relative
    catalog=Catalog(args.selection_lock)
    organized={}
    all_results={}
    for tag,p,rows in [('formal',args.main_csv,lock['formal_cases']),
                       ('additional',args.additional_csv,lock['full_coverage_additional_cases'])]:
        recorded=csv_rows(p)
        wanted={identity(r):r for r in rows}
        assert len(recorded)==len(wanted) and set(wanted)=={identity(r) for r in recorded},tag
        ordered={}
        for r in recorded:
            key=identity(r)
            assert key not in all_results and r['group']==wanted[key]['group'],key
            ordered[key]=r
            all_results[key]=r
        organized[tag]=ordered
    official=list(organized['formal'].values())
    main_stats={d:summarize([r for r in official if r['dataset']==d]) for d in sorted({r['dataset'] for r in official})}
    complete={}
    for d,n in [('SHICE',57),('BROAD',39),('Sassari',18)]:
        rows=[r for r in all_results.values() if r['dataset']==d]
        assert len(rows)==n
        complete[d]=summarize(rows)
    own_by_device={g:summarize([r for r in all_results.values() if r['dataset']=='SHICE' and r['group']==g])
                   for g in ('R1','R2','R3','R4','R5','R6')}
    spec=json.loads((ROOT/'review/final-anchor-spec.json').read_bytes())
    replays=[]
    for dataset,records in spec['anchors'].items():
        for record in records:
            published=organized['formal'][dataset,record]
            row={'dataset':dataset,'record':record,'status':published['status']}
            if published.get('candidate_deg'):
                raw,_=catalog.native(dataset,record)
                independent=predictor.score(raw)
                assert int(published['scored_frames'])==independent['scored_frames']
                assert published['scoring_indices_sha256']==independent['scoring_indices_sha256']
                delta=max(abs(float(published[k])-independent[k]) for k in ('full_vqf_deg','candidate_deg','physical_deg'))
                assert delta<2e-4,(dataset,record,delta)
                row.update(independent=independent,max_abs_error_deg=delta)
            replays.append(row)
            print(dataset,record,row.get('max_abs_error_deg',row['status']),flush=True)
    assert len(replays)==spec['n_anchors']==14
    result={'lock_sha256':sha256(args.selection_lock),'main_csv_sha256':sha256(args.main_csv),
            'main_table':args.main_csv.resolve().relative_to(ROOT).as_posix(),
            'additional_table':args.additional_csv.resolve().relative_to(ROOT).as_posix(),
            'additional_csv_sha256':sha256(args.additional_csv),'protocol_denominators':spec['records_in_main'],
            'main':main_stats,'main_improved_domains':sum(v.get('improvement_percent',0.)>0. for v in main_stats.values()),
            'main_complete_domains':sum(v['requested']==v['scored'] for v in main_stats.values()),
            'full':complete,'self_devices':own_by_device,'preregistered_raw_replays':replays,
            'raw_replays_scored':sum('max_abs_error_deg' in v for v in replays),
            'overall_rows':dict(Counter(r['status'] for r in all_results.values()))}
    (ROOT/'review/final-verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print('formal wins',result['main_improved_domains'],'complete domains',result['main_complete_domains'])


if __name__=='__main__':
    main()
