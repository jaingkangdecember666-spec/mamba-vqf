"""Score three predeclared causal VQF parameter settings at exactly the published native indices."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from lock_from_frontier import digest,read
from native_catalog import Catalog
from optical_score import metric


def fixed_scores(raw):
    signal,hz,truth,mask=raw
    observed=np.isfinite(signal).all(1)
    indices=np.flatnonzero(mask & observed)
    if not len(indices):
        raise ValueError('No published native rows')
    latest=np.maximum.accumulate(np.where(observed,np.arange(len(observed)),-1))
    clean=signal[latest.clip(min=0)].copy()
    clean[latest<0]=0.
    acc=np.ascontiguousarray(clean[:,:3]);gyro=np.ascontiguousarray(clean[:,3:])
    scores={}
    for name,opts in [('default_deg',{}),('fast_deg',{'tauAcc':1.5}),('slow_deg',{'tauAcc':6.0})]:
        state=VQF(1./hz,**opts).updateBatch(gyro,acc)
        up=Rotation.from_quat(state['quat6D'][indices],scalar_first=True).inv().apply([0.,0.,1.])
        scores[name]=metric(up,truth[indices])
    return {'scored_frames':len(indices),'scoring_indices_sha256':hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest(),**scores}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--development',action='store_true')
    ap.add_argument('--selection-lock',type=Path)
    ap.add_argument('--main-csv',type=Path)
    args=ap.parse_args()
    if args.development==bool(args.selection_lock) or bool(args.selection_lock)!=bool(args.main_csv):
        ap.error('Use --development, or both --selection-lock and --main-csv')
    formal=bool(args.selection_lock)
    path=args.main_csv if formal else ROOT/'C020/sequences.csv'
    candidate='candidate_deg' if formal else 'C020_deg'
    with path.open(encoding='utf-8-sig',newline='')as f:
        source=list(csv.DictReader(f))
    assert len(source)==(365 if formal else 213)
    keys={(r['dataset'],r['record']) for r in source}
    assert len(keys)==len(source)
    catalog=Catalog(args.selection_lock)
    if formal:
        locked=read(args.selection_lock)
        assert keys=={(r['dataset'],r['record']) for r in locked['formal_cases']}
    output=[]
    for row in source:
        d,n=row['dataset'],row['record']
        result={k:row[k]for k in ('dataset','record','group','status')}
        if row.get('full_vqf_deg') and row.get(candidate):
            raw,_=catalog.native(d,n)
            values=fixed_scores(raw)
            assert values['scoring_indices_sha256']==row['scoring_indices_sha256']
            assert values['scored_frames']==int(row['scored_frames'])
            assert abs(values['default_deg']-float(row['full_vqf_deg']))<2e-4,(d,n)
            result.update(values,candidate_deg=float(row[candidate]))
        output.append(result)
        if len(output)%60==0:print(len(output),d,flush=True)
    domains={}
    for d in dict.fromkeys(r['dataset']for r in output):
        domain=[r for r in output if r['dataset']==d]
        scored=[r for r in domain if 'default_deg' in r]
        summary={'requested':len(domain),'scored':len(scored),'statuses':dict(Counter(r['status'] for r in domain))}
        if scored:
            summary['means']={k:fmean(r[k] for r in scored)for k in ('default_deg','fast_deg','slow_deg','candidate_deg')}
            summary['candidate_improving_records']={k:sum(r['candidate_deg']<r[k] for r in scored)for k in ('default_deg','fast_deg','slow_deg')}
        domains[d]=summary
    suffix='final' if formal else 'development'
    target=ROOT/'review'/('vqf_response_controls_'+suffix+'.json')
    evidence={'scope':'Full six-axis VQF on raw native input, with all ordinary parameters including bias estimation at default except tauAcc. Three distinct fixed settings are applied to every dataset. No source identifier, model-dependent mask, per-dataset parameter fitting or new model selection.',
          'selection_lock_sha256':digest(args.selection_lock) if formal else None,'source_csv_sha256':digest(path),
          'source_candidate_column':candidate,'domains':domains,'records':output}
    target.write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
    for d,s in domains.items():print(d,s.get('means'),flush=True)


if __name__=='__main__':
    main()

