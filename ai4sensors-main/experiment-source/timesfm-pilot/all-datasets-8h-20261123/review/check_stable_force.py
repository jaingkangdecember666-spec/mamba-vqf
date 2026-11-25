"""Compare the independent four-source fit and replay the C017 development scoring paths."""
import hashlib
import json

import numpy as np
import torch

from check_sources import ROOT,TRANSFER,weights
from check_training_batch import csv_rows,read,digest
from group_uncertainty import paired_by_group
from mixture_replay import replay
from native_catalog import Catalog
from optical_score import metric


def main():
    torch.set_num_threads(2)
    batch=ROOT/'C017'
    lock=read(batch/'coefficient-lock.json')
    source=read(ROOT/'review/stable_force_source_audit.json')
    k=lock['k']
    assert lock['predevelopment_lock'] and abs(k-source['estimated_k'])<1e-6
    assert digest(batch/'external-selection.json')==lock['external_selection_sha256']
    assert digest(ROOT/'C015/sampling.npz')==lock['C015_sampling_sha256']
    old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C014/sequences.csv')}
    rows=csv_rows(batch/'sequences.csv')
    assert len(rows)==len(old)==213 and all(r['status']=='scored' for r in rows)
    assert set(old)=={(r['dataset'],r['record']) for r in rows}
    choices={}
    for r in sorted(rows,key=lambda row:(row['dataset'],row['record'])):
        choices.setdefault(r['dataset'],r['record'])
        previous=old[r['dataset'],r['record']]
        for field in ('group','scored_frames','scoring_indices_sha256'):
            assert r[field]==previous[field],(r['record'],field)
        for left,right in [('full_vqf_deg','full_vqf_deg'),('C014_B_deg','B_deg')]:
            assert abs(float(r[left])-float(previous[right]))<1e-5
    heads={'B6':(weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt','mamba_v1'),.25,'always'),
           'B7':(weights(ROOT/'C013/seed-C006-readout.pt','mamba_v1'),.25,'rest_off')}
    mix={'B6':.5,'B7':.5}
    groups={'C014_B_deg':mix,'C017_deg':mix,'physical_only_deg':{}}
    force={'C017_deg':k,'physical_only_deg':k}
    catalog=Catalog()
    samples=[]
    for r in rows:
        if r['record']!=choices[r['dataset']]:
            continue
        (raw,hz,target,mask),_=catalog.native(r['dataset'],r['record'])
        idx=np.flatnonzero(mask & np.isfinite(raw).all(1))
        assert hashlib.sha256(idx.astype('<i8').tobytes()).hexdigest()==r['scoring_indices_sha256']
        pred=replay(raw,hz,heads,groups,idx,force)
        scores={c:metric(d,target[idx]) for c,d in pred.items()}
        gap=max(abs(v-float(r[c])) for c,v in scores.items())
        assert gap<1e-5,(r['dataset'],r['record'],gap)
        samples.append({'domain':r['dataset'],'record':r['record'],'scores_deg':scores,'max_abs_gap_deg':gap})
    assert len(samples)==8
    out={'source_record_count':len(source['records']),'source_slots':source['n_slots'],
         'independent_k':source['estimated_k'],'reported_k':k,'abs_k_error':abs(k-source['estimated_k']),
         'source_relative_losses':source['estimated_relative_losses'],'source_domains':source['domains'],
         'coefficient_lock_sha256':digest(batch/'coefficient-lock.json'),'common_development_records':len(rows),
         'domains':{c:{d:paired_by_group([r for r in rows if r['dataset']==d],'full_vqf_deg',c)
                       for d in sorted(choices)} for c in ('C017_deg','physical_only_deg')},
         'raw_replays':samples,'results_sha256':digest(batch/'results.json'),'sequences_sha256':digest(batch/'sequences.csv')}
    (ROOT/'review/C017_audit.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('independent k gap',out['abs_k_error'],'maximum native gap',max(r['max_abs_gap_deg'] for r in samples))


if __name__=='__main__':
    main()
