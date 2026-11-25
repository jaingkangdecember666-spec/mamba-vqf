"""Paired, collection-group resampling for descriptive uncertainty on dataset-level mean differences."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def paired_by_group(rows,baseline,candidate):
    sets={}
    for r in rows:
        diff=float(r[candidate])-float(r[baseline])
        sets.setdefault(r['group'],[]).append(diff)
    names=sorted(sets)
    sums=np.array([sum(sets[k]) for k in names]);sizes=np.array([len(sets[k]) for k in names])
    observed=float(sums.sum()/sizes.sum())
    metrics={'records':int(sizes.sum()),'collection_groups':len(names),'candidate_minus_full_vqf_deg':observed,
             'improving_records':sum(x<0 for s in sets.values() for x in s),
             'improving_groups':int((sums<0).sum()),'groups':dict(zip(names,(sums/sizes).tolist()))}
    if len(names)>1:
        loo=(sums.sum()-sums)/(sizes.sum()-sizes)
        metrics['leave_one_group_out_deg']=[float(loo.min()),float(loo.max())]
    if len(names)>=5:
        rng=np.random.default_rng(20261123)
        take=rng.integers(0,len(names),size=(8000,len(names)))
        distribution=sums[take].sum(1)/sizes[take].sum(1)
        metrics['paired_cluster_percentile_95_deg']=np.percentile(distribution,[2.5,97.5]).tolist()
    return metrics


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('csv',type=Path);parser.add_argument('candidate');parser.add_argument('output',type=Path)
    parser.add_argument('--baseline',default='full_vqf_deg')
    args=parser.parse_args()
    with args.csv.open(encoding='utf-8-sig',newline='') as f:allrows=list(csv.DictReader(f))
    available=[r for r in allrows if r.get(args.candidate) and r.get(args.baseline)]
    report={'source_sha256':hashlib.sha256(args.csv.read_bytes()).hexdigest(),'method':'Paired record means. Bootstrap whole collection groups with replacement, retaining all their records and original record-equal weights. No inference interval for fewer than five independent groups. Descriptive when reused for selection.',
            'domain':{}}
    for dataset in sorted({r['dataset'] for r in allrows}):
        included=[r for r in available if r['dataset']==dataset]
        item=paired_by_group(included,args.baseline,args.candidate)
        item['denominator']=sum(r['dataset']==dataset for r in allrows)
        report['domain'][dataset]=item
        print(dataset,item['records'],item['collection_groups'],round(item['candidate_minus_full_vqf_deg'],6),item.get('paired_cluster_percentile_95_deg'),item.get('leave_one_group_out_deg'))
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()

