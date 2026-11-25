"""Replay a registered Mamba recipe on all 213 developmental records, without exposing sealed inputs."""
import argparse
import csv
import json
from datetime import datetime
from time import perf_counter

import torch

from check_sources import ROOT
from lock_from_frontier import inference,read,digest
from locked_replay import Predictor
from native_catalog import Catalog


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('batch')
    ap.add_argument('column')
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=ap.parse_args()
    torch.set_num_threads(2)
    recipes={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    spec=inference((args.batch,args.column),recipes)
    predictor=Predictor(spec,args.device)
    catalog=Catalog()
    src=ROOT/args.batch/'sequences.csv'
    with src.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len({(r['dataset'],r['record']) for r in rows})==213
    out=[]
    begin=perf_counter()
    for r in rows:
        raw,_=catalog.native(r['dataset'],r['record'])
        score=predictor.score(raw)
        assert score['scoring_indices_sha256']==r['scoring_indices_sha256'] and score['scored_frames']==int(r['scored_frames'])
        gap=max(abs(score['full_vqf_deg']-float(r['full_vqf_deg'])),abs(score['candidate_deg']-float(r[args.column])))
        assert gap<2e-4,(r['dataset'],r['record'],gap)
        out.append({'dataset':r['dataset'],'record':r['record'],'group':r['group'],'max_abs_gap_deg':gap,**score})
        if len(out)%40==0:
            print(len(out),r['dataset'],'seconds',round(perf_counter()-begin,1),flush=True)
    target=ROOT/'review'/f'development_replay_{args.batch}_{args.column}.json'
    summary={'cutoff':datetime.now().astimezone().isoformat(),'device':args.device,'torch_threads':torch.get_num_threads(),
             'duration_s':perf_counter()-begin,'recipe':spec,'batch':args.batch,'column':args.column,
             'source_csv_sha256':digest(src),'n_records':len(out),'sum_scoring_frames':sum(r['scored_frames'] for r in out),
             'max_abs_gap_deg':max(r['max_abs_gap_deg'] for r in out),'rows':out}
    target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print('independent development replay',summary['n_records'],summary['sum_scoring_frames'],summary['max_abs_gap_deg'],summary['duration_s'])


if __name__=='__main__':
    main()
