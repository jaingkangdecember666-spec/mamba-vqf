"""Inspect two genuinely external training traversals before choosing a new fit."""
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

from lock_from_frontier import ROOT, digest, inference, read
from locked_replay import Predictor
from penn_handheld import BASE, Penn
from vqf_response_controls import fixed_scores


def main():
    torch.set_num_threads(2)
    ds = Penn()
    source = read(BASE/'preprocessing-lock.json')
    recipes = {(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    names = ('C023','C027','C030','C031')
    models = {name:Predictor(inference((name,name+'_deg'),recipes)) for name in names}
    rows = []
    for item in source['source_records']:
        name = item['record']
        assert ds.roles[name] == 'external_training'
        raw,quality = ds.native(name)
        controls = fixed_scores(raw)
        row = dict(record=name,quality=quality,fixed_fast=controls['fast_deg'],default=controls['default_deg'],fixed_slow=controls['slow_deg'])
        for tag,model in models.items():
            score = model.score(raw)
            assert controls['scoring_indices_sha256'] == quality['scoring_indices_sha256'] == score['scoring_indices_sha256']
            assert abs(score['full_vqf_deg']-row['default']) < 1e-5
            row[tag] = score['candidate_deg']
        rows.append(row)
    assert {r['record'] for r in rows} == {'af','as'}
    result = {'created_at':datetime.now().astimezone().isoformat(),'scope':'Two A training sequences only, after training preprocessing was frozen. Same two A tracks may be used for fitting later. B has not been accessed through the numeric adapter.',
              'training_preprocessing_sha256':digest(BASE/'preprocessing-lock.json'),'rows':rows,
              'means':{k:sum(r[k] for r in rows)/len(rows) for k in ('fixed_fast','default','fixed_slow',*names)}}
    path = ROOT/'review/penn_training_controls.json'
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['means'])


if __name__=='__main__':
    main()
