"""Score the two preregistered B runs from the Penn benchmark after overall model locking."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import fmean

import torch

from lock_from_frontier import ROOT, digest, read
from locked_replay import SelectedPredictor
from penn_handheld import BASE, Penn
from vqf_response_controls import fixed_scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('selection_lock',type=Path)
    lock = p.parse_args().selection_lock
    torch.set_num_threads(2)
    ds = Penn(lock)
    model = SelectedPredictor(lock)
    source = read(BASE/'preprocessing-lock.json')
    manifest = {r['record']:r for r in read(BASE/'source-files.json')['records']}
    folder = ROOT/'review/final-penn'
    folder.mkdir(exist_ok=True)
    rows = []
    fields = ['full_vqf_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg','candidate_deg','physical_deg']
    for name in source['sealed_records']:
        assert ds.roles[name]=='sealed_after_model_lock'
        row = {'dataset':'PennCOSYVIO','record':name}
        try:
            for key in ('imu','pose'):
                file = manifest[name][key]
                assert digest(BASE/file['path'])==file['sha256']
            raw,quality = ds.native(name)
            reference = fixed_scores(raw)
            score = model.score(raw)
            assert reference['scoring_indices_sha256']==quality['scoring_indices_sha256']==score['scoring_indices_sha256']
            assert abs(reference['default_deg']-score['full_vqf_deg']) < 1e-5
            row.update(status='scored',fixed_vqf_1p5_deg=reference['fast_deg'],fixed_vqf_6_deg=reference['slow_deg'],
                       quality=quality,**score)
        except (OSError,AssertionError,RuntimeError,ValueError) as exc:
            row.update(status='evaluation_failed',detail=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    assert {r['record'] for r in rows}=={'bf','bs'}
    scored = [r for r in rows if r['status']=='scored']
    with (folder/'sequences.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','record','status','detail','scored_frames','scoring_indices_sha256',*fields],extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    summary = {'completed_at':datetime.now().astimezone().isoformat(),'selection_lock_sha256':digest(lock),
               'preprocessing_lock_sha256':digest(BASE/'preprocessing-lock.json'),'requested':len(rows),'scored':len(scored),
               'records_improved':sum(r['candidate_deg']<r['full_vqf_deg'] for r in scored),
               'means':{k:fmean(r[k] for r in scored) for k in fields} if scored else {},'rows':rows,
               'scope':'Two B benchmark records at different speeds, one instrument on approximately the A training route. Marker pose graph smoothed by the original authors.'}
    (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(summary['scored'],summary['means'])


if __name__=='__main__':
    main()
