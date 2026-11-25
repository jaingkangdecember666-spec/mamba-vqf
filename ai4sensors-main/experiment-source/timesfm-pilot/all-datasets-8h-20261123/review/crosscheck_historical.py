"""Join the frozen full-coverage run to the four archived six-axis comparisons on common scoring rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import fmean

import numpy as np

from check_sources import ROOT
from native_catalog import Catalog


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('lock',type=Path)
    parser.add_argument('sequence_csv',type=Path,nargs='+')
    parser.add_argument('--candidate',default='candidate_deg')
    args=parser.parse_args()
    catalog=Catalog(args.lock)
    archive=ROOT.parent/'overnight-20261122'/'review'/'baseline_audit.json'
    prior=read(archive)
    for relative,expected in prior['source_sha256'].items():
        assert digest(ROOT.parent/relative)==expected,relative
    current={}
    for path in args.sequence_csv:
        for row in csv_rows(path):
            if not row.get(args.candidate):
                continue
            key=row['dataset'],row['record']
            if key in current:
                for col in ('scoring_indices_sha256','scored_frames',args.candidate,'full_vqf_deg'):
                    assert row[col]==current[key][col],(key,col)
            current[key]=row
    corrections=ROOT.parent/'tilt-correction'
    r6=read(corrections/'test.json')['r6_records']
    ext=read(corrections/'external_test.json')['datasets']
    old={'R6':r6}
    for name,data in ext.items():
        old[name]=[{'recording':s['recording'],**s['protocols']['main']} for s in data['records']]
    gru=read(ROOT.parent/'gru-tilt'/'test.json')['tests']
    scored=[]; summary={}
    for name,rows in old.items():
        d='SHICE' if name=='R6' else name
        controls={r['recording']:r for r in gru[name]['per_sequence']}
        assert set(controls)=={r['recording'] for r in rows}
        for previous in rows:
            recording=previous['recording']
            row=current[d,recording]
            raw,_=catalog.native(d,recording)
            signal,_,_,eligible=raw
            ids=np.flatnonzero(eligible & np.isfinite(signal).all(axis=1))
            fingerprint=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
            assert fingerprint==row['scoring_indices_sha256']
            assert len(ids)==int(row['scored_frames'])==previous['scoring_frames']==controls[recording]['scoring_frames']
            if name=='R6':
                with np.load(ROOT.parent/'transfer-mamba-20261123'/'batch8'/'scoring-masks.npz') as f:
                    saved=f[recording]
                    recorded=(np.flatnonzero(saved) if saved.dtype.kind=='b' else saved)
                    np.testing.assert_array_equal(ids,recorded)
            scores={k:previous['rmse_deg'][k] for k in ('full_vqf','physical_only','timesfm')}
            scores['gru']=controls[recording]['rmse_deg']['gru']
            gap=abs(float(row['full_vqf_deg'])-scores['full_vqf'])
            assert gap<2e-4,(d,recording,gap)
            scored.append({'domain':name,'record':recording,'scored_frames':len(ids),'scoring_indices_sha256':fingerprint,
                           'reconstructed_default_gap_deg':gap,'candidate':float(row[args.candidate]),**scores})
        items=[r for r in scored if r['domain']==name]
        means={k:fmean(r[k] for r in items) for k in ('candidate','full_vqf','physical_only','timesfm','gru')}
        expected=prior['historical_baselines'][name]['macro_rmse_deg']
        assert all(abs(means[k]-expected[k])<1e-8 for k in ('full_vqf','physical_only','timesfm','gru'))
        summary[name]={'records':len(items),'scoring_frames':sum(r['scored_frames'] for r in items),
                       'macro_rmse_deg':means,'candidate_wins':{k:sum(r['candidate']<r[k] for r in items)
                       for k in ('full_vqf','physical_only','timesfm','gru')}}
    assert {k:v['records'] for k,v in summary.items()}=={'R6':10,'BROAD':39,'Sassari':18}
    former=ROOT.parent/'transfer-mamba-20261123/batch8/all-sequences.csv'
    verified_transfer=read(former.with_name('results.json'))
    assert digest(former)==verified_transfer['csv_sha256']
    joint=[]
    for original in csv_rows(former):
        if original['device'] not in ('R5','R6'):continue
        record=original['sequence'];row=current['SHICE',record]
        assert int(original['frames'])==int(row['scored_frames']) and original['scoring_indices_sha256']==row['scoring_indices_sha256'],record
        assert abs(float(original['full_vqf'])-float(row['full_vqf_deg']))<2e-4,record
        joint.append({'record':record,'device':original['device'],'full_vqf':float(row['full_vqf_deg']),
                      'old_transfer_mamba':float(original['transfer_mamba']),'current':float(row[args.candidate]),
                      'scoring_indices_sha256':row['scoring_indices_sha256']})
    assert len(joint)==len({r['record']for r in joint})==20
    historic_groups={}
    for key in ('R5','R6','R5+R6'):
        members=[r for r in joint if r['device']==key or key=='R5+R6']
        historic_groups[key]={'records':len(members),'improves_old_records':sum(r['current']<r['old_transfer_mamba']for r in members),
                       **{m:fmean(r[m]for r in members)for m in ('full_vqf','old_transfer_mamba','current')}}
    details={'selection_lock_sha256':digest(args.lock),'audit_sha256':digest(archive),
             'previous_transfer':{'source_sha256':digest(former),'groups':historic_groups,'records':joint},
             'current_csv_sha256':{str(p):digest(p) for p in args.sequence_csv},'domains':summary,'rows':scored}
    (ROOT/'review/historical_final_join.json').write_text(json.dumps(details,indent=2)+'\n',encoding='utf8')
    print({d:r['macro_rmse_deg'] for d,r in summary.items()})


if __name__=='__main__':
    main()
