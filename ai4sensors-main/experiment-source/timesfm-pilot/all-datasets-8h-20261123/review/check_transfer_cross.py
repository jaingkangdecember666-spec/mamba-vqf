"""Join the executing agent's native predictions to independent replays of every supplementary transfer recording."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from lock_from_frontier import ROOT,REVIEW,read,digest


def rows(path):
    with path.open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))


def index(records):
    result={(r['dataset'],r['record']):r for r in records}
    assert len(records)==len(result)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('worker_csv',type=Path);args=ap.parse_args()
    sel=ROOT/'selection-lock.json';sha=digest(sel)
    sources=[REVIEW/'final-supplemental/sequences.csv',REVIEW/'final-davis/sequences.csv',REVIEW/'final-penn/sequences.csv']
    ref=index([r for p in sources for r in rows(p)]); other=index(rows(args.worker_csv))
    alternate_table=REVIEW/'native-alternate-transfer.csv'
    alternate=index(rows(alternate_table))
    assert ref.keys()==other.keys()==alternate.keys() and len(ref)==33
    checked=defaultdict(list);missing=[]
    for key,own in ref.items():
        original=other[key];native=alternate[key]
        if not own.get('candidate_deg') or not original.get('candidate_deg'):
            assert not own.get('candidate_deg') and not original.get('candidate_deg'),(key,own['status'],original['status'])
            assert not native.get('candidate_deg') and native['status']=='source_unavailable'
            missing.append({'dataset':key[0],'record':key[1],'replay_status':own['status'],'execution_status':original['status'],'alternate_status':native['status']});continue
        assert original['scoring_indices_sha256']==own['scoring_indices_sha256']==native['scoring_indices_sha256'],key
        assert int(original['scored_frames'])==int(own['scored_frames'])==int(native['scored_frames']),key
        gaps={k:max(abs(float(original[k])-float(x[k]))for x in (own,native)) for k in ('full_vqf_deg','physical_deg','candidate_deg')}
        assert max(gaps.values())<2e-4,(key,gaps)
        checked[key[0]].append({'record':key[1],'scored_frames':int(own['scored_frames']),'gaps_deg':gaps})
    summaries={d:{'records':len(r),'max_abs_gap_deg':max(z for x in r for z in x['gaps_deg'].values()),'scored_frames':sum(x['scored_frames']for x in r)}for d,r in checked.items()}
    result={'selection_lock_sha256':sha,'requested':len(ref),'checked_records':sum(len(r)for r in checked.values()),
            'unscored_in_both':missing,'by_source':summaries,'details':checked,
            'score_tables_sha256':{p.resolve().relative_to(ROOT).as_posix():digest(p)for p in [args.worker_csv,*sources,alternate_table]},
            'scope':'Three executions: official and supplemental runs share SelectedPredictor. The alternate uses the development C036/C038 forward and C017 quaternion metric. Common elements include the locked model, VQF library, original IMU, reference alignment, and some block feature constructors.'}
    (REVIEW/'final-transfer-crosscheck.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    main_table=ROOT/'final/main.csv'
    official=index(rows(main_table))
    anchors=index(rows(REVIEW/'native-alternate-main.csv'))
    expected={(d,n) for d,names in read(REVIEW/'final-anchor-spec.json')['anchors'].items() for n in names}
    assert len(anchors)==14 and anchors.keys()==expected
    checks=[]
    for key,alternate_row in anchors.items():
        stored=official[key]
        assert alternate_row['status']==stored['status'],key
        if not stored.get('candidate_deg'):
            checks.append({'dataset':key[0],'record':key[1],'status':stored['status']});continue
        assert int(alternate_row['scored_frames'])==int(stored['scored_frames']),key
        assert alternate_row['scoring_indices_sha256']==stored['scoring_indices_sha256'],key
        gap=max(abs(float(alternate_row[k])-float(stored[k])) for k in ('full_vqf_deg','physical_deg','candidate_deg'))
        assert gap<2e-4,(key,gap)
        checks.append({'dataset':key[0],'record':key[1],'status':stored['status'],'scored_frames':int(stored['scored_frames']),'max_abs_gap_deg':gap})
    native_proof=read(REVIEW/'native-alternate-summary.json')
    assert native_proof['lock_sha256']==sha
    for path,file_hash in native_proof['hashes'].items():assert digest(ROOT/path)==file_hash,path
    report={'selection_lock_sha256':sha,'execution_main_sha256':digest(main_table),'rows':checks,
            'checked_records':sum('max_abs_gap_deg'in r for r in checks),'supplemental_checked':result['checked_records'],
            'max_main_gap_deg':max(r['max_abs_gap_deg'] for r in checks if 'max_abs_gap_deg'in r),
            'scope':result['scope'],'alternate_dependencies':native_proof}
    (REVIEW/'final-alternate-check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(summaries,missing)


if __name__=='__main__':main()
