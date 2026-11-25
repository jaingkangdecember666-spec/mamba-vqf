"""Separate authorization, actual development, locked evaluation, and independent verification times."""
import csv
import json
from collections import Counter
from datetime import datetime

from lock_from_frontier import ROOT,REVIEW,read,digest


def local_stamp(p):
    return datetime.fromtimestamp(p.stat().st_mtime).astimezone()


def main():
    lock=read(ROOT/'selection-lock.json')
    check=read(REVIEW/'final-verification.json');counts=read(REVIEW/'campaign-counts.json')
    protocol=read(ROOT/'protocol.json')
    assert check['lock_sha256']==digest(ROOT/'selection-lock.json')
    begin=datetime.fromisoformat(protocol['start']);end=datetime.fromisoformat(protocol['end'])
    locked=datetime.fromisoformat(lock['locked_at'])
    first=datetime.fromisoformat(counts['batches'][0]['started_at'])
    latest=datetime.fromisoformat(counts['batches'][-1]['completed_at'])
    files=[ROOT/check['main_table'],ROOT/check['additional_table'],REVIEW/'final-verification.json',
           REVIEW/'final-supplemental/summary.json',REVIEW/'final-davis/summary.json',REVIEW/'vqf_response_controls_final.json',
           REVIEW/'final-geometry.json',REVIEW/'tum_final_reference.json',REVIEW/'final_self_mount_predictions.json',
           REVIEW/'oxford_final_sensitivity.json',REVIEW/'historical_final_join.json',REVIEW/'selected-training-sources.json',
           REVIEW/'final-latency-cpu.json',REVIEW/'final-latency-cuda.json',REVIEW/'final-causality.json',REVIEW/'final-transfer-crosscheck.json',
           REVIEW/'transfer-correction-directions.json',REVIEW/'final-alternate-check.json',
           REVIEW/'development_replay_C038_C038_deg.json',REVIEW/'source-side-transfer.json',ROOT/'final/exposure-audit.json',
           REVIEW/'all-native-verification.json',REVIEW/'latency-environment.json',REVIEW/'development-vqf-thresholds.json',
           REVIEW/'development-fixed-vqf-threshold-audit.json',REVIEW/'shice-native-latency-cpu.json',REVIEW/'shice-native-latency-cuda.json',
           ROOT/'final/historical-models/main.csv',ROOT/'final/historical-models/transfer.csv',ROOT/'final/historical-models/summary.json',
           REVIEW/'previous-model-native-anchors.json',REVIEW/'historical-models-verified.json',
           ROOT/'final/historical-models/tcn-backend-attribution/summary.json',
           ROOT/'final/shice-clock-sensitivity/summary.json',ROOT/'final/shice-clock-sensitivity/verification.json',
           REVIEW/'shice-clock-anchors.json',REVIEW/'shice-clock-verified.json',REVIEW/'guidance-verification.json']
    if (ROOT/'external-sources/penn-handheld/preprocessing-lock.json').is_file():
        files.append(REVIEW/'final-penn/summary.json')
    job=read(ROOT/'final/process.json')
    assert job['selection_sha256']==check['lock_sha256'] and job['status'] in ('complete','complete_with_errors')
    panels={}
    for tag,n in [('main',365),('additional',76),('transfer',33)]:
        table=ROOT/'final'/f'{tag}.csv'
        assert digest(table)==job['tables'][tag]['sha256']
        with table.open(encoding='utf-8-sig',newline='')as stream:rows=list(csv.DictReader(stream))
        receipts=[read(p)for p in sorted((ROOT/'final/receipts'/tag).glob('*.json'))]
        assert len(receipts)==len(rows)==n
        for actual,flat in zip(receipts,rows,strict=True):
            assert actual['selection_sha256']==job['selection_sha256']
            for key,val in flat.items():assert val==str(actual.get(key,'') if actual.get(key) is not None else ''),(tag,key,actual['record'])
        states=dict(Counter(r['status'] for r in receipts))
        assert states==job['tables'][tag]['status_counts']
        panels[tag]={'requested':n,'states':states,'actual_attempts':sum(r['attempt']for r in receipts),
                     'repeat_attempts':sum(r['attempt']-1 for r in receipts),'csv_sha256':digest(table)}
    start_job=datetime.fromisoformat(job['started_at']);end_job=datetime.fromisoformat(job['finished_at'])
    assert begin<=locked<=start_job<=end_job
    stamps={p.relative_to(ROOT).as_posix():local_stamp(p) for p in files if p.is_file()}
    assert begin<=first<=latest<=locked<=min(stamps.values())
    chronology='review/guidance-verification.json'
    numeric_stamps={path:time for path,time in stamps.items() if path!=chronology}
    finish=max(numeric_stamps.values())
    result={'window_start':begin.isoformat(),'window_end':end.isoformat(),'authorized_seconds':int((end-begin).total_seconds()),
            'first_registered_batch':first.isoformat(),'last_development_result':latest.isoformat(),'locked_at':locked.isoformat(),
            'science_completed_at':finish.isoformat(),'science_elapsed_from_start_s':(finish-begin).total_seconds(),
            'guidance_verified_at':stamps[chronology].isoformat(),'evidence_checked_through_at':max(stamps.values()).isoformat(),
            'registered_development_process_seconds':counts['completed_job_interval_union_s'],
            'lock_to_last_scientific_artifact_s':(finish-locked).total_seconds(),
            'formal_job_seconds':(end_job-start_job).total_seconds(),'formal_job_started_at':job['started_at'],
            'formal_job_finished_at':job['finished_at'],'formal_panels':panels,
            'main_file_written_at':stamps[check['main_table']].isoformat(),'actual_verified_scientific_outputs':{k:v.isoformat() for k,v in numeric_stamps.items()},
            'pending_named_outputs':[p.relative_to(ROOT).as_posix() for p in files if not p.is_file()],
            'evidence_clock':'Job timestamps for batch starts and finishes. Local last-write times for final results and independent audit files. HTML generation and rendering take place after these scientific milestones.'}
    (REVIEW/'delivery-timing.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['science_elapsed_from_start_s'],result['lock_to_last_scientific_artifact_s'],result['pending_named_outputs'])


if __name__=='__main__':main()
