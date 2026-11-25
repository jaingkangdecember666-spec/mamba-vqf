"""Audit the locked transfer results against archived frames and the development receipts."""
import csv
import hashlib
import json
import statistics as st
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REVIEW = ROOT / 'review'
PREVIOUS = ROOT.parent / 'overnight-20261122'
MODELS = ('full_vqf', 'strict_mamba', 'strict_tcn', 'transfer_mamba', 'repeat_mamba')


def read(path):
    return json.loads(path.read_text('utf-8-sig'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def paired(group):
    assert group
    result = {'records': len(group), 'frames': sum(r['frames'] for r in group)}
    for m in MODELS:
        ordered = sorted((r[m] - r['full_vqf'], r['sequence']) for r in group)
        delta = [x[0] for x in ordered]
        result[m] = {'rmse_deg': st.fmean(r[m] for r in group), 'mean_delta_deg': st.fmean(delta),
                     'median_delta_deg': st.median(delta), 'wins': sum(x < -1e-9 for x in delta),
                     'best': ordered[0], 'worst': ordered[-1]}
    return result


def main():
    lock = read(REVIEW / 'final-transfer-lock.json')
    strict = read(REVIEW / 'strict-external-lock.json')
    assert digest(ROOT / 'external-split.json') == strict['source_split_sha256'] == lock['source_split_sha256']
    for source, sha in lock['evidence'].items():
        assert digest(ROOT/source) == sha, source
    for key, model in lock['models'].items():
        assert digest(ROOT / model['path']) == model['sha256'], key
        assert model['gain'] == 1
    current = rows(ROOT / 'batch8/all-sequences.csv')
    receipt = read(ROOT / 'batch8/results.json')
    assert receipt['lock_sha256'] == digest(REVIEW/'final-transfer-lock.json')
    assert receipt['started_at'] >= lock['locked_at'] and receipt['optimizer_steps'] == 0
    assert not receipt['failures'] and receipt['heldout_records_scored'] == 20
    assert receipt['csv_sha256'] == digest(ROOT/'batch8/all-sequences.csv')
    assert receipt['masks_sha256'] == digest(ROOT/'batch8/scoring-masks.npz')
    assert len(current) == len({r['sequence'] for r in current}) == 57
    dev = {r['sequence']:r for r in read(ROOT / 'batch4/results.json')['sequences']}
    guided = {stage:{r['sequence']:r for r in read(ROOT/f'batch{stage}/results.json')['shice_sequences']} for stage in (6,7)}
    historical = read(PREVIOUS / 'rounds/R00001/result.json')
    history = {r['name']: (r['frames'], r['rmse_deg']) for split in ('dev_metrics', 'r5')
               for r in historical[split]['sequences']}
    old_final = read(PREVIOUS / 'rounds/R01826/result.json')['datasets']['R6']['sequences']
    history.update({r['name']: (r['frames'], r['rmse_deg']['full_vqf']) for r in old_final})
    mask_by_device = read(PREVIOUS / 'review/final_audit.json')['raw_shice_score_indices']
    masks = {name:v['score_frame_indices_sha256'] for split in mask_by_device.values() for name,v in split.items()}
    masks.update({k:v['scoring_indices_sha256'] for k,v in dev.items()})
    independent = read(REVIEW / 'final_raw_anchors.json')
    anchor_map = {r['name']:r for r in independent}
    assert {'R5', 'R6'} <= {k.split('-')[0] for k in anchor_map}
    target = []
    deviations, anchor_deviations = [], []
    for row in current:
        name = row['sequence']
        device, program = name.split('-')
        assert row['device'] == device and int(row['frames']) == history[name][0]
        assert row['scoring_indices_sha256'] == masks[name]
        item = {'sequence': name, 'device': device, 'program': int(program), 'frames':int(row['frames']),
                'scoring_indices_sha256': masks[name], **{m: float(row[m]) for m in MODELS}}
        assert all(0 < item[m] < 180 for m in MODELS)
        deviations.append(abs(item['full_vqf'] - history[name][1]))
        if name in dev:
            assert abs(item['strict_mamba'] - dev[name]['rmse_deg']['mamba_v1']) < 1e-5
            assert abs(item['strict_tcn'] - dev[name]['rmse_deg']['tcn']) < 1e-5
            for stage, label in ((6,'transfer_mamba'),(7,'repeat_mamba')):
                row = guided[stage][name]
                assert row['frames'] == item['frames'] and row['scoring_indices_sha256'] == masks[name]
                assert abs(item[label]-row[f'batch{stage}_mamba_deg']) < 1e-9
        if name in anchor_map:
            a = anchor_map[name]
            assert a['indices_sha256'] == masks[name] and a['frames'] == item['frames']
            anchor_deviations.append({'sequence':name, **{m: abs(a['rmse_deg'][m]-item[m]) for m in MODELS}})
        target.append(item)
    assert set(history) == set(masks) == {x['sequence'] for x in target}
    assert len(anchor_deviations) == len(anchor_map)
    assert max(deviations) < 2e-5
    assert max(r[m] for r in anchor_deviations for m in MODELS) < 1e-4, anchor_deviations
    counts = Counter(r['device'] for r in target)
    assert counts == {'R1':7, 'R2':10, 'R3':10, 'R4':10, 'R5':10, 'R6':10}
    raw_groups = {name:[r for r in target if r['device']==name] for name in sorted(counts)}
    development = [r for r in target if r['sequence'] in dev]
    holdout = [r for r in target if r['sequence'] not in dev]
    groups = {name:paired(v) for name,v in (raw_groups | {'development': development, 'holdout':holdout,'all':target}).items()}
    assert groups['development']['frames']==8328711 and groups['holdout']['frames']==4610054
    worker = receipt['summary']
    worker_groups = worker['by_device'] | {'holdout':worker['R5+R6'],'all':worker['all_57'],'development':worker['R1–R4']}
    for name, group in groups.items():
        declared = worker_groups[name]
        assert (group['frames'],group['records']) == (declared['scoring_frames'],declared['records'])
        for m in MODELS:
            assert abs(group[m]['rmse_deg'] - declared['macro_rmse_deg'][m]) < 1e-12
            if m != 'full_vqf':
                assert group[m]['wins'] == declared['wins_vs_full_vqf'][m]
    leave_one = {}
    for m in MODELS[1:]:
        leave_one[m] = [st.fmean(r[m]-r['full_vqf'] for r in holdout if r['program']!=program) for program in range(1,11)]
    proof={'source_sha256':{p:digest(ROOT/p) for p in ('batch8/all-sequences.csv','batch8/results.json','review/final-transfer-lock.json',
               'review/strict-external-lock.json','review/final_raw_anchors.json')}, 'groups':groups, 'same_masks':len(masks),
           'largest_historical_baseline_gap_deg':max(deviations), 'raw_anchor_deviations_deg':anchor_deviations,
           'heldout_leave_one_program_out_delta_deg':leave_one,
           'historical_r6':{m:st.fmean(r['rmse_deg'][m] for r in old_final) for m in old_final[0]['rmse_deg']}}
    (REVIEW/'transfer-audit.json').write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'records':len(target), 'frames':groups['all']['frames'], 'holdout':groups['holdout']}, ensure_ascii=False))


if __name__=='__main__':
    main()
