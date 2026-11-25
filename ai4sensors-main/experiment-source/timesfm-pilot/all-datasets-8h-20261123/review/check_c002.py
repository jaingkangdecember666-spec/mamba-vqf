"""Rebuild C002 aggregate results, masks and representative native predictions."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from check_array import base_inputs
from check_sources import DATA, ROOT, TRANSFER, score, tum, weights, yt


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def mask_digest(mask):
    return hashlib.sha256(np.flatnonzero(mask).astype('<i8').tobytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    old = {(r['dataset'], r['record']): r for r in table(ROOT/'C001/sequences.csv')}
    rows = table(ROOT/'C002/sequences.csv')
    published = json.loads((ROOT/'C002/results.json').read_text('utf8'))
    poses = table(ROOT/'C002/array-sensor-poses.csv')
    inventory = json.loads((ROOT/'C001/inventory.json').read_text('utf8'))
    records = [r for r in inventory['records'] if r['side']=='development' and r['dataset']=='IMU-Array']
    array_ref = np.memmap(DATA/'IMU-Array-main/Calibration/Group1_ref_data.bin', dtype='<f8', mode='r').reshape(-1,7)
    mapped = {r['record']:r for r in records}
    score_keys = ['full_vqf_deg','b6_full_deg','b6_gain_025_deg']
    scores = defaultdict(list)
    anchor_errors = []
    head = weights(TRANSFER/'batch6/mamba-intrinsic-selected.pt', 'mamba_v1')
    with np.load(ROOT/'C001/scoring-masks.npz') as first, np.load(ROOT/'C002/scoring-masks.npz') as second:
        assert len(rows)==len(set((r['dataset'],r['record']) for r in rows))==len(second.files)==85
        for row in rows:
            domain, name = row['dataset'], row['record']
            mask = second[row['mask_key']]
            assert len(mask)==int(row['native_frames']) and mask.sum()==int(row['scored_frames'])
            assert mask_digest(mask)==row['scoring_indices_sha256']
            raw = None
            if (domain, name) in old:
                previous = old[domain,name]
                assert np.array_equal(mask,first[previous['mask_key']])
                assert float(previous['full_vqf_deg'])==float(row['full_vqf_deg'])
                assert float(previous['b6_intrinsic_deg'])==float(row['b6_full_deg'])
                if domain=='TUM-VI':
                    raw = tum(int(name.removeprefix('room')))
                if domain=='YT_Server' and name==inventory['yt']['selected_valid'][0]:
                    raw = yt(name)
            else:
                raw_array, audit = base_inputs(mapped[name], array_ref)
                assert np.array_equal(mask,raw_array[3])
                pose_rows = [x for x in poses if x['record']==name]
                assert len(pose_rows)==6
                for pose in pose_rows:
                    axis = 'XYZ'.index(pose['pose'][1])
                    sign = 1 if pose['pose'][0]=='+' else -1
                    pose_mask = mask & (raw_array[2][:,axis]*sign>.99)
                    assert pose_mask.sum()==int(pose['scored_frames']) and mask_digest(pose_mask)==pose['scoring_indices_sha256']
                for key in score_keys:
                    pooled = (sum(float(x[key])**2*int(x['scored_frames']) for x in pose_rows)/mask.sum())**.5
                    assert abs(pooled-float(row[key]))<1e-10
                if name in (records[0]['record'],records[-1]['record']):
                    raw=raw_array
            if raw is not None:
                signal, hz, truth, allowed = raw
                assert np.array_equal(mask,allowed & np.isfinite(signal).all(1))
                observed = score(signal,hz,truth,mask,{'frozen':head},{'frozen':{'b6_full':1.,'b6_gain_025':.25}},1.,{'frozen'})
                discrepancy=max(abs(observed['rmse_deg'][key.removesuffix('_deg')]-float(row[key])) for key in score_keys)
                assert discrepancy<1e-6
                anchor_errors.append({'domain':domain,'record':name,'max_error_deg':discrepancy,'rmse_deg':observed['rmse_deg']})
            scores[domain].append([float(row[k]) for k in score_keys])
    means={}
    for domain, values in scores.items():
        a=np.array(values);means[domain]=dict(zip(score_keys,a.mean(0).tolist()))
        assert all(abs(v-published['summary'][domain]['macro_rmse_deg'][k])<1e-12 for k,v in means[domain].items())
        means[domain]['paired_gain_025_wins']=int(np.sum(a[:,2]<a[:,0]))
        means[domain]['records']=len(values)
    result={'rows':len(rows),'matching_c001_controls_and_masks':len(old),'array_native_masks':len(records),
            'array_pose_masks_and_pooled_scores':len(poses),'means':means,'raw_replay':anchor_errors,
            'source_results_sha256':hashlib.sha256((ROOT/'C002/results.json').read_bytes()).hexdigest()}
    (ROOT/'review/C002_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(len(rows),len(poses),means,anchor_errors)


if __name__=='__main__':
    main()

