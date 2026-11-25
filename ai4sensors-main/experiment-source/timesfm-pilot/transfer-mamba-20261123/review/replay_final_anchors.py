"""Replay four locked SHICE holdout records from the original double-precision CSV files."""
import hashlib
import json

import torch

from replay_native import ROOT, read, score, weights
from target_raw import source_record


def main():
    lock = read(ROOT / 'review/final-transfer-lock.json')
    torch.set_num_threads(2)
    models = {}
    canonical = set()
    for label, spec in lock['models'].items():
        path = ROOT / spec['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
        device = 'cuda' if spec['kind'] == 'tcn' else 'cpu'
        models[label] = weights(path, spec['kind'], device)
        if spec.get('intrinsic'):
            canonical.add(label)
    policies = {name:{name:1.} for name in models}
    anchors = []
    for name in lock['independent_raw_checks']:
        scored = score(*source_record(name), models, policies, 1., canonical)
        anchors.append({'name':name, **scored})
        print(name, scored['frames'], scored['rmse_deg'], flush=True)
    assert len(anchors) == 4
    (ROOT / 'review/final_raw_anchors.json').write_text(json.dumps(anchors, indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
