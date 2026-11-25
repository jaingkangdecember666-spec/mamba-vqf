"""Independent source-native score checks for both externally trained intrinsic Mamba seeds."""
import json
import torch
from replay_native import ROOT, external_case, read, score, weights


def main():
    torch.set_num_threads(2)
    lock = read(ROOT/'review/final-transfer-lock.json')
    seeds = ('transfer_mamba','repeat_mamba')
    models = {m:weights(ROOT/lock['models'][m]['path'],'mamba_v1') for m in seeds}
    stages = [read(ROOT/f'batch{i}/results.json')['external_sequences'] for i in (6,7)]
    indices = [{r['sequence']:r for r in s} for s in stages]
    checks = []
    for r in read(ROOT/'external-split.json')['records']:
        if r['split'] != 'validation':
            continue
        result = score(*external_case(r),models,{m:{m:1.} for m in seeds},1.,set(seeds))
        errors = []
        for s,(idx,label) in enumerate(zip(indices,seeds),6):
            row=idx[r['name']]
            assert row['scoring_indices_sha256']==result['indices_sha256'] and row['frames']==result['frames']
            errors.append(abs(result['rmse_deg'][label]-row[f'batch{s}_mamba_deg']))
        assert max(errors) < 1e-4, (r['name'],errors)
        checks.append({'name':r['name'],'domain':r['domain'],**result,'max_abs_delta_deg':max(errors)})
    assert len(checks)==18 and sum(r['frames'] for r in checks)==663577
    (ROOT/'review/intrinsic_external_replay.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
    print(len(checks), max(r['max_abs_delta_deg'] for r in checks))


if __name__=='__main__':
    main()
