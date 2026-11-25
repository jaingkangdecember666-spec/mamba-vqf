"""Freeze previously agreed candidates and scoring behavior before opening new R5/R6 predictions."""
import json
from datetime import datetime, timedelta, timezone
import torch
from audit_transfer import ROOT, REVIEW, digest, read


def main():
    assert not (ROOT / 'batch8/results.json').exists()
    events = [json.loads(line) for line in (REVIEW/'decisions.jsonl').read_text('utf-8').splitlines()]
    assert [e['id'] for e in events][-1] == 'B07-dispatch'
    strict = read(REVIEW/'strict-external-lock.json')
    models = {}
    for kind, label in (('mamba_v1', 'strict_mamba'), ('tcn', 'strict_tcn')):
        info = strict['checkpoints'][kind]
        models[label] = {'kind':kind,'path':info['path'],'sha256':info['sha256'],'epoch':info['epoch'],
                         'seed':info['seed'],'gain':1.,'intrinsic':False,'role':'只用外部反馈先行锁定'}
    for stage, label, seed, role in ((6,'transfer_mamba',1729,'主要迁移候选'),(7,'repeat_mamba',314159,'预定换种子复验')):
        receipt = read(ROOT / f'batch{stage}/results.json')
        path = (ROOT / f'batch{stage}' / receipt['weights']).resolve()
        models[label] = {'kind':'mamba_v1','path':path.relative_to(ROOT).as_posix(),'sha256':receipt['weights_sha256'],
                         'epoch':receipt['training']['selected_epoch'],'seed':seed,'gain':1.,'intrinsic':True,'role':role}
    checkpoints = {}
    for label, model in models.items():
        assert digest(ROOT/model['path']) == model['sha256']
        checkpoint = torch.load(ROOT/model['path'], map_location='cpu', weights_only=True)
        assert checkpoint['seed'] == model['seed'] and checkpoint['epoch'] == model['epoch']
        assert checkpoint['split_sha256'] == strict['source_split_sha256'] and checkpoint['window'] == 64
        checkpoints[label] = checkpoint
    primary, repeat = checkpoints['transfer_mamba'], checkpoints['repeat_mamba']
    assert all(torch.equal(primary[k], repeat[k]) for k in ('mean','std'))
    assert any(not torch.equal(v,repeat['state_dict'][k]) for k,v in primary['state_dict'].items())
    evidence=['external-split.json','review/strict-external-lock.json','batch4/results.json','batch5/results.json',
              'batch6/results.json','batch7/results.json','batch6/intrinsic.py','review/decisions.jsonl']
    proof={'locked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),
           'source_split_sha256':digest(ROOT/'external-split.json'),'evidence':{f:digest(ROOT/f) for f in evidence},
           'models':models,'baseline':'默认六轴 FullVQF 相同原始 IMU 与评分帧',
           'primary':'transfer_mamba 对照 full_vqf；每条原生评分帧的倾角 RMSE 后按记录等权。',
           'holdout':'R5、R6 各 10 条，单独及预定合并 20 条。合并与单设备均值同时报告。',
           'secondary':'strict_mamba / strict_tcn 外部先锁；repeat_mamba 初始化种子复验，不择优、不集成；保留全57条描述。',
           'target_exposure':'R1–R4 已用于诊断与训练表示方案的取舍。自采 57 条均不用于参数拟合、归一化、安装校准或挑选 epoch。旧实验曾公布 R5、R6 其他模型得分，本实验的留出定义限于新模型。',
           'heldout_prediction_files_seen_in_current_campaign':False,
           'independent_raw_checks':['R5-1','R5-7','R6-2','R6-9']}
    with (REVIEW/'final-transfer-lock.json').open('x',encoding='utf8') as stream:
        json.dump(proof,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(proof['locked_at'],digest(REVIEW/'final-transfer-lock.json'))


if __name__=='__main__':
    main()
