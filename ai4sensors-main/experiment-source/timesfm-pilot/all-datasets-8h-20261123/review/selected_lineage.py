"""Report which external acquisitions actually contributed to the selected estimator, deduplicated by source and record."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from lock_from_frontier import ROOT,read,digest


def identities(path,arr,domain='domain',name='record'):
    return {(r[domain],r[name])for r in read(ROOT/path)[arr]}


def base_panel():
    with np.load(ROOT/'C019/training-inputs.npz')as f:
        return set(zip(map(str,f['domain']),map(str,f['source_record_name'])))


def generate(lock_path):
    lock=read(lock_path)
    model=lock['review_inference'];rows=[];union=set()
    def add(title,stage,records,source):
        union.update(records)
        rows.append({'component':title,'trained_at':stage,'source_counts':dict(sorted(Counter(d for d,n in records).items())),
                     'distinct_input_records':len(records),'evidence':source,'evidence_sha256':digest(ROOT/source)})
    history='C013/seed-lineage.json'
    original=identities(history,'historical_train_records',name='sequence')
    sixty_three=identities('C004/protocol.json','source_records')
    sixty_seven=identities('C017/external-selection.json','records')
    full=base_panel()
    assert [len(a) for a in (original,sixty_three,sixty_seven,full)]==[39,63,67,139]
    assert original<=sixty_three<=sixty_seven<=full
    for h in model['heads']:
        if h['weights'].endswith('batch6/mamba-intrinsic-selected.pt'):
            add('种子 1729 的 Mamba 主干与方向输出','既有 B6',original,history)
        elif h['weights']=='C013/seed-C006-readout.pt':
            add('种子 314159 的 Mamba 主干','既有 B7',original,history)
            add('种子 314159 主干的方向输出层','C013',sixty_three,'C013/training.json')
        else:
            raise ValueError(f"Document source identity for {h['weights']}")
    if model.get('stable_force'):
        add('方向稳定时的比力修正系数','C017',sixty_seven,'C017/external-selection.json')
    for extra in model.get('additions',[]):
        if extra['weights']!='C020/mamba-end-to-end.pt':
            raise ValueError(f"Document the extra Mamba source identity for {extra['weights']}")
        add('可训练的历史状态 Mamba，5,595 个参数','C020',full,'C020/training.json')
    if model.get('response_bank') or model.get('response_router'):
        add('快慢重力估计的外训先验系数','C022',full,'C022/external-selection.json')
    rule=model.get('response_router')
    if rule:
        batch=Path(rule['weights']).parent.name
        receipt=next(f for f in ('solver-receipt.json','synthesis-verification.json','training.json') if (ROOT/batch/f).is_file())
        manifest=read(ROOT/batch/receipt)
        supplied=next((ROOT/batch/f for f in ('training-inputs.npz','teacher-slots.npz') if (ROOT/batch/f).is_file()),None)
        if supplied is not None:
            with np.load(supplied)as inputs:
                sources=set(np.unique(inputs['domain'])) if 'domain' in inputs.files else set(manifest['domains'])
        elif batch=='C038':
            source_path=ROOT/batch/'source-replay.npz'
            if source_path.exists():
                with np.load(source_path)as inputs:sources=set(np.unique(inputs['domain']))
            else:
                assert manifest.get('original_forward_audit_records')==141
                with np.load(ROOT/'C033/training-inputs.npz')as inputs:sources=set(np.unique(inputs['domain']))
        elif receipt=='synthesis-verification.json':
            with np.load(ROOT/batch/'source-replay.npz')as inputs:sources=set(np.unique(inputs['domain']))
        elif manifest.get('solver_kind')=='neural_readout':
            with np.load(ROOT/'C033/training-inputs.npz') as inputs:sources=set(np.unique(inputs['domain']))
        elif 'domains' in manifest:
            sources=set(manifest['domains'])
        else:
            panel='C026/training-inputs.npz'
            assert read(ROOT/batch/'protocol.json')['inputs_sha256'][panel]==digest(ROOT/panel)
            with np.load(ROOT/panel)as inputs:sources=set(np.unique(inputs['domain']))
        selected=set(full)
        if 'DAVIS' in sources:
            selected|={('DAVIS',r['record'])for r in read(ROOT/'C024/training-preparation.json')['rows']}
        if any('euroc' in s.lower() for s in sources):
            selected|={('EuRoC',r['record'])for r in read(ROOT/'review/euroc_bias_response.json')['records']}
        if 'PennCOSYVIO' in sources:
            selected|={('PennCOSYVIO',r['record'])for r in read(ROOT/'external-sources/penn-handheld/preprocessing-lock.json')['source_records']}
        if batch not in ('C023','C024','C025','C026','C027','C028','C029','C030','C031','C033','C035','C038'):
            # Confirm new runs' receipts before publishing additional mechanisms or cohorts.
            raise ValueError(f'Document the selected source lineage for {batch}')
        expected=manifest.get('source_records',manifest.get('original_recordings'))
        if isinstance(expected,int):assert len(selected)==expected,(batch,len(selected),expected)
        if manifest.get('solver_kind')=='neural_readout':
            add('原始 Mamba 历史响应选择器的起始权重','C023',full,'C023/training.json')
            add('用五个外部来源更新的 50 个历史响应权重',batch,selected,batch+'/'+receipt)
        elif receipt=='solver-receipt.json':
            add('已冻结的 Mamba 历史响应选择器','C023',full,'C023/training.json')
            add('由外训拟合的四个物理系数',batch,selected,batch+'/'+receipt)
        elif batch=='C038':
            assert receipt=='synthesis-verification.json' and manifest['new_optimizer_steps']==0 and manifest['original_forward_audit_records']==len(selected)==141
            add('第一个父模型的 50 个历史响应权重','C023',full,'C023/training.json')
            add('第二个父模型在五个来源上约束训练的 50 个权重','C035',selected,'C035/solver-receipt.json')
            add('对两个父模型的读出权重取逐参数中点','C038',selected,'C038/'+receipt)
        elif receipt=='synthesis-verification.json':
            assert manifest['new_optimizer_steps']==0 and manifest['original_forward_audit_records']==len(selected)
            add('两种训练顺序的后半段权重平均','C023、C027 → '+batch,selected,batch+'/'+receipt)
        else:
            add('发布时选择重力响应的模型',batch,selected,batch+'/'+receipt)
    if gate:=model.get('raw_anchor_blend'):
        add('依据运动特征混合默认方向与模型方向的四个外训系数','C034',
            full|{('PennCOSYVIO',r['record']) for r in read(ROOT/'external-sources/penn-handheld/preprocessing-lock.json')['source_records']},gate['receipt'])
    if cap:=model.get('physical_angle_cap'):
        add('Mamba 预测相对实时物理方向的最大转角',Path(cap['receipt']).parent.name,
            full|{('PennCOSYVIO',r['record'])for r in read(ROOT/'external-sources/penn-handheld/preprocessing-lock.json')['source_records']},cap['receipt'])
    return {'selection_lock_sha256':digest(lock_path),'components':rows,'distinct_external_source_records':len(union),
            'unique_by_source':dict(sorted(Counter(d for d,n in union).items())),
            'records':sorted([{'source':d,'record':n}for d,n in union],key=lambda r:(r['source'],r['record'])),
            'own_fitting_records':0,'development_self_records':37,
            'scope':'Counts by original acquisition or recorded sensor channel, deduplicated within the named source across components. Only external sources that entered selected weights or constants are included. Online per-record state estimation uses previous six-axis input. SHICE R1–R4 influenced development decisions; SHICE records never entered fitting or normalization.'}


if __name__=='__main__':
    path=ROOT/'selection-lock.json'
    if not path.is_file():raise SystemExit('The overall selection must be sealed before the final lineage is published.')
    result=generate(path)
    (ROOT/'review/selected-training-sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(result['unique_by_source'],result['distinct_external_source_records'])
