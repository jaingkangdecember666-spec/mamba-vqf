"""Publish the audited transfer study as a report chapter and a compact linked Markdown report."""
import json
from datetime import datetime
from statistics import fmean

from audit_transfer import MODELS, REVIEW, ROOT, digest, read, rows


def deg(value):
    return f'{value:.3f}°'


def effect(value, reference):
    direction = '降低' if value < reference else '升高'
    return f'{direction} {abs(value-reference):.3f}°（{abs(value/reference-1):.1%}）'


def main():
    proof = read(REVIEW/'transfer-audit.json')
    groups = proof['groups']
    hold, dev = groups['holdout'], groups['development']
    lock = read(REVIEW/'final-transfer-lock.json')
    result_paths = ['results.json', *(f'batch{i}/results.json' for i in range(2,9))]
    batches = [None, *(read(ROOT/p) for p in result_paths)]
    _, b1, b2, b3, b4, b5, b6, b7, b8 = batches
    start = datetime.fromisoformat(json.loads((REVIEW/'decisions.jsonl').read_text('utf8').splitlines()[0])['at'])
    end = datetime.fromisoformat(b8['completed_at'])
    minutes, seconds = divmod(int((end-start).total_seconds()),60)
    started_fit = read(ROOT/'run-protocol.json')['started_at']
    first_lock = read(REVIEW/'strict-external-lock.json')['locked_at']
    methods = ['纯 FullVQF','原坐标 Mamba','原坐标 TCN','内禀坐标 Mamba','同设置换种子 Mamba']
    columns = ['数据','条数', '纯 FullVQF','原坐标 Mamba','原坐标 TCN','内禀 Mamba','换种子复验']
    group_rows = []
    for key,name in [('holdout','封存 R5+R6'),('R5','R5'),('R6','R6'),('development','诊断 R1–R4'),('all','自采全部')]:
        g=groups[key]
        group_rows.append([name,str(g['records']),*(deg(g[m]['rmse_deg']) for m in MODELS)])
    fit_count = sum(len(b['training']) for b in [b1,b3]) + sum(b['training']['fits'] for b in (b5,b6,b7))
    histories = [v['history'] for b in (b1,b3) for v in b['training'].values()] + [b['training']['history'] for b in (b5,b6,b7)]
    assert len(histories) == fit_count == 7 and all(len(h)==6 for h in histories)
    assert all((h[-1].get('optimizer_steps') or h[-1].get('steps')) == 192 for h in histories)
    evidence, sections = {}, []

    def section(key,title,body,header,values,files,links=()):
        fields = [f'c{i}' for i in range(len(header))]
        assert all(len(row)==len(header) for row in values)
        evidence[key] = {'rows':[dict(zip(fields,row)) for row in values],
                         'source':{'label':title,'files':['../../transfer-mamba-20261123/'+p for p in files],
                                   'executedAt':end.isoformat()}}
        sections.append({'id':key,'title':title,'body':body,'queryId':key,
                         'columns':[{'field':f,'label':h} for f,h in zip(fields,header)]})
        parts.append('## '+title+'\n\n'+body)
        if values:
            parts.append('\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',
                                    *['| '+' | '.join(row)+' |' for row in values]]))
        if links:
            parts.append(' · '.join('['+caption+']('+url+')' for caption,url in links))

    primary, repeat = hold['transfer_mamba'],hold['repeat_mamba']
    baseline = hold['full_vqf']['rmse_deg']
    parts = ['# 外部训练的 Mamba 在自采转台上的迁移实验',
             f'FZR-MiniPC · 2026 年 11 月 23 日 · 北京时间 {start:%H:%M:%S} 至 {end:%H:%M:%S}，共 {minutes} 分 {seconds} 秒。']
    intro = (f'在本轮封存的 R5、R6 两台设备、20 条记录上，只用 BROAD 和 Sassari 训练的内禀坐标 Mamba '
             f'得到 **{deg(primary["rmse_deg"])}**，相对默认 FullVQF 的 **{deg(baseline)}** '
             f'{effect(primary["rmse_deg"],baseline)}，改善 **{primary["wins"]}/20 条**。'
             f'只换网络初始化种子得到 {deg(repeat["rmse_deg"])}，改善 {repeat["wins"]}/20 条。'
             f'在看任何新模型的自采分数之前就锁定的原坐标 Mamba 和 TCN 分别为 {deg(hold["strict_mamba"]["rmse_deg"])} 和 {deg(hold["strict_tcn"]["rmse_deg"])}。'
             '表内是每条记录先计算倾角 RMSE，再给各记录相同权重，越小越好。')
    section('transfer-mamba','外部训练的 Mamba 在自采数据上表现如何',intro,columns,group_rows,
            ['batch8/all-sequences.csv','review/transfer-audit.json','review/final-transfer-lock.json'],
            [('57 条逐记录成绩与评分帧指纹','../batch8/all-sequences.csv'),('独立对账','transfer-audit.json'),('最终锁定清单','final-transfer-lock.json')])
    splits=[['BROAD','27 条 / 15 个程序族','12 条 / 5 个程序族','同一程序的字母复本、时长或幅度变体在同一侧'],
            ['Sassari','慢、快两次运动，共 12 路','中速一次运动的 6 路','每次运动的 6 个传感器共用光学参考，整组划分'],
            ['SHICE R1–R4','0 条','37 条迁移诊断','R1 有 7 条，R2–R4 各 10 条。用于判断改动方向'],
            ['SHICE R5 / R6','0 条','各 10 条，最终一次评分','锁定两份种子、普通 Mamba、TCN 后开封，两台按相同程序测量']]
    section('transfer-split','外部训练、自采诊断与封存设备怎么划分',
            '源数据有 42 份 MAT 文件。39 条外部训练序列提供 60,202 个合格窗口（BROAD 33,652，Sassari 26,550）。外部验证 18 条、663,577 个原生帧。按运动或试验分组后才取窗口。'
            '自采数据沿用正式 CSV 和编码器质量掩码。原有 59 条文件，R1-2 缺采，R1-3 与 R1-4 因内框编码器 A2 的独立时钟对齐待核查而剔除。剩余 57 条均未参与梯度更新、归一化、安装角校准或选训练轮。'
            'R1–R4 用来判断后续结构，R5、R6 的新模型得分只在本轮最终锁定后计算。旧实验曾评测过这两台设备。相同程序跨设备复用，结果衡量的是现有转台程序的跨设备迁移。TUM-VI 未纳入本轮。',
            ['数据集','训练','验证或测试','分组依据'],splits,['external-split.json','fit-scope.json','review/strict-external-lock.json','batch4/protocol.json','review/final-transfer-lock.json'],
            [('外部分组与文件哈希','../external-split.json'),('训练窗口清单','../fit-scope.json'),('首次锁定','strict-external-lock.json')])

    external_cols=['模型或推断设置','BROAD 12 条','Sassari 6 路','两域等权']
    def from_json(table,key,field):
        return [table[d][field][key] for d in ('BROAD','Sassari')]
    def from_csv(stage,field):
        entries=rows(ROOT/f'batch{stage}/external-sequences.csv')
        assert len(entries)==18
        return [fmean(float(r[field]) for r in entries if (r.get('group') or r.get('domain'))==d) for d in ('BROAD','Sassari')]
    comparison=[('纯 FullVQF',from_json(b1['summary'],'full_vqf','macro_rmse_deg')),
                ('初版 Mamba',from_json(b1['summary'],'mamba_v1','macro_rmse_deg')),
                ('初版 TCN',from_json(b1['summary'],'tcn','macro_rmse_deg')),
                ('初版 Mamba，常数缩放',from_json(b2['summary'],'mamba_constant','macro_rmse_deg')),
                ('初版 Mamba，角速度门控',from_json(b2['summary'],'mamba_angular_gate','macro_rmse_deg')),
                ('对齐监督的原坐标 Mamba',from_json(b3['summary'],'mamba_v1_selected_full','rmse_deg')),
                ('对齐监督的原坐标 TCN',from_json(b3['summary'],'tcn_selected_full','rmse_deg')),
                ('旋转增强 Mamba',from_csv(5,'batch5_mamba_deg')),('内禀坐标 Mamba',from_csv(6,'batch6_mamba_deg')),
                ('内禀坐标 Mamba，换种子',from_csv(7,'batch7_mamba_deg'))]
    external_replay = read(REVIEW/'intrinsic_external_replay.json')
    external_gap = max(r['max_abs_delta_deg'] for r in external_replay)
    section('transfer-external','在两套外部验证数据上的表现',
            '每个模型从 6 个训练轮中按 18 条外部验证序列的等权均值保留最佳检查点。表末列用于跨模型分析，先分别计算两域均值，再各占一半。'
            '外部先锁的 TCN 在两域均值上优于内禀坐标 Mamba。Sassari 的 6 个验证通道来自同一次中速运动。'
            f'从原始 MAT 文件重放全部 18 条的两个内禀坐标模型，逐条分数与执行方的最大差为 {external_gap:.9f}°。',external_cols,
            [[label,deg(v[0]),deg(v[1]),deg(fmean(v))] for label,v in comparison],
            ['results.json','batch2/results.json','batch3/results.json',*(f'batch{k}/external-sequences.csv' for k in (5,6,7))],
            [('内禀坐标逐条外部验证','../batch6/external-sequences.csv'),('换种子逐条外部验证','../batch7/external-sequences.csv'),('原始外部数据独立重放','intrinsic_external_replay.json')])

    stage_rows = [
        ['1','2','外部训练 Mamba 和同预算 TCN','Mamba 改善 Sassari，BROAD 慢平移 A 从 0.299° 升至 0.894°'],
        ['2','0','固定 Mamba 权重，比较统一缩放和角速度门控','两域等权分别为 0.959°、1.064°，停用角速度门控'],
        ['3','2','把发布时刻的训练标签转回输入末帧坐标再重训','两域等权 Mamba 0.958°、TCN 0.928°，冻结两份权重'],
        ['4','0','先用两份冻结权重评测 R1–R4','Mamba 2.555°，只赢纯滤波 6/37 条。两种模型四台设备均退化'],
        ['5','1','只在外部训练窗口施加随机安装旋转','Mamba 2.447°，比上版改善 33/37 条，对纯滤波胜 12/37 条，其中 10 条在 R2'],
        ['6','1','从每窗重力和运动求正交坐标，固定网络预算','Mamba 2.305°，比两种上一版均改善 37/37 条，对纯滤波胜 33/37 条'],
        ['7','1','只把网络初始化种子从 1729 改为 314159',f'R1–R4 得 {deg(dev["repeat_mamba"]["rmse_deg"])}，复验与主种子均保留'],
        ['8','0','锁定四份权重及纯 FullVQF，统一打开 R5、R6',f'主种子在 20 条留出记录上的平均误差为 {deg(primary["rmse_deg"])}，复验为 {deg(repeat["rmse_deg"])}']]
    section('transfer-rounds','八个小批次分别改了什么',
            f'每批交回逐条成绩，审查后才派发下一批。北京时间从 {start:%H:%M:%S} 首次登记小批到 {end:%H:%M:%S} 最终评分落盘，共 {minutes} 分 {seconds} 秒。'
            f'第一批正式训练进程启动于 {started_fit[11:19]}。共完成 **8 批、{fit_count} 次有效训练、6 种训练配置**，含一次只换初始化种子的复训。每次固定 192 个优化步，共 1,344 步。'
            '推断设置、检查点、逐设备评分和审查重放均单独记账。开头两次调度请求因模型容量不足而中止，当时尚未开始训练。',
            ['批次','新训练数','本批改动','审查后得到的结果'], stage_rows,
            [*result_paths,'review/decisions.jsonl'],[('逐批决策及时间戳','decisions.jsonl')])

    dev_rows=[]
    for d in ('R1','R2','R3','R4','R5','R6'):
        g=groups[d]
        dev_rows.append([d,str(g['records']),deg(g['full_vqf']['rmse_deg']),deg(g['transfer_mamba']['rmse_deg']),
                         str(g['transfer_mamba']['wins'])+'/'+str(g['records']),deg(g['repeat_mamba']['rmse_deg'])])
    lo = proof['heldout_leave_one_program_out_delta_deg']['transfer_mamba']
    best, worst=primary['best'],primary['worst']
    section('transfer-devices','哪些设备和程序受益',
            f'表内胜场按同一条记录的倾角 RMSE 低于纯滤波计。R5、R6 的 20 条记录共有 {hold["frames"]:,} 个评分帧，自采 57 条合计 {groups["all"]["frames"]:,} 帧。'
            f'主种子在留出记录中相对纯滤波变化最大的一好一差是 {best[1]}（{best[0]:+.3f}°）和 {worst[1]}（{worst[0]:+.3f}°）。'
            f'每次将同一程序在两台设备上的记录一起移除，余下 18 条的平均误差差值在 {min(lo):+.3f}° 至 {max(lo):+.3f}° 之间。'
            'R1–R4 曾用于选择表达方式，57 条合计仅描述整库表现。',
            ['设备','条数','纯 FullVQF','内禀 Mamba','主种子胜场','换种子复验'],dev_rows,
            ['batch8/all-sequences.csv','review/transfer-audit.json'],[('逐记录与帧数','../batch8/all-sequences.csv')])

    a=[read(REVIEW/f'axis_response_{s}.json') for s in ('strict','so3','intrinsic')]
    section('transfer-method','时间对齐与安装方向怎样影响预测',
            'Mamba v1 使用单个 5,595 参数的选择性状态空间模块，从随机权重训练。TCN 有 5,503 个参数。两者都在默认六轴 FullVQF 上预测一个限幅 6° 的倾角修正，使用过去 0.64 秒的 IMU、重力方向、陀螺零偏和静止标记，每秒更新 10 次。'
            'BROAD 的 33,652 个有效训练窗口中，28,844 个的最后输入比标签发布时刻早 3.5 毫秒。将两时刻当作同一个身体坐标系，监督方向的 RMS 差达到 0.643°。第 3 批只修正了这个坐标变换。'
            '换轴复核在 576 个外部窗口进行，对同一段信号施加 4 种安装旋转，比较还原到原坐标后的预测。随机旋转增强减小了预测对安装方向的敏感度，也缩小了修正本身。内禀坐标以末帧重力为竖轴、窗内横向加速度为横轴，必要时改用横向角速度。',
            ['模型','同一运动换轴后的预测差 RMS','原信号上的修正幅度 RMS'],
            [[n,f'{v["frame_change_rms_deg"]:.7f}°',deg(v['correction_rms_deg'])] for n,v in zip(['原坐标','旋转增强','内禀坐标'],a)],
            ['review/training_target_audit.json','review/axis_response_strict.json','review/axis_response_so3.json','review/axis_response_intrinsic.json'],
            [('标签时间审查','training_target_audit.json'),('Mamba 逐步状态重放','batch1_mamba_parity.json'),('所用 Mamba v1 官方实现','https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mamba/modeling_mamba.py')])

    history = proof['historical_r6']
    anchor_max = max(r[m] for r in proof['raw_anchor_deviations_deg'] for m in MODELS)
    section('transfer-history','同一台 R6 与以往本地训练结果相比',
            f'R6 的 10 条记录采用相同的评分帧。上一轮使用自采数据调参的锁定候选为 {deg(history["locked"])}，本轮外训主候选为 {deg(groups["R6"]["transfer_mamba"]["rmse_deg"])}。训练条件不同，这列历史成绩用于判断到现有方案的距离。'
            f'本轮先于 {first_lock[11:19]} 锁定普通 Mamba/TCN，再于 {lock["locked_at"][11:19]} 锁定最终四份权重。57 条评分掩码均与历史对上。'
            f'独立读取 4 条封存记录的原始 CSV 并重新推断，五种方法的逐条误差与执行方最大相差 {anchor_max:.8f}°。'
            '所有成绩只衡量倾角。外部训练时的速度标签与传感器安装条件，以及本轮只有两台封存设备，限定了可外推范围。',
            ['R6 对照','倾角 RMSE'],[[n,deg(history[k])] for n,k in [('默认 FullVQF','full_vqf'),('历史原轻量修正','physical_only'),('历史 TimesFM','timesfm'),('历史 GRU','gru'),('历史锁定候选 N00035','locked')]]+
            [[methods[i],deg(groups['R6'][MODELS[i]]['rmse_deg'])] for i in (3,4)],
            ['../overnight-20261122/rounds/R01826/result.json','review/final_raw_anchors.json','review/transfer-audit.json'],
            [('旧实验锁定 R6 成绩','../../overnight-20261122/rounds/R01826/result.json'),('原始 CSV 独立重放','final_raw_anchors.json')])

    (REVIEW/'REPORT.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    (REVIEW/'report-addendum.json').write_text(json.dumps({'queries':evidence,'sections':sections},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'{len(sections)} sections, {sum(len(q["rows"]) for q in evidence.values())} table rows')


if __name__ == '__main__':
    main()
