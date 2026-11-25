"""Produce the formal report sections strictly from completed, independently checked artifacts."""
import csv
import json
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parent.parent
REVIEW=ROOT/'review'
ORDER=('SHICE','BROAD','Sassari','TUM-VI','IMU-Array','OxIOD','RepoIMU','YT_Server')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def angle(x):
    return f'{x:.5f}°'


def append(add,markdown):
    verified=read(REVIEW/'final-verification.json')
    lock=read(ROOT/'selection-lock.json')
    candidate=lock['selected']
    short_column=candidate['column'].removesuffix('_deg')
    name=candidate['batch'] if candidate['batch']==short_column else f"{candidate['batch']} / {short_column}"
    values=[]
    for d in ORDER:
        stats=verified['main'][d]
        if stats['scored']:
            m=stats['means']
            nums=[angle(m['full_vqf_deg']),angle(m['physical_deg']),angle(m['candidate_deg']),
                  f"{stats['improvement_percent']:+.3f}%",str(stats['groups']['improving_records'])]
        else:
            nums=['—','—','—','—','—']
        values.append([d,f"{stats['scored']}/{stats['requested']}",*nums])
    n=sum(s['scored'] for s in verified['main'].values())
    add('all8-final','八套数据的锁模终测',
        f"北京时间 {lock['locked_at'][11:19]}，按 213 条开发记录的预定排序选定 {name}。同一配方用于全部数据。365 条正式记录中 {n} 条可评分，"
        f"{verified['main_improved_domains']}/8 个域的逐记录平均倾角 RMSE 低于默认 FullVQF，{verified['main_complete_domains']}/8 个域有完整分母。"
        '纯物理列保留同一前端和物理修正，取消各个 Mamba 修正。BROAD 的 12 条和 Sassari 的 6 路在本轮参与过开发排序，终表沿用相同记录作为基准。自采 R5/R6 在本轮按设备封存，过去的独立实验曾公布过成绩。',
        ['数据集','已评分/预定','默认 FullVQF','纯物理','冻结模型','相对改善','改善记录数'],values,
        ['selection-lock.json',verified['main_table'],'review/final-verification.json','review/final-anchor-spec.json'])
    markdown.append(f"[正式逐记录成绩](../{verified['main_table']}) · [选定配方与哈希](../selection-lock.json) · [正式分母与独立原生重放](final-verification.json)")
    coordination=REVIEW/'guidance-verification.json'
    if coordination.is_file():
        g=read(coordination)
        assert g['lock_sha256']==verified['lock_sha256']
        with (REVIEW/'guidance-receipts.csv').open(encoding='utf-8-sig',newline='')as src:
            dispatches=list(csv.DictReader(src))
        examples={'C001':'先冻结划分，复测已有 Mamba','C010':'对照恢复原始陀螺后的外训',
                  'C027':'只改变训练抽样次序','C037':'核查一维求解器遗漏的小角度区间',
                  'C038':'待上批汇总恢复后，检验两份读出权重的中点'}
        values=[[r['batch'],examples[r['batch']],r['received_in_worker'][11:19],r['registered_job_start'][11:19],
                 r['registered_result'][11:19],r['records_scored']] for r in dispatches if r['batch'] in examples]
        add('all8-supervision','38 个小批怎样由两个聊天共同推进',
            f"逐批将冻结的决策账本与执行聊天实际收到的消息、作业起止记录对齐。{g['separate_batch_directives_received']}/38 个小批都在登记作业开始前接到本聊天单独下达的指令。"
            f"首批之后的 37 次交接中，{g['after_previous_summary']} 次在上一批完成汇总后发出。"
            f"另一次是 C038，指令发出时已有 C037 全部 213 条逐序列成绩，汇总还在恢复，C038 实际启动时上一批已经收尾。"
            f"核对的 {g['strictly_nonoverlapping_registered_batches']} 对相邻作业时间均无重叠。表中时间为北京时间，完整 38 条回执附在下方。",
            ['小批','本次指令','执行聊天收到','作业开始','结果登记','开发记录'],values,
            ['review/guidance-verification.json','review/guidance-receipts.csv','review/decisions.jsonl','review/campaign-counts.json'])
        markdown.append('[38 次实际送达、作业起止与消息指纹](guidance-receipts.csv) · [每批为什么这样改](decisions.jsonl)')
    extra_files=[REVIEW/'final-supplemental/summary.json',REVIEW/'final-davis/summary.json',REVIEW/'final-penn/summary.json']
    if all(p.is_file() for p in extra_files):
        x,davis,penn=map(read,extra_files)
        assert x['hashes']['selection_lock']==davis['selection_lock_sha256']==penn['selection_lock_sha256']==verified['lock_sha256']
        all_metrics=[(d,'主评',verified['main'][d]['scored'],verified['main'][d]['requested'],verified['main'][d].get('means',{})) for d in ORDER]
        for d in ('EuRoC','RIDI'):
            r=x['domains'][d]
            all_metrics.append((d,'补充',r['statuses'].get('scored',0),r['requested'],r.get('means',{})))
        all_metrics.extend([('DAVIS','补充',davis['records'],3,davis['means']),('PennCOSYVIO','补充',penn['scored'],2,penn['means'])])
        outcomes=[];wins=[];losses=[];unscored=[]
        for d,kind,scored,total,m in all_metrics:
            if m:
                gap=m['candidate_deg']-m['full_vqf_deg']
                (wins if gap<0 else losses).append(d)
                cells=[angle(m['full_vqf_deg']),angle(m['candidate_deg']),f'{gap:+.5f}°',f"{100*(1.-m['candidate_deg']/m['full_vqf_deg']):+.2f}%"]
            else:
                unscored.append(d);cells=['—']*4
            outcomes.append([d,kind,f'{scored}/{total}',*cells])
        weak='。误差较高的有 '+'、'.join(losses) if losses else ''
        missing='。尚缺分数的数据集有 '+'、'.join(unscored) if unscored else ''
        exposed=read(ROOT/'final/exposure-audit.json')['development_main_reuse']['records']
        reused={d for d,_ in exposed}
        separate={d for d,_,_,_,_ in all_metrics}-reused
        comparison=f"单看本轮评估记录与开发记录分开的 {len(separate)} 套，有 {len(separate & set(wins))} 套胜过默认 VQF。"
        chart_rows=add('all8-final-cross-source','同一模型在全部有评估计划的数据上是否超过 VQF',
            f'同一冻结模型在 {len(wins)}/{len(all_metrics)} 套数据上的平均倾角 RMSE 严格低于默认 FullVQF{weak}{missing}。{comparison}每条记录按预测与参考重力方向的夹角计算 RMSE，再在数据集内等权平均。指标衡量横滚、俯仰方向误差，航向角不参与评分。八套主评来自 E 盘，另外四套检验外部迁移。RIDI 的两条缺档保留在预定分母中。',
            ['数据集','评估范围','实评/预定','默认 FullVQF','冻结模型','模型减默认','相对改善'],outcomes,
            ['review/final-verification.json','final/exposure-audit.json',*[p.relative_to(ROOT).as_posix() for p in extra_files]])
        for row,(_,_,_,_,m)in zip(chart_rows,all_metrics):
            row['gain_fraction']=1.-m['candidate_deg']/m['full_vqf_deg'] if m else None
        markdown.append('![同一冻结模型在十二套数据上的相对误差改善](final-domain-gains.png)')
    dev_file=REVIEW/'development_replay_C038_C038_deg.json'
    if dev_file.exists():
        dev=read(dev_file)
        assert dev['batch']==candidate['batch'] and dev['n_records']==213
        values=[]
        formal_ids={(r['dataset'],r['record']) for r in lock['formal_cases']}
        for d in ORDER:
            records=[r for r in dev['rows'] if r['dataset']==d]
            shared=sum((d,r['record']) in formal_ids for r in records)
            before=fmean(r['full_vqf_deg'] for r in records)
            after=fmean(r['candidate_deg'] for r in records)
            formal=verified['main'][d]
            values.append([d,str(len(records)),f'{100*(1.-after/before):+.3f}%',
                           f"{formal['scored']}/{formal['requested']}",
                           f"{formal['improvement_percent']:+.3f}%" if formal['scored'] else '—',str(shared)])
        add('all8-validation-transfer','开发时的优势在正式记录上保留了多少',
            f"锁定第 38 批后，用审查端实现完整重放 213 条开发记录、{dev['sum_scoring_frames']:,} 个原生评分时刻，与执行端计算的最大逐条误差差值为 {dev['max_abs_gap_deg']:.3g}°。两列均相对默认 FullVQF，正值表示改善。最后一列统计开发与正式名单中的同名原始记录。",
            ['数据集','开发记录','开发改善','正式覆盖','正式改善','两份名单共享记录'],values,
            ['review/development_replay_C038_C038_deg.json','C038/sequences.csv',verified['main_table'],'selection-lock.json'])
    causality=REVIEW/'final-causality.json'
    if causality.is_file():
        proof=read(causality);assert proof['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        for r in proof['cases']:
            if r['status']!='checked':
                values.append([r['dataset'],r['record'],r['status'],r['detail'],'—','—'])
                continue
            original,drop=r['comparisons']
            num=lambda c:f"{max(c['max_direction_change_deg']['candidate_deg'].values()):.3g}°"
            values.append([r['dataset'],r['record'],f"{r['native_rate_hz']:.1f} Hz",str(original['paired_past_rows']),num(original),num(drop)])
        n=sum(r['status']=='checked'for r in proof['cases'])
        add('all8-final-causality','改变后续数据会不会改变过去的估计',
            f'12 套数据各取一条，{n} 条通过预先登记的前缀测试。截短原始 IMU，或改写尚未到达的加速度、角速度和缺包位置，比较之前全部有效时刻的预测方向。最后一列另在已到达数据中人为删去两包，再对同一段过去作比较。容差为 0.0001°。',
            ['数据集','记录','原生频率','历史有效行','原始输入最大角差','历史丢两包后的最大角差'],values,
            ['review/final-causality.json','review/check_locked_causality.py','review/FINAL_PROTOCOL.md'])
    lineage=REVIEW/'selected-training-sources.json'
    extended=REVIEW/'all-native-verification.json'
    if extended.is_file():
        full_check=read(extended);assert full_check['lock_sha256']==verified['lock_sha256']
        ranges={'main':'八套主评','transfer':'四套补充','additional':'全库补足'}
        values=[]
        for k,n in ranges.items():
            panel=full_check['panels'][k]
            values.append([n,f"{panel['verified']}/{panel['requested']}",f"{panel['native_frames']:,}",
                           *[f"{panel['max_difference_deg'][col]:.3g}°" for col in ('full_vqf_deg','physical_deg','candidate_deg')]])
        add('all8-full-implementation','逐条用两份推理实现重算正式成绩',
            f"主表发布后，把原先登记的抽查扩大到全部记录。{full_check['requested']} 条中有 {full_check['scored']} 条可评分，共 {full_check['native_frames']:,} 个原生评价时刻。正式作业使用审查端的推理实现，复核使用开发端的模型前向与四元数误差计算。两端共享原件、参考对齐、VQF 库、部分分块特征和冻结权重。逐条原生索引、帧数及三列分数均对账，{full_check['verified']} 条在 0.0002° 容差内。两条 RIDI 缺失原件在两端均保留缺分。",
            ['范围','两端通过/预定','原生评价时刻','默认最大差值','纯物理最大差值','Mamba 最大差值'],values,
            ['review/all-native-verification.json','review/full-native-verification/plan.json','review/verify_all_native.py',
             'review/alternate_native.py',verified['main_table'],verified['additional_table'],'final/transfer.csv'])
    audit=REVIEW/'final-transfer-crosscheck.json'
    directions=REVIEW/'transfer-correction-directions.json'
    source_check=REVIEW/'source-side-transfer.json'
    if source_check.is_file():
        diag=read(source_check)
        assert diag['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        agreement=0
        for d,scope in diag['domains'].items():
            seen,held=scope['source_side'],scope['formal_transfer']
            agreement+=int((seen['gain_percent']>0)==(held['gain_percent']>0))
            values.append([d,str(seen['records']),str(scope['source_used_by_winner']),angle(seen['full_vqf_deg']),
                           angle(seen['candidate_deg']),f"{seen['gain_percent']:+.2f}%",f"{held['gain_percent']:+.2f}%"])
        add('all8-source-side-transfer','外部测试的胜负，在原有来源上能否看出来',
            f"锁模后，用相同权重重放开发期间已有的 15 条外部源记录，按每份原件中可用的整段时刻计算误差。这四个来源有 {agreement}/4 个在来源侧与正式迁移检验中呈现相同胜负。RIDI 四条、Penn 两条参与了最终权重的训练。EuRoC 和 DAVIS 仅在其他候选中参与训练。EuRoC V1 采用已归档的时段、还原的原始陀螺和原发布姿态。此项回顾用于定位外部退步，没有改变选型。",
            ['来源','来源侧记录','进入最终配方的源记录','来源侧默认 VQF','来源侧模型','来源侧改善','正式迁移改善'],values,
            ['review/source-side-transfer.json','review/check_source_side_transfer.py','review/selected-training-sources.json',*list(diag['test_tables_sha256'])])
    if directions.is_file():
        diagnose=read(directions);assert diagnose['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        for d,item in diagnose['domains'].items():
            values.append([d,str(item['records']),angle(item['physical_deg']),angle(item['candidate_deg']),
                           f"{item['record_delta_deg']:+.5f}°",f"{item['records_improved_over_physical']}/{item['records']}",
                           f"{item['away_fraction_given_active']:.1%}",f"{item['lower_error_fraction_given_active']:.1%}"])
        add('all8-transfer-directions','外部迁移时，神经修正把估计推向了哪里',
            '两列均使用相同的物理前端。冻结模型在此基础上启用已训练的 Mamba 分支。本表是锁模后的误差分析。逐帧比较完整模型和纯物理的重力方向，变化超过 0.01° 时，用球面切向量检查修正是否背离参考，也直接比较修正前后的夹角。先在每条录制内计算，再按记录等权平均。',
            ['外部来源','记录','纯物理 RMSE','冻结模型 RMSE','模型减纯物理','模型赢物理的记录','修正背离参考','修正后夹角更小'],values,
            ['review/transfer-correction-directions.json','review/describe_transfer_residuals.py',*list(diagnose['external_table_hashes'])])
    if audit.is_file():
        t=read(audit);assert t['selection_lock_sha256']==verified['lock_sha256']
        values=[[d,str(r['records']),f"{r['scored_frames']:,}",f"{r['max_abs_gap_deg']:.3g}°"]for d,r in t['by_source'].items()]
        add('all8-transfer-audit','两份预测实现对同一条外部原件的结果',
            f"四套补充检验登记 {t['requested']} 条，{t['checked_records']} 条可评分，另有 {len(t['unscored_in_both'])} 条原件缺档。主作业和补充脚本共用一套推理；另用开发阶段的 C038 神经推理与 C017 四元数误差实现，从原件独立计算三列。两套实现共享冻结权重、VQF 库、部分分块特征及参考对齐。逐条核对原生行索引哈希、帧数和角误差，容许差值 0.0002°。",
            ['外部来源','已交叉核对记录','原生行','三项误差最大差值'],values,
            ['review/final-transfer-crosscheck.json','review/check_transfer_cross.py',*list(t['score_tables_sha256'])])
        independent=read(REVIEW/'final-alternate-check.json')
        markdown.append(f"八个主评域预登记的 14 条原生锚点中，{independent['checked_records']} 条也用开发端实现独立重放。默认 VQF、纯物理和 Mamba 三项中，最大差值为 {independent['max_main_gap_deg']:.3g}°。[逐锚点结果与代码指纹](final-alternate-check.json)")
    if lineage.is_file():
        chain=read(lineage)
        assert chain['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        for component in chain['components']:
            sources='、'.join(f'{k} {v}'for k,v in component['source_counts'].items())
            values.append([component['component'],component['trained_at'],sources,str(component['distinct_input_records'])])
        add('all8-selected-training','最终模型的每个部件由哪些数据训练',
            f"把同一来源的同一次采集或传感器通道跨部件去重，最终配方实际用了 {chain['distinct_external_source_records']} 条外部记录。自采数据进入离线拟合与归一化的数量为 0。R1–R4 的 37 条开发记录影响了选型；部署时每条记录只从已到达的六轴观测估计在线状态。",
            ['冻结部件','训练批次','实际外训来源与记录数','去重记录'],values,
            ['review/selected-training-sources.json','selection-lock.json']+sorted({c['evidence'] for c in chain['components']}))
        markdown.append('[实际参与的外部记录名单与每个部件的凭据](selected-training-sources.json)')
    exposure=ROOT/'final/exposure-audit.json'
    if exposure.is_file():
        e=read(exposure);assert e['selection_sha256']==verified['lock_sha256']
        overlap=e['direct_fitting_overlap']['counts'];yt=e['yt']
        add('all8-exposure','训练、选型与终测的原件有没有重叠',
            f"将最终模型的 {e['fitting_acquisitions']} 条外训原件按来源、原件名与三份评估清单相交。开发与主评重用 {e['development_main_reuse']['count']} 条，来自 BROAD 12 条和 Sassari 6 路。YT 外训与正式 test 的采集前缀没有重合，但 test 的 {len(yt['overlapping_leading_codes_with_main']['fitting'])} 个文件名首段代号在外训出现过，覆盖 {yt['main_rows_with_overlapping_leading_codes']['fitting']} 条 test 记录。尚无发布方身份映射来证明跨人独立。",
            ['清单','登记数','与外训原件重合','用途'],
            [['八套主评','365',str(overlap['main']),'正式比较，含开发期复用的 18 条'],
             ['四套补充测试','33',str(overlap['transfer']),'独立原件，31 条可评分'],
             ['另外的全库补足','76',str(overlap['additional']),'补足自采、BROAD、Sassari 的全库汇总']],
            ['final/exposure-audit.json','review/selected-training-sources.json',verified['main_table'],verified['additional_table'],'final/transfer.csv'])
    timing=REVIEW/'delivery-timing.json'
    if timing.is_file():
        tm=read(timing)
        stamps=[('授权开始',tm['window_start']),('第一批作业登记',tm['first_registered_batch']),
                ('最后一批开发成绩落盘',tm['last_development_result']),('锁定唯一终测模型',tm['locked_at']),
                ('正式主表落盘',tm['main_file_written_at']),('正式单作业结束',tm['formal_job_finished_at']),
                ('最后一项数值核验落盘',tm['science_completed_at']),('38 次指导送达记录核完',tm['guidance_verified_at']),('授权截止',tm['window_end'])]
        add('all8-final-elapsed','八小时窗口的实际用时',
            f"从授权到最后一项科学核验实际经过 {tm['science_elapsed_from_start_s']/3600:.3f} 小时。开发阶段各作业的登记进程去除重叠后为 {tm['registered_development_process_seconds']/60:.1f} 分钟。单一正式评分作业耗时 {tm['formal_job_seconds']/60:.1f} 分钟，逐条核对三张表共 {sum(p['requested'] for p in tm['formal_panels'].values())} 份凭据，实际额外重试 {sum(p['repeat_attempts'] for p in tm['formal_panels'].values())} 次。锁模到最后一项科学核验经过 {tm['lock_to_last_scientific_artifact_s']/60:.1f} 分钟，其中包括可与评分同时执行的复核。报告构建与页面验证另有时戳。",
            ['事件','北京时间'],[[label,iso[:10]+' '+iso[11:19]]for label,iso in stamps],
            ['review/delivery-timing.json','review/campaign-counts.json','selection-lock.json'])
    filters=REVIEW/'vqf_response_controls_final.json'
    if filters.is_file():
        controls=read(filters)
        assert controls['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        versus=[0,0,0]
        for d in ORDER:
            r=controls['domains'][d]
            m=r.get('means',{})
            cells=[angle(m[k]) for k in ('fast_deg','default_deg','slow_deg','candidate_deg')] if m else ['—']*4
            beaten=sum(m['candidate_deg']<m[k] for k in ('fast_deg','default_deg','slow_deg')) if m else 0
            if m:
                for i,k in enumerate(('fast_deg','default_deg','slow_deg')):versus[i]+=int(m['candidate_deg']<m[k])
            values.append([d,f"{r['scored']}/{r['requested']}",*cells,f'{beaten}/3' if m else '—'])
        control_sources=['review/vqf_response_controls_final.json','review/vqf_response_controls.py',verified['main_table']]
        if all(p.is_file() for p in extra_files):
            extra,dc,pc=map(read,extra_files)
            for d,n,total,m in [('EuRoC',extra['domains']['EuRoC']['statuses'].get('scored',0),3,extra['domains']['EuRoC']['means']),
                                ('RIDI',extra['domains']['RIDI']['statuses'].get('scored',0),25,extra['domains']['RIDI']['means']),
                                ('DAVIS',dc['records'],3,dc['means']),('PennCOSYVIO',pc['scored'],2,pc['means'])]:
                fields=('fixed_vqf_1p5_deg','full_vqf_deg','fixed_vqf_6_deg')
                cells=[angle(m[k]) for k in (*fields,'candidate_deg')] if m else ['—']*4
                beaten=sum(m['candidate_deg']<m[k] for k in fields) if m else 0
                if m:
                    for i,k in enumerate(fields):versus[i]+=int(m['candidate_deg']<m[k])
                values.append([d,f'{n}/{total}',*cells,f'{beaten}/3' if m else '—'])
            control_sources.extend(p.relative_to(ROOT).as_posix() for p in extra_files)
        wins_all=[row[0] for row in values if row[-1]=='3/3']
        exposed_domains={d for d,_ in read(ROOT/'final/exposure-audit.json')['development_main_reuse']['records']}
        separated={row[0] for row in values}-exposed_domains
        distinct_wins=sorted(set(wins_all)&separated)
        add('all8-final-vqf-controls','冻结模型与三套固定 VQF 参数对照',
            f'冻结模型与固定 1.5 秒、3 秒、6 秒 VQF 分别比较，在 {versus[0]}、{versus[1]}、{versus[2]} 套数据上占优。对每套数据要求同时优于三种设置时，有 {len(wins_all)}/{len(values)} 套通过（{"、".join(wins_all) or "无"}）。在本轮开发与评估记录分开的 {len(separated)} 套中，{len(distinct_wins)} 套达到这个标准。三个 VQF 使用相同原始输入、采样率、评分掩码与其余库参数，仅改变加速度时间常数。主要胜场按默认 3 秒计算。',
            ['数据集','评分覆盖','VQF 1.5 秒','默认 VQF 3 秒','VQF 6 秒','冻结模型','优于几套设置'],values,
            control_sources)
        markdown.append('[三套固定 VQF 的全部配对成绩](vqf_response_controls_final.json)')
    dev_controls=REVIEW/'development-vqf-thresholds.json'
    worker_review=REVIEW/'development-fixed-vqf-threshold-audit.json'
    if dev_controls.is_file() and worker_review.is_file():
        comparison=read(dev_controls);replica=read(worker_review)
        assert comparison['selection_sha256']==verified['lock_sha256'] and comparison['histogram']==replica['passing_domain_count_histogram']
        values=[]
        for d in ORDER:
            r=comparison['domains'][d];w=replica['domain_thresholds'][d]
            assert abs(r['threshold_deg']-w['threshold_rmse_deg'])<1e-9
            assert abs(r['minimum_any_tested_deg']-w['best_candidate_rmse_deg'])<1e-9
            assert r['number_of_policies_beating_all_three']==w['passing_configs']
            values.append([d,str(r['records']),r['strongest_tau_s']+' 秒',angle(r['threshold_deg']),angle(r['selected_deg']),
                           angle(r['minimum_any_tested_deg']),str(r['number_of_policies_beating_all_three'])])
        missed='、'.join(d for d in ORDER if comparison['domains'][d]['number_of_policies_beating_all_three']==0)
        add('all8-dev-thresholds','开发阶段的 42 种配置能否胜过三种 VQF',
            f"两方独立检查开发端已有的 213 条记录。{comparison['compared']} 种配置有完整共同评分，另有 {comparison['partial_excluded']} 种覆盖不全。以每域三种 VQF 的最低记录均值为门槛，11 种配置通过 1 域、12 种通过 2 域、19 种通过 3 域。{missed} 五域没有任何配置超过门槛。本表只使用开发记录，展示进一步提升精度的实际差距。",
            ['开发域','记录','最低 VQF 对应设置','最低 VQF','C038','已试配置中最低值','超过门槛的配置数'],values,
            ['review/development-vqf-thresholds.json','review/development-fixed-vqf-threshold-audit.json',
             'review/development_frontier.json','review/vqf_response_controls_development.json'])
    measured=[]
    formal=verified['main']['SHICE']
    own=verified['self_devices']
    for label,r in [('自采全库',verified['full']['SHICE']),('自采 R5+R6 留出',formal),*[(x,own[x]) for x in own]]:
        if r['scored']:
            measured.append([label,f"{r['scored']}/{r['requested']}",angle(r['means']['full_vqf_deg']),
                             angle(r['means']['candidate_deg']),f"{r['improvement_percent']:+.3f}%",str(r['groups']['improving_records'])])
    add('all8-final-self','自采设备的实际收益','R1–R4 的 37 条用于开发判断。R5 和 R6 各 10 条在本轮锁模后评分。全库汇总给每条记录相同权重。',
        ['设备或范围','已评分/预定','默认 FullVQF','冻结模型','相对改善','改善记录数'],measured,
        [verified['main_table'],verified['additional_table'],'review/final-verification.json'])
    history=REVIEW/'historical_final_join.json'
    if history.is_file() and 'previous_transfer' in read(history):
        transfer=read(history)['previous_transfer']
        values=[]
        for key,stats in transfer['groups'].items():
            values.append([key,str(stats['records']),angle(stats['full_vqf']),angle(stats['old_transfer_mamba']),
                           angle(stats['current']),f"{stats['current']-stats['old_transfer_mamba']:+.5f}°",
                           str(stats['improves_old_records'])])
        add('all8-self-transfer','两次外训所得 Mamba 在同一批自采留出上的变化',
            '前一轮的外训 Mamba 和本轮模型都曾在这 20 条 R5、R6 上评分。本表逐条核对原生帧数、索引哈希及默认 VQF，用相同记录检查本次开发带来的改变。两次实验共用的自采留出已有历史曝光。',
            ['设备','记录','默认 FullVQF','前次外训 Mamba','本轮冻结模型','本轮减前次','优于前次的记录'],values,
            ['review/historical_final_join.json','../transfer-mamba-20261123/batch8/all-sequences.csv',verified['main_table']])
    clock_path=REVIEW/'shice-clock-verified.json'
    if clock_path.is_file():
        clock=read(clock_path);assert clock['selection_sha256']==verified['lock_sha256']
        both=clock['groups']['R5+R6'];gains=[x['gain_default_pct'] for x in both['offsets'].values()]
        values=[]
        for key in ('R5+R6','R5','R6'):
            grp=clock['groups'][key]
            for shift,value in grp['offsets'].items():
                m=value['means'];record_n=grp['records']
                values.append([key,f"{int(shift):+d}",str(record_n),angle(m['vqf3_deg']),angle(m['vqf1p5_deg']),
                               angle(m['c038_deg']),f"{value['gain_default_pct']:+.3f}%",
                               f"{value['wins']['vqf3_deg']}/{record_n}",f"{value['wins']['vqf1p5_deg']}/{record_n}"])
        add('all8-self-encoder-clock','自采收益对编码器时间偏差的敏感性',
            f"保持三种方法的预测方向固定，把每个时刻 i 的预测同编码器第 i+k 行的重力方向比较。"
            f"原生间隔约 1 毫秒，五种偏移在计算前一次登记。20 条自采原件各取前后连续 11 行都有效的中心，"
            f"共同评分 {both['common_frames']:,} 行，占正式 {both['formal_frames']:,} 行的 {both['common_frames']/both['formal_frames']:.2%}。"
            f"在这五种条件下，C038 相对默认 VQF 的合并平均改善为 {min(gains):.2f}% 至 {max(gains):.2f}%。"
            f"固定 1.5 秒的 VQF 在每种偏移的合并均值上均更低。两条原件的三个方法、五种偏移经另一条实现复算，最大差 "
            f"{clock['raw_anchor_max_gap_deg']:.3g}°。正式报告仍采用原始编码器对齐和完整有效帧。此次检验衡量设定偏移的影响，实际时钟偏差未知。",
            ['设备','参考偏移（原生行）','记录','默认 VQF','VQF 1.5 秒','C038','较默认改善','胜默认条数','胜快 VQF 条数'],
            values,['final/shice-clock-sensitivity/protocol.json','final/shice-clock-sensitivity/nominal-full.csv',
                    'final/shice-clock-sensitivity/sequences.csv','review/shice-clock-verified.json','review/shice-clock-anchors.json'])
        markdown.append('[20 条记录的原名义帧与正式分数](../final/shice-clock-sensitivity/nominal-full.csv) · [五种偏移的全部 100 条成绩](../final/shice-clock-sensitivity/sequences.csv)')
    rotated=REVIEW/'final_self_mount_predictions.json'
    if rotated.is_file():
        g=read(rotated)
        assert g['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        for case,label in [('original','原始安装坐标'),('x90','绕 x 轴 90°'),('cycle_xyz','xyz 循环置换'),('oblique_8deg','绕斜轴 8°')]:
            cells=[]
            for device in ('R5','R6'):
                r=g['by_device'][device][case]
                gap=r['candidate_minus_default_deg']
                cells.extend([f'{gap:+.5f}°' if gap is not None else '—',f"{r['improved_records']}/{r['scored']}"])
            values.append([label,*cells])
        add('all8-final-mount','自采 R5 和 R6 的安装旋转复核',
            '对同一批转台编码器参考和六轴观测施加三次预定坐标旋转，再从头运行两种方法。表中为冻结模型减默认 VQF 的逐记录平均误差，以及同一设备中改善的记录数。这些重放覆盖已有动作，未新增采集。',
            ['坐标条件','R5 平均角度差','R5 改善记录','R6 平均角度差','R6 改善记录'],values,
            ['review/final_self_mount_predictions.json','review/mount_prediction_audit.py',verified['main_table']])
        markdown.append('[自采 20 条记录的安装旋转与逐帧一致性](final_self_mount_predictions.json)')
        if 'motion_by_device' in g:
            values=[]
            for label,key in [('低于 0.05','below_0p05'),('0.05 至低于 0.5','0p05_to_0p5'),
                              ('0.5 至低于 2','0p5_to_2'),('2 及以上','at_least_2')]:
                for device in ('R5','R6'):
                    part=g['motion_by_device'][device][key];m=part['record_means_deg']
                    values.append([device,label,str(part['records']),f"{part['scored_frames']:,}",
                                   angle(m['full_vqf_deg']) if m else '—',angle(m['candidate_deg']) if m else '—',
                                   str(part['improved_records']) if m else '—'])
            add('all8-final-self-speed','自采两台留出设备上，不同转速分别改善多少',
                '在每条记录的原始安装坐标上，按实测陀螺大小分档。每条记录在该档累计达到一秒有效观测才计算 RMSE，各记录等权。表中显示每个分档的实际覆盖与逐记录胜场。',
                ['设备','角速度（rad/s）','有支持的记录','评分原生行','默认 VQF','冻结模型','改善记录数'],values,
                ['review/final_self_mount_predictions.json','review/mount_prediction_audit.py','review/FINAL_PROTOCOL.md'])
    evidence=[]
    def interval(x):
        return ' 至 '.join(f'{a:+.4f}°' for a in x) if x else '—'
    for d in ORDER:
        r=verified['main'][d]
        if not r['scored']:
            continue
        groups=r['groups']
        ci=groups.get('paired_cluster_percentile_95_deg')
        loo=groups.get('leave_one_group_out_deg')
        evidence.append([d,f"{groups['improving_groups']}/{groups['collection_groups']}",interval(loo),interval(ci),
                         r['largest_regression_record'],f"{r['largest_candidate_minus_default_deg']:+.4f}°"])
    group_sources=['review/final-verification.json','review/group_uncertainty.py',verified['main_table']]
    from group_uncertainty import paired_by_group
    independent_sets=[('final-supplemental',{'RIDI':None,'EuRoC':None}),('final-davis',{'DAVIS':'poster'}),('final-penn',{'PennCOSYVIO':'同一仪器与路线'})]
    for directory,sets in independent_sets:
        file=REVIEW/directory/'sequences.csv'
        if not file.is_file():continue
        with file.open(encoding='utf-8-sig',newline='')as f: original=list(csv.DictReader(f))
        group_sources.append(file.relative_to(ROOT).as_posix())
        for d,group in sets.items():
            rows=[dict(r,group=group or r.get('group'))for r in original if r['dataset']==d and r.get('candidate_deg')]
            if not rows:continue
            g=paired_by_group(rows,'full_vqf_deg','candidate_deg')
            gap=lambda r:float(r['candidate_deg'])-float(r['full_vqf_deg'])
            worst=max(rows,key=gap)
            evidence.append([d,f"{g['improving_groups']}/{g['collection_groups']}",interval(g.get('leave_one_group_out_deg')),
                             interval(g.get('paired_cluster_percentile_95_deg')),worst['record'],f"{gap(worst):+.4f}°"])
    add('all8-final-groups','按独立采集组检查胜负',
        '角度差取冻结模型减默认 VQF。逐次留一组反映单次采集影响。达到五组时才以整组为单位重抽样 8,000 次。阵列的 16 路车载通道来自同一试验。补充检验中，RIDI 按受试者分组，EuRoC 同一房间、DAVIS 同一场景和宾大同一仪器路线各计一组。',
        ['数据集','改善组/总组','留一组后角度差','组重抽样 95% 区间','差值最大的记录','该记录角度差'],evidence,
        group_sources)
    supplemental=REVIEW/'final-supplemental/summary.json'
    if supplemental.exists():
        s=read(supplemental)
        assert s['hashes']['selection_lock']==verified['lock_sha256']
        values=[]
        for d in ('EuRoC','RIDI'):
            r=s['domains'][d]
            scored=r['statuses'].get('scored',0)
            m=r.get('means')
            values.append([d,f"{scored}/{r['requested']}",angle(m['fixed_vqf_1p5_deg']) if m else '—',
                          angle(m['full_vqf_deg']) if m else '—',angle(m['fixed_vqf_6_deg']) if m else '—',
                          angle(m['candidate_deg']) if m else '—',f"{100*(1-m['candidate_deg']/m['full_vqf_deg']):+.3f}%" if m else '—',
                          str(r.get('improving_records','—'))])
        add('all8-supplemental','两套外训来源上的额外迁移检验',
            'EuRoC 的三条取自作者单列的 Vicon Room 2 采集组，使用采集的原始陀螺和作者发布的融合参考状态。RIDI 按官方 test 全 25 条登记，参考为同一手机的 Tango 视觉惯性姿态。作者压缩包中缺 hang_leg_new3、huayi_handheld_test1。两套均采用同一冻结模型。',
            ['数据来源','已评分/预定','VQF 1.5 秒','默认 FullVQF','VQF 6 秒','冻结模型','较默认改善','改善记录数'],values,
            ['review/final-supplemental/sequences.csv','review/final-supplemental/summary.json','external-sources/sealed-transfer-tests/protocol.json'])
        markdown.append('[两个补充数据集逐条成绩](final-supplemental/sequences.csv) · [原件核对和缺档](../external-sources/sealed-transfer-tests/source-files.json)')
    davis=REVIEW/'final-davis/summary.json'
    if davis.is_file():
        s=read(davis)
        assert s['selection_lock_sha256']==verified['lock_sha256'] and s['native_records_predeclared']==3
        values=[]
        keys=('fixed_vqf_1p5_deg','full_vqf_deg','fixed_vqf_6_deg','physical_deg','candidate_deg')
        for row in s['rows']:
            vals=[angle(row[k]) for k in keys] if row['status']=='scored' else ['—']*len(keys)
            values.append([row['record'],str(row.get('scored_frames','—')),*vals])
        if s['means']:
            values.append(['可评分记录等权均值',f"{s['total_scoring_frames']:,}",*[angle(s['means'][k]) for k in keys]])
        add('all8-davis','DAVIS 手持相机在留出场景的实测',
            f"poster 场景预定的三次采集中，{s['records']} 次获得分数，{s['records_improved']} 次优于默认 FullVQF。该场景与六条外训采集共用同一仪器，三种运动均使用同一冻结配方。评分取光学有效的原生 IMU 行，三套普通 VQF 均在相同行重新计算。",
            ['留出采集','评分帧','VQF 1.5 秒','默认 VQF 3 秒','VQF 6 秒','纯物理','冻结模型'],values,
            ['review/final-davis/sequences.csv','review/final-davis/summary.json','external-sources/davis-handheld/preprocessing-lock.json'])
        markdown.append('[poster 逐段实测与每段参考核验](final-davis/summary.json) · [DAVIS 训练和封存划分](../external-sources/davis-handheld/protocol.json)')
    penn=REVIEW/'final-penn/summary.json'
    if penn.is_file():
        s=read(penn)
        assert s['selection_lock_sha256']==verified['lock_sha256'] and s['requested']==2
        keys=('fixed_vqf_1p5_deg','full_vqf_deg','fixed_vqf_6_deg','physical_deg','candidate_deg')
        values=[]
        for r in s['rows']:
            vals=[angle(r[k]) for k in keys] if r['status']=='scored' else ['—']*len(keys)
            values.append([r['record'].upper(),str(r.get('scored_frames','—')),*vals])
        if s['means']:
            frames=sum(r['scored_frames'] for r in s['rows'] if r['status']=='scored')
            values.append(['两条等权均值（帧数合计）',f"{frames:,}",*[angle(s['means'][k]) for k in keys]])
        add('all8-final-penn','PennCOSYVIO 的两次手持留出采集',
            f"作者划分为测试的 B-F 和 B-S 使用同一冻结模型，{s['scored']}/2 条完成，{s['records_improved']} 条优于默认 VQF。两条同走一条近似路线，速度不同，仪器与外训 A 组相同。参考采用作者的光学标记姿态图。",
            ['留出记录','有效原生行','VQF 1.5 秒','默认 VQF','VQF 6 秒','纯物理','冻结模型'],values,
            ['review/final-penn/sequences.csv','review/final-penn/summary.json','external-sources/penn-handheld/preprocessing-lock.json'])
        markdown.append('[B 组逐条成绩与对应的光学和原始陀螺交叉核验](final-penn/summary.json) · [作者发布的坐标变换](https://daniilidis-group.github.io/penncosyvio/extrinsic_calib/)')
    history=REVIEW/'historical_final_join.json'
    if history.exists():
        h=read(history)
        assert h['selection_lock_sha256']==verified['lock_sha256']
        values=[]
        for d in ('R6','BROAD','Sassari'):
            r=h['domains'][d];m=r['macro_rmse_deg']
            values.append([d,str(r['records']),*[angle(m[k]) for k in ('full_vqf','physical_only','timesfm','gru','candidate')]])
        add('all8-final-baselines','与原有四种方法按相同记录对账',
            '每条记录独立重建原生评分行。R6 比较 10 条，BROAD 全库 39 条、Sassari 全库 18 路包含曾参与外训的记录。此表的物理特征头来自原实验，与本轮主表中的纯物理消融分别计算。',
            ['范围','记录','默认 FullVQF','原物理特征头','TimesFM','GRU','冻结 Mamba'],values,
            ['review/historical_final_join.json','../overnight-20261122/review/baseline_audit.json'])
    archived_models=REVIEW/'historical-models-verified.json'
    if archived_models.is_file():
        checked=read(archived_models)
        assert checked['selection_sha256']==verified['lock_sha256']
        domains=checked['domains']
        values=[]
        for d in (*ORDER,'RIDI','EuRoC','DAVIS','PennCOSYVIO'):
            row=domains[d]
            m=row['means_deg']
            values.append([d,f"{row['scored']}/{row['requested']}",
                           *[angle(m[k]) for k in ('full_vqf_deg','transfer_mamba_deg','strict_tcn_deg','candidate_deg')],
                           f"{row['records_c038_beats']['transfer_mamba_deg']}/{row['scored']}",
                           f"{row['records_c038_beats']['strict_tcn_deg']}/{row['scored']}"])
        wins=checked['wins_by_domain']
        brief=(f"上一轮冻结的迁移 Mamba 和 TCN 均保留原始权重、归一化和 1.0 倍修正。这次在相同的 "
               f"{checked['scored_native_rows']} 条可评分记录上重新运行两种旧模型。C038 的数据集平均误差在 "
               f"{wins['transfer_mamba_deg']}/12 套低于旧 Mamba，在 {wins['strict_tcn_deg']}/12 套低于旧 TCN。"
               f"两条 RIDI 缺档计入登记条数。18 条预先选定的原件又经上一轮的评分代码在 CPU 上复算，"
               f"最大差 {checked['native_anchor_max_gap_deg']:.3g}°。表内均为记录等权的倾角 RMSE，最后两列为 C038 逐记录胜数。")
        flagged=20-checked['original_archive_rows_passing']
        if flagged:
            brief+=(f"与上轮 R5/R6 归档逐条比对时，20 条的原生评分帧完全一致。首轮 CPU 分数有 {flagged} 条超出原设的旧模型 "
                    f"0.00001° 核对线。TCN 最大差 {checked['original_archive_max_gap_deg']['strict_tcn']:.6f}°，"
                    f"旧 Mamba 最大差 {checked['original_archive_max_gap_deg']['transfer_mamba']:.3g}°。这些原始 CPU 数值均保留在表内。")
            backend=checked['tcn_backend']
            brief+=(f"用上轮的 RTX 3090 和原代码重算，这 {backend['original_backend_matching']} 条与原档最大差 "
                    f"{backend['max_archive_gap_deg']:.3g}°。在差值最大的 {backend['probe']}，仅关闭 cuDNN 的 TF32 就使 CPU 与 CUDA 的差值从 "
                    f"{backend['probe_cpu_vs_cuda_deg']:.6f}° 降至 {backend['probe_cpu_vs_cuda_no_tf32_deg']:.3g}°。")
        add('all8-frozen-external-models','在十二套数据上重放上一轮的 Mamba 和 TCN',brief,
            ['数据集','可评分/登记','默认 FullVQF','旧迁移 Mamba','旧 TCN','本轮 C038','胜旧 Mamba 条数','胜旧 TCN 条数'],values,
            ['review/historical-models-verified.json','final/historical-models/main.csv','final/historical-models/transfer.csv',
             'review/previous-model-native-anchors.json','final/historical-models/tcn-backend-attribution/summary.json','../transfer-mamba-20261123/review/final-transfer-lock.json'])
        markdown.append('[两种旧模型主评逐条成绩](../final/historical-models/main.csv) · [四个补充数据集逐条成绩](../final/historical-models/transfer.csv) · [与原归档及 18 条原生锚点的核对](historical-models-verified.json)')
    optical=REVIEW/'oxford_final_sensitivity.json'
    if optical.is_file():
        review=read(optical)
        assert review['selection_lock_sha256']==verified['lock_sha256']
        cases=[('nominal','标称'),('earlier_10ms','时间提前 10 ms'),('later_10ms','时间推迟 10 ms'),
               ('body_x_-1deg','绕 x 轴 −1°'),('body_x_+1deg','绕 x 轴 +1°'),('body_y_-1deg','绕 y 轴 −1°'),('body_y_+1deg','绕 y 轴 +1°')]
        subsets=[review['summary'][key] for key in ('all','independently_checked')]
        values=[]
        for case,label in cases:
            results=[]
            for scope in subsets:
                item=scope['candidate_minus_default_deg'].get(case)
                results.append(f"{item['candidate_deg']:+.5f}°" if item else '—')
            values.append([label,*results])
        common=sum(row['common_frames']for row in review['sequences'])
        add('all8-final-oxford','Oxford 留出结果对参考时间与安装方向的敏感性',
            f"登记 68 条，全体有分数的 {subsets[0]['records']} 条和独立配准检验通过的 {subsets[1]['records']} 条分别汇总。各情形在共同的 {common:,} 个原生时刻比较冻结模型减默认 VQF 的误差。负值表示冻结模型更准。 ±1° 和 ±10 ms 是预先选取的扰动量，统计置信区间见采集组表。",
            ['相同参考对两种方法的变化','全部可评分记录','独立参考核验通过'],values,
            ['review/oxford_final_sensitivity.json',verified['main_table']])
        markdown.append('[Oxford 每条的共同掩码与七种参考结果](oxford_final_sensitivity.json)')
    geometry=REVIEW/'final-geometry.json'
    if geometry.is_file():
        g=read(geometry)
        assert g['lock_sha256']==verified['lock_sha256']
        rows=[]
        for d,item in g['datasets'].items():
            for scope,label in [('all','全体评分记录'),('alternative_changes_reference','备用参考实际改变的记录')]:
                if scope not in item:continue
                pair=item[scope]
                angles=[pair[k]['candidate_deg']-pair[k]['full_vqf_deg']for k in ('primary','alternative')]
                rows.append([d,label,str(item['scored'] if scope=='all' else item['changed']),
                    *[f'{v:+.5f}°' for v in angles],str(item['records_with_flags'])])
        add('all8-final-geometry','车载阵列和摆段对安装参考的敏感性',
            '保持同一冻结预测和评分帧。车载阵列将前缀拟合安装方向换成标称安装方向。三节摆在前缀激励不足时，另用开发阶段登记的夹具竖直先验。角度差为冻结模型减默认 VQF，数据集主表沿用其预先指定的参考。',
            ['数据集','记录范围','记录数','原参考角度差','备用参考角度差','全数据集有核验标记的记录'],rows,
            ['review/final-geometry.json','review/final-anchor-spec.json',verified['main_table']])
        markdown.append('[逐记录安装角、核验标记及备用参考成绩](final-geometry.json)')
    tum=REVIEW/'tum_final_reference.json'
    if tum.is_file():
        t=read(tum)
        assert t['lock_sha256']==verified['lock_sha256']
        values=[]
        for result in t['rows']:
            p=result.get('raw_paired_score')
            values.append([result['room'],str(result['dynamic_optical_pairs']),f"{result['optical_vs_gyro_direction_median_deg']:.2f}°",
                f"{result['optical_to_gyro_displacement_median_ratio']:.3f}",f"{result['quiet_acc_vs_optical_vertical_median_deg']:.2f}°" if result['quiet_rows'] else '—',
                angle(p['full_vqf_deg']) if p else '—',angle(p['candidate_deg']) if p else '—'])
        add('all8-final-tum','TUM-VI 四条留出轨迹的独立重算',
            '同一动捕室内逐条从 IMU 和光学 CSV 重建时钟与有效帧。另取光学角速度和实测陀螺均超过 0.4 rad/s 的时段，以实测陀螺积分检查同期光学转动的方向和大小。准静止段比较的是加速度计方向与光学竖直。该物理核验沿用原标定与时间对齐。',
            ['轨迹','配对动态时段','光学与陀螺轴夹角中位','光学/陀螺角位移中位','静止比力与光学竖直中位','默认 VQF','冻结模型'],values,
            ['review/tum_final_reference.json','review/tum_reference_quality.py',verified['main_table']])
        markdown.append('[四条轨迹的原生帧、光学缺口与独立重算](tum_final_reference.json) · [校准数据说明，TUM-VI 原论文](https://cvg.cit.tum.de/_media/spezial/bib/schubert2018vidataset.pdf)')
    timing_paths=sorted(REVIEW.glob('final-latency-*.json'))
    if timing_paths:
        values=[];loads=[]
        hardware=read(REVIEW/'latency-environment.json')
        runtime=hardware['runtime']
        for path in timing_paths:
            timer=read(path)
            assert timer['selection_lock_sha256']==verified['lock_sha256']
            assert timer['torch_version']==runtime['torch'] and timer['vqf_version']==runtime['vqf']
            loads.append(f"{timer['device'].upper()} {timer['checkpoint_load_ms']:.1f} ms")
            for i,row in enumerate(timer['trials'],1):
                values.append([timer['device'].upper(),str(i),f"{row['baseline_packet']['p95_us']:.1f} µs",
                               f"{row['all_candidate_packets']['p95_us']:.1f} µs",f"{row['first_release_us']/1000:.2f} ms",f"{row['warm_release_packet']['p95_us']/1000:.2f} ms",
                               f"{row['warm_release_packet']['p99_us']/1000:.2f} ms",f"{row['batch_one_vs_offline_max_angle_deg']:.2g}°"])
        add('all8-final-latency','冻结模型在 FZR-MiniPC 的逐包延迟',
            f"FZR-MiniPC 使用 {hardware['processor']}（{hardware['physical_cores']} 核 {hardware['logical_processors']} 线程）和 {runtime['gpu']}。PyTorch {runtime['torch']}，VQF {runtime['vqf']}，单个 PyTorch CPU 计算线程，Mamba 使用顺序内核。每个后端装载权重的实测时间为 {'、'.join(loads)}。200 Hz 手机原始包已读入内存，每轮 6,000 个包，5 ms 到一个新包，每 100 ms 发布一次修正。保留首次发布耗时，另取前 16 次发布之后的稳态结果。计时包含因果校准、VQF、神经网络和方向重建。",
            ['后端','次数','默认单包 P95','候选单包 P95','首次发布','候选稳态发布 P95','候选稳态发布 P99','逐包与离线最大角差'],values,
            ['review/'+p.name for p in timing_paths]+['review/latency-environment.json','review/stream_latency.py'])
    native_timing=sorted(REVIEW.glob('shice-native-latency-*.json'))
    if native_timing:
        values=[]
        for path in native_timing:
            bench=read(path)
            assert bench['selection_sha256']==verified['lock_sha256'] and bench['repeat_count']==3
            for name in ('R5-1','R6-1'):
                trials=[r for r in bench['rows'] if r['record']==name]
                assert len(trials)==3
                def span(numbers,precision):
                    return f"{min(numbers):.{precision}f}–{max(numbers):.{precision}f}"
                assert len({r['packets']for r in trials})==1 and len({r['observed_missing_input_rows']for r in trials})==1
                release=[r['warm_release']['p95_us']/1000 for r in trials]
                outside=[100.*r['packets_over_native_interval']/r['packets'] for r in trials]
                values.append([bench['device'].upper(),name,f"{trials[0]['hz']:.3f} Hz",str(trials[0]['packets']),
                               str(trials[0]['observed_missing_input_rows']),
                               span([r['default_packet']['p95_us'] for r in trials],1)+' µs',
                               span([r['candidate_packet']['p95_us'] for r in trials],1)+' µs',span(release,2)+' ms',
                               span(outside,2)+'%',f"{max(r['max_online_offline_gap_deg']['candidate_deg']for r in trials):.3g}°"])
        add('all8-self-native-latency','按自采设备约 1 kHz 的实际时钟逐包计算',
            '另取预登记的两条自采锚点，各从采集开始连续回放 60 秒。六轴信号、实际采样率和原件丢包标记均保留。每个后端独立初始化 3 次，表内为三轮测量范围。单包计时包含因果缺样处理、滤波、网络及方向还原。神经修正每 100 ms 发布一次。超过原生包间隔的比例统计实测计算时间，实际采集丢包数另列。所有输出方向逐时刻与冻结离线实现核对。',
            ['后端','自采原件','实测采样率','每轮包数','原件缺观测行','默认包 P95','候选包 P95','含发布的包 P95','处理超过一个包间隔','在线与离线最大角差'],values,
            ['review/'+p.name for p in native_timing]+['review/shice_stream_latency.py','review/final-anchor-spec.json','review/latency-environment.json'])
