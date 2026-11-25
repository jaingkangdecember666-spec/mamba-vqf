"""Build the in-progress eight-domain chapter from independently audited scoring tables."""
import csv
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
REVIEW=ROOT/'review'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def table(header,values):
    return '\n'.join(['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|',
                       *['| '+' | '.join(row)+' |' for row in values]])


def main():
    proof=read(REVIEW/'C005_C006_audit.json')
    counts=read(REVIEW/'campaign-counts.json') if (REVIEW/'campaign-counts.json').exists() else None
    final=(REVIEW/'final-verification.json').is_file() and (ROOT/'selection-lock.json').is_file()
    cutoff=counts['evidence_cutoff'] if counts else read(ROOT/'C006/results.json')['completed_at']
    timing=REVIEW/'delivery-timing.json'
    if final:
        cutoff=read(timing)['evidence_checked_through_at'] if timing.is_file() else datetime.now().astimezone().isoformat(timespec='seconds')
    started=read(ROOT/'protocol.json')['start']
    elapsed=int((datetime.fromisoformat(cutoff)-datetime.fromisoformat(started)).total_seconds())
    stage='锁模终测与审查' if final else '开发阶段审查稿'
    documents={};sections=[];markdown=['# 八套 IMU 数据集上的六轴倾角实验',
        f'{stage} · 北京时间 {started[:10]} {started[11:19]} 至 {cutoff[:10]} {cutoff[11:19]} 的证据，覆盖 {elapsed//60} 分 {elapsed%60} 秒。授权窗口至 11 月 24 日 06:56:31。']
    def add(key,title,body,headers,values,files):
        fields=[f'c{i}' for i in range(len(headers))]
        documents[key]={'rows':[dict(zip(fields,row)) for row in values],
                        'source':{'label':title,'files':['../../all-datasets-8h-20261123/'+p for p in files],
                                  'executedAt':cutoff}}
        sections.append({'id':key,'title':title,'body':body,'queryId':key,'columns':[{'field':f,'label':name} for f,name in zip(fields,headers)]})
        markdown.append('## '+title+'\n\n'+body)
        if headers:markdown.append(table(headers,values))
        return documents[key]['rows']
    if final:
        from final_report import append
        append(add,markdown)
        cross=next((s for s in sections if s['id']=='all8-final-cross-source'),None)
        if cross:
            lock=read(ROOT/'selection-lock.json'); own=read(REVIEW/'final-verification.json')['main']['SHICE']
            markdown[0]='# 八小时六轴倾角实验（自采与十一套外部数据）'
            summary=(f"冻结模型 **{lock['selected']['batch']}**。共完成 **{counts['completed_named_batches']} 个小批、"
                     f"{counts['new_or_inherited_candidate_policies_tested']} 种不同配置**。"+cross['body'])
            if own['scored']:
                summary+=f" 自采 R5、R6 共 {own['scored']} 条，逐记录平均误差由 {own['means']['full_vqf_deg']:.5f}° 降至 {own['means']['candidate_deg']:.5f}°。" if own['improvement_percent']>0 else f" 自采 R5、R6 共 {own['scored']} 条，默认与冻结模型分别为 {own['means']['full_vqf_deg']:.5f}°、{own['means']['candidate_deg']:.5f}°。"
            if 'all8-final-vqf-controls' in documents:
                controls=documents['all8-final-vqf-controls']['rows']
                strict=[r['c0'] for r in controls if r['c6']=='3/3']
                peek=next(r for r in controls if r['c0']=='SHICE')
                summary+=f" 要同时赢过 1.5、3、6 秒三个 VQF 设置，只有 **{len(strict)}/{len(controls)} 套**通过（{'、'.join(strict) or '无'}）。自采上的 1.5 秒 VQF 为 {peek['c2']}。"
            markdown.insert(2,summary+' [训练、开发、终测怎样划分](SPLITS.md) · [逐批指导依据](decisions.jsonl) · [模型与评分名单](../selection-lock.json)')
    order=('SHICE','BROAD','Sassari','TUM-VI','IMU-Array','OxIOD','RepoIMU','YT_Server')
    if not final and counts:
        frontier=read(REVIEW/'development_frontier.json')
        best=next(r for r in frontier['full_coverage'] if r['weights'] or r['components'])
        values=[]
        for d in order:
            r=best['domains'][d]
            values.append([d,str(r['records']),f"{r['baseline_deg']:.5f}°",f"{r['candidate_deg']:.5f}°",
                           f"{(1.-r['ratio'])*100:+.3f}%",f"{r['improving_records']}/{r['records']}"])
        add('all8-current','最新开发候选和八套数据的实际差值',
            f"截至 {counts['through_batch']} 已完成 {counts['completed_named_batches']} 批。按 213 条完整开发记录排序，{best['batch']}（{best['column']}）在 {best['winning_domains']}/8 个域的逐记录平均倾角误差低于默认 FullVQF。表内每个数据集分别判断，逐记录胜数同时列出。正式 365 条和另外 76 条全库记录等待统一锁模后评分。", 
            ['开发域','记录','默认 FullVQF','当前候选','相对改善','改善记录数'],values,
            ['review/development_frontier.json','review/campaign-counts.json',best['batch']+'/sequences.csv'])
        markdown.append('[逐批纠偏依据](decisions.jsonl) · [训练、开发与留出怎样划分](SPLITS.md)')
    descriptions=['自采 R1–R4','历史留出的 5 个运动组','一次中速运动，6 路','同一动捕室的两条开发轨迹','同一静态标定的 16 路','66 条手机 Vicon 记录','62 条杆件和摆段记录','12 个前缀，参考来源未明']
    domain=proof['domains']
    values=[]
    for d,desc in zip(order,descriptions):
        stats=domain[d];m=stats['means']
        values.append([d,str(stats['records']),desc,f"{m['full_vqf_deg']:.5f}°",f"{m['c003_deg']:.5f}°",f"{m['B_rest_gated_deg']:.5f}°",f"{m['B_rest_gated_deg']-m['full_vqf_deg']:+.5f}°"])
    add('all8-development','前六批的八套开发结果',
        '截至 C006，共完成 6 个小批，前 4 批覆盖最多 85 条记录。C005 和 C006 均对 8 套数据的 213 条开发记录评分，失败 0 条。C003 与 C006 的八个均值都低于默认六轴 FullVQF。C006 在两条 TUM 轨迹均改善，在 Oxford 全体开发记录上的优势仅 0.00055°。',
        ['数据集','条数','评估范围','默认 FullVQF','C003','C006','C006 减默认'],values,
        ['C006/sequences.csv','C005/results.json','review/C005_C006_audit.json'])
    markdown.append('[213 条逐记录结果](../C006/sequences.csv) · [独立原生重放](C005_C006_audit.json) · [按采集组重抽样](C005_group_uncertainty.json)')
    splits=[['SHICE','0','R1–R4，37 条','R5/R6，20 条','2 条时钟排除，R1-2 缺档；R5/R6 旧实验曾评过'],
            ['BROAD','27 条、15 组','12 条、5 组','沿用 12 条验证基准','程序变体同组；已用于历史选模'],
            ['Sassari','慢/快，各 6 路','中速 6 路','沿用中速 6 路基准','同一运动六路同步；已用于历史选模'],
            ['TUM-VI','0','room1/2','room3–6，4 条','同一动捕室，按完整轨迹隔离'],
            ['IMU-Array','0','六姿态标定，16 路','整次车载，16 路','两次物理采集，组合与补偿副本排除'],
            ['OxIOD','0','4 类场景 65 条与 1 条 handbag 补充','另四类场景 67 条与 1 条 pocket 补充','28 条 Tango 为另一刚体'],
            ['RepoIMU','0','62 条','26 条','按器械和试验编号分组，7 条无姿态'],
            ['YT_Server','910 条可选，C004 用 24 条','12 个前缀各 1 条','官方 test 213 条','训练与诊断均排除 169 个测试前缀']]
    if counts and counts['through_batch']>='C019':
        splits[-1][1]='910 条可选；C004 用 24 条，C019 用 96 条'
    splits.extend([
        ['RIDI','训练专用受试者 hao 的 4 种佩戴条件','4 条的训练侧诊断','官方测试 25 条，归档可取 23 条','按发布方记录与受试者核对，两个缺档保留分母'],
        ['EuRoC','Vicon Room 1 的 3 条','3 条的训练侧诊断','Vicon Room 2 的 3 条','按房间隔离，easy、medium、difficult 各一条'],
        ['DAVIS','shapes、boxes 各 3 条','6 条的训练侧诊断','poster 的 3 条','按场景隔离，共用仪器和实验室'],
        ['PennCOSYVIO','A-F、A-S 两条','A 组的训练侧诊断','B-F、B-S 两条','按作者划分整条录制，仪器相同、路线近似']])
    add('all8-partitions','十二套数据的训练、开发与测试划分',
        '八套主评的初始清单共 2,525 条。随后登记四套外部迁移检验。37 条自采开发记录仅用于分析和选型，20 条留出按整台设备隔离。BROAD、Sassari 继续使用历史验证组。YT 将同一采集前缀的相邻片段放在同一侧。初始外训 24 条的 16 个文件名首段代号，以及 12 条开发记录的 11 个代号，都出现在官方测试的 26 个代号中。本次可检验跨采集试验表现，跨人独立性尚未得到证实。',
        ['数据集','各批用过的外训源','开发','本轮留出或既有基准','分组与排除'],splits,['C001/inventory.json','review/SPLITS.md','review/yt_group_audit.json','review/split_identity_audit.json','external-sources/sealed-transfer-tests/protocol.json','external-sources/davis-handheld/preprocessing-lock.json','external-sources/penn-handheld/preprocessing-lock.json'])
    markdown.append('[完整划分、有效帧口径与参考来源](SPLITS.md) · [原始逐条清单](../C001/inventory.json)')
    batches=[['C001','0','冻结旧 Mamba','核对 5 个开发域，TUM 两条轨迹平均回退。'],
             ['C002','0','全域修正乘 0.25','TUM 均值下降 0.00123°；阵列 6 个静态朝向均值均回退。'],
             ['C003','0','在线估计加速度计零偏','阵列 16 路有 30.10% 评分帧激活；其他 69 条均未激活。'],
             ['C004','2','BROAD/Sassari 与加入 YT 的外训配对','同一种子、同一主干，分别更新输出层的 75 个参数各 160 步。YT 训练组与正式 test 和诊断 valid 前缀无交集。'],
             ['C005','0','按角速度连续衰减','阵列均值恢复优势，Oxford 比默认高 0.00161°。八套数据均有逐条成绩。'],
             ['C006','0','仅在 VQF 确认静止时关闭修正','TUM 两条轨迹均改善；Oxford 的均值优势缩小到 0.00055°。']]
    add('all8-decisions','前六个小批如何纠偏',
        '每一批由审查方核对上一批的逐记录差值，再给同一个执行聊天派发一个假设。前三批沿用外训的 5,595 参数 Mamba，C004 才开始本轮新的权重训练。两个输出层从完全相同的主干与归一化出发。',
        ['批次','新训练次数','控制的改动','结果'],batches,['review/decisions.jsonl','C004/training-steps.csv','review/C004_audit.json'])
    markdown.append('[逐批决定与纠错记录](decisions.jsonl) · [配对训练与源清单](../C004/protocol.json)')
    add('all8-references','哪些参考支持独立的精度结论',
        '52/66 条 Oxford 开发记录通过独立时段的时间、角速度和竖直方向检查。RepoIMU 的 T 形杆有快速运动导致陀螺读数饱和，三节摆的参考竖直与欠激励安装角存在不确定性。YT 四元数的生成系统未知。阵列的 16 路标定属于同一次运动。开发集的八个平均值均下降，仍不足以证明在八套全新采集的物理真值上均有优势。',
        ['核验','覆盖','结果'],[
            ['相同评分帧','共同开发 213 条','所有行核对原生帧索引摘要；旧 85 条控制分数逐条一致'],
            ['阵列零偏','16 路、6,849 次更新','独立公式从原始比力重建；与六姿态反向对照中心的中位距离 0.0021 m/s²'],
            ['不同采样率','25–285.714 Hz','未来样本扰动不改变历史输入；五档 100 Hz 及以上逐比特匹配原实现'],
            ['Oxford 时空配准','66 条全部计入','拟合、几何检验、评分位于不重叠时段；独立运动竖直检验的记录中位角 0.77°'],
            ['既有 TUM 结果','两条轨迹','C003 单独移除一条轨迹后，另一条的结果可从低 0.01381° 变成高 0.01136°']],
        ['review/optical_development_calibration.json','review/C003_audit.json','review/sample_rate_audit.json','review/C005_group_uncertainty.json'])
    markdown.append('[数据协议与证据索引](README.md) · [上一项独立归档的外训实验](../../transfer-mamba-20261123/review/REPORT.md)')
    if counts and counts['completed_named_batches']>6:
        study=[]
        for candidate in counts['candidates']:
            batch=candidate['batch']
            if batch<'C007':continue
            with (ROOT/batch/'sequences.csv').open(encoding='utf-8-sig',newline='') as f:
                scored=[r for r in csv.DictReader(f) if r.get(candidate['column'])]
            if not scored:continue
            means={}
            for d in order:
                local=[r for r in scored if r['dataset']==d]
                if local:
                    means[d]=[sum(float(r[k]) for r in local)/len(local) for k in ('full_vqf_deg',candidate['column'])]
            worst=max(means,key=lambda d:means[d][1]/means[d][0])
            wins=sum(row[1]<row[0] for row in means.values())
            delta=means[worst][1]-means[worst][0]
            study.append([batch,candidate['label'],f'{wins}/{len(means)}',f"{means['SHICE'][1]:.5f}°",
                          f"{means['TUM-VI'][1]:.5f}°",f"{means['OxIOD'][1]:.5f}°",f'{worst} {delta:+.5f}°'])
        add('all8-guided','外部训练与逐批纠偏',
            '各行均对同一开发面板统计。最弱域按候选与默认的相对误差排序，表中保留该域带符号的实际角度差。TUM 由同一动捕室内的两条开发轨迹组成，自采由 R1–R4 组成。单域的进步可能与另一个域的回退同时发生。',
            ['批次','本次单独检验的改动','均值改善域','自采开发','TUM-VI','Oxford','最弱域差值'],study,
            ['review/decisions.jsonl','review/campaign-counts.json',*[r['batch']+'/sequences.csv' for r in counts['batches'] if r['batch']>='C007']])
        minutes=counts['completed_job_interval_union_s']/60
        recovery=''
        if (ROOT/'C022/retry-launch.json').is_file() and counts['through_batch']>='C022':
            preflight=read(ROOT/'C022/retry-launch.json')
            assert preflight['previous_fit_count']==0
            recovery+=' 第 22 批启动前，审查方给已登记的辅助脚本增添输出，旧指纹触发预检中断。记录首个进程与变更后，实际系数只拟合了一次。'
        if counts.get('recoveries_existing_scores_only',0):
            restored=read(ROOT/'C037/recovery.json')
            recovery+=(f" 第 37 批完成评分后，审查方修改了一个已登记哈希的依赖文件，导致汇总写入中断。恢复原文件后重新核对 {restored['verified_locked_input_count']} 个指纹，保留全部 {restored['verified_existing_development_rows']} 条开发分数，额外重跑记录为 {restored['replay_rows']} 条。")
        add('all8-accounting','已完成批次与计算量',
            f"截至北京时间 {cutoff[:10]} {cutoff[11:19]}，开发作业有进程时戳的区间去重后合计 {minutes:.1f} 分钟。每种实际模型、归一化、物理前端、增益和门控组合计一次配置。"+recovery,
            ['计数项目','实测数量'],[
                ['完成的小批',str(counts['completed_named_batches'])],['不同候选配置，含物理消融',str(counts['new_or_inherited_candidate_policies_tested'])],
                ['固定 FullVQF 比较设置',f"{counts['default_reference_policies']} 套默认与 {counts['additional_fixed_vqf_benchmarks']} 套非默认"],
                ['不同神经网络权重，含继承模型',str(counts['different_neural_parameter_sets_including_inherited'])],
                ['本轮实际训练作业与优化步',f"{counts['optimizer_run_count']} 次，{counts['optimizer_steps']:,} 步"],
                ['有计时器或起止戳的拟合阶段，部分含保存与校验',
                 f"{counts['fits_with_logged_timer_or_boundary_timestamps']} 次，合计 {counts['logged_optimizer_wall_seconds_range'][0]:.1f} 至 {counts['logged_optimizer_wall_seconds_range'][1]:.1f} 秒"],
                ['有单独计时的开发回放',f"{counts.get('batches_with_development_scoring_timer',0)} 批，合计 {counts.get('logged_development_scoring_seconds',0)/60:.1f} 分钟"],
                ['单独用外训数据拟合物理系数',str(counts['external_scalar_fits'])+' 次'],
                ['整批候选与旧配置完全相同的复跑', '、'.join(counts.get('whole_candidate_replay_batches',[])) or '0'],
                ['联合求解多个物理系数',f"{counts.get('external_multicoefficient_solves',0)} 次，{counts.get('external_multicoefficient_iterations',0)} 次求解迭代"],
                ['从已有更新合成的新检查点，新增训练步为零',str(counts.get('averaged_checkpoints_without_new_training',0))],
                ['逐批有分数的记录条目',f"{counts['scored_record_by_batch_entries']:,}"],
                ['已完成小批中的记录评分失败',str(counts['record_failures'])],
                ['有日志的程序错误，修复后继续',str(len(counts['documented_execution_errors']))],
                ['求解器返回成功、独立复算找到更优可行值的批次','、'.join(counts.get('documented_solver_search_misses',[])) or '0'],
                ['保留原权重恢复评分的次数',str(counts['successful_post_lock_evaluation_restarts'])],
                ['逐条评分已经完成，仅恢复写入汇总的次数',str(counts.get('recoveries_existing_scores_only',0))],
                ['完整复用既有记录分数的对照列',str(counts['fully_previously_scored_method_columns'])],
                ['含部分既有分数的对照列',str(counts['partly_previously_scored_method_columns'])],
                ['同一配置、原始记录和掩码重复出现的分数',f"{counts['relisted_score_cells']:,}"],
                ['首次启动时复用的既有逐记录对照',str(counts['archived_score_pairs_reused_at_start'])]],
            ['review/campaign-counts.json','review/candidate_recipes.json','review/decisions.jsonl'])
        markdown.append('[每一小批起止、去重规则及控制复现](campaign-counts.json) · [执行指令与判断](decisions.jsonl)')
    if counts and counts['through_batch']>='C009':
        source='../external-sources/euroc-original-inputs/'
        external=read(ROOT/'external-sources/euroc-original-inputs/source-audit.json')
        original=read(ROOT/'external-sources/euroc-original-inputs/original-imu-verification.json')
        add('all8-source-check','EuRoC 外训源的预处理和原件对账',
            f"该外训来源有 3 条轨迹、66,346 个 200 Hz 时刻，均来自 Vicon Room 1。C009 用公开 HDF5；转换作者对输入角速度扣除了参考系统估计的零偏。我们逐时刻恢复陀螺仪读数，以 easy 的 {original['time_matched_training_rows']:,} 行采集原件核对，最大绝对差 {original['max_abs_reconstructed_gyro_difference_rad_s']:.2g} rad/s。OpenVINS 修订了 easy 的姿态参考。以下静止检验使用源传感器同一批低动态时刻。",
            ['轨迹','同步原生行','预扣陀螺零偏模长中位数','原姿态对静止比力中位角','修订后中位角'],[
                [r['record'],f"{r['hdf5_source_rows']:,}",f"{r['original_bias_magnitude_median_rad_s']:.5f} rad/s",
                 f"{r['original_quiet_vertical_median_deg']:.3f}°",f"{r['alternative_quiet_vertical_median_deg']:.3f}°"]
                for r in external['records']],['external-sources/euroc-original-inputs/source-audit.json',
                'external-sources/euroc-original-inputs/original-imu-verification.json','C009/training-preparation.json'])
        markdown.append('[输入与两种标签来源]('+source+'source-audit.json) · [原始采集对账]('+source+'original-imu-verification.json) · [公开转换代码]('+external['conversion']+') · [参考修订说明]('+external['correction_provenance']+')')
    if counts and counts['through_batch'] >= 'C011':
        compare=read(REVIEW/'C011_audit.json')
        markdown.append('C011 用 63 条外部训练记录的 15,360 个发布时刻拟合一个公共系数，独立重算与执行值之差为 '
                        f"{compare['abs_k_difference']:.2g}。同一系数用于所有数据集。相对 C006，自采由 2.37035° 降至 2.36249°，"
                        'Oxford 由 3.80531° 升至 3.80595°，比默认高 0.00010°。八个开发域各从原始数据重放一条后，最大误差差值为 '
                        f"{max(r['max_replay_gap_deg'] for r in compare['native_anchors']):.2g}°。"
                        '[独立原生重放](C011_audit.json) · [Oxford 共同参考扰动](oxford_development_sensitivity.json)')
    if counts and counts['through_batch'] >= 'C013':
        seed_audit=read(REVIEW/'C013_audit.json')
        with (ROOT/'C013/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            records=list(csv.DictReader(f))
        seed_rows=[]
        fields=['full_vqf_deg','c003_deg','seed_C003_deg','C006_deg','seed_C006_deg']
        for d in order:
            group=[r for r in records if r['dataset']==d]
            seed_rows.append([d,*[f"{sum(float(r[k]) for r in group)/len(group):.5f}°" for k in fields]])
        add('all8-initialization','更换 Mamba 初始化种子后的表现',
            f"两个 Mamba 的初始化种子分别为 1729、314159。原外训抽样与归一化相同，按同一外部准则分别选中第 6 和第 5 个 epoch。C013 固定第二个模型回放 C003，再用原 C004-B 的 {seed_audit['same_source_draws']:,} 个外训槽拟合一次输出层，复现 C006 配方。第二个种子下，两种配方各赢 7 个开发域，分别在 TUM-VI 和 BROAD 失守。",
            ['数据集','默认 VQF','C003 · 1729','C003 · 314159','C006 · 1729','C006 · 314159'],seed_rows,
            ['C013/sequences.csv','C013/seed-lineage.json','review/C013_audit.json'])
        markdown.append('[两个种子的逐记录成绩](../C013/sequences.csv) · [六次独立原生重放与源抽样核对](C013_audit.json)')
    if counts and counts['through_batch'] >= 'C014' and (REVIEW/'SHICE_development_motion.json').exists():
        own=read(REVIEW/'SHICE_development_motion.json')['groups']
        bins=[('all','全部评分时刻'),('gyro_lt_0.05','低于 0.05 rad/s'),
              ('gyro_0.05_to_0.5','0.05 至低于 0.5 rad/s'),('gyro_0.5_to_2','0.5 至低于 2 rad/s')]
        vals=[]
        for k,name in bins:
            r=own[k]
            vals.append([name,f"{r['supporting_records']} 条",f"{r['frames']:,}",f"{r['record_equal_baseline_deg']:.5f}°",
                         f"{r['record_equal_C014_B_deg']:.5f}°",f"{r['record_equal_delta_deg']:+.5f}°",
                         str(r['records_improved'])])
        add('all8-self-motion','自采在不同转速下的收益',
            '审查方从头重放了 37 条自采开发记录。C014-B 将两种初始化所得的修正各取一半，所有原始评分行的比力模长均处于重力大小的 ±5% 内。低角速度段 37 条均改善；在 0.5 至 2 rad/s 的 30 条记录中，9 条改善，逐记录差值中位数为回退 0.018°。各档记录数由实际有评分帧的记录决定。',
            ['角速度模长','记录数','评分帧','默认 VQF','C014-B','平均差','改善记录数'],vals,
            ['review/SHICE_development_motion.json','review/C014_audit.json','C014/sequences.csv'])
        markdown.append('[自采 37 条按转速的逐记录重放](SHICE_development_motion.json)')
    if counts and counts['through_batch'] >= 'C015' and (REVIEW/'C015_audit.json').exists():
        proof=read(REVIEW/'C015_audit.json')['RIDI_substitution']
        rid=read(ROOT/'external-sources/ridi-phone-train/source-audit.json')
        native={r['record']:r for r in rid['rows']}
        add('all8-phone-source','四种手机携带方式的外部训练',
            'C015 每步用 8 个手机训练时刻替换原来的 96 个外训槽中的 8 个，160 步共 1,280 个手机槽，另 14,080 个原槽的记录与时刻完全一致。四段都来自官方训练参与者 hao，输入为原始加速度与陀螺，参考是手机 Tango 视觉惯性姿态。自采只参加评估。手机外训使 TUM 和 Oxford 的开发均值较 C006 下降，但 BROAD 的 0.64784° 高于默认的 0.64547°。',
            ['外训记录','放置','可用发布时刻','实际训练槽','不同训练时刻','运动时陀螺方向检验'],[
                [r['record'],{'bag':'手提包','body':'躯干','handheld':'手持','leg':'腿部'}[native[r['record']]['placement']],
                 f"{r['eligible_releases']:,}",str(r['executed_draws']),str(r['unique_releases_drawn']),
                 f"{native[r['record']]['gyro_reference_median_deg']:.2f}°"] for r in proof['source_inputs']],
            ['C015/sampling.npz','review/C015_audit.json','external-sources/ridi-phone-train/source-audit.json'])
        markdown.append('[RIDI 官方项目]('+rid['project']+') · [来源与抽样复算](C015_audit.json)')
    if counts and counts['through_batch']>='C017':
        audit=read(REVIEW/'C017_audit.json')
        with (ROOT/'C017/SHICE-speed-strata.csv').open(encoding='utf-8-sig',newline='') as f:
            strata=list(csv.DictReader(f))
        values=[]
        for key,label in [('lt_0.05','低于 0.05'),('0.05_to_0.5','0.05 至低于 0.5'),('0.5_to_2','0.5 至低于 2')]:
            valid=[r for r in strata if r['stratum']==key and r['status']=='scored']
            wins=sum(float(r['C017_deg'])<float(r['full_vqf_deg']) for r in valid)
            angles=[sum(float(r[k]) for r in valid)/len(valid) for k in ('full_vqf_deg','C014_B_deg','C017_deg')]
            values.append([label,str(len(valid)),f"{sum(int(r['scored_frames']) for r in valid):,}",
                           *[f'{v:.5f}°' for v in angles],f'{wins}/{len(valid)}'])
        add('all8-stable-force','C017 改善了快速转动时的自采倾角',
            '审查方发现 C014-B 在较快转动段仅改善 9/30 条，随后把同一对 Mamba 固定，只在转动中比力方向稳定时加入补偿。一个系数完全由外部 67 条训练记录的 15,360 个时刻决定。在完全相同的自采评分行上，新版本改善了较快转动的 28/30 条，八个开发域的均值均低于默认。独立重算的系数与执行记录相差 '
            f"{audit['abs_k_error']:.2g}，八域各一次原生重放的分数最大差 {max(r['max_abs_gap_deg'] for r in audit['raw_replays']):.2g}°。",
            ['角速度模长（rad/s）','记录数','评分帧','默认 VQF','原 C014-B','C017','C017 改善记录'],values,
            ['C017/SHICE-speed-strata.csv','C017/coefficient-lock.json','review/C017_audit.json','C017/sequences.csv'])
        markdown.append('[C017 全部 213 条](../C017/sequences.csv) · [外部训练源的同系数复核](C017_audit.json)')
    if counts and counts['through_batch']>='C018':
        with (ROOT/'C018/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        values=[]
        for d in order:
            selected=[r for r in rows if r['dataset']==d]
            means=[sum(float(r[k]) for r in selected)/len(selected) for k in ('full_vqf_deg','C017_deg','A_deg','B_deg')]
            values.append([d,str(len(selected)),*[f'{v:.5f}°' for v in means],f'{100*(1.-means[-1]/means[0]):.3f}%'])
        add('all8-temporal','较早的 Mamba 状态是否有用',
            'C018 固定 C017，只用相同的四个外训来源和 15,360 个发布时刻，分别训练两个 75 参数输出层。A 读取窗口末端的状态，B 读取前 56 个状态的均值与后 8 个状态均值之差。B 在八套开发数据上均比 C017 进一步下降，A 的自采误差更低，但 Oxford 收益缩小。审查方重算了两臂 320 步的更新前后损失，并从四个域的原始记录重放两臂。表内均值逐记录等权。',
            ['数据集','开发记录','默认 VQF','原 C017','末端状态 A','历史状态差 B','B 相对默认改善'],values,
            ['C018/sequences.csv','C018/training.json','review/C018_audit.json','C018/old-context-intervention.json'])
        markdown.append('[C018 的逐记录结果](../C018/sequences.csv) · [时序干预与独立重算](C018_audit.json)')
    if counts and counts['through_batch']>='C020' and (REVIEW/'C020_audit.json').is_file():
        audit=read(REVIEW/'C020_audit.json')
        panel=read(ROOT/'C019/training-preparation.json')
        with (ROOT/'C020/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        values=[]
        for d in order:
            local=[r for r in rows if r['dataset']==d]
            means=[sum(float(r[k])for r in local)/len(local)for k in ('full_vqf_deg','C018_B_deg','C019_deg','C020_deg')]
            values.append([d,str(len(local)),*[f'{v:.5f}°' for v in means],f'{means[3]-means[2]:+.5f}°'])
        add('all8-full-training','扩大外训覆盖和解冻 Mamba 各有多大收益',
            f"C019 把 YT 的训练采集从 24 次扩到 96 次，保留 1,180 个原 YT 时刻，从 72 次新增采集中抽取 {panel['YT_slots_replaced']:,} 个时刻。另三个来源的 {panel['other_source_slots_exact']:,} 个时刻完全一致。C020 沿用这 15,360 个时刻，端到端训练新增分支的 5,595 个参数。"
            f"审查方用原始梯度和 AdamW 更新量重建了 {audit['actual_steps']} 步的全部权重，并在 CPU 独立重算第 1、80、160 步的完整梯度。其余两个 Mamba 和物理前端固定。C020 比 C019 在七套开发数据上略有改善，Sassari 略有回退。",
            ['数据集','记录','默认 VQF','24 次采集 C018-B','96 次采集 C019','端到端 C020','C020 减 C019'],values,
            ['C020/sequences.csv','C019/training-preparation.json','review/C019_audit.json','review/C020_audit.json'])
        markdown.append('[C020 逐记录成绩](../C020/sequences.csv) · [外训样本与 160 步权重审查](C020_audit.json) · [C018-B 全部 213 条独立原生重放](development_replay_C018_B_deg.json)')
    if counts and counts['through_batch']>='C021' and (REVIEW/'C021_audit.json').is_file():
        with (ROOT/'C021/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        values=[]
        for d in order:
            selected=[r for r in rows if r['dataset']==d]
            vals=[sum(float(r[k]) for r in selected)/len(selected) for k in ('full_vqf_deg','C020_deg','C021_deg')]
            values.append([d,*[f'{v:.5f}°' for v in vals],f'{vals[2]-vals[1]:+.5f}°'])
        add('all8-long-history','2.56 秒历史改善自采，但失去 BROAD 优势',
            '同一组外部时刻、初始权重和 160 次更新，新增 Mamba 分支每四个 10 ms 观测块合为一个 40 ms 块。较长历史仍有 15,312 个有效外训时刻，另 48 个来自 Sassari 开始阶段，增量为零。自采误差进一步降低约 0.01956°，BROAD 却比默认高约 0.01291°，该配置在 7/8 个开发域领先。审查方按全域目标保留短窗 C020 作下一批的中心。',
            ['数据集','默认 VQF','0.64 秒 C020','2.56 秒 C021','长窗减短窗'],values,
            ['C021/sequences.csv','review/C021_audit.json','C021/feature-normalization.json','C021/old-context-intervention.json','review/decisions.jsonl'])
        markdown.append('[短长窗的同源配对与所有梯度复核](C021_audit.json) · [逐条开发结果](../C021/sequences.csv)')
    if counts and counts['through_batch']>='C023' and (REVIEW/'C023_audit.json').is_file():
        with (ROOT/'C023/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        values=[]
        for d in order:
            panel=[r for r in rows if r['dataset']==d]
            m=[sum(float(r[k]) for r in panel)/len(panel) for k in ('full_vqf_deg','C020_deg','C022_deg','C023_deg')]
            values.append([d,*[f'{x:.5f}°' for x in m],f'{100*(1.-m[-1]/m[0]):.3f}%'])
        add('all8-response-router','Mamba 能否判断何时信任较快的重力估计',
            'C022 在 139 条外部记录上拟合一个固定增益。自采均值进一步下降，但 TUM 的两条开发轨迹合计回退。C023 改用同一 Mamba 的 24 项历史状态，从 15,360 个相同时刻训练 50 个权重和偏置，逐次发布时混合快、默认和慢三种重力响应。它在八个开发域均领先默认，最弱域优势达到 1.045%。审查方独立计算了 160 步的全部 8,000 项参数更新，又从五条原件重放。',
            ['开发域','默认 VQF','原 C020','固定混合 C022','依历史混合 C023','C023 相对默认改善'],values,
            ['C023/sequences.csv','C023/router-lock.json','review/C022_audit.json','review/C023_audit.json','review/decisions.jsonl'])
        markdown.append('[固定系数与外训的因果状态权重](C023_audit.json) · [完整 213 条逐记录对照](../C023/sequences.csv)')
    davis=ROOT/'external-sources/davis-handheld/preprocessing-lock.json'
    if davis.is_file():
        plan=read(davis);rows=[]
        for obj in plan['source_records']:
            rows.append([obj['record'],'参与外训',str(obj['native_rows']),str(obj['scored_rows']),
                         f"{obj['gyro_optical_median_angle_deg']:.2f}°",f"{obj['optical_over_gyro_median_angular_displacement']:.3f}"])
        for name in plan['sealed_records']:rows.append([name,'锁模后整段评估','—','—','—','—'])
        add('all8-davis-source','新增手持外训数据怎样划分和验证',
            'RPG DAVIS240C 的六轴 IMU 以约 998 Hz 采样，光学以约 200 Hz 记录姿态。shapes 与 boxes 两个场景共六次采集参与外训，poster 整场景的三次采集用于锁模后检验。冻结清单先按名称和场景登记，再对训练侧换算坐标、核对原始 IMU 与光学转动，并记录有效行。所有场景使用同一台仪器。',
            ['场景与运动','本次用途','IMU 原生行','可评分行','光学与陀螺轴中位夹角','光学/陀螺角位移中位'],rows,
            ['external-sources/davis-handheld/protocol.json','external-sources/davis-handheld/preprocessing-lock.json','external-sources/davis-handheld/source-files.json'])
        markdown.append('[按完整场景的划分和来源](SPLITS.md) · [原始 IMU、光学与哈希](../external-sources/davis-handheld/source-files.json) · [作者的采集说明](https://rpg.ifi.uzh.ch/davis_data.html)')
    if counts and counts['through_batch']>='C024' and (REVIEW/'C024_audit.json').exists():
        audit=read(REVIEW/'C024_audit.json')
        with (ROOT/'C024/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        values=[]
        for d in order:
            panel=[r for r in rows if r['dataset']==d]
            means=[sum(float(r[k]) for r in panel)/len(panel) for k in ('full_vqf_deg','C023_deg','C024_deg')]
            wins=sum(float(r['C024_deg'])<float(r['full_vqf_deg'])for r in panel)
            values.append([d,*[f'{v:.5f}°'for v in means],f'{wins}/{len(panel)}'])
        add('all8-davis-training','加入六段外训后，自采 37 条全赢，TUM 与 Oxford 回退',
            f"C024 在原先 139 条外训上增加 DAVIS 六条。原有 15,360 个时刻逐一相同，另加的 1,280 个时刻在各自原件内无重复。审查方用六份原始 IMU 与光学文件复算训练中心、目标和快慢修正，逐一复核六次采集对全部 {audit['actual_steps']} 步的梯度。自采降至 2.21687°；两个开发域高于默认，八域排序保留 C023。",
            ['开发域','默认 VQF','原 C023','增加外训 C024','C024 优于默认的记录'],values,
            ['C024/sequences.csv','C024/training-preparation.json','C024/router-lock.json','review/C024_audit.json'])
        markdown.append('[六条真实外训的参与及原生开发重放](C024_audit.json) · [八个目标域的逐条成绩](../C024/sequences.csv)')
    if counts and counts['through_batch']>='C025' and (REVIEW/'C025_audit.json').exists():
        methods=read(ROOT/'C025/training.json')['domains']
        values=[]
        for d in ('BROAD','Sassari','YT_Server','RIDI','DAVIS'):
            t=methods[d]
            values.append([d,str(t['slots']),f"{t['C024_relative_error']:.4f}",f"{t['C025_relative_error']:.4f}"])
        add('all8-motion-context','运动特征改善外训损失，五个开发域领先默认',
            'C025 保留 16,640 个外训槽，在原 24 维历史状态之外使用过去两秒的角速度、比力模长变化、世界坐标内的比力方向离散度和陀螺零偏估计。RIDI 和 DAVIS 的外训损失下降，YT 上升。213 条开发记录上，BROAD 和 Sassari 均值改善，TUM 与 Oxford 仍高于默认，YT 也略高。表中损失除以同一源上零步 C022 的损失。',
            ['外训来源','训练槽','C024 相对外训损失','C025 相对外训损失'],values,
            ['C025/training.json','C025/sequences.csv','review/C025_audit.json','review/source_motion_router_diagnostic.json'])
        markdown.append('[58 个参数和 160 步更新的独立复核](C025_audit.json) · [3,000 个真实包逐包与整段实现的一致性](C025_stream_parity.json)')
    bias_check=REVIEW/'euroc_bias_response.json'
    if bias_check.is_file():
        paired=read(bias_check)
        labels={'raw':'保留原陀螺读数','fixed_prefix_bias_variation':'减去初始 10 秒的一个固定零偏','publisher_bias_variation':'按发布方逐时刻零偏作对照'}
        vals=[]
        for key,label in labels.items():
            d=paired['means'][key]
            vals.append([label,*[f'{d[k]:.5f}°' for k in ('fast_deg','default_deg','slow_deg')]])
        add('all8-external-bias','同一运动中，陀螺零偏可反转快慢 VQF 的排名',
            '三条 EuRoC V1 的加速度、姿态标签和有效时刻完全相同。每条原始陀螺记录分别对照两个使用外部已知零偏构造的输入版本。三条记录上均发生快慢排名反转。这些版本用于解释传感器质量对模型的影响；真实测试只允许因果六轴输入。',
            ['同三次采集的输入条件','VQF 1.5 秒','默认 VQF 3 秒','VQF 6 秒'],vals,
            ['review/euroc_bias_response.json','external-sources/euroc-original-inputs/source-audit.json','review/euroc_paired_sources.py'])
        markdown.append('[三条原件与每个条件逐条成绩](euroc_bias_response.json) · [原始陀螺的对账](../external-sources/euroc-original-inputs/original-imu-verification.json)')
    paired_audit=REVIEW/'C026_audit.json'
    seed_audit=REVIEW/'C027_audit.json'
    if counts and counts['through_batch']>='C026' and paired_audit.is_file():
        choices=['C023','C025','C026']
        if counts['through_batch']>='C027' and seed_audit.is_file():choices.append('C027')
        methods={}
        for code in choices:
            with (ROOT/code/'sequences.csv').open(encoding='utf-8-sig',newline='')as f:
                methods[code]=list(csv.DictReader(f))
        values=[]
        for domain in order:
            def average(code,col):
                cells=[float(r[col])for r in methods[code] if r['dataset']==domain]
                return sum(cells)/len(cells)
            values.append([domain,f"{average('C023','full_vqf_deg'):.5f}°",
                           *[f"{average(code,code+'_deg'):.5f}°" for code in choices]])
        description='C026 在原来的 145 条外训上加入 3 次 EuRoC 采集。每次取 160 个相同发布时刻，分别输入原始陀螺与减去初始固定零偏的陀螺。训练中共 148 条原件、17,600 个输入槽。审查方重建了全部 480 对时刻和 160 步梯度。Oxford 刚刚低于默认，TUM 和 YT 的均值仍高于默认。'
        if 'C027' in choices:
            description+=' C027 回到 C023 的 139 条外训，用预设种子 53129 重排 160 个小批次，每批的 96 个时刻、初始化、学习率与步数相同。'
        refs=[c+'/sequences.csv'for c in choices]+['review/C026_audit.json','C026/sampling-lock.json']
        if 'C027' in choices:refs.extend(['review/C027_audit.json','C027/sampling-lock.json'])
        add('all8-paired-and-seed','外训零偏对照与训练顺序复现',description,
            ['开发域','默认 VQF',*choices],values,refs)
        markdown.append('[EuRoC 配对采集、梯度与原始重放](C026_audit.json)')
    if counts and counts['through_batch']>='C028' and (REVIEW/'C028_audit.json').is_file():
        codes=['C026','C028']
        if counts['through_batch']>='C029' and (REVIEW/'C029_audit.json').is_file():codes.append('C029')
        values=[];stats={}
        for code in codes:
            with (ROOT/code/'sequences.csv').open(encoding='utf-8-sig',newline='')as source:
                scored=list(csv.DictReader(source))
            for d in order:
                rows=[r for r in scored if r['dataset']==d]
                stats[code,d]=sum(float(r[code+'_deg'])for r in rows)/len(rows)
                stats['default',d]=sum(float(r['full_vqf_deg'])for r in rows)/len(rows)
        for d in order:values.append([d,f"{stats['default',d]:.5f}°",*[f"{stats[c,d]:.5f}°"for c in codes]])
        body='C028 用外部参考直接判断快、慢和原估计三种响应中哪个误差最小，对难以区分的时刻降低监督权重。17,600 个时刻、148 次原始采集及推理输入与 C026 相同。TUM 和 YT 的开发误差下降，但两者仍高于默认。'
        if 'C029' in codes:
            body+=' C029 用相同标签另加入两个实时输入，即快慢估计的夹角和该角度与陀螺偏置的乘积。八个开发域中有 '+str(sum(stats['C029',d]<stats['default',d] for d in order))+' 个低于默认。'
        add('all8-geometry-teacher','直接学习响应选择与新增实时夹角',body,
            ['开发域','默认 FullVQF',*codes],values,
            [*[c+'/sequences.csv' for c in codes],'review/external_response_teacher.json','review/C028_audit.json']
            +(['review/C029_audit.json','review/response_disagreement.json'] if 'C029' in codes else []))
        markdown.append('[17,600 个监督样本及每一步更新的独立复核](C028_audit.json)')
    if (REVIEW/'C031_audit.json').is_file() and counts and counts['through_batch']>='C031':
        cases=('C023','C027','C030','C031')
        values=[]; lookup={}
        for batch in cases:
            with (ROOT/batch/'sequences.csv').open(encoding='utf-8-sig',newline='') as src:
                rows=list(csv.DictReader(src))
            for domain in order:
                part=[r for r in rows if r['dataset']==domain]
                for method in ('full_vqf_deg',batch+'_deg'):
                    lookup[domain,method]=sum(float(r[method]) for r in part)/len(part)
        for d in order:
            values.append([d,*[f'{lookup[d,col]:.5f}°' for col in ('full_vqf_deg',*[c+'_deg' for c in cases])]])
        add('all8-calibration-order','外训误差最低与训练顺序稳定性', 
            'C030 固定 C023，只拟合四个实时可用的物理系数，压低七种外训条件中最差的相对角度误差。最差条件仍改善 0.12%，但 TUM-VI 开发误差从 1.47639° 升至 1.57554°。C031 则把原训练和顺序复现各最后 80 步的真实权重向量等权平均，得到一套 50 参数选择器。它在八个开发域均优于默认，最小优势出现在 TUM-VI，只有 0.00396°。',
            ['开发集','默认 FullVQF',*cases],values,
            [*[c+'/sequences.csv' for c in cases],'review/C030_audit.json','review/C031_audit.json','review/translation_source_predictions.json'])
        markdown.append('[七种外训条件和实际求解梯度](C030_audit.json) · [两个训练次序的 160 个权重向量复核](C031_audit.json) · [EuRoC 外训输入加倍水平加速度的受控分析](translation_source_predictions.json)')
    if (REVIEW/'C032_audit.json').is_file() and counts and counts['through_batch']>='C032':
        s32=read(REVIEW/'C032_audit.json')
        add('all8-calibration-replay','为什么第 32 批算一次复跑',
            '用相同外训记录两次训练的差异，限制第 30 批物理校正的幅度。校正本身小于训练次序造成的波动，因此缩放系数取 1。神经权重和四个物理系数逐值相同，第 32 批已实际重放的 213 条每条都得到第 30 批的分数。',
            ['核对项','实测值'],[['有效时刻 / 原始时刻',f"{s32['admitted_slots']:,} / {s32['original_slots']:,}"],
            ['次序改变导致的对数几率差，加权 RMS',f"{s32['weighted_RMS_order']:.6f}"],
            ['第 30 批校正的对数几率差，加权 RMS',f"{s32['weighted_RMS_physical']:.6f}"],
            ['缩放系数',str(s32['gamma'])],['两批逐条分数的最大差',f"{s32['max_candidate_vs_C030_difference_deg']:.6f}°"]],
            ['review/C032_audit.json','C032/calibration.json','C032/sequences.csv'])
    penn=ROOT/'external-sources/penn-handheld/preprocessing-lock.json'
    if penn.is_file():
        s=read(penn)
        values=[[r['record'].upper(),f"{r['native_rows']:,}",f"{r['scored_frames']:,}",str(r['gyro_optical_dynamic_pairs']),
                 f"{r['gyro_optical_median_angle_deg']:.2f}°",f"{r['optical_over_gyro_median_displacement']:.3f}",
                 f"{r['quiet_reference_vs_force_median_deg']:.2f}°"] for r in s['source_records']]
        add('all8-penn-source','PennCOSYVIO 手持实测怎样划分', 
            '按作者原划分，A-F、A-S 供外部训练和检查，B-F、B-S 整条留作锁模后检验。四次采集使用同一台 VI-Sensor，沿近似相同的室内外路线快走或慢走。用厂家和作者提供的坐标关系把发布的标记姿态转到 IMU 坐标，与 200 Hz 原始惯性数据按时间匹配。论文给出的姿态图平均角度不确定度为 3.5°，且会平滑个别缺少可见标记的帧。',
            ['外训记录','原始 IMU 行','可评分行','动态配对区间','转轴夹角中位','光学/陀螺转角中位比','静段比力与参考竖直夹角中位'],
            values,['external-sources/penn-handheld/protocol.json','external-sources/penn-handheld/source-files.json','external-sources/penn-handheld/preprocessing-lock.json'])
        markdown.append('[原始信号与时间对齐核验](../external-sources/penn-handheld/preprocessing-lock.json) · [发布方论文，第 IV、VII 节](https://daniilidis-group.github.io/penncosyvio/docs/penncosyvio.pdf)')
    if (REVIEW/'C033_audit.json').is_file() and counts and counts['through_batch']>='C033':
        official=read(ROOT/'C033/penn-training-source-scores.json')
        compare=[]
        for row in official['records']:
            m=row['full_record_tilt_RMSE_deg']
            compare.append([row['record'].upper(),'外训整段',str(row['scored_frames'])+' 帧',*[f'{m[k]:.5f}°' for k in ('full_vqf_deg','C023','C033')]])
        with (ROOT/'C033/sequences.csv').open(encoding='utf-8-sig',newline='')as f: observations=list(csv.DictReader(f))
        for d in order:
            data=[r for r in observations if r['dataset']==d]
            compare.append([d,'开发',str(len(data))+' 条',*[f"{sum(float(r[k]) for r in data)/len(data):.5f}°" for k in ('full_vqf_deg','C023_deg','C033_deg')]])
        add('all8-penn-fit','加入宾大外训后，哪些开发集获益',
            'C033 保留原来四个外训来源的全部 15,360 个训练时刻。160 次更新中，每次从 A-F 和 A-S 各加一个不重复的有效发布时刻，五个来源在损失中等权。独立复核发现两条新增采集在每一步均有非零梯度。新模型在自采开发上由 2.27666° 降至 2.25041°，在 TUM-VI 上由 1.47639° 升至 1.59124°。宾大 A 的完整轨迹属于训练侧评估，B 的两条单列为终测。',
            ['数据或录制名','在本轮的用途','评分量','默认 FullVQF','C023','加入外训的 C033'],compare,
            ['review/C033_audit.json','C033/penn-training-source-scores.json','C033/sequences.csv','C033/training.json'])
        markdown.append('[全部 160 步与 320 个新增观测的独立核验](C033_audit.json) · [五个外训来源的结果](../C033/training.json)')
    if (REVIEW/'C034_audit.json').is_file() and counts and counts['through_batch']>='C034':
        audit=read(REVIEW/'C034_audit.json')
        with (ROOT/'C034/sequences.csv').open(encoding='utf-8-sig',newline='')as f: dev=list(csv.DictReader(f))
        rows=[]
        for d in order:
            subset=[r for r in dev if r['dataset']==d]
            means=[sum(float(r[k])for r in subset)/len(subset)for k in ('full_vqf_deg','C023_deg','C034_deg')]
            rows.append([d,str(len(subset)),*[f'{v:.6f}°' for v in means],f'{means[2]-means[0]:+.6f}°'])
        add('all8-c034','五个外训域都改善后，阵列开发集仍出现回退',
            f"第 34 批在 141 条外部记录上拟合四个物理系数，按当前运动情况混合 C023 与默认 VQF 的预测。已重新计算求解器的 {audit['checked_distinct_solver_points']} 个参数点，并逐包对照 3,000 个时刻。宾大 A 两条完整训练轨迹合计的等权平均误差为 1.64971°，默认 VQF 为 1.65151°。开发集按各域原始记录等权，阵列上的微小差距保留六位小数。",
            ['开发集','记录','默认 FullVQF','C023','C034','C034 减默认'],rows,
            ['review/C034_audit.json','C034/solver-receipt.json','C034/sequences.csv','C034/penn-training-source-scores.json'])
    if (REVIEW/'C035_audit.json').is_file() and counts and counts['through_batch']>='C035':
        fit=read(ROOT/'C035/training.json'); audit=read(REVIEW/'C035_audit.json')
        rows=[]
        for domain,j in fit['domains'].items():
            vals=j['source_record_macro_RMSE_deg']
            rows.append([domain,str(j['records']),str(j['slots']),*[f'{vals[k]:.5f}°'for k in ('C023','C033','C035')],
                         f"{100*(1.-vals['C035']/vals['C023']):+.3f}%"])
        add('all8-trust','控制参数改变量后，五个外训来源能否同时改善',
            f"第 35 批从第 23 批出发，用两次不同训练顺序间实测的参数距离 {audit['radius']:.5f} 限制本次改动。独立复算 {audit['checked_solver_points']} 个求解点，核过 {audit['accepted_parameter_changes']} 次真实权重变化。五个来源对第 23 批均改善约 0.25%。八套开发集均优于默认 VQF，最弱一套 TUM-VI 的优势为 0.896%，小于第 23 批最弱域的 1.045%。",
            ['外训来源','原件','训练时刻','原 C023','加入宾大的 C033','约束更新 C035','C035 相对 C023'],rows,
            ['review/C035_audit.json','C035/training.json','C035/solver-receipt.json','C035/sequences.csv'])
    if (REVIEW/'C036_audit.json').is_file() and counts and counts['through_batch']>='C036':
        issue=read(REVIEW/'C036_solver_gap.json')
        prior=read(REVIEW/'C036_audit.json')
        lines=[['第 36 批',f"{prior['trained_cap_deg']:.6f}°",f"{prior['source_objective']:.8f}",
                '局部程序最小正试点为 '+f"{issue['min_positive_tested_cap_deg']:.5f}°，小角区间被跳过。"]]
        artifacts=['review/C036_audit.json','review/C036_solver_gap.json','C036/solver-receipt.json']
        if (REVIEW/'C037_certificate_check.json').is_file() and counts['through_batch']>='C037':
            corrected=read(REVIEW/'C037_certificate_check.json')
            lines.append(['第 37 批',f"{corrected['cap_deg']:.6f}°",f"{corrected['upper_bound']:.8f}",
                         f"核对 {corrected['leaves']} 个终止区间，完整搜索范围内可能遗漏的改善至多 {corrected['global_gap']:.8f}。"])
            artifacts+=['C037/certificate.json','review/C037_certificate_check.json','C037/sequences.csv']
        add('all8-cap-search','约束神经修正幅度时，求解器漏过了小角度',
            '训练按每个来源的记录等权，目标是压低五个来源中相对默认 VQF 最差的误差比。第 36 批选出零度，213 条开发结果逐条等同于第 22 批的纯物理对照。BROAD 在零度处的右导数为每度 −0.25282，说明从零略微增加上限就会改善这个目标。',
            ['批次','神经校正角度上限','五源最差误差比','求解范围复核'],lines,artifacts)
    if (REVIEW/'C038_audit.json').is_file() and counts and counts['through_batch']>='C038':
        with (ROOT/'C038/sequences.csv').open(encoding='utf-8-sig',newline='')as f: trial=list(csv.DictReader(f))
        values=[]
        for d in order:
            subset=[r for r in trial if r['dataset']==d];means=[sum(float(r[k]) for r in subset)/len(subset)for k in ('full_vqf_deg','C023_deg','C035_deg','C038_deg')]
            values.append([d,*[f'{v:.5f}°'for v in means],f'{100*(1-means[3]/means[0]):+.3f}%'])
        add('all8-midpoint','最后一批兼顾自采、RepoIMU 与 TUM-VI',
            '相对 C023，C035 在四台自采开发设备上均进一步改善；RepoIMU 逐次剔除任一采集组仍改善，而 TUM 两条开发轨迹分别回退 0.01020° 和 0.02217°。最后一次只检验两套同构读出权重的等权中点。141 条外训原件的归一化、源索引与两份父模型逐参数对账，新增优化步数为零。各开发记录均由这一个中点模型实际推理。',
            ['开发集','默认 VQF','父模型 C023','父模型 C035','中点 C038','C038 相对改善'],values,
            ['C038/sequences.csv','C038/synthesis-verification.json','review/C038_audit.json','review/C023_C035_development_shift.json'])
    sensitivity=REVIEW/'oxford_C018_B_deg_sensitivity.json'
    if sensitivity.exists():
        s=read(sensitivity)
        scenarios={'nominal':'名义参考','earlier_10ms':'提前 10 ms','later_10ms':'推迟 10 ms',
                   'body_x_-1deg':'x 轴 −1°','body_x_+1deg':'x 轴 +1°','body_y_-1deg':'y 轴 −1°','body_y_+1deg':'y 轴 +1°'}
        values=[]
        for key,label in scenarios.items():
            values.append([label,*[f"{s['summary'][k]['candidate_minus_default_deg'][key]['candidate_deg']:+.5f}°"
                                  for k in ('all','independently_checked')]])
        add('all8-oxford-reference','Oxford 的微小优势对参考安装角敏感',
            '在 66 条开发记录的共同评分行上，C018-B 比默认低 0.00987°；采用独立时间和几何检验通过的 52 条时，低 0.01197°。两种 ±10 ms 时间扰动下仍领先。将相同参考沿手机 x 轴转 +1° 时，两个范围的均值都转为回退。扰动范围用于解释差值量级。',
            ['共同参考条件','全部 66 条，候选减默认','核验通过 52 条，候选减默认'],values,
            ['review/oxford_C018_B_deg_sensitivity.json','C018/sequences.csv'])
    sensitivity=REVIEW/'oxford_C023_C023_deg_sensitivity.json'
    if sensitivity.exists():
        s=read(sensitivity)
        keys=[('nominal','标称'),('earlier_10ms','提前 10 ms'),('later_10ms','推迟 10 ms'),
              ('body_x_-1deg','x 轴 −1°'),('body_x_+1deg','x 轴 +1°'),('body_y_-1deg','y 轴 −1°'),('body_y_+1deg','y 轴 +1°')]
        values=[[label,*[f"{s['summary'][scope]['candidate_minus_default_deg'][key]['candidate_deg']:+.5f}°"
                          for scope in ('all','independently_checked')]] for key,label in keys]
        add('all8-oxford-c023','C023 在 Oxford 的七种参考条件下都领先默认 VQF',
            '全体 66 条和通过独立几何检查的 52 条分别计算。两种方法在同一组共同有效帧上配对。早期 C018-B 在参考绕 x 轴 +1° 时胜负会翻转，C023 在预定的七种条件下均保留优势。这些扰动展示参考变化的影响范围。',
            ['参考扰动','全部 66 条，候选减默认','独立核验 52 条，候选减默认'],values,
            ['review/oxford_C023_C023_deg_sensitivity.json','C023/sequences.csv'])
    controls=REVIEW/'vqf_response_controls_development.json'
    if controls.is_file():
        v=read(controls)
        values=[]
        for d in order:
            item=v['domains'][d]
            m=item['means']
            columns=('fast_deg','default_deg','slow_deg')
            values.append([d,str(item['scored']),*[f'{m[k]:.5f}°' for k in (*columns,'candidate_deg')],
                str(sum(m['candidate_deg']<m[k] for k in columns))+'/3'])
        add('all8-vqf-settings','普通 VQF 改参数后，八个开发域的比较会变',
            '所有设置使用同一库的完整六轴 VQF，每套设置统一用于全部数据。C020 的八个开发均值均优于默认时间常数 3 秒。把固定的 1.5 秒与 6 秒设置也列入对照时，C020 同时赢三套的只有 BROAD、Sassari 和阵列。自采开发设备更受益于较快的重力响应，TUM-VI 和 Oxford 开发记录更受益于较慢响应。正式检验沿用这三个设置。',
            ['数据集','记录','较快 VQF 1.5 秒','默认 VQF 3 秒','较慢 VQF 6 秒','Mamba C020','C020 优于几套'],values,
            ['review/vqf_response_controls_development.json','review/vqf_response_controls.py','C020/sequences.csv'])
        markdown.append('[三套固定 VQF 的 213 条同帧结果](vqf_response_controls_development.json)')
    (REVIEW/'REPORT.md').write_text('\n\n'.join(markdown)+'\n',encoding='utf8')
    (REVIEW/'report-addendum.json').write_text(json.dumps({'queries':documents,'sections':sections},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(len(sections),sum(len(q['rows']) for q in documents.values()),cutoff)


if __name__=='__main__':main()

