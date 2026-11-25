"""Refresh the local report using the independent round and baseline reviews."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from statistics import fmean, median

from audit_rounds import BATCH, HERE, audit, read, write
from audit_timing import describe_timing


DATA = HERE / "report-src" / "src" / "data.json"
BEIJING = timezone(timedelta(hours=8))
NAMES = {"baseline": "纯 FullVQF 基准复算", "filter": "滤波器参数", "ridge": "轻量残差修正",
         "tcn": "因果时间卷积残差", "equivariant":"旋转等变修正", "joint":"设备与程序同时留出复核",
         "development_latency": "开发集计时复核", "install_rotation": "固定安装方向修正",
         "hybrid": "默认与快速滤波器混合",
         "legacy_head": "按训练折重训的原轻量头",
         "regime_audit": "速度、静止与插值邻域复核",
         "ensemble": "多个模型的倾角方向等权平均",
         "cpu_readout_audit": "同权重 CPU 推断与耗时复核",
         "optimizer_anchor_profile": "同配置重跑核对训练与评分耗时",
         "quaternion_audit": "逐帧四元数与因果前缀复核",
         "stream_gru": "FullVQF 加循环残差模型",
         "reservoir_readout": "固定随机储备池和无记忆对照",
         "causal_dct_readout": "六轴短窗余弦系数与时间置换对照",
         "mounted_sobol_ridge_diagnostic": "固定 Sobol 特征的岭回归复核",
         "ridge_cap_pair_diagnostic": "相同预测下 4° 与 6° 修正上限配对",
         "accelerometer_bias_audit": "加速度计零偏三种处理的同帧对照",
         "accelerometer_bias_head_audit": "零偏补偿与轻量头的三种子配对复核",
         "accelerometer_relative_gain_audit": "加速度计三轴相对增益辨识与倾角对照",
         "bias_physical_head": "由配对实验登记的零偏轻量头候选",
         "ridge_seed_diagnostic": "随机特征岭回归的独立种子复验",
         "mounted_ridge_alpha_seed_diagnostic": "安装旋转岭回归的正则化与种子复验",
         "sensor_hold_diagnostic": "轻量头更新间隔内的坐标系对照",
         "timesfm_encoder_diagnostic": "TimesFM 编码器的等价输出与延迟",
         "forecast_backbone_diagnostic": "冻结预测模型的开发集对照",
         "ema_readout": "按实际时间衰减的线性时序修正",
         "selection": "R5 选模", "final": "锁模后测试"}


def refresh():
    tally = audit()
    write(HERE / "round_audit.json", tally)
    historical = read(HERE / "baseline_audit.json")
    report = read(DATA)
    checked = tally["completed"]
    evidence_time = max(datetime.fromisoformat(r["evidence_ended_at"]) for r in checked).astimezone(BEIJING)
    report["buildStatus"] = "creating"
    report["generatedAt"] = tally["reviewed_utc"]
    report["report"]["asOf"] = evidence_time.date().isoformat()
    tables = report["queries"]
    family_rows = []
    for kind, count in tally["completed_by_kind"].items():
        winner = tally["best_development_by_kind"].get(kind)
        durations = [r["elapsed_seconds_in_closed_attempts"] for r in checked
                     if r["kind"] == kind and r["elapsed_seconds_in_closed_attempts"] is not None]
        cost = f"{median(durations):g} 秒" if durations else "—"
        if durations and len(durations) != count:
            cost += f"（{len(durations)}/{count} 轮有记录）"
        family_rows.append({"group":NAMES.get(kind,kind), "rounds":count,
                            "median_time":cost,
                            "best":f"{winner['score_deg']:.3f}°" if winner else "—",
                            "round_id":winner["id"] if winner else "—"})
    tables["rounds"]["rows"] = family_rows
    round_sources = tables["rounds"]["source"]["files"]
    for source in ("audit_timing.py", "verify_score_profiles.py"):
        if source not in round_sources:
            round_sources.append(source)
    for query in tables.values():
        query["source"]["executedAt"] = tally["reviewed_utc"]
    original = read(BATCH / "rounds/R00001/result.json")
    device_score = original["dev_metrics"]["macro_rmse_deg"]
    best = tally["best_development"]
    time_label = evidence_time.strftime("%m 月 %d 日 %H:%M:%S")
    n = tally["development_candidate_runs"]
    distinct = tally["distinct_development_settings_excluding_seed"]
    development_summary = f"## 阶段审查\n\n截至北京时间 {time_label}，已核对 **{len(checked)} 轮**，其中 **{n} 轮**直接训练或评分候选配置。"
    if best:
        development_summary += (f" 当前开发误差最低的是 `{best['id']}`，为 **{best['score_deg']:.3f}°**。"
                    f"纯 FullVQF 在同批 37 条开发记录上的误差为 **{device_score:.3f}°**。")
    counts = (f"以独立轮号、可复算的配置指纹和落盘的结果三项核对。本次开发候选涉及 **{distinct} 种**去掉随机种子后仍不同的配置。"
              f"直接评分的候选轮中，有 **{tally['same_setting_direct_repetitions']} 轮**在相同设置下更换或复用种子。"
              "每次训练内部的 epoch 和交叉验证折不另计轮次。表中开发成绩取留设备与留程序两项宏平均的均值。")
    profiles = tally["direct_development_score_profiles"]
    counts += (f" 在这 {n:,} 个直接评分轮中，按相同顺序比较每条记录在两种留出方式下的未舍入误差，"
               f"共得到 {profiles['distinct_score_vectors']:,} 组不同的逐记录成绩。"
               f"有 {profiles['different_setting_groups']} 组不同设置产生了逐项相同的成绩，其中 {profiles['cap_only_groups']} 组只改变了修正角度上限。")
    cap_path = HERE / "ridge_6_8_cap_replay.json"
    if cap_path.is_file():
        from verify_ridge_caps import summarize
        caps = summarize(read(cap_path))
        device, motion = caps["device"], caps["motion"]
        counts += (f" 在 M00103 和 R01581 两轮中，独立重放了 8 个训练折。留设备的 {device['updates']:,} 次更新中，"
                   f"限幅前最大修正角为 {device['largest_unclipped_deg']:.3f}°，留程序的 {motion['updates']:,} 次更新中为 "
                   f"{motion['largest_unclipped_deg']:.3f}°。6° 与 8° 两种上限在两种留出方式下分别改变了 "
                   f"{device['different_on_score_frames']:,}/{device['score_frames']:,} 和 "
                   f"{motion['different_on_score_frames']:,}/{motion['score_frames']:,} 个评分帧上的修正向量。")
        for source in ("ridge_6_8_cap_replay.json", "verify_ridge_caps.py"):
            if source not in round_sources:
                round_sources.append(source)
    counts += " 单轮耗时取该类别的中位数，每个轮号先合计其各次尝试从启动到结束的时间，账本精度为秒。"
    counts += describe_timing(tally["timing_through_last_completed"])
    if "hybrid" in tally["completed_by_kind"]:
        counts += " 双滤波器的两个端点分别复现原滤波器和快速滤波器，按所执行的对照配置计数。"
    derived = tally["candidate_entries_from_paired_experiments"]
    if derived:
        counts += (f" 另有 {len(derived)} 个轮号从已完成的三种子配对实验登记入围配置。"
                   f"它们计入全部完成轮号和不同配置数，新增独立训练次数为零。")
    if tally["independent_ridge_seed_runs"]:
        counts += (f" 总完成轮号还包含 {len(tally['independent_ridge_seed_runs'])} 轮独立种子复验，"
                   "使用三份已经登记的配置并重新拟合，单列在诊断轮中。")
    if tally["sensor_hold_checks"]:
        counts += (f" 另有 {len(tally['sensor_hold_checks'])} 轮更新间隔对照，"
                   "用已有轻量头比较两种保持方式，计入诊断轮。不同候选配置数单独统计。")
    if tally["forecast_feature_jobs"]:
        counts += (f" 新预测模型对照已核验 {len(tally['forecast_feature_jobs'])} 个推断计时或特征提取轮号，"
                   "这些轮号不含修正头拟合。")
    for trial in tally["forecast_comparisons"]:
        counts += (f" 预测特征的 {len(trial['arms'])} 个对照组集中记在配对诊断轮 {trial['id']} 中，"
                   f"实际拟合 {trial['fits']} 次。")
    for trial in tally["sobol_checks"]:
        counts += (f" {trial['id']} 计一个诊断轮，其中两档正则化各拟合 24 个读出，"
                   "共用一份确定性特征和 24 个折内安装旋转。这两个设置单列在固定 Sobol 特征的比较中。")
    if any(k != "completed" for k in tally["status_counts"]):
        names = {"running":"运行中","planned":"待运行","cutoff":"到时中断","failed":"失败","rejected":"已淘汰","reused":"复用历史"}
        extra = "、".join(f"{v} 轮{names.get(k,k)}" for k,v in tally["status_counts"].items() if k != "completed")
        counts += f" 同一快照另有{extra}。"
    excluded = tally["completed_excluded_from_ranking"]
    if excluded:
        counts += (f" 有 {len(excluded)} 个已完成的早期循环残差轮号，因为训练隐状态跨越了禁止输入区段，"
                   "仅计入实际完成轮号，其成绩已排除出有效候选数和排名。")
    withdrawn = sum(r["status"] == "rejected" and r["attempts"] == 0 for r in tally["other_statuses"])
    if withdrawn:
        counts += f" 已淘汰的轮号中有 {withdrawn} 个在启动前撤回。"
    failures = tally["failed_attempts"]
    if failures:
        recovered = list(dict.fromkeys(a["id"] for a in failures if a["snapshot_status"] == "completed"))
        counts += f" 按每次启动计，账本共记载 {len(failures)} 次失败。"
        if recovered:
            counts += " 其中 " + "、".join(recovered) + " 在原轮号下重跑成功。"
    samples = historical["raw_replay_samples"]
    max_diff = max(abs(v["raw_replay_full_vqf_deg"] - v["historical_deg"]) for v in samples.values())
    report["report"]["review"] = {
        "byline": "FZR-MiniPC · 北京时间 2026 年 11 月 22 日 23:03 至 11 月 23 日 09:03",
        "summary": development_summary + " 当前展示开发阶段的成绩。R5 将按已登记的规则选模，随后对锁定配置进行外部复测。",
        "counts": counts,
        "comparisonTitle": "此前各方法在三套数据上的实测",
        "comparison": "这张表用于确定起点。三套数据分别包含 R6 的 10 条记录、BROAD 的 39 条记录和 Sassari 三次运动的 18 路传感器。每种方法在同一套数据内共享评分帧，每条序列等权。",
        "extraSections": [
            {"id":"verification","title":"复核依据","queryId":"historical",
             "body":f"历史基线的 67 条记录已逐条核对评分帧并重算宏平均。另从原始 CSV 和 MAT 抽取 4 条，重跑纯 FullVQF，再用 SciPy 从四元数独立计算倾角。抽查与原结果的最大绝对差为 {max_diff:.7f}°。SHICE 参考来自转台编码器，现有标签尚无绝对 1° 精度认证。"},
            {"id":"split-method","title":"设备、运动和评分帧","queryId":"rounds",
             "body":"R1–R4 的 37 条记录用于开发，R1-3 和 R1-4 因编码器对时待复核而剔除。学习模型在这四台设备中轮流留出整台设备，并按转台程序另做一组留出。R5 的 10 条记录用于最后选模。R6 及两套外部数据在先前研究中已评分，本轮沿用它们做锁模后的对照。"},
        ],
    }
    weights = tally["shortlist_development_weights"]
    if weights["count"] > 1:
        scored = weights["weighting"]
        summaries = []
        for mode,label in (("record","每条记录等权"),("device","四台设备等权"),("program","四组程序等权")):
            outcome = scored[mode]
            summaries.append({"mode":label,"first":"、".join(outcome["leaders"]),"score":f"{outcome['mean_deg']:.5f}°",
                              "next":outcome["next"] or "—","gap":f"{outcome['gap_deg']:.5f}°" if outcome["next"] else "—"})
        tables["dev_weights"] = {"rows":summaries,"source":{"label":"短名单候选的 R1–R4 逐记录分数",
            "files":["round_audit.json","verify_dev_weights.py","verify_final.py","../selection_rules.py"],"executedAt":tally["reviewed_utc"]}}
        section = next(s for s in report["report"]["review"]["extraSections"] if s["id"]=="split-method")
        first = set(scored["record"]["leaders"])
        omissions = weights["omissions"]
        device_hits = sum(bool(first.intersection(omissions[k]["leaders"])) for k in weights["device_counts"])
        program_hits = sum(bool(first.intersection(omissions[k]["leaders"])) for k in weights["program_counts"])
        qualifier = "已登记的 R5 名单" if weights["filed"] else "按当时开发证据复算的 R5 预选名单"
        device_sizes = "、".join(str(n) for n in weights["device_counts"].values())
        motion_sizes = "、".join(str(n) for n in weights["program_counts"].values())
        section["body"] += (f" 以下取 {datetime.fromisoformat(weights['at']).astimezone(BEIJING):%H:%M:%S} {qualifier}中的 {weights['count']} 个候选。"
                            f"R1 至 R4 分别有 {device_sizes} 条记录，四个程序组分别有 {motion_sizes} 条。"
                            "每条记录先平均留设备与留程序两项误差，再改变各记录的权重。设备或程序等权时，组内记录等权。"
                            f"沿用已有预测，轮流从计分中去掉一台设备或一组程序，原首位仍列第一的次数分别为 {device_hits}/4 和 {program_hits}/4。")
        controls = weights["controls"]
        if controls["install_rotation"]:
            section["body"] += f" 名单含纯 FullVQF {controls['baseline'][0]} 和固定安装旋转 {'、'.join(controls['install_rotation'])}。"
        section["body"] += " 首轮纯 FullVQF 已另行记载 R5 的基准成绩。审查重建短名单时，只向选择器提供开发指标、配置和开发阶段的准入记录。"
        section.update(queryId="dev_weights",columns=[{"field":k,"label":v} for k,v in
                       (("mode","开发记录权重"),("first","首位候选"),("score","首位误差"),("next","次名候选"),("gap","与次名差距"))])
        nominees = weights["nomination_evidence"]
        nomination_rows = [{"round":r["round_id"],"method":NAMES.get(r["kind"],r["kind"]),
                            "dev":f"{r['development_rmse_deg']:.5f}°","why":"。".join(e["explanation"] for e in r["evidence"])}
                           for r in nominees["entries"]]
        source_files = ["round_audit.json","verify_shortlist_reasons.py","../selection_rules.py"]
        nomination_body = (f"{weights['count']} 个候选共有 {nominees['reason_count']} 条入围理由。按 R1–R4 的分数、训练种子、"
                           "折内输入和各方法的入围条件逐条复算。有多个理由的候选只在表中出现一次。")
        if change := nominees.get("wording_change"):
            ratio = change["coverage"]
            effective = datetime.fromisoformat(change["effective_at"])
            nomination_body += (f" 门控 H00010 的 1% 门槛取各条记录完整回退比例的等权平均，{ratio['record_count']} 条均值为 {ratio['macro_fraction']:.5%}。"
                                f"若合并帧数则为 {ratio['fallback_frames']:,}/{ratio['frames']:,}，即 {ratio['pooled_fraction']:.5%}。"
                                f"{effective:%H:%M:%S} 更正了这项理由的文字。同一开发快照的 {change['candidate_count']} 个候选、顺序和分数逐项一致。")
            source_files.extend(["../hybrid-fallback-shortlist-reason-audit.json","../rounds/H00010/result.json"])
            if raw := change.get("native_raw_replay"):
                nomination_body += (f" 从 {raw['records']} 条原始 CSV 重新运行两个滤波器并独立重建门控，"
                                    f"各记录的完整回退帧数逐一吻合，倾角 RMSE 最大差 {raw['max_abs_rmse_difference_deg']:.2g}°。")
                source_files.extend(["gate_fallback_raw_replay.json","replay_gate_fallback.py"])
        tables["r5_nominations"] = {"rows":nomination_rows,"source":{"label":"开发候选、种子与入围规则复算",
                                     "files":source_files,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"r5-nominations","title":"R5 入围依据" if weights["filed"] else "R5 入围预览",
            "queryId":"r5_nominations","body":nomination_body,"columns":[{"field":k,"label":v} for k,v in
            (("round","轮号"),("method","方法"),("dev","开发误差"),("why","入围依据"))]})
    for cap_pair in tally["cap_pair_checks"]:
        axes = {"device":"留设备","motion":"留程序","joint_holdout":"同时留设备和程序"}
        cap_rows = []
        for seed,values in cap_pair["seeds"].items():
            for axis,label in axes.items():
                value = values[axis]
                hits = [value["clip"][c]["updates_over_cap_used_on_score_frames"] for c in ("4.0","6.0")]
                cap_rows.append({"seed":seed,"axis":label,"four":f"{value['mean_deg']['4.0']:.6f}°","six":f"{value['mean_deg']['6.0']:.6f}°",
                                 "delta":f"{value['delta_deg']:+.6f}°","records":f"{value['wins']} / {value['losses']} / {37-value['wins']-value['losses']}",
                                 "hits":f"{hits[0]:,} / {hits[1]:,}"})
        tables["paired_caps"] = {"rows":cap_rows,"source":{"label":"三种子上限配对与每次限幅的统计",
            "files":["../ridge-cap-4deg-6deg-protocol.json","../rounds/O00001/result.json","ridge_cap_pair_start.json","verify_cap_pair.py"],
            "executedAt":tally["reviewed_utc"]}}
        combined = {axis:fmean(s[axis]["delta_deg"] for s in cap_pair["seeds"].values()) for axis in axes}
        if cap_pair.get("published_sha256"):
            tables["paired_caps"]["source"]["files"].append("../ridge-cap-4deg-6deg-summary.json")
        six_counts = [value[axis]["clip"]["6.0"] for value in cap_pair["seeds"].values() for axis in axes]
        body = (f"R01745 与 R01700 仅相差一步修正的上限。原监督器在 O00001 中用三个搜索种子和 24 个开发折配对验证，"
                f"共用 {cap_pair['shared_mount_fits']} 个折内安装旋转、{cap_pair['ridge_fits']} 次读出拟合及 {cap_pair['pairs']} 组未截断预测。"
                f"将上限从 4° 调至 6°，三种子误差差值在留设备、留程序、联合留出上依次为 "
                f"{combined['device']:+.6f}°、{combined['motion']:+.6f}°、{combined['joint_holdout']:+.6f}°。"
                f"其中 {cap_pair['matched_anchor_scores']} 个已有逐记录成绩可直接复核，最大差 {cap_pair['anchor_max_abs_difference_deg']:.2g}°。"
                "末列统计两档上限各自发生截断且影响评分帧的更新次数。"
                f"各折、各种子累加，6° 上限触发 {sum(c['updates_over_cap'] for c in six_counts)} 次，"
                f"其中 {sum(c['updates_over_cap_used_on_score_frames'] for c in six_counts)} 次作用于评分帧。"
                f"从当时的开发账本中去掉这一诊断轮，{cap_pair['preview_count']} 个预选项及顺序一致。")
        if raw := cap_pair.get("qr_replay"):
            body += (f" 针对 R4 的变化，用独立 QR 解复算联合留出折 R4/M4，并重放两条原始记录，共 {raw['frames']:,} 个评分帧。"
                     f"两档上限逐条触发计数相符，四项倾角误差的最大差为 {raw['max_abs_rmse_difference_deg']:.2g}°。")
            tables["paired_caps"]["source"]["files"].extend(["ridge_cap_pair_raw_replay.json","replay_cap_pair_fold.py"])
        report["report"]["review"]["extraSections"].append({"id":"paired-caps","title":"4° 和 6° 上限对倾角的影响","queryId":"paired_caps",
            "body":body,"columns":[{"field":k,"label":v} for k,v in (("seed","种子"),("axis","验证方式"),("four","4° 上限"),
            ("six","6° 上限"),("delta","6° 减 4°"),("records","改善 / 变差 / 持平"),("hits","影响评分的截断更新（4° / 6°）"))]})
    if tally["relative_gain_checks"]:
        support = tally["relative_gain_checks"][-1]
        sources = ["../accelerometer-relative-gain-development-protocol.json","../rounds/A00006/result.json",
                   "verify_relative_gains.py","round_audit.json"]
        body = (f"从 {support['records']} 条自采开发记录的 {support['static_blocks']:,} 个静止块得到 "
                f"{support['pose_centroids']} 个记录内姿势均值。在训练折固定已有零偏，只允许三轴相对增益变化。"
                f"{support['checked_folds']} 折中有 {len(support['supported'])} 折满足姿势覆盖与跨训练设备一致性，"
                f"实际应用的单轴增益与 1 最大相差 {support['max_gain_change']:.4%}。")
        section = {"id":"relative-gains","title":"三轴加速度计增益能否改善倾角","queryId":"rounds","body":body}
        if tally["relative_gain_scores"]:
            trial = tally["relative_gain_scores"][-1]
            rows = []
            for axis,label in (("device","留设备"),("motion","留运动程序"),("joint_holdout","同时留设备与程序")):
                a = trial["axes"][axis]
                values = a["scores"]
                rows.append({"axis":label,"zero":f"{values['zero']:.6f}°","bias":f"{values['fixed_fold_bias']:.6f}°",
                             "gain":f"{values['fixed_fold_bias_plus_relative_gain']:.6f}°",
                             "coverage":f"{a['gain_records']}/{trial['records_per_axis']}"})
            sources.append(f"../rounds/{trial['id']}/result.json")
            section["queryId"] = "gain_tilt"
            section["columns"] = [{"field":f,"label":l} for f,l in (("axis","开发划分"),("zero","未校准"),
                ("bias","只校零偏"),("gain","零偏加相对增益"),("coverage","增益实际启用记录"))]
            joint = trial["axes"]["joint_holdout"]["scores"]
            diff = joint["fixed_fold_bias_plus_relative_gain"]-joint["fixed_fold_bias"]
            section["body"] += (f" 联合留出的平均倾角误差相对只校零偏变化 {diff:+.6f}°。达到 0.015° 改善且通过各项对照，才进入随机种子复验。"
                                + (" 本轮满足复验条件。" if trial["promotion"] else " 本轮未达到复验条件。"))
            replay_path = HERE / "relative_gain_raw_replay.json"
            if replay_path.is_file():
                replay = read(replay_path)
                assert replay["result_sha256"] == hashlib.sha256((BATCH / "rounds" / trial["id"] / "result.json").read_bytes()).hexdigest()
                frames = sum(r["frames"] for r in replay["records"])
                worst = max(m["absolute_difference_deg"] for r in replay["records"] for m in r["scores"].values())
                section["body"] += f" 从原始 CSV 重放了增益启用的 R3-3 和回退的 R4-2，覆盖 {frames:,} 个评分帧，三种处理与逐条成绩最大相差 {worst:.7f}°。"
                sources.append("relative_gain_raw_replay.json")
            tables["gain_tilt"] = {"rows":rows,"source":{"label":"逐条核对训练折与原始倾角","files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append(section)
    seeds = tally["random_feature_ridge_seed_check"]
    if seeds["complete_common_seed_sets"]:
        family_count = seeds["complete_common_seed_sets"]
        rows = [{"seed":str(s), "mean":f"{avg:.3f}°", "wins":n}
                for s,avg,n in zip(seeds["seeds"],seeds["seed_means_deg"],seeds["unique_lowest_counts"],strict=True)]
        tables["ridge_seeds"] = {"rows":rows, "source":{"label":"物理特征随机岭回归的同配置种子对照",
            "files":["round_audit.json","../round_index.json","audit_rounds.py"], "executedAt":tally["reviewed_utc"]}}
        body = (f"共有 {seeds['settings_with_random_features']} 种配置采用物理输入和随机特征，其中 {family_count} 种在同一组三个种子下都完成了评分。"
                "表中用这批完整配置比较种子，每种配置等权，误差取留设备与留程序两项的均值。"
                f"同一配置在三个种子之间的最大与最小误差之差，中位数为 {seeds['median_within_setting_range_deg']:.3f}°。"
                "搜索过程根据已有开发成绩挑选配置进行复跑，这些差异描述的是本批入选复跑的配置。")
        if seeds["ties"]:
            body += f" 另有 {seeds['ties']} 种配置并列最低。"
        columns = [{"field":"seed","label":"随机种子"},{"field":"mean","label":"开发误差均值"},{"field":"wins","label":"单独最低的配置数"}]
        paired = tally["matched_regularization_gain"]
        if paired:
            by_seed = {str(p["seed"]):p for p in paired["pairs"]}
            for row in rows:
                pair = by_seed[row["seed"]]
                row.update(old=f"{pair['old_deg']:.5f}°",new=f"{pair['new_deg']:.5f}°")
            columns.extend([{"field":"old","label":"R01022 配置"},{"field":"new","label":"R01442 配置"}])
            tables["ridge_seeds"]["source"]["files"].append("verify_ridge_seeds.py")
            body += (f" 表末两列是已完成复跑的 25 Hz 配置，岭惩罚系数从 0.0024752 降至 0.0017121，修正增益从 0.980 降至 0.966。"
                     f"相同三个种子等权后，开发误差由 {paired['old_mean_deg']:.5f}° 变为 {paired['new_mean_deg']:.5f}°。"
                     f"新配置减去原配置，留设备为 {paired['axes']['device']['delta_deg']:+.5f}°，留程序为 {paired['axes']['motion']['delta_deg']:+.5f}°。")
        report["report"]["review"]["extraSections"].append({"id":"ridge-seeds","title":"随机特征模型换种子后的表现",
            "queryId":"ridge_seeds","body":body,"columns":columns})
    diagnostic = tally["independent_ridge_seed_runs"]
    if diagnostic:
        from verify_ridge_seeds import compare_complete_summary, matched_summary
        families = matched_summary(diagnostic)
        successful = sum(r["status"] == "complete" for r in diagnostic)
        scored = sum(g["records"] for r in diagnostic for g in r["coverage"].values())
        fits = sum(r["successful_fits"] for r in diagnostic)
        body = (f"先固定单轮领先的 R00915、三个原种子均值领先的 R00900 和早期的 R00193，再为三者使用相同的三个新种子。"
                f"截至本次审查，{successful}/9 组取得三种划分的全部成绩，共核实 {fits} 次成功拟合、{scored} 条逐记录评分。"
                "这些轮号单列为开发复验，每个轮号对应一份旧配置和一个新种子。它们不参与生成候选或 R5 排序。")
        if successful < len(diagnostic):
            body += f" 其中 {len(diagnostic)-successful} 个已落盘的诊断轮评分有缺额。"
        overview = []
        for f in families:
            axes = f["axes"]
            diff = axes["joint_holdout"]["paired_mean_deg"]
            overview.append({"source":f["source"], "seeds":len(f["seeds"]), **{a:f"{v['mean_deg']:.3f}°" for a,v in axes.items()},
                             "paired":f"{diff:+.3f}°" if diff is not None and f["source"] != "R00193" else "—"})
        if successful == 9:
            candidate = next(f for f in families if f["source"] == "R00915")
            per_record = candidate["joint_paired_per_record_deg"]
            devices = sorted({name.split("-")[0] for name in per_record})
            device_deltas = []
            for device in devices:
                differences = [v for name,v in per_record.items() if name.split("-")[0] == device]
                device_deltas.append(f"{device} {fmean(differences):+.3f}°")
            body += (f" R00915 在三个新种子上的联合留出误差相对 R00193 的平均变化为 {candidate['axes']['joint_holdout']['paired_mean_deg']:+.3f}°。"
                     f"逐条先对种子取平均，{sum(v<0 for v in per_record.values())}/{len(per_record)} 条记录改善，各设备的平均差值为 "
                     + "、".join(device_deltas)+"。")
        files = ["../ridge-independent-seed-protocol.json","verify_ridge_seeds.py", "round_audit.json",*["../"+r["result"] for r in diagnostic]]
        published = BATCH / "ridge-independent-seed-summary.json"
        if successful == 9 and published.is_file():
            summary = read(published)
            if all(len(v["successful_seeds"]) == 3 for v in summary["by_family"].values()):
                compare_complete_summary(families,summary)
                files.append("../ridge-independent-seed-summary.json")
        replay_path = HERE / "ridge_seed_fold_replay.json"
        if replay_path.is_file():
            replay = read(replay_path)
            source = next(r for r in diagnostic if r["round_id"] == replay["round_id"])
            with (BATCH / source["result"]).open("rb") as stream:
                assert replay["result_sha256"] == hashlib.file_digest(stream,"sha256").hexdigest()
            records = replay["checked"]
            gap = max(r["abs_delta_deg"] for r in records)
            body += (f" 另取首个联合留出折，用相同物理特征和独立的加权 QR 求解重新拟合 {replay['train_labels']:,} 个训练标签。"
                     f"在 {len(records)} 条留出记录的 {sum(r['frames'] for r in records):,} 帧上复算倾角，逐记录 RMSE 最大相差 {gap:.8f}°。")
            files.append("ridge_seed_fold_replay.json")
        tables["fresh_seeds"] = {"rows":overview, "source":{"label":"预先固定配置的新种子复验","files":files,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"fresh-ridge-seeds","title":"用新种子复验搜索收益",
            "queryId":"fresh_seeds","body":body+" 表中每条记录等权，再对完整种子取平均。最后一列是联合留出时相对早期配置 R00193 的同种子误差差值。",
            "columns":[{"field":"source","label":"来源配置"},{"field":"seeds","label":"完成种子"},{"field":"device","label":"留设备"},
                       {"field":"motion","label":"留程序"},{"field":"joint_holdout","label":"联合留出"},{"field":"paired","label":"相对早期配置"}]})
    feature_jobs = tally["forecast_feature_jobs"]
    if feature_jobs:
        plan = read(BATCH/"forecast-backbone-development-protocol.json")
        labels = {"timestral_209m":"Timestral-209M","timestral_28m_fallback":"Timestral-28M",
                  "timesfm_3":"TimesFM-3","physical_zero":"纯物理对照"}
        by_model, sources = {}, ["../forecast-backbone-development-protocol.json","verify_forecast_probe.py"]
        for job in feature_jobs:
            item = by_model.setdefault(job["model"],{"cache":{},"p95":None})
            for cache in job["caches"]:
                existing = item["cache"].setdefault(cache["name"],cache)
                assert existing["sha256"] == cache["sha256"]
            if "p95_ms" in job:
                item["p95"] = job["p95_ms"]
            sources.append("../rounds/"+job["id"]+"/result.json")
        feature_rows, replayed = [], []
        for key, evidence in by_model.items():
            caches = evidence["cache"]
            raw_file = HERE/("forecast_probe_"+key+".json")
            if raw_file.exists() and plan["release_record"] in caches:
                raw = read(raw_file)
                assert raw["cache_sha256"] == caches[plan["release_record"]]["sha256"] and raw["model"] == key
                replayed.append(labels[key])
                sources.append(raw_file.name)
            physical = key == "physical_zero"
            feature_rows.append({"method":labels[key],"records":"零特征对照" if physical else f"{len(caches)}/{len(plan['development_records'])}",
                "windows":"按输入生成零列" if physical else f"{sum(c['windows'] for c in caches.values()):,}",
                "p95":f"{evidence['p95']:.2f} ms" if evidence["p95"] is not None else "待计时"})
        body = ("三臂使用同一段六轴历史。每次取过去 64 个 100 Hz 输入块，以第 1、4、8 个未来块的中位预测减去最后观测，"
                "得到 18 个特征。纯物理对照在相同位置填零。表中窗口数只计已核对索引和文件指纹的缓存。"
                "P95 为本机单次预测、读出及输出修正的合计，计时从滤波状态和输入块可用时开始。探测使用同维度固定读出权重。")
        if replayed:
            body += (" 在 R3-3 的首、中、末三个更新位置，另用原始六轴 CSV 和官方 CPU 接口重建预测，"
                     f"已复核{'、'.join(replayed)}的特征顺序和数值。")
        tables["forecast_inputs"] = {"rows":feature_rows,"source":{"label":"冻结预测模型的开发集输入与同机计时",
                                     "files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"forecast-inputs","title":"两种冻结预测模型在自采数据上如何比较",
            "queryId":"forecast_inputs","body":body,"columns":[{"field":"method","label":"模型"},{"field":"records","label":"已核对记录"},
            {"field":"windows","label":"已提取窗口"},{"field":"p95","label":"单次更新 P95"}]})
        for trial in tally["forecast_comparisons"]:
            names = {"device":"留设备","motion":"留程序","joint_holdout":"同时留设备与程序"}
            scores = trial["scores"]
            rows = [{"arm":labels[arm],"axis":names[axis],"records":str(trial["coverage"][axis]["records"]),
                     "error":f"{values[arm]:.4f}°","change":f"{values[arm]-values['physical_zero']:+.4f}°",
                     "delay":f"{trial['delayed_scores'][axis][arm]:.4f}°" if arm in trial["delayed_scores"].get(axis,{}) else "—"}
                    for axis,values in scores.items() for arm in trial["arms"]]
            body = (f"已直接核对 {trial['paired_rows']} 条记录与留出划分的组合。每个组合在同一批评分帧上比较三臂，"
                    f"共 {trial['fits']} 次修正头拟合。下面的差值均相对同折纯物理对照，各条记录等权。"
                    f"按预登记的误差与时延条件，有 {len(trial['preliminary_pass'])} 个预测模型通过初筛。")
            if trial["delayed_scores"]:
                body += " 末列将每次预测修正按该模型实测的单次 P95 固定延后，再对同一批帧评分。"
            failures = trial["failed_probes"]
            if failures and all("Official TimesFM predictor discrepancy" in f["last_error"] for f in failures):
                numbers = "、".join(f["id"] for f in failures)
                body += f" {numbers} 的 TimesFM 接口一致性检查失败，后续通过官方预测接口完成了全部记录的评分。"
            key = "forecast_scores_"+trial["id"]
            tables[key] = {"rows":rows,"source":{"label":"预测特征的同折结果",
                           "files":["../rounds/"+trial["id"]+"/result.json","round_audit.json"],"executedAt":tally["reviewed_utc"]}}
            report["report"]["review"]["extraSections"].append({"id":"forecast-score-"+trial["id"],"title":"预测特征有没有改善实测倾角",
                "queryId":key,"body":body,"columns":[{"field":"arm","label":"方法"},{"field":"axis","label":"留出方式"},
                {"field":"records","label":"记录数"},{"field":"error","label":"倾角 RMSE"},{"field":"change","label":"相对纯物理"},
                {"field":"delay","label":"按 P95 延后"}]})
        sensor_file = HERE/"sensor_forecast_audit.json"
        if sensor_file.is_file() and any(t["id"]=="F00030" for t in tally["forecast_comparisons"]):
            from inspect_sensor_forecasts import check_saved
            evidence = read(sensor_file)
            check_saved(evidence)
            rows = []
            for entry in evidence["summary"]:
                label = "加速度" if entry["signal"] == "acc" else "角速度"
                rows.append({"signal":f"{label}（{entry['unit']}）","horizon":f"{entry['horizon_ms']} ms",
                             **{k:f"{v:.5f}" for k,v in entry["record_macro_rmse"].items()}})
            body = (f"三臂倾角试验结束后，用 {len(evidence['records'])} 条开发记录的原始六轴数据复查了预测本身。"
                    f"在 {evidence['issued_windows']:,} 个更新窗口中，有 {evidence['eligible_windows']:,} 个窗口的历史至目标均处于有效区间，"
                    "且最后观测块和三个目标块均有完整的原始传感器读数。四个方法在这些共同窗口上比较。"
                    "线性外推沿用最近四个 10 ms 块的斜率。先在每条记录内对三个轴计算分量 RMSE，再对记录等权平均。"
                    "这项事后复核单独记录传感器预测精度，候选入围仍按上表的倾角与时延条件判定。")
            longest = next(r for r in evidence["summary"] if r["signal"]=="acc" and r["horizon_ms"]==80)
            values = longest["record_macro_rmse"]
            body += (f" 在 80 ms 加速度预测上，Timestral 的误差为 {values['timestral_209m']:.5f} m/s²，"
                     f"线性外推为 {values['linear_4blocks']:.5f} m/s²，"
                     f"{longest['record_wins_vs_linear']['timestral_209m']}/{evidence['covered_records']} 条记录的误差下降。")
            tables["sensor_prediction_errors"] = {"rows":rows,"source":{"label":"原始六轴目标与两种简单预测对照",
                        "files":["sensor_forecast_audit.json","inspect_sensor_forecasts.py","../rounds/F00030/result.json"],
                        "executedAt":evidence["reviewed_utc"]}}
            report["report"]["review"]["extraSections"].append({"id":"sensor-prediction-errors","title":"未来 10 至 80 毫秒的传感器读数预测得有多准",
                "queryId":"sensor_prediction_errors","body":body,
                "columns":[{"field":"signal","label":"信号与单位"},{"field":"horizon","label":"预测时距"},{"field":"persistence","label":"保持末值"},
                {"field":"linear_4blocks","label":"线性外推"},{"field":"timesfm_3","label":"TimesFM-3"},{"field":"timestral_209m","label":"Timestral-209M"}]})
    if tally["timesfm_encoder_checks"]:
        result = tally["timesfm_encoder_checks"][0]
        a,b = (result["ms"][k]["p95_ms"] for k in ("full_forward","encoder_only"))
        body = (f"在 4 条开发记录的 {result['windows']} 个窗口比较了完整前向和跳过预测输出层的编码器。两条路径返回相同的嵌入、"
                f"修正向量和四元数。每条路径计时 {result['timed_per_arm']} 次，完整前向 P95 为 {a:.2f} ms，仅编码器为 {b:.2f} ms。")
        report["report"]["review"]["extraSections"].append({"id":"encoder-timing","title":"TimesFM 只算编码器能节省多少时间",
            "queryId":"rounds","body":body})
    if tally["sensor_hold_checks"]:
        comparisons, sources = [], ["../sensor-frame-hold-development-protocol.json","verify_hold_geometry.py"]
        fits = 0
        for trial in tally["sensor_hold_checks"]:
            fits += trial["fits"]
            sources.append("../rounds/"+trial["id"]+"/result.json")
            for axis, label in (("device","留设备"),("motion","留程序"),("joint_holdout","同时留设备与程序")):
                values = trial["axes"][axis]
                comparisons.append({"seed":str(trial["seed"]),"axis":label,"world":f"{values['world_deg']:.5f}°",
                                    "sensor":f"{values['sensor_deg']:.5f}°","delta":f"{values['delta_deg']:+.5f}°"})
        initial = next(r for r in tally["sensor_hold_checks"] if r["id"] == "S00001")
        delta = initial["axes"]["joint_holdout"]["delta_deg"]
        verdict = "达到" if initial["screen_passed"] else "未达到"
        body = (f"轻量修正头每 100 毫秒更新一次。每折的两种方法共享同一套权重，只改变两次更新间的修正方式。一种保持世界坐标系中的修正，"
                f"另一种把修正旋转保留在传感器坐标系，并应用到每一帧的基准姿态。已完成 {len(tally['sensor_hold_checks'])} 个诊断轮号，"
                f"共拟合 {fits} 次。首个种子的联合留出误差变化为 {delta:+.5f}°，{verdict}预先登记的复验条件。")
        raw = initial["raw_checks"]
        if raw:
            checked_frames = sum(r["frames"] for r in raw)
            max_gap = max(r["max_rmse_gap_deg"] for r in raw)
            body += (f" 用原始 IMU 和实验前留存的四元数导出，独立复算两条开发记录的 {checked_frames:,} 个评分帧。"
                     f"两种保持方式在各速度档及更新后各时间段的 RMSE 与执行结果最多相差 {max_gap:.8f}°。")
            sources.append("hold_geometry_replay.json")
        tables["sensor_hold"] = {"rows":comparisons,"source":{"label":"同权重的逐原始帧保持方式对照",
                                 "files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"hold-geometry","title":"修正在两次更新之间怎样保持",
            "queryId":"sensor_hold","body":body,"columns":[{"field":"seed","label":"种子"},{"field":"axis","label":"留出方式"},
            {"field":"world","label":"世界坐标系"},{"field":"sensor","label":"传感器坐标系"},{"field":"delta","label":"误差变化"}]})
    state_rows = tally["fixed_time_state_runs"]
    if state_rows:
        shown = [{"id":r["id"], "time":"、".join(f"{s:g}" for s in r["horizons_seconds"]) or "无记忆",
                  "device":f"{r['device_deg']:.3f}°","motion":f"{r['motion_deg']:.3f}°","joint":f"{r['joint_deg']:.3f}°",
                  "reset":f"{r['same_weight_reset_joint_deg']:.3f}°"} for r in state_rows]
        sources = ["../ema-state-development-protocol.json","verify_ema.py",*["../rounds/"+r["id"]+"/result.json" for r in state_rows]]
        body = (f"用同一批 10 Hz 物理特征和确定性岭回归，比较无记忆与三组固定衰减时间。"
                f"已完成 {len(state_rows)}/4 档，各做 24 折。最后一列保持已拟合的权重，每次更新时将状态清零。"
                f"按开发阶段预设的四项标准，已完成配置中有 {sum(r['r5_qualified'] for r in state_rows)} 档达到 R5 入围条件。")
        if len(state_rows) == 4:
            best_state = min((r for r in state_rows if r["horizons_seconds"]),key=lambda r:r["joint_deg"])
            original = read(BATCH / "rounds/J00016/result.json")["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]
            body += (f" 有记忆的三档中，联合留出误差最低的是 {best_state['id']}，为 {best_state['joint_deg']:.3f}°，"
                     f"既有轻量头为 {original:.3f}°。")
        source = HERE / "ema_state_replay.json"
        if source.is_file():
            evidence = read(source)
            body += (f" 审查从 {evidence['record']} 的原始时钟重建更新时刻。把 {evidence['forbidden_updates_perturbed']} 个禁用更新的输入改变后，"
                     f"合格训练时段的状态逐项相同。{evidence['weighted_history_comparisons']} 次直接对历史样本加权求和，也复现了指数状态。")
            sources.append("ema_state_replay.json")
        tables["fixed_states"] = {"rows":shown,"source":{"label":"固定衰减时序模型与同权重对照","files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"fixed-time-states","title":"短记忆与长记忆的开发集对照",
            "queryId":"fixed_states","body":body,"columns":[{"field":"id","label":"轮号"},{"field":"time","label":"时间常数（秒）"},
            {"field":"device","label":"留设备"},{"field":"motion","label":"留程序"},{"field":"joint","label":"联合留出"},{"field":"reset","label":"清零后联合留出"}]})
    joint = tally["joint_checks"]
    if joint:
        by_id = {r["id"]:r for r in checked}
        best_joint = min(joint, key=lambda r:r["rmse_deg"])
        parent = by_id[best_joint["candidate_id"]]
        variant = "去掉 TimesFM 特征的配对配置" if best_joint["without_embedding"] else ""
        separate = sum(r["separate"] for r in joint)
        body = (f"共有 {len(joint)} 个轮号取得同时留出设备和程序的成绩。其中 {separate} 轮为单列复核，"
                f"{len(joint)-separate} 轮在候选实验内完成了这项评分。最低单轮为 "
                f"{NAMES.get(parent['kind'],parent['kind'])} `{parent['id']}`{variant}，误差 {best_joint['rmse_deg']:.3f}°。"
                "每折的训练记录同时排除验证设备与验证程序，随后在 R1–R4 的 37 条记录上按序列等权汇总。")
        section = {"id":"joint-holdout","title":"同时遇到新设备与新程序时","queryId":"rounds","body":body}
        for family in tally["training_device_balance_families"].values():
            if parent["id"] in family["rounds"] and not family["qualified"]:
                section["body"] += (f" 该配置所属的训练设备加权方案在 {family['seeds']} 个匹配种子上的联合误差相对原轻量头变化 "
                                    f"{family['joint_delta_deg']:+.5f}°。预登记要求平均下降至少 {family['needed_improvement_deg']:.3f}°，这组未获 R5 入围资格。")
        matched = tally["matched_ridge_update_rates"]
        if matched:
            pairs = matched["pairs"]
            section["body"] += (f" 在相同的三个开发种子下，25 Hz 的 R01022 家族平均 {fmean(p['rmse_25'] for p in pairs):.4f}°，"
                                f"50 Hz 的 R01267 家族平均 {fmean(p['rmse_50'] for p in pairs):.4f}°。"
                                f"先在每条记录上对种子取平均，50 Hz 配置有 {matched['records_lower_at_50hz']}/{matched['records_per_seed']} 条误差较低。"
                                "两套配置的正则化强度和修正增益也不同，表中差值衡量整套设置的变化。")
            devices = matched["paired_by_group"]["device"]
            programs = [g["mean_50_minus_25_deg"] for g in matched["paired_by_group"]["program"].values()]
            section["body"] += (f" 50 Hz 减 25 Hz 的误差差值在 R1 为 {devices['R1']['mean_50_minus_25_deg']:+.4f}°"
                                f"（{devices['R1']['lower_at_50hz']}/{devices['R1']['records']} 条改善），"
                                f"在 R3 为 {devices['R3']['mean_50_minus_25_deg']:+.4f}°"
                                f"（{devices['R3']['lower_at_50hz']}/{devices['R3']['records']} 条改善）。"
                                f"按四组预定运动程序汇总，组内差值为 {min(programs):+.4f}° 至 {max(programs):+.4f}°。")
            shown = [{"seed":str(p["seed"]),"slow":f"{p['rmse_25']:.5f}°","fast":f"{p['rmse_50']:.5f}°",
                      "difference":f"{p['mean_difference_deg']:+.5f}°"} for p in pairs]
            files = ["round_audit.json","../ridge-25hz-50hz-joint-protocol.json","verify_ridge_seeds.py"]
            if matched["published_summary_reconciled"]:
                files.append("../ridge-25hz-50hz-joint-summary.json")
            tables["joint_rates"] = {"rows":shown,"source":{"label":"三个匹配种子的同时留出复核","files":files,"executedAt":tally["reviewed_utc"]}}
            section.update(queryId="joint_rates",columns=[{"field":f,"label":l} for f,l in
                           (("seed","开发种子"),("slow","25 Hz 配置"),("fast","50 Hz 配置"),("difference","50 Hz 减 25 Hz"))])
            mounting = tally["matched_ridge_installation"]
            if mounting:
                by_seed = {str(p["seed"]):p["axes"]["joint_holdout"] for p in mounting["pairs"]}
                for row in shown:
                    pair = by_seed[row["seed"]]
                    row["mounted"] = f"{pair['after_deg']:.5f}°"
                    row["mount_delta"] = f"{pair['difference_deg']:+.5f}°"
                section["columns"].extend([{"field":"mounted","label":"50 Hz 加安装旋转"},{"field":"mount_delta","label":"旋转带来的差值"}])
                values = mounting["axes"]
                outcome = "达到" if mounting["paired_evidence_gate"] else "未达到"
                section["body"] += (f" 在同一 50 Hz 配置上，只增加从各训练折估计的安装旋转，三个种子等权的联合误差由 "
                                    f"{values['joint_holdout']['before_deg']:.5f}° 变为 {values['joint_holdout']['after_deg']:.5f}°。"
                                    f"留设备变化 {values['device']['difference_deg']:+.5f}°，留程序变化 {values['motion']['difference_deg']:+.5f}°。"
                                    f"本次{outcome}预登记的配对门槛。每个训练折重新拟合安装旋转和岭回归，两项各 {mounting['fold_fits_per_stage']} 次。")
                grouped = values["joint_holdout"]["per_record_groups"]
                devs = grouped["groups"]
                section["body"] += (f" 三个种子的差值逐记录平均后，{grouped['wins']}/{mounting['records']} 条改善。"
                                    f"R4 改善 {devs['R4']['wins']}/{devs['R4']['records']} 条，R1 改善 {devs['R1']['wins']}/{devs['R1']['records']} 条。"
                                    f"若从同一批预测中去掉程序组 M2（程序 2、6、10），其余记录的平均差值为 {devs['M2']['mean_without_group_deg']:+.5f}°。")
                files.extend(["../ridge-50hz-mount-matched-protocol.json","ridge_50hz_mount_reference.json","verify_mount_pair.py"])
                if "published_sha256" in mounting:
                    files.append("../ridge-50hz-mount-matched-summary.json")
                fold_replay = HERE/"ridge_50hz_mount_fold_replay.json"
                if fold_replay.is_file():
                    from replay_mount_fold import check_saved as check_mount_replay
                    evidence = check_mount_replay(read(fold_replay))
                    files.extend([fold_replay.name,"replay_mount_fold.py"])
                    section["body"] += (f" 在种子 20261123 的 R2/M2 折，对三条原始记录的 {evidence['frames']:,} 个评分帧重放了两种配置。"
                                        f"正式评分入口和逐帧四元数换算与原成绩的最大差值为 {evidence['max_difference_deg']:.7f}°。")
        report["report"]["review"]["extraSections"].insert(0,section)
    alpha_trial = tally["mounted_alpha"]
    if alpha_trial:
        labels = {"device":"留设备","motion":"留程序","joint_holdout":"同时留设备与程序"}
        rows = []
        for group,name in (("search_seeds","搜索用种子"),("diagnostic_seeds","另行复验的种子")):
            for axis,s in alpha_trial["axes"][group].items():
                rows.append({"group":name,"axis":labels[axis],"seeds":len(s["seeds"]),
                             "low":f"{s['low_deg']:.5f}°","high":f"{s['high_deg']:.5f}°","delta":f"{s['delta_deg']:+.5f}°",
                             "record_wins":f"{s['paired_records']['wins']}/37"})
        sources = ["../ridge-50hz-mount-alpha-protocol.json","round_audit.json","verify_mounted_alpha.py"]
        body = ("在 50 Hz 岭回归上固定采样、安装旋转、修正增益、特征及网络宽度，只将正则化系数从 0.0018789 提至 0.0036222。"
                "表中每个种子分别比较两套配置在相同划分、相同记录上的误差，再对种子取平均。改善记录数按每条记录的种子平均差计算。"
                f"搜索用种子补做了 {alpha_trial['joint_fits']} 个设备与程序同时留出的训练折。"
                f"三个另行复验的种子共完成 {alpha_trial['diagnostic_fits']}/48 个训练折，每折都重新估计安装旋转和回归系数。"
                "种子复验只用开发数据，入围仍依原规则。")
        if "published" in alpha_trial:
            sources.append("../ridge-50hz-mount-alpha-summary.json")
            checked_alpha = alpha_trial["published"]
            body += (f" 已从各轮结果复核汇总中的 {checked_alpha['published_score_pairs']} 对逐记录成绩及评分帧指纹。"
                     f"在补测汇总时点，将这六个诊断轮从入围程序的输入中移除，预览名单中的 "
                     f"{checked_alpha['provisional_shortlist_count_at_completion']} 项及顺序均相同。")
        for group,label in (("search_seeds","搜索所用的"),("diagnostic_seeds","另行复验的")):
            stats = alpha_trial["axes"][group]
            if all(len(stats.get(a,{}).get("seeds",[])) == 3 for a in ("device","motion")):
                per_seed = [fmean(pair) for pair in zip(stats["device"]["seed_deltas_deg"],stats["motion"]["seed_deltas_deg"],strict=True)]
                change = fmean(per_seed)
                body += (f" 对{label}三个种子，将留设备与留程序的 RMSE 等权平均，"
                         f"新强度的误差{'降低' if change < 0 else '升高'} {abs(change):.5f}°，"
                         f"其中 {sum(v<0 for v in per_seed)}/3 个种子改善。")
        qr_path = HERE/"mounted_alpha_qr_replay.json"
        if qr_path.is_file():
            from replay_mounted_alpha import check_saved
            qr = check_saved(read(qr_path))
            sources.extend([qr_path.name,"replay_mounted_alpha.py"])
            body += (f" 独立审查在种子 20261123 的留 R1 折，用 {qr['training_updates']:,} 个训练更新通过 QR 分解重算岭回归。"
                     f"提高正则化后，除截距外的系数范数从 {qr['norms'][0]:.5f} 降到 {qr['norms'][1]:.5f}。"
                     f"重放全部 {qr['records']} 条 R1 原始记录的 {qr['total_frames']:,} 个评分帧，"
                     f"两种配置所得的 {qr['score_comparisons']} 个 RMSE 与原轮最大相差 {qr['max_abs_difference_deg']:.7f}°。")
        tables["mounted_alpha"] = {"rows":rows,"source":{"label":"两档正则化强度在搜索与复验种子上的逐记录成绩",
            "files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"mounted-alpha","title":"提高 50 Hz 岭回归的正则化强度后",
            "queryId":"mounted_alpha","body":body,"columns":[{"field":k,"label":s} for k,s in
            (("group","种子来源"),("axis","验证方式"),("seeds","已配对种子"),("low","原强度"),("high","新强度"),("delta","新减原"),("record_wins","改善记录"))]})
    for trial in tally["sobol_checks"]:
        axes = {"device":"留设备","motion":"留程序","joint_holdout":"同时留设备与程序"}
        display = []
        for axis,label in axes.items():
            for alpha,result in trial["axes"][axis].items():
                paired = result["search"]
                display.append({"axis":label,"alpha":alpha,"fixed":f"{result['fixed_deg']:.5f}°",
                                "search":f"{paired['reference_mean_deg']:.5f}°","delta":f"{paired['fixed_minus_reference_deg']:+.5f}°",
                                "records":f"{paired['records_improved']}/37"})
        sources = ["../mounted-sobol-ridge-protocol.json","../rounds/"+trial["id"]+"/result.json","../mounted-sobol-fixed-tanh.npz",
                   "verify_sobol.py","round_audit.json"]
        story = ("在 50 Hz 安装旋转岭回归上，用同一组 128 个 Sobol 点固定非线性特征，其余设置与随机投影对照相同。"
                 "审查用 PyTorch 重新生成点集，用 Python 标准库计算正态分位数，与实验保存的 8,448 个投影权重、128 个偏移逐项相等。"
                 "每个正则化系数只产生一组确定性成绩。表中对照是同档正则化的三个搜索种子，逐记录取平均。"
                 f"每种划分均有 {trial['records_per_axis']} 条记录和 {trial['raw_frames_per_axis']:,} 个共同评分帧。"
                 "两档固定投影都改善了留程序均值，在留设备和同时留出两者时误差均上升。")
        arms = sorted(trial["axes"]["motion"],key=float)
        motion_group = [trial["axes"]["motion"][k]["search"]["by_group"]["M2"] for k in arms]
        devices = [trial["axes"]["device"][k]["search"]["by_group"] for k in arms]
        remaining = trial["records_per_axis"]-motion_group[0]["records"]
        story += (f" 留程序的收益集中在第 2 组（程序 2、6、10）。两档正则化在这一组分别有 "
                  f"{motion_group[0]['wins']}/{motion_group[0]['records']} 和 {motion_group[1]['wins']}/{motion_group[1]['records']} 条改善。"
                  f"其余 {remaining} 条的平均变化分别为 {motion_group[0]['mean_without_group_deg']:+.5f}° 和 "
                  f"{motion_group[1]['mean_without_group_deg']:+.5f}°。单独留出设备时，R1 与 R2 共 "
                  f"{devices[0]['R1']['records']+devices[0]['R2']['records']} 条记录，两档强度分别有 "
                  f"{devices[0]['R1']['wins']+devices[0]['R2']['wins']} 条和 {devices[1]['R1']['wins']+devices[1]['R2']['wins']} 条改善。")
        if "published" in trial:
            sources.append("../mounted-sobol-ridge-summary.json")
            proof = trial["published"]
            story += (f" 已回查批次汇总中的 {proof['score_pairs']} 对逐记录成绩。"
                      f"用汇总时已取得的开发结果重建候选预览，移除这个诊断轮后 {proof['preview_size']} 个条目与顺序一致。")
        raw_path = HERE/"sobol_raw_fold_replay.json"
        if raw_path.is_file():
            from verify_sobol import check_replay
            raw = check_replay(read(raw_path))
            sources.append(raw_path.name)
            story += (f" 在同时留出 R2 和第 2 组运动程序的一折，用加权 QR 分解独立重算两档系数，并重放 "
                      f"{raw['records']} 条原始记录的 {raw['total_frames']:,} 个评分帧。"
                      f"{raw['comparisons']} 个逐记录 RMSE 与实验原值最大相差 {raw['max_abs_degree_gap']:.7f}°。")
        tables["fixed_sobol"] = {"rows":display,"source":{"label":"固定和随机投影在相同开发折和原生评分帧上的比较",
                                "files":sources,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"fixed-sobol","title":"固定非线性特征后的留出误差",
            "body":story,"queryId":"fixed_sobol","columns":[{"field":k,"label":s} for k,s in
            (("axis","验证方式"),("alpha","正则化系数"),("fixed","固定投影"),("search","搜索种子均值"),("delta","固定减随机"),("records","改善记录"))]})
    for ensemble in tally["independent_filter_ensembles"]:
        first, second = ensemble["members"]
        display = []
        for axis, label in (("device","留设备"),("motion","留运动程序"),("joint_holdout","同时留设备与程序")):
            scores = ensemble["scores"][axis]
            display.append({"axis":label,"first":f"{scores['member_degrees'][first]:.5f}°",
                            "second":f"{scores['member_degrees'][second]:.5f}°","pair":f"{scores['ensemble_deg']:.5f}°",
                            "change":f"{scores['versus_better_member_deg']:+.5f}°"})
        ensemble_joint = ensemble["scores"]["joint_holdout"]
        query = "mixed_filter_" + ensemble["id"]
        tables[query] = {"rows":display,"source":{"label":"两套独立滤波器的同帧逐记录对照",
            "files":["../independent-filter-ensemble-development-protocol.json","../rounds/"+ensemble["id"]+"/result.json",
                     "verify_mixed_ensemble.py","round_audit.json"],"executedAt":tally["reviewed_utc"]}}
        description = (f"轻量头 {first} 以 10 Hz 更新，岭回归 {second} 以 50 Hz 更新。逐帧在传感器坐标系中等权合成两者的竖直方向估计。"
                       f"24 折共做 {ensemble['member_fits']} 次成员拟合，每个验证方向均覆盖原有 {ensemble['records_per_axis']} 条记录。"
                       f"联合留出时，两者在参考方向切平面上的误差向量余弦逐记录平均为 {ensemble_joint['correlations']['tangent_error_cosine']:.3f}，1 表示完全同向。")
        if not ensemble["accuracy_gate_passed"]:
            description += " 组合未满足入围标准，双滤波器的部署延迟未测。"
        report["report"]["review"]["extraSections"].append({"id":"mixed-filter-"+ensemble["id"],"title":"物理轻量头和岭回归合用的倾角误差",
            "queryId":query,"body":description,"columns":[{"field":f,"label":l} for f,l in (("axis","开发划分"),("first",first+" 轻量头"),
                ("second",second+" 岭回归"),("pair","等权组合"),("change","组合减去较优成员"))]})
    for ensemble in tally["shared_seed_ensembles"]:
        members = ensemble["members"]
        rows = []
        for axis, label in (("device","留设备"),("motion","留运动程序"),("joint_holdout","同时留设备与程序")):
            values = ensemble["scores"][axis]
            rows.append({"axis":label,**{k:f"{values['member_degrees'][k]:.5f}°" for k in members},
                         "together":f"{values['ensemble_deg']:.5f}°","difference":f"{values['versus_better_member_deg']:+.5f}°"})
        query = "shared_heads_"+ensemble["id"]
        sources = ["../three-seed-ensemble-development-protocol.json","../rounds/"+ensemble["id"]+"/result.json","verify_mixed_ensemble.py"]
        tables[query] = {"rows":rows,"source":{"label":"固定三个种子的轻量头和逐帧等权组合","files":sources,"executedAt":tally["reviewed_utc"]}}
        joint_cosine = ensemble["scores"]["joint_holdout"]["correlations"]["tangent_error_cosine"]
        body = (f"三个轻量头共用一个 FullVQF 和同一折内估计的安装旋转，模型配置相同，只改变训练初始化。"
                f"共核对 {ensemble['member_fits']} 次折内拟合，以及三种划分中各 {ensemble['records_per_axis']} 条记录的原生评分帧。"
                f"联合留出时，三个成员两两相比的误差方向余弦平均为 {joint_cosine:.3f}，1 表示完全同向。")
        if not ensemble["accuracy_gate_passed"]:
            body += " 相对表现最好的单头，组合未满足进入 R5 的精度要求。"
        report["report"]["review"]["extraSections"].append({"id":"seed-ensemble-"+ensemble["id"],"title":"同一个轻量头用三个种子训练后合用的效果",
            "queryId":query,"body":body,"columns":[{"field":"axis","label":"开发划分"},*[{"field":k,"label":k} for k in members],
            {"field":"together","label":"三者等权合用"},{"field":"difference","label":"组合减去最佳单头"}]})
    for trial in tally["dct_checks"]:
        shown = [{"axis":label,**{k:f"{v:.5f}°" for k,v in trial["scores"][axis].items()}} for axis,label in
                 (("device","留设备"),("motion","留程序"),("joint_holdout","同时留设备与程序"))]
        files = ["../dct-development-protocol.json","../rounds/"+trial["id"]+"/result.json","verify_dct.py"]
        body = (f"每次更新只取过去 0.64 秒的六轴数据。每轴提取 4 个低阶离散余弦系数，再用岭回归修正倾角。"
                f"相同的 {trial['windows']:,} 个更新窗口分别保留原顺序、固定打乱时间顺序，或仅使用物理特征。"
                f"三种读出共拟合 {trial['fits']} 次，使用相同的 {trial['rotations']} 个训练折安装旋转。"
                "扩试条件要求联合误差比两种岭对照和原轻量头各低至少 0.020°，单独留设备和留程序时相对原轻量头最多高 0.020°。"
                + ("这次达到扩试条件。" if trial["passed"] else "这次未达到扩试条件。"))
        if trial.get("raw_windows"):
            body += f" 从两条原始记录独立重建 {trial['raw_windows']:,} 个窗口，核对了每个窗口的时间方向、发布时刻和两套非零系数。"
            files.append("dct_native_feature_replay.json")
        key = "dct_"+trial["id"]
        tables[key] = {"rows":shown,"source":{"label":"同窗、同帧的三个岭读出与原轻量头","files":files,"executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"dct-"+trial["id"],"title":"保留 0.64 秒波形的时间顺序是否有收益",
            "queryId":key,"body":body,"columns":[{"field":k,"label":label} for k,label in
                (("axis","留出方式"),("ordered_dct","原时间顺序"),("permuted_dct","打乱时间顺序"),
                 ("physical_zero","只用物理特征"),("legacy_head","原轻量头 P00022"))]})
    for trial in tally["reservoir_checks"]:
        shown = []
        for axis, label in (("device","留设备"),("motion","留运动程序"),("joint_holdout","同时留设备与程序")):
            values = trial["scores"][axis]
            shown.append({"axis":label,**{k:f"{v:.5f}°" for k,v in values.items()}})
        query = "reservoir_" + trial["id"]
        tables[query] = {"rows":shown,"source":{"label":"储备池、无记忆对照和同种子轻量头的逐序列成绩",
            "files":["../reservoir-development-protocol.json","../rounds/"+trial["id"]+"/result.json","verify_reservoir.py"],
            "executedAt":tally["reviewed_utc"]}}
        body = (f"两个读出都用相同的 34 个物理输入、32 个随机状态特征和训练折安装旋转。一个保留上次状态，另一个在每次更新前清零。"
                f"审查从种子 {trial['seed']} 重新生成固定矩阵，核对 {trial['member_fits']} 次拟合和每种划分的 {trial['records']} 条记录。")
        if not trial["passed"]:
            body += f" 延续状态的模型未达到预定复跑门槛。同种子轻量头 {trial['head']} 为已有对照。"
        report["report"]["review"]["extraSections"].append({"id":"reservoir-"+trial["id"],"title":"保留上一次状态能否改善倾角估计",
            "queryId":query,"body":body,"columns":[{"field":f,"label":label} for f,label in
            (("axis","开发划分"),("continuous","保留状态"),("reset_each_update","每次清零"),("legacy_head","已有轻量头"))]})
    optimizer_plan = BATCH / "optimizer-development-stage.json"
    if optimizer_plan.is_file():
        complete = {r["id"]:r for r in checked}
        duration_tests = [e for e in read(optimizer_plan)["first_screen"] if e["kind"] == "legacy_head"
                          and set(e["changed"]) == {"epochs"}
                          and all(e[k] in complete for k in ("round_id","anchor","timed_anchor_replay"))]
        if duration_tests:
            experiment = max(duration_tests,key=lambda e:e["changed"]["epochs"])
            anchor_id, test_id = experiment["anchor"], experiment["round_id"]
            before = read(BATCH / "rounds" / anchor_id / "round.json")
            after = read(BATCH / "rounds" / test_id / "round.json")
            assert {**before["config"],**experiment["changed"]} == after["config"]
            a, b = before["dev_metrics"], after["dev_metrics"]
            wall_a = read(BATCH / complete[experiment["timed_anchor_replay"]]["result"])["timing"]["fit_seconds"]
            wall_b = read(BATCH / complete[test_id]["result"])["timing"]["fit_seconds"]
            body = (f"以原轻量头 `{anchor_id}` 和 `{test_id}` 为例，保持同一随机种子、学习率与输入特征，"
                    f"把训练时长从 {before['config']['epochs']} 个 epoch 改为 {after['config']['epochs']} 个。"
                    f"留程序 RMSE 从 {a['motion']['macro_rmse_deg']:.3f}° 变为 {b['motion']['macro_rmse_deg']:.3f}°，"
                    f"留设备从 {a['device']['macro_rmse_deg']:.3f}° 变为 {b['device']['macro_rmse_deg']:.3f}°。"
                    f"同机各折合计的模型拟合时间分别为 {wall_a:.2f} 秒和 {wall_b:.2f} 秒。")
            same = {j["candidate_id"]:j["rmse_deg"] for j in joint if not j["without_embedding"]}
            if anchor_id in same and test_id in same:
                body += f" 在同一折同时留出设备和程序时，误差从 {same[anchor_id]:.3f}° 变为 {same[test_id]:.3f}°。"
            report["report"]["review"]["extraSections"].append({"id":"training-duration","title":"训练时长对新程序和新设备的影响","queryId":"rounds","body":body})
    dev_export_file = HERE / "development_quaternion_audit.json"
    if any(r["id"] == "A00001" for r in checked):
        result = read(BATCH / "rounds/A00001/result.json")
        arms = result["summary"]["joint_holdout"]
        count = result["eligible_development_records"]
        active = result["online_records_with_support"]
        scores = [{"axis":axis,"method":m,"macro_rmse_deg":v["macro_rmse_deg"]} for axis,methods in result["summary"].items() for m,v in methods.items()]
        tables["accelerometer_bias"] = {"rows":scores,"source":{"label":"加速度计零偏开发集对照",
            "files":["../rounds/A00001/result.json","../accelerometer-bias-development-stage.json","development_accelerometer_spheres.json"],
            "executedAt":result["completed_at"]}}
        body = (f"保持基础滤波器、安装校准和评分帧一致，在 {count} 条开发记录上比较三种方案。"
                f"同时留出设备与程序时，零补偿为 {arms['zero']['macro_rmse_deg']:.3f}°，"
                f"从训练折静止姿态拟合的固定偏置为 {arms['fixed_fold']['macro_rmse_deg']:.3f}°。"
                f"从每条记录自己的过去数据在线估计，只有 {active}/{count} 条真正启用了补偿，整体为 {arms['online_shrunk']['macro_rmse_deg']:.3f}°。")
        if active == 0:
            body += " 本次在线方案全程退回零补偿，未取得启用后的误差数据。"
        report["report"]["review"]["extraSections"].append({"id":"accelerometer-bias","title":"静止姿态能否校正加速度计零偏",
            "queryId":"accelerometer_bias","body":body})
    if any(r["id"] == "A00002" for r in checked):
        source = BATCH / "rounds/A00002/result.json"
        result = read(source)
        metrics = result["summary"]["joint_holdout"]
        per_record = []
        for r in result["sequences"]["joint_holdout"]:
            obs = result["online_audit"][r["name"]]
            if obs["first_applied_seconds"] is None:
                continue
            base, applied = r["rmse_deg"]["zero"],r["rmse_deg"]["online_shrunk"]
            per_record.append({"name":r["name"],"first":f"{obs['first_applied_seconds']:.1f}",
                               "frames":f"{obs['scored_frames_with_nonzero_estimate']:,}","base":f"{base:.3f}°",
                               "online":f"{applied:.3f}°","change":f"{applied-base:+.3f}°"})
        total_frames = sum(r["scored_frames_with_nonzero_estimate"] for r in result["online_audit"].values())
        share = total_frames/metrics["zero"]["scoring_frames"]
        body = (f"在六姿态试验之后，单列降低姿态数门槛的开发集复核，其余条件不变。要求至少五个姿态时，"
                f"{result['online_records_with_support']}/{result['eligible_development_records']} 条记录启用，"
                f"覆盖 {total_frames:,} 个评分帧（{share:.1%}）。联合留出误差从 "
                f"{metrics['zero']['macro_rmse_deg']:.3f}° 降至 {metrics['online_shrunk']['macro_rmse_deg']:.3f}°。"
                "四台设备都只在转台程序 10 上启用，所以这些结果只覆盖一种运动程序。")
        files = ["../rounds/A00002/result.json","../accelerometer-bias-five-pose-sensitivity.json"]
        proof_path = HERE / "accelerometer_online_replay.json"
        if proof_path.is_file():
            proof = read(proof_path)
            with source.open("rb") as stream:
                assert proof["round_result_sha256"] == hashlib.file_digest(stream,"sha256").hexdigest()
            assert {r["name"] for r in proof["records"] if r["active_scored_frames"]} == {r["name"] for r in per_record}
            body += (f" 独立读取 {len(proof['records'])} 条原始 CSV，用各时间点以前的静止样本重建零偏和滤波输出。"
                     "下表的差值以整条记录的相同评分帧计算，包含启用前的时段。")
            files.extend(["accelerometer_online_replay.json","inspect_accelerometers.py"])
        if any(r["id"] == "A00003" for r in checked):
            matched = read(BATCH / "rounds/A00003/result.json")
            grouped = [(run["metrics"]["joint_holdout"],run["metrics"]["device"]) for run in matched["results"]]
            parts = []
            for method,label in (("frozen_head","沿用原权重"),("retrained_head","同种子重新拟合")):
                changes = [j[method]["macro_rmse_deg"]-j["zero"]["macro_rmse_deg"] for j,_ in grouped]
                parts.append(f"{label}时平均变化 {fmean(changes):+.4f}°，{sum(v<0 for v in changes)}/{len(changes)} 个种子改善")
            body += " 在轻量修正头上另做三种子的同帧配对，联合留出结果为" + "，".join(parts) + "。"
            files.append("../rounds/A00003/result.json")
        if any(r["id"] == "L00008" for r in checked):
            from inspect_accelerometers import benchmark_scope
            from verify_final import raw_shice_samples

            latency = read(BATCH / "rounds/L00008/result.json")
            name = latency["protocol"]["benchmark_record"]
            assert name in {r["name"] for r in per_record}
            raw = raw_shice_samples(name)
            scopes = {}
            for candidate in latency["benchmarks"]:
                if candidate["kind"] != "bias_physical_head":
                    continue
                identity = candidate["development_round"]
                entry = read(BATCH / "rounds" / identity / "round.json")
                assert entry["config_sha256"] == candidate["config_sha256"]
                timing = candidate["measurements"]
                assert timing["record"] == name
                scope = benchmark_scope(raw, timing, entry["config"], result["fit_protocol"], latency["protocol"]["native_updates"])
                scopes[identity] = {**scope, "block_p95_ms": timing["online_bias_block_update"]["p95_ms"]}
            if scopes:
                write(HERE / "bias_latency_scope.json", {"timing_round": "L00008", "by_candidate": scopes})
                parts = [f"{key} 有 {s['frames_with_active_estimate']:,}/{s['measured_native_frames']:,} 帧处于校准启用时段，"
                         f"其中 {s['observed_packets_with_bias']:,} 帧收到新数据，"
                         f"{s['measured_blocks']} 个计时块中有 {len(s['accepted_refits_after_samples'])} 次重新估计成功，"
                         f"按块处理 P95 为 {s['block_p95_ms']:.3f} ms" for key,s in scopes.items()]
                body += f" 在 {name} 上逐帧重放了同机计时区间。" + "。".join(parts) + "。"
                files.extend(["../rounds/L00008/result.json", "bias_latency_scope.json"])
        tables["bias_active"] = {"rows":per_record,"source":{"label":"五姿态在线零偏实际覆盖","files":files,
                                                             "executedAt":tally["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"bias-active","title":"降低姿态门槛后的四条记录",
            "queryId":"bias_active","body":body,"columns":[{"field":"name","label":"记录"},{"field":"first","label":"首次启用（秒）"},
            {"field":"frames","label":"启用的评分帧"},{"field":"base","label":"零补偿"},{"field":"online","label":"在线补偿"},{"field":"change","label":"差值"}]})
    recurrent_proof = HERE / "recurrent_context_audit.json"
    if recurrent_proof.is_file() and "C00011" in {r["id"] for r in checked if r["validated_development_candidate"]}:
        raw = read(recurrent_proof)
        intervention = raw.get("independent_intervention")
        if intervention:
            model = read(BATCH / "rounds/C00011/result.json")
            state_rows = [{"axis":axis,**{mode:score["macro_rmse_deg"] for mode,score in modes.items()}}
                          for axis,modes in model["state_controls"].items()]
            tables["recurrent_context"] = {"rows":state_rows,"source":{"label":"分段训练与同权重状态对照",
                "files":["recurrent_context_audit.json","verify_recurrent_context.py","../rounds/C00011/result.json"],
                "executedAt":raw["reviewed_utc"]}}
            controls = {r["axis"]:r for r in state_rows}
            motion, joint_state = controls["motion"], controls["joint_holdout"]
            body = (f"37 条开发记录共有 {raw['eligible_updates']:,} 个合格训练标签，"
                    f"其中 {raw['affected_records']} 条在标签之间出现过禁用输入。对 {intervention['record']} 的 "
                    f"{intervention['forbidden_updates_perturbed']} 个禁用更新单独更改输入，修订后的训练张量、标签、权重及归一化结果逐项一致。"
                    "反向传播检查显示，下一训练段的隐状态对段前输入的梯度为零。"
                    f"修订后的 `C00011` 使用同一组权重，逐更新清零和连续保留状态的留程序误差分别为 "
                    f"{motion['reset_each_update']:.3f}° 与 {motion['continuous']:.3f}°，"
                    f"同时留出设备和程序时分别为 {joint_state['reset_each_update']:.3f}° 与 {joint_state['continuous']:.3f}°。")
            from verify_recurrent_context import review_state_gains
            screened = review_state_gains({r["id"] for r in checked if r["validated_development_candidate"]})
            if screened:
                strongest = min(screened,key=lambda r:r["joint"])
                body += (f" 已核对 {len(screened)} 个修订后的循环模型轮号，按预设条件有 {sum(r['qualifies'] for r in screened)} 个"
                         f"满足换种子复验门槛。联合留出最低的是 `{strongest['id']}`，误差 {strongest['joint']:.3f}°，"
                         f"同种子的原轻量头 `{strongest['control_id']}` 为 {strongest['same_seed_control']:.3f}°。")
                tables["recurrent_context"]["source"]["files"].append("../stream-gru-development-summary.json")
            report["report"]["review"]["extraSections"].append({"id":"recurrent-context","title":"循环记忆在什么条件下有帮助",
                "queryId":"recurrent_context","body":body})
    if dev_export_file.is_file():
        exports = read(dev_export_file)
        completed_ids = {r["id"] for r in checked}
        verified = [r for r in exports["outputs"] if r["round_id"] in completed_ids]
        if verified:
            tables["quaternion_outputs"] = {"rows":verified,"source":{"label":"从逐帧输出和原始 CSV 复算倾角",
                "files":["development_quaternion_audit.json","../quaternion-output-stage.json","verify_final.py"],
                "executedAt":exports.get("reviewed_utc",tally["reviewed_utc"])}}
            if "final" not in tally["completed_by_kind"]:
                methods = len({r["candidate"] for r in verified})
                records = sorted({r["name"] for r in verified})
                total = sum(r["native_frames"] for r in verified)
                largest = max(r["absolute_delta_vs_direction_deg"] for r in verified)
                body = (f"从 {'、'.join(records)} 的原始 CSV 重新核对时间戳和编码器方向，并读取 {methods} 个候选导出的 "
                        f"{len(verified)} 份 wxyz 四元数文件，共 {total:,} 个输出。与开发轮直接按竖直方向计算的逐条倾角 RMSE "
                        f"最大相差 {largest:.7f}°。{sum(r['protocol_passed'] for r in verified)}/{len(verified)} 份通过了单位范数、倾角与前缀一致性检验，"
                        "绝对航向精度未测。")
                report["report"]["review"]["extraSections"].append({"id":"quaternion-output","title":"逐帧姿态输出复核",
                    "queryId":"quaternion_outputs","body":body})
    clock_receipt = HERE/"development_reference_timing.json"
    if clock_receipt.is_file() and dev_export_file.is_file():
        from inspect_reference_timing import check_saved as check_timing
        clock = read(clock_receipt)
        check_timing(clock,read(dev_export_file)["outputs"])
        rows = []
        for name,item in clock["records"].items():
            summary = item["summary"]
            display = {"name":name,"frames":f"{item['common_frames']:,} / {item['original_frames']:,}"}
            for offset in ("earlier","later"):
                low,high = (summary[offset][k] for k in ("min_change_deg","max_change_deg"))
                display[offset] = f"{low:+.5f}° 至 {high:+.5f}°"
            display["flips"] = f"{summary['earlier']['reversed_pairs']} / {summary['later']['reversed_pairs']}"
            rows.append(display)
        tables["reference_timing"] = {"rows":rows,"source":{"label":"两条开发记录的原始编码器时间轴与九个候选的四元数",
            "files":["development_reference_timing.json","inspect_reference_timing.py","development_quaternion_audit.json"],"executedAt":clock["reviewed_utc"]}}
        report["report"]["review"]["extraSections"].append({"id":"reference-timing","title":"编码器参考移动一个原始采样时刻的影响",
            "queryId":"reference_timing","body":"使用已导出姿态的两条开发记录和九个候选。固定每一帧的预测，分别与上一帧、本帧、下一帧的编码器参考比较。"
            "三种情况只取都能评分的连续帧，间隔约 1 ms。表内为相对本帧参考的倾角 RMSE 变化范围。这个扰动只衡量这两条记录对设定偏移的敏感度，实际同步误差仍未知。",
            "columns":[{"field":k,"label":v} for k,v in (("name","开发记录"),("frames","共同帧 / 原评分帧"),
                ("earlier","用前一帧参考"),("later","用后一帧参考"),("flips","候选顺序反转对数（前 / 后）"))]})
    proposal_file = HERE/"proposal_replay.json"
    if "final" in tally["completed_by_kind"]:
        from verify_proposals import audit as audit_proposals
        write(proposal_file,audit_proposals(tally["reviewed_utc"]))
    if proposal_file.is_file():
        from verify_baselines import digest
        proposal = read(proposal_file)
        assert all(digest(BATCH/name) == sha for name,sha in proposal["source_sha256"].items())
        assert proposal["matches"] == sum(r["matches"] for r in proposal["results"]) and proposal["proposals"] == len(proposal["results"])
        concurrent = len(proposal.get("concurrent_registrations",[]))
        verification = next(s for s in report["report"]["review"]["extraSections"] if s["id"] == "verification")
        verification["body"] += (f" 截至 {datetime.fromisoformat(proposal['through']).astimezone(BEIJING):%H:%M:%S}，"
            f"用账本中当时已有的开发分数、配置和任务登记信息重放参数提议与种子复跑。"
            f"{proposal['matches']:,}/{proposal['proposals']:,} 次得到了相同的配置、来源轮号与调整理由。"
            "重放时禁止策略打开数据文件。")
        if concurrent:
            verification["body"] += f" 其中 {concurrent} 次与其他任务登记几乎同时发生，采用上一轮完成时已登记的清单才复现。"
        historical_source = tables["historical"]["source"]
        historical_source["files"] = list(dict.fromkeys([*historical_source["files"],"proposal_replay.json","verify_proposals.py"]))
    if "final" in tally["completed_by_kind"]:
        from final_narrative import finish
        report = finish(report, tally)
    else:
        status = read(BATCH/"status.json")
        if datetime.now(timezone.utc) >= datetime.fromisoformat(status["window_end"]):
            from partial_report import finish_partial
            report = finish_partial(report,tally,status)
        elif weights["filed"]:
            from partial_report import show_live_selection
            show_live_selection(report,weights,development_summary)
    for study in ("transfer-mamba-20261123", "all-datasets-8h-20261123"):
        chapter = BATCH.parent / study / "review/report-addendum.json"
        if not chapter.is_file():
            continue
        addendum = read(chapter)
        report["queries"].update(addendum["queries"])
        sections = report["report"]["review"]["extraSections"]
        ids = {section["id"] for section in addendum["sections"]}
        sections[:] = [*addendum["sections"], *(section for section in sections if section["id"] not in ids)]
        if study=="all-datasets-8h-20261123" and "all8-final-cross-source" in ids:
            latest=BATCH.parent/study/"review"
            timeline=read(latest/"delivery-timing.json")
            totals=read(latest/"campaign-counts.json")
            verification=read(latest/"final-verification.json")
            selected=read(latest.parent/"selection-lock.json")["selected"]["batch"]
            newest=next(s for s in sections if s["id"]=="all8-final-cross-source")
            control_rows=report["queries"]["all8-final-vqf-controls"]["rows"]
            strict=[r["c0"] for r in control_rows if r["c6"]=="3/3"]
            shice_fast=next(r["c2"] for r in control_rows if r["c0"]=="SHICE")
            own=verification["main"]["SHICE"]
            own_text=(f"自采 R5/R6 的 {own['scored']} 条留出记录上，默认 FullVQF 为 {own['means']['full_vqf_deg']:.5f}°，"
                      f"冻结模型为 {own['means']['candidate_deg']:.5f}°。") if own['scored'] else "自采 R5/R6 的覆盖缺额见主表。"
            story=report["report"]["review"]
            old_summary=story["summary"].removeprefix("## 实验结论\n\n")
            story["byline"]=f"FZR-MiniPC · 科学核验覆盖北京时间 {timeline['window_start'][:10]} {timeline['window_start'][11:19]} 至 {timeline['science_completed_at'][:10]} {timeline['science_completed_at'][11:19]} · 八小时授权窗口至 {timeline['window_end'][11:19]}"
            story["summary"]=(f"## 本轮八小时实验\n\n共完成 **{totals['completed_named_batches']} 个小批、{totals['new_or_inherited_candidate_policies_tested']} 种不同候选配置**。"
                              f"统一锁定 {selected} 后，八套主要数据中有 **{verification['main_improved_domains']}/8 套**的逐记录平均倾角误差低于默认 FullVQF。{own_text}"
                              f"\n\n{newest['body']} 若要求同时赢过三种预设 VQF，仅 {len(strict)}/{len(control_rows)} 套通过（{'、'.join(strict) or '无'}）。自采的 1.5 秒 VQF 为 {shice_fast}。")
            story["counts"]=old_summary+"\n\n"+story["counts"]
    write(DATA, report)
    print(json.dumps({"through": time_label, "completed": len(checked), "candidate_runs": n,
                      "distinct_candidate_settings": distinct}, ensure_ascii=False))


if __name__ == "__main__":
    refresh()
