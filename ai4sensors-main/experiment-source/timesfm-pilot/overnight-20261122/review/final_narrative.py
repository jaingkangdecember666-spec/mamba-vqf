"""Compose the overnight report only from the confirmed lock and completed evidence."""

from datetime import datetime, timedelta
from statistics import fmean, median

from audit_rounds import BATCH, HERE, parameter_identity, read, write
from verify_final import audit_final


METHODS = {"locked":"本批锁定配置", "full_vqf":"纯 FullVQF", "physical_only":"原轻量修正",
           "timesfm":"原 TimesFM 修正", "gru":"原两层 GRU"}


def table_query(rows, label, sources, detail):
    return {"rows":rows,"source":{"label":label,"files":sources,"evidenceFlow":[{"title":"复核方法","detail":detail}]}}


def add_dataset_partitions(report, selection, proof, tally):
    import csv
    from verify_baselines import SHICE
    from audit_rounds import check_fold

    dev = set(selection["training"])
    r5 = set(selection["validation"])
    r6 = set(selection["held_out"]["R6"])
    assert dev.isdisjoint(r5 | r6) and r5.isdisjoint(r6)
    accepted = dev | r5 | r6
    quality_file = SHICE/"Label_quality_audit_20261116"/"recording_recommendations.csv"
    with quality_file.open(encoding="utf-8-sig", newline="") as f:
        inventory = {row["recording"]:row for row in csv.DictReader(f)}
    raw_files = {p.stem for p in SHICE.glob("R*_aligned_formal/R*-*.csv")}
    assert set(inventory) == raw_files
    excluded = sorted(raw_files - accepted)
    assert accepted <= raw_files and excluded == ["R1-3", "R1-4"]
    assert all("A2_independent_clock_requires_external_confirmation" in inventory[n]["review_reasons"] for n in excluded)
    devices = {row["sensor"] for row in inventory.values()}
    programs = {row["run"] for row in inventory.values()}
    assert {f"{device}-{program}" for device in devices for program in programs} - raw_files == {"R1-2"}

    dev_path = f"rounds/{proof['selected_id']}/result.json"
    development = read(BATCH/dev_path)
    fold_rows = []
    for fold in development["folds"]:
        held = check_fold(fold, fold["axis"], dev)
        numbers = sorted({int(name.split("-")[1]) for name in held})
        group = ("设备 " + fold["fold"] if fold["axis"] == "device"
                 else "程序 " + "、".join(map(str,numbers)))
        fold_rows.append({"fold":fold["fold"],"group":group,"training":str(len(fold["training"])),"validation":str(len(held))})
    for axis in ("device","motion"):
        rows = development["sequences"][axis]
        assert {row["name"] for row in rows} == dev and len(rows) == len(dev)
    by_record = {row["name"]:row["frames"] for row in development["sequences"]["device"]}
    assert by_record == {row["name"]:row["frames"] for row in development["sequences"]["motion"]}
    tests = proof["tests"]
    final_sequences = read(BATCH/"rounds"/proof["final_id"]/"result.json")["datasets"]
    broad_ids = {row["name"].split("_",1)[0] for row in final_sequences["BROAD"]["sequences"]}
    assert broad_ids == {f"{i:02}" for i in range(1,40)}
    sassari = {row["name"] for row in final_sequences["Sassari"]["sequences"]}
    sensors = sorted({name.split("/")[1] for name in sassari})
    assert len(sensors) == 6 and sassari == {f"{speed}/{sensor}" for sensor in sensors for speed in ("slow","medium","fast")}
    r5_scores = selection["selected"]["r5"]
    assert r5_scores["records"] == len(r5) and tests["R6"]["records"] == len(r6)
    suites = [
        ("开发与训练","自采 R1–R4",len(dev),sum(by_record.values())),
        ("选模验证","自采 R5，程序 1–10",len(r5),r5_scores["scoring_frames"]),
        ("自采终测","自采 R6，程序 1–10",len(r6),tests["R6"]["scoring_frames"]),
        ("外部复测","BROAD 序列 01–39",tests["BROAD"]["records"],tests["BROAD"]["scoring_frames"]),
        ("外部复测","Sassari，慢、中、快三次运动各 6 路",tests["Sassari"]["records"],tests["Sassari"]["scoring_frames"]),
    ]
    candidates = len(selection["r5_candidates"])
    body = (f"自采 SHICE 按整条转台记录划分。R1–R6 是六台设备，连字符后是运动程序编号，例如 R2-6 表示第二台设备的第 6 号程序。"
            f"正式目录有 {len(raw_files)} 条记录，其中缺少 R1-2。{'、'.join(excluded)} 因内框编码器 A2 的独立时钟对齐待核查而排除，"
            f"共纳入 {len(accepted)} 条。按记录数，开发、选模、自采终测占比分别约为 {len(dev)/len(accepted):.1%}、"
            f"{len(r5)/len(accepted):.1%}、{len(r6)/len(accepted):.1%}。同一条记录的全部帧和历史窗口随记录归组。\n\n"
            f"候选根据 R1–R4 的开发结果入围。需要在本批学习的参数用全部 {len(dev)} 条开发记录拟合，"
            f"随后用 R5 的 {len(r5)} 条记录对 {candidates} 项候选评分并锁定权重。R5 的标签只用于选模。"
            "R6 检验另一台自采设备。R6、BROAD、Sassari 在以往实验中已有成绩，本批锁定模型后才在这三套数据上与四个既有基线正式对照。")
    tum = proof.get("optional_tum_vi")
    if tum and tum["records"]:
        room_names = sorted(room["name"] for room in tum["per_room"])
        rooms = "room1–room6" if room_names == [f"room{i}" for i in range(1,7)] else "、".join(room_names)
        suites.append(("独立外测",f"TUM-VI {rooms}",tum["records"],tum["scoring_frames"]))
        body += f" 新增的独立检验覆盖 TUM-VI 的 {tum['records']} 个房间序列，比较锁定模型与纯 FullVQF。"
    body += (f" Sassari 每次运动均含 {'、'.join(sensors)} 六路传感器记录。"
             "\n\n表内评分帧为各记录通过本数据集评分掩码的帧数。自采记录跳过起始 10 秒，并检查实测包、参考标签和上下文。"
             "开发集在不同折重复评估，表内对每个有效帧只计一次。成绩先按记录计算倾角 RMSE，再对记录等权平均。")
    source_files = [quality_file.as_posix(), (SHICE/"字段说明.md").as_posix(), "../"+dev_path, "../selection-lock.json",
                    f"../rounds/{proof['selection_id']}/result.json", f"../rounds/{proof['final_id']}/result.json",
                    "../final_eval.py", "../../tilt_data.py", "final_audit.json"]
    if tum:
        source_files.append("../tum-vi-independent.json")
    queries = report["queries"]
    queries["dataset_partitions"] = table_query(
        [{"role":role,"scope":scope,"records":str(count),"frames":f"{frames:,}"} for role,scope,count,frames in suites],
        "自采清单、选模锁和最终逐序列成绩",source_files,"逐条对照原始 CSV、质量审计、训练及留出名单。帧数取自各阶段实际评分记录。")

    fold_body = (f"在同一批 {len(dev)} 条开发记录上分别做两组四折验证。留设备时，一整台设备的记录用于验证。"
                 "留程序时，所有开发设备上属于该程序组的记录一起用于验证。入选模型在每折用其余记录拟合模型参数、归一化参数和安装旋转。"
                 "每条记录在每种划分下各验证一次。两种划分各自先对逐记录误差取平均，再各占一半得到开发分。")
    fold_sources = ["../"+dev_path,"../learning.py","../joint_holdout.py","audit_rounds.py"]
    joint = next((item for item in tally["joint_checks"] if item["candidate_id"]==proof["selected_id"]
                  and item["separate"] and not item["without_embedding"]), None)
    if joint:
        joint_path = f"rounds/{joint['id']}/result.json"
        folds = read(BATCH/joint_path)["folds"]
        example = next(fold for fold in folds if fold["fold"]=="R2/M2")
        held = check_fold(example,"joint_holdout",dev)
        record_names = "、".join(sorted(held,key=lambda n:int(n.split("-")[1])))
        fold_body += (f"\n\n部分候选另做 {len(folds)} 折联合留出，训练时排除指定设备的全部记录和指定程序组在各设备上的全部记录，"
                      f"验证时只取两者交集。例如入选模型的 R2/M2 折用其余 {len(example['training'])} 条拟合，只评 {record_names}。")
        fold_sources.append("../"+joint_path)
    queries["development_folds"] = table_query(fold_rows,"入选模型的逐折训练和验证名单",fold_sources,
        "逐折核对记录的设备和程序编号，验证名单与训练名单互斥。")
    sections = report["report"]["review"]["extraSections"]
    sections[:0] = [
        {"id":"dataset-split","title":"数据集划分","queryId":"dataset_partitions","body":body,
         "columns":[{"field":k,"label":v} for k,v in (("role","用途"),("scope","数据范围"),("records","记录数"),("frames","有效评分帧"))]},
        {"id":"development-folds","title":"开发集内部怎样交叉验证","queryId":"development_folds","body":fold_body,
         "columns":[{"field":k,"label":v} for k,v in (("fold","验证折"),("group","整组留出"),("training","训练记录数"),("validation","验证记录数"))]},
    ]


def add_locked_explanation(report, proof):
    from replay_locked_components import check_saved
    from verify_locked_supplement import audit as verify_supplement

    comp_path = HERE/"locked_component_replay.json"
    if not comp_path.is_file():
        return
    data = check_saved(read(comp_path),proof)
    story,queries = report["report"]["review"],report["queries"]
    sources = ["locked_component_replay.json","replay_locked_components.py","../selection-lock.json","final_audit.json"]
    sets = data["datasets"]
    cases = {suite:len(group["records"]) for suite,group in sets.items()}
    rows = [{"suite":suite,"count":str(cases[suite]),**{k:f"{v:.5f}°" for k,v in group["means"].items()}} for suite,group in sets.items()]
    proof["locked_components"] = {"cases":cases,"mean_deg":{s:g["means"] for s,g in sets.items()},
                                  "mount_angle_deg":data["mount_angle_deg"],"residual_bound_deg":data["residual_bound_deg"]}
    queries["locked_components"] = table_query(rows,"锁模后的逐序列误差分解",sources,
        "重放默认滤波器及锁定的快速滤波器，在相同评分帧上应用同一安装旋转，再对照完整模型的原始重放成绩。")
    local = sets["R6"]["means"]
    default_gain = local["default"]-local["locked"]
    first_gain = local["default"]-local["fast"]
    body = (f"自采 R6 上，把 FullVQF 的加速度时间常数改为锁定值 {data['filter_parameters']['tauAcc']} 秒，误差先降 {first_gain:.5f}°。"
            f"加上锁定的安装旋转再降 {local['fast']-local['mounted']:.5f}°，TCN 残差再降 {local['mounted']-local['locked']:.5f}°。")
    if default_gain > 0:
        body += f" 按这一固定顺序分解，第一步占总降幅的 {first_gain/default_gain:.1%}。"
        wins = proof["tests"]["R6"]["vs"]["physical_only"]["wins"]
        story["summary"] += (f"\n\n按默认滤波、快速滤波、安装旋转、TCN 的顺序重放，"
                             f"快速滤波贡献了自采 R6 总降幅的 **{first_gain/default_gain:.1%}**。"
                             f"与原轻量修正逐条比较，锁定配置在 {cases['R6']} 条记录中改善了 {wins} 条。")
    if "TUM-VI" in sets:
        tum = sets["TUM-VI"]["means"]
        story["summary"] += (f" 在独立的 TUM-VI {cases['TUM-VI']} 个房间序列上，"
                             f"锁定配置为 **{tum['locked']:.3f}°**，纯 FullVQF 为 {tum['default']:.3f}°。")
    body += (" 外部三套数据也对全部序列使用相同步骤。它们的误差在快速滤波这一步已明显上升。"
             "外部拆分是看到正式总成绩后的解释性分析，使用同一份锁定配置和各序列原有的评分帧。")
    story["extraSections"].insert(0,{"id":"locked-components","title":"自采收益和外部误差来自哪一步","queryId":"locked_components","body":body,
        "columns":[{"field":k,"label":v} for k,v in (("suite","数据集"),("count","记录数"),("default","默认 FullVQF"),
        ("fast","仅快速滤波"),("mounted","再加安装旋转"),("locked","再加 TCN 残差"))]})
    receipt = HERE/"locked_tcn_supplement_audit.json"
    if not receipt.is_file():
        return
    saved = read(receipt)
    supplemental = verify_supplement(proof,data,saved["parity"])
    assert saved == supplemental
    proof["locked_tcn_supplement"] = supplemental
    probes = supplemental["timing"]
    for key,label,hz in (("native","锁后补测 · 原生滤波与安装旋转",proof["latency"]["locked"]["nominal_native_hz"]),
                         ("cpu","锁后补测 · TCN 更新（CPU 双线程）",10),("cuda","锁后补测 · TCN 更新（CUDA）",10)):
        queries["latency"]["rows"].append({"action":label,"hz":f"{hz:.1f}","p50":f"{probes[key]['p50_ms']:.4f}","p95":f"{probes[key]['p95_ms']:.4f}"})
    sources = ["../locked-tcn-explanation/protocol.json","../locked-tcn-explanation/result.json","locked_tcn_supplement_audit.json","verify_locked_supplement.py"]
    queries["latency"]["source"]["files"].extend(sources)
    queries["locked_components"]["source"]["files"].extend(sources[:2])
    compared = supplemental["parity"]
    timing = next(s for s in story["extraSections"] if s["id"]=="timing")
    timing["body"] += (f" 锁后在同一条 R6-3 上，两个后端对相同的 {supplemental['update_samples_per_backend']} 个更新位置逐次配对计时。"
                       "输入来自已形成的因果帧块，计时包含特征生成、张量传输、模型前向、修正方向和 CUDA 同步。"
                       f"CPU 双线程与 CUDA 的更新 P95 各为 {probes['cpu']['p95_ms']:.3f} 和 {probes['cuda']['p95_ms']:.3f} ms。"
                       f"另计 {supplemental['native_samples']:,} 个原生帧的滤波与安装旋转。"
                       f"独立重放 {compared['releases']:,} 次更新、{compared['frames']:,} 个评分帧，两后端在评分帧上的方向最大相差 {compared['max_direction_delta_deg']:.6f}°。")
    story["counts"] += (f" 锁后分解和同机计时在 {datetime.fromisoformat(supplemental['started_at']):%H:%M:%S} 至 "
                        f"{datetime.fromisoformat(supplemental['ended_at']):%H:%M:%S} 单独执行，使用冻结权重，新增训练次数为零。")


def direction(a, b):
    if abs(a - b) < 5e-5:
        return "与对照相当"
    return f"{'降低' if a < b else '增加'} {abs(a-b):.3f}°（{abs(a-b)/b:.1%}）"


def timing_rows(item):
    old = item["prior_comparators"]
    locked = item["locked"]
    methods = [("纯 FullVQF 逐帧",locked["nominal_native_hz"],old["full_vqf_per_frame_ms"]),
               ("原轻量修正更新",10,old["correction_update_ms"]["physical_only"]),
               ("原 TimesFM 修正更新",10,old["correction_update_ms"]["timesfm"]),
               ("原两层 GRU 每步 · CPU",item["gru"]["step_hz"],item["gru"]["CPU"]),
               ("原两层 GRU 每步 · CUDA",item["gru"]["step_hz"],item["gru"]["CUDA"]),
               ("本批锁定方法逐帧",locked["nominal_native_hz"],locked["native_step"])]
    if "correction_update" in locked:
        methods.append(("本批锁定方法更新",locked["correction_hz"],locked["correction_update"]))
    if "apply_held_correction" in locked:
        methods.append(("本批应用已有修正",locked["nominal_native_hz"],locked["apply_held_correction"]))
    if "cpu_correction_update" in locked:
        methods.append(("本批锁定方法更新 · CPU",locked["correction_hz"],locked["cpu_correction_update"]))
    if "gate_update" in locked:
        methods.append(("本批双滤波器门控更新（已含在逐帧耗时内）",locked["gate_hz"],locked["gate_update"]))
    if "online_bias_block_update" in locked:
        methods.append(("零偏估计按块更新（已含在逐帧耗时内）",locked["bias_block_hz"],locked["online_bias_block_update"]))
    return [{"action":label, "hz":f"{hz:.1f}", "p50":f"{(m['p50'] if 'p50' in m else m['p50_ms']):.4f}",
             "p95":f"{(m['p95'] if 'p95' in m else m['p95_ms']):.4f}"} for label,hz,m in methods]


def hourly_rows(tally, by_id):
    status = read(BATCH / "status.json")
    start = datetime.fromisoformat(status["window_start"])
    end = datetime.fromisoformat(status["window_end"])
    elapsed = (end-start).total_seconds()
    hours = int((elapsed + 3599) // 3600)
    buckets = [{"rounds":0,"candidates":0,"score":None} for _ in range(hours)]
    for row in tally["completed"]:
        finished = datetime.fromisoformat(row["evidence_ended_at"])
        idx = int((finished-start).total_seconds() // 3600)
        if idx < 0:
            raise ValueError("One round ended before the experiment window")
        while len(buckets) <= idx:
            buckets.append({"rounds":0,"candidates":0,"score":None})
        bucket = buckets[idx]
        bucket["rounds"] += 1
        if row["validated_development_candidate"]:
            score = by_id[row["id"]]["dev_metrics"]["selection_rmse_deg"]
            if "candidate_scores_from" not in row:
                bucket["candidates"] += 1
            prior = bucket["score"]
            bucket["score"] = min(score, prior) if prior is not None else score
    assert sum(b["candidates"] for b in buckets) == tally["development_candidate_runs"]
    output = []
    cumulative = None
    for pos, bucket in enumerate(buckets):
        if bucket["score"] is not None:
            cumulative = min(cumulative,bucket["score"]) if cumulative is not None else bucket["score"]
        begin = start + timedelta(hours=pos)
        stop = begin + timedelta(hours=1)
        output.append({"period":f"{begin:%H:%M}–{stop:%H:%M}","total":str(bucket["rounds"]),
                       "candidates":str(bucket["candidates"]),
                       "best":f"{cumulative:.3f}°" if cumulative is not None else "—"})
    return output


def finish(report, tally):
    from verify_score_profiles import compare_r5

    proof = audit_final()
    declared = tally["shortlist_development_weights"]
    assert declared["filed"] and declared["shortlist_sha256"] == proof["shortlist_replay"]["shortlist_sha256"]
    selection = read(BATCH/"rounds"/proof["selection_id"]/"result.json")
    score_pairs = compare_r5(tally["direct_development_score_profiles"]["groups"],selection)
    proof["r5_for_identical_development_scores"] = score_pairs
    queries = report["queries"]
    story = report["report"]["review"]
    result_path = next(r["result"] for r in tally["completed"] if r["id"] == proof["final_id"])
    sources = ["final_audit.json","../"+result_path,"../selection-lock.json","verify_final.py","verify_external.py"]
    tests = proof["tests"]
    rows = [{"method":label,**{ds:t["macro_rmse_deg"][m] for ds,t in tests.items()}} for m,label in METHODS.items()]
    queries["historical"] = table_query(rows,"锁模后同帧复测",sources,"对照原逐序列得分与帧数，再对本批各方法逐序列等权平均。")
    n, settings = tally["development_candidate_runs"], tally["distinct_development_settings_excluding_seed"]
    r6 = tests["R6"]["macro_rmse_deg"]
    total = len(tally["completed"])
    broad,sassari = (tests[ds]["macro_rmse_deg"] for ds in ("BROAD","Sassari"))
    story["summary"] = (f"## 实验结论\n\n本批有 **{n} 轮直接训练或评分候选**。计及配对试验中登记的候选，共核对 **{settings} 种不同配置**。另计基准复算、附加留出验证、R5 选模与锁模复测，各阶段合计 **{total} 个已完成轮号**。\n\n"
                        f"R5 选中的 `{proof['selected_id']}` 在自采 R6 的 10 条记录上达到 **{r6['locked']:.3f}°**。"
                        f"相对纯 FullVQF 的 {r6['full_vqf']:.3f}°，误差{direction(r6['locked'],r6['full_vqf'])}。"
                        f"相对原轻量修正的 {r6['physical_only']:.3f}°，误差{direction(r6['locked'],r6['physical_only'])}。\n\n"
                        f"BROAD 上锁定配置为 **{broad['locked']:.3f}°**，纯 FullVQF 为 {broad['full_vqf']:.3f}°。"
                        f"Sassari 上分别为 **{sassari['locked']:.3f}°** 和 {sassari['full_vqf']:.3f}°。")
    when = tally["timing_through_last_completed"]
    first = datetime.fromisoformat(when["first_closed_attempt_started"])
    last = datetime.fromisoformat(when["latest_completed_round_ended"])
    hours = (last-first).total_seconds()/3600
    story["counts"] += (f" 首轮启动于 {first:%m-%d %H:%M:%S}，最后一轮结束于 {last:%m-%d %H:%M:%S}，"
                        f"首尾相距 {hours:.2f} 小时。按账本中起止时间合并重叠后，有完整记录的尝试占用 {when['wall_seconds_in_union_of_closed_attempts']/3600:.2f} 小时。"
                        f"已完成轮中，额外重试了 {tally['completed_attempts_beyond_first']} 次。")
    until = datetime.fromisoformat(read(BATCH/"status.json")["window_end"])
    reserve = int((until-last).total_seconds())
    if reserve > 0:
        story["counts"] += (f" 正式复测完成时，距预留窗口结束还有 {reserve//3600} 小时 {(reserve//60)%60} 分 {reserve%60} 秒。"
                            "监督器随后依原计划结束，账本按实际结束时间计时。")
    story["comparisonTitle"] = "自采 R6 与两套外部基准的同帧结果"
    story["comparison"] = (f"SHICE R6 含 10 条记录，BROAD 含 39 条，Sassari 为 3 次运动的 18 路传感器。"
                           f"锁定配置与纯 FullVQF 配对比较，R6 有 {tests['R6']['vs']['full_vqf']['wins']}/10 条改善，"
                           f"BROAD 为 {tests['BROAD']['vs']['full_vqf']['wins']}/39 条，Sassari 为 {tests['Sassari']['vs']['full_vqf']['wins']}/18 路。")
    r6_light = tests["R6"]["vs"]["physical_only"]
    story["comparison"] += (f" 与原轻量修正逐条比较，R6 有 {r6_light['wins']}/10 条改善，"
                            f"误差差值的中位数为 {r6_light['median_locked_minus_comparator_deg']:+.3f}°。")
    labels = {"rotation":"无干扰旋转","translation":"无干扰平移","combined":"无干扰组合运动","disturbed":"带干扰的记录",
              "slow":"慢速","medium":"中速","fast":"快速"}
    external_rows = [{"group":f"{ds} · {labels[key]}","count":str(item["records"]),
                      **{m:f"{item['mean_deg'][m]:.3f}°" for m in METHODS}}
                     for ds in ("BROAD","Sassari") for key,item in tests[ds]["motion_groups"].items()]
    queries["external_motions"] = table_query(external_rows,"按原始采集条件核对两套外部数据",sources,
        "直接读取采集文件名确定分组。每条序列等权，再按各组记录数加权还原各方法的全库成绩。")
    story["extraSections"].insert(0,{"id":"external-motions","title":"锁定配置在不同外部运动条件下的表现","queryId":"external_motions",
        "body":"BROAD 的 39 条记录按采集文件名分成四组。Sassari 的每个速度对应一次运动，同一次运动由六路传感器同步记录。各格为组内逐序列等权平均倾角 RMSE。",
        "columns":[{"field":"group","label":"数据与采集条件"},{"field":"count","label":"记录数"},
                   *({"field":m,"label":label} for m,label in METHODS.items())]})
    if all("calibration" in tests[ds] for ds in ("R6","BROAD","Sassari")):
        active = [f"{ds} 为 {t['calibration']['active_records']}/{t['records']} 条、{t['calibration']['active_frames']:,} 个评分帧"
                  for ds,t in tests.items()]
        story["comparison"] += " 在线零偏实际启用的范围为" + "，".join(active) + "。"
    r6_replays = proof["raw_locked_r6_by_record"]
    verified = next(s for s in story["extraSections"] if s["id"] == "verification")
    shortlist_proof = proof["shortlist_replay"]
    native = proof["raw_shice_score_indices"]
    verified["body"] += (f" 按短名单登记时已完成的 {shortlist_proof['completed_rounds_available']} 个轮号重建筛选，"
                         f"得到相同的 {shortlist_proof['count']} 个条目、顺序和入围理由。"
                         "另外只向筛选程序提供开发分数、配置和入围所需的复核标记，省去独立种子及其他诊断记录，所得名单也逐项相同。"
                         f"从原始自采文件重建 {len(native['R5'])} 条 R5 与 {len(native['R6'])} 条 R6 的评分帧索引，"
                         "核对了每个成功评分的 R5 候选及 R6 正式成绩保存的逐记录索引指纹。")
    if proof["per_record_checkpoint"]:
        receipt = proof["per_record_checkpoint"]
        verified["body"] += f" 正式轮单独落盘的 {sum(receipt['counts'].values())} 条序列记录，其评分帧指纹及五种方法的分数均与套件汇总一致。"
    if proof["baseline_source_sha256"]:
        verified["body"] += " 正式轮引用的三份历史成绩和三份模型权重已与实验开始时的文件指纹对账，历史增益配置也已复核。"
    largest = max(r6_replays.values(), key=lambda r:r["absolute_difference_deg"])
    verified["body"] += (f" 对本批入选方法，另从全部 {len(r6_replays)} 条 R6 原始对齐 CSV 重建六轴输入、参考方向和评分帧，"
                         f"用锁定权重重新推断，共 {sum(r['frames'] for r in r6_replays.values()):,} 个评分帧。"
                         f"逐条 RMSE 与正式结果的最大差值为 {largest['absolute_difference_deg']:.7f}°（{largest['record']}）。")
    deltas = [(v,ds,method) for ds,t in tests.items() for method,v in t["maximum_absolute_difference_from_historical_per_record_deg"].items()]
    gap, dataset, method = max(deltas)
    verified["body"] += f" 四个既有基线在三套数据上的正式复测与历史逐序列成绩比较，最大差值为 {gap:.7f}°（{dataset}，{METHODS[method]}）。"
    external = proof["raw_external_replays"]
    locked_raw = [r for records in external.values() for r in records if "raw_locked_deg" in r]
    verified["body"] += (f" 审查另从原始 MAT 文件重建 BROAD 的 {len(external['BROAD'])} 条记录和 Sassari 的 "
                         f"{len(external['Sassari'])} 路数据的逐帧评分索引，逐条重跑纯 FullVQF。"
                         f"同一批 {len(locked_raw)} 条记录均用锁定权重重新推断，"
                         f"与正式轮的逐记录误差最大相差 {max(r['locked_abs_difference_deg'] for r in locked_raw):.7f}°。")
    dev_q = proof["development_quaternion_exports"]
    formal_q, exported = proof["locked_quaternion_verification"], proof["raw_quaternion_export_audit"]
    if dev_q["outputs"] or formal_q:
        write(HERE / "development_quaternion_audit.json",dev_q)
        queries["quaternion_outputs"] = table_query(dev_q["outputs"] + ([exported] if exported else []),
            "原始编码器与逐帧姿态导出",["development_quaternion_audit.json",*sources],
            "从原始编码器四元数独立恢复参考方向，在相同时间戳和掩码上复算输出倾角。")
        body = "输出按原采样顺序保存 IMU 到世界坐标的 wxyz 四元数。倾角由预测的竖直方向评分，绝对航向未测。"
        if dev_q["outputs"]:
            n = len(dev_q["outputs"])
            body += (f"开发阶段有 {dev_q['candidates']} 个候选在 {len(dev_q['unique_source_records'])} 条原始记录上导出 {n} 份结果，"
                     f"{dev_q['passed_outputs']}/{n} 份通过单位范数、倾角一致性与截断前缀检验。本次审查逐份用原始 CSV 的时间戳和编码器方向重算倾角。")
        if formal_q and exported:
            cert = formal_q["result"]
            state = "通过" if cert["passed"] else "未通过"
            max_prefix = max(p["max_rotation_delta_deg"] for p in cert["prefix_checks"])
            body += (f" 锁模后在 {cert['name']} 导出 {cert['native_frames']:,} 个四元数，使用 {cert['scored_frames']:,} 个评分帧，"
                     f"独立重算的倾角 RMSE 为 {exported['raw_quaternion_rmse_deg']:.4f}°。"
                     f"三个截断前缀与整段回放的最大姿态差为 {max_prefix:.6f}°，该记录{state}预定核对。")
        elif formal_q:
            body += f" 锁模后四元数检验未取得导出文件，执行记录为 {formal_q.get('error',formal_q['status'])}。"
        story["extraSections"].append({"id":"quaternion-output","title":"逐帧姿态输出和前缀一致性","queryId":"quaternion_outputs","body":body})
    lat = proof["latency"]
    queries["latency"] = table_query(timing_rows(lat),"同机延迟复测",sources,"分列原生帧、每次修正和应用已有修正的 P50/P95，时钟单位均为毫秒。")
    hb = "SHICE 采用转台编码器参考，同批各方法使用相同评分帧。R6 仅留出一台设备，以前的实验已观察过三套数据的结果。"
    split_section = next(s for s in story["extraSections"] if s["id"] == "split-method")
    split_section["body"] += " " + hb
    if score_pairs:
        paired = [p for p in score_pairs if p["status"] == "compared"]
        split_section["body"] += (f" 正式名单中有 {len(score_pairs)} 对候选在开发集上的逐记录误差完全相同。"
                                  f"其中 {len(paired)} 对均取得 R5 成绩，{sum(p['identical'] for p in paired)} 对在 R5 上也逐条相同。")
    delay = proof["delay_replay"]
    lat_text = (f"在 {lat['machine']} 上离线计时，GPU 型号为 {lat['accelerator']}。每帧处理与每次修正的频率不同，表内分别列出。"
                "修正头使用已准备的输入块计时。端到端设备采集和队列等待尚无实测。")
    scope = proof.get("bias_latency_scope")
    if scope:
        lat_text += (f" 原始记录 {scope['record']} 的计时区间有 {scope['measured_native_frames']:,} 个原生帧，"
                     f"其中 {scope['frames_with_active_estimate']:,} 帧处于在线补偿启用时段，"
                     f"有 {scope['observed_packets_with_bias']:,} 帧对新收到的数据应用了补偿。"
                     f"{scope['measured_blocks']} 个计时块内有 {len(scope['accepted_refits_after_samples'])} 次重新估计成功。")
    if delay:
        lat_text += f" 按本机实测 P95 将修正延后 {delay['delay_ms']:.3f} ms 再应用，R6 误差为 {delay['R6']['macro_rmse_deg']:.3f}°。"
    cpu = proof["locked_cpu_verification"]
    if cpu and cpu["status"] in {"within_recordwise_tolerance","exceeds_tolerance"}:
        host_times = cpu["latency"]["timings"]
        lat_text += (f" 在 {cpu['latency']['recording']} 上用同一权重测单次更新，NumPy CPU 的 P95 为 "
                     f"{host_times['numpy_cpu']['correction_update']['p95_ms']:.3f} ms，PyTorch CUDA 为 "
                     f"{host_times['torch_cuda']['correction_update']['p95_ms']:.3f} ms。R6 十条记录中，两者单条倾角 RMSE "
                     f"最大相差 {cpu['largest_recordwise_tilt_rmse_delta_deg']:.6f}°（复现容差 {cpu['tolerance_deg']:.4f}°）。")
    elif cpu and cpu["status"] == "failed":
        lat_text += f" 锁模后的 CPU 复现未完成，记录的错误为 {cpu.get('error','未提供错误信息')}。"
    timing_section = {"id":"timing","title":"同机算法耗时","queryId":"latency","body":lat_text,
                      "columns":[{"field":"action","label":"步骤"},{"field":"hz","label":"频率 Hz"},
                                 {"field":"p50","label":"P50 ms"},{"field":"p95","label":"P95 ms"}]}
    story["extraSections"].insert(0,timing_section)
    cfg = proof["selected_config"]
    details = "，".join(f"`{k}={v}`" for k,v in cfg.items() if k not in ("kind","parameters"))
    if cfg.get("parameters"):
        details += "，" + "，".join(f"`{k}={v}`" for k,v in cfg["parameters"].items())
    story["extraSections"].insert(0,{"id":"locked-choice","title":"选中的配置与锁定时间","queryId":"historical",
        "body":f"模型类型为 `{cfg['kind']}`，R5 倾角误差为 **{proof['r5_rmse_deg']:.3f}°**。参数包括 {details}。"
               f"配置于 {datetime.fromisoformat(proof['locked_at']):%m-%d %H:%M:%S} 锁定。权重 SHA-256 为 `{proof['weights_sha256']}`。"})
    extra = proof["optional_tum_vi"]
    if extra and extra["records"]:
        score, base = extra["locked"]["macro_rmse_deg"], extra["full_vqf"]["macro_rmse_deg"]
        q = [{"room":r["name"],"frames":f"{r['frames']:,}","locked":f"{r['rmse_deg']:.3f}°",
              "pure":f"{r['baseline_deg']:.3f}°","difference":f"{r['rmse_deg']-r['baseline_deg']:+.3f}°"} for r in extra["per_room"]]
        missing = [r["name"] + " (" + r["status"] + ")" for r in extra["failures"]] + extra["unreported_rooms"]
        missing_text = " 未取得评分的房间为 " + "、".join(missing) + "。" if missing else ""
        replays = extra["raw_replays"]
        quiet = {r["name"]:r["quiet_gravity_check"] for r in replays}
        for room in q:
            check = quiet[room["room"]]
            room["quiet"] = (f"{check['qualifying_blocks']} 窗 / {check['median_deg']:.2f}°"
                             if check["qualifying_blocks"] else "0 窗")
        reviewed = (f" 独立读取 {len(replays)} 个房间的原始 CSV，用整数纳秒时间轴重建评分帧，逐房间重放纯滤波器和锁定模型。"
                    f"锁定模型与正式成绩的最大差值为 {max(r['locked_replay_abs_difference_deg'] for r in replays):.7f}°。"
                    "低动态检查列出完整一秒窗的数量，以及加速度方向与动捕竖直方向夹角的中位数。该角度也受传感器偏差与残余运动影响。")
        queries["independent"] = table_query(q,"TUM-VI 独立检验",["../tum-vi-independent.json","../tum-vi-protocol.json","final_audit.json","verify_tum.py"],"逐房间按预设时间配准、评分帧和动捕参考计算，再对取得成绩的房间等权平均。")
        story["extraSections"].insert(1,{"id":"untuned-external","title":"新增的 TUM-VI 房间留出","queryId":"independent",
            "body":f"用同一份权重完成 {extra['records']}/6 个房间，共 {extra['scoring_frames']:,} 帧。锁定配置的宏平均倾角误差为 {score:.3f}°，相对纯 FullVQF 的 {base:.3f}°，误差{direction(score,base)}。"
                   "预设协议在传感器与动捕的校准时间轴上匹配四元数，排除超过 25 ms 的动捕间隔和每条记录的前 10 秒。该组数据未参与本批选模。" + reviewed + missing_text,
            "columns":[{"field":"room","label":"房间"},{"field":"frames","label":"评分帧"},{"field":"locked","label":"锁定配置"},
                       {"field":"pure","label":"纯 FullVQF"},{"field":"difference","label":"配对差值"},{"field":"quiet","label":"低动态方向核对"}]})
    else:
        outcome = proof["optional_tum_outcome"]
        if extra and extra["failures"]:
            why = "、".join(r["name"] + " (" + r["status"] + ")" for r in extra["failures"])
        elif isinstance(outcome, dict):
            why = outcome.get("error") or outcome.get("status") or str(outcome)
        else:
            why = "本批未生成房间评分文件"
        story["extraSections"].insert(1,{"id":"untuned-external","title":"TUM-VI 独立检验的覆盖","queryId":"historical",
            "body":f"本批未取得可核验的 TUM-VI 房间成绩。执行记录为 {why}。"})
    full_index = read(BATCH / "round_index.json")
    by_id = {r["round_id"]:r for r in full_index}
    hourly = hourly_rows(tally, by_id)
    queries["by_hour"] = table_query(hourly,"实验窗口内每小时的轮次",["round_audit.json","../ledger.jsonl","../round_index.json"],
                                     "按完成时间归入自实验开始起的连续一小时，计算该时段的完成量与截至当时的最佳开发分。")
    story["extraSections"].append({"id":"hourly-progress","title":"十小时窗口中的轮次进展","queryId":"by_hour",
        "body":"每个轮号归入其完成时段。累计最佳只使用 R1–R4 的候选开发评分。北京时间 23 时段始于 11 月 22 日，零时以后为 11 月 23 日。",
        "columns":[{"field":"period","label":"北京时间"},{"field":"total","label":"全部完成"},
                   {"field":"candidates","label":"其中候选"},{"field":"best","label":"累计最佳开发 RMSE"}]})
    r6_records = read(BATCH / result_path)["datasets"]["R6"]["sequences"]
    per_record = []
    for r in sorted(r6_records,key=lambda row:int(row["name"].split("-")[1])):
        values = r["rmse_deg"]
        per_record.append({"name":r["name"],"frames":f"{r['frames']:,}","pure":f"{values['full_vqf']:.3f}°",
                           "light":f"{values['physical_only']:.3f}°","locked":f"{values['locked']:.3f}°",
                           "change":f"{values['locked']-values['physical_only']:+.3f}°"})
    paired_residuals = {r["name"]:r["rmse_deg"]["locked"]-r["rmse_deg"]["physical_only"] for r in r6_records}
    omissions = {n:fmean(v for k,v in paired_residuals.items() if k!=n) for n in paired_residuals}
    proof["r6_leave_one_out_vs_physical_deg"] = omissions
    reversals = [n for n,v in omissions.items() if v*fmean(paired_residuals.values())<0]
    sensitivity_text = (" 单独去掉 "+" 或 ".join(reversals)+" 后，其余九条的平均差值分别为 "
                        +"、".join(f"{omissions[n]:+.4f}°" for n in reversals)+"，平均优劣随之反转。") if reversals else ""
    queries["r6_records"] = table_query(per_record,"自采 R6 逐条对比",sources,"每条记录在相同评分帧上计算各方法的倾角误差，再配对相减。")
    story["extraSections"].insert(0,{"id":"self-records","title":"自采 R6 各条转台记录","queryId":"r6_records",
        "body":"最后一列为锁定配置减去原轻量修正的倾角 RMSE。负值表示这条记录的误差下降。"+sensitivity_text,
        "columns":[{"field":"name","label":"记录"},{"field":"frames","label":"评分帧"},{"field":"pure","label":"纯 FullVQF"},
                   {"field":"light","label":"原轻量"},{"field":"locked","label":"锁定配置"},{"field":"change","label":"与原轻量之差"}]})
    selection_source = read(BATCH / by_id[proof["selection_id"]]["result"])
    scored_shortlist = {c["round_id"]:c for c in selection_source["r5_candidates"]}
    failed_shortlist = {c["round_id"] for c in selection_source["failed_candidates"]}
    nomination_table = queries["r5_nominations"]
    assert {r["round"] for r in nomination_table["rows"]} == set(scored_shortlist) | failed_shortlist
    for row in nomination_table["rows"]:
        candidate = scored_shortlist.get(row["round"])
        if candidate is None:
            row.update(r5="—", outcome="评分失败")
        else:
            outcome = "锁定" if row["round"]==proof["selected_id"] else ("通过门槛" if candidate["eligible_for_selection"] else "未通过门槛")
            row.update(r5=f"{candidate['r5']['macro_rmse_deg']:.5f}°",outcome=outcome)
    nomination_section = next(s for s in story["extraSections"] if s["id"]=="r5-nominations")
    nomination_section["columns"].extend([{"field":"r5","label":"R5 误差"},{"field":"outcome","label":"R5 结果"}])
    nomination_table["source"]["files"].append("../"+by_id[proof["selection_id"]]["result"])
    original_validation = read(BATCH.parent / "tilt-correction/validation.json")
    prior_gain = original_validation["gains"]["physical_only"]
    incumbent_score = original_validation["r5_macro_rmse_deg"][f"physical_only@{prior_gain:g}"]
    same_family = [r for r in full_index if r["status"]=="completed" and parameter_identity(r["config"])==parameter_identity(cfg)
                   and r["dev_metrics"] and "selection_rmse_deg" in r["dev_metrics"]]
    selection_section = next(s for s in story["extraSections"] if s["id"]=="locked-choice")
    replay = proof["shortlist_replay"]
    if seen := replay.get("live_filing_observation"):
        if not seen["progress_file_present"] and not seen["selection_lock_present"]:
            request = replay["supervisor_request"]
            selection_section["body"] += (f" 监督器最初提交 {request['count']} 项，选模进程从开发数据重建正式的 {replay['count']} 项。"
                                          f"审查于 {datetime.fromisoformat(seen['observed_at']):%H:%M:%S} 在首份候选成绩写出前记录了名单指纹。")
    selection_section["body"] += (f" R5 共成功评分 {len(selection_source['r5_candidates'])} 个入围配置，失败 {len(selection_source['failed_candidates'])} 个。"
                                  f"原轻量头在同口径 R5 上的历史参照为 {incumbent_score:.3f}°。")
    incumbent = selection_source.get("r5_fixed_historical_incumbent") or {}
    if incumbent.get("r5") and incumbent.get("status") != "historical_reuse_after_recomputation_error":
        selection_section["body"] += f" 本批原样复算该头得到 {incumbent['r5']['macro_rmse_deg']:.3f}°。"
    elif incumbent.get("status") == "historical_reuse_after_recomputation_error":
        selection_section["body"] += " 本批重放旧头遇到错误，执行方逐条核对旧分数和评分帧后引用了历史值。"
    if incumbent.get("r5"):
        selection_section["body"] += f" 相对旧头，本批锁定配置在 R5 的误差{direction(proof['r5_rmse_deg'],incumbent['r5']['macro_rmse_deg'])}。"
    influence = proof["r5_leave_one_record_out"]
    if influence:
        if influence["full_runner_up"]:
            selection_section["body"] += (f" 全部 R5 记录上，入选模型与次名 `{influence['full_runner_up']}` "
                                          f"相差 {influence['full_runner_up_margin_deg']:.4f}°。")
        selection_section["body"] += (f" 锁模后，轮流拿掉一条 R5 记录，在其余记录上按原规则复算，"
                                      f"同一配置在 {influence['original_wins']}/{len(influence['omissions'])} 次中仍排第一。")
    if len(same_family) > 1:
        values = [r["dev_metrics"]["selection_rmse_deg"] for r in same_family]
        selection_section["body"] += (f" 入选配置有 {len(same_family)} 组种子成绩，开发误差中位数为 {median(values):.3f}°，"
                                      f"范围为 {min(values):.3f}° 至 {max(values):.3f}°。")
        if cfg["kind"]=="tcn":
            family_ids = {r["round_id"] for r in same_family}
            joint_by_parent = {r["candidate_id"]:r["rmse_deg"] for r in tally["joint_checks"]
                               if r["candidate_id"] in family_ids and not r["without_embedding"]}
            if set(joint_by_parent)==family_ids:
                family_detail = "、".join(f"{r['seed']}（{r['round_id']}）{joint_by_parent[r['round_id']]:.5f}°"
                                         for r in sorted(same_family,key=lambda r:r["seed"]))
                tested = sorted(family_ids.intersection(scored_shortlist))
                selection_section["body"] += (f" {len(same_family)} 个训练种子在同时留出设备和程序时的误差分别为 {family_detail}。"
                                              f"该家族进入正式 R5 评分的是 {'、'.join(tested)}。")
    diffs = []
    for row in full_index:
        if row["batch"] == "matched-ablation" and row["status"] == "completed":
            other = by_id[row["relation"]]
            diffs.append(row["dev_metrics"]["selection_rmse_deg"]-other["dev_metrics"]["selection_rmse_deg"])
    if diffs:
        body = (f"共完成 {len(diffs)} 组物理特征与加入冻结 TimesFM 特征的配对实验。加入 TimesFM 后，"
                f"{sum(d < 0 for d in diffs)} 组开发误差下降。加入后的平均误差减去物理特征组为 {fmean(diffs):+.3f}°。"
                "该比较对应本批修正头、更新频率和筛选方法。")
        story["extraSections"].insert(0,{"id":"ablation","title":"TimesFM 特征带来的变化","queryId":"rounds","body":body})
    joint = tally["joint_checks"]
    if joint:
        section = next(s for s in story["extraSections"] if s["id"] == "joint-holdout")
        body = section["body"]
        joint_scores = {r["candidate_id"]:r["rmse_deg"] for r in joint if not r["without_embedding"]}
        if proof["selected_id"] in joint_scores:
            body += f" 最终入选配置在同一复核中的误差为 {joint_scores[proof['selected_id']]:.3f}°。"
        paired_path = BATCH / "regime-response-stage.json"
        if paired_path.is_file():
            for family in read(paired_path).get("completed_seed_family_pairs", []):
                settings = by_id[family["family"]]["config"]
                if settings.get("installation") != "rest" or not settings.get("train_dynamic_boost"):
                    continue
                pairs = [(by_id[p["new_round"]], by_id[p["control"]]) for p in family["paired_deltas"]]
                assert len({test["seed"] for test, _ in pairs}) == len(pairs)
                assert all(test["seed"] == control["seed"] for test, control in pairs)
                diffs = [test["dev_metrics"]["selection_rmse_deg"] - control["dev_metrics"]["selection_rmse_deg"]
                         for test, control in pairs]
                stricter = [joint_scores[test["round_id"]] - joint_scores[control["round_id"]]
                            for test, control in pairs if test["round_id"] in joint_scores and control["round_id"] in joint_scores]
                if diffs and len(stricter) == len(diffs):
                    body += (f" 对原轻量头采用静止段安装校准、提高动态段训练权重后，{len(pairs)} 个种子与各自对照相比，"
                             f"常规开发误差平均变化 {fmean(diffs):+.4f}°，同时留出误差平均变化 {fmean(stricter):+.4f}°，"
                             f"其中 {sum(d < 0 for d in stricter)}/{len(stricter)} 个种子的同时留出误差下降。")
        section["body"] = body
    if "R00047" in by_id and by_id["R00047"].get("previous_evidence"):
        story["extraSections"].append({"id":"data-repair","title":"长窗口训练样本的修订","queryId":"rounds",
            "body":"早期的 128 块窗口沿用了 64 块上下文掩码。审查后，训练筛选改为覆盖完整窗口，旧产物保存在原轮目录中。R00047 以同一轮号重跑，计入一次有效配置和一次额外尝试。"})
    if "A00001" in by_id and by_id["A00001"].get("previous_evidence"):
        section = next(s for s in story["extraSections"] if s["id"] == "accelerometer-bias")
        section["body"] += " 首次回放对零补偿也重新做了浮点运算。为逐值复现对照，A00001 在同一轮号修正后重跑，两次尝试均保留。"
    add_locked_explanation(report,proof)
    add_dataset_partitions(report,selection,proof,tally)
    close_path = HERE/"window_close.json"
    if close_path.is_file():
        from verify_baselines import digest
        close = read(close_path)
        checked_at = datetime.fromisoformat(close["checked_at"])
        assert checked_at >= until and datetime.fromisoformat(close["window_end"]) == until
        assert close["registered"] == tally["indexed_rounds"] and close["status_counts"] == tally["status_counts"]
        assert close["locked_candidate"] == proof["selected_id"] and close["status"] == "completed"
        files = close["final_sha256"]
        assert len(files) == close["monitored_files"] and all(digest(BATCH/name) == value for name,value in files.items())
        assert close["all_same"] == (not close["content_changes"] and files == close["initial_sha256"])
        first_observation = datetime.fromisoformat(close["observed_from"])
        observed = (f"从 {first_observation:%H:%M:%S} 跟踪至窗口结束，"
                    f"于 {checked_at:%H:%M:%S} 复查 {len(files)} 份实验原件，涵盖账本、入围名单、锁定权重及最终成绩。")
        if close["all_same"]:
            observed += " 此期间各文件的内容指纹一致。"
        else:
            observed += f" 此期间记录到 {len(close['content_changes'])} 次文件内容变化，报告采用截止时点复查的版本。"
        verification = next(s for s in story["extraSections"] if s["id"]=="verification")
        verification["body"] += " " + observed
        queries["rounds"]["source"]["files"].append("window_close.json")
        proof["window_close"] = {"verified_at":close["checked_at"],"receipt_sha256":digest(close_path),
                                 "reviewed_sources":len(files),"all_same":close["all_same"]}
    write(HERE/"final_audit.json",proof)
    report["buildStatus"] = "complete"
    return report
