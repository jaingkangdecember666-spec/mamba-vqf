"""Close the time window using verified persisted results when a formal round did not finish."""

from datetime import datetime
from math import isfinite
from statistics import fmean

from audit_rounds import BATCH, HERE, check_scores, fingerprint, read, write
from final_narrative import METHODS, hourly_rows, table_query
from verify_final import (check_baseline_references, check_frame_evidence, check_r5_scores, checkpoint_records, compare_checkpoint_row,
                          digest, original_rows, reproduce_shortlist, shice_frame_evidence, verify_r5_selection)


def selection_progress(manifest, progress, index, frames, native):
    replay = reproduce_shortlist(index,manifest)
    planned = [r["round_id"] for r in manifest["selected_before_r5"]]
    assert {"baseline","install_rotation"} <= {r["kind"] for r in manifest["selected_before_r5"]}
    scored, failed = [], {}
    if progress is not None:
        assert progress["shortlist_sha256"] == replay["shortlist_sha256"]
        assert datetime.fromisoformat(progress["at"]) >= datetime.fromisoformat(manifest["at"])
        for candidate in progress["r5_candidates"]:
            check_r5_scores(candidate,index,frames,native)
            scored.append(candidate["round_id"])
        failed = {r["round_id"]:r["error"] for r in progress["failed_candidates"]}
        assert len(failed) == len(progress["failed_candidates"]) and all(failed.values())
    attempted = [*scored,*failed]
    assert len(set(attempted)) == len(attempted)
    assert set(attempted) == set(planned[:len(attempted)]), "R5 检查点跳过了已登记顺序中的候选"
    return {"at":progress["at"] if progress else manifest["at"],"shortlist_replay":replay,
            "planned":len(planned),"scored":scored,"failed":failed,"unreported":planned[len(attempted):]}


def show_live_selection(report, weights, development_summary):
    """Describe committed R5 checkpoints while the official external evaluation is pending."""
    index = {r["round_id"]:r for r in read(BATCH/"round_index.json")}
    filed = []
    for item in index.values():
        folder = BATCH/"rounds"/item["round_id"]
        if item["config"]["kind"]!="selection" or not (folder/"shortlist.json").is_file():
            continue
        manifest = read(folder/"shortlist.json")
        if manifest["at"]==weights["at"] and manifest["shortlist_sha256"]==weights["shortlist_sha256"]:
            filed.append((folder,manifest))
    (folder,manifest), = filed
    lock_path = BATCH/"selection-lock.json"
    lock = read(lock_path) if lock_path.is_file() else None
    progress_path = folder/"selection-progress.json"
    progress = read(progress_path) if progress_path.is_file() else None
    baseline, = weights["controls"]["baseline"]
    baseline_metrics = read(BATCH/index[baseline]["result"])["r5"]
    frames = {s["name"]:s["frames"] for s in baseline_metrics["sequences"]}
    native = {n:shice_frame_evidence(n) for n in frames} if progress and progress["r5_candidates"] else {}
    evidence = selection_progress(manifest,progress,index,frames,native)
    assert evidence["shortlist_replay"]["shortlist_sha256"] == weights["shortlist_sha256"]
    scored = {r["round_id"]:r for r in progress["r5_candidates"]} if progress else {}
    when = datetime.fromisoformat(evidence["at"])
    message = (f" 截至 {when:%H:%M:%S}，R5 的 {evidence['planned']} 项正式候选中，"
               f"{len(scored)} 项完成十条记录的评分，{len(evidence['failed'])} 项失败，"
               f"{len(evidence['unreported'])} 项待出结果。")
    selected_id = None
    if lock:
        assert fingerprint({k:v for k,v in lock.items() if k!="lock_sha256"}) == lock["lock_sha256"]
        assert (BATCH/lock["weights"]).parent.resolve() == folder.resolve() and digest(BATCH/lock["weights"])==lock["weights_sha256"]
        assert lock["shortlist_sha256"]==manifest["shortlist_sha256"] and not evidence["unreported"]
        assert datetime.fromisoformat(manifest["at"]) <= when <= datetime.fromisoformat(lock["locked_at"])
        selected = lock["selected"]
        selected_id = selected["round_id"]
        assert selected["config_sha256"]==fingerprint(selected["config"])==index[selected_id]["config_sha256"]
        assert selected["r5"]==scored[selected_id]["r5"] and selected["r5_sequences"]==scored[selected_id]["r5_sequences"]
        at = datetime.fromisoformat(lock["locked_at"])
        message += (f" {at:%H:%M:%S} 锁定 {selected_id}，R5 倾角 RMSE 为 {selected['r5']['macro_rmse_deg']:.5f}°。"
                    "已核对配置、权重和名单的指纹。锁模后各数据集的逐序列结果正在审查。")
        evidence["lock"] = {"round":selected_id,"at":lock["locked_at"],"sha256":lock["lock_sha256"],"weights_sha256":lock["weights_sha256"]}
        official_path = folder/"result.json"
        if official_path.is_file():
            official = read(official_path)
            assert official["lock_sha256"]==lock["lock_sha256"]
            formal_ids = {r["round_id"] for r in manifest["selected_before_r5"]}
            influence = verify_r5_selection(official,index[baseline],baseline_metrics,index,formal_ids,frames,native,selected)
            evidence["eligibility_checked"] = len(official["r5_candidates"])
            evidence["leave_one_out"] = influence
            if influence:
                message += (f" 用全部 R5 候选重算准入条件和排序得到同一配置，逐次去掉一条记录后，"
                            f"它在 {influence['original_wins']}/{len(influence['omissions'])} 次中仍列首位。")
    else:
        message += " 正式锁定结果待出。"
    story = report["report"]["review"]
    story["summary"] = development_summary+message
    seen = evidence["shortlist_replay"].get("live_filing_observation")
    if seen and not seen["progress_file_present"] and not seen["selection_lock_present"]:
        section = next(s for s in story["extraSections"] if s["id"]=="r5-nominations")
        request = evidence["shortlist_replay"]["supervisor_request"]
        section["body"] += (f" 监督器提交 {request['count']} 项，选模进程按完整开发规则生成正式的 {evidence['planned']} 项。"
                            f"审查于 {datetime.fromisoformat(seen['observed_at']):%H:%M:%S} 现场记录了名单指纹，早于第一份 R5 候选成绩。")
    table = report["queries"]["r5_nominations"]
    source_files = ["../"+(folder/"shortlist.json").relative_to(BATCH).as_posix()]
    if progress:
        source_files.append("../"+progress_path.relative_to(BATCH).as_posix())
    if lock:
        source_files.append("../selection-lock.json")
    table["source"]["files"].extend(source_files)
    for row in table["rows"]:
        key = row["round"]
        if key in scored:
            row.update(r5=f"{scored[key]['r5']['macro_rmse_deg']:.5f}°",outcome="锁定" if key==selected_id else "已评分")
        else:
            row.update(r5="—",outcome="评分失败" if key in evidence["failed"] else "待出结果")
    section = next(s for s in story["extraSections"] if s["id"]=="r5-nominations")
    section["columns"].extend([{"field":"r5","label":"R5 误差"},{"field":"outcome","label":"评分进度"}])
    write(HERE/"r5_live_audit.json",evidence)


def audit_partial_dataset(group, expected):
    rows = group["sequences"]
    names = {r["name"] for r in rows}
    assert rows and len(rows) == len(names) and names <= set(expected), "Unexpected record list"
    frames = {r["name"]:expected[r["name"]]["scoring_frames"] for r in rows}
    means, deltas = {}, {}
    for method in METHODS:
        scalar = [{"name":r["name"],"frames":r["frames"],"rmse_deg":r["rmse_deg"][method]} for r in rows]
        for row in scalar:
            assert row["frames"] == frames[row["name"]] > 0 and isfinite(row["rmse_deg"]) and 0 <= row["rmse_deg"] <= 180
        if "summary" in group:
            check_scores(scalar,group["summary"][method],frames)
        means[method] = fmean(r["rmse_deg"] for r in scalar)
        if method != "locked":
            deltas[method] = max(abs(r["rmse_deg"]-expected[r["name"]]["rmse_deg"][method]) for r in scalar)
    return {"records":len(rows),"planned":len(expected),"frames":sum(frames.values()),"means":means,
            "missing":sorted(set(expected)-names),"max_delta_vs_historical_baselines":deltas}


def closing_evidence(index, status):
    proof = {"window_end":status["window_end"],"supervisor_phase":status["phase"],"last_heartbeat":status["heartbeat_at"],
             "lock":None,"tests":{},"issues":[],"latency_verified":False,"tum_verified_rooms":[],"r5_progress":[]}
    native = {}
    for step in index.values():
        if step["config"]["kind"] != "selection":
            continue
        folder = BATCH / "rounds" / step["round_id"]
        if not (folder/"shortlist.json").is_file():
            continue
        try:
            manifest = read(folder/"shortlist.json")
            progress = read(folder/"selection-progress.json") if (folder/"selection-progress.json").is_file() else None
            base = next(s for s in index.values() if s["config"]["kind"] == "baseline" and s["status"] == "completed")
            frames = {s["name"]:s["frames"] for s in read(BATCH/base["result"])["r5"]["sequences"]}
            if progress and progress["r5_candidates"] and not native:
                native = {n:shice_frame_evidence(n) for n in frames}
            checked = selection_progress(manifest,progress,index,frames,native)
            proof["r5_progress"].append({**checked,"round":step["round_id"]})
        except (AssertionError,KeyError,ValueError,OSError,StopIteration) as exc:
            proof["issues"].append(f"{step['round_id']} 的 R5 进度未通过复核，"+(str(exc) or type(exc).__name__))
    lock_path = BATCH / "selection-lock.json"
    if not lock_path.is_file():
        proof["issues"].append("没有落盘的选模锁定文件。")
        return proof
    try:
        lock = read(lock_path)
        chosen = lock["selected"]
        key = lock["lock_sha256"]
        assert fingerprint({k:v for k,v in lock.items() if k != "lock_sha256"}) == key, "锁定文件指纹不符"
        assert digest(BATCH/lock["weights"]) == lock["weights_sha256"], "权重指纹不符"
        assert fingerprint(chosen["config"]) == index[chosen["round_id"]]["config_sha256"] == chosen["config_sha256"], "入选配置指纹不符"
        manifest = read((BATCH/lock["weights"]).parent/"shortlist.json")
        entries = manifest["selected_before_r5"]
        assert fingerprint(entries) == lock["shortlist_sha256"], "R5 短名单指纹不符"
        canonical = reproduce_shortlist(index,manifest)
        assert {"baseline","install_rotation"} <= {r["kind"] for r in entries}, "短名单对照不全"
        assert datetime.fromisoformat(manifest["at"]) <= datetime.fromisoformat(lock["locked_at"]), "锁定时间顺序错误"
        ref = read(BATCH/"rounds/R00001/result.json")["r5"]["sequences"]
        expected = {r["name"]:r["frames"] for r in ref}
        assert set(r["name"] for r in chosen["r5_sequences"]) == set(expected) == set(lock["validation"]), "R5 记录不全"
        check_scores(chosen["r5_sequences"],chosen["r5"],expected)
        for series in chosen["r5_sequences"]:
            name = series["name"]
            check_frame_evidence(series,native[name] if name in native else shice_frame_evidence(name))
        proof["lock"] = {"id":chosen["round_id"],"at":lock["locked_at"],"sha256":key,"weights_sha256":lock["weights_sha256"],
                         "r5_deg":chosen["r5"]["macro_rmse_deg"],"shortlisted":len(entries),"shortlist_replay":canonical}
    except (AssertionError, KeyError, ValueError, OSError) as exc:
        proof["issues"].append("锁模证据未通过复核，" + (str(exc) or type(exc).__name__))
        return proof
    snapshots, record_paths = [], []
    for step in index.values():
        if step["config"]["kind"] != "final" or step["config"].get("selection_sha256") != key:
            continue
        folder = BATCH / "rounds" / step["round_id"]
        if (folder/"final-records-progress.json").is_file():
            record_paths.append(folder/"final-records-progress.json")
        for path in (folder/"result.json",folder/"final-progress.json"):
            if path.is_file():
                try:
                    data = read(path)
                    assert data.get("lock") == lock, "部分测试使用了其他锁模结果"
                    assert datetime.fromisoformat(data["started_at"]) >= datetime.fromisoformat(lock["locked_at"]), "部分测试早于锁模"
                    snapshots.append((data,path))
                except (AssertionError,KeyError,ValueError,OSError) as exc:
                    proof["issues"].append(f"{step['round_id']}/{path.name} 无法核验，" + (str(exc) or type(exc).__name__))
    if snapshots or record_paths:
        data = {}
        if snapshots:
            data,path = max(snapshots,key=lambda item:(len(item[0].get("datasets",{})),item[1].stat().st_mtime_ns))
            proof["source"] = str(path.relative_to(BATCH))
            if data.get("references"):
                try:
                    proof["baseline_source_sha256"] = check_baseline_references(data["references"])
                except (AssertionError,KeyError,TypeError,ValueError,OSError) as exc:
                    proof["issues"].append("正式轮引用的历史基线文件与开始时的记录不一致，"+(str(exc) or type(exc).__name__))
        existing = original_rows()
        journal = {ds:{} for ds in existing}
        proof["record_checkpoints"] = []
        for path in record_paths:
            try:
                receipt = read(path)
                recorded = checkpoint_records(receipt,lock,existing)
                for ds, by_name in recorded.items():
                    for name in by_name.keys() & journal[ds].keys():
                        compare_checkpoint_row(by_name[name],journal[ds][name])
                for ds, by_name in recorded.items():
                    journal[ds].update(by_name)
                proof["record_checkpoints"].append({"source":str(path.relative_to(BATCH)),"sha256":digest(path),
                    "at":receipt["at"],"counts":{ds:len(rows) for ds,rows in recorded.items()}})
            except (AssertionError,KeyError,TypeError,ValueError,OSError) as exc:
                proof["issues"].append(f"{path.parent.name} 逐序列检查点未通过核对，"+(str(exc) or type(exc).__name__))
        for ds, expected in existing.items():
            group = data.get("datasets",{}).get(ds)
            rows = dict(journal[ds])
            if group:
                try:
                    audit_partial_dataset(group,expected)
                    for row in group["sequences"]:
                        name = row["name"]
                        if name in rows:
                            try:
                                compare_checkpoint_row(row,rows[name])
                            except (AssertionError,KeyError,TypeError,ValueError) as exc:
                                proof["issues"].append(f"{ds}/{name} 的逐序列检查点与套件输出不一致，"+(str(exc) or type(exc).__name__))
                        rows[name] = row
                except (AssertionError,KeyError,TypeError,ValueError) as exc:
                    proof["issues"].append(ds+" 的套件汇总未通过复核，"+(str(exc) or type(exc).__name__))
            if rows:
                try:
                    checked = audit_partial_dataset({"sequences":list(rows.values())},expected)
                    names = set(rows)
                    if ds == "R6":
                        frame_proofs = {name:shice_frame_evidence(name) for name in names}
                    else:
                        from verify_external import external_frame_evidence
                        frame_proofs = external_frame_evidence(ds,names)
                    for row in rows.values():
                        check_frame_evidence(row,frame_proofs[row["name"]])
                    proof["tests"][ds] = {**checked, "raw_frame_indices":frame_proofs,
                                          "per_sequence":{n:r["rmse_deg"] for n,r in rows.items()}}
                except (AssertionError,KeyError,TypeError,ValueError,OSError) as exc:
                    proof["issues"].append(ds+" 的部分成绩未通过逐记录复核，"+(str(exc) or type(exc).__name__))
        if data.get("latency") and "R6" in proof["tests"]:
            from verify_final import check_latency
            try:
                check_latency(data,{n:r["scoring_frames"] for n,r in existing["R6"].items()})
                proof["latency_verified"] = True
                proof["latency"] = data["latency"]
            except (AssertionError,KeyError,ValueError) as exc:
                proof["issues"].append("同机计时未通过复核，"+(str(exc) or type(exc).__name__))
    external = BATCH / "tum-vi-independent.json"
    if external.is_file():
        import numpy as np
        from verify_tum import coverage_evidence, replay_rooms
        try:
            data = read(external)
            assert data["selection_lock_sha256"] == key and data["protocol"] == read(BATCH/"tum-vi-protocol.json"), "TUM-VI 锁模或协议不一致"
            assert datetime.fromisoformat(data["at"]) >= datetime.fromisoformat(lock["locked_at"]), "TUM-VI 时间早于锁模"
            rooms = data["results"]
            proof["tum_coverage"] = coverage_evidence(data,data["protocol"])
            proof["tum_failures"] = data["failures"]
            with np.load(BATCH/lock["weights"],allow_pickle=False) as src:
                weights = {k:src[k] for k in src.files}
            try:
                replays = replay_rooms(rooms,data["protocol"],chosen["config"],weights)
            except (AssertionError,KeyError,ValueError,OSError,RuntimeError):
                replays = []
                for room in rooms:
                    try:
                        replays.extend(replay_rooms([room],data["protocol"],chosen["config"],weights))
                    except (AssertionError,KeyError,ValueError,OSError,RuntimeError) as exc:
                        proof["issues"].append(room["name"]+" 的原始重放未通过，"+(str(exc) or type(exc).__name__))
            verified = {r["name"] for r in replays}
            proof["tum_verified_rooms"] = [{"name":r["name"],"frames":r["frames"],"locked_deg":r["rmse_deg"],"pure_deg":r["baseline_deg"]}
                                           for r in rooms if r["name"] in verified]
            proof["tum_raw_replays"] = replays
        except (AssertionError,KeyError,ValueError,OSError,RuntimeError) as exc:
            proof["issues"].append("TUM-VI 部分成绩未通过复核，"+(str(exc) or type(exc).__name__))
    return proof


def finish_partial(report, tally, status):
    index = {r["round_id"]:r for r in read(BATCH/"round_index.json")}
    proof = closing_evidence(index,status)
    proof["reviewed_utc"] = tally["reviewed_utc"]
    write(HERE/"window_close_audit.json",proof)
    story = report["report"]["review"]
    window = tally["timing_through_last_completed"]
    first = datetime.fromisoformat(window["first_closed_attempt_started"])
    last = datetime.fromisoformat(window["latest_completed_round_ended"])
    story["summary"] = (f"## 到期审查\n\n十小时窗口已结束，正式测试轮未完成。已核对 **{len(tally['completed'])} 个完成轮号**，"
                        f"其中 {tally['development_candidate_runs']} 轮直接训练或评分候选。计及配对试验中登记的候选，共核对 {tally['distinct_development_settings_excluding_seed']} 种不同配置。")
    story["counts"] += (f" 已完成轮中的额外尝试有 {tally['completed_attempts_beyond_first']} 次。"
                        f"有完整账本的尝试从 {first:%m-%d %H:%M:%S} 到 {last:%m-%d %H:%M:%S}，"
                        f"首尾相距 {(last-first).total_seconds()/3600:.2f} 小时，合并重叠后的记录时长为 {window['wall_seconds_in_union_of_closed_attempts']/3600:.2f} 小时。")
    outcome = proof["lock"]
    availability = (f"锁定配置为 `{outcome['id']}`，R5 倾角误差 {outcome['r5_deg']:.3f}°。"
                    if outcome else "本批没有可核验的锁定配置。")
    for item in proof["r5_progress"]:
        availability += (f" 选模轮 {item['round']} 的最后检查点在 {datetime.fromisoformat(item['at']):%m-%d %H:%M:%S}。"
                         f"{item['planned']} 个入围候选中，{len(item['scored'])} 个取得全部 R5 记录的逐帧索引与分数，"
                         f"{len(item['failed'])} 个留下失败记录，{len(item['unreported'])} 个尚无完整记录。")
    for ds in ("R6","BROAD","Sassari"):
        result = proof["tests"].get(ds)
        if result:
            availability += (f" {ds} 取得 {result['records']}/{result['planned']} 条的同帧成绩，锁定配置 {result['means']['locked']:.3f}°、"
                             f"纯 FullVQF {result['means']['full_vqf']:.3f}°、原轻量头 {result['means']['physical_only']:.3f}°、"
                             f"原 TimesFM {result['means']['timesfm']:.3f}°、原 GRU {result['means']['gru']:.3f}°。")
        else:
            availability += f" {ds} 未取得可核验的锁模后成绩。"
    rooms = proof["tum_verified_rooms"]
    availability += f" TUM-VI 核实 {len(rooms)}/6 个房间。"
    if "tum_coverage" in proof:
        coverage = proof["tum_coverage"]["counts"]
        availability += (f" 运行清单登记成功 {coverage['success']} 个、执行失败 {coverage['failed']} 个、"
                         f"截止前未启动 {coverage['not_started_before_cutoff']} 个、尚无终态 {coverage['pending']} 个。")
    if rooms:
        availability += f" 已评分房间等权平均为 {fmean(r['locked_deg'] for r in rooms):.3f}°，纯 FullVQF 为 {fmean(r['pure_deg'] for r in rooms):.3f}°。"
        qualified = [r["quiet_gravity_check"]["qualifying_blocks"] for r in proof["tum_raw_replays"]]
        availability += f" 其中 {sum(n > 0 for n in qualified)} 个房间含满足低动态方向核对条件的完整秒窗，共 {sum(qualified)} 个。"
    availability += " 同机延迟已有核验记录。" if proof["latency_verified"] else " 同机延迟缺少完整可核验记录。"
    if proof["issues"]:
        availability += " " + " ".join(proof["issues"])
    story["summary"] += "\n\n" + availability
    chart = hourly_rows(tally,index)
    report["queries"]["by_hour"] = table_query(chart,"到期时的实际轮次",["round_audit.json","window_close_audit.json"],"以轮号结束时间统计。")
    story["extraSections"].insert(0,{"id":"window-hours","title":"各时段实际完成轮数","queryId":"by_hour",
        "body":f"实验进程最后一次心跳是 {datetime.fromisoformat(status['heartbeat_at']):%m-%d %H:%M:%S}，状态为 `{status['phase']}`。",
        "columns":[{"field":"period","label":"北京时间"},{"field":"total","label":"全部完成"},{"field":"candidates","label":"其中候选"},{"field":"best","label":"累计最佳开发 RMSE"}]})
    report["buildStatus"] = "complete"
    return report
