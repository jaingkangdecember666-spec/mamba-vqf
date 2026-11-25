"""Check the evidence behind each R5 nomination independently of the shortlist builder."""

from collections import Counter, defaultdict
from datetime import datetime
from hashlib import sha256
from statistics import fmean, median

from audit_rounds import BATCH, fingerprint, parameter_identity, read


def verify_wording_change(index, frames):
    receipt = read(BATCH/"hybrid-fallback-shortlist-reason-audit.json")
    before_text, after_text = (receipt["reason_" + side] for side in ("before","after"))
    after_source = (BATCH/"review"/"selection-rules-070016.txt").read_bytes()
    assert after_source.count(after_text.encode()) == 1 and before_text.encode() not in after_source
    before_source = after_source.replace(after_text.encode(),before_text.encode())
    assert sha256(before_source).hexdigest() == receipt["selector_source_before_sha256"] == "1b207638565b496c21007af7ba36d758d447c4f3719f4e0cb67674365d8196f6"
    assert sha256(after_source).hexdigest() == receipt["selector_source_after_sha256"] == "ad5e7e798001be2da3fadfc257175cc0e4993a63b4b36c9ff0671283e5ab7fd4"
    at = datetime.fromisoformat(receipt["comparison_snapshot_at"])
    registered = [r for r in index.values() if datetime.fromisoformat(r["planned_at"]) <= at]
    assert len(registered) == receipt["comparison_snapshot_rows"]
    snapshot = [r for r in registered if r["status"]=="completed" and r["config"]["kind"] not in {"final","selection"}
                and datetime.fromisoformat(r["ended_at"]) <= at]
    def historical_read(path):
        if path.name == "stream-gru-development-summary.json":
            # The only GRU field used by the archived selector was empty in the 02:00:55 filing.
            summary = read(path)
            if datetime.fromisoformat(summary["updated_at"]) <= at:
                assert sha256(path.read_bytes()).hexdigest() == "aef1efb6d48c12381a039df28c4bd1dc67e87d40e4373aa826bdc6052517e275"
                assert summary["r5_approved"] == []
            return {"r5_approved":[]}
        assert path.name == "result.json" and path.parent.name in ids
        return read(path)
    ids = {r["round_id"] for r in snapshot}
    variants = []
    for source in (before_source,after_source):
        module = {}
        exec(compile(source,"selection-rules-070016","exec"),module)
        module["read_json"] = historical_read
        variants.append(module["build_shortlist"](snapshot))
    old,new = variants
    assert [r["round_id"] for r in old] == receipt["candidate_ids_before"] == receipt["candidate_ids_after"] == [r["round_id"] for r in new]
    modified = [a["round_id"] for a,b in zip(old,new) if a != b]
    assert modified == ["H00010"] and before_text in next(r["reasons"] for r in old if r["round_id"]=="H00010")
    assert [{**r,"reasons":[after_text if t==before_text else t for t in r["reasons"]]} for r in old] == new
    previous = read(BATCH/"shortlist-preview.json")
    assert datetime.fromisoformat(previous["created_at"]) < at < datetime.fromisoformat(receipt["effective_at"])
    assert any(r["round_id"]=="H00010" and before_text in r["reasons"] for r in previous["candidates"])
    gate = index["H00010"]
    assert sha256((BATCH/gate["result"]).read_bytes()).hexdigest() == receipt["H00010"]["result_sha256"]
    ratio = fallback_coverage(gate,frames)
    assert ratio["pooled_fraction"] < receipt["threshold_fraction"] == .01 <= ratio["macro_fraction"]
    audit = {"at":receipt["comparison_snapshot_at"],"effective_at":receipt["effective_at"],"snapshot_rows":len(registered),
            "completed_development_rows":len(snapshot),"candidate_count":len(old),"changed_reasons_for":modified,"coverage":ratio,
            "before_entries_sha256":fingerprint(old),"after_entries_sha256":fingerprint(new),"id_order_sha256":fingerprint(receipt["candidate_ids_before"])}
    raw_path = BATCH/"review"/"gate_fallback_raw_replay.json"
    if raw_path.is_file():
        raw = read(raw_path)
        sources = {r["name"]:r for r in read(BATCH/gate["result"])["sequences"]}
        assert raw["source_result_sha256"] == receipt["H00010"]["result_sha256"]
        assert len(raw["records"]) == len(frames) and {r["name"]:r["frames"] for r in raw["records"]} == frames
        for row in raw["records"]:
            expected = sources[row["name"]]
            assert row["fallback_frames"] == expected["default_fallback_frames"]
            assert abs(row["rmse_deg"]-expected["rmse_deg"]) == row["abs_rmse_difference_deg"] < 1e-5
        assert all(abs(raw[key]-ratio[key])<1e-12 for key in ("frames","fallback_frames","macro_fraction","pooled_fraction"))
        audit["native_raw_replay"] = {"records":len(raw["records"]),"frames":raw["frames"],
                                      "max_abs_rmse_difference_deg":max(r["abs_rmse_difference_deg"] for r in raw["records"])}
    return audit


def wording_at(entries, when):
    receipt = read(BATCH/"hybrid-fallback-shortlist-reason-audit.json")
    if when >= datetime.fromisoformat(receipt["effective_at"]):
        return entries
    return [{**r,"reasons":[receipt["reason_before"] if t==receipt["reason_after"] else t for t in r["reasons"]]} for r in entries]


def metric(row):
    data = row["dev_metrics"]
    if row["config"]["kind"] == "baseline":
        return fmean(r["rmse_deg"] for r in data["sequences"])
    score = fmean(data[axis]["macro_rmse_deg"] for axis in ("device","motion"))
    assert abs(score-data["selection_rmse_deg"]) < 1e-8
    return score


def families(rows, minimum):
    groups = defaultdict(dict)
    for row in rows:
        key = parameter_identity(row["config"])
        assert row["seed"] not in groups[key]
        groups[key][row["seed"]] = row
    return [list(seeds.values()) for seeds in groups.values() if len(seeds) >= minimum]


def middle(rows):
    m = median(metric(r) for r in rows)
    return min(rows,key=lambda r:(abs(metric(r)-m),r["round_id"]))


def fallback_coverage(row, frames):
    result = read(BATCH/row["result"])
    sequences = result["sequences"]
    assert isinstance(sequences,list) and len(sequences) == len(frames)
    assert {s["name"]:s["frames"] for s in sequences} == frames
    assert all(isinstance(s["default_fallback_frames"],int) and 0<=s["default_fallback_frames"]<=s["frames"] for s in sequences)
    total, hits = sum(frames.values()), sum(s["default_fallback_frames"] for s in sequences)
    macro = fmean(s["default_fallback_frames"]/s["frames"] for s in sequences)
    assert abs(macro-result["mean_default_fallback_fraction"]) < 1e-12
    return {"macro_fraction":macro,"pooled_fraction":hits/total,"fallback_frames":hits,"frames":total,"record_count":len(frames)}


def audit(available, shortlist, frames, cutoff):
    by_kind = defaultdict(list)
    eligible = {}
    for r in sorted(available,key=lambda r:r["round_id"]):
        kind = r["config"]["kind"]
        if kind not in {"baseline","filter","install_rotation","ridge","tcn","legacy_head","hybrid","equivariant","ensemble","stream_gru","bias_physical_head"}:
            continue
        if kind in {"ensemble","bias_physical_head"} and read(BATCH/r["result"]).get("eligible_for_r5") is not True:
            continue
        if kind in {"ridge","tcn","equivariant"} and r["config"]["context"] > 64:
            if read(BATCH/r["result"]).get("mask_version") != "causal-bin-v2-full-window":
                continue
        if kind == "stream_gru":
            context = r["config"].get("recurrent_training_context")
            if context != "split-state-at-forbidden-context-v2" or read(BATCH/r["result"]).get("recurrent_training_context") != context:
                continue
        assert r["status"] == "completed" and datetime.fromisoformat(r["ended_at"]) <= cutoff
        eligible[r["round_id"]] = r
        by_kind[kind].append(r)
    for rows in by_kind.values():
        rows.sort(key=lambda r:(metric(r),r["round_id"]))
    claims = defaultdict(dict)
    def add(row,reason,explanation,**proof):
        rid = row["round_id"]
        assert reason not in claims[rid]
        claims[rid][reason] = {"explanation":explanation,**proof}

    pure, = by_kind["baseline"]
    add(pure,"native pure FullVQF recomputed in this batch","纯 FullVQF 基准复算")
    ranks = (("install_rotation",2,"best fixed installation rotations by grouped development","固定安装旋转"),
             ("filter",3,"best unlearned filters by grouped development","滤波器"),
             ("ridge",4,"best residual candidates by combined development macro RMSE","岭回归"))
    for kind,count,reason,label in ranks:
        rows = by_kind[kind]
        for rank,r in enumerate(rows[:count],1):
            add(r,reason,f"{label}开发均值第 {rank}/{len(rows)}",rank=rank,pool=len(rows))
    ridge = by_kind["ridge"]
    for axis,label in (("device","留设备"),("motion","留程序")):
        if ridge:
            winner = min(ridge,key=lambda r:(r["dev_metrics"][axis]["macro_rmse_deg"],r["round_id"]))
            add(winner,f"best residual under {axis} holdout",f"岭回归{label}误差最低",pool=len(ridge),rmse_deg=winner["dev_metrics"][axis]["macro_rmse_deg"])
    physical = [r for r in ridge if not r["config"]["embedding"]]
    encoded = [r for r in ridge if r["config"]["embedding"]]
    if physical:
        add(physical[0],"best physical-only readout","物理特征岭回归最低",pool=len(physical))
    if encoded:
        add(encoded[0],"best frozen TimesFM readout","冻结 TimesFM 特征岭回归最低",pool=len(encoded))
    def family_proof(group):
        return {"family_mean_deg":fmean(metric(r) for r in group),"seeds":{str(r["seed"]):r["round_id"] for r in group}}
    def ordered_families(rows):
        return sorted(families(rows,3),key=lambda g:fmean(metric(r) for r in g))
    tcns = ordered_families(by_kind["tcn"])
    if tcns and (not physical or fmean(map(metric,tcns[0])) <= metric(physical[0])+.15):
        g = tcns[0]
        add(middle(g),"median-development seed of a competitive, replicated causal TCN family","重复 TCN 配置的中位种子",**family_proof(g))
    replicated = ordered_families([r for r in physical if r["config"]["hidden"]])
    original = next((g for g in replicated if any(r["round_id"]=="R00193" for r in g)),None)
    for g,label,name in ((replicated[0] if replicated else None,"best replicated random-feature ridge","重复家族均值最低"),
                         (original,"original leading R00193 random-feature ridge","原 R00193 家族")):
        if g:
            evidence = family_proof(g)
            add(middle(g),f"median-development seed of the {label} family",f"{name}的中位种子",**evidence)
            add(min(g,key=metric),f"best individual seed of the {label} family",f"{name}的最佳种子",**evidence)
    legacy = ordered_families([r for r in by_kind["legacy_head"] if r["batch"] in {"legacy-causal-head-screen","legacy-causal-head-seed-repeat"}])
    for g in legacy[:2]:
        evidence = family_proof(g)
        if evidence["family_mean_deg"] <= fmean(map(metric,legacy[0]))+.15:
            add(middle(g),f"median seed of an original causal physical head family averaging {evidence['family_mean_deg']:.6f} deg",
                "原轻量头重复家族的中位种子",**evidence)

    joint = defaultdict(dict)
    for trial in sorted(available,key=lambda r:r["round_id"]):
        cfg = trial["config"]
        if cfg["kind"] != "joint" or cfg.get("audit_only") or cfg.get("matched_without_embedding"):
            continue
        parent = eligible.get(cfg["parent_round"])
        if not parent:
            continue
        assert cfg["parent_config_sha256"] == parent["config_sha256"]
        seed = parent["seed"]
        family = joint[parameter_identity(parent["config"])]
        value = trial["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]
        if seed in family:
            assert family[seed][0]["round_id"] == parent["round_id"] and abs(family[seed][1]-value)<1e-8
        family[seed] = (parent,value,trial["round_id"])
    ranked = sorted(joint.values(),key=lambda g:(fmean(r[1] for r in g.values()),min(r[2] for r in g.values())))
    kinds = set()
    for g in ranked:
        parents,values,rounds = zip(*g.values())
        selected = middle(parents)
        kind = selected["config"]["kind"]
        if kind in kinds:
            continue
        mean = fmean(values)
        ids = sorted(rounds)
        add(selected,f"leading joint unseen-device and unseen-program family for {kind}, {mean:.6f} deg across {', '.join(ids)}",
            f"该方法联合留出均值最低（{mean:.4f}°）",family_joint_deg=mean,rounds=ids,**family_proof(parents))
        kinds.add(kind)
        if len(kinds)==3:
            break
    recurrent_file = BATCH/"stream-gru-development-summary.json"
    if recurrent_file.is_file():
        report = read(recurrent_file)
        if datetime.fromisoformat(report["updated_at"]) <= cutoff:
            for approved in report["r5_approved"]:
                if approved["round_id"] in eligible:
                    add(eligible[approved["round_id"]],"replicated correctly segmented recurrent residual with joint memory benefit","循环模型通过联合留出和种子复验")
    gated = [r for r in by_kind["hybrid"] if r["config"]["mode"]=="gated"]
    coverage = {r["round_id"]:fallback_coverage(r,frames) for r in gated}
    if gated:
        add(gated[0],"best development error among default-fast causal gates","门控滤波开发均值最低",pool=len(gated))
        nearby = [r for r in gated if metric(r) <= metric(gated[0])+.1]
        safest = min(nearby,key=lambda r:(r["dev_metrics"]["device"]["worst_delta_deg"],metric(r)))
        add(safest,"best worst-record delta among competitive default-fast causal gates","相近门控配置中最差记录增量最低",pool=len(nearby))
        observed = [r for r in gated if coverage[r["round_id"]]["macro_fraction"] >= .01]
        if observed:
            c = coverage[observed[0]["round_id"]]
            add(observed[0],"fallback_coverage",f"逐记录回退比例均值 {c['macro_fraction']:.3%}，达标门控中误差最低",**c)
    for row in by_kind["ensemble"]:
        add(row,"predeclared equal-weight ensemble met joint-gain and single-axis safety criteria","等权组合通过联合留出和单项门槛")
    for row in by_kind["bias_physical_head"]:
        add(row,"record-local causal accelerometer offset passed three matched development seeds","在线加速度计零偏通过三种子配对",seed=row["seed"])
    inv = by_kind["equivariant"]
    if inv and (not physical or metric(inv[0]) <= metric(physical[0])+.15):
        add(inv[0],"competitive invariant physical correction","旋转等变修正达到开发误差门槛")
    assert len(shortlist)==len({s["round_id"] for s in shortlist}) and set(claims)=={s["round_id"] for s in shortlist}
    output = []
    for entry in shortlist:
        rid = entry["round_id"]
        reasons = ["fallback_coverage" if "fallback" in t and "1%" in t else t for t in entry["reasons"]]
        assert Counter(reasons) == Counter(claims[rid].keys()), (rid,reasons,list(claims[rid]))
        assert abs(metric(eligible[rid])-entry["development_rmse_deg"]) < 1e-8
        output.append({**entry,"evidence":[claims[rid][key] for key in reasons]})
    return {"entries":output,"reason_count":sum(len(v) for v in claims.values()),"fallback_coverage":coverage,
            "entry_ids_sha256":fingerprint([r["round_id"] for r in output]),"eligible_counts":{k:len(v) for k,v in by_kind.items()}}
