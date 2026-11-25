"""Independently reconcile a snapshot of the worker's round index with per-round evidence."""

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

from audit_timing import timing_summary


HERE = Path(__file__).resolve().parent
BATCH = HERE.parent

if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes
    win32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_file = win32.CreateFileW
    open_file.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                          wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    open_file.restype = wintypes.HANDLE


def read_bytes_shared(path):
    """Allow the supervisor to atomically replace an index while the reviewer reads it."""
    if os.name != "nt":
        return path.read_bytes()
    generic_read, share_read_write_delete, open_existing = 0x80000000, 7, 3
    handle = open_file(str(path), generic_read, share_read_write_delete, None, open_existing, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except Exception:
        win32.CloseHandle(wintypes.HANDLE(handle))
        raise
    with os.fdopen(fd, "rb") as stream:
        return stream.read()


def read(path):
    return json.loads(read_bytes_shared(path))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write(path, value):
    staging = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staging.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def parameter_identity(config):
    """Count settings independently of the seed for an otherwise identical run."""
    parameters = {k:v for k,v in config.items() if k not in {"seed", "random_seed"}}
    if config["kind"] == "ensemble":
        # The source and joint round IDs refer to runs that vary with the member seed.
        members = [(parameter_identity(m["config"]),m["weight"]) for m in config["members"]]
        parameters["members"] = sorted(members)
    if config["kind"] == "bias_physical_head":
        parameters["head"] = parameter_identity(config["head"])
        parameters.pop("matched_experiment_sha256",None)
        parameters.pop("paired_zero_round",None)
    return fingerprint(parameters)


def ridge_seed_summary(candidates):
    seeds = (20261122, 20261123, 20261124)
    settings = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        cfg = row["config"]
        if cfg["kind"] == "ridge" and cfg["hidden"] > 0 and not cfg["embedding"]:
            settings[parameter_identity(cfg)][cfg["seed"]].append(row)
    matched, wins = [], Counter()
    for key, runs in sorted(settings.items()):
        if not all(seed in runs for seed in seeds):
            continue
        scores = [fmean(r["score_deg"] for r in runs[seed]) for seed in seeds]
        lowest = [seed for seed, score in zip(seeds, scores) if abs(score - min(scores)) <= 1e-9]
        wins[lowest[0] if len(lowest) == 1 else "tie"] += 1
        matched.append({"settings_sha256":key, "rounds_by_seed":[[r["id"] for r in runs[s]] for s in seeds],
                        "scores_deg":scores, "range_deg":max(scores)-min(scores)})
    return {"seeds":list(seeds), "settings_with_random_features":len(settings), "complete_common_seed_sets":len(matched),
            "median_within_setting_range_deg":median(r["range_deg"] for r in matched) if matched else None,
            "seed_means_deg":[fmean(r["scores_deg"][i] for r in matched) for i in range(len(seeds))] if matched else [],
            "unique_lowest_counts":[wins[s] for s in seeds], "ties":wins["tie"], "matched":matched}


def check_scores(records, metrics, expected_frames):
    names = {r["name"] for r in records}
    assert len(records) == len(names) == metrics["records"]
    for r in records:
        assert r["frames"] == expected_frames[r["name"]], r["name"]
        assert math.isfinite(r["rmse_deg"]) and 0 <= r["rmse_deg"] <= 180
    assert sum(r["frames"] for r in records) == metrics["scoring_frames"]
    assert math.isclose(fmean(r["rmse_deg"] for r in records), metrics["macro_rmse_deg"], abs_tol=1e-7)
    assert math.isclose(max(r["rmse_deg"] for r in records), metrics["worst_rmse_deg"], abs_tol=1e-7)


def check_fold(fold, axis, expected, fit_free=False):
    train, valid = set(fold["training"]), set(fold["validation"])
    assert (train or fit_free) and valid and train.isdisjoint(valid)
    assert len(train) == len(fold["training"]) and len(valid) == len(fold["validation"])
    assert axis in {"device","motion","joint_holdout"}
    example_device, example_program = next(iter(valid)).split("-")
    groups = ((1,5,9),(2,6,10),(3,7),(4,8))
    group = next(g for g in groups if int(example_program) in g)
    same_device = {n for n in expected if n.split("-")[0] == example_device}
    same_program = {n for n in expected if int(n.split("-")[1]) in group}
    motion_label = f"M{groups.index(group)+1}"
    partitions = {"device":(same_device,expected-same_device,example_device),
                  "motion":(same_program,expected-same_program,motion_label),
                  "joint_holdout":(same_device & same_program,expected-(same_device | same_program),example_device+"/"+motion_label)}
    held, permitted, label = partitions[axis]
    assert valid == held and (train == permitted or (fit_free and not train)), (axis,label)
    if "fold" in fold:
        assert fold["fold"] == label
    return valid


def check_device_weights(config, folds):
    mode = config.get("training_device_balance")
    if not mode:
        return
    for fold in folds:
        counts = Counter(name.split("-")[0] for name in fold["training"])
        fit = fold["fitting"]
        assert fit["record_count_per_training_device"] == dict(counts)
        epochs = fit["group_balance_training_only"]
        assert len(epochs) == config["epochs"]
        for step in epochs:
            mass = step["device_objective_mass"]
            assert set(mass) == set(counts) and min(mass.values()) > 0
            assert math.isclose(sum(mass.values()),1.0,abs_tol=1e-7)
            cap = config.get("max_worst_device_multiplier",1) if mode == "capped_worst" else 1
            assert max(mass.values()) / min(mass.values()) <= cap + 1e-7
            if "train_device_residual_loss" in step:
                losses = step["train_device_residual_loss"]
                assert set(losses) == set(counts)
                hardest = min(losses,key=lambda name:(-losses[name],name))
                assert step["most_difficult_training_device"] == hardest


def device_balance_qualification(index, completed_ids):
    plan_path = BATCH / "device-balance-development-stage.json"
    summary_path = BATCH / "device-balance-development-summary.json"
    if not plan_path.is_file() or not summary_path.is_file():
        return {}
    plan, published = read(plan_path), read(summary_path)
    assert published["stage"] == plan
    if any(r["round_id"] not in completed_ids for r in plan["rounds"]):
        return {}
    groups = defaultdict(list)
    for spec in plan["rounds"]:
        candidate, grouped, joint = (index[spec[k]] for k in ("round_id","grouped_control","joint_control"))
        assert datetime.fromisoformat(plan["planned_at"]) <= datetime.fromisoformat(candidate["started_at"])
        assert candidate["seed"] == grouped["seed"] == joint["seed"] == spec["seed"]
        assert candidate["config"]["training_device_balance"] == spec["mode"]
        assert joint["config"]["parent_config_sha256"] == grouped["config_sha256"]
        assert {k:v for k,v in candidate["config"].items() if k not in {"training_device_balance","max_worst_device_multiplier"}} == grouped["config"]
        new, old = candidate["dev_metrics"]["device"],grouped["dev_metrics"]["device"]
        worst_change = max(d["macro_rmse_deg"] for d in new["per_device"].values()) - max(d["macro_rmse_deg"] for d in old["per_device"].values())
        groups[spec["mode"]].append({"id":candidate["round_id"],"seed":spec["seed"],"worst_device_delta":worst_change,
                    "device_delta":new["macro_rmse_deg"]-old["macro_rmse_deg"],
                    "joint_delta":candidate["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]-joint["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]})
    limits = plan["promotion"]
    summary = {}
    for mode, rows in groups.items():
        assert len(rows) == len({r["seed"] for r in rows}) == 3
        joint_delta = fmean(r["joint_delta"] for r in rows)
        joint_wins = sum(r["joint_delta"]<0 for r in rows)
        device_delta = fmean(r["device_delta"] for r in rows)
        worst_delta = fmean(r["worst_device_delta"] for r in rows)
        qualifies = (joint_delta <= -limits["required_mean_joint_rmse_improvement_deg"] and
                     joint_wins >= limits["minimum_joint_paired_wins"] and device_delta <= limits["max_mean_leave_device_degradation_deg"])
        reference = published["families"][mode]
        assert reference["qualifies_for_r5"] == qualifies and reference["observed_seed_count"] == len(rows)
        assert reference["axes"]["joint_holdout"]["paired_wins"] == joint_wins
        for v,axis,field in ((joint_delta,"joint_holdout","paired_mean_delta_deg"),(device_delta,"device","paired_mean_delta_deg"),
                             (worst_delta,"device","mean_worst_device_delta_deg")):
            assert math.isclose(v,reference["axes"][axis][field],abs_tol=1e-10)
        summary[mode] = {"rounds":[r["id"] for r in rows],"seeds":len(rows),"joint_delta_deg":joint_delta,"wins":joint_wins,
                         "device_delta_deg":device_delta,"worst_device_delta_deg":worst_delta,"qualified":qualifies,
                         "needed_improvement_deg":limits["required_mean_joint_rmse_improvement_deg"]}
    return summary


def check_development(result, expected, frames):
    scores = result["dev_metrics"]
    rows = result["sequences"]
    if isinstance(rows, list):
        assert result.get("fixed_before_development") is True and "folds" not in result
    axes = ["device", "motion"]
    if "joint_holdout" in scores:
        axes.append("joint_holdout")
    for axis in axes:
        seq = rows if isinstance(rows, list) else rows[axis]
        assert {r["name"] for r in seq} == expected
        check_scores(seq, scores[axis], frames)
    assert math.isclose(fmean(scores[a]["macro_rmse_deg"] for a in ("device", "motion")),
                        scores["selection_rmse_deg"], abs_tol=1e-7)
    if "folds" in result:
        seen = {axis:[] for axis in axes}
        for fold in result["folds"]:
            axis = fold["axis"]
            seen[axis].extend(check_fold(fold, axis, expected))
        assert all(len(items) == len(expected) and set(items) == expected for items in seen.values())
        if "timing" in result:
            totals = result["timing"]
            stages = ("prepare_seconds","fit_seconds","score_seconds")
            for stage in stages:
                parts = [f["timing"][stage] for f in result["folds"]]
                assert all(t >= 0 for t in parts) and math.isclose(sum(parts),totals[stage],abs_tol=1e-6)
            assert sum(totals[k] for k in stages) <= totals["total_wall_seconds"] + 0.01


def check_ensemble(config, result):
    protocol = result["protocol"]
    assert [m["weight"] for m in config["members"]] == protocol["fixed_weights"]
    assert math.isclose(sum(protocol["fixed_weights"]), 1.0)
    registered = config.get("independent_filters",False) or config.get("shared_head_seed_ensemble",False)
    previous_folds = {}
    for member in config["members"]:
        source = read(BATCH / "rounds" / member["round_id"] / "round.json")
        assert source["config_sha256"] == fingerprint(member["config"])
        prior_result = read(BATCH / source["result"])
        prior = prior_result["sequences"]
        joint = read(BATCH / "rounds" / member["joint_round"] / "result.json")
        assert joint["parent_round"] == member["round_id"]
        if registered:
            previous_folds[member["round_id"]] = {(f["axis"],f["fold"]):f for f in prior_result["folds"]}
            previous_folds[member["round_id"]].update({("joint_holdout",f["fold"]):f for f in joint["folds"]})
        joint_rows = joint["sequences"]
        comparisons = {**prior,"joint_holdout":joint_rows["joint_holdout"] if isinstance(joint_rows,dict) else joint_rows}
        for axis, rows in result["sequences"].items():
            originals = {r["name"]:r for r in comparisons[axis]}
            for row in rows:
                old = originals[row["name"]]
                assert old["frames"] == row["frames"]
                gap = abs(old["rmse_deg"] - row["members_rmse_deg"][member["round_id"]])
                assert gap <= 1e-4
                if registered:
                    identity = member["round_id"]
                    assert old["rmse_deg"] == row["historical_member_rmse_deg"][identity]
                    assert math.isclose(gap,row["member_reproduction_abs_delta_deg"][identity],abs_tol=1e-12)
    deltas = {}
    for axis, rows in result["sequences"].items():
        better = min(fmean(r["members_rmse_deg"][m["round_id"]] for r in rows) for m in config["members"])
        deltas[axis] = fmean(r["rmse_deg"] for r in rows) - better
        assert math.isclose(deltas[axis], result["relative_to_best_individual"][axis], abs_tol=1e-8)
    qualifies = (deltas["joint_holdout"] <= -protocol["r5_inclusion_min_joint_improvement_deg"]
                 and all(deltas[a] <= protocol["r5_inclusion_max_single_axis_degradation_deg"] for a in ("device","motion")))
    if registered:
        from verify_mixed_ensemble import check_fixed_ensemble
        return check_fixed_ensemble(config,result,previous_folds,qualifies)
    assert qualifies == result["eligible_for_r5"]


def check_recurrent(config, result, frames):
    assert config["recurrent_training_context"] == result["recurrent_training_context"]
    reference = {r["name"]:r for r in read(HERE / "recurrent_context_audit.json")["records"]}
    assert set(result["sequences"]) == set(result["state_controls"]) == {"device","motion","joint_holdout"}
    for fold in result["folds"]:
        names = fold["training"]
        counts = fold["training_label_counts"]
        audit = fold["training_log"]["training_context_audit"]
        assert set(counts) == set(names)
        assert [counts[n] for n in names] == audit["per_record_label_counts"]
        assert sum(counts.values()) == audit["labels"]
        assert sum(audit["segments_per_record"]) == audit["independent_state_segments"]
        if config["update_hz"] == 10 and config["context"] == 64:
            assert audit["per_record_label_counts"] == [reference[n]["eligible_updates"] for n in names]
            segments = [reference[n]["disjoint_context_periods_containing_eligible_updates"] for n in names]
            assert audit["segments_per_record"] == segments
    for axis, rows in result["sequences"].items():
        controls = result["state_controls"][axis]
        assert set(controls) == {"continuous","reset_2_seconds","reset_each_update"}
        for mode, score in controls.items():
            measurements = [{**r,"rmse_deg":r["same_weight_state_controls_deg"][mode]} for r in rows]
            check_scores(measurements,score,frames)
        assert all(math.isclose(r["rmse_deg"],r["same_weight_state_controls_deg"]["continuous"],abs_tol=1e-8) for r in rows)


def check_regimes(config, result, frames):
    parent = read(BATCH / "rounds" / config["parent_round"] / "round.json")
    assert result["parent_config_sha256"] == parent["config_sha256"] == config["parent_config_sha256"]
    protocol = result["protocol"]
    assert fingerprint(protocol) == config["protocol_sha256"]
    assert protocol["common_scored_frames_preserved"] and result["augmentation_or_masks_changed"] is False
    source = read(BATCH / parent["result"])["sequences"]
    assert set(result["comparisons"]) == {"device", "motion"}
    for axis, rows in result["comparisons"].items():
        original = {r["name"]:r["rmse_deg"] for r in source[axis]}
        assert len(rows) == len(frames) and {r["name"] for r in rows} == set(frames)
        for row in rows:
            name = row["name"]
            assert row["frames"] == frames[name] and math.isclose(row["original_rmse_deg"],original[name],abs_tol=1e-7)
            assert abs(row["rmse_deg"]-original[name]) <= protocol["reconstruction_rmse_tolerance_deg"]
            for groups in row["slices"].values():
                assert sum(g["frames"] for g in groups.values()) == frames[name]
                total = sum(g["squared_error"] for g in groups.values())
                assert math.isclose(math.sqrt(total/frames[name]),row["rmse_deg"],abs_tol=1e-7)


def check_accelerometer(config, result, frames):
    parent = read(BATCH / "rounds" / config["mount_control_round"] / "round.json")
    joint = read(BATCH / "rounds" / config["joint_mount_control_round"] / "round.json")
    assert parent["config_sha256"] == config["mount_control_config_sha256"] == joint["config"]["parent_config_sha256"]
    assert fingerprint(result["fit_protocol"]) == config["protocol_sha256"]
    protocol = result["fit_protocol"]
    minimum = config.get("minimum_online_poses",protocol["minimum_distinct_poses"])
    assert result.get("online_minimum_distinct_poses",minimum) == minimum
    assert result.get("fixed_minimum_pooled_pose_clusters",protocol["minimum_distinct_poses"]) == protocol["minimum_distinct_poses"]
    assert result["unchanged_frame_masks_and_mounting"] and result["fixed_fits_use_only_current_training_fold"]
    original = dict(read(BATCH / parent["result"])["sequences"])
    jrows = read(BATCH / joint["result"])["sequences"]
    original["joint_holdout"] = jrows["joint_holdout"] if isinstance(jrows,dict) else jrows
    audits = result["online_audit"]
    assert set(audits) == set(frames) and result["eligible_development_records"] == len(frames)
    for name, a in audits.items():
        assert frames[name] == a["scored_frames"] and 0 <= a["scored_frames_with_nonzero_estimate"] <= frames[name]
        assert (a["first_applied_native_index"] is None) == (a["applied_original_packets"] == 0)
        if a["first_applied_native_index"] is None:
            assert a["scored_frames_with_nonzero_estimate"] == 0
        for update in a["eligible_updates"]:
            assert update["pose_count"] >= minimum and update["static_blocks"] >= protocol["minimum_static_blocks"]
            assert update["orientation_cov_eigenvalues"][0] >= protocol["minimum_orientation_cov_eigenvalue"]
            assert update["orientation_condition_number"] <= protocol["maximum_orientation_condition"]
            gain = min(update["pose_count"]/(update["pose_count"]+protocol["online_shrink_pose_pseudocount"]),protocol["online_max_shrink_multiplier"])
            assert math.isclose(gain,update["multiplier"],abs_tol=1e-9)
            assert all(math.isclose(raw*gain,used,abs_tol=1e-9) for raw,used in zip(update["offset_m_s2"],update["shrunk_offset_m_s2"],strict=True))
    assert result["online_records_with_support"] == sum(a["first_applied_native_index"] is not None for a in audits.values())
    for axis, controls in original.items():
        rows = result["sequences"][axis]
        refs = {r["name"]:r for r in controls}
        assert {r["name"] for r in rows} == set(frames)
        folds = {f["fold"]:f for f in result["folds"] if f["axis"] == axis}
        assert {r["fold"] for r in rows} == set(folds)
        for label, fold in folds.items():
            assert check_fold(fold,axis,set(frames)) == {r["name"] for r in rows if r["fold"] == label}
        for row in rows:
            name = row["name"]
            three = row["rmse_deg"]
            assert set(three) == {"zero","fixed_fold","online_shrunk"}
            assert math.isclose(three["zero"],refs[name]["rmse_deg"],abs_tol=1e-5) and row["frames"] == refs[name]["frames"]
            if not folds[row["fold"]]["fixed_bias"]["eligible"]:
                assert three["zero"] == three["fixed_fold"]
            if audits[name]["applied_original_packets"] == 0:
                assert math.isclose(three["online_shrunk"],three["zero"],abs_tol=1e-5)
        for method, metrics in result["summary"][axis].items():
            by_method = [{**r,"rmse_deg":r["rmse_deg"][method]} for r in rows]
            check_scores(by_method,metrics,frames)
            for device, score in metrics["per_device"].items():
                check_scores([r for r in by_method if r["name"].startswith(device+"-")],score,frames)
    prefixes = result["causal_prefix_replay"]
    assert {"R2-10","R4-2"} <= set(prefixes) <= set(frames) and all(len(v) == 3 for v in prefixes.values())
    exact = all(all(row[k] == 0 for k in ("max_abs_corrected_input_difference","max_abs_bias_difference","max_abs_vqf_quaternion_difference"))
                for checks in prefixes.values() for row in checks)
    assert result["exact_prefix_replay"] == exact


def check_bias_heads(config, result, frames):
    assert result["config"] == config and result["same_masks_and_fold_mounting"] and result["development_only"]
    control = read(BATCH/"rounds"/config["bias_rule_round"]/"result.json")
    stage = read(BATCH/"accelerometer-bias-light-matched-stage.json")
    expected = {r["round_id"]:r for r in stage["anchors"]}
    assert len(result["results"]) == len(expected) == len({r["seed"] for r in result["results"]})
    active = set()
    assert set(result["activation"]) == set(frames)
    for name, coverage in result["activation"].items():
        original = control["online_audit"][name]
        assert coverage["score_frames"] == frames[name]
        assert coverage["active_score_frames"] == original["scored_frames_with_nonzero_estimate"]
        assert coverage["first_applied_seconds"] == original["first_applied_seconds"]
        if coverage["active_score_frames"]:
            active.add(name)
    for run in result["results"]:
        identity = run["reference_round"]
        anchor = read(BATCH/"rounds"/identity/"round.json")
        assert anchor["config_sha256"] == expected[identity]["config_sha256"] and anchor["config"]["seed"] == run["seed"]
        joint_id = run["reference_joint_round"]
        assert expected[identity]["joint_round"] == joint_id
        paired = dict(read(BATCH/anchor["result"])["sequences"])
        paired["joint_holdout"] = read(BATCH/"rounds"/joint_id/"result.json")["sequences"]["joint_holdout"]
        assert set(run["sequences"]) == set(run["metrics"]) == set(paired)
        for axis, rows in run["sequences"].items():
            assert {r["name"] for r in rows} == set(frames)
            reference = {r["name"]:r for r in paired[axis]}
            per_fold = [f for f in run["folds"] if f["axis"] == axis]
            seen = [name for fold in per_fold for name in check_fold(fold,axis,set(frames))]
            assert len(seen) == len(frames) and set(seen) == set(frames)
            for row in rows:
                prior = reference[row["name"]]
                assert math.isclose(row["rmse_deg"]["zero"],prior["rmse_deg"],abs_tol=1e-5) and row["frames"] == prior["frames"]
                if row["name"] not in active:
                    assert row["rmse_deg"]["zero"] == row["rmse_deg"]["frozen_head"]
            for method, metrics in run["metrics"][axis].items():
                values = [{**r,"rmse_deg":r["rmse_deg"][method]} for r in rows]
                check_scores(values,metrics,frames)
                for device, score in metrics["per_device"].items():
                    check_scores([r for r in values if r["name"].startswith(device+"-")],score,frames)
                for support, flag in (("supported",True),("unsupported",False)):
                    delta = fmean(r["rmse_deg"][method]-r["rmse_deg"]["zero"] for r in rows if (r["name"] in active) == flag)
                    assert math.isclose(delta,metrics[support+"_records_mean_delta_deg"],abs_tol=1e-8)


def check_derived_bias_candidate(config, result):
    parent_id = result["derived_from_actual_matched_scored_folds"]
    assert result["not_an_additional_fit"] and result["source_seed"] == config["seed"]
    path = BATCH/"rounds"/parent_id/"result.json"
    with path.open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == config["matched_experiment_sha256"]
    trials = read(path)["results"]
    rule = read(BATCH/"accelerometer-bias-light-matched-stage.json")["promotion"]
    method = config["training_input"]
    effect = {axis:[r["metrics"][axis][method]["macro_rmse_deg"]-r["metrics"][axis]["zero"]["macro_rmse_deg"] for r in trials]
              for axis in ("device","joint_holdout")}
    assert len(trials) == len({r["seed"] for r in trials}) == 3
    qualifies = (fmean(effect["joint_holdout"]) <= -rule["minimum_three_seed_average_joint_improvement_deg"]
                 and sum(v<0 for v in effect["joint_holdout"]) >= rule["minimum_matched_seed_joint_wins"]
                 and fmean(effect["device"]) <= rule["maximum_leave_device_increase_deg"])
    assert qualifies == result["eligible_for_r5"]
    source = sorted(trials,key=lambda r:r["metrics"]["joint_holdout"][method]["macro_rmse_deg"])[1]
    assert source["reference_round"] == config["paired_zero_round"] and source["seed"] == config["seed"]
    assert config["head"] == read(BATCH/"rounds"/source["reference_round"]/"round.json")["config"]
    for axis, rows in source["sequences"].items():
        assert result["dev_metrics"][axis] == source["metrics"][axis][method]
        assert result["sequences"][axis] == [{**r,"rmse_deg":r["rmse_deg"][method]} for r in rows]
    return parent_id


def check_quaternion_record(record, protocol, expected_frames, expected_rmse):
    assert record["scored_frames"] == expected_frames and 0 < expected_frames <= record["native_frames"]
    assert math.isclose(record["direct_direction_rmse_deg"],expected_rmse,abs_tol=1e-7)
    gap = abs(record["quaternion_tilt_rmse_deg"] - expected_rmse)
    assert math.isclose(record["absolute_tilt_rmse_delta_deg"],gap,abs_tol=1e-8)
    with (BATCH / record["quaternion_file"]).open("rb") as file:
        assert hashlib.file_digest(file,"sha256").hexdigest() == record["quaternion_sha256"]
    prefixes = record["prefix_checks"]
    assert [r["prefix_native_frames"] for r in prefixes] == [int(record["native_frames"]*f) for f in protocol["prefix_fractions"]]
    assert all(0 <= r["p95_rotation_delta_deg"] <= r["max_rotation_delta_deg"] for r in prefixes)
    updates = record.get("online_update_comparison") or []
    within = (0 <= record["max_unit_norm_error"] <= protocol["max_norm_error"]
              and gap <= protocol["max_scored_rmse_difference_deg"]
              and all(x["max_rotation_delta_deg"] <= protocol["max_quaternion_prefix_rotation_deg"] for x in prefixes)
              and all(x["quaternion_rotation_difference_deg"] <= protocol["max_quaternion_prefix_rotation_deg"] for x in updates))
    assert record["passed"] == within


def check_quaternion_audit(config, result, frames):
    parent = read(BATCH / "rounds" / config["parent_round"] / "round.json")
    assert parent["config_sha256"] == result["model_config_sha256"] == config["parent_config_sha256"]
    assert result["same_existing_scored_frames"] and fingerprint(result["protocol"]) == config["protocol_sha256"]
    source = read(BATCH / parent["result"])
    expected = {(r["name"],r["axis"]) for r in config["records"]}
    records = result["records"]
    assert len(records) == len(expected) and {(r["name"],r["fold_axis"]) for r in records} == expected
    rows = source["sequences"] if "sequences" in source else source["dev_metrics"]["sequences"]
    for r in records:
        axis = r["fold_axis"]
        assert r["name"] in r["holdout"]
        check_fold({"training":r["training"],"validation":r["holdout"]},axis,set(frames))
        scores = rows[axis] if isinstance(rows,dict) else rows
        prior = next(s for s in scores if s["name"] == r["name"])
        check_quaternion_record(r,result["protocol"],frames[r["name"]],prior["rmse_deg"])
    assert (result["output_status"] == "passed") == all(r["passed"] for r in records)


def check_cpu_readout(config, result, frames):
    assert result["tested_no_holdout_partitions"] and result["model_source"] == config["parent_round"]
    parent = read(BATCH / "rounds" / config["parent_round"] / "round.json")
    assert parent["config_sha256"] == result["model_config_sha256"] == config["parent_config_sha256"]
    checks = result["checks"]
    expected_keys = {(axis,name) for axis in ("device","motion","joint_holdout") for name in frames}
    assert {(r["axis"],r["record"]) for r in checks} == expected_keys and len(checks) == len(expected_keys)
    for r in checks:
        assert r["score_frames"] == frames[r["record"]]
        difference = r["numpy_cpu_tilt_rmse_deg"] - r["gpu_tilt_rmse_deg"]
        assert math.isclose(r["tilt_delta_deg"],difference,abs_tol=1e-8)
    assert math.isclose(max(abs(r["tilt_delta_deg"]) for r in checks), result["max_abs_tilt_rmse_delta_deg"], abs_tol=1e-8)
    lat = result["latency"]
    assert lat["recording"] == config["record"] and config["record"] in frames
    assert lat["timed_windows"] > 0 and lat["warmup_windows"] >= 0
    for runtime in lat["timings"].values():
        for timing in runtime.values():
            assert 0 < timing["p50_ms"] <= timing["p95_ms"] <= timing["max_ms"]


def audit():
    from verify_score_profiles import score_vector, summarize as summarize_profiles

    indexed = read_bytes_shared(BATCH / "round_index.json")
    rounds = json.loads(indexed)
    ledger_last = {}
    open_attempts = {}
    timed_attempts = []
    with (BATCH / "ledger.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if "round_id" in event:
                key = event["round_id"]
                ledger_last[key] = event
                action = event["event"]
                when = datetime.fromisoformat(event["recorded_at"])
                if action == "running":
                    open_attempts[key] = when
                elif action in {"completed", "failed", "cutoff", "rejected", "interrupted_attempt"} and key in open_attempts:
                    timed_attempts.append((open_attempts.pop(key), when, key, action))
    ids = [r["round_id"] for r in rounds]
    known_ids = set(ids)
    assert len(ids) == len(known_ids)
    isolated = {r["round_id"] for r in rounds if r["config"]["kind"] in {"ridge_seed_diagnostic","mounted_ridge_alpha_seed_diagnostic","mounted_sobol_ridge_diagnostic","ridge_cap_pair_diagnostic"}}
    assert all(r.get("relation") not in isolated for r in rounds if r["batch"] in {"adaptive-refinement","seed-repeat"})
    reference = read(BATCH / "rounds/R00001/result.json")
    ref = {r["name"]: r["frames"] for r in reference["dev_metrics"]["sequences"]}
    r5_frames = {r["name"]: r["frames"] for r in reference["r5"]["sequences"]}
    expected = set(ref)
    checked, candidates, pending, joint_checks = [], [], [], []
    independent_seeds = []
    time_states = []
    sensor_hold_checks = []
    forecast_feature_jobs = []
    encoder_checks = []
    forecast_comparisons = []
    relative_gain_checks = []
    relative_gain_scores = []
    mixed_ensemble_checks = []
    reservoir_checks = []
    dct_checks = []
    sobol_checks = []
    cap_pair_checks = []
    ranking_exclusions = {}
    derived_candidates = {}
    best_by_kind = {}
    scored_settings = set()
    profiles = defaultdict(list)
    for row in rounds:
        key = row["round_id"]
        config = row["config"]
        status = row["status"]
        assert fingerprint(config) == row["config_sha256"], key
        partitions = row["partitions"]
        if config["kind"] == "development_latency":
            training = partitions["fit"] if "fit" in partitions else partitions["development"]
            assert set(training) == expected and config["record"] in expected, key
            assert set(partitions.get("benchmark",[config["record"]])) == {config["record"]}, key
        else:
            assert set(partitions["development"]) == expected, key
            if "excluded" in partitions:
                assert set(partitions["excluded"]) == {"R1-3", "R1-4"}, key
            assert expected.isdisjoint({"R1-3", "R1-4"})
        if "r5" in partitions:
            assert set(partitions["r5"]) == set(r5_frames), key
        if "r6" in partitions:
            assert set(partitions["r6"]) == {f"R6-{i}" for i in range(1, 11)}, key
        if status != "completed":
            pending.append({"id": key, "kind": config["kind"], "status": status, "attempts": row["attempts"]})
            continue
        detailed = read(BATCH / "rounds" / key / "round.json")
        assert detailed == row, key
        assert ledger_last[key]["status"] == status, key
        result = read(BATCH / row["result"])
        assert row["started_at"] and row["ended_at"] and row["return_code"] == 0, key
        assert datetime.fromisoformat(row["ended_at"]) >= datetime.fromisoformat(row["started_at"]), key
        if "configuration_sha256" in result:
            assert result["configuration_sha256"] == row["config_sha256"], key
        if config.get("context", 0) > 64:
            assert result.get("mask_version") == "causal-bin-v2-full-window", key
        scores = result.get("dev_metrics")
        assert scores == row["dev_metrics"], key
        if config["kind"] == "baseline":
            check_scores(scores["sequences"], scores, ref)
            check_scores(result["r5"]["sequences"], result["r5"], r5_frames)
            best_by_kind["baseline"] = {"id":key,"score_deg":scores["macro_rmse_deg"]}
        elif config["kind"] == "optimizer_anchor_profile":
            parent = read(BATCH / "rounds" / config["parent"] / "round.json")
            assert result["anchor_same_config_not_new_trial"] and result["profile_replay_for"] == parent["round_id"]
            assert config["parent_config_sha256"] == parent["config_sha256"]
            check_development(result,expected,ref)
            prior = read(BATCH / parent["result"])["sequences"]
            for axis in ("device","motion"):
                rows = {r["name"]:r["rmse_deg"] for r in prior[axis]}
                gap = max(abs(row["rmse_deg"]-rows[row["name"]]) for row in result["sequences"][axis])
                assert gap <= 1e-4 and math.isclose(gap,result["maximum_original_rmse_delta_deg"][axis],abs_tol=1e-8)
        elif config["kind"] == "stream_gru" and result.get("recurrent_training_context") != "split-state-at-forbidden-context-v2":
            check_development(result, expected, ref)
            ranking_exclusions[key] = "recurrent_training_context_not_segmented"
        elif scores and "selection_rmse_deg" in scores and "sequences" in result:
            check_development(result, expected, ref)
            if config["kind"] == "bias_physical_head" and result.get("not_an_additional_fit"):
                derived_candidates[key] = check_derived_bias_candidate(config,result)
            else:
                profiles[score_vector(result["sequences"],ref)].append(key)
            if config["kind"] == "stream_gru":
                check_recurrent(config,result,ref)
            if config["kind"] == "ema_readout":
                from verify_ema import check_trial
                time_states.append(check_trial(row,result,ref))
            if config["kind"] == "legacy_head":
                assert result["historical_weights_used_in_fit"] is False, key
                check_device_weights(config, result["folds"])
                if config.get("rotation_augmentation_deg"):
                    for fold in result["folds"]:
                        epochs = fold["fitting"]["rotation_jitter_training_only"]
                        assert len(epochs) == config["epochs"]
                        for epoch in epochs:
                            rows = epoch["training"]
                            assert len(rows) == len(fold["training"])
                            assert {r["record"] for r in rows} == set(fold["training"])
                            assert all(abs(r["rotation_angle_deg"]) <= config["rotation_augmentation_deg"] for r in rows)
            if config["kind"] == "ensemble":
                mixed = check_ensemble(config, result)
                if mixed:
                    mixed_ensemble_checks.append(mixed)
            candidates.append({"id": key, "kind": config["kind"], "batch": row["batch"],
                               "score_deg": scores["selection_rmse_deg"], "config": config})
            scored_settings.add(parameter_identity(config))
            prior = best_by_kind.get(config["kind"])
            if prior is None or scores["selection_rmse_deg"] < prior["score_deg"]:
                best_by_kind[config["kind"]] = {"id":key,"score_deg":scores["selection_rmse_deg"]}
        elif config["kind"] == "joint":
            seq = result["sequences"]
            if isinstance(seq, dict):
                assert set(seq) == {"joint_holdout"}, key
                seq = seq["joint_holdout"]
            assert set(r["name"] for r in seq) == expected
            check_scores(seq,scores["joint_holdout"],ref)
            validation = []
            parent = read(BATCH / "rounds" / config["parent_round"] / "round.json")
            assert config["protocol"] == "leave-device-and-program-union-v1" and result["uses_heldout_splits"] is False, key
            assert parent["config_sha256"] == config["parent_config_sha256"] == result["parent_config_sha256"], key
            assert result["parent_round"] == parent["round_id"] == row["relation"], key
            if row["seed"] is not None:
                assert row["seed"] == parent["seed"], key
            if parent["config"].get("context",0) > 64:
                assert result["mask_version"] == "causal-bin-v2-full-window", key
            check_device_weights(parent["config"],result["folds"])
            fit_free = parent["config"]["kind"] in {"baseline","filter","hybrid"}
            for fold in result["folds"]:
                names = check_fold(fold,"joint_holdout",expected,fit_free)
                individual = [r for r in seq if r["fold"] == fold["fold"]]
                assert {r["name"] for r in individual} == names, (key,fold["fold"])
                check_scores(individual,fold["validation_summary"],ref)
                validation.extend(names)
            assert len(validation) == len(expected) and set(validation) == expected
        elif config["kind"] == "regime_audit":
            check_regimes(config, result, ref)
        elif config["kind"] == "accelerometer_bias_audit":
            check_accelerometer(config,result,ref)
        elif config["kind"] == "accelerometer_bias_head_audit":
            check_bias_heads(config,result,ref)
        elif config["kind"] == "accelerometer_relative_gain_audit" and config["stage"] == "identifiability":
            from verify_relative_gains import check_identification
            relative_gain_checks.append(check_identification(row,result,ref))
        elif config["kind"] == "accelerometer_relative_gain_audit" and config["stage"] == "tilt":
            from verify_relative_gains import check_tilt
            relative_gain_scores.append(check_tilt(row,result,ref))
        elif config["kind"] == "ridge_seed_diagnostic":
            from verify_ridge_seeds import check_job
            independent_seeds.append(check_job(row,result,reference["dev_metrics"]["sequences"]))
        elif config["kind"] == "reservoir_readout":
            from verify_reservoir import check_trial
            reservoir_checks.append(check_trial(row,result,ref))
        elif config["kind"] == "causal_dct_readout":
            from verify_dct import check_trial
            dct_checks.append(check_trial(row,result,ref))
        elif config["kind"] == "mounted_sobol_ridge_diagnostic":
            from verify_sobol import check_trial
            sobol_checks.append(check_trial(row,result,ref))
        elif config["kind"] == "ridge_cap_pair_diagnostic":
            from verify_cap_pair import check_trial
            cap_pair_checks.append(check_trial(row,result,ref))
        elif config["kind"] == "sensor_hold_diagnostic":
            from verify_hold_geometry import check_job
            sensor_hold_checks.append(check_job(row,result,ref))
        elif config["kind"] == "timesfm_encoder_diagnostic":
            from verify_forecast_probe import check_encoder
            encoder_checks.append(check_encoder(row,result))
        elif config["kind"] == "forecast_backbone_diagnostic" and config["action"] in {"probe","extract"}:
            from verify_forecast_probe import check_feature_job
            forecast_feature_jobs.append(check_feature_job(row,result))
        elif config["kind"] == "forecast_backbone_diagnostic" and config["action"] == "paired_score":
            from verify_forecast_probe import check_paired_score
            forecast_comparisons.append(check_paired_score(row,result,reference["dev_metrics"]["sequences"]))
        elif config["kind"] == "cpu_readout_audit":
            assert set(partitions["benchmark"]) <= expected
            check_cpu_readout(config, result, ref)
        elif config["kind"] == "quaternion_audit":
            check_quaternion_audit(config,result,ref)
        elif config["kind"] == "development_latency":
            protocol = result["protocol"]
            assert protocol["benchmark_record"] == config["record"] and protocol["development_only"], key
            assert len(result["benchmarks"]) + len(result["failures"]) == len(config["benchmarks"]), key
            for measurement in result["benchmarks"]:
                source = read(BATCH / "rounds" / measurement["development_round"] / "round.json")["config"]
                model = {**source, "embedding":0} if measurement["matched_without_embedding"] else source
                assert measurement["config_sha256"] == fingerprint(model)
        if scores and "joint_holdout" in scores and key not in ranking_exclusions:
            separate = config["kind"] in {"joint","optimizer_anchor_profile"}
            parent_id = (config.get("parent_round") or config["parent"]) if separate else key
            joint_checks.append({"id":key,"candidate_id":parent_id,
                                 "separate":separate,"without_embedding":bool(config.get("matched_without_embedding")),
                                 "rmse_deg":scores["joint_holdout"]["macro_rmse_deg"]})
        checked.append({"id": key, "kind": config["kind"], "batch": row["batch"],
                        "seed": row["seed"], "attempts": row["attempts"],
                        "status": status, "result": row["result"], "evidence_ended_at": row["ended_at"]})
    cutoff = max(datetime.fromisoformat(r["evidence_ended_at"]) for r in checked)
    closed = [(start,end,rid) for start,end,rid,_ in timed_attempts if end <= cutoff and rid in known_ids]
    by_id = {r["round_id"]:r for r in rounds}
    terminal_dates = [datetime.fromisoformat(r["ended_at"]) for r in rounds
                      if r.get("ended_at") and r["status"] in {"completed","failed","cutoff","rejected"}]
    snapshot_end = max([cutoff, *terminal_dates])
    failed_attempts = [{"id":rid,"started_at":start.isoformat(),"ended_at":end.isoformat(),
                        "snapshot_status":by_id[rid]["status"]} for start,end,rid,action in timed_attempts
                       if rid in by_id and end <= snapshot_end and action == "failed"]
    per_round = Counter()
    for start,end,rid in closed:
        assert end >= start, rid
        per_round[rid] += (end-start).total_seconds()
    candidate_ids = {c["id"] for c in candidates}
    direct = [c for c in candidates if c["id"] not in derived_candidates]
    direct_settings = {parameter_identity(c["config"]) for c in direct}
    for row in checked:
        row["elapsed_seconds_in_closed_attempts"] = per_round.get(row["id"])
        row["validated_development_candidate"] = row["id"] in candidate_ids
        if row["id"] in derived_candidates:
            row["candidate_scores_from"] = derived_candidates[row["id"]]
    intervals = sorted((start,end) for start,end,rid in closed)
    merged = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(end, merged[-1][1])
    from verify_ridge_seeds import check_matched_joint_rates, regularization_pair
    from verify_dev_weights import review_shortlist
    from verify_mount_pair import audit_pairs
    from verify_mounted_alpha import audit as audit_mounted_alpha
    completed_ids = {r["id"] for r in checked}
    rate_comparison = check_matched_joint_rates(by_id,completed_ids,ref)
    shortlist_weights = review_shortlist(by_id,completed_ids,cutoff,ref)
    score_profiles = summarize_profiles(profiles,by_id,{r["id"] for r in direct},set(shortlist_weights["config_sha256"]),ref)
    return {"reviewed_utc": datetime.now(timezone.utc).isoformat(),
            "round_index_sha256": hashlib.sha256(indexed).hexdigest(), "indexed_rounds": len(rounds),
            "status_counts": dict(Counter(r["status"] for r in rounds)),
            "completed_by_kind": dict(Counter(r["kind"] for r in checked)),
            "completed_excluded_from_ranking":ranking_exclusions,
            "joint_checks":joint_checks,
            "random_feature_ridge_seed_check":ridge_seed_summary(candidates),
            "independent_ridge_seed_runs":independent_seeds,
            "matched_ridge_update_rates":rate_comparison,
            "matched_regularization_gain":regularization_pair(by_id,completed_ids,ref),
            "matched_ridge_installation":audit_pairs(by_id,completed_ids,ref),
            "mounted_alpha":audit_mounted_alpha(by_id,completed_ids,ref,set(shortlist_weights["config_sha256"])),
            "shortlist_development_weights":shortlist_weights,
            "direct_development_score_profiles":score_profiles,
            "training_device_balance_families":device_balance_qualification(by_id,completed_ids),
            "fixed_time_state_runs":time_states,
            "sensor_hold_checks":sensor_hold_checks,
            "forecast_feature_jobs":forecast_feature_jobs, "timesfm_encoder_checks":encoder_checks,
            "forecast_comparisons":forecast_comparisons,
            "relative_gain_checks":relative_gain_checks, "relative_gain_scores":relative_gain_scores,
            "independent_filter_ensembles":[r for r in mixed_ensemble_checks if not r["shared_filter_states"]],
            "shared_seed_ensembles":[r for r in mixed_ensemble_checks if r["shared_filter_states"]],
            "reservoir_checks":reservoir_checks,
            "dct_checks":dct_checks,
            "sobol_checks":sobol_checks,
            "cap_pair_checks":cap_pair_checks,
            "completed_by_batch": dict(Counter(r["batch"] for r in checked)),
            "best_development_by_kind": best_by_kind,
            "development_candidate_runs": len(direct), "scored_candidate_entries":len(candidates),
            "candidate_entries_from_paired_experiments":derived_candidates,
            "distinct_direct_candidate_settings":len(direct_settings),
            "same_setting_direct_repetitions":len(direct)-len(direct_settings),
            "distinct_development_settings_excluding_seed": len(scored_settings),
            "completed_attempts_beyond_first": sum(max(0, r["attempts"] - 1) for r in checked),
            "failed_attempts": failed_attempts,
            "timing_through_last_completed": timing_summary(merged, len(intervals), cutoff, read(BATCH / "status.json")),
            "best_development": min(candidates, default=None, key=lambda x: x["score_deg"]),
            "completed": checked, "other_statuses": pending}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "round_audit.json")
    args = parser.parse_args()
    result = audit()
    write(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"completed", "other_statuses", "random_feature_ridge_seed_check"}},
                     indent=2, ensure_ascii=False, allow_nan=False))
