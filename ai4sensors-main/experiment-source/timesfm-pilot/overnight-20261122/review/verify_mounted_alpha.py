"""Compare matched regularization strengths for the 50 Hz mounted ridge on R1–R4."""

import hashlib
import re
from datetime import datetime
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, check_scores, fingerprint, read
from replay_mounted_alpha import ALPHAS
from verify_baselines import digest
from verify_mount_pair import REFERENCE, compare_axis, group_differences
from verify_ridge_seeds import SEEDS


PLAN = BATCH / "ridge-50hz-mount-alpha-protocol.json"
PLAN_SHA = "f6a01d9bfadd5bf9d29cab6170bef15a41ee796d15fb20e472709c427587a469"
GROUPED = ("device", "motion")


def summarize(pairs, axis):
    usable = [p for p in pairs if axis in p["axes"]]
    if not usable:
        return None
    results = [p["axes"][axis] for p in usable]
    return {"seeds":[p["seed"] for p in usable], "low_deg":fmean(r["before_deg"] for r in results),
            "high_deg":fmean(r["after_deg"] for r in results), "delta_deg":fmean(r["difference_deg"] for r in results),
            "seed_deltas_deg":[r["difference_deg"] for r in results], "paired_records":group_differences(usable,axis)}


def check_independent(row, result, registered, reference, frames):
    model = registered["config"]
    assert row["config"]["protocol_sha256"] == PLAN_SHA and row["config"]["kind"] == "mounted_ridge_alpha_seed_diagnostic"
    assert row["round_id"] == result["round_id"] == registered["round_id"] and row["seed"] == model["seed"]
    assert row["config"]["evaluated_model_sha256"] == registered["config_sha256"] == fingerprint(model)
    assert result["evaluated_model_config"] == model and result["evaluated_model_sha256"] == fingerprint(model)
    assert set(row["partitions"]) == {"development","excluded"} and set(row["partitions"]["development"]) == set(frames)
    assert not row["dev_metrics"] and not result["dev_metrics"] and result["adaptive_parent_pool"] is result["eligible_for_r5"] is False
    assert result["mask_version"] == "causal-bin-v2-full-window" and result["failed_fits"] == 0
    assert result["new_installation_fits"] == result["new_ridge_fits"] == len(result["folds"]) == 8
    metrics = result["independent_dev_metrics"]
    assert set(result["sequences"]) == set(GROUPED)
    for axis in GROUPED:
        rows = result["sequences"][axis]
        assert {r["name"] for r in rows} == set(frames)
        check_scores(rows,metrics[axis],frames)
        folds = [f for f in result["folds"] if f["axis"] == axis]
        assert len(folds) == len({f["fold"] for f in folds}) == 4
        for f in folds:
            names = check_fold(f,axis,set(frames))
            individual = [s for s in rows if s["fold"] == f["fold"]]
            assert names == {s["name"] for s in individual}
            check_scores(individual,f["validation_summary"],frames)
            ref = reference["folds"][axis+":"+f["fold"]]
            assert sorted(f["training"]) == ref["training"] and sorted(names) == ref["validation"]
            assert abs(np.linalg.norm(f["installation_quaternion_wxyz"])-1) < 2e-6
            rots = [Rotation.from_quat(q,scalar_first=True) for q in (f["installation_quaternion_wxyz"],ref["wxyz"])]
            assert np.rad2deg((rots[0].inv()*rots[1]).magnitude()) < 2e-5
    assert abs(metrics["selection_rmse_deg"]-fmean(metrics[a]["macro_rmse_deg"] for a in GROUPED)) < 1e-9
    return {**result, "dev_metrics":metrics}


def reconcile(published, result, index, ref):
    assert published["protocol_sha256"] == PLAN_SHA
    assert published["new_joint_readout_fits"] == result["joint_fits"] == 48
    assert published["new_grouped_independent_readout_fits"] == result["diagnostic_fits"] == 48
    assert published["new_installation_fits"] == 96 and published["failed_runs"] == 0
    assert published["reused_lower_alpha_joint_rounds"] == 3 and published["independent_jobs_leave_ordinary_candidate_preview_identical"] is True
    count = 0
    fixed = ("lower_alpha_deg","higher_alpha_deg","higher_minus_lower_deg")
    reference_folds = {tuple(key.split(":",1)):value for key,value in ref["folds"].items()}
    for key,rows,across in (("search_seeds","regular","regular_across_seed"),
                            ("diagnostic_seeds","independent_seed_diagnoses","independent_across_seed")):
        values = {}
        for ours,theirs in zip(result[key],published[rows],strict=True):
            assert ours["seed"] == theirs["seed"]
            assert (ours["low"],ours["high"]) == (theirs["lower_alpha_grouped"],theirs["higher_alpha_grouped"])
            triples = {}
            for axis,measured in ours["axes"].items():
                if axis == "joint_holdout":
                    ids = (ours["joint_low"],ours["joint_high"])
                    assert ids == (theirs["lower_alpha_joint"],theirs["higher_alpha_joint"])
                else:
                    ids = (ours["low"],ours["high"])
                originals = []
                for rid in ids:
                    sequences = read(BATCH/index[rid]["result"])["sequences"]
                    original = sequences[axis] if isinstance(sequences,dict) else sequences
                    originals.append({r["name"]:r for r in original})
                paired = theirs["sequences"][axis]
                assert len(paired) == len(ref["source_records"]) and {r["name"] for r in paired} == set(ref["source_records"])
                for row in paired:
                    name = row["name"]
                    old,new = (m[name] for m in originals)
                    assert row["fold"] == old["fold"] == new["fold"] and row["frames"] == ref["source_records"][name]["frames"]
                    assert row["score_frame_indices_sha256"] == ref["source_records"][name]["score_index_sha256"]
                    np.testing.assert_allclose([row["lower_alpha_deg"],row["higher_alpha_deg"],row["higher_minus_lower_deg"]],
                                               [old["rmse_deg"],new["rmse_deg"],measured["by_record"][name]],rtol=0,atol=1e-10)
                count += len(paired)
                triples[axis] = np.array([measured["before_deg"],measured["after_deg"],measured["difference_deg"]])
            triples["combined_device_program"] = (triples["device"]+triples["motion"])/2
            for axis,stats in triples.items():
                np.testing.assert_allclose([theirs["macro_rmse_deg"][axis][f] for f in fixed],stats,rtol=0,atol=1e-9)
                values.setdefault(axis,[]).append(stats)
            certificates = {(f["axis"],f["fold"]):f for f in theirs["same_mount_evidence"]}
            expected = {key for key in reference_folds if key[0] in ours["axes"]}
            assert set(certificates) == expected and len(certificates) == len(theirs["same_mount_evidence"])
            for (axis,fold),cert in certificates.items():
                saved = reference_folds[axis,fold]
                assert sorted(cert["training"]) == saved["training"] and sorted(cert["validation"]) == saved["validation"]
                q1,q2 = [Rotation.from_quat(q,scalar_first=True) for q in (cert["shared_mount_quaternion_wxyz"],saved["wxyz"])]
                assert np.rad2deg((q1.inv()*q2).magnitude()) < 2e-5
        for axis,vectors in values.items():
            for j,label in enumerate(fixed):
                observations = [v[j] for v in vectors]
                s = published[across][axis][label]
                np.testing.assert_allclose([s["mean"],*s["range"]],[fmean(observations),min(observations),max(observations)],rtol=0,atol=1e-9)
    jobs = {p["joint_high"] for p in result["search_seeds"]}|set(result["independent_status"])
    assert {r["round"] for r in published["actual_runs"]} == jobs and len(published["actual_runs"]) == 9
    native_pids = {}
    for data in published["actual_runs"]:
        rid = data["round"]
        job = index[rid]
        assert data["started_at"] == job["started_at"] and data["ended_at"] == job["ended_at"]
        assert data["result_sha256"] == result["result_sha256"][rid]
        with (BATCH/job["stdout"]).open(encoding="utf-8-sig") as log:
            match = re.match(r"^mounted alpha (?:joint|independent) (\w+) python worker (\d+) ",log.readline())
        assert match and match[1] == rid
        native_pids[rid] = int(match[2])
        assert data["actual_python_pid"] in (None,native_pids[rid])
    assert native_pids[published["first_launch"]["round"]] == published["first_launch"]["python_pid"]
    live = read(HERE/"mounted_alpha_start.json")
    job = index[live["round_id"]]
    assert live["native_pid"] == native_pids[live["round_id"]] and live["supervisor_pid"] == published["first_launch"]["supervisor_pid"]
    assert live["native_parent"] == live["launcher_pid"] and live["parent_round"] == job["config"]["parent_round"]
    assert live["index_status"] == "running" and live["round_id"] in live["native_command"]
    assert datetime.fromisoformat(job["started_at"]) <= datetime.fromisoformat(live["observed_at"]) <= datetime.fromisoformat(job["ended_at"])
    from verify_final import preview_without_diagnostic
    size, signature = preview_without_diagnostic(index,published["at"],"mounted_ridge_alpha_seed_diagnostic")
    assert count == 555
    return {"published_score_pairs":count,"native_pids":native_pids,"live_observation":live["round_id"],
            "provisional_shortlist_count_at_completion":size,"provisional_shortlist_sha256":signature}


def audit(index, completed, frames, shortlist):
    if not PLAN.is_file():
        return None
    assert digest(PLAN) == PLAN_SHA
    plan, ref = read(PLAN), read(REFERENCE)
    assert set(plan["records"]) == set(frames) and sum(frames.values()) == plan["common_scored_frames_per_axis"]
    assert {n:r["frames"] for n,r in ref["source_records"].items()} == frames
    assert all(digest(BATCH/name) == sha for name,sha in plan["historical_scoring_code_sha256"].items())
    assert plan["independent_results_used_in_adaptive_or_r5_pools"] is False
    cache_key = fingerprint({"version":plan["original_mask_version"],"parameters":{}})[:16]
    for n,frames_in_record in frames.items():
        with np.load(BATCH/"cache"/cache_key/(n+".npz"),allow_pickle=False) as data:
            native_index = data["index"].astype("<i8",copy=False)
        assert len(native_index) == frames_in_record and np.all(np.diff(native_index)>0)
        assert hashlib.sha256(native_index.tobytes()).hexdigest() == ref["source_records"][n]["score_index_sha256"]
    for axis,fold,training,valid in [*plan["grouped_validation_folds"],*(["joint_holdout",*r] for r in plan["joint_validation_folds"])]:
        check_fold({"fold":fold,"training":training,"validation":valid},axis,set(frames))
    registered = datetime.fromisoformat(plan["registered_at"])
    candidates, diagnostics, hashes = [], [], {}
    joint_fits = independent_fits = 0
    for spec in plan["pairs"]:
        low, high = spec["lower_alpha"], spec["with_mount"]
        seed = spec["seed"]
        assert low["config"]["alpha"] == ALPHAS[0] and low["config"]["seed"] == seed
        assert high["config"] == {**low["config"],"alpha":ALPHAS[1]}
        axes = {}
        for key,names in (("round",GROUPED),("joint_round",("joint_holdout",))):
            if not all(m[key] in completed for m in (low,high)):
                continue
            results = []
            for member in (low,high):
                rid = member[key]
                job = index[rid]
                path = BATCH/job["result"]
                hashes[rid] = digest(path)
                cfg = member["config"]
                assert job["seed"] == seed
                result = read(path)
                if key == "round":
                    assert job["config"] == cfg and hashes[rid] == member["grouped_result_sha256"]
                else:
                    assert result["evaluated_model_sha256"] == fingerprint(cfg) and result["evaluated_model_config"] == cfg
                    assert result["new_installation_fits"] == result["new_ridge_fits"] == 16
                    assert job["config"]["parent_round"] == member["round"] and job["config"]["parent_config_sha256"] == fingerprint(cfg)
                    if member is low:
                        assert hashes[rid] == low["joint_result_sha256"]
                    else:
                        assert registered <= datetime.fromisoformat(job["started_at"])
                        joint_fits += 16
                results.append(result)
            for axis in names:
                axes[axis] = compare_axis(axis,*results,ref)
        candidates.append({"seed":seed,"low":low["round"],"high":high["round"],"joint_low":low["joint_round"],"joint_high":high["joint_round"],"axes":axes})
    assert [p["seed"] for p in candidates] == [20261122,20261123,20261124]
    items = {(r["config"]["seed"],r["config"]["alpha"]):r for r in plan["independent_seed_rows"]}
    assert len(plan["independent_seed_rows"]) == 6 and set(items) == {(s,a) for s in SEEDS for a in ALPHAS}
    wrappers = {r["round_id"] for r in items.values()}
    assert len(wrappers) == 6 and wrappers.isdisjoint(shortlist)
    base = plan["pairs"][0]["lower_alpha"]["config"]
    context = read(BATCH/index[plan["pairs"][0]["lower_alpha"]["round"]]["result"])["training_context_audit"]
    for seed in SEEDS:
        pair, axes = [], {}
        ids = [items[seed,a]["round_id"] for a in ALPHAS]
        for alpha,rid in zip(ALPHAS,ids,strict=True):
            spec = items[seed,alpha]
            assert spec["config"] == {**base,"seed":seed,"alpha":alpha}
            if rid in completed:
                row = index[rid]
                assert registered <= datetime.fromisoformat(row["started_at"])
                path = BATCH/row["result"]
                hashes[rid] = digest(path)
                pair.append(check_independent(row,read(path),spec,ref,frames))
                assert pair[-1]["training_context_audit"] == context
                independent_fits += 8
        if len(pair) == 2:
            axes = {a:compare_axis(a,*pair,ref) for a in GROUPED}
        diagnostics.append({"seed":seed,"low":ids[0],"high":ids[1],"axes":axes})
    sources = {"plan_sha256":PLAN_SHA,"result_sha256":hashes}
    comparisons = {"search_seeds":candidates,"diagnostic_seeds":diagnostics}
    summaries = {group:{a:summary for a in (*GROUPED,"joint_holdout") if (summary := summarize(pairs,a))} for group,pairs in comparisons.items()}
    result = {**sources,**comparisons,"axes":summaries,"joint_fits":joint_fits,"diagnostic_fits":independent_fits,
              "independent_status":{rid:index[rid]["status"] if rid in index else "unregistered" for rid in sorted(wrappers)}}
    summary_file = BATCH/"ridge-50hz-mount-alpha-summary.json"
    if joint_fits == independent_fits == 48 and summary_file.is_file():
        result["published"] = reconcile(read(summary_file),result,index,ref)
        result["published"]["sha256"] = digest(summary_file)
    return result
