"""Reconcile the two angular caps on the same three seeds, folds, and scoring frames."""

from collections import Counter
from datetime import datetime
from math import isclose
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, check_scores, fingerprint, read
from verify_baselines import digest
from verify_mount_pair import REFERENCE


PLAN = BATCH/"ridge-cap-4deg-6deg-protocol.json"
PLAN_SHA = "c8890ae53022ddecb5f001426abf26a519dce616377ed2d6f0f19b08416dda7b"
AXES = ("device","motion","joint_holdout")
CAPS = ("4.0","6.0")


def check_trial(job, result, frames):
    plan, ref = read(PLAN), read(REFERENCE)
    start = read(HERE/"ridge_cap_pair_start.json")
    assert digest(PLAN) == PLAN_SHA == job["config"]["protocol_sha256"]
    assert all(digest(BATCH/name)==sha for name,sha in plan["historical_scoring_code_sha256"].items())
    assert job["round_id"] == plan["round_id"] == start["round_id"] == "O00001"
    assert datetime.fromisoformat(plan["registered_at"]) <= datetime.fromisoformat(job["started_at"]) <= datetime.fromisoformat(start["observed_at"]) <= datetime.fromisoformat(job["ended_at"])
    assert start["supervisor_pid"] == plan["supervisor_at_registration"] and start["native_pid"] == result["python_pid"]
    assert start["native_parent"] == start["launcher_pid"] and "O00001" in start["native_command"]
    assert result["eligible_for_r5"] is plan["eligible_for_r5"] is plan["shortlist_policy_changed"] is False
    assert result["dev_metrics"] is job["dev_metrics"] is None and job["seed"] is None and result["failed_fits"] == 0
    assert set(job["partitions"]) == {"development","excluded"} and set(job["partitions"]["development"]) == set(frames)
    seeds = plan["search_seeds"]
    assert seeds == [20261122,20261123,20261124] and plan["caps_deg"] == [4.,6.]
    cfg = plan["shared_config"]
    assert all(fingerprint({**cfg,"cap_deg":float(cap),"seed":seed}) == plan["expected_model_configurations"][f"{seed}/{cap}"]
               for seed in seeds for cap in CAPS)
    assert len(plan["expected_model_configurations"]) == 6 and len(plan["folds"]) == len(result["folds"]) == 24
    valid_fold = {}
    for f,(axis,label,train,held) in zip(result["folds"],plan["folds"],strict=True):
        assert (f["axis"],f["fold"],f["training"],f["validation"]) == (axis,label,train,held)
        check_fold(f,axis,set(frames))
        previous = ref["folds"][axis+":"+label]
        assert sorted(train)==previous["training"] and sorted(held)==previous["validation"]
        q = Rotation.from_quat(f["mount_quaternion_wxyz"],scalar_first=True)
        expected = Rotation.from_quat(previous["wxyz"],scalar_first=True)
        assert np.rad2deg((q.inv()*expected).magnitude()) < 2e-5
        assert f["ridge_fits"]==len(seeds) and f["unclipped_prediction_vectors"]==len(held)*len(seeds)
        for name in held:
            assert (axis,name) not in valid_fold
            valid_fold[axis,name] = label
    assert result["ridge_fits"]==sum(f["ridge_fits"] for f in result["folds"])==72
    assert result["shared_mount_fits"]==24 and result["unclipped_prediction_arrays"]==sum(f["unclipped_prediction_vectors"] for f in result["folds"])==333
    anchors = {}
    for a in plan["existing_or_queued"]:
        folder = BATCH/"rounds"/a["round_id"]
        entry = read(folder/"round.json")
        assert entry["config_sha256"]==a["configuration_sha256"]
        if a["category"] == "grouped":
            assert a["configuration_sha256"] == plan["expected_model_configurations"][f"{a['seed']}/{a['cap_deg']}"]
        if a["status_at_registration"]=="completed":
            assert a["result_sha256"]==digest(folder/"result.json")
        if entry["status"]!="completed" or datetime.fromisoformat(entry["ended_at"])>datetime.fromisoformat(job["started_at"]):
            continue
        saved = read(folder/"result.json")["sequences"]
        axes = {"joint_holdout":saved} if isinstance(saved,list) else saved
        for axis,rows in axes.items():
            for r in rows:
                anchors[a["seed"],str(a["cap_deg"]),axis,r["name"]] = (a["round_id"],r)
    all_stats = {}
    max_gap = 0.
    verified_anchors = 0
    for seed in map(str,seeds):
        assert set(result["sequences_by_seed"][seed]) == set(AXES)
        stats = {}
        for axis,rows in result["sequences_by_seed"][seed].items():
            assert len(rows)==len(frames) and {r["name"] for r in rows}==set(frames)
            declared = result["summary_by_seed"][seed][axis]
            for cap in CAPS:
                check_scores([{**r,"rmse_deg":r["rmse_deg"][cap]} for r in rows],declared["errors"][cap],frames)
                totals = Counter()
                for r in rows:
                    n = r["name"]
                    assert r["fold"]==valid_fold[axis,n] and r["score_frame_indices_sha256"]==ref["source_records"][n]["score_index_sha256"]
                    c = r["clip"][cap]
                    assert all(type(value) is int and value >= 0 for value in c.values())
                    assert c["updates_over_cap_used_on_score_frames"] <= c["updates_over_cap"] <= c["all_candidate_updates"]
                    assert c["updates_over_cap_used_on_score_frames"] <= c["original_score_frames_using_clipped_update"] <= frames[n]
                    totals.update(c)
                    saved = anchors.get((int(seed),cap,axis,n))
                    proof = r["existing_score_reproduction"].get(cap)
                    assert bool(saved) == bool(proof)
                    if saved:
                        rid, original = saved
                        gap = abs(original["rmse_deg"]-r["rmse_deg"][cap])
                        assert rid==proof["round"] and gap==proof["absolute_delta_deg"] < 1e-5
                        assert original["frames"]==r["frames"] and original["fold"]==r["fold"]
                        max_gap = max(max_gap,gap)
                        verified_anchors += 1
                assert dict(totals)==declared["clip"][cap]
            diffs = [r["rmse_deg"]["6.0"]-r["rmse_deg"]["4.0"] for r in rows]
            for r,delta in zip(rows,diffs,strict=True):
                low,high = (r["clip"][c] for c in CAPS)
                assert all(high[k] <= low[k] for k in low) and high["all_candidate_updates"]==low["all_candidate_updates"]
                assert delta==r["six_minus_four_deg"]
                if low["original_score_frames_using_clipped_update"]==0:
                    assert abs(delta)<=1e-9
            wins,losses = sum(x < -1e-10 for x in diffs),sum(x > 1e-10 for x in diffs)
            assert [wins,losses,len(rows)-wins-losses] == [declared[k] for k in ("improved_at_six","degraded_at_six","equal_at_six")]
            assert isclose(fmean(diffs),declared["mean_six_minus_four_deg"],abs_tol=1e-12)
            stats[axis] = {"mean_deg":{c:fmean(r["rmse_deg"][c] for r in rows) for c in CAPS},
                           "delta_deg":fmean(diffs),"wins":wins,"losses":losses,"clip":declared["clip"]}
        all_stats[seed] = stats
    assert verified_anchors==len(anchors) and max_gap==result["maximum_existing_anchor_difference_deg"]
    from verify_final import preview_without_diagnostic
    index = {r["round_id"]:r for r in read(BATCH/"round_index.json")}
    size,signature = preview_without_diagnostic(index,job["ended_at"],job["config"]["kind"])
    proof = {"id":job["round_id"],"result_sha256":digest(BATCH/job["result"]),"plan_sha256":PLAN_SHA,
            "pairs":333,"matched_anchor_scores":verified_anchors,"anchor_max_abs_difference_deg":max_gap,"seeds":all_stats,
            "shared_mount_fits":24,"ridge_fits":72,"observed_worker":start["native_pid"],
            "preview_count":size,"preview_sha256":signature}
    summary_path = BATCH/"ridge-cap-4deg-6deg-summary.json"
    if summary_path.is_file():
        published = read(summary_path)
        assert published["round"] == job["round_id"] and published["complete_at"]==job["ended_at"]
        assert published["protocol_sha256"]==PLAN_SHA and published["file_sha256"]==proof["result_sha256"]
        assert published["seed_results"]==result["summary_by_seed"]
        assert (published["new_shared_ridge_fits"],published["one_mount_per_fold"],published["shared_unclipped_prediction_arrays"])==(72,24,333)
        assert published["failed_fits"]==0 and published["anchored_score_maximum_absolute_difference_deg"]==max_gap
        assert published["launch"]["python_pid"]==start["native_pid"] and published["launch"]["supervisor_pid"]==start["supervisor_pid"]
        assert published["ordinary_shortlist_preview_identical_without_diagnostic"]
        for a,summary in published["summary_across_seeds"].items():
            deltas = [s[a]["delta_deg"] for s in all_stats.values()]
            assert np.allclose([summary["mean_six_minus_four_deg"],*summary["paired_seed_range_deg"]],
                               [fmean(deltas),min(deltas),max(deltas)],rtol=0,atol=1e-12)
        proof["published_sha256"] = digest(summary_path)
    receipt = HERE/"ridge_cap_pair_raw_replay.json"
    if receipt.is_file():
        raw = read(receipt)
        assert raw["seed"] == 20261124 and raw["fold"]=="R4/M4" and raw["training"]==ref["folds"]["joint_holdout:R4/M4"]["training"]
        assert raw["plan_sha256"]==PLAN_SHA and raw["source_result_sha256"]==proof["result_sha256"]
        assert set(raw["records"])==set(ref["folds"]["joint_holdout:R4/M4"]["validation"]) and raw["training_updates"]>0
        saved = {r["name"]:r for r in result["sequences_by_seed"]["20261124"]["joint_holdout"] if r["fold"]=="R4/M4"}
        gaps = []
        for n,r in raw["records"].items():
            assert r["score_frame_indices_sha256"] == saved[n]["score_frame_indices_sha256"] and r["frames"]==frames[n]
            for c in CAPS:
                observed = r["caps"][c]
                assert all(observed[k]==v for k,v in saved[n]["clip"][c].items())
                gap = abs(observed["rmse_deg"]-saved[n]["rmse_deg"][c])
                assert gap==observed["abs_rmse_difference_deg"]<2e-5
                gaps.append(gap)
        proof["qr_replay"] = {"records":sorted(raw["records"]),"frames":sum(frames[n] for n in saved),
                              "training_updates":raw["training_updates"],"max_abs_rmse_difference_deg":max(gaps)}
    return proof
