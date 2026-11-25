"""Reconcile fixed-weight development ensembles against their frozen members."""

import hashlib
from collections import Counter
from datetime import datetime
from math import isclose
from statistics import fmean

import numpy as np

from audit_rounds import BATCH, fingerprint, read
from verify_baselines import digest


def check_fixed_ensemble(config, result, previous_folds, qualifies):
    three = config.get("shared_head_seed_ensemble",False)
    if three:
        path = BATCH/"three-seed-ensemble-development-protocol.json"
        key, expected_sha = "three_seed_protocol_sha256","957a65172dd7f3e416623e3a55f44e5c6bbf89093ae0167102991c71aed1b196"
    else:
        path = BATCH/"independent-filter-ensemble-development-protocol.json"
        key, expected_sha = "independent_filters_protocol_sha256","1079fd1cc043f224ff443ca0d64f41a25d374089401e6eef0e66afb3774fa5b1"
    plan = read(path)
    assert digest(path) == config[key] == result[key] == expected_sha
    assert plan["round_id"] == result["round_id"]
    assert datetime.fromisoformat(plan["registered_at"]) <= datetime.fromisoformat(result["started_at"])
    assert result["protocol"] == plan["criteria"] and config["members"] == plan["members"] == result["members"]
    for name,h in plan["anchor_result_sha256"].items():
        assert digest(BATCH/name) == h
    assert result["shared_filter_states"] is three and result["exact_matching_native_indices"] is True
    members = config["members"]
    if three:
        assert [m["round_id"] for m in members] == ["P00005","P00021","P00022"]
        assert [m["config"]["seed"] for m in members] == [1729,20261122,20261123]
        assert all(m["weight"] == 1/3 and m["config"] == {**members[0]["config"],"seed":m["config"]["seed"]} for m in members)
    else:
        assert plan["uses_only_development_splits"]
        assert [m["round_id"] for m in members] == ["P00022","R01269"]
        assert {m["config"]["seed"] for m in members} == {20261123}
    assert result["accuracy_gate_passed"] == qualifies
    if not qualifies or result["admission_pending_engineering"]:
        assert result["eligible_for_r5"] is False
    cache_dirs = [BATCH/"cache"/fingerprint({"version":"causal-bin-v2-full-window","parameters":m["config"]["parameters"]})[:16]
                  for m in members]
    assert len(set(cache_dirs)) == (1 if three else 2)
    ids = [m["round_id"] for m in members]
    folds = {(f["axis"],f["fold"]):f for f in result["folds"]}
    assert len(result["folds"]) == len(folds) == 24
    assert Counter(k[0] for k in folds) == {"device":4,"motion":4,"joint_holdout":16}
    for key, f in folds.items():
        assert f["actual_model_fits"] == {i:1 for i in ids}
        for identity in ids:
            old = previous_folds[identity][key]
            assert set(old["training"]) == set(f["training"]) and set(old["validation"]) == set(f["validation"])
            mount = old.get("installation_quaternion_wxyz")
            observed = f["body_installation_by_member_wxyz"][identity]
            if mount is None:
                assert observed is None
            else:
                q, ref = np.asarray(observed), np.asarray(mount)
                assert abs(np.linalg.norm(q)-1) < 1e-6
                assert min(np.linalg.norm(q-ref),np.linalg.norm(q+ref)) < 1e-6
                if three:
                    common = np.asarray(f["body_installation_by_member_wxyz"][ids[0]])
                    assert min(np.linalg.norm(q-common),np.linalg.norm(q+common)) < 1e-6
    fit_count = len(members)*len(folds)
    assert fit_count == plan["expected_real_head_fits" if three else "expected_fits"] == result["real_member_fits"]
    names = {r["name"] for r in result["sequences"]["joint_holdout"]}
    frames, indices_sha = {}, {}
    for n in names:
        with np.load(BATCH.parent/"gru-tilt/data"/f"{n}.npz",allow_pickle=False) as native:
            indices = np.flatnonzero(native["score"])
        for folder in set(cache_dirs):
            with np.load(folder/f"{n}.npz",allow_pickle=False) as cached:
                np.testing.assert_array_equal(cached["index"],indices)
        frames[n] = len(indices)
        if three:
            indices_sha[n] = hashlib.sha256(np.asarray(indices,dtype="<i8").tobytes()).hexdigest()
    summaries = {}
    for axis, records in result["sequences"].items():
        assert {r["name"]:r["frames"] for r in records} == frames and sum(frames.values()) == plan["scored_frames_each_axis"]
        for row in records:
            assert row["name"] in folds[axis,row["fold"]]["validation"]
            if three:
                assert row["score_frame_indices_sha256"] == indices_sha[row["name"]]
        metrics = result["dev_metrics"][axis]
        correlations = {}
        for field in ("angular_error_pearson_r","tangent_error_cosine"):
            values = [r[field] for r in records if r[field] is not None]
            if values:
                assert all(-1-1e-8 <= v <= 1+1e-8 for v in values)
                assert isclose(fmean(values),metrics["mean_per_record_"+field],abs_tol=1e-10)
                correlations[field] = fmean(values)
        averages = {i:fmean(r["members_rmse_deg"][i] for r in records) for i in ids}
        assert all(isclose(v,metrics["members_macro_rmse_deg"][i],abs_tol=1e-10) for i,v in averages.items())
        summaries[axis] = {"member_degrees":averages,"ensemble_deg":fmean(r["rmse_deg"] for r in records),
                           "versus_better_member_deg":result["relative_to_best_individual"][axis],"correlations":correlations}
    return {"id":result["round_id"],"members":ids,"scores":summaries,"shared_filter_states":three,"records_per_axis":len(names),"frames_per_axis":sum(frames.values()),
            "member_fits":fit_count,"accuracy_gate_passed":qualifies,"eligible_for_r5":result["eligible_for_r5"]}
