"""Check the identifiability and cross-device agreement of the two relative axis gains."""

from collections import Counter
from datetime import datetime
from itertools import combinations
from math import isclose

import numpy as np

from audit_rounds import BATCH, HERE, check_fold, check_scores, read, write
from verify_baselines import digest


PROTOCOL = BATCH / "accelerometer-relative-gain-development-protocol.json"
PROTOCOL_SHA256 = "c2b9a37a7c64a8be0de2b5b78cc736a5100245d74bbfc3afbf66f3da63d764e4"


def radial_objective(groups, logs):
    """Return per-record normalized loss, gradient, and direction covariance in sensor XYZ."""
    objective = 0.
    gradient = np.zeros(3)
    covariance = np.zeros((3,3))
    for points in groups:
        axes = np.square(points * np.exp(logs))
        square = axes.sum(axis=1)
        log_radius = np.log(square) / 2
        deviations = log_radius-log_radius.mean()
        directions = axes/square[:,None]
        centered = directions-directions.mean(axis=0)
        objective += np.mean(deviations**2)
        gradient += np.mean(deviations[:,None]*centered,axis=0)
        covariance += centered.T @ centered / len(centered)
    return np.sqrt(objective/len(groups)), gradient/len(groups), covariance/len(groups)


def check_estimate(fit, names, catalog, center, limits):
    assert set(fit["training_records"]) == set(names) and len(fit["training_records"]) == len(names)
    groups = [np.asarray(catalog[n]["poses"])-center for n in names if catalog[n]["poses"]]
    assert fit["centroids"] == sum(map(len,groups)) and len(groups) == fit["contributing_records"]
    assert fit["records_with_three_poses"] == sum(len(g)>=3 for g in groups)
    assert fit["static_blocks"] == sum(catalog[n]["audit"]["stationary_blocks"] for n in names)
    if (fit["static_blocks"] < limits["minimum_static_blocks"] or fit["centroids"] < limits["minimum_pose_centroids"]
            or fit["records_with_three_poses"] < limits["minimum_records_with_three_poses"]):
        assert "relative_normal_eigenvalues" not in fit
        return False
    loss, _, covariance = radial_objective(groups,np.zeros(3))
    eigen = np.linalg.eigvalsh(covariance)[1:]
    np.testing.assert_allclose(eigen,fit["relative_normal_eigenvalues"],rtol=0,atol=1e-11)
    condition = eigen[1]/max(eigen[0],1e-12)
    assert isclose(fit["relative_normal_condition"],condition,rel_tol=1e-9,abs_tol=1e-9)
    if eigen[0] < limits["min_relative_design_eigenvalue"] or condition > limits["max_relative_design_condition"]:
        assert "unshrunk_log_gain_xyz" not in fit
        return False
    raw = np.asarray(fit["unshrunk_log_gain_xyz"])
    assert abs(raw.sum()) < 1e-10
    if max(abs(raw)) > limits["max_raw_absolute_log_gain"]:
        return False
    _, gradient, _ = radial_objective(groups,raw)
    assert max(abs(gradient-gradient.mean())) < 2e-6
    log_gain = np.asarray(fit["shrunk_log_gain_xyz"])
    loss_adjusted, gradient, _ = radial_objective(groups,log_gain)
    assert abs(log_gain.sum()) < 1e-10
    assert isclose(fit["centered_log_radius_rmse_unit"],loss,abs_tol=1e-11)
    assert isclose(fit["centered_log_radius_rmse_gain"],loss_adjusted,abs_tol=1e-11)
    assert max(abs(gradient-gradient.mean()+limits["ridge_lambda"]*log_gain)) < 2e-6
    return max(abs(log_gain)) <= limits["maximum_final_absolute_log_gain"]


def check_identification(row, result, names):
    protocol = read(PROTOCOL)
    assert digest(PROTOCOL) == row["config"]["protocol_sha256"] == result["protocol_sha256"] == PROTOCOL_SHA256
    assert datetime.fromisoformat(protocol["created_at"]) <= datetime.fromisoformat(row["started_at"])
    assert row["round_id"] == result["round_id"] == protocol["diagnostic_round"] and not result["evaluated_tilt"]
    assert set(protocol["records"]) == set(names) and result["records"] == len(names) == 37
    previous = BATCH / protocol["source_A00002_path"]
    assert digest(previous) == protocol["source_A00002_sha256"]
    old = read(previous)
    fixed = {(f["axis"],f["fold"]):f["fixed_bias"] for f in old["folds"]}
    source = BATCH / result["static_pose_catalog"]
    assert digest(source) == result["static_pose_catalog_sha256"]
    catalog = read(source)
    assert set(catalog) == set(names)
    for name, entry in catalog.items():
        info, original = entry["audit"], old["online_audit"][name]
        assert len(entry["poses"]) == len(info["block_counts"]) == info["pose_clusters"] == original["pose_clusters"]
        assert sum(info["block_counts"]) == info["stationary_blocks"] == original["stationary_blocks"]
        points = np.asarray(entry["poses"]).reshape(-1,3)
        assert np.isfinite(points).all()
    assert result["total_pose_centroids"] == sum(len(r["poses"]) for r in catalog.values())
    assert result["total_static_blocks"] == sum(r["audit"]["stationary_blocks"] for r in catalog.values())
    limits = protocol["identification"]
    eligible = []
    excluded = Counter()
    assert len(result["folds"]) == len(fixed) == 24
    assert {(f["axis"],f["fold"]) for f in result["folds"]} == set(fixed)
    for f in result["folds"]:
        axis, label, training = f["axis"], f["fold"], f["training"]
        check_fold({**f,"validation":f["held"]},axis,set(names))
        baseline = fixed[axis,label]
        fit = f["relative_gain"]
        assert f["archived_fixed_bias"] == baseline
        supported = False
        if baseline["eligible"]:
            center = np.asarray(baseline["offset_m_s2"])
            np.testing.assert_allclose(fit["fixed_center_m_s2"],center,atol=1e-12,rtol=0)
            pooled = check_estimate(fit,training,catalog,center,limits)
            by_device = fit["per_training_device"]
            assert set(by_device) == {n.split("-")[0] for n in training}
            supported_devices = []
            for device, part in sorted(by_device.items()):
                device_names = [n for n in training if n.split("-")[0] == device]
                assert part["eligible"] == check_estimate(part,device_names,catalog,center,limits)
                if part["eligible"]:
                    supported_devices.append(device)
            assert supported_devices == fit["supported_training_devices"]
            differences = {f"{a}/{b}":max(abs(np.asarray(by_device[a]["unshrunk_log_gain_xyz"])-by_device[b]["unshrunk_log_gain_xyz"]))
                           for a,b in combinations(supported_devices,2)}
            assert set(differences) == set(fit["pairwise_max_absolute_unshrunk_log_gain_difference"])
            assert all(isclose(v,fit["pairwise_max_absolute_unshrunk_log_gain_difference"][k],abs_tol=1e-11)
                       for k,v in differences.items())
            supported = (pooled and len(supported_devices) >= limits["minimum_supporting_training_devices"]
                         and max(differences.values(),default=0) <= limits["max_pairwise_absolute_log_gain_difference"])
        assert fit["eligible"] == supported
        gain = np.exp(fit["shrunk_log_gain_xyz"]) if supported else np.ones(3)
        np.testing.assert_allclose(fit["correction_gain_xyz"],gain,rtol=0,atol=1e-11)
        if supported:
            eligible.append({"axis":axis,"fold":label,"gain":gain.tolist(),"held":sorted(f["held"])})
        else:
            excluded[fit["reason"]] += 1
    assert len(eligible) == result["supported_fold_count"] and bool(eligible) == result["can_score_actual_tilt"]
    return {"id":row["round_id"],"records":len(names),"pose_centroids":result["total_pose_centroids"],
            "static_blocks":result["total_static_blocks"],"supported":eligible,"fallback":dict(excluded),
            "checked_folds":len(fixed),"max_gain_change":max((abs(g-1.) for r in eligible for g in r["gain"]),default=0.)}


def check_tilt(row, result, frames):
    protocol = read(PROTOCOL)
    cfg = row["config"]
    parent = BATCH / "rounds" / cfg["diagnostic_round"] / "result.json"
    assert digest(PROTOCOL) == cfg["protocol_sha256"] == PROTOCOL_SHA256
    assert digest(parent) == cfg["diagnostic_sha256"] == result["diagnostic_sha256"]
    assert row["round_id"] == result["round_id"] == protocol["conditional_tilt_round"]
    assert result["new_gains_fitted_in_this_round"] == 0 and not result["admitted_for_r5"]
    diagnosis = read(parent)
    old = read(BATCH/protocol["source_A00002_path"])
    axes = ("device","motion","joint_holdout")
    methods = protocol["evaluation"]["arms"]
    zero, bias, gain = methods
    diagnostic = {(f["axis"],f["fold"]):f for f in diagnosis["folds"]}
    old_folds = {(f["axis"],f["fold"]):f for f in old["folds"]}
    folds = {(f["axis"],f["fold"]):f for f in result["folds"]}
    assert set(diagnostic) == set(folds) == set(old_folds) and len(result["folds"]) == len(folds) == 24
    for key, f in folds.items():
        d = diagnostic[key]
        assert d["training"] == f["training"] and d["held"] == f["held"]
        assert f["common_mount_rotation_matrix"] == old_folds[key]["common_mount_rotation_matrix"]
        fixed = d["archived_fixed_bias"]
        assert f["fixed_bias_m_s2"] == (fixed["offset_m_s2"] if fixed["eligible"] else [0,0,0])
        assert f["relative_gain_eligible"] == d["relative_gain"]["eligible"]
        assert f["relative_gain_xyz"] == d["relative_gain"]["correction_gain_xyz"]
        assert f["relative_gain_reason"] == d["relative_gain"]["reason"]
    out = {}
    anchor_gap = 0.
    packet_counts = {}
    for axis in axes:
        scores = result["per_record_scores"][axis]
        assert set(r["name"] for r in scores) == set(frames) and len(scores) == len(frames)
        comparison = {r["name"]:r for r in old["sequences"][axis]}
        for r in scores:
            name, error = r["name"],r["rmse_deg"]
            assert set(error) == set(methods)
            f = folds[axis,r["fold"]]
            assert name in f["held"] and r["relative_gain_reason"] == f["relative_gain_reason"]
            assert isclose(r["gain_minus_bias_deg"],error[gain]-error[bias],abs_tol=1e-10)
            other = comparison[name]["rmse_deg"]
            anchor_gap = max(anchor_gap,abs(error[zero]-other["zero"]),abs(error[bias]-other["fixed_fold"]))
            if not f["relative_gain_eligible"]:
                assert error[gain] == error[bias]
            for method, active in zip(methods,(False,any(f["fixed_bias_m_s2"]),any(x!=1 for x in f["relative_gain_xyz"])),strict=True):
                packets = r["active_original_packets"][method]
                assert r["active_score_frames"][method] == (frames[name] if active else 0)
                if active:
                    assert packets >= frames[name] and packet_counts.setdefault(name,packets) == packets
                else:
                    assert packets == 0
        metrics = result["metrics"][axis]
        for method in methods:
            rows = [{**r,"rmse_deg":r["rmse_deg"][method]} for r in scores]
            item = metrics[method]
            check_scores(rows,item,frames)
            assert set(item["per_device"]) == {n.split("-")[0] for n in frames}
            for device, avg in item["per_device"].items():
                check_scores([r for r in rows if r["name"].startswith(device+"-")],avg,frames)
            assert item["applied_records"] == sum(r["active_score_frames"][method]>0 for r in rows)
            assert item["applied_scored_frames"] == sum(r["active_score_frames"][method] for r in rows)
            assert item["applied_original_packets"] == sum(r["active_original_packets"][method] for r in rows)
        macro = {k:metrics[k]["macro_rmse_deg"] for k in methods}
        for key,base in (("relative_gain_minus_bias_deg",bias),("relative_gain_minus_zero_deg",zero)):
            assert isclose(metrics[key],macro[gain]-macro[base],abs_tol=1e-10)
        out[axis] = {"scores":macro,"gain_records":metrics[gain]["applied_records"],
                     "gain_frames":metrics[gain]["applied_scored_frames"],"scored_frames":metrics[zero]["scoring_frames"]}
    assert anchor_gap <= 1e-6 and isclose(result["unchanged_original_and_fold_bias_control_max_delta_deg"],anchor_gap,abs_tol=1e-12)
    s = {axis:out[axis]["scores"] for axis in axes}
    joint = s["joint_holdout"]
    checks = {"joint_gain_improves_at_least_0_015_deg_against_fixed_bias":joint[gain]-joint[bias] <= -.015,
              "joint_gain_no_worse_than_no_calibration":joint[gain]-joint[zero] <= 1e-6,
              "device_gain_no_regression_against_fixed_bias":s["device"][gain]-s["device"][bias] <= 1e-6,
              "program_gain_no_regression_against_fixed_bias":s["motion"][gain]-s["motion"][bias] <= 1e-6}
    assert result["promotion_checks"] == checks and result["worth_evaluating_with_repeated_head_seeds"] == all(checks.values())
    return {"id":row["round_id"],"axes":out,"records_per_axis":len(frames),"checks":checks,"promotion":all(checks.values())}


def replay_tilt():
    from scipy.spatial.transform import Rotation
    from vqf import VQF
    from verify_final import raw_shice_samples

    plan = read(PROTOCOL)
    path = BATCH / "rounds" / plan["conditional_tilt_round"] / "result.json"
    result = read(path)
    methods = plan["evaluation"]["arms"]
    evidence = []
    for name in ("R3-3","R4-2"):
        raw = raw_shice_samples(name)
        score = next(r for r in result["per_record_scores"]["joint_holdout"] if r["name"] == name)
        fold = next(f for f in result["folds"] if f["axis"] == "joint_holdout" and f["fold"] == score["fold"])
        mask = raw["score"]
        corrected = raw["imu"].astype(np.float64)
        corrected[np.logical_or.accumulate(raw["observed"]),:3] -= fold["fixed_bias_m_s2"]
        scaled = corrected.copy()
        scaled[:,:3] *= fold["relative_gain_xyz"]
        assert int(mask.sum()) == score["frames"]
        out = {"name":name,"fold":fold["fold"],"source_sha256":digest(raw["source"]),"frames":score["frames"],
               "gain_active":fold["relative_gain_eligible"],"scores":{}}
        assert out["gain_active"] == (name == "R3-3")
        for method, signal in zip(methods,(raw["imu"].astype(np.float64),corrected,scaled),strict=True):
            q = VQF(raw["dt"], **plan["evaluation"]["parameters"]).updateBatch(
                np.ascontiguousarray(signal[:,3:]),np.ascontiguousarray(signal[:,:3]))["quat6D"]
            prediction = Rotation.from_quat(q[mask],scalar_first=True).inv().apply((0,0,1)) @ fold["common_mount_rotation_matrix"]
            ref = raw["reference"][mask]
            angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(prediction,ref),axis=1),np.einsum("ij,ij->i",prediction,ref)))
            rmse = float(np.sqrt(np.mean(angle**2)))
            gap = abs(rmse-score["rmse_deg"][method])
            assert gap < 2e-4, (name,method,gap)
            packets = score["active_original_packets"][method]
            assert packets == 0 or packets == int(raw["observed"].sum())
            out["scores"][method] = {"raw_rmse_deg":rmse,"absolute_difference_deg":gap}
        evidence.append(out)
    return {"result_sha256":digest(path),"records":evidence}


if __name__ == "__main__":
    write(HERE / "relative_gain_raw_replay.json",replay_tilt())
