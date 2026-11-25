"""Training-only test of two relative accelerometer gains at a fixed sphere center."""

import os
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from bias_calibration import PoseCloud, pooled_fit, read_native, stationary_candidates
from common import BATCH, digest, read_json, stamp, write_json
from stream_gru import planned_folds


PROTOCOL = BATCH / "accelerometer-relative-gain-development-protocol.json"
BASIS = np.array([[1., 1.], [-1., 1.], [0., -2.]]) / np.sqrt([2., 6.])


def shape_matrices(groups, center):
    directions, sqrt_weights, group_sizes = [], [], []
    for group in groups:
        shifted = group - center
        norms = np.linalg.norm(shifted, axis=1)
        if np.any(norms < 1e-6):
            raise ValueError("Degenerate gravity direction at the fixed sphere center")
        directions.append(shifted)
        sqrt_weights.append(np.sqrt(1. / (len(groups) * len(group))))
        group_sizes.append(len(group))
    points = np.concatenate(directions)
    roots = np.repeat(sqrt_weights, group_sizes)
    split = np.cumsum(group_sizes)[:-1]

    def center_groups(values):
        return np.concatenate([group - group.mean(axis=0) for group in np.split(values, split)])

    j0 = (np.square(points) / np.square(points).sum(axis=1, keepdims=True)) @ BASIS
    design = center_groups(j0) * roots[:, None]

    def residual(theta):
        log_radius = np.log(np.linalg.norm(points * np.exp(BASIS @ theta), axis=1))
        return center_groups(log_radius) * roots

    return design, residual


def estimate(records, samples, center, limits):
    groups = [samples[name]["poses"] for name in records if len(samples[name]["poses"])]
    result = {"training_records": records, "contributing_records": len(groups),
              "centroids": sum(map(len, groups)), "records_with_three_poses": sum(len(g) >= 3 for g in groups),
              "static_blocks": sum(samples[name]["audit"]["stationary_blocks"] for name in records),
              "eligible": False, "correction_gain_xyz": [1., 1., 1.]}
    if (result["centroids"] < limits["minimum_pose_centroids"] or result["static_blocks"] < limits["minimum_static_blocks"]
            or result["records_with_three_poses"] < limits["minimum_records_with_three_poses"]):
        return {**result, "reason": "too_few_static_poses_within_records"}
    design, residual = shape_matrices(groups, center)
    eigen = np.linalg.eigvalsh(design.T @ design)
    condition = float(eigen[-1] / max(eigen[0], 1e-12))
    result.update(relative_normal_eigenvalues=eigen.tolist(), relative_normal_condition=condition)
    if eigen[0] < limits["min_relative_design_eigenvalue"] or condition > limits["max_relative_design_condition"]:
        return {**result, "reason": "insufficient_squared_direction_coverage_for_two_contrasts"}
    raw = least_squares(residual, np.zeros(2), bounds=(-0.12, 0.12))
    raw_log = BASIS @ raw.x
    result["unshrunk_log_gain_xyz"] = raw_log.tolist()
    if not raw.success or np.max(np.abs(raw_log)) > limits["max_raw_absolute_log_gain"]:
        return {**result, "reason": "unshrunk_gain_exceeds_registered_physical_range"}
    penalty = np.sqrt(limits["ridge_lambda"])
    fitted = least_squares(lambda theta: np.r_[residual(theta), penalty * theta], np.zeros(2),
                           bounds=(-0.12, 0.12))
    log_gain = BASIS @ fitted.x
    result.update(shrunk_log_gain_xyz=log_gain.tolist(), centered_log_radius_rmse_unit=float(np.linalg.norm(residual(np.zeros(2)))),
                  centered_log_radius_rmse_gain=float(np.linalg.norm(residual(fitted.x))))
    if not fitted.success or np.max(np.abs(log_gain)) > limits["maximum_final_absolute_log_gain"]:
        return {**result, "reason": "shrunk_gain_exceeds_registered_physical_range"}
    return {**result, "reason": "two_relative_contrasts_supported", "eligible": True,
            "correction_gain_xyz": np.exp(log_gain).tolist()}


def evaluate_gain(training, samples, fixed, limits):
    if not fixed["eligible"]:
        return {"eligible": False, "reason": "archived_fold_sphere_center_not_supported",
                "correction_gain_xyz": [1., 1., 1.], "per_training_device": {}}
    center = np.asarray(fixed["offset_m_s2"], dtype=np.float64)
    pooled = estimate(training, samples, center, limits)
    devices = {name.split("-")[0] for name in training}
    by_device = {device: estimate([n for n in training if n.startswith(device + "-")], samples, center, limits)
                 for device in sorted(devices)}
    supported = [device for device, fit in by_device.items() if fit["eligible"]]
    # Agreement is tested on the unshrunk diagnostics, before regularization can conceal disagreement.
    differences = {f"{a}/{b}": float(np.max(np.abs(np.asarray(by_device[a]["unshrunk_log_gain_xyz"])
                                                  - np.asarray(by_device[b]["unshrunk_log_gain_xyz"]))))
                   for a, b in combinations(supported, 2)}
    result = {**pooled, "per_training_device": by_device, "supported_training_devices": supported,
              "pairwise_max_absolute_unshrunk_log_gain_difference": differences,
              "fixed_center_m_s2": center.tolist()}
    if not pooled["eligible"]:
        return result
    if len(supported) < limits["minimum_supporting_training_devices"]:
        return {**result, "eligible": False, "reason": "too_few_independently_supported_training_devices",
                "correction_gain_xyz": [1., 1., 1.]}
    if max(differences.values(), default=0.) > limits["max_pairwise_absolute_log_gain_difference"]:
        return {**result, "eligible": False, "reason": "independent_training_device_gains_disagree",
                "correction_gain_xyz": [1., 1., 1.]}
    np.testing.assert_allclose(np.prod(result["correction_gain_xyz"]), 1., rtol=0., atol=1e-12)
    return result


def catalog(records, archived, destination):
    samples = {}
    for name in records:
        source = read_native(name)
        _, accepted, _ = stationary_candidates(source["untreated"], source["original_ok"], float(source["dt"]))
        cloud = PoseCloud()
        for point in accepted:
            cloud.observe(point)
        poses = cloud.centroids()
        previous = archived["online_audit"][name]
        if len(accepted) != previous["stationary_blocks"] or len(poses) != previous["pose_clusters"]:
            raise ValueError(f"A00002 static support did not reproduce: {name}")
        samples[name] = {"poses": poses, "audit": {"stationary_blocks": len(accepted),
                                                  "pose_clusters": len(poses), "block_counts": list(cloud.counts)}}
        print(name, "static pose centroids", len(poses), "blocks", len(accepted), flush=True)
    write_json(destination, {name: {**entry, "poses": entry["poses"].tolist()} for name, entry in samples.items()})
    return samples


def synthetic(limits):
    rng = np.random.default_rng(20261123)
    points = rng.normal(size=(70, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    desired = np.exp([.012, -.008, -.004])
    center = np.array([.12, -.08, .07])
    shifted = points * 9.80665 / desired + center
    items = {"synthetic": {"poses": shifted, "audit": {"stationary_blocks": 70}}}
    result = estimate(list(items), items, center, limits)
    if not result["eligible"]:
        raise ValueError(f"Positive synthetic shape not identified: {result}")
    np.testing.assert_allclose(result["unshrunk_log_gain_xyz"], np.log(desired), atol=1e-6, rtol=0.)
    np.testing.assert_allclose(np.prod(result["correction_gain_xyz"]), 1., atol=1e-12, rtol=0.)
    assert np.max(np.abs(result["shrunk_log_gain_xyz"])) < np.max(np.abs(np.log(desired)))
    degenerate = shifted.copy()
    degenerate[:] = np.array([0., 0., 9.8]) + center
    items["synthetic"]["poses"] = degenerate
    fallback = estimate(list(items), items, center, limits)
    assert not fallback["eligible"] and fallback["correction_gain_xyz"] == [1., 1., 1.]
    return {"known_relative_gains_recovered": True, "geometric_mean_is_one": True,
            "shrinkage_reduces_gain_size": True, "degenerate_shape_falls_back": True}


def run(job, destination):
    config = job["config"]
    if config["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Frozen relative gain protocol changed")
    protocol = read_json(PROTOCOL)
    source = BATCH / protocol["source_A00002_path"]
    if digest(source) != protocol["source_A00002_sha256"]:
        raise ValueError("Archived offset experiment changed")
    archived = read_json(source)
    print("relative accelerometer gain", config["stage"], "worker pid", os.getpid(), stamp(), flush=True)
    if config["stage"] == "tilt":
        from relative_gain_score import run as score

        score(job, destination, protocol, archived)
        return
    checks = synthetic(protocol["identification"])
    catalog_path = destination.parent / "static-pose-catalog.json"
    records = protocol["records"]
    samples = catalog(records, archived, catalog_path)
    folds = []
    archived_folds = {(row["axis"], row["fold"]): row for row in archived["folds"]}
    for axis, fold, training, held in planned_folds(records):
        old = archived_folds[axis, fold]
        if old["training"] != training or old["validation"] != held:
            raise ValueError("Unexpected change in archived fold roles")
        fixed = old["fixed_bias"]
        reproduction = pooled_fit(training, samples)
        if reproduction["eligible"] != fixed["eligible"]:
            raise ValueError("Archived center coverage did not reproduce")
        if "offset_m_s2" in fixed:
            np.testing.assert_allclose(reproduction["offset_m_s2"], fixed["offset_m_s2"], atol=1e-10, rtol=0.)
        relative = evaluate_gain(training, samples, fixed, protocol["identification"])
        folds.append({"axis": axis, "fold": fold, "training": training, "held": held,
                      "archived_fixed_bias": fixed, "relative_gain": relative})
        print(axis, fold, relative["reason"], relative.get("supported_training_devices", []), flush=True)
    supported = [row for row in folds if row["relative_gain"]["eligible"]]
    assert len(folds) == 24
    write_json(destination, {"round_id": job["round_id"], "completed_at": stamp(), "worker_pid": os.getpid(),
               "protocol_sha256": config["protocol_sha256"], "synthetic": checks, "folds": folds,
               "static_pose_catalog": str(catalog_path.relative_to(BATCH)), "static_pose_catalog_sha256": digest(catalog_path),
               "records": len(records), "total_pose_centroids": sum(len(row["poses"]) for row in samples.values()),
               "total_static_blocks": sum(row["audit"]["stationary_blocks"] for row in samples.values()),
               "can_score_actual_tilt": bool(supported), "supported_fold_count": len(supported), "evaluated_tilt": False})
