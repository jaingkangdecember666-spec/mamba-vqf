"""Paired native-frame replay for conditionally identified diagonal gain fits."""

from time import perf_counter

import numpy as np

from bias_calibration import audit_output, compensate, read_native
from common import BATCH, aggregate, by_device, digest, read_json, stamp, write_json
from features import fetch_record


METHODS = ("zero", "fixed_fold_bias", "fixed_fold_bias_plus_relative_gain")
TOTAL_FRAMES = 8328711


def calibrated_inputs(source, offset, gain):
    bias_corrected = compensate(source["untreated"], source["original_ok"], offset)
    if np.array_equal(gain, np.ones(3)):
        return bias_corrected, bias_corrected
    scaled = bias_corrected.copy()
    scaled[:, :3] *= gain
    return bias_corrected, scaled


def comparisons(rows):
    result = {}
    baseline = {row["name"]: row["rmse_deg"]["zero"] for row in rows}
    for method in METHODS:
        values = [{"name": row["name"], "frames": row["frames"], "rmse_deg": row["rmse_deg"][method],
                   "baseline_deg": baseline[row["name"]]} for row in rows]
        result[method] = {**aggregate(values), "per_device": by_device(values),
                          "applied_records": sum(row["active_score_frames"][method] > 0 for row in rows),
                          "applied_scored_frames": sum(row["active_score_frames"][method] for row in rows),
                          "applied_original_packets": sum(row["active_original_packets"][method] for row in rows)}
    correction = result[METHODS[2]]["macro_rmse_deg"]
    result["relative_gain_minus_bias_deg"] = correction - result[METHODS[1]]["macro_rmse_deg"]
    result["relative_gain_minus_zero_deg"] = correction - result[METHODS[0]]["macro_rmse_deg"]
    return result


def run(job, destination, protocol, archive):
    started = perf_counter()
    config = job["config"]
    parent = BATCH / "rounds" / config["diagnostic_round"] / "result.json"
    if digest(parent) != config["diagnostic_sha256"]:
        raise ValueError("Relative gain diagnostic changed before scoring")
    diagnosis = read_json(parent)
    if not diagnosis["can_score_actual_tilt"] or diagnosis["protocol_sha256"] != config["protocol_sha256"]:
        raise ValueError("Expected conditionally supported preregistered fold gains")
    source_folds = {(r["axis"], r["fold"]): r for r in archive["folds"]}
    old = {axis: {row["name"]: row for row in rows} for axis, rows in archive["sequences"].items()}
    records = protocol["records"]
    parameters = protocol["evaluation"]["parameters"]
    cache = {name: fetch_record(name, parameters) for name in records}
    sources = {name: read_native(name) for name in records}
    folds, predictions = [], {}
    prefix = None
    worst_anchor_difference = 0.
    for fold in diagnosis["folds"]:
        axis, label = fold["axis"], fold["fold"]
        verified = source_folds[axis, label]
        if verified["training"] != fold["training"] or verified["validation"] != fold["held"]:
            raise ValueError("Unexpected change in tilt fold training or evaluation records")
        mount = np.asarray(verified["common_mount_rotation_matrix"], dtype=np.float64)
        bias = fold["archived_fixed_bias"]
        relative = fold["relative_gain"]
        offset = np.asarray(bias.get("offset_m_s2", np.zeros(3))) if bias["eligible"] else np.zeros(3)
        gain = np.asarray(relative["correction_gain_xyz"], dtype=np.float64)
        if not relative["eligible"] and not np.array_equal(gain, np.ones(3)):
            raise ValueError("A rejected gain did not fall back to identity")
        fold_rows = []
        for name in fold["held"]:
            source = sources[name]
            raw = cache[name]
            np.testing.assert_array_equal(raw["index"], np.flatnonzero(source["score"]))
            count = len(raw["index"])
            dt = float(source["dt"])
            zero = audit_output(None, raw, dt, parameters, mount, "zero")
            base_bias, corrected = calibrated_inputs(source, offset, gain) if bias["eligible"] else (None, None)
            offset_mode = audit_output(base_bias, raw, dt, parameters, mount, "fixed") if bias["eligible"] else zero
            gain_mode = audit_output(corrected, raw, dt, parameters, mount, "fixed") if relative["eligible"] else offset_mode
            expected = old[axis][name]
            gap = max(abs(zero - expected["rmse_deg"]["zero"]), abs(offset_mode - expected["rmse_deg"]["fixed_fold"]))
            if gap > 1e-6 or count != expected["frames"]:
                raise ValueError(f"Common offset or uncalibrated control changed: {axis}/{name} by {gap}")
            worst_anchor_difference = max(worst_anchor_difference, gap)
            bias_on = bool(np.any(offset))
            gain_on = bool(not np.array_equal(gain, np.ones(3)))
            effective = [0, count if bias_on else 0, count if gain_on else 0]
            packets = int(source["original_ok"].sum())
            fold_rows.append({"name": name, "fold": label, "frames": count, "pure_fullvqf_deg": expected["pure_fullvqf_deg"],
                              "rmse_deg": dict(zip(METHODS, (zero, offset_mode, gain_mode), strict=True)),
                              "gain_minus_bias_deg": gain_mode - offset_mode,
                              "active_score_frames": dict(zip(METHODS, effective, strict=True)),
                              "active_original_packets": dict(zip(METHODS, (0, packets if bias_on else 0, packets if gain_on else 0), strict=True)),
                              "relative_gain_reason": relative["reason"]})
            if gain_on and prefix is None:
                n = len(corrected) // 2
                truncated = {"untreated": source["untreated"][:n], "original_ok": source["original_ok"][:n]}
                short_bias, short_gain = calibrated_inputs(truncated, offset, gain)
                np.testing.assert_array_equal(short_bias, base_bias[:n])
                np.testing.assert_array_equal(short_gain, corrected[:n])
                np.testing.assert_array_equal(short_bias[:, 3:], short_gain[:, 3:])
                prefix = {"name": name, "axis": axis, "fold": label, "native_prefix": n,
                          "corrected_input_prefix_exact": True, "gyroscope_unchanged": True}
        predictions.setdefault(axis, []).extend(fold_rows)
        folds.append({"axis": axis, "fold": label, "training": fold["training"], "held": fold["held"],
                      "common_mount_rotation_matrix": mount.tolist(), "fixed_bias_m_s2": offset.tolist(),
                      "relative_gain_xyz": gain.tolist(), "relative_gain_eligible": relative["eligible"],
                      "relative_gain_reason": relative["reason"]})
        print(axis, label, "gain", gain.tolist(), flush=True)
    for rows in predictions.values():
        if sorted(row["name"] for row in rows) != sorted(records) or sum(row["frames"] for row in rows) != TOTAL_FRAMES:
            raise ValueError("Paired tilt replay did not preserve original development coverage")
    metrics = {axis: comparisons(rows) for axis, rows in predictions.items()}
    checks = {
        "joint_gain_improves_at_least_0_015_deg_against_fixed_bias": metrics["joint_holdout"]["relative_gain_minus_bias_deg"] <= -.015,
        "joint_gain_no_worse_than_no_calibration": metrics["joint_holdout"]["relative_gain_minus_zero_deg"] <= 1e-6,
        "device_gain_no_regression_against_fixed_bias": metrics["device"]["relative_gain_minus_bias_deg"] <= 1e-6,
        "program_gain_no_regression_against_fixed_bias": metrics["motion"]["relative_gain_minus_bias_deg"] <= 1e-6,
    }
    write_json(destination, {"round_id": job["round_id"], "completed_at": stamp(), "diagnostic_sha256": config["diagnostic_sha256"],
               "folds": folds, "per_record_scores": predictions, "metrics": metrics, "prefix": prefix,
               "unchanged_original_and_fold_bias_control_max_delta_deg": worst_anchor_difference,
               "new_gains_fitted_in_this_round": 0, "promotion_checks": checks,
               "worth_evaluating_with_repeated_head_seeds": all(checks.values()), "admitted_for_r5": [],
               "wall_seconds": perf_counter() - started})
