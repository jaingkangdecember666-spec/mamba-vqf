"""Fit and score matched forecasting features on the three frozen development axes."""

from hashlib import sha256
from time import perf_counter

import numpy as np

import learning
from common import BATCH, aggregate, by_device, digest, read_json, stamp, write_json
from experiment import baseline_reference
from forecast_backbone import PHYSICAL, cache_file, checked_record, load_features
from stream_gru import planned_folds


AXES = ("device", "motion", "joint_holdout")
TOTAL_FRAMES = 8328711


def state_hash(state):
    sha = sha256()
    for key, values in sorted(state.items()):
        sha.update(key.encode("utf-8"))
        sha.update(values.tobytes())
    return sha.hexdigest()


def summarize(axis_rows, arms, score_key="rmse_deg"):
    control = {row["name"]: row["rmse_deg"][PHYSICAL] for row in axis_rows}
    result = {}
    for arm in arms:
        scores = [{"name": row["name"], "frames": row["frames"], "rmse_deg": row[score_key][arm],
                   "baseline_deg": control[row["name"]]} for row in axis_rows]
        summary = aggregate(scores)
        result[arm] = {**summary, "per_device": by_device(scores),
                       "delta_vs_physical_deg": summary["macro_rmse_deg"] - float(np.mean(list(control.values())))}
    return result


def admission(metrics, stage, job_config, complete):
    checks = {}
    for arm, identity in job_config["probe_rounds"].items():
        reference = BATCH / "rounds" / identity / "result.json"
        probe = read_json(reference)
        latency = probe["composite_latency"]["p95_ms"]
        values = {axis: metrics.get(axis, {}).get(arm, {}).get("delta_vs_physical_deg") for axis in AXES}
        conditions = {"full_original_coverage": complete, "p95_at_most_100ms": latency <= 100.,
                      "joint_improvement_at_least_0_015_deg": values["joint_holdout"] is not None and values["joint_holdout"] <= -.015,
                      "device_holdout_no_regression": values["device"] is not None and values["device"] <= 1e-9,
                      "program_holdout_no_regression": values["motion"] is not None and values["motion"] <= 1e-9}
        checks[arm] = {"passed_preliminary_screen": all(conditions.values()), "conditions": conditions,
                       "probe": str(reference.relative_to(BATCH)), "probe_result_sha256": digest(reference),
                       "p95_ms": latency, "paired_delta_deg": values}
    return checks


def run(job, destination, stage):
    started = perf_counter()
    job_config = job["config"]
    config = stage["readout"]
    arms = tuple(job_config["arms"])
    delays = {arm: read_json(BATCH / "rounds" / identity / "result.json")["composite_latency"]["p95_ms"]
              for arm, identity in job_config["probe_rounds"].items()}
    expected = stage["development_records"]
    missing = {arm: [name for name in expected if not cache_file(arm, name).exists()] for arm in arms if arm != PHYSICAL}
    records = [name for name in expected if not any(name in gaps for gaps in missing.values())]
    if len(records) != len(expected) and not job_config["allow_partial"]:
        raise ValueError(f"Forecast extraction is incomplete: {missing}")
    historical = baseline_reference()
    data, features, provenance = {}, {}, {}
    for name in records:
        view, blocks, indices = checked_record(name, stage)
        view["q_scored"] = learning.attach_native(name, config)["q_scored"]
        data[name] = view
        features[name] = {arm: load_features(arm, name, indices, job_config["protocol_sha256"]) for arm in arms}
        provenance[name] = {"windows": len(blocks), "scoring_frames": len(view["index"]),
                            "feature_hashes": {arm: digest(cache_file(arm, name)) for arm in arms if arm != PHYSICAL}}

    folds = []
    axes = {axis: [] for axis in AXES}
    dropped = []
    progress = destination.parent / "score-progress.json"
    for axis, fold, train, held in planned_folds(records):
        if not train or not held:
            dropped.append({"axis": axis, "fold": fold, "training_records": len(train), "held_records": len(held)})
            continue
        preparation_start = perf_counter()
        mount = learning.installation_fit([data[name] for name in train], config)
        prepared = {name: learning.prepare(learning.apply_install(data[name], mount, include_reference=name in held), config)
                    for name in train + held}
        rows = {name: {"name": name, "fold": fold, "frames": len(prepared[name]["index"]),
                       "full_vqf_rmse_deg": historical[name], "rmse_deg": {}, "p95_delay_rmse_deg": {}}
                for name in held}
        marks = {}
        for arm in arms:
            combined = {name: {**view, "matrix": np.concatenate((view["matrix"], features[name][arm]), axis=1)}
                        for name, view in prepared.items()}
            start = perf_counter()
            state = learning.fit([combined[name] for name in train], config)
            fitted = perf_counter()
            for name, row in rows.items():
                row["rmse_deg"][arm] = learning.evaluate(combined[name], state, config)
            primary_done = perf_counter()
            if arm in delays:
                for name, row in rows.items():
                    row["p95_delay_rmse_deg"][arm] = learning.evaluate(combined[name], state, config, delays[arm])
            marks[arm] = {"fit_seconds": fitted - start, "paired_scoring_seconds": primary_done - fitted,
                          "delay_replay_seconds": perf_counter() - primary_done,
                          "fit_sha256": state_hash(state), "input_width": len(state["mean"])}
        for row in rows.values():
            row["delta_vs_physical_deg"] = {arm: row["rmse_deg"][arm] - row["rmse_deg"][PHYSICAL] for arm in arms if arm != PHYSICAL}
        axes[axis].extend(rows.values())
        folds.append({"axis": axis, "fold": fold, "training": train, "held": held,
                      "mount_quaternion_body_wxyz": mount.tolist(),
                      "same_training_masks_per_arm": True,
                      "training_label_updates_per_arm": sum(int(prepared[n]["train"].sum()) for n in train),
                      "fit_count": len(arms), "scored_rows": len(held),
                      "wall_seconds": perf_counter() - preparation_start, "arms": marks})
        print(axis, fold, {arm: round(float(np.mean([r["rmse_deg"][arm] for r in rows.values()])), 6) for arm in arms}, flush=True)
        write_json(progress, {"finished_folds": folds, "fit_count": sum(row["fit_count"] for row in folds), "sequences": axes})

    coverage = {}
    for axis, paired in axes.items():
        paired.sort(key=lambda row: row["name"])
        coverage[axis] = {"records": len(paired), "scoring_frames": sum(row["frames"] for row in paired)}
        if len({row["name"] for row in paired}) != len(paired):
            raise ValueError(f"Duplicate score record on {axis}")
    full = len(records) == len(expected) and len(folds) == 24 and all(
        row == {"records": 37, "scoring_frames": TOTAL_FRAMES} for row in coverage.values())
    if not full and not job_config["allow_partial"]:
        raise ValueError(f"Expected full coverage. Measured: {coverage}")
    metrics = {axis: summarize(rows, arms) for axis, rows in axes.items() if rows}
    delay_metrics = {axis: summarize(rows, delays, "p95_delay_rmse_deg") for axis, rows in axes.items() if rows}
    screening = admission(metrics, stage, job_config, full)
    write_json(destination, {"round_id": job["round_id"], "completed_at": stamp(), "wall_seconds": perf_counter() - started,
               "protocol_sha256": job_config["protocol_sha256"], "models": list(arms), "readout": config,
               "coverage": coverage, "records": provenance, "missing_cache_records": missing, "dropped_folds": dropped,
               "all_original_development_coverage": full, "folds": folds, "metrics": metrics, "per_record_scores": axes,
               "forecast_p95_delay_replay": {"delay_ms_by_model": delays, "metrics": delay_metrics,
                                             "reference": "immediately applied physical control; fixed compute delays only",
                                             "used_for_preliminary_screen": False},
               "deterministic_fit_count": sum(row["fit_count"] for row in folds),
               "paired_record_count": sum(row["scored_rows"] for row in folds),
               "preliminary_candidate_checks": screening, "admitted_for_r5": []})
