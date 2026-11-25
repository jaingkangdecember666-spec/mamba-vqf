"""Summarize matched seeds for device-balanced training across the original folds."""

from statistics import mean, stdev

from common import BATCH, read_json, stamp, write_json


AXES = ("device", "motion", "joint_holdout")


def read_round(identity):
    info = read_json(BATCH / "rounds" / identity / "round.json")
    return info, read_json(BATCH / info["result"]) if info["status"] == "completed" else None


def verify_fold_weights(folds, mode):
    for fold in folds:
        train = {name.split("-")[0] for name in fold["training"]}
        examples = fold["fitting"]["group_balance_training_only"]
        if any(set(epoch["device_objective_mass"]) != train for epoch in examples):
            raise ValueError(f"Device mass in {fold['fold']} does not match the current training fold")
        for epoch in examples:
            weights = list(epoch["device_objective_mass"].values())
            if abs(sum(weights) - 1.) > 1e-7 or (mode == "capped_worst" and max(weights) * len(weights) > 1.5 + 1e-7):
                raise ValueError(f"Weight normalization or cap exceeded in {fold['fold']}")
            if mode == "equal_devices" and max(weights) != min(weights):
                raise ValueError(f"Equal-device objective was unequal in {fold['fold']}")


def compare(candidate, control, axis):
    a, b = candidate["dev_metrics"][axis], control["dev_metrics"][axis]
    if a["records"] != b["records"] or a["scoring_frames"] != b["scoring_frames"]:
        raise ValueError("Matched experiment coverage changed")
    a_rows = candidate["sequences"][axis]
    b_rows = control["sequences"]
    if isinstance(b_rows, dict):
        b_rows = b_rows[axis]
    if ({r["name"]: r["frames"] for r in a_rows} != {r["name"]: r["frames"] for r in b_rows}):
        raise ValueError("Matched experiment scores changed frame denominators")
    newer = {name: s["macro_rmse_deg"] for name, s in a["per_device"].items()}
    older = {name: s["macro_rmse_deg"] for name, s in b["per_device"].items()}
    a_worst, b_worst = max(newer, key=newer.get), max(older, key=older.get)
    return {"new_deg": a["macro_rmse_deg"], "control_deg": b["macro_rmse_deg"],
            "paired_delta_deg": a["macro_rmse_deg"] - b["macro_rmse_deg"],
            "scoring_frames": a["scoring_frames"],
            "per_device_new_deg": newer, "per_device_control_deg": older,
            "per_device_delta_deg": {key: value-older[key] for key, value in newer.items()},
            "worst_device": {"candidate": a_worst, "control": b_worst,
                             "candidate_deg": newer[a_worst], "control_deg": older[b_worst],
                             "delta_deg": newer[a_worst]-older[b_worst]}}


def report():
    stage = read_json(BATCH / "device-balance-development-stage.json")
    groups = {}
    outstanding = []
    for line in stage["rounds"]:
        current, candidate = read_round(line["round_id"])
        if not candidate:
            outstanding.append({"round_id": current["round_id"], "status": current["status"]})
            continue
        grouped, control = read_round(line["grouped_control"])
        _, joint = read_round(line["joint_control"])
        comparable = {k: v for k, v in current["config"].items()
                      if k not in {"training_device_balance", "max_worst_device_multiplier"}}
        if comparable != grouped["config"]:
            raise ValueError(f"Another training setting changed in {current['round_id']}")
        verify_fold_weights(candidate["folds"], line["mode"])
        result = {**line, "axes": {axis: compare(candidate, joint if axis == "joint_holdout" else control, axis) for axis in AXES}}
        groups.setdefault(line["mode"], []).append(result)
    combined = {}
    cutoff = stage["promotion"]
    for label, rows in groups.items():
        metrics = {}
        for axis in AXES:
            effects = [r["axes"][axis]["paired_delta_deg"] for r in rows]
            worst = [r["axes"][axis]["worst_device"]["delta_deg"] for r in rows]
            dev_deltas = {name: mean(r["axes"][axis]["per_device_delta_deg"][name] for r in rows)
                          for name in rows[0]["axes"][axis]["per_device_delta_deg"]}
            metrics[axis] = {"mean_new_deg": mean(r["axes"][axis]["new_deg"] for r in rows),
                             "mean_control_deg": mean(r["axes"][axis]["control_deg"] for r in rows),
                             "new_seed_sample_std_deg": stdev(r["axes"][axis]["new_deg"] for r in rows) if len(rows)>1 else None,
                             "control_seed_sample_std_deg": stdev(r["axes"][axis]["control_deg"] for r in rows) if len(rows)>1 else None,
                             "paired_mean_delta_deg": mean(effects), "paired_range_deg": [min(effects), max(effects)],
                             "paired_wins": sum(x < 0 for x in effects), "mean_worst_device_delta_deg": mean(worst),
                             "mean_per_device_delta_deg": dev_deltas}
        qualifies = (len(rows) == 3 and metrics["joint_holdout"]["paired_mean_delta_deg"] <= -cutoff["required_mean_joint_rmse_improvement_deg"]
                     and metrics["joint_holdout"]["paired_wins"] >= cutoff["minimum_joint_paired_wins"]
                     and metrics["device"]["paired_mean_delta_deg"] <= cutoff["max_mean_leave_device_degradation_deg"])
        representative = sorted(rows, key=lambda row: row["axes"]["joint_holdout"]["new_deg"])[len(rows)//2]["round_id"]
        combined[label] = {"observed_seed_count": len(rows), "axes": metrics, "by_seed": rows,
                           "qualifies_for_r5": qualifies, "median_joint_candidate": representative}
    result = {"updated_at": stamp(), "stage": stage, "outstanding": outstanding, "families": combined}
    write_json(BATCH / "device-balance-development-summary.json", result)
    print({label: {axis: round(m["paired_mean_delta_deg"],6) for axis,m in data["axes"].items()}
            for label,data in combined.items()}, outstanding, flush=True)


if __name__ == "__main__":
    report()
