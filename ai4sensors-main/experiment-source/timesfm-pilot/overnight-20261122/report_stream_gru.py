"""Compare valid recurrent-context trials with same-frame memory and physical controls."""

from statistics import mean

from common import BATCH, read_json, stamp, write_json


STAGE = BATCH / "stream-gru-development-stage.json"
REVISION = "split-state-at-forbidden-context-v2"
AXES = ("device", "motion", "joint_holdout")
LEGACY_CONTROLS = {1729:("P00005","J00015"), 20261122:("P00021","J00017"), 20261123:("P00022","J00016")}


def load(identity):
    row = read_json(BATCH / "rounds" / identity / "round.json")
    result = read_json(BATCH / row["result"]) if row["status"] == "completed" else None
    return row, result


def axes_of(grouped, joint):
    _, ordinary = load(grouped)
    _, simultaneous = load(joint)
    return {axis: (simultaneous if axis == "joint_holdout" else ordinary)["dev_metrics"][axis] for axis in AXES}


def report():
    stage = read_json(STAGE)
    tcn = axes_of("N00035", "J00010")
    rule = stage["repeat_rule"]
    results = []
    for entry in stage["designs"]:
        round_, result = load(entry["round_id"])
        metrics = {"round_id":round_["round_id"],"status":round_["status"],"seed":round_["seed"],
                   "relation":round_["relation"],"config":round_["config"]}
        if result:
            if result.get("recurrent_training_context") != REVISION:
                raise ValueError(f"Invalid recurrent-context run entered report: {round_['round_id']}")
            grouped, joint = LEGACY_CONTROLS[round_["seed"]]
            control = axes_of(grouped,joint)
            per_axis = {}
            for axis in AXES:
                score = result["dev_metrics"][axis]
                reference = control[axis]
                if (score["records"],score["scoring_frames"]) != (reference["records"],reference["scoring_frames"]):
                    raise ValueError("Recurrent comparison did not use the original scored frame counts")
                counterfactual = result["state_controls"][axis]
                baseline = score["macro_rmse_deg"]
                per_axis[axis] = {"continuous_deg":baseline,
                    "reset_every_2_seconds_deg":counterfactual["reset_2_seconds"]["macro_rmse_deg"],
                    "reset_every_update_deg":counterfactual["reset_each_update"]["macro_rmse_deg"],
                    "continuous_delta_vs_2_seconds_deg":baseline-counterfactual["reset_2_seconds"]["macro_rmse_deg"],
                    "continuous_delta_vs_each_update_deg":baseline-counterfactual["reset_each_update"]["macro_rmse_deg"],
                    "same_seed_legacy_control_deg":reference["macro_rmse_deg"],
                    "delta_vs_same_seed_legacy_deg":baseline-reference["macro_rmse_deg"],
                    "tcn_control_deg":tcn[axis]["macro_rmse_deg"],"worst_record":score["worst_record"],
                    "worst_record_deg":score["worst_rmse_deg"],
                    "device_macro_deg":{key:x["macro_rmse_deg"] for key,x in score["per_device"].items()},
                    "worse_record_count_vs_fullvqf":score["worse_records"]}
            joint_metrics = per_axis["joint_holdout"]
            keep = (joint_metrics["continuous_delta_vs_each_update_deg"] <= -rule["min_continuous_joint_improvement_over_reset_each_deg"]
                    and joint_metrics["delta_vs_same_seed_legacy_deg"] <= rule["max_joint_degradation_vs_physical_control_deg"]
                    and per_axis["device"]["delta_vs_same_seed_legacy_deg"] <= rule["max_device_degradation_vs_physical_control_deg"])
            metrics.update(joint_state_repeat_qualified=keep,axes=per_axis,timing=result["timing"])
        results.append(metrics)
        print(metrics["round_id"],metrics["status"],{k:tuple(round(row[x],4) for x in (
            "continuous_deg","continuous_delta_vs_2_seconds_deg","continuous_delta_vs_each_update_deg"))
            for k,row in metrics.get("axes",{}).items()},flush=True)
    families = {}
    for candidate in results:
        if candidate["status"] != "completed":
            continue
        family = dict(candidate["config"])
        family.pop("seed",None)
        from common import fingerprint
        families.setdefault(fingerprint(family), []).append(candidate)
    approved = []
    for rows in families.values():
        if len({r["seed"] for r in rows}) < 3:
            continue
        met = sum(r["joint_state_repeat_qualified"] for r in rows)
        gain = mean(r["axes"]["joint_holdout"]["delta_vs_same_seed_legacy_deg"] for r in rows)
        if met >= 2 and gain <= 0.005:
            selected = sorted(rows,key=lambda row:row["axes"]["joint_holdout"]["continuous_deg"])[len(rows)//2]
            approved.append({"round_id":selected["round_id"],"qualified_seeds":met,"mean_paired_joint_delta_deg":gain})
    write_json(BATCH / "stream-gru-development-summary.json",{"updated_at":stamp(),"stage":stage,
        "results":results,"r5_approved":approved,"legacy_controls_same_seed":LEGACY_CONTROLS,"tcn_control":"N00035 / J00010"})


if __name__ == "__main__":
    report()
