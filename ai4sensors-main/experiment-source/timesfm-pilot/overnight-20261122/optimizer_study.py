"""Run and summarize matched learning-rate and training-duration comparisons."""

from statistics import mean, median

from common import BATCH, EXCLUDED, fingerprint, names, read_json, stamp, write_json


STAGE = BATCH / "optimizer-development-stage.json"
ANCHORS = {"legacy_head": ("P00022", "G00001"), "tcn": ("N00035", "G00002")}
PROTOCOL = {"development_only": True, "same_seed_and_features_as_anchor": True,
            "device_and_motion_folds_unchanged": True, "optimizer_checkpoints_are_not_trials": True,
            "profile_replay_is_not_a_new_trial": True,
            "seconds_measured_separately": ["fold_preparation", "full_training_call", "fold_scoring"]}


def round_record(identity):
    return read_json(BATCH / "rounds" / identity / "round.json")


def schedule(identity, config, batch, relation, hypothesis, existing=None):
    target = BATCH / "rounds" / identity / "round.json"
    key = fingerprint(config)
    if target.is_file():
        if read_json(target)["config_sha256"] != key:
            raise ValueError(f"Attempting to change frozen config {identity}")
        return identity
    if existing is None:
        existing = {row["config_sha256"]: row["round_id"] for row in
                    (read_json(p) for p in (BATCH / "rounds").glob("*/round.json"))}
    if key in existing:
        return existing[key]
    write_json(target, {"round_id": identity, "config": config, "config_sha256": key, "seed": config.get("seed"),
                       "batch": batch, "relation": relation, "hypothesis": hypothesis,
                       "partitions": {"development": names("dev"), "excluded": list(EXCLUDED)},
                       "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned",
                       "attempts": 0, "dev_metrics": None, "result": f"rounds/{identity}/result.json",
                       "stdout": f"rounds/{identity}/run.log"})
    existing[key] = identity
    return identity


def queue():
    index = {row["config_sha256"]: row["round_id"] for row in
             (read_json(p) for p in (BATCH / "rounds").glob("*/round.json"))}
    variants = {
        "legacy_head": (
            ("P00071", {"epochs": 1}, "Stop after one epoch under the original constant learning rate"),
            ("P00072", {"epochs": 4}, "Extend the original constant-rate training to four epochs"),
            ("P00073", {"epochs": 8}, "Check whether continued original-rate training overfits held devices"),
            ("P00074", {"lr": 0.00015}, "Halve the learning rate at the original two-epoch horizon"),
            ("P00075", {"lr": 0.0006}, "Double the learning rate at the original two-epoch horizon"),
        ),
        "tcn": (
            ("N00051", {"epochs": 3, "scheduler_horizon_epochs": 6}, "Train only the first three epochs of the original six-epoch cosine"),
            ("N00052", {"epochs": 12, "scheduler_horizon_epochs": 6}, "Continue at the original cosine floor for six more epochs"),
            ("N00053", {"lr": 0.0006}, "Halve the peak and floor learning rates at the original horizon"),
            ("N00054", {"lr": 0.0024}, "Double the peak and floor learning rates at the original horizon"),
        ),
    }
    plans = []
    for kind, settings in variants.items():
        original, profile_id = ANCHORS[kind]
        parent = round_record(original)
        config = parent["config"]
        profile = {"kind": "optimizer_anchor_profile", "parent": original,
                   "parent_config_sha256": parent["config_sha256"], "seed": config["seed"]}
        saved_profile = schedule(profile_id, profile, "optimizer-anchor-profile", original,
                                 "Retime exactly the same development anchor, counted only as a profiling replay", index)
        for identity, delta, hypothesis in settings:
            model = {**config, **delta}
            saved = schedule(identity, model, "optimizer-duration-rate-screen", original, hypothesis, index)
            plans.append({"kind": kind, "round_id": saved, "anchor": original, "timed_anchor_replay": saved_profile,
                          "changed": delta, "hypothesis": hypothesis})
    write_json(STAGE, {"created_at": stamp(), "protocol": PROTOCOL, "first_screen": plans, "followup": []})
    print([r["round_id"] for r in plans], flush=True)


def profile(job):
    parent = round_record(job["config"]["parent"])
    config = parent["config"]
    if fingerprint(config) != job["config"]["parent_config_sha256"]:
        raise ValueError(f"Anchor configuration drift in {parent['round_id']}")
    if config["kind"] == "tcn":
        from temporal import run
    else:
        from legacy_head import run
    replay = run(parent)
    old = read_json(BATCH / parent["result"])
    gaps = {}
    for axis in ("device", "motion"):
        original = {row["name"]: row for row in old["sequences"][axis]}
        replayed = replay["sequences"][axis]
        if any(row["frames"] != original[row["name"]]["frames"] for row in replayed):
            raise ValueError("Anchor no longer scores the same native frames")
        gaps[axis] = max(abs(row["rmse_deg"]-original[row["name"]]["rmse_deg"]) for row in replayed)
        if gaps[axis] > 1e-4:
            raise ValueError(f"Profiling replay differs from original {axis} anchor")
    return {**replay, "profile_replay_for": parent["round_id"], "anchor_same_config_not_new_trial": True,
            "maximum_original_rmse_delta_deg": gaps}


def curves(result, kind):
    output = {}
    for axis in ("device", "motion"):
        logs = [fold["fit"] if kind == "tcn" else fold["fitting"] for fold in result["folds"] if fold["axis"] == axis]
        field = "train_loss" if kind == "tcn" else "weighted_training_loss"
        output[axis] = {"training_curve_mean": [mean(epoch) for epoch in zip(*(entry[field] for entry in logs), strict=True)],
                        "epoch_seconds_median": [median(epoch) for epoch in zip(*(entry["epoch_seconds"] for entry in logs), strict=True)],
                        "learning_rate_by_epoch": logs[0]["learning_rates"]}
    return output


def report():
    stage = read_json(STAGE)
    result = []
    for entry in [*stage["first_screen"], *stage["followup"]]:
        item = round_record(entry["round_id"])
        paired = round_record(entry["timed_anchor_replay"])
        row = {**entry, "status": item["status"], "seed": item["seed"], "config": item["config"]}
        if item["status"] == "completed":
            source = read_json(BATCH / item["result"])
            row.update(timing=source.get("timing"), curves=curves(source, entry["kind"])
                       if source["folds"][0].get("timing") else None)
            original = round_record(entry["anchor"])["dev_metrics"]
            current = source["dev_metrics"]
            row["axes"] = {axis: {"macro_deg": current[axis]["macro_rmse_deg"],
                                "paired_delta_deg": current[axis]["macro_rmse_deg"]-original[axis]["macro_rmse_deg"],
                                "worst_device_deg": max(stats["macro_rmse_deg"] for stats in current[axis]["per_device"].values()),
                                "worst_record_deg": current[axis]["worst_rmse_deg"]} for axis in ("device", "motion")}
            row["paired_combined_delta_deg"] = current["selection_rmse_deg"]-original["selection_rmse_deg"]
            if paired["status"] == "completed":
                control = read_json(BATCH / paired["result"])
                row["timed_anchor"] = {"profile_round": paired["round_id"], "source_round": control["profile_replay_for"],
                                       "same_seed": control["profile_replay_for"] == entry["anchor"], "timing": control["timing"],
                                       "curves": curves(control, entry["kind"])}
        result.append(row)
    joint, families = [], {}
    for entry in stage.get("joint_followup", {}).get("rows", []):
        audit = round_record(entry["round_id"])
        control = round_record(entry["matched_joint_control"])
        actual_seed = round_record(entry["parent_round"])["seed"]
        control_seed = round_record(control["config"]["parent_round"])["seed"]
        row = {**entry, "status": audit["status"], "seed": actual_seed}
        if audit["status"] == "completed" and control["status"] == "completed":
            first = read_json(BATCH / audit["result"])
            a = first["dev_metrics"]["joint_holdout"]
            b = control["dev_metrics"]["joint_holdout"]
            if actual_seed != control_seed or (a["records"],a["scoring_frames"]) != (b["records"],b["scoring_frames"]):
                raise ValueError("Optimization audit does not match its joint-fold control")
            row.update(candidate_rmse=a["macro_rmse_deg"], control_rmse=b["macro_rmse_deg"],
                       paired_delta_deg=a["macro_rmse_deg"]-b["macro_rmse_deg"],
                       worst_device_delta_deg=max(x["macro_rmse_deg"] for x in a["per_device"].values())
                       - max(x["macro_rmse_deg"] for x in b["per_device"].values()),
                       by_device_delta_deg={key:value["macro_rmse_deg"]-b["per_device"][key]["macro_rmse_deg"]
                                            for key,value in a["per_device"].items()},
                       scoring_frames=a["scoring_frames"], timing=first.get("timing"))
            families.setdefault(entry["seed_family_audit"], []).append(row)
        joint.append(row)
    group_summary = {}
    for family, rows in families.items():
        effects = [row["paired_delta_deg"] for row in rows]
        group_summary[family] = {"observed_seeds": len(rows), "mean_joint_delta_deg": mean(effects),
                                  "joint_delta_range_deg": [min(effects), max(effects)],
                                  "mean_worst_device_delta_deg": mean(row["worst_device_delta_deg"] for row in rows)}
    write_json(BATCH / "optimizer-development-summary.json", {"updated_at": stamp(), "stage": stage, "results": result,
                 "paired_joint_results": joint, "joint_seed_summary": group_summary})
    if group_summary:
        print(group_summary, flush=True)
    for row in result:
        print(row["round_id"], row["status"], row.get("paired_combined_delta_deg"), row.get("timing"), flush=True)


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv:
        report()
    else:
        queue()
