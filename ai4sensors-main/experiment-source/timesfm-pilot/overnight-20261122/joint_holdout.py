"""Score fixed candidates with both a device and a program group excluded from fitting."""

from common import BATCH, EXCLUDED, aggregate, by_device, fingerprint, names, read_json, stamp, write_json
from experiment import baseline_reference
from features import VERSION, fetch_record
from learning import MOTION_FOLDS


def joint_folds(records):
    for device in ("R1", "R2", "R3", "R4"):
        for number, programs in enumerate(MOTION_FOLDS, 1):
            same_device = {name for name in records if name.startswith(device + "-")}
            same_program = {name for name in records if int(name.split("-")[1]) in programs}
            held = sorted(same_device & same_program)
            training = sorted(set(records) - (same_device | same_program))
            if held:
                yield f"{device}/M{number}", training, held


def run(item):
    parent = item["config"]["parent_round"]
    source = read_json(BATCH / "rounds" / parent / "round.json")
    config = source["config"]
    if source["status"] != "completed" or fingerprint(config) != item["config"]["parent_config_sha256"]:
        raise ValueError("Joint audit requires the frozen, completed development configuration")

    if item["config"].get("matched_without_embedding"):
        if config["kind"] != "ridge" or not config["embedding"] or config.get("installation"):
            raise ValueError("The frozen paired ablation requires an unmounted embedding ridge")
        config = {**config, "embedding": 0}

    if config["kind"] == "legacy_head":
        from legacy_head import folds_score

        plan = [("joint_holdout", label, training, held) for label, training, held in joint_folds(names("dev"))]
        result = folds_score(config, plan)
        for _, _, training, held in plan:
            if set(training) & set(held):
                raise ValueError("Joint fold had overlapping train and score records")
        return {**result, "parent_round": parent, "parent_config_sha256": item["config"]["parent_config_sha256"],
                "evaluated_model_config": config, "evaluated_model_sha256": fingerprint(config),
                "uses_heldout_splits": False}

    import learning

    records = names("dev")
    baseline = baseline_reference()
    kind = config["kind"]
    predictions = []
    evidence = []
    training_audit = {}
    if kind == "filter":
        from experiment import filter_trial
        fixed = {row["name"]: row for row in filter_trial(config)["sequences"]["device"]}
    elif kind == "install_rotation":
        import install_rotation
        data = {name: fetch_record(name, config["parameters"]) for name in records}
        statistics = {name: install_rotation.sufficient_statistics(part, config) for name, part in data.items()}
    elif kind == "tcn":
        import temporal
        import torch
        search_device = torch.device("cuda")
        mounted = config.get("installation")
        if mounted:
            raw = {name: learning.attach_native(name, config) for name in records}
            data = {}
        else:
            data = {name: temporal.prepare(fetch_record(name, config["parameters"]), config) for name in records}
        training_audit = {name: part["training_context"] for name, part in data.items()}
    elif kind == "ridge" and not config.get("installation"):
        data = {name: learning.for_record(name, config) for name in records}
        training_audit = {name: part["training_context"] for name, part in data.items()}
    else:
        raise ValueError(f"Unsupported joint audit configuration {config}")

    for label, training, held in joint_folds(records):
        fit_log = None
        if kind == "filter":
            fold_rows = [{**fixed[name], "fold": label} for name in held]
        else:
            if kind == "install_rotation":
                fitted = install_rotation.fit_statistics([statistics[name] for name in training], config)
                fit_log = {**fitted, "fitted": fitted["fitted"].tolist()}
                scores = {name: install_rotation.evaluate(data[name], fitted, config) for name in held}
            elif kind == "tcn":
                q = None
                if mounted:
                    q = learning.installation_fit([raw[name] for name in training], config)
                    data = {name: temporal.prepare(learning.apply_install(raw[name], q, include_reference=name in held), config)
                            for name in training + held}
                    training_audit.update({name: part["training_context"] for name, part in data.items()})
                state, fit_log = temporal.fit([data[name] for name in training], config, search_device)
                if q is not None:
                    fit_log["installation_quaternion_wxyz"] = q.tolist()
                model = temporal.make_model(state, config, search_device)
                scores = {name: temporal.evaluate(data[name], config, model, search_device) for name in held}
            else:
                state = learning.fit([data[name] for name in training], config)
                scores = {name: learning.evaluate(data[name], state, config) for name in held}
            fold_rows = [{"name": name, "fold": label, "frames": len(data[name]["index"]),
                          "baseline_deg": baseline[name], "rmse_deg": scores[name]} for name in held]
        predictions.extend(fold_rows)
        summary = aggregate(fold_rows)
        evidence.append({"fold": label, "training": training if kind != "filter" else [], "validation": held,
                         "excluded_from_fit": sorted(set(records) - set(training)), "training_log": fit_log,
                         "validation_summary": summary})
        print(label, summary, flush=True)
    if sorted(row["name"] for row in predictions) != sorted(records):
        raise ValueError("Each development record must appear in exactly one joint validation fold")
    combined = {**aggregate(predictions), "per_device": by_device(predictions)}
    return {"dev_metrics": {"joint_holdout": combined}, "sequences": predictions, "folds": evidence,
            "mask_version": VERSION, "training_context_audit": training_audit, "parent_round": parent,
            "parent_config_sha256": item["config"]["parent_config_sha256"],
            "evaluated_model_config": config, "evaluated_model_sha256": fingerprint(config), "uses_heldout_splits": False,
            "purpose": "Development generalization audit only. The R5 shortlist rules remain fixed."}


def queue():
    from selection_rules import dev_value, eligible

    rows = [read_json(path) for path in sorted((BATCH / "rounds").glob("*/round.json"))]
    pool = [row for row in rows if eligible(row)]
    by_id = {row["round_id"]: row for row in pool}
    best_filter = min((row for row in pool if row["config"]["kind"] == "filter"), key=dev_value)
    best_rotation = min((row for row in pool if row["config"]["kind"] == "install_rotation"), key=dev_value)
    best_tcn = read_json(BATCH / "tcn-family-stage.json")["tcn_in_preview"][0]["round_id"]
    best_timesfm = min((row for row in pool if row["config"]["kind"] == "ridge" and row["config"]["embedding"]
                        and not row["config"].get("installation")), key=dev_value)
    candidates = (best_filter, best_rotation, by_id["R00193"], by_id[best_tcn], best_timesfm)
    planned = []
    for i, parent in enumerate(candidates, 1):
        identity = f"J{i:05d}"
        destination = BATCH / "rounds" / identity / "round.json"
        config = {"kind": "joint", "parent_round": parent["round_id"],
                  "parent_config_sha256": parent["config_sha256"], "protocol": "leave-device-and-program-union-v1"}
        if destination.is_file():
            if read_json(destination)["config"] != config:
                raise ValueError(f"Frozen audit changed for {identity}")
        else:
            item = {"round_id": identity, "config": config, "config_sha256": fingerprint(config),
                    "seed": parent["seed"], "batch": "joint-holdout-audit", "relation": parent["round_id"],
                    "hypothesis": "Audit generalization when both device and motion program group are absent during fit",
                    "partitions": {"development": names("dev"), "excluded": list(EXCLUDED)},
                    "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned",
                    "attempts": 0, "dev_metrics": None, "result": f"rounds/{identity}/result.json",
                    "stdout": f"rounds/{identity}/run.log"}
            write_json(destination, item)
        planned.append({"round_id": identity, "parent_round": parent["round_id"], "parent_kind": parent["config"]["kind"]})
    result = {"created_at": stamp(), "candidates": planned, "pure_fullvqf_ref": "R00001",
              "validation_unit": "device and program group intersection", "training_exclusion": "device or program group",
              "partitions": "R1–R4 grouped development"}
    write_json(BATCH / "joint-holdout-stage.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    queue()
