"""Run one development experiment without opening held-out test data."""

from pathlib import Path

from common import BATCH, aggregate, body_up, names, read_json, record, rmse, stamp, vqf_outputs, write_json


def baseline_reference():
    source = read_json(BATCH / "rounds" / "R00001" / "result.json")
    return {row["name"]: row["rmse_deg"] for row in source["dev_metrics"]["sequences"]}


def filter_trial(config):
    baseline = baseline_reference()
    rows = []
    for name in names("dev"):
        src = record(name)
        state = vqf_outputs(src["imu"], src["dt"], config["parameters"])
        valid = src["score"]
        estimate = body_up(state["quat6D"][valid])
        rows.append({"name": name, "frames": int(valid.sum()), "baseline_deg": baseline[name],
                     "rmse_deg": rmse(estimate, src["ref"][valid])})
    counts = aggregate(rows)
    return {"dev_metrics": {"device": counts, "motion": counts,
                            "selection_rmse_deg": counts["macro_rmse_deg"]},
            "sequences": {"device": rows, "motion": rows},
            "parameters_are_fixed": True}


def run(item, destination):
    started = stamp()
    config = item["config"]
    if config["kind"] == "filter":
        result = filter_trial(config)
    elif config["kind"] == "install_rotation":
        from install_rotation import run as run_rotation
        result = run_rotation(item)
    elif config["kind"] == "equivariant":
        from equivariant import run as run_equivariant
        result = run_equivariant(item)
    elif config["kind"] == "ema_readout":
        from ema_readout import run as run_ema
        result = run_ema(item)
    elif config["kind"] == "reservoir_readout":
        from reservoir_readout import run as run_reservoir
        result = run_reservoir(item)
    elif config["kind"] == "causal_dct_readout":
        from dct_readout import run as run_dct
        result = run_dct(item)
    elif config["kind"] == "mounted_ridge_alpha_seed_diagnostic":
        from mounted_ridge_alpha import run_seed
        result = run_seed(item, destination.parent)
    elif config["kind"] == "mounted_sobol_ridge_diagnostic":
        from sobol_ridge import run as run_sobol
        result = run_sobol(item)
    elif config["kind"] == "ridge_cap_pair_diagnostic":
        from ridge_cap_pair import run as run_cap_pair
        result = run_cap_pair(item)
    elif config["kind"] == "stream_gru":
        from stream_gru import run as run_gru
        result = run_gru(item)
    elif config["kind"] == "optimizer_anchor_profile":
        from optimizer_study import profile
        result = profile(item)
    elif config["kind"] == "ensemble":
        from ensemble import run as run_ensemble
        result = run_ensemble(item)
    elif config["kind"] == "legacy_head":
        from legacy_head import run as run_head
        result = run_head(item)
    elif config["kind"] == "hybrid":
        from hybrid_filter import run as run_hybrid
        result = run_hybrid(item)
    elif config["kind"] == "joint":
        if item["batch"] == "mount-50hz-ridge-matched-development":
            from mounted_ridge_joint import run as run_joint
        elif item["batch"] == "mount-50hz-ridge-alpha-joint":
            from mounted_ridge_alpha import run_joint
        else:
            from joint_holdout import run as run_joint
        result = run_joint(item)
    elif config["kind"] == "tcn":
        from temporal import run as run_temporal
        result = run_temporal(item)
    else:
        from learning import run_trial
        result = run_trial(item, destination.parent)
    result.update(round_id=item["round_id"], configuration_sha256=item["config_sha256"], started_at=started, ended_at=stamp())
    write_json(destination, result)
    print(result["dev_metrics"], flush=True)
