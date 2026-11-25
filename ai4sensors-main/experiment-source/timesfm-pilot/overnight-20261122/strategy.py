"""Checkpoint-driven strategy; its first phase starts while the learning branch is prepared."""

from pathlib import Path

from common import BATCH, LOCK, fingerprint, now, read_json


def filter_config(parameters):
    return {"kind": "filter", "parameters": parameters, "seed": None}


def ranked(items, kind=None):
    selected = [r for r in items if r["status"] == "completed" and r.get("dev_metrics")
                and "selection_rmse_deg" in r["dev_metrics"]
                and (kind is None or r["config"]["kind"] == kind)]
    return sorted(selected, key=lambda r: (r["dev_metrics"]["selection_rmse_deg"],
                                           r["dev_metrics"]["device"]["worst_rmse_deg"], r["round_id"]))


def unused(items, config):
    return fingerprint(config) not in {item["config_sha256"] for item in items}


def first_filters(items):
    configs = [{"tauAcc": value} for value in (0.5, 0.8, 1.3, 2., 4.5, 7., 12.)] + [
        {"motionBiasEstEnabled": False}, {"restBiasEstEnabled": False},
        {"restThAcc": 0.22}, {"restThAcc": 0.8}, {"restThGyr": 0.8}, {"restThGyr": 3.5},
        {"biasSigmaMotion": 0.04}, {"biasSigmaMotion": 0.2}, {"biasSigmaRest": 0.01}, {"biasSigmaRest": 0.06},
    ]
    for parameters in configs:
        config = filter_config(parameters)
        if unused(items, config):
            return config, "Screen native-rate FullVQF response and bias estimation against the exact SHICE masks", "filter-screen", None
    return None


def next_job(items):
    if any(item["config"]["kind"] == "final" for item in items):
        return None
    if now() >= LOCK:
        from advanced_strategy import final_job
        return final_job(items)
    initial = first_filters(items)
    if initial is not None:
        return initial
    if (BATCH / "advanced_strategy.py").is_file():
        from advanced_strategy import next_experiment
        return next_experiment(items)
    # Keep testing valid local perturbations of the best observed filter while setup continues.
    winner = ranked(items, "filter")[0]
    original = winner["config"]["parameters"]
    for step in range(1, 1001):
        value = round(max(0.2, original.get("tauAcc", 3.) * (0.7 + step * 0.007)), 5)
        parameters = {**original, "tauAcc": value}
        config = filter_config(parameters)
        if unused(items, config):
            return config, "Refine inclination time constant around the best development filter", "filter-refinement", winner["round_id"]
    raise RuntimeError("No unique filter settings available")
