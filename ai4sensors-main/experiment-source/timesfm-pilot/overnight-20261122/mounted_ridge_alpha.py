"""Run the frozen mounted-ridge alpha diagnostics off the ordinary selection pool."""

import os
from time import perf_counter

import learning
from common import BATCH, digest, fingerprint, read_json, stamp
from mounted_ridge_joint import run as run_mounted_joint


PROTOCOL = BATCH / "ridge-50hz-mount-alpha-protocol.json"
BATCH_NAME = "mount-50hz-ridge-alpha-joint"


def run_joint(item):
    print("mounted alpha joint", item["round_id"], "python worker", os.getpid(), stamp(), flush=True)
    return run_mounted_joint(item, PROTOCOL)


def run_seed(item, folder):
    started = perf_counter()
    stage = read_json(PROTOCOL)
    wrapper = item["config"]
    if wrapper["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("The independent alpha protocol changed")
    registered = next(row for row in stage["independent_seed_rows"] if row["round_id"] == item["round_id"])
    evaluated = registered["config"]
    if (registered["config_sha256"] != fingerprint(evaluated)
            or wrapper["evaluated_model_sha256"] != registered["config_sha256"]
            or wrapper["seed"] != evaluated["seed"]):
        raise ValueError("Unexpected independent seed/configuration")
    for source, sha in stage["historical_scoring_code_sha256"].items():
        if digest(BATCH / source) != sha:
            raise ValueError(f"Historical scoring implementation changed: {source}")
    print("mounted alpha independent", item["round_id"], "python worker", os.getpid(),
          evaluated["seed"], evaluated["alpha"], stamp(), flush=True)
    output = learning.run_trial({**item, "config": evaluated}, folder)
    grouped = [(row["axis"], row["fold"], row["training"], row["validation"]) for row in output["folds"]]
    if grouped != [(axis, fold, training, held) for axis, fold, training, held in stage["grouped_validation_folds"]]:
        raise ValueError("Independent alpha diagnosis did not use the frozen grouped folds")
    for axis, rows in output["sequences"].items():
        if sorted(row["name"] for row in rows) != sorted(stage["records"]) or sum(row["frames"] for row in rows) != 8328711:
            raise ValueError(f"Lost scoring frames in {axis}")
    return {**output, "independent_dev_metrics": output["dev_metrics"], "dev_metrics": None,
            "evaluated_model_config": evaluated, "evaluated_model_sha256": fingerprint(evaluated),
            "new_ridge_fits": len(grouped), "new_installation_fits": len(grouped), "failed_fits": 0,
            "python_pid": os.getpid(), "elapsed_seconds": perf_counter() - started,
            "adaptive_parent_pool": False, "eligible_for_r5": False}
