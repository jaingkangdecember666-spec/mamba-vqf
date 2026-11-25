"""Queue rotation controls and repair legacy windows without replacing the active supervisor."""

import os
import shutil

from common import BATCH, EXCLUDED, fingerprint, names, read_json, stamp, write_json
from features import VERSION, training_support


def validate_support():
    import numpy as np

    lengths = np.r_[10, 11, np.repeat(10, 200)]
    ends = np.r_[0, np.cumsum(lengths)]
    context = np.ones(ends[-1], dtype=bool)
    context[ends[30] + 1] = False
    prefixes = np.r_[0, np.cumsum(~context)]
    source = {"block_ends": ends, "bad_to_block": prefixes[ends], "label_ok": np.ones(len(lengths), dtype=bool)}
    indices = np.array([63, 80, 129, 180])
    for length in (24, 64, 80, 96, 128, 160, 200):
        result, _ = training_support(source, indices, length)
        support = max(64, length)
        expected = np.array([i + 1 >= support and context[ends[max(0, i + 1 - support)]:ends[i + 1]].all() for i in indices])
        np.testing.assert_array_equal(result, expected)


def queue_rotations():
    rounds = BATCH / "rounds"
    completed = [read_json(path) for path in sorted(rounds.glob("*/round.json"))]
    best = min((row for row in completed if row["config"]["kind"] == "filter" and row["status"] == "completed"),
               key=lambda row: row["dev_metrics"]["selection_rmse_deg"])
    designs = [("body", "all", 1., {}), ("world", "all", 1., {}), ("body", "rest", 1., {}),
               ("body", "all", 1., best["config"]["parameters"]), ("body", "all", 0.5, {})]
    created = []
    for i, (coordinates, motion, shrink, parameters) in enumerate(designs, 1):
        identity = f"I{i:05d}"
        directory = rounds / identity
        if (directory / "round.json").exists():
            continue
        config = {"kind": "install_rotation", "coordinates": coordinates, "fit_motion": motion,
                  "shrinkage": shrink, "parameters": parameters, "seed": None}
        item = {"round_id": identity, "batch": "installation-control", "hypothesis": "A shared proper rotation may explain repeatable SHICE tilt offsets; fit train folds only",
                "config": config, "config_sha256": fingerprint(config), "seed": None,
                "relation": best["round_id"] if parameters else "R00001",
                "partitions": {"development": names("dev"), "r5": names("r5"), "r6": names("r6"),
                               "excluded": list(EXCLUDED), "test_known_from_history": True},
                "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"}
        write_json(directory / "round.json", item)
        created.append(identity)
    return created


def repair_legacy():
    repaired, still_running = [], []
    for source in sorted((BATCH / "rounds").glob("*/round.json")):
        row = read_json(source)
        if row["config"]["kind"] != "ridge" or row["config"]["context"] <= 64:
            continue
        if row["status"] == "running":
            still_running.append(row["round_id"])
            continue
        if row["status"] != "completed":
            continue
        result = BATCH / row["result"]
        if not result.exists() or read_json(result).get("mask_version") == VERSION:
            continue
        evidence = source.parent / "legacy-context-v1"
        evidence.mkdir(exist_ok=True)
        shutil.copy2(source, evidence / "round.json")
        os.replace(result, evidence / "result.json")
        row.update(status="planned", ended_at=None, dev_metrics=None, context_repair_requested_at=stamp(),
                   previous_evidence=str(evidence.relative_to(BATCH)), previous_result_invalid_for_selection=True)
        write_json(source, row)
        repaired.append(row["round_id"])
    return repaired, still_running


if __name__ == "__main__":
    validate_support()
    queued = queue_rotations()
    repaired, running = repair_legacy()
    audit = {"timestamp": stamp(), "mask_version": VERSION, "synthetic_support_check": "pass",
             "queued_rotation_rounds": queued, "requeued_same_round_id": repaired,
             "long_window_rounds_currently_running": running}
    write_json(BATCH / "context-and-rotation-audit.json", audit)
    print(audit, flush=True)
