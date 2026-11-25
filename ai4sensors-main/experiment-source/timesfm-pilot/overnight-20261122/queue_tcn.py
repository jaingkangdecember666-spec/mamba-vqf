"""Queue a bounded TCN experiment group, and repeat only competitive families."""

import argparse

from common import BATCH, EXCLUDED, fingerprint, names, read_json, stamp, write_json
from selection_rules import dev_value, eligible


def schedule(rounds, identity, config, label, relation):
    target = BATCH / "rounds" / identity / "round.json"
    sha = fingerprint(config)
    if target.is_file() or any(row["config_sha256"] == sha for row in rounds):
        return False
    item = {"round_id": identity, "batch": label,
            "hypothesis": "Learn a causal temporal residual over native FullVQF and compare on the same device and program folds",
            "config": config, "config_sha256": sha, "seed": config["seed"], "relation": relation,
            "partitions": {"development": names("dev"), "r5": names("r5"), "r6": names("r6"),
                           "excluded": list(EXCLUDED), "test_known_from_history": True},
            "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
            "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"}
    write_json(target, item)
    return True


def physically_comparable(rows):
    candidates = [row for row in rows if eligible(row) and row["config"]["kind"] == "ridge"
                  and not row["config"]["embedding"] and not row["config"].get("installation")
                  and row["config"]["update_hz"] == 10]
    return min(candidates, key=dev_value)


def screen(rounds, parent):
    physical = parent["config"]
    config = {key: physical[key] for key in ("parameters", "features", "update_hz", "gain", "cap_deg", "acc_gate", "gyro_gate")}
    config.update(kind="tcn", context=64, embedding=0, channels=24, lr=0.0012, weight_decay=0.01,
                  epochs=6, steps=18, batch_size=192, seed=1729)
    variants = ({}, {"context": 32}, {"context": 128}, {"cap_deg": min(2.0, physical["cap_deg"] * 0.6)},
                {"weight_decay": 0.06})
    created = []
    for i, change in enumerate(variants, 1):
        identity = f"N{i:05d}"
        if schedule(rounds, identity, {**config, **change}, "causal-tcn-screen", parent["round_id"]):
            created.append(identity)
    return created


def repeat(rounds, parent):
    initial = [row for row in rounds if row["batch"] == "causal-tcn-screen"]
    if len(initial) < 5 or any(row["status"] in {"planned", "running"} for row in initial):
        return {"state": "screening_in_progress", "created": []}
    supported = [row for row in initial if eligible(row)]
    rules = lambda row: (dev_value(row) <= dev_value(parent) + 0.15
                         or (dev_value(row) <= dev_value(parent) + 0.25
                             and any(row["dev_metrics"][axis]["macro_rmse_deg"] < parent["dev_metrics"][axis]["macro_rmse_deg"]
                                     for axis in ("device", "motion"))))
    contenders = sorted((row for row in supported if rules(row)), key=dev_value)[:2]
    if not contenders:
        return {"state": "screened_but_no_competitive_family", "created": [],
                "scores": [{"round_id": row["round_id"], "dev_deg": dev_value(row)} for row in supported]}
    created = []
    for i, row in enumerate(contenders):
        for j, seed in enumerate((20261122, 20261123), 1):
            identity = f"N{21 + i * 2 + j:05d}"
            if schedule(rounds, identity, {**row["config"], "seed": seed}, "causal-tcn-seed-repeat", row["round_id"]):
                created.append(identity)
    return {"state": "replicate_competitive_families", "created": created,
            "contenders": [row["round_id"] for row in contenders]}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replicate", action="store_true")
    args = cli.parse_args()
    rows = [read_json(path) for path in sorted((BATCH / "rounds").glob("*/round.json"))]
    if args.replicate:
        original = read_json(BATCH / "tcn-screen-stage.json")["physical_control"]
        parent = next(row for row in rows if row["round_id"] == original)
    else:
        parent = physically_comparable(rows)
    result = repeat(rows, parent) if args.replicate else {"state": "screening", "created": screen(rows, parent)}
    envelope = {"timestamp": stamp(), "physical_control": parent["round_id"], "physical_control_dev_deg": dev_value(parent), **result}
    name = "tcn-repeat-stage.json" if args.replicate else "tcn-screen-stage.json"
    write_json(BATCH / name, envelope)
    print(envelope, flush=True)
