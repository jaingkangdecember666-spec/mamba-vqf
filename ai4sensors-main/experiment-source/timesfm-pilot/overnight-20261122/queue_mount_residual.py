"""Queue matched mounting-plus-residual trials in the current live batch."""

from common import BATCH, EXCLUDED, fingerprint, names, read_json, stamp, write_json
from selection_rules import dev_value, eligible


def queue():
    results = [read_json(p) for p in sorted((BATCH / "rounds").glob("*/round.json"))]
    parents = sorted([row for row in results if eligible(row) and row["config"]["kind"] == "ridge"
                      and not row["config"]["embedding"] and not row["config"].get("installation")], key=dev_value)
    if not parents:
        raise ValueError("No valid physical residual was scored before mounting integration")
    best = [parents[0]]
    distinct = next((row for row in parents if row["config"]["parameters"] != best[0]["config"]["parameters"]), None)
    if distinct:
        best.append(distinct)
    existing = {row["config_sha256"] for row in results}
    queued = []
    for idx, row in enumerate(best):
        for j, (tag, strength) in enumerate((("all", 1.), ("all", 0.5), ("rest", 1.))):
            identity = f"M{idx * 3 + j + 1:05d}"
            config = {**row["config"], "installation": tag, "mount_shrink": strength}
            sha = fingerprint(config)
            target = BATCH / "rounds" / identity / "round.json"
            if target.is_file() or sha in existing:
                continue
            item = {"round_id": identity, "batch": "installation-plus-residual", "hypothesis": "Fit a shared mounting rotation on the training fold, then learn only the remaining tilt residual with a matched physical-only readout",
                    "config": config, "config_sha256": sha, "seed": config["seed"], "relation": row["round_id"],
                    "partitions": {"development": names("dev"), "r5": names("r5"), "r6": names("r6"),
                                   "excluded": list(EXCLUDED), "test_known_from_history": True},
                    "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                    "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"}
            write_json(target, item)
            queued.append(identity)
    return queued


if __name__ == "__main__":
    scheduled = queue()
    write_json(BATCH / "mount-residual-queue.json", {"queued_at": stamp(), "rounds": scheduled})
    print(scheduled, flush=True)
