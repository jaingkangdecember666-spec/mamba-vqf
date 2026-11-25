"""Replay adaptive and seed-repeat proposals from historical development fields."""

import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from audit_rounds import BATCH, HERE, fingerprint, read, write
from verify_baselines import digest


def audit(through):
    import json
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    import advanced_strategy, selection_rules, strategy

    horizon = datetime.fromisoformat(through)
    frames = (BATCH/"ledger.jsonl").read_bytes().splitlines(keepends=True)
    entries = [json.loads(line) for line in frames if line.endswith(b"\n")]
    registered = sorted(read(BATCH/"round_index.json"),key=lambda r:r["round_id"])
    roster = []
    for row in registered:
        cfg = row["config"]
        roster.append((row["planned_at"],{"round_id":row["round_id"],"config_sha256":row["config_sha256"],"batch":row["batch"],
            "config":cfg if cfg["kind"] in {"ridge","filter"} else {"kind":cfg["kind"]},"status":"planned","dev_metrics":None}))
    source = {name:digest(BATCH/name) for name in ("strategy.py","advanced_strategy.py")}
    now = horizon
    def denied(*args, **kwargs):
        raise AssertionError("Proposal attempted to read a file")
    results, finished, mismatches, concurrent = [], {}, [], []
    previous_finish = None
    classes = {"adaptive-refinement","seed-repeat"}
    with ExitStack() as stack:
        for module in (advanced_strategy,strategy):
            stack.enter_context(patch.object(module,"now",side_effect=lambda: now))
        for module in (strategy,advanced_strategy,selection_rules):
            stack.enter_context(patch.object(module,"read_json",side_effect=denied))
        stack.enter_context(patch.object(Path,"open",side_effect=denied))
        stack.enter_context(patch("builtins.open",side_effect=denied))
        for event in entries:
            identity = event.get("round_id")
            if not identity or datetime.fromisoformat(event["recorded_at"]) > horizon:
                continue
            if event["event"] in {"completed","failed","cutoff"}:
                previous_finish = event["recorded_at"]
            if event["event"] == "planned" and event["batch"] in classes:
                now = datetime.fromisoformat(event["planned_at"])
                prior = [finished.get(item["round_id"],item) for timestamp,item in roster if timestamp < event["planned_at"]]
                config,why,category,parent = strategy.next_job(prior)
                def same(config,why,category,parent):
                    return (config == event["config"] and fingerprint(config) == event["config_sha256"] and
                            (why,category,parent) == (event["hypothesis"],event["batch"],event["relation"]))
                match = same(config,why,category,parent)
                basis = "registrations_before_plan"
                if not match and previous_finish is not None:
                    added = [item["round_id"] for timestamp,item in roster if previous_finish < timestamp < event["planned_at"]]
                    if added:
                        previous = [r for r in prior if r["round_id"] not in added]
                        match = same(*strategy.next_job(previous))
                        if match:
                            basis = "last_completion_before_concurrent_registrations"
                            concurrent.append({"id":identity,"previous_completion":previous_finish,"planned_at":event["planned_at"],
                                               "omitted_registrations":added})
                results.append({"id":identity,"at":event["planned_at"],"category":event["batch"],"matches":match,"basis":basis})
                if not match:
                    mismatches.append({"id":identity,"prior_items":len(prior),"replayed_parent":parent,"recorded_parent":event["relation"],
                                       "replayed_sha256":fingerprint(config),"recorded_sha256":event["config_sha256"]})
            if event["config"]["kind"] not in {"ridge","filter"}:
                continue
            if event["event"] != "completed":
                finished.pop(identity,None)
                continue
            source_metrics = event["dev_metrics"]
            scores = {"selection_rmse_deg":source_metrics["selection_rmse_deg"]}
            for axis in ("device","motion"):
                scores[axis] = {k:source_metrics[axis][k] for k in ("macro_rmse_deg","worst_rmse_deg","worst_delta_deg")}
            finished[identity] = {"round_id":identity,"config_sha256":event["config_sha256"],"config":event["config"],
                                  "batch":event["batch"],"status":"completed","dev_metrics":scores}
    assert results and len({r["id"] for r in results}) == len(results)
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"through":through,"source_sha256":source,
            "proposals":len(results),"matches":sum(r["matches"] for r in results),"counts_by_category":dict(Counter(r["category"] for r in results)),
            "first":results[0]["at"],"last":results[-1]["at"],"results":results,"concurrent_registrations":concurrent,"mismatches":mismatches}


if __name__ == "__main__":
    result = audit(sys.argv[1] if len(sys.argv)>1 else datetime.now().astimezone().isoformat())
    write(HERE/"proposal_replay.json",result)
    print(result["proposals"],result["matches"],result["mismatches"][:3])
