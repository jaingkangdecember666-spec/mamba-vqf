"""Publish a reviewable index of unique trials, attempts and end-of-window evidence."""

from collections import Counter

from common import BATCH, digest, fingerprint, read_json, stamp, write_json


def experiment_index(final_round_id=None):
    indexed = []
    statuses = Counter()
    exact, families = set(), set()
    seeds = 0
    for source in sorted((BATCH / "rounds").glob("*/round.json")):
        row = read_json(source)
        path = BATCH / row["result"]
        evidence = read_json(path) if path.is_file() else None
        status = row["status"]
        if row["round_id"] == final_round_id and evidence and evidence.get("core_completed_at"):
            status = "completed_core_artifact" if "completed_at" not in evidence else "completed"
        config = row["config"]
        kind = config["kind"]
        development_trial = kind in {"baseline", "filter", "ridge", "install_rotation", "tcn", "equivariant", "hybrid", "legacy_head", "ensemble", "stream_gru", "ema_readout"}
        count_status = status
        if development_trial and status == "completed" and kind in {"ridge", "tcn", "equivariant"} and config["context"] > 64:
            if not evidence or evidence.get("mask_version") != "causal-bin-v2-full-window":
                count_status = "invalid_legacy_window"
        if kind == "stream_gru" and status == "completed" and (
                not evidence or evidence.get("recurrent_training_context") != "split-state-at-forbidden-context-v2"):
            count_status = "invalid_recurrent_context_v1"
        if development_trial:
            statuses[count_status] += 1
            if count_status == "completed":
                exact.add(row["config_sha256"])
                families.add(fingerprint({key: value for key, value in config.items() if key != "seed"}))
                seeds += int("seed-repeat" in row["batch"])
        indexed.append({"round_id": row["round_id"], "batch": row["batch"], "kind": kind, "status": count_status,
                        "config_sha256": row["config_sha256"], "seed": row["seed"], "relation": row["relation"],
                        "started_at": row["started_at"], "ended_at": row["ended_at"], "attempts": row.get("attempts", 0),
                        "dev_metrics": row["dev_metrics"], "record": str(source.relative_to(BATCH)),
                        "result": row["result"], "result_sha256": digest(path) if path.is_file() else None,
                        "log": row["stdout"], "previous_evidence": row.get("previous_evidence")})
    snapshot = {"created_at": stamp(), "final_round_id": final_round_id,
                "development_counts": dict(statuses), "completed_exact_configurations": len(exact),
                "completed_configurations_ignoring_seed": len(families), "completed_seed_repeats": seeds,
                "retry_attempts_not_new_configurations": sum(max(row["attempts"] - 1, 0) for row in indexed),
                "historical_result_reuse_not_new_development_round": True,
                "selection_and_final_not_counted_as_new_configuration": True,
                "index": indexed}
    write_json(BATCH / "experiment-index.json", snapshot)
    return snapshot


if __name__ == "__main__":
    report = experiment_index()
    print({key: value for key, value in report.items() if key != "index"}, flush=True)
