"""Benchmark the reviewed development candidates on one development stream."""

import os

from common import BATCH, PILOT, digest, fingerprint, names, read_json, stamp, write_json


def run(job, destination):
    import torch
    from final_eval import bench_locked, fit_candidate
    from tilt_model import load_backbone

    config = job["config"]
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    rows, failures = [], []
    backbone = None
    for entry in config["benchmarks"]:
        identity = entry["id"]
        parent = read_json(BATCH / "rounds" / identity / "round.json")
        model = parent["config"]
        if fingerprint(model) != entry["config_sha256"]:
            raise ValueError(f"Parent configuration {identity} was modified")
        if entry["embedding_off"]:
            model = {**model, "embedding": 0}
        try:
            if model.get("embedding") and backbone is None:
                backbone = load_backbone(PILOT / "checkpoint", device)
            state = fit_candidate(model, names("dev"))
            result = bench_locked(model, state, backbone, device, config["record"])
            rows.append({"development_round": identity, "matched_without_embedding": entry["embedding_off"],
                         "config_sha256": fingerprint(model), "kind": model["kind"], "measurements": result})
            print(rows[-1], flush=True)
        except Exception as error:
            failures.append({"development_round": identity, "embedding_off": entry["embedding_off"], "error": repr(error)})
            print("Development latency failed", failures[-1], flush=True)
    report = {"round_id": job["round_id"], "created_at": stamp(), "benchmarks": rows, "failures": failures,
              "device": torch.cuda.get_device_name(), "host": os.environ["COMPUTERNAME"],
              "protocol": {"benchmark_record": config["record"], "batch": 1, "development_only": True,
                           "native_updates": 3000, "correction_updates": 100, "correction_warmup": 10,
                           "native_excludes_ingest_and_bin_construction": True}}
    write_json(destination, report)


def queue():
    stage = BATCH / "joint-holdout-stage.json"
    plan = read_json(stage)
    identities = [plan["pure_fullvqf_ref"], *[entry["parent_round"] for entry in plan["candidates"][:5]]]
    entries = [{"id": value, "embedding_off": False} for value in identities]
    entries.append({"id": identities[-1], "embedding_off": True})
    for entry in entries:
        parent = read_json(BATCH / "rounds" / entry["id"] / "round.json")
        entry["config_sha256"] = parent["config_sha256"]
    config = {"kind": "development_latency", "record": "R3-3", "benchmarks": entries}
    identity = "L00001"
    destination = BATCH / "rounds" / identity / "round.json"
    if not destination.is_file():
        write_json(destination, {"round_id": identity, "config": config, "config_sha256": fingerprint(config),
                                "seed": None, "batch": "development-latency", "relation": None,
                                "hypothesis": "Measure comparable batch-one native-rate and correction-update costs on one development stream",
                                "partitions": {"benchmark": [config["record"]], "fit": names("dev")},
                                "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned",
                                "attempts": 0, "dev_metrics": None, "result": f"rounds/{identity}/result.json",
                                "stdout": f"rounds/{identity}/run.log"})
    print(identity, "source", str(stage), digest(stage))


if __name__ == "__main__":
    queue()
