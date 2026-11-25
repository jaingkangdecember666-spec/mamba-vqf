"""Reproduce the SHICE baseline before selecting any overnight candidates."""

import argparse
from pathlib import Path

import numpy as np

from common import CACHE, HISTORY, PILOT, aggregate, body_up, digest, names, read_json, record, rmse, stamp, vqf_outputs, write_json


def run(output):
    rows = {}
    started = stamp()
    prior = read_json(HISTORY / "validation.json")
    old = {row["recording"]: row["rmse_deg"]["full_vqf"] for row in prior["r5_records"]}
    for split in ("dev", "r5"):
        members = []
        for name in names(split):
            inp = record(name)
            valid = inp["score"]
            vqf = vqf_outputs(inp["imu"], inp["dt"], {})
            score = rmse(body_up(vqf["quat6D"][valid]), inp["ref"][valid])
            if name in old and abs(score - old[name]) > 1e-5:
                raise ValueError(f"Baseline did not reproduce for {name}: {score} vs {old[name]}")
            members.append({"name": name, "rmse_deg": score, "frames": int(valid.sum()),
                            "input_cache_sha256": digest(CACHE / f"{name}.npz")})
        rows[split] = {**aggregate(members), "sequences": members}
        print(split, aggregate(members), flush=True)
    history = (HISTORY / "validation.json", HISTORY / "test.json", HISTORY / "external_test.json",
               HISTORY / "physical_only_head.pt", HISTORY / "timesfm_head.pt",
               PILOT / "gru-tilt" / "test.json", PILOT / "gru-tilt" / "longer" / "gru.pt")
    evidence = [{"path": str(path), "sha256": digest(path)} for path in history]
    write_json(output, {"stage": "baseline_recomputed", "started_at": started, "ended_at": stamp(),
                        "dev_metrics": rows["dev"], "r5": rows["r5"],
                        "prior_r5_max_abs_error_deg": max(abs(row["rmse_deg"] - old[row["name"]]) for row in rows["r5"]["sequences"]),
                        "historical_evidence_index": evidence})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    run(parser.parse_args().result)
