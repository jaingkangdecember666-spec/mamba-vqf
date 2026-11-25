"""Select new experiments from completed development evidence and reserve time for blind scoring."""

import random
from datetime import timedelta

from common import BATCH, LOCK, fingerprint, now, read_json
from selection_rules import build_shortlist
from strategy import filter_config, ranked, unused


SELECT_AT = LOCK - timedelta(minutes=20)


def learner(parameters, **changes):
    return {"kind": "ridge", "parameters": parameters, "features": "motion", "context": 64,
            "update_hz": 10, "alpha": 0.006, "gain": 0.85, "cap_deg": 5., "acc_gate": 0.,
            "gyro_gate": 0., "hidden": 0, "embedding": 0, "seed": 0, **changes}


def keep_filter_params(items):
    ordered = ranked(items, "filter")
    return [row["config"]["parameters"] for row in ordered[:3]] + [{}]


def initial_learners(parameters):
    return [learner(parameters, **changes) for changes in (
        {}, {"features": "compact"}, {"features": "quadratic"}, {"alpha": 0.0008}, {"alpha": 0.035},
        {"gain": 0.55}, {"gain": 1.0}, {"cap_deg": 2.5}, {"acc_gate": 0.045}, {"gyro_gate": 0.8},
        {"acc_gate": 0.06, "gyro_gate": 1.3}, {"context": 32}, {"context": 128},
        {"update_hz": 5}, {"update_hz": 20}, {"update_hz": 25},
        {"hidden": 64, "seed": 1729}, {"hidden": 128, "seed": 1729},
    )]


def score_key(row, axis):
    score = row["dev_metrics"][axis]
    return score["macro_rmse_deg"], score["worst_rmse_deg"], row["round_id"]


def final_job(items):
    chosen = next((row for row in items if row["config"]["kind"] == "selection" and row["status"] == "completed"), None)
    if chosen:
        selected = read_json(BATCH / chosen["result"])
        return ({"kind": "final", "selection_round": chosen["round_id"], "selection_sha256": selected["lock_sha256"],
                 "seed": selected["selected"]["config"].get("seed")},
                "Score the locked candidate and all four historical methods on identical held-out frames", "final", chosen["round_id"])
    return selection_job(items)


def selection_job(items):
    ids = [row["round_id"] for row in build_shortlist(items)]
    return ({"kind": "selection", "candidate_rounds": ids, "seed": None},
            "Evaluate only the development shortlist on R5 and serialize the winning configuration and weights", "r5-selection", None)


def next_experiment(items):
    if any(r["config"]["kind"] in {"selection", "final"} and r["status"] == "completed" for r in items):
        return final_job(items)
    if now() >= SELECT_AT:
        return selection_job(items)

    filters = keep_filter_params(items)
    for parameters in (filters[0], {}):
        for config in initial_learners(parameters):
            if unused(items, config):
                return (config, "Compare causal residual features, shrinkage and update cadence under device and motion holdout",
                        "residual-screen", None)

    # Use matched physical-only configurations for each frozen TimesFM ablation.
    ridges = ranked(items, "ridge")
    plain = [row for row in ridges if row["config"]["embedding"] == 0 and row["config"]["update_hz"] == 10]
    source_rounds = sorted([row for row in plain if row["batch"] == "residual-screen"],
                           key=lambda row: (row["dev_metrics"]["selection_rmse_deg"], row["round_id"]))[:4]
    for row in source_rounds:
        for embed in (8, 16):
            config = {**row["config"], "embedding": embed}
            if unused(items, config):
                return (config, "Add frozen TimesFM representations to the same features, fold, gate and readout",
                        "matched-ablation", row["round_id"])

    completed_count = len(ridges)
    if completed_count % 16 < 3 and ridges:
        top_random = [row for row in ridges if row["config"]["hidden"] or row["config"]["embedding"]][:3]
        for row in top_random:
            for seed in (20261122, 20261123, 20261124):
                config = {**row["config"], "seed": seed}
                if unused(items, config):
                    return config, "Repeat the same random feature configuration with an independent seed", "seed-repeat", row["round_id"]

    # Favor both device and program performance and observe the worst-record tail.
    winners = ridges[:6]
    winners += sorted(ridges, key=lambda row: score_key(row, "device"))[:3]
    winners += sorted(ridges, key=lambda row: score_key(row, "motion"))[:3]
    winners += sorted(ridges, key=lambda row: max(row["dev_metrics"][c]["worst_rmse_deg"] for c in ("device", "motion")))[:3]
    if not winners:
        return selection_job(items)
    rng = random.Random(20261122 + len(items))
    for attempt in range(400):
        parent = rng.choice(winners)
        config = parent["config"].copy()
        dev = parent["dev_metrics"]["device"]
        motion = parent["dev_metrics"]["motion"]
        worst_delta = max(dev["worst_delta_deg"], motion["worst_delta_deg"])
        modes = ["gain", "alpha", "context", "reliability", "cap", "rate", "features", "filter", "hidden"]
        if worst_delta > 0.25:
            modes += ["reliability", "cap", "gain"]
        if motion["macro_rmse_deg"] > dev["macro_rmse_deg"] + 0.15:
            modes += ["reliability", "features"]
        mutations = rng.sample(range(len(modes)), k=rng.choice((1, 1, 2, 3)))
        labels = []
        for index in mutations:
            mode = modes[index]
            labels.append(mode)
            if mode == "gain":
                config["gain"] = round(min(1.2, max(0.12, config["gain"] * rng.uniform(0.72, 1.24))), 3)
            elif mode == "alpha":
                config["alpha"] = round(min(0.35, max(0.00015, config["alpha"] * 10**rng.uniform(-0.6, 0.6))), 7)
            elif mode == "context":
                config["context"] = rng.choice((24, 32, 48, 64, 80, 96, 128, 160, 200))
            elif mode == "reliability":
                config["acc_gate"] = rng.choice((0., 0.025, 0.04, 0.07, 0.12, 0.2))
                config["gyro_gate"] = rng.choice((0., 0.45, 0.65, 0.9, 1.3, 1.8, 2.5))
            elif mode == "cap":
                config["cap_deg"] = rng.choice((1., 1.5, 2., 3., 4., 5., 6., 8.))
            elif mode == "rate":
                config["update_hz"] = rng.choice((4, 5, 10, 20, 25, 50))
            elif mode == "features":
                config["features"] = rng.choice(("compact", "motion", "quadratic"))
            elif mode == "filter":
                config["parameters"] = rng.choice(filters)
            elif mode == "hidden":
                config["hidden"] = rng.choice((0, 24, 48, 64, 96, 128, 192))
        if config["embedding"]:
            config["update_hz"] = 10
            config["context"] = 64
        if config["hidden"] or config["embedding"]:
            config["seed"] = config["seed"] or 1729
        else:
            config["seed"] = 0
        if unused(items, config):
            hypothesis = (f"Probe {', '.join(labels)} around {parent['round_id']}; "
                          f"parent device {dev['macro_rmse_deg']:.3f}°, motion {motion['macro_rmse_deg']:.3f}°, "
                          f"worst delta {worst_delta:.3f}°")
            return config, hypothesis, "adaptive-refinement", parent["round_id"]
    return selection_job(items)
