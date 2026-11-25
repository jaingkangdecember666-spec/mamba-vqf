"""Replay both residual caps from one fitted ridge prediction per fold and seed."""

import os
from hashlib import sha256
from time import perf_counter

import numpy as np

import learning
from common import BATCH, HISTORY, Z, aggregate, digest, fingerprint, names, read_json, rmse, rotate, stamp, unit
from features import VERSION
from stream_gru import planned_folds


PROTOCOL = BATCH / "ridge-cap-4deg-6deg-protocol.json"
CAPS = (4., 6.)


def replay(part, raw, configs):
    up = part["up"]
    projected = raw.copy()
    projected -= (projected * up).sum(axis=1, keepdims=True) * up
    magnitude = np.linalg.norm(projected, axis=1)
    available = np.searchsorted(part["ready"], part["index"], side="right")
    support = np.bincount(available, minlength=len(raw) + 1)[1:]
    scores, statistics = {}, {}
    for cap in CAPS:
        tangent = learning.constrain_delta(part, raw, configs[cap])
        corrected = unit(rotate(unit(part["q"]), tangent) + Z)
        output = np.concatenate((Z[None, :], corrected), axis=0)[available]
        scores[str(cap)] = rmse(output, part["ref_world"])
        hit = magnitude > np.tan(np.deg2rad(cap))
        statistics[str(cap)] = {"all_candidate_updates": len(raw), "updates_over_cap": int(hit.sum()),
                               "updates_over_cap_used_on_score_frames": int(np.count_nonzero(hit & (support > 0))),
                               "original_score_frames_using_clipped_update": int(support[hit].sum())}
    return scores, statistics


def existing_anchors(stage):
    control = {}
    for entry in stage["existing_or_queued"]:
        path = BATCH / "rounds" / entry["round_id"]
        source = read_json(path / "round.json")
        if source["config_sha256"] != entry["configuration_sha256"]:
            raise ValueError("Existing cap comparison configuration changed")
        if source["status"] != "completed":
            continue
        result_path = path / "result.json"
        if entry.get("result_sha256") and digest(result_path) != entry["result_sha256"]:
            raise ValueError("Historical cap score changed after registration")
        result = read_json(result_path)
        if result.get("mask_version") != VERSION:
            raise ValueError("Historic cap anchor requires the full 96-block training windows")
        if entry["category"] == "joint":
            sequences = {"joint_holdout": result["sequences"]}
        else:
            sequences = result["sequences"]
        for axis, rows in sequences.items():
            control[entry["seed"], entry["cap_deg"], axis] = ({r["name"]: r for r in rows}, entry["round_id"])
    return control


def summarize(outputs):
    result = {}
    for seed, axes in outputs.items():
        result[seed] = {}
        for axis, rows in axes.items():
            rmse_by_cap = {}
            clip_by_cap = {}
            for cap in map(str, CAPS):
                rmse_by_cap[cap] = aggregate([{"name": row["name"], "frames": row["frames"],
                                               "rmse_deg": row["rmse_deg"][cap]} for row in rows])
                clip_by_cap[cap] = {key: sum(row["clip"][cap][key] for row in rows)
                                    for key in rows[0]["clip"][cap]}
            differences = np.asarray([row["six_minus_four_deg"] for row in rows])
            result[seed][axis] = {"errors": rmse_by_cap, "clip": clip_by_cap, "mean_six_minus_four_deg": float(differences.mean()),
                                  "improved_at_six": int((differences < -1e-10).sum()),
                                  "degraded_at_six": int((differences > 1e-10).sum()),
                                  "equal_at_six": int((np.abs(differences) <= 1e-10).sum())}
    return result


def run(item):
    since = perf_counter()
    stage = read_json(PROTOCOL)
    if item["config"]["protocol_sha256"] != digest(PROTOCOL) or item["round_id"] != stage["round_id"]:
        raise ValueError("Unexpected cap-pair protocol or identity")
    for filename, sha in stage["historical_scoring_code_sha256"].items():
        if digest(BATCH / filename) != sha:
            raise ValueError(f"Historical scoring implementation changed: {filename}")
    base = stage["shared_config"]
    records = names("dev")
    plan = planned_folds(records)
    if plan != [tuple(fold) for fold in stage["folds"]]:
        raise ValueError("Cap pair development folds changed")
    print("ridge cap pair", item["round_id"], "python worker", os.getpid(), stamp(), flush=True)
    controls = existing_anchors(stage)
    raw = {name: learning.attach_native(name, base) for name in records}
    frame_ids = {}
    for name, data in raw.items():
        with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as original:
            np.testing.assert_array_equal(data["index"], np.flatnonzero(original["score"]))
        frame_ids[name] = sha256(np.asarray(data["index"], dtype="<i8").tobytes()).hexdigest()
    seeds = stage["search_seeds"]
    outputs = {str(seed): {axis: [] for axis in ("device", "motion", "joint_holdout")} for seed in seeds}
    evidence = []
    maximum_delta = 0.
    for axis, fold, training, held in plan:
        q = learning.installation_fit([raw[name] for name in training], base)
        prepared = {name: learning.prepare(learning.apply_install(raw[name], q, include_reference=name in held), base)
                    for name in training + held}
        fit_count, prediction_count = 0, 0
        for seed in seeds:
            caps = {cap: {**base, "seed": seed, "cap_deg": cap} for cap in CAPS}
            state = learning.fit([prepared[name] for name in training], caps[4.])
            fit_count += 1
            for name in held:
                data = prepared[name]
                candidate = learning.design(data["matrix"], state) @ state["coef"]
                prediction_count += 1
                scores, clip = replay(data, candidate, caps)
                anchor_check = {}
                for cap in CAPS:
                    entry = controls.get((seed, cap, axis))
                    if entry is None:
                        continue
                    original, anchor = entry
                    previous = original[name]
                    delta = abs(previous["rmse_deg"] - scores[str(cap)])
                    if delta > 1e-6 or previous["frames"] != len(data["index"]):
                        raise ValueError(f"Existing score did not reproduce: {anchor} {axis}/{name}")
                    maximum_delta = max(maximum_delta, delta)
                    anchor_check[str(cap)] = {"round": anchor, "absolute_delta_deg": delta}
                outputs[str(seed)][axis].append({"name": name, "fold": fold, "frames": len(data["index"]),
                                               "score_frame_indices_sha256": frame_ids[name], "rmse_deg": scores,
                                               "six_minus_four_deg": scores["6.0"] - scores["4.0"], "clip": clip,
                                               "existing_score_reproduction": anchor_check})
        evidence.append({"axis": axis, "fold": fold, "training": training, "validation": held,
                         "mount_quaternion_wxyz": q.tolist(), "ridge_fits": fit_count,
                         "unclipped_prediction_vectors": prediction_count})
        print(axis, fold, "shared fits", fit_count, "shared residual arrays", prediction_count, flush=True)
    for axes in outputs.values():
        for axis, rows in axes.items():
            if sorted(row["name"] for row in rows) != sorted(records) or sum(row["frames"] for row in rows) != 8328711:
                raise ValueError(f"Lost historical development score frames in {axis}")
    return {"dev_metrics": None, "sequences_by_seed": outputs, "summary_by_seed": summarize(outputs), "folds": evidence,
            "ridge_fits": sum(fold["ridge_fits"] for fold in evidence), "shared_mount_fits": len(evidence),
            "unclipped_prediction_arrays": sum(fold["unclipped_prediction_vectors"] for fold in evidence),
            "maximum_existing_anchor_difference_deg": maximum_delta, "eligible_for_r5": False,
            "failed_fits": 0, "python_pid": os.getpid(), "elapsed_seconds": perf_counter() - since}
