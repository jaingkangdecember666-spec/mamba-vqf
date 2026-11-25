"""Matched causal echo-state and reset-each-update linear tilt readouts."""

import os
from hashlib import sha256
from time import perf_counter

import numpy as np

import learning
from common import BATCH, HISTORY, aggregate, by_device, digest, errors, fingerprint, names, read_json, record, stamp, write_json
from ensemble import measured_prior
from experiment import baseline_reference
from features import VERSION, cache_record
from stream_gru import planned_folds, prepare, training_segments


PROTOCOL = BATCH / "reservoir-development-protocol.json"
ARMS = ("continuous", "reset_each_update")
PHYSICAL_INPUTS = 34
WIDTH = 32
TARGET_OPERATOR_NORM = .80
LEAK = .50


def fixed_weights(seed):
    rng = np.random.default_rng(seed)
    drive = rng.normal(0., .55 / np.sqrt(PHYSICAL_INPUTS), size=(WIDTH, PHYSICAL_INPUTS))
    recurrent = rng.normal(0., 1. / np.sqrt(WIDTH), size=(WIDTH, WIDTH))
    recurrent *= TARGET_OPERATOR_NORM / np.linalg.norm(recurrent, ord=2)
    bias = rng.uniform(-.15, .15, size=WIDTH)
    return {"drive": drive, "recurrent": recurrent, "bias": bias}


def weight_hash(weights):
    sha = sha256()
    for name, value in sorted(weights.items()):
        sha.update(name.encode("utf-8"))
        sha.update(np.asarray(value, dtype="<f8").tobytes())
    return sha.hexdigest()


def input_scale(training):
    supervised = [p["matrix"][p["train"]].astype(np.float64) for p in training]
    if not all(len(x) for x in supervised):
        raise ValueError("A training record has no valid labels")
    mean = np.mean([x.mean(axis=0) for x in supervised], axis=0)
    squares = np.mean([np.square(x).mean(axis=0) for x in supervised], axis=0)
    return mean, np.sqrt(np.maximum(squares - mean**2, 9e-6))


def state_sequences(physical, scaler, weights, allowed=None):
    if physical.ndim != 2 or physical.shape[1] != PHYSICAL_INPUTS:
        raise ValueError("Unexpected physical input width")
    center, scale = scaler
    scaled = ((physical.astype(np.float64) - center) / scale).clip(-8., 8.)
    width = len(weights["bias"])
    allowed = np.ones(len(physical), dtype=bool) if allowed is None else allowed
    if len(allowed) != len(physical):
        raise ValueError("Reservoir state must share the original update mask")
    continuous = np.zeros((len(scaled), width), dtype=np.float32)
    reset_each = np.zeros_like(continuous)
    state = np.zeros(width, dtype=np.float64)
    drive_weights, recurrent, bias = weights["drive"], weights["recurrent"], weights["bias"]
    for i, vector in enumerate(scaled):
        if not allowed[i]:
            state.fill(0.)
            continue
        drive = drive_weights @ vector + bias
        state = (1. - LEAK) * state + LEAK * np.tanh(drive + recurrent @ state)
        continuous[i] = state
        reset_each[i] = LEAK * np.tanh(drive)
    return dict(zip(ARMS, (continuous, reset_each), strict=True))


def make_designs(part, scaler, weights, *, training=False):
    up = part["matrix"][:, :3]
    states = state_sequences(part["matrix"], scaler, weights, part["state_input_ok"] if training else None)
    shared = {**part, "up": up}
    if training:
        delta = part["target"] - up
        shared["target"] = delta - (delta * up).sum(axis=1, keepdims=True) * up
    return {method: {**shared, "matrix": np.column_stack((part["matrix"], states[method]))} for method in ARMS}


def synthetic(weights):
    rng = np.random.default_rng(7)
    physical = rng.standard_normal((34, PHYSICAL_INPUTS), dtype=np.float32)
    scaler = (np.zeros(PHYSICAL_INPUTS), np.ones(PHYSICAL_INPUTS))
    allowed = np.ones(len(physical), dtype=bool)
    allowed[12:15] = False
    disturbed = physical.copy()
    disturbed[:15] += 2.
    normal = state_sequences(physical, scaler, weights, allowed)
    altered = state_sequences(disturbed, scaler, weights, allowed)
    positive = state_sequences(disturbed, scaler, weights)
    original_unmasked = state_sequences(physical, scaler, weights)
    for arm in ARMS:
        np.testing.assert_array_equal(normal[arm][15:], altered[arm][15:])
        np.testing.assert_array_equal(normal[arm][12:15], 0.)
        prefix = state_sequences(physical[:20], scaler, weights, allowed[:20])[arm]
        np.testing.assert_array_equal(prefix, normal[arm][:20])
    effect = float(np.max(np.abs(positive["continuous"][15:] - original_unmasked["continuous"][15:])))
    if effect < 1e-4:
        raise ValueError("Synthetic boundary test lacked a measurable state-dependence control")
    return {"forbidden_inputs_and_earlier_state_cannot_contaminate_next_segment": True,
            "states_at_forbidden_windows_are_zero": True, "exact_state_prefix": True,
            "positive_unsegmented_state_change": effect}


def raw_prefix_check(name, full, mount, scaler, weights, states, config):
    source = record(name)
    stop = int(len(source["imu"]) * .57)
    cached = cache_record(source["imu"][:stop], float(source["dt"]), source["score"][:stop],
                          source["context"][:stop], source["ref"][:stop], config["parameters"], True,
                          keep_scored_quaternion=True)
    part = prepare(learning.apply_install(cached, mount, include_reference=True), config)
    prefixes = make_designs(part, scaler, weights)
    result = {"record": name, "native_prefix_frames": stop, "arms": {}}
    for arm in ARMS:
        common = prefixes[arm]
        wide = full[arm]
        np.testing.assert_array_equal(common["ready"], wide["ready"][:len(common["ready"])])
        np.testing.assert_array_equal(common["index"], wide["index"][wide["index"] < stop])
        np.testing.assert_allclose(common["matrix"], wide["matrix"][:len(common["ready"])], atol=2e-7, rtol=0.)
        original_direction, _ = learning.frame_directions(wide, states[arm], config)
        prefix_direction, _ = learning.frame_directions(common, states[arm], config)
        gap = float(np.max(errors(prefix_direction, original_direction[:len(common["index"])])))
        if gap > 1e-4:
            raise ValueError(f"Reservoir native prefix shifted: {arm} {gap}°")
        result["arms"][arm] = {"updates": len(common["ready"]), "scored_frames": len(common["index"]),
                                "maximum_direction_change_deg": gap}
    return result


def run(item):
    started = perf_counter()
    config = item["config"]
    protocol = read_json(PROTOCOL)
    if config["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Fixed reservoir protocol changed")
    candidate = protocol["first_round"]
    if item["round_id"] != candidate["round_id"] or config["seed"] != candidate["seed"]:
        raise ValueError("The preregistered first round has a different identity")
    weights_path = BATCH / candidate["matrix_file"]
    if digest(weights_path) != candidate["matrix_file_sha256"]:
        raise ValueError("The fixed dynamic matrices changed on disk")
    with np.load(weights_path, allow_pickle=False) as opened:
        weights = {k: opened[k] for k in opened.files}
    if weight_hash(weights) != candidate["matrix_content_sha256"] or weight_hash(fixed_weights(config["seed"])) != weight_hash(weights):
        raise ValueError("The preregistered seed did not reproduce all reservoir weights")
    for anchor, sha in candidate["anchor_result_sha256"].items():
        if digest(BATCH / anchor) != sha:
            raise ValueError(f"Archived lightweight control changed: {anchor}")
    print("reservoir", item["round_id"], "python worker", os.getpid(), "starting", stamp(), flush=True)
    checks = synthetic(weights)
    records = names("dev")
    raw = {name: learning.attach_native(name, config) for name in records}
    expected = measured_prior({"round_id": candidate["head_round"], "joint_round": candidate["joint_round"]})
    baseline = baseline_reference()
    score_indices = {}
    for name, data in raw.items():
        with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(data["index"], np.flatnonzero(old["score"]))
        score_indices[name] = sha256(np.asarray(data["index"], dtype="<i8").tobytes()).hexdigest()

    predictions, fold_results = {}, []
    prefix_checked = None
    for axis, label, training, held in planned_folds(records):
        q = learning.installation_fit([raw[n] for n in training], config)
        base = {name: prepare(learning.apply_install(raw[name], q, include_reference=name in held), config,
                              for_training=name in training) for name in training + held}
        scale = input_scale([base[n] for n in training])
        designs = {name: make_designs(part, scale, weights, training=name in training) for name, part in base.items()}
        fit_started = perf_counter()
        states = {arm: learning.fit([designs[name][arm] for name in training], config) for arm in ARMS}
        fit_seconds = perf_counter() - fit_started
        rows = []
        for name in held:
            immutable = expected[axis][name]
            current = designs[name]
            n = len(base[name]["index"])
            if n != immutable["frames"]:
                raise ValueError(f"Historical lightweight scoring coverage changed for {name}")
            scores = {arm: learning.evaluate(current[arm], states[arm], config) for arm in ARMS}
            rows.append({"name": name, "fold": label, "frames": n, "score_frame_indices_sha256": score_indices[name],
                         "baseline_deg": baseline[name], "rmse_deg": scores["continuous"],
                         "reset_each_update_rmse_deg": scores["reset_each_update"],
                         "legacy_head_rmse_deg": immutable["rmse_deg"]})
            if axis == "device" and name == "R2-10":
                prefix_checked = raw_prefix_check(name, current, q, scale, weights, states, config)
        predictions.setdefault(axis, []).extend(rows)
        fold_results.append({"axis": axis, "fold": label, "training": training, "validation": held, "model_fits": 2,
                             "readout_fit_seconds": fit_seconds, "mount_wxyz": q.tolist(),
                             "projection_scale_sha256": weight_hash(dict(zip(("center", "scale"), scale, strict=True))),
                             "training_labels": {n: int(base[n]["train"].sum()) for n in training},
                             "training_state_segments": {n: sum(1 for _ in training_segments(base[n])) for n in training}})
        print(axis, label, aggregate(rows)["macro_rmse_deg"], flush=True)

    metrics = {}
    for axis, rows in predictions.items():
        if sorted(row["name"] for row in rows) != sorted(records) or sum(row["frames"] for row in rows) != 8328711:
            raise ValueError(f"Lost an original native scoring frame in {axis}")
        reset = [{**row, "rmse_deg": row["reset_each_update_rmse_deg"]} for row in rows]
        legacy = [{**row, "rmse_deg": row["legacy_head_rmse_deg"]} for row in rows]
        stats = {method: {**aggregate(group), "per_device": by_device(group)}
                 for method, group in (("continuous", rows), ("reset_each_update", reset), ("legacy_head", legacy))}
        stats["continuous_minus_reset_deg"] = stats["continuous"]["macro_rmse_deg"] - stats["reset_each_update"]["macro_rmse_deg"]
        stats["continuous_minus_legacy_deg"] = stats["continuous"]["macro_rmse_deg"] - stats["legacy_head"]["macro_rmse_deg"]
        metrics[axis] = stats
    gate = protocol["promotion"]
    minimum = gate["joint_minimum_improvement_deg"]
    single = gate["axis_maximum_regression_deg"]
    criteria = {"joint_beats_reset": metrics["joint_holdout"]["continuous_minus_reset_deg"] <= -minimum,
                "joint_beats_legacy": metrics["joint_holdout"]["continuous_minus_legacy_deg"] <= -minimum,
                "held_device_acceptable": metrics["device"]["continuous_minus_legacy_deg"] <= single,
                "held_program_acceptable": metrics["motion"]["continuous_minus_legacy_deg"] <= single}
    if len(fold_results) != 24 or prefix_checked is None:
        raise ValueError("Incomplete reservoir development protocol")
    return {"dev_metrics": None, "axis_metrics": metrics, "sequences": predictions, "folds": fold_results,
            "actual_readout_fits": sum(fold["model_fits"] for fold in fold_results), "legacy_head_refits": 0,
            "reservoir_parameter_sha256": weight_hash(weights), "mask_version": VERSION, "matched_seed": config["seed"],
            "synthetic_boundary_checks": checks, "native_prefix_check": prefix_checked, "promotion_checks": criteria,
            "promotion_screen_passed": all(criteria.values()), "admitted_for_r5": [], "python_pid": os.getpid(),
            "elapsed_seconds": perf_counter() - started}
