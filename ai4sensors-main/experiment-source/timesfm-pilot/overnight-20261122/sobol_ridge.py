"""One fixed Sobol tanh basis for two frozen mounted ridge penalties."""

import os
from hashlib import sha256
from time import perf_counter

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

import learning
from common import BATCH, aggregate, by_device, digest, names, read_json, stamp
from experiment import baseline_reference
from features import VERSION
from stream_gru import planned_folds


PROTOCOL = BATCH / "mounted-sobol-ridge-protocol.json"
WEIGHTS = BATCH / "mounted-sobol-fixed-tanh.npz"
FEATURE_COUNT = 128


def construct(width):
    samples = qmc.Sobol(d=width + 1, scramble=False).random_base2(m=7)
    strata = np.floor(samples * FEATURE_COUNT).astype(np.int64)
    if not np.all(np.sort(strata, axis=0) == np.arange(FEATURE_COUNT)[:, None]):
        raise ValueError("Each Sobol dimension must visit each of the 128 intervals once")
    points = (strata + .5) / FEATURE_COUNT
    projection = ndtri(points[:, :-1]).T.astype(np.float32) / np.sqrt(width)
    bias = (-.8 + 1.6 * points[:, -1]).astype(np.float32)
    return {"random": projection, "shift": bias, "midpoint_strata": strata}


def normal_equations(records, projection, shift):
    pairs = [(data["matrix"][data["train"]], data["target"][data["train"]]) for data in records]
    if not all(len(x) for x, _ in pairs):
        raise ValueError("One or more training records has no auditable update labels")
    means = np.stack([x.mean(axis=0, dtype=np.float64) for x, _ in pairs]).mean(axis=0)
    squares = np.stack([np.square(x, dtype=np.float64).mean(axis=0) for x, _ in pairs]).mean(axis=0)
    state = {"mean": means.astype(np.float32), "std": np.sqrt(np.maximum(squares - means**2, 9e-6)).astype(np.float32),
             "random": projection, "shift": shift}
    size = len(means) + len(shift) + 1
    gram = np.zeros((size, size), dtype=np.float64)
    target = np.zeros((size, 3), dtype=np.float64)
    for x, y in pairs:
        features = learning.design(x, state).astype(np.float64)
        weight = 1. / (len(pairs) * len(features))
        gram += (features.T @ features) * weight
        target += (features.T @ y) * weight
    return state, gram, target


def solve(state, gram, target, alpha):
    diagonal = np.full(len(gram), alpha, dtype=np.float64)
    diagonal[0] = 1e-9
    return {**state, "coef": np.linalg.solve(gram + np.diag(diagonal), target).astype(np.float32)}


def verify_existing_solver(width):
    rng = np.random.default_rng(8721)
    training = [{"matrix": rng.standard_normal((33, width), dtype=np.float32),
                 "target": rng.standard_normal((33, 3), dtype=np.float32),
                 "train": np.arange(33) % 3 != 0} for _ in range(3)]
    cfg = {"alpha": .0018789, "hidden": FEATURE_COUNT, "seed": 20261122}
    original = learning.fit(training, cfg)
    rebuilt = solve(*normal_equations(training, original["random"], original["shift"]), cfg["alpha"])
    for name in original:
        np.testing.assert_array_equal(original[name], rebuilt[name])
    return {"same_seeded_projection_gives_exact_existing_readout": True, "synthetic_record_count": len(training)}


def run(item):
    started = perf_counter()
    stage = read_json(PROTOCOL)
    if item["config"]["protocol_sha256"] != digest(PROTOCOL) or item["round_id"] != stage["round_id"]:
        raise ValueError("The Sobol experiment differs from its preregistered setup")
    if digest(WEIGHTS) != stage["operator_sha256"]:
        raise ValueError("Sobol projections changed after registration")
    for source, sha in stage["historical_scoring_code_sha256"].items():
        if digest(BATCH / source) != sha:
            raise ValueError(f"Historical scoring implementation changed: {source}")
    with np.load(WEIGHTS, allow_pickle=False) as loaded:
        weights = {key: loaded[key] for key in loaded.files}
    expected = construct(stage["input_dimensions"])
    for key, arr in expected.items():
        np.testing.assert_array_equal(weights[key], arr)
    checks = verify_existing_solver(stage["input_dimensions"])
    print("Sobol ridge", item["round_id"], "python worker", os.getpid(), stamp(), flush=True)
    config = stage["shared_config"]
    records = names("dev")
    raw = {name: learning.attach_native(name, config) for name in records}
    baseline = baseline_reference()
    alphas = stage["alphas"]
    models = {str(alpha): {**config, "alpha": alpha} for alpha in alphas}
    sequences, evidence, audits = {}, [], {}
    for axis, fold, training, held in planned_folds(records):
        before = perf_counter()
        q = learning.installation_fit([raw[name] for name in training], config)
        designs = {name: learning.prepare(learning.apply_install(raw[name], q, include_reference=name in held), config)
                   for name in training + held}
        audits.update({name: part["training_context"] for name, part in designs.items()})
        shared_at = perf_counter()
        equations = normal_equations([designs[name] for name in training], weights["random"], weights["shift"])
        solve_at = perf_counter()
        rows = {name: {"name": name, "fold": fold, "frames": len(designs[name]["index"]), "full_vqf_deg": baseline[name],
                       "score_frame_indices_sha256": sha256(np.asarray(designs[name]["index"], dtype="<i8").tobytes()).hexdigest(),
                       "rmse_deg": {}} for name in held}
        fits = {}
        for key, actual in models.items():
            arm_started = perf_counter()
            state = solve(*equations, actual["alpha"])
            fit_done = perf_counter()
            for name in held:
                rows[name]["rmse_deg"][key] = learning.evaluate(designs[name], state, actual)
            fits[key] = {"fits": 1, "solve_seconds": fit_done - arm_started,
                         "score_seconds": perf_counter() - fit_done}
        sequences.setdefault(axis, []).extend(rows.values())
        evidence.append({"axis": axis, "fold": fold, "training": training, "validation": held,
                         "installation_quaternion_wxyz": q.tolist(), "fit_by_alpha": fits,
                         "prepare_seconds": shared_at - before, "shared_normal_equation_seconds": solve_at - shared_at})
        print(axis, fold, {key: float(np.mean([row["rmse_deg"][key] for row in rows.values()])) for key in models}, flush=True)
    axis_scores = {}
    for axis, rows in sequences.items():
        if sorted(row["name"] for row in rows) != sorted(records) or sum(row["frames"] for row in rows) != 8328711:
            raise ValueError(f"Original score coverage changed on {axis}")
        axis_scores[axis] = {}
        for key in models:
            values = [{"name": row["name"], "frames": row["frames"], "baseline_deg": row["full_vqf_deg"],
                       "rmse_deg": row["rmse_deg"][key]} for row in rows]
            axis_scores[axis][key] = {**aggregate(values), "per_device": by_device(values)}
    return {"dev_metrics": None, "axis_metrics": axis_scores, "folds": evidence, "sequences": sequences,
            "input_dimensions": stage["input_dimensions"], "training_context_audit": audits, "mask_version": VERSION,
            "implementation_parity": checks, "operator_sha256": stage["operator_sha256"],
            "unique_deterministic_configurations": len(models), "actual_ridge_fits": len(evidence) * len(models),
            "actual_installation_fits": len(evidence), "failures": 0, "python_pid": os.getpid(),
            "eligible_for_r5": False, "seed_repeats": 0, "elapsed_seconds": perf_counter() - started}
