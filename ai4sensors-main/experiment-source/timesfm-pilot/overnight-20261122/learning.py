"""Train and score lightweight world-frame tilt corrections on independent folds."""

import os

import numpy as np

from common import BATCH, HISTORY, Z, aggregate, body_up, by_device, names, rmse, rotate, unit
from experiment import baseline_reference
from features import VERSION, fetch_record, physics, training_support


MOTION_FOLDS = ((1, 5, 9), (2, 6, 10), (3, 7), (4, 8))


def folds(records):
    for device in ("R1", "R2", "R3", "R4"):
        yield "device", device, [n for n in records if n.startswith(device + "-")]
    for number, programs in enumerate(MOTION_FOLDS, 1):
        yield "motion", f"M{number}", [n for n in records if int(n.split("-")[1]) in programs]


def projection(embedding_width, count, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((embedding_width, count), dtype=np.float32) / np.sqrt(embedding_width)


def projected(name, count, seed):
    if not count:
        return None
    folder = BATCH / "projected" / f"{count}-{seed}"
    folder.mkdir(exist_ok=True, parents=True)
    target = folder / f"{name}.npy"
    if target.is_file():
        return np.load(target, allow_pickle=False)
    embedding = np.load(HISTORY / "features" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    weights = projection(embedding.shape[-1], count, seed)
    result = (embedding.astype(np.float32) @ weights).reshape(len(embedding), -1)
    staging = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    with staging.open("wb") as stream:
        np.save(stream, result, allow_pickle=False)
    os.replace(staging, target)
    return result


def prepare(data, config, embedding=None):
    at, matrix, acc, gyr = physics(data, config)
    if embedding is not None:
        if config["update_hz"] != 10 or len(matrix) != len(embedding):
            raise ValueError("TimesFM embedding does not align with the selected update grid")
        matrix = np.concatenate((matrix, embedding), axis=1)
    ups = data["up"][at]
    targets = data["target"][at] - ups
    targets -= (targets * ups).sum(axis=1, keepdims=True) * ups
    training, audit = training_support(data, at, config["context"])
    return {"matrix": matrix, "target": targets, "train": training, "training_context": audit,
            "ready": data["ready"][at], "q": data["q"][at], "up": ups,
            "acc": acc, "gyr": gyr, "rest": data["rest"][at], "index": data["index"],
            "ref_world": data["ref_world"], "dt": float(data["dt"])}


def for_record(name, config):
    embedding = projected(name, config.get("embedding", 0), config["seed"])
    return prepare(fetch_record(name, config["parameters"]), config, embedding)


def multiply_quaternion(a, b):
    real = a[..., :1] * b[:1] - (a[..., 1:] * b[1:]).sum(axis=-1, keepdims=True)
    imag = a[..., :1] * b[1:] + b[:1] * a[..., 1:] + np.cross(a[..., 1:], b[1:])
    return np.concatenate((real, imag), axis=-1)


def apply_install(data, q_mount, include_reference=True):
    q = multiply_quaternion(data["q"], q_mount)
    adjusted = {**data, "q": q, "up": body_up(q).astype(np.float32)}
    if include_reference:
        full = multiply_quaternion(data["q_scored"], q_mount)
        adjusted["ref_world"] = rotate(unit(full), data["ref_body_scored"]).astype(np.float32)
    return adjusted


def attach_native(name, config):
    from features import fetch_scored_quaternion

    data = fetch_record(name, config["parameters"])
    return {**data, "q_scored": fetch_scored_quaternion(name, config["parameters"])}


def installation_fit(training, config):
    from scipy.spatial.transform import Rotation
    from install_rotation import fit_statistics, sufficient_statistics

    mounting = {"coordinates": "body", "fit_motion": config["installation"], "shrinkage": config["mount_shrink"]}
    state = fit_statistics([sufficient_statistics(data, mounting) for data in training], mounting)
    return Rotation.from_matrix(state["fitted"]).as_quat(scalar_first=True).astype(np.float32)


def design(matrix, state):
    x = ((matrix - state["mean"]) / state["std"]).clip(-12, 12)
    parts = [np.ones((len(x), 1), dtype=np.float32), x]
    if "random" in state:
        parts.append(np.tanh(x @ state["random"] + state["shift"]))
    return np.concatenate(parts, axis=1)


def fit(records, config):
    count = len(records)
    raw = []
    for data in records:
        mask = data["train"]
        if not mask.any():
            continue
        raw.append((data["matrix"][mask], data["target"][mask]))
    if len(raw) != count:
        raise ValueError("One or more training records have no auditable update labels")
    means = np.stack([x.mean(axis=0, dtype=np.float64) for x, _ in raw]).mean(axis=0)
    squares = np.stack([np.square(x, dtype=np.float64).mean(axis=0) for x, _ in raw]).mean(axis=0)
    state = {"mean": means.astype(np.float32), "std": np.sqrt(np.maximum(squares - means**2, 9e-6)).astype(np.float32)}
    hidden = config.get("hidden", 0)
    if hidden:
        rng = np.random.default_rng(config["seed"])
        state["random"] = rng.standard_normal((len(means), hidden), dtype=np.float32) / np.sqrt(len(means))
        state["shift"] = rng.uniform(-0.8, 0.8, hidden).astype(np.float32)
    dim = len(means) + hidden + 1
    ata = np.zeros((dim, dim), dtype=np.float64)
    aty = np.zeros((dim, 3), dtype=np.float64)
    for x, y in raw:
        transformed = design(x, state).astype(np.float64)
        scale = 1. / (count * len(transformed))
        ata += (transformed.T @ transformed) * scale
        aty += (transformed.T @ y) * scale
    diagonal = np.full(dim, config["alpha"])
    diagonal[0] = 1e-9
    state["coef"] = np.linalg.solve(ata + np.diag(diagonal), aty).astype(np.float32)
    return state


def delta_at_updates(data, state, config):
    delta = design(data["matrix"], state) @ state["coef"]
    return constrain_delta(data, delta, config)


def constrain_delta(data, delta, config):
    delta = delta.copy()
    up = data["up"]
    delta -= (delta * up).sum(axis=1, keepdims=True) * up
    angle_cap = np.tan(np.deg2rad(config["cap_deg"]))
    delta *= (angle_cap / np.linalg.norm(delta, axis=1, keepdims=True).clip(min=1e-9)).clip(max=1.)
    gate = np.full(len(delta), config["gain"], dtype=np.float32)
    if config.get("acc_gate"):
        gate *= np.exp(-0.5 * (data["acc"] / config["acc_gate"])**2)
    if config.get("gyro_gate"):
        gate *= np.exp(-0.5 * (data["gyr"] / config["gyro_gate"])**2)
    delta *= gate[:, None]
    return delta


def frame_directions(data, state, config, delay_ms=0.):
    delta = delta_at_updates(data, state, config)
    up = unit(rotate(unit(data["q"]), delta) + Z)
    delay = int(np.ceil(delay_ms / (1000 * data["dt"])))
    active = np.searchsorted(data["ready"] + delay, data["index"], side="right")
    estimates = np.concatenate((Z[None, :], up), axis=0)[active]
    return estimates, data["ref_world"]


def evaluate(data, state, config, delay_ms=0.):
    return rmse(*frame_directions(data, state, config, delay_ms))


def run_trial(item, path):
    config = item["config"]
    sequence_names = names("dev")
    expected = baseline_reference()
    compound = config.get("installation")
    if compound:
        raw = {name: attach_native(name, config) for name in sequence_names}
        embeddings = {name: projected(name, config.get("embedding", 0), config["seed"]) for name in sequence_names}
        audits = {}
    else:
        data = {name: for_record(name, config) for name in sequence_names}
        audits = {name: part["training_context"] for name, part in data.items()}
    by_axis = {"device": [], "motion": []}
    fold_index = []
    for axis, label, validation in folds(sequence_names):
        training = [n for n in sequence_names if n not in validation]
        if compound:
            q = installation_fit([raw[n] for n in training], config)
            data = {n: prepare(apply_install(raw[n], q, include_reference=n in validation), config, embeddings[n]) for n in sequence_names}
            audits.update({n: part["training_context"] for n, part in data.items()})
        state = fit([data[n] for n in training], config)
        fold_rows = []
        for n in validation:
            row = {"name": n, "fold": label, "frames": len(data[n]["index"]),
                   "baseline_deg": expected[n], "rmse_deg": evaluate(data[n], state, config)}
            by_axis[axis].append(row)
            fold_rows.append(row)
        fold_index.append({"axis": axis, "fold": label, "training": training, "validation": validation,
                           "installation_quaternion_wxyz": q.tolist() if compound else None,
                           "validation_summary": aggregate(fold_rows)})
    metrics = {axis: {**aggregate(rows), "per_device": by_device(rows)} for axis, rows in by_axis.items()}
    metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"] + metrics["motion"]["macro_rmse_deg"]) / 2
    return {"dev_metrics": metrics, "sequences": by_axis, "folds": fold_index, "mask_version": VERSION,
            "training_context_audit": audits}
