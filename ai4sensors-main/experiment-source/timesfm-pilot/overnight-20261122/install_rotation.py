"""A fixed proper rotation fitted on training records only, with no identifiers at inference."""

import numpy as np
from scipy.spatial.transform import Rotation

from common import Z, aggregate, by_device, names, rmse, unit
from experiment import baseline_reference
from features import fetch_record
from learning import folds


def sufficient_statistics(data, config):
    use = data["rest_scored"].astype(bool) if config["fit_motion"] == "rest" else np.ones(len(data["index"]), dtype=bool)
    frames = int(np.count_nonzero(use))
    if frames < 30:
        return None
    if config["coordinates"] == "body":
        pred = data["base_up_scored"][use].astype(np.float64)
        target = data["ref_body_scored"][use].astype(np.float64)
        stats = (pred.T @ target) / frames
    else:
        stats = data["ref_world"][use].mean(axis=0, dtype=np.float64)
    return stats, frames


def fit_statistics(stats, config):
    choices = [entry for entry in stats if entry is not None]
    if not choices:
        raise ValueError("No eligible training poses for the fixed rotation")
    mean = np.mean([entry[0] for entry in choices], axis=0)
    if config["coordinates"] == "body":
        left, _, right = np.linalg.svd(mean)
        proper = np.eye(3)
        proper[2, 2] = np.sign(np.linalg.det(left @ right))
        matrix = left @ proper @ right
        rv = Rotation.from_matrix(matrix.T).as_rotvec() * config["shrinkage"]
        fitted = Rotation.from_rotvec(rv).as_matrix().T
    else:
        target = unit(mean)
        axis = np.cross(Z, target)
        magnitude = np.linalg.norm(axis)
        rv = axis * (np.arctan2(magnitude, target @ Z) / max(magnitude, 1e-12)) * config["shrinkage"]
        fitted = Rotation.from_rotvec(rv).apply(Z)
    return {"fitted": fitted, "rotation_deg": float(np.rad2deg(np.linalg.norm(rv))),
            "training_records_used": len(choices), "training_frames_used": int(sum(entry[1] for entry in choices))}


def frame_directions(data, state, config):
    if config["coordinates"] == "body":
        return data["base_up_scored"] @ state["fitted"], data["ref_body_scored"]
    return np.broadcast_to(state["fitted"], data["ref_world"].shape), data["ref_world"]


def evaluate(data, state, config):
    return rmse(*frame_directions(data, state, config))


def run(item):
    config = item["config"]
    sequences = names("dev")
    baseline = baseline_reference()
    data = {n: fetch_record(n, config["parameters"]) for n in sequences}
    stats = {n: sufficient_statistics(source, config) for n, source in data.items()}
    outputs = {"device": [], "motion": []}
    evidence = []
    for axis, label, held in folds(sequences):
        training = [n for n in sequences if n not in held]
        fit = fit_statistics([stats[n] for n in training], config)
        rows = [{"name": n, "fold": label, "frames": len(data[n]["index"]), "baseline_deg": baseline[n],
                 "rmse_deg": evaluate(data[n], fit, config)} for n in held]
        outputs[axis].extend(rows)
        evidence.append({"axis": axis, "fold": label, "training": training, "validation": held,
                         "rotation": {**fit, "fitted": fit["fitted"].tolist()},
                         "validation_summary": aggregate(rows)})
    metrics = {axis: {**aggregate(rows), "per_device": by_device(rows)} for axis, rows in outputs.items()}
    metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"] + metrics["motion"]["macro_rmse_deg"]) / 2
    return {"dev_metrics": metrics, "sequences": outputs, "folds": evidence,
            "inference_inputs": ["native causal VQF attitude"], "fit_uses_identifiers": False}
