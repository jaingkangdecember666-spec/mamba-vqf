"""Low-capacity tilt corrections formed from physical vectors and invariant gains."""

import numpy as np

import learning
from common import Z, aggregate, by_device, names, rmse, rotate, unit
from experiment import baseline_reference
from features import SCALE, VERSION, block_indices, fetch_record, moment, training_support


def features(data, config):
    imu = data["windows"] / SCALE
    at = block_indices(len(imu), config["update_hz"])
    up = data["up"][at]
    recent, medium, long = [moment(imu, at, width) for width in (4, 16, config["context"])]
    acc = recent[:, :3]
    gyro = recent[:, 3:]
    acceleration = np.linalg.norm(acc, axis=1)
    angular_speed = np.linalg.norm(gyro, axis=1)
    agreement = (unit(acc) * up).sum(axis=1)
    columns = [acceleration - 1., angular_speed, agreement, np.linalg.norm(acc - medium[:, :3], axis=1),
               np.linalg.norm(medium[:, :3] - long[:, :3], axis=1), np.linalg.norm(gyro - medium[:, 3:], axis=1),
               np.linalg.norm(data["bias"][at], axis=1), data["rest"][at]]
    vectors = [unit(acc), unit(medium[:, :3]), unit(long[:, :3])]
    if config["basis"] == "dynamic":
        vectors += [np.cross(up, gyro), np.cross(up, gyro - medium[:, 3:]), acc - medium[:, :3]]
    basis = np.stack(vectors, axis=2).astype(np.float32)
    basis -= up[:, :, None] * np.einsum("nc,ncm->nm", up, basis)[:, None, :]
    scalars = np.stack(columns, axis=1).astype(np.float32)
    return at, basis, scalars


def prepare(data, config):
    at, basis, scalars = features(data, config)
    train, audit = training_support(data, at, config["context"])
    up = data["up"][at]
    delta = data["target"][at] - up
    delta -= up * (delta * up).sum(axis=1, keepdims=True)
    return {"basis": basis, "scalars": scalars, "q": data["q"][at], "ready": data["ready"][at],
            "index": data["index"], "ref_world": data["ref_world"], "dt": float(data["dt"]),
            "target": delta, "train": train, "training_context": audit}


def transform(values, state):
    x = ((values - state["mean"]) / state["std"]).clip(-6., 6.)
    return np.column_stack((np.ones(len(x), dtype=np.float32), x))


def fit(records, config):
    arrays = [(row["basis"][row["train"]], row["scalars"][row["train"]], row["target"][row["train"]]) for row in records]
    if not all(len(x) for _, x, _ in arrays):
        raise ValueError("No verified training labels in one equivariant training sequence")
    mean = np.mean([x.mean(axis=0, dtype=np.float64) for _, x, _ in arrays], axis=0)
    square = np.mean([np.square(x, dtype=np.float64).mean(axis=0) for _, x, _ in arrays], axis=0)
    state = {"mean": mean.astype(np.float32), "std": np.sqrt(np.maximum(square - mean**2, 9e-6)).astype(np.float32)}
    count, features = arrays[0][0].shape[-1], len(mean) + 1
    n = count * features
    cross = np.zeros((n, n), dtype=np.float64)
    targets = np.zeros(n, dtype=np.float64)
    for basis, x, target in arrays:
        phi = transform(x, state)
        design = (basis[..., None] * phi[:, None, None, :]).reshape(-1, n).astype(np.float64)
        scale = 1. / (len(arrays) * len(x))
        cross += (design.T @ design) * scale
        targets += (design.T @ target.reshape(-1)) * scale
    fitted = np.linalg.solve(cross + np.eye(n) * config["alpha"], targets)
    state["coef"] = fitted.reshape(count, features).astype(np.float32)
    return state


def delta(basis, scalars, state, config):
    amplitudes = transform(scalars, state) @ state["coef"].T
    prediction = np.einsum("ncm,nm->nc", basis, amplitudes)
    limit = np.tan(np.deg2rad(config["cap_deg"]))
    prediction *= (limit / np.linalg.norm(prediction, axis=1, keepdims=True).clip(min=1e-9)).clip(max=1.)
    gain = np.full(len(scalars), config["gain"], dtype=np.float32)
    if config["acc_gate"]:
        gain *= np.exp(-0.5 * (scalars[:, 0] / config["acc_gate"])**2)
    if config["gyro_gate"]:
        gain *= np.exp(-0.5 * (scalars[:, 1] / config["gyro_gate"])**2)
    return prediction * gain[:, None]


def evaluate(data, state, config, delay_ms=0.):
    prepared = prepare(data, config)
    residual = delta(prepared["basis"], prepared["scalars"], state, config)
    up = unit(Z + rotate(unit(prepared["q"]), residual))
    delay = int(np.ceil(delay_ms / (1000 * prepared["dt"])))
    active = np.searchsorted(prepared["ready"] + delay, prepared["index"], side="right")
    return rmse(np.vstack((Z[None, :], up))[active], prepared["ref_world"])


def run(item):
    config = item["config"]
    sequences = names("dev")
    expected = baseline_reference()
    raw = {name: learning.attach_native(name, config) if config.get("installation") else fetch_record(name, config["parameters"]) for name in sequences}
    outputs = {"device": [], "motion": []}
    evidence = []
    context = {}
    for axis, label, held in learning.folds(sequences):
        training = [name for name in sequences if name not in held]
        q = learning.installation_fit([raw[n] for n in training], config) if config.get("installation") else None
        data = {n: learning.apply_install(source, q, n in held) if q is not None else source for n, source in raw.items()}
        train = [prepare(data[n], config) for n in training]
        context.update({n: a["training_context"] for n, a in zip(training, train, strict=True)})
        state = fit(train, config)
        rows = [{"name": n, "fold": label, "frames": len(raw[n]["index"]), "baseline_deg": expected[n],
                 "rmse_deg": evaluate(data[n], state, config)} for n in held]
        outputs[axis].extend(rows)
        evidence.append({"axis": axis, "fold": label, "training": training, "validation": held,
                         "installation_quaternion_wxyz": q.tolist() if q is not None else None,
                         "validation_summary": aggregate(rows)})
    results = {axis: {**aggregate(rows), "per_device": by_device(rows)} for axis, rows in outputs.items()}
    results["selection_rmse_deg"] = (results["device"]["macro_rmse_deg"] + results["motion"]["macro_rmse_deg"]) / 2.
    return {"dev_metrics": results, "sequences": outputs, "folds": evidence, "mask_version": VERSION,
            "training_context_audit": context, "inference_inputs": ["causal six-axis IMU", "causal FullVQF state"]}
