"""Causal sensor features and a reusable record cache for each filter configuration."""

import os

import numpy as np

from common import BATCH, Z, body_up, fingerprint, record, rotate, unit, vqf_outputs


G = 9.80665
SCALE = np.array([G, G, G, 1., 1., 1.], dtype=np.float32)
VERSION = "causal-bin-v2-full-window"


def bins(imu, dt, nominal_shice):
    ratio = 10. if nominal_shice else 0.01 / dt
    if ratio < 1 - 1e-8:
        raise ValueError("The benchmark assumes at least 100 samples/s")
    num = int(np.floor(len(imu) / ratio + 1e-9))
    edges = np.arange(num + 1) * ratio
    stops = np.floor(edges + 1e-9).astype(np.int64)
    prefix = np.vstack((np.zeros((1, 6)), np.cumsum(imu, axis=0, dtype=np.float64)))
    samples = (prefix[stops[1:]] - prefix[stops[:-1]]) / np.diff(stops)[:, None]
    actual = stops[1:] - 1
    ready = np.ceil(edges[1:] - 1e-9).astype(np.int64) - 1
    return samples.astype(np.float32), actual, ready


def cache_record(imu, dt, score, context, truth, parameters, nominal_shice, keep_scored_quaternion=False):
    result = vqf_outputs(imu, dt, parameters)
    windows, actual, ready = bins(imu, dt, nominal_shice)
    base = result["quat6D"]
    q = base[actual].astype(np.float32)
    at_score = np.flatnonzero(score)
    world_ref = rotate(unit(base[at_score]), truth[at_score]).astype(np.float32)
    context_missing = np.r_[0, np.cumsum(~context, dtype=np.int64)]
    block_ends = np.r_[0, actual + 1]
    bad_to_block = context_missing[block_ends]
    data = {"windows": windows, "ready": ready, "q": q,
            "up": body_up(base[actual]).astype(np.float32), "bias": result["bias"][actual].astype(np.float32),
            "rest": result["restDetected"][actual].astype(np.float32), "target": truth[actual].astype(np.float32),
            "label_ok": score[actual], "block_ends": block_ends, "bad_to_block": bad_to_block,
            "index": at_score, "ref_world": world_ref, "dt": np.asarray(dt),
            "base_up_scored": body_up(base[at_score]).astype(np.float32),
            "ref_body_scored": truth[at_score].astype(np.float32), "rest_scored": result["restDetected"][at_score]}
    if keep_scored_quaternion:
        data["q_scored"] = base[at_score].astype(np.float32)
    return data


def fetch_record(name, parameters):
    family = fingerprint({"version": VERSION, "parameters": parameters})[:16]
    root = BATCH / "cache" / family
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.npz"
    if not path.is_file():
        item = record(name)
        filtered = cache_record(item["imu"], float(item["dt"]), item["score"], item["context"], item["ref"], parameters, True)
        staged = path.with_name(f"{name}.{os.getpid()}.tmp")
        with staged.open("wb") as stream:
            np.savez(stream, **filtered)
        os.replace(staged, path)
        return filtered
    with np.load(path, allow_pickle=False) as found:
        return {key: found[key] for key in found.files}


def fetch_scored_quaternion(name, parameters):
    key = fingerprint(parameters)[:16]
    folder = BATCH / "scored-quaternion" / key
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.npy"
    if not path.is_file():
        source = record(name)
        raw = vqf_outputs(source["imu"], source["dt"], parameters)["quat6D"]
        value = raw[source["score"]].astype(np.float32)
        staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with staged.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(staged, path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


def moment(values, at, width):
    a = np.maximum(0, at + 1 - width)
    b = at + 1
    accumulated = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0, dtype=np.float64)))
    return ((accumulated[b] - accumulated[a]) / (b - a)[:, None]).astype(np.float32)


def block_indices(count, update_hz):
    if 100 % update_hz:
        raise ValueError("Update frequency must divide the causal 100 Hz input rate")
    phase = 100 // update_hz
    first = ((64 + phase - 1) // phase) * phase - 1
    return np.arange(first, count, phase)


def training_support(cached, update_blocks, context_blocks):
    """Keep the 64-block audit floor, extending it to cover every actual feature input."""
    support = max(64, context_blocks)
    ends = update_blocks + 1
    starts = ends - support
    has_history = starts >= 0
    starts = np.maximum(0, starts)
    bad = cached["bad_to_block"][ends] - cached["bad_to_block"][starts]
    label_ok = cached["label_ok"][update_blocks]
    eligible = label_ok & has_history & (bad == 0)
    audit = {"mask_version": VERSION, "support_blocks_100hz": support,
             "update_labels_available": int(label_ok.sum()), "training_labels_eligible": int(eligible.sum()),
             "excluded_for_context_or_history": int((label_ok & ~eligible).sum()),
             "min_native_input_rows": int(np.min(np.diff(cached["block_ends"]))) * support,
             "max_native_input_rows": int(np.max(np.diff(cached["block_ends"]))) * support}
    return eligible, audit


def physics(cached, config):
    source = cached["windows"] / SCALE
    at = block_indices(len(source), config["update_hz"])
    up = cached["up"][at]
    recent, medium, long = (moment(source, at, width) for width in (4, 16, config["context"]))
    std = np.sqrt(np.maximum(moment(source**2, at, config["context"]) - long**2, 0.))
    agreement = unit(recent[:, :3]) - up
    acc_departure = np.abs(np.linalg.norm(recent[:, :3], axis=1) - 1.)
    gyro_motion = np.linalg.norm(recent[:, 3:], axis=1)
    values = [up, cached["bias"][at], cached["rest"][at, None], recent, medium, long, std, agreement]
    if config["features"] in {"motion", "quadratic"}:
        values += [acc_departure[:, None], gyro_motion[:, None], recent - medium,
                   medium - long, np.cross(up, recent[:, :3])]
    if config["features"] == "quadratic":
        values += [up**2, up * np.roll(up, 1, axis=1), up * gyro_motion[:, None],
                   up * acc_departure[:, None], agreement * gyro_motion[:, None]]
    return at, np.concatenate(values, axis=1).astype(np.float32), acc_departure, gyro_motion
