"""Diagnose motion and missed-packet regimes on the original development frames."""

import os

import numpy as np

from common import BATCH, SHICE, errors, fingerprint, names, read_json, record, stamp, write_json
from features import G, fetch_record


PROTOCOL = {
    "id": "native-regimes-v1",
    "angular_speed_history_ms": 100.0,
    "speed_breaks_deg_s": [1.0, 20.0, 60.0, 120.0],
    "gravity_magnitude_tolerance_g_for_stationary": 0.05,
    "interpolated_packet_age_breaks_ms": [50.0, 200.0, 1000.0],
    "reference_for_static_secondary": "native default FullVQF restDetected on the identical frame",
    "causal_window": "trailing including current observed frame only",
    "interpolated_packet_flags": "sample_index and is_interpolated columns from original SHICE aligned CSV",
    "partitions": ["R1", "R2", "R3", "R4"],
    "common_scored_frames_preserved": True,
    "gradient_partition_unchanged": True,
    "reconstruction_rmse_tolerance_deg": 0.02,
}

LABELS = {
    "speed": ["under_1", "1_to_20", "20_to_60", "60_to_120", "at_least_120"],
    "static": ["non_stationary", "stationary_gyro_and_gravity"],
    "native_rest": ["not_rest", "rest"],
    "interpolated_packet_age": ["under_50ms", "50_to_200ms", "200_to_1000ms", "at_least_1000ms_or_none"],
}


def causal_mean(values, count):
    limits = np.arange(1, len(values) + 1)
    before = np.maximum(0, limits - count)
    prefix = np.r_[0., np.cumsum(values, dtype=np.float64)]
    return (prefix[limits] - prefix[before]) / (limits - before)


def cached_strata(name):
    folder = BATCH / "regime-strata" / PROTOCOL["id"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{name}.npz"
    if target.is_file():
        with np.load(target, allow_pickle=False) as stored:
            return {key: stored[key] for key in stored.files}
    native = record(name)
    vqf = fetch_record(name, {})
    frames = np.flatnonzero(native["score"])
    np.testing.assert_array_equal(vqf["index"], frames)
    raw_file = SHICE / f"{name.split('-')[0]}_aligned_formal" / f"{name}.csv"
    with raw_file.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        timeline = np.loadtxt(stream, delimiter=",", quotechar='"', usecols=(header.index("sample_index"),
                              header.index("is_interpolated")), ndmin=2)
    native_index = np.arange(len(native["imu"]))
    np.testing.assert_array_equal(timeline[:, 0], native_index)
    gaps = timeline[:, 1].astype(bool)
    if gaps[frames].any():
        raise ValueError(f"An interpolated packet is included in original score mask {name}")
    previous_gap = np.maximum.accumulate(np.where(gaps, native_index, -1))
    dt = float(native["dt"])
    age = np.where(previous_gap[frames] >= 0, (frames - previous_gap[frames]) * dt * 1000., np.inf)
    width = round(PROTOCOL["angular_speed_history_ms"] / (1000. * dt))
    speed = causal_mean(np.rad2deg(np.linalg.norm(native["imu"][:, 3:], axis=1)), width)[frames]
    magnitude_error = causal_mean(np.abs(np.linalg.norm(native["imu"][:, :3], axis=1) / G - 1.), width)[frames]
    categories = {"index": frames, "speed": np.searchsorted(PROTOCOL["speed_breaks_deg_s"], speed, side="right").astype(np.uint8),
                  "static": ((speed < 1.) & (magnitude_error < PROTOCOL["gravity_magnitude_tolerance_g_for_stationary"])).astype(np.uint8),
                  "native_rest": vqf["rest_scored"].astype(np.uint8),
                  "interpolated_packet_age": np.searchsorted(PROTOCOL["interpolated_packet_age_breaks_ms"], age, side="right").astype(np.uint8),
                  "base_squared": np.square(errors(vqf["base_up_scored"], vqf["ref_body_scored"]), dtype=np.float64),
                  "native_interpolation_flags": np.asarray(int(gaps.sum()))}
    temporary = folder / f"{name}.{os.getpid()}.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **categories)
    os.replace(temporary, target)
    return categories


def predict_frames(data, config, state, device, fitted):
    import learning

    kind = config["kind"]
    if config.get("installation"):
        data = learning.apply_install(data, state["installation_quaternion_wxyz"])
    if kind == "filter":
        return data["base_up_scored"], data["ref_body_scored"]
    if kind == "install_rotation":
        from install_rotation import frame_directions

        return frame_directions(data, state, config)
    if kind == "ridge":
        prepared = learning.prepare(data, config)
        return learning.frame_directions(prepared, state, config)
    if kind == "legacy_head":
        import legacy_head

        return legacy_head.frame_directions(legacy_head.prepare(data, config), config, fitted, device)
    if kind == "tcn":
        import temporal

        return temporal.frame_directions(temporal.prepare(data, config), config, fitted, device)
    raise ValueError(f"Cannot profile {kind}")


def summarize_slices(squared, strata):
    result = {}
    for axis, labels in LABELS.items():
        group = strata[axis]
        count = np.bincount(group, minlength=len(labels))
        current = np.bincount(group, weights=squared, minlength=len(labels))
        baseline = np.bincount(group, weights=strata["base_squared"], minlength=len(labels))
        if count.sum() != len(squared):
            raise ValueError(f"Some score frames are missing from {axis}")
        scores = {}
        for key, n, x, old in zip(labels, count, current, baseline, strict=True):
            scores[key] = {"frames": int(n), "squared_error": float(x), "baseline_squared_error": float(old),
                           "rmse_deg": float(np.sqrt(x/n)) if n else None,
                           "baseline_rmse_deg": float(np.sqrt(old/n)) if n else None}
        result[axis] = scores
    return result


def run(job, destination):
    import torch

    import learning
    from final_eval import cached_candidate, fit_candidate

    parent = job["config"]["parent_round"]
    parent_job = read_json(BATCH / "rounds" / parent / "round.json")
    config = parent_job["config"]
    if fingerprint(config) != job["config"]["parent_config_sha256"]:
        raise ValueError("The development candidate changed before its regime audit")
    historical = read_json(BATCH / parent_job["result"])["sequences"]
    expected = {axis: {row["name"]: row for row in rows} for axis, rows in historical.items()}
    all_names = names("dev")
    device = torch.device("cuda")
    data = {name: cached_candidate(name, config) for name in all_names}
    categories = {name: cached_strata(name) for name in all_names}
    comparisons = {"device": [], "motion": []}
    for axis, label, held in learning.folds(all_names):
        training = [name for name in all_names if name not in held]
        state = fit_candidate(config, training)
        fitted = None
        if config["kind"] == "legacy_head":
            from legacy_head import make_model
            fitted = make_model(state, device)
        if config["kind"] == "tcn":
            from temporal import make_model
            fitted = make_model(state, config, device)
        for name in held:
            part = data[name]
            strata = categories[name]
            np.testing.assert_array_equal(strata["index"], part["index"])
            pairs = predict_frames(part, config, state, device, fitted)
            squared = np.square(errors(*pairs), dtype=np.float64)
            row = expected[axis][name]
            value = float(np.sqrt(squared.mean()))
            old = row["rmse_deg"]
            if len(squared) != row["frames"] or abs(value-old) > PROTOCOL["reconstruction_rmse_tolerance_deg"]:
                raise ValueError(f"Original score failed reconstruction in {axis}/{name}: {value} vs {old}")
            comparisons[axis].append({"name": name, "fold": label, "frames": len(squared),
                                      "rmse_deg": value, "original_rmse_deg": old,
                                      "reconstruction_delta_deg": value-old,
                                      "native_interpolation_flags": int(strata["native_interpolation_flags"]),
                                      "slices": summarize_slices(squared, strata)})
    for axis, rows in comparisons.items():
        if sorted(row["name"] for row in rows) != sorted(all_names):
            raise ValueError(f"Missing original record in {axis}")
    write_json(destination, {"round_id": job["round_id"], "parent_round": parent, "parent_config_sha256": fingerprint(config),
                             "created_at": stamp(), "augmentation_or_masks_changed": False,
                             "protocol": PROTOCOL, "comparisons": comparisons})
    print(parent, "maximum absolute RMSE reconstruction difference",
          max(abs(row["reconstruction_delta_deg"]) for rows in comparisons.values() for row in rows), flush=True)


def queue():
    candidates = ("R00021", "I00004", "N00035", "P00022", "P00004", "R00193")
    plan = []
    for number, parent in enumerate(candidates, 1):
        identity = f"Q{number:05d}"
        reference = read_json(BATCH / "rounds" / parent / "round.json")
        if reference["status"] != "completed":
            raise ValueError(f"Development parent {parent} is incomplete")
        config = {"kind": "regime_audit", "parent_round": parent,
                  "parent_config_sha256": reference["config_sha256"], "protocol_sha256": fingerprint(PROTOCOL),
                  "seed": reference["seed"]}
        destination = BATCH / "rounds" / identity / "round.json"
        if not destination.is_file():
            write_json(destination, {"round_id": identity, "config": config, "config_sha256": fingerprint(config), "seed": reference["seed"],
                        "batch": "motion-and-packet-regime-audit", "relation": parent,
                        "hypothesis": "Reconstruct original grouped outcomes and localize errors by motion and original interpolation age",
                        "partitions": {"development": names("dev")},
                        "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                        "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"})
        plan.append({"audit_round": identity, "parent_round": parent})
    write_json(BATCH / "regime-development-stage.json", {"planned_at": stamp(), "protocol": PROTOCOL, "rounds": plan})
    print(plan, flush=True)


if __name__ == "__main__":
    queue()
