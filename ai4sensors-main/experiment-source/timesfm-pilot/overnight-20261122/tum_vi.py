"""Fixed-configuration, dense-mocap TUM-VI verification after the primary test."""

import csv

import numpy as np

from common import BATCH, DATA, Z, aggregate, body_up, now, read_json, rmse, stamp, unit, vqf_outputs, write_json


def stream_csv(path, first, last):
    timestamps, values = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if row:
                timestamps.append(int(row[0]))
                values.append(tuple(float(token) for token in row[first:last]))
    times = np.asarray(timestamps, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError(f"Timestamps must strictly increase in {path}")
    if data.shape != (len(times), last - first):
        raise ValueError(f"Unexpected field width in {path}")
    return times, data


def dense_benchmark_frames(imu_ns, imu, mocap_ns, pose, protocol):
    right = np.searchsorted(mocap_ns, imu_ns, side="right")
    inside = (right > 0) & (right < len(mocap_ns))
    right_safe = right.clip(1, len(mocap_ns) - 1)
    left = right_safe - 1
    reference_ok = np.isfinite(pose).all(axis=1) & (np.linalg.norm(pose, axis=1) > 0)
    duration = mocap_ns[right_safe] - mocap_ns[left]
    eligible = (inside & reference_ok[left] & reference_ok[right_safe]
                & (duration <= protocol["max_consecutive_mocap_gap_ns"]) & np.isfinite(imu).all(axis=1)
                & (imu_ns - imu_ns[0] >= protocol["initial_exclusion_ns"]))
    if not eligible.any():
        raise ValueError("No dense valid mocap brackets after the fixed warmup mask")
    left_ok, right_ok = left[eligible], right_safe[eligible]
    a, b = unit(pose[left_ok]), unit(pose[right_ok])
    fraction = ((imu_ns[eligible] - mocap_ns[left_ok]) / duration[eligible])[:, None]
    dot = (a * b).sum(axis=1, keepdims=True)
    b = np.where(dot < 0., -b, b)
    theta = np.arccos(np.abs(dot).clip(0., 1.))
    denominator = np.sin(theta)
    slerp = (np.sin((1 - fraction) * theta) * a + np.sin(fraction * theta) * b) / denominator.clip(min=1e-12)
    interpolated = unit(np.where(theta < 1e-6, (1 - fraction) * a + fraction * b, slerp))
    ref = np.zeros((len(imu_ns), 3), dtype=np.float32)
    ref[eligible] = body_up(interpolated).astype(np.float32)
    return eligible, ref


def room_data(root, room, protocol):
    prefix = root / f"dataset-room{room}_512_16"
    matches = list(prefix.rglob("mav0/imu0/data.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected one native IMU file for room {room}, found {len(matches)}")
    imu_file = matches[0]
    mocap_file = imu_file.parents[1] / "mocap0" / "data.csv"
    imu_ns, readings = stream_csv(imu_file, 1, 7)
    mocap_ns, poses = stream_csv(mocap_file, 4, 8)
    step = np.diff(imu_ns)
    median = float(np.median(step))
    if np.max(np.abs(step - median)) > median * protocol["max_relative_imu_step_jitter"]:
        raise ValueError(f"room{room}: irregular native IMU clock exceeds predeclared tolerance")
    imu = readings[:, (3, 4, 5, 0, 1, 2)]
    good = np.isfinite(imu).all(axis=1)
    mask, truth = dense_benchmark_frames(imu_ns, imu, mocap_ns, poses, protocol)
    from tilt_data import causal_fill

    clean = causal_fill(imu, good)
    stats = {"name": f"room{room}", "frames": int(mask.sum()), "imu_frames": len(imu_ns), "mocap_frames": len(mocap_ns),
             "imu_dt_ns": median, "mocap_gaps_above_threshold": int(np.sum(np.diff(mocap_ns) > protocol["max_consecutive_mocap_gap_ns"])),
             "imu_path": str(imu_file), "mocap_path": str(mocap_file)}
    return clean, median * 1e-9, mask, good, truth, stats


def run(root, cfg, state, lock_hash, backbone, device, cutoff):
    from final_eval import cache_external_candidate, embed_external, score_candidate

    protocol = read_json(BATCH / "tum-vi-protocol.json")
    expected = [f"room{room}" for room in protocol["rooms"]]
    rows, failures = [], []
    statuses = {}
    destination = BATCH / "tum-vi-independent.json"

    def persist(*, final):
        if final and set(statuses) != set(expected):
            raise ValueError("Final TUM-VI coverage did not account for every planned room")
        manifest = [{"name":name,"status":statuses.get(name,"pending")} for name in expected]
        count = {status:sum(row["status"] == status for row in manifest)
                 for status in ("success","failed","not_started_before_cutoff","pending")}
        snapshot = {"at":stamp(),"selection_lock_sha256":lock_hash,"protocol":protocol,
                    "phase":"final" if final else "progress","room_manifest":manifest,
                    "coverage":{"planned":len(expected),**count},"results":rows,"failures":failures,
                    "locked":aggregate(rows) if rows else None,
                    "pure_full_vqf":aggregate([{**row,"rmse_deg":row["baseline_deg"]} for row in rows]) if rows else None}
        write_json(destination,snapshot)
        return snapshot

    for room, name in zip(protocol["rooms"], expected, strict=True):
        if now() >= cutoff:
            statuses[name] = "not_started_before_cutoff"
            failures.append({"name":name,"status":statuses[name]})
            continue
        try:
            imu, dt, mask, good, truth, row = room_data(root, room, protocol)
            pure = vqf_outputs(imu, dt, {})["quat6D"]
            baseline = rmse(body_up(pure[mask], Z), truth[mask])
            cached = cache_external_candidate(imu, dt, mask, good, truth, cfg)
            latent = embed_external(cached, cfg, backbone, device)
            row.update(baseline_deg=baseline, rmse_deg=score_candidate(cached, cfg, state, latent))
            rows.append(row)
            statuses[name] = "success"
            print("TUM-VI", row["name"], row["rmse_deg"], row["baseline_deg"], flush=True)
        except Exception as error:
            statuses[name] = "failed"
            failures.append({"name":name,"status":statuses[name],"error":repr(error)})
            print("TUM-VI", room, repr(error), flush=True)
        persist(final=False)
    return persist(final=True)
