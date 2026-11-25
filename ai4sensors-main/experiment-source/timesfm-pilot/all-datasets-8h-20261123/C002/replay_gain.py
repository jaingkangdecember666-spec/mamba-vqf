"""Evaluate one globally damped, frozen B6 correction on the authorized development records."""

from datetime import datetime
from pathlib import Path
import json
import os
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C001 = CAMPAIGN / "C001"
sys.path.insert(0, str(C001))

import replay as c1
import numpy as np
import torch
from evaluate_external_tilt import Recording
from scipy.io import loadmat


base = c1.base
intrinsic = c1.intrinsic
read, write, digest, stamp = c1.read, c1.write, c1.digest, c1.stamp
DATA, PILOT, TRANSFER = c1.DATA, c1.PILOT, c1.TRANSFER
METHODS = ("full_vqf_deg", "b6_full_deg", "b6_gain_025_deg")
GAIN = .25
POSES = {"-X": (-1, 0, 0), "+X": (1, 0, 0), "-Y": (0, -1, 0),
         "+Y": (0, 1, 0), "-Z": (0, 0, -1), "+Z": (0, 0, 1)}


def corrections(signal, dt, model, mean, std):
    quat, index, release, histories, _, _ = base.causal_inputs(signal, 1. / dt)
    frame, counts = intrinsic.intrinsic_windows(histories)
    delta = intrinsic.predict(model, histories, frame, mean, std)
    return (quat, quat[index], release, delta, dt), counts


def compare(native, mask, reference):
    quat, at_update, release, delta, dt = native
    scores = intrinsic.score_attitudes(quat, reference, mask, at_update, release, {"frozen": delta},
                                        {"b6_full": ("frozen", 1., False),
                                         "b6_gain_025": ("frozen", GAIN, False)}, dt)
    base.require(np.isfinite(list(scores.values())).all(), "Nonfinite inclination RMSE")
    pure, full, damp = (scores[k] for k in ("full_vqf", "b6_full", "b6_gain_025"))
    return {**dict(zip(METHODS, (pure, full, damp), strict=True)), "full_minus_pure_deg": full - pure,
            "gain_025_minus_pure_deg": damp - pure, "gain_025_minus_full_deg": damp - full}


def source_for(row, spec, yt, medium, split, c1_protocol):
    domain, name = row["dataset"], row["record"]
    if domain == "TUM-VI":
        base.require(name in ("room1", "room2"), "Unexpected TUM room")
        signal, dt, mask, _, reference, meta = c1.room_data(DATA / "TUM-VI", int(name[-1]), c1_protocol["tum"])
        base.require(digest(Path(meta["imu_path"])) == row["imu_file_sha256"]
                     and digest(Path(meta["mocap_path"])) == row["mocap_file_sha256"], "TUM raw sources changed")
        return signal, dt, mask, reference
    if domain == "YT_Server":
        base.require(name in c1_protocol["yt"]["records"] and spec["official_split"] == "valid", "Unexpected YT key")
        signal, dt, mask, reference, stats = c1.yt_record(yt[name], spec["native_frames"])
        base.require(stats["source_field_sha256"] == row["source_field_sha256"], "YT raw fields changed")
        return signal, dt, mask, reference
    if domain == "SHICE":
        base.require(row["group"] in ("R1", "R2", "R3", "R4"), "Unexpected SHICE device")
        dt, signal, reference, mask = intrinsic.shice.raw_record(name)
        return signal, dt, mask, reference
    if domain == "BROAD":
        path = Path(spec["path"])
        base.require(digest(path) == split["source_file_sha256"][str(path)], f"BROAD raw file changed: {name}")
        mat = loadmat(path, variable_names=("imu_acc", "imu_gyr", "opt_quat", "movement", "sampling_rate"))
        imu = np.column_stack((mat["imu_acc"], mat["imu_gyr"]))
        item = Recording(name, spec["group"], imu, mat["opt_quat"], mat["movement"].ravel().astype(bool),
                         float(mat["sampling_rate"].item()))
    else:
        base.require(domain == "Sassari" and name.startswith("medium/"), "Unexpected source")
        truth = medium["Qs"]
        item = Recording(name, "medium", medium[name.split("/")[1]][:, 1:7], truth,
                         np.ones(len(truth), dtype=bool), 100.)
    reference, label_ok = base.reference_direction(item.quaternion)
    observed = np.isfinite(item.imu).all(axis=1)
    # Pass original rows to the same causal fill used by B6.
    return item.imu, 1. / item.frequency, observed & label_ok & item.eligible, reference


def row_base(domain, name, group, reference_status, mask, key, stats, counts):
    return {"dataset": domain, "record": name, "group": group, "reference_status": reference_status,
            "native_frames": len(mask), "scored_frames": int(mask.sum()), "mask_key": key,
            "scoring_indices_sha256": intrinsic.mask_digest(mask), **stats, "frame_sources": counts}


def replay_prior(row, spec, yt, medium, split, c1_protocol, old_masks, model, mean, std):
    signal, dt, mask, reference = source_for(row, spec, yt, medium, split, c1_protocol)
    np.testing.assert_array_equal(mask, old_masks[row["mask_key"]])
    native, counts = corrections(signal, dt, model, mean, std)
    result = row_base(row["dataset"], row["record"], row["group"], row["reference_status"], mask,
                      f"c001_{row['mask_key']}", compare(native, mask, reference), counts)
    base.require(result["scoring_indices_sha256"] == row["scoring_indices_sha256"]
                 and result["scored_frames"] == row["scored_frames"], "C001 frame identity changed")
    errors = {k: result[k] - row[old] for k, old in (
        ("full_vqf_deg", "full_vqf_deg"), ("b6_full_deg", "b6_intrinsic_deg"))}
    base.require(max(map(abs, errors.values())) <= 1e-6, f"C001 control score changed: {errors}")
    result["recomputed_minus_c001_baseline_and_full_deg"] = errors
    return result, mask


def reference_array(path):
    ref = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 7)
    base.require(ref.shape == (387840, 7), "Array reference length changed")
    t = ref[:, 0]
    base.require(np.isfinite(t).all() and np.all(np.diff(t) > 0), "Invalid array reference timestamps")
    raw_force = ref[:, 4:]
    length = np.linalg.norm(raw_force, axis=1)
    labeled = np.isfinite(length) & (length > 0.)
    up = np.zeros_like(raw_force)
    up[labeled] = raw_force[labeled] / length[labeled, None]
    rounded = np.rint(up).astype(np.int8)
    np.testing.assert_allclose(up, rounded, rtol=0., atol=1e-10)
    base.require({tuple(v) for v in np.unique(rounded[labeled], axis=0)} == set(POSES.values()), "Array pose axes changed")
    return t, up, labeled, {key: np.all(rounded == axis, axis=1) & labeled for key, axis in POSES.items()}


def replay_array(spec, anchor, raw_ref, raw_hash, model, mean, std, at):
    path = Path(spec["path"])
    base.require("Calibration" in path.parts and "original data" in path.parts and digest(path) == raw_hash,
                 "Unexpected array source")
    channels = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 7)
    ref_times, reference, labeled, pose_masks = raw_ref
    base.require(len(channels) == spec["native_frames"] == anchor["frames"] == 387840, "Array native length changed")
    times = channels[:, 0]
    offset = float(np.max(np.abs(times - ref_times)))
    step = np.diff(times)
    dt = float(np.median(step))
    base.require(offset < 1e-7 and abs(dt - .005) <= 1e-7 and np.all(step > 0)
                 and np.max(np.abs(step - dt)) < .01 * dt, "Array clock no longer matches reference")
    imu = np.column_stack((channels[:, 4:], np.deg2rad(channels[:, 1:4])))
    mask = (times - times[0] >= 10.) & labeled & np.isfinite(imu).all(axis=1)
    code = intrinsic.mask_digest(mask)
    base.require(int(mask.sum()) == anchor["eligible_frames"] == 296400 and code == anchor["sha256"],
                 "Array native scoring mask differs from prior audit")
    native, counts = corrections(imu, dt, model, mean, std)
    name = spec["record"]
    row = row_base("IMU-Array", name, spec["group"], "same_time_known_static_specific_force", mask,
                   f"array_{at:02}", compare(native, mask, reference), counts)
    row.update({"native_dt_s": dt, "max_reference_clock_offset_s": offset, "raw_file_sha256": raw_hash})
    if "scores" in anchor:
        expected = anchor["scores"]["rmse_deg"]
        mismatch = {"full_vqf_deg": row["full_vqf_deg"] - expected["full_vqf"],
                    "b6_full_deg": row["b6_full_deg"] - expected["candidate"]}
        base.require(max(map(abs, mismatch.values())) <= 1e-5, f"Independent array anchor differs: {mismatch}")
        row["primary_minus_independent_anchor_deg"] = mismatch
    poses = []
    for name, pose_mask in pose_masks.items():
        eligible = mask & pose_mask
        count = int(eligible.sum())
        base.require(count == anchor["known_static_pose_frame_counts"][str(list(POSES[name]))], "Array pose coverage changed")
        poses.append({"record": spec["record"], "pose": name, "scored_frames": count,
                      "scoring_indices_sha256": intrinsic.mask_digest(eligible), **compare(native, eligible, reference)})
    base.require(sum(r["scored_frames"] for r in poses) == int(mask.sum()), "Pose partition does not cover array mask")
    return row, mask, poses


def summarize(rows):
    if not rows:
        return {"records": 0}
    means = {k: float(np.mean([r[k] for r in rows])) for k in METHODS}
    signs = {}
    for first, second, name in ((METHODS[1], METHODS[0], "full_vs_pure"),
                                (METHODS[2], METHODS[0], "gain_025_vs_pure"),
                                (METHODS[2], METHODS[1], "gain_025_vs_full")):
        signs[name] = {"wins": sum(r[first] < r[second] for r in rows), "losses": sum(r[first] > r[second] for r in rows),
                       "ties": sum(r[first] == r[second] for r in rows)}
    return {"records": len(rows), "scored_frames": sum(r["scored_frames"] for r in rows), "macro_rmse_deg": means,
            "full_minus_pure_deg": means[METHODS[1]] - means[METHODS[0]],
            "gain_025_minus_pure_deg": means[METHODS[2]] - means[METHODS[0]],
            "gain_025_minus_full_deg": means[METHODS[2]] - means[METHODS[1]], "paired_record_counts": signs}


def main():
    started = stamp()
    lock_path = HERE / "protocol.json"
    base.require(not lock_path.exists(), "C002 already started")
    global_protocol = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(global_protocol["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(global_protocol["end"]), "Outside the campaign window")
    inv = read(C001 / "inventory.json")
    previous = read(C001 / "results.json")
    c1_protocol = read(C001 / "protocol.json")
    masks_path = C001 / "scoring-masks.npz"
    base.require(digest(C001 / "inventory.json") == previous["inventory_sha256"] == c1_protocol["inventory_sha256"]
                 and digest(masks_path) == previous["masks_sha256"] and digest(C001 / "protocol.json") == previous["protocol_sha256"]
                 and digest(CAMPAIGN / "protocol.json") == inv["campaign_protocol_sha256"]
                 and not previous["failures"] and len(previous["sequences"]) == 69, "C001 evidence changed")
    incumbent = PILOT / global_protocol["incumbent"]
    base.require(digest(incumbent) == global_protocol["incumbent_sha256"] == previous["incumbent_sha256"],
                 "Incumbent has changed")
    b6_dir = TRANSFER / "batch6"
    b6_protocol = read(b6_dir / "protocol.json")
    base.require(all(digest(b6_dir / name) == sha for name, sha in b6_protocol["implementation_sha256"].items())
                 and all(digest(TRANSFER / name) == b6_protocol["source_sha256"][name]
                         for name in ("train_pair.py", "models.py", ".deps/transformers/models/mamba/modeling_mamba.py")),
                 "Frozen B6 inference implementation changed")
    specs = {(x["dataset"], x["record"]): x for x in inv["records"] if x["c001"] != "none"}
    array_specs = [x for x in inv["records"] if x["dataset"] == "IMU-Array" and x["side"] == "development"]
    array_audit_path = CAMPAIGN / "review/array_calibration_audit.json"
    array_audit = {x["name"]: x for x in read(array_audit_path)}
    base.require({x["record"] for x in array_specs} == set(array_audit) and len(array_specs) == 16, "Array panel changed")
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    raw_hashes = {x["record"]: digest(Path(x["path"])) for x in array_specs}
    ref_sha = digest(ref_path)
    evidence = {"c001_results_sha256": digest(C001 / "results.json"), "c001_protocol_sha256": digest(C001 / "protocol.json"),
                "c001_inventory_sha256": digest(C001 / "inventory.json"), "c001_masks_sha256": digest(masks_path),
                "array_review_sha256": digest(array_audit_path), "array_reference_sha256": ref_sha,
                "array_raw_sha256": raw_hashes, "checkpoint_sha256": digest(incumbent)}
    protocol = {"started_at": started, "pid": os.getpid(), "optimizer_steps": 0,
                "gains": {"full_vqf": 0., "b6_full": 1., "b6_gain_025": GAIN},
                "intervention": "Multiply the frozen B6 native tangent residual before the original direction normalization and zero-order hold.",
                "frozen_evidence": evidence, "c001_records": [[x["dataset"], x["record"]] for x in previous["sequences"]],
                "array_calibration_records": [x["record"] for x in array_specs],
                "array": {"same_physical_trial": True, "gyro_input_unit": "deg/s", "converted_gyro_unit": "rad/s",
                          "reference": str(ref_path), "initial_exclusion_s": 10.,
                          "mask": "Finite original IMU and finite nonzero static reference on the same native clock row.",
                          "pose_axes": POSES, "aggregation": "Pool native frames for each sensor and pose across repeated static runs, then mean of per-sensor RMSE across 16 synchronized channels.",
                          "expected_native_frames_per_sensor": 387840, "expected_scored_frames_per_sensor": 296400},
                "aggregation": "Native-frame tilt RMSE within record, then equal records within development domain. Any exact tie is neutral.",
                "scope": "69 C001 development records and 16 synchronized calibration channels. YT reference remains provisional."}
    write(lock_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "protocol_sha256": digest(lock_path), "status": "running"})
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    frozen = torch.load(incumbent, weights_only=True, map_location="cpu")
    base.require(frozen["model"] == c1.MODEL and frozen["seed"] == 1729 and frozen["epoch"] == 6
                 and frozen["window"] == 64 and frozen["split_sha256"] == inv["prior_external_split_sha256"],
                 "Frozen weight metadata changed")
    model = base.ResidualHead(c1.MODEL)
    model.load_state_dict(frozen["state_dict"])
    model.eval().requires_grad_(False).to(device)
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    split_path = TRANSFER / "external-split.json"
    base.require(digest(split_path) == inv["prior_external_split_sha256"], "External split changed")
    split = read(split_path)
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium data changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    raw_ref = reference_array(ref_path)
    rows, failures, masks, poses = [], [], {}, []
    with c1.h5py.File(inv["yt"]["path"], "r") as yt, np.load(masks_path, allow_pickle=False) as old_masks:
        for row in previous["sequences"]:
            domain, name = row["dataset"], row["record"]
            try:
                result, mask = replay_prior(row, specs[domain, name], yt["valid"], medium, split, c1_protocol,
                                            old_masks, model, mean, std)
            except (AssertionError, ValueError, OSError, RuntimeError, KeyError) as exc:
                print("FAILED", domain, name, repr(exc), flush=True)
                failures.append({"dataset": domain, "record": name, "error": repr(exc)})
                continue
            rows.append(result)
            masks[result["mask_key"]] = mask
            print(domain, name, result["scored_frames"], *(result[k] for k in METHODS), flush=True)
    for at, spec in enumerate(array_specs):
        name = spec["record"]
        try:
            result, mask, samples = replay_array(spec, array_audit[name], raw_ref, raw_hashes[name], model, mean, std, at)
        except (AssertionError, ValueError, OSError, RuntimeError, KeyError) as exc:
            print("FAILED", "IMU-Array", name, repr(exc), flush=True)
            failures.append({"dataset": "IMU-Array", "record": name, "error": repr(exc)})
            continue
        rows.append(result)
        masks[result["mask_key"]] = mask
        poses.extend(samples)
        print("IMU-Array", name, result["scored_frames"], *(result[k] for k in METHODS), flush=True)

    csv_path = HERE / "sequences.csv"
    cols = ("dataset", "record", "group", "reference_status", "native_frames", "scored_frames",
            "mask_key", "scoring_indices_sha256", *METHODS, "full_minus_pure_deg",
            "gain_025_minus_pure_deg", "gain_025_minus_full_deg")
    intrinsic.shice.save_csv(csv_path, [{k: r[k] for k in cols} for r in rows])
    pose_csv = HERE / "array-sensor-poses.csv"
    if poses:
        intrinsic.shice.save_csv(pose_csv, poses)
    mask_path = HERE / "scoring-masks.npz"
    stage = mask_path.with_suffix(".tmp")
    with stage.open("wb") as stream:
        np.savez_compressed(stream, **masks)
    stage.replace(mask_path)
    by_domain = {domain: summarize([r for r in rows if r["dataset"] == domain])
                 for domain in sorted({x["dataset"] for x in previous["sequences"]} | {"IMU-Array"})}
    by_domain["YT_Server"]["reference_status"] = "provisional_unknown_reference_producer"
    pose_summary = {pose: summarize([r for r in poses if r["pose"] == pose]) for pose in POSES}
    control_diffs = [v["recomputed_minus_c001_baseline_and_full_deg"] for v in rows
                     if "recomputed_minus_c001_baseline_and_full_deg" in v]
    complete = stamp()
    write(HERE / "results.json", {"started_at": started, "completed_at": complete, "pid": os.getpid(),
          "optimizer_steps": 0, "protocol_sha256": digest(lock_path), "gains": protocol["gains"],
          "evidence": evidence, "csv_sha256": digest(csv_path), "masks_sha256": digest(mask_path),
          "array_per_sensor_pose_csv_sha256": digest(pose_csv) if poses else None,
          "maximum_c001_control_delta_deg": max((abs(v) for x in control_diffs for v in x.values()), default=None),
          "summary": by_domain, "array_static_poses": pose_summary, "array_physical_trials": 1,
          "failures": failures, "sequences": rows, "array_sensor_poses": poses})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": complete,
                                  "protocol_sha256": digest(lock_path), "status": "complete" if not failures else "partial"})
    print(json.dumps({"summary": by_domain, "array_poses": pose_summary, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
