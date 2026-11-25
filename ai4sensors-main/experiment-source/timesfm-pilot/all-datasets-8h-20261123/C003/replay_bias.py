"""Score one causal accelerometer-bias strategy on the frozen 85 development records."""

from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C002 = CAMPAIGN / "C002"
sys.path.insert(0, str(C002))

import replay_gain as c2
import numpy as np
import torch
from online_bias import Settings, estimate, synthetic_checks
from scipy.io import loadmat

C001, DATA, PILOT, TRANSFER = c2.C001, c2.DATA, c2.PILOT, c2.TRANSFER
base, intrinsic = c2.base, c2.intrinsic
read, write, digest, stamp = c2.read, c2.write, c2.digest, c2.stamp
METHODS = ("full_vqf_deg", "b6_gain_025_deg", "debiased_full_vqf_deg", "debiased_b6_gain_025_deg")
CONTRASTS = (("raw_b6_vs_pure", 1, 0), ("debiased_vqf_vs_pure", 2, 0), ("debiased_b6_vs_pure", 3, 0),
             ("debiased_b6_vs_raw_b6", 3, 1), ("debiased_b6_vs_debiased_vqf", 3, 2))


def raw_shice(name):
    src = intrinsic.shice
    device = name.split("-")[0]
    path = src.ROOT / f"{device}_aligned_formal" / f"{name}.csv"
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        columns = ("sample_index", "time_s", *src.CHANNELS, *src.QUATERNION, "is_interpolated")
        values = np.loadtxt(stream, delimiter=",", quotechar='"', usecols=[header.index(k) for k in columns], ndmin=2)
    np.testing.assert_array_equal(values[:, 0], np.arange(len(values)))
    time, imu, quat = values[:, 1], values[:, 2:8], values[:, 8:12]
    step = np.diff(time)
    dt = float(np.median(step))
    base.require(dt > 0 and np.allclose(step, dt, rtol=1e-5, atol=1e-9), f"SHICE timing changed: {name}")
    observed = (values[:, -1] == 0) & np.isfinite(imu).all(axis=1)
    pose = np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > 0)
    reference = np.zeros((len(time), 3), dtype=np.float32)
    reference[pose] = src.body_up(quat[pose], src.REF_UP)
    with np.load(src.ROOT / "Label_quality_audit_20261116" / name / "screening_masks.npz", allow_pickle=False) as masks:
        np.testing.assert_array_equal(masks["sample_index"], values[:, 0])
        score = (time >= src.WARMUP_S) & observed & pose & masks["context_ok"].astype(bool) & masks["observed_loss_ok"].astype(bool)
    return imu, dt, observed, reference, score, {}


def sources(row, spec, info, medium, yt, split, c1_proto, array, array_hashes):
    domain, name = row["dataset"], row["record"]
    if domain == "SHICE":
        base.require(row["group"] in ("R1", "R2", "R3", "R4"), "Unexpected SHICE device")
        return raw_shice(name)
    if domain == "TUM-VI":
        base.require(name in ("room1", "room2"), "Unexpected TUM room")
        signal, dt, mask, good, reference, meta = c2.c1.room_data(DATA / "TUM-VI", int(name[-1]), c1_proto["tum"])
        base.require(digest(Path(meta["imu_path"])) == info["imu_file_sha256"]
                     and digest(Path(meta["mocap_path"])) == info["mocap_file_sha256"], "TUM raw evidence changed")
        return signal, dt, good, reference, mask, {}
    if domain == "YT_Server":
        base.require(name in c1_proto["yt"]["records"] and spec["official_split"] == "valid", "Unexpected YT key")
        node = yt[name]
        signal, dt, mask, reference, stats = c2.c1.yt_record(node, spec["native_frames"])
        base.require(stats["source_field_sha256"] == info["source_field_sha256"], "YT raw evidence changed")
        observed = np.ones(len(signal), dtype=bool)
        if stats["finite_imu_frames"] != len(signal):
            observed = np.isfinite(node["acc"][:]).all(axis=1) & np.isfinite(node["gyr"][:]).all(axis=1)
        return signal, dt, observed, reference, mask, {}
    if domain == "IMU-Array":
        path = Path(spec["path"])
        base.require(spec["side"] == "development" and path.is_relative_to(DATA / "IMU-Array-main/Calibration")
                     and "original data" in path.parts and digest(path) == array_hashes[name],
                     f"Array raw evidence changed: {name}")
        native = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 7)
        ref_time, reference, labeled, pose_masks = array
        time = native[:, 0]
        np.testing.assert_array_equal(time, ref_time)
        dt = float(np.median(np.diff(time)))
        signal = np.column_stack((native[:, 4:7], np.deg2rad(native[:, 1:4])))
        observed = np.isfinite(signal).all(axis=1)
        mask = observed & labeled & (time - time[0] >= 10.)
        base.require(len(mask) == 387840 and int(mask.sum()) == 296400, "Array raw coverage changed")
        return signal, dt, observed, reference, mask, pose_masks
    signal, dt, mask, reference = c2.source_for(info, spec, yt, medium, split, c1_proto)
    return signal, dt, np.isfinite(signal).all(axis=1), reference, mask, {}


def score(native, mask, reference):
    quat, at_update, release, delta, dt = native
    scores = intrinsic.score_attitudes(quat, reference, mask, at_update, release, {"frozen": delta},
                                      {"quarter": ("frozen", .25, False)}, dt)
    base.require(np.isfinite(list(scores.values())).all(), "Nonfinite inclination RMSE")
    return scores["full_vqf"], scores["quarter"]


def replay(row, spec, info, medium, yt, split, c1_proto, array, array_hashes, old_masks, model, mean, std, cfg):
    raw, dt, observed, reference, mask, axis_masks = sources(row, spec, info, medium, yt, split, c1_proto, array, array_hashes)
    np.testing.assert_array_equal(mask, old_masks[row["mask_key"]])
    digest_mask = intrinsic.mask_digest(mask)
    base.require(digest_mask == row["scoring_indices_sha256"] and int(mask.sum()) == row["scored_frames"],
                 "C002 native mask changed")
    # The estimator operates on the uncorrected original readings and original packet validity.
    bias, active, diagnostic, updates = estimate(raw, observed, dt, cfg)
    original_imu = base.causal_fill(raw, observed)
    original_native, _ = c2.corrections(original_imu, dt, model, mean, std)
    if active.any():
        modified = original_imu.copy()
        modified[:, :3] -= bias
        debiased_native, _ = c2.corrections(modified, dt, model, mean, std)
        np.testing.assert_array_equal(original_native[2], debiased_native[2])
    else:
        debiased_native = original_native
    control = score(original_native, mask, reference)
    candidate = score(debiased_native, mask, reference) if active.any() else control
    differences = {k: value - row[k] for k, value in zip(METHODS[:2], control, strict=True)}
    base.require(max(map(abs, differences.values())) <= 1e-6, f"C002 controls changed: {differences}")
    first = diagnostic["first_active_native_index"]
    diagnostic.update({"first_active_time_s": first * dt if first is not None else None,
                       "active_scored_frames": int((mask & active).sum()), "native_dt_s": dt})
    result = {**{k: row[k] for k in ("dataset", "record", "group", "reference_status", "native_frames",
                                   "scored_frames", "mask_key", "scoring_indices_sha256")},
              **dict(zip(METHODS, (*control, *candidate), strict=True)), "bias": diagnostic,
              "control_minus_c002_deg": differences}
    poses = []
    for axis, reference_mask in axis_masks.items():
        use = mask & reference_mask
        no_bias = score(original_native, use, reference)
        with_bias = score(debiased_native, use, reference) if active.any() else no_bias
        count = int(use.sum())
        coverage = int((active & use).sum())
        poses.append({"record": row["record"], "pose": axis, "scored_frames": count,
                      "scoring_indices_sha256": intrinsic.mask_digest(use), "active_scored_frames": coverage,
                      "active_scored_fraction": coverage / count,
                      **dict(zip(METHODS, (*no_bias, *with_bias), strict=True))})
    return result, poses, updates


def summary(rows):
    if not rows:
        return {}
    values = {method: float(np.mean([r[method] for r in rows])) for method in METHODS}
    differences = {}
    for label, first, second in CONTRASTS:
        a, b = METHODS[first], METHODS[second]
        differences[label] = {"delta_macro_deg": values[a] - values[b], "wins": sum(r[a] < r[b] for r in rows),
                              "losses": sum(r[a] > r[b] for r in rows), "ties": sum(r[a] == r[b] for r in rows)}
    frames = sum(row["scored_frames"] for row in rows)
    active_scored = sum((row["bias"] if "bias" in row else row)["active_scored_frames"] for row in rows)
    return {"records": len(rows), "scored_frames": frames, "active_scored_frames": active_scored,
            "active_scored_fraction": active_scored / frames, "macro_rmse_deg": values, "contrasts": differences,
            "records_with_active_scored_frames": sum((r["bias"] if "bias" in r else r)["active_scored_frames"] > 0 for r in rows)}


def main():
    started = stamp()
    protocol_path = HERE / "protocol.json"
    base.require(not protocol_path.exists(), "C003 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    previous = read(C002 / "results.json")
    c2_protocol = read(C002 / "protocol.json")
    inv = read(C001 / "inventory.json")
    c1_previous = read(C001 / "results.json")
    c1_proto = read(C001 / "protocol.json")
    evidence = c2_protocol["frozen_evidence"]
    masks_path = C002 / "scoring-masks.npz"
    base.require(digest(C002 / "protocol.json") == previous["protocol_sha256"] and digest(masks_path) == previous["masks_sha256"]
                 and digest(C001 / "inventory.json") == evidence["c001_inventory_sha256"]
                 and digest(C001 / "results.json") == evidence["c001_results_sha256"]
                 and digest(C001 / "protocol.json") == evidence["c001_protocol_sha256"]
                 and not previous["failures"] and len(previous["sequences"]) == 85, "C002 evidence changed")
    specs = {(x["dataset"], x["record"]): x for x in inv["records"] if x["c001"] != "none"
             or (x["dataset"] == "IMU-Array" and x["side"] == "development")}
    matched = {(x["dataset"], x["record"]) for x in previous["sequences"]}
    base.require(set(specs) == matched and len(matched) == 85, "Unexpected partition coverage")
    original_rows = {(x["dataset"], x["record"]): x for x in c1_previous["sequences"]}
    incumbent = PILOT / campaign["incumbent"]
    base.require(digest(incumbent) == campaign["incumbent_sha256"] == evidence["checkpoint_sha256"],
                 "Frozen B6 weight hash changed")
    prior_b6 = read(TRANSFER / "batch6/protocol.json")
    base.require(all(digest(TRANSFER / "batch6" / file) == value
                     for file, value in prior_b6["implementation_sha256"].items()), "B6 inference implementation changed")
    cfg = Settings()
    protocol = {"started_at": started, "pid": os.getpid(), "optimizer_steps": 0, "gain": .25,
                "checkpoint_sha256": digest(incumbent), "c002_results_sha256": digest(C002 / "results.json"),
                "c002_masks_sha256": digest(masks_path), "c002_protocol_sha256": digest(C002 / "protocol.json"),
                "estimator_sha256": digest(HERE / "online_bias.py"), "thresholds": asdict(cfg),
                "signal": "Original accelerometer and gyro only, in m/s² and rad/s; missing-packet blocks excluded from estimation.",
                "geometry": "Equal confirmed direction centers, with online arithmetic averaging within each cluster. Fit sphere center and radius, require non-coplanar support and bounded residual/center/radius.",
                "release": "Disjoint complete one-second blocks; projected norm bounded smoothing after the complete block. Bias becomes active only at its following native frame. Until the first valid fit use zero; later rejected fits retain the most recent accepted update.",
                "score": "Frozen 85 C002 indices. Raw/debiased VQF and fixed 0.25 B6; same 64-window/release/hold, state reset per record.",
                "reference": "Used at evaluation only. YT reference producer remains unknown. Calibration has one physical trial.",
                "records": [[x["dataset"], x["record"]] for x in previous["sequences"]]}
    write(protocol_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "protocol_sha256": digest(protocol_path),
                                  "status": "running"})
    tests = synthetic_checks(cfg)
    write(HERE / "synthetic-checks.json", tests)
    print("synthetic checks", tests, flush=True)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    state = torch.load(incumbent, map_location="cpu", weights_only=True)
    base.require(state["model"] == c2.c1.MODEL and state["seed"] == 1729 and state["epoch"] == 6 and state["window"] == 64,
                 "Unexpected B6 metadata")
    model = base.ResidualHead(state["model"])
    model.load_state_dict(state["state_dict"])
    model.eval().requires_grad_(False).to(device)
    mean, std = state["mean"].to(device), state["std"].to(device)
    split_path = TRANSFER / "external-split.json"
    base.require(digest(split_path) == inv["prior_external_split_sha256"], "BROAD/Sassari split changed")
    split = read(split_path)
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari raw evidence changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref_path) == evidence["array_reference_sha256"], "Array reference changed")
    array = c2.reference_array(ref_path)
    pose_archive = {(r["record"], r["pose"]): r for r in previous["array_sensor_poses"]}
    rows, failures, poses, updates = [], [], [], []
    with np.load(masks_path, allow_pickle=False) as masks, c2.c1.h5py.File(inv["yt"]["path"], "r") as yt:
        for row in previous["sequences"]:
            key = row["dataset"], row["record"]
            try:
                result, decomposed, switches = replay(row, specs[key], original_rows.get(key), medium, yt["valid"], split,
                                                      c1_proto, array, evidence["array_raw_sha256"], masks,
                                                      model, mean, std, cfg)
                for p in decomposed:
                    old = pose_archive[p["record"], p["pose"]]
                    base.require(p["scored_frames"] == old["scored_frames"]
                                 and p["scoring_indices_sha256"] == old["scoring_indices_sha256"]
                                 and max(abs(p[m] - old[m]) for m in METHODS[:2]) <= 1e-6,
                                 f"Array pose differs from C002: {p['record']}/{p['pose']}")
            except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
                continue
            rows.append(result)
            poses.extend(decomposed)
            updates.append({"dataset": key[0], "record": key[1], "updates": switches})
            print(*key, result["scored_frames"], "active", result["bias"]["active_scored_frames"],
                  *(result[k] for k in METHODS), flush=True)
    csv_path = HERE / "sequences.csv"
    columns = ("dataset", "record", "group", "reference_status", "native_frames", "scored_frames", "mask_key",
               "scoring_indices_sha256", *METHODS)
    intrinsic.shice.save_csv(csv_path, [
        {**{k: row[k] for k in columns}, **{k: row["bias"][k] for k in (
            "status", "first_active_native_index", "first_active_time_s", "active_native_frames", "active_scored_frames",
            "direction_clusters", "confirmed_directions", "accepted_update_count", "last_applied_bias_norm_m_s2")},
         **dict(zip(("bias_x_m_s2", "bias_y_m_s2", "bias_z_m_s2"), row["bias"]["last_applied_bias_xyz_m_s2"], strict=True)),
         "active_scored_fraction": row["bias"]["active_scored_frames"] / row["scored_frames"]}
        for row in rows])
    pose_path = HERE / "array-sensor-poses.csv"
    intrinsic.shice.save_csv(pose_path, poses)
    update_path = HERE / "bias-updates.json"
    write(update_path, updates)
    domain = {name: summary([r for r in rows if r["dataset"] == name]) for name in previous["summary"]}
    by_pose = {axis: summary([r for r in poses if r["pose"] == axis]) for axis in c2.POSES}
    completed = stamp()
    write(HERE / "results.json", {"started_at": started, "completed_at": completed, "pid": os.getpid(),
          "optimizer_steps": 0, "protocol_sha256": digest(protocol_path), "synthetic_checks_sha256": digest(HERE / "synthetic-checks.json"),
          "shared_scoring_masks_path": str(masks_path), "shared_scoring_masks_sha256": digest(masks_path),
          "csv_sha256": digest(csv_path), "array_pose_csv_sha256": digest(pose_path), "bias_updates_sha256": digest(update_path),
          "summary": domain, "array_static_poses": by_pose, "array_physical_trials": 1, "failures": failures,
          "sequences": rows, "array_sensor_poses": poses})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": completed,
                                  "protocol_sha256": digest(protocol_path), "status": "complete" if not failures else "partial"})
    print(json.dumps({"summary": domain, "array_static_poses": by_pose, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
