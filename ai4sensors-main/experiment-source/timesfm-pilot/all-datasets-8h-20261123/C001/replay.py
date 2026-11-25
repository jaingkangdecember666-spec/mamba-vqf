"""Replay the one frozen incumbent on C001 development records only."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import sys

from freeze_inventory import CAMPAIGN, DATA, HERE, PILOT, TRANSFER, digest, h5py, read, stamp, write

OVERNIGHT = PILOT / "overnight-20261122"
sys.path[:0] = [str(TRANSFER / "batch6"), str(TRANSFER / "batch4"), str(TRANSFER / "batch3"), str(OVERNIGHT)]

import fit_intrinsic as intrinsic
import numpy as np
import torch
from tilt_eval import tilt_error
from tum_vi import room_data


base = intrinsic.original
MODEL = "mamba_v1"
TUM_ROOMS = (1, 2)
G = 9.80665


def histogram(angles):
    if not len(angles):
        return {"frames": 0, "median_deg": None, "p95_deg": None, "max_deg": None}
    return {"frames": len(angles), "median_deg": float(np.median(angles)),
            "p95_deg": float(np.quantile(angles, .95)), "max_deg": float(np.max(angles))}


def quiet_gravity(imu, dt, score, truth):
    width = round(1. / dt)
    acc, gyro = imu[:, :3], imu[:, 3:]
    slow = np.linalg.norm(gyro, axis=1) <= .05
    near_g = np.abs(np.linalg.norm(acc, axis=1) - G) <= .20
    low = score & slow & near_g
    blocks, indices = [], []
    for first in range(0, len(score) - width + 1, width):
        s = slice(first, first + width)
        if low[s].all() and np.linalg.norm(acc[s] - acc[s].mean(axis=0), axis=1).max() <= .12:
            blocks.append(first)
            indices.extend(range(first, first + width))
    selected = np.asarray(indices, dtype=np.int64)
    low_angles = tilt_error(acc[low], truth[low])
    block_angles = tilt_error(acc[selected], truth[selected])
    return {"low_rate_near_g": histogram(low_angles),
            "complete_quiet_blocks": {"count": len(blocks), "samples_per_block": width,
                                      "first_indices": blocks, **histogram(block_angles)}}


def yt_record(group, expected_frames):
    acc, gyro, times, quat = (group[field][:] for field in ("acc", "gyr", "ts", "qua"))
    base.require(acc.shape == gyro.shape == (expected_frames, 3) and times.shape == (expected_frames,)
                 and quat.shape == (expected_frames, 4), f"YT shape changed: {group.name}")
    step = np.diff(times)
    dt = .005
    base.require(np.isfinite(times).all() and len(step) and np.all(step > 0)
                 and np.max(np.abs(step - dt)) <= .01 * dt, f"YT nonuniform native timing: {group.name}")
    raw = np.column_stack((acc, gyro))
    observed = np.isfinite(raw).all(axis=1)
    pose = np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat, axis=1) > 0)
    mask = (times - times[0] >= 10.) & observed & pose
    truth = np.zeros((len(raw), 3), dtype=np.float64)
    truth[pose] = base.body_up(quat[pose], base.EST_UP)
    source_sha = {}
    for name, field in (("acc", acc), ("gyr", gyro), ("ts", times), ("qua", quat)):
        source_sha[name] = hashlib.sha256(np.ascontiguousarray(field, dtype="<f8").tobytes()).hexdigest()
    return base.causal_fill(raw, observed), dt, mask, truth, {
        "reference_status": "provisional_unknown_reference_producer",
        "source_field_sha256": source_sha, "dt_s": dt, "ts_median_step_s": float(np.median(step)),
        "ts_min_step_s": float(step.min()), "ts_max_step_s": float(step.max()),
        "duration_s": float(times[-1] - times[0]), "finite_imu_frames": int(observed.sum()),
        "finite_nonzero_reference_frames": int(pose.sum()), "valid_paired_frames_before_warmup": int((observed & pose).sum()),
    }


def score(name, dataset, group, imu, dt, mask, reference, stats, model, mean, std):
    base.require(mask.any(), f"No paired scoring frames after exclusions: {name}")
    quat, index, release, histories, _, _ = base.causal_inputs(imu, 1. / dt)
    frame, counts = intrinsic.intrinsic_windows(histories)
    delta = intrinsic.predict(model, histories, frame, mean, std)
    rmse = intrinsic.score_attitudes(quat, reference, mask, quat[index], release,
                                    {MODEL: delta}, {MODEL: (MODEL, 1., False)}, dt)
    base.require(np.isfinite(list(rmse.values())).all(), f"Nonfinite score for {name}")
    result = {"dataset": dataset, "record": name, "group": group, "origin": "new_raw_replay",
              "native_frames": len(mask), "scored_frames": int(mask.sum()),
              "scoring_indices_sha256": intrinsic.mask_digest(mask), "full_vqf_deg": rmse["full_vqf"],
              "b6_intrinsic_deg": rmse[MODEL], "update_windows": len(release),
              "frame_source_counts": counts, **stats}
    result["gravity_diagnostic"] = quiet_gravity(imu, dt, mask, reference)
    return result


def tum_stats(meta, mask, observed, prior):
    imu_sha = digest(Path(meta["imu_path"]))
    mocap_sha = digest(Path(meta["mocap_path"]))
    old_mask_sha = hashlib.sha256(mask.tobytes()).hexdigest()
    base.require((imu_sha, mocap_sha, old_mask_sha, int(mask.sum()), meta["mocap_gaps_above_threshold"]) ==
                 (prior["imu_sha256"], prior["mocap_sha256"], prior["mask_sha256"],
                  prior["frames"], prior["large_mocap_gaps"]), f"Prior TUM raw audit mismatch: {meta['name']}")
    old_quiet = prior["quiet_gravity_check"]
    return {"reference_status": "audited_calibrated_mocap",
            "imu_file_sha256": imu_sha, "mocap_file_sha256": mocap_sha,
            "mocap_frames": meta["mocap_frames"], "mocap_gaps_above_25ms": meta["mocap_gaps_above_threshold"],
            "dt_s": float(meta["imu_dt_ns"] * 1e-9), "finite_imu_frames": int(observed.sum()),
            "reproduced_prior_raw_mask": True, "prior_quiet_blocks": old_quiet["qualifying_blocks"],
            "prior_quiet_median_deg": old_quiet["median_deg"]}


def reuses(inv, previous, masks_path, archived_masks):
    expected = {(r["dataset"], r["record"]): r for r in inv["records"] if r["c001"] == "reused_score"}
    source_rows = {**{(r["group"], r["sequence"]): r for r in previous["external_sequences"]},
                   **{("SHICE", r["sequence"]): r for r in previous["shice_sequences"]}}
    base.require(set(expected) == set(source_rows) and digest(masks_path) == previous["masks_sha256"],
                 "Archived B6 coverage or mask file changed")
    out = []
    with np.load(masks_path, allow_pickle=False) as masks:
        for (domain, name), source in source_rows.items():
            mask = masks[source["mask_key"]]
            sha = intrinsic.mask_digest(mask)
            base.require(sha == source["scoring_indices_sha256"] and int(mask.sum()) == source["frames"],
                         f"Archived frame indices differ: {domain}/{name}")
            identifier = f"b6_{source['mask_key']}"
            archived_masks[identifier] = mask
            out.append({"dataset": domain, "record": name, "group": expected[domain, name]["group"],
                        "origin": "reused_b6_raw_replay_and_mask", "native_frames": len(mask),
                        "scored_frames": int(mask.sum()), "scoring_indices_sha256": sha, "mask_key": identifier,
                        "full_vqf_deg": source["full_vqf_deg"], "b6_intrinsic_deg": source["batch6_mamba_deg"],
                        "reference_status": "archived_audited_reference", "archived_source": "transfer-mamba-20261123/batch6/results.json"})
    return out


def summarize(rows, inv):
    summary = {}
    for domain, counts in inv["record_denominators"].items():
        subset = [r for r in rows if r["dataset"] == domain]
        attempts = sum(r["c001"] != "none" for r in inv["records"] if r["dataset"] == domain)
        base_mean = float(np.mean([r["full_vqf_deg"] for r in subset])) if subset else None
        model_mean = float(np.mean([r["b6_intrinsic_deg"] for r in subset])) if subset else None
        summary[domain] = {"inventory_records": counts["records"], "planned_c001_records": attempts,
                           "scored_records": len(subset), "scored_native_denominator": sum(r["native_frames"] for r in subset),
                           "scoring_frames": sum(r["scored_frames"] for r in subset),
                           "reuse_count": sum(row["origin"] != "new_raw_replay" for row in subset),
                           "full_vqf_macro_deg": base_mean, "b6_intrinsic_macro_deg": model_mean,
                           "paired_strict_record_wins": sum(r["b6_intrinsic_deg"] < r["full_vqf_deg"] for r in subset),
                           "status": "deferred" if attempts == 0 else "partial" if len(subset) < attempts else
                                     "development_proxy_reference_only" if domain == "YT_Server" else "development"}
    return summary


def main():
    started = stamp()
    out = HERE / "results.json"
    protocol_path = HERE / "protocol.json"
    base.require(not out.exists() and not protocol_path.exists(), "C001 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside authorized experiment window")
    inv_path = HERE / "inventory.json"
    inv = read(inv_path)
    base.require(inv["campaign_protocol_sha256"] == digest(CAMPAIGN / "protocol.json"), "Campaign protocol changed")
    b6_dir = TRANSFER / "batch6"
    incumbent = PILOT / campaign["incumbent"]
    b6 = read(b6_dir / "results.json")
    b6_protocol = read(b6_dir / "protocol.json")
    base.require(digest(incumbent) == campaign["incumbent_sha256"] == b6["weights_sha256"]
                 and digest(b6_dir / "protocol.json") == b6["protocol_sha256"], "Frozen B6 evidence differs")
    base.require(digest(b6_dir / "intrinsic.py") == b6_protocol["implementation_sha256"]["intrinsic.py"]
                 and digest(b6_dir / "fit_intrinsic.py") == b6_protocol["implementation_sha256"]["fit_intrinsic.py"],
                 "Frozen intrinsic implementation differs")
    for name in ("train_pair.py", "models.py", ".deps/transformers/models/mamba/modeling_mamba.py"):
        base.require(digest(TRANSFER / name) == b6_protocol["source_sha256"][name], f"Frozen source differs: {name}")
    old_tum_protocol_path = OVERNIGHT / "tum-vi-protocol.json"
    tum_protocol = read(old_tum_protocol_path)
    tum_protocol = {key: tum_protocol[key] for key in ("max_consecutive_mocap_gap_ns", "initial_exclusion_ns",
                                                       "max_relative_imu_step_jitter", "world_up")}
    old_tum_audit_path = OVERNIGHT / "review/final_audit.json"
    old_tum = {r["name"]: r for r in read(old_tum_audit_path)["optional_tum_vi"]["raw_replays"]
               if r["name"] in ("room1", "room2")}
    yt_selected = inv["yt"]["selected_valid"]
    yt_expected = {r["record"]: r for r in inv["records"] if r["dataset"] == "YT_Server" and r["c001"] == "new_score"}
    base.require(set(yt_expected) == set(yt_selected) and len(yt_expected) == 12, "YT frozen shortlist mismatch")
    base.require(Path(inv["yt"]["path"]).stat().st_size == inv["yt"]["file_size"], "YT metadata fingerprint changed")
    protocol = {"started_at": started, "pid": os.getpid(), "inventory_sha256": digest(inv_path),
                "incumbent_sha256": campaign["incumbent_sha256"], "optimizer_steps": 0, "gain": 1.,
                "external_b6_results_sha256": digest(b6_dir / "results.json"),
                "tum": {**tum_protocol, "rooms": list(TUM_ROOMS), "historical_protocol_sha256": digest(old_tum_protocol_path),
                        "raw_audit_source_sha256": digest(old_tum_audit_path)},
                "yt": {"split": "valid", "records": yt_selected, "nominal_dt_s": .005,
                       "max_relative_jitter": .01, "initial_exclusion_s": 10.,
                       "mask": "Native rows with observed finite IMU and finite nonzero same-row qua. No gap interpolation or fitted mount.",
                       "reference": "Unknown producer, wxyz / +Z field convention; proxy diagnostics only."},
                "quiet_gravity": {"gyro_max_rad_s": .05, "acc_norm_departure_max_m_s2": .20,
                                  "block_acc_departure_max_m_s2": .12, "nonoverlapping_block_s": 1.,
                                  "diagnostic_scope": "Same native scoring frames, with both per-frame near-g/low-rate and complete quiet-block summaries."},
                "metric": "One native-frame inclination RMSE per record, then arithmetic mean. Default 6D FullVQF and frozen full-gain B6 share the mask.",
                "deferred": ["IMU-Array", "OxIOD", "RepoIMU"]}
    write(protocol_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "program": str(Path(__file__)),
                                  "protocol_sha256": digest(protocol_path), "status": "running"})
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    state = torch.load(incumbent, map_location="cpu", weights_only=True)
    base.require(state["model"] == MODEL and state["seed"] == 1729 and state["epoch"] == 6 and state["window"] == 64
                 and state["cap_deg"] == base.CAP and state["split_sha256"] == inv["prior_external_split_sha256"],
                 "Unexpected locked weight metadata")
    model = base.ResidualHead(MODEL)
    model.load_state_dict(state["state_dict"])
    model.eval().requires_grad_(False).to(device)
    mean, std = state["mean"].to(device), state["std"].to(device)
    masks = {}
    rows = reuses(inv, b6, b6_dir / "scoring-masks.npz", masks)
    failures = []
    for room in TUM_ROOMS:
        name = f"room{room}"
        try:
            imu, dt, mask, observed, truth, meta = room_data(DATA / "TUM-VI", room, tum_protocol)
            stats = tum_stats(meta, mask, observed, old_tum[name])
            row = score(name, "TUM-VI", name, imu, dt, mask, truth, stats, model, mean, std)
            row["prior_pure_vqf_abs_diff_deg"] = abs(row["full_vqf_deg"] - old_tum[name]["pure_replay_deg"])
            base.require(row["prior_pure_vqf_abs_diff_deg"] < 2e-4, f"TUM VQF did not reproduce for {name}")
            row["mask_key"] = f"tum_{name}"
            masks[row["mask_key"]] = mask
            rows.append(row)
            print("TUM-VI", name, row["scored_frames"], row["full_vqf_deg"], row["b6_intrinsic_deg"], flush=True)
        except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
            failures.append({"dataset": "TUM-VI", "record": name, "error": repr(exc)})
            print("TUM-VI failed", name, repr(exc), flush=True)
    with h5py.File(inv["yt"]["path"], "r") as data:
        valid = data["valid"]
        for at, name in enumerate(yt_selected):
            spec = yt_expected[name]
            try:
                imu, dt, mask, truth, stats = yt_record(valid[name], spec["native_frames"])
                row = score(name, "YT_Server", spec["group"], imu, dt, mask, truth, stats, model, mean, std)
                row["mask_key"] = f"yt_{at:02}"
                masks[row["mask_key"]] = mask
                rows.append(row)
                print("YT_Server", name, row["scored_frames"], row["full_vqf_deg"], row["b6_intrinsic_deg"],
                      row["gravity_diagnostic"]["complete_quiet_blocks"]["count"], flush=True)
            except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
                failures.append({"dataset": "YT_Server", "record": name, "error": repr(exc)})
                print("YT_Server failed", name, repr(exc), flush=True)

    for row in rows:
        row["delta_deg"] = row["b6_intrinsic_deg"] - row["full_vqf_deg"]
    columns = ("dataset", "record", "group", "origin", "native_frames", "scored_frames", "mask_key",
               "scoring_indices_sha256", "full_vqf_deg", "b6_intrinsic_deg", "delta_deg", "reference_status")
    csv_path = HERE / "sequences.csv"
    intrinsic.shice.save_csv(csv_path, [{k: row[k] for k in columns} for row in rows])
    mask_path = HERE / "scoring-masks.npz"
    temp = mask_path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **masks)
    temp.replace(mask_path)
    summary = summarize(rows, inv)
    completed = stamp()
    write(out, {"started_at": started, "completed_at": completed, "pid": os.getpid(),
                "optimizer_steps": 0, "incumbent_sha256": campaign["incumbent_sha256"],
                "inventory_sha256": digest(inv_path), "protocol_sha256": digest(protocol_path),
                "csv_sha256": digest(csv_path), "masks_sha256": digest(mask_path),
                "summary": summary, "failures": failures, "sequences": rows})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": completed,
                                  "program": str(Path(__file__)), "protocol_sha256": digest(protocol_path),
                                  "status": "complete" if not failures else "complete_with_record_failures"})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
