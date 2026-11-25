"""Compare two EuRoC label choices at the fixed C009 draws and restored gyro input."""

import csv
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C009 = CAMPAIGN / "C009"
sys.path.insert(0, str(C009))

import fit_euroc as c9
import numpy as np
import torch
from scipy.io import loadmat

c8, c6, c5, c4, c3, c2 = c9.c8, c9.c6, c9.c5, c9.c4, c9.c3, c9.c2
base, intrinsic = c9.base, c9.intrinsic
read, write, digest, stamp, array_sha = c9.read, c9.write, c9.digest, c9.stamp, c9.array_sha
PILOT, DATA, TRANSFER = c9.PILOT, c9.DATA, c9.TRANSFER
SOURCE = CAMPAIGN / "external-sources" / "euroc-original-inputs"
SAMPLING = C009 / "sampling.npz"
CONTROLS = c9.METHODS
METHODS = (*CONTROLS, "A_deg", "B_deg")
ARMS = ("A", "B")


def state_csv(spec):
    name = spec["record"]
    return SOURCE / (("V1_01_easy_original" if name == c9.EUROC_ORDER[0] else name) + ".csv")


def restored_windows(spec, old_release, old_last, old_first, draws, cfg):
    name = spec["record"]
    easy = name == c9.EUROC_ORDER[0]
    keys = ("utc_ns", "acc_m_s2", "gyr_rad_s", "original_wxyz", "alternative_wxyz", "alternative_valid", "movement")
    with np.load(SOURCE / spec["prepared_file"], allow_pickle=False) as data:
        base.require(set(data.files) == set(keys), f"Prepared layout changed for {name}")
        clock, acc, gyro, original, alternative, alt_valid, movement = (data[k] for k in keys)
    n = spec["hdf5_source_rows"]
    base.require(clock.shape == movement.shape == alt_valid.shape == (n,) and clock.dtype == np.int64
                 and acc.shape == gyro.shape == (n, 3) and original.shape == alternative.shape == (n, 4)
                 and movement.dtype == alt_valid.dtype == np.bool_ and np.all(np.diff(clock) > 0),
                 f"Invalid prepared rows for {name}")
    timestamp_ranges = [int(np.diff(clock).min()), int(np.diff(clock).max())]
    base.require(timestamp_ranges == spec["native_clock_interval_ns"], f"Prepared clock interval changed for {name}")
    published = state_csv(spec)
    utc = np.loadtxt(published, delimiter=",", usecols=(0,), skiprows=1, dtype=np.int64)
    bias = np.loadtxt(published, delimiter=",", usecols=(11, 12, 13), skiprows=1)
    np.testing.assert_array_equal(clock, utc)
    with c2.c1.h5py.File(c9.EUROC / (name + ".hdf5"), "r") as old:
        original_acc = np.column_stack([old[k][:] for k in ("acc_x", "acc_y", "acc_z")])
        original_gyro = np.column_stack([old[k][:] for k in ("gyr_x", "gyr_y", "gyr_z")])
        original_pose = np.column_stack([old[k][:] for k in ("opt_a", "opt_b", "opt_c", "opt_d")])
        np.testing.assert_array_equal(movement, old["movement_mask"][:].astype(bool))
    np.testing.assert_array_equal(acc, original_acc)
    np.testing.assert_array_equal(original, original_pose)
    reconstruction_error = float(np.max(np.abs(gyro - (original_gyro.astype(np.float64) + bias))))
    base.require(reconstruction_error <= 1e-12, f"Gyro restoration differs from released-state bias for {name}")
    if not easy:
        np.testing.assert_array_equal(alternative, original)
    raw = np.column_stack((acc, gyro))
    observed = np.isfinite(raw).all(axis=1)
    truth_a, pose_a = base.reference_direction(original)
    truth_b, pose_b = base.reference_direction(alternative)
    q, index, release, history, complete, bias_diag = c4.inertial_replay(raw, observed, 1. / c9.EUROC_RATE, cfg, training=True)
    # Preserve the C009 200 Hz release grid and filter warm-up. Never resample the candidate pool.
    at = np.searchsorted(release, old_release)
    base.require(np.all(at < len(release)), f"C009 release missing for {name}")
    np.testing.assert_array_equal(release[at], old_release)
    np.testing.assert_array_equal(index[at], old_last)
    np.testing.assert_array_equal(index[at] + 1 - 2 * base.WINDOW, old_first)
    base.require(np.all(complete[at] & observed[old_release] & movement[old_release] & pose_a[old_release]
                        & pose_b[old_release] & alt_valid[old_release] & (old_first >= c9.WARMUP_NATIVE)),
                 f"A frozen C009 release failed common A/B eligibility for {name}")
    windows = history[at]
    frame = intrinsic.build_frame(windows)
    base.require(frame.valid.all(), f"A frozen C009 release no longer has a valid intrinsic frame for {name}")
    canonical = intrinsic.transform_inputs(windows, frame)
    orientation = c8.normalize(q[old_last])
    inverse = orientation.copy()
    inverse[:, 1:] *= -1.
    q_release = c8.normalize(q[old_release])
    targets = []
    world_errors = []
    for truth in (truth_a, truth_b):
        world = c8.rotate(q_release, truth[old_release])
        at_input = c8.rotate(inverse, world)
        error = float(np.max(np.abs(c8.rotate(orientation, at_input) - world)))
        base.require(error <= 2e-12, f"Native point alignment differs for {name}")
        world_errors.append(error)
        targets.append(intrinsic.transform_directions(at_input, frame))
    changed = np.any(targets[0] != targets[1], axis=1)
    base.require(np.all(changed) if easy else not changed.any(), f"Unexpected A/B label changes for {name}")
    angular = c8.tilt_error(targets[0][draws], targets[1][draws])
    audit = {"record": name, "native_rows": n, "clock_interval_ns": timestamp_ranges,
             "original_acc_and_quaternion_match_C009_exactly": True,
             "gyro_restoration_max_abs_error_rad_s": reconstruction_error,
             "gyro_max_abs_change_vs_C009_rad_s": float(np.max(np.abs(gyro - original_gyro))),
             "C009_eligible_windows": len(at), "C010_common_eligible_windows": int(frame.valid.sum()),
             "alternative_covered_C009_native_releases": int(alt_valid[old_release].sum()),
             "C009_draws": len(draws), "changed_A_B_unique_labels": int(changed.sum()),
             "changed_A_B_draw_labels": int(changed[draws].sum()),
             "target_angle_median_draws_deg": float(np.median(angular)),
             "max_target_angle_draws_deg": float(angular.max()),
             "target_world_roundtrip_max_abs_by_arm": dict(zip(ARMS, world_errors)),
             "C009_release_native_sha256": array_sha(old_release, "<i8"),
             "C010_release_native_sha256": array_sha(release[at], "<i8"),
             "C010_raw_input_sha256": array_sha(raw, "<f8"), "C010_canonical_input_sha256": array_sha(canonical, "<f4"),
             "target_sha256_by_arm": {a: array_sha(t, "<f4") for a, t in zip(ARMS, targets)},
             "C003_active_native_frames": bias_diag["active_native_frames"]}
    return canonical, targets, audit, (raw, observed, 1. / c9.EUROC_RATE)


def fit_arm(arm, model, inputs, targets, indices, euroc, mean, std, initial_loss):
    torch.manual_seed(c4.SEED)
    model.train().requires_grad_(True)
    backbone = [p for name, p in model.named_parameters() if not name.startswith("output.")]
    head = list(model.output.parameters())
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": c8.BACKBONE_LR},
                                   {"params": head, "lr": c4.LR}], lr=c4.LR, weight_decay=c4.WD)
    history = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, started = perf_counter(), stamp()
    for step, (idx, is_euroc) in enumerate(zip(indices, euroc, strict=True), 1):
        optimizer.zero_grad(set_to_none=True)
        error = (c8.prediction(model, inputs[idx], mean, std) - targets[idx]).square().sum(dim=1)
        loss = error.mean()
        measured = float(loss.detach())
        if step == 1:
            base.require(measured == initial_loss, f"Arm {arm} started with different weight, input or target")
        loss.backward()
        base.require(all(p.grad is not None for p in model.parameters()), f"Arm {arm} has disconnected parameters")
        norms = torch.stack([p.grad.detach().norm() for p in (*backbone, *head)])
        trunk_norm, head_norm = (float(n) for n in (norms[:len(backbone)].norm(), norms[len(backbone):].norm()))
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.))
        base.require(np.isfinite([measured, trunk_norm, head_norm, gradient]).all() and trunk_norm > 0. and head_norm > 0.,
                     f"Arm {arm} has invalid gradients at step {step}")
        optimizer.step()
        row = {"arm": arm, "step": step, "loss": measured, "gradient_norm": gradient,
               "backbone_gradient_l2": trunk_norm, "output_gradient_l2": head_norm,
               "retained_slots": 80, "euroc_slots": 16,
               "retained_slot_loss": float(error.detach()[~is_euroc].mean()),
               "euroc_slot_loss": float(error.detach()[is_euroc].mean())}
        history.append(row)
        if step in (1, 40, 80, 120, c4.STEPS):
            print("training", row, flush=True)
    torch.cuda.synchronize()
    return model.eval().requires_grad_(False), history, {"started_at": started, "completed_at": stamp(),
            "seconds": perf_counter() - start, "optimizer_steps": len(history),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}


def summarize(rows, denominator):
    values = {k: float(np.mean([r[k] for r in rows])) for k in METHODS} if rows else {}
    return {"records": len(rows), "denominator": denominator, "scored_frames": sum(r["scored_frames"] for r in rows),
            "macro_rmse_deg": values,
            "A_minus_C009_deg": values["A_deg"] - values["C009_deg"] if rows else None,
            "B_minus_A_deg": values["B_deg"] - values["A_deg"] if rows else None,
            "A_wins_vs_C009": sum(r["A_deg"] < r["C009_deg"] for r in rows),
            "B_wins_vs_A": sum(r["B_deg"] < r["A_deg"] for r in rows)}


def replay(prior, inventory, campaign, split, models, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(p) for p in (c9.C001 / "results.json", c9.C001 / "protocol.json",
                                             c9.C003 / "results.json", c9.C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "One or more records are outside the frozen development panel")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(ref_path)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    scorer = {a: (a, c4.GAIN, False) for a in ARMS}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(c8.MASKS, allow_pickle=False) as masks, np.load(c8.GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            domain, name = key = old["dataset"], old["record"]
            try:
                if domain in ("OxIOD", "RepoIMU"):
                    _, q, at, release, history, mask, ref, qa, _, _ = c5.optical_payload(specs[key], cfg)
                    q = q.as_quat(scalar_first=True)
                    axes = {}
                    base.require(qa["quality_flags"] == old["reference_quality_flags"], "Optical QA changed")
                else:
                    raw, dt, observed, ref, mask, axes = c3.sources(map3[key], specs[key], map1.get(key), medium, yt["valid"],
                                                                  split, c1p, array, evidence["array_raw_sha256"])
                    q, at, release, history, _, _ = c4.inertial_replay(raw, observed, dt, cfg)
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                base.require(intrinsic.mask_digest(mask) == old["scoring_indices_sha256"]
                             and int(mask.sum()) == old["scored_frames"], "C009 scoring frames changed")
                rest = c6.rest_flags(history)
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                deltas = {}
                for arm, model in models.items():
                    delta = c4.evaluate_windows(model, {arm: model.output}, history, mean, std)[arm]
                    delta[rest] = 0.
                    deltas[arm] = delta

                def score(use, archive):
                    values = intrinsic.score_attitudes(q, ref, use, q[at], release, deltas, scorer, 1.)
                    a, b = values["A"], values["B"]
                    base.require(np.isfinite([a, b]).all(), "Nonfinite paired score")
                    return {"A_deg": a, "B_deg": b, "A_minus_C009_deg": a - archive["C009_deg"], "B_minus_A_deg": b - a}

                scored = {**old, **score(mask, old)}
                detail = []
                for axis, pose in axes.items():
                    use = mask & pose
                    anchor = old_poses[name, axis]
                    base.require(intrinsic.mask_digest(use) == anchor["scoring_indices_sha256"]
                                 and int(use.sum()) == anchor["scored_frames"], "IMU-Array pose mask changed")
                    detail.append({**anchor, **score(use, anchor)})
                rows.append(scored)
                poses.extend(detail)
                print(*key, scored["scored_frames"], *(scored[m] for m in METHODS), flush=True)
            except (AssertionError, ValueError, OSError, RuntimeError, KeyError, IndexError) as exc:
                failures.append({"dataset": domain, "record": name, "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    by_key = {(r["dataset"], r["record"]): r for r in rows}
    errors = {(r["dataset"], r["record"]): r["error"] for r in failures}
    fields = ("dataset", "record", "status", "reference_status", "scored_frames", "mask_key",
              "scoring_indices_sha256", "group", "instrument", "scenario", "rest_scored_fraction",
              *METHODS, "A_minus_C009_deg", "B_minus_A_deg", "error")
    output = []
    for old in prior["sequences"]:
        key = old["dataset"], old["record"]
        pair = by_key.get(key, {})
        row = {**old, **pair, "status": "scored" if pair else "failed", "error": errors.get(key, "")}
        output.append({k: row.get(k, "") for k in fields})
    path, pose_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv"
    intrinsic.shice.save_csv(path, output)
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    panels = {d: summarize([r for r in rows if r["dataset"] == d], prior["summary"][d]["denominator"])
              for d in campaign["datasets"]}
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_status"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_status"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": panels, "optical_subsets": {k: summarize(v, prior["optical_subsets"][k]["denominator"]) for k, v in subsets.items()},
            "csv_sha256": digest(path), "pose_csv_sha256": digest(pose_path) if poses else None,
            "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    process_path, protocol_path = HERE / "process.json", HERE / "protocol.json"
    base.require(not protocol_path.exists() and not any((HERE / (a + ".pt")).exists() for a in ARMS), "C010 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p9, t9, s9, prior = (read(C009 / n) for n in ("protocol.json", "training.json", "process.json", "results.json"))
    base.require(s9["status"] == "complete" and len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C009 / "results.json") == s9["result_sha256"]
                 and digest(C009 / "protocol.json") == s9["protocol_sha256"] == prior["protocol_sha256"]
                 and digest(C009 / "training.json") == prior["training_sha256"]
                 and digest(C009 / "sequences.csv") == prior["csv_sha256"]
                 and digest(C009 / "array-sensor-poses.csv") == prior["pose_csv_sha256"]
                 and all(digest(C009 / path) == sha for path, sha in t9["artifacts_sha256"].items())
                 and all(digest(PILOT / path) == sha for path, sha in p9["files_sha256"].items())
                 and digest(c8.MASKS) == p9["scoring_masks_sha256"] == prior["scoring_masks_sha256"]
                 and digest(c8.GATES) == p9["gate_traces_sha256"], "C009 freeze, results, or masked panel changed")
    expected_opt = {"kind": "AdamW", "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH,
                    "output_lr": c4.LR, "backbone_lr": c8.BACKBONE_LR, "weight_decay": c4.WD, "grad_clip": 2.}
    base.require(p9["optimizer"] == expected_opt and p9["gain"] == c4.GAIN, "C009 optimizer or gain changed")
    path_audit, path_imu_check = SOURCE / "source-audit.json", SOURCE / "original-imu-verification.json"
    audit, raw_check = read(path_audit), read(path_imu_check)
    sources = audit["records"]
    base.require(tuple(r["record"] for r in sources) == c9.EUROC_ORDER
                 and sum(s["hdf5_source_rows"] for s in sources) == 66346
                 and all(s["prepared_file"] == s["record"] + ".npz"
                         and digest(SOURCE / s["prepared_file"]) == s["prepared_sha256"]
                         and digest(state_csv(s)) == s["original_state"]["file_sha256"] for s in sources)
                 and digest(SOURCE / "V1_01_easy.csv") == sources[0]["corrected_state"]["file_sha256"]
                 and raw_check["original_state_csv_numeric_parity"] and raw_check["time_matched_training_rows"] == 28712
                 and raw_check["max_abs_reconstructed_gyro_difference_rad_s"] <= 2.98e-8
                 and raw_check["fields_read"]["state_groundtruth_estimate0"]["sha256"] == digest(state_csv(sources[0]))
                 and sources[0]["corrected_pose_covered_rows"] == 28712 and sources[0]["corrected_label_rows_excluded"] == 0,
                 "EuRoC original gyro or alternative easy label bundle changed")
    incumbent = PILOT / campaign["incumbent"]
    base.require(digest(incumbent) == p9["initial_checkpoint_sha256"] == campaign["incumbent_sha256"], "C009 initialization changed")
    p8, prep8 = (read(c9.C008 / n) for n in ("protocol.json", "training-preparation.json"))
    records = p9["old_source_records"]
    base.require(records == p8["source_records"] and len(records) == 63
                 and p9["validation_records"] == [[r["dataset"], r["record"]] for r in prior["sequences"]],
                 "Frozen source or development list changed")
    files = {**p9["files_sha256"]}
    paths = [Path(__file__).resolve(), path_audit, path_imu_check, SOURCE / "V1_01_easy.csv",
             *(SOURCE / s["prepared_file"] for s in sources), *(state_csv(s) for s in sources),
             *(C009 / n for n in ("protocol.json", "process.json", "training.json", "results.json", "sequences.csv",
                                  "array-sensor-poses.csv", "sampling.npz"))]
    files.update({str(p.relative_to(PILOT)): digest(p) for p in paths})
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": files,
                "optimizer": expected_opt, "gain": c4.GAIN, "arms": {
                    "A": "Restored acquisition gyro on all three EuRoC sources; original released orientations on every source.",
                    "B": "Same shared A inputs and draw tensor; OpenVINS revised easy orientation only. Original medium and difficult labels."
                },
                "source_records": sources, "old_source_records": records, "yt_prefix_exclusion": p9["yt_prefix_exclusion"],
                "raw_easy_independent_verification": raw_check, "source_audit": audit,
                "initial_checkpoint_sha256": digest(incumbent), "normalization_sha256": p9["normalization_sha256"],
                "sampling_npz_sha256": digest(SAMPLING), "masks_sha256": digest(c8.MASKS), "gate_traces_sha256": digest(c8.GATES),
                "validation_records": p9["validation_records"], "native_schedule": "Reuse every record and release in C009 sampling.npz. Complete eligible history must precede each archived C009 release. Retain all 80 original slots per step.",
                "target": p9["training_target"], "loss": p9["training_loss"], "candidate_replay": p9["replay_policy"],
                "scored_controls": "Locked per-record FullVQF, C003, C006, C008 and C009 values and pose values copied from the verified C009 result. A and B computed on the same native masks, original releases and baseline rest flags.",
                "source_qualification": "Original released bias reversal checked against all three timestamped original states; independent raw acquisition CSV parity applies only to easy. No reference-derived correction is used in development evaluation.",
                "scope": "Exactly one fit from B6 per arm and the original 213 development records. Final holdouts remain sealed."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "started_at": started, "pid": os.getpid(),
                             "protocol_sha256": protocol_sha, **extra})

    status("preparing")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "CUDA is unavailable")
    device = torch.device("cuda")
    frozen = torch.load(incumbent, weights_only=True, map_location="cpu")
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    base.require(frozen["model"] == "mamba_v1" and sum(p.numel() for p in core.parameters()) == 5595
                 and sum(p.numel() for p in core.output.parameters()) == 75
                 and c4.tensor_hash({k: frozen[k] for k in ("mean", "std")}) == p9["normalization_sha256"],
                 "B6 model or fixed normalization changed")
    core.eval().requires_grad_(False).to(device)
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    cfg = c3.Settings(**read(c9.C003 / "protocol.json")["thresholds"])
    inventory = read(c9.C001 / "inventory.json")
    c4_prepared = read(c9.C004 / "training-preparation.json")["records"]
    with np.load(c9.C004 / "schedules.npz", allow_pickle=False) as item:
        schedule, offsets, old_releases = (item[k] for k in ("B", "record_offsets", "release_indices"))
    with np.load(SAMPLING, allow_pickle=False) as file:
        ledger = {key: file[key] for key in file.files}
    np.testing.assert_array_equal(schedule, ledger["C008_global_window"])
    mask = ledger["replacement_mask"]
    np.testing.assert_array_equal(mask, np.arange(96)[None, :] % 6 == np.arange(160)[:, None] % 6)
    np.testing.assert_array_equal(ledger["C009_release_native"][~mask], old_releases[schedule[~mask]])
    np.testing.assert_array_equal(ledger["C009_global_window"][~mask], schedule[~mask])
    np.testing.assert_array_equal(ledger["C009_record_ordinal"][~mask], np.searchsorted(offsets[1:], schedule[~mask], side="right"))
    base.require(array_sha(schedule, "<i8") == p9["B_schedule_sha256"], "Original C008 draw identity changed")
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        canonical, target, remap, old_audits, causal, _ = c8.point_windows(records, c4_prepared, schedule, offsets,
                                                                       old_releases, yt["train"], cfg)
    base.require(old_audits == prep8["records"] and array_sha(canonical, "<f4") == prep8["canonical_inputs_sha256"]
                 and array_sha(target, "<f4") == prep8["point_targets_sha256"], "Retained canonical inputs or labels changed")
    starts = ledger["euroc_eligible_offsets"]
    track, local = (ledger[k] for k in ("euroc_drawn_ordinal", "euroc_drawn_eligible_index"))
    base.require(len(starts) == 4 and len(track) == len(local) == 2560 and np.all(local >= 0)
                 and np.all((track >= 0) & (track < 3)) and np.all(local < np.diff(starts)[track]), "EuRoC draw map is invalid")
    pool = starts[track] + local
    np.testing.assert_array_equal(ledger["C009_global_window"][mask], offsets[-1] + pool)
    np.testing.assert_array_equal(ledger["C009_record_ordinal"][mask], len(records) + track)
    np.testing.assert_array_equal(ledger["C009_release_native"][mask], ledger["euroc_eligible_release_native"][pool])
    restored = []
    for i, source in enumerate(sources):
        sl = slice(starts[i], starts[i + 1])
        native = [ledger[key][sl] for key in ("euroc_eligible_release_native", "euroc_eligible_input_last_native",
                                              "euroc_eligible_history_first_native")]
        restored.append(restored_windows(source, *native, local[track == i], cfg))
    base.require([r[2]["C009_draws"] for r in restored] == [854, 853, 853], "Per-source draw count changed")
    for source, item in zip(sources, restored, strict=True):
        causal[source["record"]] = item[3]
    shared = np.concatenate([canonical, *(x[0] for x in restored)])
    targets = {arm: np.concatenate([target, *(x[1][i] for x in restored)]) for i, arm in enumerate(ARMS)}
    draw = remap.copy()
    draw[mask] = len(canonical) + pool
    np.testing.assert_array_equal(draw[~mask], remap[~mask])
    changed = np.any(targets["A"][draw] != targets["B"][draw], axis=-1)
    np.testing.assert_array_equal(changed, ledger["C009_record_ordinal"] == 63)
    np.testing.assert_array_equal(shared[draw[~mask]], canonical[remap[~mask]])
    for result in targets.values():
        np.testing.assert_array_equal(result[draw[~mask]], target[remap[~mask]])
    batch_rows = []
    for s, take in enumerate(draw):
        common_hash = array_sha(shared[take], "<f4")
        batch_rows.append({"step": s + 1, "A_input_sha256": common_hash, "B_input_sha256": common_hash,
                           "A_target_sha256": array_sha(targets["A"][take], "<f4"),
                           "B_target_sha256": array_sha(targets["B"][take], "<f4"),
                           "changed_easy_target_slots": int(changed[s].sum()),
                           "C009_record_ordinal_sha256": array_sha(ledger["C009_record_ordinal"][s], "<i8"),
                           "C009_native_release_sha256": array_sha(ledger["C009_release_native"][s], "<i8")})
    draw_path = HERE / "batch-audit.csv"
    intrinsic.shice.save_csv(draw_path, batch_rows)
    prep_path = HERE / "training-preparation.json"
    common_sha = array_sha(shared, "<f4")
    write(prep_path, {"protocol_sha256": protocol_sha, "sampling_npz_sha256": digest(SAMPLING),
                      "batch_audit_sha256": digest(draw_path), "draw_matrix_sha256": array_sha(draw, "<i8"),
                      "C009_native_release_matrix_sha256": array_sha(ledger["C009_release_native"], "<i8"),
                      "shared_input_sha256_by_arm": dict.fromkeys(ARMS, common_sha),
                      "target_pool_sha256_by_arm": {a: array_sha(t, "<f4") for a, t in targets.items()},
                      "old_records": old_audits, "old_slots": 12800, "euroc_slots": 2560,
                      "B_changed_easy_only_draw_labels": int(changed.sum()), "euroc_records": [x[2] for x in restored]})
    base.require(all(digest(PILOT / path) == sha for path, sha in files.items())
                 and digest(SAMPLING) == protocol["sampling_npz_sha256"], "C010 inputs changed after locking")
    print("paired source lock passed", "shared_input", common_sha, "easy-only changed targets", int(changed.sum()), flush=True)
    inputs, draws, euroc_slots = [torch.from_numpy(x).to(device) for x in (shared, draw, mask)]
    model_hash = c4.tensor_hash(core.state_dict())
    models, fits, all_steps = {}, {}, []
    for arm in ARMS:
        model = deepcopy(core)
        base.require(c4.tensor_hash(model.state_dict()) == model_hash, "A/B initialization differs")
        labels = torch.from_numpy(targets[arm]).to(device)
        with torch.inference_mode():
            init_loss = float((c8.prediction(model, inputs[draws[0]], mean, std) - labels[draws[0]]).square().sum(dim=1).mean())
        status("fitting", arm=arm, completed_arms=list(fits), preparation_sha256=digest(prep_path))
        model, history, info = fit_arm(arm, model, inputs, labels, draws, euroc_slots, mean, std, init_loss)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        path = HERE / f"{arm}.pt"
        stage = path.with_suffix(".tmp")
        torch.save({**frozen, "state_dict": state, "C010_arm": arm, "C010_fit": info, "protocol_sha256": protocol_sha,
                    "sampling_sha256": digest(SAMPLING), "initial_parameters_sha256": model_hash, "input_sha256": common_sha}, stage)
        stage.replace(path)
        saved = torch.load(path, weights_only=True, map_location="cpu")
        base.require(all(torch.equal(saved[k], frozen[k]) for k in ("mean", "std")), "Fixed normalization changed")
        model.load_state_dict(saved["state_dict"])
        layers_path = HERE / f"{arm}-layer-norms.json"
        write(layers_path, c8.layer_differences(frozen["state_dict"], saved["state_dict"]))
        prefix_path = HERE / f"{arm}-prefix-audit.json"
        write(prefix_path, c8.final_prefix(model, causal, mean, std, cfg))
        models[arm] = model
        fits[arm] = {**info, "weight_file": path.name, "weight_sha256": digest(path),
                     "layer_norms_sha256": digest(layers_path), "prefix_audit_sha256": digest(prefix_path),
                     "initial_loss": init_loss, "last_step_loss": history[-1]["loss"]}
        all_steps.extend({**row, **batch_rows[i]} for i, row in enumerate(history))
        intrinsic.shice.save_csv(HERE / "training-steps.csv", all_steps)
    base.require(c4.tensor_hash(core.state_dict()) == model_hash, "B6 initialization mutated between arms")
    steps_path = HERE / "training-steps.csv"
    training_path = HERE / "training.json"
    write(training_path, {"fits": fits, "actual_fit_count": len(fits), "total_optimizer_steps": len(all_steps),
                          "GPU": torch.cuda.get_device_name(device), "torch": str(torch.__version__),
                          "training_steps_sha256": digest(steps_path), "preparation_sha256": digest(prep_path)})
    del inputs, targets, labels, draws, shared, canonical
    status("scoring", completed_arms=list(fits), training_sha256=digest(training_path))
    print("C010 paired 160-step weights persisted; scoring the frozen 213 development records", flush=True)
    scoring_at = perf_counter()
    result = replay(prior, inventory, campaign, read(TRANSFER / "external-split.json"), models, mean, std, cfg)
    completed = stamp()
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_sha,
                   "training_sha256": digest(training_path), "masks_sha256": protocol["masks_sha256"],
                   "control_result_sha256": digest(C009 / "results.json"), "scoring_seconds": perf_counter() - scoring_at,
                   "total_seconds": perf_counter() - wall})
    result_path = HERE / "results.json"
    write(result_path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(result_path))
    print("C010 coverage", len(result["sequences"]), "failures", result["failures"], "seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process = HERE / "process.json"
        if process.exists():
            last = read(process)
            if last["pid"] == os.getpid():
                write(process, {**last, "status": "failed", "phase": last["status"],
                                "completed_at": stamp(), "failure": repr(error)})
        raise
