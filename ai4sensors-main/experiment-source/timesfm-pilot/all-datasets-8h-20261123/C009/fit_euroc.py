"""Run the C009 external-source substitution once, then score the frozen development panel."""

import csv
from collections import Counter
from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C008 = CAMPAIGN / "C008"
sys.path.insert(0, str(C008))

import fit_backbone as c8
import numpy as np
import torch
from scipy.io import loadmat

c7, c6, c5, c4, c3, c2 = c8.c7, c8.c6, c8.c5, c8.c4, c8.c3, c8.c2
base, intrinsic = c8.base, c8.intrinsic
read, write, stamp, digest = c8.read, c8.write, c8.stamp, c8.digest
array_sha = c8.array_sha
C001, C002, C003, C004, C006 = c8.C001, c8.C002, c8.C003, c8.C004, c8.C006
DATA, PILOT, TRANSFER = c8.DATA, c8.PILOT, c8.TRANSFER
EUROC = CAMPAIGN / "external-sources" / "euroc-vicon-room1"
EUROC_ORDER = ("V1_01_easy", "V1_02_medium", "V1_03_difficult")
EUROC_SEED = 271828
EUROC_RATE = 200
WARMUP_NATIVE = 10 * EUROC_RATE
METHODS = ("full_vqf_deg", "c003_deg", "C006_deg", "C008_deg", "C009_deg")


def euroc_windows(spec, quality, cfg):
    path = EUROC / spec["file"]
    acc_names = ("acc_x", "acc_y", "acc_z")
    gyro_names = ("gyr_x", "gyr_y", "gyr_z")
    pose_names = ("opt_a", "opt_b", "opt_c", "opt_d")
    with c2.c1.h5py.File(path, "r") as file:
        base.require(set(file.keys()) == set(spec["fields"]) == {*acc_names, *gyro_names, *pose_names, "movement_mask", "dt"}
                     and all(list(file[k].shape) == shape for k, shape in spec["fields"].items())
                     and dict(file.attrs) == spec["attributes"], f"Unexpected schema in {path.name}")
        raw = np.column_stack([file[k][:] for k in (*acc_names, *gyro_names)]).astype(np.float64)
        quaternion = np.column_stack([file[k][:] for k in pose_names]).astype(np.float64)
        time_step = file["dt"][:]
        movement = file["movement_mask"][:]
    n = len(raw)
    dt = 1. / EUROC_RATE
    base.require(n == quality["rows"] and np.allclose(time_step, dt, rtol=0., atol=2e-10)
                 and np.all(np.isin(movement, [0, 1])) and int(movement.sum()) == quality["movement_rows"],
                 f"Unexpected sample grid or mask in {path.name}")
    observed = np.isfinite(raw).all(axis=1)
    ref, pose = base.reference_direction(quaternion)
    base.require(int((observed & pose).sum()) == quality["finite_with_pose"], f"Valid-pose count changed in {path.name}")
    q, index, release, histories, complete, bias = c4.inertial_replay(raw, observed, dt, cfg, training=True)
    # At 200 Hz every completed 100 Hz input block spans exactly two native frames.
    native_per_block = EUROC_RATE // 100
    np.testing.assert_array_equal(index, release)
    first = index + 1 - native_per_block * base.WINDOW
    score = observed & pose & movement.astype(bool)
    before_geometry = complete & score[release] & (first >= WARMUP_NATIVE)
    use = np.flatnonzero(before_geometry)
    window = histories[use]
    frame = intrinsic.build_frame(window)
    body_at_input = c8.normalize(q[index[use]])
    inverse = body_at_input.copy()
    inverse[:, 1:] *= -1.
    point_world = c8.rotate(c8.normalize(q[release[use]]), ref[release[use]])
    point_local_body = c8.rotate(inverse, point_world)
    err = float(np.max(np.abs(c8.rotate(body_at_input, point_local_body) - point_world)))
    base.require(err <= 2e-12 and frame.valid.any(), f"Invalid EuRoC point alignment for {path.name}")
    keep = np.flatnonzero(frame.valid)
    frame_kept = type(frame)(frame.basis[keep], frame.source[keep], frame.anchor[keep])
    target = intrinsic.transform_directions(point_local_body[keep], frame_kept)
    np.testing.assert_allclose(np.einsum("ni,nji->nj", target, frame_kept.basis), point_local_body[keep], rtol=0., atol=1e-7)
    features = intrinsic.transform_inputs(window[keep], frame_kept)
    chosen = use[keep]
    sample = keep[np.linspace(0, len(keep) - 1, min(32, len(keep)), dtype=int)]
    audit = {"domain": "EuRoC", "record": spec["record"], "file_sha256": spec["sha256"], "native_rows": n,
             "sample_rate_hz": EUROC_RATE, "declared_dt_min_max": [float(time_step.min()), float(time_step.max())],
             "finite_imu_rows": int(observed.sum()), "valid_pose_rows": int(pose.sum()),
             "movement_rows": int(movement.sum()), "all_scheduled_windows": len(release),
             "complete_point_windows_before_warmup_filter": int(np.count_nonzero(complete & score[release])),
             "eligible_windows": len(use), "intrinsic_degenerate_windows": len(use) - len(keep),
             "usable": len(keep), "first_usable_history_native_index": int(first[chosen[0]]),
             "first_usable_release_native_index": int(release[chosen[0]]),
             "last_usable_release_native_index": int(release[chosen[-1]]),
             "active_native_frames": bias["active_native_frames"], "target_world_roundtrip_max_abs": err,
             "canonical_inputs_sha256": array_sha(features, "<f4"), "point_targets_sha256": array_sha(target, "<f4"),
             "eligible_release_sha256": array_sha(release[chosen], "<i8")}
    return features, target, release[chosen], index[chosen], first[chosen], audit, (raw, observed, dt), window[sample].copy()


def mix_draws(schedule, offsets, releases, draw_index, european):
    replacement = (np.arange(schedule.shape[1])[None, :] % 6) == (np.arange(schedule.shape[0])[:, None] % 6)
    base.require(np.all(replacement.sum(axis=1) == 16), "Expected 16 replacements in every batch")
    rng = np.random.Generator(np.random.PCG64(EUROC_SEED))
    ordinal = np.arange(int(replacement.sum())) % len(european)
    euro_offsets = np.r_[0, np.cumsum([len(item[2]) for item in european])]
    local = np.asarray([rng.integers(len(european[k][2])) for k in ordinal], dtype=np.int64)
    pool = local + euro_offsets[ordinal]
    euro_releases, euro_last, euro_first = (np.concatenate([item[k] for item in european]) for k in (2, 3, 4))
    original_record = np.searchsorted(offsets[1:], schedule, side="right")
    original_release = releases[schedule]
    mixed_record, mixed_release, mixed_global = (a.copy() for a in (original_record, original_release, schedule))
    mixed_record[replacement] = len(offsets) - 1 + ordinal
    mixed_release[replacement] = euro_releases[pool]
    mixed_global[replacement] = offsets[-1] + pool
    remap = draw_index.copy()
    remap[replacement] = int(draw_index.max()) + 1 + pool
    np.testing.assert_array_equal(remap[~replacement], draw_index[~replacement])
    np.testing.assert_array_equal(mixed_global[~replacement], schedule[~replacement])
    np.testing.assert_array_equal(mixed_release[~replacement], original_release[~replacement])
    path = HERE / "sampling.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, replacement_mask=replacement, C008_global_window=schedule,
                            C008_record_ordinal=original_record, C008_release_native=original_release,
                            C009_global_window=mixed_global, C009_record_ordinal=mixed_record, C009_release_native=mixed_release,
                            euroc_eligible_offsets=euro_offsets, euroc_eligible_release_native=euro_releases,
                            euroc_eligible_input_last_native=euro_last, euroc_eligible_history_first_native=euro_first,
                            euroc_drawn_ordinal=ordinal, euroc_drawn_eligible_index=local)
    return remap, replacement, mixed_record, pool, path


def fit(model, canonical, target, draw_index, replacement, mean, std, expected_first):
    torch.manual_seed(c4.SEED)
    named = dict(model.named_parameters())
    backbone = [p for key, p in named.items() if not key.startswith("output.")]
    head = list(model.output.parameters())
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": c8.BACKBONE_LR},
                                   {"params": head, "lr": c4.LR}], lr=c4.LR, weight_decay=c4.WD)
    history = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started, start = stamp(), perf_counter()
    for step, (idx, euroc) in enumerate(zip(draw_index, replacement, strict=True), 1):
        optimizer.zero_grad(set_to_none=True)
        errors = (c8.prediction(model, canonical[idx], mean, std) - target[idx]).square().sum(dim=1)
        loss = errors.mean()
        number = float(loss.detach())
        if step == 1:
            base.require(number == expected_first, f"Mixed batch changed before optimization: {number}")
        loss.backward()
        base.require(all(p.grad is not None for p in named.values()), "A Mamba parameter is disconnected")
        norms = torch.stack([p.grad.detach().norm() for p in (*backbone, *head)])
        trunk_norm, head_norm = (float(x) for x in (norms[:len(backbone)].norm(), norms[len(backbone):].norm()))
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.))
        base.require(np.isfinite([number, grad, trunk_norm, head_norm]).all() and trunk_norm > 0. and head_norm > 0.,
                     f"Invalid end-to-end backprop at step {step}")
        optimizer.step()
        row = {"arm": "C009", "step": step, "loss": number, "gradient_norm": grad,
               "backbone_gradient_l2": trunk_norm, "output_gradient_l2": head_norm,
               "retained_C008_slots": 80, "euroc_slots": 16,
               "retained_slot_loss": float(errors.detach()[~euroc].mean()),
               "euroc_slot_loss": float(errors.detach()[euroc].mean())}
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
            "C009_delta_macro_deg": {k: values["C009_deg"] - values[k] for k in METHODS[:-1]} if rows else {},
            "C009_wins_vs": {k: sum(r["C009_deg"] < r[k] for r in rows) for k in METHODS[:-1]}}


def replay(prior, c008, inventory, campaign, split, core, control, tuned, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(path) for path in (C001 / "results.json", C001 / "protocol.json",
                                                C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    frozen = [(r["dataset"], r["record"]) for r in prior["sequences"]]
    base.require(frozen == [(r["dataset"], r["record"]) for r in c008["sequences"]] and len(set(frozen)) == 213,
                 "C008 score order or coverage changed")
    keys = set(frozen)
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(set(specs) == keys and all(r["side"] in ("development", "development_previously_used_for_selection")
                                            for r in specs.values()), "A replay record is outside development")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(ref)
    old_poses = {(r["record"], r["pose"]): r for r in c008["array_sensor_poses"]}
    layers = {"c003": core.output, "C006": control.output}
    rows, failures, poses = [], [], []
    match = ("scored_frames", "scoring_indices_sha256", "full_vqf_deg", "c003_deg", "C006_deg")
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(c8.MASKS, allow_pickle=False) as masks, np.load(c8.GATES, allow_pickle=False) as gates:
        for key, old, archive in zip(frozen, prior["sequences"], c008["sequences"], strict=True):
            try:
                # The shared evaluator calls its one supplied candidate C008. Here it evaluates the C009 weights.
                row, detail = c8.evaluate(old, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                          evidence["array_raw_sha256"], masks, gates, core, layers, tuned, mean, std, cfg)
                base.require(all(row[k] == archive[k] for k in (*match, "mask_key", "reference_status")),
                             "C008 archive comparison does not match the fresh controls")
                row["C009_deg"] = row.pop("C008_deg")
                row["C008_deg"] = archive["C008_deg"]
                for r in detail:
                    prev = old_poses[r["record"], r["pose"]]
                    base.require(all(r[k] == prev[k] for k in match), "C008 pose controls or scoring masks changed")
                    r["C009_deg"] = r.pop("C008_deg")
                    r["C008_deg"] = prev["C008_deg"]
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, ValueError, OSError, RuntimeError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    by_key = {(r["dataset"], r["record"]): r for r in rows}
    errored = {(r["dataset"], r["record"]): r["error"] for r in failures}
    fields = ("dataset", "record", "status", "reference_status", "scored_frames", "mask_key",
              "scoring_indices_sha256", "group", "instrument", "scenario", "rest_scored_fraction", *METHODS, "error")
    output = []
    for key, old in zip(frozen, prior["sequences"], strict=True):
        measured = by_key.get(key, {})
        archive = {**old, **measured, "status": "scored" if measured else "failed", "error": errored.get(key, "")}
        output.append({k: measured.get(k, "") if k in METHODS else archive.get(k, "") for k in fields})
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
    errors = [abs(x) for r in rows for x in r["controls_minus_prior_deg"].values()]
    return {"summary": panels, "optical_subsets": {k: summarize(v, prior["optical_subsets"][k]["denominator"]) for k, v in subsets.items()},
            "csv_sha256": digest(path), "pose_csv_sha256": digest(pose_path) if poses else None,
            "maximum_absolute_control_error_deg": max(errors, default=None), "sequences": rows,
            "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    checkpoint = HERE / "mamba-end-to-end.pt"
    base.require(not protocol_path.exists() and not checkpoint.exists(), "C009 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p8, t8, prep8, s8, r8 = (read(C008 / n) for n in ("protocol.json", "training.json", "training-preparation.json",
                                                    "process.json", "results.json"))
    base.require(s8["status"] == "complete" and not r8["failures"] and len(r8["sequences"]) == 213
                 and digest(C008 / "protocol.json") == s8["protocol_sha256"] == r8["protocol_sha256"]
                 and digest(C008 / "results.json") == s8["results_sha256"]
                 and digest(C008 / "training.json") == r8["training_sha256"]
                 and all(digest(C008 / path) == sha for path, sha in t8["artifacts_sha256"].items())
                 and all(digest(PILOT / path) == sha for path, sha in p8["files_sha256"].items()),
                 "C008 freeze or score evidence changed")
    prior, p3, inventory = (read(path) for path in (C006 / "results.json", C003 / "protocol.json", C001 / "inventory.json"))
    c4_records = read(C004 / "training-preparation.json")["records"]
    records = p8["source_records"]
    base.require(Counter(r["domain"] for r in records) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and len(prior["sequences"]) == 213 and not prior["failures"]
                 and prior["shared_masks_sha256"] == p8["scoring_masks_sha256"] == r8["scoring_masks_sha256"] == digest(c8.MASKS)
                 and prior["gate_traces_sha256"] == p8["gate_traces_sha256"] == digest(c8.GATES),
                 "C008 source scope, development masks or gate changed")
    expected_opt = {"kind": "AdamW", "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH,
                    "output_lr": c4.LR, "backbone_lr": c8.BACKBONE_LR, "weight_decay": c4.WD, "grad_clip": 2.}
    base.require(p8["optimizer"] == expected_opt and p8["gain"] == c4.GAIN, "C008 optimization or inference policy changed")
    source_path, quality_path = EUROC / "source-manifest.json", EUROC / "reference-quality.json"
    original_manifest = read(source_path)
    specs, quality = original_manifest["records"], read(quality_path)
    base.require(tuple(s["record"] for s in specs) == tuple(q["record"] for q in quality) == EUROC_ORDER
                 and sum(q["rows"] for q in quality) == 66346
                 and all(s["room"] == "V1" and s["partition"] == "external_training_source"
                         and s["file"] == f"{s['record']}.hdf5" and digest(EUROC / s["file"]) == s["sha256"]
                         and (EUROC / s["file"]).stat().st_size == s["bytes"] for s in specs),
                 "EuRoC manifest, quality records, or HDF5 checksums changed")
    expected_files = {**p8["files_sha256"]}
    locked_paths = [HERE / "fit_euroc.py", source_path, quality_path, *(EUROC / s["file"] for s in specs),
                    *(C008 / n for n in ("fit_backbone.py", "protocol.json", "results.json", "training.json", "process.json",
                                         "training-preparation.json", "mamba-end-to-end.pt")),
                    C001 / "results.json", C001 / "protocol.json", C002 / "protocol.json", C003 / "results.json"]
    expected_files.update({str(p.relative_to(PILOT)): digest(p) for p in locked_paths})
    incumbent = PILOT / campaign["incumbent"]
    bpath = C004 / read(C004 / "training.json")["arms"]["B"]["weights"]
    base.require(digest(incumbent) == p8["initial_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(bpath) == p8["C006_B_checkpoint_sha256"], "B6 initialization or C006 readout changed")
    with np.load(C004 / "schedules.npz", allow_pickle=False) as a:
        schedule, offsets, releases = (a[key] for key in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and array_sha(schedule, "<i8") == p8["B_schedule_sha256"] == prep8["B_schedule_sha256"]
                 and np.array_equal(np.diff(offsets), [r["usable"] for r in c4_records]), "Original C008 samples changed")
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": expected_files,
                "optimizer": expected_opt, "gain": c4.GAIN, "initial_checkpoint_sha256": digest(incumbent),
                "normalization_sha256": p8["normalization_sha256"], "B_schedule_sha256": p8["B_schedule_sha256"],
                "old_source_records": records, "euroc_records": specs, "euroc_reference_quality": quality,
                "yt_prefix_exclusion": p8["yt_prefix_exclusion"], "sampling": {
                    "rule": "Zero-based step s and slot j; replace j%6==s%6. Process replaced slots in row-major order.",
                    "per_step_old_slots": 80, "per_step_euroc_slots": 16,
                    "euroc_track_assignment": "Round robin continuously over the three manifest tracks in order; independently draw one eligible window uniformly with replacement within each assigned track.",
                    "euroc_rng": "numpy.random.Generator(PCG64)", "euroc_rng_seed": EUROC_SEED,
                    "euroc_warmup": "First 2000 native samples used for causal VQF/accelerometer-estimator warmup only. Both complete training input history and release-point label must be outside warmup."
                },
                "euroc_source_qualification": {
                    "input_label_zh": "参考状态预去偏的外部训练源",
                    "input": "Frozen RIANN HDF5, 200 Hz grid; accelerometer m/s^2, gyroscope rad/s. The converter aligns measured IMU to reference-state timestamps by nearest neighbor and subtracts the reference-state gyro bias before writing the gyro fields.",
                    "target": "Published fused state attitude, opt_a...d in wxyz, world vertical +Z. OpenVINS documents a known original orientation accuracy issue for V1_01_easy; the registered labels are used as shipped in this one run.",
                    "conversion_source": "https://raw.githubusercontent.com/daniel-om-weber/riann_dev/435274141c08ab318c01be904de9754943b4feed/riann/prep/euroc.py",
                    "V1_01_easy_orientation_source": "https://docs.openvins.com/gs-datasets.html",
                    "interpretation": "This run jointly changes motion-domain coverage and uses a source whose gyro is reference-precorrected; it does not isolate the effect of motion diversity on raw-IMU generalization.",
                    "target_online_path": "All eight development domains retain the original sensor-only causal C003 path and C006 rest gate for the candidate, with no reference-state bias correction or EuRoC inputs at evaluation."
                },
                "training_target": p8["target"], "training_loss": p8["loss"], "replay_policy": p8["replay"],
                "C008_control": "Reuse the locked C008 scores after checking each row has exactly the fresh FullVQF/C003/C006 scores and scoring-frame identity.",
                "validation_records": p8["validation_records"], "scoring_masks_sha256": digest(c8.MASKS),
                "gate_traces_sha256": digest(c8.GATES), "scope": "One fit. Score the same 213 development records only. EuRoC is training only."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "started_at": started, "pid": os.getpid(), "protocol_sha256": protocol_sha, **extra})

    status("preparing")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "No CUDA device")
    device = torch.device("cuda")
    frozen, old_b = (torch.load(p, weights_only=True, map_location="cpu") for p in (incumbent, bpath))
    base.require(c4.tensor_hash({k: frozen[k] for k in ("mean", "std")}) == p8["normalization_sha256"]
                 and all(torch.equal(frozen[k], old_b[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, old_b["state_dict"][k]) for k, v in frozen["state_dict"].items() if not k.startswith("output.")),
                 "Normalization or shared control backbone changed")
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    base.require(frozen["model"] == "mamba_v1" and sum(p.numel() for p in core.parameters()) == 5595
                 and sum(p.numel() for p in core.output.parameters()) == 75, "B6 parameter count changed")
    core.eval().requires_grad_(False).to(device)
    tuned, control = deepcopy(core), deepcopy(core)
    control.load_state_dict(old_b["state_dict"])
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        canonical, target, draw_index, old_audits, causal, rotation = c8.point_windows(
            records, c4_records, schedule, offsets, releases, yt["train"], cfg)
    base.require(old_audits == prep8["records"] and array_sha(canonical, "<f4") == prep8["canonical_inputs_sha256"]
                 and array_sha(target, "<f4") == prep8["point_targets_sha256"], "C008 source windows or point targets changed")
    european = [euroc_windows(s, q, cfg) for s, q in zip(specs, quality, strict=True)]
    causal["EuRoC"] = european[0][6]
    rotation.extend((item[5], item[7]) for item in european)
    indices, mask, record_ord, pool, sample_path = mix_draws(schedule, offsets, releases, draw_index, european)
    all_x = np.concatenate([canonical, *(x[0] for x in european)])
    all_y = np.concatenate([target, *(x[1] for x in european)])
    np.testing.assert_array_equal(all_x[indices[~mask]], canonical[draw_index[~mask]])
    np.testing.assert_array_equal(all_y[indices[~mask]], target[draw_index[~mask]])
    old_audits = [{**a, "retained_draws": int(np.count_nonzero(record_ord == i))} for i, a in enumerate(old_audits)]
    euroc_audits = [{**x[5], "draws": int(np.count_nonzero(record_ord == len(records) + i))}
                    for i, x in enumerate(european)]
    base.require([x["draws"] for x in euroc_audits] == [854, 853, 853]
                 and sum(r["retained_draws"] for r in old_audits) == 12800, "Source mixing count changed")
    sampling = {"protocol_sha256": protocol_sha, "sampling_npz_sha256": digest(sample_path),
                "replacement_mask_sha256": array_sha(mask, "?"), "retained_slots": 12800, "euroc_slots": 2560,
                "distinct_drawn_euroc_windows": len(np.unique(pool)), "old_rows": old_audits,
                "new_rows": euroc_audits, "draw_matrix_shape": list(indices.shape),
                "record_ordinal": "Original 63 C008 source rows, then the three EuRoC manifest rows.",
                "retained_identity": "Exact row/slot, raw canonical input and point label equality to C008.",
                "mixed_canonical_pool_sha256": array_sha(all_x, "<f4"), "mixed_point_target_pool_sha256": array_sha(all_y, "<f4")}
    prep_path = HERE / "training-preparation.json"
    write(prep_path, sampling)
    samples, labels, draws, flags = (torch.from_numpy(x).to(device) for x in (all_x, all_y, indices, mask))
    source_row = torch.from_numpy(draw_index[0]).to(device)
    with torch.inference_mode():
        pure_loss = float((c8.prediction(core, samples[source_row], mean, std) - labels[source_row]).square().sum(dim=1).mean())
        mixed_loss = float((c8.prediction(core, samples[draws[0]], mean, std) - labels[draws[0]]).square().sum(dim=1).mean())
    base.require(pure_loss == t8["initial_loss"] == t8["archived_C004_B_first_loss"],
                 "Original C008 first batch does not reproduce")
    base.require(all(digest(PILOT / path) == sha for path, sha in expected_files.items())
                 and digest(sample_path) == sampling["sampling_npz_sha256"], "Frozen inputs changed before training")
    print("C009 source and draw lock passed", "original", pure_loss, "mixed", mixed_loss,
          "eligible EuRoC windows", [x["usable"] for x in euroc_audits], flush=True)
    status("fitting", optimizer_steps_planned=160, sampling_sha256=digest(sample_path))
    tuned, history, fit_time = fit(tuned, samples, labels, draws, flags, mean, std, mixed_loss)
    state = {k: v.detach().cpu().clone() for k, v in tuned.state_dict().items()}
    stage = checkpoint.with_suffix(".tmp")
    torch.save({**frozen, "state_dict": state, "protocol_sha256": protocol_sha, "sampling_sha256": digest(sample_path),
                "C009_fit": fit_time}, stage)
    stage.replace(checkpoint)
    steps = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(steps, history)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    base.require(all(torch.equal(frozen[k], saved[k]) for k in ("mean", "std")), "Input normalization changed")
    tuned.load_state_dict(saved["state_dict"])
    del samples, labels, draws, flags, all_x, all_y, canonical, target
    norms, prefix_path, rotation_path = (HERE / n for n in ("layer-norms.json", "prefix-audit.json", "equivariance.json"))
    write(norms, c8.layer_differences(frozen["state_dict"], saved["state_dict"]))
    write(prefix_path, c8.final_prefix(tuned, causal, mean, std, cfg))
    write(rotation_path, c8.rotated_audit(tuned, rotation, mean, std))
    training = {"fits": 1, "fit": fit_time, "original_C008_initial_loss": pure_loss, "initial_mixed_loss": mixed_loss,
                "first_step_loss": history[0]["loss"], "last_step_loss": history[-1]["loss"], "torch": str(torch.__version__),
                "gpu": torch.cuda.get_device_name(device), "artifacts_sha256": {path.name: digest(path) for path in
                (checkpoint, sample_path, prep_path, steps, norms, prefix_path, rotation_path)}}
    training_path = HERE / "training.json"
    write(training_path, training)
    status("scoring", fit_completed=True, training_sha256=digest(training_path))
    print("C009 fixed-step weights persisted; scoring the frozen 213 development records", flush=True)
    scored_at = perf_counter()
    result = replay(prior, r8, inventory, campaign, read(TRANSFER / "external-split.json"), core, control, tuned, mean, std, cfg)
    completed = stamp()
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_sha,
                   "training_sha256": digest(training_path), "scoring_masks_sha256": protocol["scoring_masks_sha256"],
                   "scoring_seconds": perf_counter() - scored_at, "total_seconds": perf_counter() - wall})
    results_path = HERE / "results.json"
    write(results_path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, result_sha256=digest(results_path))
    print("C009 coverage", len(result["sequences"]), "failures", result["failures"], "total_seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status_path = HERE / "process.json"
        if status_path.exists():
            old_status = read(status_path)
            if old_status["pid"] == os.getpid():
                write(status_path, {**old_status, "status": "failed", "phase": old_status["status"],
                                    "completed_at": stamp(), "failure": repr(error)})
        raise
