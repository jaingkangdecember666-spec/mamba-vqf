"""Evaluate one global B readout motion gate on the locked development panel."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C004 = CAMPAIGN / "C004"
REVIEW = CAMPAIGN / "review"
sys.path[:0] = [str(C004), str(REVIEW)]

import fit_readouts as c4
import numpy as np
import torch
from native_blocks import block_inputs
from optical_reference import extract
from optical_score import infer_canonical
from replay_native import metric
from scipy.io import loadmat


c3, c2, base, intrinsic = c4.c3, c4.c2, c4.base, c4.intrinsic
read, write, stamp, digest = c4.read, c4.write, c4.stamp, c4.digest
C001, C002, C003, DATA, PILOT, TRANSFER = c4.C001, c4.C002, c4.C003, c4.DATA, c4.PILOT, c4.TRANSFER
GAIN, SCALE = .25, .10
METHODS = ("full_vqf_deg", "c003_deg", "B_fixed_deg", "B_gated_deg")
RELATIVE_FLAG = "relative_level_and_repeated_samples"


def gate_for(history):
    w = history[..., 3:6].astype(np.float64) - history[..., 9:12].astype(np.float64)
    mean_square = np.einsum("ntc,ntc->n", w, w) / history.shape[1]
    return np.sqrt(mean_square), mean_square / (mean_square + SCALE**2)


def quantiles(v):
    return dict(zip(("min", "p05", "median", "p95", "max"),
                    np.quantile(v, (0., .05, .5, .95, 1.)).tolist(), strict=True))


def replay_controls(q_raw, q, end, release, history, mask, truth, model, layers, mean, std):
    delta = c4.evaluate_windows(model, layers, history, mean, std)
    r, gate = gate_for(history)
    base.require(np.isfinite(gate).all() and ((gate >= 0.) & (gate <= 1.)).all(), "Invalid physical gate")
    delta["B_gated"] = delta["B_fixed"] * gate[:, None]
    candidates = {k: (k, GAIN, False) for k in delta}

    def scores(mask_for_score):
        pure = intrinsic.score_attitudes(q_raw, truth, mask_for_score, q_raw[end], release, {}, {}, 1.)["full_vqf"]
        result = intrinsic.score_attitudes(q, truth, mask_for_score, q[end], release, delta, candidates, 1.)
        return {"full_vqf_deg": pure, "front_end_vqf_deg": result["full_vqf"],
                **{k + "_deg": result[k] for k in delta}}

    return scores, r, gate


def audit_gate_prefix():
    rng = np.random.default_rng(1729)
    probe = rng.normal(0., .2, (3500, 6))
    probe[:, 2] += 9.80665
    rows = []
    for hz in (90., 100., 200.):
        _, w, _, release = block_inputs(probe, hz)
        p = len(release) // 2
        cut = int(release[p]) + 1
        changed = probe.copy()
        changed[cut:] = rng.normal(0., 10., changed[cut:].shape)
        _, small, _, release2 = block_inputs(probe[:cut], hz)
        _, altered, _, release3 = block_inputs(changed, hz)
        np.testing.assert_array_equal(release2, release[:p + 1])
        np.testing.assert_array_equal(release3[:p + 1], release2)
        for original, truncated, perturbed in zip(gate_for(w), gate_for(small), gate_for(altered), strict=True):
            np.testing.assert_array_equal(original[:p + 1], truncated)
            np.testing.assert_array_equal(original[:p + 1], perturbed[:p + 1])
        rows.append({"native_rate_hz": hz, "release_index": int(release[p]), "same_gate_prefix": True})
    return rows


def gate_stats(mask, release, r, gate):
    lookup = np.searchsorted(release, np.flatnonzero(mask), side="right")
    applied = np.r_[0., gate][lookup]
    return {"updates": len(gate), "r_rms_rad_s_by_update": quantiles(r), "gate_by_update": quantiles(gate),
            "gate_by_native_scoring_frame": quantiles(applied), "mean_gate_scored": float(applied.mean()),
            "mean_effective_gain_scored": float(GAIN * applied.mean()),
            "scored_frames_at_gate_le_0p5": int((applied <= .5).sum())}


def row_for(spec, mask, key, scores, gate_summary, active, status):
    return {"dataset": spec["dataset"], "record": spec["record"], "group": spec["group"],
            "status": "scored", "reference_status": status, "native_frames": len(mask), "scored_frames": int(mask.sum()),
            "scoring_indices_sha256": intrinsic.mask_digest(mask), "mask_key": key, **scores,
            "bias_active_scored_frames": active, "gate": gate_summary}


def old_replay(row, archived, spec, old_source, medium, yt, split, c1p, calibration, raw_sha, old_masks,
               core, layers, mean, std, cfg):
    raw, dt, observed, truth, mask, poses = c3.sources(row, spec, old_source, medium, yt, split, c1p, calibration, raw_sha)
    np.testing.assert_array_equal(mask, old_masks[row["mask_key"]])
    clean = base.causal_fill(raw, observed)
    q_raw = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
    q, end, release, history, _, diag = c4.inertial_replay(raw, observed, dt, cfg)
    base.require(diag["first_active_native_index"] == row["bias"]["first_active_native_index"],
                 f"C003 bias activation changed: {row['record']}")
    scorer, r, gate = replay_controls(q_raw, q, end, release, history, mask, truth, core, layers, mean, std)
    scores = scorer(mask)
    errors = {"full_vqf_deg": scores["full_vqf_deg"] - archived["full_vqf_deg"],
              "c003_deg": scores["c003_deg"] - archived["c003_deg"],
              "B_fixed_deg": scores["B_fixed_deg"] - archived["B_deg"]}
    if max(map(abs, errors.values())) > 1e-6:
        raise ValueError(f"C004 control scores differ: {errors}")
    result = row_for(spec, mask, row["mask_key"], scores, gate_stats(mask, release, r, gate),
                     row["bias"]["active_scored_frames"], row["reference_status"])
    base.require(result["scoring_indices_sha256"] == archived["scoring_indices_sha256"] == row["scoring_indices_sha256"],
                 "Archived score indices changed")
    result["control_minus_c004_deg"] = errors
    detail = []
    for axis, where in poses.items():
        use = mask & where
        detail.append({"record": row["record"], "pose": axis, "scored_frames": int(use.sum()),
                       "scoring_indices_sha256": intrinsic.mask_digest(use), **scorer(use),
                       **gate_stats(use, release, r, gate)})
    return result, mask, (release, r, gate), detail


def optical_payload(spec, cfg):
    raw, hz, truth, mask, audit = extract(spec)
    observed = np.isfinite(raw).all(axis=1)
    if not mask.any():
        raise ValueError("No eligible optical suffix frames")
    base.require(np.all(observed[mask]), "Optical score mask includes invalid IMU")
    clean = base.causal_fill(raw, observed)
    raw_q, history, end, release = block_inputs(clean, hz)
    bias, active, diag, _ = c3.estimate(raw, observed, 1. / hz, cfg)
    if active.any():
        filtered = clean.copy()
        filtered[:, :3] -= bias
        q, history, changed_end, changed_release = block_inputs(filtered, hz)
        np.testing.assert_array_equal(changed_end, end)
        np.testing.assert_array_equal(changed_release, release)
    else:
        q = raw_q
    return raw_q, q, end, release, history, mask, truth, audit, active, diag


def repeat_anchor(raw_q, q, end, release, history, mask, ref, b6, bhead, mean, std):
    indices = np.flatnonzero(mask)
    z = np.array([0., 0., 1.])
    lookup = np.searchsorted(release, indices, side="right")
    raw_truth = raw_q[indices].apply(ref[indices])
    target = q[indices].apply(ref[indices])
    c003_delta = q[end].apply(infer_canonical((b6, mean, std, base.CAP), history))
    b_delta = q[end].apply(infer_canonical((bhead, mean, std, base.CAP), history))
    centered_gyro = history[:, :, 3:6].astype(float) - history[:, :, 9:12].astype(float)
    motion = (np.linalg.norm(centered_gyro, axis=2) ** 2).mean(axis=1)
    factor = motion / (motion + .01)
    results = {"full_vqf_deg": metric(np.broadcast_to(z, raw_truth.shape), raw_truth),
               "front_end_vqf_deg": metric(np.broadcast_to(z, target.shape), target)}
    for name, offset in (("c003_deg", c003_delta), ("B_fixed_deg", b_delta), ("B_gated_deg", b_delta * factor[:, None])):
        tip = z + GAIN * offset
        directions = np.vstack((z, tip / np.linalg.norm(tip, axis=1, keepdims=True)))[lookup]
        results[name] = metric(directions, target)
    return results


def optical_replay(spec, old_ref, anchor, core, bhead, layers, mean, std, cfg, pos):
    start = perf_counter()
    raw, q, end, release, history, mask, truth, audit, active, diag = optical_payload(spec, cfg)
    base.require(audit["scoring_frames"] == old_ref["scoring_frames"] and audit["native_frames"] == old_ref["native_frames"]
                 and audit["quality_flags"] == old_ref["quality_flags"]
                 and audit["applied_imu_minus_optical_rows"] == old_ref["applied_imu_minus_optical_rows"],
                 "Frozen optical geometry changed")
    np.testing.assert_allclose(audit["mount_xyzw"], old_ref["mount_xyzw"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(audit["world_up"], old_ref["world_up"], rtol=0, atol=1e-6)
    scorer, r, gate = replay_controls(raw.as_quat(scalar_first=True), q.as_quat(scalar_first=True),
                                      end, release, history, mask, truth, core, layers, mean, std)
    scores = scorer(mask)
    flags = audit["quality_flags"]
    quality = "verified" if not flags else "geometry_verified_relative_level" if flags == [RELATIVE_FLAG] else "quality_flagged"
    result = row_for(spec, mask, f"optical_{pos:03}", scores, gate_stats(mask, release, r, gate),
                     int((active & mask).sum()), quality)
    result.update({"reference_geometry": audit, "bias_first_active_native_index": diag["first_active_native_index"],
                   "elapsed_s": perf_counter() - start})
    repeat = None
    if anchor is not None:
        independent = repeat_anchor(raw, q, end, release, history, mask, truth, core, bhead, mean, std)
        archive_names = {"full_vqf_deg": "full_vqf", "front_end_vqf_deg": "front_end_vqf", "c003_deg": "C003"}
        from_archive = {key: independent[key] - anchor["rmse_deg"][field] for key, field in archive_names.items()}
        from_primary = {key: value - independent[key] for key, value in scores.items()}
        same_indices = result["scoring_indices_sha256"] == anchor["mask_sha256"] and result["scored_frames"] == anchor["scored_native_frames"]
        repeat = {"dataset": spec["dataset"], "record": spec["record"], "same_archived_mask": same_indices,
                  "independent_rmse_deg": independent, "independent_minus_archived_deg": from_archive,
                  "primary_minus_independent_deg": from_primary, "passed": bool(same_indices
                  and max(map(abs, (*from_archive.values(), *from_primary.values()))) < 2e-4)}
    return result, mask, (release, r, gate), repeat


def summarize(rows, denominator):
    valid = [r for r in rows if r["status"] == "scored"]
    means = {k: float(np.mean([r[k] for r in valid])) for k in METHODS} if valid else {}
    return {"denominator": denominator, "rated": len(valid), "failures": denominator - len(valid),
            "native_frames": sum(r.get("native_frames", 0) for r in rows),
            "scoring_frames": sum(r["scored_frames"] for r in valid), "macro_rmse_deg": means,
            "gate_wins_vs_pure": sum(r["B_gated_deg"] < r["full_vqf_deg"] for r in valid),
            "gate_wins_vs_C003": sum(r["B_gated_deg"] < r["c003_deg"] for r in valid),
            "gate_wins_vs_B_fixed": sum(r["B_gated_deg"] < r["B_fixed_deg"] for r in valid),
            "mean_effective_gain_per_record": float(np.mean([r["gate"]["mean_effective_gain_scored"] for r in valid])) if valid else None}


def main():
    started = stamp()
    protocol_path = HERE / "protocol.json"
    base.require(not protocol_path.exists(), "C005 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    inv = read(C001 / "inventory.json")
    old_c1 = read(C001 / "results.json")
    c1p = read(C001 / "protocol.json")
    c3r, c3p = (read(C003 / f) for f in ("results.json", "protocol.json"))
    c4r, c4p = (read(C004 / f) for f in ("results.json", "protocol.json"))
    c2p = read(C002 / "protocol.json")
    masks_path = C002 / "scoring-masks.npz"
    bpath = C004 / c4r["weights"]["B"]["weights"]
    base_path = PILOT / campaign["incumbent"]
    base.require(digest(bpath) == c4r["weights"]["B"]["weights_sha256"] and digest(base_path) == campaign["incumbent_sha256"]
                 and digest(masks_path) == c4r["shared_scoring_masks_sha256"] == c3r["shared_scoring_masks_sha256"]
                 and digest(C004 / "protocol.json") == c4r["protocol_sha256"] and digest(C003 / "protocol.json") == c3r["protocol_sha256"]
                 and digest(C003 / "online_bias.py") == c3p["estimator_sha256"]
                 and digest(C001 / "inventory.json") == old_c1["inventory_sha256"]
                 and not c4r["failures"] and len(c4r["sequences"]) == 85, "Frozen controls changed")
    opt = [r for r in inv["records"] if r["side"] == "development" and r["dataset"] in ("OxIOD", "RepoIMU")]
    base.require(sum(r["dataset"] == "OxIOD" for r in opt) == 66 and len(opt) == 128, "Optical panel changed")
    reference_audit = {(r["dataset"], r["record"]): r for r in read(REVIEW / "optical_development_calibration.json")}
    anchors = {(r["domain"], r["record"]): r for r in read(REVIEW / "optical_anchor_replay.json")}
    base.require(set(reference_audit) == {(r["dataset"], r["record"]) for r in opt}
                 and len(anchors) == 10 and set(anchors) <= set(reference_audit), "Reviewed optical panels differ")
    files = ("optical_reference.py", "native_blocks.py", "optical_score.py", "check_mount_pilots.py", "check_oxiod_timing.py",
             "mount_check.py", "optical_translation.py", "check_sources.py", "sample_rate_audit.json",
             "optical_development_calibration.json", "optical_anchor_replay.json")
    protocol = {"started_at": started, "pid": os.getpid(), "optimizer_steps": 0,
                "c003_controls_sha256": digest(C003 / "results.json"), "c004_controls_sha256": digest(C004 / "results.json"),
                "b6_sha256": digest(base_path), "B_weights_sha256": digest(bpath), "inventory_sha256": digest(C001 / "inventory.json"),
                "reference_sha256": {name: digest(REVIEW / name) for name in files},
                "independent_replay_code_sha256": digest(TRANSFER / "review/replay_native.py"),
                "legacy_masks_sha256": digest(masks_path),
                "scale_rad_s": SCALE, "fixed_gain": GAIN,
                "gate": "On each completed 64-block input, r=sqrt(mean(||gyroscope - VQF gyro bias||^2)); multiply the frozen B tangent residual by 0.25*r^2/(r^2+0.10^2) before native held-direction normalization.",
                "front_end": "Frozen C003 causal accelerometer bias. Optical fit/check prefixes define only the reference; score the reviewed suffix.",
                "optical_subset_rule": {"OxIOD_verified": "No quality flags.",
                                        "RepoIMU_geometry_verified": "No flags other than relative_level_and_repeated_samples. Pendulum remains relative reference."},
                "legacy_records": [[r["dataset"], r["record"]] for r in c3r["sequences"]],
                "optical_records": [[r["dataset"], r["record"]] for r in opt], "anchor_records": list(anchors)}
    write(protocol_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "protocol_sha256": digest(protocol_path), "status": "running"})
    write(HERE / "gate-prefix-audit.json", audit_gate_prefix())
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    old, new = (torch.load(path, weights_only=True, map_location="cpu") for path in (base_path, bpath))
    base.require(new["c004_arm"] == "B" and new["c004_gain"] == GAIN and old["seed"] == 1729
                 and all(torch.equal(old[m], new[m]) for m in ("mean", "std"))
                 and all(torch.equal(v, new["state_dict"][k]) for k, v in old["state_dict"].items() if not k.startswith("output.")),
                 "B6 and B frozen backbone or normalization differs")
    core = base.ResidualHead(old["model"])
    core.load_state_dict(old["state_dict"])
    core.eval().requires_grad_(False).to(device)
    bhead = deepcopy(core)
    bhead.load_state_dict(new["state_dict"])
    layers = {"c003": core.output, "B_fixed": bhead.output}
    mean, std = (old[k].to(device) for k in ("mean", "std"))
    cfg = c3.Settings(**c3p["thresholds"])
    specs = {(r["dataset"], r["record"]): r for r in inv["records"] if r["c001"] != "none"
             or (r["dataset"] == "IMU-Array" and r["side"] == "development")}
    c1_map = {(r["dataset"], r["record"]): r for r in old_c1["sequences"]}
    c4_map = {(r["dataset"], r["record"]): r for r in c4r["sequences"]}
    split = read(TRANSFER / "external-split.json")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    array_evidence = c2p["frozen_evidence"]
    base.require(digest(ref_path) == array_evidence["array_reference_sha256"], "Array reference changed")
    array = c2.reference_array(ref_path)
    rows, traces, masks, poses, repeats, failures = [], {}, {}, [], [], []
    with np.load(masks_path, allow_pickle=False) as old_masks, c2.c1.h5py.File(inv["yt"]["path"], "r") as yt:
        for row in c3r["sequences"]:
            key = row["dataset"], row["record"]
            try:
                result, mask, trace, details = old_replay(row, c4_map[key], specs[key], c1_map.get(key), medium, yt["valid"],
                    split, c1p, array, array_evidence["array_raw_sha256"], old_masks, core, layers, mean, std, cfg)
                masks[result["mask_key"]] = mask
                traces[result["mask_key"]] = np.column_stack(trace)
                rows.append(result)
                poses.extend(details)
                print(*key, *(result[m] for m in METHODS), flush=True)
            except (AssertionError, KeyError, RuntimeError, ValueError, OSError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("CONTROL FAILED", *key, repr(exc), flush=True)
    max_control = max(abs(v) for row in rows for v in row["control_minus_c004_deg"].values()) if rows else None
    write(HERE / "legacy-control-audit.json", {"completed_at": stamp(), "scored": len(rows), "failures": failures,
                                              "maximum_absolute_control_difference_deg": max_control,
                                              "per_record": rows})
    if failures:
        write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": stamp(),
                                      "status": "stopped_for_control_diagnosis", "failures": failures})
        raise ValueError("Control discrepancies must be diagnosed before starting optical replay")
    print("85 original controls passed; starting optical development only", max_control, flush=True)
    for pos, spec in enumerate(opt):
        key = spec["dataset"], spec["record"]
        old_ref = reference_audit[key]
        try:
            result, mask, trace, repeat = optical_replay(spec, old_ref, anchors.get(key), core, bhead, layers, mean, std, cfg, pos)
            masks[result["mask_key"]] = mask
            traces[result["mask_key"]] = np.column_stack(trace)
            rows.append(result)
            if repeat is not None:
                repeats.append(repeat)
            print(*key, result["scored_frames"], *(result[m] for m in METHODS), result["reference_status"], flush=True)
        except (AssertionError, KeyError, RuntimeError, ValueError, OSError, IndexError) as exc:
            fail = {"dataset": key[0], "record": key[1], "error": repr(exc)}
            failures.append(fail)
            rows.append({**fail, "group": spec["group"], "status": "unscorable",
                         "native_frames": old_ref.get("native_frames", 0), "reference_geometry": old_ref})
            print("OPTICAL FAILED", *key, repr(exc), flush=True)

    column = ("dataset", "record", "group", "status", "reference_status", "native_frames", "scored_frames", "mask_key",
              "scoring_indices_sha256", *METHODS, "front_end_vqf_deg", "bias_active_scored_frames")
    sequence_path, pose_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv"
    intrinsic.shice.save_csv(sequence_path, [{**{k: r.get(k, "") for k in column},
        "median_scored_gate": r.get("gate", {}).get("gate_by_native_scoring_frame", {}).get("median", ""),
        "mean_effective_gain": r.get("gate", {}).get("mean_effective_gain_scored", ""),
        "quality_flags": "|".join(r.get("reference_geometry", {}).get("quality_flags", [])),
        "error": r.get("error", "")} for r in rows])
    # Keep all gate statistics and native frame data in the companion results/masks files.
    intrinsic.shice.save_csv(pose_path, [{k: r[k] for k in ("record", "pose", "scored_frames", "scoring_indices_sha256", *METHODS,
                                                            "mean_gate_scored", "mean_effective_gain_scored")} for r in poses])
    for name, data in (("scoring-masks.npz", masks), ("gate-traces.npz", traces)):
        with (HERE / name).open("wb") as stream:
            np.savez_compressed(stream, **data)
    anchor_path = HERE / "anchor-replays.json"
    write(anchor_path, {"planned": 10, "reproduced": len(repeats), "passed": sum(r["passed"] for r in repeats),
                        "rows": repeats})
    summary = {}
    for domain in campaign["datasets"]:
        panel = [r for r in rows if r["dataset"] == domain]
        summary[domain] = summarize(panel, len(panel))
        summary[domain]["file_inventory_denominator"] = inv["record_denominators"][domain]["records"]
    ox = [r for r in rows if r["dataset"] == "OxIOD"]
    repo = [r for r in rows if r["dataset"] == "RepoIMU"]
    subsets = {"oxiod_verified": [r for r in ox if r.get("reference_status") == "verified"],
               "repo_stick": [r for r in repo if r["group"].startswith("TStick/")],
               "repo_pendulum": [r for r in repo if r["group"].startswith("Pendulum/")],
               "repo_geometry_verified": [r for r in repo if r.get("reference_status") in ("verified", "geometry_verified_relative_level")],
               "repo_stick_verified": [r for r in repo if r["group"].startswith("TStick/") and r.get("reference_status") == "verified"],
               "repo_pendulum_geometry_verified_relative": [r for r in repo if r.get("reference_status") == "geometry_verified_relative_level"]}
    completed = stamp()
    write(HERE / "results.json", {"started_at": started, "completed_at": completed, "pid": os.getpid(),
          "optimizer_steps": 0, "protocol_sha256": digest(protocol_path), "sequences_csv_sha256": digest(sequence_path),
          "pose_csv_sha256": digest(pose_path), "masks_sha256": digest(HERE / "scoring-masks.npz"),
          "gate_traces_sha256": digest(HERE / "gate-traces.npz"), "gate_trace_columns": ["release_index", "r_rms_rad_s", "gate_factor"],
          "anchor_replays_sha256": digest(anchor_path), "legacy_control_max_abs_error_deg": max_control,
          "summary": summary, "optical_subsets": {k: summarize(v, len(v)) for k, v in subsets.items()},
          "array_pose_rows": poses, "sequences": rows, "failures": failures,
          "anchor_check_count": sum(r["passed"] for r in repeats)})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": completed,
          "status": "complete" if not failures and len(repeats) == 10 and all(r["passed"] for r in repeats) else "complete_with_exceptions",
          "protocol_sha256": digest(protocol_path)})
    print(json.dumps({"summary": summary, "failed": failures, "anchors_passed": sum(r["passed"] for r in repeats)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
