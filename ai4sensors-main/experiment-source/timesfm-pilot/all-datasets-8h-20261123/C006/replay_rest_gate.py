"""Replay the locked 213-record development panel with one VQF rest gate."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C005 = CAMPAIGN / "C005"
sys.path.insert(0, str(C005))

import replay_motion_gate as c5
import numpy as np
import torch
from scipy.io import loadmat


c4, c3, c2, base, intrinsic = c5.c4, c5.c3, c5.c2, c5.base, c5.intrinsic
read, write, stamp, digest = c5.read, c5.write, c5.stamp, c5.digest
C001, C002, C003, C004, DATA, PILOT, TRANSFER = c5.C001, c5.C002, c5.C003, c5.C004, c5.DATA, c5.PILOT, c5.TRANSFER
METHODS = ("full_vqf_deg", "c003_deg", "B_fixed_deg", "B_gated_deg", "B_rest_gated_deg")
GAIN = .25
LOW_RATE = .10
CONDITIONS = ("low_rate_nonrest", "confirmed_rest")


def rest_flags(history):
    flag = history[:, -1, 12]
    base.require(np.all((flag == 0.) | (flag == 1.)), "Last completed block contains an invalid VQF rest flag")
    return flag == 1.


def check_prefix():
    output = []
    for hz in (90., 100., 200.):
        signal = np.zeros((round(hz * 15), 6))
        signal[:, 2] = 9.80665
        signal[round(hz * 5):round(hz * 9), 3] = .30
        _, w, _, release = c5.block_inputs(signal, hz)
        flag = rest_flags(w)
        assert flag.any() and (~flag).any()
        at = len(release) // 2
        cutoff = int(release[at]) + 1
        changed = signal.copy()
        changed[cutoff:, 3:6] = .70
        _, short, _, release_short = c5.block_inputs(signal[:cutoff], hz)
        _, after, _, _ = c5.block_inputs(changed, hz)
        np.testing.assert_array_equal(release_short, release[:at + 1])
        np.testing.assert_array_equal(rest_flags(short), flag[:at + 1])
        np.testing.assert_array_equal(rest_flags(after)[:at + 1], flag[:at + 1])
        output.append({"native_hz": hz, "rest_updates": int(flag.sum()), "other_updates": int((~flag).sum()),
                       "prefix_release_index": int(release[at]), "same_rest_prefix": True})
    return output


def diagnose(mask, release, r, soft, rest, acceleration):
    lookup = np.searchsorted(release, np.flatnonzero(mask), side="right")
    count = np.bincount(lookup, minlength=len(release) + 1)[1:]
    conditions = {"low_rate_nonrest": (r <= LOW_RATE) & ~rest, "confirmed_rest": rest}
    details = {}
    for name, valid in conditions.items():
        frames = int(count[valid].sum())
        n = int(valid.sum())
        hard = (~rest[valid]).astype(np.float64)
        weights = count[valid]
        sums = {
            "soft_gain_sum": float(np.dot(weights, GAIN * soft[valid])),
            "hard_gain_sum": float(np.dot(weights, GAIN * hard)),
            "acceleration_variation_sum_m_s2": float(np.dot(weights, acceleration[valid])),
            "soft_nonzero_frames": int(weights[soft[valid] > 0.].sum()),
            "hard_nonzero_frames": int(weights[hard > 0.].sum()),
        }
        details[name] = {"updates": n, "scored_frames": frames, **sums,
                         "mean_soft_gain": sums["soft_gain_sum"] / frames if frames else None,
                         "mean_hard_gain": sums["hard_gain_sum"] / frames if frames else None,
                         "mean_acceleration_variation_m_s2": sums["acceleration_variation_sum_m_s2"] / frames if frames else None,
                         "acceleration_variation_by_update_m_s2": c5.quantiles(acceleration[valid]) if n else None}
    active_rest = int(count[rest].sum())
    return {"updates": len(release), "rest_updates": int(rest.sum()), "rest_update_fraction": float(rest.mean()),
            "scored_frames_with_rest": active_rest, "rest_scored_fraction": active_rest / int(mask.sum()),
            "scored_frames_before_first_update": int(np.count_nonzero(lookup == 0)),
            "mean_effective_rest_gain_on_scored_frames": float(GAIN * count[~rest].sum() / mask.sum()),
            "conditions": details}


def scorer(q_raw, q, at, release, windows, ref, core, layers, mean, std):
    delta = c4.evaluate_windows(core, layers, windows, mean, std)
    r, soft = c5.gate_for(windows)
    rest = rest_flags(windows)
    acc = windows[..., :3].astype(np.float64)
    change = acc - acc.mean(axis=1, keepdims=True)
    variation = np.sqrt(np.einsum("ntc,ntc->n", change, change) / windows.shape[1])
    delta["B_gated"] = delta["B_fixed"] * soft[:, None]
    delta["B_rest_gated"] = np.where(rest[:, None], 0., delta["B_fixed"])
    candidates = {name: (name, GAIN, False) for name in delta}

    def score(mask):
        default = intrinsic.score_attitudes(q_raw, ref, mask, q_raw[at], release, {}, {}, 1.)["full_vqf"]
        fitted = intrinsic.score_attitudes(q, ref, mask, q[at], release, delta, candidates, 1.)
        output = {"full_vqf_deg": default, "front_end_vqf_deg": fitted["full_vqf"],
                  **{name + "_deg": fitted[name] for name in delta}}
        base.require(np.isfinite(list(output.values())).all(), "Nonfinite inclination RMSE")
        return output

    return score, (r, soft, rest, variation)


def classify(row):
    domain, name = row["dataset"], row["record"]
    scenario = row["group"]
    if domain == "OxIOD":
        tokens = name.split("/")
        instrument = tokens[1] if tokens[0] == "multi devices" else "unclassified_phone"
    elif domain == "RepoIMU":
        instrument = name.split("/")[0]
    elif domain == "SHICE":
        instrument, scenario = name.split("-")
    elif domain == "Sassari":
        instrument = "".join(x for x in name.split("/")[1] if x.isalpha())
    elif domain == "IMU-Array":
        instrument = "array_bank_" + name.rsplit("/", 1)[-1].split("_")[1]
    elif domain == "YT_Server":
        instrument = "filename_code_" + name.split("_")[1]
    else:
        instrument = {"TUM-VI": "TUM_VI_rig", "BROAD": "BROAD_rig"}[domain]
    return {"reference_QA": row["reference_status"], "instrument": instrument, "scenario": scenario}


def evaluate(old, spec, contexts):
    c3_row, c1_row, medium, yt, split, c1p, array, hashes, masks, core, layers, mean, std, cfg = contexts
    domain = old["dataset"]
    if domain in ("OxIOD", "RepoIMU"):
        raw, fit, at, release, windows, mask, ref, geometry, _, _ = c5.optical_payload(spec, cfg)
        prior = old["reference_geometry"]
        base.require((geometry["applied_imu_minus_optical_rows"], geometry["quality_flags"]) ==
                     (prior["applied_imu_minus_optical_rows"], prior["quality_flags"]), "C005 optical calibration changed")
        for key in ("mount_xyzw", "world_up"):
            np.testing.assert_allclose(geometry[key], prior[key], rtol=0., atol=1e-6)
        q_raw, q = raw.as_quat(scalar_first=True), fit.as_quat(scalar_first=True)
        poses = {}
    else:
        raw, dt, observed, ref, mask, poses = c3.sources(c3_row, spec, c1_row, medium, yt, split, c1p, array, hashes)
        clean = base.causal_fill(raw, observed)
        q_raw = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
        q, at, release, windows, _, dia = c4.inertial_replay(raw, observed, dt, cfg)
        base.require(dia["first_active_native_index"] == c3_row["bias"]["first_active_native_index"], "C003 front end changed")
    np.testing.assert_array_equal(mask, masks[old["mask_key"]])
    base.require(old["scoring_indices_sha256"] == intrinsic.mask_digest(mask), "C005 scoring indices changed")
    scoring, values = scorer(q_raw, q, at, release, windows, ref, core, layers, mean, std)
    scores = scoring(mask)
    errors = {key: scores[key] - old[key] for key in METHODS[:-1]}
    base.require(max(map(abs, errors.values())) <= 1e-6, f"C005 controls did not reproduce: {errors}")
    identity = ("dataset", "record", "group", "status", "reference_status", "native_frames", "scored_frames",
                "mask_key", "scoring_indices_sha256")
    diag = diagnose(mask, release, *values)
    result = {**{k: old[k] for k in identity}, **classify(old), **scores, "rest": diag, "control_minus_C005_deg": errors}
    if "reference_geometry" in old:
        result["reference_quality_flags"] = old["reference_geometry"]["quality_flags"]
    detail = []
    for axis, pose_mask in poses.items():
        use = mask & pose_mask
        detail.append({"record": old["record"], "pose": axis, "scored_frames": int(use.sum()),
                       "scoring_indices_sha256": intrinsic.mask_digest(use), **scoring(use),
                       "rest": diagnose(use, release, *values)})
    return result, np.column_stack((release, *values)), detail


def summarize(rows):
    valid = [r for r in rows if r["status"] == "scored"]
    result = {"denominator": len(rows), "scored_records": len(valid)}
    if not valid:
        return result
    frames = sum(r["scored_frames"] for r in valid)
    scores = {m: float(np.mean([r[m] for r in valid])) for m in METHODS}
    coverage = sum(r["rest"]["scored_frames_with_rest"] for r in valid)
    details = {}
    for k in CONDITIONS:
        chosen = [r["rest"]["conditions"][k] for r in valid]
        count = sum(x["scored_frames"] for x in chosen)
        details[k] = {"updates": sum(x["updates"] for x in chosen), "scored_frames": count,
                      "mean_soft_gain": sum(x["soft_gain_sum"] for x in chosen) / count if count else None,
                      "mean_hard_gain": sum(x["hard_gain_sum"] for x in chosen) / count if count else None,
                      "soft_nonzero_frames": sum(x["soft_nonzero_frames"] for x in chosen),
                      "hard_nonzero_frames": sum(x["hard_nonzero_frames"] for x in chosen),
                      "mean_acceleration_variation_m_s2": sum(x["acceleration_variation_sum_m_s2"] for x in chosen) / count if count else None}
    return {**result, "scored_frames": frames, "macro_rmse_deg": scores,
            "rest_scored_frames": coverage, "rest_scored_fraction": coverage / frames,
            "rest_wins_by_control": {m: sum(r["B_rest_gated_deg"] < r[m] for r in valid) for m in METHODS[:-1]},
            "conditions": details}


def main():
    started = stamp()
    out_protocol = HERE / "protocol.json"
    base.require(not out_protocol.exists(), "C006 already started")
    global_protocol = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(global_protocol["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(global_protocol["end"]), "Outside campaign window")
    prior = read(C005 / "results.json")
    previous_protocol = read(C005 / "protocol.json")
    inventory_path = C001 / "inventory.json"
    inventory = read(inventory_path)
    masks_path = C005 / "scoring-masks.npz"
    base.require(len(prior["sequences"]) == 213 and not prior["failures"] and
                 digest(C005 / "protocol.json") == prior["protocol_sha256"]
                 and digest(masks_path) == prior["masks_sha256"]
                 and digest(inventory_path) == previous_protocol["inventory_sha256"], "C005 protocol, mask or inventory changed")
    base.require(all(digest(c5.REVIEW / path) == value for path, value in previous_protocol["reference_sha256"].items()),
                 "A frozen optical reference dependency changed")
    c1r, c1p = (read(C001 / f) for f in ("results.json", "protocol.json"))
    c3r, c3p = (read(C003 / f) for f in ("results.json", "protocol.json"))
    c2p = read(C002 / "protocol.json")
    c4r = read(C004 / "results.json")
    hashes = c2p["frozen_evidence"]
    weights = PILOT / global_protocol["incumbent"]
    bpath = C004 / c4r["weights"]["B"]["weights"]
    base.require(digest(weights) == previous_protocol["b6_sha256"] == global_protocol["incumbent_sha256"]
                 and digest(bpath) == previous_protocol["B_weights_sha256"] == c4r["weights"]["B"]["weights_sha256"]
                 and digest(C003 / "results.json") == previous_protocol["c003_controls_sha256"]
                 and digest(C003 / "online_bias.py") == c3p["estimator_sha256"], "Frozen front end or B weights changed")
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if r["c001"] != "none"
             or (r["dataset"] in ("OxIOD", "RepoIMU", "IMU-Array") and r["side"] == "development")}
    records = [(r["dataset"], r["record"]) for r in prior["sequences"]]
    base.require(len(set(records)) == 213 and set(records) == set(specs), "Unexpected development inventory")
    lock = {"started_at": started, "pid": os.getpid(), "optimizer_steps": 0,
            "C005_results_sha256": digest(C005 / "results.json"), "C005_native_masks_sha256": digest(masks_path),
            "B_checkpoint_sha256": digest(bpath), "B6_checkpoint_sha256": digest(weights),
            "C003_bias_protocol_sha256": digest(C003 / "protocol.json"), "development_inventory_sha256": digest(inventory_path),
            "optical_reference_sha256": previous_protocol["reference_sha256"], "records": records,
            "new_policy": "At each original release, read restDetected in the newest completed feature block on the C003 bias-corrected default 6D VQF path. A true flag issues zero residual; otherwise the same frozen B residual is multiplied by 0.25. Hold until the next original release.",
            "diagnostic": "Low-rate nonrest means last-block rest flag false and C005 r<=0.1 rad/s. Acceleration variation is vector RMS around the 64-block mean, from the shared feature input. Condition statistics use the held release for each native scoring frame; no effect on scores or gates.",
            "reference_qualification": "Retain all optical QA flags and relative pendulum reference; YT label remains provisional.",
            "oversize_diagnostic_threshold_deg": 45.}
    write(out_protocol, lock)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "status": "running",
                                  "protocol_sha256": digest(out_protocol)})
    write(HERE / "prefix-audit.json", check_prefix())
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    old, new = (torch.load(path, weights_only=True, map_location="cpu") for path in (weights, bpath))
    base.require(new["c004_arm"] == "B" and new["c004_gain"] == GAIN and all(torch.equal(old[k], new[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, new["state_dict"][k]) for k, v in old["state_dict"].items() if not k.startswith("output.")),
                 "Model normalization or shared backbone changed")
    core = base.ResidualHead(old["model"])
    core.load_state_dict(old["state_dict"])
    core.eval().requires_grad_(False).to(device)
    bhead = deepcopy(core)
    bhead.load_state_dict(new["state_dict"])
    layers = {"c003": core.output, "B_fixed": bhead.output}
    mean, std = old["mean"].to(device), old["std"].to(device)
    cfg = c3.Settings(**c3p["thresholds"])
    c3_rows = {(r["dataset"], r["record"]): r for r in c3r["sequences"]}
    c1_rows = {(r["dataset"], r["record"]): r for r in c1r["sequences"]}
    split_path = TRANSFER / "external-split.json"
    base.require(digest(split_path) == inventory["prior_external_split_sha256"], "External split changed")
    split = read(split_path)
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref) == hashes["array_reference_sha256"], "Array source changed")
    array = c2.reference_array(ref)
    rows, failures, poses, traces = [], [], [], {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(masks_path, allow_pickle=False) as masks:
        for row in prior["sequences"]:
            key = row["dataset"], row["record"]
            try:
                result, trace, axis = evaluate(row, specs[key], (c3_rows.get(key), c1_rows.get(key), medium, yt["valid"], split,
                                                c1p, array, hashes["array_raw_sha256"], masks, core, layers, mean, std, cfg))
                traces[row["mask_key"]] = trace
                rows.append(result)
                poses.extend(axis)
                print(*key, result["scored_frames"], *(result[m] for m in METHODS),
                      "rest", result["rest"]["rest_scored_fraction"], flush=True)
            except (AssertionError, KeyError, RuntimeError, ValueError, OSError, IndexError) as exc:
                failure = {"dataset": key[0], "record": key[1], "error": repr(exc)}
                rows.append({**failure, "status": "failed", **classify(row)})
                failures.append(failure)
                print("FAILED", *key, repr(exc), flush=True)
    csv_path = HERE / "sequences.csv"
    fields = ("dataset", "record", "group", "status", "reference_status", "native_frames", "scored_frames", "mask_key",
              "scoring_indices_sha256", *METHODS, "front_end_vqf_deg", "instrument", "scenario")
    intrinsic.shice.save_csv(csv_path, [{**{k: r.get(k, "") for k in fields},
         "rest_updates": r.get("rest", {}).get("rest_updates", ""),
         "rest_scored_frames": r.get("rest", {}).get("scored_frames_with_rest", ""),
         "rest_scored_fraction": r.get("rest", {}).get("rest_scored_fraction", ""),
         "quality_flags": "|".join(r.get("reference_quality_flags", [])), "error": r.get("error", "")} for r in rows])
    by_domain = {domain: summarize([r for r in rows if r["dataset"] == domain]) for domain in global_protocol["datasets"]}
    optical = [r for r in rows if r["dataset"] in ("OxIOD", "RepoIMU")]
    subgroups = {"Oxford_verified": [r for r in optical if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
                 "Repo_stick": [r for r in optical if r["instrument"] == "TStick"],
                 "Repo_pendulum": [r for r in optical if r["instrument"] == "Pendulum"],
                 "Repo_geometry_verified": [r for r in optical if r["dataset"] == "RepoIMU" and
                                            r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    groups = {}
    for k in ("reference_QA", "instrument", "scenario"):
        groups[k] = {}
        for domain, label in sorted({(r["dataset"], r[k]) for r in rows}):
            groups[k][f"{domain}/{label}"] = summarize([r for r in rows if r["dataset"] == domain and r[k] == label])
    group_path, pose_path = HERE / "group-summaries.json", HERE / "array-sensor-poses.csv"
    write(group_path, groups)
    intrinsic.shice.save_csv(pose_path, [{**{k: r[k] for k in ("record", "pose", "scored_frames", "scoring_indices_sha256", *METHODS)},
                                         "rest_scored_fraction": r["rest"]["rest_scored_fraction"]} for r in poses])
    traces_path = HERE / "gate-traces.npz"
    with traces_path.open("wb") as stream:
        np.savez_compressed(stream, **traces)
    control = max((abs(v) for r in rows for v in r.get("control_minus_C005_deg", {}).values()), default=None)
    outliers = [{k: r[k] for k in ("record", "reference_quality_flags", *METHODS)}
                for r in optical if r["status"] == "scored" and r["instrument"] == "TStick" and r["full_vqf_deg"] > 45.]
    complete = stamp()
    write(HERE / "results.json", {"started_at": started, "completed_at": complete, "pid": os.getpid(),
          "optimizer_steps": 0, "protocol_sha256": digest(out_protocol), "shared_masks_sha256": digest(masks_path),
          "shared_masks_path": str(masks_path), "csv_sha256": digest(csv_path), "array_pose_csv_sha256": digest(pose_path),
          "group_summaries_sha256": digest(group_path), "gate_traces_sha256": digest(traces_path),
          "gate_trace_columns": ["release_index", "gyro_rms_rad_s", "soft_factor", "rest_flag", "acc_rms_variation_m_s2"],
          "prefix_audit_sha256": digest(HERE / "prefix-audit.json"), "control_max_abs_delta_deg": control,
          "summary": by_domain, "optical_subsets": {k: summarize(v) for k, v in subgroups.items()},
          "oversize_stick_reference_diagnostics": outliers, "array_sensor_poses": poses,
          "sequences": rows, "failures": failures})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": complete,
                                  "status": "complete" if not failures else "partial", "protocol_sha256": digest(out_protocol)})
    print(json.dumps({"summary": by_domain, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
