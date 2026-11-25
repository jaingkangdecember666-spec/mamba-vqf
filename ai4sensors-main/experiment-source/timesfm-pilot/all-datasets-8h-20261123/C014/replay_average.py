"""Replay the two fixed world-horizontal residual averages with frozen C003/C006 branches."""

from copy import deepcopy
from datetime import datetime
import inspect
import os
from pathlib import Path
from shutil import copyfile
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C013 = CAMPAIGN / "C013"
sys.path[:0] = [str(C013), str(CAMPAIGN / "C011")]

import compare_seed as c13
import numpy as np
import torch
from replay_long_state import front_end
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from tilt_eval import normalize, rotate, tilt_error

c6, c5, c4, c3, c2 = c13.c6, c13.c5, c13.c4, c13.c3, c13.c2
base, intrinsic = c13.base, c13.intrinsic
read, write, digest, stamp = c13.read, c13.write, c13.digest, c13.stamp
C001, C002, C003, C004, C006 = c13.C001, c13.C002, c13.C003, c13.C004, c13.C006
DATA, PILOT, TRANSFER, MASKS, GATES = c13.DATA, c13.PILOT, c13.TRANSFER, c13.MASKS, c13.GATES
UP = np.array([0., 0., 1.])
NAMES = ("c003_deg", "C006_deg", "seed_C003_deg", "seed_C006_deg")
METHODS = ("full_vqf_deg", *NAMES, "A_deg", "B_deg")
AB = ("A_deg", "B_deg")


def correction_vectors(q_last, window, b6, c006, b7, seed_c006, mean, std):
    # A shares one set of B6 features, including on the frame that enables the C006 readout.
    b6_deltas = c4.evaluate_windows(b6, {"c003_deg": b6.output, "C006_deg": c006.output}, window, mean, std)
    b7_deltas = c4.evaluate_windows(b7, {"seed_C003_deg": b7.output, "seed_C006_deg": seed_c006.output}, window, mean, std)
    rest = c6.rest_flags(window)
    deltas = {**b6_deltas, **b7_deltas}
    for key in ("C006_deg", "seed_C006_deg"):
        deltas[key][rest] = 0.
    rotation = normalize(q_last)
    world = {name: rotate(rotation, deltas[name] * c4.GAIN) for name in NAMES}
    cap = float(np.tan(np.deg2rad(base.CAP)) * c4.GAIN)
    base.require(all(np.isfinite(v).all() and np.linalg.norm(v, axis=1).max() <= cap + 2e-7 for v in world.values()),
                 "A branch exceeded its original post-gain tangent bound")
    removed_z = max(float(np.max(np.abs(v[:, 2]))) for v in world.values())
    base.require(removed_z <= 1e-6, "A branch did not map into world horizontal")
    horizontal = {name: np.column_stack((v[:, :2], np.zeros(len(v)))) for name, v in world.items()}
    combined = {k: .5 * (horizontal["c003_deg"] + horizontal[m])
                for k, m in zip(AB, ("C006_deg", "seed_C006_deg"), strict=True)}
    for v in combined.values():
        np.testing.assert_array_equal(v[rest], .5 * horizontal["c003_deg"][rest])
    return {**world, **combined}, rest, removed_z


def synthetic(hz):
    t = np.arange(round(hz * 12.05)) / hz
    m = t >= 3.
    phase = t[m] - 3.
    raw = np.zeros((len(t), 6))
    raw[:, 2] = 9.80665
    raw[m, 0] = 1.2 * np.sin(1.1 * phase)
    raw[m, 1] = .8 * np.sin(2.3 * phase)
    raw[m, 3] = .12 * np.sin(.65 * phase)
    raw[m, 4] = .16 * np.cos(.91 * phase)
    raw[m, 5] = .19 * np.cos(.41 * phase)
    observed = np.ones(len(t), dtype=bool)
    missing = round(4.6 * hz)
    observed[missing:missing + 4] = False
    return raw, observed


def prefix_audit(cfg, b6, c006, b7, seed_c006, mean, std):
    probes = []
    for hz in (48., 100., 200.):
        raw, full_observed = synthetic(hz)
        optical_hz = hz if hz < 100. else None
        dt = 1. / hz
        _, q, at, released, history, _, _, _ = front_end(raw, full_observed, dt, cfg, optical_hz)
        cuts = (min(19, len(released) - 1), len(released) // 2)
        for index in cuts:
            start = index - 3
            selection = slice(start, index + 1)
            end = int(released[index]) + 1
            altered = raw.copy()
            altered[end:, :3] += [11., -17., 8.]
            altered[end:, 3:] -= 2.
            changed_observed = full_observed.copy()
            changed_observed[end::2] = False
            base_vectors, status, _ = correction_vectors(q[at[selection]], history[selection],
                                                          b6, c006, b7, seed_c006, mean, std)
            for imu, observed in ((raw[:end], full_observed[:end]), (altered, changed_observed)):
                _, q2, at2, r2, hist2, _, _, _ = front_end(imu, observed, dt, cfg, optical_hz)
                np.testing.assert_array_equal(at[:index + 1], at2[:index + 1])
                np.testing.assert_array_equal(released[:index + 1], r2[:index + 1])
                np.testing.assert_array_equal(history[:index + 1], hist2[:index + 1])
                candidate, rest, _ = correction_vectors(q2[at2[selection]], hist2[selection],
                                                          b6, c006, b7, seed_c006, mean, std)
                np.testing.assert_array_equal(status, rest)
                for name in (*NAMES, *AB):
                    np.testing.assert_array_equal(base_vectors[name], candidate[name])
            probes.append({"native_hz": hz, "input_last": int(at[index]), "release": int(released[index]),
                           "compared_releases": len(status), "rest_releases": int(status.sum()),
                           "exact_truncation_and_mutated_future": True})
    base.require(any(r["rest_releases"] > 0 for r in probes)
                 and any(r["rest_releases"] < r["compared_releases"] for r in probes), "Missing rest or motion prefix audit")
    return {"probes": probes, "compared_branches": [*NAMES, *AB]}


def rotation_audit(cfg, b6, c006, b7, seed_c006, mean, std):
    rotations = [*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))]
    raw, observed = synthetic(200.)
    dt = .005
    _, q, at, release, window, _, _, _ = front_end(raw, observed, dt, cfg)
    to_world = Rotation.from_quat(q[at], scalar_first=True)
    to_body_release = Rotation.from_quat(q[release], scalar_first=True).inv()
    vectors, rest, _ = correction_vectors(q[at], window, b6, c006, b7, seed_c006, mean, std)
    frozen = {k: normalize(UP + d) for k, d in vectors.items()}
    vertical = np.broadcast_to(UP, vectors["A_deg"].shape).copy()
    frozen_body = {k: to_body_release.apply(v) for k, v in {**frozen, "full_vqf_deg": vertical}.items()}
    exact, native = [], []
    for name, change in zip(("90deg_x", "90deg_y", "90deg_z", "63deg_(1,2,-1)"), rotations, strict=True):
        rotated_window = window.copy()
        rotated_window[..., :12] = change.apply(window[..., :12].reshape(-1, 3)).reshape(len(window), window.shape[1], 12)
        q_rot = (to_world * change.inv()).as_quat(scalar_first=True)
        rotated, r2, _ = correction_vectors(q_rot, rotated_window, b6, c006, b7, seed_c006, mean, std)
        np.testing.assert_array_equal(rest, r2)
        errors = {k: float(tilt_error(frozen[k], normalize(UP + d)).max()) for k, d in rotated.items()}
        base.require(max(errors.values()) <= .0001, f"Fixed-world installation equivariance failed for {name}")
        exact.append({"rotation": name, "windows": len(window), "rest_windows": int(rest.sum()), "maximum_errors_deg": errors})
        rotated_native = change.apply(raw.reshape(-1, 3)).reshape(raw.shape)
        _, r_q, r_at, r_release, r_windows, _, _, _ = front_end(rotated_native, observed, dt, cfg)
        np.testing.assert_array_equal(at, r_at)
        np.testing.assert_array_equal(release, r_release)
        transformed, r3, _ = correction_vectors(r_q[r_at], r_windows, b6, c006, b7, seed_c006, mean, std)
        np.testing.assert_array_equal(rest, r3)
        r_inv = Rotation.from_quat(r_q[r_release], scalar_first=True).inv()
        native_directions = {**{k: normalize(UP + d) for k, d in transformed.items()}, "full_vqf_deg": vertical}
        raw_errors = {k: float(tilt_error(frozen_body[k], change.inv().apply(r_inv.apply(v))).max())
                      for k, v in native_directions.items()}
        native.append({"rotation": name, "windows": len(window), "maximum_errors_deg": raw_errors})
    return {"fixed_gauge": exact, "fixed_gauge_comparisons": sum(r["windows"] for r in exact),
            "raw_sensor_replay": native, "raw_sensor_comparisons": sum(r["windows"] for r in native)}


def make_scorer(raw_q, q, at, release, window, ref, models, mean, std):
    vectors, rest, z = correction_vectors(q[at], window, *models, mean, std)
    held_units = {k: np.vstack((UP, normalize(UP + v))) for k, v in vectors.items()}

    def score(mask):
        indices = np.flatnonzero(mask)
        active = np.searchsorted(release, indices, side="right")
        raw_reference = rotate(normalize(raw_q[indices]), ref[indices])
        reference = rotate(normalize(q[indices]), ref[indices])
        pure = tilt_error(np.broadcast_to(UP, raw_reference.shape), raw_reference)
        result = {"full_vqf_deg": float(np.sqrt(np.mean(pure**2)))}
        for k, units in held_units.items():
            angle = tilt_error(units[active], reference)
            result[k] = float(np.sqrt(np.mean(angle**2)))
        base.require(np.isfinite(list(result.values())).all(), "Nonfinite development scores")
        return result, int(np.r_[False, rest][active].sum())

    return score, rest, z


def compare_control(measured, archived):
    gap = max(abs(measured[k] - archived[k]) for k in METHODS[:-2])
    base.require(gap <= 1e-6, f"Archived C013 controls changed by {gap}")
    return gap


def summarize(rows):
    scored = [r for r in rows if r["status"] == "scored"]
    macro = {k: float(np.mean([r[k] for r in scored])) for k in METHODS} if scored else {}
    return {"denominator": len(rows), "scored_records": len(scored), "scored_frames": sum(r["scored_frames"] for r in scored),
            "rest_scored_frames": sum(r["rest_scored_frames"] for r in scored), "macro_rmse_deg": macro,
            "delta_vs_default_deg": {k: macro[k] - macro["full_vqf_deg"] for k in AB} if scored else {},
            "B_minus_A_deg": macro["B_deg"] - macro["A_deg"] if scored else None,
            "wins_vs_default": {k: sum(r[k] < r["full_vqf_deg"] for r in scored) for k in AB},
            "B_wins_vs_A": sum(r["B_deg"] < r["A_deg"] for r in scored)}


def replay(prior, inventory, campaign, split, models, mean, std, cfg, masks_path):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "Replay does not match the frozen development scope")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(array_path)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    identity = ("dataset", "record", "group", "reference_QA", "instrument", "scenario", "reference_status",
                "mask_key", "scored_frames", "scoring_indices_sha256")
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(masks_path, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            d, name = key = old["dataset"], old["record"]
            try:
                if d in ("OxIOD", "RepoIMU"):
                    raw_q, q, at, release, windows, mask, ref, qa, _, _ = c5.optical_payload(specs[key], cfg)
                    raw_q, q = raw_q.as_quat(scalar_first=True), q.as_quat(scalar_first=True)
                    base.require(qa["quality_flags"] == old["reference_quality_flags"], "Optical reference classification changed")
                    axes = {}
                else:
                    raw, dt, observed, ref, mask, axes = c3.sources(map3[key], specs[key], map1.get(key), medium, yt["valid"],
                                                                     split, c1p, array, evidence["array_raw_sha256"])
                    clean = base.causal_fill(raw, observed)
                    raw_q = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
                    q, at, release, windows, _, bias = c4.inertial_replay(raw, observed, dt, cfg)
                    base.require(bias["first_active_native_index"] == map3[key]["bias"]["first_active_native_index"],
                                 "C003 bias activation changed")
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                base.require(intrinsic.mask_digest(mask) == old["scoring_indices_sha256"]
                             and int(mask.sum()) == old["scored_frames"] and np.all(at <= release), "Native scoring identity changed")
                score, rest, horizontal_z = make_scorer(raw_q, q, at, release, windows, ref, models, mean, std)
                frozen_gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, frozen_gate[:, 0])
                np.testing.assert_array_equal(rest, frozen_gate[:, 3])
                measures, rest_scored = score(mask)
                gap = compare_control(measures, old)
                base.require(rest_scored == old["rest_scored_frames"], "Rest-scored coverage changed")
                row = {**{k: old[k] for k in identity}, **measures, "status": "scored", "rest_scored_frames": rest_scored,
                       "maximum_control_error_deg": gap, "max_abs_removed_vertical_residual": horizontal_z,
                       "reference_quality_flags": old["reference_quality_flags"],
                       "B_minus_A_deg": measures["B_deg"] - measures["A_deg"]}
                detail = []
                for axis, subset in axes.items():
                    use = mask & subset
                    prev = old_poses[name, axis]
                    signature = intrinsic.mask_digest(use)
                    base.require(signature == prev["scoring_indices_sha256"] and int(use.sum()) == prev["scored_frames"],
                                 "IMU-Array pose scoring indices changed")
                    values, pose_rest = score(use)
                    g = compare_control(values, prev)
                    detail.append({"record": name, "pose": axis, "scored_frames": int(use.sum()),
                                   "scoring_indices_sha256": signature, **values, "rest_scored_frames": pose_rest,
                                   "maximum_control_error_deg": g, "B_minus_A_deg": values["B_deg"] - values["A_deg"]})
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": d, "record": name, "error": repr(exc)})
                rows.append({**{k: old[k] for k in identity}, "status": "failed", "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    csv_path, pose_path, group_path = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json"))
    fields = (*identity, "status", "rest_scored_frames", *METHODS, "B_minus_A_deg", "maximum_control_error_deg", "error")
    intrinsic.shice.save_csv(csv_path, [{k: row.get(k, "") for k in fields} for row in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    summary = {d: summarize([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]}
    groups = {field: {f"{d}/{g}": summarize([r for r in rows if r["dataset"] == d and r[field] == g])
                      for d, g in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("reference_QA", "instrument", "scenario")}
    write(group_path, groups)
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": summary, "optical_subsets": {k: summarize(rs) for k, rs in subsets.items()},
            "maximum_control_error_deg": max((r["maximum_control_error_deg"] for r in rows if r["status"] == "scored"), default=None),
            "max_abs_removed_vertical_residual": max((r["max_abs_removed_vertical_residual"] for r in rows if r["status"] == "scored"), default=None),
            "csv_sha256": digest(csv_path), "pose_csv_sha256": digest(pose_path) if poses else None,
            "group_summaries_sha256": digest(group_path), "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    base.require(not protocol_path.exists() and not (HERE / "results.json").exists(), "C014 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p13, t13, r13, s13, lineage = (read(C013 / n) for n in ("protocol.json", "training.json", "results.json",
                                                          "process.json", "seed-lineage.json"))
    p6, r6 = (read(C006 / n) for n in ("protocol.json", "results.json"))
    p3 = read(C003 / "protocol.json")
    inventory_path, split_path = C001 / "inventory.json", TRANSFER / "external-split.json"
    base.require(s13["status"] == "complete" and not r13["failures"] and len(r13["sequences"]) == 213
                 and digest(C013 / "results.json") == s13["results_sha256"]
                 and digest(C013 / "protocol.json") == r13["protocol_sha256"] == s13["protocol_sha256"]
                 and digest(C013 / "training.json") == r13["training_sha256"]
                 and digest(C013 / "sequences.csv") == r13["csv_sha256"]
                 and digest(C013 / "array-sensor-poses.csv") == r13["pose_csv_sha256"]
                 and all(digest(C013 / p) == sha for p, sha in t13["artifacts_sha256"].items())
                 and all(digest(PILOT / p) == sha for p, sha in p13["files_sha256"].items())
                 and digest(C006 / "protocol.json") == r6["protocol_sha256"]
                 and digest(MASKS) == r6["shared_masks_sha256"] == r13["scoring_masks_sha256"] == p13["scoring_masks_sha256"]
                 and digest(GATES) == r6["gate_traces_sha256"] == p13["gate_traces_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == p3["estimator_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and all(digest(c5.REVIEW / name) == sha for name, sha in p6["optical_reference_sha256"].items()),
                 "Archived controls, masks, gates or sources changed")
    identity = ("dataset", "record", "scored_frames", "mask_key", "scoring_indices_sha256")
    for old, current in zip(r6["sequences"], r13["sequences"], strict=True):
        base.require(all(old[k] == current[k] for k in identity)
                     and all(old[k] == current[k] for k in ("full_vqf_deg", "c003_deg"))
                     and old["B_rest_gated_deg"] == current["C006_deg"], "C013 differs from original C006 panel")
    w6 = PILOT / campaign["incumbent"]
    c6_weight = C004 / read(C004 / "training.json")["arms"]["B"]["weights"]
    w7 = c13.B7 / read(c13.B7 / "selection.json")["weights"]
    seed_weight = C013 / "seed-C006-readout.pt"
    checks = {w6: p6["B6_checkpoint_sha256"], c6_weight: p6["B_checkpoint_sha256"],
              w7: t13["B7_unchanged_sha256"], seed_weight: t13["artifacts_sha256"][seed_weight.name]}
    base.require(all(digest(path) == sha for path, sha in checks.items())
                 and digest(w6) == lineage["B6"]["checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(w7) == lineage["B7"]["checkpoint_sha256"], "Branch checkpoint identity changed")
    b6_data, c6_data, b7_data, seed_data = (torch.load(path, weights_only=True, map_location="cpu") for path in checks)
    base.require(b6_data["seed"] == 1729 and b7_data["seed"] == seed_data["seed"] == 314159
                 and c6_data["c004_arm"] == "B" and c6_data["c004_gain"] == c4.GAIN == .25
                 and seed_data["C013_optimizer_steps"] == 160 and seed_data["source_B7_sha256"] == digest(w7)
                 and seed_data["protocol_sha256"] == digest(C013 / "protocol.json")
                 and all(torch.equal(b6_data[k], state[k]) for state in (c6_data, b7_data, seed_data) for k in ("mean", "std")),
                 "Branch optimizer origin or fixed normalization changed")
    for a, b, historical_sha in ((b6_data, c6_data, lineage["B6_backbone_sha256"]),
                                 (b7_data, seed_data, lineage["B7_backbone_sha256"])):
        backbone = {k: v for k, v in a["state_dict"].items() if not k.startswith("output.")}
        base.require(c4.tensor_hash(backbone) == historical_sha
                     and all(torch.equal(v, b["state_dict"][k]) for k, v in backbone.items()),
                     "Readout branch no longer shares archived anchor features")
    base.require(lineage["B6_backbone_sha256"] != lineage["B7_backbone_sha256"], "B does not use the actual second seed")
    files = dict(p13["files_sha256"])
    extra = {Path(inspect.getfile(x)).resolve() for x in (front_end, normalize, rotate, tilt_error, c13, c4, c6, c5)}
    extra.update((Path(__file__).resolve(), CAMPAIGN / "protocol.json", inventory_path, split_path, MASKS, GATES,
                  C013 / "protocol.json", C013 / "results.json", C013 / "process.json", C013 / "training.json",
                  C013 / "sequences.csv", C013 / "array-sensor-poses.csv", C013 / "seed-lineage.json", C006 / "results.json",
                  *checks.keys()))
    files.update({str(p.relative_to(PILOT)): digest(p) for p in extra})
    records = [[r["dataset"], r["record"]] for r in r13["sequences"]]
    protocol = {"pid": os.getpid(), "started_at": started, "files_sha256": files, "trained_models": 0,
                "fitted_coefficients": 0, "weight_per_branch": .5, "gain_per_branch": c4.GAIN,
                "A": "0.5*seed1729 C003 + 0.5*seed1729 C006", "B": "0.5*seed1729 C003 + 0.5*C013 seed314159 C006",
                "common_front_end": "C003 causal accelerometer bias, original VQF, original native input-last and release schedule",
                "combination": "Each branch uses its original tangent projection and norm cap. Apply its 0.25 gain and rest gate, rotate to world at the common input-last. Take the world horizontal residuals at the common release, average 1/2 + 1/2, add world up and normalize once. Hold to the next release.",
                "rest": "The C006 second branch has zero residual on static releases. The C003 half is unmodified; the weights stay 1/2.",
                "compute": {"A_backbones": 1, "B_backbones": 2, "common_front_ends": 1, "B6_head_sharing": True, "B7_head_sharing": True},
                "development_records": records, "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "scope": "The original 213 development records, two prescribed averages and frozen control replays. No fit and no final holdout."}
    write(protocol_path, protocol)
    protocol_hash = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "pid": os.getpid(), "started_at": started,
                             "protocol_sha256": protocol_hash, **extra})

    status("auditing")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "Original control CUDA inference path unavailable")
    device = torch.device("cuda")
    b6 = base.ResidualHead(b6_data["model"])
    b6.load_state_dict(b6_data["state_dict"])
    b6.eval().requires_grad_(False).to(device)
    c006 = deepcopy(b6)
    c006.load_state_dict(c6_data["state_dict"])
    b7 = base.ResidualHead(b7_data["model"])
    b7.load_state_dict(b7_data["state_dict"])
    b7.eval().requires_grad_(False).to(device)
    seed_c006 = deepcopy(b7)
    seed_c006.load_state_dict(seed_data["state_dict"])
    models = (b6, c006, b7, seed_c006)
    mean, std = b6_data["mean"].to(device), b6_data["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    prefix_path, rotation_path = HERE / "prefix-audit.json", HERE / "rotation-audit.json"
    write(prefix_path, prefix_audit(cfg, *models, mean, std))
    write(rotation_path, rotation_audit(cfg, *models, mean, std))
    masks_replay = HERE / "scoring-masks.npz"
    copyfile(MASKS, masks_replay)
    base.require(digest(masks_replay) == protocol["scoring_masks_sha256"], "Copied score masks differ")
    status("development_replay")
    print("C014 fixed 1/2+1/2 rules and audits complete; replaying 213 records", flush=True)
    scoring_at = perf_counter()
    result = replay(r13, read(inventory_path), campaign, read(split_path), models, mean, std, cfg, masks_replay)
    completed = stamp()
    base.require(all(digest(path) == sha for path, sha in checks.items()), "Frozen weights changed during evaluation")
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_hash,
                   "optimizer_steps": 0, "fitted_coefficients": 0, "weight_per_branch": .5,
                   "scoring_masks_sha256": digest(masks_replay), "prefix_audit_sha256": digest(prefix_path),
                   "rotation_audit_sha256": digest(rotation_path), "scoring_seconds": perf_counter() - scoring_at,
                   "total_seconds": perf_counter() - wall})
    results_path = HERE / "results.json"
    write(results_path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(results_path))
    print("C014 coverage", len(result["sequences"]), "failures", result["failures"], "seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process_path = HERE / "process.json"
        if process_path.exists():
            status = read(process_path)
            if status["pid"] == os.getpid():
                write(process_path, {**status, "phase": status["status"], "status": "failed",
                                     "completed_at": stamp(), "failure": repr(error)})
        raise
