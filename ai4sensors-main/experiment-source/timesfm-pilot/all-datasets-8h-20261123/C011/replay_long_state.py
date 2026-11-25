"""Select one causal world-horizontal EMA gain externally and replay the frozen development panel."""

from collections import Counter
from datetime import datetime
import hashlib
import inspect
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C006 = CAMPAIGN / "C006"
C005 = CAMPAIGN / "C005"
sys.path.insert(0, str(C006))

import replay_rest_gate as c6
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
from tilt_eval import normalize, rotate, tilt_error

from minimax import CERTIFICATE_TOL, GRID_INTERVALS, TIE_TOL, Curve, select_scalar

c5, c4, c3, c2 = c6.c5, c6.c4, c6.c3, c6.c2
base, intrinsic = c6.base, c6.intrinsic
read, write, digest, stamp = c6.read, c6.write, c6.digest, c6.stamp
C001, C002, C003, C004, DATA, PILOT, TRANSFER = c6.C001, c6.C002, c6.C003, c6.C004, c6.DATA, c6.PILOT, c6.TRANSFER
MASKS, GATES = C005 / "scoring-masks.npz", C006 / "gate-traces.npz"
UP = np.array([0., 0., 1.])
GRAVITY, TAU, CAP = 9.80665, 2., float(np.tan(np.deg2rad(6.)))
DOMAINS = ("BROAD", "Sassari", "YT_Server")
METHODS = ("full_vqf_deg", "C006_deg", "C011_deg", "physical_only_deg")


def array_sha(data, dtype):
    return hashlib.sha256(np.ascontiguousarray(data, dtype=dtype).tobytes()).hexdigest()


def front_end(raw, observed, dt, cfg, optical_hz=None):
    """The existing C006 front end, also returning the native acceleration supplied to VQF."""
    bias, active, diagnostic, _ = c3.estimate(raw, observed, dt, cfg)
    clean = base.causal_fill(raw, observed)
    modified = clean.copy() if active.any() else clean
    if active.any():
        modified[:, :3] -= bias
    if optical_hz is not None:
        q0, windows, at, release = c5.block_inputs(clean, optical_hz)
        q0 = q0.as_quat(scalar_first=True)
        complete = None
        if active.any():
            q, windows, later_at, later_release = c5.block_inputs(modified, optical_hz)
            q = q.as_quat(scalar_first=True)
    else:
        source = raw.copy()
        source[~observed] = np.nan
        q0, at, release, windows, _, complete = base.causal_inputs(source, 1. / dt)
        if active.any():
            q, later_at, later_release, windows, _, _ = base.causal_inputs(modified, 1. / dt)
    if active.any():
        np.testing.assert_array_equal(at, later_at)
        np.testing.assert_array_equal(release, later_release)
    else:
        q = q0
    base.require(np.all(at <= release), "Input-last index is after publication")
    return q0, q, at, release, windows, complete, modified[:, :3], diagnostic


def ema_at(q, acceleration, observed, dt, at):
    """Evaluate the zero-initialized EMA at the requested native indices."""
    native = np.flatnonzero(observed)
    state = np.zeros((len(at), 3), dtype=np.float64)
    if len(native):
        horizontal = rotate(normalize(q[native]), acceleration[native])[:, :2] / GRAVITY
        decay = float(np.exp(-dt / TAU))
        filtered = lfilter([1. - decay], [1., -decay], horizontal, axis=0)
        last_observed = np.searchsorted(native, at, side="right") - 1
        active = last_observed >= 0
        state[active, :2] = filtered[last_observed[active]]
    return state


def world_base(core, layer, window, q_last, mean, std):
    rest = c6.rest_flags(window)
    delta = c4.evaluate_windows(core, {"C006": layer}, window, mean, std)["C006"]
    delta[rest] = 0.
    return UP + rotate(normalize(q_last), delta * c4.GAIN), rest


def addition(state, rest, k):
    extra = k * state
    extra[rest] = 0.
    size = np.linalg.norm(extra, axis=1)
    saturated = size > CAP
    extra[saturated] *= (CAP / size[saturated])[:, None]
    return extra, saturated


def link_units(c006, state, rest, k):
    extra, saturated = addition(state, rest, k)
    return (normalize(c006), normalize(c006 + extra), normalize(UP + extra)), saturated


def training_source(spec, archive, source_releases, yt, mats, cfg, core, layer, mean, std):
    raw, observed, dt, score, truth, hashes = c4.train_raw(spec, yt, mats)
    _, q, at, release, history, complete, acc, bias = front_end(raw, observed, dt, cfg)
    eligible = complete & score[release]
    positions = np.flatnonzero(eligible)
    frame = intrinsic.build_frame(history[positions])
    use = positions[frame.valid]
    np.testing.assert_array_equal(release[use], source_releases)
    base.require(len(positions) == archive["eligible_windows"] and len(use) == archive["usable"]
                 and hashes == archive["raw_hashes"] and bias["active_native_frames"] == archive["active_native_frames"],
                 f"Frozen C004 source changed: {spec['record']}")
    branch, rest = world_base(core, layer, history[use], q[at[use]], mean, std)
    state = ema_at(q, acc, observed, dt, at[use])
    true_world = rotate(normalize(q[release[use]]), truth[release[use]])
    base.require(np.isfinite(branch).all() and np.isfinite(state).all() and np.isfinite(true_world).all()
                 and np.allclose(np.linalg.norm(true_world, axis=1), 1., rtol=0., atol=2e-7), "Invalid training directions")
    return (branch, state, rest, true_world), {**spec, "raw_hashes": hashes, "eligible_windows": len(positions), "usable": len(use),
            "C003_active_native_frames": bias["active_native_frames"],
            "release_native_sha256": array_sha(release[use], "<i8"), "input_last_native_sha256": array_sha(at[use], "<i8"),
            "point_reference_world_sha256": array_sha(true_world, "<f8"), "ema_at_input_last_sha256": array_sha(state, "<f8"),
            "rest_sha256": array_sha(rest, "?")}, at[use], (raw, observed, dt)


def train_coefficient(draws_by_record, stats):
    curves = []
    for domain in DOMAINS:
        records = [data for data, row in zip(draws_by_record, stats, strict=True) if row["domain"] == domain]
        sizes = np.array([len(r[0]) for r in records])
        branch, state, rest, truth = (np.concatenate([r[i] for r in records]) for i in range(4))
        weights = np.repeat(1. / (len(records) * sizes), sizes)
        curves.append(Curve(branch, np.where(rest[:, None], 0., state), truth, weights, CAP))
    coefficient, numerics = select_scalar(curves)
    by_record = []
    for data, row in zip(draws_by_record, stats, strict=True):
        branch, state, rest, truth = data
        _, capped = addition(state, rest, coefficient)
        (control, combined, physical), _ = link_units(branch, state, rest, coefficient)
        b, after, ablation = (float(np.mean(np.sum((v - truth)**2, axis=1))) for v in (control, combined, physical))
        by_record.append({**row, "C004_B_draws": len(rest), "rest_draws": int(rest.sum()),
                          "saturated_draws": int(capped.sum()), "C006_loss": b, "C011_loss": after,
                          "physical_only_loss": ablation, "C011_ratio": after / b})
    per_domain = {}
    for domain, curve in zip(DOMAINS, curves, strict=True):
        relevant = [r for r in by_record if r["domain"] == domain]
        means = {k: float(np.mean([r[k] for r in relevant])) for k in ("C006_loss", "C011_loss", "physical_only_loss")}
        ratio = means["C011_loss"] / means["C006_loss"]
        base.require(abs(ratio - curve.ratio(coefficient)) < 2e-11, "Scalar objective and direct normalized squared loss disagree")
        per_domain[domain] = {**means, "C011_to_C006_ratio": ratio, "records": len(relevant),
                              "ratio_at_zero": curve.ratio(0.), "ratio_at_one": curve.ratio(1.),
                              **{k: sum(row[k] for row in relevant) for k in ("C004_B_draws", "rest_draws", "saturated_draws")}}
    return coefficient, {"selection": numerics, "domains": per_domain, "records": by_record}


def prefix_checks(probes, cfg, core, layer, mean, std, k):
    results = []
    # Evaluate the recurrence against scalar updates and check exactly preserved gap states.
    q = np.tile([1., 0., 0., 0.], (8, 1))
    acc = np.column_stack((np.arange(8), -np.arange(8), np.full(8, GRAVITY)))
    valid = np.array([False, True, True, False, False, True, False, True])
    state = ema_at(q, acc, valid, .01, np.arange(len(q)))
    decay = np.exp(-.01 / TAU)
    manual = np.zeros_like(state)
    previous = np.zeros(3)
    for i in range(len(q)):
        if valid[i]:
            previous = decay * previous + (1. - decay) * (acc[i] * [1., 1., 0.]) / GRAVITY
        manual[i] = previous
    np.testing.assert_allclose(state, manual, rtol=0., atol=2e-17)
    np.testing.assert_array_equal(state[3:5], np.broadcast_to(state[2], (2, 3)))
    np.testing.assert_array_equal(state[6], state[5])
    for domain, (raw, observed, dt) in probes.items():
        orig = front_end(raw, observed, dt, cfg)
        _, q, at, release, history, _, acc, _ = orig
        center = len(release) // 2
        use = slice(center - 3, center + 1)
        end = release[center] + 1
        perturbed = raw.copy()
        perturbed[end:, :3] += [25., -10., 50.]
        perturbed[end:, 3:] -= 3.
        changed_valid = observed.copy()
        changed_valid[end::3] = False
        m = ema_at(q, acc, observed, dt, at[use])
        branch, rest = world_base(core, layer, history[use], q[at[use]], mean, std)
        directions, _ = link_units(branch, m, rest, k)
        for data, ob in ((raw[:end], observed[:end]), (perturbed, changed_valid)):
            _, q2, at2, release2, history2, _, acc2, _ = front_end(data, ob, dt, cfg)
            np.testing.assert_array_equal(at[:center + 1], at2[:center + 1])
            np.testing.assert_array_equal(release[:center + 1], release2[:center + 1])
            np.testing.assert_array_equal(history[:center + 1], history2[:center + 1])
            m2 = ema_at(q2, acc2, ob, dt, at2[use])
            np.testing.assert_array_equal(m, m2)
            b2, r2 = world_base(core, layer, history2[use], q2[at2[use]], mean, std)
            score2, _ = link_units(b2, m2, r2, k)
            np.testing.assert_array_equal(directions, score2)
        results.append({"domain": domain, "native_rate_hz": 1. / dt, "last_input_native": int(at[center]),
                        "last_release_native": int(release[center]), "exact_prefix_and_future_mutation": True})
    return {"probes": results, "zero_initial_state": bool(np.all(state[0] == 0.)),
            "missing_packet_holds_exactly": True, "scalar_recurrence_max_abs": float(np.max(np.abs(state - manual)))}


def installation_checks(probes, cfg, core, layer, mean, std, k):
    fixed_rotations = [*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                       Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))]
    results = []
    for domain, (raw, observed, dt) in probes.items():
        # Keeping the VQF world gauge fixed isolates sensor installation from heading initialization.
        _, q, at, release, history, _, acc, _ = front_end(raw, observed, dt, cfg)
        take = np.linspace(0, len(at) - 1, min(32, len(at)), dtype=int)
        at = at[take]
        engine = Rotation.from_quat(q, scalar_first=True)
        state = ema_at(q, acc, observed, dt, at)
        branch, rest = world_base(core, layer, history[take], q[at], mean, std)
        world, _ = link_units(branch, state, rest, k)
        for i, change in enumerate(fixed_rotations):
            new_q = (engine * change.inv()).as_quat(scalar_first=True)
            moved = history[take].copy()
            moved[..., :12] = change.apply(moved[..., :12].reshape(-1, 3)).reshape(moved.shape[0], moved.shape[1], 12)
            state2 = ema_at(new_q, change.apply(acc), observed, dt, at)
            branch2, rest2 = world_base(core, layer, moved, new_q[at], mean, std)
            np.testing.assert_array_equal(rest2, rest)
            world2, _ = link_units(branch2, state2, rest2, k)
            errors = np.asarray([tilt_error(a, b) for a, b in zip(world, world2, strict=True)])
            ema_error = float(np.max(np.abs(state2 - state)))
            base.require(ema_error <= 2e-12 and errors.max() <= .0001, "Fixed-gauge installation rotation audit failed")
            results.append({"domain": domain, "rotation": i, "windows": len(take), "ema_max_abs": ema_error,
                            "max_control_deg": float(errors[0].max()), "max_C011_deg": float(errors[1].max()),
                            "max_physical_only_deg": float(errors[2].max())})

    # Rerun the actual sensor path at fixed mounting rotations, allowing VQF's world heading gauge to change.
    dt = .005
    time = np.arange(2400) * dt
    moving = time >= 2.
    raw = np.zeros((len(time), 6))
    raw[:, 2] = GRAVITY
    ph = time[moving] - 2.
    raw[moving, 0] = 1.3 * np.sin(1.1 * ph)
    raw[moving, 1] = .7 * np.sin(2.2 * ph)
    raw[moving, 3] = .15 * np.sin(.6 * ph)
    raw[moving, 4] = .14 * np.cos(.9 * ph)
    raw[moving, 5] = .18 * np.cos(.35 * ph)
    observed = np.ones(len(raw), dtype=bool)
    observed[510:517] = False

    def body_replay(imu):
        _, q, at, release, history, _, acc, _ = front_end(imu, observed, dt, cfg)
        state = ema_at(q, acc, observed, dt, at)
        world, rest = world_base(core, layer, history, q[at], mean, std)
        links, _ = link_units(world, state, rest, k)
        inv = Rotation.from_quat(q[release], scalar_first=True).inv()
        return release, rest, [inv.apply(u) for u in (np.broadcast_to(UP, world.shape), *links)], inv.apply(state)

    release, rest, original, original_ema = body_replay(raw)
    full_path = []
    for i, change in enumerate(fixed_rotations):
        rotated = change.apply(raw.reshape(-1, 3)).reshape(raw.shape)
        release2, rest2, moved, moved_ema = body_replay(rotated)
        np.testing.assert_array_equal(release, release2)
        np.testing.assert_array_equal(rest, rest2)
        maxes = [float(tilt_error(a, change.inv().apply(b)).max()) for a, b in zip(original, moved, strict=True)]
        state_delta = float(np.max(np.abs(original_ema - change.inv().apply(moved_ema))))
        full_path.append({"rotation": i, "windows": len(release), "default_vqf_max_deg": maxes[0],
                          "C006_max_deg": maxes[1], "C011_max_deg": maxes[2], "physical_only_max_deg": maxes[3],
                          "ema_body_max_abs": state_delta})
    return {"fixed_world_gauge": results, "fixed_world_comparisons": sum(r["windows"] for r in results),
            "synthetic_raw_sensor_path": full_path,
            "synthetic_raw_max_C011_deg": max(r["C011_max_deg"] for r in full_path)}


def score_native(q0, q, at, release, refs, mask, unit, rest, capped):
    i = np.flatnonzero(mask)
    held = np.searchsorted(release, i, side="right")
    target = rotate(normalize(q[i]), refs[i])
    raw_target = rotate(normalize(q0[i]), refs[i])
    raw_angle = tilt_error(np.broadcast_to(UP, raw_target.shape), raw_target)
    values = [float(np.sqrt(np.mean(raw_angle**2)))]
    for direction in unit:
        angle = tilt_error(np.vstack((UP, direction))[held], target)
        values.append(float(np.sqrt(np.mean(angle**2))))
    base.require(np.isfinite(values).all(), "Nonfinite angular score")
    counts = {"rest_scored_frames": int(np.r_[False, rest][held].sum()),
              "saturated_scored_frames": int(np.r_[False, capped][held].sum()),
              "scored_frames_before_first_release": int(np.count_nonzero(held == 0))}
    return dict(zip(METHODS, values, strict=True)), counts


def coverage(rows):
    keys = ("scored_frames", "rest_scored_frames", "saturated_scored_frames", "scored_frames_before_first_release")
    return {key: sum(r[key] for r in rows) for key in keys}


def summary(rows, denominator):
    means = {m: float(np.mean([r[m] for r in rows])) for m in METHODS} if rows else {}
    return {"records": len(rows), "denominator": denominator, **coverage(rows), "macro_rmse_deg": means,
            "C011_delta_macro_deg": {m: means["C011_deg"] - means[m] for m in METHODS if m != "C011_deg"} if rows else {},
            "C011_wins_vs": {m: sum(r["C011_deg"] < r[m] for r in rows) for m in METHODS if m != "C011_deg"}}


def replay(old, inventory, campaign, split, core, layer, mean, std, cfg, k):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in old["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "Scoring inventory includes a nondevelopment record")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "Array reference changed")
    array = c2.reference_array(array_path)
    old_poses = {(r["record"], r["pose"]): r for r in old["array_sensor_poses"]}
    rows, poses, failures, traces = [], [], [], {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for prior in old["sequences"]:
            domain, name = key = prior["dataset"], prior["record"]
            try:
                if domain in ("OxIOD", "RepoIMU"):
                    raw, hz, truth, mask, qa = c5.extract(specs[key])
                    observed = np.isfinite(raw).all(axis=1)
                    base.require(mask.any() and observed[mask].all() and qa["quality_flags"] == prior["reference_quality_flags"],
                                 "Optical score or QA changed")
                    dt, optical_hz, axes = 1. / hz, hz, {}
                else:
                    raw, dt, observed, truth, mask, axes = c3.sources(map3[key], specs[key], map1.get(key), medium, yt["valid"],
                                                                     split, c1p, array, evidence["array_raw_sha256"])
                    optical_hz = None
                q0, q, at, release, window, _, acc, diag = front_end(raw, observed, dt, cfg, optical_hz)
                if optical_hz is None:
                    base.require(diag["first_active_native_index"] == map3[key]["bias"]["first_active_native_index"],
                                 "C003 bias activation changed")
                np.testing.assert_array_equal(mask, masks[prior["mask_key"]])
                base.require(intrinsic.mask_digest(mask) == prior["scoring_indices_sha256"]
                             and int(mask.sum()) == prior["scored_frames"], "Mask identity changed")
                c006, rest = world_base(core, layer, window, q[at], mean, std)
                previous = gates[prior["mask_key"]]
                np.testing.assert_array_equal(release, previous[:, 0])
                np.testing.assert_array_equal(rest, previous[:, 3])
                state = ema_at(q, acc, observed, dt, at)
                units, capped = link_units(c006, state, rest, k)
                np.testing.assert_array_equal(units[0][rest], units[1][rest])
                if k == 0.:
                    np.testing.assert_array_equal(units[0], units[1])
                scores, counts = score_native(q0, q, at, release, truth, mask, units, rest, capped)
                comparisons = {m: scores[m] - prior[o] for m, o in (("full_vqf_deg", "full_vqf_deg"), ("C006_deg", "B_rest_gated_deg"))}
                base.require(max(map(abs, comparisons.values())) <= 1e-6, f"C006/FullVQF controls changed {comparisons}")
                row = {k: prior[k] for k in ("dataset", "record", "group", "instrument", "scenario", "reference_status",
                                            "scored_frames", "scoring_indices_sha256", "mask_key")}
                row.update(scores, **counts, release_count=len(release), rest_releases=int(rest.sum()),
                           saturated_releases=int(capped.sum()), control_minus_C006_deg=comparisons, status="scored")
                detail = []
                for axis, pose in axes.items():
                    use = mask & pose
                    was = old_poses[name, axis]
                    pose_scores, pose_counts = score_native(q0, q, at, release, truth, use, units, rest, capped)
                    base.require(intrinsic.mask_digest(use) == was["scoring_indices_sha256"]
                                 and abs(pose_scores["C006_deg"] - was["B_rest_gated_deg"]) <= 1e-6
                                 and abs(pose_scores["full_vqf_deg"] - was["full_vqf_deg"]) <= 1e-6,
                                 "IMU-Array control or mask changed")
                    detail.append({"record": name, "pose": axis, "scored_frames": int(use.sum()),
                                   "scoring_indices_sha256": was["scoring_indices_sha256"], **pose_scores, **pose_counts})
                rows.append(row)
                poses.extend(detail)
                traces[prior["mask_key"]] = np.column_stack((at, release, state, rest, capped))
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, ValueError, OSError, RuntimeError, KeyError, IndexError) as exc:
                failures.append({"dataset": domain, "record": name, "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    seq_path, pose_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv"
    trace_path = HERE / "release-traces.npz"
    fields = ("dataset", "record", "group", "instrument", "scenario", "status", "reference_status", "mask_key",
              "scoring_indices_sha256", "scored_frames", "release_count", "rest_releases", "saturated_releases",
              "rest_scored_frames", "saturated_scored_frames", "scored_frames_before_first_release", *METHODS, "error")
    indexed = {(r["dataset"], r["record"]): r for r in rows}
    errors = {(r["dataset"], r["record"]): r["error"] for r in failures}
    csv_rows = []
    for r in old["sequences"]:
        key = r["dataset"], r["record"]
        actual = indexed.get(key, {})
        entry = {**r, **actual, "status": "scored" if actual else "failed", "error": errors.get(key, "")}
        csv_rows.append({f: actual.get(f, "") if f in METHODS else entry.get(f, "") for f in fields})
    intrinsic.shice.save_csv(seq_path, csv_rows)
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    with trace_path.open("wb") as stream:
        np.savez_compressed(stream, **traces)
    panels = {d: summary([r for r in rows if r["dataset"] == d], old["summary"][d]["denominator"])
              for d in campaign["datasets"]}
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_status"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_status"] in ("verified", "geometry_verified_relative_level")]}
    control_error = [abs(value) for r in rows for value in r["control_minus_C006_deg"].values()]
    return {"summary": panels, "optical_subsets": {key: summary(rs, old["optical_subsets"][key]["denominator"])
                                                   for key, rs in subsets.items()},
            "coverage": coverage(rows), "maximum_absolute_control_error_deg": max(control_error, default=None),
            "sequences_csv_sha256": digest(seq_path), "array_pose_csv_sha256": digest(pose_path) if poses else None,
            "release_traces_sha256": digest(trace_path), "release_trace_columns": ["input_last", "release",
                    "world_ema_x", "world_ema_y", "world_ema_z", "rest_gate", "saturated_addition"],
            "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    lock_path = HERE / "coefficient-lock.json"
    base.require(not protocol_path.exists() and not lock_path.exists(), "C011 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p4, r4, t4, prep4 = (read(C004 / n) for n in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    p6, prior = (read(C006 / n) for n in ("protocol.json", "results.json"))
    c3p = read(C003 / "protocol.json")
    inventory_path = C001 / "inventory.json"
    inventory = read(inventory_path)
    schedule_path = C004 / "schedules.npz"
    split_path = TRANSFER / "external-split.json"
    incumbent = PILOT / campaign["incumbent"]
    control_path = C004 / t4["arms"]["B"]["weights"]
    base.require(len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C006 / "protocol.json") == prior["protocol_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_path) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(MASKS) == prior["shared_masks_sha256"] and digest(GATES) == prior["gate_traces_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == c3p["estimator_sha256"]
                 and digest(split_path) == p4["source_split_sha256"] == inventory["prior_external_split_sha256"]
                 and digest(incumbent) == p4["base_checkpoint_sha256"] == p6["B6_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(control_path) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"],
                 "Frozen control/weights/sources changed")
    base.require(all(digest(c5.REVIEW / file) == v for file, v in p6["optical_reference_sha256"].items()),
                 "Frozen optical reference implementation changed")
    cfg = c3.Settings(**c3p["thresholds"])
    records, archived = p4["source_records"], prep4["records"]
    base.require(Counter(r["domain"] for r in records) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and all(all(row[k] == v for k, v in source.items()) for source, row in zip(records, archived, strict=True))
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "Original external source scope changed")
    with np.load(schedule_path, allow_pickle=False) as file:
        schedule, offsets, source_releases = (file[k] for k in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and np.array_equal(np.diff(offsets), [a["usable"] for a in archived])
                 and np.all((schedule >= 0) & (schedule < len(source_releases))), "C004-B schedule changed")
    source_paths = {Path(inspect.getfile(x)).resolve() for x in (c6, c5, c4, c3, c2, c2.c1, base, base.ResidualHead,
                    base.input_blocks, intrinsic, intrinsic.build_frame, rotate, select_scalar)}
    source_paths.update(c5.REVIEW / file for file in p6["optical_reference_sha256"])
    source_paths.update((Path(__file__).resolve(), CAMPAIGN / "protocol.json", C003 / "online_bias.py",
                         C004 / "protocol.json", C004 / "training-preparation.json", C004 / "training.json", schedule_path,
                         C006 / "protocol.json", C006 / "results.json", MASKS, GATES, inventory_path, split_path, incumbent,
                         control_path, C003 / "protocol.json", C003 / "results.json", C002 / "protocol.json",
                         C001 / "protocol.json", C001 / "results.json", c5.REVIEW / "long_state_source_audit.json",
                         c5.REVIEW / "mamba_history_audit.json"))
    frozen, fitted = (torch.load(p, weights_only=True, map_location="cpu") for p in (incumbent, control_path))
    base.require(fitted["c004_arm"] == "B" and fitted["c004_gain"] == p4["inference_gain"] == c4.GAIN == .25
                 and all(torch.equal(frozen[k], fitted[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, fitted["state_dict"][k]) for k, v in frozen["state_dict"].items() if not k.startswith("output."))
                 and c4.tensor_hash({k: fitted[k] for k in ("mean", "std")}) == p4["frozen_normalization_sha256"],
                 "C006 network or fixed normalization changed")
    core = base.ResidualHead(fitted["model"])
    core.load_state_dict(fitted["state_dict"])
    source_paths.add(Path(inspect.getfile(type(core.sequence))).resolve())
    files = {str(p.relative_to(PILOT)): digest(p) for p in sorted(source_paths)}
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": files,
                "neural_optimizer_steps": 0, "source_records": records,
                "yt_prefix_exclusion": p4["yt_prefix_exclusion"], "C004_B_schedule_sha256": digest(schedule_path),
                "C004_B_draw_matrix_sha256": array_sha(schedule, "<i8"), "C004_B_schedule_shape": list(schedule.shape),
                "C006_checkpoint_sha256": digest(control_path), "C006_gain": c4.GAIN, "C006_masks_sha256": digest(MASKS),
                "C006_gate_traces_sha256": digest(GATES), "tau_s": TAU, "gravity_m_s2": GRAVITY,
                "extra_cap_tan": CAP, "extra_cap_deg": 6., "k_range": [0., 1.],
                "alpha": "1-exp(-dt/2). At missing native samples m holds its previous state.",
                "branch": "After causal C003 bias correction, rotate each real native accelerometer to the corresponding VQF world. Remove vertical, divide by gravity, update a zero-initialized horizontal EMA. Sample at input-last and publish at release, holding the output until the next original release.",
                "application": "Keep C006 rest gate and gain. On nonrest releases add the independently norm-capped k*m to the pre-normalization C006 world direction. On rest set k*m to zero. Physical-only uses the same C003 VQF world up and EMA with the neural term removed.",
                "objective": "Original C004-B sampled point labels, expressed in C003 VQF world at the release. Per record average the squared unit-direction chord loss over its scheduled draws, including rest (added term zero in both arms). Average those means with equal record weight by source. Minimize the maximum of the three ratios of domain mean C011 loss to domain mean C006 loss.",
                "solver": {"method": "Endpoints and grid plus local refinement, then an interval lower-bound certificate",
                           "initial_grid_intervals": GRID_INTERVALS, "certificate_tolerance": CERTIFICATE_TOL,
                           "tie_tolerance_choose_smaller": TIE_TOL},
                "development_records": [[r["dataset"], r["record"]] for r in prior["sequences"]],
                "scope": "One scalar, frozen from the 63 external source records before development replay. No database/record-specific coefficient, EuRoC training, SHICE fitting, or final holdout scoring."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "started_at": started, "pid": os.getpid(), "protocol_sha256": protocol_sha, **extra})

    status("external_preparation")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "Original C006 GPU path is unavailable")
    device = torch.device("cuda")
    core.eval().requires_grad_(False).to(device)
    layer = core.output
    mean, std = fitted["mean"].to(device), fitted["std"].to(device)
    pool, stats, indices, probes = [], [], [], {}
    mats = {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for j, (spec, ref) in enumerate(zip(records, archived, strict=True)):
            bank, stat, at, probe = training_source(spec, ref, source_releases[offsets[j]:offsets[j + 1]],
                                                   yt["train"], mats, cfg, core, layer, mean, std)
            pool.append(bank)
            stats.append(stat)
            indices.append(at)
            probes.setdefault(spec["domain"], probe)
    all_windows = [np.concatenate([item[i] for item in pool]) for i in range(4)]
    draws = schedule.ravel()
    ordinal = np.searchsorted(offsets[1:], draws, side="right")
    drawn = [array[draws] for array in all_windows]
    by_source = [tuple(x[ordinal == j] for x in drawn) for j in range(len(records))]
    base.require(all(len(item[0]) == archived[j]["B_draws"] > 0 for j, item in enumerate(by_source)),
                 "A source draw count differs from C004-B")
    train_path = HERE / "training-windows.npz"
    with train_path.open("wb") as stream:
        np.savez_compressed(stream, original_B_schedule=schedule, record_ordinal=ordinal.reshape(schedule.shape),
                            release_native=source_releases[schedule], input_last_native=np.concatenate(indices)[schedule],
                            C006_direction=drawn[0], horizontal_ema=drawn[1], rest_gate=drawn[2], point_reference=drawn[3])
    status("external_scalar_selection", training_windows_sha256=digest(train_path))
    selection_started = perf_counter()
    k, fit = train_coefficient(by_source, stats)
    fit_path = HERE / "external-selection.json"
    write(fit_path, {**fit, "selection_seconds": perf_counter() - selection_started, "training_windows_sha256": digest(train_path)})
    base.require(all(digest(PILOT / path) == sha for path, sha in files.items()), "Locked source changed before publication")
    write(lock_path, {"k": k, "tau_s": TAU, "gravity_m_s2": GRAVITY, "cap_tan": CAP, "C006_gain": c4.GAIN,
                      "protocol_sha256": protocol_sha, "training_windows_sha256": digest(train_path),
                      "external_selection_sha256": digest(fit_path), "locked_at": stamp()})
    locked_sha = digest(lock_path)
    print("C011 gain locked", k, "domains", {d: v["C011_to_C006_ratio"] for d, v in fit["domains"].items()},
          "certificate", fit["selection"]["certificate_gap"], flush=True)
    prefix_path, mount_path = HERE / "prefix-audit.json", HERE / "installation-audit.json"
    write(prefix_path, prefix_checks(probes, cfg, core, layer, mean, std, k))
    write(mount_path, installation_checks(probes, cfg, core, layer, mean, std, k))
    del all_windows, pool, by_source, drawn, probes
    status("development_replay", coefficient_lock_sha256=locked_sha)
    scoring_start = perf_counter()
    result = replay(prior, inventory, campaign, read(split_path), core, layer, mean, std, cfg, k)
    completed = stamp()
    base.require(digest(lock_path) == locked_sha, "C011 coefficient file changed during replay")
    result.update({"started_at": started, "completed_at": completed, "k": k, "neural_optimizer_steps": 0,
                   "protocol_sha256": protocol_sha, "coefficient_lock_sha256": locked_sha, "masks_sha256": digest(MASKS),
                   "prefix_audit_sha256": digest(prefix_path), "installation_audit_sha256": digest(mount_path),
                   "scoring_seconds": perf_counter() - scoring_start, "total_seconds": perf_counter() - wall})
    path = HERE / "results.json"
    write(path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(path))
    print("C011 coverage", len(result["sequences"]), "failures", result["failures"], "seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = HERE / "process.json"
        if path.exists():
            prior = read(path)
            if prior["pid"] == os.getpid():
                write(path, {**prior, "phase": prior["status"], "status": "failed",
                             "completed_at": stamp(), "failure": repr(error)})
        raise
