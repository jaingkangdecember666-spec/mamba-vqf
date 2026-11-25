"""C017: one externally fitted stable specific-force correction around C014-B."""
from pathlib import Path
from time import perf_counter
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / name) for name in ("C014", "C015", "C011")]

import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation
import replay_average as c14
import fit_phone as c15
from minimax import Curve, select_scalar
from replay_long_state import array_sha, front_end

c6, c5, c4, c3, c2 = c14.c6, c14.c5, c14.c4, c14.c3, c14.c2
base, intrinsic = c14.base, c14.intrinsic
read, write, digest, stamp = c14.read, c14.write, c14.digest, c14.stamp
DATA, PILOT, TRANSFER, MASKS, GATES = c14.DATA, c14.PILOT, c14.TRANSFER, c14.MASKS, c14.GATES
C001, C002, C003, C004, C006 = c14.C001, c14.C002, c14.C003, c14.C004, c14.C006
UP = c14.UP
TAU = .25
MAX_SPREAD = np.deg2rad(3.)
CAP = float(np.tan(np.deg2rad(6.)))
DOMAINS = ("BROAD", "Sassari", "YT_Server", "RIDI")
METHODS = ("full_vqf_deg", "C014_B_deg", "C017_deg", "physical_only_deg")
IDENTITY = ("dataset", "record", "group", "instrument", "scenario", "reference_QA", "reference_status",
            "mask_key", "scored_frames", "scoring_indices_sha256")


def window_starts(dt, count):
    first_block = (base.WINDOW + base.HOP - 1) // base.HOP * base.HOP - base.WINDOW
    start_blocks = first_block + base.HOP * np.arange(count, dtype=np.int64)
    return np.floor(start_blocks * ((1. / dt) / 100.) + 1e-9).astype(np.int64)


def force_at(q, acceleration, observed, dt, at, starts, rest):
    """Advance only on nonzero valid packets, then read the last visible state."""
    size = np.linalg.norm(acceleration, axis=1)
    use = observed & np.isfinite(size) & (size > 0.)
    native = np.flatnonzero(use)
    if not len(native):
        return np.zeros((len(at), 3)), np.zeros(len(at), dtype=bool), np.full(len(at), -1, dtype=np.int64)
    world = c14.rotate(c14.normalize(q[native]), acceleration[native] / size[native, None])
    decay = float(np.exp(-dt / TAU))
    weighted = lfilter([1. - decay], [1., -decay], world, axis=0)
    j = np.searchsorted(native, at, side="right") - 1
    last = np.where(j >= 0, native[j.clip(min=0)], -1)
    visible = (last >= 0) & (last >= starts)
    state = np.zeros((len(at), 3))
    state[visible] = weighted[j[visible]]
    length = np.linalg.norm(state, axis=1)
    spread = np.sqrt(2. * np.maximum(0., 1. - length))
    enabled = visible & ~rest & (length > 0.) & (spread < MAX_SPREAD)
    direction = np.zeros_like(state)
    direction[enabled, :2] = state[enabled, :2] / length[enabled, None]
    base.require(np.all(last <= at) and np.isfinite(direction).all(), "Invalid causal force state")
    return direction, enabled, np.where(visible, last, -1)


def branch_at(q_last, history, models, mean, std):
    b6, b7 = models
    a = c4.evaluate_windows(b6, {"a": b6.output}, history, mean, std)["a"]
    b = c4.evaluate_windows(b7, {"b": b7.output}, history, mean, std)["b"]
    rest = c6.rest_flags(history)
    b[rest] = 0.
    rotation = c14.normalize(q_last)
    wa = c14.rotate(rotation, a * c4.GAIN)
    wb = c14.rotate(rotation, b * c4.GAIN)
    base.require(np.max(np.abs(np.r_[wa[:, 2], wb[:, 2]])) <= 1e-6, "C014-B world tangent changed")
    wa[:, 2] = wb[:, 2] = 0.
    return UP + .5 * (wa + wb), rest


def additions(force, k):
    norm = np.linalg.norm(force, axis=1)
    cap_k = np.divide(CAP, norm, out=np.full(len(norm), np.inf), where=norm > 0.)
    return force * np.minimum(k, cap_k)[:, None], cap_k < k


def integrity_models(protocol, campaign):
    b6_file = PILOT / campaign["incumbent"]
    b7_file = ROOT / "C013/seed-C006-readout.pt"
    for path in (b6_file, b7_file):
        base.require(digest(path) == protocol["files_sha256"][str(path.relative_to(PILOT))], "C014-B frozen branch hash changed")
    b6_state, b7_state = (torch.load(p, map_location="cpu", weights_only=True) for p in (b6_file, b7_file))
    base.require(b6_state["seed"] == 1729 and b7_state["seed"] == 314159 and c4.GAIN == .25, "C014-B branch recipe changed")
    base.require(all(torch.equal(b6_state[k], b7_state[k]) for k in ("mean", "std")), "C014-B normalization changed")
    device = torch.device("cuda")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "C014-B CUDA path unavailable")
    models = []
    for state in (b6_state, b7_state):
        model = base.ResidualHead(state["model"])
        model.load_state_dict(state["state_dict"])
        models.append(model.eval().requires_grad_(False).to(device))
    return models, b6_state["mean"].to(device), b6_state["std"].to(device), b6_file, b7_file


def trained_phone(spec, prior):
    with np.load(c15.RIDI / spec["file"], allow_pickle=False) as data:
        time, a, g, obs, quat, valid = (data[k] for k in ("time_ns", "acc_m_s2", "gyr_rad_s", "imu_observed", "tango_wxyz", "reference_valid"))
    raw = np.column_stack((a, g))
    c15.verify_phone_input(spec, raw, obs, time)
    for k, x, dtype in (("raw_imu", raw, "<f8"), ("original_accelerometer", a, "<f8"), ("original_gyro", g, "<f8"),
                       ("tango_wxyz", quat, "<f8"), ("original_imu_mask", obs, "?"), ("original_reference_mask", valid, "?")):
        base.require(array_sha(x, dtype) == prior[k + "_sha256"], f"Original phone {k} changed")
    truth, pose = base.reference_direction(quat)
    mask = valid & pose & obs & (time - time[0] >= c15.WARMUP_NS)
    return raw, obs, 1. / c15.RATE, mask, truth, {"raw_imu": prior["raw_imu_sha256"], "tango": prior["tango_wxyz_sha256"]}


def training(inventory, p15, cfg, models, mean, std):
    prep = read(ROOT / "C015/training-preparation.json")
    p16 = read(ROOT / "C016/training-preparation.json")
    sample_path = ROOT / "C015/sampling.npz"
    archived_path = ROOT / "C016/training-windows.npz"
    base.require(digest(sample_path) == p16["C015_sampling_sha256"] == prep["sampling_sha256"]
                 and digest(archived_path) == p16["training_windows_sha256"], "Frozen C015 draw evidence changed")
    with np.load(sample_path, allow_pickle=False) as samples, np.load(archived_path, allow_pickle=False) as archived:
        ordinal = samples["C015_record_ordinal"].ravel()
        native = samples["C015_native_release"].ravel()
        draws = samples["C015_global_window"].ravel()
        for x, k in ((ordinal, "source_record_ordinal"), (native, "native_release"), (draws, "C015_global_window")):
            np.testing.assert_array_equal(x, archived[k])
        orig_last, orig_rest = archived["native_input_last"], archived["rest_gate"]
        weight = archived["per_draw_weight"]
    sources = p15["legacy_train_sources"] + p15["ridi_sources"]
    base.require(len(ordinal) == 15360 and len(sources) == 67, "Incorrect training population")
    b, f, y = (np.empty((len(ordinal), 3)) for _ in range(3))
    support, last_valid = np.zeros(len(ordinal), dtype=bool), np.full(len(ordinal), -1, dtype=np.int64)
    names, records, mats = [], [], {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, spec in enumerate(sources):
            started = perf_counter()
            domain, name = spec.get("domain", "RIDI"), spec["record"]
            names.append(domain)
            take = np.flatnonzero(ordinal == i)
            base.require(len(take) > 0, f"No original slots for {name}")
            if domain == "RIDI":
                raw, obs, dt, mask, truth, hashes = trained_phone(spec, prep["ridi_eligibility"][i - 63])
            else:
                raw, obs, dt, mask, truth, hashes = c4.train_raw(spec, yt["train"], mats)
                base.require(hashes == prep["original_sources"][i]["raw_hashes"], f"External source {name} changed")
            _, q, at, release, windows, complete, acc, bias = front_end(raw, obs, dt, cfg)
            unique, inverse = np.unique(native[take], return_inverse=True)
            position = np.searchsorted(release, unique)
            np.testing.assert_array_equal(release[position], unique)
            base.require(np.all(mask[unique] & complete[position]), f"Ineligible archived release for {name}")
            np.testing.assert_array_equal(at[position][inverse], orig_last[take])
            b_unique, rest = branch_at(q[at[position]], windows[position], models, mean, std)
            np.testing.assert_array_equal(rest[inverse], orig_rest[take])
            f_unique, on, last = force_at(q, acc, obs, dt, at[position], window_starts(dt, len(at))[position], rest)
            b[take], f[take], y[take] = b_unique[inverse], f_unique[inverse], c14.rotate(c14.normalize(q[unique]), truth[unique])[inverse]
            support[take], last_valid[take] = on[inverse], last[inverse]
            records.append({"domain": domain, "record": name, "slots": len(take), "rest": int(orig_rest[take].sum()),
                            "enabled": int(support[take].sum()), "support_rate": float(support[take].mean()),
                            "nonzero_tangent": int(np.count_nonzero(np.linalg.norm(f[take], axis=1))),
                            "source_sha256": hashes, "C003_active_native_frames": bias["active_native_frames"],
                            "native_release_sha256": array_sha(native[take], "<i8"),
                            "native_input_last_sha256": array_sha(orig_last[take], "<i8"),
                            "valid_force_input_sha256": array_sha(last_valid[take], "<i8"),
                            "seconds": perf_counter() - started})
            print("training", domain, name, len(take), int(support[take].sum()), flush=True)
    per_slot = np.asarray(names)[ordinal]
    independent_weight = np.array([1. / (np.count_nonzero(ordinal == i) * names.count(names[i])) for i in range(len(names))])[ordinal]
    np.testing.assert_array_equal(weight, independent_weight)
    base.require(np.all(last_valid <= orig_last) and np.all(orig_last <= native), "Training uses an unreleased sample")
    data_path = HERE / "training-slots.npz"
    np.savez_compressed(data_path, source_record_ordinal=ordinal, global_window=draws, native_release=native,
                        native_input_last=orig_last, valid_force_input=last_valid, rest=orig_rest, support=support,
                        C014_B=b, force=f, truth=y, weight=weight, domain=per_slot)
    evidence = {"C015_sampling_sha256": digest(sample_path), "source_slots_sha256": digest(data_path),
                "draw_count": len(ordinal), "record_count": len(sources), "records": records}
    return (b, f, y, weight, per_slot, support, ordinal), evidence

def select_k(training_data, evidence):
    b, f, y, weight, slot_domain, support, ordinal = training_data
    masks = [slot_domain == d for d in DOMAINS]
    curves = [Curve(b[t], f[t], y[t], weight[t], CAP) for t in masks]
    start = perf_counter()
    k, certificate = select_scalar(curves)
    extra, capped = additions(f, k)
    losses = {name: np.sum((c14.normalize(v) - y)**2, axis=1)
              for name, v in (("C014_B", b), ("C017", b + extra), ("physical_only", UP + extra), ("C003_front_end", np.broadcast_to(UP, y.shape)))}
    domains = {}
    for name, use, curve in zip(DOMAINS, masks, curves, strict=True):
        row = {key: float(weight[use] @ val[use]) for key, val in losses.items()}
        ratio = row["C017"] / row["C014_B"]
        base.require(abs(ratio - curve.ratio(k)) < 2e-11, "External normalized loss audit disagrees")
        domains[name] = {"records": len(np.unique(ordinal[use])), "slots": int(use.sum()),
                         "enabled": int(support[use].sum()), "support_rate_by_slot": float(support[use].mean()),
                         "support_rate_record_equal": float(weight[use] @ support[use]), "capped": int(capped[use].sum()),
                         "relative_to_C014_B": ratio, "mean_chord_loss": row}
    for i, row in enumerate(evidence["records"]):
        use = ordinal == i
        row["capped"] = int(capped[use].sum())
        row["mean_chord_loss"] = {key: float(val[use].mean()) for key, val in losses.items()}
    return k, {"coefficient_k": k, "certificate": certificate, "domains": domains,
               "solver_seconds": perf_counter() - start, **evidence}


def audit(cfg, k):
    audit_rows = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, observed = c14.synthetic(hz)
        # Distinguish a missing packet from an observed zero acceleration packet.
        raw[round(5.4 * hz), :3] = 0.
        _, q, at, release, w, _, acc, _ = front_end(raw, observed, dt, cfg, optical)
        starts = window_starts(dt, len(at))
        rest = c6.rest_flags(w)
        force, enabled, last = force_at(q, acc, observed, dt, at, starts, rest)
        decay = np.exp(-dt / TAU)
        direct = np.zeros((len(raw), 3))
        current = np.zeros(3)
        last_good = -1
        sampled = np.full(len(raw), -1)
        rotation = Rotation.from_quat(q, scalar_first=True)
        world = rotation.apply(acc)
        for j, (a, good) in enumerate(zip(world, observed, strict=True)):
            anorm = np.linalg.norm(a)
            if good and anorm > 0.:
                current = decay * current + (1. - decay) * (a / anorm)
                last_good = j
            direct[j] = current
            sampled[j] = last_good
        size = np.linalg.norm(direct[at], axis=1)
        expected_on = (sampled[at] >= starts) & (sampled[at] >= 0) & ~rest & (size > 0.) & (np.sqrt(2. * np.maximum(0., 1. - size)) < MAX_SPREAD)
        np.testing.assert_array_equal(enabled, expected_on)
        np.testing.assert_allclose(force[enabled, :2], direct[at[enabled], :2] / size[enabled, None], rtol=0., atol=1e-13)
        np.testing.assert_array_equal(last, np.where(sampled[at] >= starts, sampled[at], -1))
        extra, clipped = additions(force, k)
        base.require(np.all(np.linalg.norm(extra, axis=1) <= CAP + 1e-14) and np.all(extra[rest] == 0.), "Cap or rest gate failed")
        cut_candidates = {len(at) // 2, len(at) - 2}
        gap = np.flatnonzero(release > at)
        if len(gap):
            cut_candidates.add(int(gap[len(gap) // 2]))
        for j in sorted(cut_candidates):
            stop = int(at[j]) + 1
            edited = raw.copy()
            edited[stop:, :3] += [14., -8., 6.]
            edited[stop:, 3:] += .94
            changed_mask = observed.copy()
            changed_mask[stop + 1::2] = False
            # Full mutation tests the release sample when release > input-last.
            _, q2, at2, release2, w2, _, a2, _ = front_end(edited, changed_mask, dt, cfg, optical)
            end = j + 1
            np.testing.assert_array_equal(at2[:end], at[:end])
            np.testing.assert_array_equal(release2[:end], release[:end])
            replay, on, latest = force_at(q2, a2, changed_mask, dt, at2[:end], starts[:end], c6.rest_flags(w2[:end]))
            np.testing.assert_array_equal(replay, force[:end])
            np.testing.assert_array_equal(on, enabled[:end])
            np.testing.assert_array_equal(latest, last[:end])
            truncated = int(release[j]) + 1
            _, qp, ap, rp, wp, _, accp, _ = front_end(raw[:truncated], observed[:truncated], dt, cfg, optical)
            np.testing.assert_array_equal(rp[:end], release[:end])
            np.testing.assert_array_equal(ap[:end], at[:end])
            past, _, _ = force_at(qp, accp, observed[:truncated], dt, ap[:end], starts[:end], c6.rest_flags(wp[:end]))
            np.testing.assert_array_equal(past, force[:end])
        fixed_errors, native_errors, native_gate_changes = [], [], []
        for rot in (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                    Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))):
            q_change = (rotation * rot.inv()).as_quat(scalar_first=True)
            f2, on2, l2 = force_at(q_change, rot.apply(acc), observed, dt, at, starts, rest)
            np.testing.assert_array_equal(on2, enabled)
            np.testing.assert_array_equal(l2, last)
            fixed_errors.append(float(np.max(np.abs(f2 - force))))
            rotated = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, q3, at3, rel3, w3, _, acc3, _ = front_end(rotated, observed, dt, cfg, optical)
            np.testing.assert_array_equal(at3, at)
            np.testing.assert_array_equal(rel3, release)
            f3, on3, _ = force_at(q3, acc3, observed, dt, at3, starts, c6.rest_flags(w3))
            unit = c14.normalize(UP + extra)
            unit3 = c14.normalize(UP + additions(f3, k)[0])
            original_body = rotation[release].inv().apply(unit)
            transformed_body = rot.inv().apply(Rotation.from_quat(q3[release], scalar_first=True).inv().apply(unit3))
            native_errors.append(float(c14.tilt_error(original_body, transformed_body).max()))
            native_gate_changes.append(int(np.count_nonzero(on3 != enabled)))
        base.require(max(fixed_errors) <= 1e-12, "Fixed-world mounting rotation failed")
        audit_rows.append({"native_hz": hz, "windows": len(at), "force_gate_open": int(enabled.sum()), "rest": int(rest.sum()),
                           "capped": int(clipped.sum()), "prefix_cut_count": len(cut_candidates),
                           "release_after_input_count": int(len(gap)), "max_fixed_gauge_force_delta": max(fixed_errors),
                           "raw_mount_max_body_tilt_deg": max(native_errors), "raw_mount_gate_changes": max(native_gate_changes),
                           "direct_recurrence_and_prefix_match": True})
    n = 251
    quat = np.broadcast_to([1., 0., 0., 0.], (n, 4))
    acceleration = np.tile([1., 0., 9.8], (n, 1))
    observed = np.ones(n, dtype=bool)
    observed[-2] = False
    acceleration[-1] = 0.
    points = np.array([0, n - 3, n - 2, n - 1, n - 1, n - 1])
    starts = np.array([0, 0, 0, 0, 0, n - 2])
    rest = np.array([False, False, False, False, True, False])
    force, enabled, last = force_at(quat, acceleration, observed, .01, points, starts, rest)
    np.testing.assert_array_equal(enabled, [False, True, True, True, False, False])
    np.testing.assert_array_equal(last, [0, n - 3, n - 3, n - 3, n - 3, -1])
    np.testing.assert_array_equal(force[1:4], np.broadcast_to(force[1], (3, 3)))
    # Exercise both sides of the cap without changing the locked coefficient.
    delta, capped = additions(np.array([[1., 0., 0.], [0., 0., 0.]]), 1.)
    np.testing.assert_array_equal(capped, [True, False])
    np.testing.assert_allclose(delta, [[CAP, 0., 0.], [0., 0., 0.]], rtol=0., atol=1e-15)
    return audit_rows


def payload(key, spec, map3, map1, medium, yt, split, c1p, array, evidence, cfg):
    if key[0] in ("OxIOD", "RepoIMU"):
        raw, hz, ref, mask, qa = c5.extract(spec)
        obs = np.isfinite(raw).all(axis=1)
        base.require(np.all(obs[mask]), "Optical validity changed")
        data = front_end(raw, obs, 1. / hz, cfg, hz)
        return data, raw, obs, 1. / hz, ref, mask, {}, qa["quality_flags"]
    raw, dt, obs, ref, mask, axes = c3.sources(map3[key], spec, map1.get(key), medium, yt, split, c1p, array,
                                                evidence["array_raw_sha256"])
    data = front_end(raw, obs, dt, cfg)
    base.require(data[-1]["first_active_native_index"] == map3[key]["bias"]["first_active_native_index"], "C003 activation changed")
    return data, raw, obs, dt, ref, mask, axes, None


def measure(indices, refs, raw_q, q, release, estimates):
    active = np.searchsorted(release, indices, side="right")
    raw_ref = c14.rotate(c14.normalize(raw_q[indices]), refs[indices])
    ref = c14.rotate(c14.normalize(q[indices]), refs[indices])
    errors = {METHODS[0]: c14.tilt_error(np.broadcast_to(UP, raw_ref.shape), raw_ref)**2}
    for name, units in estimates.items():
        errors[name] = c14.tilt_error(units[active], ref)**2
    return errors


def scores(errors, use=slice(None)):
    return {name: float(np.sqrt(np.mean(values[use]))) for name, values in errors.items()}


def check_controls(result, previous):
    gap = max(abs(result[a] - previous[b]) for a, b in (("full_vqf_deg", "full_vqf_deg"), ("C014_B_deg", "B_deg")))
    base.require(gap <= 1e-6 and np.isfinite(list(result.values())).all(), f"C014-B frozen controls differ by {gap}")
    return gap


def summary(rows):
    kept = [row for row in rows if row["status"] == "scored"]
    macro = {k: float(np.mean([r[k] for r in kept])) for k in METHODS} if kept else {}
    return {"records": len(kept), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in kept),
            "macro_rmse_deg": macro, "wins_vs_C014_B": sum(r["C017_deg"] < r["C014_B_deg"] for r in kept),
            "delta_vs_C014_B": macro["C017_deg"] - macro["C014_B_deg"] if kept else None}

def replay(prior, inventory, campaign, split, cfg, models, mean, std, k):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in src["sequences"]} for src in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(s["side"] in ("development", "development_previously_used_for_selection") for s in specs.values()),
                 "Unknown or held-out record in replay")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(array_path)
    old_poses = {(a["record"], a["pose"]): a for a in prior["array_sensor_poses"]}
    rows, poses, strata, failures = [], [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            started = perf_counter()
            key = old["dataset"], old["record"]
            identity = {f: old[f] for f in IDENTITY}
            try:
                data, raw, observed, dt, ref, mask, axes, quality = payload(key, specs[key], map3, map1, medium,
                                                                           yt["valid"], split, c1p, array, evidence, cfg)
                q0, q, at, release, windows, _, acc, _ = data
                if quality is not None:
                    base.require(quality == old["reference_quality_flags"], "Optical classification changed")
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                indices = np.flatnonzero(mask)
                base.require(len(indices) == old["scored_frames"] and intrinsic.mask_digest(mask) == old["scoring_indices_sha256"],
                             "C014 native mask changed")
                head_at = perf_counter()
                mamba, rest = branch_at(q[at], windows, models, mean, std)
                head_s = perf_counter() - head_at
                frozen = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, frozen[:, 0])
                np.testing.assert_array_equal(rest, frozen[:, 3])
                force_started = perf_counter()
                force, enabled, last = force_at(q, acc, observed, dt, at, window_starts(dt, len(at)), rest)
                extra, capped = additions(force, k)
                force_s = perf_counter() - force_started
                estimates = {name: np.vstack((UP, c14.normalize(v))) for name, v in
                             zip(METHODS[1:], (mamba, mamba + extra, UP + extra), strict=True)}
                errors = measure(indices, ref, q0, q, release, estimates)
                values = scores(errors)
                gap = check_controls(values, old)
                lookup = np.searchsorted(release, indices, side="right")
                active_score = np.r_[False, enabled][lookup]
                rest_score = np.r_[False, rest][lookup]
                base.require(int(rest_score.sum()) == old["rest_scored_frames"], "C014-B rest coverage changed")
                new_poses, new_strata = [], []
                for pose, subset in axes.items():
                    use = subset[indices]
                    old_pose = old_poses[key[1], pose]
                    sha = intrinsic.mask_digest(mask & subset)
                    base.require(sha == old_pose["scoring_indices_sha256"] and int(use.sum()) == old_pose["scored_frames"],
                                 "Native pose mask changed")
                    result = scores(errors, use)
                    new_poses.append({"record": key[1], "pose": pose, "scoring_indices_sha256": sha, "scored_frames": int(use.sum()),
                                      "control_max_abs_delta_deg": check_controls(result, old_pose), **result})
                if key[0] == "SHICE":
                    speed = np.linalg.norm(raw[indices, 3:], axis=1)
                    close_to_g = np.abs(np.linalg.norm(raw[indices, :3], axis=1) / 9.80665 - 1.) < .05
                    for label, use in (("lt_0.05", speed < .05), ("0.05_to_0.5", (.05 <= speed) & (speed < .5)),
                                       ("0.5_to_2", (.5 <= speed) & (speed < 2.)), ("ge_2", speed >= 2.)):
                        for stratum, frames in ((label, use), (label + "_within_5pct_g", use & close_to_g)):
                            if frames.any():
                                new_strata.append({"record": key[1], "stratum": stratum, "status": "scored",
                                                   "scoring_indices_sha256": array_sha(indices[frames], "<i8"),
                                                   "scored_frames": int(frames.sum()), "enabled_frames": int(active_score[frames].sum()),
                                                   **scores(errors, frames)})
                row = {**identity, **values, "status": "scored", "control_max_abs_delta_deg": gap,
                       "force_state_sha256": array_sha(force, "<f8"), "valid_force_input_sha256": array_sha(last, "<i8"),
                       "native_release_sha256": array_sha(release, "<i8"), "native_input_last_sha256": array_sha(at, "<i8"),
                       "release_count": len(release), "enabled_releases": int(enabled.sum()), "capped_releases": int(capped.sum()),
                       "enabled_scoring_frames": int(active_score.sum()), "max_extra_norm": float(np.linalg.norm(extra, axis=1).max()),
                       "Mamba_replay_seconds": head_s, "force_replay_seconds": force_s, "seconds": perf_counter() - started}
                rows.append(row)
                poses.extend(new_poses)
                strata.extend(new_strata)
                print("development", *key, *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - started})
                print("FAILED", *key, repr(exc), flush=True)
    seq_path, pose_path, strata_path = (HERE / p for p in ("sequences.csv", "array-sensor-poses.csv", "SHICE-speed-strata.csv"))
    fields = dict.fromkeys(k for row in rows for k in row)
    intrinsic.shice.save_csv(seq_path, [{k: r.get(k, "") for k in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    if strata:
        intrinsic.shice.save_csv(strata_path, strata)
    return {"summary": {d: summary([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "pooled_records": summary(rows),
            "SHICE_speed_strata": {k: summary([r for r in strata if r["stratum"] == k]) for k in sorted({r["stratum"] for r in strata})},
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifact_sha256": {p.name: digest(p) for p in (seq_path, pose_path, strata_path) if p.is_file()}}


def main():
    started = perf_counter()
    launcher = read(HERE / "launch.json")
    base.require(os.getpid() == launcher["pid"], "C017 did not stay in its registered process")
    p14, p15, campaign, p3, initial = (read(p) for p in (ROOT / "C014/protocol.json", ROOT / "C015/protocol.json",
                                                         ROOT / "protocol.json", C003 / "protocol.json", HERE / "input-verification.json"))
    for path, expected in initial["files_sha256"].items():
        base.require(digest(ROOT / path) == expected, f"Frozen input changed since launch: {path}")
    base.require(p14["scoring_masks_sha256"] == p15["scoring_masks_sha256"] == digest(MASKS)
                 and p14["gate_traces_sha256"] == p15["gate_traces_sha256"] == digest(GATES)
                 and p14["development_records"] == p15["development_records"], "C014 and C015 panels disagree")
    inventory_path, split_path = C001 / "inventory.json", TRANSFER / "external-split.json"
    actual_files = {**initial["files_sha256"]}
    for path in (inventory_path, split_path, Path(__file__), HERE / "launch.py", ROOT / "C016/training-preparation.json"):
        actual_files[str(path.relative_to(PILOT))] = digest(path)
    for path in (inventory_path, split_path):
        base.require(digest(path) == p14["files_sha256"][str(path.relative_to(PILOT))], "Development inventory changed")
    models, mean, std, b6_file, b7_file = integrity_models(p14, campaign)
    cfg = c3.Settings(**p3["thresholds"])
    protocol_path = HERE / "protocol.json"
    protocol = {**launcher, "stage": "registered", "files_sha256": actual_files, "single_process": True,
                "development_records": p14["development_records"], "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "fixed_weights": {str(p.relative_to(PILOT)): digest(p) for p in (b6_file, b7_file)},
                "fixed_C014_B": "0.125 times B6 C003 plus 0.125 times B7 C013 readout with the latter rest-gated",
                "fit": "C011 bounded global minimax on actual C015 slots, equal records in each retained source and equal draws within record",
                "force": {"tau_s": TAU, "spread_threshold_deg_strict": 3., "extra_cap_deg": 6.,
                          "zero_start": True, "invalid_or_zero_packets_hold": True,
                          "last_visible": "last valid packet within the 64-block window at input-last or earlier",
                          "rest_gate": "original C006 release flag"}}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **details):
        write(HERE / "process.json", {"status": phase, "pid": os.getpid(), "started_at": launcher["started_at"],
                                     "updated_at": stamp(), "protocol_sha256": protocol_sha, **details})

    status("external_source_preparation")
    print("C017 source pass started", os.getpid(), flush=True)
    inventory = read(inventory_path)
    points, train = training(inventory, p15, cfg, models, mean, std)
    status("external_scalar_fit", slots=train["draw_count"])
    k, selection = select_k(points, train)
    source_path = HERE / "external-selection.json"
    write(source_path, selection)
    lock_path = HERE / "coefficient-lock.json"
    write(lock_path, {"k": k, "locked_at": stamp(), "pid": os.getpid(), "protocol_sha256": protocol_sha,
                      "external_selection_sha256": digest(source_path), "C015_sampling_sha256": train["C015_sampling_sha256"],
                      "predevelopment_lock": True, "no_parameter_updates_after_lock": True})
    sha = digest(lock_path)
    status("coefficient_locked", k=k, coefficient_lock_sha256=sha)
    print("C017 coefficient locked", k, {d: v["relative_to_C014_B"] for d, v in selection["domains"].items()}, flush=True)
    audits_path = HERE / "causal-rotation-audit.json"
    write(audits_path, {"coefficient_lock_sha256": sha, "probes": audit(cfg, k)})
    status("development_replay", k=k, coefficient_lock_sha256=sha)
    before = perf_counter()
    prior = read(ROOT / "C014/results.json")
    result = replay(prior, inventory, campaign, read(split_path), cfg, models, mean, std, k)
    base.require(digest(lock_path) == sha and all(digest(PILOT / p) == v for p, v in protocol["fixed_weights"].items()),
                 "Coefficients or fixed branches changed after lock")
    result_path = HERE / "results.json"
    result.update({"pid": os.getpid(), "started_at": launcher["started_at"], "completed_at": stamp(), "k": k,
                   "protocol_sha256": protocol_sha, "coefficient_lock_sha256": sha, "causal_rotation_audit_sha256": digest(audits_path),
                   "development_replay_seconds": perf_counter() - before, "total_seconds": perf_counter() - started,
                   "scoring_masks_sha256": digest(MASKS), "final_holdout_scored": False})
    write(result_path, result)
    complete = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    status("complete" if complete else "incomplete", results_sha256=digest(result_path), k=k, scored=result["pooled_records"]["records"])
    print("C017 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    if main():
        raise RuntimeError("C017 incomplete; see results.json")
