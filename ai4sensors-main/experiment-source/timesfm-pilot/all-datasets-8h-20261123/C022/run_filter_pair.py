"""C022: one gated mixture of auxiliary VQF response directions around fixed C020."""
from pathlib import Path
from statistics import fmean
from time import perf_counter
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C020"), str(ROOT / "C011"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import run_temporal as c20
from minimax import Curve, select_scalar
from filter_response_bank import response_delta as independent_delta

c19, c18, c17, c14, c6, c4, c3, c2 = c20.c19, c20.c18, c20.c17, c20.c14, c20.c6, c20.c4, c20.c3, c20.c2
base, intrinsic = c20.base, c20.intrinsic
read, write, sha, stamp, array_sha = c20.read, c20.write, c20.sha, c20.stamp, c20.array_sha
DATA, PILOT, TRANSFER, MASKS, GATES = c20.DATA, c20.PILOT, c20.TRANSFER, c20.MASKS, c20.GATES
UP, DOMAINS, STEPS, BATCH = c20.UP, c20.DOMAINS, c20.STEPS, c20.BATCH
CAP = float(np.tan(np.deg2rad(6.)))
METHODS = ("full_vqf_deg", "C020_deg", "C022_deg", "physical_only_deg")


def tangent(world):
    horizontal = np.column_stack((world[:, :2] / np.maximum(world[:, 2:3], 1.e-6), np.zeros(len(world))))
    length = np.linalg.norm(horizontal, axis=1)
    return horizontal * np.minimum(1., CAP / np.maximum(length, 1.e-12))[:, None]


def responses(q, q_fast, q_slow):
    width = (len(q), 4)
    base.require(q.shape == q_fast.shape == q_slow.shape == width, "VQF response branches use different native cuts")
    q = c14.normalize(q)
    up = np.broadcast_to(UP, (len(q), 3))
    out = []
    for auxiliary in (q_fast, q_slow):
        inv = c14.normalize(auxiliary)
        inv[:, 1:] *= -1.
        out.append(tangent(c14.rotate(q, c14.rotate(inv, up))))
    return out


def vf(gyro, accel, dt, tau):
    return base.VQF(dt, tauAcc=tau).updateBatch(gyro, accel)["quat6D"]


def filters(raw, observed, acceleration, q, dt, at, independence=False, check_default=False):
    gyro = np.ascontiguousarray(base.causal_fill(raw, observed)[:, 3:], dtype=np.float64)
    acc = np.ascontiguousarray(acceleration, dtype=np.float64)
    q_fast, q_slow = (vf(gyro, acc, dt, tau)[at] for tau in (1.5, 6.))
    fast, slow = responses(q[at], q_fast, q_slow)
    combined = (2. * fast + slow) / 3.
    base.require(np.all(np.linalg.norm(combined, axis=1) <= CAP + 2e-15) and np.all(combined[:, 2] == 0.),
                 "Filter response violated the horizontal bound")
    reference_gap = None
    if independence:
        expect = independent_delta(*(Rotation.from_quat(x, scalar_first=True) for x in (q[at], q_fast, q_slow)))
        reference_gap = float(np.max(np.abs(combined - expect)))
        base.require(reference_gap <= 1e-12, "Error in the independent tauAcc 2:1 vector cross-check")
    default_gap = None
    if check_default:
        own = vf(gyro, acc, dt, 3.)
        expected = c14.normalize(q)
        own = c14.normalize(own)
        signs = np.where((own * expected).sum(axis=1, keepdims=True) < 0., -1., 1.)
        default_gap = float(np.max(np.abs(own * signs - expected)))
        base.require(default_gap <= 3e-10, "Default tauAcc=3 does not reproduce central VQF")
    return fast, slow, combined, q_fast, q_slow, reference_gap, default_gap


def base_packet(q, at, release, history, acc, observed, dt, fixed):
    models, mean, std, learned, average, scale, k17 = fixed
    block_q = q[at]
    c17_center, rest = c17.branch_at(block_q, history, models, mean, std)
    force, _, last_force = c17.force_at(q, acc, observed, dt, at, c17.window_starts(dt, len(release)), rest)
    base.require(np.all(at <= release) and np.all(last_force <= at), "Central branch used an unreleased input")
    physical = UP + c17.additions(force, k17)[0]
    center = c17_center + c17.additions(force, k17)[0]
    feat = c18.features(history, block_q, learned, mean, std)
    old, old_delta = c20.learned_direction(learned, feat, center, rest, average, scale)
    return old, center + old_delta, physical, rest, feat.valid


def load_fixed(protocol):
    files = protocol["input_file_sha256"]
    base.require(all(sha(ROOT / n) == h for n, h in files.items())
                 and all(sha(PILOT / n) == h for n, h in protocol["inherited_C020_files_sha256"].items()),
                 "Frozen C020 control or method file changed")
    campaign, p14 = read(ROOT / "protocol.json"), read(ROOT / "C014/protocol.json")
    models, mean, std, _, _ = c17.integrity_models(p14, campaign)
    from copy import deepcopy
    learned = deepcopy(models[0])
    ckpt = torch.load(ROOT / "C020/mamba-end-to-end.pt", weights_only=True, map_location=mean.device)
    learned.load_state_dict(ckpt["state_dict"])
    learned.eval().requires_grad_(False)
    for own, k in ((mean, "B6_input_mean"), (std, "B6_input_std")):
        torch.testing.assert_close(own, ckpt[k], atol=0., rtol=0.)
    base.require(ckpt["optimizer_steps"] == 160 and ckpt["architecture"]["window"] == 64, "C020 checked weights have a different architecture")
    return (models, mean, std, learned, ckpt["C019_feature_mean"], ckpt["C019_feature_std"], protocol["C017_k"]), campaign


@torch.inference_mode()
def training_direction(data, history, fixed):
    _, mean, std, learned, average, scale, _ = fixed
    device = mean.device
    p = c18.to_device
    center = p(data["center"], device)
    up, world, truth, enabled = (p(v, device) for v in (data["canonical_up"], data["world_basis"], data["truth"], data["valid"] & ~data["rest"]))
    out, offsets = [], []
    for i in range(0, len(history), 256):
        s = slice(i, i + 256)
        x = base.normalized(p(np.array(history[s], copy=True), device), mean, std)
        y, d = c20.predict(learned, x, up[s], world[s], enabled[s], center[s], average, scale)
        out.append(y.cpu().numpy())
        offsets.append(d.cpu().numpy())
    directions = np.concatenate(out)
    delta = np.concatenate(offsets)
    c020 = data["center"] + delta
    np.testing.assert_allclose(c14.normalize(c020), directions, rtol=0., atol=5e-16)
    base.require(np.all(delta[data["rest"] | ~data["valid"]] == 0.) and np.all(c020[:, 2] == 1.),
                 "Unexpected saved C020 center or rest behavior")
    return c020, directions


def source_slots(inventory, data, fixed, cfg):
    p15 = read(ROOT / "C015/protocol.json")
    p15prep = read(ROOT / "C015/training-preparation.json")
    p19prep = read(ROOT / "C019/training-preparation.json")
    p20prep = read(ROOT / "C020/training-preparation.json")
    panel = read(ROOT / "review/yt_training_panel.json")
    old_specs, _, new, approved, _ = c19.frozen_panel(inventory, panel, p15)
    names = p19prep["source_records"]
    base.require(names == [r["record"] for r in old_specs] + p19prep["new_YT_trial_order"]
                 and len(names) == 139 and set(new) == set(names[len(old_specs):]), "Unregistered external source list")
    history = np.load(ROOT / "C020/training-canonical-history.npy", mmap_mode="r", allow_pickle=False)
    native, ordinal = data["native_release"], data["source_record_ordinal"]
    center, init_direction = training_direction(data, history, fixed)
    weights = data["record_equal_weights"]
    usable = data["valid"] & ~data["rest"]
    q_cuts, q_at_release, q_fast, q_slow = (np.empty((len(ordinal), 4)) for _ in range(4))
    fast, slow, mixture = (np.empty((len(ordinal), 3)) for _ in range(3))
    rows, mats = [], {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, name in enumerate(names):
            started = perf_counter()
            slots = np.flatnonzero(ordinal == i)
            domain = str(data["domain"][slots[0]])
            np.testing.assert_array_equal(data["source_record_name"][slots], name)
            np.testing.assert_array_equal(data["domain"][slots], domain)
            old = p20prep["records"][i]
            base.require(old["domain"] == domain and old["record"] == name and old["ordinal"] == i
                         and old["native_release_sha256"] == array_sha(native[slots], "<i8"), "C020 slot identity changed")
            if i < len(old_specs):
                spec = old_specs[i]
            else:
                check = approved[name]
                base.require(yt["train"][name].name == check["source"], "New YT segment is out of allowed training scope")
                spec = {"record": name, "domain": "YT_Server", "native_frames": check["native_frames"]}
            if domain == "RIDI":
                raw, observed, dt, score, truth, fields = c17.trained_phone(spec, p15prep["ridi_eligibility"][i - 63])
            else:
                raw, observed, dt, score, truth, fields = c4.train_raw(spec, yt["train"], mats)
            base.require(fields == old["source_sha256"], "C020 physical signal or label hash changed")
            _, q, at, release, hist, complete, acc, dia = c17.front_end(raw, observed, dt, cfg)
            unique, inv = np.unique(native[slots], return_inverse=True)
            positions = np.searchsorted(release, unique)
            np.testing.assert_array_equal(release[positions], unique)
            cuts = at[positions]
            base.require(np.all(complete[positions] & score[unique]) and np.all(cuts <= unique), "Original C020 slot is no longer causal with a valid target")
            np.testing.assert_array_equal(cuts[inv], data["native_input_last"][slots])
            picked = hist[positions]
            frame = intrinsic.build_frame(picked)
            rest = c6.rest_flags(picked)
            np.testing.assert_array_equal(rest[inv], data["rest"][slots])
            np.testing.assert_array_equal(frame.valid[inv], data["valid"][slots])
            np.testing.assert_array_equal(intrinsic.transform_inputs(picked, frame)[inv], history[slots])
            target = c14.rotate(c14.normalize(q[unique]), truth[unique])
            np.testing.assert_array_equal(target[inv], data["truth"][slots])
            q_cuts[slots], q_at_release[slots] = q[cuts][inv], q[unique][inv]
            f, s, h, fq, sq, gap, central_gap = filters(raw, observed, acc, q, dt, cuts, independence=True, check_default=True)
            q_fast[slots], q_slow[slots] = fq[inv], sq[inv]
            fast[slots], slow[slots], mixture[slots] = f[inv], s[inv], h[inv]
            rows.append({"domain": domain, "record": name, "ordinal": i, "slots": len(slots), "distinct_releases": len(unique),
                         "active_slots": int(usable[slots].sum()), "native_release_sha256": array_sha(native[slots], "<i8"),
                         "native_input_last_sha256": array_sha(cuts[inv], "<i8"), "source_sha256": fields,
                         "default_quat_max_abs_vs_tau3": central_gap, "independent_2to1_max_abs": gap,
                         "q_default_input_sha256": array_sha(q_cuts[slots], "<f8"), "target_sha256": array_sha(target[inv], "<f8"),
                         "source_seconds": perf_counter() - started, "debias_active_native_frames": dia["active_native_frames"]})
            print("C022 source", domain, name, len(slots), int(usable[slots].sum()), flush=True)
    base.require(np.all(np.isfinite(mixture)) and len(rows) == 139, "Training filter vectors are incomplete")
    raw_mix = mixture.copy()
    mixture[~usable] = 0.
    source = {"center": center, "zero_direction": init_direction, "truth": data["truth"], "response": mixture,
              "ungated_response": raw_mix, "fast": fast, "slow": slow, "q_default_input_last": q_cuts, "q_default_release": q_at_release,
              "q_fast_input_last": q_fast, "q_slow_input_last": q_slow, "rest": data["rest"], "intrinsic_valid": data["valid"],
              "enabled": usable, "domain": data["domain"], "source_record_ordinal": ordinal, "source_record_name": data["source_record_name"],
              "record_equal_weights": weights, "native_release": native, "native_input_last": data["native_input_last"]}
    path = HERE / "training-slots.npz"
    np.savez_compressed(path, **source)
    return source, {"records": rows, "slots": len(ordinal), "active_slots": int(usable.sum()), "array_sha256": sha(path),
                    "C020_canonical_histories_sha256": p20prep["history_sha256"]}

def chord(base_direction, truth):
    return np.sum((c14.normalize(base_direction) - truth)**2, axis=1)


def zero_slopes(source, independent):
    baseline, truth = source["center"], source["truth"]
    size = np.linalg.norm(baseline, axis=1, keepdims=True)
    u = baseline / size
    error = u - truth
    loss = np.sum(error**2, axis=1)
    active = source["enabled"]
    derivative = {}
    for label, v in (("fast", source["fast"]), ("slow", source["slow"]), ("mixture", source["response"])):
        d = (v - u * np.sum(u * v, axis=1, keepdims=True)) / size
        derivative[label] = np.where(active, 2. * np.sum(error * d, axis=1), 0.)
    old = read(ROOT / "C020/training.json")["source_scores"]["domains"]
    weight, domains = source["record_equal_weights"], source["domain"]
    out = {}
    for d in DOMAINS:
        use = domains == d
        denom = float(weight[use] @ loss[use])
        base.require(abs(weight[use].sum() - 1.) < 1e-12 and abs(denom - old[d]["record_equal_chord_loss"]["C020"]) <= 2e-11,
                     "k=0 external replay differs from the locked C020 source score")
        slopes = {key: float(weight[use] @ a[use] / denom) for key, a in derivative.items()}
        reference = independent["summary"][d]["all"]
        expect = (2. * reference["fast_fractional_loss_slope"] + reference["slow_fractional_loss_slope"]) / 3.
        gap = max(abs(slopes["fast"] - reference["fast_fractional_loss_slope"]),
                  abs(slopes["slow"] - reference["slow_fractional_loss_slope"]), abs(slopes["mixture"] - expect))
        base.require(int((use & active).sum()) == reference["slots"] and abs(slopes["mixture"] - (2. * slopes["fast"] + slopes["slow"]) / 3.) <= 2e-14
                     and slopes["mixture"] < 0. and gap < 2e-5, f"Independent zero-derivative discrepancy for {d}: {gap}")
        out[d] = {"slots": int(use.sum()), "active": int((use & active).sum()), "C020_loss_at_zero": denom,
                  "archived_C020_loss_at_zero": old[d]["record_equal_chord_loss"]["C020"], "fractional_slopes": slopes,
                  "review_fractional_mixture_slope": expect, "max_abs_review_slope_difference": gap}
    return out


def fit_coefficient(source, slopes):
    v, center, truth, weight, domains = (source[k] for k in ("response", "center", "truth", "record_equal_weights", "domain"))
    use = [domains == d for d in DOMAINS]
    curves = [Curve(center[sel], v[sel], truth[sel], weight[sel], CAP) for sel in use]
    started = perf_counter()
    k, certificate = select_scalar(curves)
    loss0, loss1 = chord(center, truth), chord(center + k * v, truth)
    score = {}
    for d, sel, curve in zip(DOMAINS, use, curves, strict=True):
        old, new = float(weight[sel] @ loss0[sel]), float(weight[sel] @ loss1[sel])
        ratio = new / old
        base.require(abs(ratio - curve.ratio(k)) <= 2e-11, "C022 certified loss differs from direct normalization")
        score[d] = {**slopes[d], "C022_loss_at_lock": new, "loss_ratio": ratio, "ratio_at_one": curve.ratio(1.)}
    records = []
    for i in np.unique(source["source_record_ordinal"]):
        used = source["source_record_ordinal"] == i
        records.append({"ordinal": int(i), "domain": str(source["domain"][used][0]), "record": str(source["source_record_name"][used][0]),
                        "count": int(used.sum()), "enabled": int(source["enabled"][used].sum()),
                        "C020_chord_loss": float(loss0[used].mean()), "C022_chord_loss": float(loss1[used].mean())})
    return k, {"C022_k": k, "fit_count": 1, "compute_seconds": perf_counter() - started,
               "minimax_certificate": certificate, "domains": score, "records": records}


def held(q, at, release, hist, acc, raw, observed, dt, k, fixed, independent=False):
    orig, C020, physical, rest, valid = base_packet(q, at, release, hist, acc, observed, dt, fixed)
    before = perf_counter()
    _, _, combined, _, _, _, _ = filters(raw, observed, acc, q, dt, at, independence=independent)
    filter_seconds = perf_counter() - before
    enabled = valid & ~rest
    extra = k * np.where(enabled[:, None], combined, 0.)
    # Exact freeze of the central output where the new filter response is zero.
    predicted = orig.copy()
    changed = np.any(extra != 0., axis=1)
    predicted[changed] = c14.normalize(C020[changed] + extra[changed])
    np.testing.assert_array_equal(predicted[~enabled], orig[~enabled])
    if k == 0.:
        np.testing.assert_array_equal(predicted, orig)
    return {"C020_deg": orig, "C022_deg": predicted, "physical_only_deg": c14.normalize(physical + extra)}, extra, rest, enabled, filter_seconds


def audit(cfg, k, fixed):
    rows = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, obs = c14.synthetic(hz)
        _, q, at, release, hist, _, acc, _ = c17.front_end(raw, obs, dt, cfg, optical)
        full, extra, rest, enabled, _ = held(q, at, release, hist, acc, raw, obs, dt, k, fixed, independent=True)
        bundle0, _, _, _, _ = held(q, at, release, hist, acc, raw, obs, dt, 0., fixed)
        np.testing.assert_array_equal(bundle0["C020_deg"], bundle0["C022_deg"])
        tests = {len(at) // 2, len(at) - 2}
        late = np.flatnonzero(release > at)
        if len(late):
            tests.add(int(late[len(late) // 2]))
        differences = dict.fromkeys(METHODS[1:], 0.)
        for j in sorted(tests):
            # Modify every native row after the last input, including the publication row when it is distinct.
            changed = raw.copy()
            changed[at[j] + 1:, :3] += [17., -8., 7.]
            changed[at[j] + 1:, 3:] += 1.1
            changed_obs = obs.copy()
            changed_obs[at[j] + 2::2] = False
            end, stop = j + 1, int(release[j]) + 1
            for x, o in ((changed, changed_obs), (raw[:stop], obs[:stop])):
                _, qr, ar, rr, h, _, a, _ = c17.front_end(x, o, dt, cfg, optical)
                np.testing.assert_array_equal(at[:end], ar[:end])
                np.testing.assert_array_equal(release[:end], rr[:end])
                v, delta, gate, on, _ = held(qr, ar[:end], rr[:end], h[:end], a, x, o, dt, k, fixed)
                np.testing.assert_array_equal(gate, rest[:end])
                np.testing.assert_array_equal(on, enabled[:end])
                np.testing.assert_allclose(delta, extra[:end], rtol=0., atol=2e-15)
                for key in differences:
                    differences[key] = max(differences[key], float(c20.angles(v[key], full[key][:end]).max()))
        base.require(max(differences.values()) <= 1e-5, "C022 uses information past its input-last cut")
        gyro = np.ascontiguousarray(base.causal_fill(raw, obs)[:, 3:], dtype=np.float64)
        acc64 = np.ascontiguousarray(acc, dtype=np.float64)
        rot0, r_f, r_s, r_release = (Rotation.from_quat(a, scalar_first=True) for a in
                                      (q[at], vf(gyro, acc64, dt, 1.5)[at], vf(gyro, acc64, dt, 6.)[at], q[release]))
        rot_stats = []
        for label, rot in zip(("90x", "90y", "90z", "63_1_2_-1"),
                                (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))), strict=True):
            fix0, fixf, fixs = ((r * rot.inv()).as_quat(scalar_first=True) for r in (rot0, r_f, r_s))
            f, s = responses(fix0, fixf, fixs)
            gauge_delta = k * np.where(enabled[:, None], (2. * f + s) / 3., 0.)
            gauge_gap = float(np.max(np.abs(gauge_delta - extra)))
            base.require(gauge_gap <= 1e-12, "Auxiliary displacement changes in a rotated body basis")
            rotated = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, q3, at3, release3, h3, _, a3, _ = c17.front_end(rotated, obs, dt, cfg, optical)
            np.testing.assert_array_equal(at3, at)
            np.testing.assert_array_equal(release3, release)
            pred, _, _, on, _ = held(q3, at3, release3, h3, a3, rotated, obs, dt, k, fixed)
            mapped = Rotation.from_quat(q3[release3], scalar_first=True).inv()
            body = {name: float(c20.angles(r_release.inv().apply(full[name]), rot.inv().apply(mapped.apply(v))).max())
                    for name, v in pred.items()}
            rot_stats.append({"rotation": label, "fixed_world_response_max_abs": gauge_gap, "native_body_angle_max_deg": body,
                              "native_active_gate_changes": int((enabled != on).sum())})
        rows.append({"native_hz": hz, "releases": len(at), "delayed_publication_releases": len(late), "cuts": sorted(tests),
                     "exact_zero_gain_control": True, "future_and_truncation_max_angle_deg": differences, "rotations": rot_stats})
    return rows

def metrics(rows):
    okay = [r for r in rows if r["status"] == "scored"]
    macro = {m: fmean(r[m] for r in okay) for m in METHODS} if okay else {}
    return {"records": len(okay), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in okay),
            "macro_tilt_rmse_deg": macro, "C022_delta_deg": {m: macro["C022_deg"] - macro[m] for m in METHODS if m != "C022_deg"} if okay else {},
            "C022_wins": {m: sum(r["C022_deg"] < r[m] for r in okay) for m in METHODS if m != "C022_deg"}}


def controls(result, archived):
    difference = max(abs(result[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg"))
    base.require(np.isfinite(list(result.values())).all() and difference <= 1e-6, "Locked C020 and FullVQF no longer reproduce")
    return difference


def broad_scores(key, path, observed, ref, mask, rest, release, q0, q, estimates, official_errors):
    movement = loadmat(path, variable_names=("movement",))["movement"].ravel().astype(bool)
    labeled = np.isfinite(ref).all(axis=1) & (np.linalg.norm(ref, axis=1) > .5)
    np.testing.assert_array_equal(movement & observed & labeled, mask)
    ix = np.flatnonzero(mask)
    lookup = np.searchsorted(release, ix, side="right")
    native_rest = np.r_[False, rest][lookup]
    results = []
    for label, use in (("official_movement", np.ones(len(ix), dtype=bool)),
                       ("VQF_rest_within_official_movement", native_rest),
                       ("VQF_nonrest_within_official_movement", ~native_rest)):
        if use.any():
            results.append({"record": key[1], "stratum": label, "status": "scored", "scored_frames": int(use.sum()),
                            "scoring_indices_sha256": array_sha(ix[use], "<i8"), **c17.scores(official_errors, use)})
    still = np.flatnonzero(~movement & observed & labeled)
    if len(still):
        results.append({"record": key[1], "stratum": "raw_movement_flag_0_supplement_only", "status": "scored", "scored_frames": len(still),
                        "scoring_indices_sha256": array_sha(still, "<i8"), **c17.scores(c17.measure(still, ref, q0, q, release, estimates))})
    return results


def replay(prior, inventory, campaign, split, cfg, k, fixed):
    c1r, c1p, c3r, c2p = (read(ROOT / n) for n in ("C001/results.json", "C001/protocol.json", "C003/results.json", "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in src["sequences"]} for src in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "C022 replay leaves the frozen development panel")
    medium_path, array_path = DATA / "Sassari/medium_v5.mat", DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(medium_path) == split["source_file_sha256"][str(medium_path)]
                 and sha(array_path) == evidence["array_reference_sha256"], "Frozen reference at replay changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(array_path)
    poses_archive = {(a["record"], a["pose"]): a for a in prior["array_sensor_poses"]}
    rows, poses, broad, failures = [], [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for a in prior["sequences"]:
            tick = perf_counter()
            key = a["dataset"], a["record"]
            identity = {k: a[k] for k in c17.IDENTITY}
            try:
                pack, raw, observed, dt, ref, mask, axes, _ = c17.payload(key, specs[key], map3, map1, medium, yt["valid"],
                                                                          split, c1p, array, evidence, cfg)
                q0, q, at, release, hist, _, acc, _ = pack
                np.testing.assert_array_equal(mask, masks[a["mask_key"]])
                ix = np.flatnonzero(mask)
                base.require(len(ix) == a["scored_frames"] and array_sha(ix, "<i8") == a["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == a["native_input_last_sha256"]
                             and array_sha(release, "<i8") == a["native_release_sha256"], "Original C020 native score or release indices changed")
                start_pred = perf_counter()
                result, delta, rest, enabled, filter_seconds = held(q, at, release, hist, acc, raw, observed, dt, k, fixed)
                prediction_seconds = perf_counter() - start_pred
                gate = gates[a["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                estimates = {key: np.vstack((UP, arr)) for key, arr in result.items()}
                error = c17.measure(ix, ref, q0, q, release, estimates)
                score = c17.scores(error)
                gap = controls(score, a)
                lookup = np.searchsorted(release, ix, side="right")
                unchanged = ~np.r_[False, enabled][lookup]
                native_rest = np.r_[False, rest][lookup]
                np.testing.assert_array_equal(error["C022_deg"][unchanged], error["C020_deg"][unchanged])
                base.require(int(native_rest.sum()) == a["static_scoring_frames"], "Original C020 rest frames changed")
                detail = []
                for pose, use in axes.items():
                    sel = use[ix]
                    old = poses_archive[key[1], pose]
                    sig = array_sha(ix[sel], "<i8")
                    base.require(int(sel.sum()) == old["scored_frames"] and sig == old["scoring_indices_sha256"], "Array native pose scoring changed")
                    val = c17.scores(error, sel)
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(sel.sum()), "scoring_indices_sha256": sig,
                                   **val, "control_max_abs_delta_deg": controls(val, old)})
                if key[0] == "BROAD":
                    broad.extend(broad_scores(key, Path(specs[key]["path"]), observed, ref, mask, rest, release, q0, q, estimates, error))
                rows.append({**identity, **score, "status": "scored", "control_max_abs_delta_deg": gap,
                             "native_input_last_sha256": a["native_input_last_sha256"], "native_release_sha256": a["native_release_sha256"],
                             "original_rest_scoring_frames": int(native_rest.sum()), "active_releases": int(enabled.sum()),
                             "gated_scoring_frames": int(unchanged.sum()), "auxiliary_delta_sha256": array_sha(delta, "<f8"),
                             "max_added_tangent": float(np.linalg.norm(delta, axis=1).max()), "auxiliary_filter_seconds": filter_seconds,
                             "prediction_seconds": prediction_seconds, "seconds": perf_counter() - tick})
                poses.extend(detail)
                print("development", *key, *(score[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - tick})
                print("FAILED", *key, repr(exc), flush=True)
    seq, pose, groups, broad_file = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json", "BROAD-native-motion.csv"))
    columns = dict.fromkeys(k for r in rows for k in r)
    intrinsic.shice.save_csv(seq, [{k: r.get(k, "") for k in columns} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pose, poses)
    if broad:
        intrinsic.shice.save_csv(broad_file, broad)
    groups_out = {"BROAD_trial_motion": {g: metrics([x for x in rows if x["dataset"] == "BROAD" and x["group"] == g])
                                         for g in sorted({x["group"] for x in rows if x["dataset"] == "BROAD"})},
                  "BROAD_native_motion": {g: metrics([x for x in broad if x["stratum"] == g]) for g in sorted({x["stratum"] for x in broad})},
                  "Oxford_carry": {g: metrics([x for x in rows if x["dataset"] == "OxIOD" and x["group"] == g])
                                    for g in sorted({x["group"] for x in rows if x["dataset"] == "OxIOD"})},
                  "group": {f"{d}/{g}": metrics([x for x in rows if x["dataset"] == d and x["group"] == g])
                            for d, g in sorted({(x["dataset"], x["group"]) for x in rows})}}
    write(groups, groups_out)
    return {"pooled_records": metrics(rows), "domains": {d: metrics([x for x in rows if x["dataset"] == d]) for d in campaign["datasets"]},
            "BROAD_trial_motion": groups_out["BROAD_trial_motion"], "BROAD_native_motion": groups_out["BROAD_native_motion"],
            "Oxford_carry": groups_out["Oxford_carry"], "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifacts_sha256": {p.name: sha(p) for p in (seq, pose, groups, broad_file) if p.is_file()}}

def status(protocol, phase, **data):
    write(HERE / "process.json", {"status": phase, "pid": os.getpid(), "original_fit_pid": protocol["pid"],
                                  "started_at": protocol["started_at"], "updated_at": stamp(),
                                  "protocol_sha256": sha(HERE / "protocol.json"), **data})


def after_lock(protocol, fixed, campaign, inventory, cfg, recovery_sha=None):
    lock_path = HERE / "coefficient-lock.json"
    locked = read(lock_path)
    sig = sha(lock_path)
    base.require(locked["fit_count"] == 1 and locked["C020_branch_lock_sha256"] == sha(ROOT / "C020/branch-lock.json")
                 and locked["external_selection_sha256"] == sha(HERE / "external-selection.json")
                 and locked["training_slots_sha256"] == sha(HERE / "training-slots.npz")
                 and locked["protocol_sha256"] == sha(HERE / "protocol.json"), "C022 coefficient provenance mismatch")
    k = locked["k"]
    info = {"coefficient_lock_sha256": sig, "recovery_sha256": recovery_sha}
    status(protocol, "causal_rotation_audits", **info)
    audit_path = HERE / "causal-rotation-audit.json"
    write(audit_path, {**info, "checks": audit(cfg, k, fixed)})
    status(protocol, "development_replay", **info)
    print("C022 coefficient frozen, replaying 213 development records", flush=True)
    start = perf_counter()
    previous = read(ROOT / "C020/results.json")
    result = replay(previous, inventory, campaign, read(TRANSFER / "external-split.json"), cfg, k, fixed)
    base.require(sha(lock_path) == sig and all(sha(ROOT / n) == h for n, h in protocol["input_file_sha256"].items())
                 and all(sha(PILOT / n) == h for n, h in protocol["inherited_C020_files_sha256"].items()), "A fixed artifact changed")
    result.update({**info, "pid": os.getpid(), "original_fit_pid": protocol["pid"], "started_at": protocol["started_at"],
                   "completed_at": stamp(), "k": k, "protocol_sha256": sha(HERE / "protocol.json"),
                   "scoring_masks_sha256": sha(MASKS), "causal_rotation_audit_sha256": sha(audit_path),
                   "C020_branch_lock_sha256": locked["C020_branch_lock_sha256"],
                   "development_replay_seconds": perf_counter() - start,
                   "auxiliary_filter_seconds_summed_across_records": sum(r["auxiliary_filter_seconds"] for r in result["sequences"] if r["status"] == "scored"),
                   "final_holdout_scored": False})
    path = HERE / "results.json"
    write(path, result)
    good = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if good else "incomplete", **info, scored=result["pooled_records"]["records"], results_sha256=sha(path))
    print("C022 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not good:
        raise RuntimeError("Incomplete C022 development replay; coefficient remains locked")


def main():
    receipts = (("retry-launch.json", "retry-input-verification.json") if "--preflight-retry" in sys.argv[1:]
                else ("launch.json", "input-verification.json"))
    launch, init = (read(HERE / n) for n in receipts)
    files = init["files_sha256"]
    base.require(os.getpid() == launch["pid"] == init["pid"]
                 and all(sha(ROOT / n) == h for n, h in files.items()), "C022 registration or read-only files changed")
    p20, p17, k17, reviewed = (read(ROOT / x) for x in ("C020/protocol.json", "C017/protocol.json", "C017/coefficient-lock.json",
                                                       "review/auxiliary_filter_directions.json"))
    base.require(p20["development_records"] == p17["development_records"] and len(p20["development_records"]) == 213
                 and p20["scoring_masks_sha256"] == sha(MASKS) and p17["gate_traces_sha256"] == sha(GATES)
                 and reviewed["native_records"] == 139 and reviewed["slots"] == 15360, "C022 panel differs")
    review_path = ROOT / "review/auxiliary_filter_training_slots.npz"
    files[str(review_path.relative_to(ROOT))] = sha(review_path)
    protocol = {**launch, "code_sha256": sha(Path(__file__)), "input_file_sha256": files,
                "inherited_C020_files_sha256": p20["C017_dependencies_sha256"], "C017_k": k17["k"],
                "frozen": "All C020 and prior neural and stable force weights; same C003 bias estimation; no history or model training.",
                "candidate": "Same corrected acceleration and gyro to VQF tauAcc 1.5 and 6 s, compared with default tauAcc 3 at the native input-last.",
                "response": "Rotate each auxiliary body up into default world, project xy divided by max(z,1e-6), independently cap to tan(6 deg), mix 2:1.",
                "gate": "Original 64x10ms valid intrinsic frame and non-rest. Add k times mixed response before final normalization on native release.",
                "physical_ablation": "C003 front end and C017 fixed stable-force physical term plus gated C022 k term. All neural residuals removed.",
                "selection": "One certified scalar minimax in [0,1], endpoints included, smaller k for objective ties within C011 tolerance. Each record equal within its source; each sampled draw equal within record.",
                "development_records": p20["development_records"], "scoring_masks_sha256": sha(MASKS),
                "BROAD_support": "Headline uses existing native movement mask. Raw movement=0 valid-label rows are supplementary only, on the same 12 development trials."}
    protocol_path = HERE / "protocol.json"
    write(protocol_path, protocol)
    fixed, campaign = load_fixed(protocol)
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    inventory = read(ROOT / "C001/inventory.json")
    status(protocol, "external_training_input_preparation")
    data = c20.load_schedule(files)
    source, prep = source_slots(inventory, data, fixed, cfg)
    with np.load(review_path, allow_pickle=False) as archived:
        for key, old in (("truth", "truth"), ("enabled", "active"), ("source_record_ordinal", "record_ordinal"), ("domain", "domain"),
                         ("record_equal_weights", "record_equal_weights"), ("native_input_last", "input_last"), ("native_release", "release")):
            np.testing.assert_array_equal(source[key], archived[old])
        independent_errors = {key: float(np.max(np.abs(source[key] - archived[key]))) for key in ("center", "fast", "slow")}
    base.require(independent_errors["center"] <= 2e-7 and max(independent_errors["fast"], independent_errors["slow"]) <= 2e-7,
                 "Independent physical vector or C020 CPU center discrepancy")
    del data
    slopes = zero_slopes(source, reviewed)
    prep_path, slope_path = HERE / "training-preparation.json", HERE / "zero-slopes.json"
    write(prep_path, {**prep, "independent_array_max_abs_differences": independent_errors})
    write(slope_path, slopes)
    print("C022 independent zero point", {d: row["fractional_slopes"]["mixture"] for d, row in slopes.items()}, flush=True)
    status(protocol, "single_scalar_fit", training_preparation_sha256=sha(prep_path), zero_slopes_sha256=sha(slope_path))
    lock_path = HERE / "coefficient-lock.json"
    base.require(not lock_path.exists(), "The single scalar fit is already locked")
    k, fit = fit_coefficient(source, slopes)
    ext = HERE / "external-selection.json"
    write(ext, {"pid": os.getpid(), **fit, "zero_slopes_sha256": sha(slope_path), "training_preparation_sha256": sha(prep_path)})
    with lock_path.open("x", encoding="utf-8") as f:
        import json
        json.dump({"pid": os.getpid(), "locked_at": stamp(), "k": k, "fit_count": 1,
                   "protocol_sha256": sha(protocol_path), "C020_branch_lock_sha256": files["C020/branch-lock.json"],
                   "C019_sampling_lock_sha256": files["C019/sampling-lock.json"], "external_selection_sha256": sha(ext),
                   "training_slots_sha256": prep["array_sha256"]}, f, indent=2)
        f.write("\n")
    print("C022 locked", k, {d: v["loss_ratio"] for d, v in fit["domains"].items()}, flush=True)
    after_lock(protocol, fixed, campaign, inventory, cfg)


def resume():
    protocol, locked, state = (read(HERE / n) for n in ("protocol.json", "coefficient-lock.json", "process.json"))
    base.require(state["status"] != "complete" and locked["pid"] == protocol["pid"] == state["original_fit_pid"], "Only a previously locked incomplete C022 may resume")
    evidence = {f: sha(HERE / f) for f in ("failure.json", "failure-traceback.log", "results.json") if (HERE / f).is_file()}
    path = HERE / "replay-recovery.json"
    write(path, {"pid": os.getpid(), "original_fit_pid": locked["pid"], "started_at": stamp(), "fit_repeated": False,
                 "old_evidence_sha256": evidence, "coefficient_lock_sha256": sha(HERE / "coefficient-lock.json")})
    fixed, campaign = load_fixed(protocol)
    inventory = read(ROOT / "C001/inventory.json")
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    after_lock(protocol, fixed, campaign, inventory, cfg, sha(path))


if __name__ == "__main__":
    if "--resume" in sys.argv[1:]:
        resume()
    else:
        main()
