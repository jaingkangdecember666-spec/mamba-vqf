"""Repeat C004-B once with native release-hold direction targets."""

from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C006 = CAMPAIGN / "C006"
sys.path.insert(0, str(C006))

import replay_rest_gate as c6
import numpy as np
import torch
from scipy.io import loadmat
from tilt_eval import normalize, rotate, tilt_error
from torch.nn import functional as F


c5, c4, c3, c2 = c6.c5, c6.c4, c6.c3, c6.c2
base, intrinsic = c6.base, c6.intrinsic
read, write, stamp, digest = c6.read, c6.write, c6.stamp, c6.digest
C001, C002, C003, C004, C005, DATA, PILOT, TRANSFER = c6.C001, c6.C002, c6.C003, c6.C004, c6.C005, c6.DATA, c6.PILOT, c6.TRANSFER
METHODS = ("full_vqf_deg", "c003_deg", "C006_deg", "C007_deg")


def unpack_targets(q, index, release, score, ref):
    ends = np.r_[release[1:], len(q)]
    eligible = np.flatnonzero(score)
    world = np.zeros((len(score), 3), dtype=np.float64)
    world[eligible] = rotate(normalize(q[eligible]), ref[eligible])
    clock = np.r_[0, np.cumsum(score, dtype=np.int64)]
    counts = clock[ends] - clock[release]
    total = np.vstack((np.zeros((1, 3)), np.cumsum(world, axis=0, dtype=np.float64)))
    means = (total[ends] - total[release]) / counts.clip(min=1)[:, None]
    orientation = normalize(q[index])
    inverse = orientation.copy()
    inverse[:, 1:] *= -1.
    return world, ends, counts, means, orientation, inverse


def training_source(spec, prior, yt, mats, core, mean, std, cfg):
    raw, observed, dt, score, ref, hashes = c4.train_raw(spec, yt, mats)
    q, index, release, windows, complete, bias = c4.inertial_replay(raw, observed, dt, cfg, training=True)
    eligible = complete & score[release]
    eligible_pos = np.flatnonzero(eligible)
    world, ends, counts, means, origin, inverse = unpack_targets(q, index, release, score, ref)
    source_release = release[eligible_pos]
    base.require(len(source_release) and np.all(counts[eligible_pos] > 0)
                 and np.all(ends > release) and ends[-1] == len(q), "Invalid training hold boundaries")
    point = rotate(inverse[eligible_pos], world[source_release])
    target = rotate(inverse[eligible_pos], means[eligible_pos])
    err = float(np.max(np.abs(rotate(origin[eligible_pos], target) - means[eligible_pos])))
    base.require(err < 2e-12, "Hold target world round trip failed")
    latent, up, frame = c4.features(core, windows[eligible_pos], mean, std)
    keep = frame.valid
    frames_used = eligible_pos[keep]
    base.require(keep.any() and len(frames_used) == prior["usable"] and len(source_release) == prior["eligible_windows"]
                 and hashes == prior["raw_hashes"] and bias["active_native_frames"] == prior["active_native_frames"],
                 f"C004 feature or raw-source eligibility changed: {spec['record']}")
    point_local = intrinsic.transform_directions(point, frame)[keep]
    target_local = intrinsic.transform_directions(target, frame)[keep]
    strength = np.linalg.norm(means[frames_used], axis=1)
    base.require(np.isfinite(strength).all() and (strength <= 1. + 2e-7).all(), "Invalid unnormalized hold target")
    differences = tilt_error(point[keep], target[keep])
    spread = np.sqrt(np.maximum(0., 1. - strength**2))
    selected_counts = counts[frames_used]
    audit = {**spec, "raw_hashes": hashes, "eligible_point_windows": len(eligible_pos), "used_windows": len(frames_used),
             "scoring_frames_across_used_hold_targets": int(selected_counts.sum()), "one_frame_hold_targets": int((selected_counts == 1).sum()),
             "labels_per_hold_min": int(selected_counts.min()), "labels_per_hold_max": int(selected_counts.max()),
             "labels_per_hold_mean": float(selected_counts.mean()),
             "point_to_hold_rms_deg": float(np.sqrt(np.mean(differences**2))),
             "point_to_hold_p95_deg": float(np.quantile(differences, .95)), "point_to_hold_max_deg": float(differences.max()),
             "mean_target_norm": float(strength.mean()), "minimum_target_norm": float(strength.min()),
             "within_hold_mean_chord_dispersion": float(spread.mean()),
             "world_roundtrip_max_abs": err, "last_window_ends_exclusive": int(ends[-1])}
    features = np.concatenate((latent[keep], up[keep]), axis=1)
    # Retain a real interval for an independent framewise objective check.
    probe = np.flatnonzero(keep)[len(frames_used) // 2]
    k = eligible_pos[probe]
    a, b = release[k], ends[k]
    explicit = rotate(np.broadcast_to(inverse[k], (int(counts[k]), 4)), world[a:b][score[a:b]]) @ frame.basis[probe]
    base.require(len(explicit) == int(counts[k]), "Explicit frame list does not match segment denominator")
    np.testing.assert_allclose(explicit.mean(axis=0), target_local[len(frames_used) // 2], rtol=0., atol=6e-8)
    probe = {"domain": spec["domain"], "record": spec["record"], "target_frames": len(explicit),
             "x": latent[probe].astype(np.float64), "up": up[probe].astype(np.float64), "directions": explicit}
    return (np.c_[features, target_local], np.c_[features, point_local], release[frames_used], audit,
            probe, (raw, observed, dt))


def gradients_equivalent(initial, probes, width):
    weight = initial.weight.detach().cpu().double().requires_grad_(True)
    bias = initial.bias.detach().cpu().double().requires_grad_(True)
    samples = [*probes, {**probes[0], "domain": "synthetic_singleton", "record": "one_valid_frame",
                         "directions": np.array([[0., 0., 1.]]), "target_frames": 1}]
    rows = []
    for p in samples:
        x, up, truth = (torch.as_tensor(p[k], dtype=torch.float64) for k in ("x", "up", "directions"))
        pred = F.normalize(up + c4.GAIN * base.tangent(F.linear(x.reshape(1, width), weight, bias),
                                                       up.reshape(1, 3), base.CAP)[0], dim=0)
        framewise = (pred[None, :] - truth).square().sum(dim=1).mean()
        target = truth.mean(dim=0)
        averaged = (pred - target).square().sum()
        constant = truth.square().sum(dim=1).mean() - target.square().sum()
        ga = torch.autograd.grad(framewise, (weight, bias), retain_graph=True)
        gb = torch.autograd.grad(averaged, (weight, bias))
        grad_diff = max(float((a - b).abs().max()) for a, b in zip(ga, gb, strict=True))
        shifted_loss = float((framewise - averaged - constant).detach().abs())
        base.require(grad_diff <= 1e-12 and shifted_loss <= 1e-12, "Hold objective gradients differ from native-frame objective")
        rows.append({"domain": p["domain"], "record": p["record"], "target_frames": p["target_frames"],
                     "parameter_gradient_max_abs_difference": grad_diff, "loss_minus_constant_abs_difference": shifted_loss})
    return rows


@torch.inference_mode()
def initial_point_control(initial, point_data, indices, old_first, device):
    x, up, target = torch.from_numpy(point_data[indices]).to(device).split((initial.in_features, 3, 3), dim=1)
    prediction = F.normalize(up + c4.GAIN * base.tangent(initial(x), up, base.CAP), dim=1)
    loss = float((prediction - target).square().sum(dim=1).mean())
    gap = abs(loss - float(old_first["loss"]))
    base.require(gap <= 1e-10, "Could not reproduce C004-B initialization and point-target loss")
    return {"step": 1, "recomputed_point_loss": loss, "archived_C004_B_loss": float(old_first["loss"]), "abs_difference": gap}


def causal_check(probes, core, layer, mean, std, cfg):
    rows = []
    for domain, (raw, valid, dt) in probes.items():
        q, idx, release, x, _, _ = c4.inertial_replay(raw, valid, dt, cfg)
        center = len(release) // 2
        n = int(release[center]) + 1
        corrupted = raw.copy()
        corrupted[n:, :3] += np.array([1., 2., -3.])
        corrupted[n:, 3:] -= .8
        _, ix, when, hx, _, _ = c4.inertial_replay(corrupted, valid, dt, cfg)
        _, cut_idx, cut_when, cut_hist, _, _ = c4.inertial_replay(raw[:n], valid[:n], dt, cfg)
        np.testing.assert_array_equal(cut_when, release[:center + 1])
        np.testing.assert_array_equal(when[:center + 1], cut_when)
        np.testing.assert_array_equal(ix[:center + 1], cut_idx)
        np.testing.assert_array_equal(cut_idx, idx[:center + 1])
        np.testing.assert_array_equal(hx[:center + 1], cut_hist)
        np.testing.assert_array_equal(cut_hist, x[:center + 1])
        deltas = []
        for window in (x[center - 3:center + 1], hx[center - 3:center + 1], cut_hist[-4:]):
            d = c4.evaluate_windows(core, {"new": layer}, window, mean, std)["new"]
            deltas.append(np.where(c6.rest_flags(window)[:, None], 0., c4.GAIN * d))
        np.testing.assert_array_equal(deltas[0], deltas[1])
        np.testing.assert_array_equal(deltas[0], deltas[2])
        rows.append({"domain": domain, "last_checked_release": n - 1, "same_input_prefix": True,
                     "same_predictions_truncation_and_future_mutation": True})
    return rows


def evaluate(prior, spec, map3, map1, medium, yt, split, c1p, array, hashes, old_masks, old_gate,
             core, layers, mean, std, cfg):
    key = (prior["dataset"], prior["record"])
    if prior["dataset"] in ("OxIOD", "RepoIMU"):
        raw_q, corrected_q, at, release, histories, mask, ref, geometry, _, _ = c5.optical_payload(spec, cfg)
        raw_q, corrected_q = raw_q.as_quat(scalar_first=True), corrected_q.as_quat(scalar_first=True)
        poses = {}
        base.require(geometry["quality_flags"] == prior["reference_quality_flags"], "Optical QA classification changed")
    else:
        sample, dt, observed, ref, mask, poses = c3.sources(map3[key], spec, map1.get(key), medium, yt, split, c1p, array, hashes)
        clean = base.causal_fill(sample, observed)
        raw_q = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
        corrected_q, at, release, histories, _, _ = c4.inertial_replay(sample, observed, dt, cfg)
    np.testing.assert_array_equal(mask, old_masks[prior["mask_key"]])
    base.require(intrinsic.mask_digest(mask) == prior["scoring_indices_sha256"], "C006 mask identity changed")
    previous_gate = old_gate[prior["mask_key"]]
    flag = c6.rest_flags(histories)
    np.testing.assert_array_equal(release, previous_gate[:, 0])
    np.testing.assert_array_equal(flag, previous_gate[:, 3])
    delta = c4.evaluate_windows(core, layers, histories, mean, std)
    for key_gate in ("C006", "C007"):
        delta[key_gate][flag] = 0.
    candidates = {name: (name, c4.GAIN, False) for name in delta}

    def compute(use):
        pure = intrinsic.score_attitudes(raw_q, ref, use, raw_q[at], release, {}, {}, 1.)["full_vqf"]
        scores = intrinsic.score_attitudes(corrected_q, ref, use, corrected_q[at], release, delta, candidates, 1.)
        return {"full_vqf_deg": pure, **{k + "_deg": scores[k] for k in delta}}

    scores = compute(mask)
    errors = {m: scores[m] - prior[o] for m, o in (("full_vqf_deg", "full_vqf_deg"), ("c003_deg", "c003_deg"),
                                                   ("C006_deg", "B_rest_gated_deg"))}
    base.require(np.isfinite(list(scores.values())).all() and max(map(abs, errors.values())) <= 1e-6,
                 f"C006 controls did not reproduce: {errors}")
    fields = ("dataset", "record", "group", "reference_status", "scored_frames", "mask_key",
              "scoring_indices_sha256", "instrument", "scenario")
    row = {**{k: prior[k] for k in fields}, **scores, "status": "scored",
           "reference_quality_flags": prior.get("reference_quality_flags", []),
           "rest_scored_fraction": prior["rest"]["rest_scored_fraction"], "controls_minus_prior_deg": errors}
    per_pose = []
    for axis, pose in poses.items():
        use = mask & pose
        per_pose.append({"record": prior["record"], "pose": axis, "scored_frames": int(use.sum()),
                         "scoring_indices_sha256": intrinsic.mask_digest(use), **compute(use)})
    return row, per_pose


def summarize(rows, denominator):
    out = {"records": len(rows), "denominator": denominator}
    if rows:
        values = {k: float(np.mean([r[k] for r in rows])) for k in METHODS}
        out.update({"scored_frames": sum(r["scored_frames"] for r in rows), "macro_rmse_deg": values,
                    "C007_wins_vs": {m: sum(r["C007_deg"] < r[m] for r in rows) for m in METHODS[:-1]},
                    "C007_delta_macro_deg": {m: values["C007_deg"] - values[m] for m in METHODS[:-1]}})
    return out


def main():
    started = stamp()
    protocol_path = HERE / "protocol.json"
    base.require(not protocol_path.exists(), "C007 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p4, r4, t4, prep4 = (read(C004 / p) for p in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    r6, p6 = (read(C006 / p) for p in ("results.json", "protocol.json"))
    c3p = read(C003 / "protocol.json")
    cfg = c3.Settings(**c3p["thresholds"])
    inv = read(C001 / "inventory.json")
    schedule_file = C004 / "schedules.npz"
    old_masks, old_gate = C005 / "scoring-masks.npz", C006 / "gate-traces.npz"
    base.require(len(r6["sequences"]) == 213 and not r6["failures"]
                 and digest(C006 / "protocol.json") == r6["protocol_sha256"]
                 and digest(old_masks) == r6["shared_masks_sha256"] and digest(old_gate) == r6["gate_traces_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_file) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(C004 / "training-steps.csv") == t4["training_steps_csv_sha256"]
                 and digest(C003 / "online_bias.py") == c3p["estimator_sha256"], "Frozen C004 or C006 evidence changed")
    base.require(all(digest(c5.REVIEW / name) == value for name, value in p6["optical_reference_sha256"].items()),
                 "Optical reference sources changed")
    records = p4["source_records"]
    archived_training = prep4["records"]
    base.require(len(records) == len(archived_training) == 63 and len(p4["yt_prefix_exclusion"]["selected_prefixes"]) == 24
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "The C004-B source panel changed")
    for spec, saved in zip(records, archived_training, strict=True):
        base.require(all(saved[k] == v for k, v in spec.items()), "C004 training source identity changed")
    incumbent = PILOT / campaign["incumbent"]
    bpath = C004 / t4["arms"]["B"]["weights"]
    base.require(digest(incumbent) == p4["base_checkpoint_sha256"] == campaign["incumbent_sha256"] == p6["B6_checkpoint_sha256"]
                 and digest(bpath) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"],
                 "Initial or predecessor weights changed")
    base.require((p4["seed"], p4["steps_per_arm"], p4["batch_size"], p4["lr"], p4["weight_decay"], p4["inference_gain"]) ==
                 (c4.SEED, c4.STEPS, c4.BATCH, c4.LR, c4.WD, c4.GAIN), "C004 optimizer settings changed")
    with np.load(schedule_file, allow_pickle=False) as archived:
        schedule = archived["B"]
        offsets = archived["record_offsets"]
        old_releases = archived["release_indices"]
    base.require(schedule.shape == (160, 96) and len(offsets) == len(records) + 1, "C004-B sequence schedule changed")
    protocol = {"started_at": started, "pid": os.getpid(), "source_records": records, "yt_prefix_exclusion": p4["yt_prefix_exclusion"],
                "C004_training_protocol_sha256": digest(C004 / "protocol.json"), "C006_protocol_sha256": digest(C006 / "protocol.json"),
                "initial_checkpoint_sha256": digest(incumbent), "C004_B_schedule_file_sha256": digest(schedule_file),
                "B_schedule_matrix_sha256": __import__("hashlib").sha256(schedule.astype("<i8").tobytes()).hexdigest(),
                "initial_output_parameter_sha256": p4["initial_B_output_parameter_sha256"],
                "frozen_backbone_sha256": p4["frozen_backbone_parameter_sha256"],
                "frozen_normalization_sha256": p4["frozen_normalization_sha256"],
                "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH, "gain": c4.GAIN,
                "optimizer": {"kind": p4["optimizer"], "lr": c4.LR, "weight_decay": c4.WD, "grad_clip": 2.},
                "scope": "One C004-B initialization fit, 75 output parameters, only the target changes.",
                "target": "Per eligible release, each eligible native reference direction in [release,next_release) is moved into VQF world with q at that native frame. Average uniformly, transform by inverse q at input last native sample and then the frozen window's intrinsic basis. Leave mean unnormalized. Final exclusive bound is native record length; one valid native frame is sufficient.",
                "loss": "Same C004-B squared direction distance at fixed gain 0.25, with unnormalized mean target. Equal draw weights and equal eligible native frames within each sampled hold interval.",
                "inference": "Same causal C003 accelerometer front end, C006 last-block VQF restDetected gate and fixed gain 0.25 outside rest.",
                "validation_records": [[r["dataset"], r["record"]] for r in r6["sequences"]],
                "validation_masks_sha256": digest(old_masks)}
    write(protocol_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "status": "running",
                                  "protocol_sha256": digest(protocol_path)})
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    frozen, old_b = (torch.load(path, map_location="cpu", weights_only=True) for path in (incumbent, bpath))
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    core.eval().requires_grad_(False).to(device)
    bhead = deepcopy(core)
    bhead.load_state_dict(old_b["state_dict"])
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    base.require(c4.tensor_hash(core.output.state_dict()) == protocol["initial_output_parameter_sha256"]
                 and c4.tensor_hash({"mean": mean, "std": std}) == protocol["frozen_normalization_sha256"]
                 and c4.tensor_hash({k: v for k, v in core.state_dict().items() if not k.startswith("output.")}) ==
                 protocol["frozen_backbone_sha256"], "Initial tensors did not match C004-B")
    split_path = TRANSFER / "external-split.json"
    base.require(digest(split_path) == p4["source_split_sha256"], "Training split changed")
    split = read(split_path)
    memos, training, control, target_stats, grad_probes, causal_probes = {}, [], [], [], {}, {}
    with c2.c1.h5py.File(inv["yt"]["path"], "r") as yt:
        for i, spec in enumerate(records):
            data, point, releases, audit, grad_probe, causal_probe = training_source(
                spec, archived_training[i], yt["train"], memos, core, mean, std, cfg)
            np.testing.assert_array_equal(releases, old_releases[offsets[i]:offsets[i + 1]])
            training.append(data)
            control.append(point)
            target_stats.append(audit)
            grad_probes.setdefault(spec["domain"], grad_probe)
            causal_probes.setdefault(spec["domain"], causal_probe)
    target_path = HERE / "target-stats.json"
    write(target_path, target_stats)
    with (C004 / "training-steps.csv").open(newline="", encoding="utf-8") as stream:
        import csv
        old_first = next(r for r in csv.DictReader(stream) if r["arm"] == "B" and r["step"] == "1")
    point_parity = initial_point_control(core.output, np.concatenate(control), schedule[0], old_first, device)
    grad_audit = gradients_equivalent(core.output, list(grad_probes.values()), core.output.in_features)
    write(HERE / "training-target-audit.json", {"first_C004_B_point_loss": point_parity, "loss_gradient_checks": grad_audit,
                                                "identical_C004_native_release_indices": True})
    print("target and C004-B schedule checks passed", point_parity, flush=True)
    data = torch.from_numpy(np.concatenate(training)).to(device)
    del control, training, memos
    head, history, fit_times = c4.optimize("C007", core.output, data, schedule, core.output.in_features, device)
    del data
    checkpoint = HERE / "hold-readout.pt"
    tensor_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    tensor_state.update({"output." + k: v.detach().cpu() for k, v in head.state_dict().items()})
    stage = checkpoint.with_suffix(".tmp")
    torch.save({**frozen, "state_dict": tensor_state, "c007_seed": c4.SEED, "c007_steps": c4.STEPS,
                "c007_gain": c4.GAIN, "protocol_sha256": digest(protocol_path)}, stage)
    stage.replace(checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    changed = set(k for k, v in saved["state_dict"].items() if not torch.equal(v, frozen["state_dict"][k]))
    base.require(changed == {"output.weight", "output.bias"}
                 and all(torch.equal(saved[k], frozen[k]) for k in ("mean", "std")),
                 "Unexpected frozen parameter change")
    head.load_state_dict({k.removeprefix("output."): v for k, v in saved["state_dict"].items() if k.startswith("output.")})
    layers = {"c003": core.output, "C006": bhead.output, "C007": head}
    history_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(history_path, history)
    causal = causal_check(causal_probes, core, head, mean, std, cfg)
    write(HERE / "prefix-audit.json", causal)
    training_record = {"fit": fit_times, "fits": 1, "weights_sha256": digest(checkpoint),
                       "training_steps_sha256": digest(history_path), "output_sha256": c4.tensor_hash(head.state_dict()),
                       "target_stats_sha256": digest(target_path), "target_audit_sha256": digest(HERE / "training-target-audit.json"),
                       "prefix_audit_sha256": digest(HERE / "prefix-audit.json")}
    write(HERE / "training.json", training_record)
    print("C007 fixed-step checkpoint persisted; starting the 213 frozen development scores", flush=True)

    c1r, c1p = (read(C001 / f) for f in ("results.json", "protocol.json"))
    c3r = read(C003 / "results.json")
    evidence = read(C002 / "protocol.json")["frozen_evidence"]
    map1 = {(r["dataset"], r["record"]): r for r in c1r["sequences"]}
    map3 = {(r["dataset"], r["record"]): r for r in c3r["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inv["records"] if r["c001"] != "none"
             or (r["dataset"] in ("OxIOD", "RepoIMU", "IMU-Array") and r["side"] == "development")}
    base.require(len(specs) == 213 and set(specs) == {(r["dataset"], r["record"]) for r in r6["sequences"]}, "Scope changed")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari development data changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref_path) == evidence["array_reference_sha256"], "Array calibration reference changed")
    array = c2.reference_array(ref_path)
    old_poses = {(r["record"], r["pose"]): r for r in r6["array_sensor_poses"]}
    validation_at = stamp()
    rows, failures, poses = [], [], []
    with c2.c1.h5py.File(inv["yt"]["path"], "r") as yt, np.load(old_masks, allow_pickle=False) as masks, np.load(old_gate, allow_pickle=False) as traces:
        for prior in r6["sequences"]:
            key = prior["dataset"], prior["record"]
            try:
                row, details = evaluate(prior, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                        evidence["array_raw_sha256"], masks, traces, core, layers, mean, std, cfg)
                for detail in details:
                    old = old_poses[detail["record"], detail["pose"]]
                    base.require(detail["scoring_indices_sha256"] == old["scoring_indices_sha256"]
                                 and abs(detail["C006_deg"] - old["B_rest_gated_deg"]) <= 1e-6, "C006 pose mask or value changed")
                rows.append(row)
                poses.extend(details)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, ValueError, OSError, RuntimeError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    csv_path, pose_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv"
    fields = ("dataset", "record", "group", "status", "reference_status", "scored_frames", "mask_key", "scoring_indices_sha256",
              "instrument", "scenario", "rest_scored_fraction", *METHODS)
    row_index = {(r["dataset"], r["record"]): r for r in rows}
    failure_index = {(r["dataset"], r["record"]): r["error"] for r in failures}
    intrinsic.shice.save_csv(csv_path, [{**{k: row_index.get((r["dataset"], r["record"]), {}).get(k, r.get(k, ""))
                                         if k not in METHODS else row_index.get((r["dataset"], r["record"]), {}).get(k, "")
                                         for k in fields if k != "status"},
        "status": "failed" if (r["dataset"], r["record"]) in failure_index else "scored",
        "error": failure_index.get((r["dataset"], r["record"]), "")} for r in r6["sequences"]])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    panels = {d: summarize([r for r in rows if r["dataset"] == d], r6["summary"][d]["denominator"])
              for d in campaign["datasets"]}
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_status"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_status"] in ("verified", "geometry_verified_relative_level")]}
    completed = stamp()
    errors = [abs(value) for row in rows for value in row["controls_minus_prior_deg"].values()]
    write(HERE / "results.json", {"started_at": started, "validation_started_at": validation_at, "completed_at": completed,
          "pid": os.getpid(), "protocol_sha256": digest(protocol_path), "training_sha256": digest(HERE / "training.json"),
          "training": training_record, "scoring_masks_sha256": digest(old_masks), "csv_sha256": digest(csv_path),
          "pose_csv_sha256": digest(pose_path) if poses else None, "maximum_absolute_control_error_deg": max(errors, default=None),
          "summary": panels, "optical_subsets": {k: summarize(v, r6["optical_subsets"][k]["denominator"]) for k, v in subsets.items()},
          "sequences": rows, "array_sensor_poses": poses, "failures": failures})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": completed,
                                  "status": "complete" if not failures else "partial", "protocol_sha256": digest(protocol_path)})
    print(json.dumps({"summary": panels, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
