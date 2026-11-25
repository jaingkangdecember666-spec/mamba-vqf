"""Fit the whole B6 Mamba once, with the original C004-B point targets and draws."""

import csv
from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import inspect
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN / "C007"))

import fit_hold_targets as c7
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from tilt_eval import normalize, rotate, tilt_error
from torch.nn import functional as F

c6, c5, c4, c3, c2 = c7.c6, c7.c5, c7.c4, c7.c3, c7.c2
base, intrinsic = c7.base, c7.intrinsic
read, write, stamp, digest = c7.read, c7.write, c7.stamp, c7.digest
C001, C002, C003, C004, C005, C006 = c7.C001, c7.C002, c7.C003, c7.C004, c7.C005, c7.C006
DATA, PILOT, TRANSFER = c7.DATA, c7.PILOT, c7.TRANSFER
METHODS = ("full_vqf_deg", "c003_deg", "C006_deg", "C008_deg")
BACKBONE_LR = c4.LR * .1
MASKS = C005 / "scoring-masks.npz"
GATES = C006 / "gate-traces.npz"


def array_sha(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def point_windows(records, archived, schedule, offsets, old_releases, train, cfg):
    unique, draw_index = np.unique(schedule.ravel(), return_inverse=True)
    slices = np.searchsorted(unique, offsets)
    matrices, targets, audit, causal, rotation = [], [], [], {}, []
    mats = {}
    for i, (spec, prior) in enumerate(zip(records, archived, strict=True)):
        raw, observed, dt, score, truth, hashes = c4.train_raw(spec, train, mats)
        q, index, release, history, complete, bias = c4.inertial_replay(raw, observed, dt, cfg, training=True)
        eligible = complete & score[release]
        index, release, history = index[eligible], release[eligible], history[eligible]
        source = normalize(q[index])
        inverse = source.copy()
        inverse[:, 1:] *= -1.
        reference_world = rotate(normalize(q[release]), truth[release])
        target = rotate(inverse, reference_world)
        err = float(np.max(np.abs(rotate(source, target) - reference_world)))
        frame = intrinsic.build_frame(history)
        valid = np.flatnonzero(frame.valid)
        np.testing.assert_array_equal(release[valid], old_releases[offsets[i]:offsets[i + 1]])
        base.require(err <= 2e-12 and hashes == prior["raw_hashes"]
                     and bias["active_native_frames"] == prior["active_native_frames"]
                     and len(history) == prior["eligible_windows"] and len(valid) == prior["usable"],
                     f"Point-source identity changed for {spec['record']}")
        positions = unique[slices[i]:slices[i + 1]] - offsets[i]
        picked = valid[positions]
        selected = type(frame)(frame.basis[picked], frame.source[picked], frame.anchor[picked])
        canonical = intrinsic.transform_inputs(history[picked], selected)
        local = intrinsic.transform_directions(target[picked], selected)
        recovered = np.einsum("ni,nji->nj", local, selected.basis)
        np.testing.assert_allclose(recovered, target[picked], rtol=0, atol=1e-7)
        matrices.append(canonical)
        targets.append(local)
        sample = valid[np.linspace(0, len(valid) - 1, min(32, len(valid)), dtype=int)]
        rotation.append((spec, history[sample].copy()))
        causal.setdefault(spec["domain"], (raw, observed, dt))
        audit.append({**spec, "raw_hashes": hashes, "eligible_windows": len(history),
                      "usable": len(valid), "B_draws": int(((schedule >= offsets[i]) & (schedule < offsets[i+1])).sum()),
                      "distinct_drawn_windows": len(picked), "drawn_release_indices_sha256": array_sha(release[picked], "<i8"),
                      "canonical_inputs_sha256": array_sha(canonical, "<f4"),
                      "point_targets_sha256": array_sha(local, "<f4"), "point_world_roundtrip_max_abs": err})
    canonical, target = np.concatenate(matrices), np.concatenate(targets)
    draw_index = draw_index.reshape(schedule.shape)
    np.testing.assert_array_equal(unique[draw_index], schedule)
    base.require(len(canonical) == len(target) == len(unique) and sum(x["B_draws"] for x in audit) == schedule.size,
                 "Remapped inputs or labels changed the number of draws")
    return canonical, target, draw_index, audit, causal, rotation


def prediction(model, window, mean, std):
    up = window[:, -1, 6:9]
    delta = base.tangent(model(base.normalized(window, mean, std)), up, base.CAP)
    return F.normalize(up + c4.GAIN * delta, dim=1)


def fit(model, canonical, target, draw_index, mean, std, archived_first):
    torch.manual_seed(c4.SEED)
    named = dict(model.named_parameters())
    backbone = [p for key, p in named.items() if not key.startswith("output.")]
    head = list(model.output.parameters())
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": BACKBONE_LR},
                                   {"params": head, "lr": c4.LR}], lr=c4.LR, weight_decay=c4.WD)
    history = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    started = stamp()
    for step, idx in enumerate(draw_index, 1):
        optimizer.zero_grad(set_to_none=True)
        loss = (prediction(model, canonical[idx], mean, std) - target[idx]).square().sum(dim=1).mean()
        number = float(loss.detach())
        if step == 1:
            base.require(abs(number - archived_first) <= 1e-10, f"C004-B initial point loss changed: {number:.15g}")
        loss.backward()
        base.require(all(p.grad is not None for p in named.values()), "A Mamba parameter is disconnected")
        norms = torch.stack([p.grad.detach().norm() for p in (*backbone, *head)])
        trunk_norm, head_norm = (float(x) for x in (norms[:len(backbone)].norm(), norms[len(backbone):].norm()))
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 2.))
        base.require(np.isfinite([number, trunk_norm, head_norm, norm]).all() and trunk_norm > 0. and head_norm > 0.,
                     f"Invalid real backprop at step {step}")
        optimizer.step()
        row = {"arm": "C008", "step": step, "loss": number, "gradient_norm": norm,
               "backbone_gradient_l2": trunk_norm, "output_gradient_l2": head_norm}
        history.append(row)
        if step in (1, 40, 80, 120, c4.STEPS):
            print("training", row, flush=True)
    torch.cuda.synchronize()
    return model.eval().requires_grad_(False), history, {"started_at": started, "completed_at": stamp(),
            "seconds": perf_counter() - start, "optimizer_steps": len(history),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()}


def layer_differences(initial, final):
    by_layer = {}
    for name, old in initial.items():
        layer = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
        by_layer.setdefault(layer, []).append(name)
    rows = []
    for layer, names in by_layer.items():
        a = torch.cat([initial[k].detach().cpu().flatten().double() for k in names])
        b = torch.cat([final[k].detach().cpu().flatten().double() for k in names])
        change = b - a
        rows.append({"layer": layer, "parameters": a.numel(), "initial_l2": float(a.norm()),
                     "final_l2": float(b.norm()), "difference_l2": float(change.norm()),
                     "difference_max_abs": float(change.abs().max()), "changed_parameters": int(torch.count_nonzero(change))})
    base.require(all(row["difference_l2"] > 0 for row in rows), "An expected trainable layer did not move")
    return rows


@torch.inference_mode()
def final_prefix(model, causal, mean, std, cfg):
    probes = c7.causal_check(causal, model, model.output, mean, std, cfg)
    for row in probes:
        raw, observed, dt = causal[row["domain"]]
        window_end = row["last_checked_release"] + 1
        windows = c4.inertial_replay(raw[:window_end], observed[:window_end], dt, cfg)[3][-4:]
        x = torch.from_numpy(intrinsic.transform_inputs(windows, intrinsic.build_frame(windows))).to(mean.device)
        x = base.normalized(x, mean, std)
        cutoff = 37
        original = model(x, all_steps=True)[:, :cutoff]
        altered = x.clone()
        altered[:, cutoff:] *= -5.
        trunc_err = float((original - model(x[:, :cutoff], all_steps=True)).abs().max())
        future_err = float((original - model(altered, all_steps=True)[:, :cutoff]).abs().max())
        base.require(max(trunc_err, future_err) <= 2e-5, "Inner Mamba prefix changed")
        row.update({"inner_steps_checked": cutoff, "inner_truncation_max_abs": trunc_err,
                    "inner_future_mutation_max_abs": future_err})
    return probes


def rotated_audit(model, rotation, mean, std):
    rotations = [*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))]
    records, all_errors = [], []
    for spec, raw in rotation:
        frame = intrinsic.build_frame(raw)
        flags = c6.rest_flags(raw)
        up = normalize(raw[:, -1, 6:9].astype(np.float64))
        delta = intrinsic.predict_delta(model, raw, mean, std, frame=frame)
        corrected = normalize(up + c4.GAIN * np.where(flags[:, None], 0., delta))
        errors, switches = [], []
        for change in rotations:
            moved = raw.copy()
            moved[..., :12] = change.apply(raw[..., :12].reshape(-1, 3)).reshape(raw.shape[0], raw.shape[1], 12)
            moved_frame = intrinsic.build_frame(moved)
            np.testing.assert_array_equal(c6.rest_flags(moved), flags)
            moved_delta = intrinsic.predict_delta(model, moved, mean, std, frame=moved_frame)
            moved_up = normalize(moved[:, -1, 6:9].astype(np.float64))
            recovered = change.inv().apply(normalize(moved_up + c4.GAIN * np.where(flags[:, None], 0., moved_delta)))
            errors.append(tilt_error(corrected, recovered).tolist())
            switches.append(int(np.count_nonzero((frame.source != moved_frame.source) | (frame.anchor != moved_frame.anchor))))
        all_errors.append(np.asarray(errors))
        records.append({**spec, "windows": len(raw), "rest_windows": int(flags.sum()),
                        "anchor_or_source_switches": switches, "rotation_errors_deg": errors})
    pooled = np.concatenate(all_errors, axis=1)
    base.require(np.isfinite(pooled).all(), "Nonfinite rotation audit")
    return {"windows": pooled.shape[1], "comparisons": pooled.size, "passed_at_1e_4_deg": bool(pooled.max() <= .0001),
            "rotation_labels": ["90deg_x", "90deg_y", "90deg_z", "63deg_about_(1,2,-1)"],
            "rms_deg": float(np.sqrt(np.mean(pooled**2))), "p95_deg": float(np.quantile(pooled, .95)),
            "max_deg": float(pooled.max()), "records": records}


def evaluate(prior, spec, map3, map1, medium, yt, split, c1p, array, hashes, masks, gates,
             core, layers, tuned, mean, std, cfg):
    d, name = key = prior["dataset"], prior["record"]
    if d in ("OxIOD", "RepoIMU"):
        raw_q, corrected_q, at, release, history, mask, ref, qa, _, _ = c5.optical_payload(spec, cfg)
        raw_q, corrected_q = raw_q.as_quat(scalar_first=True), corrected_q.as_quat(scalar_first=True)
        poses = {}
        base.require(qa["quality_flags"] == prior["reference_quality_flags"], f"Optical QA changed for {name}")
    else:
        raw, dt, observed, ref, mask, poses = c3.sources(map3[key], spec, map1.get(key), medium, yt, split, c1p, array, hashes)
        sample = base.causal_fill(raw, observed)
        raw_q = base.VQF(dt).updateBatch(np.ascontiguousarray(sample[:, 3:]), np.ascontiguousarray(sample[:, :3]))["quat6D"]
        corrected_q, at, release, history, _, _ = c4.inertial_replay(raw, observed, dt, cfg)
    np.testing.assert_array_equal(mask, masks[prior["mask_key"]])
    base.require(intrinsic.mask_digest(mask) == prior["scoring_indices_sha256"], "C006 score mask changed")
    gate = c6.rest_flags(history)
    old_gate = gates[prior["mask_key"]]
    np.testing.assert_array_equal(release, old_gate[:, 0])
    np.testing.assert_array_equal(gate, old_gate[:, 3])
    deltas = c4.evaluate_windows(core, layers, history, mean, std)
    deltas.update(c4.evaluate_windows(tuned, {"C008": tuned.output}, history, mean, std))
    for method in ("C006", "C008"):
        deltas[method][gate] = 0.
    candidates = {key: (key, c4.GAIN, False) for key in deltas}

    def compute(use):
        pure = intrinsic.score_attitudes(raw_q, ref, use, raw_q[at], release, {}, {}, 1.)["full_vqf"]
        scores = intrinsic.score_attitudes(corrected_q, ref, use, corrected_q[at], release, deltas, candidates, 1.)
        return {"full_vqf_deg": pure, **{key + "_deg": scores[key] for key in deltas}}

    scores = compute(mask)
    controls = {"full_vqf_deg": "full_vqf_deg", "c003_deg": "c003_deg", "C006_deg": "B_rest_gated_deg"}
    errors = {key: scores[key] - prior[old] for key, old in controls.items()}
    base.require(np.isfinite(list(scores.values())).all() and max(map(abs, errors.values())) <= 1e-6,
                 f"Could not reproduce controls for {name}: {errors}")
    fields = ("dataset", "record", "group", "reference_status", "scored_frames", "mask_key",
              "scoring_indices_sha256", "instrument", "scenario")
    row = {**{key: prior[key] for key in fields}, **scores, "status": "scored",
           "reference_quality_flags": prior.get("reference_quality_flags", []),
           "rest_scored_fraction": prior["rest"]["rest_scored_fraction"], "controls_minus_prior_deg": errors}
    detail = []
    for axis, pose in poses.items():
        use = mask & pose
        detail.append({"record": name, "pose": axis, "scored_frames": int(use.sum()),
                       "scoring_indices_sha256": intrinsic.mask_digest(use), **compute(use)})
    return row, detail


def summarize(rows, denominator):
    base.require(len(rows) <= denominator, "Duplicate scored records")
    values = {key: float(np.mean([r[key] for r in rows])) for key in METHODS} if rows else {}
    return {"records": len(rows), "denominator": denominator, "scored_frames": sum(r["scored_frames"] for r in rows),
            "macro_rmse_deg": values,
            "C008_delta_macro_deg": {key: values["C008_deg"] - values[key] for key in METHODS[:-1]} if rows else {},
            "C008_wins_vs": {key: sum(r["C008_deg"] < r[key] for r in rows) for key in METHODS[:-1]}}


def replay(prior, inventory, campaign, split, core, control, tuned, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(path) for path in (C001 / "results.json", C001 / "protocol.json",
                                                C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    frozen_keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in frozen_keys}
    base.require(len(frozen_keys) == 213 and frozen_keys == set(specs)
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "A replay record is outside the frozen development scope")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    ref = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(ref)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    layers = {"c003": core.output, "C006": control.output}
    rows, failures, poses = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            key = old["dataset"], old["record"]
            try:
                row, detail = evaluate(old, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                       evidence["array_raw_sha256"], masks, gates, core, layers, tuned, mean, std, cfg)
                for r in detail:
                    prev = old_poses[r["record"], r["pose"]]
                    base.require(r["scoring_indices_sha256"] == prev["scoring_indices_sha256"]
                                 and abs(r["C006_deg"] - prev["B_rest_gated_deg"]) <= 1e-6, "C006 pose changed")
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
    for old in prior["sequences"]:
        key = old["dataset"], old["record"]
        row = by_key.get(key, {})
        source = {**old, **row, "status": "scored" if row else "failed", "error": errored.get(key, "")}
        output.append({k: row.get(k, "") if k in METHODS else source.get(k, "") for k in fields})
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
    wall_start = perf_counter()
    started = stamp()
    protocol_path = HERE / "protocol.json"
    process_path = HERE / "process.json"
    checkpoint = HERE / "mamba-end-to-end.pt"
    base.require(not protocol_path.exists() and not checkpoint.exists(), "C008 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p4, r4, t4, prep4 = (read(C004 / n) for n in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    p6, prior = (read(C006 / n) for n in ("protocol.json", "results.json"))
    p3 = read(C003 / "protocol.json")
    inventory_path = C001 / "inventory.json"
    inventory = read(inventory_path)
    schedule_path = C004 / "schedules.npz"
    original_steps = C004 / "training-steps.csv"
    split_path = TRANSFER / "external-split.json"
    incumbent = PILOT / campaign["incumbent"]
    bpath = C004 / t4["arms"]["B"]["weights"]
    base.require(len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C006 / "protocol.json") == prior["protocol_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_path) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(original_steps) == t4["training_steps_csv_sha256"]
                 and digest(MASKS) == prior["shared_masks_sha256"] and digest(GATES) == prior["gate_traces_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == p3["estimator_sha256"]
                 and digest(split_path) == p4["source_split_sha256"] == inventory["prior_external_split_sha256"]
                 and digest(incumbent) == p4["base_checkpoint_sha256"] == p6["B6_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(bpath) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"],
                 "Frozen sources, parameters or controls changed")
    base.require(all(digest(c5.REVIEW / name) == sha for name, sha in p6["optical_reference_sha256"].items()),
                 "Frozen optical reference source changed")
    records, archived = p4["source_records"], prep4["records"]
    base.require(Counter(spec["domain"] for spec in records) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and all(all(saved[key] == val for key, val in spec.items()) for spec, saved in zip(records, archived, strict=True))
                 and len(p4["yt_prefix_exclusion"]["selected_prefixes"]) == 24
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "External source manifest changed")
    base.require((p4["seed"], p4["steps_per_arm"], p4["batch_size"], p4["lr"], p4["weight_decay"], p4["inference_gain"]) ==
                 (c4.SEED, c4.STEPS, c4.BATCH, c4.LR, c4.WD, c4.GAIN), "C004 optimizer settings changed")
    with np.load(schedule_path, allow_pickle=False) as item:
        schedule, offsets, releases = (item[key] for key in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and len(offsets) == len(records) + 1
                 and np.array_equal(np.diff(offsets), [x["usable"] for x in archived]), "C004 B draw map changed")
    with original_steps.open(newline="", encoding="utf-8") as f:
        archived_first = float(next(x["loss"] for x in csv.DictReader(f) if x["arm"] == "B" and x["step"] == "1"))
    frozen, old_b = (torch.load(path, weights_only=True, map_location="cpu") for path in (incumbent, bpath))
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    head = {key: val for key, val in core.named_parameters() if key.startswith("output.")}
    base.require(frozen["model"] == "mamba_v1" and sum(p.numel() for p in core.parameters()) == 5595
                 and sum(p.numel() for p in head.values()) == 75, "Mamba parameter set changed")
    base.require(old_b["c004_arm"] == "B" and old_b["c004_gain"] == c4.GAIN
                 and all(torch.equal(frozen[k], old_b[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, old_b["state_dict"][k]) for k, v in frozen["state_dict"].items() if not k.startswith("output."))
                 and c4.tensor_hash(core.output.state_dict()) == p4["initial_B_output_parameter_sha256"]
                 and c4.tensor_hash({k: v for k, v in core.state_dict().items() if not k.startswith("output.")}) == p4["frozen_backbone_parameter_sha256"]
                 and c4.tensor_hash({k: frozen[k] for k in ("mean", "std")}) == p4["frozen_normalization_sha256"],
                 "B6 initialization, normalization, or C006 control changed")
    source_paths = {Path(inspect.getfile(obj)).resolve() for obj in
                    (c7, c6, c5, c4, c3, c2, c2.c1, base, intrinsic, intrinsic.build_frame, base.ResidualHead,
                     type(core.sequence), rotate)}
    source_paths.update((Path(__file__).resolve(), C003 / "online_bias.py"))
    source_paths.update(c5.REVIEW / name for name in p6["optical_reference_sha256"])
    artifacts = (CAMPAIGN / "protocol.json", C004 / "protocol.json", C004 / "training-preparation.json",
                 original_steps, schedule_path, C006 / "protocol.json", C006 / "results.json", MASKS, GATES,
                 inventory_path, split_path, incumbent, bpath, C003 / "protocol.json")
    input_files = {str(path.relative_to(PILOT)): digest(path) for path in sorted(source_paths | set(artifacts))}
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": input_files, "source_records": records,
                "yt_prefix_exclusion": p4["yt_prefix_exclusion"],
                "validation_records": [[r["dataset"], r["record"]] for r in prior["sequences"]],
                "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "initial_checkpoint_sha256": digest(incumbent), "C006_B_checkpoint_sha256": digest(bpath),
                "B_schedule_sha256": array_sha(schedule, "<i8"), "B_schedule_shape": list(schedule.shape),
                "normalization_sha256": p4["frozen_normalization_sha256"],
                "optimizer": {"kind": "AdamW", "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH,
                              "output_lr": c4.LR, "backbone_lr": BACKBONE_LR, "weight_decay": c4.WD, "grad_clip": 2.},
                "trainable_parameters": {"output": 75, "backbone": 5520}, "gain": c4.GAIN,
                "target": "Original point label at release: rotate with causal VQF at release to world, then inverse VQF at input-last and into intrinsic frame.",
                "loss": "Same original C004-B normalized squared direction distance and global gradient clipping. No training rest gate.",
                "replay": p6["new_policy"], "fit_scope": "Exactly one fit, 63 external source records; evaluation on the 213 archived development records only."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"pid": os.getpid(), "started_at": started, "status": phase,
                             "protocol_sha256": protocol_sha, **extra})

    status("preparing")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "The authorized GPU replay path is unavailable")
    device = torch.device("cuda")
    core.eval().requires_grad_(False).to(device)
    tuned, control = deepcopy(core), deepcopy(core)
    control.load_state_dict(old_b["state_dict"])
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        canonical, target, draw_index, audits, causal, rotation = point_windows(
            records, archived, schedule, offsets, releases, yt["train"], cfg)
    prep_path = HERE / "training-preparation.json"
    write(prep_path, {"records": audits, "distinct_drawn_windows": len(canonical), "draws": schedule.size,
                      "canonical_inputs_sha256": array_sha(canonical, "<f4"), "point_targets_sha256": array_sha(target, "<f4"),
                      "B_schedule_sha256": protocol["B_schedule_sha256"], "protocol_sha256": protocol_sha})
    inputs, labels, draws = (torch.from_numpy(x).to(device) for x in (canonical, target, draw_index))
    with torch.inference_mode():
        start_loss = float((prediction(core, inputs[draws[0]], mean, std) - labels[draws[0]]).square().sum(dim=1).mean())
    base.require(abs(start_loss - archived_first) <= 1e-10, f"Initial C004-B parity failed before fitting: {start_loss:.15g}")
    print("source/schedule/init lock passed", {"draws": schedule.size, "windows": len(canonical),
          "C004_B_first_loss": archived_first, "C008_initial_loss": start_loss, "elapsed_s": perf_counter() - wall_start}, flush=True)
    base.require(all(digest(PILOT / name) == sha for name, sha in input_files.items()), "A frozen input changed before the fit")
    status("fitting", optimizer_steps_planned=c4.STEPS, preparation_sha256=digest(prep_path))
    tuned, history, fit_time = fit(tuned, inputs, labels, draws, mean, std, archived_first)
    state = {k: v.detach().cpu().clone() for k, v in tuned.state_dict().items()}
    stage = checkpoint.with_suffix(".tmp")
    torch.save({**frozen, "state_dict": state, "C008_fit": fit_time, "protocol_sha256": protocol_sha}, stage)
    stage.replace(checkpoint)
    history_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(history_path, history)
    del inputs, labels, draws, canonical, target, draw_index
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    base.require(all(torch.equal(saved[k], frozen[k]) for k in ("mean", "std")), "Fixed external normalization changed")
    tuned.load_state_dict(saved["state_dict"])
    layers = layer_differences(frozen["state_dict"], saved["state_dict"])
    layers_path = HERE / "layer-norms.json"
    write(layers_path, layers)
    audits_at = perf_counter()
    prefix = final_prefix(tuned, causal, mean, std, cfg)
    rotation_checks = rotated_audit(tuned, rotation, mean, std)
    prefix_path, rotation_path = HERE / "prefix-audit.json", HERE / "equivariance.json"
    write(prefix_path, prefix)
    write(rotation_path, rotation_checks)
    training = {"fits": 1, "fit": fit_time, "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__), "cuda": torch.version.cuda, "initial_loss": start_loss,
                "archived_C004_B_first_loss": archived_first, "first_actual_step_loss": history[0]["loss"],
                "last_actual_step_loss": history[-1]["loss"], "audit_seconds": perf_counter() - audits_at,
                "artifacts_sha256": {path.name: digest(path) for path in (checkpoint, prep_path, history_path,
                                                                         layers_path, prefix_path, rotation_path)}}
    training_path = HERE / "training.json"
    write(training_path, training)
    status("scoring", fit_completed=True, training_sha256=digest(training_path))
    print("C008 weights and 160 losses persisted; scoring the frozen 213 records", flush=True)
    scoring_at = perf_counter()
    result = replay(prior, inventory, campaign, read(split_path), core, control, tuned, mean, std, cfg)
    completed = stamp()
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_sha,
                   "training_sha256": digest(training_path), "scoring_masks_sha256": protocol["scoring_masks_sha256"],
                   "scoring_seconds": perf_counter() - scoring_at, "total_seconds": perf_counter() - wall_start})
    results_path = HERE / "results.json"
    write(results_path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(results_path))
    print("C008 coverage", len(result["sequences"]), "failures", result["failures"], "total_seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    main()
