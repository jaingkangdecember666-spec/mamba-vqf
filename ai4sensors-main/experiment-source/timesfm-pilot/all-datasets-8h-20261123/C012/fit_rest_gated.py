"""Fit one C004-B output layer with the C006 rest condition applied in training."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import csv
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
from tilt_eval import normalize, rotate
from torch.nn import functional as F

c6, c5, c4, c3, c2 = c7.c6, c7.c5, c7.c4, c7.c3, c7.c2
base, intrinsic = c7.base, c7.intrinsic
read, write, digest, stamp = c7.read, c7.write, c7.digest, c7.stamp
C001, C002, C003, C004, C005, C006 = c7.C001, c7.C002, c7.C003, c7.C004, c7.C005, c7.C006
PILOT, DATA, TRANSFER = c7.PILOT, c7.DATA, c7.TRANSFER
MASKS, GATES = C005 / "scoring-masks.npz", C006 / "gate-traces.npz"
DOMAINS = ("BROAD", "Sassari", "YT_Server")
METHODS = ("full_vqf_deg", "C006_deg", "C012_deg")


def array_sha(a, dtype):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=dtype).tobytes()).hexdigest()


def point_source(spec, archived, original_releases, yt, mats, core, mean, std, cfg):
    raw, observed, dt, score, truth, hashes = c4.train_raw(spec, yt, mats)
    q, at, release, history, complete, bias = c4.inertial_replay(raw, observed, dt, cfg, training=True)
    eligible = complete & score[release]
    at, release, history = at[eligible], release[eligible], history[eligible]
    q_last = normalize(q[at])
    inverse = q_last.copy()
    inverse[:, 1:] *= -1.
    world = rotate(normalize(q[release]), truth[release])
    aligned = rotate(inverse, world)
    err = float(np.max(np.abs(rotate(q_last, aligned) - world)))
    latent, up, frame = c4.features(core, history, mean, std)
    target = intrinsic.transform_directions(aligned, frame)
    valid = frame.valid
    np.testing.assert_array_equal(release[valid], original_releases)
    np.testing.assert_allclose(np.einsum("ni,nji->nj", target[valid], frame.basis[valid]), aligned[valid], rtol=0., atol=1e-7)
    base.require(err <= 2e-12 and valid.any() and len(history) == archived["eligible_windows"]
                 and int(valid.sum()) == archived["usable"] and hashes == archived["raw_hashes"]
                 and bias["active_native_frames"] == archived["active_native_frames"],
                 f"Frozen C004 eligibility or point target changed for {spec['record']}")
    values = np.c_[latent[valid], up[valid], target[valid]]
    rest = c6.rest_flags(history)[valid]
    evidence = {**spec, "raw_hashes": hashes, "eligible_windows": len(history), "usable": len(values),
                "C003_active_native_frames": bias["active_native_frames"], "point_world_roundtrip_max_abs": err,
                "point_features_targets_sha256": array_sha(values, "<f4"), "rest_flags_sha256": array_sha(rest, "?"),
                "release_native_sha256": array_sha(release[valid], "<i8"),
                "input_last_native_sha256": array_sha(at[valid], "<i8")}
    return values, rest, at[valid], evidence, (raw, observed, dt)


def per_draw_error(layer, batch, rest=None):
    x, up, target = batch.split((layer.in_features, 3, 3), dim=1)
    delta = base.tangent(layer(x), up, base.CAP)
    if rest is not None:
        delta = torch.where(rest[:, None], torch.zeros_like(delta), delta)
    return (F.normalize(up + c4.GAIN * delta, dim=1) - target).square().sum(dim=1)


def first_step(initial, batch, rest, historical):
    probe = deepcopy(initial).requires_grad_(True)
    ungated = per_draw_error(probe, batch)
    ungated_mean = ungated.mean()
    old_grad = torch.autograd.grad(ungated_mean, tuple(probe.parameters()))
    old_norm = float(torch.stack([g.norm() for g in old_grad]).norm())
    old_loss = float(ungated_mean.detach())
    base.require(old_loss == float(historical["loss"])
                 and abs(old_norm - float(historical["gradient_norm"])) <= 1e-10, "C004-B first-step control changed")
    error = per_draw_error(probe, batch, rest)
    params = tuple(probe.parameters())
    mean96 = error.mean()
    nonrest_contribution = error[~rest].sum() / c4.BATCH
    rest_contribution = error[rest].sum() / c4.BATCH
    grad = torch.autograd.grad(mean96, params, retain_graph=True)
    nonrest_grad = torch.autograd.grad(nonrest_contribution, params, retain_graph=True)
    rest_grad = torch.autograd.grad(rest_contribution, params)
    rest_max = max(float(x.abs().max()) for x in rest_grad)
    diff = max(float((a - b).abs().max()) for a, b in zip(grad, nonrest_grad, strict=True))
    rest_prediction_error = (F.normalize(batch[:, -6:-3], dim=1) - batch[:, -3:]).square().sum(dim=1)
    torch.testing.assert_close(error[rest], rest_prediction_error[rest], rtol=0., atol=0.)
    torch.testing.assert_close(error[~rest], ungated[~rest], rtol=0., atol=0.)
    base.require(rest.any() and rest_max == 0. and diff <= 1e-10, "Rest contribution sends gradient or batch denominator changed")
    return {"denominator": len(batch), "rest_draws": int(rest.sum()), "C004_B_archived_loss": float(historical["loss"]),
            "ungated_C004_B_replay_loss": old_loss, "archived_ungated_grad_l2": float(historical["gradient_norm"]),
            "ungated_replay_grad_l2": old_norm, "gated_loss": float(mean96.detach()),
            "gated_grad_l2": float(torch.stack([g.norm() for g in grad]).norm()),
            "rest_loss_contribution_divided_by_96": float(rest_contribution.detach()),
            "nonrest_loss_contribution_divided_by_96": float(nonrest_contribution.detach()),
            "rest_only_parameter_grad_max_abs": rest_max, "gated_vs_nonrest_grad_max_abs": diff}


def optimize(initial, data, flags, schedule, step_counts, audited_loss, device):
    torch.manual_seed(c4.SEED)
    layer = deepcopy(initial).requires_grad_(True).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=c4.LR, weight_decay=c4.WD)
    draws = torch.from_numpy(schedule).to(device)
    history = []
    torch.cuda.synchronize()
    seconds, started = perf_counter(), stamp()
    for step, (indices, counts) in enumerate(zip(draws, step_counts, strict=True), 1):
        optimizer.zero_grad(set_to_none=True)
        rest = flags[indices]
        errors = per_draw_error(layer, data[indices], rest)
        loss = errors.mean()
        value = float(loss.detach())
        if step == 1:
            base.require(value == audited_loss, f"C012 first actual step drifted to {value:.15g}")
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(layer.parameters(), 2.))
        base.require(np.isfinite([value, grad]).all(), f"Nonfinite step {step}")
        optimizer.step()
        row = {"arm": "C012", "step": step, "loss": value, "gradient_norm": grad, "denominator": len(indices),
               "rest_contribution": float(errors.detach()[rest].sum() / c4.BATCH),
               "nonrest_contribution": float(errors.detach()[~rest].sum() / c4.BATCH), **counts}
        history.append(row)
        if step in (1, 40, 80, 120, c4.STEPS):
            print("training", row, flush=True)
    torch.cuda.synchronize()
    return layer.eval().requires_grad_(False), history, {"started_at": started, "completed_at": stamp(),
            "steps": len(history), "seconds": perf_counter() - seconds}


def summary(rows):
    good = [r for r in rows if r["status"] == "scored"]
    measures = {m: float(np.mean([r[m] for r in good])) for m in METHODS} if good else {}
    return {"denominator": len(rows), "scored_records": len(good), "scored_frames": sum(r["scored_frames"] for r in good),
            "macro_rmse_deg": measures, "rest_scored_frames": sum(r["rest_scored_frames"] for r in good),
            "C012_delta_macro_deg": {m: measures["C012_deg"] - measures[m] for m in METHODS[:-1]} if good else {},
            "C012_wins_vs": {m: sum(r["C012_deg"] < r[m] for r in good) for m in METHODS[:-1]}}


def replay(prior, inventory, campaign, split, core, layers, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "A scoring record is outside frozen development")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(array_path)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            key = old["dataset"], old["record"]
            try:
                # The shared frozen replay names its only candidate C007; here that slot holds C012.
                row, detail = c7.evaluate(old, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                         evidence["array_raw_sha256"], masks, gates, core, layers, mean, std, cfg)
                row["C012_deg"] = row.pop("C007_deg")
                row.pop("c003_deg")
                row["reference_QA"] = old["reference_QA"]
                row["rest_scored_frames"] = old["rest"]["scored_frames_with_rest"]
                for pose in detail:
                    saved = old_poses[pose["record"], pose["pose"]]
                    base.require(pose["scoring_indices_sha256"] == saved["scoring_indices_sha256"]
                                 and abs(pose["C006_deg"] - saved["B_rest_gated_deg"]) <= 1e-6
                                 and abs(pose["full_vqf_deg"] - saved["full_vqf_deg"]) <= 1e-6,
                                 "Original IMU-Array pose control changed")
                    pose["C012_deg"] = pose.pop("C007_deg")
                    pose.pop("c003_deg")
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**{k: old[k] for k in ("dataset", "record", "group", "instrument", "scenario", "reference_QA",
                                                    "reference_status", "mask_key", "scored_frames", "scoring_indices_sha256")},
                             "status": "failed", "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    output, pose_path, groups_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv", HERE / "group-summaries.json"
    fields = ("dataset", "record", "group", "instrument", "scenario", "status", "reference_QA", "mask_key", "scoring_indices_sha256",
              "scored_frames", "rest_scored_frames", "rest_scored_fraction", *METHODS, "error")
    intrinsic.shice.save_csv(output, [{k: row.get(k, "") for k in fields} for row in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    domains = {d: summary([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]}
    groups = {k: {f"{d}/{name}": summary([r for r in rows if r["dataset"] == d and r[k] == name])
                  for d, name in sorted({(r["dataset"], r[k]) for r in rows})} for k in ("reference_QA", "instrument", "scenario")}
    write(groups_path, groups)
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": domains, "optical_subsets": {k: summary(v) for k, v in subsets.items()},
            "csv_sha256": digest(output), "pose_csv_sha256": digest(pose_path) if poses else None,
            "group_summaries_sha256": digest(groups_path),
            "control_max_abs_delta_deg": max((abs(v) for r in rows for v in r.get("controls_minus_prior_deg", {}).values()), default=None),
            "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    checkpoint = HERE / "rest-trained-readout.pt"
    base.require(not protocol_path.exists() and not checkpoint.exists(), "C012 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p4, r4, t4, prep4 = (read(C004 / k) for k in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    p6, prior = (read(C006 / k) for k in ("protocol.json", "results.json"))
    p3 = read(C003 / "protocol.json")
    inventory_path = C001 / "inventory.json"
    inventory = read(inventory_path)
    schedule_path, old_steps = C004 / "schedules.npz", C004 / "training-steps.csv"
    split_path = TRANSFER / "external-split.json"
    incumbent = PILOT / campaign["incumbent"]
    old_b = C004 / t4["arms"]["B"]["weights"]
    c11 = CAMPAIGN / "C011"
    c11_witness = c11 / "training-windows.npz"
    base.require(len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C006 / "protocol.json") == prior["protocol_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_path) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(old_steps) == t4["training_steps_csv_sha256"]
                 and digest(MASKS) == prior["shared_masks_sha256"] and digest(GATES) == prior["gate_traces_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == p3["estimator_sha256"]
                 and digest(split_path) == p4["source_split_sha256"] == inventory["prior_external_split_sha256"]
                 and digest(incumbent) == p4["base_checkpoint_sha256"] == p6["B6_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(old_b) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"]
                 and digest(c11_witness) == read(c11 / "coefficient-lock.json")["training_windows_sha256"],
                 "Frozen archive, mask, C006 weights, or independent rest witness changed")
    base.require(all(digest(c5.REVIEW / name) == sha for name, sha in p6["optical_reference_sha256"].items()),
                 "Frozen optical reference implementation changed")
    base.require((p4["seed"], p4["steps_per_arm"], p4["batch_size"], p4["lr"], p4["weight_decay"], p4["inference_gain"]) ==
                 (c4.SEED, c4.STEPS, c4.BATCH, c4.LR, c4.WD, c4.GAIN), "C004 optimizer or gain changed")
    records, archived = p4["source_records"], prep4["records"]
    base.require(Counter(r["domain"] for r in records) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and all(all(r[k] == v for k, v in q.items()) for q, r in zip(records, archived, strict=True))
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "External source manifest changed")
    with np.load(schedule_path, allow_pickle=False) as file:
        schedule, offsets, releases = (file[k] for k in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and np.array_equal(np.diff(offsets), [r["usable"] for r in archived])
                 and np.all((schedule >= 0) & (schedule < len(releases))), "C004-B sampling changed")
    with old_steps.open(encoding="utf-8", newline="") as stream:
        first_archive = next(r for r in csv.DictReader(stream) if r["arm"] == "B" and r["step"] == "1")
    frozen, trained = (torch.load(path, map_location="cpu", weights_only=True) for path in (incumbent, old_b))
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    base.require(sum(x.numel() for x in core.output.parameters()) == 75 and trained["c004_arm"] == "B"
                 and trained["c004_gain"] == c4.GAIN
                 and all(torch.equal(frozen[k], trained[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, trained["state_dict"][k]) for k, v in frozen["state_dict"].items() if not k.startswith("output."))
                 and c4.tensor_hash(core.output.state_dict()) == p4["initial_B_output_parameter_sha256"]
                 and c4.tensor_hash({k: v for k, v in frozen["state_dict"].items() if not k.startswith("output.")}) == p4["frozen_backbone_parameter_sha256"]
                 and c4.tensor_hash({k: frozen[k] for k in ("mean", "std")}) == p4["frozen_normalization_sha256"],
                 "B6 initialization, normalization, or control backbone changed")
    paths = {Path(inspect.getfile(x)).resolve() for x in (c7, c6, c5, c4, c3, c2, c2.c1, base, base.ResidualHead,
                                                         type(core.sequence), intrinsic, intrinsic.build_frame, rotate)}
    paths.update(c5.REVIEW / name for name in p6["optical_reference_sha256"])
    paths.update((Path(__file__).resolve(), incumbent, old_b, schedule_path, old_steps, split_path, inventory_path, MASKS, GATES,
                  CAMPAIGN / "protocol.json", C004 / "protocol.json", C004 / "training-preparation.json", C004 / "training.json",
                  C006 / "protocol.json", C006 / "results.json", C003 / "protocol.json", C003 / "online_bias.py",
                  C002 / "protocol.json", C001 / "protocol.json", C001 / "results.json", C003 / "results.json",
                  c11 / "coefficient-lock.json", c11_witness))
    files = {str(p.relative_to(PILOT)): digest(p) for p in sorted(paths)}
    protocol = {"pid": os.getpid(), "started_at": started, "files_sha256": files, "source_records": records,
                "yt_prefix_exclusion": p4["yt_prefix_exclusion"], "C004_B_schedule_sha256": digest(schedule_path),
                "C004_B_draw_matrix_sha256": array_sha(schedule, "<i8"), "C004_B_schedule_shape": list(schedule.shape),
                "initial_checkpoint_sha256": digest(incumbent), "initial_output_parameter_sha256": p4["initial_B_output_parameter_sha256"],
                "frozen_backbone_parameter_sha256": p4["frozen_backbone_parameter_sha256"],
                "frozen_normalization_sha256": p4["frozen_normalization_sha256"], "C006_weight_sha256": digest(old_b),
                "development_records": [[r["dataset"], r["record"]] for r in prior["sequences"]],
                "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "optimizer": {"kind": "AdamW", "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH,
                              "lr": c4.LR, "weight_decay": c4.WD, "clip_grad_norm": 2.},
                "gain": c4.GAIN, "trainable_parameters": {"output.weight": 72, "output.bias": 3},
                "target": "The original C004-B release-point label aligned and expressed in the intrinsic input-last frame.",
                "training_gate": "Use VQF restDetected at the C006 input-last completed block. Set the predicted neural delta to zero on rest before multiplying by gain and normalizing. Retain every original sample's squared direction error in the mean with denominator 96.",
                "inference": p6["new_policy"], "scope": "Exactly one output-layer fit on C004-B external training only. Score the original 213 development records. All final holdouts sealed."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "started_at": started, "pid": os.getpid(),
                             "protocol_sha256": protocol_sha, **extra})

    status("external_preparation")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "C004 CUDA inference path unavailable")
    device = torch.device("cuda")
    core.eval().requires_grad_(False).to(device)
    control = deepcopy(core)
    control.load_state_dict(trained["state_dict"])
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    pool, flags, inputs_last, sources, causal, mats = [], [], [], [], {}, {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, (spec, ref) in enumerate(zip(records, archived, strict=True)):
            data, rest, at, audit, probe = point_source(spec, ref, releases[offsets[i]:offsets[i + 1]], yt["train"], mats,
                                                       core, mean, std, cfg)
            pool.append(data)
            flags.append(rest)
            inputs_last.append(at)
            sources.append(audit)
            causal.setdefault(spec["domain"], probe)
    all_data, all_rest, all_last = np.concatenate(pool), np.concatenate(flags), np.concatenate(inputs_last)
    sampled = all_rest[schedule]
    ordinal = np.searchsorted(offsets[1:], schedule, side="right")
    code = np.array([DOMAINS.index(r["domain"]) for r in records])[ordinal]
    counts = []
    for i, (where, rest) in enumerate(zip(code, sampled, strict=True)):
        row = {"rest_total": int(rest.sum()), "nonrest_total": int((~rest).sum())}
        for j, d in enumerate(DOMAINS):
            domain = where == j
            row[d + "_draws"] = int(domain.sum())
            row[d + "_rest"] = int(np.count_nonzero(domain & rest))
        base.require(all(row[d + "_draws"] == 32 for d in DOMAINS) and len(rest) == 96, f"Step {i+1} changed")
        counts.append(row)
    totals = {d: sum(x[d + "_rest"] for x in counts) for d in DOMAINS}
    base.require(totals == {"BROAD": 0, "Sassari": 2799, "YT_Server": 0}, f"Rest count diverged: {totals}")
    with np.load(c11_witness, allow_pickle=False) as witness:
        np.testing.assert_array_equal(witness["original_B_schedule"], schedule)
        np.testing.assert_array_equal(witness["release_native"], releases[schedule])
        np.testing.assert_array_equal(witness["input_last_native"], all_last[schedule])
        np.testing.assert_array_equal(witness["rest_gate"].reshape(schedule.shape), sampled)
    for i, (a, ref) in enumerate(zip(sources, archived, strict=True)):
        a["draws"] = int(np.count_nonzero(ordinal == i))
        a["rest_draws"] = int(np.count_nonzero((ordinal == i) & sampled))
        base.require(a["draws"] == ref["B_draws"], "A source got a different count of draws")
    witness = HERE / "training-windows.npz"
    with witness.open("wb") as stream:
        np.savez_compressed(stream, original_B_schedule=schedule, record_ordinal=ordinal, native_release=releases[schedule],
                            native_input_last=all_last[schedule], vqf_rest=sampled, C004_points=all_data[schedule])
    prep_path = HERE / "training-preparation.json"
    write(prep_path, {"protocol_sha256": protocol_sha, "sampling_npz_sha256": digest(witness), "records": sources,
                      "draws": schedule.size, "rest_draws_by_domain": totals, "rest_draws": int(sampled.sum()),
                      "C011_independent_rest_witness_exact": True, "drawn_rest_flags_sha256": array_sha(sampled, "?")})
    data, flag = torch.from_numpy(all_data).to(device), torch.from_numpy(all_rest).to(device)
    ix0 = torch.from_numpy(schedule[0]).to(device)
    audit = first_step(core.output, data[ix0], flag[ix0], first_archive)
    audit_path = HERE / "first-step-control.json"
    write(audit_path, {**audit, "first_batch_counts": counts[0]})
    base.require(all(digest(PILOT / path) == sha for path, sha in files.items()), "Frozen inputs changed before fitting")
    print("C012 source and first-step checks passed", audit, "rest by domain", totals, flush=True)
    status("fitting", planned_steps=160, preparation_sha256=digest(prep_path))
    layer, history, runtime = optimize(core.output, data, flag, schedule, counts, audit["gated_loss"], device)
    stage = checkpoint.with_suffix(".tmp")
    state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    state.update({"output." + k: v.detach().cpu().clone() for k, v in layer.state_dict().items()})
    torch.save({**frozen, "state_dict": state, "C012_steps": len(history), "protocol_sha256": protocol_sha}, stage)
    stage.replace(checkpoint)
    steps_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(steps_path, history)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    changed = {k for k, v in saved["state_dict"].items() if not torch.equal(v, frozen["state_dict"][k])}
    base.require(changed == {"output.weight", "output.bias"}
                 and all(torch.equal(saved[k], frozen[k]) for k in ("mean", "std")), "Frozen backbone or input normalization changed")
    layer.load_state_dict({k.removeprefix("output."): saved["state_dict"][k] for k in changed})
    prefix_path = HERE / "prefix-audit.json"
    write(prefix_path, c7.causal_check(causal, core, layer, mean, std, cfg))
    del data, flag, all_data, all_rest, pool, flags, causal, mats
    training_path = HERE / "training.json"
    write(training_path, {"fit_count": 1, "fit": runtime, "device": torch.cuda.get_device_name(device),
                          "changed_params": sorted(changed), "rest_draws_by_domain": totals,
                          "files_sha256": {p.name: digest(p) for p in (checkpoint, steps_path, prep_path, audit_path, prefix_path, witness)}})
    status("development_replay", training_sha256=digest(training_path))
    print("C012 fixed-step weights persisted; replaying 213 frozen development records", flush=True)
    scoring = perf_counter()
    final = replay(prior, inventory, campaign, read(split_path), core,
                   {"c003": core.output, "C006": control.output, "C007": layer}, mean, std, cfg)
    completed = stamp()
    final.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_sha, "optimizer_steps": len(history),
                  "training_sha256": digest(training_path), "scoring_masks_sha256": digest(MASKS),
                  "rest_draws_by_domain": totals, "scoring_seconds": perf_counter() - scoring, "total_seconds": perf_counter() - wall})
    result_path = HERE / "results.json"
    write(result_path, final)
    status("partial" if final["failures"] else "complete", completed_at=completed, result_sha256=digest(result_path))
    print("C012 coverage", len(final["sequences"]), "failures", final["failures"], "seconds", final["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process = HERE / "process.json"
        if process.exists():
            p = read(process)
            if p["pid"] == os.getpid():
                write(process, {**p, "phase": p["status"], "status": "failed", "completed_at": stamp(), "failure": repr(error)})
        raise
