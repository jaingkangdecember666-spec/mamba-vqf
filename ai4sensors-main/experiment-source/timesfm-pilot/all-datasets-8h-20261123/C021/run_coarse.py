"""C021: train one coarse-history Mamba branch on the exact C020 releases."""
from collections import Counter
from copy import deepcopy
import inspect
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C020"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
import run_temporal as c20
from multirate_blocks import long_windows

c19, c18, c17, c14 = c20.c19, c20.c18, c20.c17, c20.c14
c6, c4, c3, c2, base, intrinsic = c20.c6, c20.c4, c20.c3, c20.c2, c20.base, c20.intrinsic
read, write, sha, stamp, array_sha = c20.read, c20.write, c20.sha, c20.stamp, c20.array_sha
DATA, PILOT, TRANSFER, MASKS, GATES = c20.DATA, c20.PILOT, c20.TRANSFER, c20.MASKS, c20.GATES
UP, STEPS, BATCH, GAIN, DOMAINS = c20.UP, c20.STEPS, c20.BATCH, c20.GAIN, c20.DOMAINS
METHODS = (*c20.METHODS, "C021_deg")
BACKBONE_LR, HEAD_LR, WD, CLIP = c20.BACKBONE_LR, c20.HEAD_LR, c20.WD, c20.CLIP
SCHEDULE_FIELDS = c20.SCHEDULE_FIELDS


def ordered_coarse(fine, at, release):
    base.require(len(fine) == len(at) == len(release) and np.all(at <= release)
                 and np.all(np.diff(at) > 0) and np.all(np.diff(release) > 0), "Coarse expects a complete ordered release stream")
    if len(fine) > 1:
        np.testing.assert_array_equal(fine[1:, :-10], fine[:-1, 10:])
    coarse, ready = long_windows(fine)
    np.testing.assert_array_equal(ready, 63 + 10 * np.arange(len(fine)) >= 255)
    np.testing.assert_array_equal(coarse[ready, -1, 6:], fine[ready, -1, 6:])
    return coarse, ready


def verify_locked_schedule(files):
    data = c20.load_schedule(files)
    before = read(ROOT / "C020/training-preparation.json")
    for key, value in (("C019_center_sha256", array_sha(data["center"], "<f8")),
                       ("C019_release_target_sha256", array_sha(data["truth"], "<f8")),
                       ("C019_source_record_ordinals_sha256", array_sha(data["source_record_ordinal"].reshape(STEPS, BATCH), "<i8")),
                       ("C019_native_release_sha256", array_sha(data["native_release"].reshape(STEPS, BATCH), "<i8"))):
        base.require(before[key] == value, "Dose, center or native label differs from C020")
    with (ROOT / "C020/optimizer-steps.jsonl").open(encoding="utf-8") as f:
        history = [json.loads(line) for line in f]
    base.require(len(history) == STEPS, "C020 optimizer trace length changed")
    for i, row in enumerate(history):
        sl = slice(i * BATCH, (i + 1) * BATCH)
        base.require(row["step"] == i + 1 and row["rest_slots"] == int(data["rest"][sl].sum())
                     and all(array_sha(data[k][sl], "<i8") == row[k + "_sha256"] for k in SCHEDULE_FIELDS[:3]),
                     "Static C020/C021 per-batch verification failed")
    base.require(list(inspect.signature(long_windows).parameters) == ["fine"]
                 and list(inspect.signature(c17.force_at).parameters) == ["q", "acceleration", "observed", "dt", "at", "starts", "rest"],
                 "An inherited input helper changed its signature")
    return data, before


def reconstruct(inventory, data, prior, models, mean, std, cfg, k):
    p15 = read(ROOT / "C015/protocol.json")
    p15prep = read(ROOT / "C015/training-preparation.json")
    p19prep = read(ROOT / "C019/training-preparation.json")
    panel = read(ROOT / "review/yt_training_panel.json")
    originals, _, _, approved, _ = c19.frozen_panel(inventory, panel, p15)
    sources = p19prep["source_records"]
    fine_canonical = np.load(ROOT / "C020/training-canonical-history.npy", mmap_mode="r", allow_pickle=False)
    rows, mats, probes, domains = [], {}, [], set()
    canonical = np.zeros_like(fine_canonical)
    up, basis = np.zeros_like(data["canonical_up"]), np.zeros_like(data["world_basis"])
    valid, available = np.zeros_like(data["valid"]), np.zeros_like(data["valid"])
    pooled = np.zeros((len(valid), 24), dtype=np.float32)
    ordinal, native = data["source_record_ordinal"], data["native_release"]
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, name in enumerate(sources):
            started = perf_counter()
            draw = np.flatnonzero(ordinal == i)
            d = str(data["domain"][draw[0]])
            np.testing.assert_array_equal(data["source_record_name"][draw], name)
            np.testing.assert_array_equal(data["domain"][draw], d)
            old = prior["records"][i]
            base.require(old["ordinal"] == i and old["record"] == name and old["domain"] == d
                         and old["slots"] == len(draw) and old["native_release_sha256"] == array_sha(native[draw], "<i8"),
                         "C020 training provenance no longer matches")
            if i < len(originals):
                spec = originals[i]
            else:
                item = approved[name]
                base.require(yt["train"][name].name == item["source"], "Outside approved official YT train")
                spec = {"domain": "YT_Server", "record": name, "native_frames": item["native_frames"]}
            if d == "RIDI":
                raw, observed, dt, score, truth, hashes = c17.trained_phone(spec, p15prep["ridi_eligibility"][i - 63])
            else:
                raw, observed, dt, score, truth, hashes = c4.train_raw(spec, yt["train"], mats)
            base.require(hashes == old["source_sha256"], "An original C020 signal or reference changed")
            _, q, at, release, fine, complete, acc, bias = c17.front_end(raw, observed, dt, cfg)
            coarse, ready = ordered_coarse(fine, at, release)
            native_rows, inv = np.unique(native[draw], return_inverse=True)
            pos = np.searchsorted(release, native_rows)
            np.testing.assert_array_equal(release[pos], native_rows)
            np.testing.assert_array_equal(at[pos][inv], data["native_input_last"][draw])
            base.require(np.all(complete[pos] & score[native_rows]), f"Original eligibility changed for {name}")
            y = c14.rotate(c14.normalize(q[native_rows]), truth[native_rows])
            np.testing.assert_array_equal(y[inv], data["truth"][draw])
            selected_fine, selected_coarse, qlast = fine[pos], coarse[pos], q[at[pos]]
            rest = c6.rest_flags(selected_fine)
            np.testing.assert_array_equal(rest[inv], data["rest"][draw])
            local = intrinsic.transform_inputs(selected_fine, intrinsic.build_frame(selected_fine))
            np.testing.assert_array_equal(local[inv], fine_canonical[draw])
            base.require(array_sha(selected_fine, "<f4") == old["raw_history_sha256"], "C020 10 ms history digest changed")
            center, r = c17.branch_at(qlast, selected_fine, models, mean, std)
            np.testing.assert_array_equal(r, rest)
            force, _, _ = c17.force_at(q, acc, observed, dt, at[pos], c17.window_starts(dt, len(at))[pos], rest)
            center += c17.additions(force, k)[0]
            center_gap = float(np.max(np.abs(center[inv] - data["center"][draw])))
            base.require(center_gap <= 2e-7, f"C017 center no longer matches for {name}")
            frame = intrinsic.build_frame(selected_coarse)
            local_coarse = intrinsic.transform_inputs(selected_coarse, frame)
            feat = c18.features(selected_coarse, qlast, models[0], mean, std, frame=frame)
            okay = ready[pos] & feat.valid
            canonical[draw], up[draw], basis[draw], pooled[draw] = (x[inv] for x in (local_coarse, feat.up, feat.world_basis, feat.pooled[:, 1]))
            valid[draw], available[draw] = okay[inv], ready[pos][inv]
            if d not in domains:
                active = np.flatnonzero(okay & ~rest)
                if len(active) >= 3:
                    for j in active[np.linspace(0, len(active) - 1, 3, dtype=int)]:
                        probes.append({"domain": d, "record": name, "release": int(native_rows[j]), "fine": selected_fine[j],
                                       "coarse": selected_coarse[j], "quat": qlast[j],
                                       "center": data["center"][draw[np.flatnonzero(inv == j)[0]]]})
                    domains.add(d)
            active_all = np.flatnonzero(ready)
            rows.append({"ordinal": i, "domain": d, "record": name, "slots": len(draw), "unique_native_releases": len(native_rows),
                         "coarse_ready_slots": int(available[draw].sum()), "coarse_valid_slots": int(valid[draw].sum()),
                         "coarse_valid_nonrest_slots": int((valid[draw] & ~data["rest"][draw]).sum()),
                         "ready_at_stream_start_false_count": int((~ready).sum()),
                         "first_coarse_release": int(release[active_all[0]]) if len(active_all) else None,
                         "first_coarse_release_elapsed_s": float((release[active_all[0]] + 1) * dt) if len(active_all) else None,
                         "native_release_sha256": array_sha(native[draw], "<i8"), "C020_fine_canonical_sha256": array_sha(local, "<f4"),
                         "coarse_canonical_sha256": array_sha(local_coarse, "<f4"), "target_sha256": array_sha(y[inv], "<f8"),
                         "C020_center_sha256": array_sha(data["center"][draw], "<f8"), "C017_center_reconstruction_max_abs": center_gap,
                         "rest_sha256": array_sha(rest[inv], "?"), "source_sha256": hashes, "seconds": perf_counter() - started})
            print("C021 coarse slots", d, name, len(draw), int(valid[draw].sum()), flush=True)
    base.require(len(rows) == 139 and len(probes) == 12 and np.isfinite(canonical).all() and valid.any() and np.all(valid <= available),
                 "Coarse reconstruction coverage incomplete")
    native_fields = {key: data[key] for key in (*SCHEDULE_FIELDS, "source_record_name", "domain", "rest", "center", "truth", "record_equal_weights")}
    archive, history, sample = (HERE / n for n in ("training-coarse-inputs.npz", "training-canonical-history.npy", "history-probes.npz"))
    np.savez_compressed(archive, **native_fields, coarse_ready=available, coarse_valid=valid, coarse_up=up, coarse_world_basis=basis,
                        initial_coarse_B_features=pooled)
    np.save(history, canonical, allow_pickle=False)
    np.savez_compressed(sample, **{key: np.stack([r[key] for r in probes]) for key in ("domain","record","release","fine","coarse","quat","center")})
    coarse_data = {**native_fields, "valid": valid, "ready": available, "canonical_up": up, "world_basis": basis}
    proof = {"records": rows, "slots": len(valid), "coarse_ready": int(available.sum()), "coarse_valid": int(valid.sum()),
             "coarse_valid_nonrest": int((valid & ~data["rest"]).sum()), "C020_fine_exact_match": True,
             "C017_center_passed_independent_reconstruction": max(row["C017_center_reconstruction_max_abs"] for row in rows),
             "archive_sha256": {p.name: sha(p) for p in (archive, history, sample)},
             "C020_native_release_sha256": array_sha(native, "<i8"), "C020_source_ordinal_sha256": array_sha(ordinal, "<i8"),
             "C020_center_sha256": array_sha(data["center"], "<f8"), "C020_target_sha256": array_sha(data["truth"], "<f8"),
             "C020_rest_sha256": array_sha(data["rest"], "?")}
    return coarse_data, canonical, pooled, proof


def normalization(data, original, coarse, fine_mean, fine_std):
    valid = data["valid"]
    mean, std = c18.moments(coarse[valid])
    same_mean, same_std = c18.moments(original["pooled"][valid, 1])
    archived_mean, archived_std = fine_mean.cpu().numpy(), fine_std.cpu().numpy()
    fresh_mean, fresh_std = c18.moments(original["pooled"][:, 1])
    np.testing.assert_array_equal(fresh_mean, archived_mean)
    np.testing.assert_array_equal(fresh_std, archived_std)
    shift = (original["pooled"][valid, 1] - same_mean) / same_std - (original["pooled"][valid, 1] - archived_mean) / archived_std
    proof = {"weighting": "Equal per valid sampled slot, population std (ddof=0), original C019 floor. Includes duplicate and rest slots when valid.",
             "C019_frozen_fine_slots": len(valid), "matched_coarse_and_fine_slots": int(valid.sum()),
             "difference_due_to_subset_only": {"mean_max_abs": float(np.max(np.abs(archived_mean - same_mean))),
                                                "std_max_abs": float(np.max(np.abs(archived_std - same_std))),
                                                "fine_normalized_feature_shift_rms": float(np.sqrt(np.mean(shift.astype(np.float64)**2)))},
             "coarse_vs_same_subset_fine": {"mean_max_abs": float(np.max(np.abs(mean - same_mean))),
                                            "std_max_abs": float(np.max(np.abs(std - same_std)))},
             "coarse": {"mean": mean.tolist(), "std": std.tolist()},
             "fine_matched_subset": {"mean": same_mean.tolist(), "std": same_std.tolist()},
             "fine_C019_original": {"mean": archived_mean.tolist(), "std": archived_std.tolist()},
             "std_floor": c18.STD_MIN}
    return mean, std, proof

def prefit(branch, data, canonical, mean, std, average, scale):
    device = mean.device
    up, world, base_unit, ref, usable = (c18.to_device(x[:BATCH], device) for x in
          (data["canonical_up"], data["world_basis"], data["center"], data["truth"], data["valid"] & ~data["rest"]))
    x = base.normalized(c18.to_device(canonical[:BATCH], device), mean, std)
    with torch.no_grad():
        direction, delta = c20.predict(branch, x, up, world, usable, base_unit, average, scale)
        base.require(direction.shape == delta.shape == (96, 3) and c20.pooled_history(branch, x).shape == (96, 24),
                     "Prediction helper did not return the specified 96 x 3 direction or 96 x 24 feature")
        torch.testing.assert_close(direction, c18.unit(base_unit), rtol=0., atol=0.)
        base.require(torch.count_nonzero(delta).item() == 0, "First coarse branch is not the zero C017 augmentation")
        first_loss = float((direction - ref).square().sum(-1).mean())
    c20_loss = read(ROOT / "C020/pretraining-checks.json")["first_step_loss"]
    base.require(first_loss == c20_loss, "C021 first optimizer loss differs from the C020 zero head")
    np.testing.assert_array_equal(c20.angles(UP[None, :], UP[None, :]), [0.])
    return {"checked_steps": 160, "first_loss": first_loss, "exact_zero_residual": True, "exact_C017_first_direction": True,
            "first_usable_in_96": int(usable.sum()), "singleton_angle_shape_verified": True, "prediction_shape": list(direction.shape)}


def fit(branch, original_b6, data, canonical, mean, std, average, scale, first_loss, protocol_hash):
    torch.manual_seed(1729)
    started, tick = stamp(), perf_counter()
    original_hash = c4.tensor_hash(branch.state_dict())
    base.require(original_hash == read(ROOT / "C020/training.json")["initial_state_tensor_sha256"], "C021 B6/zero-head initialization differs")
    layout, components = c20.parameter_layout(branch)
    named = tuple(branch.named_parameters())
    base.require(all(p.data_ptr() != old.data_ptr() and not old.requires_grad
                     for (_, p), (_, old) in zip(named, original_b6.named_parameters(), strict=True)), "New branch shares original B6 storage")
    params = [p for name, p in named if not name.startswith("output.")]
    device = mean.device
    x = base.normalized(c18.to_device(canonical, device), mean, std)
    up, basis, center, truth = (c18.to_device(data[key], device) for key in ("canonical_up", "world_basis", "center", "truth"))
    enabled = c18.to_device(data["valid"] & ~data["rest"], device)
    branch.train().requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": params, "lr": BACKBONE_LR}, {"params": list(branch.output.parameters()), "lr": HEAD_LR}],
                                  lr=HEAD_LR, weight_decay=WD, foreach=False, fused=False)
    initial = c20.vector(named, lambda n, p: p).astype(np.float64)
    store = {key: [] for key in ("grad_raw", "grad_clipped", "decay_delta", "adam_gradient_delta")}
    checked_unusable = False
    step_path = HERE / "optimizer-steps.jsonl"
    torch.cuda.reset_peak_memory_stats()
    with step_path.open("x", encoding="utf-8") as log:
        for step in range(STEPS):
            part = slice(step * BATCH, (step + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            prediction, residual = c20.predict(branch, x[part], up[part], basis[part], enabled[part], center[part], average, scale)
            errors = (prediction - truth[part]).square().sum(-1)
            loss = errors.mean()
            if step == 0:
                base.require(float(loss.detach()) == first_loss and torch.count_nonzero(residual).item() == 0, "Changed real first step")
            unavailable = ~enabled[part]
            if not checked_unusable and unavailable.any():
                grads = torch.autograd.grad(errors[unavailable].sum(), [p for _, p in named], retain_graph=True)
                base.require(all(torch.count_nonzero(g).item() == 0 for g in grads), "Inactive coarse slot supplied gradients")
                checked_unusable = True
            loss.backward()
            base.require(all(p.grad is not None for _, p in named), "C021 has disconnected parameters")
            raw = c20.vector(named, lambda n, p: p.grad)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(branch.parameters(), CLIP))
            clipped = c20.vector(named, lambda n, p: p.grad)
            before = c20.vector(named, lambda n, p: p).astype(np.float64)
            decay_only = c20.vector(named, lambda n, p: p.detach() * (1. - layout[n]["lr"] * WD)).astype(np.float64)
            optimizer.step()
            final = c20.vector(named, lambda n, p: p).astype(np.float64)
            a, b = decay_only - before, final - decay_only
            for key, val in zip(store, (raw, clipped, a, b), strict=True):
                store[key].append(val)
            if step == 0:
                base.require(np.all(raw[:5520] == 0.) and np.all(b[:5520] == 0.) and np.any(raw[5520:] != 0.), "Bad zero-head attribution")
            loss_scalar = float(loss.detach())
            base.require(np.isfinite(np.r_[loss_scalar, grad_norm, raw, final]).all(), "C021 nonfinite optimizer step")
            row = {"pid": os.getpid(), "step": step + 1, "loss": loss_scalar, "global_grad_norm_before_clip": grad_norm,
                   "sources": c18.source_loss(errors.detach().cpu().numpy(), data["domain"][part]),
                   "available": int(data["ready"][part].sum()), "valid": int(data["valid"][part].sum()),
                   "batch_active": int(enabled[part].sum()), "rest": int(data["rest"][part].sum()),
                   "weight_after_sha256": array_sha(final, "<f8"), "decay_sha256": array_sha(a, "<f8"),
                   "adaptation_sha256": array_sha(b, "<f8"),
                   **{key + "_sha256": array_sha(data[key][part], "<i8") for key in SCHEDULE_FIELDS}}
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            if step + 1 in (1, 2, 40, 80, 120, 160):
                print("C021 fit", step + 1, loss_scalar, grad_norm, flush=True)
    base.require(checked_unusable, "No inactive slots checked for zero gradient")
    seconds = perf_counter() - tick
    trace = {key: np.stack(val) for key, val in store.items()}
    trace.update({"initial_weights": initial, "final_weights": final})
    param = c20.attribute(trace, layout, components)
    branch.eval().requires_grad_(False)
    tensor_path, attrib_path, checkpoint = (HERE / n for n in ("optimizer-tensors.npz", "parameter-attribution.json", "mamba-coarse.pt"))
    np.savez_compressed(tensor_path, **trace)
    write(attrib_path, {"layout_order": list(layout), "parameters": param, "optimizer_tensors_sha256": sha(tensor_path),
                       "weight_decay_split": "Float32 decoupled multiply, followed by actual AdamW adaptive update as in C020."})
    cfg = {"kind": branch.kind, "input_features": 13, "hidden_size": 24, "steps": 64, "block_ms": 40,
           "history_feature": "mean(H[:56])-mean(H[-8:])", "output_width": 3, "gain": GAIN,
           "cap_deg": base.CAP, "mixer_config": branch.sequence.config.to_dict()}
    torch.save({"state_dict": {k: p.detach().cpu() for k, p in branch.state_dict().items()}, "model": branch.kind,
                "configuration": cfg, "B6_input_mean": mean.cpu(), "B6_input_std": std.cpu(),
                "feature_mean": average.cpu(), "feature_std": scale.cpu(), "optimizer_steps": STEPS,
                "initial_state_tensor_sha256": original_hash, "protocol_sha256": protocol_hash}, checkpoint)
    return {"pid": os.getpid(), "started_at": started, "completed_at": stamp(), "fit_count": 1, "optimizer_steps": STEPS,
            "trainable_parameters": 5595, "seconds": seconds, "inactive_slot_gradient_zero": True,
            "initial_state_tensor_sha256": original_hash, "architecture": cfg, "weight_sha256": sha(checkpoint),
            "artifacts_sha256": {p.name: sha(p) for p in (step_path, tensor_path, attrib_path, checkpoint)}}

def snapshot(q, at, release, fine, acc, observed, dt, k, bank, coarse_bank, use=slice(None)):
    coarse, ready = ordered_coarse(fine, at, release)
    directions, deltas, rest, center, _ = c20.snapshot(q, at, release, fine, acc, observed, dt, k, bank, use)
    learned, average, scale = coarse_bank
    feat = c18.features(coarse[use], q[at[use]], learned, bank[1], bank[2])
    effective_rest = rest | ~ready[use]
    directions["C021_deg"], deltas["C021_deg"] = c20.learned_direction(learned, feat, center, effective_rest, average, scale)
    base.require(np.all(deltas["C021_deg"][effective_rest | ~feat.valid] == 0.), "Coarse inactive release did not hold C017")
    return directions, deltas, rest, ready[use], feat.valid & ready[use], center


def interventions(bank, coarse_bank):
    with np.load(HERE / "history-probes.npz", allow_pickle=False) as f:
        a = {key: f[key] for key in f.files}
    fine, coarse, quat, center = (a[k] for k in ("fine", "coarse", "quat", "center"))
    rest = c6.rest_flags(fine)
    base.require(len(coarse) == 12 and not rest.any(), "Controlled history intervention should use 12 active windows")
    learned, average, scale = coarse_bank
    mean, std = bank[1:3]
    frame = intrinsic.build_frame(coarse)
    f0 = c18.features(coarse, quat, learned, mean, std, frame=frame)
    original, original_delta = c20.learned_direction(learned, f0, center, rest, average, scale)
    controls, _ = c20.readout_bundle(fine, quat, center, rest, bank)
    changed = coarse.copy()
    changed[:, :32, :3] += np.asarray([.35, -.25, .15], dtype=np.float32)
    changed[:, :32, 3:6] += np.asarray([.035, -.025, .015], dtype=np.float32)
    np.testing.assert_array_equal(changed[:, 32:], coarse[:, 32:])
    # All perturbed coarse blocks precede the entire 640 ms input to the fine branch.
    same_fine, _ = c20.readout_bundle(fine, quat, center, rest, bank)
    np.testing.assert_array_equal(controls["C020_deg"], same_fine["C020_deg"])
    result = {}
    for label, chosen in (("fixed_original_basis", frame), ("recomputed_basis", None)):
        feat = c18.features(changed, quat, learned, mean, std, frame=chosen)
        v, delta = c20.learned_direction(learned, feat, center, rest, average, scale)
        angle = c20.angles(original, v)
        difference = np.linalg.norm(delta - original_delta, axis=-1)
        result[label] = [{"domain": str(a["domain"][i]), "record": str(a["record"][i]), "native_release": int(a["release"][i]),
                          "input_sha256": array_sha(coarse[i], "<f4"), "edited_input_sha256": array_sha(changed[i], "<f4"),
                          "C020_output_change_deg": 0., "C021_output_change_deg": float(angle[i]),
                          "C021_post_gain_tangent_change": float(difference[i])} for i in range(len(coarse))]
    return {"older_coarse_blocks_modified": 32, "length_of_perturbed_history_s": 1.28, "fine_640ms_input_unchanged": True,
            "C017_center_held_fixed": True, "accel_perturbation_m_s2": [.35, -.25, .15], "gyro_perturbation_rad_s": [.035, -.025, .015],
            "variants": result}


def audits(cfg, k, bank, coarse_bank):
    probes = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, obs = c14.synthetic(hz)
        _, q, at, release, w, _, acc, _ = c17.front_end(raw, obs, dt, cfg, optical)
        dir0, delta0, rest, ready, valid, center = snapshot(q, at, release, w, acc, obs, dt, k, bank, coarse_bank)
        base.require(ready.any() and not ready[0] and np.any(~rest & valid), "Timing fixture lacks coarse or early cases")
        first = int(np.flatnonzero(ready)[0])
        split = {first - 1, first, len(w) // 2}
        delayed = np.flatnonzero((release > at) & ready)
        if len(delayed):
            split.add(int(delayed[len(delayed) // 2]))
        max_angle = 0.
        for j in sorted(split):
            changed = raw.copy()
            changed[at[j] + 1:, :3] += [16., -9., 8.]
            changed[at[j] + 1:, 3:] += 1.1
            changed_obs = obs.copy()
            changed_obs[at[j] + 2::2] = False
            end = j + 1
            truncate = int(release[j]) + 1
            for x, good in ((changed, changed_obs), (raw[:truncate], obs[:truncate])):
                _, qr, ar, rr, h, _, ac, _ = c17.front_end(x, good, dt, cfg, optical)
                np.testing.assert_array_equal(ar[:end], at[:end])
                np.testing.assert_array_equal(rr[:end], release[:end])
                g, d, static, ready2, _, _ = snapshot(qr, ar, rr, h, ac, good, dt, k, bank, coarse_bank, slice(0, end))
                np.testing.assert_array_equal(ready2, ready[:end])
                np.testing.assert_array_equal(static, rest[:end])
                np.testing.assert_allclose(d["C021_deg"], delta0["C021_deg"][:end], atol=3e-8, rtol=0)
                max_angle = max(max_angle, float(c20.angles(g["C021_deg"], dir0["C021_deg"][:end]).max()))
        coarse, _ = ordered_coarse(w, at, release)
        q_last = Rotation.from_quat(q[at], scalar_first=True)
        q_pub = Rotation.from_quat(q[release], scalar_first=True)
        learned, average, scale = coarse_bank
        angles, physical = [], []
        for rot in (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                    Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))):
            wc = coarse.copy()
            wc[:, :, :12] = rot.apply(coarse[:, :, :12].reshape(-1, 3)).reshape(len(w), 64, 12)
            feat = c18.features(wc, (q_last * rot.inv()).as_quat(scalar_first=True), learned, bank[1], bank[2])
            o, _ = c20.learned_direction(learned, feat, center, rest | ~ready, average, scale)
            angles.append(float(c20.angles(dir0["C021_deg"], o).max()))
            rotated = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, qr, ar, rr, hr, _, acr, _ = c17.front_end(rotated, obs, dt, cfg, optical)
            np.testing.assert_array_equal(release, rr)
            newest, _, _, rrdy, _, _ = snapshot(qr, ar, rr, hr, acr, obs, dt, k, bank, coarse_bank)
            np.testing.assert_array_equal(rrdy, ready)
            moved_back = rot.inv().apply(Rotation.from_quat(qr[rr], scalar_first=True).inv().apply(newest["C021_deg"]))
            physical.append(float(c20.angles(q_pub.inv().apply(dir0["C021_deg"]), moved_back).max()))
        base.require(max_angle <= 1e-5 and max(angles) <= 1e-4, "Coarse branch future or rotation audit failed")
        probes.append({"native_hz": hz, "windows": len(w), "first_coarse_release": int(release[first]),
                       "first_coarse_input_last": int(at[first]), "early_windows": first,
                       "prefix_or_future_edit_max_angle_deg": max_angle, "cuts": sorted(split),
                       "fixed_world_rotation_max_deg": max(angles), "raw_replay_rotation_max_deg": max(physical)})
    return probes

def metrics(rows):
    scored = [row for row in rows if row["status"] == "scored"]
    macro = {k: fmean(a[k] for a in scored) for k in METHODS} if scored else {}
    return {"records": len(scored), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in scored),
            "macro_tilt_rmse_deg": macro,
            "C021_delta_deg": {k: macro["C021_deg"] - macro[k] for k in METHODS[:-1]} if scored else {},
            "C021_wins": {k: sum(a["C021_deg"] < a[k] for a in scored) for k in METHODS[:-1]}}


def compare_controls(values, old):
    gap = max(abs(values[k] - old[k]) for k in METHODS[:-1])
    base.require(np.isfinite(list(values.values())).all() and gap <= 1e-6, f"C020 or other frozen control changed by {gap}")
    return gap


def replay(prior, inventory, campaign, split, cfg, k, bank, coarse_bank):
    c1r, c1p, c3r, c2p = (read(ROOT / n) for n in ("C001/results.json", "C001/protocol.json", "C003/results.json", "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in a["sequences"]} for a in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(a["side"] in ("development", "development_previously_used_for_selection") for a in specs.values()),
                 "C021 replay crossed the development boundary")
    mp, ap = DATA / "Sassari/medium_v5.mat", DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp) == split["source_file_sha256"][str(mp)] and sha(ap) == evidence["array_reference_sha256"],
                 "Frozen reference file verification failed")
    medium = loadmat(mp, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(ap)
    pose_index = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures, coverage = [], [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for a in prior["sequences"]:
            start = perf_counter()
            key = a["dataset"], a["record"]
            identity = {f: a[f] for f in c17.IDENTITY}
            try:
                data, _, observed, dt, truth, mask, axes, _ = c17.payload(key, specs[key], map3, map1, medium, yt["valid"],
                                                                          split, c1p, array, evidence, cfg)
                q0, q, at, release, fine, _, acc, _ = data
                np.testing.assert_array_equal(mask, masks[a["mask_key"]])
                ix = np.flatnonzero(mask)
                base.require(mask.sum() == a["scored_frames"] and array_sha(ix, "<i8") == a["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == a["native_input_last_sha256"]
                             and array_sha(release, "<i8") == a["native_release_sha256"], "C021/C020 native mask or release differs")
                infer = perf_counter()
                output, deltas, rest, ready, usable, _ = snapshot(q, at, release, fine, acc, observed, dt, k, bank, coarse_bank)
                pred_s = perf_counter() - infer
                gate = gates[a["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, truth, q0, q, release, {m: np.vstack((UP, v)) for m, v in output.items()})
                score = c17.scores(errs)
                gap = compare_controls(score, a)
                lookup = np.searchsorted(release, ix, side="right")
                static = np.r_[False, rest][lookup]
                not_ready = ~np.r_[False, ready][lookup]
                no_fit = static | ~np.r_[False, usable][lookup]
                np.testing.assert_array_equal(errs["C021_deg"][no_fit], errs["C017_deg"][no_fit])
                base.require(int(static.sum()) == a["static_scoring_frames"], "C020 static population differs")
                details = []
                for pose, portion in axes.items():
                    sub = portion[ix]
                    old = pose_index[key[1], pose]
                    sig = array_sha(ix[sub], "<i8")
                    base.require(int(sub.sum()) == old["scored_frames"] and sig == old["scoring_indices_sha256"], "Pose scoring frame mismatch")
                    values = c17.scores(errs, sub)
                    details.append({"record": key[1], "pose": pose, "scored_frames": int(sub.sum()), "scoring_indices_sha256": sig,
                                    **values, "control_max_abs_delta_deg": compare_controls(values, old)})
                sections = []
                for label, use in (("coarse_ready", ~not_ready), ("coarse_unavailable", not_ready)):
                    if use.any():
                        sections.append({"dataset": key[0], "record": key[1], "section": label, "scored_frames": int(use.sum()),
                                         "scoring_indices_sha256": array_sha(ix[use], "<i8"), "status": "scored", **c17.scores(errs, use)})
                added = deltas["C021_deg"]
                rows.append({**identity, **score, "status": "scored", "control_max_abs_delta_deg": gap,
                             "native_input_last_sha256": a["native_input_last_sha256"], "native_release_sha256": a["native_release_sha256"],
                             "coarse_ready_releases": int(ready.sum()), "coarse_early_releases": int((~ready).sum()),
                             "coarse_valid_releases": int(usable.sum()), "coarse_active_releases": int(np.count_nonzero(np.any(added != 0., axis=1))),
                             "coarse_unavailable_scoring_frames": int(not_ready.sum()), "static_scoring_frames": int(static.sum()),
                             "coarse_ready_sha256": array_sha(ready, "?"), "C021_delta_sha256": array_sha(added, "<f8"),
                             "C021_max_post_gain_norm": float(np.linalg.norm(added, axis=1).max()),
                             "prediction_seconds": pred_s, "seconds": perf_counter() - start})
                coverage.extend(sections)
                poses.extend(details)
                print("development", *key, *(score[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - start})
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    seq, pos, group, availability = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json", "availability-scores.csv"))
    columns = dict.fromkeys(k for row in rows for k in row)
    intrinsic.shice.save_csv(seq, [{k: r.get(k, "") for k in columns} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pos, poses)
    if coverage:
        intrinsic.shice.save_csv(availability, coverage)
    groups = {field: {f"{d}/{label}": metrics([r for r in rows if r["dataset"] == d and r[field] == label])
                     for d, label in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("group", "reference_QA", "instrument", "scenario")}
    groups["coarse_coverage"] = {f"{d}/{label}": metrics([r for r in coverage if r["dataset"] == d and r["section"] == label])
                                 for d, label in sorted({(r["dataset"], r["section"]) for r in coverage})}
    write(group, groups)
    return {"pooled_records": metrics(rows), "domains": {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "coverage": {label: metrics([r for r in coverage if r["section"] == label]) for label in ("coarse_ready", "coarse_unavailable")},
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifact_sha256": {item.name: sha(item) for item in (seq, pos, group, availability) if item.is_file()}}

def load_controls(protocol):
    files = protocol["inherited_file_sha256"]
    base.require(all(sha(ROOT / n) == h for n, h in files.items())
                 and all(sha(PILOT / n) == h for n, h in protocol["frozen_B6_C017_files_sha256"].items()),
                 "Inherited control or helper changed")
    campaign, p14 = read(ROOT / "protocol.json"), read(ROOT / "C014/protocol.json")
    models, mean, std, controls, averages, scales = c20.controls_and_bank(p14, campaign, files)
    stored = torch.load(ROOT / "C020/mamba-end-to-end.pt", weights_only=True, map_location=mean.device)
    base.require(stored["optimizer_steps"] == 160 and stored["architecture"]["window"] == 64, "Archived C020 checkpoint differs")
    for a, b in ((stored["B6_input_mean"], mean), (stored["B6_input_std"], std),
                 (stored["C019_feature_mean"], averages[1]), (stored["C019_feature_std"], scales[1])):
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
    fine = deepcopy(models[0])
    fine.load_state_dict(stored["state_dict"])
    fine.eval().requires_grad_(False)
    return (models, mean, std, controls, averages, scales, fine), campaign


def write_status(protocol, phase, **items):
    write(HERE / "process.json", {"pid": os.getpid(), "original_fitting_pid": protocol["pid"],
                                 "started_at": protocol["started_at"], "updated_at": stamp(),
                                 "status": phase, "protocol_sha256": sha(HERE / "protocol.json"), **items})


def after_lock(protocol, bank, campaign, inventory, cfg, recovery_sha=None):
    started = perf_counter()
    lock_path, saved = HERE / "branch-lock.json", HERE / "mamba-coarse.pt"
    locked = read(lock_path)
    lock_hash = sha(lock_path)
    base.require(locked["protocol_sha256"] == sha(HERE / "protocol.json") and locked["C021_checkpoint_sha256"] == sha(saved)
                 and locked["training_sha256"] == sha(HERE / "training.json") and locked["optimizer_steps"] == 160
                 and locked["C020_branch_lock_sha256"] == sha(ROOT / "C020/branch-lock.json")
                 and locked["C019_sampling_lock_sha256"] == sha(ROOT / "C019/sampling-lock.json")
                 and locked["C017_coefficient_lock_sha256"] == sha(ROOT / "C017/coefficient-lock.json"),
                 "Locked coarse checkpoint or inherited identity changed")
    models, mean, std, *_ = bank
    checkpoint = torch.load(saved, map_location=mean.device, weights_only=True)
    learned = deepcopy(models[0])
    learned.load_state_dict(checkpoint["state_dict"])
    learned.eval().requires_grad_(False)
    for a, b in ((checkpoint["B6_input_mean"], mean), (checkpoint["B6_input_std"], std)):
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
    coarse_bank = learned, checkpoint["feature_mean"], checkpoint["feature_std"]
    k = protocol["C017_k"]
    tag = {"branch_lock_sha256": lock_hash, "recovery_sha256": recovery_sha}
    write_status(protocol, "learned_coarse_audits", **tag)
    intervention_path, audit_path = HERE / "old-context-intervention.json", HERE / "causal-rotation-audit.json"
    write(audit_path, {**tag, "checks": audits(cfg, k, bank, coarse_bank)})
    write(intervention_path, {**tag, **interventions(bank, coarse_bank)})
    write_status(protocol, "development_replay", **tag)
    print("C021 coarse branch locked; 213-record replay started", flush=True)
    tick = perf_counter()
    result = replay(read(ROOT / "C020/results.json"), inventory, campaign, read(TRANSFER / "external-split.json"), cfg, k, bank, coarse_bank)
    base.require(sha(lock_path) == lock_hash and sha(saved) == locked["C021_checkpoint_sha256"]
                 and all(sha(ROOT / p) == h for p, h in protocol["inherited_file_sha256"].items())
                 and all(sha(PILOT / p) == h for p, h in protocol["frozen_B6_C017_files_sha256"].items()),
                 "A locked file changed after replay")
    result.update({**tag, "pid": os.getpid(), "original_fitting_pid": protocol["pid"], "started_at": protocol["started_at"],
                   "completed_at": stamp(), "protocol_sha256": sha(HERE / "protocol.json"),
                   "C020_branch_lock_sha256": locked["C020_branch_lock_sha256"],
                   "C019_sampling_lock_sha256": locked["C019_sampling_lock_sha256"],
                   "scoring_masks_sha256": sha(MASKS), "causal_rotation_audit_sha256": sha(audit_path),
                   "old_context_intervention_sha256": sha(intervention_path), "final_holdout_scored": False,
                   "post_lock_seconds": perf_counter() - started, "development_replay_seconds": perf_counter() - tick})
    result_path = HERE / "results.json"
    write(result_path, result)
    okay = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    write_status(protocol, "complete" if okay else "incomplete", **tag, scored=result["pooled_records"]["records"],
                 results_sha256=sha(result_path))
    print("C021 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not okay:
        raise RuntimeError("C021 replay incomplete; branch lock is preserved")


def main():
    launch, initial = (read(HERE / n) for n in ("launch.json", "input-verification.json"))
    files = initial["files_sha256"]
    base.require(os.getpid() == launch["pid"] == initial["pid"], "Lost registration")
    p20, p17, lock = (read(ROOT / n) for n in ("C020/protocol.json", "C017/protocol.json", "C017/coefficient-lock.json"))
    base.require(len(p20["development_records"]) == 213 and p20["development_records"] == p17["development_records"]
                 and p20["scoring_masks_sha256"] == sha(MASKS) and p17["gate_traces_sha256"] == sha(GATES),
                 "Development panel changed")
    base.require(p20["optimizer"] == {"kind": "AdamW", "seed": 1729, "steps": STEPS, "batch": BATCH,
                 "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "weight_decay": WD, "global_clip": CLIP,
                 "foreach": False, "fused": False}, "C020 optimizer configuration changed")
    protocol_path = HERE / "protocol.json"
    protocol = {**launch, "code_sha256": sha(Path(__file__)), "inherited_file_sha256": files,
                "frozen_B6_C017_files_sha256": p20["C017_dependencies_sha256"], "C017_k": lock["k"],
                "coarse_window": "64 x 40 ms; four causal 10 ms blocks per token. Average six-axis values, take terminal VQF state.",
                "window_builder": "review/multirate_blocks.py::long_windows called on full ordered per-record release stream before sampling.",
                "inactive": "Before 256 visible 10 ms blocks, or during C020 static or degenerate geometry gates, the extra residual is exactly zero.",
                "feature_normalization": "Frozen B6 13-channel normalization; new feature moments on exactly the coarse geometrically valid C020 slots.",
                "gain": GAIN, "tangent_cap_deg": base.CAP, "feature": "mean(H[:56])-mean(H[-8:])",
                "optimizer": p20["optimizer"], "development_records": p20["development_records"], "scoring_masks_sha256": sha(MASKS)}
    write(protocol_path, protocol)
    bank, campaign = load_controls(protocol)
    models, mean, std, controls, averages, scales, _ = bank
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    inventory = read(ROOT / "C001/inventory.json")
    write_status(protocol, "static_schedule_check")
    orig_data, prior = verify_locked_schedule(files)
    write_status(protocol, "reconstruct_complete_ordered_streams")
    data, canonical, features, prep = reconstruct(inventory, orig_data, prior, models, mean, std, cfg, lock["k"])
    prep_path, norm_path, preflight_path = (HERE / n for n in ("training-preparation.json", "feature-normalization.json", "pretraining-checks.json"))
    write(prep_path, prep)
    avg, sd, norm = normalization(data, orig_data, features, averages[1], scales[1])
    write(norm_path, norm)
    del orig_data, features
    avg, sd = c18.to_device(avg, mean.device), c18.to_device(sd, mean.device)
    learned = deepcopy(models[0])
    torch.nn.init.zeros_(learned.output.weight)
    torch.nn.init.zeros_(learned.output.bias)
    check = prefit(learned, data, canonical, mean, std, avg, sd)
    write(preflight_path, check)
    print("C021 preflight complete", {**check, "coarse_ready": prep["coarse_ready"], "coarse_valid": prep["coarse_valid"],
                                     "matched_norm_difference": norm["difference_due_to_subset_only"]}, flush=True)
    write_status(protocol, "single_coarse_branch_fit", preparation_sha256=sha(prep_path), normalizer_sha256=sha(norm_path))
    initial_controls = tuple(c4.tensor_hash(m.state_dict()) for m in models)
    result = fit(learned, models[0], data, canonical, mean, std, avg, sd, check["first_loss"], sha(protocol_path))
    base.require(tuple(c4.tensor_hash(m.state_dict()) for m in models) == initial_controls
                 and all(p.grad is None for m in models for p in m.parameters()), "Frozen C017 branches changed during training")
    del data, canonical
    train_path, locked = HERE / "training.json", HERE / "branch-lock.json"
    write(train_path, {**result, "pretraining_checks_sha256": sha(preflight_path), "feature_normalization_sha256": sha(norm_path),
                       "reconstructed_training_sha256": sha(prep_path), "frozen_C017_source_tensor_sha256": list(initial_controls)})
    write(locked, {"pid": os.getpid(), "locked_at": stamp(), "protocol_sha256": sha(protocol_path), "optimizer_steps": 160,
                   "C021_checkpoint_sha256": result["weight_sha256"], "training_sha256": sha(train_path),
                   "C019_sampling_lock_sha256": files["C019/sampling-lock.json"], "C020_branch_lock_sha256": files["C020/branch-lock.json"],
                   "C017_coefficient_lock_sha256": files["C017/coefficient-lock.json"]})
    after_lock(protocol, bank, campaign, inventory, cfg)


def resume():
    protocol, lock = (read(HERE / n) for n in ("protocol.json", "branch-lock.json"))
    state = read(HERE / "process.json")
    base.require(state["status"] != "complete" and state["original_fitting_pid"] == lock["pid"] == protocol["pid"],
                 "Only a locked incomplete C021 can resume")
    previous = [n for n in ("failure.json", "results.json") if (HERE / n).is_file()]
    ticket = HERE / "replay-recovery.json"
    write(ticket, {"pid": os.getpid(), "started_at": stamp(), "fitting_pid": lock["pid"], "optimizer_steps_repeated": 0,
                   "branch_lock_sha256": sha(HERE / "branch-lock.json"), "recovery_code_sha256": sha(Path(__file__)),
                   "previous_failure_evidence_sha256": {n: sha(HERE / n) for n in previous}})
    bank, campaign = load_controls(protocol)
    inventory = read(ROOT / "C001/inventory.json")
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    after_lock(protocol, bank, campaign, inventory, cfg, sha(ticket))


if __name__ == "__main__":
    if "--resume" in sys.argv[1:]:
        resume()
    else:
        main()
