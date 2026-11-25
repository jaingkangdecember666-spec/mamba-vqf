"""Fit and compare two frozen-B6 temporal readouts around the locked C017 center."""
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from time import perf_counter
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "C017"))

import replay_stable_force as c17
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn
from torch.nn import functional as F

c14, c6, c4, c3, c2 = c17.c14, c17.c6, c17.c4, c17.c3, c17.c2
base, intrinsic = c17.base, c17.intrinsic
read, write, sha, stamp, array_sha = c17.read, c17.write, c17.digest, c17.stamp, c17.array_sha
UP, DOMAINS = c17.UP, c17.DOMAINS
MASKS, GATES, PILOT, DATA, TRANSFER = c17.MASKS, c17.GATES, c17.PILOT, c17.DATA, c17.TRANSFER
WIDTH, STEPS, BATCH = 24, 160, 96
GAIN, LR, WD, CLIP, STD_MIN = .125, .0003, .01, 2., 1e-4
METHODS = ("full_vqf_deg", "C017_deg", "A_deg", "B_deg")


@dataclass
class Features:
    pooled: np.ndarray
    up: np.ndarray
    world_basis: np.ndarray
    valid: np.ndarray
    hidden: np.ndarray | None


def hidden(core, x):
    h = core.input(x)
    return h + core.sequence(core.norm(h))


@torch.inference_mode()
def features(windows, q_last, core, mean, std, frame=None, keep_hidden=False):
    base.require(windows.shape[1:] == (64, 13), "Frozen B6 window shape changed")
    if frame is None:
        frame = intrinsic.intrinsic_windows(windows)[0]
    n = len(windows)
    pooled = np.zeros((n, 2, WIDTH), dtype=np.float32)
    up = np.zeros((n, 3), dtype=np.float32)
    all_hidden = np.zeros((n, 64, WIDTH), dtype=np.float32) if keep_hidden else None
    for start in range(0, n, 256):
        positions = np.arange(start, min(start + 256, n))
        use = positions[frame.valid[positions]]
        if not len(use):
            continue
        fixed = type(frame)(frame.basis[use], frame.source[use], frame.anchor[use])
        canonical = intrinsic.transform_inputs(windows[use], fixed)
        h = hidden(core, base.normalized(torch.from_numpy(canonical).to(mean.device), mean, std))
        pooled[use, 0] = h[:, -1].cpu().numpy()
        pooled[use, 1] = (h[:, :56].mean(dim=1) - h[:, -8:].mean(dim=1)).cpu().numpy()
        up[use] = canonical[:, -1, 6:9]
        if keep_hidden:
            all_hidden[use] = h.cpu().numpy()
    rotation = c14.normalize(q_last)
    world = np.stack([c14.rotate(rotation, frame.basis[:, :, k]) for k in range(3)], axis=-1)
    base.require(np.isfinite(pooled).all() and np.isfinite(world).all(), "Invalid frozen B6 representation")
    return Features(pooled, up, world, frame.valid, all_hidden)


def to_device(x, device):
    return torch.as_tensor(x, device=device)


def tensors(f, rest, center, device):
    return tuple(to_device(a, device) for a in (f.pooled, f.up, f.world_basis, f.valid & ~rest, center))


def residual(layer, z, up, world_basis, enabled):
    # Original B6 tangent projection and cap, then transform to the shared world tangent.
    intrinsic_delta = base.tangent(layer(z), up, base.CAP)
    intrinsic_delta = intrinsic_delta.masked_fill(~enabled[:, None], 0.)
    world = torch.bmm(world_basis, intrinsic_delta.to(world_basis.dtype).unsqueeze(-1)).squeeze(-1)
    horizontal = torch.cat((world[:, :2], torch.zeros_like(world[:, 2:])), dim=-1)
    return GAIN * horizontal


def unit(center):
    return F.normalize(center, dim=-1)


def moments(pooled):
    # Empirical population moments over the actual draws, including repeated and static slots.
    x = pooled.astype(np.float64)
    return x.mean(axis=0).astype(np.float32), np.maximum(x.std(axis=0), STD_MIN).astype(np.float32)


def training_features(inventory, p15, cfg, core, mean, std, k, lock):
    c15 = c17.c15
    prep = read(ROOT / "C015/training-preparation.json")
    prev = read(ROOT / "C017/external-selection.json")
    sample_path, center_path = ROOT / "C015/sampling.npz", ROOT / "C017/training-slots.npz"
    base.require(sha(sample_path) == lock["C015_sampling_sha256"] == prep["sampling_sha256"]
                 and sha(center_path) == prev["source_slots_sha256"], "C015 slot or C017 center file changed")
    with np.load(sample_path, allow_pickle=False) as ledger, np.load(center_path, allow_pickle=False) as saved:
        fixed = {name: saved[name] for name in saved.files}
        for a, b in (("source_record_ordinal", "C015_record_ordinal"), ("native_release", "C015_native_release"),
                     ("global_window", "C015_global_window")):
            base.require(ledger[b].shape == (STEPS, BATCH), "C015 step shape changed")
            np.testing.assert_array_equal(fixed[a], ledger[b].ravel())
    ordinal, native, last, rest = (fixed[key] for key in ("source_record_ordinal", "native_release", "native_input_last", "rest"))
    center = fixed["C014_B"] + c17.additions(fixed["force"], k)[0]
    source = p15["legacy_train_sources"] + p15["ridi_sources"]
    base.require(len(source) == 67 and len(ordinal) == STEPS * BATCH and np.all(last <= native), "Frozen source populations changed")
    pooled = np.empty((len(native), 2, WIDTH), dtype=np.float32)
    up = np.empty((len(native), 3), dtype=np.float32)
    world = np.empty((len(native), 3, 3))
    valid = np.empty(len(native), dtype=bool)
    rows, mats, probes = [], {}, []
    probed_domains = set()
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, spec in enumerate(source):
            started = perf_counter()
            domain, name = spec.get("domain", "RIDI"), spec["record"]
            draw = np.flatnonzero(ordinal == i)
            np.testing.assert_array_equal(fixed["domain"][draw], domain)
            if domain == "RIDI":
                raw, observed, dt, mask, reference, hashes = c17.trained_phone(spec, prep["ridi_eligibility"][i - 63])
            else:
                raw, observed, dt, mask, reference, hashes = c4.train_raw(spec, yt["train"], mats)
                base.require(hashes == prep["original_sources"][i]["raw_hashes"], f"Archived raw fields changed: {name}")
            previous = prev["records"][i]
            base.require((domain, name, hashes) == (previous["domain"], previous["record"], previous["source_sha256"]),
                         "C017 slot source changed")
            _, q, at, release, histories, complete, _, bias = c17.front_end(raw, observed, dt, cfg)
            unique, inverse = np.unique(native[draw], return_inverse=True)
            position = np.searchsorted(release, unique)
            np.testing.assert_array_equal(release[position], unique)
            np.testing.assert_array_equal(at[position][inverse], last[draw])
            np.testing.assert_array_equal(c6.rest_flags(histories[position])[inverse], rest[draw])
            base.require(np.all(mask[unique] & complete[position]) and bias["active_native_frames"] == previous["C003_active_native_frames"],
                         "Frozen training release or C003 activation changed")
            np.testing.assert_array_equal(c14.rotate(c14.normalize(q[unique]), reference[unique])[inverse], fixed["truth"][draw])
            x, q_last = histories[position], q[at[position]]
            feat = features(x, q_last, core, mean, std, keep_hidden=True)
            base.require(feat.valid.all(), "Training slot has degenerate intrinsic frame")
            pooled[draw], up[draw], world[draw], valid[draw] = (f[inverse] for f in (feat.pooled, feat.up, feat.world_basis, feat.valid))
            if i in (0, 27, 39, 63):
                historical = c4.features(core, x, mean, std)[0]
                np.testing.assert_array_equal(historical, feat.pooled[:, 0])
            if domain not in probed_domains:
                eligible = np.flatnonzero(~c6.rest_flags(x) & feat.valid)
                if len(eligible) >= 3:
                    selected = eligible[np.linspace(0, len(eligible) - 1, 3, dtype=int)]
                    for j in selected:
                        slot = int(draw[np.flatnonzero(inverse == j)[0]])
                        probes.append({"domain": domain, "record": name, "native_release": int(unique[j]),
                                       "history": x[j], "q_last": q_last[j], "center": center[slot]})
                    probed_domains.add(domain)
            rows.append({"ordinal": i, "domain": domain, "record": name, "slots": len(draw), "rest": int(rest[draw].sum()),
                         "unique_input_windows": len(x), "input_history_sha256": array_sha(x, "<f4"),
                         "frozen_hidden_sha256": array_sha(feat.hidden, "<f4"), "native_release_sha256": array_sha(native[draw], "<i8"),
                         "source_sha256": hashes, "seconds": perf_counter() - started})
            print("C018 features", domain, name, len(draw), flush=True)
    base.require(probed_domains == set(DOMAINS), "Missing paired active old-context probe")
    out = HERE / "training-inputs.npz"
    np.savez_compressed(out, pooled=pooled, canonical_up=up, world_basis=world, valid=valid, rest=rest, center=center,
                        truth=fixed["truth"], source_record_ordinal=ordinal, native_release=native, native_input_last=last,
                        global_window=fixed["global_window"], domain=fixed["domain"], record_equal_weights=fixed["weight"])
    proof = {"C015_sampling_sha256": sha(sample_path), "C017_training_slots_sha256": sha(center_path),
             "training_inputs_sha256": sha(out), "step_slot_shape": [STEPS, BATCH], "slots": len(native),
             "slot_order_sha256": array_sha(fixed["global_window"].reshape(STEPS, BATCH), "<i8"),
             "native_releases_sha256": array_sha(native, "<i8"), "records": rows}
    return Features(pooled, up, world, valid, None), fixed, center, probes, proof

def exact_zero_heads(layers, f, rest, center, truth, averages, scales):
    device = averages.device
    pooled, up, world, enabled, center_t = tensors(f, rest, center, device)
    expected = unit(center_t)
    target = to_device(truth, device)
    modes = []
    for i, layer in enumerate(layers):
        z = (pooled[:, i] - averages[i]) / scales[i]
        delta = residual(layer, z, up, world, enabled)
        prediction = unit(center_t + delta)
        torch.testing.assert_close(prediction, expected, rtol=0., atol=0.)
        base.require(torch.count_nonzero(delta).item() == 0, "Zero readout changed C017")
        modes.append({"arm": "AB"[i], "zero_residual_every_slot": True, "exact_equal_C017_in_runtime": True,
                      "first_batch_loss": float((prediction[:BATCH] - target[:BATCH]).square().sum(-1).mean().detach())})
    error = float(np.max(np.abs(expected.cpu().numpy() - c14.normalize(center))))
    base.require(error <= 3e-16, "Float64 base direction does not reproduce archived C017 normalization")
    return {"slots": len(rest), "initial_unit_max_abs_difference_numpy": error, "arms": modes}


def source_loss(loss, domain):
    return {d: {"slots": int(np.count_nonzero(domain == d)), "mean_chord_loss": float(np.mean(loss[domain == d]))}
            for d in DOMAINS if np.any(domain == d)}


def flat_params(layer, use_grad=False):
    return torch.cat([(p.grad if use_grad else p.detach()).reshape(-1) for p in layer.parameters()]).detach().cpu().numpy()


def fit_arm(i, layer, f, fixed, center, averages, scales, protocol_hash, first_check):
    start, started = perf_counter(), stamp()
    device = averages.device
    pooled, up, world, enabled, center_t = tensors(f, fixed["rest"], center, device)
    z = (pooled[:, i] - averages[i]) / scales[i]
    truth = to_device(fixed["truth"], device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=LR, weight_decay=WD)
    name = "AB"[i]
    path = HERE / f"{name}-steps.jsonl"
    rest_gradient_checked = False
    with path.open("x", encoding="utf-8") as out:
        for step in range(STEPS):
            sl = slice(step * BATCH, (step + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            before = flat_params(layer)
            delta = residual(layer, z[sl], up[sl], world[sl], enabled[sl])
            err = (unit(center_t[sl] + delta) - truth[sl]).square().sum(-1)
            loss = err.mean()
            rest = fixed["rest"][sl]
            if rest.any() and not rest_gradient_checked:
                grads = torch.autograd.grad(err[to_device(rest, device)].sum(), tuple(layer.parameters()), retain_graph=True)
                base.require(all(torch.count_nonzero(g).item() == 0 for g in grads), "Rest-only loss leaks gradient")
                rest_gradient_checked = True
            before_loss = float(loss.detach())
            if step == 0:
                base.require(before_loss == first_check, "First optimizer batch differs from the zero readout audit")
            loss.backward()
            raw_grad = flat_params(layer, use_grad=True)
            grad_norm = float(torch.nn.utils.clip_grad_norm_(layer.parameters(), CLIP))
            clipped = flat_params(layer, use_grad=True)
            optimizer.step()
            after = flat_params(layer)
            with torch.no_grad():
                update = residual(layer, z[sl], up[sl], world[sl], enabled[sl])
                after_loss = (unit(center_t[sl] + update) - truth[sl]).square().sum(-1).cpu().numpy()
            base.require(np.isfinite(np.r_[before_loss, grad_norm, after_loss, after, raw_grad]).all(), "Nonfinite optimizer step")
            domain = fixed["domain"][sl]
            per_source = source_loss(err.detach().cpu().numpy(), domain)
            for d, values in source_loss(after_loss, domain).items():
                per_source[d]["post_step_mean_chord_loss"] = values["mean_chord_loss"]
            row = {"arm": name, "step": step + 1, "mean_chord_loss": before_loss, "mean_chord_loss_after": float(after_loss.mean()),
                   "grad_norm_before_clip": grad_norm, "grad_before_clip": raw_grad.tolist(), "grad_after_clip": clipped.tolist(),
                   "weights_before": before.tolist(), "weights_after": after.tolist(), "sources": per_source,
                   "nonrest_slots": int((~rest).sum()), "rest_slots": int(rest.sum()),
                   "draw_global_window_sha256": array_sha(fixed["global_window"][sl], "<i8"),
                   "record_ordinals_sha256": array_sha(fixed["source_record_ordinal"][sl], "<i8"),
                   "native_release_sha256": array_sha(fixed["native_release"][sl], "<i8")}
            print(json.dumps(row, allow_nan=False), file=out, flush=True)
            if step + 1 in (1, 40, 80, 120, 160):
                print("C018 fit", name, step + 1, before_loss, grad_norm, flush=True)
    base.require(rest_gradient_checked and len(after) == 75 and np.any(after != 0.), f"{name} fitting or rest check incomplete")
    layer.eval().requires_grad_(False)
    saved = HERE / f"{name}-readout.pt"
    torch.save({"state_dict": {key: val.detach().cpu() for key, val in layer.state_dict().items()},
                "feature_mean": averages[i].cpu(), "feature_std": scales[i].cpu(),
                "feature": ("H[-1]", "mean(H[:56])-mean(H[-8:])")[i],
                "steps": STEPS, "gain": GAIN, "protocol_sha256": protocol_hash}, saved)
    with torch.inference_mode():
        direction = unit(center_t)
        delta = residual(layer, z, up, world, enabled)
        after = unit(center_t + delta)
        torch.testing.assert_close(after[~enabled], direction[~enabled], rtol=0., atol=0.)
        old_loss = (direction - truth).square().sum(-1).cpu().numpy()
        new_loss = (after - truth).square().sum(-1).cpu().numpy()
        norms = delta.norm(dim=1).cpu().numpy()
    domains = {}
    for d in DOMAINS:
        use = fixed["domain"] == d
        weight = fixed["weight"][use]
        base.require(abs(weight.sum() - 1.) < 1e-12, "Invalid record-equal source weights")
        opened = norms[use] > 0.
        domains[d] = {"draws": int(use.sum()), "nonzero_readout_slots": int(opened.sum()),
                      "mean_post_gain_tangent_norm": float(norms[use].mean()), "max_post_gain_tangent_norm": float(norms[use].max()),
                      "slot_equal_C017": float(old_loss[use].mean()), "slot_equal_fitted": float(new_loss[use].mean()),
                      "record_equal_C017": float(weight @ old_loss[use]), "record_equal_fitted": float(weight @ new_loss[use])}
    return {"arm": name, "pid": os.getpid(), "started_at": started, "completed_at": stamp(), "seconds": perf_counter() - start,
            "optimizer_steps": STEPS, "trainable_parameters": len(flat_params(layer)), "weights_l2": float(np.linalg.norm(flat_params(layer))),
            "rest_gradient_zero": rest_gradient_checked, "rest_deployed_delta_max_abs": float(norms[fixed["rest"]].max()),
            "domains": domains, "artifacts_sha256": {p.name: sha(p) for p in (path, saved)}}


@torch.inference_mode()
def directions(f, rest, center, layers, averages, scales):
    pooled, up, world, enabled, center_t = tensors(f, rest, center, averages.device)
    values, offsets = {}, {}
    old = c14.normalize(center)
    for i, (name, layer) in enumerate(zip(METHODS[2:], layers, strict=True)):
        z = (pooled[:, i] - averages[i]) / scales[i]
        delta = residual(layer, z, up, world, enabled)
        normalized = unit(center_t + delta)
        a = delta.cpu().numpy()
        base.require(np.all(a[rest] == 0.) and np.all(a[~f.valid] == 0.)
                     and np.max(np.linalg.norm(a, axis=1)) <= GAIN * np.tan(np.deg2rad(base.CAP)) + 1e-8,
                     "Readout gate or original tilt cap changed")
        active = np.any(a != 0., axis=1)
        prediction = old.copy()
        prediction[active] = normalized.cpu().numpy()[active]
        values[name], offsets[name] = prediction, a
    return values, offsets


def window_output(q, at, release, w, acc, observed, dt, models, mean, std, lock_k,
                  layers, averages, scales, select=slice(None)):
    q_last = q[at[select]]
    window = w[select]
    center, rest = c17.branch_at(q_last, window, models, mean, std)
    force, _, valid_last = c17.force_at(q, acc, observed, dt, at[select], c17.window_starts(dt, len(at))[select], rest)
    base.require(np.all(at[select] <= release[select]) and np.all(valid_last <= at[select]), "Unreleased input used")
    center += c17.additions(force, lock_k)[0]
    f = features(window, q_last, models[0], mean, std)
    a_b, offsets = directions(f, rest, center, layers, averages, scales)
    return {"C017_deg": c14.normalize(center), **a_b}, offsets, rest, f.valid, center


def old_context_probes(probes, core, mean, std, layers, averages, scales):
    x, q, center = (np.stack([r[key] for r in probes]) for key in ("history", "q_last", "center"))
    f = features(x, q, core, mean, std)
    rest = c6.rest_flags(x)
    base.require(len(x) == 12 and not rest.any() and f.valid.all(), "Old-context probes are not active")
    changed = x.copy()
    changed[:, :32, :3] += np.array([.35, -.25, .15], dtype=np.float32)
    changed[:, :32, 3:6] += np.array([.035, -.025, .015], dtype=np.float32)
    np.testing.assert_array_equal(changed[:, 32:], x[:, 32:])
    basis = intrinsic.intrinsic_windows(x)[0]
    original, offsets = directions(f, rest, center, layers, averages, scales)
    trials = {}
    for label, use in (("fixed_original_basis", basis), ("recomputed_basis", None)):
        candidate = features(changed, q, core, mean, std, frame=use)
        output, new_offsets = directions(candidate, rest, center, layers, averages, scales)
        trials[label] = [{"domain": row["domain"], "record": row["record"], "native_release": row["native_release"],
                          "input_sha256": array_sha(x[j], "<f4"), "perturbed_input_sha256": array_sha(changed[j], "<f4"),
                          "valid_after_edit": bool(candidate.valid[j]),
                          "features_delta_l2": {k: float(np.linalg.norm(candidate.pooled[j, i] - f.pooled[j, i]))
                                                for i, k in enumerate(METHODS[2:])},
                          "output_angle_change_deg": {k: float(c14.tilt_error(original[k][j:j + 1], output[k][j:j + 1])[0]) for k in output},
                          "post_gain_delta_change_l2": {k: float(np.linalg.norm(new_offsets[k][j] - offsets[k][j])) for k in output}}
                         for j, row in enumerate(probes)]
    return {"support": len(x), "old_blocks_changed": 32, "recent_blocks_unchanged": 32, "shared_C017_center_held_fixed": True,
            "accel_perturbation_m_s2": [.35, -.25, .15], "gyro_perturbation_rad_s": [.035, -.025, .015], "trials": trials}

def audits(cfg, models, mean, std, k, layers, averages, scales):
    prefix, rotation = [], []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, observed = c14.synthetic(hz)
        _, q, at, release, windows, _, acc, _ = c17.front_end(raw, observed, dt, cfg, optical)
        result, offsets, rest, _, center = window_output(q, at, release, windows, acc, observed, dt, models, mean, std,
                                                           k, layers, averages, scales)
        max_prefix = dict.fromkeys(METHODS[1:], 0.)
        max_delta = dict.fromkeys(METHODS[2:], 0.)
        cut_set = {len(at) // 2, len(at) - 2}
        between = np.flatnonzero(release > at)
        if len(between):
            cut_set.add(int(between[len(between) // 2]))
        for j in sorted(cut_set):
            changed = raw.copy()
            changed[at[j] + 1:, :3] += [14., -8., 6.]
            changed[at[j] + 1:, 3:] += .94
            future_mask = observed.copy()
            future_mask[at[j] + 2::2] = False
            truncate = int(release[j]) + 1
            for x, good in ((changed, future_mask), (raw[:truncate], observed[:truncate])):
                _, q2, at2, release2, w2, _, acc2, _ = c17.front_end(x, good, dt, cfg, optical)
                end = j + 1
                np.testing.assert_array_equal(at[:end], at2[:end])
                np.testing.assert_array_equal(release[:end], release2[:end])
                np.testing.assert_array_equal(windows[:end], w2[:end])
                p, d, r, _, _ = window_output(q2, at2, release2, w2, acc2, good, dt, models, mean, std,
                                               k, layers, averages, scales, slice(0, end))
                np.testing.assert_array_equal(rest[:end], r)
                for name in p:
                    max_prefix[name] = max(max_prefix[name], float(c14.tilt_error(result[name][:end], p[name]).max()))
                for name in d:
                    max_delta[name] = max(max_delta[name], float(np.max(np.abs(offsets[name][:end] - d[name]))))
        base.require(max(max_prefix.values()) <= 1e-5 and max(max_delta.values()) <= 3e-8, "New branch reads beyond the authorized window")
        frame = intrinsic.intrinsic_windows(windows)[0]
        use = np.flatnonzero(frame.valid & ~rest)[::7]
        view = type(frame)(frame.basis[use], frame.source[use], frame.anchor[use])
        canonical = intrinsic.transform_inputs(windows[use], view)
        with torch.inference_mode():
            x = base.normalized(to_device(canonical, mean.device), mean, std)
            h = hidden(models[0], x)[:, :39]
            changed = x.clone()
            changed[:, 39:, :6] *= -3.
            inner = max(float((h - hidden(models[0], changed)[:, :39]).abs().max()),
                        float((h - hidden(models[0], x[:, :39])).abs().max()))
        base.require(inner <= 2e-5, "Frozen hidden states use later tokens in fixed coordinates")
        prefix.append({"native_hz": hz, "cuts": sorted(cut_set), "release_after_input_count": len(between),
                       "direction_max_change_deg": max_prefix, "residual_max_abs_change": max_delta,
                       "inner_frozen_hidden_prefix_max_abs": inner, "rest_windows": int(rest.sum())})
        to_world = Rotation.from_quat(q[at], scalar_first=True)
        native_ref = Rotation.from_quat(q[release], scalar_first=True)
        for rotname, rot in zip(("90x", "90y", "90z", "63_1_2_-1"),
                                (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))), strict=True):
            new_w = windows.copy()
            new_w[..., :12] = rot.apply(windows[..., :12].reshape(-1, 3)).reshape(len(windows), 64, 12)
            q_rot = (to_world * rot.inv()).as_quat(scalar_first=True)
            ft = features(new_w, q_rot, models[0], mean, std)
            values, d = directions(ft, rest, center, layers, averages, scales)
            fixed = {key: float(c14.tilt_error(result[key], values[key]).max()) for key in values}
            delta_diff = {key: float(np.max(np.abs(d[key] - offsets[key]))) for key in d}
            base.require(max(fixed.values()) <= 1e-4, "Readout installation rotation differs in a fixed world")
            changed = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, q3, at3, rel3, w3, _, acc3, _ = c17.front_end(changed, observed, dt, cfg, optical)
            np.testing.assert_array_equal(at, at3)
            np.testing.assert_array_equal(release, rel3)
            native, _, r3, _, _ = window_output(q3, at3, rel3, w3, acc3, observed, dt, models, mean, std,
                                                  k, layers, averages, scales)
            inv3 = Rotation.from_quat(q3[rel3], scalar_first=True).inv()
            native_err = {key: float(c14.tilt_error(native_ref.inv().apply(result[key]),
                                                       rot.inv().apply(inv3.apply(v))).max()) for key, v in native.items()}
            rotation.append({"native_hz": hz, "rotation": rotname, "windows": len(windows),
                             "fixed_world_max_deg": fixed, "fixed_world_post_gain_residual_max_abs": delta_diff,
                             "raw_installation_max_body_tilt_deg": native_err, "raw_installation_rest_changes": int((rest != r3).sum())})
    return {"prefix": prefix, "rotation": rotation}


def metrics(rows):
    valid = [r for r in rows if r["status"] == "scored"]
    macro = {m: fmean(r[m] for r in valid) for m in METHODS} if valid else {}
    return {"records": len(valid), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in valid),
            "macro_tilt_rmse_deg": macro,
            "delta_vs_C017_deg": {m: macro[m] - macro["C017_deg"] for m in METHODS[2:]} if valid else {},
            "wins_vs_C017": {m: sum(r[m] < r["C017_deg"] for r in valid) for m in METHODS[2:]},
            "B_wins_vs_A": sum(r["B_deg"] < r["A_deg"] for r in valid)}


def compare_center(values, archived):
    error = max(abs(values[k] - archived[k]) for k in METHODS[:2])
    base.require(error <= 1e-6 and np.isfinite(list(values.values())).all(), f"Frozen C017/FullVQF controls differ by {error}")
    return error


def replay(prior, inventory, campaign, cfg, split, models, mean, std, k, layers, averages, scales):
    c1r, c1p, c3r, c2p = (read(p) for p in (ROOT / "C001/results.json", ROOT / "C001/protocol.json", ROOT / "C003/results.json",
                                             ROOT / "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "Found out-of-scope records in C018 replay")
    mp, ap = DATA / "Sassari/medium_v5.mat", DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp) == split["source_file_sha256"][str(mp)] and sha(ap) == evidence["array_reference_sha256"],
                 "Frozen reference checksums changed")
    medium = loadmat(mp, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(ap)
    old_poses = {(a["record"], a["pose"]): a for a in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            started = perf_counter()
            key = old["dataset"], old["record"]
            identity = {field: old[field] for field in c17.IDENTITY}
            try:
                data, _, observed, dt, ref, mask, axes, _ = c17.payload(key, specs[key], map3, map1, medium, yt["valid"], split,
                                                                         c1p, array, evidence, cfg)
                q0, q, at, release, windows, _, acc, _ = data
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                indices = np.flatnonzero(mask)
                base.require(len(indices) == old["scored_frames"] and intrinsic.mask_digest(mask) == old["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == old["native_input_last_sha256"]
                             and array_sha(release, "<i8") == old["native_release_sha256"], "C017 native timeline or scoring frames changed")
                before = perf_counter()
                prediction, offsets, rest, valid, _ = window_output(q, at, release, windows, acc, observed, dt, models, mean, std,
                                                                      k, layers, averages, scales)
                prediction_seconds = perf_counter() - before
                native_gates = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, native_gates[:, 0])
                np.testing.assert_array_equal(rest, native_gates[:, 3])
                estimates = {m: np.vstack((UP, direction)) for m, direction in prediction.items()}
                errors = c17.measure(indices, ref, q0, q, release, estimates)
                score = c17.scores(errors)
                gap = compare_center(score, old)
                lookup = np.searchsorted(release, indices, side="right")
                static = np.r_[False, rest][lookup]
                for key_arm in METHODS[2:]:
                    np.testing.assert_array_equal(errors["C017_deg"][static], errors[key_arm][static])
                details = []
                for pose, subset in axes.items():
                    use = subset[indices]
                    archived = old_poses[key[1], pose]
                    index_sha = array_sha(indices[use], "<i8")
                    base.require(index_sha == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"],
                                 "C017 pose scoring indices changed")
                    val = c17.scores(errors, use)
                    details.append({"record": key[1], "pose": pose, "scoring_indices_sha256": index_sha, "scored_frames": int(use.sum()),
                                    "control_max_abs_delta_deg": compare_center(val, archived), **val})
                residuals = {}
                for mode, off in offsets.items():
                    label = mode[0]
                    magnitude = np.linalg.norm(off, axis=1)
                    angular = c14.tilt_error(prediction["C017_deg"], prediction[mode])
                    residuals.update({f"{label}_delta_sha256": array_sha(off, "<f8"),
                                      f"{label}_active_releases": int(np.count_nonzero(magnitude)),
                                      f"{label}_post_gain_norm_mean": float(magnitude.mean()),
                                      f"{label}_post_gain_norm_max": float(magnitude.max()),
                                      f"{label}_tilt_increment_deg_max": float(angular.max()),
                                      f"{label}_static_max_abs": float(np.max(np.abs(off[rest]), initial=0.))})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_delta_deg": gap,
                             "native_release_sha256": old["native_release_sha256"], "native_input_last_sha256": old["native_input_last_sha256"],
                             "static_scoring_frames": int(static.sum()), "rest_releases": int(rest.sum()), "scored_releases": len(release),
                             "valid_nonrest_releases": int((valid & ~rest).sum()), **residuals,
                             "paired_prediction_seconds": prediction_seconds, "seconds": perf_counter() - started})
                poses.extend(details)
                print("development", *key, *(score[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - started})
                print("FAILED", *key, repr(exc), flush=True)
    seq, pos, groups = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json"))
    fields = dict.fromkeys(f for r in rows for f in r)
    intrinsic.shice.save_csv(seq, [{f: r.get(f, "") for f in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pos, poses)
    group_summaries = {field: {f"{d}/{label}": metrics([r for r in rows if r["dataset"] == d and r[field] == label])
                              for d, label in sorted({(r["dataset"], r[field]) for r in rows})}
                       for field in ("group", "reference_QA", "instrument", "scenario")}
    write(groups, group_summaries)
    return {"pooled_records": metrics(rows), "domains": {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifacts_sha256": {p.name: sha(p) for p in (seq, pos, groups) if p.is_file()}}

def main():
    start = perf_counter()
    launch, initial = (read(HERE / n) for n in ("launch.json", "input-verification.json"))
    base.require(os.getpid() == launch["pid"] == initial["pid"], "C018 did not stay in the registered PID")
    for name, expected in initial["files_sha256"].items():
        base.require(sha(ROOT / name) == expected, f"Preflight input changed: {name}")
    p14, p15, p17, campaign, lock = (read(ROOT / name) for name in ("C014/protocol.json", "C015/protocol.json",
                              "C017/protocol.json", "protocol.json", "C017/coefficient-lock.json"))
    lock_hash = sha(ROOT / "C017/coefficient-lock.json")
    base.require(lock_hash == initial["files_sha256"]["C017/coefficient-lock.json"]
                 and lock["predevelopment_lock"] and lock["no_parameter_updates_after_lock"], "C017 coefficient was not frozen")
    base.require(p17["scoring_masks_sha256"] == p15["scoring_masks_sha256"] == sha(MASKS)
                 and p17["gate_traces_sha256"] == p15["gate_traces_sha256"] == sha(GATES)
                 and p17["development_records"] == p15["development_records"] and len(p17["development_records"]) == 213,
                 "Development panel no longer matches C017")
    k = lock["k"]
    models, mean, std, b6, b7 = c17.integrity_models(p14, campaign)
    cfgp = ROOT / "C003/protocol.json"
    cfg = c3.Settings(**read(cfgp)["thresholds"])
    inventory_path, split_path = ROOT / "C001/inventory.json", TRANSFER / "external-split.json"
    important = (inventory_path, split_path, ROOT / "C003/online_bias.py", cfgp, ROOT / "C004/fit_readouts.py", b6, b7)
    before = {str(p.relative_to(PILOT)): sha(p) for p in important}
    base.require(all(p14["files_sha256"][name] == value for name, value in before.items()), "An inherited C017 dependency changed")
    # Restore two separately archived branches with frozen B6 input normalization.
    base.require(all(not p.requires_grad for m in models for p in m.parameters()), "C017 branch is trainable")
    norms_sha = array_sha(torch.stack((mean, std)).cpu().numpy(), "<f4")
    protocol_path = HERE / "protocol.json"
    protocol = {**launch, "source_worker_sha256": sha(Path(__file__)), "inherited_input_sha256": before,
                "C017_coefficient_lock_sha256": lock_hash, "C017_gain": k,
                "C015_sampling_sha256": lock["C015_sampling_sha256"], "C017_stable_force_recipe": p17["force"],
                "scoring_masks_sha256": sha(MASKS), "gate_traces_sha256": sha(GATES),
                "development_records": p17["development_records"], "B6_input_normalization_sha256": norms_sha,
                "arms": {"A": "H[-1]", "B": "mean(H[:56])-mean(H[-8:])"},
                "new_feature_normalization": "each arm/channel has empirical population moments over the exact ordered C015 slots including repeats and rest",
                "std_floor": STD_MIN, "per_arm_parameters": 75, "steps_per_arm": STEPS, "batch": BATCH,
                "optimizer": {"name": "AdamW", "learning_rate": LR, "weight_decay": WD, "gradient_clip": CLIP},
                "gain": GAIN, "linear_output_scale": .1, "tangent_cap_deg": base.CAP,
                "loss": "batch mean squared chord distance in normalized world directions at native release, including constant rest losses",
                "release": "frozen C017 center + new projected/capped 0.125 world-horizontal tangent on nonrest; shared normalization",
                "C017_preflight_files_sha256": initial["files_sha256"]}
    write(protocol_path, protocol)
    protocol_hash = sha(protocol_path)

    def status(phase, **details):
        write(HERE / "process.json", {"status": phase, "pid": os.getpid(), "started_at": launch["started_at"],
                                      "updated_at": stamp(), "protocol_sha256": protocol_hash, **details})

    status("feature_extraction")
    print("C018 external features started", os.getpid(), flush=True)
    inventory = read(inventory_path)
    f, fixed, center, probes, evidence = training_features(inventory, p15, cfg, models[0], mean, std, k, lock)
    input_path = HERE / "training-preparation.json"
    write(input_path, evidence)
    moment_mean, moment_std = moments(f.pooled)
    averages, scales = (to_device(a, mean.device) for a in (moment_mean, moment_std))
    moments_path = HERE / "feature-normalization.json"
    write(moments_path, {"A": {"mean": moment_mean[0].tolist(), "std": moment_std[0].tolist()},
                         "B": {"mean": moment_mean[1].tolist(), "std": moment_std[1].tolist()},
                         "B6_input_normalization_sha256": norms_sha, "source_slots": len(f.pooled), "std_floor": STD_MIN,
                         "C015_sampling_sha256": evidence["C015_sampling_sha256"]})
    layers = [nn.Linear(WIDTH, 3).to(mean.device) for _ in range(2)]
    for layer in layers:
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        base.require(sum(p.numel() for p in layer.parameters() if p.requires_grad) == 75, "Bad readout parameter count")
    zero = exact_zero_heads(layers, f, fixed["rest"], center, fixed["truth"], averages, scales)
    write(HERE / "zero-initialization-audit.json", zero)
    training = []
    for i, layer in enumerate(layers):
        status(f"fitting_{'AB'[i]}")
        training.append(fit_arm(i, layer, f, fixed, center, averages, scales, protocol_hash, zero["arms"][i]["first_batch_loss"]))
        for key, val in torch.load(HERE / f"{'AB'[i]}-readout.pt", map_location=mean.device, weights_only=True)["state_dict"].items():
            torch.testing.assert_close(val, layer.state_dict()[key], rtol=0., atol=0.)
    training_path = HERE / "training.json"
    write(training_path, {"fit_count": 2, "single_pid": os.getpid(), "arms": training, "C015_sampling_sha256": evidence["C015_sampling_sha256"],
                          "normalization_sha256": sha(moments_path), "training_preparation_sha256": sha(input_path),
                          "zero_initialization_sha256": sha(HERE / "zero-initialization-audit.json")})
    pair_path = HERE / "pair-lock.json"
    write(pair_path, {"pid": os.getpid(), "locked_at": stamp(), "trained_arms": ["A", "B"],
                      "training_sha256": sha(training_path), "protocol_sha256": protocol_hash,
                      "C017_coefficient_lock_sha256": lock_hash,
                      "readout_sha256": {m: sha(HERE / f"{m}-readout.pt") for m in "AB"}})
    pair_hash = sha(pair_path)
    status("learned_branch_audits", pair_lock_sha256=pair_hash)
    probe_path = HERE / "old-context-intervention.json"
    write(probe_path, {"pair_lock_sha256": pair_hash, **old_context_probes(probes, models[0], mean, std, layers, averages, scales)})
    audit_path = HERE / "causal-rotation-audit.json"
    write(audit_path, {"pair_lock_sha256": pair_hash, **audits(cfg, models, mean, std, k, layers, averages, scales)})
    status("development_replay", pair_lock_sha256=pair_hash)
    print("C018 A/B locked, audits passed, replaying 213 development records", flush=True)
    eval_start = perf_counter()
    result = replay(read(ROOT / "C017/results.json"), inventory, campaign, cfg, read(split_path), models, mean, std, k,
                    layers, averages, scales)
    base.require(sha(ROOT / "C017/coefficient-lock.json") == lock_hash and sha(pair_path) == pair_hash
                 and all(sha(PILOT / name) == value for name, value in before.items())
                 and all(sha(HERE / f"{m}-readout.pt") == h for m, h in read(pair_path)["readout_sha256"].items()),
                 "A frozen dependency changed before finalizing C018")
    result.update({"pid": os.getpid(), "started_at": launch["started_at"], "completed_at": stamp(),
                   "protocol_sha256": protocol_hash, "pair_lock_sha256": pair_hash, "C017_coefficient_lock_sha256": lock_hash,
                   "development_replay_seconds": perf_counter() - eval_start, "total_seconds": perf_counter() - start,
                   "scoring_masks_sha256": sha(MASKS), "causal_rotation_audit_sha256": sha(audit_path),
                   "old_context_intervention_sha256": sha(probe_path), "final_holdout_scored": False})
    output = HERE / "results.json"
    write(output, result)
    success = not result["failures"] and len(result["array_sensor_poses"]) == 96 and result["pooled_records"]["records"] == 213
    status("complete" if success else "incomplete", results_sha256=sha(output), pair_lock_sha256=pair_hash,
           scored=result["pooled_records"]["records"])
    print("C018 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not success:
        raise RuntimeError("C018 development checks incomplete")


if __name__ == "__main__":
    main()
