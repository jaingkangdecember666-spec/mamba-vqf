"""C019: one B-type readout with a frozen, broader YT training trial panel."""
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from time import perf_counter
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "C018"))

import run_pair as c18
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn

c17, c14, c6, c4, c3, c2 = c18.c17, c18.c14, c18.c6, c18.c4, c18.c3, c18.c2
base, intrinsic = c18.base, c18.intrinsic
read, write, sha, stamp, array_sha = c18.read, c18.write, c18.sha, c18.stamp, c18.array_sha
PILOT, DATA, TRANSFER, MASKS, GATES = c18.PILOT, c18.DATA, c18.TRANSFER, c18.MASKS, c18.GATES
UP, SEED, DOMAINS = c18.UP, 20261123, c18.DOMAINS
STEPS, BATCH, GAIN = c18.STEPS, c18.BATCH, c18.GAIN
METHODS = ("full_vqf_deg", "C017_deg", "C018_B_deg", "C019_deg")
NEW_FEATURES = ("pooled", "canonical_up", "world_basis", "valid", "rest", "center", "truth", "native_input_last")


def stable_order(items, namespace):
    return sorted(items, key=lambda v: (sha256(f"C019|{SEED}|{namespace}|{v}".encode("utf-8")).digest(), v))


def frozen_panel(inventory, panel, p15):
    base.require(panel["inventory_sha256"] == sha(ROOT / "C001/inventory.json")
                 and panel["original_protocol_sha256"] == sha(ROOT / "C004/protocol.json"),
                 "Panel inventory or original training protocol hash changed")
    rows = panel["records"]
    names = [r["record"] for r in rows]
    yt = [r for r in inventory["records"] if r["dataset"] == "YT_Server"]
    eligible = [r for r in yt if r["official_split"] == "train" and r["side"] == "external_training_candidate"]
    chosen = {r["record"]: r for r in eligible if r["record"] in names}
    forbidden = {r["group"] for r in yt if r["official_split"] == "test" or r["side"].startswith("development")}
    declared_old = [r["record"] for r in rows if r["in_original_24"]]
    declared_new = [r["record"] for r in rows if not r["in_original_24"]]
    sources = p15["legacy_train_sources"] + p15["ridi_sources"]
    old_trial = {r["record"]: i for i, r in enumerate(sources) if r.get("domain") == "YT_Server"}
    groups = {r["group"] for r in rows}
    base.require(len(eligible) == panel["eligible_records"] == 910 and len({r["group"] for r in eligible}) == panel["eligible_trial_groups"]
                 and len(names) == len(set(names)) == len(chosen) == len(groups) == 96 and len(declared_new) == 72
                 and set(declared_old) == set(old_trial) and len(old_trial) == 24 and not groups & forbidden
                 and not panel["intersection_with_development_and_official_test_trial_groups"], "Frozen panel membership differs")
    base.require(Counter(r["activity_code"] for r in rows) == panel["activity_counts"]
                 and Counter(r["leading_identifier"] for r in rows) == panel["leading_identifier_counts"]
                 and sorted(panel["activity_counts"].values()) == [19, 19, 19, 19, 20]
                 and len(panel["leading_identifier_counts"]) == 27, "Frozen panel diversity metadata changed")
    row_by_name = {r["record"]: r for r in rows}
    for name, spec in chosen.items():
        r = row_by_name[name]
        leading, code, *_ = name.split("_")
        base.require((spec["group"], spec["native_frames"], spec["source"]) ==
                     (r["group"], r["native_frames"], "/train/" + name)
                     and (leading, code) == (r["leading_identifier"], r["activity_code"]),
                     "Frozen trial identity, split or native length differs")
    return sources, old_trial, declared_new, chosen, row_by_name


def selection_plan(inventory, panel, p15, original):
    sources, old_trial, new_names, chosen, panel_rows = frozen_panel(inventory, panel, p15)
    ordinal = original["source_record_ordinal"]
    yt_old = original["domain"] == "YT_Server"
    retained = np.zeros(len(ordinal), dtype=bool)
    for name, i in old_trial.items():
        draw = np.flatnonzero(ordinal == i)
        base.require(len(draw) > 0 and np.all(yt_old[draw]), f"Original YT slots missing for {name}")
        retained[draw[::4]] = True
    replaced = yt_old & ~retained
    old_slots = stable_order(np.flatnonzero(replaced).tolist(), "replaced-slot-order")
    new_order = stable_order(new_names, "new-trial-order")
    assignment = {name: np.sort(np.asarray(old_slots[j::len(new_order)], dtype=np.int64))
                  for j, name in enumerate(new_order)}
    counts = [len(x) for x in assignment.values()]
    base.require(int(yt_old.sum()) == 4685 and sum(counts) == int(replaced.sum())
                 and min(counts) > 0 and max(counts) - min(counts) <= 1, "YT dose or balanced assignment changed")
    all_slots = np.concatenate(list(assignment.values()))
    np.testing.assert_array_equal(np.sort(all_slots), np.flatnonzero(replaced))
    return sources, old_trial, new_order, chosen, panel_rows, retained, replaced, assignment


def load_original(initial):
    prep = read(ROOT / "C018/training-preparation.json")
    fit = read(ROOT / "C018/training.json")
    path = ROOT / "C018/training-inputs.npz"
    base.require(sha(path) == prep["training_inputs_sha256"] == initial["C018/training-inputs.npz"]
                 and sha(ROOT / "C018/training-preparation.json") == fit["training_preparation_sha256"],
                 "Exact C018 source input archive changed")
    with np.load(path, allow_pickle=False) as a, np.load(ROOT / "C015/sampling.npz", allow_pickle=False) as ledger:
        original = {key: a[key] for key in a.files}
        for old, new in (("global_window", "C015_global_window"), ("native_release", "C015_native_release"),
                         ("source_record_ordinal", "C015_record_ordinal")):
            base.require(ledger[new].shape == (STEPS, BATCH), "Original training batch dimensions changed")
            np.testing.assert_array_equal(original[old], ledger[new].ravel())
    return original


def prepare(inventory, p15, panel, orig, cfg, models, mean, std, k):
    sources, old_trial, new_order, chosen, panel_rows, retained, replace, assignments = selection_plan(inventory, panel, p15, orig)
    data = {key: arr.copy() for key, arr in orig.items() if key != "record_equal_weights"}
    data["retained_original_YT"] = retained
    data["replacement_YT"] = replace
    slots = {}
    source_names = [r["record"] for r in sources] + new_order
    panel_by_name = {r["record"]: r for r in panel["records"]}
    old_slot_proof = read(ROOT / "C018/training-preparation.json")["records"]
    for name, j in old_trial.items():
        flat = np.flatnonzero(retained & (orig["source_record_ordinal"] == j))
        releases = orig["native_release"][flat]
        slots[name] = {**panel_by_name[name], "slots": len(flat), "unique_native_releases": len(np.unique(releases)),
                       "flat_slots_sha256": array_sha(flat, "<i8"), "sampled_release_sha256": array_sha(releases, "<i8"),
                       "source_sha256": old_slot_proof[j]["source_sha256"]}
    new_base = len(sources)
    new_evidence = []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        official_train = yt["train"]
        for j, name in enumerate(new_order):
            started = perf_counter()
            inv = chosen[name]
            spec = {"domain": "YT_Server", "record": name, "native_frames": inv["native_frames"]}
            base.require(official_train[name].name == inv["source"] and name in panel_rows, "Source is not approved official train")
            raw, observed, dt, mask, ref, hashes = c4.train_raw(spec, official_train, {})
            _, q, at, release, windows, complete, acc, bias = c17.front_end(raw, observed, dt, cfg)
            candidate = np.flatnonzero(complete & mask[release])
            geometry = intrinsic.build_frame(windows[candidate])
            eligible = candidate[geometry.valid]
            allowed = release[eligible]
            count = len(assignments[name])
            base.require(len(allowed) > 0 and np.all(at[eligible] <= allowed), f"No eligible causal release for {name}")
            ordered = np.asarray(stable_order(allowed.tolist(), "eligible-release|" + name), dtype=np.int64)
            selected = np.resize(ordered, count)
            # Distinct where possible; if an eligible pool is exhausted only then cycle.
            base.require(len(np.unique(selected)) == min(count, len(allowed)), "Unexpected repeated new release")
            native, inverse = np.unique(selected, return_inverse=True)
            pos = np.searchsorted(release, native)
            np.testing.assert_array_equal(release[pos], native)
            last = at[pos]
            x, q_last = windows[pos], q[last]
            feat = c18.features(x, q_last, models[0], mean, std, keep_hidden=True)
            center, rest = c17.branch_at(q_last, x, models, mean, std)
            force, support, last_valid = c17.force_at(q, acc, observed, dt, last, c17.window_starts(dt, len(at))[pos], rest)
            center += c17.additions(force, k)[0]
            truth = c14.rotate(c14.normalize(q[native]), ref[native])
            base.require(feat.valid.all() and np.all(last <= native) and np.all(last_valid <= last)
                         and np.isfinite(truth).all() and np.allclose(np.linalg.norm(truth, axis=1), 1., rtol=0., atol=2e-7),
                         "New C019 slot violates timeline or direction geometry")
            draw = assignments[name]
            for key, value in zip(NEW_FEATURES, (feat.pooled, feat.up, feat.world_basis, feat.valid, rest, center, truth, last), strict=True):
                data[key][draw] = value[inverse]
            data["native_release"][draw] = selected
            data["source_record_ordinal"][draw] = new_base + j
            slots[name] = {**panel_by_name[name], "slots": count, "unique_native_releases": len(native),
                           "eligible_native_releases": len(allowed), "rest_slots": int(rest[inverse].sum()),
                           "C017_force_gate_slots": int(support[inverse].sum()),
                           "flat_slots_sha256": array_sha(draw, "<i8"), "sampled_release_sha256": array_sha(selected, "<i8"),
                           "eligible_release_sha256": array_sha(allowed, "<i8"), "source_sha256": hashes}
            new_evidence.append({"record": name, "history_sha256": array_sha(x, "<f4"), "full_hidden_sha256": array_sha(feat.hidden, "<f4"),
                                 "C003_active_native_frames": bias["active_native_frames"], "seconds": perf_counter() - started})
            print("C019 YT new trial", name, count, len(native), len(allowed), flush=True)
    base.require(np.all(data["native_input_last"] <= data["native_release"]) and np.all(data["valid"])
                 and int((data["domain"] == "YT_Server").sum()) == 4685, "Replaced slot timeline or YT dose changed")
    for f in ("global_window", "domain"):
        np.testing.assert_array_equal(data[f], orig[f])
    for f in (*NEW_FEATURES, "source_record_ordinal", "native_release"):
        np.testing.assert_array_equal(data[f][~replace], orig[f][~replace])
    index = data["source_record_ordinal"]
    base.require(len(np.unique(index)) == len(source_names) == 139 and len(slots) == 96, "A frozen YT trial was omitted")
    domain_order = [r.get("domain", "RIDI") for r in sources] + ["YT_Server"] * len(new_order)
    domain_per_source = Counter(domain_order)
    count = np.bincount(index, minlength=len(source_names))
    data["record_equal_weights"] = 1. / (count[index] * np.asarray([domain_per_source[domain] for domain in domain_order])[index])
    data["source_record_name"] = np.asarray(source_names)[index]
    path = HERE / "training-inputs.npz"
    np.savez_compressed(path, **data)
    evidence = {"original_YT_slots": 4685, "YT_original_slots_retained": int(retained.sum()), "YT_slots_replaced": int(replace.sum()),
                "other_source_slots_exact": int(np.count_nonzero(data["domain"] != "YT_Server")),
                "new_YT_slots_min": min(len(v) for v in assignments.values()), "new_YT_slots_max": max(len(v) for v in assignments.values()),
                "new_YT_unique_release_total": sum(slots[name]["unique_native_releases"] for name in new_order),
                "original_YT_unique_release_total": sum(slots[name]["unique_native_releases"] for name in old_trial),
                "panel_sha256": sha(ROOT / "review/yt_training_panel.json"), "C015_sampling_sha256": sha(ROOT / "C015/sampling.npz"),
                "training_inputs_sha256": sha(path), "native_release_sha256": array_sha(data["native_release"].reshape(STEPS, BATCH), "<i8"),
                "source_record_ordinal_sha256": array_sha(index.reshape(STEPS, BATCH), "<i8"),
                "retained_original_YT_mask_sha256": array_sha(retained, "?"), "replacement_YT_mask_sha256": array_sha(replace, "?"),
                "new_YT_trial_order": new_order, "source_records": source_names,
                "trials": [slots[r["record"]] for r in panel["records"]], "new_source_input_fingerprints": new_evidence}
    return data, evidence

def fit(data, feature_mean, feature_std, layer, protocol_hash, schedule_hash):
    started = perf_counter()
    device = feature_mean.device
    feat = c18.Features(data["pooled"], data["canonical_up"], data["world_basis"], data["valid"], None)
    pooled, up, basis, enabled, center = c18.tensors(feat, data["rest"], data["center"], device)
    z = (pooled[:, 1] - feature_mean) / feature_std
    truth = c18.to_device(data["truth"], device)
    params = tuple(layer.parameters())
    loss_per = lambda angles: (angles - truth).square().sum(-1)
    with torch.no_grad():
        origin = c18.unit(center)
        init = c18.residual(layer, z, up, basis, enabled)
        zero = c18.unit(center + init)
        torch.testing.assert_close(zero, origin, rtol=0., atol=0.)
        base.require(torch.count_nonzero(init).item() == 0, "C019 initialization is not exactly C017")
        zero_loss = loss_per(zero)
    zero_path = HERE / "zero-initialization-audit.json"
    write(zero_path, {"slots": len(z), "exact_equal_C017": True, "first_step_mean_chord_loss": float(zero_loss[:BATCH].mean())})
    optimizer = torch.optim.AdamW(params, lr=c18.LR, weight_decay=c18.WD)
    track = HERE / "optimizer-steps.jsonl"
    freeze_proof = False
    with track.open("x", encoding="utf-8") as log:
        for step in range(STEPS):
            batch = slice(step * BATCH, (step + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            previous = c18.flat_params(layer)
            residual = c18.residual(layer, z[batch], up[batch], basis[batch], enabled[batch])
            losses = (c18.unit(center[batch] + residual) - truth[batch]).square().sum(-1)
            mean_loss = losses.mean()
            if step == 0:
                torch.testing.assert_close(mean_loss, zero_loss[:BATCH].mean(), rtol=0., atol=0.)
            subset = {d: data["domain"][batch] == d for d in DOMAINS}
            yt_groups = {"original_24": data["retained_original_YT"][batch], "new_72": data["replacement_YT"][batch]}
            source_grad = {}
            ys = {}
            for key, mask in {**subset, **yt_groups}.items():
                if mask.any():
                    selected = losses[c18.to_device(mask, device)]
                    # Use the optimizer denominator for each source contribution.
                    grad = torch.autograd.grad(selected.sum() / BATCH, params, retain_graph=True)
                    vector = torch.cat([g.reshape(-1) for g in grad]).detach().cpu().numpy()
                    entry = {"slots": int(mask.sum()), "mean_chord_loss": float(selected.mean().detach()),
                             "grad_l2": float(np.linalg.norm(vector)), "grad_before_global_clip": vector.tolist()}
                else:
                    entry = {"slots": 0, "mean_chord_loss": None, "grad_l2": 0., "grad_before_global_clip": [0.] * 75}
                (source_grad if key in DOMAINS else ys)[key] = entry
            rest = data["rest"][batch]
            if rest.any() and not freeze_proof:
                probe = torch.autograd.grad(losses[c18.to_device(rest, device)].sum(), params, retain_graph=True)
                base.require(all(torch.count_nonzero(g).item() == 0 for g in probe), "Rest-only slots affected C019 gradients")
                freeze_proof = True
            mean_loss.backward()
            raw = c18.flat_params(layer, use_grad=True)
            component = np.sum([r["grad_before_global_clip"] for r in source_grad.values()], axis=0)
            yt_component = np.sum([r["grad_before_global_clip"] for r in ys.values()], axis=0)
            source_gap = float(np.max(np.abs(component - raw)))
            yt_gap = float(np.max(np.abs(yt_component - source_grad["YT_Server"]["grad_before_global_clip"])))
            base.require(max(source_gap, yt_gap) <= 2e-8, "Per-source gradients do not reconstruct the optimized gradient")
            norm = float(torch.nn.utils.clip_grad_norm_(params, c18.CLIP))
            clipped = c18.flat_params(layer, use_grad=True)
            optimizer.step()
            after = c18.flat_params(layer)
            with torch.no_grad():
                moved = c18.residual(layer, z[batch], up[batch], basis[batch], enabled[batch])
                after_loss = (c18.unit(center[batch] + moved) - truth[batch]).square().sum(-1).cpu().numpy()
            for key, mask in {**subset, **yt_groups}.items():
                if mask.any():
                    group = source_grad if key in DOMAINS else ys
                    group[key]["post_step_mean_chord_loss"] = float(after_loss[mask].mean())
            base.require(np.isfinite(np.r_[float(mean_loss.detach()), after_loss, raw, clipped, after]).all(), "Nonfinite C019 step")
            row = {"step": step + 1, "pid": os.getpid(), "loss": float(mean_loss.detach()), "post_step_loss": float(after_loss.mean()),
                   "gradient_l2_before_global_clip": norm, "gradient_before_clip": raw.tolist(), "gradient_after_clip": clipped.tolist(),
                   "weights_before": previous.tolist(), "weights_after": after.tolist(), "source_gradients": source_grad,
                   "YT_gradient_breakdown": ys, "source_gradient_reconstruction_max_abs": source_gap,
                   "YT_gradient_reconstruction_max_abs": yt_gap, "rest_slots": int(rest.sum()),
                   "C015_flat_slot_order_sha256": array_sha(data["global_window"][batch], "<i8"),
                   "native_releases_sha256": array_sha(data["native_release"][batch], "<i8"),
                   "source_record_ordinals_sha256": array_sha(data["source_record_ordinal"][batch], "<i8")}
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            if step + 1 in (1, 40, 80, 120, 160):
                print("C019 fit B", step + 1, row["loss"], norm, flush=True)
    base.require(freeze_proof and len(after) == 75 and np.any(after != 0.), "C019 fit or rest proof incomplete")
    layer.eval().requires_grad_(False)
    weights = HERE / "readout.pt"
    torch.save({"state_dict": {k: p.detach().cpu() for k, p in layer.state_dict().items()},
                "feature_mean": feature_mean.cpu(), "feature_std": feature_std.cpu(),
                "feature": "mean(H[:56])-mean(H[-8:])", "steps": STEPS, "gain": GAIN,
                "protocol_sha256": protocol_hash, "sampling_lock_sha256": schedule_hash}, weights)
    with torch.inference_mode():
        delta = c18.residual(layer, z, up, basis, enabled)
        final = c18.unit(center + delta)
        torch.testing.assert_close(final[~enabled], origin[~enabled], rtol=0., atol=0.)
        before_loss, final_loss = zero_loss.cpu().numpy(), loss_per(final).cpu().numpy()
        magnitude = delta.norm(dim=-1).cpu().numpy()
    def summarize(mask):
        w = data["record_equal_weights"][mask]
        w = w / w.sum()
        return {"slots": int(mask.sum()), "records": len(np.unique(data["source_record_ordinal"][mask])),
                "mean_loss_C017_record_equal": float(w @ before_loss[mask]), "mean_loss_C019_record_equal": float(w @ final_loss[mask]),
                "mean_loss_C017_slot_equal": float(before_loss[mask].mean()), "mean_loss_C019_slot_equal": float(final_loss[mask].mean()),
                "max_post_gain_tangent": float(magnitude[mask].max()), "active_slots": int(np.count_nonzero(magnitude[mask]))}
    return {"pid": os.getpid(), "fit_count": 1, "optimizer_steps": STEPS, "trainable_parameters": len(c18.flat_params(layer)),
            "seconds": perf_counter() - started, "weight_norm": float(np.linalg.norm(c18.flat_params(layer))),
            "static_readout_delta_max_abs": float(magnitude[data["rest"]].max()), "rest_only_gradient_exact_zero": freeze_proof,
            "domains": {d: summarize(data["domain"] == d) for d in DOMAINS},
            "YT_subgroups": {name: summarize(data[mask]) for name, mask in
                             (("original_24", "retained_original_YT"), ("new_72", "replacement_YT"))},
            "file_sha256": {p.name: sha(p) for p in (weights, track, zero_path)}}


@torch.inference_mode()
def paired_readouts(feat, center, rest, layers, averages, scales):
    pooled, up, basis, enabled, center_t = c18.tensors(feat, rest, center, averages[0].device)
    base_unit = c14.normalize(center)
    directions, deltas = {"C017_deg": base_unit}, {}
    for name, layer, average, scale in zip(METHODS[2:], layers, averages, scales, strict=True):
        z = (pooled[:, 1] - average) / scale
        d = c18.residual(layer, z, up, basis, enabled)
        offsets = d.cpu().numpy()
        base.require(np.all(offsets[rest | ~feat.valid] == 0.)
                     and np.max(np.linalg.norm(offsets, axis=-1)) <= GAIN * np.tan(np.deg2rad(base.CAP)) + 1e-8,
                     "Frozen B static gate or tangent cap changed")
        moving = np.any(offsets != 0., axis=1)
        estimate = base_unit.copy()
        estimate[moving] = c18.unit(center_t + d).cpu().numpy()[moving]
        directions[name], deltas[name] = estimate, offsets
    return directions, deltas


def replay_window(q, at, release, windows, acceleration, observed, dt, models, mean, std, k, layers, averages, scales, use=slice(None)):
    last, chosen = at[use], windows[use]
    center, rest = c17.branch_at(q[last], chosen, models, mean, std)
    force, _, obs_last = c17.force_at(q, acceleration, observed, dt, last, c17.window_starts(dt, len(at))[use], rest)
    base.require(np.all(obs_last <= last) and np.all(last <= release[use]), "Unknown native index reached a release")
    center += c17.additions(force, k)[0]
    feat = c18.features(chosen, q[last], models[0], mean, std)
    return (*paired_readouts(feat, center, rest, layers, averages, scales), rest, center)

def audit(cfg, models, mean, std, k, layers, averages, scales):
    notes = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, observed = c14.synthetic(hz)
        _, q, at, release, w, _, acc, _ = c17.front_end(raw, observed, dt, cfg, optical)
        directions, offsets, rest, center = replay_window(q, at, release, w, acc, observed, dt, models, mean, std,
                                                           k, layers, averages, scales)
        j = len(at) // 2
        between = np.flatnonzero(release > at)
        if len(between):
            j = int(between[len(between) // 2])
        end = j + 1
        changed = raw.copy()
        changed[at[j] + 1:, :3] += [14., -8., 6.]
        changed[at[j] + 1:, 3:] += .94
        stop = int(release[j]) + 1
        prefix = dict.fromkeys(METHODS[1:], 0.)
        for x, obs in ((changed, observed), (raw[:stop], observed[:stop])):
            _, q2, a2, r2, w2, _, ac2, _ = c17.front_end(x, obs, dt, cfg, optical)
            np.testing.assert_array_equal(at[:end], a2[:end])
            np.testing.assert_array_equal(release[:end], r2[:end])
            np.testing.assert_array_equal(w[:end], w2[:end])
            out, d, gate, _ = replay_window(q2, a2, r2, w2, ac2, obs, dt, models, mean, std, k, layers, averages, scales, slice(0, end))
            np.testing.assert_array_equal(rest[:end], gate)
            for key in out:
                prefix[key] = max(prefix[key], float(c14.tilt_error(directions[key][:end], out[key]).max()))
            for name in d:
                np.testing.assert_allclose(d[name], offsets[name][:end], rtol=0., atol=3e-8)
        basis_audit = dict.fromkeys(METHODS[2:], 0.)
        q_last = Rotation.from_quat(q[at], scalar_first=True)
        for rot in (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                    Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))):
            changed_w = w.copy()
            changed_w[..., :12] = rot.apply(w[..., :12].reshape(-1, 3)).reshape(len(w), 64, 12)
            feat = c18.features(changed_w, (q_last * rot.inv()).as_quat(scalar_first=True), models[0], mean, std)
            values, _ = paired_readouts(feat, center, rest, layers, averages, scales)
            for key in basis_audit:
                basis_audit[key] = max(basis_audit[key], float(c14.tilt_error(directions[key], values[key]).max()))
        base.require(max(prefix.values()) <= 1e-5 and max(basis_audit.values()) <= 1e-4, "C019 learned readout fails causal or rotation audit")
        notes.append({"native_hz": hz, "windows": len(w), "inclusive_release": int(release[j]), "inclusive_input_last": int(at[j]),
                      "has_separate_release_and_input": bool(release[j] > at[j]), "prefix_max_delta_degrees": prefix,
                      "fixed_world_rotation_max_delta_degrees": basis_audit, "rest_windows": int(rest.sum())})
    return notes


def metrics(rows):
    good = [r for r in rows if r["status"] == "scored"]
    values = {k: fmean(r[k] for r in good) for k in METHODS} if good else {}
    return {"records": len(good), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg": values,
            "C019_delta_deg": {m: values["C019_deg"] - values[m] for m in METHODS[:-1]} if good else {},
            "C019_wins": {m: sum(r["C019_deg"] < r[m] for r in good) for m in METHODS[:-1]}}


def compare(score, old):
    gap = max(abs(score[k] - old[prior]) for k, prior in zip(METHODS[:3], ("full_vqf_deg", "C017_deg", "B_deg"), strict=True))
    base.require(gap <= 1e-6 and np.isfinite(list(score.values())).all(), f"Frozen score controls changed by {gap}")
    return gap


def replay(previous, inventory, campaign, split, cfg, models, mean, std, k, layers, averages, scales):
    c1r, c1p, c3r, c2p = (read(ROOT / p) for p in ("C001/results.json", "C001/protocol.json", "C003/results.json", "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in previous["sequences"]}
    spec = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(spec) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in spec.values()),
                 "Replay would leave the frozen development panel")
    medium_path = DATA / "Sassari/medium_v5.mat"
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(medium_path) == split["source_file_sha256"][str(medium_path)] and sha(array_path) == evidence["array_reference_sha256"],
                 "Archived reference source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(array_path)
    old_poses = {(p["record"], p["pose"]): p for p in previous["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in previous["sequences"]:
            tick = perf_counter()
            key = old["dataset"], old["record"]
            identity = {f: old[f] for f in c17.IDENTITY}
            try:
                data, _, observed, dt, truth, mask, axes, _ = c17.payload(key, spec[key], map3, map1, medium, yt["valid"],
                                                                          split, c1p, array, evidence, cfg)
                q0, q, at, release, w, _, acc, _ = data
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                ix = np.flatnonzero(mask)
                base.require(len(ix) == old["scored_frames"] and array_sha(ix, "<i8") == old["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == old["native_input_last_sha256"]
                             and array_sha(release, "<i8") == old["native_release_sha256"], "Frozen native score or release identity changed")
                infer = perf_counter()
                out, deltas, rest, _ = replay_window(q, at, release, w, acc, observed, dt, models, mean, std, k, layers, averages, scales)
                inference_seconds = perf_counter() - infer
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                score_errors = c17.measure(ix, truth, q0, q, release, {m: np.vstack((UP, value)) for m, value in out.items()})
                values = c17.scores(score_errors)
                error = compare(values, old)
                lookup = np.searchsorted(release, ix, side="right")
                static = np.r_[False, rest][lookup]
                for name in METHODS[2:]:
                    np.testing.assert_array_equal(score_errors[name][static], score_errors["C017_deg"][static])
                details = []
                for label, mask_pose in axes.items():
                    use = mask_pose[ix]
                    prior = old_poses[key[1], label]
                    digest = array_sha(ix[use], "<i8")
                    base.require(int(use.sum()) == prior["scored_frames"] and digest == prior["scoring_indices_sha256"], "Array pose changed")
                    detail = c17.scores(score_errors, use)
                    details.append({"record": key[1], "pose": label, "scored_frames": int(use.sum()), "scoring_indices_sha256": digest,
                                    **detail, "control_max_abs_delta_deg": compare(detail, prior)})
                d = deltas["C019_deg"]
                rows.append({**identity, **values, "status": "scored", "control_max_abs_delta_deg": error,
                             "static_scoring_frames": int(static.sum()), "rest_releases": int(rest.sum()),
                             "C019_delta_sha256": array_sha(d, "<f8"), "C019_readout_post_gain_max_norm": float(np.linalg.norm(d, axis=1).max()),
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "C019_active_releases": int(np.count_nonzero(np.any(d != 0., axis=1))),
                             "paired_prediction_seconds": inference_seconds, "seconds": perf_counter() - tick})
                poses.extend(details)
                print("development", *key, *(values[m] for m in METHODS), flush=True)
            except (ValueError, RuntimeError, KeyError, OSError, AssertionError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - tick})
                print("FAILED", *key, repr(exc), flush=True)
    csv_path, pose_path, group_path = (HERE / name for name in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json"))
    fields = dict.fromkeys(k for row in rows for k in row)
    intrinsic.shice.save_csv(csv_path, [{k: r.get(k, "") for k in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    groups = {field: {f"{domain}/{label}": metrics([r for r in rows if r["dataset"] == domain and r[field] == label])
                      for domain, label in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("group", "reference_QA", "instrument", "scenario")}
    write(group_path, groups)
    return {"pooled_records": metrics(rows), "domains": {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifacts_sha256": {p.name: sha(p) for p in (csv_path, pose_path, group_path) if p.is_file()}}

def main():
    wall = perf_counter()
    launch, initial = (read(HERE / name) for name in ("launch.json", "input-verification.json"))
    base.require(os.getpid() == launch["pid"] == initial["pid"], "C019 lost its registered PID")
    files = initial["files_sha256"]
    base.require(all(sha(ROOT / p) == h for p, h in files.items()), "A frozen C019 preflight input changed")
    campaign, p14, p15, p17, p18, lock, pair, panel = (read(ROOT / f) for f in
          ("protocol.json", "C014/protocol.json", "C015/protocol.json", "C017/protocol.json", "C018/protocol.json",
           "C017/coefficient-lock.json", "C018/pair-lock.json", "review/yt_training_panel.json"))
    lock_hash = files["C017/coefficient-lock.json"]
    base.require(p18["C017_coefficient_lock_sha256"] == pair["C017_coefficient_lock_sha256"] == lock_hash
                 and lock["C015_sampling_sha256"] == files["C015/sampling.npz"] and lock["no_parameter_updates_after_lock"],
                 "C017 fixed coefficients or C015 order changed")
    base.require(p17["development_records"] == p18["development_records"] and len(p18["development_records"]) == 213
                 and p17["scoring_masks_sha256"] == p18["scoring_masks_sha256"] == sha(MASKS)
                 and p17["gate_traces_sha256"] == p18["gate_traces_sha256"] == sha(GATES),
                 "C017/C018 scoring panel differs")
    recipe = p18["optimizer"]
    base.require(recipe == {"name": "AdamW", "learning_rate": c18.LR, "weight_decay": c18.WD, "gradient_clip": c18.CLIP}
                 and p18["gain"] == GAIN and p18["std_floor"] == c18.STD_MIN and p18["steps_per_arm"] == STEPS
                 and p18["batch"] == BATCH and p18["tangent_cap_deg"] == base.CAP, "Frozen C018 B fit recipe changed")
    cfg_path = ROOT / "C003/protocol.json"
    cfg = c3.Settings(**read(cfg_path)["thresholds"])
    inventory_path, split_path = ROOT / "C001/inventory.json", TRANSFER / "external-split.json"
    frozen = p18["inherited_input_sha256"]
    base.require(all(sha(PILOT / p) == h for p, h in frozen.items()), "Original B6/C017 source hash changed")
    inventory = read(inventory_path)
    p = frozen_panel(inventory, panel, p15)
    source_names = p[3]
    base.require(len(source_names) == 96, "C019 exact panel not active")
    models, mean, std, frozen_b6, frozen_b7 = c17.integrity_models(p14, campaign)
    control_data = torch.load(ROOT / "C018/B-readout.pt", map_location=mean.device, weights_only=True)
    base.require(control_data["feature"] == "mean(H[:56])-mean(H[-8:])" and control_data["gain"] == GAIN
                 and control_data["protocol_sha256"] == pair["protocol_sha256"] and control_data["steps"] == STEPS,
                 "Archived B control differs from its lock")
    control = nn.Linear(c18.WIDTH, 3, device=mean.device)
    control.load_state_dict(control_data["state_dict"])
    control.eval().requires_grad_(False)
    protocol_path = HERE / "protocol.json"
    rule_path = HERE / "selection-rule.json"
    rule = {"seed": SEED, "panelfile_sha256": files["review/yt_training_panel.json"],
            "old_24": "For each original YT trial in flattened C015 order retain occurrences with zero-based ordinal divisible by four, preserving their original native releases.",
            "other_sources": "Retain exactly all BROAD, Sassari and RIDI slot positions, record ordinals and release/input-last indices; reuse their C018 cached numerical input.",
            "new_72_assignment": "Sort removed flat slot ordinals by SHA256 digest and original ordinal as tie-break. Sort new trial IDs by separate SHA256 namespace and ID as tie-break. Assign slots round-robin over sorted trials then sort each trial's assigned flat slots numerically.",
            "new_72_release": "For each official train approved trial, retain complete observed original-schedule releases with valid original reference and valid intrinsic frame. Sort native release indices by per-trial SHA256 and numeric release as tie-break. Assign first N to assigned flat slots, cycling in same order only if eligible pool is exhausted.",
            "SHA256_encoding": "UTF-8 C019|20261123|<namespace>|<item>. Namespaces replaced-slot-order, new-trial-order, eligible-release|<trial ID>.",
            "global_window_field": "The archived C015 global window ordinal is a legacy slot identity; for each replacement the actual input address is C019 record ordinal and native release.",
            "follow_up_scores_used_for_selection": False}
    write(rule_path, rule)
    protocol = {**launch, "worker_sha256": sha(Path(__file__)), "frozen_file_sha256": files, "inherited_input_sha256": frozen,
                "C017_coefficient_lock_sha256": lock_hash, "C017_k": lock["k"], "C019_YT_rule_sha256": sha(rule_path),
                "training_recipe_inherited_from_C018_B": recipe, "same_C017_stable_force_recipe": p17["force"],
                "normalized_feature": control_data["feature"], "feature_std_floor": c18.STD_MIN,
                "normalization": "Re-estimate only the new B channel population moments over the 15360 effective training slots.",
                "gain": GAIN, "trainable_parameters": 75, "optimizer_steps": STEPS, "batch_size": BATCH,
                "development_records": p18["development_records"], "scoring_masks_sha256": sha(MASKS)}
    write(protocol_path, protocol)
    protocol_hash = sha(protocol_path)

    def status(name, **details):
        write(HERE / "process.json", {"status": name, "pid": os.getpid(), "started_at": launch["started_at"],
                                      "updated_at": stamp(), "protocol_sha256": protocol_hash, **details})

    status("external_source_extraction", selection_rule_sha256=sha(rule_path))
    old = load_original(files)
    data, evidence = prepare(inventory, p15, panel, old, cfg, models, mean, std, lock["k"])
    del old
    prep_path = HERE / "training-preparation.json"
    write(prep_path, evidence)
    sample_lock = HERE / "sampling-lock.json"
    write(sample_lock, {"pid": os.getpid(), "locked_at": stamp(), "protocol_sha256": protocol_hash,
                        "selection_rule_sha256": sha(rule_path), "training_preparation_sha256": sha(prep_path),
                        "training_inputs_sha256": evidence["training_inputs_sha256"], "native_release_sha256": evidence["native_release_sha256"],
                        "record_ordinal_sha256": evidence["source_record_ordinal_sha256"], "YT_slots": evidence["original_YT_slots"],
                        "regrouped_YT_slots": evidence["YT_slots_replaced"], "retained_YT_slots": evidence["YT_original_slots_retained"],
                        "all_96_panel_trials_present": True})
    sample_hash = sha(sample_lock)
    print("C019 sample locked", sample_hash, "retained", evidence["YT_original_slots_retained"], "replaced", evidence["YT_slots_replaced"],
          "new per trial", [evidence["new_YT_slots_min"], evidence["new_YT_slots_max"]], flush=True)
    base.require(sha(HERE / "training-inputs.npz") == evidence["training_inputs_sha256"], "Locked schedule changed before fitting")
    norm_mean, norm_std = c18.moments(data["pooled"][:, 1])
    norm_path = HERE / "feature-normalization.json"
    write(norm_path, {"mean": norm_mean.tolist(), "std": norm_std.tolist(), "slots": len(data["pooled"]), "std_floor": c18.STD_MIN,
                      "C019_sampling_lock_sha256": sample_hash, "B6_input_normalization_sha256": p18["B6_input_normalization_sha256"]})
    average, scale = (c18.to_device(a, mean.device) for a in (norm_mean, norm_std))
    learned = nn.Linear(c18.WIDTH, 3, device=mean.device)
    nn.init.zeros_(learned.weight)
    nn.init.zeros_(learned.bias)
    status("fitting_one_B_readout", sampling_lock_sha256=sample_hash)
    training = fit(data, average, scale, learned, protocol_hash, sample_hash)
    training_path = HERE / "training.json"
    write(training_path, {**training, "sampling_lock_sha256": sample_hash, "feature_normalization_sha256": sha(norm_path)})
    del data
    weight_path = HERE / "readout.pt"
    check = torch.load(weight_path, weights_only=True, map_location=mean.device)
    for name, val in check["state_dict"].items():
        torch.testing.assert_close(val, learned.state_dict()[name], rtol=0., atol=0.)
    torch.testing.assert_close(average, check["feature_mean"], rtol=0., atol=0.)
    torch.testing.assert_close(scale, check["feature_std"], rtol=0., atol=0.)
    checkpoint = HERE / "readout-lock.json"
    write(checkpoint, {"pid": os.getpid(), "locked_at": stamp(), "protocol_sha256": protocol_hash,
                       "sampling_lock_sha256": sample_hash, "training_sha256": sha(training_path),
                       "frozen_C018_B_weight_sha256": files["C018/B-readout.pt"], "C019_weight_sha256": sha(weight_path),
                       "C017_coefficient_lock_sha256": lock_hash, "optimizer_steps": STEPS})
    frozen_hash = sha(checkpoint)
    layers = (control, learned)
    averages = (control_data["feature_mean"], average)
    scales = (control_data["feature_std"], scale)
    status("learned_readout_audit", readout_lock_sha256=frozen_hash)
    audit_path = HERE / "causal-rotation-audit.json"
    write(audit_path, {"readout_lock_sha256": frozen_hash, "checks": audit(cfg, models, mean, std, lock["k"], layers, averages, scales)})
    status("development_replay", readout_lock_sha256=frozen_hash)
    print("C019 readout locked; auditing 213 development records", flush=True)
    eval_start = perf_counter()
    result = replay(read(ROOT / "C018/results.json"), inventory, campaign, read(split_path), cfg, models, mean, std,
                    lock["k"], layers, averages, scales)
    base.require(sha(checkpoint) == frozen_hash and sha(sample_lock) == sample_hash and sha(weight_path) == read(checkpoint)["C019_weight_sha256"]
                 and all(sha(ROOT / file) == h for file, h in files.items())
                 and all(sha(PILOT / file) == h for file, h in frozen.items()), "A frozen C017/C018/C019 receipt changed in replay")
    result.update({"pid": os.getpid(), "started_at": launch["started_at"], "completed_at": stamp(),
                   "protocol_sha256": protocol_hash, "sampling_lock_sha256": sample_hash, "readout_lock_sha256": frozen_hash,
                   "scoring_masks_sha256": sha(MASKS), "C017_coefficient_lock_sha256": lock_hash,
                   "causal_rotation_audit_sha256": sha(audit_path), "development_replay_seconds": perf_counter() - eval_start,
                   "total_seconds": perf_counter() - wall, "final_holdout_scored": False})
    output = HERE / "results.json"
    write(output, result)
    success = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    status("complete" if success else "incomplete", scored=result["pooled_records"]["records"], results_sha256=sha(output),
           sampling_lock_sha256=sample_hash, readout_lock_sha256=frozen_hash)
    print("C019 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not success:
        raise RuntimeError("C019 development replay incomplete")


if __name__ == "__main__":
    main()
