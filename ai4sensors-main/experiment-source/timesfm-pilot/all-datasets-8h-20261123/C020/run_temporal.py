"""C020 end-to-end historical-state Mamba branch around the frozen C017 center."""
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
sys.path.insert(0, str(ROOT / "C019"))

import run_panel as c19
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn

c18, c17, c14 = c19.c18, c19.c17, c19.c14
c6, c4, c3, c2 = c19.c6, c19.c4, c19.c3, c19.c2
base, intrinsic = c19.base, c19.intrinsic
read, write, sha, stamp, array_sha = c19.read, c19.write, c19.sha, c19.stamp, c19.array_sha
DATA, PILOT, TRANSFER, MASKS, GATES = c19.DATA, c19.PILOT, c19.TRANSFER, c19.MASKS, c19.GATES
UP, STEPS, BATCH, GAIN, DOMAINS = c19.UP, c19.STEPS, c19.BATCH, c19.GAIN, c19.DOMAINS
BACKBONE_LR, HEAD_LR, WD, CLIP = 3e-5, .0003, .01, 2.
METHODS = ("full_vqf_deg", "C017_deg", "C018_B_deg", "C019_deg", "C020_deg")
SCHEDULE_FIELDS = ("global_window", "source_record_ordinal", "native_release", "native_input_last")


def angles(a, b):
    base.require(a.shape == b.shape and a.ndim == 2 and a.shape[1] == 3, "Angle helper expects paired [N,3] arrays")
    return c14.tilt_error(a, b)


def load_schedule(files):
    p19 = ROOT / "C019"
    lock = read(p19 / "sampling-lock.json")
    require = base.require
    require(sha(p19 / "training-inputs.npz") == lock["training_inputs_sha256"] == files["C019/training-inputs.npz"],
            "Exact C019 numerical input archive does not match the lock")
    with np.load(p19 / "training-inputs.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    require(all(data[k].shape == (STEPS * BATCH,) and np.issubdtype(data[k].dtype, np.integer) for k in SCHEDULE_FIELDS)
            and data["pooled"].shape == (15360, 2, 24) and data["canonical_up"].shape == (15360, 3)
            and data["world_basis"].shape == (15360, 3, 3) and data["center"].shape == data["truth"].shape == (15360, 3)
            and data["rest"].dtype == np.dtype(bool) and data["valid"].all()
            and np.all(data["native_input_last"] <= data["native_release"]) and np.all(data["native_input_last"] >= 0),
            "C019 training input shape, gate, or causality changed")
    counts = Counter(str(x) for x in data["domain"])
    require(counts == {"BROAD": 4720, "Sassari": 4675, "YT_Server": 4685, "RIDI": 1280}
            and len(np.unique(data["source_record_ordinal"])) == 139, "Locked source counts changed")
    with (p19 / "optimizer-steps.jsonl").open("r", encoding="utf-8") as stream:
        previous = [json.loads(line) for line in stream]
    require(len(previous) == STEPS, "C019 training receipt is incomplete")
    for i, row in enumerate(previous):
        part = slice(i * BATCH, (i + 1) * BATCH)
        require(row["step"] == i + 1 and np.count_nonzero(data["rest"][part]) == row["rest_slots"], "Batch identity changed")
        for field, key in zip(SCHEDULE_FIELDS[:3], ("C015_flat_slot_order_sha256", "source_record_ordinals_sha256",
                                                   "native_releases_sha256"), strict=True):
            require(array_sha(data[field][part], "<i8") == row[key], f"Batch {i + 1} C019 {field} differs")
    with np.load(ROOT / "C015/sampling.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(f["C015_global_window"], data["global_window"].reshape(STEPS, BATCH))
    require(array_sha(data["native_release"].reshape(STEPS, BATCH), "<i8") == lock["native_release_sha256"]
            and array_sha(data["source_record_ordinal"].reshape(STEPS, BATCH), "<i8") == lock["record_ordinal_sha256"],
            "Actual slot order does not match C019 lock")
    return data


def aligned_histories(inventory, data, cfg, b6, mean, std):
    c15 = c17.c15
    p15 = read(ROOT / "C015/protocol.json")
    p15prep = read(ROOT / "C015/training-preparation.json")
    p19prep = read(ROOT / "C019/training-preparation.json")
    panel = read(ROOT / "review/yt_training_panel.json")
    original, old_yt, new_names, chosen, _, = c19.frozen_panel(inventory, panel, p15)
    records = p19prep["source_records"]
    base.require(records == [r["record"] for r in original] + p19prep["new_YT_trial_order"]
                 and set(records[len(original):]) == set(new_names) and len(records) == 139,
                 "C019 source-record ordinal table changed")
    archived = {r["record"]: r for r in p19prep["trials"]}
    p17_records = read(ROOT / "C017/external-selection.json")["records"]
    ordinal, native, expected_last = (data[k] for k in ("source_record_ordinal", "native_release", "native_input_last"))
    canonical = np.empty((len(native), 64, 13), dtype=np.float32)
    mats, evidence, probes, probe_domains = {}, [], [], set()
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, name in enumerate(records):
            start = perf_counter()
            draw = np.flatnonzero(ordinal == i)
            domain = str(data["domain"][draw[0]])
            np.testing.assert_array_equal(data["source_record_name"][draw], name)
            np.testing.assert_array_equal(data["domain"][draw], domain)
            if i < len(original):
                spec = original[i]
            else:
                inv = chosen[name]
                base.require(yt["train"][name].name == inv["source"], "New trial path is outside approved official train")
                spec = {"domain": "YT_Server", "record": name, "native_frames": inv["native_frames"]}
            if domain == "RIDI":
                raw, observed, dt, score, truth, hashes = c17.trained_phone(spec, p15prep["ridi_eligibility"][i - 63])
            else:
                raw, observed, dt, score, truth, hashes = c4.train_raw(spec, yt["train"], mats)
            expected_source = (archived[name] if domain == "YT_Server" else p17_records[i])["source_sha256"]
            base.require(hashes == expected_source, f"Original training source changed: {name}")
            _, q, at, release, history, complete, _, diag = c17.front_end(raw, observed, dt, cfg)
            native_ids, inverse = np.unique(native[draw], return_inverse=True)
            pos = np.searchsorted(release, native_ids)
            np.testing.assert_array_equal(native_ids, release[pos])
            np.testing.assert_array_equal(at[pos][inverse], expected_last[draw])
            base.require(np.all(complete[pos] & score[native_ids]), "C019 slot no longer has a complete native window and label")
            selected = history[pos]
            qlast = q[at[pos]]
            np.testing.assert_array_equal(c6.rest_flags(selected)[inverse], data["rest"][draw])
            y = c14.rotate(c14.normalize(q[native_ids]), truth[native_ids])[inverse]
            np.testing.assert_array_equal(y, data["truth"][draw])
            frame = intrinsic.build_frame(selected)
            base.require(frame.valid.all(), f"Locked C019 intrinsic frame is degenerate for {name}")
            local = intrinsic.transform_inputs(selected, frame)
            canonical[draw] = local[inverse]
            f = c18.features(selected, qlast, b6, mean, std, frame=frame)
            np.testing.assert_array_equal(f.up[inverse], data["canonical_up"][draw])
            np.testing.assert_array_equal(f.world_basis[inverse], data["world_basis"][draw])
            difference = np.max(np.abs(f.pooled[inverse] - data["pooled"][draw]), axis=(0, 2))
            base.require(np.all(difference <= 2e-5), f"B6 initial hidden representation diverges for {name}: {difference}")
            if domain == "YT_Server" and name in old_yt:
                base.require(array_sha(native[draw], "<i8") == archived[name]["sampled_release_sha256"], "Retained YT order changed")
            if domain not in probe_domains:
                active = np.flatnonzero(~c6.rest_flags(selected))
                if len(active) >= 3:
                    for j in active[np.linspace(0, len(active) - 1, 3, dtype=int)]:
                        a = draw[np.flatnonzero(inverse == j)[0]]
                        probes.append({"domain": domain, "record": name, "native_release": int(native_ids[j]),
                                       "raw": selected[j], "quat": qlast[j], "center": data["center"][a]})
                    probe_domains.add(domain)
            evidence.append({"domain": domain, "record": name, "ordinal": i, "slots": len(draw), "unique_releases": len(native_ids),
                             "C003_active_native_frames": diag["active_native_frames"], "raw_history_sha256": array_sha(selected, "<f4"),
                             "canonical_history_sha256": array_sha(local, "<f4"), "native_release_sha256": array_sha(native[draw], "<i8"),
                             "H_last_max_abs_difference": float(difference[0]), "H_history_diff_max_abs_difference": float(difference[1]),
                             "source_sha256": hashes, "seconds": perf_counter() - start})
            print("C020 reconstructed", domain, name, len(draw), len(native_ids), flush=True)
    base.require(len(probes) == 12 and len(evidence) == 139 and np.isfinite(canonical).all(), "C020 history or intervention coverage incomplete")
    history_path, probes_path = HERE / "training-canonical-history.npy", HERE / "history-probes.npz"
    np.save(history_path, canonical, allow_pickle=False)
    np.savez_compressed(probes_path, raw=np.stack([r["raw"] for r in probes]), quat=np.stack([r["quat"] for r in probes]),
                        center=np.stack([r["center"] for r in probes]), native_release=np.asarray([r["native_release"] for r in probes]),
                        domain=np.asarray([r["domain"] for r in probes]), record=np.asarray([r["record"] for r in probes]))
    return canonical, {"history_shape": list(canonical.shape), "history_sha256": sha(history_path),
                       "probe_inputs_sha256": sha(probes_path), "records": evidence,
                       "H_last_max_abs_difference": max(r["H_last_max_abs_difference"] for r in evidence),
                       "H_history_diff_max_abs_difference": max(r["H_history_diff_max_abs_difference"] for r in evidence)}


def pooled_history(model, x):
    h = c18.hidden(model, x)
    return h[:, :56].mean(dim=1) - h[:, -8:].mean(dim=1)


def predict(model, x, up, basis, enabled, center, feature_mean, feature_std):
    f = (pooled_history(model, x) - feature_mean) / feature_std
    offset = c18.residual(model.output, f, up, basis, enabled)
    return c18.unit(center + offset), offset


def preflight(model, data, canonical, mean, std, feature_mean, feature_std):
    require = base.require
    signature = list(inspect.signature(c17.force_at).parameters)
    require(signature == ["q", "acceleration", "observed", "dt", "at", "starts", "rest"], "force_at argument order changed")
    other_signatures = {"tilt_error": list(inspect.signature(c14.tilt_error).parameters),
                        "paired_readouts": list(inspect.signature(c19.paired_readouts).parameters)}
    require(other_signatures["tilt_error"] == ["a", "b"] and other_signatures["paired_readouts"] ==
            ["feat", "center", "rest", "layers", "averages", "scales"], "Archived replay helper signature changed")
    np.testing.assert_array_equal(angles(np.broadcast_to(UP, (1, 3)), np.broadcast_to(UP, (1, 3))), [0.])
    np.testing.assert_array_equal(angles(np.broadcast_to(UP, (2, 3)), np.broadcast_to(UP, (2, 3))), [0., 0.])
    device = mean.device
    with torch.no_grad():
        x = base.normalized(c18.to_device(canonical[:BATCH], device), mean, std)
        up, basis, enabled, center = (c18.to_device(data[k][:BATCH], device) for k in ("canonical_up", "world_basis", "valid", "center"))
        enabled &= ~c18.to_device(data["rest"][:BATCH], device)
        pred, offset = predict(model, x, up, basis, enabled, center, feature_mean, feature_std)
        require(pred.shape == offset.shape == (96, 3) and pooled_history(model, x).shape == (96, 24), "Prediction helper shape changed")
        torch.testing.assert_close(pred, c18.unit(center), rtol=0., atol=0.)
        require(torch.count_nonzero(offset).item() == 0, "Zero readout fails initial C017 replay")
        first = float((pred - c18.to_device(data["truth"][:BATCH], device)).square().sum(-1).mean())
    archive = read(ROOT / "C019/zero-initialization-audit.json")
    require(first == archive["first_step_mean_chord_loss"], "Initial C019/C020 slot loss differs after matching the zero readout")
    return {"native_order_batches_checked": 160, "n_slots": len(canonical), "force_at": signature, **other_signatures,
            "singleton_and_batched_angles_checked": True, "prediction_batch_shape": list(pred.shape),
            "readout_feature_shape": [96, 24], "first_step_exact_equal_C017": True, "first_step_loss": first}

def parameter_layout(model):
    layout, offset = {}, 0
    for name, p in model.named_parameters():
        layout[name] = {"shape": list(p.shape), "start": offset, "stop": offset + p.numel(),
                        "lr": HEAD_LR if name.startswith("output.") else BACKBONE_LR}
        offset += p.numel()
    xproj = layout["sequence.x_proj.weight"]
    inner = xproj["shape"][1]
    dt_rank, ssm_size = model.sequence.time_step_rank, model.sequence.ssm_state_size
    first = xproj["start"]
    components = {}
    for label, width in (("dt", dt_rank), ("B", ssm_size), ("C", ssm_size)):
        stop = first + width * inner
        components[f"sequence.x_proj.weight[{label}]"] = {"start": first, "stop": stop, "shape": [width, inner], "lr": BACKBONE_LR}
        first = stop
    base.require(first == xproj["stop"] and offset == 5595, "Unexpected copied Mamba parameter layout")
    return layout, components


def vector(named, transform):
    return torch.cat([transform(name, p).detach().reshape(-1) for name, p in named]).cpu().numpy()


def attribute(traces, layout, components):
    out = {}
    for name, meta in {**layout, **components}.items():
        part = slice(meta["start"], meta["stop"])
        grad = traces["grad_raw"][:, part]
        decay = traces["decay_delta"][:, part]
        adaptive = traces["adam_gradient_delta"][:, part]
        learned = traces["final_weights"][part] - traces["initial_weights"][part]
        raw = np.linalg.norm(grad.astype(np.float64), axis=1)
        out[name] = {**meta, "initial_to_final_l2": float(np.linalg.norm(learned)), "initial_to_final_nonzero": int(np.count_nonzero(learned)),
                     "steps_with_nonzero_gradient": int(np.count_nonzero(raw)),
                     "first_nonzero_gradient_step": int(np.flatnonzero(raw)[0] + 1) if np.any(raw) else None,
                     "max_raw_gradient_l2": float(raw.max()), "max_nonzero_gradient_values_per_step": int(np.count_nonzero(grad, axis=1).max()),
                     "sum_of_step_decay_norms": float(np.linalg.norm(decay, axis=1).sum()),
                     "sum_of_step_adaptive_update_norms": float(np.linalg.norm(adaptive, axis=1).sum()),
                     "net_decay_vector_l2": float(np.linalg.norm(decay.sum(axis=0))),
                     "net_adaptive_vector_l2": float(np.linalg.norm(adaptive.sum(axis=0))),
                     "net_decay_plus_adaptive_reconstruction_max_abs": float(np.max(np.abs(decay.sum(axis=0) + adaptive.sum(axis=0) - learned)))}
        base.require(out[name]["net_decay_plus_adaptive_reconstruction_max_abs"] <= 1e-12, "Parameter delta did not decompose")
    for name, row in out.items():
        if name.startswith("sequence."):
            base.require(row["steps_with_nonzero_gradient"] > 0 and row["sum_of_step_adaptive_update_norms"] > 0.,
                         f"SSM tensor has no measured adaptive update: {name}")
    return out


def training_sources(model, control19, data, x, center, up, basis, enabled, truth, feature_mean, feature_std):
    with torch.inference_mode():
        reference = c18.unit(center)
        original_z = (c18.to_device(data["pooled"][:, 1], x.device) - feature_mean) / feature_std
        d19 = c18.residual(control19, original_z, up, basis, enabled)
        control = c18.unit(center + d19)
        result, offsets = [], []
        for start in range(0, len(x), 256):
            use = slice(start, start + 256)
            now, d = predict(model, x[use], up[use], basis[use], enabled[use], center[use], feature_mean, feature_std)
            result.append(now)
            offsets.append(d)
        fitted, delta = torch.cat(result), torch.cat(offsets)
        torch.testing.assert_close(fitted[~enabled], reference[~enabled], rtol=0., atol=0.)
        direction = dict(zip(("C017", "C019", "C020"), (reference, control, fitted), strict=True))
        losses = {name: (pred - truth).square().sum(-1).cpu().numpy() for name, pred in direction.items()}
    def summary(mask):
        weight = data["record_equal_weights"][mask]
        weight = weight / weight.sum()
        return {"slots": int(mask.sum()), "records": len(np.unique(data["source_record_ordinal"][mask])),
                "slot_equal_chord_loss": {key: float(value[mask].mean()) for key, value in losses.items()},
                "record_equal_chord_loss": {key: float(weight @ value[mask]) for key, value in losses.items()}}
    return {"domains": {d: summary(data["domain"] == d) for d in DOMAINS},
            "YT_subgroups": {label: summary(data[key]) for label, key in
                             (("original_24", "retained_original_YT"), ("new_72", "replacement_YT"))},
            "static_max_abs_new_residual": float(delta[~enabled].abs().max()),
            "overall_max_post_gain_tangent_norm": float(delta.norm(dim=1).max())}


def train(model, inherited, control19, data, canonical, mean, std, feature_mean, feature_std, first_loss, protocol_hash):
    torch.manual_seed(1729)
    started = stamp()
    tick = perf_counter()
    device = mean.device
    before_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
    layout, components = parameter_layout(model)
    named = tuple(model.named_parameters())
    source_named = tuple(inherited.named_parameters())
    base.require(all(not p.requires_grad and p.grad is None for _, p in source_named)
                 and not any(p.data_ptr() == old.data_ptr() for (_, p), (_, old) in zip(named, source_named, strict=True)),
                 "New branch shares storage or gradients with B6")
    backbone = [p for name, p in named if not name.startswith("output.")]
    head = list(model.output.parameters())
    base.require(sum(p.numel() for p in backbone) == 5520 and sum(p.numel() for p in head) == 75,
                 "Expected exactly 5520 backbone and 75 readout parameters")
    x = base.normalized(c18.to_device(canonical, device), mean, std)
    up, basis, center, truth = (c18.to_device(data[k], device) for k in ("canonical_up", "world_basis", "center", "truth"))
    enable = c18.to_device(data["valid"] & ~data["rest"], device)
    model.train().requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": BACKBONE_LR}, {"params": head, "lr": HEAD_LR}],
                                  lr=HEAD_LR, weight_decay=WD, foreach=False, fused=False)
    first_weights = vector(named, lambda n, p: p).astype(np.float64)
    grids = {key: [] for key in ("grad_raw", "grad_clipped", "decay_delta", "adam_gradient_delta")}
    trace_path = HERE / "optimizer-steps.jsonl"
    first_probe = False
    torch.cuda.reset_peak_memory_stats()
    with trace_path.open("x", encoding="utf-8") as stream:
        for i in range(STEPS):
            sl = slice(i * BATCH, (i + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            prediction, residual = predict(model, x[sl], up[sl], basis[sl], enable[sl], center[sl], feature_mean, feature_std)
            loss = (prediction - truth[sl]).square().sum(-1)
            total = loss.mean()
            if not i:
                base.require(float(total.detach()) == first_loss and torch.count_nonzero(residual).item() == 0,
                             "First real zero initialization drifted")
            if not first_probe and np.any(data["rest"][sl]):
                rest = c18.to_device(data["rest"][sl], device)
                rest_grad = torch.autograd.grad(loss[rest].sum(), tuple(p for _, p in named), retain_graph=True)
                base.require(all(torch.count_nonzero(g).item() == 0 for g in rest_grad), "Static loss supplies nonzero gradient")
                first_probe = True
            total.backward()
            base.require(all(p.grad is not None for _, p in named), "A trainable input/norm/mixer/readout tensor is disconnected")
            raw = vector(named, lambda n, p: p.grad)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            clipped = vector(named, lambda n, p: p.grad)
            prev = vector(named, lambda n, p: p).astype(np.float64)
            # Apply exactly the first (decoupled decay) operation of the configured AdamW to detached copies.
            pure_decay = vector(named, lambda n, p: p.detach() * (1. - layout[n]["lr"] * WD)).astype(np.float64)
            optimizer.step()
            after = vector(named, lambda n, p: p).astype(np.float64)
            decay, adaptive = pure_decay - prev, after - pure_decay
            if not i:
                pivot = layout["output.weight"]["start"]
                base.require(np.all(raw[:pivot] == 0.) and np.all(adaptive[:pivot] == 0.) and np.any(raw[pivot:] != 0.),
                             "Zero head must silence the first backbone gradient and adaptive update")
            base.require(np.isfinite(np.r_[float(total.detach()), norm, raw, after]).all(), f"C020 nonfinite at step {i + 1}")
            for key, val in zip(grids, (raw, clipped, decay, adaptive), strict=True):
                grids[key].append(val)
            value = loss.detach().cpu().numpy()
            per_source = c18.source_loss(value, data["domain"][sl])
            row = {"step": i + 1, "pid": os.getpid(), "batch_mean_chord_loss": float(total.detach()),
                   "global_gradient_norm_before_clip": norm, "source_losses": per_source, "rest_slots": int(data["rest"][sl].sum()),
                   "decay_delta_sha256": array_sha(decay, "<f8"), "adaptive_delta_sha256": array_sha(adaptive, "<f8"),
                   "weights_after_sha256": array_sha(after, "<f8"), "global_window_sha256": array_sha(data["global_window"][sl], "<i8"),
                   "source_record_ordinal_sha256": array_sha(data["source_record_ordinal"][sl], "<i8"),
                   "native_release_sha256": array_sha(data["native_release"][sl], "<i8")}
            print(json.dumps(row, allow_nan=False), file=stream, flush=True)
            if i + 1 in (1, 2, 40, 80, 120, 160):
                print("C020 fit", i + 1, row["batch_mean_chord_loss"], norm, flush=True)
    base.require(first_probe and len(grids["grad_raw"]) == STEPS, "Fit count or static gradient coverage differs")
    elapsed = perf_counter() - tick
    traces = {key: np.stack(val) for key, val in grids.items()}
    traces.update({"initial_weights": first_weights, "final_weights": after})
    proof = attribute(traces, layout, components)
    model.eval().requires_grad_(False)
    tensor_path, attr_path, weights = (HERE / n for n in ("optimizer-tensors.npz", "parameter-attribution.json", "mamba-end-to-end.pt"))
    np.savez_compressed(tensor_path, **traces)
    write(attr_path, {"layout_order": list(layout), "parameter_attribution": proof,
                      "meaning": "Each step float32 weight immediately after the decoupled AdamW multiplication minus prior weight is decay. Remaining actual update includes first/second moments and clipping. Net norms use signed summed vectors.",
                      "tensor_trace_sha256": sha(tensor_path), "first_step_trunk_gradient_and_adaptive_update_exact_zero": True})
    start_hash = c4.tensor_hash(before_state)
    architecture = {"hidden_size": 24, "input_size": 13, "state_size": model.sequence.ssm_state_size,
                    "conv_kernel": model.sequence.conv_kernel_size, "intermediate_size": model.sequence.intermediate_size,
                    "time_step_rank": model.sequence.time_step_rank, "layers": 1, "window": 64,
                    "frozen_feature_normalization": "C019 history difference mean and population std",
                    "pooling": "mean(H[:56])-mean(H[-8:])", "output_dimension": 3, "gain": GAIN, "tangent_cap_deg": base.CAP}
    torch.save({"model": model.kind, "state_dict": {k: p.detach().cpu() for k, p in model.state_dict().items()},
                "B6_input_mean": mean.cpu(), "B6_input_std": std.cpu(), "C019_feature_mean": feature_mean.cpu(),
                "C019_feature_std": feature_std.cpu(), "architecture": architecture, "optimizer_steps": STEPS,
                "protocol_sha256": protocol_hash, "initial_state_tensor_sha256": start_hash}, weights)
    outputs = training_sources(model, control19, data, x, center, up, basis, enable, truth, feature_mean, feature_std)
    return {"pid": os.getpid(), "started_at": started, "completed_at": stamp(), "fit_count": 1, "optimizer_steps": STEPS,
            "trainable_parameters": 5595, "fit_seconds": elapsed, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "first_step_backbone_only_decay": True, "rest_only_gradient_exact_zero": first_probe, "source_scores": outputs,
            "initial_state_tensor_sha256": start_hash, "architecture": architecture,
            "artifacts_sha256": {p.name: sha(p) for p in (tensor_path, attr_path, weights, trace_path)}}

@torch.inference_mode()
def learned_direction(model, feat, center, rest, feature_mean, feature_std):
    pooled, up, world, enabled, center_t = c18.tensors(feat, rest, center, feature_mean.device)
    z = (pooled[:, 1] - feature_mean) / feature_std
    offset = c18.residual(model.output, z, up, world, enabled)
    delta = offset.cpu().numpy()
    base.require(np.all(delta[rest | ~feat.valid] == 0.)
                 and np.max(np.linalg.norm(delta, axis=-1)) <= GAIN * np.tan(np.deg2rad(base.CAP)) + 1e-8,
                 "New branch violates the gate or cap")
    moving = np.any(delta != 0., axis=1)
    predicted = c14.normalize(center)
    predicted[moving] = c18.unit(center_t + offset).cpu().numpy()[moving]
    return predicted, delta


def readout_bundle(w, q_last, center, rest, bank, frame=None):
    models, mean, std, controls, averages, scales, learned = bank
    base_features = c18.features(w, q_last, models[0], mean, std, frame=frame)
    directions, offsets = c19.paired_readouts(base_features, center, rest, controls, averages, scales)
    new_features = c18.features(w, q_last, learned, mean, std, frame=frame)
    directions["C020_deg"], offsets["C020_deg"] = learned_direction(learned, new_features, center, rest, averages[1], scales[1])
    return directions, offsets


def snapshot(q, at, release, w, acc, observed, dt, k, bank, use=slice(None)):
    models, mean, std, *_ = bank
    last, windows = at[use], w[use]
    center, rest = c17.branch_at(q[last], windows, models, mean, std)
    f, support, input_last = c17.force_at(q, acc, observed, dt, last, c17.window_starts(dt, len(at))[use], rest)
    base.require(np.all(input_last <= last) and np.all(last <= release[use]), "C020 chose an unreleased input")
    center += c17.additions(f, k)[0]
    return (*readout_bundle(windows, q[last], center, rest, bank), rest, center, support)


def interventions(bank):
    with np.load(HERE / "history-probes.npz", allow_pickle=False) as file:
        data = {k: file[k] for k in file.files}
    raw, q, center = (data[k] for k in ("raw", "quat", "center"))
    base.require(raw.shape == (12, 64, 13) and q.shape == (12, 4) and center.shape == (12, 3), "Intervention array dimensions changed")
    rest = c6.rest_flags(raw)
    base.require(not rest.any(), "Intervention must use active releases")
    probed = raw.copy()
    probed[:, :32, :3] += np.array([.35, -.25, .15], dtype=np.float32)
    probed[:, :32, 3:6] += np.array([.035, -.025, .015], dtype=np.float32)
    np.testing.assert_array_equal(probed[:, 32:], raw[:, 32:])
    original_frame = intrinsic.build_frame(raw)
    base_result, base_offset = readout_bundle(raw, q, center, rest, bank, original_frame)
    results = {}
    for name, frame in (("fixed_original_basis", original_frame), ("recomputed_basis", None)):
        changed, offset = readout_bundle(probed, q, center, rest, bank, frame)
        keys = METHODS[3:]
        measured = {k: angles(base_result[k], changed[k]) for k in keys}
        residuals = {k: np.linalg.norm(offset[k] - base_offset[k], axis=1) for k in keys}
        results[name] = [{"record": str(data["record"][i]), "domain": str(data["domain"][i]), "native_release": int(data["native_release"][i]),
                          "old_window_sha256": array_sha(raw[i], "<f4"), "edited_window_sha256": array_sha(probed[i], "<f4"),
                          "output_angle_change_deg": {k: float(measured[k][i]) for k in keys},
                          "post_gain_tangent_change_l2": {k: float(residuals[k][i]) for k in keys}} for i in range(len(raw))]
    return {"probes": 12, "older_steps_modified": 32, "latest_steps_exact_unchanged": 32,
            "C017_center_and_rest_held_fixed": True, "accel_perturbation_m_s2": [.35, -.25, .15],
            "gyro_perturbation_rad_s": [.035, -.025, .015], "variants": results}


def audits(cfg, k, bank):
    models, mean, std, controls, averages, scales, learned = bank
    relations = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, observed = c14.synthetic(hz)
        _, q, at, release, w, _, acc, _ = c17.front_end(raw, observed, dt, cfg, optical)
        original, offsets, rest, center, _ = snapshot(q, at, release, w, acc, observed, dt, k, bank)
        prefix = dict.fromkeys(METHODS[1:], 0.)
        delayed = np.flatnonzero(release > at)
        for j in sorted({len(at) // 2, len(at) - 2, int(delayed[len(delayed) // 2]) if len(delayed) else len(at) // 2}):
            altered = raw.copy()
            altered[at[j] + 1:, :3] += [14., -8., 6.]
            altered[at[j] + 1:, 3:] += .94
            altered_obs = observed.copy()
            altered_obs[at[j] + 2::2] = False
            limit = int(release[j]) + 1
            end = j + 1
            for signal, obs in ((altered, altered_obs), (raw[:limit], observed[:limit])):
                _, q2, at2, release2, w2, _, acc2, _ = c17.front_end(signal, obs, dt, cfg, optical)
                np.testing.assert_array_equal(release[:end], release2[:end])
                np.testing.assert_array_equal(at[:end], at2[:end])
                np.testing.assert_array_equal(w[:end], w2[:end])
                outs, delta, rest2, _, _ = snapshot(q2, at2, release2, w2, acc2, obs, dt, k, bank, slice(0, end))
                np.testing.assert_array_equal(rest[:end], rest2)
                for name, v in outs.items():
                    prefix[name] = max(prefix[name], float(angles(original[name][:end], v).max()))
                for name, v in delta.items():
                    np.testing.assert_allclose(v, offsets[name][:end], rtol=0., atol=3e-8)
        frame = intrinsic.build_frame(w)
        probe = np.flatnonzero(frame.valid & ~rest)[::7]
        frame_probe = type(frame)(frame.basis[probe], frame.source[probe], frame.anchor[probe])
        with torch.inference_mode():
            x = base.normalized(c18.to_device(intrinsic.transform_inputs(w[probe], frame_probe), mean.device), mean, std)
            latest = 39
            h = c18.hidden(learned, x)[:, :latest]
            altered = x.clone()
            altered[:, latest:, :6] *= -3.
            hidden_error = max(float((c18.hidden(learned, x[:, :latest]) - h).abs().max()),
                               float((c18.hidden(learned, altered)[:, :latest] - h).abs().max()))
        base.require(max(prefix.values()) <= 1e-5 and hidden_error <= 2e-5, "Learned branch failed prefix causality")
        rot_metrics = []
        q_last = Rotation.from_quat(q[at], scalar_first=True)
        r_release = Rotation.from_quat(q[release], scalar_first=True)
        for label, rot in zip(("90x", "90y", "90z", "63_1_2_-1"),
                (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))), strict=True):
            transformed = w.copy()
            transformed[..., :12] = rot.apply(w[..., :12].reshape(-1, 3)).reshape(len(w), 64, 12)
            rotated, _ = readout_bundle(transformed, (q_last * rot.inv()).as_quat(scalar_first=True), center, rest, bank)
            gauge = {name: float(angles(original[name], v).max()) for name, v in rotated.items()}
            base.require(max(gauge.values()) <= 1e-4, "Learned branch failed fixed-world installation rotation")
            changed = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, q3, at3, rel3, w3, _, acc3, _ = c17.front_end(changed, observed, dt, cfg, optical)
            np.testing.assert_array_equal(at, at3)
            np.testing.assert_array_equal(release, rel3)
            native, _, rest3, _, _ = snapshot(q3, at3, rel3, w3, acc3, observed, dt, k, bank)
            inverse = Rotation.from_quat(q3[rel3], scalar_first=True).inv()
            physical = {name: float(angles(r_release.inv().apply(original[name]), rot.inv().apply(inverse.apply(v))).max())
                        for name, v in native.items()}
            rot_metrics.append({"rotation": label, "fixed_world_max_deg": gauge, "raw_replay_body_max_deg": physical,
                                "raw_replay_rest_changes": int(np.count_nonzero(rest3 != rest))})
        relations.append({"native_hz": hz, "windows": len(w), "release_later_than_input_windows": len(delayed),
                          "prefix_max_deg": prefix, "hidden_prefix_max_abs": hidden_error, "rotations": rot_metrics})
    return relations

def metrics(rows):
    good = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in good) for k in METHODS} if good else {}
    return {"records": len(good), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg": macro,
            "C020_delta_deg": {m: macro["C020_deg"] - macro[m] for m in METHODS[:-1]} if good else {},
            "C020_wins": {m: sum(r["C020_deg"] < r[m] for r in good) for m in METHODS[:-1]}}


def compare_archive(values, old):
    difference = max(abs(values[k] - old[k]) for k in METHODS[:-1])
    base.require(np.isfinite(list(values.values())).all() and difference <= 1e-6, f"Frozen C019 controls changed by {difference}")
    return difference


def replay(prior, inventory, campaign, split, cfg, k, bank):
    c1r, c1p, c3r, c2p = (read(ROOT / p) for p in ("C001/results.json", "C001/protocol.json", "C003/results.json", "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obs["sequences"]} for obs in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    spec = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == len(spec) == 213 and set(spec) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in spec.values()),
                 "C020 replay is outside the registered 213 development records")
    medium_path, array_path = DATA / "Sassari/medium_v5.mat", DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(medium_path) == split["source_file_sha256"][str(medium_path)] and sha(array_path) == evidence["array_reference_sha256"],
                 "C019 archived reference source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(array_path)
    previous_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            start = perf_counter()
            key = old["dataset"], old["record"]
            identity = {f: old[f] for f in c17.IDENTITY}
            try:
                data, _, observed, dt, truth, mask, axes, _ = c17.payload(key, spec[key], map3, map1, medium, yt["valid"], split,
                                                                          c1p, array, evidence, cfg)
                q0, q, at, release, w, _, acc, _ = data
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                ix = np.flatnonzero(mask)
                base.require(len(ix) == old["scored_frames"] and array_sha(ix, "<i8") == old["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == old["native_input_last_sha256"]
                             and array_sha(release, "<i8") == old["native_release_sha256"], "Native scoring or release timeline differs")
                inference = perf_counter()
                dirs, offsets, rest, _, _ = snapshot(q, at, release, w, acc, observed, dt, k, bank)
                elapsed = perf_counter() - inference
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                error = c17.measure(ix, truth, q0, q, release, {m: np.vstack((UP, d)) for m, d in dirs.items()})
                values = c17.scores(error)
                gap = compare_archive(values, old)
                static = np.r_[False, rest][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(error["C020_deg"][static], error["C017_deg"][static])
                detail = []
                for pose, take in axes.items():
                    used = take[ix]
                    previous = previous_poses[key[1], pose]
                    score_hash = array_sha(ix[used], "<i8")
                    base.require(score_hash == previous["scoring_indices_sha256"] and int(used.sum()) == previous["scored_frames"],
                                 "Native array pose mask changed")
                    val = c17.scores(error, used)
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(used.sum()), "scoring_indices_sha256": score_hash,
                                   **val, "control_max_abs_delta_deg": compare_archive(val, previous)})
                added = offsets["C020_deg"]
                rows.append({**identity, **values, "status": "scored", "control_max_abs_delta_deg": gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "C020_world_delta_sha256": array_sha(added, "<f8"),
                             "C020_active_releases": int(np.count_nonzero(np.any(added != 0., axis=1))),
                             "C020_max_post_gain_tangent": float(np.linalg.norm(added, axis=1).max()),
                             "static_scoring_frames": int(static.sum()), "rest_releases": int(rest.sum()),
                             "paired_prediction_seconds": elapsed, "seconds": perf_counter() - start})
                poses.extend(detail)
                print("development", *key, *(values[m] for m in METHODS), flush=True)
            except (ValueError, RuntimeError, KeyError, OSError, AssertionError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - start})
                print("FAILED", *key, repr(exc), flush=True)
    seq, pose_csv, group = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json"))
    fields = dict.fromkeys(k for row in rows for k in row)
    intrinsic.shice.save_csv(seq, [{k: row.get(k, "") for k in fields} for row in rows])
    if poses:
        intrinsic.shice.save_csv(pose_csv, poses)
    write(group, {f: {f"{d}/{cat}": metrics([r for r in rows if r["dataset"] == d and r[f] == cat])
                       for d, cat in sorted({(r["dataset"], r[f]) for r in rows})}
                  for f in ("group", "reference_QA", "instrument", "scenario")})
    return {"pooled_records": metrics(rows), "domains": {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifacts_sha256": {p.name: sha(p) for p in (seq, pose_csv, group) if p.is_file()}}


def controls_and_bank(p14, campaign, files):
    models, mean, std, _, _ = c17.integrity_models(p14, campaign)
    bank, averages, scales = [], [], []
    for file in ("C018/B-readout.pt", "C019/readout.pt"):
        base.require(sha(ROOT / file) == files[file], f"Frozen control changed: {file}")
        ckpt = torch.load(ROOT / file, map_location=mean.device, weights_only=True)
        base.require(ckpt["feature"] == "mean(H[:56])-mean(H[-8:])" and ckpt["gain"] == GAIN
                     and ckpt["steps"] == STEPS, "Frozen history readout format changed")
        layer = nn.Linear(c18.WIDTH, 3, device=mean.device)
        layer.load_state_dict(ckpt["state_dict"])
        bank.append(layer.eval().requires_grad_(False))
        averages.append(ckpt["feature_mean"])
        scales.append(ckpt["feature_std"])
    return models, mean, std, bank, averages, scales


def main():
    start = perf_counter()
    launch, initial = (read(HERE / x) for x in ("launch.json", "input-verification.json"))
    files = initial["files_sha256"]
    base.require(os.getpid() == launch["pid"] == initial["pid"], "C020 lost its registered PID")
    base.require(all(sha(ROOT / name) == h for name, h in files.items()), "Inherited C020 input changed")
    p14, p17, p18, p19, p8, campaign, k_lock = (read(ROOT / x) for x in (
        "C014/protocol.json", "C017/protocol.json", "C018/protocol.json", "C019/protocol.json", "C008/protocol.json",
        "protocol.json", "C017/coefficient-lock.json"))
    optimizer = p8["optimizer"]
    base.require(optimizer["kind"] == "AdamW" and optimizer["steps"] == STEPS and optimizer["batch_size"] == BATCH
                 and optimizer["seed"] == 1729 and np.isclose(optimizer["backbone_lr"], BACKBONE_LR, rtol=1e-15, atol=0.)
                 and optimizer["output_lr"] == HEAD_LR and optimizer["weight_decay"] == WD and optimizer["grad_clip"] == CLIP,
                 "C008 optimizer protocol differs")
    base.require(p17["development_records"] == p18["development_records"] == p19["development_records"]
                 and len(p19["development_records"]) == 213
                 and p19["scoring_masks_sha256"] == p17["scoring_masks_sha256"] == sha(MASKS)
                 and p17["gate_traces_sha256"] == sha(GATES), "Frozen C020 development panel differs")
    frozen = p18["inherited_input_sha256"]
    base.require(all(sha(PILOT / name) == h for name, h in frozen.items()), "C017 B6/B7 dependency changed")
    models, mean, std, controls, averages, scales = controls_and_bank(p14, campaign, files)
    learned = deepcopy(models[0])
    nn.init.zeros_(learned.output.weight)
    nn.init.zeros_(learned.output.bias)
    feature_mean, feature_std = averages[1], scales[1]
    k = k_lock["k"]
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    inventory = read(ROOT / "C001/inventory.json")
    protocol_path = HERE / "protocol.json"
    protocol = {**launch, "worker_sha256": sha(Path(__file__)), "inherited_file_sha256": files, "C017_dependencies_sha256": frozen,
                "C017_coefficient_lock_sha256": files["C017/coefficient-lock.json"], "C017_k": k,
                "C019_sampling_lock_sha256": files["C019/sampling-lock.json"],
                "learned": "Independent copy of B6 input/norm/MambaMixer plus zero 24→3 head. All new branch parameters trained.",
                "B6_input_normalization_sha256": array_sha(torch.stack((mean, std)).cpu().numpy(), "<f4"),
                "C019_B_feature_normalization_sha256": array_sha(torch.stack((feature_mean, feature_std)).cpu().numpy(), "<f4"),
                "optimizer": {"kind": "AdamW", "seed": 1729, "steps": STEPS, "batch": BATCH, "backbone_lr": BACKBONE_LR,
                              "head_lr": HEAD_LR, "weight_decay": WD, "global_clip": CLIP, "foreach": False, "fused": False},
                "training_target": "Native release reference in same world frame as frozen C017, optimizing the actual final unit vector chord loss.",
                "readout_feature": "mean(H[:56])-mean(H[-8:])", "gain": GAIN, "cap_deg": base.CAP,
                "static_gate": "C019 last-block rest and same original valid intrinsic frame gate.",
                "development_records": p19["development_records"], "scoring_masks_sha256": sha(MASKS),
                "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                            "transformers": sys.modules["transformers"].__version__}}
    write(protocol_path, protocol)
    protocol_hash = sha(protocol_path)

    def status(phase, **items):
        write(HERE / "process.json", {"status": phase, "pid": os.getpid(), "started_at": launch["started_at"],
                                      "updated_at": stamp(), "protocol_sha256": protocol_hash, **items})

    status("static_schedule_check")
    data = load_schedule(files)
    status("reconstruct_exact_histories")
    canonical, prep = aligned_histories(inventory, data, cfg, models[0], mean, std)
    prep_path, check_path = HERE / "training-preparation.json", HERE / "pretraining-checks.json"
    write(prep_path, {**prep, "C019_sampling_lock_sha256": files["C019/sampling-lock.json"],
                      "C019_native_release_sha256": array_sha(data["native_release"].reshape(STEPS, BATCH), "<i8"),
                      "C019_source_record_ordinals_sha256": array_sha(data["source_record_ordinal"].reshape(STEPS, BATCH), "<i8"),
                      "C019_center_sha256": array_sha(data["center"], "<f8"), "C019_release_target_sha256": array_sha(data["truth"], "<f8")})
    checks = preflight(learned, data, canonical, mean, std, feature_mean, feature_std)
    write(check_path, checks)
    status("one_end_to_end_fit", training_preparation_sha256=sha(prep_path), pretraining_checks_sha256=sha(check_path))
    print("C020 pretraining checks passed", checks, flush=True)
    snapshot_hashes = tuple(c4.tensor_hash(m.state_dict()) for m in models)
    done = train(learned, models[0], controls[1], data, canonical, mean, std, feature_mean, feature_std,
                 checks["first_step_loss"], protocol_hash)
    base.require(tuple(c4.tensor_hash(m.state_dict()) for m in models) == snapshot_hashes
                 and all(p.grad is None for m in models for p in m.parameters()), "A C017 model was touched by training")
    del data, canonical
    training_path, branch_path = HERE / "training.json", HERE / "mamba-end-to-end.pt"
    write(training_path, {**done, "pretraining_checks_sha256": sha(check_path), "training_preparation_sha256": sha(prep_path),
                          "frozen_C017_state_tensor_sha256": list(snapshot_hashes)})
    branch = torch.load(branch_path, map_location=mean.device, weights_only=True)
    for key, value in branch["state_dict"].items():
        torch.testing.assert_close(value, learned.state_dict()[key], rtol=0., atol=0.)
    branch_lock = HERE / "branch-lock.json"
    write(branch_lock, {"pid": os.getpid(), "locked_at": stamp(), "protocol_sha256": protocol_hash,
                        "C017_coefficient_lock_sha256": files["C017/coefficient-lock.json"],
                        "C019_sampling_lock_sha256": files["C019/sampling-lock.json"],
                        "C020_full_branch_sha256": sha(branch_path), "training_sha256": sha(training_path), "optimizer_steps": STEPS})
    lock_hash = sha(branch_lock)
    status("learned_branch_audits", branch_lock_sha256=lock_hash)
    bank = (models, mean, std, controls, averages, scales, learned)
    audit_path, history_path = HERE / "causal-rotation-audit.json", HERE / "old-context-intervention.json"
    write(audit_path, {"branch_lock_sha256": lock_hash, "probes": audits(cfg, k, bank)})
    write(history_path, {"branch_lock_sha256": lock_hash, **interventions(bank)})
    status("development_replay", branch_lock_sha256=lock_hash)
    print("C020 branch locked; 213-record development replay started", flush=True)
    eval_start = perf_counter()
    prior = read(ROOT / "C019/results.json")
    result = replay(prior, inventory, campaign, read(TRANSFER / "external-split.json"), cfg, k, bank)
    base.require(sha(branch_lock) == lock_hash and sha(branch_path) == read(branch_lock)["C020_full_branch_sha256"]
                 and all(sha(ROOT / name) == h for name, h in files.items())
                 and all(sha(PILOT / name) == h for name, h in frozen.items()), "Frozen inputs or checkpoint changed after replay")
    result.update({"pid": os.getpid(), "started_at": launch["started_at"], "completed_at": stamp(), "protocol_sha256": protocol_hash,
                   "branch_lock_sha256": lock_hash, "C019_sampling_lock_sha256": files["C019/sampling-lock.json"],
                   "development_replay_seconds": perf_counter() - eval_start, "total_seconds": perf_counter() - start,
                   "causal_rotation_audit_sha256": sha(audit_path), "old_context_intervention_sha256": sha(history_path),
                   "scoring_masks_sha256": sha(MASKS), "final_holdout_scored": False})
    output = HERE / "results.json"
    write(output, result)
    complete = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    status("complete" if complete else "incomplete", scored=result["pooled_records"]["records"], results_sha256=sha(output),
           branch_lock_sha256=lock_hash)
    print("C020 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not complete:
        raise RuntimeError("C020 incomplete replay")


if __name__ == "__main__":
    main()
