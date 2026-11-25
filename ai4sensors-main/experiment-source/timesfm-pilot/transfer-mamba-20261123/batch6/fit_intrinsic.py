"""Fit and score one Mamba in causal intrinsic coordinates."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

BATCH6 = Path(__file__).resolve().parent
TRANSFER = BATCH6.parent
sys.path[:0] = [str(TRANSFER), str(TRANSFER / "batch3"), str(TRANSFER / "batch4")]

import train_pair as original
import retrain_aligned as aligned
import replay_shice as shice
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F
from tilt_eval import normalize, score_attitudes, tilt_error

from intrinsic import ACC, ACC_FLOOR, DEGENERATE, GYRO, GYRO_FLOOR, build_frame, predict_delta, transform_directions, transform_inputs


KIND = "mamba_v1"
METHODS = ("full_vqf", "batch3_mamba", "batch5_mamba", "batch6_mamba")


def count_frames(frame):
    return {"acceleration": int(np.count_nonzero(frame.source == ACC)),
            "gyro_minus_bias": int(np.count_nonzero(frame.source == GYRO)),
            "degenerate": int(np.count_nonzero(frame.source == DEGENERATE))}


def geometry(frame):
    basis = frame.basis[frame.valid]
    if not len(basis):
        return {"orthogonality_max_abs": 0., "det_max_abs_from_one": 0.}
    orth = float(np.max(np.abs(basis.transpose(0, 2, 1) @ basis - np.eye(3))))
    det = float(np.max(np.abs(np.linalg.det(basis) - 1.)))
    original.require(max(orth, det) <= 2e-12, "Invalid intrinsic basis")
    return {"orthogonality_max_abs": orth, "det_max_abs_from_one": det}


def transform_training(train, target, device):
    features, directions, counts, moment_rows = [], [], [], []
    start = 0
    for part in train:
        raw = part.histories[part.eligible_training]
        if len(raw) == 0:
            continue
        frame = build_frame(raw)
        canonical = transform_inputs(raw, frame)
        reference = target[start:start + len(raw)]
        mapped = transform_directions(reference, frame)
        np.testing.assert_array_equal(raw[..., 12], canonical[..., 12])
        valid = frame.valid
        np.testing.assert_allclose(np.einsum("ni,nji->nj", mapped[valid], frame.basis[valid]),
                                   reference[valid], rtol=0, atol=1e-7)
        if valid.any():
            values = canonical[valid].astype(np.float64)
            moment_rows.append((part.domain, values.mean(axis=(0, 1)), np.square(values).mean(axis=(0, 1))))
        counts.append({"domain": part.domain, "sequence": part.item.name, "windows": len(raw),
                       **count_frames(frame), **geometry(frame)})
        features.append(canonical)
        directions.append(mapped)
        start += len(raw)
    original.require(start == len(target), "Training windows and labels are misaligned")
    domains = [np.mean([(mean, square) for key, mean, square in moment_rows if key == domain], axis=0)
               for domain in original.DOMAINS]
    mean, square = np.mean(domains, axis=0)
    std = np.sqrt(np.maximum(square - mean**2, 1e-6))
    x, y = (torch.from_numpy(np.concatenate(values)).to(device) for values in (features, directions))
    validity = torch.linalg.vector_norm(x[:, -1, 6:9], dim=-1) > .5
    original.require(int(validity.sum()) == sum(row["windows"] - row["degenerate"] for row in counts),
                     "Geometry or training validity mismatch")
    moments = {"records": counts, "input_mean": mean.astype(np.float32).tolist(),
               "input_std": std.astype(np.float32).tolist(),
               "degenerate_windows": sum(row["degenerate"] for row in counts)}
    arrays = (x, x[:, -1, 6:9], y, validity)
    return arrays, (torch.tensor(moments[k], dtype=torch.float32, device=device) for k in ("input_mean", "input_std")), moments


def intrinsic_windows(histories):
    frame = build_frame(histories)
    return frame, {**count_frames(frame), **geometry(frame)}


def predict(model, histories, frame, mean, std):
    deltas = []
    for first in range(0, len(histories), 256):
        sl = slice(first, first + 256)
        subset = type(frame)(frame.basis[sl], frame.source[sl], frame.anchor[sl])
        deltas.append(predict_delta(model, histories[sl], mean, std, frame=subset))
    result = np.concatenate(deltas)
    original.require(np.isfinite(result).all() and np.all(result[~frame.valid] == 0.), "Invalid or nonzero degenerate delta")
    return result


def external_rows(model, prepared, mean, std):
    rows = []
    for part, frame, counts in prepared:
        delta = predict(model, part.histories, frame, mean, std)
        score = score_attitudes(part.base, part.reference, part.score, part.base[part.index], part.scheduled,
                                {KIND: delta}, {KIND: (KIND, 1., False)}, 1. / part.item.frequency)
        rows.append({"rmse_deg": score, "frames": int(part.score.sum()), **counts})
    return rows


def fit(model, arrays, batches, prepared, mean, std):
    started = perf_counter()
    when = original.stamp()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history, best, best_epoch, state = [], float("inf"), None, None
    x, up, target, valid = arrays
    degenerate_draws = 0
    for epoch in range(1, original.EPOCHS + 1):
        model.train()
        losses = []
        for ids in batches[(epoch - 1) * original.STEPS:epoch * original.STEPS]:
            optimizer.zero_grad(set_to_none=True)
            delta = original.tangent(model(original.normalized(x[ids], mean, std)), up[ids], original.CAP)
            predicted = F.normalize(up[ids] + delta.masked_fill(~valid[ids, None], 0.))
            # A degenerate window's inherent no-correction error is constant with respect to the model.
            error = (predicted - target[ids]).square().sum(dim=-1)
            loss = error.masked_fill(~valid[ids], 0.).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(loss.detach())
        model.eval()
        degenerate_draws += int((~valid[batches[(epoch - 1) * original.STEPS:epoch * original.STEPS]]).sum())
        metric = original.macro(external_rows(model, prepared, mean, std), KIND)
        mean_loss = float(torch.stack(losses).mean())
        original.require(np.isfinite(metric) and np.isfinite(mean_loss), "Nonfinite training or validation result")
        history.append({"epoch": epoch, "optimizer_steps": epoch * original.STEPS,
                        "train_mean_squared_direction_distance_excluding_constant_degenerate_error": mean_loss,
                        "external_validation_macro_rmse_deg": metric})
        print(KIND, history[-1], flush=True)
        if metric < best:
            best, best_epoch = metric, epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, {"fit_started_at": when, "fit_completed_at": original.stamp(), "fit_seconds": perf_counter() - started,
                   "fits": 1, "executed_optimizer_steps": len(batches), "selected_epoch": best_epoch,
                   "degenerate_batch_draws": degenerate_draws, "history": history}


def audit_causality(model, prepared, mean, std):
    rows = []
    for domain in original.DOMAINS:
        part, _, _ = next(row for row in prepared if row[0].domain == domain)
        probe = len(part.scheduled) // 2
        end = part.scheduled[probe] + 1
        _, _, release, sliced, _, _ = original.causal_inputs(part.item.imu[:end], part.item.frequency)
        modified = part.item.imu.copy()
        modified[end:] = np.nan
        _, _, changed_release, changed, _, _ = original.causal_inputs(modified, part.item.frequency)
        np.testing.assert_array_equal(release, part.scheduled[:probe + 1])
        np.testing.assert_array_equal(release, changed_release[:probe + 1])
        np.testing.assert_array_equal(sliced, part.histories[:probe + 1])
        np.testing.assert_array_equal(sliced, changed[:probe + 1])
        histories = part.histories[probe - 3:probe + 1]
        canonical = torch.from_numpy(transform_inputs(histories, build_frame(histories))).to(mean.device)
        x = original.normalized(canonical, mean, std)
        with torch.inference_mode():
            cutoff = 37
            prefix = model(x, all_steps=True)[:, :cutoff]
            altered = x.clone()
            altered[:, cutoff:] *= -5.
            future_error = float((prefix - model(altered, all_steps=True)[:, :cutoff]).abs().max())
            prefix_error = float((prefix - model(x[:, :cutoff], all_steps=True)).abs().max())
        original.require(max(future_error, prefix_error) <= 2e-5, "Mamba hidden-prefix invariance failed")
        delta = predict_delta(model, sliced[-4:], mean, std)
        changed_delta = predict_delta(model, changed[probe - 3:probe + 1], mean, std)
        np.testing.assert_array_equal(delta, changed_delta)
        rows.append({"domain": domain, "sequence": part.item.name, "release_native_index": int(end - 1),
                     "exact_schedule_and_past_features": True, "release_delta_future_error": 0.,
                     "inner_prefix_max_abs": prefix_error, "inner_future_mutation_max_abs": future_error})
    return {"semantics": "The complete 64-block window chooses one basis at its release. No predictions are published at internal tokens. Inner prefix check keeps the complete window's basis fixed.",
            "probes": rows}


def equivariance(model, prepared, mean, std):
    rotations = [*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))]
    records = []
    for part, _, _ in prepared:
        eligible = np.flatnonzero(part.score[part.scheduled])
        take = eligible[np.linspace(0, len(eligible) - 1, 32, dtype=int)]
        raw = part.histories[take]
        frame = build_frame(raw)
        up = normalize(raw[:, -1, 6:9].astype(np.float64))
        direction = normalize(up + predict_delta(model, raw, mean, std, frame=frame))
        errors, switches = [], []
        for rotation in rotations:
            moved = raw.copy()
            vectors = raw[..., :12].reshape(len(raw), original.WINDOW, 4, 3)
            moved[..., :12] = rotation.apply(vectors.reshape(-1, 3)).reshape(len(raw), original.WINDOW, 12)
            moved_frame = build_frame(moved)
            changed = predict_delta(model, moved, mean, std, frame=moved_frame)
            rotated_up = normalize(moved[:, -1, 6:9].astype(np.float64))
            recovered = rotation.inv().apply(normalize(rotated_up + changed))
            errors.append(tilt_error(direction, recovered).tolist())
            switches.append(int(np.count_nonzero((frame.source != moved_frame.source) |
                                                  (frame.anchor != moved_frame.anchor))))
        records.append({"domain": part.domain, "sequence": part.item.name, "errors_deg": errors,
                        "anchor_or_source_switches": switches, "source_counts": count_frames(frame),
                        "correction_deg": tilt_error(direction, up).tolist()})
    error = np.concatenate([row["errors_deg"] for row in records], axis=1)
    correction = np.concatenate([row["correction_deg"] for row in records])
    return {"rotation_labels": ["90deg_x", "90deg_y", "90deg_z", "63deg_about_(1,2,-1)"],
            "windows": len(correction), "rotated_comparisons": error.size,
            "rms_deg": float(np.sqrt(np.mean(error**2))), "p95_deg": float(np.quantile(error, .95)),
            "max_deg": float(error.max()), "per_rotation_rms_deg": np.sqrt(np.mean(error**2, axis=1)).tolist(),
            "anchor_or_source_switches": int(sum(sum(row["anchor_or_source_switches"]) for row in records)),
            "correction_rms_deg": float(np.sqrt(np.mean(correction**2))), "records": records}


def check_fallback(model, mean, std):
    values = np.zeros((3, original.WINDOW, 13), dtype=np.float32)
    values[..., 8] = 1.
    values[0, 13, 0] = 2 * ACC_FLOOR
    values[1:, :, 9] = 1.
    values[1:, :, 3] = 1.
    values[1, 17, 3] += 2 * GYRO_FLOOR
    frame = build_frame(values)
    np.testing.assert_array_equal(frame.source, [ACC, GYRO, DEGENERATE])
    np.testing.assert_array_equal(frame.anchor, [13, 17, -1])
    np.testing.assert_array_equal(predict_delta(model, values, mean, std, frame=frame)[2], np.zeros(3))
    return {"acceleration_anchor": 13, "gyro_minus_bias_anchor": 17, "degenerate_zero_delta": True}


def mask_digest(mask):
    return hashlib.sha256(np.flatnonzero(mask).astype("<i8").tobytes()).hexdigest()


def row_for(name, group, mask, baseline, b3, b5, b6, counts, key, archive_masks):
    old_hash = mask_digest(mask)
    archive_masks[key] = mask
    return {"sequence": name, "group": group, "native_frames": len(mask), "frames": int(mask.sum()),
            "mask_key": key, "scoring_indices_sha256": old_hash, "full_vqf_deg": baseline, "batch3_mamba_deg": b3,
            "batch5_mamba_deg": b5, "batch6_mamba_deg": b6,
            "delta_vs_full_vqf_deg": b6 - baseline, "delta_vs_batch3_deg": b6 - b3, "delta_vs_batch5_deg": b6 - b5,
            **counts}


def verify_mask(row, frame_count, digest, baseline):
    original.require((row["frames"], row["scoring_indices_sha256"], row["full_vqf_deg"]) ==
                     (frame_count, digest, baseline), f"Archived scoring frames or baseline changed: {row['sequence']}")


def replay_shice(model, mean, std, archived, b5_by_name, masks):
    old_path = shice.LEGACY / "rounds/R00001/result.json"
    old_rows = {row["name"]: row for row in shice.historical_dev(old_path)["sequences"]}
    allowed = {f"{d}-{i}" for d in shice.DEVICES for i in range(1, 11)}
    original.require(len(archived) == len(old_rows) == 37 and {r["sequence"] for r in archived} == set(old_rows)
                     and set(old_rows) < allowed, "Unexpected SHICE records")
    rows = []
    for i, before in enumerate(archived):
        name = before["sequence"]
        dt, signal, reference, mask = shice.raw_record(name)
        base_q, index, release, histories, _, _ = original.causal_inputs(signal, 1. / dt)
        frame, counts = intrinsic_windows(histories)
        delta = predict(model, histories, frame, mean, std)
        scores = score_attitudes(base_q, reference, mask, base_q[index], release,
                                 {KIND: delta}, {KIND: (KIND, 1., False)}, dt)
        b5 = b5_by_name[name]
        row = row_for(name, before["device"], mask, scores["full_vqf"], before["rmse_deg"][KIND],
                      b5["batch5_mamba_deg"], scores[KIND], counts, f"shice_{i:02}", masks)
        verify_mask(row, before["frames"], before["scoring_indices_sha256"], before["rmse_deg"]["full_vqf"])
        verify_mask(row, b5["frames"], b5["scoring_indices_sha256"], b5["full_vqf_deg"])
        rows.append(row)
        print("SHICE", name, row["frames"], row["batch6_mamba_deg"], counts["degenerate"], flush=True)
    return rows


def summarize(rows):
    return {"records": len(rows), "frames": sum(row["frames"] for row in rows),
            "macro_rmse_deg": {key: float(np.mean([row[f"{key}_deg"] for row in rows])) for key in METHODS},
            "wins": {key: sum(row["batch6_mamba_deg"] < row[f"{key}_deg"] - 1e-9 for row in rows) for key in METHODS[:-1]},
            "degenerate": sum(row["degenerate"] for row in rows),
            "frame_source_counts": {key: sum(row[key] for row in rows) for key in ("acceleration", "gyro_minus_bias", "degenerate")}}


def main():
    started = original.stamp()
    weights_path = BATCH6 / "mamba-intrinsic-selected.pt"
    original.require(not weights_path.exists(), "Batch 6 has already fitted its one head")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    read = aligned.read
    b3_protocol = read(TRANSFER / "batch3/protocol.json")
    b3 = read(TRANSFER / "batch3/results.json")
    lock = read(TRANSFER / "review/strict-external-lock.json")
    sources = {f: original.digest(TRANSFER / f) for f in (
        "external-split.json", "fit-scope.json", "train_pair.py", "models.py",
        ".deps/transformers/models/mamba/modeling_mamba.py", "review/training_target_audit.json",
        "batch3/protocol.json", "batch3/results.json", "batch3/mamba_v1-selected.pt",
        "batch3/retrain_aligned.py", "batch4/replay_shice.py", "review/strict-external-lock.json",
        "review/axis_response_strict.json", "review/axis_response_so3.json",
    )}
    original.require(all(sources[k] == v for k, v in b3_protocol["source_sha256"].items() if k in sources)
                     and sources["batch3/protocol.json"] == b3["protocol_sha256"]
                     and sources["review/strict-external-lock.json"] == shice.LOCK_SHA
                     and sources["batch3/results.json"] == lock["selection_evidence_sha256"]
                     and sources["batch3/mamba_v1-selected.pt"] == lock["checkpoints"][KIND]["sha256"],
                     "Original frozen training or comparison inputs changed")
    protocol_path = BATCH6 / "protocol.json"
    protocol = {"started_at": started, "device": torch.cuda.get_device_name(device), "source_sha256": sources,
                "implementation_sha256": {name: original.digest(BATCH6 / name)
                                          for name in ("intrinsic.py", "fit_intrinsic.py")},
                "settings": b3_protocol["settings"], "target": b3_protocol["target"], "model": KIND,
                "selection": b3_protocol["selection"], "gain": 1., "training_schedule_sha256": b3_protocol["training_schedule_sha256"],
                "frame": {"z": "last completed block VQF up", "x": "largest transverse acceleration, otherwise largest transverse gyro minus bias; earliest exact tie",
                          "y": "z cross x", "acceleration_floor_m_s2": ACC_FLOOR, "gyro_floor_rad_s": GYRO_FLOOR,
                          "degenerate": "Zero delta. Keep its sampled draw and its share of batch denominator, with no label-dependent gradient."},
                "normalization": "Feature statistics from valid framed eligible external training windows only. Equal records within domain and equal domains. Original 0.001 standard-deviation floor.",
                "evaluation": "18 native external validation records, then 37 original SHICE R1-R4 records. Original release and hold. No rotation augmentation."}
    if protocol_path.exists():
        prior = read(protocol_path)
        prior["started_at"] = started
        original.require(prior == protocol, "Frozen protocol changed")
    else:
        original.write(protocol_path, protocol)

    torch.manual_seed(original.SEED)
    model = original.ResidualHead(KIND).to(device)
    original.require(sum(p.numel() for p in model.parameters()) == 5595 == protocol["settings"]["models"][KIND],
                     "Mamba model has changed")
    partitions = original.load(read(TRANSFER / "external-split.json"))
    train, validation = partitions["train"], partitions["validation"]
    raw, batches, schedule = original.make_batches(train, device)
    scope = read(TRANSFER / "fit-scope.json")
    original.require(schedule == scope["training_batch_schedule_sha256"] == protocol["training_schedule_sha256"]
                     and tuple(batches.shape) == (192, 96), "Training batch schedule changed")
    targets, label_audit = aligned.aligned_targets(train, read(TRANSFER / "review/training_target_audit.json"), raw[2])
    original.require(label_audit == read(TRANSFER / "batch3/target-audit.json"), "Aligned labels changed")
    del raw
    arrays, (mean, std), moments = transform_training(train, targets, device)
    moments.update({"batch_schedule_sha256": schedule, "training_windows": len(targets)})
    training_path = BATCH6 / "training-frame-audit.json"
    original.write(training_path, moments)
    del targets, partitions, train
    prepared = [(part, *intrinsic_windows(part.histories)) for part in validation]
    synthetic_check = check_fallback(model, mean, std)
    print("frozen intrinsic normalization and schedule", schedule, "degenerate", moments["degenerate_windows"], flush=True)
    torch.manual_seed(original.SEED)
    state, training = fit(model, arrays, batches, prepared, mean, std)
    stage = weights_path.with_suffix(".tmp")
    torch.save({"model": KIND, "state_dict": state, "mean": mean.cpu(), "std": std.cpu(), "seed": original.SEED,
                "epoch": training["selected_epoch"], "window": original.WINDOW, "cap_deg": original.CAP,
                "split_sha256": sources["external-split.json"], "intrinsic_acc_floor": ACC_FLOOR,
                "intrinsic_gyro_floor": GYRO_FLOOR}, stage)
    stage.replace(weights_path)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().requires_grad_(False)
    mean, std = checkpoint["mean"].to(device), checkpoint["std"].to(device)
    del arrays, batches
    scored = external_rows(model, prepared, mean, std)
    history_score = training["history"][training["selected_epoch"] - 1]["external_validation_macro_rmse_deg"]
    original.require(abs(original.macro(scored, KIND) - history_score) <= 1e-6, "Selected checkpoint failed reproduction")
    causality = audit_causality(model, prepared, mean, std)
    axis = equivariance(model, prepared, mean, std)
    original.require(axis["windows"] == 576 and axis["rotated_comparisons"] == 2304, "Unexpected axis audit coverage")
    axis_path = BATCH6 / "equivariance.json"
    original.write(axis_path, {**axis, "reference": {label: {key: data[key] for key in ("windows", "rotated_comparisons", "frame_change_rms_deg", "correction_rms_deg")}
                   for label, data in (("batch3", read(TRANSFER / "review/axis_response_strict.json")),
                                       ("batch5", read(TRANSFER / "review/axis_response_so3.json")))}})
    selection_path = BATCH6 / "selection.json"
    selection = {"selected_at": original.stamp(), "protocol_sha256": original.digest(protocol_path),
                 "training_audit_sha256": original.digest(training_path), "weights": weights_path.name,
                 "weights_sha256": original.digest(weights_path), "equivariance_sha256": original.digest(axis_path),
                 "training": training, "causality": causality, "fallback_check": synthetic_check,
                 "external_selected_macro_rmse_deg": history_score}
    original.write(selection_path, selection)
    print("external lock", training["selected_epoch"], history_score, "rotation rms", axis["rms_deg"], flush=True)

    b5 = read(TRANSFER / "batch5/results.json")
    b4 = read(TRANSFER / "batch4/results.json")
    original.require(original.digest(TRANSFER / "batch5" / b5["weights"]) == b5["weights_sha256"]
                     and original.digest(TRANSFER / "batch5/selection.json") == b5["selection_sha256"]
                     and original.digest(TRANSFER / "batch4/protocol.json") == b4["protocol_sha256"],
                     "Frozen comparison evidence changed")
    ext_b5 = {(row["group"], row["sequence"]): row for row in b5["external_sequences"]}
    masks, external = {}, []
    for i, (part, new, old) in enumerate(zip(validation, scored, b3["sequences"], strict=True)):
        name = part.item.name
        b5_row = ext_b5.pop((part.domain, name))
        original.require((old["domain"], old["sequence"]) == (part.domain, name), "External record order changed")
        values = row_for(name, part.domain, part.score, new["rmse_deg"]["full_vqf"],
                         old["rmse_deg"]["mamba_v1_selected_full"], b5_row["batch5_mamba_deg"], new["rmse_deg"][KIND],
                         {k: v for k, v in new.items() if k not in ("rmse_deg", "frames")}, f"external_{i:02}", masks)
        verify_mask(values, old["frames"], old["scoring_indices_sha256"], old["rmse_deg"]["full_vqf"])
        verify_mask(values, b5_row["frames"], b5_row["scoring_indices_sha256"], b5_row["full_vqf_deg"])
        external.append(values)
    original.require(not ext_b5, "Missing external records")
    shice_rows = replay_shice(model, mean, std, b4["sequences"], {r["sequence"]: r for r in b5["shice_sequences"]}, masks)
    paths = {}
    for key, rows in (("external", external), ("shice", shice_rows)):
        path = BATCH6 / f"{key}-sequences.csv"
        shice.save_csv(path, rows)
        paths[f"{key}_csv_sha256"] = original.digest(path)
    mask_path = BATCH6 / "scoring-masks.npz"
    stage = mask_path.with_suffix(".tmp")
    with stage.open("wb") as stream:
        np.savez_compressed(stream, **masks)
    stage.replace(mask_path)
    ext_domain = {d: summarize([r for r in external if r["group"] == d]) for d in original.DOMAINS}
    summary = {"external_pooled": summarize(external), "external_by_domain": ext_domain,
               "external_equal_domain_rmse_deg": {k: float(np.mean([m["macro_rmse_deg"][k] for m in ext_domain.values()]))
                                                  for k in METHODS},
               "shice_pooled": summarize(shice_rows),
               "shice_by_device": {d: summarize([r for r in shice_rows if r["group"] == d]) for d in shice.DEVICES}}
    original.write(BATCH6 / "results.json", {**selection, **paths, "started_at": started, "completed_at": original.stamp(),
                   "selection_sha256": original.digest(selection_path), "masks_sha256": original.digest(mask_path),
                   "comparison_sha256": {f: original.digest(TRANSFER / f) for f in ("batch4/results.json", "batch5/results.json")},
                   "axis_summary": {k: v for k, v in axis.items() if k != "records"}, "summary": summary,
                   "external_sequences": external, "shice_sequences": shice_rows})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
