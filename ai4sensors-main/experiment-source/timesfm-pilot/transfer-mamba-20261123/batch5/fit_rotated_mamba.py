"""Fit one aligned Mamba with training-only, per-draw Haar SO(3) rotations."""

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

BATCH5 = Path(__file__).resolve().parent
TRANSFER = BATCH5.parent
sys.path[:0] = [str(TRANSFER), str(TRANSFER / "batch3"), str(TRANSFER / "batch4")]

import train_pair as base
import retrain_aligned as aligned
import replay_shice as shice
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F


KIND = "mamba_v1"
ROTATION_SEED = 20261123
MARGIN = 1e-9


def digest_array(values, dtype):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=dtype).tobytes()).hexdigest()


def draw_rotations(shape, device):
    rng = np.random.Generator(np.random.PCG64(ROTATION_SEED))
    quaternions = rng.standard_normal((int(np.prod(shape)), 4))
    # A unit Gaussian direction on S3 pushed through the quaternion double cover is Haar SO(3).
    matrices = Rotation.from_quat(quaternions).as_matrix().reshape(*shape, 3, 3)
    determinants = np.linalg.det(matrices)
    det_error = float(np.max(np.abs(determinants - 1.)))
    orthogonal_error = float(np.max(np.abs(matrices @ matrices.swapaxes(-1, -2) - np.eye(3))))
    base.require(det_error < 2e-12 and orthogonal_error < 2e-12, "Improper rotation draw")
    audit = {"draws": int(determinants.size), "det_min": float(determinants.min()),
             "det_max": float(determinants.max()), "orthogonality_max_abs": orthogonal_error,
             "applied_matrix_float32_sha256": digest_array(matrices, "<f4")}
    return torch.from_numpy(matrices.astype(np.float32)).to(device), audit


def rotate_draws(x, up, target, matrices):
    transposed = matrices.transpose(1, 2)
    vectors = x[..., :12].reshape(len(x), x.shape[1], 4, 3)
    changed = torch.bmm(vectors.reshape(len(x), -1, 3), transposed).reshape_as(vectors)
    directions = torch.stack((up, target), dim=1)
    changed_directions = torch.bmm(directions, transposed)
    result = torch.cat((changed.flatten(2), x[..., 12:]), dim=-1)
    return result, changed_directions[:, 0], changed_directions[:, 1]


def norm_change(before, after):
    norm = torch.linalg.vector_norm(before, dim=-1)
    rotated_norm = torch.linalg.vector_norm(after, dim=-1)
    return ((rotated_norm - norm).abs() / norm.clamp_min(1.)).amax()


def angle(a, b):
    return torch.atan2(torch.linalg.cross(a, b).norm(dim=-1), (a * b).sum(dim=-1))


def audit_causal_inputs(validation, draws, device):
    records = []
    for domain in base.DOMAINS:
        part = next(item for item in validation if item.domain == domain)
        stop = len(part.item.imu) // 2
        _, _, schedule, histories, _, _ = base.causal_inputs(part.item.imu[:stop], part.item.frequency)
        np.testing.assert_array_equal(schedule, part.scheduled[:len(schedule)])
        np.testing.assert_allclose(histories, part.histories[:len(schedule)], atol=1e-6, rtol=0)
        records.append({"domain": domain, "record": part.item.name, "native_prefix_frames": stop,
                        "same_schedule_and_features": True})
    probe = validation[0]
    x = torch.from_numpy(probe.histories[:4]).to(device)
    up = torch.from_numpy(probe.up[:4]).to(device)
    target = torch.from_numpy(probe.reference[probe.scheduled[:4]].astype(np.float32)).to(device)
    matrices = draws[0, :len(x)]
    full, _, _ = rotate_draws(x, up, target, matrices)
    altered = x.clone()
    altered[:, 37:] = altered[:, 37:] * -5. + 2.
    changed, _, _ = rotate_draws(altered, -up, -target, matrices)
    prefix, _, _ = rotate_draws(x[:, :37], up, target, matrices)
    future_error = float((full[:, :37] - changed[:, :37]).abs().max())
    prefix_error = float((full[:, :37] - prefix).abs().max())
    rest_error = float((full[..., 12] - x[..., 12]).abs().max())
    base.require(max(future_error, prefix_error, rest_error) == 0., "Rotation changed causal prefix or rest")
    return {"raw_prefix": records, "transformed_prefix_blocks": 37,
            "transformed_future_and_label_mutation_max_abs": future_error,
            "transformed_truncated_prefix_max_abs": prefix_error, "rest_max_abs": rest_error}


def fit(model, arrays, batches, rotations, validation, mean, std, device):
    started_at = base.stamp()
    started = perf_counter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    x, up, target = arrays
    history, best, best_epoch, state = [], float("inf"), None, None
    geometry = torch.zeros(4, device=device)
    for epoch in range(1, base.EPOCHS + 1):
        model.train()
        losses = []
        first = (epoch - 1) * base.STEPS
        for step in range(first, first + base.STEPS):
            ids = batches[step]
            raw, gravity, truth = x[ids], up[ids], target[ids]
            changed, rotated_up, rotated_target = rotate_draws(raw, gravity, truth, rotations[step])
            with torch.no_grad():
                vectors = raw[..., :12].reshape(len(ids), base.WINDOW, 4, 3)
                rotated_vectors = changed[..., :12].reshape_as(vectors)
                errors = torch.stack((norm_change(vectors, rotated_vectors),
                                      norm_change(torch.stack((gravity, truth), dim=1),
                                                  torch.stack((rotated_up, rotated_target), dim=1)),
                                      (angle(gravity, truth) - angle(rotated_up, rotated_target)).abs().max(),
                                      (changed[..., 12] - raw[..., 12]).abs().max()))
                geometry = torch.maximum(geometry, errors)
            optimizer.zero_grad(set_to_none=True)
            delta = base.tangent(model(base.normalized(changed, mean, std)), rotated_up, base.CAP)
            direction = F.normalize(rotated_up + delta)
            loss = (direction - rotated_target).square().sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            losses.append(loss.detach())
        metric = base.macro(base.score(model, validation, mean, std, device, KIND), KIND)
        train_loss = float(torch.stack(losses).mean())
        base.require(np.isfinite(train_loss) and np.isfinite(metric), "Nonfinite training score")
        history.append({"epoch": epoch, "optimizer_steps": epoch * base.STEPS,
                        "train_mean_squared_direction_distance": train_loss,
                        "external_validation_macro_rmse_deg": metric})
        print(KIND, history[-1], flush=True)
        if metric < best:
            best, best_epoch = metric, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    checks = dict(zip(("feature_norm_scaled_error", "up_and_target_norm_scaled_error",
                       "up_target_angle_max_error_rad", "rest_max_abs"), geometry.tolist(), strict=True))
    base.require(max(checks.values()) < 2e-6 and checks["rest_max_abs"] == 0., f"Rotation audit failed: {checks}")
    return state, {"fit_started_at": started_at, "fit_completed_at": base.stamp(),
                   "fit_seconds": perf_counter() - started, "fits": 1, "executed_optimizer_steps": base.EPOCHS * base.STEPS,
                   "selected_epoch": best_epoch, "history": history}, checks


def comparison(name, group, frames, mask_sha, pure, old, new):
    base.require(all(np.isfinite([pure, old, new])), f"Nonfinite score for {name}")
    return {"sequence": name, "group": group, "frames": frames, "scoring_indices_sha256": mask_sha,
            "full_vqf_deg": pure, "batch3_mamba_deg": old, "batch5_mamba_deg": new,
            "delta_vs_full_vqf_deg": new - pure, "delta_vs_batch3_deg": new - old}


def external_replay(model, validation, mean, std, device, archived):
    rows = []
    scores = base.score(model, validation, mean, std, device, KIND)
    for part, new, old in zip(validation, scores, archived, strict=True):
        score_index = np.flatnonzero(part.score)
        mask_sha = digest_array(score_index, "<i8")
        pure = new["rmse_deg"]["full_vqf"]
        base.require((new["domain"], new["sequence"], new["frames"], mask_sha, pure) ==
                     (old["domain"], old["sequence"], old["frames"], old["scoring_indices_sha256"],
                      old["rmse_deg"]["full_vqf"]), "External baseline or frame indices changed")
        rows.append(comparison(new["sequence"], new["domain"], new["frames"], mask_sha, pure,
                               old["rmse_deg"]["mamba_v1_selected_full"], new["rmse_deg"][KIND]))
    return rows


def shice_replay(model, mean, std, device, archived):
    old_path = shice.LEGACY / "rounds/R00001/result.json"
    past = {row["name"]: row for row in shice.historical_dev(old_path)["sequences"]}
    names = {row["sequence"] for row in archived}
    allowed = {f"{device_name}-{i}" for device_name in shice.DEVICES for i in range(1, 11)}
    base.require(len(archived) == len(names) == 37 and names == set(past) and names < allowed,
                 "Unexpected SHICE inventory")
    rows = []
    for old in archived:
        name = old["sequence"]
        new = shice.replay(past[name], {KIND: (model, mean, std)}, device)
        rmse, mask_sha = new["rmse_deg"], new["scoring_indices_sha256"]
        base.require((new["frames"], mask_sha, rmse["full_vqf"]) ==
                     (old["frames"], old["scoring_indices_sha256"], old["rmse_deg"]["full_vqf"]),
                     f"SHICE baseline or scored index changed: {name}")
        row = comparison(name, old["device"], new["frames"], mask_sha, rmse["full_vqf"],
                         old["rmse_deg"][KIND], rmse[KIND])
        row["prediction_scored_frames_rms_deg"] = new["prediction_magnitude_deg"][KIND]["scored_frames_rms_deg"]
        rows.append(row)
        print("SHICE", name, row["frames"], row["batch5_mamba_deg"], flush=True)
    return rows


def summarize(rows):
    scores = {k: float(np.mean([row[f"{k}_deg"] for row in rows]))
              for k in ("full_vqf", "batch3_mamba", "batch5_mamba")}
    return {"records": len(rows), "frames": sum(row["frames"] for row in rows), "macro_rmse_deg": scores,
            "wins_vs_full_vqf": sum(row["delta_vs_full_vqf_deg"] < -MARGIN for row in rows),
            "wins_vs_batch3": sum(row["delta_vs_batch3_deg"] < -MARGIN for row in rows)}


def main():
    started_at = base.stamp()
    selection_path = BATCH5 / "selection.json"
    weight_path = BATCH5 / "mamba-so3-selected.pt"
    # Refuse a second fit once a trained checkpoint exists.
    base.require(not selection_path.exists() and not weight_path.exists(), "Batch 5 has already fitted its one head")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    read = aligned.read
    previous = read(TRANSFER / "batch3/results.json")
    previous_protocol = read(TRANSFER / "batch3/protocol.json")
    lock = read(TRANSFER / "review/strict-external-lock.json")
    scope = read(TRANSFER / "fit-scope.json")
    source_files = ("external-split.json", "fit-scope.json", "models.py", "train_pair.py",
                    ".deps/transformers/models/mamba/modeling_mamba.py", "review/training_target_audit.json")
    sha = {name: base.digest(TRANSFER / name) for name in source_files}
    base.require(all(previous_protocol["source_sha256"][name] == value for name, value in sha.items()),
                 "A frozen Batch 3 input changed")
    for name in ("batch3/protocol.json", "batch3/results.json", "batch3/retrain_aligned.py",
                 "batch3/mamba_v1-selected.pt", "batch4/protocol.json", "batch4/results.json",
                 "batch4/replay_shice.py", "review/strict-external-lock.json"):
        sha[name] = base.digest(TRANSFER / name)
    base.require(sha["review/strict-external-lock.json"] == shice.LOCK_SHA
                 and sha["batch3/protocol.json"] == previous["protocol_sha256"]
                 and sha["batch3/results.json"] == lock["selection_evidence_sha256"]
                 and sha["batch3/mamba_v1-selected.pt"] == lock["checkpoints"][KIND]["sha256"],
                 "Locked Batch 3 reference changed")
    protocol_path = BATCH5 / "protocol.json"
    protocol = {
        "started_at": started_at, "device": torch.cuda.get_device_name(device), "source_sha256": sha,
        "change": "Independent Haar SO(3) rotation of each selected external training-window occurrence. Same rotation for all 64 blocks, all four vector fields, up, and aligned target. Rest unchanged. Apply before frozen normalization.",
        "rotation_rng": "NumPy PCG64, isotropic Gaussian quaternion normalized by scipy Rotation",
        "rotation_seed": ROTATION_SEED, "training_schedule_sha256": scope["training_batch_schedule_sha256"],
        "settings": previous_protocol["settings"], "scope": "Single Mamba fit on BROAD and Sassari training windows.",
        "selection": previous_protocol["selection"], "inference_gain": 1.,
        "evaluation": "18 original external validation records and then 37 original SHICE R1–R4 records.",
        "win_margin_deg": MARGIN,
    }
    if protocol_path.exists():
        existing = read(protocol_path)
        existing["started_at"] = started_at
        base.require(existing == protocol, "Batch 5 protocol changed")
    else:
        base.write(protocol_path, protocol)

    torch.manual_seed(base.SEED)
    model = base.ResidualHead(KIND).to(device)
    base.require(sum(p.numel() for p in model.parameters()) == protocol["settings"]["models"][KIND],
                 "Mamba parameter count changed")
    partitions = base.load(read(TRANSFER / "external-split.json"))
    train, validation = partitions["train"], partitions["validation"]
    frozen = torch.load(TRANSFER / "batch3/mamba_v1-selected.pt", map_location="cpu", weights_only=True)
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    np.testing.assert_array_equal(mean.cpu().numpy(), np.asarray(scope["input_mean"], dtype=np.float32))
    np.testing.assert_array_equal(std.cpu().numpy(), np.asarray(scope["input_std"], dtype=np.float32))
    arrays, batches, batch_sha = base.make_batches(train, device)
    base.require(batch_sha == protocol["training_schedule_sha256"] == previous_protocol["training_schedule_sha256"]
                 and tuple(batches.shape) == (192, 96) and len(arrays[0]) == scope["training_windows"], "Batch matrix changed")
    target, audit = aligned.aligned_targets(train, read(TRANSFER / "review/training_target_audit.json"), arrays[2])
    base.require(audit == read(TRANSFER / "batch3/target-audit.json"), "Aligned target audit changed")
    arrays = (arrays[0], arrays[1], torch.from_numpy(target).to(device))
    draws, rotation_audit = draw_rotations(tuple(batches.shape), device)
    causality = audit_causal_inputs(validation, draws, device)
    print("validated batches and rotations", batch_sha, rotation_audit, flush=True)
    torch.manual_seed(base.SEED)
    state, training, applied_checks = fit(model, arrays, batches, draws, validation, mean, std, device)
    rotation_audit["applied_geometry"] = applied_checks
    stage = weight_path.with_suffix(".tmp")
    torch.save({"model": KIND, "state_dict": state, "mean": mean.cpu(), "std": std.cpu(),
                "seed": base.SEED, "epoch": training["selected_epoch"], "cap_deg": base.CAP, "window": base.WINDOW,
                "split_sha256": sha["external-split.json"], "rotation_seed": ROTATION_SEED}, stage)
    stage.replace(weight_path)
    restored = torch.load(weight_path, weights_only=True, map_location="cpu")
    model.load_state_dict(restored["state_dict"])
    model.eval().requires_grad_(False)
    del arrays, batches, draws, partitions, train, target
    causality["selected_model"] = base.causal_check(model, validation[0], mean, std, device)
    external_rows = external_replay(model, validation, mean, std, device, previous["sequences"])
    pooled = summarize(external_rows)
    selected_metric = training["history"][training["selected_epoch"] - 1]["external_validation_macro_rmse_deg"]
    base.require(abs(pooled["macro_rmse_deg"]["batch5_mamba"] - selected_metric) < 1e-6,
                 "Saved checkpoint failed external score reproduction")
    external_path = BATCH5 / "external-sequences.csv"
    shice.save_csv(external_path, external_rows)
    domain_metrics = {domain: summarize([r for r in external_rows if r["group"] == domain]) for domain in base.DOMAINS}
    domain_equal = {key: float(np.mean([m["macro_rmse_deg"][key] for m in domain_metrics.values()]))
                    for key in pooled["macro_rmse_deg"]}
    selection = {"selected_at": base.stamp(), "weights": weight_path.name, "weights_sha256": base.digest(weight_path),
                 "protocol_sha256": base.digest(protocol_path), "training": training, "rotation_audit": rotation_audit,
                 "causality": causality, "external_csv_sha256": base.digest(external_path),
                 "external": {"pooled": pooled, "by_domain": domain_metrics, "equal_domain_rmse_deg": domain_equal}}
    base.write(selection_path, selection)
    print("externally selected checkpoint", training["selected_epoch"], domain_equal, flush=True)

    batch4 = read(TRANSFER / "batch4/results.json")
    base.require(sha["batch4/protocol.json"] == batch4["protocol_sha256"], "Batch 4 protocol changed")
    shice_rows = shice_replay(model, restored["mean"].to(device), restored["std"].to(device), device, batch4["sequences"])
    shice_path = BATCH5 / "shice-sequences.csv"
    shice.save_csv(shice_path, shice_rows)
    shice_summary = {"all": summarize(shice_rows), "by_device": {
        name: summarize([r for r in shice_rows if r["group"] == name]) for name in shice.DEVICES}}
    base.write(BATCH5 / "results.json", {**selection, "started_at": started_at, "completed_at": base.stamp(),
               "selection_sha256": base.digest(selection_path), "shice_csv_sha256": base.digest(shice_path),
               "shice": shice_summary, "external_sequences": external_rows, "shice_sequences": shice_rows})
    print(json.dumps({"external": selection["external"], "shice": shice_summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
