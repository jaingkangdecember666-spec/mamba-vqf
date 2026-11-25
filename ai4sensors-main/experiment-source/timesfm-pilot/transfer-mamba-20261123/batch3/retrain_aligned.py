"""Repeat Batch 1 with the training reference rotated into the cut-off frame."""

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

BATCH3 = Path(__file__).resolve().parent
BATCH1 = BATCH3.parent
sys.path.insert(0, str(BATCH1))

import train_pair as original
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F
from tilt_eval import normalize, rotate, score_attitudes, tilt_error


MATCHED_EPOCH = 5


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aligned_targets(training, review, original_targets):
    records, pooled = [], {domain: [] for domain in original.DOMAINS}
    targets = []
    for part in training:
        use = part.eligible_training
        if not use.any():
            continue
        s, r = part.index[use], part.scheduled[use]
        source = normalize(part.base[s])
        reference = part.reference[r]
        # This is the same per-frame world reference used inside score_attitudes.
        world = rotate(normalize(part.base[r]), reference)
        source_inv = source.copy()
        source_inv[:, 1:] *= -1
        aligned = rotate(source_inv, world)
        np.testing.assert_allclose(rotate(source, aligned), world, rtol=0, atol=2e-12)
        actual = aligned.astype(np.float32)
        actual_world = rotate(source, actual)
        np.testing.assert_allclose(actual_world, world, rtol=0, atol=1e-7)
        # Independently check orientation convention and round-trip in SciPy.
        external_world = Rotation.from_quat(part.base[r], scalar_first=True).apply(reference)
        external_aligned = Rotation.from_quat(part.base[s], scalar_first=True).inv().apply(external_world)
        np.testing.assert_allclose(aligned, external_aligned, rtol=0, atol=2e-12)
        np.testing.assert_array_equal(actual[s == r], reference[s == r].astype(np.float32))
        angles = tilt_error(aligned, reference)
        pooled[part.domain].append(angles)
        records.append({"domain": part.domain, "sequence": part.item.name, "eligible_updates": len(r),
                        "different_sample": int(np.count_nonzero(s != r)),
                        "max_gap_ms": float(np.max(r - s) * 1000 / part.item.frequency),
                        "reference_change_rmse_deg": float(np.sqrt(np.mean(angles**2))),
                        "round_trip_max_abs": float(np.max(np.abs(rotate(source, aligned) - world))),
                        "float32_round_trip_max_abs": float(np.max(np.abs(actual_world - world)))})
        targets.append(actual)
    legacy = np.concatenate([p.reference[p.scheduled[p.eligible_training]].astype(np.float32) for p in training])
    np.testing.assert_array_equal(legacy, original_targets.cpu().numpy())
    summary = {}
    for domain, pieces in pooled.items():
        subset = [row for row in records if row["domain"] == domain]
        angle = np.concatenate(pieces)
        rmse = float(np.sqrt(np.mean(angle**2)))
        audited = review["pooled"][f"{domain}/train"]
        mismatched = sum(row["different_sample"] for row in subset)
        original.require(len(angle) == audited["eligible"] and mismatched == audited["different_sample"]
                         and abs(rmse - audited["rmse_deg"]) <= 1e-10,
                         f"Does not reproduce independent target audit for {domain}")
        summary[domain] = {"eligible_updates": len(angle), "different_sample": mismatched,
                           "reference_change_rmse_deg": rmse,
                           "round_trip_max_abs": max(row["round_trip_max_abs"] for row in subset),
                           "float32_round_trip_max_abs": max(row["float32_round_trip_max_abs"] for row in subset)}
    return np.concatenate(targets), {"domains": summary, "records": records}


def persist_weights(kind, label, state, epoch, mean, std, split_sha):
    path = BATCH3 / f"{kind}-{label}.pt"
    staging = path.with_suffix(".tmp")
    torch.save({"model": kind, "state_dict": state, "mean": mean.cpu(), "std": std.cpu(),
                "seed": original.SEED, "epoch": epoch, "cap_deg": original.CAP, "window": original.WINDOW,
                "split_sha256": split_sha}, staging)
    staging.replace(path)
    return path


def fit(kind, model, arrays, batches, validation, mean, std, device, split_sha):
    started = perf_counter()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history, best, best_epoch = [], float("inf"), None
    state, epoch5_state = None, None
    x, up, target = arrays
    # Keep Batch 1's update and selection logic, including full-gain selection on the pooled record mean.
    for epoch in range(1, original.EPOCHS + 1):
        model.train()
        losses = []
        for ids in batches[(epoch - 1) * original.STEPS:epoch * original.STEPS]:
            opt.zero_grad(set_to_none=True)
            delta = original.tangent(model(original.normalized(x[ids], mean, std)), up[ids], original.CAP)
            direction = F.normalize(up[ids] + delta)
            loss = (direction - target[ids]).square().sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            losses.append(loss.detach())
        metric = original.macro(original.score(model, validation, mean, std, device, kind), kind)
        train_loss = float(torch.stack(losses).mean())
        original.require(np.isfinite(train_loss) and np.isfinite(metric), "Nonfinite training statistics")
        history.append({"epoch": epoch, "optimizer_steps": epoch * original.STEPS,
                        "train_mean_squared_direction_distance": train_loss, "validation_macro_rmse_deg": metric})
        print(kind, history[-1], flush=True)
        if metric < best:
            best, best_epoch = metric, epoch
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == MATCHED_EPOCH:
            epoch5_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    paths = {"selected": persist_weights(kind, "selected", state, best_epoch, mean, std, split_sha),
             "epoch5": persist_weights(kind, "epoch5", epoch5_state, MATCHED_EPOCH, mean, std, split_sha)}
    info = {"training_seconds": perf_counter() - started, "history": history, "selected_epoch": best_epoch,
            "executed_optimizer_steps": len(history) * original.STEPS,
            "checkpoints": {label: {"file": path.name, "sha256": original.digest(path)}
                            for label, path in paths.items()}}
    return paths, info


@torch.inference_mode()
def replay(model, path, validation, device, gain):
    restored = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["state_dict"])
    model.eval()
    mean, std = restored["mean"].to(device), restored["std"].to(device)
    rows = []
    for part in validation:
        deltas = []
        for first in range(0, len(part.histories), 256):
            subset = slice(first, first + 256)
            x = torch.from_numpy(part.histories[subset]).to(device)
            up = torch.from_numpy(part.up[subset]).to(device)
            deltas.append(original.tangent(model(original.normalized(x, mean, std)), up, original.CAP).cpu().numpy())
        delta = np.concatenate(deltas)
        original.require(np.isfinite(delta).all(), f"Nonfinite output for {part.item.name}")
        rows.append(score_attitudes(part.base, part.reference, part.score, part.base[part.index], part.scheduled,
                                    {"model": delta}, {"full": ("model", 1., False),
                                                       "constant": ("model", gain, False)}, 1. / part.item.frequency))
    return rows, original.causal_check(model, validation[0], mean, std, device)


def save_results(rows, protocol_path, audit_path, training, old):
    methods = list(rows[0]["rmse_deg"])
    summary = {}
    for domain in (*original.DOMAINS, "pooled_sequences"):
        subset = [row for row in rows if domain == "pooled_sequences" or row["domain"] == domain]
        summary[domain] = {"sequences": len(subset), "frames": sum(row["frames"] for row in subset),
                           "rmse_deg": {method: original.macro(subset, method) for method in methods}}
    summary["equal_domains_rmse_deg"] = {method: float(np.mean([
        summary[domain]["rmse_deg"][method] for domain in original.DOMAINS
    ])) for method in methods}
    csv_path = BATCH3 / "validation-sequences.csv"
    fields = ("domain", "sequence", "frames", "scoring_indices_sha256")
    staging = csv_path.with_suffix(".tmp")
    with staging.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, (*fields, *methods))
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields}, **row["rmse_deg"]})
    staging.replace(csv_path)
    original.write(BATCH3 / "results.json", {
        "completed_at": original.stamp(), "protocol_sha256": original.digest(protocol_path),
        "target_audit_sha256": original.digest(audit_path), "validation_csv_sha256": original.digest(csv_path),
        "training": training, "batch1_epoch_reference": old, "summary": summary, "sequences": rows,
    })
    print(json.dumps(summary, indent=2), flush=True)


def main():
    original.require(not (BATCH3 / "results.json").exists(), "Batch 3 already finished")
    batch1 = read(BATCH1 / "results.json")
    first_protocol = read(BATCH1 / "run-protocol.json")
    scope = read(BATCH1 / "fit-scope.json")
    batch2 = read(BATCH1 / "batch2/results.json")
    calibration = read(BATCH1 / "batch2/calibration.json")
    gain = calibration["constant_gain"]
    sources = {name: original.digest(BATCH1 / name) for name in (
        "external-split.json", "run-protocol.json", "results.json", "fit-scope.json", "models.py", "train_pair.py",
        "batch2/calibration.json", "batch2/results.json", "review/training_target_audit.json",
        ".deps/transformers/models/mamba/modeling_mamba.py",
    )}
    original.require(sources["run-protocol.json"] == batch1["protocol_sha256"]
                     and sources["external-split.json"] == first_protocol["split_sha256"]
                     and sources["batch2/calibration.json"] == batch2["calibration_sha256"]
                     and gain == batch2["constant_gain"]
                     and sources[".deps/transformers/models/mamba/modeling_mamba.py"] ==
                     first_protocol["mamba_implementation"]["module_sha256"], "Frozen inputs changed")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    protocol_path = BATCH3 / "protocol.json"
    settings = {key: first_protocol[key] for key in (
        "seed", "models", "epochs", "steps_per_epoch", "batch_size", "lr", "weight_decay", "optimizer", "score",
    )}
    protocol = {"frozen_at": original.stamp(), "pid": os.getpid(), "device": torch.cuda.get_device_name(device),
                "source_sha256": sources, "settings": settings, "training_schedule_sha256": scope["training_batch_schedule_sha256"],
                "target": "rotate(inverse(normalize(base[index])), rotate(normalize(base[scheduled]), reference[scheduled]))",
                "scope": "BROAD and Sassari only. Change training targets only.",
                "selection": "Batch 1 rule: earliest minimum full-gain tilt RMSE, with all 18 validation records equally weighted.",
                "matched_epoch": MATCHED_EPOCH, "inference_gains": {"full": 1., "constant": gain}}
    if protocol_path.exists():
        previous = read(protocol_path)
        ignore = {"pid", "frozen_at"}
        original.require({k: v for k, v in protocol.items() if k not in ignore} ==
                         {k: v for k, v in previous.items() if k not in ignore}, "Frozen protocol changed")
    else:
        original.write(protocol_path, protocol)

    models = {}
    for kind in original.KINDS:
        torch.manual_seed(original.SEED)
        models[kind] = original.ResidualHead(kind)
        original.require(sum(p.numel() for p in models[kind].parameters()) == settings["models"][kind],
                         f"Parameter count changed for {kind}")
    partitions = original.load(read(BATCH1 / "external-split.json"))
    train, validation = partitions["train"], partitions["validation"]
    archived = {(row["domain"], row["sequence"]): row for row in batch2["sequences"]}
    prior = {(row["domain"], row["sequence"]): row for row in batch1["sequences"]}
    rows = []
    for part in validation:
        key = (part.domain, part.item.name)
        indices = np.flatnonzero(part.score).astype("<i8")
        mask_sha = hashlib.sha256(indices.tobytes()).hexdigest()
        old = archived.pop(key)
        prior_row = prior.pop(key)
        original.require(len(indices) == old["frames"] == prior_row["frames"] and
                         len(part.scheduled) == old["correction_updates"] and mask_sha == old["scoring_indices_sha256"],
                         f"Validation frame indices changed for {key}")
        rows.append({"domain": part.domain, "sequence": part.item.name, "frames": len(indices),
                     "scoring_indices_sha256": mask_sha,
                     "rmse_deg": {"full_vqf": old["rmse_deg"]["full_vqf"],
                                  "batch1_mamba_full": old["rmse_deg"]["mamba_full"],
                                  "batch1_tcn_full": prior_row["rmse_deg"]["tcn"],
                                  "batch2_mamba_constant": old["rmse_deg"]["mamba_constant"]}})
    original.require(not archived and not prior, "Missing validation records")
    mean, std = (torch.from_numpy(value).to(device) for value in original.moments(train))
    arrays, batches, schedule_sha = original.make_batches(train, device)
    original.require(schedule_sha == scope["training_batch_schedule_sha256"]
                     and mean.tolist() == scope["input_mean"] and std.tolist() == scope["input_std"]
                     and len(arrays[0]) == scope["training_windows"], "Training inputs or batch order changed")
    targets, audit = aligned_targets(train, read(BATCH1 / "review/training_target_audit.json"), arrays[2])
    audit_path = BATCH3 / "target-audit.json"
    original.write(audit_path, audit)
    print("world round-trip", audit["domains"], flush=True)
    arrays = (arrays[0], arrays[1], torch.from_numpy(targets).to(device))
    training, old = {}, {}
    for kind, model in models.items():
        torch.manual_seed(original.SEED)
        paths, info = fit(kind, model, arrays, batches, validation, mean, std, device, sources["external-split.json"])
        training[kind] = info
        old_info = batch1["training"][kind]
        old[kind] = {label: old_info["history"][epoch - 1] for label, epoch in (
            ("selected", old_info["selected_epoch"]), ("epoch5", MATCHED_EPOCH),
        )}
        for label, path in paths.items():
            epoch = info["selected_epoch"] if label == "selected" else MATCHED_EPOCH
            scores, causality = replay(model, path, validation, device, gain)
            expected = info["history"][epoch - 1]["validation_macro_rmse_deg"]
            actual = float(np.mean([score["full"] for score in scores]))
            original.require(abs(actual - expected) <= 1e-6, f"Failed to reproduce {kind}/{label} checkpoint")
            info["checkpoints"][label].update({"epoch": epoch, "causality": causality})
            for row, score in zip(rows, scores, strict=True):
                original.require(score["full_vqf"] == row["rmse_deg"]["full_vqf"], "FullVQF score changed")
                for gain_label in ("full", "constant"):
                    row["rmse_deg"][f"{kind}_{label}_{gain_label}"] = score[gain_label]
    save_results(rows, protocol_path, audit_path, training, old)


if __name__ == "__main__":
    main()
