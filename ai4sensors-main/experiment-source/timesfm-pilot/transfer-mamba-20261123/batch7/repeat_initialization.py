"""Repeat Batch 6 with a single different network initialization."""

import hashlib
import json
from pathlib import Path
import sys

BATCH7 = Path(__file__).resolve().parent
TRANSFER = BATCH7.parent
BATCH6 = TRANSFER / "batch6"
sys.path[:0] = [str(BATCH6), str(TRANSFER), str(TRANSFER / "batch3"), str(TRANSFER / "batch4")]

import fit_intrinsic as b6
import numpy as np
import torch


base = b6.original
read = b6.aligned.read
shice = b6.shice
MODEL = b6.KIND
MODEL_SEED = 314159
METHODS = ("full_vqf", "batch6_mamba", "batch7_mamba")


def parameter_sha(model):
    state = hashlib.sha256()
    for name, parameter in model.named_parameters():
        values = parameter.detach().cpu().numpy()
        state.update(json.dumps([name, values.shape, values.dtype.str], separators=(",", ":")).encode("utf-8"))
        state.update(np.ascontiguousarray(values, dtype="<f4").tobytes())
    return state.hexdigest()


def statistics_sha(mean, std):
    values = np.stack((mean.detach().cpu().numpy(), std.detach().cpu().numpy()))
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def create_model():
    # Batch 6 used this same constructor after setting the model-initialization seed.
    torch.manual_seed(base.SEED)
    reconstructed = base.ResidualHead(MODEL)
    original_sha = parameter_sha(reconstructed)
    del reconstructed
    torch.manual_seed(MODEL_SEED)
    model = base.ResidualHead(MODEL)
    new_sha = parameter_sha(model)
    base.require(new_sha != original_sha, "Model initialization did not change")
    return model, {"batch6_reconstructed_sha256": original_sha, "batch7_sha256": new_sha,
                   "algorithm": "SHA-256 of ordered parameter names, shapes, dtypes and little-endian float32 values"}


def compare(old, mask, baseline, rmse, counts, old_masks, new_masks):
    key = old["mask_key"]
    np.testing.assert_array_equal(mask, old_masks[key])
    digest = b6.mask_digest(mask)
    base.require(digest == old["scoring_indices_sha256"] and int(mask.sum()) == old["frames"]
                 and baseline == old["full_vqf_deg"], f"Scoring or baseline changed for {old['sequence']}")
    base.require(all(old[label] == n for label, n in counts.items()), f"Intrinsic frames changed for {old['sequence']}")
    base.require(np.isfinite(rmse), f"Nonfinite RMSE for {old['sequence']}")
    new_masks[key] = mask
    return {**{k: old[k] for k in ("sequence", "group", "native_frames", "frames", "mask_key",
                                  "scoring_indices_sha256", "full_vqf_deg", "batch6_mamba_deg")},
            "batch7_mamba_deg": rmse, "delta_vs_full_vqf_deg": rmse - baseline,
            "delta_vs_batch6_deg": rmse - old["batch6_mamba_deg"], **counts}


def summarize(rows):
    return {"records": len(rows), "frames": sum(row["frames"] for row in rows),
            "macro_rmse_deg": {k: float(np.mean([row[f"{k}_deg"] for row in rows])) for k in METHODS},
            "wins_vs_full_vqf": sum(row["delta_vs_full_vqf_deg"] < -1e-9 for row in rows),
            "wins_vs_batch6": sum(row["delta_vs_batch6_deg"] < -1e-9 for row in rows)}


def shice_replay(model, mean, std, archived, old_masks, new_masks):
    allowed = {f"{d}-{i}" for d in shice.DEVICES for i in range(1, 11)}
    names = {row["sequence"] for row in archived}
    base.require(len(archived) == len(names) == 37 and names < allowed and not {"R1-3", "R1-4"} & names,
                 "Unexpected SHICE inventory")
    rows = []
    for old in archived:
        name = old["sequence"]
        dt, signal, reference, mask = shice.raw_record(name)
        quat, index, release, histories, _, _ = base.causal_inputs(signal, 1. / dt)
        frame, counts = b6.intrinsic_windows(histories)
        delta = b6.predict(model, histories, frame, mean, std)
        scores = b6.score_attitudes(quat, reference, mask, quat[index], release,
                                    {MODEL: delta}, {MODEL: (MODEL, 1., False)}, dt)
        row = compare(old, mask, scores["full_vqf"], scores[MODEL], counts, old_masks, new_masks)
        rows.append(row)
        print("SHICE", name, row["frames"], row["batch7_mamba_deg"], flush=True)
    return rows


def main():
    started_at = base.stamp()
    weight_path = BATCH7 / "mamba-intrinsic-seed314159-selected.pt"
    base.require(not weight_path.exists(), "Batch 7 has already fitted its one head")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    frozen = read(BATCH6 / "selection.json")
    b6_protocol = read(BATCH6 / "protocol.json")
    old_moments = read(BATCH6 / "training-frame-audit.json")
    for key, filename in (("protocol_sha256", "protocol.json"), ("training_audit_sha256", "training-frame-audit.json"),
                          ("weights_sha256", frozen["weights"])):
        base.require(base.digest(BATCH6 / filename) == frozen[key], f"Batch 6 artifact changed: {filename}")
    for files, root in ((b6_protocol["implementation_sha256"], BATCH6), (b6_protocol["source_sha256"], TRANSFER)):
        base.require(all(base.digest(root / name) == digest for name, digest in files.items()),
                     f"Batch 6 implementation or data evidence changed under {root}")
    b6_state = torch.load(BATCH6 / frozen["weights"], map_location="cpu", weights_only=True)
    base.require(b6_state["seed"] == base.SEED == 1729 and b6_state["model"] == MODEL,
                 "Unexpected primary checkpoint")
    model, init = create_model()
    model = model.to(device)
    base.require(sum(b.numel() for b in model.parameters()) == 5595, "Changed model size")
    mean, std = b6_state["mean"].to(device), b6_state["std"].to(device)
    frozen_stats_sha = statistics_sha(mean, std)
    protocol_path = BATCH7 / "protocol.json"
    protocol = {"started_at": started_at, "device": torch.cuda.get_device_name(device),
                "batch6_selection_sha256": base.digest(BATCH6 / "selection.json"),
                "batch6_weights_sha256": frozen["weights_sha256"],
                "initialization_seed": MODEL_SEED, "sampling_seed": base.SEED, "initialization": init,
                "batch_schedule_sha256": old_moments["batch_schedule_sha256"],
                "normalization_sha256": frozen_stats_sha,
                "settings": {k: v for k, v in b6_protocol["settings"].items() if k != "seed"},
                "training_scope": "One initialization replicate. The only changed seed is the network initialization.",
                "selection": b6_protocol["selection"], "gain": 1.}
    if protocol_path.exists():
        old_protocol = read(protocol_path)
        old_protocol["started_at"] = started_at
        base.require(old_protocol == protocol, "Frozen Batch 7 protocol changed")
    else:
        base.write(protocol_path, protocol)

    data = base.load(read(TRANSFER / "external-split.json"))
    train, validation = data["train"], data["validation"]
    raw, batches, schedule = base.make_batches(train, device)
    base.require(schedule == old_moments["batch_schedule_sha256"] == b6_protocol["training_schedule_sha256"]
                 and tuple(batches.shape) == (192, 96), "Sampling schedule changed")
    targets, audit = b6.aligned.aligned_targets(train, read(TRANSFER / "review/training_target_audit.json"), raw[2])
    base.require(audit == read(TRANSFER / "batch3/target-audit.json"), "Aligned targets changed")
    del raw
    arrays, (recomputed_mean, recomputed_std), moments = b6.transform_training(train, targets, device)
    moments.update({"batch_schedule_sha256": schedule, "training_windows": len(targets)})
    base.require(moments == old_moments and statistics_sha(recomputed_mean, recomputed_std) == frozen_stats_sha,
                 "Recomputed canonical normalization changed")
    np.testing.assert_array_equal(recomputed_mean.cpu().numpy(), b6_state["mean"].numpy())
    np.testing.assert_array_equal(recomputed_std.cpu().numpy(), b6_state["std"].numpy())
    del data, train, targets
    prepared = [(part, *b6.intrinsic_windows(part.histories)) for part in validation]
    print("matched batch and normalization fingerprints", schedule, frozen_stats_sha,
          "different initialization", init, flush=True)
    torch.manual_seed(MODEL_SEED)
    state, training = b6.fit(model, arrays, batches, prepared, mean, std)
    stage = weight_path.with_suffix(".tmp")
    torch.save({"model": MODEL, "state_dict": state, "mean": mean.cpu(), "std": std.cpu(), "seed": MODEL_SEED,
                "sampling_seed": base.SEED, "epoch": training["selected_epoch"], "window": base.WINDOW,
                "cap_deg": base.CAP, "split_sha256": b6_state["split_sha256"],
                "intrinsic_acc_floor": b6_state["intrinsic_acc_floor"],
                "intrinsic_gyro_floor": b6_state["intrinsic_gyro_floor"]}, stage)
    stage.replace(weight_path)
    restored = torch.load(weight_path, map_location="cpu", weights_only=True)
    base.require(restored["seed"] == MODEL_SEED and restored["sampling_seed"] == base.SEED
                 and statistics_sha(restored["mean"], restored["std"]) == frozen_stats_sha,
                 "Saved checkpoint settings changed")
    model.load_state_dict(restored["state_dict"])
    model.eval().requires_grad_(False)
    mean, std = restored["mean"].to(device), restored["std"].to(device)
    del arrays, batches
    scored = b6.external_rows(model, prepared, mean, std)
    metric = base.macro(scored, MODEL)
    selected = training["history"][training["selected_epoch"] - 1]["external_validation_macro_rmse_deg"]
    base.require(abs(metric - selected) <= 1e-6, "Saved weights failed external score reproduction")
    causality = b6.audit_causality(model, prepared, mean, std)
    selection_path = BATCH7 / "selection.json"
    selection = {"selected_at": base.stamp(), "external_selected_macro_rmse_deg": metric, "training": training,
                 "weights": weight_path.name, "weights_sha256": base.digest(weight_path),
                 "protocol_sha256": base.digest(protocol_path), "initialization": init,
                 "initialization_seed": MODEL_SEED, "sampling_seed": base.SEED,
                 "batch_schedule_sha256": schedule, "normalization_sha256": frozen_stats_sha,
                 "recomputed_normalization_sha256": statistics_sha(recomputed_mean, recomputed_std),
                 "causality": causality}
    base.write(selection_path, selection)
    print("externally selected checkpoint", training["selected_epoch"], metric, flush=True)

    previous = read(BATCH6 / "results.json")
    base.require(previous["selection_sha256"] == protocol["batch6_selection_sha256"]
                 and previous["masks_sha256"] == base.digest(BATCH6 / "scoring-masks.npz"),
                 "Changed Batch 6 selection evidence or masks")
    masks, external = {}, []
    with np.load(BATCH6 / "scoring-masks.npz", allow_pickle=False) as old_masks:
        for part, new, old in zip(validation, scored, previous["external_sequences"], strict=True):
            base.require((old["group"], old["sequence"]) == (part.domain, part.item.name), "External ordering changed")
            counts = {k: v for k, v in new.items() if k not in ("rmse_deg", "frames")}
            external.append(compare(old, part.score, new["rmse_deg"]["full_vqf"], new["rmse_deg"][MODEL],
                                    counts, old_masks, masks))
        external_path = BATCH7 / "external-sequences.csv"
        shice.save_csv(external_path, external)
        shice_started_at = base.stamp()
        shice_rows = shice_replay(model, mean, std, previous["shice_sequences"], old_masks, masks)
        base.require(set(old_masks.files) == set(masks) and len(masks) == 55, "Incomplete frame coverage")
    shice_path = BATCH7 / "shice-sequences.csv"
    shice.save_csv(shice_path, shice_rows)
    mask_path = BATCH7 / "scoring-masks.npz"
    staging = mask_path.with_suffix(".tmp")
    with staging.open("wb") as stream:
        np.savez_compressed(stream, **masks)
    staging.replace(mask_path)

    by_domain = {d: summarize([r for r in external if r["group"] == d]) for d in base.DOMAINS}
    summary = {"external_pooled": summarize(external), "external_by_domain": by_domain,
               "external_equal_domain_rmse_deg": {k: float(np.mean([v["macro_rmse_deg"][k] for v in by_domain.values()]))
                                                  for k in METHODS},
               "shice_pooled": summarize(shice_rows),
               "shice_by_device": {d: summarize([r for r in shice_rows if r["group"] == d]) for d in shice.DEVICES}}
    base.write(BATCH7 / "results.json", {**selection, "started_at": started_at, "shice_started_at": shice_started_at,
               "completed_at": base.stamp(), "selection_sha256": base.digest(selection_path),
               "batch6_results_sha256": base.digest(BATCH6 / "results.json"), "masks_sha256": base.digest(mask_path),
               "external_csv_sha256": base.digest(external_path), "shice_csv_sha256": base.digest(shice_path),
               "summary": summary, "external_sequences": external, "shice_sequences": shice_rows})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
