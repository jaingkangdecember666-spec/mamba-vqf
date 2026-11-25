"""Score the frozen transfer models once on the SHICE final holdouts."""

import json
from pathlib import Path
import sys

BATCH8 = Path(__file__).resolve().parent
TRANSFER = BATCH8.parent
sys.path[:0] = [str(TRANSFER), str(TRANSFER / "batch3"), str(TRANSFER / "batch4"),
                str(TRANSFER / "batch6"), str(TRANSFER / "review")]

import fit_intrinsic as intrinsic
import numpy as np
import torch
from replay_native import features as independent_features
from target_raw import source_record as independent_raw
from tilt_eval import normalize, tilt_error


base = intrinsic.original
shice = intrinsic.shice
read = intrinsic.aligned.read
LOCK_SHA = "411530d332658b0d0c4c68fa9ec4494fde8cf126a68fbb1d1d66d9d3c282db69"
MODELS = ("strict_mamba", "strict_tcn", "transfer_mamba", "repeat_mamba")
METHODS = ("full_vqf", *MODELS)
COLUMNS = ("sequence", "device", "frames", "scoring_indices_sha256", *METHODS)
HOLDOUTS = ("R5", "R6")


def verify_lock():
    path = TRANSFER / "review/final-transfer-lock.json"
    base.require(base.digest(path) == LOCK_SHA, "Final transfer lock SHA-256 mismatch")
    locked = read(path)
    base.require(tuple(locked["models"]) == MODELS and locked["heldout_prediction_files_seen_in_current_campaign"] is False,
                 "Unexpected lock model scope")
    expected = {**locked["evidence"], **{item["path"]: item["sha256"] for item in locked["models"].values()}}
    actual = {name: base.digest(TRANSFER / name) for name in expected}
    base.require(actual == expected, f"Locked evidence mismatch: {[k for k in expected if actual[k] != expected[k]]}")
    base.require(actual["external-split.json"] == locked["source_split_sha256"], "Locked external split changed")
    return locked, actual


def load_model_set(locked, device):
    protocol = read(TRANSFER / "batch6/protocol.json")
    for name in ("train_pair.py", "models.py", "batch4/replay_shice.py",
                 ".deps/transformers/models/mamba/modeling_mamba.py"):
        base.require(base.digest(TRANSFER / name) == protocol["source_sha256"][name], f"Inference code changed: {name}")
    models = {}
    for label, meta in locked["models"].items():
        saved = torch.load(TRANSFER / meta["path"], weights_only=True, map_location="cpu")
        base.require(all(saved[k] == meta[k] for k in ("epoch", "seed")) and saved["model"] == meta["kind"]
                     and saved["split_sha256"] == locked["source_split_sha256"]
                     and saved["window"] == base.WINDOW and saved["cap_deg"] == base.CAP
                     and meta["gain"] == 1., f"Checkpoint metadata changed: {label}")
        if meta["intrinsic"]:
            base.require(saved["intrinsic_acc_floor"] == intrinsic.ACC_FLOOR
                         and saved["intrinsic_gyro_floor"] == intrinsic.GYRO_FLOOR, "Intrinsic fallback settings changed")
        mean, std = saved["mean"].to(device), saved["std"].to(device)
        base.require(mean.shape == std.shape == (13,) and bool(torch.isfinite(mean).all())
                     and bool(torch.isfinite(std).all()) and bool((std > 0).all()), f"Invalid normalization: {label}")
        model = base.ResidualHead(meta["kind"])
        model.load_state_dict(saved["state_dict"])
        models[label] = (model.eval().requires_grad_(False).to(device), mean, std)
    return models


def merge_development(locked, archive, mask_bundle):
    b4, b6, b7 = (read(TRANSFER / f"batch{i}/results.json") for i in (4, 6, 7))
    protocol = read(TRANSFER / "batch4/protocol.json")
    base.require(base.digest(TRANSFER / "batch4/protocol.json") == b4["protocol_sha256"], "Batch 4 protocol changed")
    models = locked["models"]
    for original, label in (("mamba_v1", "strict_mamba"), ("tcn", "strict_tcn")):
        base.require(protocol["checkpoints"][original]["sha256"] == models[label]["sha256"],
                     f"Archived strict model differs from lock: {label}")
    base.require(b6["weights_sha256"] == models["transfer_mamba"]["sha256"]
                 and b7["weights_sha256"] == models["repeat_mamba"]["sha256"], "Archived intrinsic weights differ from lock")
    maps = [{row["sequence"]: row for row in seq} for seq in
            (b4["sequences"], b6["shice_sequences"], b7["shice_sequences"])]
    base.require(len(maps[0]) == 37 and all(set(m) == set(maps[0]) for m in maps),
                 "Development record coverage differs across batches")
    b6_path, b7_path = (TRANSFER / f"batch{i}/scoring-masks.npz" for i in (6, 7))
    base.require(base.digest(b6_path) == b6["masks_sha256"] and base.digest(b7_path) == b7["masks_sha256"],
                 "Archived mask files changed")
    rows = []
    with np.load(b6_path, allow_pickle=False) as masks6, np.load(b7_path, allow_pickle=False) as masks7:
        for name, strict in maps[0].items():
            transfer, repeat = (m[name] for m in maps[1:])
            frames = strict["frames"]
            digest = strict["scoring_indices_sha256"]
            baseline = strict["rmse_deg"]["full_vqf"]
            key = transfer["mask_key"]
            original = masks6[key]
            np.testing.assert_array_equal(original, masks7[repeat["mask_key"]])
            base.require((frames, digest, baseline) == (transfer["frames"], transfer["scoring_indices_sha256"],
                         transfer["full_vqf_deg"]) == (repeat["frames"], repeat["scoring_indices_sha256"],
                         repeat["full_vqf_deg"]) and int(original.sum()) == frames
                         and intrinsic.mask_digest(original) == digest and strict["same_score_mask_as_archived"],
                         f"Development baseline or mask mismatch: {name}")
            base.require(transfer["batch3_mamba_deg"] == strict["rmse_deg"]["mamba_v1"]
                         and repeat["batch6_mamba_deg"] == transfer["batch6_mamba_deg"],
                         f"Archived score lineage mismatch: {name}")
            mask_bundle[name] = original
            archive[name] = {"source": "batch4/6/7", "native_frames": len(original),
                             "same_archived_mask": True, "same_baseline": True}
            rows.append({"sequence": name, "device": strict["device"], "frames": frames,
                         "scoring_indices_sha256": digest, "full_vqf": baseline,
                         "strict_mamba": strict["rmse_deg"]["mamba_v1"], "strict_tcn": strict["rmse_deg"]["tcn"],
                         "transfer_mamba": transfer["batch6_mamba_deg"], "repeat_mamba": repeat["batch7_mamba_deg"]})
    return rows


@torch.inference_mode()
def strict_predict(model, histories, up, mean, std):
    pieces = []
    for first in range(0, len(histories), 256):
        subset = slice(first, first + 256)
        x = torch.from_numpy(histories[subset]).to(mean.device)
        gravity = torch.from_numpy(up[subset]).to(mean.device)
        pieces.append(base.tangent(model(base.normalized(x, mean, std)), gravity, base.CAP).cpu().numpy())
    return np.concatenate(pieces)


def independent_check(name, signal, dt, reference, mask, index, release, histories, deltas, scores):
    raw_signal, hz, body_ref, independent_mask = independent_raw(name)
    np.testing.assert_array_equal(mask, independent_mask)
    np.testing.assert_array_equal(signal, raw_signal)
    base.require(hz == 1. / dt, f"Independent timestamp mismatch for {name}")
    np.testing.assert_allclose(reference[mask], body_ref[mask], rtol=0, atol=5e-8)
    native, windows, last, ready = independent_features(raw_signal, hz)
    np.testing.assert_array_equal(last, index)
    np.testing.assert_array_equal(ready, release)
    np.testing.assert_allclose(windows, histories, rtol=1e-7, atol=1e-6)
    indices = np.flatnonzero(independent_mask)
    truth_world = native[indices].apply(body_ref[indices])
    z = np.array([0., 0., 1.])
    offsets = np.searchsorted(ready, indices, side="right")
    references = {"full_vqf": np.broadcast_to(z, truth_world.shape)}
    references.update({name: np.vstack((z, normalize(z + native[last].apply(delta))))[offsets]
                       for name, delta in deltas.items()})
    differences = {label: float(np.sqrt(np.mean(tilt_error(direction, truth_world)**2))) - scores[label]
                   for label, direction in references.items()}
    base.require(max(map(abs, differences.values())) < 1e-5, f"Independent scores differ for {name}: {differences}")
    return {"same_raw_imu": True, "same_mask": True, "same_update_indices": True,
            "feature_max_abs_difference": float(np.max(np.abs(windows - histories))),
            "independent_minus_primary_rmse_deg": differences}


def score_holdout(name, locked, models):
    dt, signal, reference, mask = shice.raw_record(name)
    with np.load(shice.previous.CACHE / f"{name}.npz", allow_pickle=False) as cached:
        np.testing.assert_array_equal(mask, cached["score"])
        np.testing.assert_array_equal(reference[mask], cached["ref"][mask])
        np.testing.assert_array_equal(signal.astype(np.float32), cached["imu"])
        base.require(dt == float(cached["dt"]), f"Historical SHICE sample interval differs: {name}")
    quat, index, release, histories, _, _ = base.causal_inputs(signal, 1. / dt)
    up = base.body_up(quat[index], base.EST_UP).astype(np.float32)
    frame, count = intrinsic.intrinsic_windows(histories)
    deltas = {}
    for label, (model, mean, std) in models.items():
        if locked["models"][label]["intrinsic"]:
            deltas[label] = intrinsic.predict(model, histories, frame, mean, std)
        else:
            deltas[label] = strict_predict(model, histories, up, mean, std)
        base.require(np.isfinite(deltas[label]).all(), f"Nonfinite delta for {name}/{label}")
    scores = intrinsic.score_attitudes(quat, reference, mask, quat[index], release, deltas,
                                       {k: (k, 1., False) for k in MODELS}, dt)
    base.require(all(np.isfinite(list(scores.values()))), f"Nonfinite RMSE for {name}")
    row = {"sequence": name, "device": name.split("-")[0], "frames": int(mask.sum()),
           "scoring_indices_sha256": intrinsic.mask_digest(mask), **scores}
    audit = {"source": "raw CSV and SHICE native audit masks", "native_frames": len(mask),
             "same_archived_mask": True, "same_archived_reference_on_scored_frames": True,
             "same_archived_imu_at_float32": True, "updates": len(release), "intrinsic_frame_audit": count}
    if name in locked["independent_raw_checks"]:
        audit["independent_raw_check"] = independent_check(name, signal, dt, reference, mask, index,
                                                          release, histories, deltas, scores)
    return row, mask, audit


def summarize(rows):
    scores = {method: float(np.mean([r[method] for r in rows])) for method in METHODS} if rows else {}
    return {"records": len(rows), "scoring_frames": sum(r["frames"] for r in rows), "macro_rmse_deg": scores,
            "wins_vs_full_vqf": {method: sum(r[method] < r["full_vqf"] - 1e-9 for r in rows) for method in MODELS},
            "worse_than_full_vqf": {method: sum(r[method] > r["full_vqf"] + 1e-9 for r in rows) for method in MODELS}}


def main():
    started = base.stamp()
    results_path = BATCH8 / "results.json"
    protocol_path = BATCH8 / "protocol.json"
    base.require(not results_path.exists() and not protocol_path.exists(), "Batch 8 already started")
    locked, verified = verify_lock()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    models = load_model_set(locked, device)
    audits, masks = {}, {}
    rows = merge_development(locked, audits, masks)
    heldout = [f"{device_name}-{i}" for device_name in HOLDOUTS for i in range(1, 11)]
    base.require(set(locked["independent_raw_checks"]) <= set(heldout), "Independent probes are outside the final holdouts")
    base.write(protocol_path, {"started_at": started, "lock_sha256": LOCK_SHA, "verified_evidence_and_weights": verified,
                               "device": torch.cuda.get_device_name(device), "optimizer_steps": 0,
                               "heldout_records": heldout, "inverse_reference_world_axis": shice.REF_UP.tolist(),
                               "development_sources": ["batch4/results.json", "batch6/results.json", "batch7/results.json"],
                               "models": locked["models"], "raw_audit_records": locked["independent_raw_checks"]})
    print("verified final lock, four checkpoints and 37 merged records", flush=True)
    failed = []
    for name in heldout:
        try:
            row, mask, audit = score_holdout(name, locked, models)
        except (OSError, RuntimeError, ValueError, KeyError, AssertionError) as exc:
            failed.append({"sequence": name, "type": type(exc).__name__, "message": str(exc)})
            print("heldout scoring failed", name, repr(exc), flush=True)
            continue
        rows.append(row)
        audits[name] = audit
        masks[name] = mask
        print(name, row["frames"], {k: row[k] for k in METHODS}, flush=True)
    _, post_run_hashes = verify_lock()
    base.require(post_run_hashes == verified, "Locked files changed during final evaluation")
    base.require(len(rows) + len(failed) == 57 and len(masks) == len(rows), "Unexpected record coverage")
    result_csv = BATCH8 / "all-sequences.csv"
    output = {row["sequence"]: row for row in rows}
    for failure in failed:
        name = failure["sequence"]
        output[name] = {k: name if k == "sequence" else name.split("-")[0] if k == "device" else "" for k in COLUMNS}
    names = [*list(audits)[:37], *heldout]
    shice.save_csv(result_csv, [{key: output[name][key] for key in COLUMNS} for name in names])
    mask_path = BATCH8 / "scoring-masks.npz"
    temp = mask_path.with_suffix(".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **masks)
    temp.replace(mask_path)
    by_device = {dev: summarize([r for r in rows if r["device"] == dev]) for dev in (*shice.DEVICES, *HOLDOUTS)}
    selection = {"by_device": by_device, "R5+R6": summarize([r for r in rows if r["device"] in HOLDOUTS]),
                 "R1–R4": summarize([r for r in rows if r["device"] in shice.DEVICES]), "all_57": summarize(rows)}
    base.write(results_path, {"lock_sha256": LOCK_SHA, "started_at": started, "completed_at": base.stamp(),
                             "optimizer_steps": 0, "verified_evidence_and_weights": verified,
                             "protocol_sha256": base.digest(protocol_path), "csv_sha256": base.digest(result_csv),
                             "masks_sha256": base.digest(mask_path), "summary": selection, "failures": failed,
                             "heldout_records_scored": len(heldout) - len(failed), "development_records_merged": 37,
                             "same_archived_frame_indices_records": sum(r["same_archived_mask"] for r in audits.values()),
                             "mask_audit": audits, "sequences": rows,
                             "target_exposure": locked["target_exposure"], "primary": locked["primary"]})
    print(json.dumps({"holdout": selection["R5+R6"], "all": selection["all_57"], "failures": failed}, indent=2), flush=True)


if __name__ == "__main__":
    main()
