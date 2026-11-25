"""Evaluate the frozen SHICE correction heads on BROAD and Sassari."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.io import loadmat
from vqf import VQF

from run_pilot import CHECKPOINT_REVISION, body_up
from tilt_data import ARTIFACTS, CONTEXT, EST_UP, HOP_INPUT, WARMUP_S, causal_fill
from tilt_eval import load_head, macro_scores, predict_updates, save_json, score_attitudes
from tilt_model import contexts, encode, load_backbone, physical_features


SASSARI_SENSORS = ("XS1", "XS2", "AP1", "AP2", "SH1", "SH2")
FREQUENCY = 100


@dataclass(frozen=True)
class Recording:
    name: str
    group: str
    imu: np.ndarray
    quaternion: np.ndarray
    eligible: np.ndarray
    frequency: float


def broad_records(root):
    folder = root / "broad-main" / "broad-main" / "data_mat"
    files = sorted(folder.glob("*.mat"))
    trials = json.loads((folder / "trials.json").read_text(encoding="utf-8"))["trials"]
    if len(files) != 39 or set(trials) != {path.stem for path in files}:
        raise ValueError("BROAD MAT inventory does not match the 39 trial definitions")
    for path in files:
        data = loadmat(path, variable_names=("imu_acc", "imu_gyr", "opt_quat", "movement", "sampling_rate"))
        principal_groups = ("rotation", "translation", "combined")
        group = next((name for name in principal_groups if name in trials[path.stem]["groups"]), "disturbed")
        yield Recording(path.stem, group, np.column_stack((data["imu_acc"], data["imu_gyr"])),
                        data["opt_quat"], data["movement"].ravel().astype(bool), float(data["sampling_rate"].item()))


def sassari_records(root):
    folder = root / "Sassari"
    for speed in ("slow", "medium", "fast"):
        data = loadmat(folder / f"{speed}_v5.mat", variable_names=("Qs", *SASSARI_SENSORS))
        truth = data["Qs"]
        for sensor in SASSARI_SENSORS:
            yield Recording(f"{speed}/{sensor}", speed, data[sensor][:, 1:7], truth,
                            np.ones(len(truth), dtype=bool), 100.)


def input_blocks(imu, source_frequency):
    """Average native samples available before each completed 10 ms boundary."""
    num_blocks = int(np.floor(len(imu) * FREQUENCY / source_frequency + 1e-9))
    block_edges = np.arange(num_blocks + 1) * (source_frequency / FREQUENCY)
    stops = np.floor(block_edges + 1e-9).astype(np.int64)
    if np.any(np.diff(stops) < 1):
        raise ValueError("Every causal input block must contain native observations")
    summed = np.vstack((np.zeros((1, 6)), np.cumsum(imu, axis=0, dtype=np.float64)))
    features = (summed[stops[1:]] - summed[stops[:-1]]) / np.diff(stops)[:, None]
    last_observation = stops[1:] - 1
    available_at = np.ceil(block_edges[1:] - 1e-9).astype(np.int64) - 1
    return features.astype(np.float32), last_observation, available_at


def reference_direction(quaternions):
    valid = np.isfinite(quaternions).all(axis=1) & (np.linalg.norm(quaternions, axis=1) > 0)
    up = np.zeros((len(quaternions), 3), dtype=np.float64)
    up[valid] = body_up(quaternions[valid], EST_UP)
    return up, valid


def score_record(record, model, heads, candidates, device, batch_size, shift_check):
    n = len(record.eligible)
    if record.imu.shape != (n, 6) or record.quaternion.shape != (n, 4) or record.frequency < FREQUENCY:
        raise ValueError(f"Unexpected source schema in {record.name}")
    valid_imu = np.isfinite(record.imu).all(axis=1)
    imu = causal_fill(record.imu, valid_imu)
    dt = 1. / record.frequency
    filter_ = VQF(dt)
    result = filter_.updateBatch(np.ascontiguousarray(imu[:, 3:], dtype=np.float64),
                                np.ascontiguousarray(imu[:, :3], dtype=np.float64))
    base = result["quat6D"]
    windows, state_at, available_at = input_blocks(imu, record.frequency)
    first_update = ((CONTEXT + HOP_INPUT - 1) // HOP_INPUT) * HOP_INPUT - 1
    blocks = np.arange(first_update, len(windows), HOP_INPUT)
    if not len(blocks):
        raise ValueError(f"No correction update in {record.name}")
    update_rows = state_at[blocks]
    scheduled = available_at[blocks]
    history = contexts(windows, blocks)
    physical = physical_features(history, body_up(base[update_rows], EST_UP).astype(np.float32),
                                 result["bias"][update_rows].astype(np.float32),
                                 result["restDetected"][update_rows].astype(np.float32))
    features = None
    if any(head.embedding_width for head in heads.values()):
        features = np.concatenate([encode(model, batch, device).cpu().numpy()
                                   for batch in np.array_split(history, (len(history) + batch_size - 1) // batch_size)])
        if not np.isfinite(features).all():
            raise ValueError(f"Invalid TimesFM output in {record.name}")
    deltas = {method: predict_updates(head, physical, features if head.embedding_width else None, device)
              for method, head in heads.items()}
    if not all(np.isfinite(delta).all() for delta in deltas.values()):
        raise ValueError(f"Invalid model corrections in {record.name}")

    def score(reference, mask):
        return {"scoring_frames": int(mask.sum()),
                "rmse_deg": score_attitudes(base, reference, mask, base[update_rows], scheduled,
                                            deltas, candidates, dt)}

    reference, ref_valid = reference_direction(record.quaternion)
    mask = valid_imu & ref_valid & record.eligible
    warm = mask & (np.arange(n) * dt >= WARMUP_S)
    protocols = {"main": score(reference, mask)}
    if warm.any():
        protocols["after_10s"] = score(reference, warm)
    if shift_check:
        shifted_ref = np.vstack((reference[1:], np.zeros((1, 3))))
        shifted_valid = np.r_[ref_valid[1:], False]
        overlap = mask & shifted_valid
        protocols["same_index_matched"] = score(reference, overlap)
        protocols["reference_plus_one_sample"] = score(shifted_ref, overlap)
    return {"recording": record.name, "group": record.group, "input_hz": record.frequency,
            "raw_frames": n, "invalid_imu_frames": int((~valid_imu).sum()), "updates": len(blocks),
            "protocols": protocols}


def aggregate(records, protocol):
    scores = [row["protocols"][protocol] for row in records if protocol in row["protocols"]]
    if not scores:
        return None
    return {"sequences": len(scores), "scoring_frames": sum(row["scoring_frames"] for row in scores),
            "macro_rmse_deg": macro_scores(scores),
            "improved_vs_vqf": {method: sum(row["rmse_deg"][method] < row["rmse_deg"]["full_vqf"] for row in scores)
                                 for method in ("timesfm", "physical_only")},
            "lightweight_better_than_timesfm": sum(row["rmse_deg"]["physical_only"] < row["rmse_deg"]["timesfm"]
                                                   for row in scores)}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/数据集"))
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint", type=Path, default=ARTIFACTS.parent / "checkpoint")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    started = perf_counter()
    validation_file = args.artifacts / "validation.json"
    selected = json.loads(validation_file.read_text(encoding="utf-8"))
    gains, epochs = selected["gains"], selected["best_epochs"]
    candidates = {name: (name, gain, False) for name, gain in gains.items()}
    hashes = {validation_file.name: sha256(validation_file)}
    heads = {}
    for method in ("timesfm", "physical_only"):
        path = args.artifacts / f"{method}_head.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["selected_gain"] != gains[method] or checkpoint["selected_epoch"] != epochs[method]:
            raise ValueError(f"{path.name} does not match the R5 selection")
        if method == "timesfm" and checkpoint["backbone_revision"] != CHECKPOINT_REVISION:
            raise ValueError("TimesFM head and backbone revision do not match")
        hashes[path.name] = sha256(path)
        heads[method] = load_head(path, device)
    backbone = load_backbone(args.checkpoint, device)
    datasets = {}
    for name, source in (("BROAD", broad_records), ("Sassari", sassari_records)):
        records = []
        for item in source(args.root):
            result = score_record(item, backbone, heads, candidates, device, args.batch_size, name == "Sassari")
            records.append(result)
            print(f"{name} {item.name}: {result['protocols']['main']['rmse_deg']}", flush=True)
        protocols = sorted({key for row in records for key in row["protocols"]})
        group_names = sorted({row["group"] for row in records})
        datasets[name] = {"protocols": {key: aggregate(records, key) for key in protocols},
                          "by_group": {key: aggregate([row for row in records if row["group"] == key], "main")
                                       for key in group_names}, "records": records}
    output = {"backbone_revision": CHECKPOINT_REVISION, "checkpoint_sha256": hashes,
              "r5_gains": gains, "selected_epochs": epochs, "context_input_samples": CONTEXT,
              "input_hz": FREQUENCY, "correction_hz": FREQUENCY / HOP_INPUT,
              "time_s": perf_counter() - started, "datasets": datasets}
    save_json(args.artifacts / "external_test.json", output)
    print(json.dumps({key: value["protocols"] for key, value in datasets.items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
