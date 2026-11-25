"""Prepare causal SHICE inputs and fixed train, validation and test partitions."""

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from vqf import VQF

from run_pilot import CHANNELS, QUATERNION, body_up, tilt_error


ROOT = Path("E:/数据集/IMU_dynamic_data_SHICE")
ARTIFACTS = Path(__file__).resolve().parent / "tilt-correction"
EXCLUDED = {"R1-3", "R1-4"}
GROUPS = {"train": ("R1", "R2", "R3", "R4"), "valid": ("R5",), "test": ("R6",)}
NATIVE_PER_INPUT = 10
CONTEXT = 64
HOP_INPUT = 10
WARMUP_S = 10
EST_UP = np.array([0., 0., 1.])
REF_UP = np.array([0., -1., 0.])


def partition(root, split):
    name = root / "Label_quality_audit_20261116" / "recording_recommendations.csv"
    with name.open(encoding="utf-8-sig", newline="") as stream:
        return [row["recording"] for row in csv.DictReader(stream)
                if row["sensor"] in GROUPS[split] and row["recording"] not in EXCLUDED]


def causal_fill(imu, packet_ok):
    """Replace only missing or invalid packets using the latest observed sample."""
    rows = np.arange(len(imu))
    previous = np.maximum.accumulate(np.where(packet_ok, rows, -1))
    actual = imu[np.maximum(previous, 0)].copy()
    actual[previous < 0] = 0
    return actual


def prepare_record(root, folder, recording):
    sensor = recording.split("-")[0]
    path = root / f"{sensor}_aligned_formal" / f"{recording}.csv"
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        keys = ("sample_index", "time_s", *CHANNELS, *QUATERNION, "is_interpolated")
        values = np.loadtxt(stream, delimiter=",", quotechar='"',
                            usecols=[header.index(key) for key in keys], ndmin=2)
    rows = len(values)
    np.testing.assert_array_equal(values[:, 0], np.arange(rows))
    sample_time = values[:, 1]
    dt = float(np.median(np.diff(sample_time)))
    if not math.isfinite(dt) or dt <= 0 or not np.allclose(np.diff(sample_time), dt, rtol=1e-5, atol=1e-9):
        raise ValueError(f"Inconsistent sampling interval for {recording}")

    mask_path = root / "Label_quality_audit_20261116" / recording / "screening_masks.npz"
    with np.load(mask_path, allow_pickle=False) as mask:
        np.testing.assert_array_equal(mask["sample_index"], np.arange(rows))
        context_ok = mask["context_ok"].astype(bool)
        observed_ok = mask["observed_loss_ok"].astype(bool)
    packet_ok = (values[:, -1] == 0) & np.isfinite(values[:, 2:8]).all(axis=1)
    imu = causal_fill(values[:, 2:8], packet_ok)
    quaternions = values[:, 8:12]
    label_ok = np.isfinite(quaternions).all(axis=1) & (np.linalg.norm(quaternions, axis=1) > 0)
    reference = np.zeros((rows, 3), dtype=np.float32)
    reference[label_ok] = body_up(quaternions[label_ok], REF_UP)

    vqf = VQF(dt)
    result = vqf.updateBatch(np.ascontiguousarray(imu[:, 3:]), np.ascontiguousarray(imu[:, :3]))
    if not np.isfinite(result["quat6D"]).all():
        raise ValueError(f"Invalid VQF output in {recording}")
    base = result["quat6D"].astype(np.float32)
    base_up = body_up(result["quat6D"], EST_UP).astype(np.float32)
    blocks = len(imu) // NATIVE_PER_INPUT
    downsampled = imu[:blocks * NATIVE_PER_INPUT].reshape(blocks, NATIVE_PER_INPUT, 6).mean(axis=1)
    phase = HOP_INPUT - 1
    first_block = math.ceil(CONTEXT / HOP_INPUT) * HOP_INPUT - 1
    update_blocks = np.arange(first_block, blocks, HOP_INPUT)
    if len(update_blocks) == 0 or first_block % HOP_INPUT != phase:
        raise ValueError(f"Insufficient input context for {recording}")
    updates = (update_blocks + 1) * NATIVE_PER_INPUT - 1
    dropped_context = np.r_[0, np.cumsum(~context_ok)]
    good_training_context = dropped_context[updates + 1] == dropped_context[updates + 1 - CONTEXT * NATIVE_PER_INPUT]
    score = (sample_time >= WARMUP_S) & context_ok & observed_ok & packet_ok & label_ok
    train_ok = score[updates] & good_training_context

    print(f"{recording} rows={rows} score={int(score.sum())} training_updates={int(train_ok.sum())}", flush=True)
    target = folder / f"{recording}.npz"
    staging = target.with_suffix(".tmp")
    with staging.open("wb") as stream:
        np.savez(stream, dt=dt, windows=downsampled.astype(np.float32), updates=updates,
                 update_blocks=update_blocks, train_ok=train_ok, reference=reference, base=base,
                 base_up=base_up, score=score, bias=result["bias"][updates].astype(np.float32),
                 rest=result["restDetected"][updates].astype(np.float32))
    staging.replace(target)
    return recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--splits", nargs="+", choices=GROUPS, default=("train", "valid"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    folder = args.artifacts / "prepared"
    folder.mkdir(parents=True, exist_ok=True)
    records = [name for split in args.splits for name in partition(args.root, split)]
    pending = [name for name in records if not (folder / f"{name}.npz").is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        jobs = [executor.submit(prepare_record, args.root, folder, name) for name in pending]
        for job in jobs:
            job.result()
    print(f"Prepared {len(records)} records ({len(pending)} new)", flush=True)


if __name__ == "__main__":
    main()
