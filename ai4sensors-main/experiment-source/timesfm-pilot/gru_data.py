"""Load SHICE IMU for the RIANN-style model without changing the audit masks."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from run_pilot import CHANNELS
from tilt_data import ARTIFACTS, GROUPS, ROOT, causal_fill, partition


OUTPUT = Path(__file__).resolve().parent / "gru-tilt"
SHICE_HZ = 500.


def prepare_record(root, prepared, cache, name):
    source = root / f"{name.split('-')[0]}_aligned_formal" / f"{name}.csv"
    with source.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        keys = ("sample_index", *CHANNELS, "is_interpolated")
        actual = np.loadtxt(stream, delimiter=",", quotechar='"',
                            usecols=[header.index(key) for key in keys], ndmin=2)
    np.testing.assert_array_equal(actual[:, 0], np.arange(len(actual)))
    packet_ok = (actual[:, -1] == 0) & np.isfinite(actual[:, 1:7]).all(axis=1)
    imu = causal_fill(actual[:, 1:7], packet_ok).astype(np.float32)
    with np.load(prepared / f"{name}.npz", allow_pickle=False) as data:
        ref, score, dt = data["reference"], data["score"], data["dt"]
    with np.load(root / "Label_quality_audit_20261116" / name / "screening_masks.npz", allow_pickle=False) as audit:
        np.testing.assert_array_equal(audit["sample_index"], actual[:, 0])
        context = audit["context_ok"].astype(bool)
    if len(imu) != len(score) or not np.all(packet_ok[score] & context[score]):
        raise ValueError(f"Data and scoring mask do not align in {name}")
    staging = cache / f"{name}.tmp"
    with staging.open("wb") as stream:
        np.savez(stream, imu=imu, ref=ref, score=score, context=context, dt=dt)
    staging.replace(cache / f"{name}.npz")
    print(f"{name}: {len(imu)} input samples, {int(score.sum())} scored", flush=True)


def causal_resample(imu, dt, rate):
    """Use only complete time bins and timestamp each result at its causal availability."""
    ratio = 1. / (dt * rate)
    if ratio < 1 - 1e-9:
        raise ValueError("The causal adapter cannot increase sampling frequency")
    bins = int(np.floor(len(imu) / ratio + 1e-9))
    edges = np.arange(bins + 1) * ratio
    stops = np.floor(edges + 1e-9).astype(np.int64)
    sums = np.vstack((np.zeros((1, 6)), np.cumsum(imu, axis=0, dtype=np.float64)))
    readings = (sums[stops[1:]] - sums[stops[:-1]]) / np.diff(stops)[:, None]
    available_at = np.ceil(edges[1:] - 1e-9).astype(np.int64) - 1
    return readings.astype(np.float32), available_at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--prepared", type=Path, default=ARTIFACTS / "prepared")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--splits", nargs="+", default=("train", "valid"), choices=GROUPS)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    cache = args.output / "data"
    cache.mkdir(parents=True, exist_ok=True)
    names = [name for split in args.splits for name in partition(args.root, split)]
    pending = [name for name in names if not (cache / f"{name}.npz").is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(prepare_record, args.root, args.prepared, cache, name) for name in pending]
        for job in jobs:
            job.result()
    print(f"Cached {len(names)} SHICE sequences ({len(pending)} new)", flush=True)


if __name__ == "__main__":
    main()
