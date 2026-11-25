"""Run a local, causal TimesFM 3.0 gap replay on held-out SHICE recordings."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from timesfm3 import TimesFM3Forecaster
from vqf import VQF


HERE = Path(__file__).resolve().parent
CONTEXT = 128
GAPS = (1, 5, 20)
CHANNELS = tuple(f"imu_{kind}_{axis}" for kind in ("acc", "gyr") for axis in "xyz")
QUATERNION = tuple(f"opt_quat_{axis}" for axis in "wxyz")
METHODS = ("hold", "gyro_only", "timesfm", "observed")
CHECKPOINT_REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"
CHECKPOINT_SHA256 = {
    "config.json": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
    "model.safetensors": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
}


def load_record(root, recording):
    sensor = recording.split("-")[0]
    path = root / f"{sensor}_aligned_formal" / f"{recording}.csv"
    with path.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        columns = ("time_s", *CHANNELS, *QUATERNION, "is_interpolated")
        values = np.loadtxt(
            stream, delimiter=",", quotechar='"', usecols=[header.index(key) for key in columns],
            ndmin=2,
        )
    mask_path = root / "Label_quality_audit_20261116" / recording / "screening_masks.npz"
    with np.load(mask_path, allow_pickle=False) as mask:
        np.testing.assert_array_equal(mask["sample_index"], np.arange(len(values)))
        valid = mask["observed_loss_ok"].astype(bool) & mask["context_ok"].astype(bool)
    valid &= (values[:, -1] == 0) & np.isfinite(values).all(axis=1)
    valid &= np.linalg.norm(values[:, 7:11], axis=1) > 0
    timestamps = values[:, 0]
    dt = float(np.median(np.diff(timestamps)))
    if not np.allclose(np.diff(timestamps), dt, rtol=1e-5, atol=1e-9):
        raise ValueError(f"Irregular time grid in {recording}")
    return dt, values[:, 1:7], values[:, 7:11], valid


def choose_starts(imu, valid, dt, windows_per_band):
    horizon = max(GAPS)
    starts = np.arange(max(CONTEXT, int(np.ceil(10 / dt))), len(imu) - horizon + 1, 32)
    invalid = np.r_[0, np.cumsum(~valid)]
    starts = starts[invalid[starts - CONTEXT] == invalid[starts + horizon]]
    speed = np.degrees(np.linalg.norm(imu[:, 3:], axis=1))
    rolling = np.r_[0, np.cumsum(speed)]
    recent_speed = (rolling[starts] - rolling[starts - 32]) / 32
    chosen = []
    for band, band_starts in (("below_20_deg_s", starts[recent_speed < 20]),
                              ("at_least_20_deg_s", starts[recent_speed >= 20])):
        count = min(windows_per_band, len(band_starts))
        indices = np.linspace(0, len(band_starts) - 1, count, dtype=int)
        chosen.extend((int(start), band) for start in band_starts[indices])
    return sorted(chosen)


def body_up(quaternion, world_up):
    unit = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    vector = unit[:, 1:]
    cross = np.cross(vector, world_up)
    return world_up - 2 * unit[:, :1] * cross + 2 * np.cross(vector, cross)


def tilt_error(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=1), (a * b).sum(axis=1)))


def vqf_update(filter_, values):
    acc = np.ascontiguousarray(values[:, :3], dtype=np.float64)
    gyr = np.ascontiguousarray(values[:, 3:], dtype=np.float64)
    return filter_.updateBatch(gyr, acc)["quat6D"]


def forecast(model, imu):
    context = np.ascontiguousarray(imu.T, dtype=np.float32)
    estimate = model.predict(context, horizon=max(GAPS)).forecast.T
    if estimate.shape != (max(GAPS), len(CHANNELS)) or not np.isfinite(estimate).all():
        raise ValueError(f"Invalid TimesFM forecast shape or values: {estimate.shape}")
    return estimate


def replay_record(model, root, recording, windows_per_band):
    dt, imu, encoder, valid = load_record(root, recording)
    starts = choose_starts(imu, valid, dt, windows_per_band)
    if not starts:
        raise ValueError(f"No eligible replay windows in {recording}")

    filter_ = VQF(dt)
    last = 0
    events = []
    for start, band in starts:
        vqf_update(filter_, imu[last:start])
        last = start
        state = filter_.state
        hold = np.repeat(imu[start - 1:start], max(GAPS), axis=0)
        observed = imu[start:start + max(GAPS)]
        started = perf_counter()
        predicted = forecast(model, imu[start - CONTEXT:start])
        elapsed_ms = (perf_counter() - started) * 1000

        quaternions = {}
        for name, values in (("hold", hold), ("timesfm", predicted), ("observed", observed)):
            branch = VQF(dt)
            branch.state = state
            quaternions[name] = vqf_update(branch, values)
        branch = VQF(dt)
        branch.state = state
        gyro_quaternions = []
        gyro = np.ascontiguousarray(hold[0, 3:])
        for _ in range(max(GAPS)):
            branch.updateGyr(gyro)
            gyro_quaternions.append(branch.getQuat6D())
        quaternions["gyro_only"] = np.asarray(gyro_quaternions)

        reference = body_up(encoder[start:start + max(GAPS)], np.array([0., -1., 0.]))
        up = {name: body_up(quat, np.array([0., 0., 1.])) for name, quat in quaternions.items()}
        errors = {name: tilt_error(value, reference) for name, value in up.items()}
        displacement = {name: tilt_error(value, up["observed"]) for name, value in up.items()}
        scores = {}
        for length in GAPS:
            scores[str(length)] = {
                "tilt_squared_error": {name: float(np.square(errors[name][:length]).sum()) for name in METHODS},
                "tilt_displacement_squared": {name: float(np.square(displacement[name][:length]).sum()) for name in METHODS},
                "signal_squared_error": {
                    name: np.square(values[:length] - observed[:length]).sum(axis=0).tolist()
                    for name, values in (("hold", hold), ("timesfm", predicted))
                },
            }
        events.append({"sample_index": start, "speed_band": band, "predict_ms": elapsed_ms, "scores": scores})
    return {"recording": recording, "dt_s": dt, "events": events}


def summarize(records):
    summary = {}
    for length in GAPS:
        by_record = [[event["scores"][str(length)] for event in record["events"]] for record in records]
        all_scores = [score for group in by_record for score in group]
        entry = {"windows": len(all_scores), "frames": len(all_scores) * length}
        for metric in ("tilt_squared_error", "tilt_displacement_squared"):
            entry[metric.removesuffix("_squared_error").removesuffix("_squared") + "_macro_rmse_deg"] = {
                name: float(np.mean([np.sqrt(sum(score[metric][name] for score in group) / (length * len(group)))
                                     for group in by_record]))
                for name in METHODS
            }
        entry["signal_rmse"] = {}
        for name in ("hold", "timesfm"):
            squared = np.asarray([score["signal_squared_error"][name] for score in all_scores]).sum(axis=0)
            entry["signal_rmse"][name] = {
                "acc_m_s2": float(np.sqrt(squared[:3].sum() / (3 * entry["frames"]))),
                "gyro_rad_s": float(np.sqrt(squared[3:].sum() / (3 * entry["frames"]))),
            }
        summary[str(length)] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("E:/数据集/IMU_dynamic_data_SHICE"))
    parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoint",
                        help="Directory of the pinned local TimesFM 3.0 checkpoint.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    parser.add_argument("--windows-per-band", type=int, default=8)
    parser.add_argument("--records", nargs="+", default=[f"R6-{i}" for i in range(1, 11)])
    parser.add_argument("--smoke", action="store_true", help="Load the local weights and forecast a synthetic six-channel input.")
    args = parser.parse_args()
    if args.windows_per_band < 1 or len(args.records) != len(set(args.records)):
        parser.error("Use a positive window count and unique recording IDs.")
    for file, expected_sha256 in CHECKPOINT_SHA256.items():
        path = args.checkpoint / file
        if not path.is_file():
            parser.error(f"Local checkpoint file missing: {path}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                parser.error(f"Local checkpoint differs from the pinned revision: {path}")
    torch.set_float32_matmul_precision("high")
    started = perf_counter()
    model = TimesFM3Forecaster.from_pretrained(
        str(args.checkpoint.resolve()), device=args.device, local_files_only=True, per_core_batch_size=1,
    )
    gpu = torch.cuda.get_device_name() if args.device == "cuda" else None
    metadata = {"torch": torch.__version__, "device": args.device, "gpu": gpu,
                "model_load_s": perf_counter() - started, "checkpoint_revision": CHECKPOINT_REVISION,
                "checkpoint_sha256": CHECKPOINT_SHA256["model.safetensors"]}
    timeline = np.linspace(0, 5, CONTEXT)
    synthetic = np.sin(timeline[:, None] * np.arange(1, 7))
    for _ in range(4):
        estimate = forecast(model, synthetic)
    print(json.dumps({**metadata, "smoke_shape": list(estimate.shape)}), flush=True)
    if args.smoke:
        return

    records = []
    for recording in args.records:
        result = replay_record(model, args.root, recording, args.windows_per_band)
        records.append(result)
        print(f"{recording}: {len(result['events'])} replay windows", flush=True)
    latencies = [event["predict_ms"] for record in records for event in record["events"]]
    report = {
        **metadata, "data_root": str(args.root), "context_samples": CONTEXT,
        "windows_per_speed_band": args.windows_per_band,
        "predict_latency_ms": dict(zip(("p50", "p95", "max"), np.percentile(latencies, (50, 95, 100)).tolist())),
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if gpu else None,
        "summary": summarize(records), "records": records,
    }
    staging = args.output.with_suffix(args.output.suffix + ".tmp")
    staging.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(args.output)
    print(json.dumps({"latency": report["predict_latency_ms"], "summary": report["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
