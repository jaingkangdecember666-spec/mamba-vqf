"""Compare frozen forecasting features on the same causal IMU update stream."""

import os
import sys
from time import perf_counter, perf_counter_ns

import numpy as np
import torch

from common import BATCH, HISTORY, PILOT, digest, read_json, write_json
from features import block_indices, fetch_record, physics
from optimizer_study import schedule
from tilt_model import correction_quaternions, multiply, normalize


STAGE = BATCH / "forecast-backbone-development-protocol.json"
ENGINEERING = BATCH / "forecast-backbone-engineering.json"
TIMESFM_DECODE = BATCH / "forecast-backbone-timesfm-official-api.json"
TIMESTRAL_SOURCE = BATCH / "timestral-upstream" / "src"
CHECKPOINTS = BATCH / "forecast-checkpoints"
CACHE_ROOT = BATCH / "forecast-development-cache"
PHYSICAL = "physical_zero"
TIMESFM = "timesfm_3"
PRIMARY = "timestral_209m"
FALLBACK = "timestral_28m_fallback"
HISTORY_BLOCKS = 64
HORIZON = 8
STEP_INDEX = [0, 3, 7]
FEATURES = 18
TOTAL_WINDOWS = 87612
PARITY_ABS_LIMIT = 0.002
BATCH_OPTIONS = (16, 32, 64, 128, 256)


def local_model(model_key, device, stage):
    if model_key == PHYSICAL:
        return None, {}
    spec = stage["models"][model_key]
    if model_key == TIMESFM:
        from timesfm3 import TimesFM3Forecaster

        root = PILOT / "checkpoint"
        for filename, key in (("config.json", "config_sha256"), ("model.safetensors", "weights_sha256")):
            if digest(root / filename) != spec[key]:
                raise ValueError(f"Changed TimesFM file: {filename}")
        model = TimesFM3Forecaster.from_pretrained(str(root), device=str(device), local_files_only=True,
                                                   per_core_batch_size=max(BATCH_OPTIONS))
        model.model.requires_grad_(False)
        if not model.model.use_variate_attention:
            raise ValueError("Expected a joint six-variate checkpoint")
        return model, spec
    path = CHECKPOINTS / model_key
    manifest = read_json(path / "verified.json")
    if manifest["revision"] != spec["revision"] or manifest["source"] != spec["repo"]:
        raise ValueError("Wrong Timestral snapshot")
    for name, metadata in manifest["files"].items():
        if digest(path / name) != metadata["sha256"]:
            raise ValueError(f"Changed Timestral file: {name}")
    if manifest["files"]["model.safetensors"]["sha256"] != spec["weights_sha256"]:
        raise ValueError("Timestral weights differ from official file hash")
    sys.path.insert(0, str(TIMESTRAL_SOURCE))
    from timestral import TimestralForecaster

    model = TimestralForecaster.from_pretrained(path, device=device, local_files_only=True)
    model.model.requires_grad_(False)
    return model, {**spec, "local_files": manifest["files"]}


@torch.inference_mode()
def forecast_features(model_key, model, histories, device):
    if model_key == PHYSICAL:
        return np.zeros((len(histories), FEATURES), dtype=np.float32)
    if histories.shape[1:] != (HISTORY_BLOCKS, 6):
        raise ValueError(f"Wrong six-axis context shape: {histories.shape}")
    if model_key == TIMESFM:
        contexts = np.ascontiguousarray(histories.transpose(0, 2, 1), dtype=np.float32)
        outputs = model.predict_batch(list(contexts), horizon=HORIZON, use_symmetric_averaging=False,
                                      make_positive=False, sort_quantiles=True, use_znorm=False)
        median = np.stack([output.forecast for output in outputs])
        result = (median[:, :, STEP_INDEX] - contexts[:, :, -1:]).transpose(0, 2, 1).reshape(-1, FEATURES)
    else:
        values = torch.as_tensor(histories, dtype=torch.float32, device=device).transpose(1, 2).contiguous()
        median = model.forecast(values, prediction_length=HORIZON,
                                symmetric_averaging=False, clip_nonnegative=False).median
        result = (median[:, :, STEP_INDEX] - values[:, :, -1:]).transpose(1, 2).flatten(1).cpu().numpy()
    if tuple(median.shape) != (len(histories), 6, HORIZON):
        raise ValueError(f"Wrong median shape: {tuple(median.shape)}")
    if not np.isfinite(result).all():
        raise ValueError(f"Nonfinite {model_key} feature")
    return result


def contexts_for(inputs, updates):
    return inputs[updates[:, None] - np.arange(HISTORY_BLOCKS - 1, -1, -1)]


def checked_record(name, stage):
    if name not in stage["development_records"]:
        raise ValueError(f"Nondevelopment record: {name}")
    cached = fetch_record(name, stage["readout"]["parameters"])
    updates = block_indices(len(cached["windows"]), stage["release_rate_hz"])
    native_indices = cached["ready"][updates]
    with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(updates, archive["update_blocks"])
        np.testing.assert_array_equal(native_indices, archive["updates"])
        np.testing.assert_array_equal(cached["index"], np.flatnonzero(archive["score"]))
    return cached, updates, native_indices


def minmax_latency(values):
    return dict(zip(("p50_ms", "p95_ms", "max_ms"), np.percentile(values, (50, 95, 100)).tolist(), strict=True))


def prefix_probe(model_key, model, device):
    t = np.arange(240, dtype=np.float32) / 100
    signal = np.column_stack((np.sin(t), np.cos(2*t), 9.8-.2*t, .1*np.sin(3*t), .2*t, -.3*t)).astype(np.float32)
    truncated = signal[:150].copy()
    replaced = signal.copy()
    replaced[150:] *= -97.
    slots = block_indices(len(truncated), 10)
    expected_windows = contexts_for(truncated, slots)
    original_windows = contexts_for(signal, slots)
    modified_windows = contexts_for(replaced, slots)
    np.testing.assert_array_equal(expected_windows, original_windows)
    np.testing.assert_array_equal(expected_windows, modified_windows)
    reference = forecast_features(model_key, model, expected_windows, device)
    np.testing.assert_array_equal(reference, forecast_features(model_key, model, original_windows, device))
    np.testing.assert_array_equal(reference, forecast_features(model_key, model, modified_windows, device))
    check = {"synthetic_prefix_releases": len(slots), "future_mutation_preserved_features_exactly": True}
    if model_key == TIMESFM:
        raw = np.ascontiguousarray(expected_windows[-1].T, dtype=np.float32)
        prediction = model.predict(raw, horizon=HORIZON).forecast[:, STEP_INDEX].T
        wrapped = (prediction - expected_windows[-1, -1]).reshape(FEATURES)
        direct = forecast_features(model_key, model, expected_windows[-1:], device)[0]
        gap = float(np.max(np.abs(wrapped - direct)))
        if gap > PARITY_ABS_LIMIT:
            raise ValueError(f"Official TimesFM predictor discrepancy: {gap}")
        check["official_predictor_max_abs_delta"] = gap
    return check


def time_composite(model_key, model, cached, updates, stage, device):
    import learning

    warmup = stage["latency_warmups"]
    ids = np.linspace(0, len(updates) - 1, stage["latency_samples"] + warmup, dtype=int)
    config = stage["readout"]
    _, matrix, _, _ = physics(cached, config)
    width = matrix.shape[1] + FEATURES
    rng = np.random.default_rng(10)
    state = {"mean": np.zeros(width, dtype=np.float32), "std": np.ones(width, dtype=np.float32),
             "coef": rng.standard_normal((width + 1, 3), dtype=np.float32) * 0.001}
    times, singles = [], []
    series = cached["windows"]
    for index, which in enumerate(ids):
        start = perf_counter_ns()
        end = updates[which] + 1
        tail = slice(end - 70, end)
        history = series[end - HISTORY_BLOCKS:end][None]
        extra = forecast_features(model_key, model, history, device)
        online = {key: cached[key][tail] for key in ("windows", "up", "bias", "rest")}
        _, features, accelerometer, gyroscope = physics(online, config)
        fixed_input = {"matrix": np.concatenate((features, extra), axis=1), "up": online["up"][-1:],
                       "acc": accelerometer, "gyr": gyroscope}
        delta = learning.delta_at_updates(fixed_input, state, config)
        base = cached["q"][end - 1]
        corrected = normalize(multiply(correction_quaternions(base, delta[0]), base))
        elapsed = (perf_counter_ns() - start) / 1e6
        if not np.isfinite(corrected).all():
            raise ValueError("Nonfinite attitude output")
        np.testing.assert_allclose(features[0], matrix[which], rtol=1e-4, atol=2e-6)
        if index >= warmup:
            times.append(elapsed)
        singles.append(extra[0])
    return ids, np.stack(singles), times


def batch_profiles(model_key, model, sample, device):
    rows = []
    for size in BATCH_OPTIONS:
        sample_ids = np.arange(size) % len(sample)
        repeated = sample[sample_ids]
        torch.cuda.reset_peak_memory_stats()
        try:
            forecast_features(model_key, model, repeated, device)
            started = perf_counter()
            for _ in range(4):
                forecast_features(model_key, model, repeated, device)
            seconds = (perf_counter() - started) / 4
            rows.append({"batch": size, "seconds_per_batch": seconds, "windows_per_second": size / seconds,
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2)})
        except torch.OutOfMemoryError as error:
            rows.append({"batch": size, "memory_error": str(error)})
            torch.cuda.empty_cache()
            break
    return rows


def cache_file(model_key, name):
    return CACHE_ROOT / model_key / f"{name}.npz"


def extract_one(model_key, model, name, stage, batch, protocol_digest, device):
    cached, updates, indices = checked_record(name, stage)
    target = cache_file(model_key, name)
    if target.exists():
        load_features(model_key, name, indices, protocol_digest)
        return {"record": name, "windows": len(updates), "new_windows": 0, "cache": str(target.relative_to(BATCH)),
                "sha256": digest(target)}
    start = perf_counter()
    delta = np.empty((len(updates), FEATURES), dtype=np.float32)
    for first in range(0, len(updates), batch):
        chunk = updates[first:first + batch]
        delta[first:first + len(chunk)] = forecast_features(model_key, model, contexts_for(cached["windows"], chunk), device)
    temp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    with temp.open("wb") as stream:
        np.savez(stream, features=delta, indices=indices, protocol=np.asarray(protocol_digest),
                 model=np.asarray(model_key))
    os.replace(temp, target)
    elapsed = perf_counter() - start
    return {"record": name, "windows": len(updates), "new_windows": len(updates),
            "cache": str(target.relative_to(BATCH)), "sha256": digest(target), "seconds": elapsed,
            "batch": batch, "windows_per_second": len(updates) / elapsed}


def load_features(model_key, name, indices, protocol_digest):
    if model_key == PHYSICAL:
        return np.zeros((len(indices), FEATURES), dtype=np.float32)
    with np.load(cache_file(model_key, name), allow_pickle=False) as data:
        features = data["features"]
        np.testing.assert_array_equal(data["indices"], indices)
        if data["model"].item() != model_key or data["protocol"].item() != protocol_digest:
            raise ValueError(f"Wrong forecast cache identity: {model_key} {name}")
    if features.shape != (len(indices), FEATURES) or features.dtype != np.float32 or not np.isfinite(features).all():
        raise ValueError(f"Invalid forecast features: {model_key} {name}")
    return features


def probe(model_key, model, source, config, stage, device, destination):
    checks = prefix_probe(model_key, model, device)
    name = stage["release_record"]
    cached, updates, indices = checked_record(name, stage)
    ids, singles, time_ms = time_composite(model_key, model, cached, updates, stage, device)
    history = contexts_for(cached["windows"], updates[ids])
    profiles = batch_profiles(model_key, model, history, device) if model_key != PHYSICAL else []
    batch = max((row for row in profiles if "windows_per_second" in row),
                key=lambda row: row["windows_per_second"], default={"batch": 1})["batch"]
    batched = np.concatenate([forecast_features(model_key, model, part, device)
                              for part in np.array_split(history, (len(history) + batch - 1) // batch)])
    error = np.max(np.abs(batched - singles), axis=0)
    if float(error.max()) > PARITY_ABS_LIMIT:
        raise ValueError(f"Batch and single-window forecast features disagree: {error.tolist()}")
    latency = minmax_latency(time_ms)
    print(model_key, "preflight complete", latency, "batch", batch, "parity", float(error.max()), flush=True)
    extract = extract_one(model_key, model, name, stage, batch, config["protocol_sha256"], device) if model_key != PHYSICAL else None
    report = {"model": model_key, "weights": source, "host": os.environ["COMPUTERNAME"], "worker_pid": os.getpid(),
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "prefix_check": checks,
              "timing_record": name, "timed_windows": stage["latency_samples"], "warmup_windows": stage["latency_warmups"],
              "native_indices_timed": indices[ids[stage["latency_warmups"]:]].tolist(),
              "composite_latency": latency, "samples_ms": time_ms, "batch_profiles": profiles,
              "batch_single_max_abs_delta_by_feature": error.tolist(), "selected_extraction_batch": batch,
              "first_full_cache": extract, "planned_windows_per_backbone": TOTAL_WINDOWS,
              "benchmark_readout": "fixed dimension-matched weights for timing only, no labeled fit"}
    write_json(destination, report)


def extract(model_key, model, source, config, stage, device, destination):
    outputs = []
    for name in config["records"]:
        output = extract_one(model_key, model, name, stage, config["batch_size"], config["protocol_sha256"], device)
        outputs.append(output)
        print(model_key, output, flush=True)
        write_json(destination.parent / "cache-progress.json", {"model": model_key, "records": outputs})
    write_json(destination, {"model": model_key, "weights": source, "records": outputs,
                             "newly_extracted_windows": sum(entry["new_windows"] for entry in outputs)})


def run(job, destination):
    config = job["config"]
    stage = read_json(STAGE)
    if config["protocol_sha256"] != digest(STAGE) or config["engine_sha256"] != digest(ENGINEERING):
        raise ValueError("Changed forecast development protocol")
    if config.get("model") == TIMESFM and config.get("timesfm_decode_sha256") != digest(TIMESFM_DECODE):
        raise ValueError("TimesFM job lacks the verified public predictor decode convention")
    action = config["action"]
    if action == "paired_score":
        from forecast_backbone_score import run as score

        score(job, destination, stage)
        return
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    model_key = config["model"]
    print(model_key, action, "pid", os.getpid(), "starting", flush=True)
    model, source = local_model(model_key, device, stage)
    print(model_key, "loaded", source["revision"] if source else None, flush=True)
    operation = probe if action == "probe" else extract
    operation(model_key, model, source, config, stage, device, destination)


def queue_round(identity, action, model=None, **settings):
    config = {"kind": "forecast_backbone_diagnostic", "seed": None, "action": action,
              "protocol_sha256": digest(STAGE), "engine_sha256": digest(ENGINEERING), **settings}
    if model is not None:
        config["model"] = model
    if model == TIMESFM:
        config["timesfm_decode_sha256"] = digest(TIMESFM_DECODE)
    return schedule(identity, config, "forecast-backbone-development", None, f"Frozen forecast backbone {action}: {model}")
