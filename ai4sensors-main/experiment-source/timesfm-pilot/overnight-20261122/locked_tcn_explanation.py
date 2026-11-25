"""Explain and benchmark the frozen formal TCN on R6 after the review replay."""

import os
import platform
from time import perf_counter_ns

import numpy as np

from common import BATCH, END, Z, aggregate, digest, errors, fingerprint, now, read_json, record, rmse, rotate, stamp, unit, write_json
from final_eval import cached_candidate, historical_score_indices, verified_indices_digest


DESTINATION = BATCH / "locked-tcn-explanation"
METHODS = ("fast_fullvqf", "fast_plus_locked_mount", "locked_tcn")


def ensure_ready(protocol):
    observer_path = BATCH / protocol["start_gate"]["observer_path"]
    observer = read_json(observer_path)
    if observer["status"] != protocol["start_gate"]["required_status"]:
        raise ValueError(f"Review observer is {observer['status']}; timing has not started")
    if now() >= END:
        raise ValueError("The supplemental experiment window has ended")
    round_dir = BATCH / "rounds" / protocol["formal_round"]
    marker = read_json(round_dir / "round.json")
    status = read_json(BATCH / "status.json")
    if marker["status"] != "completed" or status["phase"] != "completed":
        raise ValueError("Formal experiment still owns the timing environment")
    snapshots = {
        BATCH / "selection-lock.json": protocol["selection_lock_file_sha256"],
        BATCH / protocol["weights_path"]: protocol["weights_sha256"],
        round_dir / "result.json": protocol["formal_result_sha256"],
        round_dir / "final-records-progress.json": protocol["formal_scoring_manifest_sha256"],
        BATCH / "ledger.jsonl": protocol["formal_ledger_sha256_before_supplement"],
        BATCH / "experiment-index.json": protocol["formal_index_sha256_before_supplement"],
    }
    for path, expected in snapshots.items():
        if digest(path) != expected:
            raise ValueError(f"Official snapshot changed before supplemental comparison: {path.name}")
    lock = read_json(BATCH / "selection-lock.json")
    lock_id = protocol["frozen_selection_lock_sha256"]
    if lock["lock_sha256"] != lock_id or fingerprint({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock_id:
        raise ValueError("The formal selection lock does not verify")
    return lock, observer, snapshots, read_json(round_dir / "result.json")


def score_r6(config, state, cuda_model, cuda_device, protocol, final):
    import learning
    import temporal

    official = {row["name"]: row for row in final["datasets"]["R6"]["sequences"]}
    expected = {row["name"]: row for row in protocol["r6_records"]}
    if set(official) != set(expected) or len(official) != 10:
        raise ValueError("Formal R6 coverage differs from the registered supplement")
    mount = state["installation_quaternion_wxyz"]
    selected = protocol["benchmark"]["record"]
    comparisons = []
    bench = None
    for spec in protocol["r6_records"]:
        name = spec["name"]
        data = cached_candidate(name, config)
        hashed = verified_indices_digest(name, historical_score_indices(name), data["index"])
        prior = official[name]
        if (hashed != spec["score_frame_indices_sha256"] or hashed != prior["score_frame_indices_sha256"]
                or len(data["index"]) != spec["frames"] or spec["frames"] != prior["frames"]):
            raise ValueError(f"R6 supplement scoring support differs for {name}")
        mounted = learning.apply_install(data, mount)
        part = temporal.prepare(mounted, config)
        plain = rmse(data["base_up_scored"], data["ref_body_scored"])
        no_residual = rmse(np.broadcast_to(Z, mounted["ref_world"].shape), mounted["ref_world"])
        restored = temporal.evaluate(part, config, cuda_model, cuda_device)
        discrepancy = abs(restored - prior["rmse_deg"]["locked"])
        if discrepancy > 1e-6:
            raise ValueError(f"Full frozen TCN did not reproduce on {name}: {discrepancy}°")
        comparisons.append({**spec, "pure_default_fullvqf_deg": prior["rmse_deg"]["full_vqf"],
                            "rmse_deg": dict(zip(METHODS, (plain, no_residual, restored), strict=True)),
                            "formal_locked_tcn_deg": prior["rmse_deg"]["locked"],
                            "full_arm_absolute_reproduction_difference_deg": discrepancy})
        print(name, comparisons[-1]["rmse_deg"], flush=True)
        if name == selected:
            bench = (mounted, part)
    if bench is None:
        raise ValueError("Registered timing record was absent")
    summary = {}
    for arm in METHODS:
        scores = [{"name": row["name"], "frames": row["frames"], "rmse_deg": row["rmse_deg"][arm],
                   "baseline_deg": row["pure_default_fullvqf_deg"]} for row in comparisons]
        summary[arm] = aggregate(scores)
    return comparisons, summary, bench


def output_parity(part, config, built):
    import temporal

    released = {}
    scored = {}
    for name, (device, (model, normal)) in built.items():
        delta = temporal.inference(part, model, normal, config, device)
        direction = unit(Z + rotate(unit(part["q"]), delta))
        selected = np.searchsorted(part["ready"], part["index"], side="right")
        released[name] = direction
        scored[name] = np.vstack((Z[None], direction))[selected]
    release_error = errors(released["cpu"], released["cuda"])
    native_error = errors(scored["cpu"], scored["cuda"])
    return {"scoring_frames": len(part["index"]), "actual_correction_releases": len(part["ready"]),
            "native_frames_scoring_indices_sha256": verified_indices_digest("R6-3", historical_score_indices("R6-3"), part["index"]),
            "same_frozen_weight_tilt_rmse_deg": {k: rmse(v, part["ref_world"]) for k, v in scored.items()},
            "mean_direction_delta_deg": float(np.mean(native_error)),
            "p95_direction_delta_deg": float(np.percentile(native_error, 95)),
            "max_direction_delta_deg": float(np.max(native_error)),
            "max_release_direction_delta_deg": float(np.max(release_error))}


def nominal_timing(values):
    return dict(zip(("p50_ms", "p95_ms", "max_ms"), np.percentile(values, (50, 95, 100)).tolist(), strict=True))


def native_timing(config, state, settings):
    import learning
    from vqf import VQF

    original = record(settings["record"])
    dt = float(original["dt"])
    gyro = np.ascontiguousarray(original["imu"][:, 3:], dtype=np.float64)
    accel = np.ascontiguousarray(original["imu"][:, :3], dtype=np.float64)
    start = settings["native_first_frame"]
    warmup = settings["native_warmup_frames"]
    total = settings["native_frames"]
    vqf = VQF(dt, **config["parameters"])
    vqf.updateBatch(gyro[:start], accel[:start])
    mount = state["installation_quaternion_wxyz"]
    samples = []
    for i in range(start, start + warmup + total):
        point = perf_counter_ns()
        vqf.update(gyro[i], accel[i])
        learning.multiply_quaternion(vqf.getQuat6D(), mount)
        if i >= start + warmup:
            samples.append((perf_counter_ns() - point) / 1e6)
    return {"sample_scope": settings["native_scope"], "sampling_hz": 1. / dt, "warmups": warmup,
            "first_timed_native_index": start + warmup, "samples": len(samples),
            "summary": nominal_timing(samples), "samples_ms": samples}


def correction_timing(native, part, config, built, settings):
    import temporal
    import torch
    from features import physics, block_indices

    warmup = settings["warmup_updates_per_backend"]
    count = settings["timed_correction_updates_per_backend"]
    indices = block_indices(len(native["windows"]), config["update_hz"])
    np.testing.assert_array_equal(native["ready"][indices], part["ready"])
    if len(indices) < warmup + count:
        raise ValueError("R6-3 has too few measured releases")
    choices = np.linspace(0, len(indices) - 1, warmup + count, dtype=int)
    positions = indices[choices]
    phase = 100 // config["update_hz"]
    context = ((max(64, config["context"]) + phase - 1) // phase) * phase
    time_ms = {key: [] for key in built}
    pair_gaps = []
    order = tuple(built)
    for i, block in enumerate(positions):
        outputs = {}
        for key in order if i % 2 == 0 else reversed(order):
            device, (model, normal) = built[key]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter_ns()
            stop = block + 1
            local = {name: native[name][stop-context:stop] for name in ("windows", "q", "up", "bias", "rest")}
            at, matrix, acceleration, gyro = physics(local, config)
            last = at[-1:]
            inputs = {"matrix": matrix[-1:], "up": local["up"][last], "acc": acceleration[-1:], "gyr": gyro[-1:],
                      "history": temporal.windows(local["windows"], last, config["context"])}
            delta = temporal.inference(inputs, model, normal, config, device)
            outputs[key] = unit(Z + rotate(unit(local["q"][last]), delta))[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (perf_counter_ns() - start) / 1e6
            if i >= warmup:
                time_ms[key].append(elapsed_ms)
            np.testing.assert_allclose(matrix[-1], part["matrix"][choices[i]], rtol=1e-5, atol=3e-6)
            np.testing.assert_array_equal(inputs["history"][0], part["history"][choices[i]])
        if i >= warmup:
            pair_gaps.append(float(errors(outputs["cpu"][None], outputs["cuda"][None])[0]))
    return {"sample_scope": settings["correction_scope"], "synchronous_single_stream": True,
            "preexisting_native_blocks": True, "cuda_post_forward_synchronization_included": True,
            "warmups_each": warmup, "samples_each": count, "update_hz": config["update_hz"],
            "sampled_original_release_indices": part["ready"][choices[warmup:]].tolist(),
            "maximum_paired_timed_update_direction_difference_deg": max(pair_gaps),
            "backends": {key: {"summary": nominal_timing(values), "samples_ms": values} for key, values in time_ms.items()}}


def run():
    import temporal
    import torch

    protocol = read_json(DESTINATION / "protocol.json")
    lock, observer, snapshots, final = ensure_ready(protocol)
    settings = protocol["benchmark"]
    observer_finished = observer.get("verified_at") or observer["checked_at"]
    started = stamp()
    write_json(DESTINATION / "status.json", {"phase": "running", "started_at": started, "pid": os.getpid(),
                                            "observer_finished_at": observer_finished})
    torch.set_num_threads(settings["cpu_threads"])
    torch.set_float32_matmul_precision("high")
    with np.load(BATCH / lock["weights"], allow_pickle=False) as data:
        state = {name: data[name] for name in data.files}
    config = lock["selected"]["config"]
    cuda = torch.device("cuda")
    built = {"cpu": (torch.device("cpu"), temporal.make_model(state, config, torch.device("cpu"))),
             "cuda": (cuda, temporal.make_model(state, config, cuda))}
    rows, summary, (native, part) = score_r6(config, state, built["cuda"][1], cuda, protocol, final)
    parity = output_parity(part, config, built)
    ref = next(row for row in rows if row["name"] == settings["record"])
    if abs(parity["same_frozen_weight_tilt_rmse_deg"]["cuda"] - ref["formal_locked_tcn_deg"]) > 1e-6:
        raise ValueError("R6-3 CUDA parity replay differs from the formal scorer")
    native_bench = native_timing(config, state, settings)
    correction_bench = correction_timing(native, part, config, built, settings)
    if any(digest(path) != expected for path, expected in snapshots.items()):
        raise ValueError("An official artifact changed during post-lock scoring")
    output = {"started_at": started, "completed_at": stamp(), "pid": os.getpid(), "lock_sha256": lock["lock_sha256"],
              "locked_weight_sha256": lock["weights_sha256"], "protocol_sha256": digest(DESTINATION / "protocol.json"),
              "observer_finished_at": observer_finished, "observer_status_checked_at": observer["checked_at"],
              "r6": {"summary": summary, "records": rows},
              "R6-3": {"output_parity": parity, "shared_native_timing": native_bench, "correction_update_timing": correction_bench},
              "cpu": platform.processor(), "gpu": torch.cuda.get_device_name(cuda), "torch": torch.__version__,
              "torch_cpu_threads": torch.get_num_threads(), "formal_artifact_hashes_unchanged": True, "refits": 0}
    write_json(DESTINATION / "result.json", output)
    write_json(DESTINATION / "status.json", {"phase": "completed", "started_at": started,
                                           "completed_at": output["completed_at"], "pid": os.getpid()})


if __name__ == "__main__":
    run()
