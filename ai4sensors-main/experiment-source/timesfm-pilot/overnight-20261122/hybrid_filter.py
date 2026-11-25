"""Causally blend native FullVQF and a faster tilt filter, with an exact default fallback."""

import numpy as np

from common import BATCH, EXCLUDED, aggregate, body_up, by_device, errors, fingerprint, names, read_json, record, rmse, stamp, unit, write_json
from experiment import baseline_reference
from features import G, fetch_record, moment


def pair(default, fast):
    if not np.array_equal(default["index"], fast["index"]) or not np.array_equal(default["ready"], fast["ready"]):
        raise ValueError("The two FullVQF branches must share all score frames and update times")
    return {"default": default, "fast": fast, "index": default["index"], "dt": float(default["dt"])}


def cached(name, config):
    return pair(fetch_record(name, {}), fetch_record(name, config["parameters"]))


def confidence(values, thresholds):
    low, high = thresholds
    value = np.clip((high - values) / (high - low), 0., 1.)
    return value * value * (3. - 2. * value)


def gate(windows, default_up, fast_up, config):
    """Inputs are aligned causal 100 Hz block means and branch outputs at block ends."""
    mode = config["mode"]
    if mode != "gated":
        return np.full(len(windows), {"default": 0., "fast": 1., "fixed": config["max_fast_weight"]}[mode])

    width = config["history_blocks"]
    at = np.arange(len(windows))
    acc = windows[:, :3] / G
    motion = np.linalg.norm(windows[:, 3:], axis=1)
    magnitude_error = moment(np.abs(np.linalg.norm(acc, axis=1) - 1.)[:, None], at, width)[:, 0]
    acc_mean = moment(acc, at, width)
    angular_speed = moment(motion[:, None], at, width)[:, 0]
    recent = moment(acc, at, 4) if "disagreement_acc_change_deg2" in config["thresholds"] else None
    weight = weight_from_signals(default_up, fast_up, magnitude_error, acc_mean, angular_speed, config, recent)
    weight[:width - 1] = 0.
    return weight


def weight_from_signals(default_up, fast_up, magnitude_error, acc_mean, angular_speed, config, recent_acc=None):
    acceleration_innovation = errors(fast_up, unit(acc_mean))
    disagreement = errors(default_up, fast_up)
    available_signals = {"disagreement_deg": disagreement, "acc_magnitude_g": magnitude_error,
                         "acc_innovation_deg": acceleration_innovation, "gyro_rad_s": angular_speed}
    if recent_acc is not None:
        available_signals["disagreement_acc_change_deg2"] = disagreement * errors(unit(recent_acc), unit(acc_mean))
    weight = np.full(np.shape(magnitude_error), config["max_fast_weight"])
    for key, limits in config["thresholds"].items():
        weight *= confidence(available_signals[key], limits)
    return weight


def mix(default_up, fast_up, weight):
    combined = unit(default_up + weight[..., None] * (fast_up - default_up))
    return np.where(weight[..., None] == 0., default_up,
                    np.where(weight[..., None] == 1., fast_up, combined))


def score(data, config, delay_ms=0.):
    default, fast = data["default"], data["fast"]
    eligible = data["index"]
    block_weight = gate(default["windows"], default["up"], fast["up"], config)
    delay = int(np.ceil(delay_ms / (1000 * data["dt"])))
    active = np.searchsorted(default["ready"] + delay, eligible, side="right")
    weights = np.r_[0., block_weight][active]
    direction = mix(default["base_up_scored"], fast["base_up_scored"], weights)
    stats = {"default_fallback_frames": int((weights == 0.).sum()),
             "fast_only_frames": int((weights == 1.).sum()), "mean_fast_weight": float(weights.mean())}
    return rmse(direction, default["ref_body_scored"]), stats


def bench(config, benchmark_record):
    from time import perf_counter_ns
    from vqf import VQF
    from final_eval import percentile

    source = record(benchmark_record)
    imu, dt = np.ascontiguousarray(source["imu"], dtype=np.float64), float(source["dt"])
    width = config["history_blocks"]
    begin = 22000
    warmup = width * 10 + 100
    cut = begin - warmup
    slow = VQF(dt)
    fast = VQF(dt, **config["parameters"])
    warm_acc = np.ascontiguousarray(imu[:cut, :3])
    warm_gyr = np.ascontiguousarray(imu[:cut, 3:])
    for branch in (slow, fast):
        branch.updateBatch(warm_gyr, warm_acc)
    block = np.empty((10, 6))
    history = np.zeros((width, 6))
    step_times, gate_times = [], []
    weight = np.asarray(0.)
    for i in range(cut, begin + 3000):
        started = perf_counter_ns()
        incoming = imu[i]
        slow.update(incoming[3:], incoming[:3])
        fast.update(incoming[3:], incoming[:3])
        slow_up = body_up(slow.getQuat6D())
        fast_up = body_up(fast.getQuat6D())
        block[i % 10] = incoming
        if i % 10 == 9:
            gate_start = perf_counter_ns()
            history[(i // 10) % width] = block.mean(axis=0)
            acc = history[:, :3] / G
            magnitude_error = np.abs(np.linalg.norm(acc, axis=1) - 1.).mean()
            speed = np.linalg.norm(history[:, 3:], axis=1).mean()
            if i - cut + 1 >= width * 10:
                mode = config["mode"]
                if mode == "gated":
                    recent = None
                    if "disagreement_acc_change_deg2" in config["thresholds"]:
                        recent = acc[(i // 10 - np.arange(4)) % width].mean(axis=0)
                    weight = weight_from_signals(slow_up, fast_up, magnitude_error, acc.mean(axis=0), speed, config, recent)
                else:
                    weight = np.asarray({"default": 0., "fast": 1., "fixed": config["max_fast_weight"]}[mode])
            if i >= begin:
                gate_times.append((perf_counter_ns() - gate_start) / 1e6)
        mix(slow_up, fast_up, weight)
        if i >= begin:
            step_times.append((perf_counter_ns() - started) / 1e6)
    return {"record": benchmark_record, "native_step": percentile(step_times), "nominal_native_hz": 1. / dt,
            "gate_update": percentile(gate_times), "gate_hz": 1. / (10 * dt),
            "native_step_includes_two_filters_gate_buffer_and_current_direction_output": True}


def run(item):
    config = item["config"]
    expected = baseline_reference()
    records = []
    for name in names("dev"):
        data = cached(name, config)
        value, metrics = score(data, config)
        if config["mode"] == "default" and abs(value - expected[name]) > 1e-5:
            raise ValueError(f"Default branch differs from pure FullVQF on {name}")
        records.append({"name": name, "frames": len(data["index"]), "baseline_deg": expected[name],
                        "rmse_deg": value, **metrics})
    macro = {**aggregate(records), "per_device": by_device(records)}
    return {"dev_metrics": {"device": macro, "motion": macro, "selection_rmse_deg": macro["macro_rmse_deg"]},
            "sequences": records, "fixed_before_development": True, "config": config,
            "native_branches": {"default": {}, "fast": config["parameters"]}, "gate_rate_hz": 100,
            "mean_default_fallback_fraction": float(np.mean([r["default_fallback_frames"] / r["frames"] for r in records])),
            "mean_fast_only_fraction": float(np.mean([r["fast_only_frames"] / r["frames"] for r in records])),
            "mean_fast_weight": float(np.mean([r["mean_fast_weight"] for r in records])),
            "inference_inputs": ["causal IMU history", "current default FullVQF tilt", "current fast FullVQF tilt"]}


def queue():
    plans = (
        ("default-branch", "default", 0., 20, {}),
        ("fast-branch", "fast", 1., 20, {}),
        ("fixed-half-blend", "fixed", 0.5, 20, {}),
        ("disagreement-only", "gated", 1., 20, {"disagreement_deg": [1.0, 3.0]}),
        ("acceleration-only", "gated", 1., 20, {"acc_magnitude_g": [0.03, 0.12], "acc_innovation_deg": [3., 12.]}),
        ("combined", "gated", 1., 20, {"disagreement_deg": [1.0, 3.0], "acc_magnitude_g": [0.03, 0.12],
                                     "acc_innovation_deg": [3., 12.]}),
        ("combined-gyro", "gated", 1., 20, {"disagreement_deg": [1.0, 3.0], "acc_magnitude_g": [0.03, 0.12],
                                         "acc_innovation_deg": [3., 12.], "gyro_rad_s": [0.8, 2.0]}),
        ("permissive", "gated", 1., 12, {"disagreement_deg": [1.5, 5.0], "acc_magnitude_g": [0.04, 0.20],
                                       "acc_innovation_deg": [4., 17.]}),
        ("conservative", "gated", 0.9, 32, {"disagreement_deg": [0.5, 2.2], "acc_magnitude_g": [0.02, 0.08],
                                          "acc_innovation_deg": [2., 8.]}),
        ("rare-disagreement-veto", "gated", 1., 20, {"disagreement_deg": [3.4, 3.9]}),
        ("changing-acceleration-veto", "gated", 1., 20, {"disagreement_acc_change_deg2": [8.5, 14.0]}),
        ("changing-acceleration-and-magnitude", "gated", 1., 20, {"disagreement_acc_change_deg2": [8.5, 14.0],
                                                                "acc_magnitude_g": [0.03, 0.11]}),
        ("rare-joint-veto", "gated", 1., 20, {"disagreement_acc_change_deg2": [11.0, 17.0],
                                             "disagreement_deg": [3.6, 4.3]}),
    )
    stage = []
    for number, (label, mode, weight, width, limits) in enumerate(plans, 1):
        identity = f"H{number:05d}"
        destination = BATCH / "rounds" / identity / "round.json"
        config = {"kind": "hybrid", "parameters": {"tauAcc": 0.2}, "mode": mode, "max_fast_weight": weight,
                  "thresholds": limits, "history_blocks": width, "seed": None}
        if destination.is_file():
            if read_json(destination)["config_sha256"] != fingerprint(config):
                raise ValueError(f"Frozen hybrid configuration changed at {identity}")
        else:
            write_json(destination, {"round_id": identity, "config": config, "config_sha256": fingerprint(config), "seed": None,
                        "batch": "causal-default-fast-hybrid", "relation": "R00021",
                        "hypothesis": label, "partitions": {"development": names("dev"), "excluded": list(EXCLUDED)},
                        "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                        "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"})
        stage.append({"round": identity, "role": label, "config": config})
    write_json(BATCH / "hybrid-development-stage.json", {"created_at": stamp(), "rounds": stage,
               "pure_default_control": "R00001", "pure_fast_control": "R00021", "holdouts_unopened": True})
    print([(r["round"], r["role"]) for r in stage], flush=True)


if __name__ == "__main__":
    queue()
