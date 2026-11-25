"""Verify and time a CPU readout for the same fitted causal physical-head weights."""

import numpy as np
from scipy.special import ndtr
import torch

from common import BATCH, Z, fingerprint, names, read_json, rotate, stamp, unit, write_json
from ensemble import planned_folds
from final_eval import cached_candidate, fit_candidate, timed
import learning
from legacy_head import evaluate, make_model, prepare
from tilt_eval import predict_updates
from tilt_model import RESIDUAL_SCALE, physical_features


class PhysicalReadout:
    def __init__(self, state):
        self.mean = state["physical_mean"]
        self.std = state["physical_std"]
        self.matrices = [np.ascontiguousarray(state[f"readout.{k}.weight"].T) for k in (0, 3, 6)]
        self.biases = [state[f"readout.{k}.bias"] for k in (0, 3, 6)]

    def predict(self, physical):
        x = ((physical - self.mean) / self.std).clip(-12., 12.)
        for matrix, bias in zip(self.matrices[:-1], self.biases[:-1], strict=True):
            x = x @ matrix + bias
            x = x * ndtr(x)
        output = np.tanh(x @ self.matrices[-1] + self.biases[-1]) * (3 * RESIDUAL_SCALE)
        up = physical[:, :3]
        return output - (output * up).sum(axis=1, keepdims=True) * up

    def __call__(self, tensor, _embedding=None):
        return torch.from_numpy(self.predict(tensor.numpy()))


def latency(state, config, gpu, cpu, benchmark_record):
    model = make_model(state, gpu)
    cpu_model = make_model(state, cpu)
    numpy_head = PhysicalReadout(state)
    raw = cached_candidate(benchmark_record, config)
    q_mount = state.get("installation_quaternion_wxyz")
    data = learning.apply_install(raw, q_mount, include_reference=False) if q_mount is not None else raw
    points = np.linspace(100, len(data["windows"])-1, 220, dtype=int)
    matrix_cache = [physical_features(data["windows"][None,i-63:i+1], data["up"][i:i+1],
                                     data["bias"][i:i+1], data["rest"][i:i+1]) for i in points]

    def update(i, method):
        p = points[i]
        feature = physical_features(data["windows"][None,p-63:p+1], data["up"][p:p+1],
                                    data["bias"][p:p+1], data["rest"][p:p+1])
        residual = method(feature) * config["gain"]
        return unit(Z + rotate(unit(data["q"][p:p+1]), residual))

    predictors = {"torch_cuda": lambda x: predict_updates(model, x, None, gpu),
                  "torch_cpu": lambda x: predict_updates(cpu_model, x, None, cpu), "numpy_cpu": numpy_head.predict}
    rows = {}
    for label, predict in predictors.items():
        rows[label] = {"forward_batch_one": timed(lambda i: predict(matrix_cache[i]), 200, 20),
                       "correction_update": timed(lambda i: update(i, predict), 200, 20)}
    return {"recording": benchmark_record, "timings": rows, "thread_count_torch": torch.get_num_threads(),
            "warmup_windows": 20, "timed_windows": 200, "nominal_update_hz": 1./(100*float(data["dt"]))}


def run(job, destination):
    config = job["config"]
    identity = config["parent_round"]
    parent = read_json(BATCH / "rounds" / identity / "round.json")
    model_config = parent["config"]
    if fingerprint(model_config) != config["parent_config_sha256"] or model_config["kind"] != "legacy_head":
        raise ValueError("CPU readout verification requires the frozen physical-head configuration")
    gpu, cpu = torch.device("cuda"), torch.device("cpu")
    records = names("dev")
    raw = {name: cached_candidate(name, model_config) for name in records}
    checks = []
    for axis, label, training, held in planned_folds(records):
        state = fit_candidate(model_config, training)
        model = make_model(state, gpu)
        original_cpu = make_model(state, cpu)
        port = PhysicalReadout(state)
        mounting = state.get("installation_quaternion_wxyz")
        for name in held:
            data = learning.apply_install(raw[name], mounting) if mounting is not None else raw[name]
            part = prepare(data, model_config)
            reference = predict_updates(model, part["matrix"], None, gpu)
            torch_cpu = predict_updates(original_cpu, part["matrix"], None, cpu)
            numpy_cpu = port.predict(part["matrix"])
            base = evaluate(part, model_config, model, gpu)
            score = evaluate(part, model_config, port, cpu)
            check = {"axis": axis, "fold": label, "record": name, "score_frames": len(part["index"]),
                     "gpu_tilt_rmse_deg": base, "numpy_cpu_tilt_rmse_deg": score, "tilt_delta_deg": score-base,
                     "max_abs_residual_torch_cpu": float(np.max(np.abs(torch_cpu-reference))),
                     "max_abs_residual_numpy_cpu": float(np.max(np.abs(numpy_cpu-reference)))}
            if abs(check["tilt_delta_deg"]) > 1e-4:
                raise ValueError(f"Score changed materially after porting: {check}")
            checks.append(check)
        print(axis, label, max(abs(row["tilt_delta_deg"]) for row in checks), flush=True)
    live_state = fit_candidate(model_config, records)
    measured = latency(live_state, model_config, gpu, cpu, config["record"])
    write_json(destination, {"started_for_round": job["round_id"], "finished_at": stamp(), "model_source": identity,
                             "model_config_sha256": fingerprint(model_config), "checks": checks, "latency": measured,
                             "max_abs_tilt_rmse_delta_deg": max(abs(row["tilt_delta_deg"]) for row in checks),
                             "tested_no_holdout_partitions": True})
    print(measured, flush=True)


def queue():
    identity = "L00005"
    source = "P00022"
    parent = read_json(BATCH / "rounds" / source / "round.json")
    target = BATCH / "rounds" / identity / "round.json"
    config = {"kind": "cpu_readout_audit", "parent_round": source, "parent_config_sha256": parent["config_sha256"], "record": "R3-3"}
    if not target.is_file():
        write_json(target, {"round_id": identity, "config": config, "config_sha256": fingerprint(config), "seed": parent["seed"],
                            "batch": "cpu-readout-audit", "relation": source,
                            "hypothesis": "Run the original causal physical head on CPU with unchanged weights and score frames",
                            "partitions": {"development": names("dev"), "benchmark": [config["record"]]},
                            "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                            "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"})
    print(identity, source, flush=True)


if __name__ == "__main__":
    queue()
