"""Compare the R5-selected GRU, pure FullVQF and the prior lightweight correction."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from vqf import VQF

from evaluate_external_tilt import broad_records, input_blocks, reference_direction, sassari_records, score_record
from evaluate_tilt import correction_update, latency_summary
from fit_gru_tilt import read_shice, validate
from gru_data import OUTPUT, SHICE_HZ, causal_resample
from gru_tilt_model import infer_sequences, load_gru, with_dt
from run_pilot import body_up, tilt_error
from tilt_data import ARTIFACTS, EST_UP, HOP_INPUT, ROOT, causal_fill, partition
from tilt_eval import full_record_scores, load_head, macro_scores, record_predictions, save_json
from tilt_model import contexts, multiply, normalize


LIGHT = "physical_only"
CHECKPOINT_SHA = "d661b824806e18555db791af4e7b52b35ad7e058702e1fa1261d30a35b29fd67"


def check_comparison(result, previous):
    if result["scoring_frames"] != previous["scoring_frames"]:
        raise ValueError(f"Scoring mask count changed in {result['recording']}")
    for method in ("full_vqf", LIGHT):
        if abs(result["rmse_deg"][method] - previous["rmse_deg"][method]) > 1e-5:
            raise ValueError(f"Previous {method} score changed in {result['recording']}")


def choose_on_r5(folders):
    candidates = []
    for folder in folders:
        report = json.loads((folder / "validation.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(folder / "gru.pt", map_location="cpu", weights_only=True)
        for key in ("training_records", "validation_records", "selected_epoch", "r5_macro_rmse_deg"):
            if checkpoint[key] != report[key]:
                raise ValueError(f"R5 report and checkpoint disagree in {folder}")
        candidates.append({"folder": str(folder.resolve()), "epoch": report["selected_epoch"],
                           "r5_macro_rmse_deg": report["r5_macro_rmse_deg"]})
    chosen = min(candidates, key=lambda row: row["r5_macro_rmse_deg"])
    return Path(chosen["folder"]) / "gru.pt", candidates


def summary(records):
    return {"records": len(records), "scoring_frames": sum(row["scoring_frames"] for row in records),
            "macro_rmse_deg": macro_scores(records), "per_sequence": records,
            "gru_beats_full_vqf": sum(row["rmse_deg"]["gru"] < row["rmse_deg"]["full_vqf"] for row in records),
            "gru_beats_lightweight": sum(row["rmse_deg"]["gru"] < row["rmse_deg"][LIGHT] for row in records)}


def shice_scores(model, head, output, artifacts, root, candidates, device):
    names = partition(root, "test")
    old = json.loads((artifacts / "test.json").read_text(encoding="utf-8"))
    lookup = {row["recording"]: row for row in old["r6_records"]}
    validation = read_shice(output, names)
    errors = validate(model, validation, device)
    records = []
    for name, error in zip(names, errors, strict=True):
        deltas = record_predictions(artifacts, name, {LIGHT: head}, device)
        result = full_record_scores(artifacts, name, deltas, candidates)
        check_comparison(result, lookup[name])
        result["rmse_deg"]["gru"] = error
        records.append(result)
    with np.load(output / "data" / "R6-3.npz", allow_pickle=False) as data:
        probe = (data["imu"], data["dt"].item(), SHICE_HZ)
    return summary(records), probe


def external_scores(name, source, model, head, artifact, dataset_root, candidates, device):
    rows = list(source(dataset_root))
    baseline = json.loads((artifact / "external_test.json").read_text(encoding="utf-8"))["datasets"][name]
    lookup = {row["recording"]: row["protocols"]["main"] for row in baseline["records"]}
    result = []
    indices = sorted(range(len(rows)), key=lambda index: len(rows[index].eligible))
    for start in range(0, len(indices), 8):
        batch = [rows[index] for index in indices[start:start + 8]]
        clean = [causal_fill(row.imu, np.isfinite(row.imu).all(axis=1)).astype(np.float32) for row in batch]
        states = infer_sequences(model, [with_dt(imu, 1. / row.frequency) for row, imu in zip(batch, clean)], device)
        for row, state in zip(batch, states, strict=True):
            if not np.isfinite(state).all():
                raise ValueError(f"Invalid GRU prediction in {row.name}")
            scores = score_record(row, None, {LIGHT: head}, candidates, device, 256, False)["protocols"]["main"]
            scores["recording"] = row.name
            check_comparison(scores, lookup[row.name])
            up, valid = reference_direction(row.quaternion)
            mask = row.eligible & valid & np.isfinite(row.imu).all(axis=1)
            if int(mask.sum()) != scores["scoring_frames"]:
                raise ValueError(f"GRU mask mismatch in {row.name}")
            error = tilt_error(state[mask], up[mask])
            scores["rmse_deg"]["gru"] = float(np.sqrt(np.mean(np.square(error))))
            result.append(scores)
    representative = rows[21] if name == "BROAD" else rows[-4]
    imu = causal_fill(representative.imu, np.isfinite(representative.imu).all(axis=1))
    return summary(sorted(result, key=lambda row: row["recording"])), (imu, 1. / representative.frequency, representative.frequency)


def measure_action(action, count, warmup):
    times = []
    for index in range(warmup + count):
        start = perf_counter_ns()
        action(index)
        if index >= warmup:
            times.append((perf_counter_ns() - start) / 1e6)
    return latency_summary(times)


@torch.inference_mode()
def measure_gru(model, imu, dt, rate, device):
    source = imu if abs(rate * dt - 1) < 1e-8 else causal_resample(imu, dt, rate)[0]
    data = with_dt(np.asarray(source, dtype=np.float32), 1. / rate)
    offset = int(10 * rate)
    hidden = None

    def step(index):
        nonlocal hidden
        signal = torch.as_tensor(data[offset + index:offset + index + 1][None, :], device=device)
        quat, hidden = model(signal, hidden)
        quat[0, 0].cpu().numpy()

    return measure_action(step, 1200, 100)


def measure_vqf_and_light(imu, dt, head, gain, device):
    acc = np.ascontiguousarray(imu[:, :3], dtype=np.float64)
    gyr = np.ascontiguousarray(imu[:, 3:], dtype=np.float64)
    baseline = VQF(dt)
    start = int(10 / dt)
    baseline.updateBatch(gyr[:start], acc[:start])

    def frame(index):
        baseline.update(gyr[start + index], acc[start + index])
        baseline.getQuat6D()

    if len(acc) < start + 5100:
        raise ValueError("Not enough native data for a representative latency sample")
    application_rows = np.linspace(start, len(imu) - 1, 5100, dtype=int)
    vqf_latency = measure_action(frame, 5000, 100)
    data = VQF(dt).updateBatch(gyr, acc)
    windows, state_at, _ = input_blocks(imu, 1. / dt)
    blocks = np.linspace(600, len(windows) - 1, 130, dtype=int)
    history = contexts(windows, blocks)
    q, bias, rest = (data[key][state_at[blocks]].astype(np.float32) for key in ("quat6D", "bias", "restDetected"))

    @torch.inference_mode()
    def update(index):
        return correction_update(None, head, history[index], q[index], bias[index], rest[index], gain, device)

    _, held, _ = update(0)
    update_latency = measure_action(update, 120, 10)

    def apply(index):
        normalize(multiply(held, data["quat6D"][application_rows[index]]))

    return vqf_latency, update_latency, measure_action(apply, 5000, 100)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--candidate-dir", type=Path, action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    path = args.artifacts / f"{LIGHT}_head.pt"
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != CHECKPOINT_SHA:
            raise ValueError("The lightweight checkpoint differs from the earlier R5 selection")
    head = load_head(path, device)
    light_validation = json.loads((args.artifacts / "validation.json").read_text(encoding="utf-8"))
    gain = light_validation["gains"][LIGHT]
    candidates = {LIGHT: (LIGHT, gain, False)}
    selected_path, candidate_summary = choose_on_r5([args.output, *args.candidate_dir])
    model, chosen = load_gru(selected_path, device)
    print(f"R5 selected {selected_path}: {chosen['r5_macro_rmse_deg']:.6f}°", flush=True)
    if chosen["training_records"] != partition(args.root, "train") or chosen["validation_records"] != partition(args.root, "valid"):
        raise ValueError("GRU candidate did not use the required train and validation sequences")
    if set(partition(args.root, "test")) & set(chosen["training_records"] + chosen["validation_records"]):
        raise ValueError("GRU checkpoint overlaps the test set")

    tests, probes = {}, {}
    tests["R6"], probes["R6"] = shice_scores(model, head, args.output, args.artifacts, args.root, candidates, device)
    print("R6", tests["R6"]["macro_rmse_deg"], flush=True)
    for name, source in (("BROAD", broad_records), ("Sassari", sassari_records)):
        tests[name], probes[name] = external_scores(name, source, model, head, args.artifacts,
                                                     args.root.parent, candidates, device)
        print(name, tests[name]["macro_rmse_deg"], flush=True)

    timings = {}
    cpu = torch.device("cpu")
    cpu_model, _ = load_gru(selected_path, cpu)
    cpu_head = load_head(path, cpu)
    for name, (imu, dt, rate) in probes.items():
        baseline, light_gpu, apply = measure_vqf_and_light(imu, dt, head, gain, device)
        _, light_cpu, _ = measure_vqf_and_light(imu, dt, cpu_head, gain, cpu)
        timings[name] = {"gru_hz": rate, "native_hz": 1. / dt, "light_update_hz": 100. / HOP_INPUT,
                         "full_vqf_cpu_frame_ms": baseline, "light_apply_cpu_frame_ms": apply,
                         "light_gpu_update_ms": light_gpu, "light_cpu_update_ms": light_cpu,
                         "gru_gpu_step_ms": measure_gru(model, imu, dt, rate, device),
                         "gru_cpu_step_ms": measure_gru(cpu_model, imu, dt, rate, cpu)}
        print(name, timings[name], flush=True)
    with selected_path.open("rb") as stream:
        selected_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"selected_epoch": chosen["selected_epoch"], "r5_macro_rmse_deg": chosen["r5_macro_rmse_deg"],
              "selected_checkpoint": str(selected_path), "selected_sha256": selected_hash,
              "r5_candidates": candidate_summary,
              "light_gain": gain, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "tests": tests, "latency": timings}
    save_json(args.output / "test.json", report)


if __name__ == "__main__":
    main()
