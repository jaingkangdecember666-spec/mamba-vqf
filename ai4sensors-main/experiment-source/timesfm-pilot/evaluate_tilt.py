"""Score the R5-selected SHICE correction on R6 and measure local inference."""

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from vqf import VQF

from run_pilot import CHECKPOINT_REVISION, body_up, load_record
from tilt_data import ARTIFACTS, CONTEXT, EST_UP, HOP_INPUT, NATIVE_PER_INPUT, ROOT, partition
from tilt_eval import full_record_scores, load_head, macro_scores, open_embedding, predict_updates, record_predictions, save_json
from tilt_model import contexts, correction_quaternions, encode, load_backbone, multiply, normalize, physical_features


BENCH_RECORD = "R6-3"
BENCH_UPDATES = 120


def latency_summary(values):
    percentiles = np.percentile(values, (50, 95, 100)).tolist()
    return dict(zip(("p50", "p95", "max"), percentiles))


@torch.inference_mode()
def correction_update(model, head, history, base, bias, rest, gain, device):
    """Compute one causal world-space correction plus the current output quaternion."""
    windows = history[None, :, :]
    up = body_up(base[None, :], EST_UP).astype(np.float32)
    physics = physical_features(windows, up, bias[None, :], np.asarray([rest], dtype=np.float32))
    latent = encode(model, windows, device) if head.embedding_width else None
    delta = head(torch.as_tensor(physics, device=device), latent).cpu().numpy()[0]
    correction = correction_quaternions(base, delta * gain)
    return delta, correction, normalize(multiply(correction, base))


def benchmark(root, artifacts, model, heads, gains, device):
    with np.load(artifacts / "prepared" / f"{BENCH_RECORD}.npz", allow_pickle=False) as data:
        update_blocks = data["update_blocks"]
        ids = np.linspace(100, len(update_blocks) - 1, BENCH_UPDATES + 10, dtype=int)
        histories = contexts(data["windows"], update_blocks[ids])
        update_rows = data["updates"][ids]
        base = data["base"][update_rows]
        physical = physical_features(histories, data["base_up"][update_rows], data["bias"][ids], data["rest"][ids])
        bias, rests = data["bias"][ids], data["rest"][ids]

    timings, single_deltas = {}, {}
    for method, head in heads.items():
        measured, deltas = [], []
        gain = gains[method] or 1.
        for index, (history, q, b, rest) in enumerate(zip(histories, base, bias, rests)):
            start = perf_counter_ns()
            delta, correction, output = correction_update(model, head, history, q, b, rest, gain, device)
            took_ms = (perf_counter_ns() - start) / 1e6
            if index >= 10:
                measured.append(took_ms)
            deltas.append(delta)
            if index == 0:
                expected_up = normalize(body_up(q[None, :], EST_UP)[0] + gain * delta)
                np.testing.assert_allclose(body_up(output[None, :], EST_UP)[0], expected_up, atol=1e-5)
                np.testing.assert_allclose(np.linalg.norm(correction), 1., atol=1e-6)
        timing = latency_summary(measured)
        timings[method] = timing
        print(f"{method}: correction P50={timing['p50']:.3f} ms P95={timing['p95']:.3f} ms", flush=True)
        single_deltas[method] = np.stack(deltas)

    embedding = open_embedding(artifacts, BENCH_RECORD)[ids]
    batched_deltas = predict_updates(heads["timesfm"], physical, embedding, device)
    batch_difference = float(np.max(np.linalg.norm(single_deltas["timesfm"] - batched_deltas, axis=1)))
    if batch_difference > 0.002:
        raise ValueError(f"Single-window and batched TimesFM residuals differ by {batch_difference:.6f}")

    dt, imu, _, _ = load_record(root, BENCH_RECORD)
    acc = np.ascontiguousarray(imu[:21000, :3])
    gyr = np.ascontiguousarray(imu[:21000, 3:])
    vqf = VQF(dt)
    base_times, frame_times = [], []
    held = correction_quaternions(base[0], single_deltas["timesfm"][0])
    for index, (a, g) in enumerate(zip(acc, gyr)):
        start = perf_counter_ns()
        vqf.update(g, a)
        q = vqf.getQuat6D()
        finished = perf_counter_ns()
        normalize(multiply(held, q))
        if index >= 1000:
            base_times.append((finished - start) / 1e6)
            frame_times.append((perf_counter_ns() - finished) / 1e6)
    return {"benchmark_record": BENCH_RECORD, "correction_windows_per_method": BENCH_UPDATES,
            "native_frames": len(base_times), "full_vqf_per_frame_ms": latency_summary(base_times),
            "quaternion_apply_per_frame_ms": latency_summary(frame_times), "correction_update_ms": timings,
            "single_vs_batched_delta_max_norm": batch_difference,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "gpu": torch.cuda.get_device_name()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    validation = json.loads((args.artifacts / "validation.json").read_text(encoding="utf-8"))
    gains = validation["gains"]
    names = partition(args.root, "test")
    if set(names) & set(validation["training_records"] + validation["validation_records"]):
        raise ValueError("Test sequences overlap with train or validation")
    heads = {name: load_head(args.artifacts / f"{name}_head.pt", device) for name in gains}
    backbone = load_backbone(args.checkpoint, device)
    timings = benchmark(args.root, args.artifacts, backbone, heads, gains, device)
    delay_ms = timings["correction_update_ms"]["timesfm"]["p95"]
    candidates = {name: (name, gain, False) for name, gain in gains.items()}
    candidates["timesfm_p95_delay"] = ("timesfm", gains["timesfm"], True)
    candidates["timesfm_unscaled"] = ("timesfm", 1., False)
    scores = []
    for name in names:
        deltas = record_predictions(args.artifacts, name, heads, device)
        result = full_record_scores(args.artifacts, name, deltas, candidates, latency_ms=delay_ms)
        scores.append(result)
        print(f"{name} {result['rmse_deg']}", flush=True)
    macro = macro_scores(scores)
    report = {"test_records": names, "scoring_frames": sum(row["scoring_frames"] for row in scores),
              "r6_macro_rmse_deg": macro, "r6_records": scores, "gains_from_r5": gains,
              "improved_records": {method: int(sum(row["rmse_deg"][method] < row["rmse_deg"]["full_vqf"]
                                                        for row in scores)) for method in candidates},
              "latency": timings, "replay_delay_ms": delay_ms, "context_input_samples": CONTEXT,
              "update_every_native_frames": HOP_INPUT * NATIVE_PER_INPUT,
              "backbone_revision": CHECKPOINT_REVISION, "torch": torch.__version__}
    save_json(args.artifacts / "test.json", report)
    print(json.dumps({"r6_macro_rmse_deg": macro, "latency": timings}, indent=2), flush=True)


if __name__ == "__main__":
    main()
