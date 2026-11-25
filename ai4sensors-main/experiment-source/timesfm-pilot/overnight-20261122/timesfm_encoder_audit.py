"""Measure the cost of the unused forecast head in the frozen TimesFM tilt path."""

from time import perf_counter_ns

import numpy as np
import torch

from common import BATCH, HISTORY, PILOT, digest, names, stamp, write_json
from evaluate_tilt import correction_update
from optimizer_study import schedule
from run_pilot import body_up
from tilt_data import EST_UP
from tilt_eval import load_head
from tilt_model import contexts, correction_quaternions, encode, load_backbone, multiply, normalize, physical_features


WEIGHTS = HISTORY / "timesfm_head.pt"
CONFIG = PILOT / "checkpoint" / "config.json"
UPSTREAM = PILOT / "upstream" / "src" / "timesfm3" / "torch" / "model.py"
RECORDS = ("R1-10", "R2-10", "R3-3", "R4-2")


@torch.inference_mode()
def encode_without_forecasts(model, windows, device):
    signal = torch.as_tensor(windows, device=device)
    values = signal.transpose(1, 2).reshape(len(signal), 6, -1, model.input_patch_len)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        values = torch.nan_to_num(values, nan=0.0).clamp(-model.value_clip, model.value_clip)
        masks = torch.zeros_like(values, dtype=torch.bool)
        is_target = torch.ones(values.shape[:-1], device=device, dtype=torch.bool)
        _, transformer_input, patch_mask, _, _ = model._preprocess(values, masks, is_target)
        effective_mask = torch.cumprod(patch_mask.int(), dim=2).bool()
        hidden, _, _ = model.transformer_stack(transformer_input, effective_mask)
    return hidden[:, :, -1].to(torch.float16).contiguous()


@torch.inference_mode()
def encoder_update(model, head, history, base, bias, rest, gain, device):
    windows = history[None]
    up = body_up(base[None], EST_UP).astype(np.float32)
    physical = physical_features(windows, up, bias[None], np.asarray([rest], dtype=np.float32))
    embedding = encode_without_forecasts(model, windows, device)
    delta = head(torch.as_tensor(physical, device=device), embedding).cpu().numpy()[0]
    correction = correction_quaternions(base, delta * gain)
    return delta, correction, normalize(multiply(correction, base))


def probe_records(config):
    for name in config["records"]:
        if name not in names("dev"):
            raise ValueError(f"Nondevelopment input {name}")
        with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as data:
            count = config["samples_per_record"] + config["warmup_per_record"]
            ids = np.linspace(100, len(data["update_blocks"]) - 1, count, dtype=int)
            history = contexts(data["windows"], data["update_blocks"][ids])
            native_rows = data["updates"][ids]
            yield name, native_rows, history, data["base"][native_rows], data["bias"][ids], data["rest"][ids]


def run(job, destination):
    config = job["config"]
    for key, path in (("head_sha256", WEIGHTS), ("checkpoint_config_sha256", CONFIG), ("upstream_model_sha256", UPSTREAM)):
        if config[key] != digest(path):
            raise ValueError(f"Source changed before timing: {path.name}")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    head = load_head(WEIGHTS, device)
    model = load_backbone(PILOT / "checkpoint", device)
    if not model.use_variate_attention:
        raise ValueError("Expected native cross-channel attention")
    comparisons, embeddings = [], []
    timings = {"full_forward": [], "encoder_only": []}
    calls = {"full_forward": correction_update, "encoder_only": encoder_update}
    keys = tuple(calls)
    for name, rows, histories, bases, biases, rests in probe_records(config):
        expected = encode(model, histories, device)
        actual = encode_without_forecasts(model, histories, device)
        torch.testing.assert_close(expected, actual, rtol=0., atol=0.)
        if not torch.isfinite(actual).all().item():
            raise ValueError(f"Nonfinite embedding in {name}")
        embeddings.append({"record": name, "windows": len(histories), "exact_in_batch": True})
        print(name, "embedding exact", flush=True)
        for i, (row, history, base, bias, rest) in enumerate(zip(rows, histories, bases, biases, rests, strict=True)):
            pair = {}
            for method in keys if i % 2 == 0 else reversed(keys):
                start = perf_counter_ns()
                outputs = calls[method](model, head, history, base, bias, rest, config["gain"], device)
                elapsed_ms = (perf_counter_ns() - start) / 1e6
                pair[method] = outputs
                if i >= config["warmup_per_record"]:
                    timings[method].append(elapsed_ms)
            for original, optimized in zip(pair[keys[0]], pair[keys[1]], strict=True):
                np.testing.assert_array_equal(original, optimized)
            comparisons.append({"record": name, "native_index": int(row),
                                "timed": i >= config["warmup_per_record"]})
    summaries = {method: dict(zip(("p50_ms", "p95_ms", "max_ms"), np.percentile(values, (50, 95, 100)).tolist(), strict=True))
                 for method, values in timings.items()}
    write_json(destination, {"round_id": job["round_id"], "created_at": stamp(), "config": config,
               "gpu": torch.cuda.get_device_name(device), "embedding_checks": embeddings,
               "outputs_exact": ["residual_vector", "correction_quaternion", "output_quaternion"],
               "probe_indices": comparisons, "latency_summary": summaries, "samples_ms": timings,
               "new_fits": 0, "used_for_selection": False})


def queue():
    config = {"kind": "timesfm_encoder_diagnostic", "seed": None, "records": list(RECORDS),
              "samples_per_record": 30, "warmup_per_record": 4, "gain": 0.75,
              "head_sha256": digest(WEIGHTS), "checkpoint_config_sha256": digest(CONFIG),
              "upstream_model_sha256": digest(UPSTREAM)}
    write_json(BATCH / "timesfm-encoder-diagnostic-protocol.json", {"at": stamp(), **config,
               "single_update_scope": "physical features, frozen representation, existing head, CPU copy, correction and output",
               "timing_order": "alternate both paths on identical windows; batch size one, unique measured windows",
               "parity": "Exact batch embeddings and exact single-update residual and both quaternions",
               "eligible_for_model_selection": False})
    return schedule("X00001", config, "timesfm-encoder-diagnostic", None,
                    "Measure pinned TimesFM without the unused forecasting output, under exact correction parity")


if __name__ == "__main__":
    print(queue())
