"""A compact causal TCN for a low-rate correction over native-rate FullVQF."""

from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import learning
from common import Z, aggregate, by_device, names, rmse, rotate, unit
from experiment import baseline_reference
from features import SCALE, VERSION, block_indices, fetch_record


class CausalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.left = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation, padding=self.left)
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.conv(x)[..., :-self.left]
        y = self.norm(y.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(y))


class TiltTCN(nn.Module):
    def __init__(self, context, width, physical_dim, dropout):
        super().__init__()
        depth = int(np.ceil(np.log2((context + 1) / 2)))
        self.stem = nn.Conv1d(6, width, 1)
        self.blocks = nn.Sequential(*(CausalBlock(width, 2**i, dropout) for i in range(depth)))
        self.out = nn.Sequential(nn.Linear(width + physical_dim, 48), nn.GELU(), nn.Dropout(dropout), nn.Linear(48, 3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, signal, physics):
        representation = self.blocks(self.stem(signal.transpose(1, 2)))[:, :, -1]
        return 0.05 * self.out(torch.cat((representation, physics), axis=-1))


def windows(values, at, length):
    history = at[:, None] - np.arange(length - 1, -1, -1, dtype=np.int64)
    return np.ascontiguousarray(values[np.maximum(0, history)] / SCALE, dtype=np.float32)


def prepare(data, config):
    base = learning.prepare(data, config)
    at = block_indices(len(data["windows"]), config["update_hz"])
    enough_history = (at + 1 >= config["context"])
    aligned = ("matrix", "target", "train", "ready", "q", "up", "acc", "gyr", "rest")
    prepared = {**base, **{key: base[key][enough_history] for key in aligned}}
    prepared["history"] = windows(data["windows"], at[enough_history], config["context"])
    return prepared


def normalizer(records, key, axes):
    usable = [row[key][row["train"]] for row in records]
    if not all(len(x) for x in usable):
        raise ValueError("A TCN training record lacks labeled complete history")
    mean = np.mean([x.mean(axis=axes, dtype=np.float64) for x in usable], axis=0)
    square = np.mean([np.square(x, dtype=np.float64).mean(axis=axes) for x in usable], axis=0)
    return mean.astype(np.float32), np.sqrt(np.maximum(square - mean**2, 9e-6)).astype(np.float32)


def normalized(signal, physical, normal):
    return (((signal - normal[0]) / normal[1]).clamp(-12, 12),
            ((physical - normal[2]) / normal[3]).clamp(-12, 12))


def fit(records, config, device):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    arrays = (*normalizer(records, "history", (0, 1)), *normalizer(records, "matrix", 0))
    normal = tuple(torch.as_tensor(value, device=device) for value in arrays)
    x, p, up, y = [torch.as_tensor(np.concatenate([r[key][r["train"]] for r in records]), device=device)
                   for key in ("history", "matrix", "up", "target")]
    count_by_record = [int(r["train"].sum()) for r in records]
    order_weights = torch.as_tensor(np.concatenate([np.full(n, 1 / n, dtype=np.float32) for n in count_by_record]), device=device)
    model = TiltTCN(config["context"], config["channels"], p.shape[1], 0.10).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    horizon = config.get("scheduler_horizon_epochs", config["epochs"])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, horizon, config["lr"] / 5)
    starting = perf_counter()
    history, epoch_times, rates = [], [], []
    for epoch in range(config["epochs"]):
        epoch_start = perf_counter()
        rates.append(float(opt.param_groups[0]["lr"]))
        model.train()
        running = torch.zeros((), device=device)
        draws = torch.multinomial(order_weights, config["steps"] * config["batch_size"], replacement=True)
        for idx in draws.split(config["batch_size"]):
            xs, ps = normalized(x[idx], p[idx], normal)
            opt.zero_grad(set_to_none=True)
            prediction = model(xs, ps)
            vertical = up[idx]
            prediction = prediction - vertical * (prediction * vertical).sum(-1, keepdim=True)
            loss = F.smooth_l1_loss(prediction / .05, y[idx] / .05, beta=0.25)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            running += loss.detach()
        if epoch < horizon:
            schedule.step()
        value = float(running.item() / config["steps"])
        if not np.isfinite(value):
            raise ValueError(f"TCN loss diverged at epoch {epoch + 1}")
        history.append(value)
        epoch_times.append(perf_counter() - epoch_start)
    state = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
    for key, value in zip(("signal_mean", "signal_std", "physical_mean", "physical_std"), arrays, strict=True):
        state[key] = value
    return state, {"train_loss": history, "parameters": sum(p.numel() for p in model.parameters()),
                   "training_updates": sum(count_by_record), "training_seconds": perf_counter() - starting,
                   "epoch_seconds": epoch_times, "learning_rates": rates, "scheduler_horizon_epochs": horizon}


def make_model(state, config, device):
    keys = ("signal_mean", "signal_std", "physical_mean", "physical_std")
    model = TiltTCN(config["context"], config["channels"], len(state["physical_mean"]), 0.10).to(device)
    metadata_keys = {*keys, "installation_quaternion_wxyz"}
    model.load_state_dict({key: torch.as_tensor(value) for key, value in state.items() if key not in metadata_keys})
    return model.eval(), tuple(torch.as_tensor(state[key], device=device) for key in keys)


@torch.inference_mode()
def inference(part, model, normal, config, device):
    prediction = []
    for start in range(0, len(part["history"]), 512):
        stop = start + 512
        x = torch.as_tensor(part["history"][start:stop], device=device)
        p = torch.as_tensor(part["matrix"][start:stop], device=device)
        prediction.append(model(*normalized(x, p, normal)).cpu().numpy())
    raw = np.concatenate(prediction)
    return learning.constrain_delta(part, raw, config)


def frame_directions(part, config, built, device, delay_ms=0.):
    model, normal = built
    delta = inference(part, model, normal, config, device)
    corrected = unit(Z + rotate(unit(part["q"]), delta))
    delay = int(np.ceil(delay_ms / (1000 * part["dt"])))
    active = np.searchsorted(part["ready"] + delay, part["index"], side="right")
    return np.vstack((Z[None, :], corrected))[active], part["ref_world"]


def evaluate(part, config, built, device, delay_ms=0.):
    return rmse(*frame_directions(part, config, built, device, delay_ms))


def run(item):
    started_at = perf_counter()
    config = item["config"]
    sequence_names = names("dev")
    expected = baseline_reference()
    device = torch.device("cuda")
    mount = config.get("installation")
    raw = {name: learning.attach_native(name, config) for name in sequence_names} if mount else None
    data = {name: prepare(fetch_record(name, config["parameters"]), config) for name in sequence_names} if not mount else {}
    outputs = {"device": [], "motion": []}
    evidence, audits = [], {}
    for axis, label, held in learning.folds(sequence_names):
        prepare_started = perf_counter()
        training = [name for name in sequence_names if name not in held]
        q = None
        if mount:
            q = learning.installation_fit([raw[name] for name in training], config)
            data = {name: prepare(learning.apply_install(part, q, include_reference=name in held), config)
                    for name, part in raw.items()}
        audits.update({name: part["training_context"] for name, part in data.items()})
        fit_started = perf_counter()
        state, log = fit([data[n] for n in training], config, device)
        score_started = perf_counter()
        model = make_model(state, config, device)
        rows = [{"name": n, "fold": label, "frames": len(data[n]["index"]), "baseline_deg": expected[n],
                 "rmse_deg": evaluate(data[n], config, model, device)} for n in held]
        timing = {"prepare_seconds": fit_started - prepare_started, "fit_seconds": score_started - fit_started,
                  "score_seconds": perf_counter() - score_started}
        scores = aggregate(rows)
        print(axis, label, scores["macro_rmse_deg"], timing, log["train_loss"], flush=True)
        outputs[axis].extend(rows)
        evidence.append({"axis": axis, "fold": label, "training": training, "validation": held,
                         "installation_quaternion_wxyz": q.tolist() if q is not None else None,
                         "validation_summary": scores, "fit": log, "timing": timing})
    metrics = {axis: {**aggregate(rows), "per_device": by_device(rows)} for axis, rows in outputs.items()}
    metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"] + metrics["motion"]["macro_rmse_deg"]) / 2.
    timing = {key: sum(fold["timing"][key] for fold in evidence) for key in ("prepare_seconds", "fit_seconds", "score_seconds")}
    timing["total_wall_seconds"] = perf_counter() - started_at
    return {"dev_metrics": metrics, "sequences": outputs, "folds": evidence, "mask_version": VERSION, "timing": timing,
            "training_context_audit": audits,
            "inference_inputs": ["causal 100 Hz six-axis history", "FullVQF state at the update"]}
