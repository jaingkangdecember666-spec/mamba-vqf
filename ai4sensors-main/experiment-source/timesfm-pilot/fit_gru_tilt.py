"""Fit the RIANN-style GRU on SHICE R1–R4 and select the checkpoint on R5."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from gru_data import OUTPUT, SHICE_HZ, causal_resample
from gru_tilt_model import RATES, AttitudeGRU, augment, infer_sequences, tilt_loss, with_dt
from run_pilot import tilt_error
from tilt_data import EST_UP, EXCLUDED, ROOT, partition
from tilt_eval import save_json


SEED = 1729
WINDOW = 768
CHUNK = 256
BATCH = 48


class TrainingSequence:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            imu, self.ref, self.score = data["imu"], data["ref"], data["score"]
            self.dt = data["dt"].item()
            usable = data["context"] & (np.arange(len(imu)) * self.dt >= 10.)
        self.stats = np.square(imu.astype(np.float64)).sum(axis=0)
        self.sums = np.vstack((np.zeros((1, 6)), np.cumsum(imu, axis=0, dtype=np.float64)))
        changes = np.flatnonzero(np.diff(np.r_[False, usable, False]))
        self.starts, self.stops = changes[::2], changes[1::2]

    def sample(self, rng, rate):
        edges = np.floor(np.arange(WINDOW + 1) / (rate * self.dt) + 1e-9).astype(np.int64)
        legal = np.maximum(self.stops - self.starts - edges[-1] + 1, 0)
        total = legal.sum()
        if not total:
            return None
        cumul = np.cumsum(legal)
        draw = rng.integers(total)
        span = np.searchsorted(cumul, draw, side="right")
        start = self.starts[span] + draw - (cumul[span - 1] if span else 0)
        edges += start
        imu = ((self.sums[edges[1:]] - self.sums[edges[:-1]]) / np.diff(edges)[:, None]).astype(np.float32)
        label_index = edges[1:] - 1
        valid = self.score[label_index].copy()
        valid[:int(np.ceil(rate * 0.35))] = False
        if valid.sum() < WINDOW // 2:
            return None
        return with_dt(imu, 1. / rate), self.ref[label_index], valid


def training_batch(inputs, rng, device):
    items = []
    for _ in range(BATCH):
        for _ in range(100):
            item = inputs[rng.integers(len(inputs))].sample(rng, rng.choice(RATES))
            if item is not None:
                items.append(item)
                break
        else:
            raise ValueError("Could not find enough valid SHICE training windows")
    return tuple(torch.as_tensor(np.stack(parts), device=device) for parts in zip(*items))


def hold_up(prediction, updates, native_frames):
    active = np.searchsorted(updates, native_frames, side="right")
    return np.concatenate((EST_UP[None, :].astype(np.float32), prediction), axis=0)[active]


def read_shice(output, names):
    items = []
    for name in names:
        with np.load(output / "data" / f"{name}.npz", allow_pickle=False) as data:
            imu, ref, score, dt = data["imu"], data["ref"], data["score"], data["dt"].item()
        readings, updates = causal_resample(imu, dt, SHICE_HZ)
        items.append((with_dt(readings, 1. / SHICE_HZ), updates, np.flatnonzero(score), ref[score]))
    return items


def validate(model, validation, device):
    predictions = infer_sequences(model, [row[0] for row in validation], device)
    scores = []
    for prediction, (_, updates, indices, reference) in zip(predictions, validation, strict=True):
        errors = tilt_error(hold_up(prediction, updates, indices), reference)
        scores.append(float(np.sqrt(np.square(errors, dtype=np.float64).mean())))
    return scores


def fit(inputs, validation, epochs, batches, device):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    energies = np.sum([sequence.stats for sequence in inputs], axis=0)
    n = sum(len(sequence.ref) for sequence in inputs)
    acc_rms = np.sqrt(energies[:3].sum() / (n * 3))
    gyr_rms = np.sqrt(energies[3:].sum() / (n * 3))
    periods = 1. / RATES
    model = AttitudeGRU([0.] * 6 + [periods.mean()], [acc_rms] * 3 + [gyr_rms] * 3 + [periods.std()]).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.002, weight_decay=0.001, decoupled_weight_decay=True)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.00015)
    scaler = torch.amp.GradScaler("cuda")
    slow = [param.detach().clone() for param in model.parameters()]
    best, best_epoch, best_scores, best_state = float("inf"), 0, None, None
    history = []
    trace_started = perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total = torch.zeros((), device=device)
        for step in range(batches):
            x, y, mask = training_batch(inputs, rng, device)
            x, y = augment(x, y)
            optimizer.zero_grad(set_to_none=True)
            hidden = None
            batch_loss = torch.zeros((), device=device)
            for start in range(0, WINDOW, CHUNK):
                interval = slice(start, start + CHUNK)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    q, hidden = model(x[:, interval].contiguous(), hidden)
                hidden = hidden.detach()
                loss = tilt_loss(q, y[:, interval], mask[:, interval]) / (WINDOW // CHUNK)
                scaler.scale(loss).backward()
                batch_loss += loss.detach()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(optimizer)
            scaler.update()
            if ((epoch - 1) * batches + step + 1) % 6 == 0:
                with torch.no_grad():
                    for average, param in zip(slow, model.parameters(), strict=True):
                        average.lerp_(param, 0.5)
                        param.copy_(average)
            total += batch_loss
        schedule.step()
        loss = (total / batches).item()
        if not np.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at epoch {epoch}")
        row = {"epoch": epoch, "train_loss_rad2": loss, "seconds": perf_counter() - trace_started}
        if epoch == 1 or epoch % 3 == 0 or epoch == epochs:
            scores = validate(model.eval(), validation, device)
            rmse = float(np.mean(scores))
            row["r5_macro_rmse_deg"] = rmse
            if rmse < best:
                best, best_epoch, best_scores = rmse, epoch, scores
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append(row)
        print(row, flush=True)
    if best_state is None:
        raise ValueError("No valid R5 checkpoint")
    return best_state, best, best_epoch, best_scores, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--save-dir", type=Path, help="Save a separate R5 candidate without replacing the primary run.")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--batches", type=int, default=24)
    args = p.parse_args()
    if min(args.epochs, args.batches) <= 0:
        p.error("Use positive epoch and batch counts")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    train = partition(args.root, "train")
    valid = partition(args.root, "valid")
    inputs = [TrainingSequence(args.output / "data" / f"{name}.npz") for name in train]
    validation = read_shice(args.output, valid)
    print(f"Training {len(train)} records, validating {len(valid)} records", flush=True)
    weights, best, epoch, scores, history = fit(inputs, validation, args.epochs, args.batches, device)
    state = {"state_dict": weights, "selected_epoch": epoch, "r5_macro_rmse_deg": best,
             "training_records": train, "validation_records": valid, "seed": SEED}
    destination = args.save_dir or args.output
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / "gru_head.tmp"
    torch.save(state, staging)
    staging.replace(destination / "gru.pt")
    report = {"training_records": train, "validation_records": valid, "excluded_records": sorted(EXCLUDED),
              "selected_epoch": epoch, "r5_macro_rmse_deg": best,
              "r5_per_sequence_rmse_deg": dict(zip(valid, scores, strict=True)),
              "r5_scoring_frames": sum(len(item[2]) for item in validation), "history": history,
              "rate_count": len(RATES), "rate_range_hz": [RATES[0], RATES[-1]], "shice_gru_hz": SHICE_HZ,
              "windows_per_epoch": BATCH * args.batches, "samples_per_window": WINDOW, "seed": SEED}
    save_json(destination / "validation.json", report)
    print(f"Selected epoch {epoch}: {best:.6f}° on R5", flush=True)


if __name__ == "__main__":
    main()
