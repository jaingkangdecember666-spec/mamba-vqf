"""Run only the externally trained Mamba v1 and TCN pairing for batch 1."""

import csv
import inspect
import json
import os
import sys
from dataclasses import dataclass
from time import perf_counter

from freeze_split import DATA, HERE, digest, stamp, write

sys.path[:0] = [str(HERE / ".deps"), str(HERE.parent)]

import numpy as np
import torch
import transformers
from torch.nn import functional as F
from vqf import VQF

from evaluate_external_tilt import broad_records, input_blocks, reference_direction, sassari_records
from models import MambaMixer, ResidualHead, tangent
from run_pilot import body_up
from tilt_data import EST_UP, causal_fill
from tilt_eval import score_attitudes


SEED = 1729
WINDOW = 64
HOP = 10
EPOCHS = 6
STEPS = 32
BATCH = 96
CAP = 6.
KINDS = ("mamba_v1", "tcn")
DOMAINS = ("BROAD", "Sassari")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class Prepared:
    domain: str
    item: object
    base: np.ndarray
    up: np.ndarray
    reference: np.ndarray
    score: np.ndarray
    index: np.ndarray
    scheduled: np.ndarray
    histories: np.ndarray
    eligible_training: np.ndarray


def causal_inputs(imu, frequency):
    valid = np.isfinite(imu).all(axis=1)
    filled = causal_fill(imu, valid)
    base = VQF(1. / frequency).updateBatch(np.ascontiguousarray(filled[:, 3:], dtype=np.float64),
                                           np.ascontiguousarray(filled[:, :3], dtype=np.float64))
    signal, last, available = input_blocks(filled, frequency)
    up = body_up(base["quat6D"][last], EST_UP)
    features = np.column_stack((signal, up, base["bias"][last], base["restDetected"][last])).astype(np.float32)
    blocks = np.arange(((WINDOW + HOP - 1) // HOP) * HOP - 1, len(signal), HOP)
    gather = blocks[:, None] - np.arange(WINDOW - 1, -1, -1)
    observed = np.r_[0, np.cumsum(~valid, dtype=np.int64)]
    ends = last[blocks] + 1
    start_blocks = blocks + 1 - WINDOW
    starts = np.r_[0, last + 1][start_blocks]
    all_observed = (observed[ends] == observed[starts]) & valid[available[blocks]]
    return base["quat6D"], last[blocks], available[blocks], features[gather], valid, all_observed


def prepare(domain, item):
    base, index, scheduled, histories, valid, complete = causal_inputs(item.imu, item.frequency)
    reference, label_ok = reference_direction(item.quaternion)
    score = valid & label_ok & item.eligible
    require(len(histories) > 0 and score.any(), f"No usable data in {domain}/{item.name}")
    return Prepared(domain, item, base, body_up(base[index], EST_UP).astype(np.float32), reference, score, index,
                    scheduled, histories, complete & score[scheduled])


def load(split):
    expected = {(row["domain"], row["name"]): row for row in split["records"]}
    for filename, sha in split["source_file_sha256"].items():
        from pathlib import Path
        require(digest(Path(filename)) == sha, f"Frozen source changed: {filename}")
    from freeze_split import BROAD
    require(digest(BROAD / "trials.json") == split["broad_trials_sha256"], "BROAD trial metadata changed")
    result = {"train": [], "validation": []}
    for domain, source in zip(DOMAINS, (broad_records, sassari_records), strict=True):
        for item in source(DATA):
            row = expected.pop((domain, item.name))
            part = prepare(domain, item)
            result[row["split"]].append(part)
            print("prepared", domain, item.name, row["split"], len(part.scheduled), int(part.score.sum()), flush=True)
    require(not expected, f"Unmatched split entries: {expected}")
    return result


def moments(training):
    domains = []
    for domain in DOMAINS:
        rows = []
        for p in training:
            if p.domain != domain or not p.eligible_training.any():
                continue
            x = p.histories[p.eligible_training].astype(np.float64)
            rows.append((x.mean(axis=(0, 1)), np.square(x).mean(axis=(0, 1))))
        require(rows, f"No training windows for {domain}")
        domains.append(np.mean(rows, axis=0))
    mean, square = np.mean(domains, axis=0)
    return mean.astype(np.float32), np.sqrt(np.maximum(square - mean**2, 1e-6)).astype(np.float32)


def normalized(x, mean, std):
    return ((x - mean) / std).clamp(-12, 12)


def make_batches(training, device):
    arrays = [[], [], []]
    pools = {domain: [] for domain in DOMAINS}
    first = 0
    for p in training:
        use = p.eligible_training
        n = int(use.sum())
        if not n:
            continue
        arrays[0].append(p.histories[use])
        arrays[1].append(p.up[use])
        arrays[2].append(p.reference[p.scheduled[use]].astype(np.float32))
        pools[p.domain].append((first, n))
        first += n
    rng = np.random.default_rng(SEED)
    schedule = []
    for _ in range(EPOCHS * STEPS):
        ids = []
        for domain in DOMAINS:
            seq = rng.integers(len(pools[domain]), size=BATCH // 2)
            ids.extend(pools[domain][i][0] + rng.integers(pools[domain][i][1]) for i in seq)
        schedule.append(rng.permutation(ids))
    matrix = np.stack(schedule).astype(np.int64)
    import hashlib
    return (tuple(torch.from_numpy(np.concatenate(x)).to(device) for x in arrays),
            torch.from_numpy(matrix).to(device), hashlib.sha256(matrix.astype("<i8").tobytes()).hexdigest())


@torch.inference_mode()
def score(model, validation, mean, std, device, name):
    model.eval()
    rows = []
    for p in validation:
        updates = []
        for first in range(0, len(p.histories), 256):
            end = first + 256
            x = torch.from_numpy(p.histories[first:end]).to(device)
            up = torch.from_numpy(p.up[first:end]).to(device)
            updates.append(tangent(model(normalized(x, mean, std)), up, CAP).cpu().numpy())
        delta = np.concatenate(updates)
        require(np.isfinite(delta).all(), f"Nonfinite output on {p.item.name}")
        scores = score_attitudes(p.base, p.reference, p.score, p.base[p.index], p.scheduled,
                                 {name: delta}, {name: (name, 1., False)}, 1. / p.item.frequency)
        rows.append({"domain": p.domain, "sequence": p.item.name, "frames": int(p.score.sum()), "rmse_deg": scores})
    return rows


def macro(rows, key):
    return float(np.mean([row["rmse_deg"][key] for row in rows]))


@torch.inference_mode()
def causal_check(model, part, mean, std, device):
    model.eval()
    x = normalized(torch.from_numpy(part.histories[:4]).to(device), mean, std)
    stop = 37
    changed = x.clone()
    changed[:, stop:] *= -5
    full = model(x, all_steps=True)[:, :stop]
    prefix = model(x[:, :stop], all_steps=True)
    changed = model(changed, all_steps=True)[:, :stop]
    gap = float((full - prefix).abs().max())
    future = float((full - changed).abs().max())
    require(max(gap, future) <= 2e-5, "Model prefix depends on future input")
    return {"input_windows": len(x), "prefix_blocks": stop, "prefix_max_abs": gap, "future_mutation_max_abs": future}


def fit(kind, model, arrays, batches, validation, mean, std, device):
    started = perf_counter()
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history, best, best_epoch = [], float("inf"), None
    state = None
    x, up, target = arrays
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for ids in batches[epoch * STEPS:(epoch + 1) * STEPS]:
            opt.zero_grad(set_to_none=True)
            delta = tangent(model(normalized(x[ids], mean, std)), up[ids], CAP)
            direction = F.normalize(up[ids] + delta)
            loss = (direction - target[ids]).square().sum(dim=-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            losses.append(loss.detach())
        rows = score(model, validation, mean, std, device, kind)
        metric = macro(rows, kind)
        train_loss = float(torch.stack(losses).mean())
        require(np.isfinite(train_loss) and np.isfinite(metric), "Nonfinite training statistics")
        history.append({"epoch": epoch + 1, "optimizer_steps": (epoch + 1) * STEPS,
                        "train_mean_squared_direction_distance": train_loss, "validation_macro_rmse_deg": metric})
        print(kind, history[-1], flush=True)
        if metric < best:
            best, best_epoch = metric, epoch + 1
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    dest = HERE / f"{kind}.pt"
    temp = dest.with_suffix(".tmp")
    torch.save({"model": kind, "state_dict": state, "mean": mean.cpu(), "std": std.cpu(),
                "seed": SEED, "best_epoch": best_epoch, "cap_deg": CAP,
                "window": WINDOW, "split_sha256": digest(HERE / "external-split.json")}, temp)
    temp.replace(dest)
    restored = torch.load(dest, weights_only=True)
    model.load_state_dict(restored["state_dict"])
    rows = score(model, validation, mean, std, device, kind)
    require(abs(macro(rows, kind) - best) <= 1e-6, "Saved weights failed score reproduction")
    return rows, {"seconds": perf_counter() - started, "history": history, "selected_epoch": best_epoch,
                  "executed_optimizer_steps": EPOCHS * STEPS, "selected_optimizer_step": best_epoch * STEPS,
                  "weights": dest.name, "weights_sha256": digest(dest),
                  "causality": causal_check(model, validation[0], mean, std, device)}


def main():
    require(not (HERE / "results.json").exists(), "Batch 1 already finished")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    require(torch.cuda.is_available(), "Expected CUDA on FZR-MiniPC")
    models = {}
    for kind in KINDS:
        torch.manual_seed(SEED)
        models[kind] = ResidualHead(kind)
    counts = {k: sum(p.numel() for p in m.parameters() if p.requires_grad) for k, m in models.items()}
    relative = abs(counts["mamba_v1"] - counts["tcn"]) / counts["mamba_v1"]
    require(relative < .08, f"Model capacities differ by {relative:.1%}")
    protocol = {"started_at": stamp(), "pid": os.getpid(), "device": torch.cuda.get_device_name(device),
                "split_sha256": digest(HERE / "external-split.json"), "seed": SEED, "models": counts,
                "relative_parameter_difference": relative, "epochs": EPOCHS, "steps_per_epoch": STEPS,
                "batch_size": BATCH, "lr": .001, "weight_decay": .01, "optimizer": "AdamW",
                "capacity": {"mamba_v1": "24 channels, one MambaMixer, state 8, expansion 2, kernel 4",
                             "tcn": "20 channels, six causal kernel-2 blocks with dilation 1,2,4,8,16,32"},
                "mamba_implementation": {"transformers_version": transformers.__version__,
                                         "module_sha256": digest(HERE / ".deps" / "transformers/models/mamba/modeling_mamba.py"),
                                         "class": MambaMixer.__qualname__, "backend": "Transformers bundled PyTorch sequential selective scan"},
                "input": {"window_blocks": WINDOW, "block_hz": 100, "update_hz": 10,
                          "channels": "six IMU block means, VQF gravity xyz, gyro bias xyz, rest",
                          "max_residual_deg": CAP, "target": "optical reference gravity at causal release, squared unit-vector distance"},
                "fit": "Training windows require observed packets throughout and eligible reference at release. Train-only feature moments and equal-domain, equal-record sampling. Fixed budget, select earliest best external-validation epoch.",
                "score": "Default FullVQF and corrections at identical native scored indices; BROAD movement mask, Sassari same-row Qs, finite IMU/reference, no extra warm-up; equal sequence mean of angular RMSE.",
                "reserved": ["TUM-VI", "all SHICE"], "no_calibration": True}
    protocol_path = HERE / "run-protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        for key in old.keys() - {"started_at", "pid"}:
            require(old[key] == protocol[key], f"Retry changed frozen protocol at {key}")
    else:
        write(protocol_path, protocol)
    print("protocol", protocol, flush=True)
    split = json.loads((HERE / "external-split.json").read_text(encoding="utf-8"))
    data = load(split)
    validation = data["validation"]
    train = data["train"]
    coverage = [{"domain": p.domain, "sequence": p.item.name, "split": s, "native_frames": len(p.score),
                 "score_frames": int(p.score.sum()), "correction_updates": len(p.scheduled),
                 "eligible_training_windows": int(p.eligible_training.sum()) if s == "train" else None}
                for s, parts in data.items() for p in parts]
    mean, std = (torch.from_numpy(v).to(device) for v in moments(train))
    arrays, batches, schedule_sha = make_batches(train, device)
    write(HERE / "fit-scope.json", {"datasets": coverage, "training_windows": len(arrays[0]),
                                   "training_batch_schedule_sha256": schedule_sha,
                                   "input_mean": mean.tolist(), "input_std": std.tolist()})
    probe = validation[0]
    stop = len(probe.item.imu) // 2
    _, _, prefix_rows, prefix_features, _, _ = causal_inputs(probe.item.imu[:stop], probe.item.frequency)
    np.testing.assert_array_equal(prefix_rows, probe.scheduled[:len(prefix_rows)])
    np.testing.assert_allclose(prefix_features, probe.histories[:len(prefix_rows)], atol=1e-6, rtol=0.)
    scores, info = {}, {}
    for kind in KINDS:
        torch.manual_seed(SEED)
        scores[kind], info[kind] = fit(kind, models[kind], arrays, batches, validation, mean, std, device)
        write(HERE / f"{kind}-validation.json", {"training": info[kind], "sequences": scores[kind]})
    rows = []
    for a, b in zip(*(scores[k] for k in KINDS), strict=True):
        require((a["domain"], a["sequence"], a["frames"], a["rmse_deg"]["full_vqf"]) ==
                (b["domain"], b["sequence"], b["frames"], b["rmse_deg"]["full_vqf"]), "Mismatched scored frames")
        rows.append({**a, "rmse_deg": {**a["rmse_deg"], **b["rmse_deg"]}})
    summary = {}
    for domain in (*DOMAINS, "all"):
        subset = [r for r in rows if domain == "all" or r["domain"] == domain]
        summary[domain] = {"sequences": len(subset), "frames": sum(r["frames"] for r in subset),
                           "macro_rmse_deg": {k: macro(subset, k) for k in ("full_vqf", *KINDS)}}
    with (HERE / "validation-sequences.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "sequence", "frames", "full_vqf", *KINDS])
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in ("domain", "sequence", "frames")}, **r["rmse_deg"]} for r in rows)
    write(HERE / "results.json", {"completed_at": stamp(), "protocol_sha256": digest(protocol_path),
                                 "training": info, "summary": summary, "sequences": rows, "raw_prefix": {
                                     "sequence": probe.item.name, "native_frames": stop, "same_ready_and_features": True},
                                 "stage": "batch_1_complete", "shice_scores_computed": False})
    print(summary, flush=True)


if __name__ == "__main__":
    main()
