"""Train two SHICE correction heads on R1–R4 and select them using R5 only."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from run_pilot import CHECKPOINT_REVISION
from tilt_data import ARTIFACTS, EXCLUDED, ROOT, partition
from tilt_eval import full_record_scores, macro_scores, open_embedding, read_updates, record_predictions, save_json
from tilt_model import TiltHead, residual_loss


SEED = 1729
GAINS = (0., 0.25, 0.5, 0.75, 1., 1.25)
BATCH_SIZE = 1024


def load_partition(artifacts, names, device):
    physical, embedding, target, labels = [], [], [], []
    counts = {}
    for index, name in enumerate(names):
        values, reference, valid = read_updates(artifacts, name)
        counts[name] = int(valid.sum())
        if not counts[name]:
            raise ValueError(f"No supervised update points for {name}")
        physical.append(values[valid])
        target.append(reference[valid])
        labels.append(np.full(counts[name], index, dtype=np.int64))
        embedding.append(open_embedding(artifacts, name)[valid])
    X, E, Y, sequence = (torch.as_tensor(np.concatenate(items), device=device)
                         for items in (physical, embedding, target, labels))
    count_tensor = torch.as_tensor(list(counts.values()), device=device)
    weight = len(X) / (len(names) * count_tensor[sequence])
    return X, E, Y, sequence, weight, counts


@torch.inference_mode()
def validation_rmse(model, validation):
    X, E, Y, sequence, _, counts = validation
    corrected = []
    for x, e in zip(X.split(BATCH_SIZE), E.split(BATCH_SIZE)):
        delta = model(x, e if model.embedding_width else None)
        corrected.append(F.normalize(x[:, :3] + delta, dim=1))
    prediction = torch.cat(corrected)
    cross = torch.linalg.cross(prediction, Y).norm(dim=1)
    angles = torch.rad2deg(torch.atan2(cross, (prediction * Y).sum(dim=1)))
    sums = torch.bincount(sequence, weights=angles.square(), minlength=len(counts))
    return (sums / torch.bincount(sequence)).sqrt().mean().item()


def train(method, train_data, validation, epochs, device):
    torch.manual_seed(SEED)
    X, E, Y, _, weight, _ = train_data
    width = E.shape[-1] if method == "timesfm" else 0
    model = TiltHead(X.mean(0), X.std(0, unbiased=False).clamp_min(0.003), width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_rmse, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = torch.zeros((), device=device)
        for batch in torch.randperm(len(X), device=device).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            x = X[batch]
            delta = model(x, E[batch] if width else None)
            loss = residual_loss(x[:, :3], Y[batch], delta) * weight[batch]
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.detach().sum()
        model.eval()
        rmse = validation_rmse(model, validation)
        record = {"epoch": epoch, "weighted_training_loss": epoch_loss.item() / len(X), "r5_update_rmse_deg": rmse}
        history.append(record)
        print(f"{method} epoch={epoch} train={record['weighted_training_loss']:.6f} R5={rmse:.6f}°", flush=True)
        if rmse < best_rmse - 1e-4:
            best_rmse, best_epoch = rmse, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch - best_epoch >= 5:
            break
    if best_state is None:
        raise ValueError(f"No finite validation loss for {method}")
    model.load_state_dict(best_state)
    return model.eval(), {"state_dict": best_state, "embedding_width": width}, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epoch count must be positive")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    started = perf_counter()
    train_names = partition(args.root, "train")
    valid_names = partition(args.root, "valid")
    train_data = load_partition(args.artifacts, train_names, device)
    validation = load_partition(args.artifacts, valid_names, device)
    print(f"Supervised windows: train={len(train_data[0])}, R5={len(validation[0])}", flush=True)

    methods = ("timesfm", "physical_only")
    heads, history, checkpoints, best_epochs = {}, {}, {}, {}
    for method in methods:
        heads[method], checkpoints[method], history[method], best_epochs[method] = train(
            method, train_data, validation, args.epochs, device,
        )
    candidates = {f"{method}@{gain:g}": (method, gain, False) for method in methods for gain in GAINS}
    scores = [full_record_scores(args.artifacts, name, record_predictions(args.artifacts, name, heads, device), candidates)
              for name in valid_names]
    macro = macro_scores(scores)
    selected = {method: min(GAINS, key=lambda gain: macro[f"{method}@{gain:g}"]) for method in methods}

    for method, checkpoint in checkpoints.items():
        path = args.artifacts / f"{method}_head.pt"
        staging = path.with_suffix(".tmp")
        torch.save({**checkpoint, "selected_gain": selected[method], "selected_epoch": best_epochs[method],
                    "backbone_revision": CHECKPOINT_REVISION if method == "timesfm" else None}, staging)
        staging.replace(path)
    report = {"training_records": train_names, "validation_records": valid_names, "excluded_records": sorted(EXCLUDED),
              "trained_updates": train_data[-1], "validation_updates": validation[-1], "seed": SEED,
              "r5_macro_rmse_deg": macro, "r5_records": scores, "gains": selected,
              "best_epochs": best_epochs, "history": history, "training_seconds": perf_counter() - started}
    save_json(args.artifacts / "validation.json", report)
    print({"R5": macro, "selected_gains": selected, "best_epochs": best_epochs}, flush=True)


if __name__ == "__main__":
    main()
