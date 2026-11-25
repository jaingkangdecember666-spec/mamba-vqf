"""Cache frozen, causal TimesFM embeddings for one or more SHICE splits."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from tilt_data import ARTIFACTS, GROUPS, ROOT, partition
from tilt_model import contexts, encode, load_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).resolve().parent / "checkpoint")
    parser.add_argument("--splits", nargs="+", choices=GROUPS, default=("train", "valid"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    torch.set_num_threads(2)
    device = torch.device("cuda")
    model = load_backbone(args.checkpoint, device)
    width = model.transformer_config.transformer.model_dims
    folder = args.artifacts / "features"
    folder.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    for split in args.splits:
        for name in partition(args.root, split):
            with np.load(args.artifacts / "prepared" / f"{name}.npz", allow_pickle=False) as data:
                windows, blocks = data["windows"], data["update_blocks"]
            shape = (len(blocks), 6, width)
            target = folder / f"{name}.npy"
            if target.is_file():
                cached = np.load(target, mmap_mode="r", allow_pickle=False)
                if cached.shape != shape or cached.dtype != np.float16:
                    raise ValueError(f"Unexpected cached feature layout in {target}")
                continue
            staging = target.with_suffix(".tmp")
            result = np.lib.format.open_memmap(staging, mode="w+", dtype=np.float16, shape=shape)
            for start in range(0, len(blocks), args.batch_size):
                batch = blocks[start:start + args.batch_size]
                features = encode(model, contexts(windows, batch), device).cpu().numpy()
                if not np.isfinite(features).all():
                    raise ValueError(f"Non-finite embeddings in {name}, batch {start}")
                result[start:start + len(batch)] = features
            result.flush()
            del result
            staging.replace(target)
            print(f"{split} {name}: {len(blocks)} windows", flush=True)
    print(f"Feature extraction finished in {perf_counter() - started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
