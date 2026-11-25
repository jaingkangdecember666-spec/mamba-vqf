"""Shared supervised data loading and full-sequence tilt scoring."""

import json

import numpy as np
import torch

from run_pilot import tilt_error
from tilt_data import EST_UP
from tilt_model import TiltHead, contexts, normalize, physical_features, rotate


def read_updates(artifacts, name):
    with np.load(artifacts / "prepared" / f"{name}.npz", allow_pickle=False) as data:
        updates = data["updates"]
        physical = physical_features(contexts(data["windows"], data["update_blocks"]),
                                     data["base_up"][updates], data["bias"], data["rest"])
        return physical, data["reference"][updates], data["train_ok"]


def open_embedding(artifacts, name):
    return np.load(artifacts / "features" / f"{name}.npy", mmap_mode="r", allow_pickle=False)


@torch.inference_mode()
def predict_updates(head, physical, embedding, device):
    updates = []
    for start in range(0, len(physical), 1024):
        end = start + 1024
        inputs = torch.as_tensor(physical[start:end], device=device)
        auxiliary = torch.as_tensor(embedding[start:end].copy(), device=device) if embedding is not None else None
        updates.append(head(inputs, auxiliary).cpu().numpy())
    return np.concatenate(updates)


def record_predictions(artifacts, name, heads, device):
    physical, _, _ = read_updates(artifacts, name)
    embedding = open_embedding(artifacts, name) if any(head.embedding_width for head in heads.values()) else None
    return {key: predict_updates(head, physical, embedding if head.embedding_width else None, device)
            for key, head in heads.items()}


def score_attitudes(base, reference_up, score, update_quaternion, updates, deltas, candidates, dt, latency_ms=0.):
    """Score native frames, holding each available world correction until the next one."""
    indices = np.flatnonzero(score)
    if len(indices) == 0:
        raise ValueError("No eligible scoring frames")
    reference = rotate(normalize(base[indices]), reference_up[indices])
    update_quaternion = normalize(update_quaternion)

    def rmse(up):
        return float(np.sqrt(np.mean(np.square(tilt_error(up, reference)))))

    baseline = rmse(np.broadcast_to(EST_UP, reference.shape))
    scores = {"full_vqf": baseline}
    for key, (method, gain, delayed) in candidates.items():
        if gain == 0:
            scores[key] = baseline
            continue
        delay = int(np.ceil(latency_ms / (dt * 1000))) if delayed else 0
        active = np.searchsorted(updates + delay, indices, side="right")
        up_world = normalize(EST_UP + rotate(update_quaternion, deltas[method] * gain))
        scores[key] = rmse(np.concatenate((EST_UP[None, :], up_world), axis=0)[active])
    return scores


def full_record_scores(artifacts, name, deltas, candidates, latency_ms=0.):
    with np.load(artifacts / "prepared" / f"{name}.npz", allow_pickle=False) as data:
        score = data["score"]
        base = data["base"]
        updates = data["updates"]
        scores = score_attitudes(base, data["reference"], score, base[updates], updates,
                                 deltas, candidates, data["dt"].item(), latency_ms)
    return {"recording": name, "scoring_frames": int(score.sum()), "correction_updates": len(updates),
            "rmse_deg": scores}


def macro_scores(records):
    return {key: float(np.mean([row["rmse_deg"][key] for row in records]))
            for key in records[0]["rmse_deg"]}


def save_json(path, data):
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(path)


def load_head(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    head = TiltHead(state["physical_mean"], state["physical_std"], saved["embedding_width"])
    head.load_state_dict(state)
    return head.to(device).eval()
