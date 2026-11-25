"""Retrain the original causal physical TiltHead on independent development folds."""

from time import perf_counter

import numpy as np
import torch

import learning
from common import BATCH, EXCLUDED, Z, aggregate, by_device, fingerprint, names, read_json, rmse, rotate, stamp, unit, write_json
from experiment import baseline_reference
from features import VERSION, block_indices, fetch_record, training_support
from fit_tilt import BATCH_SIZE
from tilt_eval import predict_updates
from tilt_model import RESIDUAL_SCALE, TiltHead, contexts, physical_features, residual_loss


def prepare(data, config, retain_augmentation=False):
    block = block_indices(len(data["windows"]), config["update_hz"])
    history = contexts(data["windows"], block)
    features = physical_features(history, data["up"][block], data["bias"][block], data["rest"][block])
    mask, audit = training_support(data, block, config["context"])
    result = {"matrix": features, "target": data["target"][block], "train": mask, "training_context": audit,
              "ready": data["ready"][block], "q": data["q"][block], "index": data["index"],
              "ref_world": data["ref_world"], "dt": float(data["dt"])}
    boost = config.get("train_dynamic_boost", 0.)
    if boost:
        speed = np.rad2deg(np.linalg.norm(history[:, -10:, 3:], axis=-1).mean(axis=1))
        result["relative_weight"] = 1. + boost * np.clip((speed - 1.) / 59., 0., 1.)
    if retain_augmentation and config.get("rotation_augmentation_deg"):
        selected = block[mask]
        result["augmentation"] = {"history": history[mask], "up": data["up"][selected],
                                  "bias": data["bias"][selected], "rest": data["rest"][selected]}
    return result


def device_groups(training_names, sequence_count):
    if training_names is None or len(training_names) != sequence_count:
        raise ValueError("Device-balanced loss needs the current training fold's recording names")
    groups = {}
    for index, name in enumerate(training_names):
        groups.setdefault(name.split("-")[0], []).append(index)
    return groups


def device_weights(sequence_weights, sizes, groups, masses, device):
    scale = np.zeros(len(sizes), dtype=np.float32)
    for key, indices in groups.items():
        scale[indices] = np.float32(masses[key] * len(sizes) / len(indices))
    weighted = sequence_weights * np.repeat(scale, sizes)
    return torch.as_tensor(weighted.astype(np.float32), device=device)


@torch.inference_mode()
def training_group_losses(head, inputs, target, sizes, groups):
    head.eval()
    losses = []
    for x, y in zip(inputs.split(BATCH_SIZE), target.split(BATCH_SIZE), strict=True):
        losses.append(residual_loss(x[:, :3], y, head(x)))
    per_record = [float(part.mean().item()) for part in torch.cat(losses).split(sizes)]
    return {name: float(np.mean([per_record[i] for i in indices])) for name, indices in groups.items()}


def rotate_training_examples(record, rng, max_angle):
    from scipy.spatial.transform import Rotation

    augmented = record["augmentation"]
    mask = record["train"]
    matrix = record["matrix"][mask].copy()
    target = record["target"][mask].copy()
    affected = rng.random(len(target)) < 0.5
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    degrees = rng.uniform(-max_angle, max_angle)
    rotation = Rotation.from_rotvec(np.deg2rad(degrees) * axis).as_matrix().astype(np.float32)
    selected = augmented["history"][affected]
    history = np.concatenate((selected[:, :, :3] @ rotation.T, selected[:, :, 3:] @ rotation.T), axis=2)
    matrix[affected] = physical_features(history, augmented["up"][affected] @ rotation.T,
                                         augmented["bias"][affected] @ rotation.T, augmented["rest"][affected])
    target[affected] = target[affected] @ rotation.T
    return matrix, target, {"rotation_angle_deg": float(degrees), "transformed_labels": int(affected.sum())}


def fit(records, config, device, training_names=None):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    sizes = [int(np.sum(p["train"])) for p in records]
    if not all(sizes):
        raise ValueError("At least one training sequence has no complete labeled context")
    x = torch.as_tensor(np.concatenate([p["matrix"][p["train"]] for p in records]), device=device)
    target = torch.as_tensor(np.concatenate([p["target"][p["train"]] for p in records]), device=device)
    if config.get("train_dynamic_boost"):
        per_record = [p["relative_weight"][p["train"]] for p in records]
        weight_array = np.concatenate([group / group.sum() * (sum(sizes) / len(records)) for group in per_record])
    else:
        weight_array = np.repeat(np.float32(sum(sizes)) / (len(records) * np.asarray(sizes)), sizes)
    mode = config.get("training_device_balance")
    masses, groups = None, None
    if mode:
        groups = device_groups(training_names, len(records))
        masses = {key: 1. / len(groups) for key in groups}
        weights = device_weights(weight_array, sizes, groups, masses, device)
    else:
        weights = torch.as_tensor(weight_array.astype(np.float32), device=device)
    balancing = []
    angular_scale = config.get("robust_error_deg")
    if angular_scale:
        squared_scale = (np.deg2rad(angular_scale) / RESIDUAL_SCALE) ** 2
    head = TiltHead(x.mean(0), x.std(0, unbiased=False).clamp_min(0.003), 0).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    history, augmentations, epoch_times = [], [], []
    angle = config.get("rotation_augmentation_deg")
    aug_rng = np.random.default_rng(config["seed"] + 40131) if angle else None
    for epoch in range(config["epochs"]):
        started_epoch = perf_counter()
        if angle:
            augmented = [rotate_training_examples(record, aug_rng, angle) for record in records]
            epoch_x = torch.as_tensor(np.concatenate([row[0] for row in augmented]), device=device)
            epoch_target = torch.as_tensor(np.concatenate([row[1] for row in augmented]), device=device)
            augmentations.append({"epoch": epoch+1, "training": [{"record": name, **row[2]}
                                  for name, row in zip(training_names, augmented, strict=True)]})
        else:
            epoch_x, epoch_target = x, target
        group_log = {"epoch": epoch+1, "device_objective_mass": masses.copy()} if mode else None
        head.train()
        epoch_loss = torch.zeros((), device=device)
        for batch in torch.randperm(len(x), device=device).split(BATCH_SIZE):
            optimizer.zero_grad(set_to_none=True)
            inputs = epoch_x[batch]
            delta = head(inputs)
            loss = residual_loss(inputs[:, :3], epoch_target[batch], delta)
            if angular_scale:
                loss = 2 * squared_scale * (torch.sqrt(1 + loss / squared_scale) - 1)
            loss = loss * weights[batch]
            loss.mean().backward()
            optimizer.step()
            epoch_loss += loss.detach().sum()
        value = epoch_loss.item() / len(x)
        if not np.isfinite(value):
            raise ValueError("Physical TiltHead training loss is not finite")
        history.append(value)
        if mode:
            if mode == "capped_worst" and epoch+1 < config["epochs"]:
                severity = training_group_losses(head, x, target, sizes, groups)
                worst = min(groups, key=lambda key: (-severity[key], key))
                multiplier = config["max_worst_device_multiplier"]
                normalizer = len(groups) - 1 + multiplier
                masses = {key: (multiplier if key == worst else 1.) / normalizer for key in groups}
                weights = device_weights(weight_array, sizes, groups, masses, device)
                group_log.update(train_device_residual_loss=severity, most_difficult_training_device=worst)
            balancing.append(group_log)
        epoch_times.append(perf_counter() - started_epoch)
    state = {key: value.detach().cpu().numpy().copy() for key, value in head.state_dict().items()}
    return state, {"weighted_training_loss": history, "training_labels": sum(sizes), "training_records": len(records),
                   "learnable_parameters": sum(p.numel() for p in head.parameters()), "epoch_seconds": epoch_times,
                   "learning_rates": [config["lr"]] * config["epochs"],
                   "rotation_jitter_training_only": augmentations if angle else None,
                   "group_balance_training_only": balancing if mode else None,
                   "record_count_per_training_device": {key: len(indices) for key, indices in groups.items()} if mode else None}


def make_model(state, device):
    weights = {k: torch.as_tensor(v) for k, v in state.items() if k != "installation_quaternion_wxyz"}
    head = TiltHead(weights["physical_mean"], weights["physical_std"], 0)
    head.load_state_dict(weights)
    return head.to(device).eval()


def frame_directions(data, config, model, device, delay_ms=0.):
    residual = predict_updates(model, data["matrix"], None, device)
    up = unit(Z + rotate(unit(data["q"]), config["gain"] * residual))
    delay = int(np.ceil(delay_ms / (1000 * data["dt"])))
    active = np.searchsorted(data["ready"] + delay, data["index"], side="right")
    return np.vstack((Z[None, :], up))[active], data["ref_world"]


def evaluate(data, config, model, device, delay_ms=0.):
    return rmse(*frame_directions(data, config, model, device, delay_ms))


def folds_score(config, plan):
    started_at = perf_counter()
    recordings = names("dev")
    expected = baseline_reference()
    device = torch.device("cuda")
    install = config.get("installation")
    raw = {name: learning.attach_native(name, config) if install else fetch_record(name, config["parameters"]) for name in recordings}
    prepared = {name: prepare(data, config, retain_augmentation=True) for name, data in raw.items()} if not install else {}
    audits = {name: row["training_context"] for name, row in prepared.items()}
    predictions, evidence = {}, []
    for axis, label, training, held in plan:
        prepare_started = perf_counter()
        q = None
        if install:
            q = learning.installation_fit([raw[name] for name in training], config)
            prepared = {name: prepare(learning.apply_install(raw[name], q, include_reference=name in held), config,
                                      retain_augmentation=name in training)
                        for name in training + held}
            audits.update({name: row["training_context"] for name, row in prepared.items()})
        fit_started = perf_counter()
        state, log = fit([prepared[name] for name in training], config, device, training)
        score_started = perf_counter()
        head = make_model(state, device)
        rows = [{"name": name, "fold": label, "frames": len(prepared[name]["index"]), "baseline_deg": expected[name],
                 "rmse_deg": evaluate(prepared[name], config, head, device)} for name in held]
        timing = {"prepare_seconds": fit_started - prepare_started, "fit_seconds": score_started - fit_started,
                  "score_seconds": perf_counter() - score_started}
        predictions.setdefault(axis, []).extend(rows)
        evidence.append({"axis": axis, "fold": label, "training": training, "validation": held, "fitting": log,
                         "validation_summary": aggregate(rows), "timing": timing,
                         "installation_quaternion_wxyz": q.tolist() if q is not None else None})
        print(label, evidence[-1]["validation_summary"]["macro_rmse_deg"], timing,
              log["weighted_training_loss"], flush=True)
    metrics = {axis: {**aggregate(rows), "per_device": by_device(rows)} for axis, rows in predictions.items()}
    if "device" in metrics and "motion" in metrics:
        metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"] + metrics["motion"]["macro_rmse_deg"]) / 2
    timing = {key: sum(fold["timing"][key] for fold in evidence) for key in ("prepare_seconds", "fit_seconds", "score_seconds")}
    timing["total_wall_seconds"] = perf_counter() - started_at
    return {"dev_metrics": metrics, "sequences": predictions, "folds": evidence, "training_context_audit": audits,
            "timing": timing,
            "mask_version": VERSION, "original_network": "tilt_model.TiltHead, embedding_width=0",
            "original_loss": "tilt_model.residual_loss", "historical_weights_used_in_fit": False,
            "inference_inputs": ["64 causal six-axis blocks", "FullVQF vertical vector, gyro bias, and rest state"]}


def run(item):
    records = names("dev")
    plan = [(axis, label, [name for name in records if name not in held], held)
            for axis, label, held in learning.folds(records)]
    if item["config"].get("training_device_balance") or item["config"].get("rotation_augmentation_deg"):
        from joint_holdout import joint_folds

        plan.extend(("joint_holdout", label, training, held) for label, training, held in joint_folds(records))
    return folds_score(item["config"], plan)


def schedule(identity, config, relation, hypothesis, batch):
    target = BATCH / "rounds" / identity / "round.json"
    sha = fingerprint(config)
    if target.is_file():
        if read_json(target)["config_sha256"] != sha:
            raise ValueError(f"Configuration already exists with a different hash: {identity}")
        return
    for found in (BATCH / "rounds").glob("*/round.json"):
        item = read_json(found)
        if item["config_sha256"] == sha:
            return item["round_id"]
    write_json(target, {"round_id": identity, "config": config, "config_sha256": sha, "seed": config["seed"],
                       "batch": batch, "relation": relation, "hypothesis": hypothesis,
                       "partitions": {"development": names("dev"), "excluded": list(EXCLUDED)},
                       "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                       "dev_metrics": None, "result": f"rounds/{identity}/result.json",
                       "stdout": f"rounds/{identity}/run.log"})
    return identity


def queue():
    base = {"kind": "legacy_head", "parameters": {}, "context": 64, "update_hz": 10, "epochs": 2,
            "gain": 0.75, "lr": 0.0003, "weight_decay": 0.01, "seed": 1729}
    desc = (
        ({}, "Match the historic physical head architecture, training recipe, two epochs, and gain"),
        ({"epochs": 6}, "Check whether more fitting hurts generalization on grouped development folds"),
        ({"gain": 0.45}, "Test attenuation of the same trained physical corrections"),
        ({"parameters": {"tauAcc": 0.2}}, "Ablate the FullVQF time constant below the original architecture"),
        ({"parameters": {"tauAcc": 0.2}, "installation": "all", "mount_shrink": 1.0}, "Fit the physical head over a fold-only mounting rotation"),
    )
    plans = []
    for n, (change, hypothesis) in enumerate(desc, 1):
        config = {**base, **change}
        identity = schedule(f"P{n:05d}", config, None, hypothesis, "legacy-causal-head-screen")
        plans.append({"round_id": identity or f"P{n:05d}", "hypothesis": hypothesis})
    write_json(BATCH / "legacy-causal-head-stage.json", {"planned_at": stamp(), "planned": plans,
               "unchanged_historical_control": "tilt-correction/physical_only_head.pt",
               "train_partition": "R1-R4 strictly grouped development", "initial_seed": 1729,
               "fitting_model_source": "tilt_model.py:TiltHead and residual_loss"})
    print(plans, flush=True)


if __name__ == "__main__":
    queue()
