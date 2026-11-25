"""One matched pair of frozen-backbone linear readout fits."""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C003 = CAMPAIGN / "C003"
sys.path.insert(0, str(C003))

import replay_bias as c3
import numpy as np
import torch
from scipy.io import loadmat
from tilt_eval import normalize, rotate
from torch.nn import functional as F


c2, base, intrinsic = c3.c2, c3.base, c3.intrinsic
C001, C002, DATA, PILOT, TRANSFER = c3.C001, c3.C002, c3.DATA, c3.PILOT, c3.TRANSFER
read, write, digest, stamp = c3.read, c3.write, c3.digest, c3.stamp
SEED, STEPS, BATCH, FRACTION = 1729, 160, 96, 32
LR, WD, GAIN = .0003, .01, .25
METHODS = ("full_vqf_deg", "c003_deg", "A_deg", "B_deg")


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_hash(values):
    sha = hashlib.sha256()
    for key, tensor in values.items():
        a = tensor.detach().cpu().numpy().astype("<f4", copy=False)
        sha.update(json.dumps([key, a.shape], separators=(",", ":")).encode("utf-8"))
        sha.update(a.tobytes())
    return sha.hexdigest()


def freeze_records(inv, split):
    source = []
    for row in split["records"]:
        if row["split"] != "train":
            continue
        path = Path(row["path"])
        relative = path.relative_to(DATA).as_posix()
        source.append({"domain": row["domain"], "record": row["name"], "group": row["group"],
                       "relative_data_path": relative, "file_sha256": split["source_file_sha256"][str(path)]})
    base.require(Counter(r["domain"] for r in source) == {"BROAD": 27, "Sassari": 12}, "External training inventory changed")
    yt = [r for r in inv["records"] if r["dataset"] == "YT_Server"]
    test = {r["group"] for r in yt if r["official_split"] == "test"}
    dev = set(inv["yt"]["selected_groups"])
    pool = defaultdict(list)
    for row in yt:
        if row["side"] == "external_training_candidate":
            base.require(row["official_split"] == "train" and row["group"] not in test | dev, "YT group eligibility changed")
            pool[row["group"]].append(row)
    groups = sorted(pool, key=lambda g: (text_hash("C004|YT_Server|" + g), g))[:24]
    base.require(len(test) == 169 and len(dev) == 12 and sum(map(len, pool.values())) == 910
                 and len(groups) == len(set(groups)) == 24 and not set(groups) & (test | dev),
                 "YT test/diagnostic exclusion failed")
    for group in groups:
        row = min(pool[group], key=lambda r: (text_hash("C004|YT_Server|train|" + r["record"]), r["record"]))
        source.append({"domain": "YT_Server", "record": row["record"], "group": group,
                       "relative_data_path": "YT_server/YT_server_200Hz.h5", "hdf5_path": row["source"],
                       "native_frames": row["native_frames"], "reference_status": "unknown_unverified_producer"})
    return source, {"candidate_records": 910, "test_records_preserved": 213, "test_prefixes": len(test),
                    "diagnostic_valid_prefixes": sorted(dev), "selected_prefixes": groups,
                    "test_overlap": sorted(set(groups) & test), "diagnostic_valid_overlap": sorted(set(groups) & dev),
                    "group_hash": "SHA256 UTF-8 C004|YT_Server|<prefix>",
                    "record_hash": "SHA256 UTF-8 C004|YT_Server|train|<record>"}


def train_raw(spec, yt, mats):
    domain, name = spec["domain"], spec["record"]
    if domain == "YT_Server":
        item = yt[name]
        clean, dt, score, reference, evidence = c2.c1.yt_record(item, spec["native_frames"])
        observed = np.ones(len(clean), dtype=bool)
        if evidence["finite_imu_frames"] != len(clean):
            observed = np.isfinite(item["acc"][:]).all(axis=1) & np.isfinite(item["gyr"][:]).all(axis=1)
        return clean, observed, dt, score, reference, evidence["source_field_sha256"]
    path = DATA / spec["relative_data_path"]
    base.require(digest(path) == spec["file_sha256"], f"Training file changed: {path}")
    if domain == "BROAD":
        mat = loadmat(path, variable_names=("imu_acc", "imu_gyr", "opt_quat", "movement", "sampling_rate"))
        raw = np.column_stack((mat["imu_acc"], mat["imu_gyr"]))
        reference, pose = base.reference_direction(mat["opt_quat"])
        eligible = mat["movement"].ravel().astype(bool)
        dt = 1. / float(mat["sampling_rate"].item())
    else:
        base.require(domain == "Sassari" and spec["group"] in ("slow", "fast"), "Unexpected source split")
        if path not in mats:
            mats[path] = loadmat(path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
        mat = mats[path]
        raw = mat[name.split("/")[1]][:, 1:7]
        reference, pose = base.reference_direction(mat["Qs"])
        eligible = np.ones(len(raw), dtype=bool)
        dt = .01
    observed = np.isfinite(raw).all(axis=1)
    return raw, observed, dt, pose & eligible & observed, reference, {"file": spec["file_sha256"]}


def inertial_replay(raw, observed, dt, cfg, training=False):
    source = raw.copy()
    source[~observed] = np.nan
    bias, active, diag, _ = c3.estimate(raw, observed, dt, cfg)
    q, index, release, histories, _, eligible = base.causal_inputs(source, 1. / dt)
    if active.any():
        clean = base.causal_fill(raw, observed)
        clean[:, :3] -= bias
        debiased, at, when, histories, _, _ = base.causal_inputs(clean, 1. / dt)
        np.testing.assert_array_equal(at, index)
        np.testing.assert_array_equal(when, release)
        q = debiased
    return q, index, release, histories, eligible if training else None, diag


@torch.inference_mode()
def features(core, windows, mean, std):
    frame = intrinsic.intrinsic_windows(windows)[0]
    projected = np.zeros((len(windows), 3), dtype=np.float32)
    latent = np.zeros((len(windows), core.output.in_features), dtype=np.float32)
    for start in range(0, len(windows), 256):
        indices = np.arange(start, min(start + 256, len(windows)))
        active = indices[frame.valid[indices]]
        if not len(active):
            continue
        transform = type(frame)(frame.basis[active], frame.source[active], frame.anchor[active])
        canonical = intrinsic.transform_inputs(windows[active], transform)
        x = base.normalized(torch.from_numpy(canonical).to(mean.device), mean, std)
        hidden = core.input(x)
        hidden = hidden + core.sequence(core.norm(hidden))
        latent[active] = hidden[:, -1].cpu().numpy()
        projected[active] = canonical[:, -1, 6:9]
    return latent, projected, frame


def prepare_source(spec, yt, mats, core, mean, std, cfg):
    raw, observed, dt, score, truth, hashes = train_raw(spec, yt, mats)
    q, index, release, history, complete, bias = inertial_replay(raw, observed, dt, cfg, training=True)
    eligible = complete & score[release]
    index, release, history = index[eligible], release[eligible], history[eligible]
    base.require(len(release) > 0, f"No training windows for {spec['record']}")
    source = normalize(q[index])
    inverse = source.copy()
    inverse[:, 1:] *= -1.
    target_world = rotate(normalize(q[release]), truth[release])
    aligned = rotate(inverse, target_world)
    world_err = float(np.max(np.abs(rotate(source, aligned) - target_world)))
    base.require(world_err <= 2e-12, "Release-frame target alignment failed")
    latent, up, frame = features(core, history, mean, std)
    canonical_target = intrinsic.transform_directions(aligned, frame)
    valid = frame.valid
    recovered = np.einsum("ni,nji->nj", canonical_target[valid], frame.basis[valid])
    np.testing.assert_allclose(recovered, aligned[valid], rtol=0, atol=1e-7)
    base.require(valid.any(), f"No nondegenerate windows for {spec['record']}")
    data = np.concatenate((latent[valid], up[valid], canonical_target[valid]), axis=1)
    return data, release[valid], {"raw_hashes": hashes, "eligible_windows": int(eligible.sum()), "usable": len(data),
                                  "degenerate_windows": int((~valid).sum()), "active_native_frames": bias["active_native_frames"],
                                  "target_world_roundtrip_max_abs": world_err}


def sample_pair(records, sizes):
    domains = defaultdict(lambda: defaultdict(list))
    for i, spec in enumerate(records):
        domains[spec["domain"]][spec["group"]].append(i)
    rng = np.random.default_rng(SEED)
    roots = np.r_[0, np.cumsum(sizes)]

    def choose(domain, count):
        groups = list(sorted(domains[domain]))
        out = []
        for g in rng.integers(len(groups), size=count):
            records_in_group = domains[domain][groups[g]]
            rec = records_in_group[rng.integers(len(records_in_group))]
            out.append(int(roots[rec] + rng.integers(sizes[rec])))
        return out

    matrix_a, matrix_b, membership = [], [], []
    for _ in range(STEPS):
        share = choose("BROAD", 32) + choose("Sassari", 32)
        alt_a = choose("BROAD", 16) + choose("Sassari", 16)
        alt_b = choose("YT_Server", 32)
        order = rng.permutation(BATCH)
        matrix_a.append(np.asarray(share + alt_a)[order])
        matrix_b.append(np.asarray(share + alt_b)[order])
        membership.append(order < BATCH - FRACTION)
    a, b = (np.asarray(m, dtype="<i8") for m in (matrix_a, matrix_b))
    common = np.asarray(membership, dtype=bool)
    base.require(np.all(common.sum(axis=1) == 64) and np.array_equal(a == b, common), "Paired batch schedule changed")
    return a, b, common, roots


def optimize(label, initial, data, schedule, input_width, device):
    started = stamp()
    torch.manual_seed(SEED)
    layer = deepcopy(initial).requires_grad_(True).to(device)
    opt = torch.optim.AdamW(layer.parameters(), lr=LR, weight_decay=WD)
    batch = torch.from_numpy(schedule).to(device)
    records = []
    for step, idx in enumerate(batch, 1):
        features_and_labels = data[idx]
        x, up, target = features_and_labels.split((input_width, 3, 3), dim=1)
        opt.zero_grad(set_to_none=True)
        delta = base.tangent(layer(x), up, base.CAP)
        corrected = F.normalize(up + GAIN * delta, dim=1)
        loss = (corrected - target).square().sum(dim=1).mean()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(layer.parameters(), 2.))
        opt.step()
        entry = {"arm": label, "step": step, "loss": float(loss.detach()), "gradient_norm": grad_norm}
        base.require(np.isfinite([entry["loss"], grad_norm]).all(), f"Nonfinite fit {label}/{step}")
        records.append(entry)
        if step in (1, 40, 80, 120, STEPS):
            print("training", entry, flush=True)
    return layer.eval().requires_grad_(False), records, {"started_at": started, "completed_at": stamp(), "steps": len(records)}


@torch.inference_mode()
def evaluate_windows(core, layers, history, mean, std):
    latent, up, frame = features(core, history, mean, std)
    corrections = {name: np.zeros((len(history), 3), dtype=np.float64) for name in layers}
    for start in range(0, len(history), 256):
        rows = np.arange(start, min(start + 256, len(history)))
        use = rows[frame.valid[rows]]
        if not len(use):
            continue
        hidden = torch.from_numpy(latent[use]).to(mean.device)
        gravity = torch.from_numpy(up[use]).to(mean.device)
        for name, layer in layers.items():
            delta = base.tangent(layer(hidden), gravity, base.CAP).cpu().numpy()
            corrections[name][use] = np.einsum("ni,nji->nj", delta, frame.basis[use])
    return corrections


def evaluate(row, spec, old_c1, medium, yt, split, c1_proto, array, hashes, masks, core, layers, mean, std, cfg):
    raw, dt, observed, reference, mask, axes = c3.sources(row, spec, old_c1, medium, yt, split, c1_proto, array, hashes)
    np.testing.assert_array_equal(mask, masks[row["mask_key"]])
    mask_sha = intrinsic.mask_digest(mask)
    base.require(mask_sha == row["scoring_indices_sha256"], "Development mask changed")
    q, index, release, history, _, bias = inertial_replay(raw, observed, dt, cfg)
    base.require(bias["first_active_native_index"] == row["bias"]["first_active_native_index"], "C003 estimator activation changed")
    deltas = evaluate_windows(core, layers, history, mean, std)
    candidates = {name: (name, GAIN, False) for name in layers}

    def metrics(mask_for_score):
        result = intrinsic.score_attitudes(q, reference, mask_for_score, q[index], release, deltas, candidates, dt)
        return {"debiased_full_vqf_deg": result["full_vqf"], **{name + "_deg": result[name] for name in layers}}

    scores = metrics(mask)
    base.require(abs(scores["c003_deg"] - row["debiased_b6_gain_025_deg"]) <= 1e-6
                 and abs(scores["debiased_full_vqf_deg"] - row["debiased_full_vqf_deg"]) <= 1e-8,
                 f"C003 baseline did not reproduce: {row['record']}")
    identity = ("dataset", "record", "group", "scored_frames", "mask_key", "scoring_indices_sha256", "reference_status")
    result = {**{key: row[key] for key in identity}, "full_vqf_deg": row["full_vqf_deg"], **scores,
              "c003_replay_minus_archived_deg": scores["c003_deg"] - row["debiased_b6_gain_025_deg"]}
    poses = []
    for key, axis in axes.items():
        use = mask & axis
        poses.append({"record": row["record"], "pose": key, "scored_frames": int(use.sum()),
                      "scoring_indices_sha256": intrinsic.mask_digest(use), **metrics(use)})
    return result, poses


def summary(rows):
    if not rows:
        return {}
    scores = {m: float(np.mean([x[m] for x in rows])) for m in METHODS}
    return {"records": len(rows), "frames": sum(x["scored_frames"] for x in rows), "macro_rmse_deg": scores,
            "wins_vs_default": {m: sum(x[m] < x["full_vqf_deg"] for x in rows) for m in METHODS[1:]},
            "wins_vs_c003": {m: sum(x[m] < x["c003_deg"] for x in rows) for m in METHODS[2:]},
            "B_minus_A_deg": scores["B_deg"] - scores["A_deg"]}


def main():
    started = stamp()
    protocol_path = HERE / "protocol.json"
    base.require(not protocol_path.exists(), "C004 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    c3_previous = read(C003 / "results.json")
    c3_proto = read(C003 / "protocol.json")
    c2_proto = read(C002 / "protocol.json")
    c1_previous = read(C001 / "results.json")
    c1_proto = read(C001 / "protocol.json")
    inv = read(C001 / "inventory.json")
    cfg = c3.Settings(**c3_proto["thresholds"])
    masks_path = C002 / "scoring-masks.npz"
    base.require(not c3_previous["failures"] and len(c3_previous["sequences"]) == 85
                 and digest(C003 / "protocol.json") == c3_previous["protocol_sha256"]
                 and digest(C002 / "protocol.json") == c3_proto["c002_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == c3_proto["estimator_sha256"]
                 and digest(masks_path) == c3_previous["shared_scoring_masks_sha256"]
                 and digest(C001 / "inventory.json") == c1_previous["inventory_sha256"], "Frozen C003 evidence changed")
    incumbent = PILOT / campaign["incumbent"]
    base.require(digest(incumbent) == campaign["incumbent_sha256"] == c3_proto["checkpoint_sha256"], "B6 weight changed")
    b6_dir = TRANSFER / "batch6"
    b6_protocol = read(b6_dir / "protocol.json")
    base.require(all(digest(b6_dir / f) == h for f, h in b6_protocol["implementation_sha256"].items())
                 and all(digest(TRANSFER / f) == b6_protocol["source_sha256"][f]
                         for f in ("train_pair.py", "models.py", ".deps/transformers/models/mamba/modeling_mamba.py")),
                 "B6 inference implementation changed")
    split_path = TRANSFER / "external-split.json"
    base.require(digest(split_path) == inv["prior_external_split_sha256"], "BROAD/Sassari split changed")
    split = read(split_path)
    records, exclusion = freeze_records(inv, split)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    frozen = torch.load(incumbent, map_location="cpu", weights_only=True)
    base.require(frozen["model"] == c2.c1.MODEL and frozen["seed"] == 1729 and frozen["epoch"] == 6, "Unexpected B6 metadata")
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    core.eval().requires_grad_(False).to(device)
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    initial = tensor_hash(core.output.state_dict())
    backbone = tensor_hash({k: v for k, v in core.state_dict().items() if not k.startswith("output.")})
    norm_sha = tensor_hash({"mean": mean, "std": std})
    protocol = {"started_at": started, "pid": os.getpid(), "base_checkpoint_sha256": digest(incumbent),
                "c003_results_sha256": digest(C003 / "results.json"), "c003_bias_protocol_sha256": digest(C003 / "protocol.json"),
                "mask_sha256": digest(masks_path), "source_split_sha256": digest(split_path),
                "initial_A_output_parameter_sha256": initial, "initial_B_output_parameter_sha256": initial,
                "frozen_backbone_parameter_sha256": backbone, "frozen_normalization_sha256": norm_sha,
                "source_records": records, "A_training_domains": ["BROAD", "Sassari"], "B_training_domains": ["BROAD", "Sassari", "YT_Server"],
                "yt_prefix_exclusion": exclusion, "seed": SEED, "steps_per_arm": STEPS, "batch_size": BATCH,
                "A_sample_count_per_batch": {"BROAD": 48, "Sassari": 48},
                "B_sample_count_per_batch": {"BROAD": 32, "Sassari": 32, "YT_Server": 32},
                "sampling": "Independent uniform group, then uniform record within group, then uniform eligible update; identical 64 positions per paired batch.",
                "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "gradient_clip": 2.,
                "trainable": ["output.weight", "output.bias"], "trainable_parameters": sum(v.numel() for v in core.output.parameters()),
                "objective": "Unit-direction squared loss in intrinsic release-window coordinates at deployment gain 0.25.",
                "inference_gain": GAIN, "checkpoint_rule": "Fixed step 160 for both arms. After both fits, score common 85-record development panel once. No validation selection."}
    write(protocol_path, protocol)
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "status": "running",
                                  "protocol_sha256": digest(protocol_path)})
    print("frozen", len(records), "source records; B YT groups", exclusion["selected_prefixes"], flush=True)
    sources, source_times, preparation, memos = [], [], [], {}
    with c2.c1.h5py.File(inv["yt"]["path"], "r") as yt:
        for spec in records:
            matrix, times, evidence = prepare_source(spec, yt["train"], memos, core, mean, std, cfg)
            sources.append(matrix)
            source_times.append(times)
            preparation.append({**spec, **evidence})
    schedules = sample_pair(records, [len(x) for x in sources])
    a, b, common, roots = schedules
    schedule_path = HERE / "schedules.npz"
    with schedule_path.open("wb") as stream:
        np.savez_compressed(stream, A=a, B=b, shared_slots=common, record_offsets=roots, release_indices=np.concatenate(source_times))
    for name, ids in (("A", a), ("B", b)):
        uses = np.bincount(np.searchsorted(roots[1:], ids.ravel(), side="right"), minlength=len(records))
        for index, info in enumerate(preparation):
            info[name + "_draws"] = int(uses[index])
    preparation_path = HERE / "training-preparation.json"
    write(preparation_path, {"completed_at": stamp(), "schedules_sha256": digest(schedule_path), "records": preparation})
    data = torch.from_numpy(np.concatenate(sources)).to(device)
    del sources, memos
    metadata, logs, layers = {}, [], {"c003": core.output}
    for name, indices in (("A", a), ("B", b)):
        layer, history, times = optimize(name, core.output, data, indices, core.output.in_features, device)
        base.require(tensor_hash(core.output.state_dict()) == initial, "Shared initialization mutated")
        layers[name] = layer
        logs.extend(history)
        state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
        state.update({f"output.{key}": value.detach().cpu() for key, value in layer.state_dict().items()})
        checkpoint = HERE / f"{name}-readout.pt"
        stage = checkpoint.with_suffix(".tmp")
        torch.save({**frozen, "state_dict": state, "c004_arm": name, "steps": STEPS, "c004_seed": SEED,
                    "c004_gain": GAIN, "c004_protocol_sha256": digest(protocol_path)}, stage)
        stage.replace(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        base.require(tensor_hash({k: v for k, v in saved["state_dict"].items() if not k.startswith("output.")}) == backbone
                     and tensor_hash({"mean": saved["mean"], "std": saved["std"]}) == norm_sha,
                     "Frozen checkpoint tensors changed")
        output_state = {k.removeprefix("output."): v for k, v in saved["state_dict"].items() if k.startswith("output.")}
        layers[name].load_state_dict(output_state)
        metadata[name] = {**times, "weights": checkpoint.name, "weights_sha256": digest(checkpoint),
                          "output_parameter_sha256": tensor_hash(output_state), "frozen_backbone_sha256": backbone,
                          "normalization_sha256": norm_sha}
    del data
    logs_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(logs_path, logs)
    write(HERE / "training.json", {"arms": metadata, "training_steps_csv_sha256": digest(logs_path),
                                   "schedules_sha256": digest(schedule_path), "shared_slots_per_batch": 64})
    print("both fits persisted, starting one shared validation", flush=True)

    specs = {(r["dataset"], r["record"]): r for r in inv["records"] if r["c001"] != "none" or
             (r["dataset"] == "IMU-Array" and r["side"] == "development")}
    c1_map = {(r["dataset"], r["record"]): r for r in c1_previous["sequences"]}
    prior_poses = {(r["record"], r["pose"]): r for r in c3_previous["array_sensor_poses"]}
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari dev source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_evidence = c2_proto["frozen_evidence"]
    ref_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(ref_path) == array_evidence["array_reference_sha256"], "Array reference changed")
    array = c2.reference_array(ref_path)
    validation_started_at = stamp()
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inv["yt"]["path"], "r") as yt, np.load(masks_path, allow_pickle=False) as masks:
        for record in c3_previous["sequences"]:
            key = record["dataset"], record["record"]
            try:
                result, details = evaluate(record, specs[key], c1_map.get(key), medium, yt["valid"], split, c1_proto,
                                           array, array_evidence["array_raw_sha256"], masks, core, layers, mean, std, cfg)
                for detail in details:
                    old = prior_poses[detail["record"], detail["pose"]]
                    base.require(detail["scoring_indices_sha256"] == old["scoring_indices_sha256"]
                                 and abs(detail["c003_deg"] - old["debiased_b6_gain_025_deg"]) <= 1e-6,
                                 "C003 pose score or mask changed")
                    detail["full_vqf_deg"] = old["full_vqf_deg"]
            except (AssertionError, OSError, ValueError, RuntimeError, KeyError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
                continue
            rows.append(result)
            poses.extend(details)
            print(*key, *(result[m] for m in METHODS), flush=True)
    csv_path, pose_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv"
    intrinsic.shice.save_csv(csv_path, rows)
    intrinsic.shice.save_csv(pose_path, poses)
    panels = {name: summary([r for r in rows if r["dataset"] == name]) for name in c3_previous["summary"]}
    pose_tables = {name: summary([r for r in poses if r["pose"] == name]) for name in c2.POSES}
    completed = stamp()
    write(HERE / "results.json", {"started_at": started, "validation_started_at": validation_started_at, "completed_at": completed,
          "pid": os.getpid(), "protocol_sha256": digest(protocol_path), "training_sha256": digest(HERE / "training.json"),
          "training_preparation_sha256": digest(preparation_path), "schedules_sha256": digest(schedule_path),
          "weights": metadata, "validation_scope": 85, "shared_scoring_masks_sha256": digest(masks_path),
          "csv_sha256": digest(csv_path), "pose_csv_sha256": digest(pose_path), "yt_reference_status": "unverified",
          "summary": panels, "array_by_pose": pose_tables, "array_physical_trials": 1, "failures": failures,
          "sequences": rows, "array_sensor_poses": poses})
    write(HERE / "process.json", {"pid": os.getpid(), "started_at": started, "completed_at": completed,
                                  "protocol_sha256": digest(protocol_path), "status": "complete" if not failures else "partial"})
    print(json.dumps({"summary": panels, "array_by_pose": pose_tables, "failures": failures}, indent=2), flush=True)


if __name__ == "__main__":
    main()
