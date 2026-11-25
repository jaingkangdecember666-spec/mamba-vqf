"""C013 compares the archived B7 seed with the two existing C006 deployment anchors."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import inspect
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN / "C012"))

import fit_rest_gated as c12
import numpy as np
import torch
from scipy.io import loadmat

c7, c6, c5, c4, c3, c2 = c12.c7, c12.c6, c12.c5, c12.c4, c12.c3, c12.c2
base, intrinsic = c12.base, c12.intrinsic
read, write, digest, stamp, array_sha = c12.read, c12.write, c12.digest, c12.stamp, c12.array_sha
C001, C002, C003, C004, C006 = c12.C001, c12.C002, c12.C003, c12.C004, c12.C006
DATA, PILOT, TRANSFER, MASKS, GATES = c12.DATA, c12.PILOT, c12.TRANSFER, c12.MASKS, c12.GATES
B6, B7 = TRANSFER / "batch6", TRANSFER / "batch7"
sys.path.insert(0, str(B7))
import repeat_initialization as historical

METHODS = ("full_vqf_deg", "c003_deg", "seed_C003_deg", "C006_deg", "seed_C006_deg")
DOMAINS = ("BROAD", "Sassari", "YT_Server")


def seed_lineage(p6, p7, s6, s7, stat6, scope, w6, w7, state6, state7):
    """Verify old selections and reconstruct the historical source schedule from its frozen counts."""
    base.require(digest(B6 / "selection.json") == p7["batch6_selection_sha256"]
                 and digest(w6) == s6["weights_sha256"] == p7["batch6_weights_sha256"]
                 and digest(w7) == s7["weights_sha256"]
                 and digest(B6 / "protocol.json") == s6["protocol_sha256"]
                 and digest(B7 / "protocol.json") == s7["protocol_sha256"]
                 and digest(B6 / "training-frame-audit.json") == s6["training_audit_sha256"]
                 and all(digest(TRANSFER / p) == sha for p, sha in p6["source_sha256"].items())
                 and all(digest(B6 / p) == sha for p, sha in p6["implementation_sha256"].items()),
                 "Historical source or checkpoint evidence changed")
    base.require(p7["settings"] == {k: v for k, v in p6["settings"].items() if k != "seed"}
                 and p7["selection"] == p6["selection"] and p7["gain"] == p6["gain"]
                 and state6["seed"] == p6["settings"]["seed"] == state7["sampling_seed"] == p7["sampling_seed"] == 1729
                 and state7["seed"] == p7["initialization_seed"] == s7["initialization_seed"] == 314159
                 and state6["split_sha256"] == state7["split_sha256"] == p6["source_sha256"]["external-split.json"]
                 and all(state6[k] == state7[k] for k in ("model", "window", "cap_deg", "intrinsic_acc_floor", "intrinsic_gyro_floor")),
                 "Historical seed, scope or model settings changed")
    base.require(all(torch.equal(state6[k], state7[k]) for k in ("mean", "std")), "B6/B7 normalization tensors differ")
    stats_hash = historical.statistics_sha(state6["mean"], state6["std"])
    base.require(stats_hash == historical.statistics_sha(state7["mean"], state7["std"]) == p7["normalization_sha256"]
                 == s7["normalization_sha256"] == s7["recomputed_normalization_sha256"], "Normalization hash differs")
    # The historical 192-step experiment used 39 sources. It is separate from the current C004-B readout schedule.
    records = [r for r in scope["datasets"] if r["split"] == "train"]
    slots = {"BROAD": [], "Sassari": []}
    first = 0
    for record in records:
        n = record["eligible_training_windows"]
        if n:
            slots[record["domain"]].append((first, n))
            first += n
    base.require({k: len(v) for k, v in slots.items()} == {"BROAD": 27, "Sassari": 12}
                 and first == scope["training_windows"] == stat6["training_windows"], "B6/B7 source window counts changed")
    rng = np.random.default_rng(p7["sampling_seed"])
    matrix = []
    for _ in range(p6["settings"]["epochs"] * p6["settings"]["steps_per_epoch"]):
        batch = []
        for pools in slots.values():
            chosen = rng.integers(len(pools), size=48)
            batch.extend(pools[i][0] + rng.integers(pools[i][1]) for i in chosen)
        matrix.append(rng.permutation(batch))
    schedule = np.asarray(matrix, dtype=np.int64)
    fingerprint = array_sha(schedule, "<i8")
    base.require(schedule.shape == (192, 96) and fingerprint == scope["training_batch_schedule_sha256"]
                 == stat6["batch_schedule_sha256"] == p6["training_schedule_sha256"]
                 == p7["batch_schedule_sha256"] == s7["batch_schedule_sha256"], "Historical schedule mismatch")
    del schedule
    _, init = historical.create_model()
    base.require(init == p7["initialization"] == s7["initialization"], "Recreated seed initializations differ from archive")
    epochs = {}
    for label, selected, state, expected in (("B6", s6, state6, 6), ("B7", s7, state7, 5)):
        training = selected["training"]
        history = training["history"]
        epoch = min(history, key=lambda r: (r["external_validation_macro_rmse_deg"], r["epoch"]))["epoch"]
        base.require(training["fits"] == 1 and training["executed_optimizer_steps"] == 192
                     and epoch == state["epoch"] == training["selected_epoch"] == expected,
                     f"{label} historical external selection differs")
        epochs[label] = {"epoch": epoch, "external_macro_deg": history[epoch - 1]["external_validation_macro_rmse_deg"],
                         "checkpoint_sha256": selected["weights_sha256"], "initialization_seed": state["seed"]}
    backbones = [{k: v for k, v in state["state_dict"].items() if not k.startswith("output.")} for state in (state6, state7)]
    changed = {k: int(torch.count_nonzero(v - backbones[1][k])) for k, v in backbones[0].items()}
    hashes = [c4.tensor_hash(v) for v in backbones]
    base.require(hashes[0] != hashes[1] and sum(changed.values()) > 0, "B7 backbone matches B6")
    return {**epochs, "same_source_schedule_sha256": fingerprint, "historical_schedule_shape": [192, 96],
            "historical_sampling_seed": p7["sampling_seed"], "historical_train_records": records,
            "historical_training_windows": first, "normalization_sha256": stats_hash, "normalization_tensors_exactly_equal": True,
            "recreated_initialization": init, "B6_backbone_sha256": hashes[0], "B7_backbone_sha256": hashes[1],
            "B6_vs_B7_changed_backbone_elements_by_tensor": changed,
            "B6_vs_B7_changed_backbone_elements": sum(changed.values())}


def metric_summary(rows):
    valid = [r for r in rows if r["status"] == "scored"]
    macro = {m: float(np.mean([r[m] for r in valid])) for m in METHODS} if valid else {}
    pairs = (("seed_C003_deg", "c003_deg"), ("seed_C006_deg", "C006_deg"))
    return {"denominator": len(rows), "scored_records": len(valid), "scored_frames": sum(r["scored_frames"] for r in valid),
            "macro_rmse_deg": macro,
            "paired_delta_deg": {a: macro[a] - macro[b] for a, b in pairs} if valid else {},
            "paired_wins": {a: sum(r[a] < r[b] for r in valid) for a, b in pairs},
            "wins_vs_default": {a: sum(r[a] < r["full_vqf_deg"] for r in valid) for a, _ in pairs}}


def score_record(old, spec, map3, map1, medium, yt, split, c1p, array, hashes, masks, gates,
                 b6, control, b7, b7_readout, mean, std, cfg):
    key = old["dataset"], old["record"]
    if key[0] in ("OxIOD", "RepoIMU"):
        raw, q, at, release, window, mask, ref, qa, _, _ = c5.optical_payload(spec, cfg)
        raw_q, q = raw.as_quat(scalar_first=True), q.as_quat(scalar_first=True)
        base.require(qa["quality_flags"] == old["reference_quality_flags"], "Optical QA changed")
        axes = {}
    else:
        raw, dt, observed, ref, mask, axes = c3.sources(map3[key], spec, map1.get(key), medium, yt, split, c1p, array, hashes)
        clean = base.causal_fill(raw, observed)
        raw_q = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
        q, at, release, window, _, bias = c4.inertial_replay(raw, observed, dt, cfg)
        base.require(bias["first_active_native_index"] == map3[key]["bias"]["first_active_native_index"], "C003 bias activation changed")
    np.testing.assert_array_equal(mask, masks[old["mask_key"]])
    base.require(intrinsic.mask_digest(mask) == old["scoring_indices_sha256"]
                 and int(mask.sum()) == old["scored_frames"], "Original score mask changed")
    rest = c6.rest_flags(window)
    gate = gates[old["mask_key"]]
    np.testing.assert_array_equal(release, gate[:, 0])
    np.testing.assert_array_equal(rest, gate[:, 3])
    delta = c4.evaluate_windows(b6, {"c003": b6.output, "C006": control.output}, window, mean, std)
    delta.update(c4.evaluate_windows(b7, {"seed_C003": b7.output, "seed_C006": b7_readout}, window, mean, std))
    for name in ("C006", "seed_C006"):
        delta[name][rest] = 0.
    options = {k: (k, c4.GAIN, False) for k in delta}

    def compute(use, prior):
        pure = intrinsic.score_attitudes(raw_q, ref, use, raw_q[at], release, {}, {}, 1.)["full_vqf"]
        scores = intrinsic.score_attitudes(q, ref, use, q[at], release, delta, options, 1.)
        row = {"full_vqf_deg": pure, **{k + "_deg": scores[k] for k in delta}}
        controls = (("full_vqf_deg", "full_vqf_deg"), ("c003_deg", "c003_deg"), ("C006_deg", "B_rest_gated_deg"))
        gap = max(abs(row[m] - prior[o]) for m, o in controls)
        base.require(np.isfinite(list(row.values())).all() and gap <= 1e-6, f"C006 controls changed by {gap}")
        return {**row, "seed_C003_delta_deg": row["seed_C003_deg"] - row["c003_deg"],
                "seed_C006_delta_deg": row["seed_C006_deg"] - row["C006_deg"], "maximum_control_error_deg": gap}

    identity = ("dataset", "record", "group", "reference_QA", "reference_status", "instrument", "scenario",
                "mask_key", "scoring_indices_sha256", "scored_frames")
    row = {**{k: old[k] for k in identity}, **compute(mask, old), "status": "scored",
           "rest_scored_frames": old["rest"]["scored_frames_with_rest"],
           "reference_quality_flags": old.get("reference_quality_flags", [])}
    return row, axes, mask, compute


def replay(prior, inventory, campaign, split, b6, control, b7, layer, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "The scoring list is outside the frozen development panel")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(array_path)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for anchor in prior["sequences"]:
            key = anchor["dataset"], anchor["record"]
            try:
                row, axes, mask, score = score_record(anchor, specs[key], map3, map1, medium, yt["valid"], split, c1p,
                                                     array, evidence["array_raw_sha256"], masks, gates, b6, control,
                                                     b7, layer, mean, std, cfg)
                detail = []
                for axis, use in axes.items():
                    include = mask & use
                    old = old_poses[key[1], axis]
                    base.require(intrinsic.mask_digest(include) == old["scoring_indices_sha256"]
                                 and int(include.sum()) == old["scored_frames"], "Array pose mask changed")
                    detail.append({"record": key[1], "pose": axis, "scored_frames": old["scored_frames"],
                                   "scoring_indices_sha256": old["scoring_indices_sha256"], **score(include, old)})
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**{k: anchor[k] for k in ("dataset", "record", "group", "instrument", "scenario",
                                                        "reference_QA", "reference_status", "mask_key",
                                                        "scored_frames", "scoring_indices_sha256")},
                             "status": "failed", "error": repr(exc)})
                print("FAILED", *key, repr(exc), flush=True)
    sequence_path, pose_path, group_path = HERE / "sequences.csv", HERE / "array-sensor-poses.csv", HERE / "group-summaries.json"
    fields = ("dataset", "record", "group", "instrument", "scenario", "status", "reference_QA", "mask_key",
              "scoring_indices_sha256", "scored_frames", "rest_scored_frames", *METHODS, "seed_C003_delta_deg",
              "seed_C006_delta_deg", "error")
    intrinsic.shice.save_csv(sequence_path, [{k: r.get(k, "") for k in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    by_domain = {d: metric_summary([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]}
    groups = {field: {f"{d}/{g}": metric_summary([r for r in rows if r["dataset"] == d and r[field] == g])
                      for d, g in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("reference_QA", "instrument", "scenario")}
    write(group_path, groups)
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
               "Repo_stick": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "TStick"],
               "Repo_pendulum": [r for r in rows if r["dataset"] == "RepoIMU" and r["instrument"] == "Pendulum"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": by_domain, "optical_subsets": {k: metric_summary(v) for k, v in subsets.items()},
            "maximum_control_error_deg": max((r["maximum_control_error_deg"] for r in rows if r["status"] == "scored"), default=None),
            "csv_sha256": digest(sequence_path), "pose_csv_sha256": digest(pose_path) if poses else None,
            "group_summaries_sha256": digest(group_path), "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    checkpoint = HERE / "seed-C006-readout.pt"
    base.require(not protocol_path.exists() and not checkpoint.exists(), "C013 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    s6, s7 = (read(p / "selection.json") for p in (B6, B7))
    hist6, hist7 = (read(p / "protocol.json") for p in (B6, B7))
    w6, w7 = (root / s["weights"] for root, s in ((B6, s6), (B7, s7)))
    base.require(w7.name == "mamba-intrinsic-seed314159-selected.pt", "Unexpected B7 checkpoint name")
    state6, state7 = (torch.load(path, map_location="cpu", weights_only=True) for path in (w6, w7))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    lineage = seed_lineage(hist6, hist7, s6, s7, read(B6 / "training-frame-audit.json"),
                           read(TRANSFER / "fit-scope.json"), w6, w7, state6, state7)
    p4, r4, t4, prep4 = (read(C004 / n) for n in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    p6, prior = (read(C006 / n) for n in ("protocol.json", "results.json"))
    p3 = read(C003 / "protocol.json")
    inventory_path, split_path = C001 / "inventory.json", TRANSFER / "external-split.json"
    inventory = read(inventory_path)
    schedule_path = C004 / "schedules.npz"
    control_path = C004 / t4["arms"]["B"]["weights"]
    c12_path = CAMPAIGN / "C012"
    witness = c12_path / "training-windows.npz"
    base.require(len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C006 / "protocol.json") == prior["protocol_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_path) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(MASKS) == prior["shared_masks_sha256"] and digest(GATES) == prior["gate_traces_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == p3["estimator_sha256"]
                 and digest(split_path) == p4["source_split_sha256"] == inventory["prior_external_split_sha256"]
                 and digest(w6) == p4["base_checkpoint_sha256"] == p6["B6_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(control_path) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"]
                 and digest(witness) == read(c12_path / "training.json")["files_sha256"][witness.name],
                 "C004-B/C006 weights, scoring masks or independent point-label archive changed")
    base.require(all(digest(c5.REVIEW / name) == sha for name, sha in p6["optical_reference_sha256"].items()),
                 "Optical reference implementation changed")
    records, archived = p4["source_records"], prep4["records"]
    base.require(Counter(r["domain"] for r in records) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and all(all(r[k] == v for k, v in src.items()) for src, r in zip(records, archived, strict=True))
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "C004 external source manifest changed")
    base.require((p4["seed"], p4["steps_per_arm"], p4["batch_size"], p4["lr"], p4["weight_decay"], p4["inference_gain"]) ==
                 (c4.SEED, c4.STEPS, c4.BATCH, c4.LR, c4.WD, c4.GAIN), "C004 optimization rule changed")
    with np.load(schedule_path, allow_pickle=False) as item:
        schedule, offsets, releases = (item[k] for k in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and np.array_equal(np.diff(offsets), [r["usable"] for r in archived]),
                 "C004-B schedule shape or original eligibility differs")
    old_b = torch.load(control_path, map_location="cpu", weights_only=True)
    base.require(old_b["c004_arm"] == "B" and old_b["c004_gain"] == c4.GAIN
                 and all(torch.equal(old_b[k], state7[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, old_b["state_dict"][k]) for k, v in state6["state_dict"].items() if not k.startswith("output.")),
                 "C006's backbone or normalization changed")
    files = {Path(inspect.getfile(m)).resolve() for m in (c12, c7, c6, c5, c4, c3, c2, c2.c1, base, base.ResidualHead,
                                                         intrinsic, intrinsic.build_frame, historical, c12.rotate)}
    files.update(TRANSFER / name for name in hist6["source_sha256"])
    files.update(B6 / name for name in hist6["implementation_sha256"])
    files.update(c5.REVIEW / name for name in p6["optical_reference_sha256"])
    files.update((Path(__file__).resolve(), CAMPAIGN / "protocol.json", B6 / "protocol.json", B7 / "protocol.json",
                  B6 / "selection.json", B7 / "selection.json", B6 / "training-frame-audit.json", w6, w7, control_path,
                  C004 / "protocol.json", C004 / "training.json", C004 / "training-preparation.json", schedule_path,
                  C006 / "protocol.json", C006 / "results.json", inventory_path, split_path, MASKS, GATES,
                  C003 / "protocol.json", C003 / "online_bias.py", C003 / "results.json", C001 / "protocol.json",
                  C001 / "results.json", C002 / "protocol.json", c12_path / "training.json", witness))
    fingerprints = {str(p.relative_to(PILOT)): digest(p) for p in sorted(files)}
    lineage_path = HERE / "seed-lineage.json"
    write(lineage_path, lineage)
    protocol = {"pid": os.getpid(), "started_at": started, "files_sha256": fingerprints,
                "seed_lineage_sha256": digest(lineage_path), "initial_B7_weight_sha256": digest(w7),
                "historical_B6_selected_epoch": 6, "historical_B7_selected_epoch": 5,
                "source_records": records, "yt_prefix_exclusion": p4["yt_prefix_exclusion"],
                "C004_B_schedule_sha256": digest(schedule_path), "C004_B_draw_matrix_sha256": array_sha(schedule, "<i8"),
                "optimizer": {"kind": "AdamW", "seed": c4.SEED, "steps": c4.STEPS, "batch_size": c4.BATCH,
                              "lr": c4.LR, "weight_decay": c4.WD, "gradient_clip": 2.},
                "target_and_loss": "Original C004-B release-point targets and full-batch squared direction loss with no training rest gate.",
                "seed_C003": "Frozen selected B7, C003 causal accelerometer bias, fixed gain 0.25, always publish, zero optimizer steps.",
                "seed_C006": "One 160-step readout-only fit from selected B7 on the original C004-B 160x96 draw matrix. Fixed gain 0.25 at inference and zero neural correction at C006 restDetected.",
                "development_records": [[r["dataset"], r["record"]] for r in prior["sequences"]],
                "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "scope": "Only the two requested B7 deployment columns. No RIDI, final holdout, or target-based epoch selection."}
    write(protocol_path, protocol)
    protocol_sha = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "started_at": started, "pid": os.getpid(),
                             "protocol_sha256": protocol_sha, **extra})

    status("external_preparation")
    base.require(torch.cuda.is_available(), "Original CUDA inference path unavailable")
    device = torch.device("cuda")
    b6 = base.ResidualHead(state6["model"])
    b6.load_state_dict(state6["state_dict"])
    b6.eval().requires_grad_(False).to(device)
    control = deepcopy(b6)
    control.load_state_dict(old_b["state_dict"])
    b7 = base.ResidualHead(state7["model"])
    b7.load_state_dict(state7["state_dict"])
    b7.eval().requires_grad_(False).to(device)
    mean, std = state7["mean"].to(device), state7["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    pool, flags, inputs_last, audits, probes, mats = [], [], [], [], {}, {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, (spec, ref) in enumerate(zip(records, archived, strict=True)):
            values, rest, at, audit, probe = c12.point_source(spec, ref, releases[offsets[i]:offsets[i+1]],
                                                            yt["train"], mats, b7, mean, std, cfg)
            pool.append(values)
            flags.append(rest)
            inputs_last.append(at)
            audits.append(audit)
            probes.setdefault(spec["domain"], probe)
    pool_data, rest_flags, at_native = np.concatenate(pool), np.concatenate(flags), np.concatenate(inputs_last)
    ordinal = np.searchsorted(offsets[1:], schedule, side="right")
    domain = np.array([DOMAINS.index(r["domain"]) for r in records])[ordinal]
    samples = pool_data[schedule]
    native = releases[schedule]
    with np.load(witness, allow_pickle=False) as old:
        np.testing.assert_array_equal(schedule, old["original_B_schedule"])
        np.testing.assert_array_equal(ordinal, old["record_ordinal"])
        np.testing.assert_array_equal(native, old["native_release"])
        np.testing.assert_array_equal(at_native[schedule], old["native_input_last"])
        np.testing.assert_array_equal(rest_flags[schedule], old["vqf_rest"])
        np.testing.assert_array_equal(samples[..., -6:], old["C004_points"][..., -6:])
        base.require(np.any(samples[..., :-6] != old["C004_points"][..., :-6]), "B7 training features equal B6")
    for i, (current, archived_source) in enumerate(zip(audits, archived, strict=True)):
        current["draws"] = int(np.count_nonzero(ordinal == i))
        base.require(current["draws"] == archived_source["B_draws"], "A C004-B source draw count changed")
    training_windows = HERE / "training-windows.npz"
    with training_windows.open("wb") as stream:
        np.savez_compressed(stream, original_B_schedule=schedule, record_ordinal=ordinal, source_domain_ordinal=domain,
                            native_release=native, native_input_last=at_native[schedule], C004_point_targets=samples[..., -3:],
                            B7_latent_and_up=samples[..., :-3])
    prep_path = HERE / "training-preparation.json"
    write(prep_path, {"protocol_sha256": protocol_sha, "source_records": audits, "draws": schedule.size,
                      "B6_witness_same_inputs_last_releases_and_point_targets": True, "B7_latent_differs_from_B6": True,
                      "B7_output_init_sha256": c4.tensor_hash(b7.output.state_dict()),
                      "training_windows_sha256": digest(training_windows)})
    base.require(all(digest(PILOT / file) == sha for file, sha in fingerprints.items()), "Input changed before fitting")
    print("B7 lineage verified", lineage["B7"], "historical schedule", lineage["same_source_schedule_sha256"],
          "different backbone elements", lineage["B6_vs_B7_changed_backbone_elements"], flush=True)
    status("fitting", planned_steps=160, preparation_sha256=digest(prep_path))
    tensor = torch.from_numpy(pool_data).to(device)
    layer, history, timing = c4.optimize("seed_C006", b7.output, tensor, schedule, b7.output.in_features, device)
    stage = checkpoint.with_suffix(".tmp")
    state = {k: v.detach().cpu().clone() for k, v in b7.state_dict().items()}
    state.update({"output." + k: v.detach().cpu().clone() for k, v in layer.state_dict().items()})
    torch.save({**state7, "state_dict": state, "source_B7_sha256": digest(w7), "C013_optimizer_steps": len(history),
                "protocol_sha256": protocol_sha}, stage)
    stage.replace(checkpoint)
    for i, row in enumerate(history):
        counts = {d + "_draws": int(np.count_nonzero(domain[i] == j)) for j, d in enumerate(DOMAINS)}
        base.require(all(v == 32 for v in counts.values()), "Per-source draw counts changed")
        row.update(counts, record_ordinals_sha256=array_sha(ordinal[i], "<i8"),
                   native_release_sha256=array_sha(native[i], "<i8"))
    steps_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(steps_path, history)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    changed = [k for k, v in saved["state_dict"].items() if not torch.equal(v, state7["state_dict"][k])]
    base.require(set(changed) == {"output.weight", "output.bias"} and all(torch.equal(saved[k], state7[k]) for k in ("mean", "std"))
                 and c4.tensor_hash(b7.state_dict()) == c4.tensor_hash(state7["state_dict"]),
                 "B7 anchor or frozen features were modified")
    layer.load_state_dict({k.removeprefix("output."): saved["state_dict"][k] for k in changed})
    # This audits the fitted branch; the frozen B7 anchor retains its historical ungated causality evidence.
    prefix = HERE / "prefix-audit.json"
    write(prefix, c7.causal_check(probes, b7, layer, mean, std, cfg))
    training = HERE / "training.json"
    write(training, {"fit_count": 1, "seed_C003_optimizer_steps": 0, "seed_C006_fit": timing,
                      "B7_unchanged_sha256": digest(w7), "fitted_readout_sha256": c4.tensor_hash(layer.state_dict()),
                      "changed_parameter_tensors": changed, "output_parameters": 75,
                      "artifacts_sha256": {f.name: digest(f) for f in (checkpoint, prep_path, steps_path, prefix, training_windows, lineage_path)}})
    del tensor, pool_data, pool, samples, flags, probes, mats
    status("development_replay", training_sha256=digest(training))
    print("C013 one readout fit persisted, beginning fixed paired replay", flush=True)
    start = perf_counter()
    result = replay(prior, inventory, campaign, read(split_path), b6, control, b7, layer, mean, std, cfg)
    completed = stamp()
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": protocol_sha,
                   "training_sha256": digest(training), "scoring_masks_sha256": digest(MASKS),
                   "scoring_seconds": perf_counter() - start, "total_seconds": perf_counter() - wall})
    path = HERE / "results.json"
    write(path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(path))
    print("C013 coverage", len(result["sequences"]), "failures", result["failures"], "seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        path = HERE / "process.json"
        if path.exists():
            previous = read(path)
            if previous["pid"] == os.getpid():
                write(path, {**previous, "phase": previous["status"], "status": "failed", "completed_at": stamp(),
                             "failure": repr(error)})
        raise
