"""Score the externally locked heads on the historical SHICE development records."""

import csv
import hashlib
import json
from pathlib import Path
import sys

BATCH4 = Path(__file__).resolve().parent
TRANSFER = BATCH4.parent
PILOT = TRANSFER.parent
LEGACY = PILOT / "overnight-20261122"
sys.path[:0] = [str(TRANSFER), str(LEGACY)]

import train_pair as external
import common as previous
import numpy as np
import torch
from run_pilot import CHANNELS, QUATERNION, body_up
from tilt_data import REF_UP, ROOT, WARMUP_S, causal_fill
from tilt_eval import normalize, rotate, score_attitudes


LOCK_SHA = "58cf429fc1a0660e2e5ced44f71873bceefd1a9d1f660f0772dec8c67e924e11"
DEVICES = ("R1", "R2", "R3", "R4")
PROGRAM_GROUPS = ((1, 5, 9), (2, 6, 10), (3, 7), (4, 8))
METHODS = ("full_vqf", *external.KINDS)
TIE_DEG = 1e-9


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def historical_dev(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip() == '"dev_metrics": {':
                content = ["{"]
                for member in source:
                    if member == "  },\n":
                        return json.loads("".join(content) + "}")
                    content.append(member)
    raise ValueError("Archived SHICE development section not found")


def load_locked_models(lock, device):
    actual_split = external.digest(TRANSFER / "external-split.json")
    external.require(actual_split == lock["source_split_sha256"], "Locked split changed")
    external.require(external.digest(TRANSFER / "batch3/results.json") == lock["selection_evidence_sha256"],
                     "Locked selection evidence changed")
    batch3 = read(TRANSFER / "batch3/results.json")
    batch3_protocol = read(TRANSFER / "batch3/protocol.json")
    external.require(external.digest(TRANSFER / "batch3/protocol.json") == batch3["protocol_sha256"],
                     "Batch 3 protocol changed")
    for relative in ("models.py", "train_pair.py", ".deps/transformers/models/mamba/modeling_mamba.py"):
        external.require(external.digest(TRANSFER / relative) == batch3_protocol["source_sha256"][relative],
                         f"Locked model implementation changed: {relative}")
    external.require(set(lock["checkpoints"]) == set(external.KINDS), "Unexpected locked models")
    models = {}
    for kind in external.KINDS:
        record = lock["checkpoints"][kind]
        path = TRANSFER / record["path"]
        selected = batch3["training"][kind]["checkpoints"]["selected"]
        external.require(path == TRANSFER / "batch3" / f"{kind}-selected.pt"
                         and external.digest(path) == record["sha256"] == selected["sha256"]
                         and record["gain"] == 1.
                         and record["epoch"] == selected["epoch"], f"Locked inference settings changed: {kind}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        external.require(state["model"] == kind and state["epoch"] == record["epoch"]
                         and state["seed"] == external.SEED == record["seed"]
                         and state["split_sha256"] == actual_split
                         and state["window"] == external.WINDOW and state["cap_deg"] == external.CAP,
                         f"Incompatible checkpoint: {kind}")
        head = external.ResidualHead(kind)
        head.load_state_dict(state["state_dict"])
        models[kind] = (head.eval().requires_grad_(False).to(device),
                        state["mean"].to(device), state["std"].to(device))
    return models


def raw_record(name):
    device = name.split("-")[0]
    source = ROOT / f"{device}_aligned_formal" / f"{name}.csv"
    audit_path = ROOT / "Label_quality_audit_20261116" / name / "screening_masks.npz"
    with source.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        keys = ("sample_index", "time_s", *CHANNELS, *QUATERNION, "is_interpolated")
        values = np.loadtxt(stream, delimiter=",", quotechar='"',
                            usecols=[header.index(key) for key in keys], ndmin=2)
    np.testing.assert_array_equal(values[:, 0], np.arange(len(values)))
    time = values[:, 1]
    increments = np.diff(time)
    dt = float(np.median(increments))
    external.require(np.isfinite(dt) and dt > 0 and np.allclose(increments, dt, rtol=1e-5, atol=1e-9),
                     f"Nonregular timestamp grid: {name}")
    samples = values[:, 2:8]
    observed = (values[:, -1] == 0) & np.isfinite(samples).all(axis=1)
    truth = values[:, 8:12]
    label_ok = np.isfinite(truth).all(axis=1) & (np.linalg.norm(truth, axis=1) > 0)
    reference = np.zeros((len(values), 3), dtype=np.float32)
    reference[label_ok] = body_up(truth[label_ok], REF_UP)
    with np.load(audit_path, allow_pickle=False) as audit:
        np.testing.assert_array_equal(audit["sample_index"], values[:, 0])
        score = ((time >= WARMUP_S) & audit["context_ok"].astype(bool) &
                 audit["observed_loss_ok"].astype(bool) & observed & label_ok)
    return dt, causal_fill(samples, observed), reference, score


def match_previous(row, dt, signal, reference, score):
    path = previous.CACHE / f"{row['name']}.npz"
    digest = external.digest(path)
    external.require(digest == row["input_cache_sha256"], f"Archived input cache changed: {row['name']}")
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(score, archive["score"])
        np.testing.assert_array_equal(reference[score], archive["ref"][score])
        cached_signal = archive["imu"]
        np.testing.assert_array_equal(signal.astype(np.float32), cached_signal)
        external.require(dt == float(archive["dt"]), f"Archived sample interval changed: {row['name']}")
        base = previous.vqf_outputs(cached_signal, dt, {})["quat6D"]
    reproduced = previous.rmse(previous.body_up(base[score]), reference[score])
    return reproduced


def magnitudes(base, index, scheduled, score_index, delta):
    world = rotate(normalize(base[index]), delta)
    angles = np.degrees(np.arctan2(np.linalg.norm(world[:, :2], axis=1), 1. + world[:, 2]))
    held = np.r_[0., angles][np.searchsorted(scheduled, score_index, side="right")]
    return {"scheduled_mean_deg": float(angles.mean()), "scheduled_p95_deg": float(np.quantile(angles, .95)),
            "scheduled_max_deg": float(angles.max()), "scored_frames_rms_deg": float(np.sqrt(np.mean(held**2)))}


@torch.inference_mode()
def replay(row, models, device):
    name = row["name"]
    dt, signal, reference, score = raw_record(name)
    old_replayed = match_previous(row, dt, signal, reference, score)
    score_index = np.flatnonzero(score).astype("<i8")
    external.require(len(score_index) == row["frames"], f"Archived frame count changed: {name}")
    base, index, scheduled, histories, _, _ = external.causal_inputs(signal, 1. / dt)
    external.require((scheduled >= index).all() and (np.diff(scheduled) > 0).all(),
                     f"Noncausal update schedule: {name}")
    up = body_up(base[index], external.EST_UP).astype(np.float32)
    deltas = {}
    for kind, (head, mean, std) in models.items():
        pieces = []
        for first in range(0, len(histories), 256):
            subset = slice(first, first + 256)
            inputs = torch.from_numpy(histories[subset]).to(device)
            gravity = torch.from_numpy(up[subset]).to(device)
            pieces.append(external.tangent(head(external.normalized(inputs, mean, std)), gravity,
                                           external.CAP).cpu().numpy())
        deltas[kind] = np.concatenate(pieces)
        external.require(np.isfinite(deltas[kind]).all(), f"Nonfinite prediction: {name}/{kind}")
    candidates = {kind: (kind, 1., False) for kind in models}
    rmse = score_attitudes(base, reference, score, base[index], scheduled, deltas, candidates, dt)
    body_double_rmse = previous.rmse(previous.body_up(base[score]), reference[score])
    device_name, program = name.split("-")
    return {"sequence": name, "device": device_name, "program": int(program), "native_frames": len(score),
            "frames": len(score_index), "scoring_indices_sha256": hashlib.sha256(score_index.tobytes()).hexdigest(),
            "same_score_mask_as_archived": True, "archived_input_cache_sha256": row["input_cache_sha256"],
            "native_frequency_hz": 1. / dt, "correction_updates": len(scheduled), "rmse_deg": rmse,
            "baseline_audit": {"archived_deg": row["rmse_deg"], "legacy_replay_deg": old_replayed,
                               "legacy_replay_minus_archived_deg": old_replayed - row["rmse_deg"],
                               "native_body_minus_legacy_replay_deg": body_double_rmse - old_replayed,
                               "current_world_minus_native_body_deg": rmse["full_vqf"] - body_double_rmse,
                               "current_minus_archived_deg": rmse["full_vqf"] - row["rmse_deg"]},
            "prediction_magnitude_deg": {
                kind: magnitudes(base, index, scheduled, score_index, delta) for kind, delta in deltas.items()
            }}


def summarize(rows):
    macro = {key: external.macro(rows, key) for key in METHODS}
    changes = {key: [row["rmse_deg"][key] - row["rmse_deg"]["full_vqf"] for row in rows]
               for key in external.KINDS}
    return {"records": len(rows), "frames": sum(row["frames"] for row in rows), "macro_rmse_deg": macro,
            "macro_delta_deg": {key: macro[key] - macro["full_vqf"] for key in changes},
            "improved_records": {key: sum(delta < -TIE_DEG for delta in values) for key, values in changes.items()},
            "worse_records": {key: sum(delta > TIE_DEG for delta in values) for key, values in changes.items()}}


def save_csv(path, rows):
    staging = path.with_suffix(".tmp")
    with staging.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    staging.replace(path)


def main():
    results_path = BATCH4 / "results.json"
    external.require(not results_path.exists(), "Batch 4 already finished")
    lock_path = TRANSFER / "review/strict-external-lock.json"
    external.require(external.digest(lock_path) == LOCK_SHA, "External lock hash mismatch")
    lock = read(lock_path)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    models = load_locked_models(lock, device)

    old_path = LEGACY / "rounds/R00001/result.json"
    baseline = historical_dev(old_path)
    old_rows = baseline["sequences"]
    names = [row["name"] for row in old_rows]
    allowed = {f"{device_name}-{program}" for device_name in DEVICES for program in range(1, 11)}
    external.require(len(names) == len(set(names)) == 37 and set(names) < allowed
                     and not {"R1-3", "R1-4"}.intersection(names), "Unexpected SHICE inventory")
    protocol_path = BATCH4 / "protocol.json"
    protocol = {"frozen_at": external.stamp(), "external_lock_sha256": LOCK_SHA, "checkpoints": lock["checkpoints"],
                "device": torch.cuda.get_device_name(device), "optimizer_steps": 0,
                "score_records": names, "archived_development_source": str(old_path),
                "window_blocks": external.WINDOW, "block_hz": 100, "update_hz": 100 // external.HOP,
                "input": "Raw aligned six-axis CSV; causal forward-fill of interpolated or invalid packets; exact external time bins from native median sample interval. Keep the locked moments and bias/rest/gravity features.",
                "score": "Default six-axis FullVQF in float64 on native timestamps. SHICE context_ok, observed_loss_ok, observed finite IMU, valid encoder quaternion and time >= 10 s. Encoder world -Y into body. Hold each correction from causal release.",
                "aggregation": "Each record contributes one native-frame tilt RMSE. Device, exact program number, and historical program groups are equal-record means.",
                "historical_program_groups": PROGRAM_GROUPS, "improvement_tolerance_deg": TIE_DEG}
    if protocol_path.exists():
        old_protocol = read(protocol_path)
        old_protocol["frozen_at"] = protocol["frozen_at"]
        external.require(old_protocol == json.loads(json.dumps(protocol)), "Frozen protocol changed")
    else:
        external.write(protocol_path, protocol)

    rows = []
    for old in old_rows:
        row = replay(old, models, device)
        rows.append(row)
        print(row["sequence"], row["frames"], row["rmse_deg"], flush=True)
    external.require(sum(row["frames"] for row in rows) == baseline["scoring_frames"], "Scoring total changed")
    groups = {f"P{i}": group for i, group in enumerate(PROGRAM_GROUPS, 1)}
    summary = {"all": summarize(rows),
               "by_device": {key: summarize([r for r in rows if r["device"] == key]) for key in DEVICES},
               "by_program": {str(key): summarize([r for r in rows if r["program"] == key]) for key in range(1, 11)},
               "by_program_group": {key: {"program_numbers": numbers,
                                          **summarize([r for r in rows if r["program"] in numbers])}
                                    for key, numbers in groups.items()}}
    csv_path = BATCH4 / "shice-sequences.csv"
    columns = ("sequence", "device", "program", "native_frames", "frames", "scoring_indices_sha256",
               "native_frequency_hz", "correction_updates")
    save_csv(csv_path, [{**{key: row[key] for key in columns}, **row["rmse_deg"], **row["baseline_audit"],
                         **{f"{kind}_{key}": value for kind, stats in row["prediction_magnitude_deg"].items()
                            for key, value in stats.items()}} for row in rows])
    group_path = BATCH4 / "shice-groups.csv"
    save_csv(group_path, [
        {"grouping": label, "group": key, "records": value["records"], "frames": value["frames"],
         **value["macro_rmse_deg"], **{f"{kind}_improved": count for kind, count in value["improved_records"].items()}}
        for label, values in (("all", {"R1-R4": summary["all"]}), ("device", summary["by_device"]),
                              ("program", summary["by_program"]), ("program_group", summary["by_program_group"]))
        for key, value in values.items()
    ])
    differences = {key: max(abs(row["baseline_audit"][key]) for row in rows) for key in (
        "legacy_replay_minus_archived_deg", "native_body_minus_legacy_replay_deg",
        "current_world_minus_native_body_deg", "current_minus_archived_deg",
    )}
    external.write(results_path, {"completed_at": external.stamp(), "protocol_sha256": external.digest(protocol_path),
                                  "sequence_csv_sha256": external.digest(csv_path),
                                  "group_csv_sha256": external.digest(group_path), "optimizer_steps": 0,
                                  "historical_macro_rmse_deg": baseline["macro_rmse_deg"],
                                  "baseline_max_abs_differences_deg": differences,
                                  "same_archived_frame_indices_records": len(rows), "summary": summary, "sequences": rows})
    print(json.dumps({"all": summary["all"], "baseline_max_abs_differences_deg": differences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
