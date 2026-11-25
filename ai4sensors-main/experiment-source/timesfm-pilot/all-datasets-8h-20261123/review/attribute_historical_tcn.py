"""Attribute the flagged historical TCN archive differences on the original CUDA backend."""
import csv
import hashlib
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from check_sources import ROOT, TRANSFER
from locked_replay import sha256
from native_blocks import block_inputs
from native_catalog import Catalog
from replay_native import metric, weights
from score_final import atomic_output, read, single_job, write_receipt

sys.path.insert(0, str(TRANSFER / "batch8"))
import final_score as batch8


HISTORICAL = ROOT / "final/historical-models"
TARGET = HISTORICAL / "tcn-backend-attribution"
OLD = TRANSFER / "batch8"
FINAL_LOCK = ROOT / "selection-lock.json"
UP = np.array([0., 0., 1.])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now().astimezone().isoformat()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def raw_for_current(catalog, name):
    (signal, hz, truth, mask), _ = catalog.native("SHICE", name)
    observed = np.isfinite(signal).all(axis=1)
    previous = np.maximum.accumulate(np.where(observed, np.arange(len(signal)), -1))
    filled = signal[previous.clip(min=0)].copy()
    filled[previous < 0] = 0.
    return filled, hz, truth, mask & observed


def current_score(q, at, release, truth, indices, delta):
    target = q[indices].apply(truth[indices])
    directions = UP + q[at].apply(delta)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    available = np.searchsorted(release, indices, side="right")
    return metric(np.vstack((UP, directions))[available], target)


def compare_one(cpu_row, old_row, archive_masks, catalog, model, tolerance, probe_tf32):
    name = cpu_row["record"]
    dt, raw, old_ref, mask = batch8.shice.raw_record(name)
    filled, hz, truth, new_mask = raw_for_current(catalog, name)
    np.testing.assert_array_equal(raw, filled)
    np.testing.assert_array_equal(mask, new_mask)
    np.testing.assert_array_equal(mask, archive_masks[name])
    indices = np.flatnonzero(mask).astype("<i8", copy=False)
    digest = hashlib.sha256(indices.tobytes()).hexdigest()
    require(len(indices) == int(cpu_row["scored_frames"]) == int(old_row["frames"]) and
            digest == cpu_row["scoring_indices_sha256"] == old_row["scoring_indices_sha256"] and hz == 1. / dt,
            "Archived and current native masks or acquisition clock differ")

    quat, last_old, release_old, historical_windows, _, _ = batch8.base.causal_inputs(raw, hz)
    historical_up = batch8.base.body_up(quat[last_old], batch8.base.EST_UP).astype(np.float32)
    historical_delta = batch8.strict_predict(model[0], historical_windows, historical_up, model[1], model[2])
    archived_backend = batch8.intrinsic.score_attitudes(
        quat, old_ref, mask, quat[last_old], release_old, {"strict_tcn": historical_delta},
        {"strict_tcn": ("strict_tcn", 1., False)}, dt,
    )
    baseline_archived_gap = archived_backend["full_vqf"] - float(old_row["full_vqf"])
    require(abs(baseline_archived_gap) < 1e-8, "Historical CUDA replay did not reproduce the archived VQF")

    q, current_windows, at, release = block_inputs(filled, hz)
    np.testing.assert_array_equal(last_old, at)
    np.testing.assert_array_equal(release_old, release)
    up = q[at].inv().apply(UP).astype(np.float32)
    features_exact = np.array_equal(current_windows, historical_windows) and np.array_equal(up, historical_up)
    same_input_delta = historical_delta if features_exact else batch8.strict_predict(
        model[0], current_windows, up, model[1], model[2],
    )
    same_input_cuda = current_score(q, at, release, truth, indices, same_input_delta)
    archived = float(old_row["strict_tcn"])
    cpu = float(cpu_row["strict_tcn_deg"])
    original_cuda = archived_backend["strict_tcn"]
    difference = original_cuda - archived
    output = {
        "record":name, "scored_frames":len(indices), "scoring_indices_sha256":digest,
        "native_mask_and_clock_exact":True, "archive_cuda_deg":archived, "first_cpu_deg":cpu,
        "replayed_original_cuda_deg":original_cuda, "current_input_cuda_deg":same_input_cuda,
        "first_cpu_minus_archived_deg":cpu-archived, "original_cuda_minus_archived_deg":difference,
        "current_cuda_minus_original_cuda_deg":same_input_cuda-original_cuda,
        "cpu_minus_current_cuda_deg":cpu-same_input_cuda,
        "historical_vqf_minus_archive_deg":baseline_archived_gap,
        "current_features_bitwise_equal_to_history":bool(features_exact),
        "max_feature_abs_diff":float(np.max(np.abs(current_windows-historical_windows))),
        "max_update_gravity_abs_diff":float(np.max(np.abs(up-historical_up))),
        "max_scored_reference_abs_diff":float(np.max(np.abs(truth[indices]-old_ref[indices]))),
        "original_cuda_within_preset_tolerance":abs(difference)<=tolerance,
        "original_first_flag":cpu_row["status"],
        "status":"attributed" if abs(difference)<=tolerance else "original_cuda_archive_mismatch",
    }
    if probe_tf32:
        allow = torch.backends.cudnn.allow_tf32
        output["original_cudnn_allow_tf32"] = allow
        if allow:
            try:
                torch.backends.cudnn.allow_tf32 = False
                fp32_delta = batch8.strict_predict(model[0], current_windows, up, model[1], model[2])
                without_tf32 = current_score(q, at, release, truth, indices, fp32_delta)
                output["controlled_current_cuda_no_tf32_deg"] = without_tf32
                output["cpu_minus_controlled_cuda_no_tf32_deg"] = cpu-without_tf32
            finally:
                torch.backends.cudnn.allow_tf32 = allow
    return output


def main():
    completed = read(HISTORICAL / "process.json")
    require(completed["status"] in ("complete","complete_with_errors"), "Finish the single CPU baseline run first")
    run = read(HISTORICAL / "run.json")
    locks = (TRANSFER / "review/final-transfer-lock.json", OLD / "all-sequences.csv", OLD / "scoring-masks.npz")
    lock_path, archive_path, masks_path = locks
    lock = read(lock_path)
    source = read(OLD / "results.json")
    model_cfg = lock["models"]["strict_tcn"]
    checkpoint = TRANSFER / model_cfg["path"]
    require(sha256(lock_path) == run["historical_selection_sha256"] and sha256(checkpoint) == model_cfg["sha256"]
            and sha256(archive_path) == source["csv_sha256"] and sha256(masks_path) == source["masks_sha256"],
            "Historical TCN evidence changed")
    require(sha256(FINAL_LOCK)==run["selection_sha256"], "Official C038 lock changed")
    first_table = HISTORICAL / "main.csv"
    require(sha256(first_table)==completed["tables"]["main"]["sha256"], "Original CPU result table changed")
    flags = [r for r in read_csv(first_table) if r["status"]=="batch8_score_mismatch"]
    require(flags and all(r["dataset"]=="SHICE" and r["record"].split("-")[0] in ("R5","R6") for r in flags),
            "Unexpected baseline flags")
    registered = read(HISTORICAL / "r5-r6-batch8-parity.json")
    require({r["record"] for r in flags}=={r["record"] for r in registered["records_detail"] if r["status"]=="batch8_score_mismatch"},
            "Flag membership differs from the original SHICE review")
    archive = {r["sequence"]:r for r in read_csv(archive_path)}
    biggest = max(flags,key=lambda r:abs(float(r["strict_tcn_minus_batch8_deg"])))["record"]
    fingerprint_paths = [*locks, checkpoint, FINAL_LOCK, first_table, HISTORICAL/"run.json", HISTORICAL/"r5-r6-batch8-parity.json",
                         Path(__file__), OLD/"final_score.py", ROOT/"review/native_blocks.py",
                         *[HISTORICAL/f"{label}.csv" for label in ("transfer",)], ROOT/"final/main.csv", ROOT/"final/transfer.csv"]
    fingerprints = {os.path.relpath(path,ROOT).replace("\\","/"):sha256(path) for path in fingerprint_paths}
    TARGET.mkdir(exist_ok=True)
    with single_job(TARGET/"review.lock"):
        torch.set_num_threads(2)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.cuda.get_device_name()
        original_device = read(OLD/"protocol.json")["device"]
        require(device == original_device, f"Original CUDA device differs: {device} vs {original_device}")
        model = weights(checkpoint,"tcn","cuda")
        tolerance = run["tolerances_deg"]["historical_r5_r6_models"]
        catalog = Catalog(FINAL_LOCK)
        rows = []
        started = perf_counter()
        start_at = now()
        with np.load(masks_path,allow_pickle=False) as mask_archive:
            for cpu_row in flags:
                name = cpu_row["record"]
                tick = perf_counter()
                try:
                    result = compare_one(cpu_row, archive[name], mask_archive, catalog, model, tolerance, name==biggest)
                except (ValueError,RuntimeError,AssertionError, OSError) as exc:
                    result = {"record":name,"status":"attribution_failed","detail":f"{type(exc).__name__}: {exc}"}
                result["seconds"] = perf_counter()-tick
                result["completed_at"] = now()
                write_receipt(TARGET/f"{name}.json",result)
                rows.append(result)
                print(name,result["status"],result.get("original_cuda_minus_archived_deg"),flush=True)
        require(all(sha256(ROOT/name)==digest for name,digest in fingerprints.items()), "Baseline input changed during attribution")
        path = TARGET/"sequences.csv"
        columns = list(dict.fromkeys(k for row in rows for k in row))
        with atomic_output(path,encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream,fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        numeric = [r for r in rows if "original_cuda_minus_archived_deg" in r]
        write_receipt(TARGET/"summary.json", {
            "pid":os.getpid(),"started_at":start_at,"completed_at":now(),"wall_seconds":perf_counter()-started,
            "scope":"Only strict_tcn and the R5/R6 records flagged by the frozen first CPU run.",
            "device":device,"original_device":original_device,"torch_version":str(torch.__version__),
            "cudnn_version":torch.backends.cudnn.version(),"cudnn_deterministic":torch.backends.cudnn.deterministic,
            "cudnn_benchmark":torch.backends.cudnn.benchmark,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
            "preset_tolerance_deg":tolerance,"first_cpu_outside_tolerance":len(flags),
            "status_counts":dict(Counter(r["status"] for r in rows)),
            "max_abs_original_cuda_vs_archive_deg":max((abs(r["original_cuda_minus_archived_deg"]) for r in numeric),default=None),
            "max_abs_current_cuda_vs_original_cuda_deg":max((abs(r["current_cuda_minus_original_cuda_deg"]) for r in numeric),default=None),
            "controlled_tf32_probe_record":biggest,"sequences_sha256":sha256(path),
            "source_sha256":fingerprints,"rows":rows,
        })
        return int(any(r["status"]!="attributed" for r in rows))


if __name__ == "__main__":
    raise SystemExit(main())
