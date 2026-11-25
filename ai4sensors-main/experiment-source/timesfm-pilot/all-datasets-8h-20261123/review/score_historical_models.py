"""Replay the two historically frozen baselines on the current main and transfer panels."""
import csv
import hashlib
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np
import torch

from check_sources import ROOT, TRANSFER
from locked_replay import sha256
from native_blocks import block_inputs
from native_catalog import Catalog
from replay_native import metric, weights
from score_final import TransferCatalog, atomic_output, read, single_job, write_receipt

sys.path.insert(0, str(TRANSFER / "batch6"))
from intrinsic import ACC_FLOOR, GYRO_FLOOR, predict_delta
from models import tangent


FINAL = ROOT / "final"
OUT = FINAL / "historical-models"
FINAL_LOCK = ROOT / "selection-lock.json"
OLD_LOCK = TRANSFER / "review/final-transfer-lock.json"
OLD_RESULTS = TRANSFER / "batch8/results.json"
OLD_PROTOCOL = TRANSFER / "batch8/protocol.json"
OLD_TABLE = TRANSFER / "batch8/all-sequences.csv"
OLD_MASKS = TRANSFER / "batch8/scoring-masks.npz"
LATENCY = ROOT / "review/shice-native-latency-cuda.json"
OLD_LOCK_SHA = "411530d332658b0d0c4c68fa9ec4494fde8cf126a68fbb1d1d66d9d3c282db69"
MODELS = ("transfer_mamba", "strict_tcn")
SCORES = ("full_vqf_deg", "candidate_deg", *[m + "_deg" for m in MODELS])
IDENTITY = ("dataset", "record", "group", "side", "setting")
VQF_TOL = 1e-8
SHICE_TOL = 1e-5
UP = np.array([0., 0., 1.])
THREADS = 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now().astimezone().isoformat()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_checkpoints(lock):
    models = {}
    protocol = read(TRANSFER / "batch6/protocol.json")
    for name in ("train_pair.py", "models.py", "batch4/replay_shice.py",
                 ".deps/transformers/models/mamba/modeling_mamba.py"):
        require(sha256(TRANSFER / name) == protocol["source_sha256"][name], f"Historical code changed: {name}")
    for name in MODELS:
        cfg = lock["models"][name]
        path = TRANSFER / cfg["path"]
        require(sha256(path) == cfg["sha256"] and cfg["gain"] == 1., f"Historical model changed: {name}")
        require(cfg["intrinsic"] == (name == "transfer_mamba"), f"Historical coordinate policy changed: {name}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        require(all(state[key] == cfg[key] for key in ("epoch", "seed"))
                and state["model"] == cfg["kind"] and state["window"] == 64 and state["cap_deg"] == 6.
                and state["split_sha256"] == lock["source_split_sha256"], f"Historical metadata changed: {name}")
        if cfg["intrinsic"]:
            require(state["intrinsic_acc_floor"] == ACC_FLOOR and state["intrinsic_gyro_floor"] == GYRO_FLOOR,
                    "Historical intrinsic thresholds changed")
        head, mean, scale, cap = weights(path, cfg["kind"], "cpu")
        require(tuple(mean.shape) == tuple(scale.shape) == (13,) and torch.isfinite(mean).all().item()
                and torch.isfinite(scale).all().item() and (scale > 0).all().item(), f"Invalid normalization: {name}")
        models[name] = head.requires_grad_(False), mean, scale, cap
    return models


@torch.inference_mode()
def historical_corrections(windows, gravity, models):
    intrinsic, mean, scale, _ = models["transfer_mamba"]
    tcn, tcn_mean, tcn_scale, cap = models["strict_tcn"]
    d_mamba = np.empty((len(windows), 3))
    d_tcn = np.empty((len(windows), 3))
    for first in range(0, len(windows), 256):
        span = slice(first, first + 256)
        features = windows[span]
        d_mamba[span] = predict_delta(intrinsic, features, mean, scale)
        x = torch.from_numpy(features)
        up = torch.from_numpy(gravity[span])
        d_tcn[span] = tangent(tcn(((x - tcn_mean) / tcn_scale).clamp(-12, 12)), up, cap).numpy()
    return {"transfer_mamba": d_mamba, "strict_tcn": d_tcn}


def score_record(raw, official, models, archive, masks):
    signal, hz, truth, mask = raw
    observed = np.isfinite(signal).all(axis=1)
    paired = mask & observed
    indices = np.flatnonzero(paired).astype("<i8", copy=False)
    index_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    require(len(indices) == int(official["scored_frames"]) and index_hash == official["scoring_indices_sha256"],
            f"Common mask differs: {len(indices)} frames / {index_hash}")
    name = official["record"]
    historical = archive.get(name) if official["dataset"] == "SHICE" else None
    if historical is not None:
        require(np.array_equal(paired, masks[name]) and len(indices) == int(historical["frames"])
                and index_hash == historical["scoring_indices_sha256"], "Historical SHICE native mask differs")

    previous = np.maximum.accumulate(np.where(observed, np.arange(len(signal)), -1))
    filled = signal[previous.clip(min=0)].copy()
    filled[previous < 0] = 0.
    q, windows, last, release = block_inputs(filled, hz)
    target = q[indices].apply(truth[indices])
    raw_vqf = metric(np.broadcast_to(UP, target.shape), target)
    expected_vqf = float(official["full_vqf_deg"])
    gap = raw_vqf - expected_vqf
    require(abs(gap) <= VQF_TOL, f"Default VQF differs from official by {gap:.15g} deg ({raw_vqf:.15g} vs {expected_vqf:.15g})")
    result = {"status": "scored", "native_frames": len(signal), "scored_frames": len(indices),
              "scoring_indices_sha256": index_hash, "rate_hz": hz, "releases": len(release),
              "causal_empty_block_compatibility": bool(hz < 100), "full_vqf_deg": raw_vqf,
              "official_full_vqf_deg": expected_vqf, "full_vqf_minus_official_deg": gap,
              "candidate_deg": float(official["candidate_deg"])}

    updates = q[last]
    gravity = updates.inv().apply(UP).astype(np.float32)
    corrections = historical_corrections(windows, gravity, models)
    available = np.searchsorted(release, indices, side="right")
    for model, delta in corrections.items():
        direction = UP + updates.apply(delta)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        value = metric(np.vstack((UP, direction))[available], target)
        require(math.isfinite(value), f"{model} produced a nonfinite tilt RMSE")
        result[model + "_deg"] = value
        result[model + "_minus_c038_deg"] = value - result["candidate_deg"]

    if historical is not None:
        differences = {
            "full_vqf": raw_vqf - float(historical["full_vqf"]),
            **{model: result[model + "_deg"] - float(historical[model]) for model in MODELS},
        }
        result["batch8_mask_exact"] = True
        result["batch8_differences_deg"] = differences
        result["batch8_full_vqf_abs_gap_deg"] = abs(differences["full_vqf"])
        for model in MODELS:
            result[model + "_minus_batch8_deg"] = differences[model]
        if abs(differences["full_vqf"]) > VQF_TOL or any(abs(differences[m]) > SHICE_TOL for m in MODELS):
            result.update(status="batch8_score_mismatch", detail=f"Batch8 pre-rounding differences in degrees: {differences}")
    return result


def score_case(receipt, official, catalog, models, archive, masks, run_hash):
    identity = {key: official[key] for key in IDENTITY}
    identity["run_sha256"] = run_hash
    if receipt.exists():
        row = read(receipt)
        require(all(row[k] == value for k, value in identity.items()), f"Receipt identity differs: {receipt}")
        if row["status"] in ("scored", "source_unavailable"):
            return row
    started = perf_counter()
    if official["status"] == "source_unavailable":
        result = {"status": "source_unavailable", "detail": official["detail"]}
    else:
        try:
            raw, _ = catalog.native(official["dataset"], official["record"])
            result = score_record(raw, official, models, archive, masks)
        except (ValueError, RuntimeError, OSError, KeyError, IndexError, AssertionError) as exc:
            result = {"status": "evaluation_failed", "detail": f"{type(exc).__name__}: {exc}"}
    row = {**identity, **result, "completed_at": now(), "scoring_seconds": perf_counter() - started}
    write_receipt(receipt, row)
    detail = row.get("detail", "")
    print(identity["dataset"], identity["record"], row["status"], round(row["scoring_seconds"], 3), detail, flush=True)
    return row


def summarize(rows):
    stats = {}
    for domain in sorted({r["dataset"] for r in rows}):
        requested = [r for r in rows if r["dataset"] == domain]
        scored = [r for r in requested if r["status"] == "scored"]
        metric_means = {k:fmean(r[k] for r in scored) for k in SCORES} if scored else {}
        stats[domain] = {"records":len(requested), "scored":len(scored), "scored_frames":sum(r["scored_frames"] for r in scored),
                         "empty_block_compatibility_records":sum(r["causal_empty_block_compatibility"] for r in scored),
                         "mean_rmse_deg":metric_means,
                         "mean_minus_c038_deg":{model:fmean(r[model + "_minus_c038_deg"] for r in scored)
                                                for model in MODELS} if scored else {},
                         "actual_scoring_seconds":sum(r["scoring_seconds"] for r in requested)}
    return stats


def write_panel(path, rows):
    cols = (*IDENTITY, "status", "detail", "native_frames", "scored_frames", "scoring_indices_sha256", "rate_hz", "releases",
            "causal_empty_block_compatibility", *SCORES, "official_full_vqf_deg", "full_vqf_minus_official_deg",
            *[m + "_minus_c038_deg" for m in MODELS], "batch8_mask_exact", "batch8_full_vqf_abs_gap_deg",
            *[m + "_minus_batch8_deg" for m in MODELS], "scoring_seconds", "completed_at")
    with atomic_output(path, encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def panel_inputs(lock_path, tables):
    catalog = Catalog(lock_path)
    transfer = TransferCatalog(lock_path)
    panels = {
        "main": (catalog, catalog.lock["formal_cases"], read_csv(tables["main"])),
        "transfer": (transfer, transfer.cases(), read_csv(tables["transfer"])),
    }
    for tag, (db, cases, rows) in panels.items():
        official = {(r["dataset"], r["record"]):r for r in rows}
        planned = {(r["dataset"], r["record"]):r for r in cases}
        require(len(official) == len(rows) == len(cases) == (365 if tag == "main" else 33) and planned.keys() == official.keys(),
                f"{tag} membership differs from the registered input")
        for identity, row in official.items():
            require(all(row[k] == v for k,v in planned[identity].items() if k in IDENTITY),
                    f"{tag} group, side, or setting differs: {identity}")
            expected = "source_unavailable" if identity in getattr(db, "missing", {}) else "scored"
            require(row["status"] == expected, f"Unexpected official status for {identity}: {row['status']}")
    return panels


def main():
    if not LATENCY.is_file():
        raise SystemExit("Wait for the completed SHICE native CUDA latency receipt")
    read(LATENCY)
    frozen = read(FINAL / "process.json")
    require(frozen["status"] == "complete" and sha256(FINAL_LOCK) == frozen["selection_sha256"], "Official table lock differs")
    tables = {tag: FINAL / f"{tag}.csv" for tag in ("main", "transfer")}
    for tag, table in tables.items():
        require(sha256(table) == frozen["tables"][tag]["sha256"], f"Official {tag} table checksum differs")
    require(sha256(OLD_LOCK) == OLD_LOCK_SHA, "Historical selection lock checksum differs")
    historical = read(OLD_LOCK)
    previous_protocol = read(OLD_PROTOCOL)
    old = read(OLD_RESULTS)
    require(previous_protocol["lock_sha256"] == OLD_LOCK_SHA and sha256(OLD_MASKS) == old["masks_sha256"],
            "Historical SHICE archive checksum differs")
    archived_names = set(previous_protocol["heldout_records"])
    archive = {row["sequence"]:row for row in read_csv(OLD_TABLE) if row["sequence"] in archived_names}
    require(len(archive) == 20, "Historical R5/R6 archive must contain 20 holdout records")
    for key, digest in historical["evidence"].items():
        require(sha256(TRANSFER / key) == digest, f"Historical frozen evidence changed: {key}")
    ref_paths = [
        FINAL_LOCK, OLD_LOCK, OLD_TABLE, OLD_MASKS, OLD_RESULTS, OLD_PROTOCOL, LATENCY, Path(__file__).resolve(),
        ROOT/"review/native_blocks.py", ROOT/"review/sample_rate_audit.json", ROOT/"review/native_catalog.py",
        ROOT/"review/score_final.py", ROOT/"review/supplemental_data.py", ROOT/"review/davis_handheld.py",
        ROOT/"review/penn_handheld.py", TRANSFER/"batch6/intrinsic.py", TRANSFER/"models.py",
        TRANSFER/"review/replay_native.py", *tables.values(),
        *[TRANSFER/historical["models"][m]["path"] for m in MODELS],
    ]
    fingerprints = {os.path.relpath(path, ROOT).replace("\\", "/"):sha256(path) for path in ref_paths}
    run = {"source_sha256": fingerprints, "selection_sha256": frozen["selection_sha256"],
           "historical_selection_sha256": OLD_LOCK_SHA, "models": {m:historical["models"][m] for m in MODELS},
           "device":"cpu", "torch_threads":THREADS, "torch_version":str(torch.__version__),
           "scope":"365 official main plus 33 preregistered transfer rows. C038 is read from the official table.",
           "release":"Frozen 64 x 10 ms feature window; 100 ms release on current native clock. Earlier observed sample and VQF state fill empty blocks.",
           "tolerances_deg":{"full_vqf":VQF_TOL,"historical_r5_r6_models":SHICE_TOL}}
    OUT.mkdir(parents=True, exist_ok=True)
    with single_job(OUT / "scoring.lock"):
        started = perf_counter()
        process = {"pid":os.getpid(),"started_at":now(),"script":str(Path(__file__).resolve()),"tables":{}}
        path = OUT / "run.json"
        if path.exists():
            require(read(path) == run, "Resume inputs differ from the original scoring job")
        else:
            write_receipt(path, run)
        run_hash = sha256(path)
        write_receipt(OUT/"process.json", {**process,"status":"loading"})
        torch.set_num_threads(THREADS)
        panels = panel_inputs(FINAL_LOCK, tables)
        require({r["record"] for r in panels["main"][2] if r["dataset"]=="SHICE"} == archived_names,
                "Current SHICE main panel differs from historical R5/R6")
        models = load_checkpoints(historical)
        all_rows = {}
        with np.load(OLD_MASKS, allow_pickle=False) as masks:
            for tag, (catalog, _, official) in panels.items():
                write_receipt(OUT/"process.json", {**process,"status":f"scoring_{tag}"})
                phase_started = perf_counter()
                receipts = OUT/"receipts"/tag
                receipts.mkdir(parents=True, exist_ok=True)
                rows = [score_case(receipts/f"{i:03d}.json", row, catalog, models, archive, masks, run_hash)
                        for i,row in enumerate(official,1)]
                require(sha256(tables[tag]) == frozen["tables"][tag]["sha256"], f"{tag} official source changed")
                output = OUT/f"{tag}.csv"
                write_panel(output, rows)
                process["tables"][tag] = {"published_at":now(),"wall_seconds":perf_counter()-phase_started,
                                          "sha256":sha256(output),"status_counts":dict(Counter(r["status"] for r in rows))}
                all_rows[tag] = rows
                print("published",tag,process["tables"][tag],flush=True)

        require(all(sha256(ROOT/name)==expected for name,expected in fingerprints.items()), "Source changed during baseline scoring")
        scored = [r for rows in all_rows.values() for r in rows if r["status"]=="scored"]
        shice = [r for r in all_rows["main"] if r["dataset"]=="SHICE"]
        checked = [r for r in shice if r.get("batch8_mask_exact")]
        errors = [r for rows in all_rows.values() for r in rows if r["status"] not in ("scored","source_unavailable")]
        summary = {**process,"completed_at":now(),"wall_seconds":perf_counter()-started,"run_sha256":run_hash,
                   "domains":{tag:summarize(rows) for tag,rows in all_rows.items()},
                   "max_full_vqf_official_abs_gap_deg":max((abs(r["full_vqf_minus_official_deg"]) for r in scored),default=None),
                   "batch8_shice":{"registered":len(shice),"exact_native_masks":len(checked),
                                   "score_parity_passes":sum(r["status"]=="scored" for r in checked),
                                   "max_abs_gaps_deg":{m:max((abs(r["batch8_differences_deg"][m]) for r in checked),default=None)
                                                       for m in ("full_vqf",*MODELS)}},
                   "unscored":[{k:r[k] for k in ("dataset","record","group","status","detail")}
                              for rows in all_rows.values() for r in rows if r["status"]!="scored"],
                   "source_sha256":fingerprints}
        write_receipt(OUT/"summary.json", summary)
        write_receipt(OUT/"process.json", {**process,"status":"complete_with_errors" if errors else "complete",
                                          "completed_at":summary["completed_at"],"wall_seconds":summary["wall_seconds"],
                                          "summary_sha256":sha256(OUT/"summary.json")})
        return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
