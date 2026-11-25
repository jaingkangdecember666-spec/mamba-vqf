"""Test a frozen SHICE predictor against five prescribed native encoder offsets."""
import csv
import hashlib
import math
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT
from locked_replay import SelectedPredictor, sha256
from native_catalog import Catalog
from optical_score import metric
from score_final import atomic_output, read, single_job, write_receipt


OUT = ROOT / "final/shice-clock-sensitivity"
PLAN = OUT / "protocol.json"
LOCK = ROOT / "selection-lock.json"
TABLE = ROOT / "final/main.csv"
FIXED = ROOT / "review/vqf_response_controls_final.json"
SHIFTS = (-5, -1, 0, 1, 5)
COLUMNS = ("c038_deg", "vqf3_deg", "vqf1p5_deg")
UP = np.array([0., 0., 1.])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now().astimezone().isoformat()


def scoring_hash(indices):
    return hashlib.sha256(indices.astype("<i8", copy=False).tobytes()).hexdigest()


def common_centers(official_mask):
    radius = max(map(abs, SHIFTS))
    width = 2 * radius + 1
    excluded = np.r_[0, np.cumsum(~official_mask, dtype=np.int64)]
    return np.flatnonzero(excluded[width:] == excluded[:-width]) + radius


def fast_vqf(signal, hz, observed, indices):
    previous = np.maximum.accumulate(np.where(observed, np.arange(len(signal)), -1))
    filled = signal[previous.clip(min=0)].copy()
    filled[previous < 0] = 0.
    gyro = np.ascontiguousarray(filled[:, 3:], dtype=np.float64)
    accel = np.ascontiguousarray(filled[:, :3], dtype=np.float64)
    states = VQF(1. / hz, tauAcc=1.5).updateBatch(gyro, accel)
    return Rotation.from_quat(states["quat6D"][indices], scalar_first=True).inv().apply(UP)


def nominal_input(official, fixed, catalog, predictor, tolerance):
    name = official["record"]
    (signal, hz, truth, mask), _ = catalog.native("SHICE", name)
    observed = np.isfinite(signal).all(axis=1)
    official_mask = mask & observed
    indices = np.flatnonzero(official_mask)
    digest = scoring_hash(indices)
    require(len(indices) == int(official["scored_frames"]) == fixed["scored_frames"]
            and digest == official["scoring_indices_sha256"] == fixed["scoring_indices_sha256"],
            "Native indices disagree with the registered masks")
    require(np.allclose(np.linalg.norm(truth[indices],axis=1),1.,rtol=0.,atol=1e-6), "Native encoder up must be unit length")
    center = common_centers(official_mask)
    require(len(center)>0 and np.all(official_mask[center+min(SHIFTS)]) and np.all(official_mask[center+max(SHIFTS)]),
            "No continuous jointly valid encoder indices")

    directions = predictor.predict(signal, hz, indices)
    scores = {
        "c038_deg":directions["candidate_deg"],
        "vqf3_deg":directions["full_vqf_deg"],
        "vqf1p5_deg":fast_vqf(signal, hz, observed, indices),
    }
    observed_nominal = {k:metric(v, truth[indices]) for k,v in scores.items()}
    expected_nominal = {
        "c038_deg":float(official["candidate_deg"]),
        "vqf3_deg":float(official["full_vqf_deg"]),
        "vqf1p5_deg":float(fixed["fast_deg"]),
    }
    deltas = {k:observed_nominal[k]-expected_nominal[k] for k in COLUMNS}
    require(all(math.isfinite(v) and abs(v)<=tolerance for v in deltas.values()),
            f"Nominal replay disagrees with official full-mask score: {deltas}")
    require(abs(float(fixed["candidate_deg"])-expected_nominal["c038_deg"])<=tolerance
            and abs(float(fixed["default_deg"])-expected_nominal["vqf3_deg"])<=tolerance,
            "Archived fixed-control table does not match the formal comparator")

    available = np.searchsorted(indices,center)
    np.testing.assert_array_equal(indices[available],center)
    predictors = {k:v[available] for k,v in scores.items()}
    scenarios = []
    for shift in SHIFTS:
        shifted_ref = truth[center+shift]
        values = {k:metric(v,shifted_ref) for k,v in predictors.items()}
        require(all(math.isfinite(v) for v in values.values()), f"Nonfinite score at offset {shift}")
        scenarios.append({"offset_native_samples":shift,"offset_ms":1e3*shift/hz,**values})
    path = Path(catalog.rows["SHICE",name]["path"])
    archive = path.parents[1]/"Label_quality_audit_20261116"/name/"screening_masks.npz"
    return {
        "status":"scored","rate_hz":hz,"native_interval_ms":1e3/hz,"native_frames":len(signal),
        "official_frames":len(indices),"official_scoring_indices_sha256":digest,
        "official_full_mask_deg":expected_nominal,"nominal_replay_full_mask_deg":observed_nominal,
        "nominal_replay_minus_official_deg":deltas,
        "common_frames":len(center),"common_scoring_indices_sha256":scoring_hash(center),
        "common_removed_frames":len(indices)-len(center),"common_first_native_index":int(center[0]),
        "common_last_native_index":int(center[-1]),"offset_scores":scenarios,
        "raw_imu_encoder_csv_sha256":sha256(path),"screening_mask_sha256":sha256(archive),
    }


def evaluate(path, official, fixed, catalog, predictor, tolerance, manifest_hash):
    identity = {"dataset":official["dataset"],"record":official["record"],"group":official["group"],
                "run_sha256":manifest_hash}
    if path.exists():
        previous = read(path)
        require(all(previous[k]==value for k,value in identity.items()), "Receipt identity differs")
        if previous["status"]=="scored":
            return previous
    started = perf_counter()
    try:
        result = nominal_input(official,fixed,catalog,predictor,tolerance)
    except (ValueError,RuntimeError,OSError,KeyError,IndexError,AssertionError) as exc:
        result = {"status":"evaluation_failed","detail":f"{type(exc).__name__}: {exc}"}
    item = {**identity,**result,"completed_at":now(),"seconds":perf_counter()-started}
    write_receipt(path,item)
    print(official["record"],item["status"],item.get("official_frames"),item.get("common_frames"),
          item.get("detail",""),flush=True)
    return item


def summarize(rows):
    means = {key:fmean(row[key] for row in rows) for key in COLUMNS}
    default = means["vqf3_deg"]
    fast = means["vqf1p5_deg"]
    return {
        "records":len(rows),"record_equal_rmse_deg":means,
        "c038_reduction_vs_vqf3_pct":100.*(1.-means["c038_deg"]/default),
        "c038_reduction_vs_vqf1p5_pct":100.*(1.-means["c038_deg"]/fast),
        "c038_wins_vs_vqf3":sum(row["c038_deg"]<row["vqf3_deg"] for row in rows),
        "c038_wins_vs_vqf1p5":sum(row["c038_deg"]<row["vqf1p5_deg"] for row in rows),
        "vqf1p5_wins_vs_vqf3":sum(row["vqf1p5_deg"]<row["vqf3_deg"] for row in rows),
        "c038_ties_vs_vqf3":sum(row["c038_deg"]==row["vqf3_deg"] for row in rows),
        "c038_ties_vs_vqf1p5":sum(row["c038_deg"]==row["vqf1p5_deg"] for row in rows),
    }


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with atomic_output(path,encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def export(rows):
    full, shifted = [],[]
    for row in rows:
        ident = {k:row[k] for k in ("dataset","record","group","rate_hz","native_interval_ms")}
        full.append({**ident,"scored_frames":row["official_frames"],
                     "scoring_indices_sha256":row["official_scoring_indices_sha256"],
                     **{f"official_{k}":v for k,v in row["official_full_mask_deg"].items()},
                     **{f"replay_{k}":v for k,v in row["nominal_replay_full_mask_deg"].items()},
                     **{f"replay_minus_official_{k}":v for k,v in row["nominal_replay_minus_official_deg"].items()}})
        shifted.extend({**ident,"common_scored_frames":row["common_frames"],
                        "common_scoring_indices_sha256":row["common_scoring_indices_sha256"],**case}
                       for case in row["offset_scores"])
    write_csv(OUT/"nominal-full.csv",full)
    write_csv(OUT/"sequences.csv",shifted)

    groups = {}
    for name in ("R5","R6","R5+R6"):
        participants = [row for row in rows if name=="R5+R6" or row["group"]==name]
        records = {row["record"] for row in participants}
        original = sum(row["official_frames"] for row in participants)
        common = sum(row["common_frames"] for row in participants)
        groups[name] = {
            "nominal_full_scored_frames":original,"common_scored_frames":common,"common_fraction":common/original,
            "nominal_full_official":summarize([row["official_full_mask_deg"] for row in participants]),
            "nominal_full_replay":summarize([row["nominal_replay_full_mask_deg"] for row in participants]),
            "common_by_offset_native_samples":{
                f"{shift:+d}":summarize([row for row in shifted if row["record"] in records and row["offset_native_samples"]==shift])
                for shift in SHIFTS},
        }
    return groups


def main():
    started_at = now()
    start = perf_counter()
    plan = read(PLAN)
    protocol_hash = sha256(PLAN)
    require(tuple(plan["shift_native_samples"])==SHIFTS, "Prescribed offsets changed")
    order = plan["authorized_records"]
    require(set(order)=={f"R{dev}-{i}" for dev in (5,6) for i in range(1,11)} and len(order)==20,
            "Only the 20 registered R5/R6 records are authorized")
    source_hashes = plan["source_sha256"]
    for name,digest in source_hashes.items():
        require(sha256(ROOT/name)==digest, f"Preregistered source changed: {name}")
    fixed_document = read(FIXED)
    require(fixed_document["selection_lock_sha256"]==source_hashes["selection-lock.json"]
            and fixed_document["source_csv_sha256"]==source_hashes["final/main.csv"],"Fixed fast VQF provenance differs")
    with TABLE.open(encoding="utf-8-sig",newline="") as stream:
        official = {row["record"]:row for row in csv.DictReader(stream) if row["dataset"]=="SHICE" and row["record"] in order}
    fixed = {row["record"]:row for row in fixed_document["records"] if row["dataset"]=="SHICE" and row["record"] in order}
    require(len(official)==len(fixed)==20 and all(official[n]["status"]==fixed[n]["status"]=="scored" for n in order),
            "Published SHICE coverage differs")
    run = {"protocol_sha256":protocol_hash,"source_sha256":source_hashes,
           "script_sha256":sha256(Path(__file__)),"device":plan["predictions_device"],
           "torch_version":str(torch.__version__),"torch_threads":plan["torch_threads"]}
    run_path = OUT/"run.json"
    with single_job(OUT/"scoring.lock"):
        if run_path.exists():
            require(read(run_path)==run,"Existing receipts have different inputs")
        else:
            write_receipt(run_path,run)
        run_hash = sha256(run_path)
        process = {"pid":os.getpid(),"started_at":started_at,"protocol_sha256":protocol_hash,"run_sha256":run_hash}
        write_receipt(OUT/"process.json",{**process,"status":"scoring","expected":20})
        torch.set_num_threads(plan["torch_threads"])
        catalog = Catalog(LOCK)
        predictor = SelectedPredictor(LOCK,plan["predictions_device"])
        receipts = OUT/"receipts"
        receipts.mkdir(exist_ok=True)
        rows = [evaluate(receipts/f"{name}.json",official[name],fixed[name],catalog,predictor,
                         plan["parity_tolerance_deg"],run_hash) for name in order]
        require(sha256(PLAN)==protocol_hash and all(sha256(ROOT/name)==digest for name,digest in source_hashes.items()),
                "A preregistered input changed during scoring")
        status_counts = dict(Counter(row["status"] for row in rows))
        successful = [row for row in rows if row["status"] == "scored"]
        complete = len(successful)==20
        groups = export(successful) if complete else {}
        summary = {
            **process,"status":"complete" if complete else "incomplete","completed_at":now(),
            "wall_seconds":perf_counter()-start,"expected_records":20,"status_counts":status_counts,
            "offset_definition":plan["sign"],"common_mask_rule":plan["eligibility"],
            "max_nominal_official_abs_delta_deg":{
                k:max((abs(r["nominal_replay_minus_official_deg"][k]) for r in successful),default=None) for k in COLUMNS},
            "minimum_native_interval_ms":min((r["native_interval_ms"] for r in successful),default=None),
            "maximum_native_interval_ms":max((r["native_interval_ms"] for r in successful),default=None),
            "groups":groups,"source_sha256":source_hashes,
            "receipt_sha256":{f"{row['record']}.json":sha256(receipts/f"{row['record']}.json") for row in rows},
            "outputs_sha256":{name:sha256(OUT/name) for name in ("nominal-full.csv","sequences.csv")} if complete else {},
            "failures":[row for row in rows if row["status"]!="scored"],
        }
        write_receipt(OUT/"summary.json",summary)
        write_receipt(OUT/"process.json",{**process,"status":summary["status"],"completed_at":summary["completed_at"],
                                          "wall_seconds":summary["wall_seconds"],"scored_records":len(successful),
                                          "summary_sha256":sha256(OUT/"summary.json")})
        return int(not complete)


if __name__ == "__main__":
    raise SystemExit(main())
