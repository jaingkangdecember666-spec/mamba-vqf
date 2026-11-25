"""Audit the locked evaluation, including chronological order and paired record scores."""

import hashlib
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, median

import numpy as np

from audit_rounds import BATCH, HERE, check_quaternion_record, check_scores, fingerprint, read, write


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def completed(index, kind):
    found = [r for r in index.values() if r["config"]["kind"] == kind and r["status"] == "completed"]
    if len(found) != 1:
        raise RuntimeError(f"Expected one completed {kind} step, currently found {len(found)}")
    return found[0]


def paired_summary(rows, baseline):
    diffs = {r["name"]: r["rmse_deg"]["locked"] - r["rmse_deg"][baseline] for r in rows}
    worst = max(diffs, key=diffs.get)
    return {"wins": sum(d < -1e-8 for d in diffs.values()), "ties": sum(abs(d) <= 1e-8 for d in diffs.values()),
            "n": len(diffs), "mean_locked_minus_comparator_deg": fmean(diffs.values()),
            "median_locked_minus_comparator_deg": median(diffs.values()), "worst_record": worst,
            "worst_delta_deg": diffs[worst]}


def r5_sensitivity(candidates, baseline_id, selected_id):
    scores = {r["round_id"]:{s["name"]:s["rmse_deg"] for s in r["r5_sequences"]} for r in candidates}
    baseline = next(r for r in candidates if r["round_id"] == baseline_id)
    base_dev = baseline["development_rmse_deg"]
    records = sorted(scores[baseline_id],key=lambda name:int(name.split("-")[1]))
    assert all(set(values) == set(records) for values in scores.values())
    eligible = sorted((r for r in candidates if r["eligible_for_selection"]),
                      key=lambda r:(r["r5"]["macro_rmse_deg"],r["development_rmse_deg"],r["r5"]["worst_rmse_deg"]))
    assert eligible[0]["round_id"] == selected_id
    runner = eligible[1] if len(eligible) > 1 else None
    by_omission = []
    for omit in records:
        subset = [s for s in records if s != omit]
        base = fmean(scores[baseline_id][s] for s in subset)
        ranking = []
        for candidate in candidates:
            identity = candidate["round_id"]
            dev = candidate["development_rmse_deg"]
            values = [scores[identity][s] for s in subset]
            avg = fmean(values)
            qualifies = identity == baseline_id or (dev < base_dev and avg < base)
            config = candidate.get("config",{})
            if config.get("kind") == "bias_physical_head":
                comparator = scores.get(config["paired_zero_round"])
                active = any(r["online_calibration"]["active_score_frames"] > 0 for r in candidate["r5_sequences"] if r["name"] != omit)
                qualifies = qualifies and active and comparator is not None and avg < fmean(comparator[n] for n in subset)
            if qualifies:
                ranking.append((identity,avg,dev,max(values)))
        winner = min(ranking,key=lambda row:row[1:])
        chosen = fmean(scores[selected_id][s] for s in subset)
        by_omission.append({"omitted":omit,"winner":winner[0],"winner_subset_deg":winner[1],
                            "locked_subset_deg":chosen,"eligible_candidates":len(ranking)})
    return {"full_runner_up":runner["round_id"] if runner else None,
            "full_runner_up_margin_deg":runner["r5"]["macro_rmse_deg"]-eligible[0]["r5"]["macro_rmse_deg"] if runner else None,
            "original_wins":sum(r["winner"] == selected_id for r in by_omission),"omissions":by_omission}


def verify_r5_selection(selection, base_step, r5, index, shortlisted_ids, r5_frames, r5_native, chosen):
    """Reconstruct all eligibility flags, the winning tie-break and record influence using R5 alone."""
    scored_ids = [r["round_id"] for r in selection["r5_candidates"]]
    scored_by_id = {r["round_id"]:r for r in selection["r5_candidates"]}
    failed_ids = [r["round_id"] for r in selection["failed_candidates"]]
    assert len(scored_ids + failed_ids) == len(shortlisted_ids) and set(scored_ids + failed_ids) == shortlisted_ids
    base_dev = base_step["dev_metrics"]["macro_rmse_deg"]
    for candidate in selection["r5_candidates"]:
        row = index[candidate["round_id"]]
        dev = check_r5_scores(candidate,index,r5_frames,r5_native)
        qualifies = row["round_id"] == base_step["round_id"] or (dev < base_dev and candidate["r5"]["macro_rmse_deg"] < r5["macro_rmse_deg"])
        if row["config"]["kind"] == "bias_physical_head":
            count = sum(r["online_calibration"]["active_score_frames"] for r in candidate["r5_sequences"])
            assert all(0 <= r["online_calibration"]["active_score_frames"] <= r["frames"] for r in candidate["r5_sequences"])
            identity = row["config"]["paired_zero_round"]
            zero = scored_by_id.get(identity)
            better = bool(zero and candidate["r5"]["macro_rmse_deg"] < zero["r5"]["macro_rmse_deg"])
            assert candidate["bias_r5_requirement"] == {"active_score_frames":count,"paired_zero_round":identity,"beats_same_seed_zero_head_on_r5":better}
            qualifies = qualifies and count > 0 and better
        assert candidate["eligible_for_selection"] == qualifies
    eligible = [r for r in selection["r5_candidates"] if r["eligible_for_selection"]]
    reconstructed = min(eligible, key=lambda r: (r["r5"]["macro_rmse_deg"], r["development_rmse_deg"], r["r5"]["worst_rmse_deg"]))
    assert reconstructed == chosen
    sensitivity = None
    if base_step["round_id"] in scored_ids:
        sensitivity = r5_sensitivity(selection["r5_candidates"],base_step["round_id"],chosen["round_id"])
    return sensitivity


def check_latency(final, r6_frame_counts):
    latency = final["latency"]
    old, current = latency["prior_comparators"], latency["locked"]
    assert latency["machine"].casefold() == read(BATCH/"status.json")["host"].casefold()
    assert old["gpu"] == latency["accelerator"] and old["benchmark_record"] == current["record"] == "R6-3"
    assert old["native_frames"] > 0 and old["correction_windows_per_method"] > 0 and latency["gru"]["step_hz"] > 0
    with np.load(BATCH.parent/"gru-tilt/data/R6-3.npz",allow_pickle=False) as native:
        assert math.isclose(current["nominal_native_hz"],1/float(native["dt"]),rel_tol=1e-8)
    operations = [old["full_vqf_per_frame_ms"],old["quaternion_apply_per_frame_ms"],latency["gru"]["CPU"],latency["gru"]["CUDA"],
                  *(old["correction_update_ms"][m] for m in ("physical_only","timesfm")),current["native_step"]]
    operations.extend(current[k] for k in ("correction_update","cpu_correction_update","gate_update","apply_held_correction","online_bias_block_update") if k in current)
    for data in operations:
        suffix = "_ms" if "p50_ms" in data else ""
        values = [data[k+suffix] for k in ("p50","p95","max")]
        assert all(math.isfinite(v) for v in values) and 0 < values[0] <= values[1] <= values[2]
    delay = final.get("locked_delay_replay")
    if delay:
        update = current["gate_update"] if "gate_update" in current else current["correction_update"]
        assert math.isclose(delay["delay_ms"],update["p95_ms"],abs_tol=1e-8)
        assert set(r["name"] for r in delay["sequences"]) == set(r6_frame_counts)
        check_scores(delay["sequences"],delay["R6"],r6_frame_counts)


def original_rows():
    root = BATCH.parent
    r6 = read(root / "tilt-correction/test.json")
    external = read(root / "tilt-correction/external_test.json")
    gru = read(root / "gru-tilt/test.json")
    rows = {"R6": {r["recording"]: r for r in r6["r6_records"]}}
    for name, data in external["datasets"].items():
        rows[name] = {r["recording"]: r["protocols"]["main"] for r in data["records"]}
    for name, data in gru["tests"].items():
        for row in data["per_sequence"]:
            other = rows[name][row["recording"]]
            assert other["scoring_frames"] == row["scoring_frames"]
            other["rmse_deg"]["gru"] = row["rmse_deg"]["gru"]
    return rows


def check_baseline_references(references):
    first = read(HERE/"baseline_audit.json")
    source_hashes = {key.replace("\\","/"):value for key,value in first["source_sha256"].items()}
    weights = first["checkpoint_sha256"]
    expected = {"R6":("tilt-correction/test.json",source_hashes["tilt-correction/test.json"]),
                "external":("tilt-correction/external_test.json",source_hashes["tilt-correction/external_test.json"]),
                "gru":("gru-tilt/test.json",source_hashes["gru-tilt/test.json"]),
                "light_weights":("tilt-correction/physical_only_head.pt",weights["physical_only_head.pt"]),
                "timesfm_weights":("tilt-correction/timesfm_head.pt",weights["timesfm_head.pt"]),
                "gru_weights":("gru-tilt/longer/gru.pt",weights["selected_gru.pt"])}
    assert set(expected) == set(references)
    verified = {}
    for key,(relative,original_sha) in expected.items():
        path = (BATCH.parent/relative).resolve()
        assert Path(references[key]["path"]).resolve() == path, key
        assert references[key]["sha256"] == digest(path) == original_sha, key
        verified[relative] = original_sha
    gain_file = "tilt-correction/validation.json"
    assert digest(BATCH.parent/gain_file) == source_hashes[gain_file] == weights["validation.json"]
    verified[gain_file] = source_hashes[gain_file]
    return verified


def raw_shice_samples(name):
    """Reconstruct SHICE samples, label orientation and masks directly from the aligned source."""
    import csv
    from scipy.spatial.transform import Rotation
    from verify_baselines import AUDIT, SHICE

    source = SHICE / f"{name.split('-')[0]}_aligned_formal" / f"{name}.csv"
    columns = ["sample_index", "time_s", "is_interpolated",
               *(f"imu_{kind}_{axis}" for kind in ("acc", "gyr") for axis in "xyz"),
               *(f"opt_quat_{axis}" for axis in "wxyz")]
    with source.open(encoding="utf-8-sig") as stream:
        headings = next(csv.reader(stream))
        raw = np.loadtxt(stream, delimiter=",", quotechar='"', usecols=[headings.index(k) for k in columns])
    indices = np.arange(len(raw))
    np.testing.assert_array_equal(raw[:, 0], indices)
    dt = float(np.median(np.diff(raw[:, 1])))
    readings, quats = raw[:, 3:9], raw[:, 9:13]
    observed = (raw[:, 2] == 0) & np.isfinite(readings).all(axis=1)
    most_recent = np.maximum.accumulate(np.where(observed, indices, -1))
    signal = readings[most_recent.clip(min=0)].astype(np.float32)
    signal[most_recent < 0] = 0
    valid_quat = np.isfinite(quats).all(axis=1) & (np.linalg.norm(quats,axis=1) > 0)
    reference = np.zeros((len(raw), 3), dtype=np.float32)
    reference[valid_quat] = Rotation.from_quat(quats[valid_quat], scalar_first=True).inv().apply((0,-1,0))
    with np.load(AUDIT / name / "screening_masks.npz", allow_pickle=False) as masks:
        np.testing.assert_array_equal(masks["sample_index"], indices)
        context = masks["context_ok"].astype(bool)
        score = (raw[:, 1] >= 10) & valid_quat & observed & context & masks["observed_loss_ok"].astype(bool)
    return {"source":source, "time_s":raw[:,1], "imu":signal, "dt":dt, "score":score, "observed":observed,
            "context":context, "reference":reference}


def shice_frame_evidence(name):
    return score_mask_evidence(raw_shice_samples(name)["score"])


def score_mask_evidence(mask):
    indices = np.flatnonzero(mask).astype("<i8", copy=False)
    return {"frames":len(indices), "score_frame_indices_sha256":hashlib.sha256(indices.tobytes()).hexdigest()}


def check_frame_evidence(row, evidence):
    assert row["frames"] == evidence["frames"], row["name"]
    assert row["score_frame_indices_sha256"] == evidence["score_frame_indices_sha256"], row["name"]


CORE_METHODS = ("full_vqf", "physical_only", "timesfm", "gru", "locked")
SHORTLIST_DIAGNOSTICS = {"ridge_seed_diagnostic", "mounted_ridge_alpha_seed_diagnostic", "mounted_sobol_ridge_diagnostic", "ridge_cap_pair_diagnostic", "sensor_hold_diagnostic",
                         "forecast_backbone_diagnostic", "timesfm_encoder_diagnostic", "reservoir_readout",
                         "causal_dct_readout", "accelerometer_relative_gain_audit"}


def checkpoint_records(snapshot, lock, expected):
    assert snapshot["selection_lock_sha256"] == lock["lock_sha256"], "逐序列清单锁定指纹不符"
    started, written = (datetime.fromisoformat(snapshot[k]) for k in ("started_at","at"))
    assert datetime.fromisoformat(lock["locked_at"]) <= started <= written, "逐序列清单时间顺序错误"
    assert isinstance(snapshot["datasets"],dict) and set(snapshot["datasets"]) <= set(expected)
    indexed = {}
    for dataset, rows in snapshot["datasets"].items():
        assert isinstance(rows,list)
        by_name = {r["name"]:r for r in rows}
        assert len(rows) == len(by_name) and set(by_name) <= set(expected[dataset]), dataset
        for name, row in by_name.items():
            assert row["frames"] == expected[dataset][name]["scoring_frames"] > 0, name
            sha = row["score_frame_indices_sha256"]
            assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), name
            for method in CORE_METHODS:
                value = row["rmse_deg"][method]
                assert math.isfinite(value) and 0 <= value <= 180, (name,method)
        indexed[dataset] = by_name
    return indexed


def compare_checkpoint_row(row, other):
    assert row["name"] == other["name"]
    check_frame_evidence(row,other)
    for method in CORE_METHODS:
        assert math.isclose(row["rmse_deg"][method],other["rmse_deg"][method],rel_tol=0,abs_tol=1e-8), (row["name"],method)


def check_r5_scores(candidate, index, frames, native):
    row = index[candidate["round_id"]]
    assert candidate["config_sha256"] == row["config_sha256"] == fingerprint(candidate["config"])
    metrics = row["dev_metrics"]
    dev = metrics.get("selection_rmse_deg",metrics.get("macro_rmse_deg"))
    assert math.isclose(candidate["development_rmse_deg"],dev,abs_tol=1e-8)
    assert {s["name"] for s in candidate["r5_sequences"]} == set(frames)
    for series in candidate["r5_sequences"]:
        check_frame_evidence(series,native[series["name"]])
    check_scores(candidate["r5_sequences"],candidate["r5"],frames)
    return dev


def replay_with_development_fields(available, cutoff):
    from unittest.mock import patch
    import selection_rules

    inputs = []
    metadata_files = {}
    seen_fields = {}
    for row in sorted(available,key=lambda r:r["round_id"]):
        if row["config"]["kind"] in SHORTLIST_DIAGNOSTICS:
            continue
        item = {key:row[key] for key in ("round_id","status","config","config_sha256","seed","batch","result")}
        original = row["dev_metrics"] or {}
        metrics = {k:original[k] for k in ("selection_rmse_deg","macro_rmse_deg") if k in original}
        for axis in ("device","motion","joint_holdout"):
            if axis in original:
                metrics[axis] = {k:original[axis][k] for k in ("macro_rmse_deg","worst_delta_deg") if k in original[axis]}
        item["dev_metrics"] = metrics
        inputs.append(item)
        if item["result"]:
            metadata_files[BATCH / item["result"]] = item["round_id"]

    def development_metadata(path):
        if path == BATCH / "stream-gru-development-summary.json":
            summary = read(path)
            approvals = summary["r5_approved"] if datetime.fromisoformat(summary["updated_at"]) <= cutoff else []
            values = {"r5_approved":[{"round_id":s["round_id"]} for s in approvals]}
        else:
            assert path in metadata_files, f"Shortlist requested an unregistered file: {path}"
            source = read(path)
            fields = ("mask_version","recurrent_training_context","eligible_for_r5","mean_default_fallback_fraction")
            values = {k:source[k] for k in fields if k in source}
        seen_fields[path.relative_to(BATCH).as_posix()] = values
        return values

    with patch.object(selection_rules,"read_json",development_metadata):
        shortlist = selection_rules.build_shortlist(inputs)
    return shortlist, {"rows":len(inputs),"omitted_diagnostic_rows":len(available)-len(inputs),
                       "metadata_files_used":len(seen_fields),"selection_code_sha256":digest(BATCH / "selection_rules.py"),
                       "input_sha256":fingerprint(inputs),"metadata_sha256":fingerprint(seen_fields)}


def preview_without_diagnostic(index, when, kind):
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from selection_rules import build_shortlist
    from verify_shortlist_reasons import wording_at

    at = datetime.fromisoformat(when)
    available = [r for r in sorted(index.values(),key=lambda r:r["round_id"]) if r["status"] == "completed"
                 and r["config"]["kind"] not in {"selection","final"} and datetime.fromisoformat(r["ended_at"]) <= at]
    assert any(r["config"]["kind"] == kind for r in available)
    full = build_shortlist(available)
    restricted,_ = replay_with_development_fields(available,at)
    assert full == restricted and full == build_shortlist([r for r in available if r["config"]["kind"] != kind])
    assert {"baseline","install_rotation"} <= {r["kind"] for r in full}
    return len(full), fingerprint(wording_at(full,at))


def reproduce_shortlist(index, manifest):
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from selection_rules import build_shortlist

    cutoff = datetime.fromisoformat(manifest["at"])
    available = [r for r in index.values() if r["status"] == "completed" and r["config"]["kind"] not in {"selection","final"}
                 and datetime.fromisoformat(r["ended_at"]) <= cutoff]
    regenerated = build_shortlist(available)
    assert regenerated == manifest["selected_before_r5"], "Development-only shortlist differs from the filed R5 shortlist"
    restricted, inputs = replay_with_development_fields(available,cutoff)
    assert restricted == regenerated, "The shortlist could not be reproduced from the permitted development fields"
    signature = fingerprint(regenerated)
    assert signature == manifest["shortlist_sha256"]
    replay = {"at":manifest["at"], "completed_rounds_available":len(available), "count":len(regenerated),
              "shortlist_sha256":signature, "restricted_development_inputs":inputs}
    requested = manifest["requested_by_supervisor"]
    actual_ids = [c["round_id"] for c in regenerated]
    assert len(requested)==len(set(requested)) and all(k in index for k in requested)
    replay["supervisor_request"] = {"count":len(requested),"added":[k for k in actual_ids if k not in requested],
                                   "omitted":[k for k in requested if k not in actual_ids]}
    observed_file = HERE/"r5_filing_observation.json"
    if observed_file.is_file():
        observed = read(observed_file)
        if observed["manifest_at"] == manifest["at"]:
            job = index[observed["selection_round"]]
            assert job["config"]["kind"]=="selection" and job["started_at"]==observed["started_at"]
            assert requested==job["config"]["candidate_rounds"]
            path = BATCH/"rounds"/job["round_id"]/"shortlist.json"
            assert digest(path)==observed["manifest_sha256"] and signature==observed["shortlist_sha256"]
            assert [c["round_id"] for c in regenerated]==observed["candidates"]
            assert [c["round_id"] for c in regenerated if c["kind"] in {"baseline","install_rotation"}]==observed["controls"]
            assert datetime.fromisoformat(job["started_at"])<=cutoff<=datetime.fromisoformat(observed["observed_at"])
            assert observed["scored"]+observed["failed"] <= len(regenerated)
            if not observed["progress_file_present"]:
                assert observed["scored"]==observed["failed"]==0
            replay["live_filing_observation"] = observed
    return replay


def audit_exported_quaternions(record, raw=None):
    from scipy.spatial.transform import Rotation

    raw = raw_shice_samples(record["name"]) if raw is None else raw
    with np.load(BATCH / record["quaternion_file"], allow_pickle=False) as stored:
        q = stored["q_wxyz"]
        np.testing.assert_array_equal(stored["sample_index"],np.arange(len(raw["time_s"])))
        np.testing.assert_array_equal(stored["time_s"],raw["time_s"])
    assert q.shape == (len(raw["time_s"]),4) and np.isfinite(q).all()
    norm_error = float(np.max(np.abs(np.linalg.norm(q,axis=1)-1)))
    assert math.isclose(norm_error,record["max_unit_norm_error"],abs_tol=1e-9)
    mask = raw["score"]
    assert int(mask.sum()) == record["scored_frames"]
    truth = raw["reference"][mask]
    predicted = Rotation.from_quat(q[mask],scalar_first=True).inv().apply((0,0,1))
    angles = np.degrees(np.arctan2(np.linalg.norm(np.cross(truth,predicted),axis=1),
                                  np.einsum("ij,ij->i",truth,predicted)))
    rmse = float(np.sqrt(np.mean(angles**2)))
    assert math.isclose(rmse,record["quaternion_tilt_rmse_deg"],abs_tol=2e-5)
    return {"name":record["name"], "source":str(raw["source"]), "quaternion_file":record["quaternion_file"],
            "native_frames":len(q), "scored_frames":int(mask.sum()), "raw_quaternion_rmse_deg":rmse,
            "maximum_norm_error":norm_error,"absolute_delta_vs_direction_deg":abs(rmse-record["direct_direction_rmse_deg"])}


def audit_development_quaternions(index):
    raw_by_record, outputs = {}, []
    for item in index:
        if item["config"]["kind"] != "quaternion_audit" or item["status"] != "completed":
            continue
        result = read(BATCH / item["result"])
        for cert in result["records"]:
            name = cert["name"]
            if name not in raw_by_record:
                raw_by_record[name] = raw_shice_samples(name)
            measured = audit_exported_quaternions(cert, raw_by_record[name])
            outputs.append({**measured,"round_id":item["round_id"],"candidate":result["parent_round"],
                            "protocol_passed":cert["passed"],
                            "max_prefix_rotation_delta_deg":max(p["max_rotation_delta_deg"] for p in cert["prefix_checks"])})
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),
            "candidates":len({v["candidate"] for v in outputs}),"unique_source_records":sorted(raw_by_record),
            "passed_outputs":sum(r["protocol_passed"] for r in outputs),"outputs":outputs}


def load_raw_backbone():
    import torch
    if str(BATCH.parent) not in sys.path:
        sys.path.insert(0, str(BATCH.parent))
    from tilt_model import load_backbone

    device = torch.device("cuda")
    return load_backbone(BATCH.parent / "checkpoint", device), device


def raw_locked_replay(name, config, state, backbone=None, device=None):
    if str(BATCH) not in sys.path:
        sys.path.insert(0, str(BATCH))
    from features import cache_record
    from final_eval import embed_external, score_candidate

    raw = raw_shice_samples(name)
    args = raw["imu"], raw["dt"], raw["score"], raw["context"], raw["reference"]
    needs_native_q = bool(config.get("installation")) or config["kind"] in {"ensemble","bias_physical_head"}
    data = cache_record(*args, config.get("parameters", {}), True, keep_scored_quaternion=needs_native_q)
    raw_coverage = None
    if config["kind"] == "bias_physical_head":
        from inspect_accelerometers import raw_offset_trace
        protocol = read(BATCH/"selection-lock.json")["frozen_bias_protocol"]["base_protocol"]
        assert fingerprint(protocol) == config["protocol_sha256"]
        trace,_,_,_ = raw_offset_trace(raw,config["minimum_online_poses"],protocol)
        active = np.linalg.norm(trace,axis=1) > 1e-12
        raw_coverage = int((active & raw["score"]).sum())
        adjusted = data
        if np.any(active & raw["observed"]):
            previous = np.maximum.accumulate(np.where(raw["observed"],np.arange(len(active)),-1))
            clean = raw["imu"].astype(np.float64)
            good = previous >= 0
            clean[good,:3] -= trace[previous[good]]
            adjusted = cache_record(clean,*args[1:],config["parameters"],True,keep_scored_quaternion=True)
        data = {"original":data,"adjusted":adjusted,"index":data["index"]}
    if config["kind"] == "hybrid":
        from hybrid_filter import pair
        data = pair(cache_record(*args, {}, True), data)
    embedding = None
    if config.get("embedding"):
        if backbone is None:
            backbone, device = load_raw_backbone()
        embedding = embed_external(data, config, backbone, device)
    obtained = float(score_candidate(data, config, state, embedding))
    return {"record":name, "source":str(raw["source"]), "source_sha256":digest(raw["source"]),
            **score_mask_evidence(raw["score"]), "locked_rmse_deg":obtained,"online_calibrated_frames":raw_coverage}


def replay_local_records(rows, config, weights):
    backbone, device = load_raw_backbone() if config.get("embedding") else (None, None)
    replays = {}
    for official in rows:
        name = official["name"]
        assert name not in replays, name
        measured = raw_locked_replay(name, config, weights, backbone, device)
        check_frame_evidence(official, measured)
        if measured["online_calibrated_frames"] is not None:
            assert measured["online_calibrated_frames"] == official["online_calibration"]["active_score_frames"], name
        measured["final_rmse_deg"] = official["rmse_deg"]["locked"]
        measured["absolute_difference_deg"] = abs(measured["locked_rmse_deg"] - measured["final_rmse_deg"])
        assert measured["absolute_difference_deg"] < 2e-4, measured
        replays[name] = measured
    return replays


def audit_final():
    index = {r["round_id"]: r for r in read(BATCH / "round_index.json")}
    final_step, selection_step = completed(index, "final"), completed(index, "selection")
    base_step = completed(index, "baseline")
    final, selection = (read(BATCH / r["result"]) for r in (final_step, selection_step))
    lock = read(BATCH / "selection-lock.json")
    chosen, lock_id = lock["selected"], lock["lock_sha256"]
    assert fingerprint({k:v for k,v in lock.items() if k != "lock_sha256"}) == lock_id
    assert lock_id == selection["lock_sha256"] == final_step["config"]["selection_sha256"]
    assert lock == final["lock"]
    baseline_sources = check_baseline_references(final["references"])
    assert digest(BATCH / lock["weights"]) == lock["weights_sha256"]
    assert chosen["config_sha256"] == index[chosen["round_id"]]["config_sha256"] == fingerprint(chosen["config"])
    if chosen["config"]["kind"] == "bias_physical_head":
        cfg = chosen["config"]
        frozen = lock["frozen_bias_protocol"]
        assert fingerprint(frozen["base_protocol"]) == cfg["protocol_sha256"] and frozen["online_minimum_poses"] == cfg["minimum_online_poses"]
        assert frozen["matched_experiment_sha256"] == cfg["matched_experiment_sha256"] == digest(BATCH/"rounds/A00003/result.json")
    shortlist = read((BATCH / selection_step["result"]).parent / "shortlist.json")
    assert fingerprint(shortlist["selected_before_r5"]) == lock["shortlist_sha256"]
    shortlist_replay = reproduce_shortlist(index,shortlist)
    shortlist_rows = shortlist["selected_before_r5"]
    shortlisted_ids = {r["round_id"] for r in shortlist_rows}
    assert len(shortlisted_ids) == len(shortlist_rows) and base_step["round_id"] in shortlisted_ids
    assert any(index[r]["config"]["kind"] == "install_rotation" for r in shortlisted_ids)
    assert all(index[r]["config"]["kind"] not in SHORTLIST_DIAGNOSTICS for r in shortlisted_ids)
    for identity in shortlisted_ids:
        candidate = index[identity]
        assert candidate["status"] == "completed"
        if candidate["config"]["kind"] == "ema_readout":
            from verify_ema import check_trial
            frames = {r["name"]:r["frames"] for r in base_step["dev_metrics"]["sequences"]}
            assert check_trial(candidate,read(BATCH / candidate["result"]),frames)["r5_qualified"]
        if candidate["config"]["kind"] == "stream_gru":
            revision = candidate["config"].get("recurrent_training_context")
            assert revision == "split-state-at-forbidden-context-v2"
            assert read(BATCH / candidate["result"])["recurrent_training_context"] == revision
    assert datetime.fromisoformat(shortlist["at"]) <= datetime.fromisoformat(lock["locked_at"]) <= datetime.fromisoformat(final["started_at"])
    assert set(lock["training"]) == set(base_step["partitions"]["development"])
    assert set(lock["validation"]) == set(base_step["partitions"]["r5"])
    r5 = read(BATCH / base_step["result"])["r5"]
    r5_frames = {r["name"]:r["frames"] for r in r5["sequences"]}
    r5_native = {name:shice_frame_evidence(name) for name in sorted(r5_frames)}
    sensitivity = verify_r5_selection(selection,base_step,r5,index,shortlisted_ids,r5_frames,r5_native,chosen)
    expected = original_rows()
    tests = {}
    methods = CORE_METHODS
    record_receipt = None
    receipt_path = (BATCH/final_step["result"]).with_name("final-records-progress.json")
    if receipt_path.is_file():
        receipt = read(receipt_path)
        journal = checkpoint_records(receipt,lock,expected)
        assert datetime.fromisoformat(receipt["at"]) <= datetime.fromisoformat(final["core_completed_at"])
        assert set(journal) == set(expected)
        for dataset, group in final["datasets"].items():
            assert set(journal[dataset]) == set(expected[dataset]), dataset
            for row in group["sequences"]:
                compare_checkpoint_row(row,journal[dataset][row["name"]])
        record_receipt = {"source":str(receipt_path.relative_to(BATCH)),"sha256":digest(receipt_path),
                          "started_at":receipt["started_at"],"at":receipt["at"],
                          "counts":{ds:len(rows) for ds,rows in journal.items()}}
    for name in ("R6", "BROAD", "Sassari"):
        details = final["datasets"][name]
        rows = details["sequences"]
        assert len(rows) == len(expected[name]) == len({r["name"] for r in rows})
        assert set(expected[name]) == {r["name"] for r in rows}
        for r in rows:
            prior = expected[name][r["name"]]
            assert r["frames"] == prior["scoring_frames"]
            assert all(math.isfinite(r["rmse_deg"][m]) and 0 <= r["rmse_deg"][m] <= 180 for m in methods)
        means = {m: fmean(r["rmse_deg"][m] for r in rows) for m in methods}
        frames = sum(r["frames"] for r in rows)
        for m in means:
            summary = details["summary"][m]
            assert math.isclose(summary["macro_rmse_deg"], means[m], abs_tol=1e-8)
            assert summary["scoring_frames"] == frames and summary["records"] == len(rows)
        prior_residuals = {m:max(abs(r["rmse_deg"][m]-expected[name][r["name"]]["rmse_deg"][m]) for r in rows)
                           for m in methods if m != "locked"}
        tests[name] = {"records": len(rows), "scoring_frames": frames, "macro_rmse_deg": means,
                       "vs": {m: paired_summary(rows, m) for m in ("full_vqf","physical_only","timesfm","gru")},
                       "maximum_absolute_difference_from_historical_per_record_deg": prior_residuals}
        if chosen["config"]["kind"] == "bias_physical_head":
            assert all(0 <= r["online_calibration"]["active_score_frames"] <= r["frames"] for r in rows)
            tests[name]["calibration"] = {"active_frames":sum(r["online_calibration"]["active_score_frames"] for r in rows),
                                         "active_records":sum(r["online_calibration"]["active_score_frames"]>0 for r in rows)}
        if name in {"BROAD","Sassari"}:
            from verify_external import motion_groups
            groups = motion_groups(name,rows,methods)
            tests[name]["motion_groups"] = groups
            if name == "Sassari":
                tests[name]["motion_means_deg"] = {g:item["mean_deg"] for g,item in groups.items()}
    for row in final["datasets"]["R6"]["sequences"]:
        name = row["name"]
        with np.load(BATCH.parent / "tilt-correction/prepared" / f"{name}.npz") as a, np.load(BATCH.parent / "gru-tilt/data" / f"{name}.npz") as b:
            assert np.array_equal(a["score"], b["score"]) and int(np.count_nonzero(b["score"])) == expected["R6"][name]["scoring_frames"]
            cached_indices = np.flatnonzero(a["score"]).astype("<i8",copy=False)
            assert hashlib.sha256(cached_indices.tobytes()).hexdigest() == row["score_frame_indices_sha256"], name
    check_latency(final,{n:r["scoring_frames"] for n,r in expected["R6"].items()})
    bias_timing = None
    if chosen["config"]["kind"] == "bias_physical_head":
        from inspect_accelerometers import benchmark_scope

        measured = final["latency"]["locked"]
        bias_timing = benchmark_scope(raw_shice_samples(measured["record"]), measured, chosen["config"],
                                      lock["frozen_bias_protocol"]["base_protocol"])
    with np.load(BATCH / lock["weights"], allow_pickle=False) as archive:
        weights = {key:archive[key] for key in archive.files}
    from verify_external import replay_suites
    external_replays = replay_suites(final["datasets"],chosen["config"],weights)
    r6_replays = replay_local_records(final["datasets"]["R6"]["sequences"], chosen["config"], weights)
    assert set(r6_replays) == set(expected["R6"])
    r6_native = {name:{k:row[k] for k in ("frames","score_frame_indices_sha256")} for name,row in r6_replays.items()}
    quaternion = final.get("locked_quaternion_verification")
    exported = None
    if quaternion:
        assert quaternion["selection_lock_sha256"] == lock_id
        if quaternion.get("result"):
            record = quaternion["result"]
            official_q = next(r for r in final["datasets"]["R6"]["sequences"] if r["name"] == record["name"])
            assert quaternion["protocol"] == read(BATCH / "quaternion-output-stage.json")["protocol"]
            check_quaternion_record(record,quaternion["protocol"],official_q["frames"],official_q["rmse_deg"]["locked"])
            assert (quaternion["status"] == "passed") == record["passed"]
            exported = audit_exported_quaternions(record)
    cpu = final.get("locked_cpu_verification")
    if cpu and cpu.get("status") != "failed":
        original = {r["name"]:r for r in final["datasets"]["R6"]["sequences"]}
        cpu_rows = cpu["R6_records"]
        assert len(cpu_rows) == len(original) and {r["name"] for r in cpu_rows} == set(original)
        for row in cpu_rows:
            match = original[row["name"]]
            assert row["frames"] == match["frames"] and math.isclose(row["cuda_rmse_deg"],match["rmse_deg"]["locked"],abs_tol=1e-8)
            delta = abs(row["numpy_cpu_rmse_deg"] - row["cuda_rmse_deg"])
            assert math.isclose(row["absolute_delta_deg"],delta,abs_tol=1e-8)
        maximum = max(r["absolute_delta_deg"] for r in cpu_rows)
        assert math.isclose(maximum,cpu["largest_recordwise_tilt_rmse_delta_deg"],abs_tol=1e-8)
        expected_status = "within_recordwise_tolerance" if maximum <= cpu["tolerance_deg"] else "exceeds_tolerance"
        assert cpu["status"] == expected_status
    extra = None
    path = BATCH / "tum-vi-independent.json"
    if path.is_file():
        from verify_tum import coverage_evidence, replay_rooms
        ext = read(path)
        protocol = read(BATCH / "tum-vi-protocol.json")
        assert ext["selection_lock_sha256"] == lock_id and ext["protocol"] == protocol
        assert datetime.fromisoformat(ext["at"]) >= datetime.fromisoformat(final["core_completed_at"])
        rows = ext["results"]
        failures = ext["failures"]
        covered = coverage_evidence(ext,protocol)
        outcome = final.get("optional_tum_vi")
        if isinstance(outcome,dict) and outcome.get("status") == "room_manifest_written":
            assert ext["phase"] == "final" and covered["counts"] == outcome["coverage"]
        frame_count = sum(r["frames"] for r in rows)
        extra = {"records":len(rows), "scoring_frames":frame_count, "failures":failures,"coverage":covered["counts"],
                 "unreported_rooms":sorted(n for n,s in covered["states"].items() if s == "pending"),
                 "locked":ext["locked"], "full_vqf":ext["pure_full_vqf"],
                 "per_room":[{k:r[k] for k in ("name","frames","imu_frames","rmse_deg","baseline_deg")} for r in rows],
                 "raw_replays":replay_rooms(rows,protocol,chosen["config"],weights)}
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(), "selected_id":chosen["round_id"],
            "selected_config":chosen["config"], "weights_sha256":lock["weights_sha256"], "locked_at":lock["locked_at"],
            "selection_id":selection_step["round_id"], "final_id":final_step["round_id"], "lock_sha256":lock_id,
            "selection_result_sha256":digest(BATCH / selection_step["result"]),
            "final_result_sha256":digest(BATCH / final_step["result"]),
            "baseline_source_sha256":baseline_sources,
            "r5_rmse_deg":chosen["r5"]["macro_rmse_deg"], "failed_shortlist_candidates":selection["failed_candidates"],
            "r5_leave_one_record_out":sensitivity,
            "shortlist_replay":shortlist_replay, "raw_shice_score_indices":{"R5":r5_native,"R6":r6_native},
            "tests":tests, "per_record_checkpoint":record_receipt, "latency":final["latency"], "bias_latency_scope":bias_timing, "delay_replay":final.get("locked_delay_replay"),
            "raw_locked_replay":r6_replays["R6-1"], "raw_locked_r6_by_record":r6_replays,
            "raw_external_replays":external_replays,
            "locked_quaternion_verification":quaternion, "raw_quaternion_export_audit":exported,
            "development_quaternion_exports":audit_development_quaternions(index.values()),
            "locked_cpu_verification":cpu,
            "optional_tum_vi":extra, "optional_tum_outcome":final.get("optional_tum_vi"),
            "test_started_at":final["started_at"], "test_core_completed_at":final["core_completed_at"]}


if __name__ == "__main__":
    audit = audit_final()
    write(HERE / "final_audit.json", audit)
    for label, result in audit["tests"].items():
        print(label, result["records"], result["scoring_frames"], result["macro_rmse_deg"])
