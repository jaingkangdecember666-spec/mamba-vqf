"""Lock the development shortlist on R5, then verify all held-out comparisons."""

import os
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from time import perf_counter_ns

import numpy as np

import install_rotation as installation
import learning
from common import BATCH, DATA, END, HISTORY, PILOT, SHICE, Z, aggregate, body_up, digest, fingerprint, names, now, read_json, record, rmse, rotate, stamp, unit, vqf_outputs, write_json
from features import VERSION, cache_record, fetch_record, physics
from final_record_progress import FinalRecordProgress
from selection_rules import build_shortlist, dev_value, eligible


def rows_now():
    return [read_json(path) for path in sorted((BATCH / "rounds").glob("*/round.json"))]


def historical_score_indices(name):
    with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as data:
        return np.flatnonzero(data["score"])


def verified_indices_digest(name, expected, actual):
    if not np.array_equal(expected, actual):
        raise ValueError(f"Score frame indices mismatch in {name}")
    return sha256(np.asarray(expected, dtype="<i8").tobytes()).hexdigest()


def fit_candidate(config, training):
    if config["kind"] == "bias_physical_head":
        from calibrated_head import fit_candidate as fit_bias_candidate
        import torch

        approved = read_json(BATCH / "accelerometer-bias-validation-protocol.json")
        if any(digest(BATCH/name) != sha for name,sha in approved["implementation_sha256"].items()):
            raise ValueError("Calibration implementation changed after development validation")
        return fit_bias_candidate(config, training, torch.device("cuda"))
    if config["kind"] == "stream_gru":
        import stream_gru
        import torch

        raw = [cached_candidate(name,config) for name in training]
        mount = learning.installation_fit(raw,config)
        prepared = [stream_gru.prepare(learning.apply_install(part,mount,include_reference=False),config,for_training=True) for part in raw]
        state,_ = stream_gru.fit(prepared,config,torch.device("cuda"))
        return {**state,"installation_quaternion_wxyz":mount}
    if config["kind"] == "ensemble":
        from ensemble import fit

        return fit(config, training)
    if config["kind"] == "legacy_head":
        import legacy_head
        import torch

        raw = [cached_candidate(name, config) for name in training]
        q = learning.installation_fit(raw, config) if config.get("installation") else None
        prepared = [legacy_head.prepare(learning.apply_install(data, q, include_reference=False) if q is not None else data,
                                        config, retain_augmentation=True) for data in raw]
        state, _ = legacy_head.fit(prepared, config, torch.device("cuda"), training)
        return {**state, "installation_quaternion_wxyz": q} if q is not None else state
    if config["kind"] == "tcn":
        import temporal
        import torch

        raw = [cached_candidate(name, config) for name in training]
        q = learning.installation_fit(raw, config) if config.get("installation") else None
        prepared = [temporal.prepare(learning.apply_install(data, q, include_reference=False) if q is not None else data, config)
                    for data in raw]
        state = temporal.fit(prepared, config, torch.device("cuda"))[0]
        return {**state, "installation_quaternion_wxyz": q} if q is not None else state
    if config["kind"] == "equivariant":
        import equivariant

        records = [cached_candidate(name, config) for name in training]
        q = learning.installation_fit(records, config) if config.get("installation") else None
        prepared = [equivariant.prepare(learning.apply_install(data, q, include_reference=False) if q is not None else data, config)
                    for data in records]
        state = equivariant.fit(prepared, config)
        return {**state, "installation_quaternion_wxyz": q} if q is not None else state
    if config["kind"] == "ridge":
        if config.get("installation"):
            raw = [fetch_record(name, config["parameters"]) for name in training]
            q = learning.installation_fit(raw, config)
            designs = [learning.prepare(learning.apply_install(data, q, include_reference=False), config,
                                         learning.projected(name, config.get("embedding", 0), config.get("seed")))
                       for name, data in zip(training, raw, strict=True)]
            return {**learning.fit(designs, config), "installation_quaternion_wxyz": q}
        return learning.fit([learning.for_record(name, config) for name in training], config)
    if config["kind"] == "install_rotation":
        stats = [installation.sufficient_statistics(fetch_record(name, config["parameters"]), config) for name in training]
        return installation.fit_statistics(stats, config)
    return {}


def score_candidate(data, config, state, embedding=None, delay_ms=0.):
    if config["kind"] == "bias_physical_head":
        from calibrated_head import score
        import torch

        return score(data, config, state, torch.device("cuda"), delay_ms)
    if config["kind"] == "ensemble":
        from ensemble import score
        import torch

        return score(data, config, state, torch.device("cuda"), delay_ms)
    if config["kind"] == "hybrid":
        from hybrid_filter import score

        return score(data, config, delay_ms)[0]
    if config.get("installation"):
        data = learning.apply_install(data, state["installation_quaternion_wxyz"])
    if config["kind"] == "stream_gru":
        import stream_gru
        import torch

        device = torch.device("cuda")
        return stream_gru.score(stream_gru.prepare(data,config),config,stream_gru.make_model(state,config,device),
                                device,delay_ms=delay_ms)
    if config["kind"] == "legacy_head":
        import legacy_head
        import torch

        device = torch.device("cuda")
        return legacy_head.evaluate(legacy_head.prepare(data, config), config,
                                    legacy_head.make_model(state, device), device, delay_ms)
    if config["kind"] == "tcn":
        import temporal
        import torch

        device = torch.device("cuda")
        return temporal.evaluate(temporal.prepare(data, config), config, temporal.make_model(state, config, device), device, delay_ms)
    if config["kind"] == "equivariant":
        import equivariant

        return equivariant.evaluate(data, state, config, delay_ms)
    if config["kind"] == "ridge":
        return learning.evaluate(learning.prepare(data, config, embedding), state, config, delay_ms)
    if config["kind"] == "install_rotation":
        return installation.evaluate(data, state, config)
    return rmse(np.broadcast_to(Z, data["ref_world"].shape), data["ref_world"])


def cached_candidate(name, config):
    if config["kind"] == "bias_physical_head":
        from calibrated_head import cached

        return cached(name, config)
    if config["kind"] == "hybrid":
        from hybrid_filter import cached

        return cached(name, config)
    if config.get("installation") or config["kind"] == "ensemble":
        return learning.attach_native(name, config)
    return fetch_record(name, config.get("parameters", {}))


def cache_external_candidate(imu, dt, score, context, truth, config):
    if config["kind"] == "bias_physical_head":
        from calibrated_head import bundle

        return bundle(imu, dt, score, context, truth, config, False, packet_ok=context)
    current = cache_record(imu, dt, score, context, truth, config.get("parameters", {}), False,
                           keep_scored_quaternion=bool(config.get("installation")) or config["kind"] == "ensemble")
    if config["kind"] != "hybrid":
        return current
    from hybrid_filter import pair

    default = cache_record(imu, dt, score, context, truth, {}, False)
    return pair(default, current)


def repetitions(items):
    groups = {}
    for item in items:
        if not eligible(item) or item["config"]["kind"] not in {"ridge", "tcn", "legacy_head"}:
            continue
        config = item["config"]
        if config["kind"] == "ridge" and not (config["hidden"] or config["embedding"]):
            continue
        family = fingerprint({k: v for k, v in config.items() if k != "seed"})
        groups.setdefault(family, []).append({"round_id": item["round_id"], "seed": item["seed"],
                                             "dev_rmse_deg": dev_value(item)})
    result = {}
    for key, rows in groups.items():
        by_seed = {}
        for row in rows:
            by_seed.setdefault(row["seed"], row)
        if len(by_seed) < 2:
            continue
        measured = sorted(by_seed.values(), key=lambda row: (row["dev_rmse_deg"], row["round_id"]))
        values = [row["dev_rmse_deg"] for row in measured]
        result[key] = {"repeats": measured, "seeds": len(measured), "mean_deg": float(np.mean(values)),
                       "std_deg": float(np.std(values, ddof=1)), "range_deg": float(np.ptp(values)),
                       "min_deg": values[0], "max_deg": values[-1],
                       "median_deg": float(np.median(values)), "best_single_seed": measured[0]}
    return result


def select(job, result_file):
    index = {item["round_id"]: item for item in rows_now()}
    requested = job["config"]["candidate_rounds"]
    # The live supervisor may have imported the previous planner before the audit correction.
    # The new isolated worker canonicalizes every shortlist before reading any R5 samples.
    shortlist = build_shortlist(index.values())
    ready = [index[row["round_id"]] for row in shortlist]
    baseline = next(row for row in ready if row["config"]["kind"] == "baseline")
    write_json(result_file.parent / "shortlist.json", {"at": stamp(), "requested_by_supervisor": requested,
              "selected_before_r5": shortlist, "shortlist_sha256": fingerprint(shortlist)})
    r5_reference = read_json(BATCH / baseline["result"])["r5"]
    r5_baseline = {row["name"]: row["rmse_deg"] for row in r5_reference["sequences"]}
    dev_baseline = dev_value(baseline)
    training, validation = names("dev"), names("r5")
    incumbent = incumbent_r5(validation, r5_baseline)
    reference_indices = {name: historical_score_indices(name) for name in validation}
    candidates, fits, failures = [], {}, []
    progress = result_file.parent / "selection-progress.json"
    for row in ready:
        config = row["config"]
        identity = row["round_id"]
        try:
            state = fit_candidate(config, training)
            scores = []
            for name in validation:
                data = cached_candidate(name, config)
                score_digest = verified_indices_digest(name, reference_indices[name], data["index"])
                embedding = learning.projected(name, config.get("embedding", 0), config.get("seed"))
                series_score = {"name": name, "frames": len(data["index"]), "baseline_deg": r5_baseline[name],
                                "rmse_deg": score_candidate(data, config, state, embedding),
                                "score_frame_indices_sha256": score_digest}
                if config["kind"] == "bias_physical_head":
                    series_score["online_calibration"] = data["online_scope"]
                scores.append(series_score)
            metrics = aggregate(scores)
            qualifies = identity == baseline["round_id"] or (dev_value(row) < dev_baseline and metrics["macro_rmse_deg"] < r5_reference["macro_rmse_deg"])
            candidates.append({"round_id": identity, "config": config, "config_sha256": row["config_sha256"],
                               "development_rmse_deg": dev_value(row), "r5": metrics, "r5_sequences": scores,
                               "eligible_for_selection": qualifies})
            fits[identity] = state
            print(identity, config["kind"], metrics, flush=True)
        except Exception as error:
            failures.append({"round_id": identity, "error": repr(error)})
            print(identity, "selection failed", repr(error), flush=True)
        write_json(progress, {"at": stamp(), "shortlist_sha256": fingerprint(shortlist), "r5_candidates": candidates,
                              "failed_candidates": failures, "r5_fixed_historical_incumbent": incumbent})
    by_round = {row["round_id"]: row for row in candidates}
    for row in candidates:
        config = row["config"]
        if config["kind"] == "bias_physical_head":
            count = sum(entry["online_calibration"]["active_score_frames"] for entry in row["r5_sequences"])
            reference = by_round.get(config["paired_zero_round"])
            beats_pair = bool(reference and row["r5"]["macro_rmse_deg"] < reference["r5"]["macro_rmse_deg"])
            row["bias_r5_requirement"] = {"active_score_frames":count,"paired_zero_round":config["paired_zero_round"],
                                          "beats_same_seed_zero_head_on_r5":beats_pair}
            row["eligible_for_selection"] = row["eligible_for_selection"] and count > 0 and beats_pair
    finalists = [row for row in candidates if row["eligible_for_selection"]]
    if not finalists:
        raise ValueError("No candidate or pure baseline was scored on R5")
    winner = min(finalists, key=lambda row: (row["r5"]["macro_rmse_deg"], row["development_rmse_deg"], row["r5"]["worst_rmse_deg"]))
    identifier = winner["round_id"]
    output = result_file.parent / "locked-weights.npz"
    staged = result_file.parent / f"locked-weights.{os.getpid()}.tmp"
    with staged.open("wb") as stream:
        np.savez(stream, **fits[identifier])
    os.replace(staged, output)
    lock = {"locked_at": stamp(), "training": training, "validation": validation,
            "selected": winner, "weights": str(output.relative_to(BATCH)), "weights_sha256": digest(output),
            "shortlist_sha256": fingerprint(shortlist),
            "rule": "Beat pure FullVQF on grouped development and R5. Bias compensation also needs active R5 scored frames and improvement over its same-seed zero-compensation head. Rank by R5 macro RMSE, development RMSE, then worst R5 record. Pure FullVQF is the fallback.",
            "held_out": {"R6": names("r6"), "BROAD": 39, "Sassari": {"channels": 18, "motions": 3},
                         "historically_observed": True, "used_for_this_selection": False}}
    if winner["config"]["kind"] == "bias_physical_head":
        from bias_calibration import PROTOCOL as bias_protocol
        from calibrated_head import verify_protocol

        verify_protocol(winner["config"])
        calibration_manifest = BATCH / "accelerometer-bias-validation-protocol.json"
        approved = read_json(calibration_manifest)
        if fingerprint(approved["protocol"]) != fingerprint(bias_protocol):
            raise ValueError("Pre-R5 calibration manifest does not match the active protocol")
        lock["frozen_bias_protocol"] = {"base_protocol":approved["protocol"],
                                        "online_minimum_poses":winner["config"]["minimum_online_poses"],
                                        "matched_experiment_sha256":winner["config"]["matched_experiment_sha256"],
                                        "implementation_sha256":approved["implementation_sha256"],
                                        "development_validation_manifest_sha256":digest(calibration_manifest),
                                        "inference":"From original packet validity and past static observations, with independent state per record."}
    lock_hash = fingerprint(lock)
    write_json(BATCH / "selection-lock.json", {**lock, "lock_sha256": lock_hash})
    write_json(result_file, {**lock, "lock_sha256": lock_hash, "r5_candidates": candidates, "failed_candidates": failures,
                             "r5_fixed_historical_incumbent": incumbent,
                             "development_seed_repeats": repetitions(index.values())})


def incumbent_r5(validation, expected):
    try:
        return recompute_incumbent_r5(validation, expected)
    except Exception as error:
        print("R5 incumbent recomputation failed", repr(error), flush=True)
        try:
            archive = read_json(HISTORY / "validation.json")
            if archive["validation_records"] != validation:
                raise ValueError("Historical R5 recording list differs from the frozen split")
            gain = archive["gains"]["physical_only"]
            method = f"physical_only@{gain:g}"
            rows = []
            for old in archive["r5_records"]:
                name = old["recording"]
                if abs(old["rmse_deg"]["full_vqf"] - expected[name]) > 1e-5:
                    raise ValueError(f"Archived R5 FullVQF score does not match {name}")
                base = record(name)
                if old["scoring_frames"] != int(base["score"].sum()):
                    raise ValueError(f"Archived R5 scoring frame count does not match {name}")
                rows.append({"name": name, "frames": old["scoring_frames"], "rmse_deg": old["rmse_deg"][method],
                             "baseline_deg": expected[name]})
            if sorted(row["name"] for row in rows) != sorted(validation):
                raise ValueError("Historical R5 records differ from the frozen split")
            metrics = aggregate(rows)
            if abs(metrics["macro_rmse_deg"] - archive["r5_macro_rmse_deg"][method]) > 1e-9:
                raise ValueError("Historical R5 macro differs from its archived per-record scores")
            weights = HISTORY / "physical_only_head.pt"
            return {"r5": metrics, "sequences": rows, "status": "historical_reuse_after_recomputation_error",
                    "error": repr(error), "gain": gain, "weights_sha256": digest(weights) if weights.is_file() else None,
                    "selected_epoch_in_previous_experiment": archive["best_epochs"]["physical_only"],
                    "source": str(HISTORY / "validation.json")}
        except Exception as archive_error:
            return {"status": "unavailable", "error": repr(error), "archive_check_error": repr(archive_error)}


def recompute_incumbent_r5(validation, expected):
    import torch
    from tilt_eval import full_record_scores, load_head, record_predictions

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    weights = HISTORY / "physical_only_head.pt"
    head = load_head(weights, device)
    archive = read_json(HISTORY / "validation.json")
    gain = archive["gains"]["physical_only"]
    known = archive["r5_macro_rmse_deg"][f"physical_only@{gain:g}"]
    rows = []
    for name in validation:
        scores = full_record_scores(HISTORY, name, record_predictions(HISTORY, name, {"physical_only": head}, device),
                                    {"physical_only": ("physical_only", gain, False)})
        if abs(scores["rmse_deg"]["full_vqf"] - expected[name]) > 1e-5:
            raise ValueError(f"Incumbent R5 score masks differ for {name}")
        rows.append({"name": name, "frames": scores["scoring_frames"], "rmse_deg": scores["rmse_deg"]["physical_only"],
                     "baseline_deg": expected[name]})
    stats = aggregate(rows)
    return {"r5": stats, "sequences": rows, "status": "recomputed_fixed_reference_not_refit", "gain": gain,
            "selected_epoch_in_previous_experiment": archive["best_epochs"]["physical_only"],
            "weights_sha256": digest(weights), "historical_r5_deg": known,
            "absolute_difference_from_history_deg": abs(stats["macro_rmse_deg"] - known)}


def embed_external(data, config, backbone, device):
    count = config.get("embedding", 0)
    if not count:
        return None
    from tilt_model import contexts, encode
    from features import block_indices

    at = block_indices(len(data["windows"]), config["update_hz"])
    history = contexts(data["windows"], at)
    blocks = [encode(backbone, history[idx:idx + 256], device).cpu().numpy()
              for idx in range(0, len(history), 256)]
    embedding = np.concatenate(blocks, axis=0)
    return (embedding.astype(np.float32) @ learning.projection(embedding.shape[-1], count, config["seed"])).reshape(len(embedding), -1)


def reference_sources():
    return {"R6": HISTORY / "test.json", "external": HISTORY / "external_test.json", "gru": PILOT / "gru-tilt" / "test.json",
            "timesfm_weights": HISTORY / "timesfm_head.pt", "light_weights": HISTORY / "physical_only_head.pt",
            "gru_weights": PILOT / "gru-tilt" / "longer" / "gru.pt"}


def accuracy_check(test, records, expected):
    comparison = {}
    for method in ("full_vqf", "timesfm", "physical_only", "gru"):
        gaps = {row["name"]: abs(row["rmse_deg"][method] - expected[row["name"]][method]) for row in records}
        worst = max(gaps, key=gaps.get)
        comparison[method] = {"max_abs_historical_delta_deg": gaps[worst], "record": worst,
                              "validation": "recomputed on matching historical score frames"}
    print(test, "historical verification", comparison, flush=True)
    return comparison


def r6_accuracy(config, state, heads, gains, gru, device, record_progress):
    from fit_gru_tilt import read_shice, validate
    from gru_data import OUTPUT as GRU_DIR
    from tilt_eval import full_record_scores, record_predictions

    historical = reference_sources()
    old = read_json(historical["R6"])
    old_gru = read_json(historical["gru"])["tests"]["R6"]["per_sequence"]
    expected = {r["recording"]: r["rmse_deg"] for r in old["r6_records"]}
    expected_frames = {r["recording"]: r["scoring_frames"] for r in old["r6_records"]}
    for row in old_gru:
        expected[row["recording"]]["gru"] = row["rmse_deg"]["gru"]
    records = []
    selection_data = {}
    methods = {k: (k, gains[k], False) for k in heads}
    sequence_names = names("r6")
    gru_inputs = read_shice(GRU_DIR, sequence_names)
    gru_scores = validate(gru, gru_inputs, device)
    for name, g, (_, _, gru_indices, _) in zip(sequence_names, gru_scores, gru_inputs, strict=True):
        previous = full_record_scores(HISTORY, name, record_predictions(HISTORY, name, heads, device), methods)
        data = cached_candidate(name, config)
        indices = historical_score_indices(name)
        score_digest = verified_indices_digest(name, indices, data["index"])
        verified_indices_digest(f"{name} (GRU)", indices, gru_indices)
        if previous["scoring_frames"] != expected_frames[name] or len(data["index"]) != expected_frames[name]:
            raise ValueError(f"Score frame count mismatch in R6 record {name}")
        embedding = learning.projected(name, config.get("embedding", 0), config.get("seed"))
        scores = {**previous["rmse_deg"], "gru": g, "locked": score_candidate(data, config, state, embedding)}
        row = {"name": name, "frames": previous["scoring_frames"], "rmse_deg": scores,
               "score_frame_indices_sha256": score_digest}
        if config["kind"] == "bias_physical_head":
            row["online_calibration"] = data["online_scope"]
        record_progress.add("R6", row)
        records.append(row)
        selection_data[name] = (data, embedding)
    return records, accuracy_check("R6", records, expected), selection_data


def external_accuracy(name, config, state, heads, gains, gru, backbone, device, record_progress):
    from evaluate_external_tilt import broad_records, reference_direction, sassari_records, score_record
    from gru_tilt_model import infer_sequences, with_dt
    from tilt_data import causal_fill

    historical = reference_sources()
    old = read_json(historical["external"])["datasets"][name]["records"]
    old_gru = read_json(historical["gru"])["tests"][name]["per_sequence"]
    expected = {r["recording"]: r["protocols"]["main"]["rmse_deg"] for r in old}
    expected_frames = {r["recording"]: r["protocols"]["main"]["scoring_frames"] for r in old}
    for row in old_gru:
        expected[row["recording"]]["gru"] = row["rmse_deg"]["gru"]
    source = broad_records if name == "BROAD" else sassari_records
    inputs = sorted(source(DATA), key=lambda row: len(row.eligible))
    outputs = []
    methods = {k: (k, gains[k], False) for k in heads}
    for start in range(0, len(inputs), 6):
        batch = inputs[start:start + 6]
        packet = [np.isfinite(row.imu).all(axis=1) for row in batch]
        imu = [causal_fill(row.imu, valid).astype(np.float32) for row, valid in zip(batch, packet, strict=True)]
        states = infer_sequences(gru, [with_dt(x, 1. / r.frequency) for x, r in zip(imu, batch, strict=True)], device)
        for item, clean, valid, gru_up in zip(batch, imu, packet, states, strict=True):
            # The same existing benchmark entry point is used for published local comparators.
            previous = score_record(item, backbone, heads, methods, device, 256, False)["protocols"]["main"]
            reference, label_ok = reference_direction(item.quaternion)
            mask = valid & label_ok & item.eligible
            if int(mask.sum()) != expected_frames[item.name] or int(mask.sum()) != previous["scoring_frames"]:
                raise ValueError(f"Score frame mask mismatch in {name} {item.name}")
            current = cache_external_candidate(clean, 1. / item.frequency, mask, valid, reference, config)
            score_digest = verified_indices_digest(f"{name} {item.name}", np.flatnonzero(mask), current["index"])
            latent = embed_external(current, config, backbone, device)
            scores = {**previous["rmse_deg"], "gru": rmse(gru_up[mask], reference[mask]),
                      "locked": score_candidate(current, config, state, latent)}
            scored = {"name": item.name, "frames": int(mask.sum()), "rmse_deg": scores, "group": item.group,
                      "score_frame_indices_sha256": score_digest}
            if config["kind"] == "bias_physical_head":
                scored["online_calibration"] = current["online_scope"]
            record_progress.add(name, scored)
            outputs.append(scored)
            print(name, item.name, scores, flush=True)
    return sorted(outputs, key=lambda row: row["name"]), accuracy_check(name, outputs, expected)


def percentile(values):
    p = np.percentile(values, (50, 95, 100)).tolist()
    return dict(zip(("p50_ms", "p95_ms", "max_ms"), p, strict=True))


def timed(action, count, warmup):
    records = []
    for index in range(count + warmup):
        start = perf_counter_ns()
        action(index)
        if index >= warmup:
            records.append((perf_counter_ns() - start) / 1e6)
    return percentile(records)


def bench_locked(config, state, backbone, device, benchmark_record="R6-3"):
    if config["kind"] == "bias_physical_head":
        from calibrated_head import benchmark

        return benchmark(config, state, device, benchmark_record)
    if config["kind"] == "stream_gru":
        from stream_gru import bench

        return bench(config,state,device,benchmark_record)
    if config["kind"] == "ensemble":
        from ensemble import bench

        return bench(config, state, device, benchmark_record)
    if config["kind"] == "hybrid":
        from hybrid_filter import bench

        return bench(config, benchmark_record)
    from vqf import VQF

    source = record(benchmark_record)
    reading, dt = source["imu"], float(source["dt"])
    acc = np.ascontiguousarray(reading[:, :3], dtype=np.float64)
    gyr = np.ascontiguousarray(reading[:, 3:], dtype=np.float64)
    index = 22000
    filter_ = VQF(dt, **config.get("parameters", {}))
    filter_.updateBatch(gyr[:index], acc[:index])

    def native(i):
        row = index + i
        filter_.update(gyr[row], acc[row])
        q = filter_.getQuat6D()
        if config.get("installation"):
            return learning.multiply_quaternion(q, state["installation_quaternion_wxyz"])
        if config["kind"] == "install_rotation" and config["coordinates"] == "body":
            return body_up(q[None, :])[0] @ state["fitted"]
        if config["kind"] == "install_rotation":
            return body_up(q[None, :], state["fitted"])[0]
        return q

    native_times = timed(native, 3000, 100)
    result = {"record": benchmark_record, "native_step": native_times, "nominal_native_hz": 1. / dt}
    if config["kind"] not in {"ridge", "tcn", "equivariant", "legacy_head"}:
        return result

    from tilt_model import encode

    cache = fetch_record(benchmark_record, config["parameters"])
    if config.get("installation"):
        cache = learning.apply_install(cache, state["installation_quaternion_wxyz"], include_reference=False)
    count = len(cache["windows"])
    phase = 100 // config["update_hz"]
    context = ((max(64, config["context"]) + phase - 1) // phase) * phase
    positions = np.linspace(context + 80, count - 1, 110, dtype=int)
    projected_count = config.get("embedding", 0)
    projection = learning.projection(1280, projected_count, config["seed"]) if projected_count else None
    if config["kind"] == "tcn":
        import temporal

        model, normal = temporal.make_model(state, config, device)
    if config["kind"] == "legacy_head":
        import legacy_head

        head = legacy_head.make_model(state, device)

    def update(i):
        stop = positions[i] + 1
        history = slice(stop - context, stop)
        local = {key: cache[key][history] for key in ("windows", "up", "q", "bias", "rest")}
        if config["kind"] == "equivariant":
            import equivariant

            at, basis, scalars = equivariant.features(local, config)
            residual = equivariant.delta(basis[-1:], scalars[-1:], state, config)
            return unit(Z + rotate(unit(local["q"][at[-1:]]), residual))
        if config["kind"] == "legacy_head":
            from tilt_model import physical_features
            from tilt_eval import predict_updates

            matrix = physical_features(local["windows"][-64:][None], local["up"][-1:],
                                       local["bias"][-1:], local["rest"][-1:])
            residual = predict_updates(head, matrix, None, device) * config["gain"]
            return unit(Z + rotate(unit(local["q"][-1:]), residual))
        at, matrix, acc_norm, gyr_rate = physics(local, config)
        last = at[-1:]
        matrix = matrix[-1:]
        if projection is not None:
            x = encode(backbone, local["windows"][-64:][None], device).cpu().numpy().astype(np.float32)
            matrix = np.concatenate((matrix, (x @ projection).reshape(1, -1)), axis=1)
        inputs = {"matrix": matrix, "up": local["up"][last], "acc": acc_norm[-1:], "gyr": gyr_rate[-1:]}
        if config["kind"] == "tcn":
            inputs["history"] = temporal.windows(local["windows"], last, config["context"])
            delta = temporal.inference(inputs, model, normal, config, device)
        else:
            delta = learning.delta_at_updates(inputs, state, config)
        return unit(Z + rotate(unit(local["q"][last]), delta))

    held = update(0)[0]
    result["correction_update"] = timed(update, 100, 10)
    frames = np.linspace(5000, len(reading) - 1, 3100, dtype=int)
    base = vqf_outputs(reading, dt, config["parameters"])["quat6D"]
    if config.get("installation"):
        base = learning.multiply_quaternion(base, state["installation_quaternion_wxyz"])
    result["apply_held_correction"] = timed(lambda i: body_up(base[frames[i]:frames[i] + 1], held), 3000, 100)
    due = cache["ready"][np.arange(phase - 1, count, phase)]
    result["correction_hz"] = float(1. / (np.median(np.diff(due)) * dt))
    return result


def latency(config, state, heads, gains, gru, backbone, device):
    import torch
    from evaluate_gru_tilt import measure_gru
    from evaluate_tilt import benchmark
    from gru_tilt_model import load_gru

    previous = benchmark(SHICE, HISTORY, backbone, heads, gains, device)
    test = record("R6-3")
    cpu_gru, _ = load_gru(reference_sources()["gru_weights"], torch.device("cpu"))
    gru_time = {"step_hz": 500., "CPU": measure_gru(cpu_gru, test["imu"], float(test["dt"]), 500., torch.device("cpu")),
                "CUDA": measure_gru(gru, test["imu"], float(test["dt"]), 500., device)}
    return {"machine": os.environ.get("COMPUTERNAME"), "accelerator": torch.cuda.get_device_name(),
            "prior_comparators": previous, "gru": gru_time, "locked": bench_locked(config, state, backbone, device)}


def verify_locked_cpu_on_r6(config, state, rows, device):
    import torch
    from cpu_readout import PhysicalReadout, latency as cpu_latency
    from legacy_head import evaluate, prepare

    cpu = torch.device("cpu")
    readout = PhysicalReadout(state)
    differences = []
    for row in rows:
        name = row["name"]
        data = cached_candidate(name, config)
        if config.get("installation"):
            data = learning.apply_install(data, state["installation_quaternion_wxyz"])
        result = prepare(data, config)
        if len(result["index"]) != row["frames"]:
            raise ValueError(f"CPU score mask in {name} does not match the locked model")
        value = evaluate(result, config, readout, cpu)
        differences.append({"name": name, "frames": row["frames"], "numpy_cpu_rmse_deg": value,
                            "cuda_rmse_deg": row["rmse_deg"]["locked"], "absolute_delta_deg": abs(value - row["rmse_deg"]["locked"])})
    largest = max(row["absolute_delta_deg"] for row in differences)
    return {"status": "within_recordwise_tolerance" if largest <= 1e-4 else "exceeds_tolerance",
            "largest_recordwise_tilt_rmse_delta_deg": largest, "tolerance_deg": 1e-4, "R6_records": differences,
            "latency": cpu_latency(state, config, device, cpu, "R6-3")}


def summarize_test(records):
    report = {}
    for key in records[0]["rmse_deg"]:
        rows = [{"name": row["name"], "frames": row["frames"], "rmse_deg": row["rmse_deg"][key],
                 "baseline_deg": row["rmse_deg"]["full_vqf"]} for row in records]
        report[key] = aggregate(rows)
    return report


def final_report(result):
    selected = result["lock"]["selected"]
    lines = ["# 夜间六轴 IMU 倾角优化", "", f"锁模轮 `{selected['round_id']}`，模型类型 `{selected['config']['kind']}`。",
             f"R5 倾角 RMSE **{selected['r5']['macro_rmse_deg']:.3f}°**，权重 SHA-256 `{result['lock']['weights_sha256']}`。",
             "", "| 数据集 | 序列 | 纯 FullVQF | 既有轻量 | 既有 TimesFM | 两层 GRU | 锁模候选 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for data, value in result["datasets"].items():
        kind = value["summary"]
        score = [f"{kind[key]['macro_rmse_deg']:.3f}°" for key in ("full_vqf", "physical_only", "timesfm", "gru", "locked")]
        lines.append(f"| {data} | {value['summary']['full_vqf']['records']} | " + " | ".join(score) + " |")
    lines += ["", "R6、BROAD、Sassari 为历史已看过的留出，只用于本批锁模后评分。Sassari 的 18 路来自 3 次运动。",
              "逐序列、每种方法的最大退化、原始延迟和复算差异见正式轮的 `result.json`。",
              "", "建议用 `experiment-index.json` 检查完成轮、配置及复跑，用 `ledger.jsonl` 核对阶段与尝试。"]
    bias_choices = result.get("r5_bias_candidates", [])
    if bias_choices:
        phrases = [f"{row['round_id']} 为 {row['r5_macro_rmse_deg']:.3f}°，校准启用评分帧 {row['active_score_frames']:,}"
                   for row in bias_choices]
        lines.extend(["", "R5 偏置候选中，" + "。".join(phrases) + "。"])
    if selected["config"]["kind"] == "bias_physical_head":
        coverage = []
        for name, data in result["datasets"].items():
            rows = data["sequences"]
            enabled = sum(row["online_calibration"]["active_score_frames"] > 0 for row in rows)
            count = sum(row["online_calibration"]["active_score_frames"] for row in rows)
            coverage.append(f"{name} {enabled}/{len(rows)} 条，{count:,} 帧")
        lines.extend(["", "锁模候选的补偿覆盖为 " + "，".join(coverage) + "。"])
    incumbent = result.get("r5_fixed_historical_incumbent") or {}
    if "r5" in incumbent:
        score = incumbent["r5"]["macro_rmse_deg"]
        origin = "复算" if incumbent["status"] == "recomputed_fixed_reference_not_refit" else "引用原运行，经基线分数和样本数核对"
        lines.extend(["", f"既有轻量头的 R5 误差为 **{score:.3f}°**，取得方式为{origin}。"])
    elif incumbent:
        lines.extend(["", "既有轻量头的 R5 复算和历史记录核对均失败，详见正式轮结果。"])
    if result.get("latency"):
        latency = result["latency"]
        previous, locked = latency["prior_comparators"], latency["locked"]
        lines.extend(["", f"在 {latency['machine']}（{latency['accelerator']}）上单条流计时。修正更新和逐帧操作分别列出。",
                      "", "| 操作 | 频率 | P50 ms | P95 ms |", "|---|---:|---:|---:|"])
        def add_latency(label, rate, values):
            p50 = values["p50_ms"] if "p50_ms" in values else values["p50"]
            p95 = values["p95_ms"] if "p95_ms" in values else values["p95"]
            lines.append(f"| {label} | {rate} Hz | {p50:.4f} | {p95:.4f} |")
        add_latency("纯 FullVQF 逐帧更新", round(locked["nominal_native_hz"]), previous["full_vqf_per_frame_ms"])
        add_latency("既有轻量头更新", 10, previous["correction_update_ms"]["physical_only"])
        add_latency("既有 TimesFM 头更新", 10, previous["correction_update_ms"]["timesfm"])
        add_latency("两层 GRU CPU 逐步更新", 500, latency["gru"]["CPU"])
        add_latency("两层 GRU CUDA 逐步更新", 500, latency["gru"]["CUDA"])
        add_latency("锁模候选逐帧更新", round(locked["nominal_native_hz"]), locked["native_step"])
        if "online_bias_block_update" in locked:
            add_latency("锁模候选零偏校准更新（包含在逐帧更新内）", round(locked["bias_block_hz"], 2), locked["online_bias_block_update"])
        if "gate_update" in locked:
            add_latency("锁模候选门控更新（包含在逐帧更新内）", round(locked["gate_hz"], 2), locked["gate_update"])
        if "correction_update" in locked:
            add_latency("锁模候选修正更新", round(locked["correction_hz"], 2), locked["correction_update"])
            if "cpu_correction_update" in locked:
                add_latency("锁模候选 CPU 修正更新", round(locked["correction_hz"], 2), locked["cpu_correction_update"])
            if "apply_held_correction" in locked:
                add_latency("锁模候选应用既有修正", round(locked["nominal_native_hz"]), locked["apply_held_correction"])
        if "locked_delay_replay" in result:
            delayed = result["locked_delay_replay"]
            lines.extend(["", f"R6 使用修正更新 P95 延迟 {delayed['delay_ms']:.3f} ms 回放后，"
                          f"锁模候选倾角误差为 {delayed['R6']['macro_rmse_deg']:.3f}°。"])
    if "locked_cpu_verification" in result:
        cpu = result["locked_cpu_verification"]
        if cpu["status"] == "within_recordwise_tolerance":
            difference = cpu["largest_recordwise_tilt_rmse_delta_deg"]
            p95 = cpu["latency"]["timings"]["numpy_cpu"]["correction_update"]["p95_ms"]
            lines.extend(["", f"同一锁模权重的 NumPy CPU 修正更新 P95 为 {p95:.3f} ms，"
                          f"R6 每条记录与 CUDA 的最大误差差值为 {difference:.5f}°。"])
        else:
            lines.extend(["", "CPU 补充验证未达到复现条件，状态和数值见正式轮结果。"])
    if "locked_quaternion_verification" in result:
        audit = result["locked_quaternion_verification"]
        if audit["status"] == "passed":
            row = audit["result"]
            peak = max(check["max_rotation_delta_deg"] for check in row["prefix_checks"])
            lines.extend(["", f"锁模权重在 {row['name']} 上已逐帧输出 wxyz 四元数。单位范数最大偏差 "
                          f"{row['max_unit_norm_error']:.2e}，截断回放最大姿态差 {peak:.5f}°，"
                          f"反算倾角 RMSE 差 {row['absolute_tilt_rmse_delta_deg']:.5f}°。"])
        else:
            lines.extend(["", "锁模四元数补充验证未通过，检查数值和错误见正式轮结果。"])
    tum_report = result.get("optional_tum_vi")
    if isinstance(tum_report, dict) and tum_report.get("status") == "room_manifest_written":
        coverage = tum_report["coverage"]
        lines.extend(["", f"TUM-VI 共 {coverage['planned']} 个房间，评分成功 {coverage['success']} 个，"
                      f"失败 {coverage['failed']} 个，截止前未启动 {coverage['not_started_before_cutoff']} 个。"
                      "逐房间状态见 `tum-vi-independent.json`。"])
    if "index_counts" in result:
        counts = result["index_counts"]
        lines.extend(["", f"本批完成 {counts['development_counts'].get('completed', 0)} 个开发轮，"
                      f"去除随机种子后 {counts['completed_configurations_ignoring_seed']} 个配置，"
                      f"种子复跑 {counts['completed_seed_repeats']} 轮，失败 {counts['development_counts'].get('failed', 0)} 轮。"
                      f"重试尝试次数 {counts['retry_attempts_not_new_configurations']} 单独计数。"])
    (BATCH / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(job, result_file):
    import torch
    from gru_tilt_model import load_gru
    from tilt_eval import load_head
    from tilt_model import load_backbone

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    lock = read_json(BATCH / "selection-lock.json")
    lock_id = job["config"]["selection_sha256"]
    if lock["lock_sha256"] != lock_id or fingerprint({k: v for k, v in lock.items() if k != "lock_sha256"}) != lock_id:
        raise ValueError("Selection lock changed before held-out scoring")
    weights = BATCH / lock["weights"]
    if digest(weights) != lock["weights_sha256"]:
        raise ValueError("Locked model weights changed before test")
    progress = result_file.parent / "final-progress.json"
    report = read_json(progress) if progress.is_file() else None
    if report is not None and report["lock"]["lock_sha256"] != lock_id:
        raise ValueError("Partial test results are tied to a different lock")
    records_progress = FinalRecordProgress(result_file.parent, lock_id, report["started_at"] if report is not None else stamp())
    if report is not None:
        if report["started_at"] != records_progress.started_at:
            raise ValueError("Formal per-record and dataset progress have different start times")
        for name, entry in report["datasets"].items():
            records_progress.reconcile(name, entry["sequences"])
    with np.load(weights, allow_pickle=False) as data:
        state = {name: data[name] for name in data.files}
    config = lock["selected"]["config"]
    if config["kind"] == "bias_physical_head":
        from calibrated_head import verify_protocol

        verify_protocol(config)
        frozen_bias = lock["frozen_bias_protocol"]
        if (fingerprint(frozen_bias["base_protocol"]) != config["protocol_sha256"]
                or frozen_bias["online_minimum_poses"] != config["minimum_online_poses"]
                or any(digest(BATCH/name) != sha for name,sha in frozen_bias["implementation_sha256"].items())):
            raise ValueError("Online bias protocol differs from the locked model")
    sources = reference_sources()
    heads = {"physical_only": load_head(sources["light_weights"], device),
             "timesfm": load_head(sources["timesfm_weights"], device)}
    gains = read_json(HISTORY / "validation.json")["gains"]
    gru, _ = load_gru(sources["gru_weights"], device)
    backbone = load_backbone(PILOT / "checkpoint", device)
    if report is None:
        report = {"lock": lock, "datasets": {},
                  "references": {key: {"path": str(path), "sha256": digest(path)} for key, path in sources.items()},
                  "started_at": records_progress.started_at}
    selection_result = read_json(BATCH / "rounds" / job["config"]["selection_round"] / "result.json")
    report["r5_fixed_historical_incumbent"] = selection_result.get("r5_fixed_historical_incumbent")
    report["r5_bias_candidates"] = [{"round_id":row["round_id"],"r5_macro_rmse_deg":row["r5"]["macro_rmse_deg"],
                                     "eligible_for_selection":row["eligible_for_selection"],**row["bias_r5_requirement"]}
                                    for row in selection_result["r5_candidates"] if "bias_r5_requirement" in row]
    data_r6 = None
    if "R6" not in report["datasets"]:
        rows, correspondence, data_r6 = r6_accuracy(config, state, heads, gains, gru, device, records_progress)
        records_progress.reconcile("R6", rows)
        report["datasets"]["R6"] = {"sequences": rows, "summary": summarize_test(rows), "reproduction": correspondence}
        write_json(progress, report)
    for name in ("BROAD", "Sassari"):
        if name not in report["datasets"]:
            rows, correspondence = external_accuracy(name, config, state, heads, gains, gru, backbone, device, records_progress)
            records_progress.reconcile(name, rows)
            report["datasets"][name] = {"sequences": rows, "summary": summarize_test(rows), "reproduction": correspondence}
            write_json(progress, report)
    if "latency" not in report:
        report["latency"] = latency(config, state, heads, gains, gru, backbone, device)
        if config["kind"] in {"ridge", "tcn", "equivariant", "hybrid", "legacy_head", "ensemble", "stream_gru", "bias_physical_head"}:
            update_key = "gate_update" if config["kind"] == "hybrid" else "correction_update"
            delay = report["latency"]["locked"][update_key]["p95_ms"]
            delayed = []
            for row in report["datasets"]["R6"]["sequences"]:
                name = row["name"]
                cached = data_r6[name] if data_r6 is not None else (
                    cached_candidate(name, config), learning.projected(name, config.get("embedding", 0), config["seed"]))
                data, embedding = cached
                delayed.append({"name": name, "frames": row["frames"], "baseline_deg": row["rmse_deg"]["full_vqf"],
                                "rmse_deg": score_candidate(data, config, state, embedding, delay)})
            report["locked_delay_replay"] = {"delay_ms": delay, "R6": aggregate(delayed), "sequences": delayed}
        write_json(progress, report)
    report["core_completed_at"] = stamp()
    write_json(result_file, report)
    final_report(report)

    if config["kind"] == "legacy_head" and "locked_cpu_verification" not in report:
        try:
            report["locked_cpu_verification"] = verify_locked_cpu_on_r6(config, state, report["datasets"]["R6"]["sequences"], device)
        except Exception as error:
            report["locked_cpu_verification"] = {"status": "failed", "error": repr(error)}
            print("CPU verification failed", repr(error), flush=True)
        write_json(progress, report)
        write_json(result_file, report)

    if "locked_quaternion_verification" not in report:
        try:
            from quaternion_audit import PROTOCOL as quaternion_protocol, verify as verify_quaternion

            probe = next(row for row in report["datasets"]["R6"]["sequences"] if row["name"] == "R6-3")
            expected = {"frames": probe["frames"], "rmse_deg": probe["rmse_deg"]["locked"]}
            audited = verify_quaternion(probe["name"], config, state, expected,
                                        result_file.parent / "R6-3-locked-q-wxyz.npz", device, backbone)
            report["locked_quaternion_verification"] = {"protocol": quaternion_protocol, "selection_lock_sha256": lock_id,
                                                        "status": "passed" if audited["passed"] else "failed", "result": audited}
            print("Locked quaternion verification", report["locked_quaternion_verification"]["status"], flush=True)
        except Exception as error:
            report["locked_quaternion_verification"] = {"status": "failed", "error": repr(error),
                                                        "selection_lock_sha256": lock_id}
            print("Locked quaternion verification failed", repr(error), flush=True)
        write_json(progress, report)
        write_json(result_file, report)

    from datetime import timedelta
    from tum_vi import run as run_tum

    protocol = read_json(BATCH / "tum-vi-protocol.json")
    remaining = (END - now()).total_seconds() / 60
    if remaining >= protocol["minimum_remaining_minutes"]:
        try:
            tum_manifest = run_tum(DATA / "TUM-VI", config, state, lock_id, backbone, device, END - timedelta(minutes=3))
            report["optional_tum_vi"] = {"status":"room_manifest_written","path":str(BATCH / "tum-vi-independent.json"),
                                         "coverage":tum_manifest["coverage"]}
        except Exception as error:
            report["optional_tum_vi"] = {"status": "failed_after_core_tests", "error": repr(error)}
            print("Optional TUM-VI evaluation failed", repr(error), flush=True)
    else:
        report["optional_tum_vi"] = {"status": "not_started_insufficient_reserved_window", "remaining_minutes": remaining}
    report["completed_at"] = stamp()
    write_json(result_file, report)
    from report_index import experiment_index

    snapshot = experiment_index(job["round_id"])
    report["index_counts"] = {key: value for key, value in snapshot.items() if key != "index"}
    write_json(result_file, report)
    experiment_index(job["round_id"])
    final_report(report)
