"""Replay frozen model forecasts on raw SHICE development windows using the public CPU interfaces."""

import sys
import math
from collections import Counter
from datetime import datetime, timezone

import numpy as np

from audit_rounds import BATCH, HERE, check_fold, check_scores, read, write
from verify_final import digest, raw_shice_samples


PROTOCOL = BATCH / "forecast-backbone-development-protocol.json"
PROTOCOL_SHA256 = "a4d9f1a4916d40f4f126beed28282c0003c22b8a93218598629f5a84e2525cf7"
RECORD = "R3-3"


def check_timing(values, published):
    values = np.asarray(values,dtype=np.float64)
    assert len(values) and np.isfinite(values).all() and min(values) > 0
    np.testing.assert_allclose(np.percentile(values,[50,95,100]),
                               [published[k] for k in ("p50_ms","p95_ms","max_ms")],atol=1e-8,rtol=0)


def check_encoder(row, result):
    path = BATCH / "timesfm-encoder-diagnostic-protocol.json"
    assert digest(path) == "e7ac414dbcee61c74e9e0c9ad0ea4f3340f4c09a8f6bf48bf1bd8ef25ef138f5"
    plan = read(path)
    cfg = row["config"]
    assert result["round_id"] == row["round_id"] and result["config"] == cfg
    assert all(plan[k] == v for k,v in cfg.items())
    assert datetime.fromisoformat(plan["at"]) <= datetime.fromisoformat(row["started_at"])
    assert not result["used_for_selection"] and not plan["eligible_for_model_selection"] and result["new_fits"] == 0
    files = {"head_sha256":"tilt-correction/timesfm_head.pt","checkpoint_config_sha256":"checkpoint/config.json",
             "upstream_model_sha256":"upstream/src/timesfm3/torch/model.py"}
    for key, relative in files.items():
        assert digest(BATCH.parent / relative) == cfg[key]
    expected = []
    assert cfg["records"] == ["R1-10","R2-10","R3-3","R4-2"]
    n = cfg["samples_per_record"] + cfg["warmup_per_record"]
    for name in cfg["records"]:
        with np.load(BATCH.parent/"tilt-correction/prepared"/(name+".npz"),allow_pickle=False) as src:
            releases = src["updates"]
            indices = releases[np.linspace(100,len(releases)-1,n,dtype=int)]
        expected.extend({"record":name,"native_index":int(idx),"timed":i>=cfg["warmup_per_record"]} for i,idx in enumerate(indices))
    assert expected == result["probe_indices"]
    assert result["embedding_checks"] == [{"record":name,"windows":n,"exact_in_batch":True} for name in cfg["records"]]
    assert set(result["outputs_exact"]) == {"residual_vector","correction_quaternion","output_quaternion"}
    for arm in ("full_forward","encoder_only"):
        assert len(result["samples_ms"][arm]) == sum(r["timed"] for r in expected)
        check_timing(result["samples_ms"][arm],result["latency_summary"][arm])
    return {"id":row["round_id"],"windows":len(expected),"timed_per_arm":sum(r["timed"] for r in expected),
            "ms":result["latency_summary"]}


def check_cache(record, model_key, stage):
    name = record["record"]
    assert name in stage["development_records"]
    path = BATCH / record["cache"]
    assert path == BATCH/"forecast-development-cache"/model_key/(name+".npz")
    assert digest(path) == record["sha256"]
    with np.load(path,allow_pickle=False) as cache, np.load(BATCH.parent/"tilt-correction/prepared"/(name+".npz"),allow_pickle=False) as native:
        np.testing.assert_array_equal(cache["indices"],native["updates"])
        assert str(cache["protocol"]) == PROTOCOL_SHA256 and str(cache["model"]) == model_key
        x = cache["features"]
        assert x.shape == (len(native["updates"]),18) and np.isfinite(x).all()
        assert record["windows"] == len(x) and record["new_windows"] in {0,len(x)}
    return {"name":name,"windows":record["windows"],"new_windows":record["new_windows"],"sha256":record["sha256"]}


def check_feature_job(row, result):
    cfg = row["config"]
    stage = read(PROTOCOL)
    engine_path = BATCH/"forecast-backbone-engineering.json"
    assert cfg["protocol_sha256"] == digest(PROTOCOL) == PROTOCOL_SHA256
    assert cfg["engine_sha256"] == digest(engine_path)
    assert datetime.fromisoformat(stage["registered_at"]) <= datetime.fromisoformat(row["started_at"])
    model_key = cfg["model"]
    assert result["model"] == model_key
    if model_key != "physical_zero":
        for key in ("repo","revision","weights_sha256"):
            assert result["weights"][key] == stage["models"][model_key][key]
    out = {"id":row["round_id"],"action":cfg["action"],"model":model_key}
    if cfg["action"] == "probe":
        assert result["timing_record"] == stage["release_record"] == RECORD
        assert result["host"].lower() == read(BATCH/"status.json")["host"].lower()
        n, warm = stage["latency_samples"], stage["latency_warmups"]
        assert len(result["samples_ms"]) == result["timed_windows"] == n and result["warmup_windows"] == warm
        check_timing(result["samples_ms"],result["composite_latency"])
        with np.load(BATCH.parent/"tilt-correction/prepared"/(RECORD+".npz"),allow_pickle=False) as src:
            indices = src["updates"]
        np.testing.assert_array_equal(indices[np.linspace(0,len(indices)-1,n+warm,dtype=int)[warm:]],result["native_indices_timed"])
        assert result["prefix_check"]["future_mutation_preserved_features_exactly"]
        error = np.asarray(result["batch_single_max_abs_delta_by_feature"])
        assert error.shape == (18,) and np.isfinite(error).all() and np.all(error>=0)
        limit = read(engine_path)["batch_single_parity_absolute_limit_sensor_units"]
        assert error.max() <= limit
        if model_key == "timesfm_3":
            assert result["prefix_check"]["official_predictor_max_abs_delta"] <= limit
        out["p95_ms"] = result["composite_latency"]["p95_ms"]
        out["samples"] = n
        caches = [result["first_full_cache"]] if result["first_full_cache"] else []
        assert len(caches) == int(model_key != "physical_zero")
    else:
        assert cfg["action"] == "extract"
        caches = result["records"]
        assert [r["record"] for r in caches] == cfg["records"]
        assert result["newly_extracted_windows"] == sum(r["new_windows"] for r in caches)
    out["caches"] = [check_cache(r,model_key,stage) for r in caches]
    return out


def check_paired_score(row, result, baseline):
    stage = read(PROTOCOL)
    cfg = row["config"]
    assert digest(PROTOCOL) == result["protocol_sha256"] == cfg["protocol_sha256"] == PROTOCOL_SHA256
    assert result["round_id"] == row["round_id"] and result["readout"] == stage["readout"]
    arms = result["models"]
    assert arms == cfg["arms"] and len(set(arms)) == len(arms)
    assert {"physical_zero","timesfm_3"} < set(arms) <= set(stage["models"])
    assert result["readout"]["hidden"] == 0 and result["readout"]["embedding"] == 0
    reference = {r["name"]:r for r in baseline}
    all_names = set(stage["development_records"])
    assert set(reference) == all_names
    names = set(result["records"])
    assert names <= all_names and names == all_names.difference(*result["missing_cache_records"].values())
    frames = {n:reference[n]["frames"] for n in names}
    eligible = {r["name"]:r["eligible_updates"] for r in read(HERE/"recurrent_context_audit.json")["records"]}
    for name, provenance in result["records"].items():
        assert provenance["scoring_frames"] == frames[name]
        assert set(provenance["feature_hashes"]) == set(arms)-{"physical_zero"}
        for arm, expected in provenance["feature_hashes"].items():
            path = BATCH/"forecast-development-cache"/arm/(name+".npz")
            assert digest(path) == expected
            with np.load(path,allow_pickle=False) as cache:
                assert len(cache["indices"]) == provenance["windows"]
    axes = ("device","motion","joint_holdout")
    folds = result["folds"]
    assert len({(f["axis"],f["fold"]) for f in folds}) == len(folds)
    mapped = {a:set() for a in axes}
    widths = set()
    for fold in folds:
        axis = fold["axis"]
        held = check_fold({**fold,"validation":fold["held"]},axis,names)
        assert fold["fit_count"] == len(arms) and fold["scored_rows"] == len(held)
        assert fold["same_training_masks_per_arm"] and set(fold["arms"]) == set(arms)
        assert fold["training_label_updates_per_arm"] == sum(eligible[n] for n in fold["training"])
        assert math.isclose(float(np.linalg.norm(fold["mount_quaternion_body_wxyz"])),1.,abs_tol=1e-6)
        for fit in fold["arms"].values():
            assert len(bytes.fromhex(fit["fit_sha256"])) == 32 and fit["fit_seconds"] > 0
            widths.add(fit["input_width"])
        mapped[axis].update((n,fold["fold"]) for n in held)
    assert len(widths) == 1 and next(iter(widths)) > 18
    assert result["deterministic_fit_count"] == sum(f["fit_count"] for f in folds)
    assert result["paired_record_count"] == sum(len(s) for s in mapped.values())
    measured = {}
    for axis in axes:
        rows = result["per_record_scores"][axis]
        assert mapped[axis] == {(r["name"],r["fold"]) for r in rows} and len(rows) == len(mapped[axis])
        assert result["coverage"][axis] == {"records":len(rows),"scoring_frames":sum(frames[r["name"]] for r in rows)}
        if not rows:
            continue
        for r in rows:
            assert math.isclose(r["full_vqf_rmse_deg"],reference[r["name"]]["rmse_deg"],abs_tol=1e-7)
            for arm in set(arms)-{"physical_zero"}:
                assert math.isclose(r["delta_vs_physical_deg"][arm],r["rmse_deg"][arm]-r["rmse_deg"]["physical_zero"],abs_tol=1e-8)
        measured[axis] = {}
        for arm in arms:
            metrics = result["metrics"][axis][arm]
            check_scores([{**r,"rmse_deg":r["rmse_deg"][arm]} for r in rows],metrics,frames)
            measured[axis][arm] = metrics["macro_rmse_deg"]
            gap = measured[axis][arm]-result["metrics"][axis]["physical_zero"]["macro_rmse_deg"]
            assert math.isclose(gap,metrics["delta_vs_physical_deg"],abs_tol=1e-8)
    complete = names == all_names and Counter(f["axis"] for f in folds) == {"device":4,"motion":4,"joint_holdout":16} and all(
        result["coverage"][a] == {"records":len(all_names),"scoring_frames":sum(frames.values())} for a in axes)
    assert complete == result["all_original_development_coverage"] and (complete or cfg["allow_partial"])
    checks = result["preliminary_candidate_checks"]
    assert set(checks) == set(cfg["probe_rounds"])
    delayed = result.get("forecast_p95_delay_replay")
    delayed_scores = {}
    for arm, check in checks.items():
        path = BATCH/"rounds"/cfg["probe_rounds"][arm]/"result.json"
        probe = read(path)
        assert check["probe_result_sha256"] == digest(path) and probe["model"] == arm
        latency = probe["composite_latency"]["p95_ms"]
        assert math.isclose(check["p95_ms"],latency,abs_tol=1e-8)
        if delayed:
            assert math.isclose(delayed["delay_ms_by_model"][arm],latency,abs_tol=1e-8)
            for axis,values in measured.items():
                seq = result["per_record_scores"][axis]
                scored = [{**r,"rmse_deg":r["p95_delay_rmse_deg"][arm]} for r in seq]
                m = delayed["metrics"][axis][arm]
                check_scores(scored,m,frames)
                assert math.isclose(m["delta_vs_physical_deg"],m["macro_rmse_deg"]-values["physical_zero"],abs_tol=1e-8)
                delayed_scores.setdefault(axis,{})[arm] = m["macro_rmse_deg"]
        delta = {a:measured[a][arm]-measured[a]["physical_zero"] if a in measured else None for a in axes}
        assert all((v is None and check["paired_delta_deg"][a] is None) or math.isclose(v,check["paired_delta_deg"][a],abs_tol=1e-8)
                   for a,v in delta.items())
        rules = {"full_original_coverage":complete,"p95_at_most_100ms":latency <= 100,
                 "joint_improvement_at_least_0_015_deg":delta["joint_holdout"] is not None and delta["joint_holdout"] <= -.015,
                 "device_holdout_no_regression":delta["device"] is not None and delta["device"] <= 1e-9,
                 "program_holdout_no_regression":delta["motion"] is not None and delta["motion"] <= 1e-9}
        assert check["conditions"] == rules and check["passed_preliminary_screen"] == all(rules.values())
    recovery = []
    summary_path = BATCH/"forecast-backbone-development-summary.json"
    if summary_path.is_file():
        summary = read(summary_path)
        if summary["paired_scores_round"] == row["round_id"]:
            assert summary["paired_scores_sha256"] == digest(BATCH/row["result"]) and summary["protocol_sha256"] == PROTOCOL_SHA256
            assert summary["coverage"] == result["coverage"] and summary["primary_rmse"] == result["metrics"]
            assert summary["preliminary_candidate_checks"] == checks and summary["p95_delay_replay"] == delayed
            extracted = 0
            for arm, item in summary["extraction"].items():
                cache = item["per_record"]
                assert item["records"] == len(cache) and item["windows"] == sum(v["windows"] for v in cache.values())
                for name in names:
                    assert cache[name]["sha256"] == result["records"][name]["feature_hashes"][arm]
                extracted += item["windows"]
            assert set(summary["extraction"]) == set(arms)-{"physical_zero"}
            assert summary["counts"] == {"real_frozen_checkpoints":len(arms)-1,"unique_extractions":extracted,
                "deterministic_fits":result["deterministic_fit_count"],"paired_record_scores":result["paired_record_count"],
                "individual_model_record_scores":result["paired_record_count"]*len(arms),"r5_admissions":len(result["admitted_for_r5"])}
            for failed in summary["failed_adapter_rounds"]:
                attempt = read(BATCH/"rounds"/failed/"round.json")
                assert attempt["status"] == "failed" and attempt["return_code"] and attempt["config"]["kind"] == cfg["kind"]
                lines = (BATCH/attempt["stdout"]).read_text(encoding="utf-8-sig",errors="replace").strip().splitlines()
                recovery.append({"id":failed,"last_error":lines[-1]})
    return {"id":row["round_id"],"fits":result["deterministic_fit_count"],"paired_rows":result["paired_record_count"],
            "arms":arms,"coverage":result["coverage"],"scores":measured,"delayed_scores":delayed_scores,
            "preliminary_pass":[a for a,c in checks.items() if c["passed_preliminary_screen"]],
            "admitted":result["admitted_for_r5"],"failed_probes":recovery}


def replay_probe(model_key):
    import torch

    plan = read(PROTOCOL)
    assert digest(PROTOCOL) == PROTOCOL_SHA256 and plan["release_record"] == RECORD
    assert plan["history_blocks"] == 64 and plan["horizon_steps"] == [1,4,8]
    assert RECORD in plan["development_records"]
    file = BATCH / "forecast-development-cache" / model_key / (RECORD+".npz")
    with np.load(file,allow_pickle=False) as cache:
        saved = cache["features"]
        indices = cache["indices"]
        assert str(cache["protocol"]) == PROTOCOL_SHA256 and str(cache["model"]) == model_key
    raw = raw_shice_samples(RECORD)
    original = raw["imu"]
    # The 1000 Hz SHICE frontend emits a 100 Hz block after observing its tenth sample.
    n_blocks = len(original)//10
    blocks = original[:n_blocks*10].reshape(n_blocks,10,6).mean(axis=1,dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(indices,np.arange(699,len(original),100))
    assert saved.shape == (len(indices),18) and np.isfinite(saved).all()
    picks = np.array([0,len(indices)//2,len(indices)-1])
    model_directory = (BATCH.parent / "checkpoint" if model_key == "timesfm_3"
                       else BATCH / "forecast-checkpoints" / model_key)
    weights = digest(model_directory / "model.safetensors")
    assert weights == plan["models"][model_key]["weights_sha256"]
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if model_key == "timesfm_3":
        from timesfm3 import TimesFM3Forecaster
        official = TimesFM3Forecaster.from_pretrained(str(model_directory),device="cpu",local_files_only=True)
    else:
        sys.path.insert(0,str(BATCH / "timestral-upstream" / "src"))
        from timestral import TimestralForecaster
        official = TimestralForecaster.from_pretrained(model_directory,device="cpu",local_files_only=True)

    samples = []
    with torch.inference_mode():
        for pick in picks:
            stop = (int(indices[pick])+1)//10
            history = np.ascontiguousarray(blocks[stop-64:stop].T)
            if model_key == "timesfm_3":
                predicted = official.predict(history,horizon=8).forecast
            else:
                input_tensor = torch.from_numpy(history).unsqueeze(0)
                forecast = official.forecast(input_tensor,prediction_length=8,symmetric_averaging=False,clip_nonnegative=False)
                predicted = forecast.median[0].cpu().numpy()
            assert predicted.shape == (6,8)
            # Flatten by forecast horizon first, then the original six measured channels.
            changes = np.concatenate([predicted[:,h-1]-history[:,-1] for h in plan["horizon_steps"]])
            assert np.isfinite(changes).all()
            diff = np.abs(changes-saved[pick])
            samples.append({"native_index":int(indices[pick]),"cache_row":int(pick),
                            "feature_abs_error":diff.tolist(),
                            "max_abs_error_per_channel":np.max(diff.reshape(3,6),axis=0).tolist()})
    return {"model":model_key, "source":str(raw["source"]), "source_sha256":digest(raw["source"]),
            "weights_sha256":weights,"cache_sha256":digest(file), "cache_windows":len(indices),
            "cache_columns":saved.shape[1],"checked_windows":samples,
            "max_abs_error":max(max(r["feature_abs_error"]) for r in samples),
            "cpu_pytorch":str(torch.__version__),"reviewed_utc":datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    for key in sys.argv[1:]:
        if key not in {"timestral_209m","timestral_28m_fallback","timesfm_3"}:
            raise ValueError("Unknown frozen forecast model: "+key)
        proof = replay_probe(key)
        write(HERE / ("forecast_probe_"+key+".json"),proof)
        print(key,proof["cache_windows"],proof["max_abs_error"])
