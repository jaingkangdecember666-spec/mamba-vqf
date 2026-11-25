"""Validate the fixed Sobol map, the held-out folds and the random-feature comparators."""

import hashlib
import sys
from datetime import datetime, timezone
from statistics import fmean, NormalDist

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, check_scores, read, write
from replay_mounted_alpha import ALPHAS, fit_by_qr, replay_raw_scores, standardize_records, training_inputs
from verify_baselines import SHICE, digest
from verify_mount_pair import REFERENCE, group_record_means
from verify_mounted_alpha import PLAN as ALPHA_PLAN, PLAN_SHA as ALPHA_SHA


PLAN = BATCH / "mounted-sobol-ridge-protocol.json"
PLAN_SHA = "c6138480496b736ae6be4cf7a3b88916ec1fbd43a508f85390e6b2e126c44a54"
RID = "Y00001"
AXES = ("device","motion","joint_holdout")


def independent_operator(plan):
    import torch

    width, count = plan["input_dimensions"], plan["construction"]["points"]
    assert (width,count) == (66,128)
    points = torch.quasirandom.SobolEngine(width+1,scramble=False).draw(count,dtype=torch.float64).numpy()
    strata = (points*count).astype(np.int64)
    np.testing.assert_array_equal(np.sort(strata,axis=0),np.tile(np.arange(count)[:,None],(1,width+1)))
    midpoints = (np.arange(count)+.5)/count
    normal = NormalDist()
    quantiles = np.asarray([normal.inv_cdf(v) for v in midpoints],dtype=np.float32)
    projection = quantiles[strata[:,:width]].T / np.sqrt(width)
    shift = (-.8+1.6*midpoints[strata[:,width]]).astype(np.float32)
    generated = {"random":projection,"shift":shift,"midpoint_strata":strata}
    path = BATCH/plan["operator_file"]
    assert digest(path) == plan["operator_sha256"]
    with np.load(path,allow_pickle=False) as frozen:
        assert set(generated) == set(frozen.files)
        for key,data in generated.items():
            np.testing.assert_array_equal(data,frozen[key])
            assert hashlib.sha256(data.tobytes()).hexdigest() == plan["projection_fingerprints"][key]
    return {"random":projection,"shift":shift}


def compare_references(scored, references):
    assert references and all(set(anchor) == set(scored) for anchor in references.values())
    by_seed = {str(seed):fmean(scored[n]-anchor[n] for n in scored) for seed,anchor in references.items()}
    mean_by_record = {n:fmean(anchor[n] for anchor in references.values()) for n in scored}
    differences = {n:scored[n]-value for n,value in mean_by_record.items()}
    assert abs(fmean(differences.values())-fmean(by_seed.values())) < 1e-10
    return {"seeds":list(references),"reference_mean_deg":fmean(mean_by_record.values()),"fixed_minus_reference_deg":fmean(differences.values()),
            "fixed_minus_seed_deg":by_seed,"fixed_minus_mean_by_record_deg":differences,"records_improved":sum(v<0 for v in differences.values()),
            "by_group":group_record_means(differences)["groups"]}


def reconcile_summary(published, proof, job, result, pair_plan):
    from verify_final import preview_without_diagnostic

    assert published["round"] == RID and published["protocol_sha256"] == PLAN_SHA and published["result_sha256"] == digest(BATCH/job["result"])
    assert published["completed_at"] == job["ended_at"] and published["first_launch"]["python_pid"] == result["python_pid"]
    assert published["ridge_fits"] == 48 and published["installation_fits"] == 24 and published["failed_fits"] == 0
    assert published["r5_admissions"] == 0 and published["formal_preview_identical_without_diagnostic"] is True
    assert published["deterministic_configurations_counted_once"] == 2
    score_pairs = 0
    for alpha,member in zip(ALPHAS,("lower_alpha","with_mount"),strict=True):
        arm = str(alpha)
        official = published["comparisons"][arm]
        for pair,target in zip(pair_plan["pairs"],official["matched"],strict=True):
            seed = pair["seed"]
            assert target["search_seed"] == seed and target["grouped_round"] == pair[member]["round"] and target["joint_round"] == pair[member]["joint_round"]
            for axis in AXES:
                source = target["joint_round"] if axis == "joint_holdout" else target["grouped_round"]
                original = read(BATCH/"rounds"/source/"result.json")["sequences"]
                originals = {r["name"]:r for r in (original[axis] if isinstance(original,dict) else original)}
                deterministic = {r["name"]:r for r in result["sequences"][axis]}
                pair_rows = target["axes"][axis]["records"]
                by_record = {r["name"]:r for r in pair_rows}
                assert len(pair_rows) == len(by_record) and set(by_record) == set(originals) == set(deterministic)
                for n,p in by_record.items():
                    s,rand = deterministic[n],originals[n]
                    assert p["fold"] == s["fold"] == rand["fold"] and p["frames"] == s["frames"] == rand["frames"]
                    assert p["score_frame_indices_sha256"] == s["score_frame_indices_sha256"]
                    assert p["sobol_rmse_deg"] == s["rmse_deg"][arm] and p["seeded_rmse_deg"] == rand["rmse_deg"]
                    assert abs(p["sobol_minus_seeded_deg"]-(p["sobol_rmse_deg"]-p["seeded_rmse_deg"])) < 1e-10
                expected = proof["axes"][axis][arm]
                assert abs(target["axes"][axis]["difference_deg"]-expected["search"]["fixed_minus_seed_deg"][str(seed)]) < 1e-9
                score_pairs += len(by_record)
        diagnostic_means = {}
        for axis in AXES:
            data = proof["axes"][axis][arm]
            assert official["scores"][axis] == data["fixed_deg"]
            effects = list(data["search"]["fixed_minus_seed_deg"].values())
            reported = official["paired_summary"][axis]
            np.testing.assert_allclose([reported["mean_sobol_minus_seeded_deg"],*reported["pairwise_range_deg"]],
                                       [fmean(effects),min(effects),max(effects)],rtol=0,atol=1e-9)
            if "diagnostic" in data:
                diagnostic_means[axis] = {seed:data["fixed_deg"]-v for seed,v in data["diagnostic"]["fixed_minus_seed_deg"].items()}
        diagnostic_means["combined_device_program"] = {s:(v+diagnostic_means["motion"][s])/2 for s,v in diagnostic_means["device"].items()}
        for axis,per_seed in diagnostic_means.items():
            v = list(per_seed.values())
            reported = official["historical_independent_seed_description_only"][axis]
            np.testing.assert_allclose([reported["mean"],*reported["range"]],[fmean(v),min(v),max(v)],rtol=0,atol=1e-9)
        for timing in ("solve_seconds","score_seconds"):
            total = sum(f["fit_by_alpha"][arm][timing] for f in result["folds"])
            assert abs(published["timing"]["per_alpha"][arm][timing]-total) < 1e-6
    assert published["timing"]["wall_seconds"] == result["elapsed_seconds"] and score_pairs == 666
    assert abs(published["timing"]["prep_seconds"]-sum(f["prepare_seconds"] for f in result["folds"])) < 1e-6
    assert abs(published["timing"]["shared_normal_equation_seconds"]-sum(f["shared_normal_equation_seconds"] for f in result["folds"])) < 1e-6
    index = {r["round_id"]:r for r in read(BATCH/"round_index.json")}
    size, signature = preview_without_diagnostic(index,published["at"],job["config"]["kind"])
    return {"score_pairs":score_pairs,"preview_size":size,"preview_sha256":signature}


def check_trial(job, result, frames):
    plan, ref = read(PLAN), read(REFERENCE)
    assert digest(PLAN) == PLAN_SHA == job["config"]["protocol_sha256"]
    assert job["round_id"] == result["round_id"] == plan["round_id"] == RID
    assert job["config"]["seed"] is None and job["dev_metrics"] is None and result["dev_metrics"] is None
    assert plan["diagnostic_only"] and plan["eligible_for_r5"] is result["eligible_for_r5"] is False
    assert set(job["partitions"]) == {"development","excluded"} and set(job["partitions"]["development"]) == set(frames)
    assert datetime.fromisoformat(plan["registered_at"]) <= datetime.fromisoformat(job["started_at"])
    assert plan["alphas"] == list(ALPHAS) and plan["shared_config"]["update_hz"] == 50 and plan["shared_config"]["context"] == 96
    assert plan["regular_pair_protocol_sha256"] == ALPHA_SHA == digest(ALPHA_PLAN)
    assert digest(BATCH/"ridge-50hz-mount-alpha-summary.json") == plan["independent_description_source_sha256"]
    assert all(digest(BATCH/f)==sha for f,sha in plan["historical_scoring_code_sha256"].items())
    assert result["operator_sha256"] == plan["operator_sha256"] and result["input_dimensions"] == plan["input_dimensions"]
    independent_operator(plan)
    assert result["actual_ridge_fits"] == 48 and result["actual_installation_fits"] == len(result["folds"]) == 24 and result["failures"] == 0
    assert result["unique_deterministic_configurations"] == 2 and result["seed_repeats"] == 0
    assert result["mask_version"] == "causal-bin-v2-full-window" and set(result["sequences"]) == set(result["axis_metrics"]) == set(AXES)
    assert result["training_context_audit"] == read(BATCH/"rounds/M00102/result.json")["training_context_audit"]
    aplan = read(ALPHA_PLAN)
    archive = read(BATCH/"ridge-50hz-mount-alpha-summary.json")
    for rid,sha in plan["existing_comparison_round_result_sha256"].items():
        assert digest(BATCH/"rounds"/rid/"result.json") == sha
    assert len(plan["existing_comparison_round_result_sha256"]) == 12
    independent_hashes = {r["round"]:r["result_sha256"] for r in archive["actual_runs"]}
    times, summaries, score_vectors = {}, {}, {}
    arms = {str(a) for a in ALPHAS}
    for axis in AXES:
        raw = result["sequences"][axis]
        rows = {r["name"]:r for r in raw}
        assert len(raw) == len(rows) == len(frames) and set(rows) == set(frames)
        folds = [f for f in result["folds"] if f["axis"] == axis]
        assert len({f["fold"] for f in folds}) == len(folds) == (16 if axis == "joint_holdout" else 4)
        for f in folds:
            held = check_fold(f,axis,set(frames))
            assert held == {n for n,r in rows.items() if r["fold"] == f["fold"]}
            target = ref["folds"][axis+":"+f["fold"]]
            assert sorted(f["training"]) == target["training"] and sorted(held) == target["validation"]
            assert abs(np.linalg.norm(f["installation_quaternion_wxyz"])-1) < 2e-6
            one,two = (Rotation.from_quat(q,scalar_first=True) for q in (f["installation_quaternion_wxyz"],target["wxyz"]))
            assert np.rad2deg((one.inv()*two).magnitude()) < 2e-5
            assert set(f["fit_by_alpha"]) == arms and all(v["fits"] == 1 for v in f["fit_by_alpha"].values())
        assert all(r["score_frame_indices_sha256"] == ref["source_records"][n]["score_index_sha256"] and set(r["rmse_deg"]) == arms for n,r in rows.items())
        summaries[axis], score_vectors[axis] = {}, {}
        for alpha,arm in zip(ALPHAS,("lower_alpha","with_mount"),strict=True):
            key = str(alpha)
            scalar = [{**r,"rmse_deg":r["rmse_deg"][key]} for r in raw]
            metrics = result["axis_metrics"][axis][key]
            check_scores(scalar,metrics,frames)
            assert set(metrics["per_device"]) == {"R1","R2","R3","R4"}
            for d,group in metrics["per_device"].items():
                check_scores([r for r in scalar if r["name"].split("-")[0]==d],group,frames)
            fixed = {r["name"]:r["rmse_deg"] for r in scalar}
            score_vectors[axis][key] = fixed
            references = {"search":{},"diagnostic":{}}
            jobs = [(p["seed"],p[arm]["joint_round" if axis=="joint_holdout" else "round"],"search") for p in aplan["pairs"]]
            if axis != "joint_holdout":
                jobs += [(j["config"]["seed"],j["round_id"],"diagnostic") for j in aplan["independent_seed_rows"] if j["config"]["alpha"]==alpha]
            for seed,rid,group in jobs:
                path = BATCH/"rounds"/rid/"result.json"
                sha = (plan["existing_comparison_round_result_sha256"] if group=="search" else independent_hashes)[rid]
                assert digest(path) == sha
                source = read(path)["sequences"]
                previous = source[axis] if isinstance(source,dict) else source
                for r in previous:
                    n = r["name"]
                    assert r["fold"] == rows[n]["fold"] and r["frames"] == rows[n]["frames"] and r["baseline_deg"] == rows[n]["full_vqf_deg"]
                references[group][seed] = {r["name"]:r["rmse_deg"] for r in previous}
            summaries[axis][key] = {"fixed_deg":metrics["macro_rmse_deg"], **{g:compare_references(fixed,arr) for g,arr in references.items() if arr}}
        times[axis] = {"prepare_seconds":sum(f["prepare_seconds"] for f in folds),"shared_equation_seconds":sum(f["shared_normal_equation_seconds"] for f in folds),
                       "solve_seconds":sum(t["solve_seconds"] for f in folds for t in f["fit_by_alpha"].values()),
                       "score_seconds":sum(t["score_seconds"] for f in folds for t in f["fit_by_alpha"].values())}
        assert all(0<v<=result["elapsed_seconds"] for v in times[axis].values())
    live = read(HERE/"sobol_start.json")
    assert live["native_pid"] == result["python_pid"] and live["native_parent"] == live["launcher_pid"]
    assert live["round_id"] == RID and live["supervisor_pid"] == plan["supervisor_pid"] and live["round_id"] in live["native_command"]
    assert datetime.fromisoformat(job["started_at"]) <= datetime.fromisoformat(live["observed_at"]) <= datetime.fromisoformat(job["ended_at"])
    proof = {"id":RID,"operator_sha256":plan["operator_sha256"],"plan_sha256":PLAN_SHA,"solver_fits":48,"installation_fits":24,
             "unique_configs":2,"axes":summaries,"per_axis_component_seconds":times,"elapsed_seconds":result["elapsed_seconds"],
             "fixed_scores_by_record":score_vectors,"raw_frames_per_axis":sum(frames.values()),"records_per_axis":len(frames)}
    summary_file = BATCH/"mounted-sobol-ridge-summary.json"
    if summary_file.is_file():
        proof["published"] = reconcile_summary(read(summary_file),proof,job,result,aplan)
        proof["published"]["sha256"] = digest(summary_file)
    return proof


def replay():
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    plan = read(PLAN)
    assert digest(PLAN) == PLAN_SHA
    path = BATCH/"rounds"/RID/"result.json"
    data = read(path)
    ref = read(REFERENCE)
    fold, = (f for f in data["folds"] if (f["axis"],f["fold"]) == ("joint_holdout","R2/M2"))
    check_fold(fold,"joint_holdout",set(ref["source_records"]))
    q = np.asarray(fold["installation_quaternion_wxyz"],dtype=np.float32)
    configs = [{**plan["shared_config"],"alpha":a} for a in ALPHAS]
    rows,support = training_inputs(fold["training"],configs[0],q)
    assert all(data["training_context_audit"][n] == v for n,v in support.items())
    state = {**standardize_records(rows),**independent_operator(plan)}
    coefs = fit_by_qr(rows,state,ALPHAS)
    raw_scores = replay_raw_scores(fold["validation"],ref["source_records"],q,configs,state,coefs)
    return {"at":datetime.now(timezone.utc).isoformat(),"round":RID,"fold":fold["fold"],"axis":fold["axis"],
            "source_sha256":digest(path),"plan_sha256":PLAN_SHA,"operator_sha256":plan["operator_sha256"],"training":fold["training"],
            "training_updates":sum(len(x) for x,_ in rows),"alphas":list(ALPHAS),"audit_refits":2,"raw_records":raw_scores}


def check_replay(receipt):
    path = BATCH/"rounds"/RID/"result.json"
    assert receipt["round"] == RID and receipt["source_sha256"] == digest(path) and receipt["plan_sha256"] == PLAN_SHA == digest(PLAN)
    assert receipt["operator_sha256"] == read(PLAN)["operator_sha256"] and receipt["alphas"] == list(ALPHAS) and receipt["audit_refits"] == 2
    assert (receipt["axis"],receipt["fold"]) == ("joint_holdout","R2/M2")
    data,ref = read(path),read(REFERENCE)
    split = ref["folds"]["joint_holdout:R2/M2"]
    assert sorted(receipt["training"]) == split["training"] and len(split["training"]) == 19
    assert receipt["training_updates"] == sum(data["training_context_audit"][n]["training_labels_eligible"] for n in split["training"])
    official = {r["name"]:r for r in data["sequences"]["joint_holdout"] if r["fold"] == "R2/M2"}
    assert set(receipt["raw_records"]) == set(official) == set(split["validation"])
    gaps = []
    for n,r in receipt["raw_records"].items():
        source = SHICE/"R2_aligned_formal"/(n+".csv")
        assert r["source"] == str(source) and r["source_sha256"] == digest(source)
        assert r["frames"] == official[n]["frames"] and r["score_frame_indices_sha256"] == official[n]["score_frame_indices_sha256"]
        for alpha,value in zip(ALPHAS,r["qr_rmse_deg"],strict=True):
            assert 0<=value<=180
            gaps.append(abs(value-official[n]["rmse_deg"][str(alpha)]))
    assert max(gaps) < 2e-4
    return {"comparisons":len(gaps),"records":len(official),"total_frames":sum(r["frames"] for r in official.values()),"max_abs_degree_gap":max(gaps)}


if __name__ == "__main__":
    receipt = replay()
    print(check_replay(receipt))
    write(HERE/"sobol_raw_fold_replay.json",receipt)
