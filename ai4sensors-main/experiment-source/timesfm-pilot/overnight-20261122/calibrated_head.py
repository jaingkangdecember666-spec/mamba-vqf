"""Frozen five-pose causal accelerometer compensation with the matched legacy physical head."""

from functools import lru_cache
from statistics import mean
from time import perf_counter

import numpy as np
import torch

import learning
from bias_calibration import OnlineBiasTracker, PROTOCOL, online_filter, read_native, stationary_candidates
from common import BATCH, Z, body_up, digest, fingerprint, names, read_json, record, rotate, stamp, unit, write_json
from features import block_indices, cache_record
from legacy_head import evaluate, fit, make_model, prepare
from optimizer_study import round_record, schedule
from tilt_eval import predict_updates
from tilt_model import physical_features


STAGE = BATCH/"accelerometer-bias-shortlist-stage.json"
METHODS = {"frozen_head":"A00004","retrained_head":"A00005"}


def verify_protocol(config):
    if config["protocol_sha256"] != fingerprint(PROTOCOL) or config["minimum_online_poses"] != 5:
        raise ValueError("Accelerometer calibration does not match the frozen pre-R5 protocol")


def bundle(imu,dt,score,context,reference,config,nominal_shice,packet_ok):
    verify_protocol(config)
    parameters = config["parameters"]
    original = cache_record(imu,dt,score,context,reference,parameters,nominal_shice,keep_scored_quaternion=True)
    corrected,trace,_,audit = online_filter(imu,packet_ok,dt,config["minimum_online_poses"])
    if audit["first_applied_native_index"] is None:
        adjusted = original
    else:
        adjusted = cache_record(corrected,dt,score,context,reference,parameters,nominal_shice,keep_scored_quaternion=True)
    np.testing.assert_array_equal(original["index"],adjusted["index"])
    return {"original":original,"adjusted":adjusted,"index":original["index"],"online_scope":{
                "first_applied_seconds":audit["first_applied_seconds"],"active_score_frames":int(np.count_nonzero(
                    np.linalg.norm(trace[score],axis=1)>1e-12)), "pose_clusters":audit["pose_clusters"]}}


@lru_cache(maxsize=6)
def known_record(name, minimum_poses, tau):
    source = read_native(name)
    raw = learning.attach_native(name,{"parameters":{"tauAcc":tau}})
    dt = float(source["dt"])
    clean,trace,_,audit = online_filter(source["untreated"],source["original_ok"],dt,minimum_poses)
    if audit["first_applied_native_index"] is None:
        adjusted = raw
    else:
        adjusted = cache_record(clean,dt,source["score"],source["context"],source["ref"],{"tauAcc":tau},True,
                                keep_scored_quaternion=True)
    np.testing.assert_array_equal(adjusted["index"],raw["index"])
    return {"original":raw,"adjusted":adjusted,"index":raw["index"],"online_scope":{
                "first_applied_seconds":audit["first_applied_seconds"],"active_score_frames":int(np.count_nonzero(
                    np.linalg.norm(trace[source["score"]],axis=1)>1e-12)),"pose_clusters":audit["pose_clusters"]}}


def cached(name,config):
    verify_protocol(config)
    return known_record(name,config["minimum_online_poses"],config["parameters"]["tauAcc"])


def fit_candidate(config,training,device):
    head = config["head"]
    raw = [cached(name,config) for name in training]
    q_mount = learning.installation_fit([data["original"] for data in raw],head)
    key = "adjusted" if config["training_input"] == "retrained_head" else "original"
    inputs = [prepare(learning.apply_install(part[key],q_mount,include_reference=False),head) for part in raw]
    state,_ = fit(inputs,head,device,training)
    return {**state,"installation_quaternion_wxyz":q_mount}


def score(data,config,state,device,delay_ms=0.):
    verify_protocol(config)
    head = config["head"]
    mounted = learning.apply_install(data["adjusted"],state["installation_quaternion_wxyz"])
    return evaluate(prepare(mounted,head),head,make_model(state,device),device,delay_ms)


def benchmark(config,state,device,benchmark_record):
    from vqf import VQF
    from final_eval import timed
    verify_protocol(config)
    source = read_native(benchmark_record)
    imu, valid, dt = source["untreated"],source["original_ok"],float(source["dt"])
    width = max(1,round(PROTOCOL["block_seconds"]/dt))
    tracker = OnlineBiasTracker(config["minimum_online_poses"])
    q_mount = state["installation_quaternion_wxyz"]
    current = np.zeros(6)
    native_filter = VQF(dt,**config["parameters"])
    cached_record = cached(benchmark_record,config)
    activated_at = cached_record["online_scope"]["first_applied_seconds"]
    start = 22000 if activated_at is None else max(22000,round(activated_at/dt)+1000)
    start = min(start,len(imu)-3100)
    refit_timings = []

    def step(j, *, measure_refit=False):
        nonlocal current
        if valid[j]:
            current = imu[j].astype(np.float64)
            current[:3] -= tracker.offset
        native_filter.update(current[3:],current[:3])
        q = learning.multiply_quaternion(native_filter.getQuat6D(),q_mount)
        if (j+1)%width==0:
            begin = j+1-width
            gate_started = perf_counter()
            _, static, _ = stationary_candidates(imu[begin:j+1],valid[begin:j+1],dt)
            if len(static):
                tracker.observe(static[0],j+1)
            if measure_refit:
                refit_timings.append((perf_counter()-gate_started)*1000.0)
        return q

    for i in range(start):
        step(i)
    native_timing = timed(lambda i:step(start+i,measure_refit=i>=100),3000,100)
    prepared = learning.apply_install(cached_record["adjusted"],q_mount,include_reference=False)
    candidates = block_indices(len(prepared["windows"]),10)
    positions = np.linspace(0,len(candidates)-1,110,dtype=int)
    model = make_model(state,device)

    def update(i):
        at = candidates[positions[i]]
        physical = physical_features(prepared["windows"][None,at-63:at+1],prepared["up"][at:at+1],
                                     prepared["bias"][at:at+1],prepared["rest"][at:at+1])
        residual = predict_updates(model,physical,None,device)*config["head"]["gain"]
        return unit(Z+rotate(unit(prepared["q"][at:at+1]),residual))

    held = update(0)[0]
    apply = timed(lambda i:body_up(prepared["q_scored"][i],held),3000,100)
    result = {"record":benchmark_record,"native_step":native_timing,"nominal_native_hz":1./dt,
              "correction_update":timed(update,100,10),"correction_hz":1./(100*dt),
              "apply_held_correction":apply,"native_includes_causal_online_bias_and_mount":True,
              "measured_bias_blocks":len(refit_timings),
              "native_benchmark_start_seconds":start*dt,
              "first_online_calibration_seconds":activated_at,
              "calibration_active_at_end_of_native_benchmark":bool(np.linalg.norm(tracker.offset)>1e-12)}
    if refit_timings:
        values = np.percentile(refit_timings,[50,95,100]).tolist()
        result["online_bias_block_update"] = dict(zip(("p50_ms","p95_ms","max_ms"),values,strict=True))
        result["bias_block_hz"] = 1./(width*dt)
    return result


def derive(job,destination):
    config = job["config"]
    evidence_path = BATCH / "rounds/A00003/result.json"
    if config["matched_experiment_sha256"] != digest(evidence_path):
        raise ValueError("Matched-bias evidence changed before shortlist materialization")
    evidence = read_json(evidence_path)
    result = next(row for row in evidence["results"] if row["seed"] == config["seed"])
    method = config["training_input"]
    metrics = {axis:row[method] for axis,row in result["metrics"].items()}
    metrics["selection_rmse_deg"] = (metrics["device"]["macro_rmse_deg"]+metrics["motion"]["macro_rmse_deg"])/2
    data = {axis:[{**row,"rmse_deg":row["rmse_deg"][method]} for row in rows] for axis,rows in result["sequences"].items()}
    write_json(destination,{"round_id":job["round_id"],"dev_metrics":metrics,"sequences":data,
        "derived_from_actual_matched_scored_folds":"A00003","not_an_additional_fit":True,
        "eligible_for_r5":True,"source_seed":config["seed"]})


def queue():
    path = BATCH / "rounds/A00003/result.json"
    scored = read_json(path)
    rule = read_json(BATCH/"accelerometer-bias-light-matched-stage.json")["promotion"]
    accepted = []
    for method, identity in METHODS.items():
        results = scored["results"]
        effect = {axis:[row["metrics"][axis][method]["macro_rmse_deg"]-row["metrics"][axis]["zero"]["macro_rmse_deg"]
                        for row in results] for axis in ("device","motion","joint_holdout")}
        passes = (mean(effect["joint_holdout"]) <= -rule["minimum_three_seed_average_joint_improvement_deg"] and
                  sum(v<0 for v in effect["joint_holdout"]) >= rule["minimum_matched_seed_joint_wins"] and
                  mean(effect["device"]) <= rule["maximum_leave_device_increase_deg"])
        median = sorted(results,key=lambda row:row["metrics"]["joint_holdout"][method]["macro_rmse_deg"])[1]
        head = round_record(median["reference_round"])["config"]
        row = {"status":"qualified" if passes else "rejected","method":method,"mean_paired_delta_deg":{k:mean(v) for k,v in effect.items()},
               "paired_joint_deltas_deg":effect["joint_holdout"],"median_joint_seed":head["seed"],
               "paired_zero_round":median["reference_round"]}
        if passes:
            config = {"kind":"bias_physical_head","head":head,"parameters":head["parameters"],"seed":head["seed"],
                      "training_input":method,"minimum_online_poses":5,"paired_zero_round":median["reference_round"],
                      "protocol_sha256":fingerprint(PROTOCOL),"matched_experiment_sha256":digest(path)}
            row["round_id"] = schedule(identity,config,"static-sphere-light-candidate-from-matched-audit","A00003",
                "Promote the median matched seed from the completed bias-head audit without rerunning it")
        accepted.append(row)
    write_json(STAGE,{"updated_at":stamp(),"evidence":"A00003","outcomes":accepted,
              "selection_rule_predeclared_before_r5":"R5 must activate on observed frames and improve against the reconstructed uncalibrated paired head."})
    print(accepted,flush=True)


if __name__ == "__main__":
    queue()
