"""Pair the actual online accelerometer compensation with fixed or retrained causal physical heads."""

from time import perf_counter

import numpy as np
import torch

import learning
from bias_calibration import online_filter, read_native
from common import BATCH, aggregate, by_device, fingerprint, names, read_json, stamp, write_json
from features import cache_record
from joint_holdout import joint_folds
from legacy_head import evaluate, fit, make_model, prepare
from optimizer_study import round_record, schedule


STAGE = BATCH / "accelerometer-bias-light-matched-stage.json"
ANCHORS = (("P00005", "J00015"), ("P00021", "J00017"), ("P00022", "J00016"))


def training_plan(records):
    grouped = [(axis,label,[name for name in records if name not in held],held)
               for axis,label,held in learning.folds(records)]
    return grouped + [("joint_holdout",name,train,held) for name,train,held in joint_folds(records)]


def expected_original(grouped,joint):
    standard = read_json(BATCH / round_record(grouped)["result"])["sequences"]
    simultaneous = read_json(BATCH / round_record(joint)["result"])["sequences"]["joint_holdout"]
    full = {**standard,"joint_holdout":simultaneous}
    return {key:{row["name"]:row for row in rows} for key,rows in full.items()}


def support_data(records, config, minimum_poses):
    out = {}
    base = {name: learning.attach_native(name,config) for name in records}
    audit = {}
    for name in records:
        source = read_native(name)
        dt = float(source["dt"])
        clean, trace, _, diag = online_filter(source["untreated"],source["original_ok"],dt,minimum_poses)
        if diag["first_applied_native_index"] is None:
            out[name] = base[name]
        else:
            out[name] = cache_record(clean,dt,source["score"],source["context"],source["ref"],config["parameters"],True,
                                     keep_scored_quaternion=True)
        np.testing.assert_array_equal(out[name]["index"],base[name]["index"])
        audit[name] = {"first_applied_seconds":diag["first_applied_seconds"],
                       "active_score_frames":int(np.count_nonzero(np.linalg.norm(trace[source["score"]],axis=1)>1e-12)),
                       "score_frames":len(base[name]["index"])}
    return base,out,audit


def run(job,destination):
    started = perf_counter()
    planned = read_json(STAGE)
    record_ids = names("dev")
    config = job["config"]
    device = torch.device("cuda")
    strong = round_record(config["representative_anchor"])
    base,corrected,activation = support_data(record_ids,strong["config"],config["minimum_online_poses"])
    support_in_scored_frames = {n for n,audit in activation.items() if audit["active_score_frames"]}
    results = []
    for item in planned["anchors"]:
        model_cfg = round_record(item["round_id"])["config"]
        if fingerprint(model_cfg)!=item["config_sha256"]:
            raise ValueError("The matched physical head was modified")
        expected = expected_original(item["round_id"],item["joint_round"])
        grouped,fold_rows = {},[]
        for axis,label,train,held in training_plan(record_ids):
            fitting_started = perf_counter()
            q_mount = learning.installation_fit([base[n] for n in train],model_cfg)
            seeds = {}
            for key,raw in (("standard",base),("adapted",corrected)):
                data = [prepare(learning.apply_install(raw[n],q_mount,include_reference=False),model_cfg) for n in train]
                seeds[key] = fit(data,model_cfg,device,train)[0]
            scoring_started = perf_counter()
            standard, adapted = (make_model(seeds[key],device) for key in ("standard","adapted"))
            rows = []
            for name in held:
                original = prepare(learning.apply_install(base[name],q_mount),model_cfg)
                identical_input = corrected[name] is base[name]
                calibrated = original if identical_input else prepare(learning.apply_install(corrected[name],q_mount),model_cfg)
                zero = evaluate(original,model_cfg,standard,device)
                score = expected[axis][name]
                if abs(score["rmse_deg"]-zero)>1e-5 or score["frames"]!=len(original["index"]):
                    raise ValueError(f"Physical match check failed on {axis}/{name}")
                retained = zero if identical_input else evaluate(calibrated,model_cfg,standard,device)
                new = evaluate(calibrated,model_cfg,adapted,device)
                rows.append({"name":name,"frames":len(original["index"]),"baseline_deg":zero,
                             "rmse_deg":{"zero":zero,"frozen_head":retained,"retrained_head":new}})
            grouped.setdefault(axis,[]).extend(rows)
            fold_rows.append({"axis":axis,"fold":label,"training":train,"validation":held,
                              "fit_seconds":scoring_started-fitting_started,"score_seconds":perf_counter()-scoring_started})
            print(item["round_id"],axis,label,flush=True)
        estimates = {}
        for axis,rows in grouped.items():
            estimates[axis] = {}
            for method in ("zero","frozen_head","retrained_head"):
                converted = [{**row,"rmse_deg":row["rmse_deg"][method]} for row in rows]
                values = {**aggregate(converted),"per_device":by_device(converted)}
                values["supported_records_mean_delta_deg"] = float(np.mean(
                    [row["rmse_deg"]-row["baseline_deg"] for row in converted if row["name"] in support_in_scored_frames]))
                values["unsupported_records_mean_delta_deg"] = float(np.mean(
                    [row["rmse_deg"]-row["baseline_deg"] for row in converted if row["name"] not in support_in_scored_frames]))
                estimates[axis][method] = values
        results.append({"seed":model_cfg["seed"],"reference_round":item["round_id"],"reference_joint_round":item["joint_round"],
                        "metrics":estimates,"folds":fold_rows,"sequences":grouped})
    write_json(destination,{"finished_at":stamp(),"config":config,"elapsed_seconds":perf_counter()-started,
        "activation":activation,"results":results,"same_masks_and_fold_mounting":True,
        "recording_id_never_used_by_estimator":True,"development_only":True})


def queue():
    source = read_json(BATCH/"rounds/A00002/result.json")
    assert source["exact_prefix_replay"] and source["online_records_with_support"]>0
    assert source["summary"]["joint_holdout"]["zero"]["macro_rmse_deg"]-source["summary"]["joint_holdout"]["online_shrunk"]["macro_rmse_deg"]>=.015
    references = [{"round_id":grouped,"joint_round":joint,"config_sha256":round_record(grouped)["config_sha256"]}
                  for grouped,joint in ANCHORS]
    write_json(STAGE,{"created_at":stamp(),"anchors":references,"bias_rule_source":"A00002",
           "recurrent_or_other_physical_options_not_changed":True,
           "arms":["zero","same fitted head with record-local online compensation","retrain same head with causal online compensation"],
           "promotion":{"minimum_three_seed_average_joint_improvement_deg":.015,"minimum_matched_seed_joint_wins":2,
                        "maximum_leave_device_increase_deg":.01}})
    cfg = {"kind":"accelerometer_bias_head_audit","representative_anchor":"P00022","minimum_online_poses":5,"bias_rule_round":"A00002","seed":None}
    print(schedule("A00003",cfg,"static-sphere-paired-physical-head","A00002",
                   "Match seeds, physical head architecture, native masks, and trained mounting while perturbing only the causal offset input"),flush=True)


if __name__ == "__main__":
    queue()
