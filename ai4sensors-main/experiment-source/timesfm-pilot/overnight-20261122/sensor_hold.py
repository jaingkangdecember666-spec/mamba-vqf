"""Compare one frozen correction head with world-frame vs sensor-frame rotation hold."""

from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import learning
from common import BATCH, Z, aggregate, body_up, by_device, errors, fingerprint, names, read_json, record, stamp, unit, write_json
from features import block_indices
from final_eval import fit_candidate
from legacy_head import make_model
from optimizer_study import round_record,schedule
from quaternion_audit import IDENTITY,inertial_states,rotation_difference,source_time
from regime_audit import LABELS,cached_strata
from stream_gru import planned_folds
from tilt_eval import predict_updates
from tilt_model import contexts,correction_quaternions,multiply,normalize,physical_features


STAGE = BATCH/"sensor-frame-hold-development-protocol.json"
LIMITS = {"minimum_joint_improvement_deg":0.015,"maximum_allowed_axis_regression_deg":0.000001}
AGE_EDGES = (25.,50.,75.,100.)
AGE_LABELS = ("0_to_25ms","25_to_50ms","50_to_75ms","75_to_100ms","at_least_100ms","before_first_correction")


def hold_two_frames(native_base, release_indices, base_at_update, body_delta):
    """World rotations left-multiply; the equivalent sensor rotations right-multiply."""
    head = normalize(base_at_update)
    correction = correction_quaternions(head,body_delta)
    inverse = head.copy()
    inverse[:,1:] *= -1
    sensor_rotation = normalize(multiply(multiply(inverse,correction),head))
    active = np.searchsorted(release_indices,np.arange(len(native_base)),side="right")
    q_world = normalize(multiply(np.vstack((IDENTITY,correction))[active],native_base))
    q_sensor = normalize(multiply(native_base,np.vstack((IDENTITY,sensor_rotation))[active]))
    return q_world,q_sensor,active


def native_predictions(base, stream, config, state, device):
    mount = state["installation_quaternion_wxyz"]
    q = normalize(multiply(base,mount))
    stream = learning.apply_install(stream,mount,include_reference=False)
    positions = block_indices(len(stream["windows"]),config["update_hz"])
    history = contexts(stream["windows"],positions)
    inputs = physical_features(history,stream["up"][positions],stream["bias"][positions],stream["rest"][positions])
    model = make_model(state,device)
    delta = predict_updates(model,inputs,None,device)*config["gain"]
    when = stream["ready"][positions]
    a,b,active = hold_two_frames(q,when,stream["q"][positions],delta)
    return a,b,active,when


def synthetic():
    n = 260
    rotation = np.column_stack((np.linspace(.01,.6,n),np.linspace(.05,-.4,n),np.linspace(.1,.2,n)))
    moving = Rotation.from_rotvec(rotation).as_quat(scalar_first=True)
    static = np.repeat(moving[:1],n,axis=0)
    releases = np.arange(24,n,30)
    delta = np.tile([.031,-.022,.017],(len(releases),1))

    zero_world,zero_body,_ = hold_two_frames(moving,releases,moving[releases],np.zeros_like(delta))
    np.testing.assert_allclose(zero_world,moving,atol=1e-13)
    np.testing.assert_allclose(zero_body,moving,atol=1e-13)
    stat_w,stat_b,_ = hold_two_frames(static,releases,static[releases],delta)
    np.testing.assert_allclose(stat_w,stat_b,atol=1e-13)
    moving_w,moving_b,_ = hold_two_frames(moving,releases,moving[releases],delta)
    np.testing.assert_allclose(moving_w[releases],moving_b[releases],atol=1e-13)
    if np.max(rotation_difference(moving_w,moving_b)) < 0.03:
        raise ValueError("Non-commuting moving control did not distinguish the two coordinate frames")
    for i,when in enumerate(releases[:-1]):
        to = releases[i+1]
        base_up = body_up(moving[when])
        corrected_up = unit(base_up+delta[i])
        correction,_ = Rotation.align_vectors(base_up,corrected_up)
        independent = correction.inv().apply(body_up(moving[when:to]))
        np.testing.assert_allclose(body_up(moving_b[when:to]),independent,atol=2e-13)
    end = 177
    preserved = releases < end
    cut_w,cut_b,_ = hold_two_frames(moving[:end],releases[preserved],moving[releases[preserved]],delta[preserved])
    np.testing.assert_array_equal(cut_w,moving_w[:end])
    np.testing.assert_array_equal(cut_b,moving_b[:end])
    mask_a = np.arange(n)%3>0
    mask_b = np.arange(n)%5>0
    tile = np.flatnonzero(mask_a & mask_b)
    np.testing.assert_array_equal(moving_b[mask_a][np.searchsorted(np.flatnonzero(mask_a),tile)],
                                  moving_b[mask_b][np.searchsorted(np.flatnonzero(mask_b),tile)])
    return {"zero_correction_same_base":True,"static_pose_two_holds_equal":True,"continuous_turn_matches_independent_geometry":True,
            "exact_quaternion_prefix":True,"mask_changes_only_readout_indices":True,
            "max_angular_distinction_on_moving_synthetic_deg":float(np.max(rotation_difference(moving_w,moving_b)))}


def slice_summary(labels,codes,world,sensor):
    count = np.bincount(codes,minlength=len(labels))
    a = np.bincount(codes,weights=np.square(world,dtype=np.float64),minlength=len(labels))
    b = np.bincount(codes,weights=np.square(sensor,dtype=np.float64),minlength=len(labels))
    return {key:{"frames":int(n),"world_squared_error":float(old),"sensor_squared_error":float(new),
                 "world_rmse_deg":float(np.sqrt(old/n)) if n else None,
                 "sensor_rmse_deg":float(np.sqrt(new/n)) if n else None}
            for key,n,old,new in zip(labels,count,a,b,strict=True)}


def anchor_scores(source,joint):
    view = read_json(BATCH/round_record(source)["result"])["sequences"]
    another = read_json(BATCH/round_record(joint)["result"])["sequences"]["joint_holdout"]
    return {axis:{r["name"]:r for r in rows} for axis,rows in {**view,"joint_holdout":another}.items()}


def run(job,destination):
    stage = read_json(STAGE)
    config = job["config"]
    row = round_record(config["parent_round"])
    head = row["config"]
    if fingerprint(head) != config["head_sha256"]:
        raise ValueError("Frozen legacy correction head configuration was changed")
    baseline = anchor_scores(config["parent_round"],config["joint_round"])
    record_names = names("dev")
    device = torch.device("cuda")
    prepared = {}
    started = perf_counter()
    for name in record_names:
        source = record(name)
        dt = float(source["dt"])
        base, stream = inertial_states(source["imu"],dt,head["parameters"])
        native = source_time(name,len(base))
        strata = cached_strata(name)
        index = np.flatnonzero(source["score"])
        np.testing.assert_array_equal(index,strata["index"])
        prepared[name] = {"source":source,"base":base,"stream":stream,"time":native,
                          "speed":strata["speed"],"index":index}
    all_axes,folds = {},[]
    prefix_test = None
    count = 0
    for axis,label,train,held in planned_folds(record_names):
        start_fit = perf_counter()
        state = fit_candidate(head,train)
        count += 1
        score_start = perf_counter()
        results = []
        for name in held:
            source = prepared[name]
            a,b,active,releases = native_predictions(source["base"],source["stream"],head,state,device)
            index = source["index"]
            truth = source["source"]["ref"][index]
            errors_world,errors_sensor = errors(body_up(a[index]),truth),errors(body_up(b[index]),truth)
            direct = float(np.sqrt(np.mean(np.square(errors_world,dtype=np.float64))))
            existing = baseline[axis][name]
            if abs(existing["rmse_deg"]-direct)>1e-4 or existing["frames"]!=len(index):
                raise ValueError(f"Stored world-hold metric does not match raw quaternion replay in {axis}/{name}")
            estimate = float(np.sqrt(np.mean(np.square(errors_sensor,dtype=np.float64))))
            applied = active[index]
            age = (source["time"][index]-source["time"][releases[np.maximum(applied-1,0)]])*1000.
            age_codes = np.searchsorted(AGE_EDGES,age,side="right")
            age_codes[applied==0] = len(AGE_LABELS)-1
            metrics = {"name":name,"fold":label,"frames":len(index),"baseline_deg":direct,"rmse_deg":estimate,
                       "world_historical_rmse_deg":existing["rmse_deg"],"world_replay_abs_delta_deg":abs(existing["rmse_deg"]-direct),
                       "score_by_update_age":slice_summary(AGE_LABELS,age_codes,errors_world,errors_sensor),
                       "score_by_speed":slice_summary(LABELS["speed"],source["speed"],errors_world,errors_sensor)}
            results.append(metrics)
            if prefix_test is None and axis=="device" and name=="R2-10":
                cut = int(len(a)*.61)
                original = source["source"]
                cut_base,cut_stream = inertial_states(original["imu"][:cut],float(original["dt"]),head["parameters"])
                cut_a,cut_b,_,_ = native_predictions(cut_base,cut_stream,head,state,device)
                prefix_test = {"name":name,"prefix_frames":cut,"world_max_quat_difference_deg":float(np.max(rotation_difference(a[:cut],cut_a))),
                               "sensor_max_quat_difference_deg":float(np.max(rotation_difference(b[:cut],cut_b))),
                               "input_inference_before_any_score_mask":True}
                if max(prefix_test["world_max_quat_difference_deg"],prefix_test["sensor_max_quat_difference_deg"])>0.001:
                    raise ValueError("End-to-end prefix differs before cutoff")
        all_axes.setdefault(axis,[]).extend(results)
        folds.append({"axis":axis,"fold":label,"training":train,"validation":held,"shared_head_fits":1,
                      "fit_seconds":score_start-start_fit,"score_seconds":perf_counter()-score_start})
        print(axis,label,aggregate(results)["macro_rmse_deg"],flush=True)
    metrics = {}
    for axis,rows in all_axes.items():
        if sorted(row["name"] for row in rows)!=sorted(record_names):
            raise ValueError("Original score coverage changed")
        new = aggregate(rows)
        old = aggregate([{**r,"rmse_deg":r["baseline_deg"]} for r in rows])
        metrics[axis] = {"world_hold":old,"sensor_hold":new,"sensor_minus_world_deg":new["macro_rmse_deg"]-old["macro_rmse_deg"],
                         "new_per_device":by_device(rows),
                         "world_per_device":by_device([{**r,"rmse_deg":r["baseline_deg"]} for r in rows])}
    passes=(metrics["joint_holdout"]["sensor_minus_world_deg"]<=-LIMITS["minimum_joint_improvement_deg"] and
            all(metrics[axis]["sensor_minus_world_deg"]<=LIMITS["maximum_allowed_axis_regression_deg"] for axis in ("device","motion")))
    reasons=[]
    if metrics["joint_holdout"]["sensor_minus_world_deg"]>-LIMITS["minimum_joint_improvement_deg"]:
        reasons.append("joint improvement did not reach the predefined threshold")
    reasons.extend(f"{axis} holdout regression" for axis in ("device","motion")
                   if metrics[axis]["sensor_minus_world_deg"]>LIMITS["maximum_allowed_axis_regression_deg"])
    write_json(destination,{"round_id":job["round_id"],"parent_round":config["parent_round"],
        "paired_checkpoint_seed":head["seed"],"synthetic_checks":synthetic(),"native_prefix_check":prefix_test,
        "metrics":metrics,"sequences":all_axes,"folds":folds,"real_head_fits":count,"elapsed_seconds":perf_counter()-started,
        "promotion_screen_passed":passes,"nonpromotion_reasons":reasons,"ranked_separately_from_model_configs":True,
        "maximum_world_reproduction_delta_deg":max(r["world_replay_abs_delta_deg"] for rows in all_axes.values() for r in rows)})


def queue():
    source = "P00022"
    r = round_record(source)
    config = {"kind":"sensor_hold_diagnostic","parent_round":source,"joint_round":"J00016",
              "head_sha256":r["config_sha256"],"seed":r["seed"]}
    audit = synthetic()
    write_json(STAGE,{"created_at":stamp(),"phase":"initial_one_seed_screen","queued_round":"S00001",
             "source_round":source,"seed":r["seed"],"checkpoint_reference_sha256":r["config_sha256"],
             "hypothesis":"Holding the shortest correction in sensor coordinates may better preserve the current tilt for a continuously turning sensor.",
             "two_output_maps":{"world":"q_world_update * q_native","sensor":"q_native * (inverse(q_native_at_update)*q_world_update*q_native_at_update)"},
             "update_age_ms_bins":list(AGE_EDGES),"quaternion_checks":audit,"promotion":LIMITS,
             "remaining_matched_seeds_if_pass":[1729,20261122],"development_only":True,
             "fold_coverage":"4 device, 4 program, 16 joint; 24 shared fits total"})
    print(schedule("S00001",config,"sensor-versus-world-hold-diagnostic",source,
                   "Measure sensor-frame and world-frame quaternion holds with the same fitted residual head and update timestamps"),flush=True)


if __name__ == "__main__":
    queue()
