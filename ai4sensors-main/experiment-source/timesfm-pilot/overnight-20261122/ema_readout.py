"""Deterministic inclination residuals from short and long time-decayed physical states."""

from time import perf_counter

import numpy as np

import learning
from common import BATCH, aggregate, by_device, fingerprint, names, read_json, record, stamp, write_json
from experiment import baseline_reference
from features import VERSION
from quaternion_audit import source_time
from stream_gru import planned_folds, prepare as prepare_physical


STAGE = BATCH/"ema-state-development-protocol.json"
HORIZONS = (0.3,2.0,8.0)
METHODS = ("continuous","same_weight_reset")


def innovations(physical, time_s, active_horizons, allowed=None, reset_each=False):
    n, channels = physical.shape
    result = np.zeros((n,len(HORIZONS),channels),dtype=np.float32)
    if not active_horizons or reset_each or not len(physical):
        return result.reshape(n,len(HORIZONS)*channels)
    if len(time_s)!=n or np.any(np.diff(time_s)<=0):
        raise ValueError("EMA inputs require strictly increasing original update timestamps")
    use = np.isin(HORIZONS,active_horizons)
    states = None
    raw = physical.astype(np.float64)
    transition = -np.expm1(-np.diff(time_s)[:,None,None]/np.asarray(HORIZONS)[None,:,None])
    for index,current in enumerate(raw):
        if allowed is not None and not allowed[index]:
            states = None
            continue
        if states is None:
            states = np.broadcast_to(current,(len(HORIZONS),channels)).copy()
        else:
            states += transition[index-1]*(current-states)
            result[index,use] = (states-current)[use]
    return result.reshape(n,len(HORIZONS)*channels)


def prepare(sensor, config, native_time, *, training=False):
    part = prepare_physical(sensor,config,for_training=training)
    matrix = part["matrix"]
    up = matrix[:,:3]
    allowed = part["state_input_ok"] if training else None
    time_s = native_time[part["ready"]]
    memory = innovations(matrix,time_s,config["horizons"],allowed)
    part.update(matrix=np.column_stack((matrix,memory)),instantaneous_matrix=matrix,up=up)
    if training:
        target = part["target"] - up
        part["target"] = target-(target*up).sum(axis=1,keepdims=True)*up
    return part


def evaluate(part, state, config, *, reset=False, delay_ms=0.):
    if reset:
        current=part["instantaneous_matrix"]
        matrix=np.zeros_like(part["matrix"])
        matrix[:,:current.shape[1]]=current
        part={**part,"matrix":matrix}
    return learning.evaluate(part,state,config,delay_ms)


def run(item):
    started = perf_counter()
    config = item["config"]
    frozen = read_json(STAGE)
    anchor = frozen["configurations"][item["round_id"]]
    if fingerprint(config)!=anchor["config_sha256"]:
        raise ValueError("Time-state experiment does not match its frozen configuration")
    records = names("dev")
    native = {name:learning.attach_native(name,config) for name in records}
    clock = {name:source_time(name,len(record(name)["imu"])) for name in records}
    baseline = baseline_reference()
    outputs,fold_evidence = {},[]
    for axis,label,train,held in planned_folds(records):
        prep_started = perf_counter()
        q = learning.installation_fit([native[name] for name in train],config)
        inputs = {name:prepare(learning.apply_install(native[name],q,include_reference=name in held),
                               config,clock[name],training=name in train) for name in train+held}
        fit_started = perf_counter()
        fitted = learning.fit([inputs[name] for name in train],config)
        scoring_started = perf_counter()
        rows = []
        for name in held:
            part = inputs[name]
            current = evaluate(part,fitted,config)
            reset = evaluate(part,fitted,config,reset=True)
            rows.append({"name":name,"fold":label,"frames":len(part["index"]),"baseline_deg":baseline[name],
                         "rmse_deg":current,"same_weight_reset_rmse_deg":reset})
        outputs.setdefault(axis,[]).extend(rows)
        timings = {"prepare_seconds":fit_started-prep_started,"fit_seconds":scoring_started-fit_started,
                   "score_seconds":perf_counter()-scoring_started}
        fold_evidence.append({"axis":axis,"fold":label,"training":train,"validation":held,"timing":timings,
                              "training_labels":{name:int(inputs[name]["train"].sum()) for name in train}})
        print(axis,label,aggregate(rows)["macro_rmse_deg"],timings,flush=True)
    info,reset_result = {},{}
    for axis, rows in outputs.items():
        if sorted(row["name"] for row in rows)!=sorted(records):
            raise ValueError(f"Lost original scored records in {axis}")
        info[axis] = {**aggregate(rows),"per_device":by_device(rows)}
        reset_result[axis] = aggregate([{**r,"rmse_deg":r["same_weight_reset_rmse_deg"]} for r in rows])
    info["selection_rmse_deg"] = (info["device"]["macro_rmse_deg"]+info["motion"]["macro_rmse_deg"])/2.
    return {"dev_metrics":info,"same_weight_reset":reset_result,"sequences":outputs,"folds":fold_evidence,
            "mask_version":VERSION,"fixed_horizons_seconds":list(HORIZONS),"active_horizons_seconds":config["horizons"],
            "deterministic_fit":True,"elapsed_seconds":perf_counter()-started,
            "filter_inputs":"current and previous IMU blocks, FullVQF state and original update clock only",
            "training_boundary":"each forbidden input window is omitted and resets all state; distinct from a missing supervision label"}


def queue():
    from optimizer_study import round_record, schedule

    p = round_record("P00022")
    base = {k:p["config"][k] for k in ("parameters","context","update_hz","installation","mount_shrink","gain")}
    base.update(kind="ema_readout",hidden=0,embedding=0,seed=None,alpha=.006,cap_deg=6.,acc_gate=0.,gyro_gate=0.)
    templates = (("D00001",[],"Same linear readout and feature dimensions, zero memory columns"),
                 ("D00002",list(HORIZONS),"Use all three fixed time constants"),
                 ("D00003",[.3],"Isolate short state at 0.3 s"),
                 ("D00004",[2.,8.],"Isolate two long time constants"))
    data = {}
    for identity,settings,label in templates:
        cfg = {**base,"horizons":settings}
        data[identity] = {"config":cfg,"config_sha256":fingerprint(cfg),"hypothesis":label}
    write_json(STAGE,{"created_at":stamp(),"configurations":data,"axis_controls":{"rotation":["I00004","J00002"],
                  "legacy_physical":["P00022","J00016"],"trainable_temporal_reference":["N00035","J00010"]},
                  "state_equation":"state += (1-exp(-observed_dt/tau))*(input-state), with zero innovation at reset",
                  "same_dimension_even_when_inactive":True,"explicit_training_boundary":"full original window mask and uninterrupted safe states",
                  "native_timestamp_field":"time_s at release index","development_only":True,"deterministic":True,
                  "promotion":{"minimum_joint_improvement_over_fitted_no_memory_deg":.015,
                               "minimum_joint_improvement_over_same_weight_reset_deg":.01,
                               "max_leave_device_degradation_vs_no_memory_deg":.015,
                               "max_joint_degradation_vs_legacy_head_deg":0.0}})
    for identity,line in data.items():
        saved = schedule(identity,line["config"],"deterministic-time-decayed-residual",p["round_id"],line["hypothesis"])
        if saved != identity:
            raise ValueError("A registered EMA arm resolved to an unexpected round ID")
    print(list(data),flush=True)


if __name__ == "__main__":
    queue()
