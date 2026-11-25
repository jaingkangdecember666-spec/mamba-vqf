"""Audit deterministic time-decayed readouts on the SHICE development folds."""

import hashlib
import math
import sys
from datetime import datetime

import numpy as np

from audit_rounds import BATCH, HERE, check_scores, fingerprint, read, read_bytes_shared, write


STAGE = BATCH / "ema-state-development-protocol.json"
PROTOCOL_SHA256 = "1ad80c21b0c6ebbb9b461df637c28bb12e995bfe9e27650e7da14746079838fb"


def check_trial(row,result,frames):
    assert hashlib.sha256(read_bytes_shared(STAGE)).hexdigest() == PROTOCOL_SHA256
    plan = read(STAGE)
    cfg = row["config"]
    key = row["round_id"]
    assert plan["configurations"][key]["config"] == cfg
    assert plan["configurations"][key]["config_sha256"] == fingerprint(cfg)
    assert datetime.fromisoformat(plan["created_at"]) <= datetime.fromisoformat(row["started_at"])
    assert result["active_horizons_seconds"] == cfg["horizons"] and result["fixed_horizons_seconds"] == [.3,2.,8.]
    assert result["deterministic_fit"] and row["seed"] is None and cfg["hidden"] == cfg["embedding"] == 0
    expected = {r["name"]:r["eligible_updates"] for r in read(HERE / "recurrent_context_audit.json")["records"]}
    assert len(result["folds"]) == 24
    for fold in result["folds"]:
        assert fold["training_labels"] == {n:expected[n] for n in fold["training"]}
    for axis, samples in result["sequences"].items():
        reset = [{**r,"rmse_deg":r["same_weight_reset_rmse_deg"]} for r in samples]
        check_scores(reset,result["same_weight_reset"][axis],frames)
        if not cfg["horizons"]:
            assert all(r["rmse_deg"] == r["same_weight_reset_rmse_deg"] for r in samples)
    base = read(BATCH / "rounds/D00001/result.json")["dev_metrics"]
    metrics = result["dev_metrics"]
    joint = metrics["joint_holdout"]["macro_rmse_deg"]
    reset = result["same_weight_reset"]["joint_holdout"]["macro_rmse_deg"]
    reference = read(BATCH / "rounds/J00016/result.json")["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]
    rule = plan["promotion"]
    benefit = base["joint_holdout"]["macro_rmse_deg"]-joint
    passed = (bool(cfg["horizons"]) and benefit >= rule["minimum_joint_improvement_over_fitted_no_memory_deg"]
              and reset-joint >= rule["minimum_joint_improvement_over_same_weight_reset_deg"]
              and metrics["device"]["macro_rmse_deg"]-base["device"]["macro_rmse_deg"] <= rule["max_leave_device_degradation_vs_no_memory_deg"]
              and joint-reference <= rule["max_joint_degradation_vs_legacy_head_deg"])
    summary_path = BATCH / "ema-state-development-summary.json"
    if summary_path.is_file():
        published = read(summary_path)
        assert published["protocol"] == plan
        shown = next((r for r in published["results"] if r["round_id"] == key and r["status"] == "completed"),None)
        if shown:
            assert passed == shown["promoted"] == (key in published["r5_qualified_rounds"])
            controls = {"rotation":"fixed_installation_control_deg","legacy_physical":"physical_head_control_deg",
                        "trainable_temporal_reference":"tcn_control_deg"}
            for axis, numbers in shown["axes"].items():
                assert math.isclose(numbers["continuous_deg"],metrics[axis]["macro_rmse_deg"],abs_tol=1e-8)
                assert math.isclose(numbers["same_fitted_weights_reset_deg"],result["same_weight_reset"][axis]["macro_rmse_deg"],abs_tol=1e-8)
                assert math.isclose(numbers["fitted_memoryless_deg"],base[axis]["macro_rmse_deg"],abs_tol=1e-8)
                for control,column in controls.items():
                    grouped, simultaneous = plan["axis_controls"][control]
                    origin = simultaneous if axis == "joint_holdout" else grouped
                    value = read(BATCH / "rounds" / origin / "round.json")["dev_metrics"][axis]["macro_rmse_deg"]
                    assert math.isclose(numbers[column],value,abs_tol=1e-8)
    return {"id":key, "horizons_seconds":cfg["horizons"], "joint_deg":joint,"same_weight_reset_joint_deg":reset,
            "memoryless_minus_current_joint_deg":benefit, "device_deg":metrics["device"]["macro_rmse_deg"],
            "motion_deg":metrics["motion"]["macro_rmse_deg"],"r5_qualified":passed}


def replay_states():
    from verify_final import raw_shice_samples
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from ema_readout import innovations,prepare
    from features import fetch_record

    cfg = read(STAGE)["configurations"]["D00002"]["config"]
    name = "R1-10"
    raw = raw_shice_samples(name)
    cached = fetch_record(name,cfg["parameters"])
    train = prepare(cached,cfg,raw["time_s"],training=True)
    online = prepare(cached,cfg,raw["time_s"])
    x = train["instantaneous_matrix"]
    ends = train["ready"]+1
    cumulative_bad = np.r_[0,np.cumsum(~raw["context"],dtype=np.int64)]
    safe = (ends >= 640) & (cumulative_bad[ends] == cumulative_bad[(ends-640).clip(min=0)])
    np.testing.assert_array_equal(train["state_input_ok"],safe)
    ticks = raw["time_s"][ends-1]
    horizons = cfg["horizons"]
    starts = np.flatnonzero(safe & ~np.r_[False,safe[:-1]])
    chosen = np.unique(np.r_[np.linspace(0,len(x)-1,13,dtype=int), starts,(starts+1).clip(max=len(x)-1)])
    largest = 0.
    checks = 0
    for sample in chosen:
        for training,part in ((True,train),(False,online)):
            vector = part["matrix"][sample,x.shape[1]:].reshape(len(horizons),-1)
            if training and not safe[sample]:
                assert not vector.any()
                continue
            first = starts[starts<=sample][-1] if training else 0
            t = ticks[first:sample+1]
            values = x[first:sample+1].astype(np.float64)
            for tau,actual in zip(horizons,vector,strict=True):
                weights = np.r_[np.exp((t[0]-t[-1])/tau),
                                -np.expm1(-np.diff(t)/tau)*np.exp((t[1:]-t[-1])/tau)]
                assert math.isclose(weights.sum(),1,abs_tol=1e-10)
                largest=max(largest,float(np.max(np.abs(weights@values-values[-1]-actual))))
                checks+=1
    assert largest < 2e-5
    changed = x.copy()
    changed[~safe] += 100
    np.testing.assert_array_equal(innovations(changed,ticks,horizons,safe)[safe],train["matrix"][safe,x.shape[1]:])
    midpoint = len(x)//2
    np.testing.assert_array_equal(innovations(x[:midpoint],ticks[:midpoint],horizons),online["matrix"][:midpoint,x.shape[1]:])
    assert not innovations(x,ticks,horizons,reset_each=True).any()
    full = online["matrix"][:,x.shape[1]:].reshape(len(x),len(horizons),-1)
    for entry in read(STAGE)["configurations"].values():
        active = entry["config"]["horizons"]
        subset = innovations(x,ticks,active).reshape(full.shape)
        for i,tau in enumerate(horizons):
            if tau in active:
                np.testing.assert_array_equal(subset[:,i],full[:,i])
            else:
                assert not subset[:,i].any()
    return {"record":name,"release_timestamps":len(ticks),"forbidden_updates_perturbed":int((~safe).sum()),
            "checked_prefix_updates":midpoint,"weighted_history_comparisons":checks,"max_abs_feature_difference":largest,
            "features":x.shape[1],"protocol_sha256":PROTOCOL_SHA256}


if __name__ == "__main__":
    receipt = replay_states()
    write(HERE / "ema_state_replay.json",receipt)
    print(receipt)
