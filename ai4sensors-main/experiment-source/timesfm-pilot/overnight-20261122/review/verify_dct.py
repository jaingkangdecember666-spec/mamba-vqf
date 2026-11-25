"""Verify the causal-window DCT comparison on native development inputs."""

import hashlib
import sys
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.fft import dct

from audit_rounds import BATCH, HERE, check_fold, check_scores, read, write
from verify_baselines import AUDIT, SHICE, digest


PLAN = BATCH/"dct-development-protocol.json"
PLAN_SHA = "10eb15253883687abba84e6cf81c507e3c8b1d65ee0c56e1217a4a6872ba6f7c"
ARMS = ("ordered_dct","permuted_dct","physical_zero")
AXES = ("device","motion","joint_holdout")


def frozen_operators(plan):
    path = BATCH/plan["operator_file"]
    assert digest(path) == plan["operator_file_sha256"] and digest(PLAN) == PLAN_SHA
    with np.load(path,allow_pickle=False) as arrays:
        basis, perm = arrays["basis"],arrays["permutation"]
    assert list(plan["transform"]["non_dc_orders"]) == [1,2,3,4] and basis.shape == (4,64)
    reference = dct(np.eye(64),type=2,axis=0,norm="ortho")[1:5]
    np.testing.assert_allclose(basis,reference,rtol=0,atol=2e-14)
    np.testing.assert_array_equal(perm,np.random.Generator(np.random.PCG64(641004)).permutation(64))
    assert perm.tolist() == plan["fixed_permutation"]["indices"]
    assert hashlib.sha256(basis.astype("<f8").tobytes()).hexdigest() == plan["transform"]["basis_sha256"]
    assert hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest() == plan["fixed_permutation"]["sha256"]
    return basis,perm


def native_support(name, count):
    with np.load(BATCH.parent/"gru-tilt/data"/(name+".npz"),allow_pickle=False) as data:
        score = data["score"].astype(bool)
    with np.load(AUDIT/name/"screening_masks.npz",allow_pickle=False) as data:
        valid = data["context_ok"].astype(bool)
        np.testing.assert_array_equal(data["sample_index"],np.arange(len(score)))
    assert len(valid) == len(score) and int(score.sum()) == count
    released = np.arange(699,len(score),100,dtype="<i8")
    bad = np.r_[0,np.cumsum(~valid)]
    train = score[released] & (bad[released+1] == bad[released-639])
    return {"native_release_count":len(released),"frames":count,
            "score_frame_indices_sha256":hashlib.sha256(np.flatnonzero(score).astype("<i8").tobytes()).hexdigest(),
            "native_release_indices_sha256":hashlib.sha256(released.tobytes()).hexdigest(),
            "earliest_history_native_release":int(released[0])},int(train.sum()),int(score[released].sum())


def check_trial(row, result, frames):
    plan = read(PLAN)
    frozen_operators(plan)
    cfg = row["config"]
    assert cfg["protocol_sha256"] == digest(PLAN) and row["round_id"] == result["round_id"] == plan["round_id"]
    assert datetime.fromisoformat(plan["registered_at"]) <= datetime.fromisoformat(row["started_at"])
    assert cfg["context"] == 64 and cfg["update_hz"] == 10 and cfg["hidden"] == cfg["embedding"] == 0
    assert cfg["alpha"] == .006 and cfg["gain"] == .75 and cfg["parameters"] == {"tauAcc":.2} and cfg["installation"] == "all"
    assert row["dev_metrics"] is None and result["dev_metrics"] is None and not result["admitted_for_r5"]
    for p,h in plan["anchor_result_sha256"].items():
        assert digest(BATCH/p) == h
    head,joint = [read(BATCH/"rounds"/rid/"result.json") for rid in ("P00022","J00016")]
    controls = {**head["sequences"],**joint["sequences"]}
    reference = {(f["axis"],f["fold"]):f for f in head["folds"]+joint["folds"]}
    support = {n:native_support(n,c) for n,c in frames.items()}
    assert {n:v[0] for n,v in support.items()} == result["scoring_evidence"]
    assert result["real_window_count"] == sum(v[0]["native_release_count"] for v in support.values())
    folds = {(f["axis"],f["fold"]):f for f in result["folds"]}
    assert set(folds) == set(reference) and len(result["folds"]) == len(folds) == 24
    assert result["fit_count"] == 72 and result["mount_fit_count"] == 24 and result["failed_fits"] == 0
    for key,fold in folds.items():
        check_fold(fold,key[0],set(frames))
        a,b = (np.asarray(f["installation_quaternion_wxyz"]) for f in (fold,reference[key]))
        assert min(np.linalg.norm(a-b),np.linalg.norm(a+b)) < 1e-6
        assert set(fold["fit_log"]) == set(ARMS) and all(v["fits"] == 1 and v["readout_inputs"] == 58 for v in fold["fit_log"].values())
        assert fold["eligible_training_updates"] == {n:support[n][1] for n in fold["training"]}
        assert set(fold["training_qualifications"]) == set(fold["training"])
        for n,meta in fold["training_qualifications"].items():
            assert meta["support_blocks_100hz"] == 64 and meta["min_native_input_rows"] == meta["max_native_input_rows"] == 640
            assert (meta["training_labels_eligible"],meta["update_labels_available"]) == support[n][1:]
    scores = {}
    for axis in AXES:
        rows = result["sequences"][axis]
        assert {r["name"] for r in rows} == set(frames) and len(rows) == len(frames)
        original = {r["name"]:r["rmse_deg"] for r in controls[axis]}
        for r in rows:
            n, values = r["name"],r["rmse_deg"]
            assert n in folds[axis,r["fold"]]["validation"] and original[n] == values["legacy_head"]
            assert r["score_frame_indices_sha256"] == support[n][0]["score_frame_indices_sha256"]
            for other in ("permuted_dct","physical_zero","legacy_head"):
                assert isclose(r["ordered_delta_deg"][other],values["ordered_dct"]-values[other],abs_tol=1e-10)
        scores[axis] = {}
        for arm in (*ARMS,"legacy_head"):
            projected = [{**r,"rmse_deg":r["rmse_deg"][arm]} for r in rows]
            check_scores(projected,result["axis_metrics"][axis][arm],frames)
            scores[axis][arm] = fmean(r["rmse_deg"] for r in projected)
    limit = plan["criteria"]
    assert limit["joint_minimum_improvement_deg"] == limit["single_axis_legacy_maximum_regression_deg"] == .02
    criteria = {"joint_beats_"+arm:scores["joint_holdout"]["ordered_dct"] <= scores["joint_holdout"][arm]-.02
                for arm in ("permuted_dct","physical_zero","legacy_head")}
    criteria.update({axis+"_acceptable":scores[axis]["ordered_dct"] <= scores[axis]["legacy_head"]+.02 for axis in ("device","motion")})
    assert criteria == result["promotion_checks"] and all(criteria.values()) == result["worth_expanding"]
    summary = BATCH/"dct-development-summary.json"
    if summary.is_file():
        filed = read(summary)
        assert filed["round"] == row["round_id"] and filed["protocol_sha256"] == PLAN_SHA
        assert filed["result_sha256"] == digest(BATCH/row["result"]) and filed["actual_start"]["python_pid"] == result["python_pid"]
        assert filed["axis_metrics"] == result["axis_metrics"] and filed["scoring_evidence"] == result["scoring_evidence"]
        assert (filed["fits"],filed["shared_mount_fits"],filed["failed_fits"],filed["r5_admissions"]) == (72,24,0,0)
        assert filed["promotion_checks"] == criteria and filed["worth_expanding"] == all(criteria.values())
    output = {"id":row["round_id"],"scores":scores,"criteria":criteria,"passed":all(criteria.values()),"fits":result["fit_count"],
            "rotations":len(folds),"records":len(frames),"scored_frames":sum(frames.values()),"windows":result["real_window_count"],
            "operator_sha256":plan["operator_file_sha256"]}
    receipt = HERE/"dct_native_feature_replay.json"
    if receipt.is_file():
        saved = read(receipt)
        assert saved["protocol_sha256"] == PLAN_SHA and set(saved["records"]) == {"R2-10","R4-2"}
        for name,e in saved["records"].items():
            for k,v in support[name][0].items():
                assert e[k] == v
            path = Path(e["source"])
            assert path == SHICE/(name.split("-")[0]+"_aligned_formal")/(name+".csv") and digest(path) == e["source_sha256"]
            assert set(e["maximum_coefficient_error"]) == set(ARMS[:2]) and all(0 <= x <= 2e-6 for x in e["maximum_coefficient_error"].values())
        output["raw_windows"] = sum(e["native_release_count"] for e in saved["records"].values())
    return output


def replay_raw():
    sys.path.insert(0,str(BATCH))
    from dct_readout import collect_raw
    from verify_final import raw_shice_samples

    plan = read(PLAN)
    basis,perm = frozen_operators(plan)
    cfg = read(BATCH/"rounds"/plan["round_id"]/"round.json")["config"]
    scale = np.array([9.80665]*3+[1]*3,dtype=np.float32)
    results = {}
    for name in ("R2-10","R4-2"):
        raw = raw_shice_samples(name)
        n = len(raw["imu"])//10
        ten = raw["imu"][:n*10].reshape(n,10,6).mean(axis=1,dtype=np.float64).astype(np.float32)
        releases = np.arange(69,n,10)
        indices = releases[:,None]+np.arange(-63,1)
        history = ten[indices].astype(np.float64)/scale
        production,attached,evidence = collect_raw([name],cfg,basis,perm)
        np.testing.assert_array_equal(production[name]["ready"][releases],releases*10+9)
        delta = {}
        for arm,window in (("ordered_dct",history),("permuted_dct",history[:,perm])):
            independent = dct(window,type=2,norm="ortho",axis=1)[:,1:5].reshape(len(releases),24)
            observed = attached[name][arm]
            np.testing.assert_allclose(observed,independent,rtol=1e-6,atol=1e-6)
            delta[arm] = float(np.max(np.abs(observed-independent)))
        assert not np.any(attached[name]["physical_zero"])
        expected = native_support(name,int(raw["score"].sum()))[0]
        assert evidence[name] == expected
        results[name] = {**expected,"source":str(raw["source"]),"source_sha256":digest(raw["source"]),"maximum_coefficient_error":delta}
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"protocol_sha256":digest(PLAN),"records":results}


if __name__ == "__main__":
    receipt = replay_raw()
    write(HERE/"dct_native_feature_replay.json",receipt)
    for n,row in receipt["records"].items():
        print(n,row["native_release_count"],row["maximum_coefficient_error"])
