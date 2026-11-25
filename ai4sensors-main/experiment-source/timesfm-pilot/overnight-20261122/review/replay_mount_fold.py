"""Reproduce the middle-seed R2/M2 development fold through the formal fit and inference paths."""

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, fingerprint, read, write
from verify_baselines import SHICE, digest
from verify_final import raw_shice_samples
from verify_mount_pair import PLAN, PLAN_SHA, REFERENCE


def replay():
    sys.path.insert(0,str(BATCH))
    from features import cache_record
    from final_eval import fit_candidate, score_candidate
    from quaternion_audit import PROTOCOL, native_replay, rotation_difference

    assert digest(PLAN) == PLAN_SHA
    plan = read(PLAN)
    trial, = [p for p in plan["pairs"] if p["seed"] == 20261123]
    reference = read(REFERENCE)
    names = set(plan["records"])
    fold_name = "R2/M2"
    expected = reference["folds"]["joint_holdout:"+fold_name]
    members, records, hashes = {}, {}, {}
    for side in ("without_mount","with_mount"):
        member = trial[side]
        rid = member["joint_round"]
        path = BATCH/"rounds"/rid/"result.json"
        hashes[str(path.relative_to(BATCH))] = digest(path)
        source = read(path)
        cfg = member["config"]
        assert source["evaluated_model_config"] == cfg and source["evaluated_model_sha256"] == fingerprint(cfg)
        assert cfg["parameters"] == {} and cfg["embedding"] == 0
        fold, = [f for f in source["folds"] if f["fold"] == fold_name]
        check_fold(fold,"joint_holdout",names)
        assert sorted(fold["training"]) == expected["training"] and sorted(fold["validation"]) == expected["validation"]
        state = fit_candidate(cfg,fold["training"])
        state_digest = {k:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k,v in state.items()}
        if side == "with_mount":
            trained = Rotation.from_quat(state["installation_quaternion_wxyz"],scalar_first=True)
            independent = Rotation.from_quat(expected["wxyz"],scalar_first=True)
            assert np.rad2deg((trained.inv()*independent).magnitude()) < 2e-5
        members[side] = cfg,state,{r["name"]:r for r in source["sequences"] if r["fold"] == fold_name}
        records[side] = {"id":rid,"config_sha256":fingerprint(cfg),"state_arrays_sha256":state_digest,"scores":{}}
    assert expected["validation"] == sorted(["R2-2","R2-6","R2-10"])
    source_files = {}
    for name in expected["validation"]:
        raw = raw_shice_samples(name)
        imu,dt,mask = raw["imu"],raw["dt"],raw["score"]
        native_index = np.flatnonzero(mask).astype("<i8",copy=False)
        actual = hashlib.sha256(native_index.tobytes()).hexdigest()
        assert actual == reference["source_records"][name]["score_index_sha256"]
        source_files[name] = {"path":str(raw["source"]),"sha256":digest(raw["source"]),"score_frames":len(native_index),
                              "score_index_sha256":actual}
        cache = cache_record(imu,dt,mask,raw["context"],raw["reference"],{},True,keep_scored_quaternion=True)
        np.testing.assert_array_equal(cache["index"],native_index)
        for side,(cfg,state,observed) in members.items():
            official = observed[name]
            assert official["frames"] == len(native_index)
            formal = float(score_candidate(cache,cfg,state))
            quat,_ = native_replay(imu,dt,cfg,state,None)
            assert quat.shape == (len(imu),4) and np.isfinite(quat).all()
            norm_error = float(np.max(np.abs(np.linalg.norm(quat,axis=1)-1)))
            predicted = Rotation.from_quat(quat[mask],scalar_first=True).inv().apply([0,0,1])
            truth = raw["reference"][mask]
            angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted,truth),axis=1),
                                          np.einsum("ij,ij->i",predicted,truth)))
            native = float(np.sqrt(np.mean(angle**2)))
            differences = [abs(s-official["rmse_deg"]) for s in (formal,native)]
            assert max(differences) <= PROTOCOL["max_scored_rmse_difference_deg"] and norm_error <= PROTOCOL["max_norm_error"]
            # Both outputs see identical IMU packets through this point. Reference labels are never passed to native_replay.
            length = int(0.59*len(imu))
            prefix,_ = native_replay(imu[:length],dt,cfg,state,None)
            prefix_error = float(np.max(rotation_difference(quat[:length],prefix)))
            assert prefix_error <= PROTOCOL["max_quaternion_prefix_rotation_deg"]
            records[side]["scores"][name] = {"original_deg":official["rmse_deg"],"formal_deg":formal,"native_deg":native,
                                             "max_difference_deg":max(differences),"norm_max_error":norm_error,
                                             "prefix_frames":length,"prefix_max_difference_deg":prefix_error}
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":PLAN_SHA,"fold":fold_name,"seed":trial["seed"],
            "training":expected["training"],"validation":expected["validation"],"source_results_sha256":hashes,
            "source_files":source_files,"members":records,"audit_refits":len(members)}


def check_saved(result):
    assert result["plan_sha256"] == digest(PLAN) == PLAN_SHA and result["audit_refits"] == 2
    assert result["seed"] == 20261123 and result["fold"] == "R2/M2"
    reference = read(REFERENCE)
    expected_fold = reference["folds"]["joint_holdout:R2/M2"]
    assert result["training"] == expected_fold["training"] and result["validation"] == expected_fold["validation"]
    names = set(expected_fold["validation"])
    assert set(result["source_files"]) == names and set(result["members"]) == {"without_mount","with_mount"}
    for n,source in result["source_files"].items():
        original = reference["source_records"][n]
        assert source["score_frames"] == original["frames"] and source["score_index_sha256"] == original["score_index_sha256"]
        path = Path(source["path"])
        assert path == SHICE/"R2_aligned_formal"/(n+".csv") and digest(path) == source["sha256"]
    differences = []
    for side,rid in (("without_mount","J00039"),("with_mount","M00112")):
        member = result["members"][side]
        path = BATCH/"rounds"/rid/"result.json"
        assert member["id"] == rid and digest(path) == result["source_results_sha256"][str(path.relative_to(BATCH))]
        measured = read(path)
        assert measured["evaluated_model_sha256"] == member["config_sha256"]
        scores = {r["name"]:r["rmse_deg"] for r in measured["sequences"] if r["fold"] == result["fold"]}
        assert set(member["scores"]) == names == set(scores)
        for n,row in member["scores"].items():
            assert row["original_deg"] == scores[n]
            assert all(np.isfinite(row[k]) and 0 <= row[k] <= 180 for k in ("formal_deg","native_deg"))
            diff = max(abs(row[k]-scores[n]) for k in ("formal_deg","native_deg"))
            assert diff == row["max_difference_deg"] and diff <= 1e-4 and 0 <= row["norm_max_error"] <= 1e-6
            assert 0 <= row["prefix_max_difference_deg"] <= 0.001 and row["prefix_frames"] > 0
            differences.append(diff)
    return {"compared":len(differences),"records":len(names),"frames":sum(v["score_frames"] for v in result["source_files"].values()),
            "max_difference_deg":max(differences)}


if __name__ == "__main__":
    result = replay()
    check_saved(result)
    write(HERE/"ridge_50hz_mount_fold_replay.json",result)
    for side,m in result["members"].items():
        print(side,m["id"],{n:v["max_difference_deg"] for n,v in m["scores"].items()})
