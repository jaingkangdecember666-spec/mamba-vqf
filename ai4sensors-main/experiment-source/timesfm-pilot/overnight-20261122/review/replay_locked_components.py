"""Measure the fast filter, fixed mounting rotation and locked TCN separately on the same scoring frames."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from audit_rounds import BATCH, HERE, fingerprint, read, write
from verify_baselines import VERTICAL, digest
from verify_external import raw_records
from verify_final import raw_shice_samples, score_mask_evidence
from verify_tum import align, read_series


ARMS = ("default","fast","mounted","locked")


def causal_fill(signal):
    valid = np.isfinite(signal).all(axis=1)
    last = np.maximum.accumulate(np.where(valid,np.arange(len(signal)),-1))
    held = signal[last.clip(min=0)].astype(np.float32)
    held[last<0] = 0.
    return held,valid


def inputs(suite, expected):
    if suite=="R6":
        for name in expected:
            raw = raw_shice_samples(name)
            yield name,raw["imu"],raw["dt"],raw["score"],raw["reference"][raw["score"]]
    elif suite in {"BROAD","Sassari"}:
        for name,_,imu,quat,eligible,dt in raw_records(suite,expected):
            held,valid = causal_fill(imu)
            mask = eligible & valid & np.isfinite(quat).all(axis=1) & np.any(quat!=0,axis=1)
            truth = Rotation.from_quat(quat[mask],scalar_first=True).inv().apply(VERTICAL)
            yield name,held,dt,mask,truth
    else:
        assert suite=="TUM-VI"
        protocol = read(BATCH/"tum-vi-protocol.json")
        for row in read(BATCH/"tum-vi-independent.json")["results"]:
            imu_ns,measurements = read_series(Path(row["imu_path"]),range(1,7))
            pose_ns,q = read_series(Path(row["mocap_path"]),range(4,8))
            held,valid = causal_fill(np.column_stack((measurements[:,3:],measurements[:,:3])))
            mask,truth = align(imu_ns,valid,pose_ns,q,protocol)
            yield row["name"],held,float(np.median(np.diff(imu_ns)))*1e-9,mask,truth[mask]


def angular_error(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum("ij,ij->i",a,b)))


def compare_native(signal,dt,mask,target,params,mount):
    gyro,acc = [np.ascontiguousarray(a,dtype=np.float64) for a in (signal[:,3:],signal[:,:3])]
    rms = {}
    for arm,setting in (("default",{}),("fast",params)):
        q = VQF(dt,**setting).updateBatch(gyro,acc)["quat6D"]
        orientation = Rotation.from_quat(q[mask],scalar_first=True)
        up = orientation.inv().apply(VERTICAL)
        rms[arm] = float(np.sqrt(np.mean(angular_error(up,target)**2)))
    mounted = mount.inv().apply(up)
    rms["mounted"] = float(np.sqrt(np.mean(angular_error(mounted,target)**2)))
    return rms,float(np.max(angular_error(up,mounted)))


def check_saved(receipt,audit):
    lock = read(BATCH/"selection-lock.json")
    assert receipt["lock_sha256"]==audit["lock_sha256"]==lock["lock_sha256"] and receipt["weights_sha256"]==audit["weights_sha256"]==digest(BATCH/lock["weights"])
    assert receipt["round"]==lock["selected"]["round_id"]=="N00035" and lock["selected"]["config"]["kind"]=="tcn"
    assert fingerprint({k:v for k,v in lock.items() if k!="lock_sha256"})==lock["lock_sha256"]
    cfg = lock["selected"]["config"]
    assert cfg["parameters"]==receipt["filter_parameters"] and cfg["gain"]==receipt["gain"] and cfg["cap_deg"]==receipt["cap_deg"]
    assert receipt["residual_bound_deg"]==float(np.rad2deg(np.arctan(cfg["gain"]*np.tan(np.deg2rad(cfg["cap_deg"])))))
    with np.load(BATCH/lock["weights"],allow_pickle=False) as state:
        angle = float(np.rad2deg(Rotation.from_quat(state["installation_quaternion_wxyz"],scalar_first=True).magnitude()))
    assert angle==receipt["mount_angle_deg"] and set(receipt["datasets"])=={"R6","BROAD","Sassari","TUM-VI"}
    official = read(BATCH/"rounds"/audit["final_id"]/"result.json")
    assert digest(BATCH/"rounds"/audit["final_id"]/"result.json")==receipt["final_result_sha256"]==audit["final_result_sha256"]
    for suite,group in receipt["datasets"].items():
        if suite=="TUM-VI":
            tum = audit["optional_tum_vi"]
            source = {r["name"]:(r["frames"],r["baseline_deg"],r["rmse_deg"]) for r in tum["per_room"]}
            signatures = {r["name"]:r["mask_sha256"] for r in tum["raw_replays"]}
            hash_key = "mask_sha256"
        else:
            rows = official["datasets"][suite]["sequences"]
            source = {r["name"]:(r["frames"],r["rmse_deg"]["full_vqf"],r["rmse_deg"]["locked"]) for r in rows}
            signatures = {r["name"]:r["score_frame_indices_sha256"] for r in rows}
            hash_key = "score_frame_indices_sha256"
        assert len(group["records"])==len(source) and set(group["records"])==set(source)
        for name,r in group["records"].items():
            n,baseline,locked = source[name]
            assert r["frames"]==n and r[hash_key]==signatures[name] and 0 <= r["max_fast_to_mounted_angle_deg"] <= angle+1e-6
            scores = r["rmse_deg"]
            assert set(scores)==set(ARMS) and all(np.isfinite(v) and 0<=v<=180 for v in scores.values())
            assert abs(scores["default"]-baseline)<2e-4 and scores["locked"]==locked
            assert abs(scores["fast"]-scores["mounted"])<=angle+1e-6
            assert abs(scores["mounted"]-scores["locked"])<=receipt["residual_bound_deg"]+2e-4
        for key in ARMS:
            assert group["means"][key]==fmean(r["rmse_deg"][key] for r in group["records"].values())
    return receipt


def main():
    lock = read(BATCH/"selection-lock.json")
    cfg = lock["selected"]["config"]
    assert lock["selected"]["round_id"]=="N00035" and cfg["installation"]=="all" and cfg["parameters"]=={"tauAcc":.2}
    assert cfg["gain"]>0 and cfg["acc_gate"]==cfg["gyro_gate"]==0
    audit = read(HERE/"final_audit.json")
    final_path = BATCH/"rounds"/audit["final_id"]/"result.json"
    formal = read(final_path)
    started = datetime.now(timezone.utc)
    with np.load(BATCH/lock["weights"],allow_pickle=False) as weights:
        mount = Rotation.from_quat(weights["installation_quaternion_wxyz"],scalar_first=True)
    datasets = {}
    for suite in ("R6","BROAD","Sassari","TUM-VI"):
        if suite=="TUM-VI":
            expected = {r["name"]:r["rmse_deg"] for r in audit["optional_tum_vi"]["per_room"]}
        else:
            expected = {r["name"]:r["rmse_deg"]["locked"] for r in formal["datasets"][suite]["sequences"]}
        records = {}
        for name,signal,dt,mask,truth in inputs(suite,expected):
            scores,rotation_gap = compare_native(signal,dt,mask,truth,cfg["parameters"],mount)
            scores["locked"] = expected[name]
            records[name] = {**score_mask_evidence(mask),"mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest(),
                             "rmse_deg":scores,"max_fast_to_mounted_angle_deg":rotation_gap}
        datasets[suite] = {"records":records,"means":{k:fmean(r["rmse_deg"][k] for r in records.values()) for k in ARMS}}
    receipt = {"started":started.isoformat(),"ended":datetime.now(timezone.utc).isoformat(),"round":lock["selected"]["round_id"],
               "lock_sha256":lock["lock_sha256"],"weights_sha256":lock["weights_sha256"],"final_result_sha256":digest(final_path),
               "filter_parameters":cfg["parameters"],"gain":cfg["gain"],"cap_deg":cfg["cap_deg"],
               "mount_angle_deg":float(np.rad2deg(mount.magnitude())),
               "residual_bound_deg":float(np.rad2deg(np.arctan(cfg["gain"]*np.tan(np.deg2rad(cfg["cap_deg"]))))),
               "datasets":datasets}
    check_saved(receipt,audit)
    write(HERE/"locked_component_replay.json",receipt)
    print({k:v["means"] for k,v in datasets.items()},flush=True)


if __name__=="__main__":
    main()
