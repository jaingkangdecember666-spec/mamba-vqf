"""Rebuild the locked room benchmarks from integer-native IMU and motion-capture CSVs."""

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from audit_rounds import BATCH, check_scores, fingerprint


PROTOCOL_SHA256 = "3d3ab0c6a11c6189c3b41d3571930f488dc89c94fda288258fd8c89ee8b030ce"


def coverage_evidence(snapshot, protocol):
    assert snapshot["protocol"] == protocol and fingerprint(protocol) == PROTOCOL_SHA256
    expected = {f"room{n}" for n in protocol["rooms"]}
    rows, failures = snapshot["results"], snapshot["failures"]
    names = {r["name"] for r in rows}
    missing = {r["name"]:r["status"] for r in failures}
    assert len(rows) == len(names) and len(failures) == len(missing)
    assert names.isdisjoint(missing) and (names | set(missing)) <= expected
    assert all(r["status"] == "not_started_before_cutoff" or (r["status"] == "failed" and r.get("error")) for r in failures)
    states = {n:"success" if n in names else missing.get(n,"pending") for n in expected}
    categories = ("success","failed","not_started_before_cutoff","pending")
    coverage = {"planned":len(expected),**{s:sum(v == s for v in states.values()) for s in categories}}
    if "room_manifest" in snapshot:
        manifest = snapshot["room_manifest"]
        assert len(manifest) == len(expected) and {r["name"]:r["status"] for r in manifest} == states
        assert snapshot["coverage"] == coverage and snapshot["phase"] in {"progress","final"}
        assert snapshot["phase"] != "final" or coverage["pending"] == 0
    frames = {r["name"]:r["frames"] for r in rows}
    assert all(0 < r["frames"] <= r["imu_frames"] for r in rows)
    for key, metric in (("locked","rmse_deg"),("pure_full_vqf","baseline_deg")):
        if rows:
            scalar = [{"name":r["name"],"frames":r["frames"],"rmse_deg":r[metric]} for r in rows]
            check_scores(scalar,snapshot[key],frames)
        else:
            assert snapshot[key] is None
    return {"counts":coverage,"states":states}


def read_series(path, columns):
    fields = [("time_ns",np.int64),("measurements",np.float64,len(columns))]
    data = np.loadtxt(path,dtype=fields,delimiter=",",usecols=(0,*columns),skiprows=1,
                      encoding="utf-8-sig",ndmin=1)
    times = data["time_ns"]
    assert len(times) >= 2 and np.all(np.diff(times) > 0), path
    return times, data["measurements"]


def align(imu_ns, observed, pose_ns, quaternion, protocol):
    valid = np.isfinite(quaternion).all(axis=1) & (np.linalg.norm(quaternion,axis=1) > 0)
    duration = np.diff(pose_ns)
    pairs = np.flatnonzero(valid[1:] & valid[:-1] & (duration <= protocol["max_consecutive_mocap_gap_ns"]))
    # Place valid mocap intervals onto the IMU grid. Consecutive intervals meet at their left endpoint.
    ends = np.searchsorted(imu_ns,pose_ns,side="left")
    coverage_delta = np.zeros(len(imu_ns)+1,dtype=np.int32)
    np.add.at(coverage_delta,ends[pairs],1)
    np.add.at(coverage_delta,ends[pairs+1],-1)
    covered = np.cumsum(coverage_delta[:-1])
    assert np.all((covered == 0) | (covered == 1))
    mask = (covered == 1) & observed & (imu_ns-imu_ns[0] >= protocol["initial_exclusion_ns"])
    truth = np.zeros((len(imu_ns),3))
    if mask.any():
        at = imu_ns[mask]
        left = np.searchsorted(pose_ns,at,side="right") - 1
        assert np.isin(left,pairs).all()
        start = Rotation.from_quat(quaternion[left],scalar_first=True)
        stop = Rotation.from_quat(quaternion[left+1],scalar_first=True)
        fraction = (at-pose_ns[left]) / duration[left]
        partial = Rotation.from_rotvec((start.inv()*stop).as_rotvec()*fraction[:,None])
        truth[mask] = (start*partial).inv().apply(protocol["world_up"])
    return mask, truth


def quiet_gravity_check(readings, mask, truth, dt_ns):
    """Compare specific force with mocap up in complete, low-dynamic native blocks."""
    width = max(1,round(1e9/dt_ns))
    blocks = []
    for start in range(0,len(mask)-width+1,width):
        win = slice(start,start+width)
        if not mask[win].all():
            continue
        gyro, acc = readings[win,:3], readings[win,3:]
        if (np.linalg.norm(gyro,axis=1).max() > 0.05
                or np.abs(np.linalg.norm(acc,axis=1)-9.80665).max() > 0.20
                or np.linalg.norm(acc-acc.mean(axis=0),axis=1).max() > 0.12):
            continue
        up = truth[win]
        # atan2 of cross and dot cancels the norm of the accelerometer vector.
        angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(acc,up),axis=1),
                                        np.einsum("ij,ij->i",acc,up)))
        blocks.append({"start_index":start,"median_deg":float(np.median(angles))})
    medians = [b["median_deg"] for b in blocks]
    return {"samples_per_block":width,"qualifying_blocks":len(blocks),"qualifying_frames":len(blocks)*width,
            "median_deg":float(np.median(medians)) if medians else None,"blocks":blocks}


def replay_rooms(rows, protocol, config, weights):
    from verify_baselines import DATA, digest

    assert fingerprint(protocol) == PROTOCOL_SHA256
    if not rows:
        return []
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from final_eval import cache_external_candidate, embed_external, score_candidate

    backbone = device = None
    if config.get("embedding"):
        import torch
        from tilt_model import load_backbone
        device = torch.device("cuda")
        backbone = load_backbone(BATCH.parent / "checkpoint",device)
    sorted_rows = sorted(rows,key=lambda row:protocol["rooms"].index(int(row["name"].removeprefix("room"))))
    results = []
    for row in sorted_rows:
        paths = [Path(row[k]).resolve() for k in ("imu_path","mocap_path")]
        imu_path, pose_path = paths
        room_root = (DATA / "TUM-VI" / f"dataset-{row['name']}_512_16").resolve()
        assert all(p.is_relative_to(room_root) for p in paths)
        assert imu_path.parents[1] == pose_path.parents[1]
        assert imu_path.parent.name == "imu0" and pose_path.parent.name == "mocap0"
        imu_ns, readings = read_series(imu_path,range(1,7))
        pose_ns, q = read_series(pose_path,range(4,8))
        assert len(imu_ns) == row["imu_frames"] and len(pose_ns) == row["mocap_frames"]
        step_ns = np.diff(imu_ns)
        dt_ns = float(np.median(step_ns))
        assert dt_ns == row["imu_dt_ns"] and np.max(np.abs(step_ns-dt_ns)) <= dt_ns*protocol["max_relative_imu_step_jitter"]
        observed = np.isfinite(readings).all(axis=1)
        mask, truth = align(imu_ns,observed,pose_ns,q,protocol)
        assert int(mask.sum()) == row["frames"] and mask.any()
        large_gaps = int((np.diff(pose_ns)>protocol["max_consecutive_mocap_gap_ns"]).sum())
        assert large_gaps == row["mocap_gaps_above_threshold"]
        previous = np.maximum.accumulate(np.where(observed,np.arange(len(imu_ns)),-1))
        causal = readings[np.maximum(previous,0)]
        causal[previous<0] = 0.
        dt = dt_ns*1e-9
        gyro, accel = np.ascontiguousarray(causal[:,:3]), np.ascontiguousarray(causal[:,3:])
        pure = VQF(dt).updateBatch(gyro,accel)["quat6D"]
        prediction = Rotation.from_quat(pure[mask],scalar_first=True).inv().apply(protocol["world_up"])
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(prediction,truth[mask]),axis=1),
                                     np.einsum("ij,ij->i",prediction,truth[mask])))
        baseline = float(np.sqrt(np.mean(angle**2)))
        error = abs(baseline-row["baseline_deg"])
        assert error < 2e-4, (row["name"],baseline,row["baseline_deg"])
        measured = {"name":row["name"],"frames":int(mask.sum()),"imu_frames":len(imu_ns),
                    "mocap_frames":len(pose_ns),"imu_sha256":digest(imu_path),"mocap_sha256":digest(pose_path),
                    "mask_sha256":hashlib.sha256(mask.tobytes()).hexdigest(),
                    "large_mocap_gaps":large_gaps,"pure_replay_deg":baseline,"pure_replay_abs_difference_deg":error}
        measured["quiet_gravity_check"] = quiet_gravity_check(readings,mask,truth,dt_ns)
        imu = np.column_stack((accel,gyro))
        cached = cache_external_candidate(imu,dt,mask,observed,truth.astype(np.float32),config)
        latent = embed_external(cached,config,backbone,device)
        value = float(score_candidate(cached,config,weights,latent))
        delta = abs(value-row["rmse_deg"])
        assert delta < 2e-4, (row["name"],value,row["rmse_deg"])
        measured.update(locked_replay_deg=value,locked_replay_abs_difference_deg=delta)
        results.append(measured)
    return results
