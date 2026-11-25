"""Describe the gravity sphere on SHICE development recordings already labeled stationary."""

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from audit_rounds import BATCH, HERE, read, write


def sphere(points):
    mean = points.mean(axis=0)
    squares = np.square(points).sum(axis=1)
    center, _, rank, singular = np.linalg.lstsq(2*(points-mean),squares-squares.mean(),rcond=None)
    radius = np.linalg.norm(points-center,axis=1)
    return {"points":len(points),"rank":int(rank),"smallest_to_largest_singular":float(singular[-1]/singular[0]),
            "center_ms2":center.tolist(),"center_magnitude_ms2":float(np.linalg.norm(center)),
            "median_radius_ms2":float(np.median(radius)),
            "uncorrected_magnitude_q05_q50_q95_ms2":np.quantile(np.linalg.norm(points,axis=1),[.05,.5,.95]).tolist()}


def inspect():
    names = read(BATCH / "rounds/R00001/round.json")["partitions"]["development"]
    devices = defaultdict(list)
    for name in names:
        with np.load(BATCH / "regime-strata/native-regimes-v1" / f"{name}.npz",allow_pickle=False) as regimes:
            sampled = regimes["index"][regimes["static"] == 1][::100]
        with np.load(BATCH.parent / "gru-tilt/data" / f"{name}.npz",allow_pickle=False) as original:
            assert original["score"][sampled].all()
            points = original["imu"][sampled,:3].astype(np.float64)
        assert len(points) >= 4 and np.isfinite(points).all()
        devices[name.split("-")[0]].append((name,points))
    assert set(devices) == {"R1","R2","R3","R4"}
    summary = {}
    for device, records in devices.items():
        summary[device] = {"all_programs":sphere(np.concatenate([p for _,p in records])),
                           "by_record":{name:sphere(points) for name,points in records}}
        for remainder, label in ((0,"even_programs"),(1,"odd_programs")):
            summary[device][label] = sphere(np.concatenate([p for n,p in records if int(n.split("-")[1])%2 == remainder]))
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"development_records":len(names),
            "sampling":"Every 100th previously scored stationary row within each R1-R4 record. Descriptive, no candidate scores.",
            "devices":summary}


def offset_estimate(points, blocks, minimum_poses, protocol):
    if len(points) < minimum_poses or blocks < protocol["minimum_static_blocks"]:
        return None
    unit = points/np.linalg.norm(points,axis=1,keepdims=True)
    eigen = np.linalg.eigvalsh(np.cov(unit,rowvar=False,bias=True))
    if eigen[0] < protocol["minimum_orientation_cov_eigenvalue"] or eigen[-1]/eigen[0] > protocol["maximum_orientation_condition"]:
        return None
    design = np.column_stack((2*points,np.ones(len(points))))
    fit, _, rank, _ = np.linalg.lstsq(design,np.square(points).sum(axis=1),rcond=None)
    assert rank == 4
    offset = fit[:3]
    radii = np.linalg.norm(points-offset,axis=1)
    low,high = np.asarray(protocol["accept_radius_g_interval"])*9.80665
    if (not low <= radii.mean() <= high or radii.std() > protocol["maximum_sphere_rms_m_s2"]
            or np.linalg.norm(offset) > protocol["maximum_offset_norm_m_s2"]):
        return None
    gain = min(len(points)/(len(points)+protocol["online_shrink_pose_pseudocount"]),protocol["online_max_shrink_multiplier"])
    return offset*gain


def raw_offset_trace(raw, minimum_poses, protocol):
    width = round(protocol["block_seconds"]/raw["dt"])
    centers, counts, changes, accepted = [], [], [], []
    good_blocks = 0
    for start in range(0,len(raw["imu"])-width+1,width):
        stop = start + width
        observed = raw["observed"][start:stop]
        if observed.mean() < protocol["min_original_packet_fraction"]:
            continue
        data = raw["imu"][start:stop][observed].astype(np.float64)
        accel, gyro = data[:,:3], data[:,3:]
        point = accel.mean(axis=0)
        if (np.sqrt(accel.var(axis=0).sum()) > protocol["max_accel_standard_deviation_m_s2"]
                or np.sqrt(np.square(gyro).sum(axis=1).mean()) > np.deg2rad(protocol["max_gyro_rms_deg_s"])
                or abs(np.linalg.norm(point)/9.80665-1) > protocol["max_gravity_departure_fraction"]):
            continue
        good_blocks += 1
        if centers:
            dots = np.asarray(centers)@point / (np.linalg.norm(centers,axis=1)*np.linalg.norm(point))
            angles = np.rad2deg(np.arccos(np.clip(dots,-1,1)))
            closest = int(np.argmin(angles))
        if centers and angles[closest] <= protocol["pose_separation_deg"]:
            counts[closest] += 1
            centers[closest] += (point-centers[closest])/counts[closest]
        else:
            centers.append(point.copy())
            counts.append(1)
        if good_blocks % protocol["online_recheck_accepted_blocks"] == 0:
            offset = offset_estimate(np.asarray(centers),good_blocks,minimum_poses,protocol)
            if offset is not None:
                accepted.append((stop,offset))
            changes.append((stop,np.zeros(3) if offset is None else offset))
    trace = np.zeros((len(raw["imu"]),3))
    for i,(start,offset) in enumerate(changes):
        stop = changes[i+1][0] if i+1 < len(changes) else len(trace)
        trace[start:stop] = offset
    return trace, good_blocks, len(centers), accepted


def benchmark_scope(raw, measurement, config, protocol, frames=3000, warmup=100):
    """Recover which measured native updates actually applied or refitted a bias."""
    from audit_rounds import fingerprint

    assert fingerprint(protocol) == config["protocol_sha256"]
    dt = raw["dt"]
    assert np.isclose(1 / dt, measurement["nominal_native_hz"], rtol=1e-8)
    start = round(measurement["native_benchmark_start_seconds"] / dt) + warmup
    stop = start + frames
    assert 0 <= start < stop <= len(raw["imu"])
    trace, _, _, estimates = raw_offset_trace(raw, config["minimum_online_poses"], protocol)
    nonzero = np.linalg.norm(trace, axis=1) > 1e-12
    applied = nonzero & raw["observed"]
    first = int(np.flatnonzero(applied)[0]) if applied.any() else None
    reported = measurement["first_online_calibration_seconds"]
    assert (reported is None) == (first is None)
    if first is not None:
        assert abs(reported - first * dt) < dt / 2
    if stop < len(nonzero):
        assert bool(nonzero[stop]) == measurement["calibration_active_at_end_of_native_benchmark"]
    width = max(1, round(protocol["block_seconds"] / dt))
    blocks = stop // width - start // width
    assert blocks == measurement["measured_bias_blocks"]
    return {"record": measurement["record"], "first_measured_sample": start, "stop_exclusive": stop,
            "sample_seconds": [start * dt, (stop - 1) * dt], "measured_native_frames": frames,
            "frames_with_active_estimate": int(nonzero[start:stop].sum()),
            "observed_packets_with_bias": int(applied[start:stop].sum()), "measured_blocks": blocks,
            "accepted_refits_after_samples": [when for when, _ in estimates if start < when <= stop]}


def replay_online(identity):
    from scipy.spatial.transform import Rotation
    from vqf import VQF
    from verify_baselines import digest
    from verify_final import raw_shice_samples

    row = read(BATCH / "rounds" / identity / "round.json")
    result = read(BATCH / row["result"])
    protocol = result["fit_protocol"]
    minimum = row["config"].get("minimum_online_poses",protocol["minimum_distinct_poses"])
    active_names = {n for n,v in result["online_audit"].items() if v["first_applied_native_index"] is not None}
    names = sorted(active_names | {"R4-2"})
    joint = result["sequences"]["joint_holdout"]
    audits = []
    for name in names:
        raw = raw_shice_samples(name)
        trace, blocks, distinct, accepted = raw_offset_trace(raw,minimum,protocol)
        info = result["online_audit"][name]
        assert (blocks,distinct) == (info["stationary_blocks"],info["pose_clusters"])
        assert [t for t,_ in accepted] == [v["first_eligible_native_index"] for v in info["eligible_updates"]]
        if accepted:
            np.testing.assert_allclose(np.asarray([v for _,v in accepted]),
                                       [u["shrunk_offset_m_s2"] for u in info["eligible_updates"]],rtol=1e-8,atol=1e-9)
        active = np.linalg.norm(trace,axis=1)>1e-12
        active_samples = np.flatnonzero(active & raw["observed"])
        first = int(active_samples[0]) if len(active_samples) else None
        assert len(active_samples) == info["applied_original_packets"] and first == info["first_applied_native_index"]
        mask = raw["score"]
        assert int((active & mask).sum()) == info["scored_frames_with_nonzero_estimate"]
        normal = raw["imu"].astype(np.float64)
        corrected = normal.copy()
        last_valid = np.maximum.accumulate(np.where(raw["observed"],np.arange(len(normal)),-1))
        good = last_valid >= 0
        corrected[good,:3] -= trace[last_valid[good]]
        score = next(r for r in joint if r["name"] == name)
        fold = next(f for f in result["folds"] if f["axis"] == "joint_holdout" and f["fold"] == score["fold"])
        mount = np.asarray(fold["common_mount_rotation_matrix"])
        reference = raw["reference"][mask]
        out = {"name":name,"source_sha256":digest(raw["source"]),"scored_frames":int(mask.sum()),
               "active_scored_frames":int((active & mask).sum()),"first_applied_native_index":first,
               "accepted_estimates":len(accepted),"first_applied_seconds":first*raw["dt"] if first is not None else None}
        scored_after = np.flatnonzero(mask) >= first if first is not None else np.zeros(int(mask.sum()),dtype=bool)
        for method, signal in (("zero",normal),("online_shrunk",corrected)):
            q = VQF(raw["dt"],**row["config"]["parameters"]).updateBatch(np.ascontiguousarray(signal[:,3:]),np.ascontiguousarray(signal[:,:3]))["quat6D"]
            predicted = Rotation.from_quat(q[mask],scalar_first=True).inv().apply((0,0,1)) @ mount
            angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted,reference),axis=1),
                                         np.einsum("ij,ij->i",predicted,reference)))
            rmse = float(np.sqrt(np.mean(angle**2)))
            diff = abs(rmse-score["rmse_deg"][method])
            assert diff < 2e-4, (name,method,diff)
            out[method] = {"raw_rmse_deg":rmse,"absolute_difference_from_round_deg":diff,
                           "before_first_activation_deg":float(np.sqrt(np.mean(angle[~scored_after]**2))) if (~scored_after).any() else None,
                           "after_first_activation_deg":float(np.sqrt(np.mean(angle[scored_after]**2))) if scored_after.any() else None}
        assert out["zero"]["before_first_activation_deg"] == out["online_shrunk"]["before_first_activation_deg"]
        audits.append(out)
    return {"round_id":identity,"config_sha256":row["config_sha256"],"round_result_sha256":digest(BATCH / row["result"]),
            "reviewed_utc":datetime.now(timezone.utc).isoformat(),
            "source_scope":"All records with active bias plus R4-2, a predeclared unactivated control.","records":audits}


if __name__ == "__main__":
    result = inspect()
    write(HERE / "development_accelerometer_spheres.json",result)
    for name, parts in result["devices"].items():
        print(name,[(k,round(parts[k]["center_magnitude_ms2"],4),round(parts[k]["smallest_to_largest_singular"],4))
                    for k in ("all_programs","even_programs","odd_programs")])
    candidate = BATCH / "rounds/A00002/result.json"
    if candidate.is_file():
        replay = replay_online("A00002")
        write(HERE / "accelerometer_online_replay.json",replay)
        for evidence in replay["records"]:
            print(evidence["name"],evidence["active_scored_frames"],evidence["online_shrunk"]["absolute_difference_from_round_deg"])
