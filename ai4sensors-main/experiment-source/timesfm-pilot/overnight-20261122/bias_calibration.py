"""Compare causal static-pose accelerometer offset estimates on development data only."""

from collections import Counter
from time import perf_counter

import numpy as np

from common import BATCH, SHICE, aggregate, body_up, by_device, fingerprint, names, read_json, record, rmse, stamp, unit, vqf_outputs, write_json
from experiment import baseline_reference
from features import G, fetch_record
from install_rotation import fit_statistics, sufficient_statistics
from joint_holdout import joint_folds
from learning import folds
from run_pilot import CHANNELS
from tilt_data import causal_fill


PROTOCOL = {
    "id":"accelerometer-static-sphere-v1",
    "block_seconds":0.1,"min_original_packet_fraction":0.95,"max_gyro_rms_deg_s":1.0,
    "max_accel_standard_deviation_m_s2":0.06,"max_gravity_departure_fraction":0.06,
    "pose_separation_deg":20.0,"minimum_distinct_poses":6,"minimum_static_blocks":24,
    "minimum_orientation_cov_eigenvalue":0.015,"maximum_orientation_condition":32.0,
    "accept_radius_g_interval":[0.96,1.04],"maximum_offset_norm_m_s2":0.65,
    "maximum_sphere_rms_m_s2":0.14,"online_recheck_accepted_blocks":5,
    "online_shrink_pose_pseudocount":6.0,"online_max_shrink_multiplier":0.8,
    "timing":"evaluate a block only after all its observations arrive; apply to native samples after the block",
    "scope":"R1-R4 only; grouped device, grouped program, and joint holdout",
    "fallback":"zero offset when insufficiently supported; one independent online state per recording",
}


def read_native(name):
    source = record(name)
    file = SHICE / f"{name.split('-')[0]}_aligned_formal" / f"{name}.csv"
    with file.open(encoding="utf-8-sig") as stream:
        header = stream.readline().strip().split(",")
        columns = ("sample_index",*CHANNELS,"is_interpolated")
        table = np.loadtxt(stream,delimiter=",",quotechar='"',usecols=[header.index(key) for key in columns],ndmin=2)
    np.testing.assert_array_equal(table[:,0],np.arange(len(source["imu"])))
    untreated = table[:,1:7].astype(np.float32)
    original_ok = (table[:,-1]==0) & np.isfinite(untreated).all(axis=1)
    np.testing.assert_allclose(causal_fill(untreated,original_ok),source["imu"],atol=2e-6,rtol=1e-6)
    if not original_ok[source["score"]].all():
        raise ValueError(f"Score mask in {name} includes an invalid native packet")
    return {**source,"untreated":untreated,"original_ok":original_ok}


class PoseCloud:
    def __init__(self):
        self.sums = []
        self.counts = []
        self.cosine = np.cos(np.deg2rad(PROTOCOL["pose_separation_deg"]))

    def observe(self, point):
        if self.sums:
            similarity = unit(np.asarray(self.sums)) @ unit(point)
            match = int(np.argmax(similarity))
            if similarity[match] >= self.cosine:
                self.sums[match] += point
                self.counts[match] += 1
                return
        self.sums.append(point.copy())
        self.counts.append(1)

    def centroids(self):
        if not self.sums:
            return np.empty((0,3),dtype=np.float64)
        return np.asarray(self.sums) / np.asarray(self.counts)[:,None]


def stationary_candidates(imu, packet_ok, dt):
    width = max(1, round(PROTOCOL["block_seconds"]/dt))
    count = len(imu)//width
    if not count:
        return np.empty(0,dtype=int),np.empty((0,3)),0
    valid = packet_ok[:count*width].reshape(count,width)
    raw = imu[:count*width].reshape(count,width,6).astype(np.float64)
    safe = np.where(valid[:,:,None],raw,0.)
    n = valid.sum(axis=1)
    divisor = np.maximum(n,1)
    avg_acc = safe[:,:,:3].sum(axis=1) / divisor[:,None]
    second_moment = np.square(safe[:,:,:3]).sum(axis=(1,2)) / divisor
    variance = np.maximum(second_moment-np.square(avg_acc).sum(axis=1),0.)
    gyro_rms = np.sqrt(np.square(safe[:,:,3:]).sum(axis=(1,2))/divisor)
    gravity_departure = np.abs(np.linalg.norm(avg_acc,axis=1)/G-1.)
    admitted = ((n >= np.ceil(width*PROTOCOL["min_original_packet_fraction"])) &
                (gyro_rms <= np.deg2rad(PROTOCOL["max_gyro_rms_deg_s"])) &
                (np.sqrt(variance) <= PROTOCOL["max_accel_standard_deviation_m_s2"]) &
                (gravity_departure <= PROTOCOL["max_gravity_departure_fraction"]))
    next_native_index = (np.flatnonzero(admitted)+1)*width
    return next_native_index, avg_acc[admitted], count


def sphere(points, weights, raw_static_blocks, minimum_poses=None):
    minimum = PROTOCOL["minimum_distinct_poses"] if minimum_poses is None else minimum_poses
    count = len(points)
    info = {"pose_count":count,"static_blocks":int(raw_static_blocks),"eligible":False}
    if count == 0:
        return {**info,"reason":"insufficient_distinct_pose_or_block_count",
                "orientation_cov_eigenvalues":None,"orientation_condition_number":None}
    weights = np.asarray(weights,dtype=np.float64)
    weights /= weights.sum()
    directions = unit(points)
    centered_dirs = directions - np.average(directions,axis=0,weights=weights)
    cov = (centered_dirs*weights[:,None]).T @ centered_dirs
    eigen = np.linalg.eigvalsh(cov)
    condition = float(eigen[-1]/max(eigen[0],1e-12))
    info.update(orientation_cov_eigenvalues=eigen.tolist(),orientation_condition_number=condition)
    if count < minimum or raw_static_blocks < PROTOCOL["minimum_static_blocks"]:
        return {**info,"reason":"insufficient_distinct_pose_or_block_count"}
    if eigen[0] < PROTOCOL["minimum_orientation_cov_eigenvalue"] or condition > PROTOCOL["maximum_orientation_condition"]:
        return {**info,"reason":"insufficient_three_dimensional_orientation_coverage"}
    center = np.average(points,axis=0,weights=weights)
    square = np.square(points).sum(axis=1)
    root_w = np.sqrt(weights)
    design = (points-center) * root_w[:,None]
    rhs = .5*(square - np.dot(weights,square))*root_w
    estimate = np.linalg.lstsq(design,rhs,rcond=None)[0]
    offset_size = float(np.linalg.norm(estimate))
    radii = np.linalg.norm(points-estimate,axis=1)
    radius = float(np.dot(weights,radii))
    residual = float(np.sqrt(np.dot(weights,(radii-radius)**2)))
    low, high = PROTOCOL["accept_radius_g_interval"]
    info.update(offset_m_s2=estimate.tolist(),offset_norm_m_s2=offset_size,
                radius_m_s2=radius,radial_rmse_m_s2=residual)
    if not (low*G <= radius <= high*G) or offset_size > PROTOCOL["maximum_offset_norm_m_s2"] or residual > PROTOCOL["maximum_sphere_rms_m_s2"]:
        return {**info,"reason":"sphere_residual_radius_or_offset_outside_fixed_bounds"}
    return {**info,"eligible":True,"reason":"coverage_and_residual_supported"}


def compensate(imu, packet_ok, bias):
    inputs = imu.astype(np.float64,copy=True)
    inputs[:,:3] -= bias
    return causal_fill(inputs,packet_ok)


class OnlineBiasTracker:
    def __init__(self, minimum_poses=None):
        self.cloud = PoseCloud()
        self.minimum_poses = minimum_poses
        self.offset = np.zeros(3)
        self.observations = 0
        self.accepted = []
        self.last_fit = None
        self.reviews = Counter()

    def observe(self, point, next_native_index):
        self.cloud.observe(point)
        self.observations += 1
        if self.observations % PROTOCOL["online_recheck_accepted_blocks"]:
            return False
        points = self.cloud.centroids()
        fit = sphere(points,np.ones(len(points)),self.observations,self.minimum_poses)
        self.last_fit = fit
        self.reviews[fit["reason"]] += 1
        if fit["eligible"]:
            effective = len(points)/(len(points)+PROTOCOL["online_shrink_pose_pseudocount"])
            effective = min(effective,PROTOCOL["online_max_shrink_multiplier"])
            self.offset = np.asarray(fit["offset_m_s2"])*effective
            self.accepted.append({"first_eligible_native_index":int(next_native_index),
                                  "shrunk_offset_m_s2":self.offset.tolist(),"multiplier":effective,**fit})
        else:
            self.offset = np.zeros(3)
        return True


def online_filter(imu, packet_ok, dt, minimum_poses=None):
    indices, candidates, blocks = stationary_candidates(imu,packet_ok,dt)
    tracker = OnlineBiasTracker(minimum_poses)
    offset = np.zeros((len(imu),3))
    since = 0
    for when,point in zip(indices,candidates,strict=True):
        last = tracker.offset
        if tracker.observe(point,when):
            offset[since:when] = last
            since = int(when)
    offset[since:] = tracker.offset
    actual_nonzero = (np.linalg.norm(offset,axis=1)>1e-12)&packet_ok
    first = np.flatnonzero(actual_nonzero)
    audit = {"stationary_blocks":len(candidates),"native_interval_blocks":blocks,
             "pose_clusters":len(tracker.cloud.counts),"rechecks":dict(tracker.reviews),"last_fit":tracker.last_fit,
             "first_applied_native_index":int(first[0]) if len(first) else None,
             "first_applied_seconds":float(first[0]*dt) if len(first) else None,
             "applied_original_packets":int(actual_nonzero.sum()),"eligible_updates":tracker.accepted}
    return compensate(imu,packet_ok,offset),offset,tracker.cloud.centroids(),audit


def pooled_fit(training, samples):
    records = [(samples[n]["poses"],samples[n]["audit"]["stationary_blocks"]) for n in training]
    populated = [points for points,_ in records if len(points)]
    if not populated:
        return {"eligible":False,"reason":"no_stable_training_records","pose_count":0,"static_blocks":0}
    points = np.concatenate(populated)
    mass = np.concatenate([np.full(len(points),1./len(points)) for points in populated])
    result = sphere(points,mass,sum(count for _,count in records))
    pooled_pose_cloud = PoseCloud()
    for point in points:
        pooled_pose_cloud.observe(point)
    return {**result,"contributing_training_records":len(populated),
            "diagnostic_distinct_pooled_poses":len(pooled_pose_cloud.counts)}


def quarters(imu,packet_ok,dt,reference,reference_offset,parameters,minimum_poses=None):
    checks = []
    base = vqf_outputs(reference,dt,parameters)["quat6D"]
    for fraction in (.27,.59,.81):
        index = int(len(imu)*fraction)
        observed, offsets, _, _ = online_filter(imu[:index],packet_ok[:index],dt,minimum_poses)
        difference = float(np.max(np.abs(observed-reference[:index])))
        angle_output = vqf_outputs(observed,dt,parameters)["quat6D"]
        checks.append({"native_prefix":index,"max_abs_corrected_input_difference":difference,
                       "max_abs_bias_difference":float(np.max(np.abs(offsets-reference_offset[:index]))),
                       "max_abs_vqf_quaternion_difference":float(np.max(np.abs(angle_output-base[:index])))})
    return checks


def audit_output(fitted, raw, dt, parameters, mounting, measurement):
    if measurement == "zero":
        return rmse(raw["base_up_scored"] @ mounting, raw["ref_body_scored"])
    native = body_up(vqf_outputs(fitted,dt,parameters)["quat6D"][raw["index"]])
    return rmse(native @ mounting, raw["ref_body_scored"])


def run(job,destination):
    started = perf_counter()
    election = job["config"]
    stage = read_json(BATCH/"accelerometer-bias-development-stage.json")
    if election["protocol_sha256"] != fingerprint(stage["protocol"]):
        raise ValueError("Frozen accelerometer-bias protocol changed before execution")
    parameters = election["parameters"]
    minimum_online_poses = election.get("minimum_online_poses",PROTOCOL["minimum_distinct_poses"])
    mount_source = read_json(BATCH/"rounds"/election["mount_control_round"]/"round.json")
    config = mount_source["config"]
    if election["mount_control_config_sha256"]!=fingerprint(config):
        raise ValueError("Frozen mount parameters changed")
    names_dev = names("dev")
    sources, samples = {}, {}
    raw = {n:fetch_record(n,parameters) for n in names_dev}
    for n in names_dev:
        source = read_native(n)
        corrected, offsets, poses, audit = online_filter(source["untreated"],source["original_ok"],float(source["dt"]),minimum_online_poses)
        scores = source["score"]
        audit["scored_frames_with_nonzero_estimate"] = int(np.count_nonzero(np.linalg.norm(offsets[scores],axis=1)>1e-12))
        audit["scored_frames"] = int(scores.sum())
        index = raw[n]["index"]
        first = audit["first_applied_native_index"]
        if first is None:
            online_up = raw[n]["base_up_scored"]
        else:
            online_up = body_up(vqf_outputs(corrected,float(source["dt"]),parameters)["quat6D"][index])
            online_up[index < first] = raw[n]["base_up_scored"][index < first]
        samples[n] = {"audit":audit,"poses":poses,"trace":offsets,"online":corrected,"online_up":online_up}
        sources[n] = source
        print(n,"online poses",audit["pose_clusters"],"first",audit["first_applied_seconds"],flush=True)
    mount_stats = {n:sufficient_statistics(raw[n],config) for n in names_dev}
    baseline = baseline_reference()
    fold_plan = [(axis,fold,[n for n in names_dev if n not in held],held) for axis,fold,held in folds(names_dev)]
    fold_plan.extend(("joint_holdout",label,training,held) for label,training,held in joint_folds(names_dev))
    expected_dev = read_json(BATCH/mount_source["result"])["sequences"]
    expected_joint = read_json(BATCH/"rounds"/election["joint_mount_control_round"]/"result.json")["sequences"]
    expected = {axis:{row["name"]:row for row in group} for axis,group in expected_dev.items()}
    expected["joint_holdout"] = {row["name"]:row for row in expected_joint}
    predictions, fold_evidence = {}, []
    for axis,label,training,held in fold_plan:
        fit = fit_statistics([mount_stats[n] for n in training],config)
        mounting = fit["fitted"]
        fixed = pooled_fit(training,samples)
        offset = np.asarray(fixed["offset_m_s2"]) if fixed["eligible"] else np.zeros(3)
        rows = []
        for n in held:
            source = sources[n]
            base = audit_output(None,raw[n],None,parameters,mounting,"zero")
            old = expected[axis][n]
            if abs(base-old["rmse_deg"])>1e-5 or old["frames"]!=len(raw[n]["index"]):
                raise ValueError(f"Zero compensation changed the frozen control in {axis}/{n}")
            if fixed["eligible"]:
                corrected = compensate(source["untreated"],source["original_ok"],offset)
                fixed_score = audit_output(corrected,raw[n],float(source["dt"]),parameters,mounting,"fixed")
            else:
                fixed_score = base
            online_score = rmse(samples[n]["online_up"] @ mounting,raw[n]["ref_body_scored"])
            rows.append({"name":n,"fold":label,"frames":len(raw[n]["index"]),"pure_fullvqf_deg":baseline[n],
                         "rmse_deg":{"zero":base,"fixed_fold":fixed_score,"online_shrunk":online_score}})
        predictions.setdefault(axis,[]).extend(rows)
        fold_evidence.append({"axis":axis,"fold":label,"training":training,"validation":held,
                              "common_mount_rotation_matrix":mounting.tolist(),"fixed_bias":fixed})
        print(axis,label,"fixed",fixed.get("offset_m_s2"),"supported",fixed["eligible"],flush=True)
    summary = {}
    for axis,group in predictions.items():
        if sorted(row["name"] for row in group)!=sorted(names_dev):
            raise ValueError(f"Missing original records in {axis}")
        summary[axis] = {}
        for method in ("zero","fixed_fold","online_shrunk"):
            scores = [{"name":r["name"],"frames":r["frames"],"rmse_deg":r["rmse_deg"][method],"baseline_deg":r["rmse_deg"]["zero"]} for r in group]
            summary[axis][method] = {**aggregate(scores),"per_device":by_device(scores)}
    activated = [n for n in names_dev if samples[n]["audit"]["first_applied_native_index"] is not None]
    probes = list(dict.fromkeys(["R2-10","R4-2",*activated[:1]]))
    prefix = {n:quarters(sources[n]["untreated"],sources[n]["original_ok"],float(sources[n]["dt"]),
                         samples[n]["online"],samples[n]["trace"],parameters,minimum_online_poses) for n in probes}
    exact = all(all(v[key]==0. for key in ("max_abs_corrected_input_difference","max_abs_bias_difference","max_abs_vqf_quaternion_difference"))
                for values in prefix.values() for v in values)
    write_json(destination,{"round_id":job["round_id"],"completed_at":stamp(),"fit_protocol":PROTOCOL,
        "online_minimum_distinct_poses":minimum_online_poses,"fixed_minimum_pooled_pose_clusters":PROTOCOL["minimum_distinct_poses"],
        "eligible_development_records":len(names_dev),"unchanged_frame_masks_and_mounting":True,
         "online_independent_per_record_uses_prior_samples_only":True,"untouched_zero_path_when_online_never_activates":True,
        "fixed_fits_use_only_current_training_fold":True,"online_records_with_support":len(activated),
        "summary":summary,"runtime_seconds":perf_counter()-started,"sequences":predictions,"folds":fold_evidence,
        "online_audit":{key:value["audit"] for key,value in samples.items()},"causal_prefix_replay":prefix,"exact_prefix_replay":exact})


def queue():
    from optimizer_study import round_record,schedule

    mount = round_record("I00004")
    config = {"kind":"accelerometer_bias_audit","mount_control_round":"I00004","joint_mount_control_round":"J00002",
              "mount_control_config_sha256":mount["config_sha256"],"parameters":mount["config"]["parameters"],
              "protocol_sha256":fingerprint(PROTOCOL),"seed":None}
    path = BATCH / "accelerometer-bias-development-stage.json"
    write_json(path,{"created_at":stamp(),"protocol":PROTOCOL,"round_id":"A00001","zero_arm_control":"I00004 / J00002",
                    "branches":["zero","fixed_fold","online_shrunk"],
                    "compare_with_matched_light_head_only_if":{"minimum_joint_improvement_deg":.015,
                        "max_leave_device_degradation_deg":.015,"online_definition_requires_exact_prefix_replay":True}})
    identity = schedule("A00001",config,"static-sphere-accelerometer-offset","I00004",
                        "Use physically supported gravity spheres to compare zero, fold-only fixed, and independent causal online offsets")
    print(identity,flush=True)


if __name__ == "__main__":
    queue()
