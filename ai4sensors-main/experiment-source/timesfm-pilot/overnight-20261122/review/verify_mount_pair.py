"""Audit the 50 Hz ridge mounting comparison using the fixed R1–R4 native sample index."""

import hashlib
from datetime import datetime, timezone
from itertools import combinations
from statistics import fmean, median

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, fingerprint, read, write
from verify_baselines import digest


PLAN = BATCH / "ridge-50hz-mount-matched-protocol.json"
PLAN_SHA = "9180ec99dd933112a182adfcf8ad9ce51a9732f2d9bec274f49ff77bd77acb03"
REFERENCE = HERE / "ridge_50hz_mount_reference.json"


def quaternion_solution(moment):
    """Maximize trace(R(q).T @ moment) using a 4-D unit-quaternion eigensystem."""
    def objective(q):
        return np.sum(Rotation.from_quat(q,scalar_first=True).as_matrix()*moment)

    basis = np.eye(4)
    diagonal = np.array([objective(q) for q in basis])
    matrix = np.diag(diagonal)
    for i,j in combinations(range(4),2):
        both = objective((basis[i]+basis[j])/np.sqrt(2))
        matrix[i,j] = matrix[j,i] = both-(diagonal[i]+diagonal[j])/2
    eigvals, vectors = np.linalg.eigh(matrix)
    assert eigvals[-1]-eigvals[-2] > 1e-8, "The training poses do not determine a unique rotation."
    return Rotation.from_quat(vectors[:,-1],scalar_first=True).as_quat(scalar_first=True,canonical=True)


def mount_reference():
    assert digest(PLAN) == PLAN_SHA
    plan = read(PLAN)
    names = set(plan["records"])
    assert len(names) == 37 and {n.split("-")[0] for n in names} == {"R1","R2","R3","R4"}
    parameters = plan["pairs"][0]["without_mount"]["config"]["parameters"]
    assert parameters == {}
    cache = BATCH/"cache"/fingerprint({"version":plan["mask_version"],"parameters":parameters})[:16]
    quaternion_dir = BATCH/"scored-quaternion"/fingerprint(parameters)[:16]
    evidence, moments = {}, {}
    for name in sorted(names):
        native = np.load(quaternion_dir/(name+".npy"),allow_pickle=False,mmap_mode="r")
        with np.load(cache/(name+".npz"),allow_pickle=False) as data:
            index, target, cached = (data[k] for k in ("index","ref_body_scored","base_up_scored"))
        assert native.shape == (len(index),4) and target.shape == cached.shape == (len(index),3)
        assert len(index) and np.all(np.diff(index) > 0)
        assert np.max(np.abs(np.linalg.norm(native,axis=1)-1)) < 2e-6
        assert np.max(np.abs(np.linalg.norm(target,axis=1)-1)) < 2e-6
        base = Rotation.from_quat(native,scalar_first=True).inv().apply([0,0,1])
        error = float(np.max(np.abs(base-cached)))
        assert error < 2e-6
        moments[name] = (base.T @ target)/len(index)
        evidence[name] = {"frames":len(index),"base_max_abs_difference":error,
                          "score_index_sha256":hashlib.sha256(index.astype("<i8").tobytes()).hexdigest(),
                          "quaternion_sha256":digest(quaternion_dir/(name+".npy")),
                          "target_array_sha256":hashlib.sha256(target.tobytes()).hexdigest()}
    assert sum(r["frames"] for r in evidence.values()) == plan["score_frames_per_validation_axis"]
    folds = {}
    schedule = list(plan["grouped_validation_folds"])+[["joint_holdout",*r] for r in plan["joint_validation_folds"]]
    for axis,label,training,held in schedule:
        check_fold({"training":training,"validation":held,"fold":label},axis,names)
        mean = np.mean([moments[n] for n in training],axis=0)
        quat = quaternion_solution(mean)
        folds[axis+":"+label] = {"training":sorted(training),"validation":sorted(held),
                                 "wxyz":quat.tolist(),"frames":sum(evidence[n]["frames"] for n in training)}
    assert len(folds) == len(schedule) == 24
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":PLAN_SHA,
            "source_records":evidence,"folds":folds}


def compare_axis(axis, old, new, reference):
    records = reference["source_records"]
    old_seq, new_seq = [item["sequences"][axis] if isinstance(item["sequences"],dict) else item["sequences"] for item in (old,new)]
    original = {s["name"]:s for s in old_seq}
    assert len(original) == len(old_seq) == len(new_seq) and set(original) == set(records)
    assert new["training_context_audit"] == old["training_context_audit"]
    folds = [f for f in new["folds"] if f.get("axis","joint_holdout") == axis]
    deltas, angles = {}, []
    for fold in folds:
        ref = reference["folds"][axis+":"+fold["fold"]]
        assert sorted(fold["training"]) == ref["training"] and sorted(fold["validation"]) == ref["validation"]
        q = (fold["training_log"] if axis == "joint_holdout" else fold)["installation_quaternion_wxyz"]
        assert abs(np.linalg.norm(q)-1) < 2e-6
        rot = Rotation.from_quat(q,scalar_first=True)
        oracle = Rotation.from_quat(ref["wxyz"],scalar_first=True)
        angles.append(float(np.rad2deg((rot.inv()*oracle).magnitude())))
    assert {f["fold"] for f in folds} == {key.split(":",1)[1] for key in reference["folds"] if key.startswith(axis+":")}
    assert len(folds) == (16 if axis == "joint_holdout" else 4) and max(angles) < 2e-5
    for s in new_seq:
        n = s["name"]
        prior = original[n]
        assert n not in deltas and n in reference["folds"][axis+":"+s["fold"]]["validation"]
        assert s["fold"] == prior["fold"] and s["baseline_deg"] == prior["baseline_deg"]
        assert s["frames"] == prior["frames"] == records[n]["frames"]
        if axis == "joint_holdout":
            assert s["score_frame_indices_sha256"] == records[n]["score_index_sha256"]
        deltas[n] = s["rmse_deg"]-prior["rmse_deg"]
    assert set(deltas) == set(records)
    group = [item["dev_metrics"][axis]["macro_rmse_deg"] for item in (old,new)]
    assert abs(group[1]-group[0]-fmean(deltas.values())) < 1e-9
    return {"before_deg":group[0],"after_deg":group[1],"difference_deg":fmean(deltas.values()),
            "by_record":deltas,"new_folds":len(folds),"max_rotation_difference_deg":max(angles)}


def reconcile_published(published, result, reference, index, plan):
    assert published["protocol_sha256"] == PLAN_SHA and published["paired_evidence_passed"] == result["paired_evidence_gate"]
    assert published["new_installation_fit_calls"] == published["new_ridge_fit_calls"] == result["fold_fits_per_stage"]
    assert published["distinct_training_fold_rotations"] == len(reference["folds"]) and published["failures"] == 0
    source_ids = {p[choice][key] for p in plan["pairs"] for choice in ("without_mount","with_mount") for key in ("round","joint_round")}
    expected_files = {rid:digest(BATCH/index[rid]["result"]) for rid in source_ids}
    assert len(published["source_results"]) == len(expected_files) and {r["round"]:r["result_sha256"] for r in published["source_results"]} == expected_files
    for pair, announced in zip(result["pairs"],published["pairs"],strict=True):
        assert pair["seed"] == announced["seed"] and pair["after_id"] == announced["with_mount_round"]
        for axis, data in pair["axes"].items():
            observed = published["across_seed"][axis]["delta_with_minus_without_deg"]
            np.testing.assert_allclose([observed["mean"],*observed["range"]],
                                       [result["axes"][axis]["difference_deg"],*result["axes"][axis]["seed_range_deg"]],rtol=0,atol=1e-9)
            metrics = announced["metrics"][axis]
            np.testing.assert_allclose([metrics["without_mount_deg"],metrics["with_mount_deg"],metrics["delta_with_minus_without_deg"]],
                                       [data["before_deg"],data["after_deg"],data["difference_deg"]],rtol=0,atol=1e-9)
            key = "without_mount_joint" if axis == "joint_holdout" else "without_mount_round"
            src = read(BATCH/index[announced[key]]["result"])["sequences"]
            orig = {r["name"]:r for r in (src[axis] if isinstance(src,dict) else src)}
            records = announced["sequences"][axis]
            assert len(records) == len(orig) and {r["name"] for r in records} == set(orig)
            for row in records:
                n = row["name"]
                ref = reference["source_records"][n]
                assert row["score_frame_indices_sha256"] == ref["score_index_sha256"] and row["frames"] == ref["frames"]
                assert row["fold"] == orig[n]["fold"]
                np.testing.assert_allclose([row["without_mount_deg"],row["with_mount_deg"],row["delta_with_minus_without_deg"]],
                                           [orig[n]["rmse_deg"],orig[n]["rmse_deg"]+data["by_record"][n],data["by_record"][n]],rtol=0,atol=1e-9)


def group_differences(pairs, axis):
    by_seed = [p["axes"][axis]["by_record"] for p in pairs]
    names = sorted(by_seed[0])
    assert all(set(s) == set(names) for s in by_seed)
    average = {n:fmean(s[n] for s in by_seed) for n in names}
    return group_record_means(average)


def group_record_means(average):
    from verify_dev_weights import PROGRAMS

    names = sorted(average)
    groups = {d:[n for n in names if n.split("-")[0] == d] for d in sorted({n.split("-")[0] for n in names})}
    groups.update({f"M{i}":[n for n in names if int(n.split("-")[1]) in group] for i,group in enumerate(PROGRAMS,1)})
    measure = {}
    for label, group in groups.items():
        inside = [average[n] for n in group]
        outside = [average[n] for n in names if n not in group]
        measure[label] = {"records":len(inside),"wins":sum(x < -1e-8 for x in inside),"mean_deg":fmean(inside),
                          "mean_without_group_deg":fmean(outside)}
    for prefix in ("R","M"):
        members = [g for name,g in measure.items() if name.startswith(prefix)]
        assert sum(g["records"] for g in members) == len(names)
        assert abs(sum(g["records"]*g["mean_deg"] for g in members)/len(names)-fmean(average.values())) < 1e-10
    return {"wins":sum(v < -1e-8 for v in average.values()),"median_deg":median(average.values()),"groups":measure}


def audit_pairs(index, completed_ids, frames):
    if not PLAN.is_file():
        return None
    assert digest(PLAN) == PLAN_SHA
    plan = read(PLAN)
    if any(p["with_mount"][k] not in completed_ids for p in plan["pairs"] for k in ("round","joint_round")):
        return None
    reference = read(REFERENCE)
    assert reference["plan_sha256"] == PLAN_SHA
    assert {n:r["frames"] for n,r in reference["source_records"].items()} == frames
    historic = read(BATCH/"ridge-25hz-50hz-joint-summary.json")["common_frames_by_record"]
    assert all(historic[n] == {k:r[k] for k in ("frames","score_index_sha256")} for n,r in reference["source_records"].items())
    axes = ("device","motion","joint_holdout")
    pairs = []
    for trial in plan["pairs"]:
        old,new = trial["without_mount"],trial["with_mount"]
        assert new["config"] == {**old["config"],"installation":"all","mount_shrink":1.0}
        for member in (old,new):
            grouped,joint = (index[member[k]] for k in ("round","joint_round"))
            assert grouped["config"] == member["config"] and grouped["config_sha256"] == member["config_sha256"]
            assert joint["config"]["parent_round"] == grouped["round_id"] and joint["config"]["parent_config_sha256"] == grouped["config_sha256"]
            assert grouped["seed"] == joint["seed"] == trial["seed"]
        by_axis = {}
        for key, directions in (("round",axes[:2]),("joint_round",axes[2:])):
            row = index[new[key]]
            assert datetime.fromisoformat(row["started_at"]) >= datetime.fromisoformat(plan["registered_at"])
            prev = BATCH/index[old[key]]["result"]
            assert digest(prev) == old["grouped_sha256" if key == "round" else "joint_sha256"]
            left, right = read(prev),read(BATCH/row["result"])
            for axis in directions:
                by_axis[axis] = compare_axis(axis,left,right,reference)
                assert abs(by_axis[axis]["before_deg"]-old["prior_rmse_deg"][axis]) < 1e-9
        pairs.append({"seed":trial["seed"],"before_id":old["round"],"after_id":new["round"],"joint_id":new["joint_round"],"axes":by_axis})
    assert [p["seed"] for p in pairs] == [20261122,20261123,20261124]
    summary = {}
    for axis in axes:
        changes = [p["axes"][axis]["difference_deg"] for p in pairs]
        summary[axis] = {"before_deg":fmean(p["axes"][axis]["before_deg"] for p in pairs),
                         "after_deg":fmean(p["axes"][axis]["after_deg"] for p in pairs),
                         "difference_deg":fmean(changes),"seed_range_deg":[min(changes),max(changes)],
                         "per_record_groups":group_differences(pairs,axis)}
    threshold = plan["paired_evidence_threshold"]
    passed = (summary["joint_holdout"]["difference_deg"] <= -threshold["three_seed_mean_joint_improvement_deg"]
              and max(summary[a]["difference_deg"] for a in ("device","motion")) <= threshold["max_axis_mean_regression_deg"])
    total = sum(x["new_folds"] for p in pairs for x in p["axes"].values())
    result = {"pairs":pairs,"axes":summary,"fold_fits_per_stage":total,"paired_evidence_gate":passed,
            "records":len(frames),"native_frames":sum(frames.values()),"plan_sha256":PLAN_SHA,
            "max_rotation_difference_deg":max(x["max_rotation_difference_deg"] for p in pairs for x in p["axes"].values())}
    published = BATCH/"ridge-50hz-mount-matched-summary.json"
    if published.is_file():
        reconcile_published(read(published),result,reference,index,plan)
        result["published_sha256"] = digest(published)
    return result


if __name__ == "__main__":
    reference = mount_reference()
    write(REFERENCE,reference)
    print(len(reference["source_records"]),"native records;",len(reference["folds"]),"independent fold rotations")
