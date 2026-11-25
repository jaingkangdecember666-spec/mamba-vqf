"""Measure when the 6/8 degree cap changes a correction in a development pair with equal scores."""

import hashlib
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, read, write
from verify_baselines import digest
from verify_mount_pair import REFERENCE


IDS = ("M00103","R01581")
AXES = ("device","motion")
CAPS = (6,8)


def activity(direction, base_up, ready, native_index, deltas):
    assert direction.shape == base_up.shape == (len(ready),3) and len(ready) > 0
    assert np.all(np.diff(ready) > 0) and np.all(np.diff(native_index) > 0) and len(native_index) > 0
    tangent = direction-(direction*base_up).sum(axis=1,keepdims=True)*base_up
    amplitude = np.linalg.norm(tangent,axis=1)
    assert np.isfinite(amplitude).all()
    angle = np.rad2deg(np.arctan(amplitude.astype(np.float64)))
    # -1 means the scored output predates the first available update.
    latest = np.searchsorted(ready,native_index,side="right")-1
    used = np.bincount(latest[latest>=0],minlength=len(ready))
    assert set(deltas) == set(CAPS) and all(v.shape == direction.shape and np.isfinite(v).all() for v in deltas.values())
    changed = np.any(deltas[6] != deltas[8],axis=1)
    counts = {}
    for cap in CAPS:
        over = amplitude > np.tan(np.deg2rad(cap))
        counts[str(cap)] = {"over_limit_updates":int(over.sum()),"over_limit_updates_used_for_scoring":int(np.count_nonzero(over & (used>0))),
                            "score_frames_with_over_limit_update":int(used[over].sum())}
    return {"updates":len(ready),"updates_used_for_scoring":int(np.count_nonzero(used)),"score_frames":len(native_index),
            "frames_before_first_update":int((latest<0).sum()),"maximum_unclipped_deg":float(angle.max()),
            "maximum_unclipped_scored_deg":float(angle[used>0].max()) if used.any() else None,
            "threshold_counts":counts,"different_update_vectors":int(changed.sum()),"different_vectors_on_score_frames":int(used[changed].sum())}


def replay():
    sys.path.insert(0,str(BATCH))
    import learning
    from final_eval import fit_candidate

    jobs = [read(BATCH/"rounds"/rid/"round.json") for rid in IDS]
    configs = [r["config"] for r in jobs]
    assert configs[1] == {**configs[0],"cap_deg":8.0} and configs[0]["cap_deg"] == 6.0
    assert all(c["seed"] == 20261124 and c["update_hz"] == 50 and c["installation"] == "all" and c["embedding"] == 0 for c in configs)
    paths = [BATCH/r["result"] for r in jobs]
    results = [read(p) for p in paths]
    ref = read(REFERENCE)
    names = set(ref["source_records"])
    first,second = [{(f["axis"],f["fold"]):f for f in r["folds"]} for r in results]
    assert set(first) == set(second) and len(first) == 8
    expected_scores = [{a:{r["name"]:r for r in data["sequences"][a]} for a in AXES} for data in results]
    validation = {a:{} for a in AXES}
    for (axis,fold),f in first.items():
        check_fold(f,axis,names)
        other = second[axis,fold]
        assert all(f[k] == other[k] for k in ("training","validation","installation_quaternion_wxyz"))
        state = fit_candidate(configs[0],f["training"])
        expected = ref["folds"][axis+":"+fold]
        assert sorted(f["training"]) == expected["training"] and sorted(f["validation"]) == expected["validation"]
        qa,qb = [Rotation.from_quat(q,scalar_first=True) for q in (state["installation_quaternion_wxyz"],expected["wxyz"])]
        assert np.rad2deg((qa.inv()*qb).magnitude()) < 2e-5
        for name in f["validation"]:
            cfg = configs[0]
            raw = learning.attach_native(name,cfg)
            part = learning.prepare(learning.apply_install(raw,state["installation_quaternion_wxyz"]),cfg)
            index = part["index"]
            assert hashlib.sha256(index.astype("<i8").tobytes()).hexdigest() == ref["source_records"][name]["score_index_sha256"]
            unconstrained = learning.design(part["matrix"],state) @ state["coef"]
            deltas = {cap:learning.constrain_delta(part,unconstrained,c) for cap,c in zip(CAPS,configs,strict=True)}
            counts = activity(unconstrained,part["up"],part["ready"],index,deltas)
            reproduced = [float(learning.evaluate(part,state,c)) for c in configs]
            source = [m[axis][name] for m in expected_scores]
            assert all(r["frames"] == len(index) and r["fold"] == fold for r in source)
            original = [r["rmse_deg"] for r in source]
            assert original[0] == original[1] and max(abs(u-v) for u,v in zip(reproduced,original)) <= 1e-5
            assert name not in validation[axis]
            validation[axis][name] = {**counts,"fold":fold,"recorded_deg":original,"reproduced_deg":reproduced}
    assert all(set(records) == names for records in validation.values())
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"rounds":list(IDS),
            "source_results_sha256":{r["round_id"]:digest(p) for r,p in zip(jobs,paths,strict=True)},
            "config_sha256":{r["round_id"]:r["config_sha256"] for r in jobs},"audit_refits":len(first),
            "axes":validation}


def summarize(result):
    assert result["rounds"] == list(IDS) and result["audit_refits"] == 8
    original = []
    for rid in IDS:
        job = read(BATCH/"rounds"/rid/"round.json")
        path = BATCH/job["result"]
        assert result["source_results_sha256"][rid] == digest(path) and result["config_sha256"][rid] == job["config_sha256"]
        original.append(read(path)["sequences"])
    output = {}
    for axis in AXES:
        measured = result["axes"][axis]
        scores = [{r["name"]:r for r in data[axis]} for data in original]
        assert set(measured) == set(scores[0]) == set(scores[1])
        for n,row in measured.items():
            assert row["recorded_deg"] == [s[n]["rmse_deg"] for s in scores] and row["recorded_deg"][0] == row["recorded_deg"][1]
            assert all(0 <= x <= 180 and abs(x-y) <= 1e-5 for x,y in zip(row["reproduced_deg"],row["recorded_deg"],strict=True))
            assert row["score_frames"] == scores[0][n]["frames"] == scores[1][n]["frames"]
            assert row["fold"] == scores[0][n]["fold"] == scores[1][n]["fold"]
            assert 0 < row["updates_used_for_scoring"] <= row["updates"] and 0 <= row["different_vectors_on_score_frames"] <= row["score_frames"]
            assert 0 <= row["maximum_unclipped_scored_deg"] <= row["maximum_unclipped_deg"] < 90
            assert 0 <= row["frames_before_first_update"] < row["score_frames"]
            assert 0 <= row["different_update_vectors"] <= row["updates"]
            assert row["different_vectors_on_score_frames"] <= row["score_frames"] - row["frames_before_first_update"]
            for cap in CAPS:
                counts = row["threshold_counts"][str(cap)]
                assert 0 <= counts["over_limit_updates_used_for_scoring"] <= counts["over_limit_updates"] <= row["updates"]
                assert counts["over_limit_updates_used_for_scoring"] <= counts["score_frames_with_over_limit_update"] <= row["score_frames"]
                if row["maximum_unclipped_deg"] < cap - 1e-5:
                    assert counts["over_limit_updates"] == 0
                if row["maximum_unclipped_deg"] > cap + 1e-5:
                    assert counts["over_limit_updates"] > 0
            at_six, at_eight = (row["threshold_counts"][str(c)] for c in CAPS)
            assert all(at_eight[k] <= v for k,v in at_six.items())
            assert row["different_update_vectors"] <= at_six["over_limit_updates"]
            assert row["different_vectors_on_score_frames"] <= at_six["score_frames_with_over_limit_update"]
        output[axis] = {"updates":sum(r["updates"] for r in measured.values()),"score_frames":sum(r["score_frames"] for r in measured.values()),
                        "different_updates":sum(r["different_update_vectors"] for r in measured.values()),
                        "different_on_score_frames":sum(r["different_vectors_on_score_frames"] for r in measured.values()),
                        "largest_unclipped_deg":max(r["maximum_unclipped_deg"] for r in measured.values()),
                        "largest_unclipped_scored_deg":max(r["maximum_unclipped_scored_deg"] for r in measured.values()),
                        "updates_exceeding_six":sum(r["threshold_counts"]["6"]["over_limit_updates"] for r in measured.values())}
    return output


if __name__ == "__main__":
    result = replay()
    sums = summarize(result)
    write(HERE/"ridge_6_8_cap_replay.json",result)
    print(sums)
