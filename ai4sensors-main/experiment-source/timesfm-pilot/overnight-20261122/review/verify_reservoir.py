"""Verify the frozen random reservoir, matched controls and native development folds."""

import hashlib
from collections import Counter
from datetime import datetime
from math import isclose
from statistics import fmean

import numpy as np

from audit_rounds import BATCH, HERE, check_fold, check_scores, read
from verify_baselines import digest


PROTOCOL = BATCH / "reservoir-development-protocol.json"
REGISTERED_SHA = "0ba3337071c1efec593c0f051a27f58725abdd5cb86218d450503c853a45aef6"
ARMS = {"continuous":"rmse_deg","reset_each_update":"reset_each_update_rmse_deg","legacy_head":"legacy_head_rmse_deg"}


def fixed_matrices(protocol):
    planned, model = protocol["first_round"], protocol["shared_model"]
    path = BATCH/planned["matrix_file"]
    assert digest(path) == planned["matrix_file_sha256"]
    width, inputs = model["width"], model["input_width"]
    assert (width,inputs,model["leak"]) == (32,34,.5)
    rng = np.random.Generator(np.random.PCG64(planned["seed"]))
    drive = rng.normal(0,.55/np.sqrt(inputs),(width,inputs))
    recurrent = rng.normal(0,1/np.sqrt(width),(width,width))
    recurrent *= .8/np.linalg.svd(recurrent,compute_uv=False)[0]
    generated = {"drive":drive,"recurrent":recurrent,"bias":rng.uniform(-.15,.15,width)}
    label = hashlib.sha256()
    with np.load(path,allow_pickle=False) as saved:
        assert set(saved.files) == set(generated)
        for key in sorted(generated):
            np.testing.assert_allclose(saved[key],generated[key],rtol=1e-13,atol=1e-14)
            label.update(key.encode("utf-8"))
            label.update(np.asarray(saved[key],dtype="<f8").tobytes())
        upper = (1-model["leak"]) + model["leak"]*np.linalg.norm(saved["recurrent"],2)
        radius = float(np.abs(np.linalg.eigvals(saved["recurrent"])).max())
    assert isclose(upper,model["contraction_upper_bound"],abs_tol=1e-12) and upper < 1
    assert isclose(radius,model["measured_spectral_radius"],abs_tol=1e-12)
    assert label.hexdigest() == planned["matrix_content_sha256"]
    return label.hexdigest()


def check_trial(row, result, frames):
    protocol = read(PROTOCOL)
    assert digest(PROTOCOL) == row["config"]["protocol_sha256"] == REGISTERED_SHA
    plan = protocol["first_round"]
    assert row["round_id"] == plan["round_id"] and row["seed"] == result["matched_seed"] == plan["seed"]
    assert datetime.fromisoformat(protocol["registered_at"]) <= datetime.fromisoformat(row["started_at"])
    assert result["reservoir_parameter_sha256"] == fixed_matrices(protocol)
    for path,sha in plan["anchor_result_sha256"].items():
        assert digest(BATCH/path) == sha
    matched, joint = (read(BATCH/"rounds"/plan[k]/"result.json") for k in ("head_round","joint_round"))
    assert joint["parent_round"] == plan["head_round"]
    controls = {**matched["sequences"],**joint["sequences"]}
    control_folds = {(f["axis"],f["fold"]):f for f in matched["folds"]+joint["folds"]}
    inputs = {r["name"]:r for r in read(HERE/"recurrent_context_audit.json")["records"]}
    names = set(frames)
    assert set(row["partitions"]["development"]) == set(inputs) == names
    folds = {(f["axis"],f["fold"]):f for f in result["folds"]}
    assert len(folds) == len(result["folds"]) == 24 and set(folds) == set(control_folds)
    assert Counter(axis for axis,_ in folds) == {"device":4,"motion":4,"joint_holdout":16}
    for (axis,label), fold in folds.items():
        check_fold(fold,axis,names)
        anchor = control_folds[axis,label]
        q, reference = (np.asarray(v) for v in (fold["mount_wxyz"],anchor["installation_quaternion_wxyz"]))
        assert min(np.linalg.norm(q-reference),np.linalg.norm(q+reference)) < 1e-6
        assert fold["model_fits"] == 2
        assert fold["training_labels"] == {n:inputs[n]["eligible_updates"] for n in fold["training"]}
        assert fold["training_state_segments"] == {n:inputs[n]["disjoint_context_periods_containing_eligible_updates"] for n in fold["training"]}
    assert result["actual_readout_fits"] == 2*len(folds) and result["legacy_head_refits"] == 0
    frame_hashes = {}
    for n,count in frames.items():
        with np.load(BATCH.parent/"gru-tilt/data"/(n+".npz"),allow_pickle=False) as native:
            indices = np.asarray(np.flatnonzero(native["score"]),dtype="<i8")
        assert len(indices) == count
        frame_hashes[n] = hashlib.sha256(indices.tobytes()).hexdigest()
    scores = {}
    for axis, records in result["sequences"].items():
        metric = result["axis_metrics"][axis]
        assert {r["name"] for r in records} == names
        old = {r["name"]:r for r in controls[axis]}
        for r in records:
            n = r["name"]
            assert r["score_frame_indices_sha256"] == frame_hashes[n] and n in folds[axis,r["fold"]]["validation"]
            assert old[n]["rmse_deg"] == r["legacy_head_rmse_deg"] and old[n]["frames"] == r["frames"]
        scores[axis] = {}
        for arm,key in ARMS.items():
            measured = [{**r,"rmse_deg":r[key]} for r in records]
            check_scores(measured,metric[arm],frames)
            scores[axis][arm] = fmean(r["rmse_deg"] for r in measured)
        for arm,key in (("reset_each_update","continuous_minus_reset_deg"),("legacy_head","continuous_minus_legacy_deg")):
            assert isclose(scores[axis]["continuous"]-scores[axis][arm],metric[key],abs_tol=1e-10)
    threshold = protocol["promotion"]
    assert threshold["joint_minimum_improvement_deg"] == threshold["axis_maximum_regression_deg"] == .020
    joint_scores = scores["joint_holdout"]
    criteria = {k:joint_scores["continuous"] <= joint_scores[v]-.020 for k,v in
                (("joint_beats_reset","reset_each_update"),("joint_beats_legacy","legacy_head"))}
    criteria.update({k:scores[a]["continuous"] <= scores[a]["legacy_head"]+.020 for k,a in
                     (("held_device_acceptable","device"),("held_program_acceptable","motion"))})
    assert criteria == result["promotion_checks"] and all(criteria.values()) == result["promotion_screen_passed"]
    assert result["admitted_for_r5"] == []
    prefix = result["native_prefix_check"]
    assert prefix["record"] == "R2-10" and set(prefix["arms"]) == {"continuous","reset_each_update"}
    assert all(s["scored_frames"] > 0 and 0 <= s["maximum_direction_change_deg"] <= 1e-4 for s in prefix["arms"].values())
    return {"id":row["round_id"],"seed":row["seed"],"member_fits":result["actual_readout_fits"],"scores":scores,
            "records":len(names),"native_frames":sum(frames.values()),"head":plan["head_round"],"criteria":criteria,
            "passed":all(criteria.values()),"prefix":prefix,"matrix_sha256":result["reservoir_parameter_sha256"]}
