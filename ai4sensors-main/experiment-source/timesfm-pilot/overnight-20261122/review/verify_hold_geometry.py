"""Reconstruct the two P00022 hold rules from raw SHICE and pre-existing V00001 outputs."""

import hashlib
import math
from datetime import datetime, timezone
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from audit_rounds import BATCH, HERE, check_fold, check_scores, read, write
from verify_final import digest, raw_shice_samples


UP = (0., 0., 1.)
STAGE = BATCH / "sensor-frame-hold-development-protocol.json"
INITIAL_PROTOCOL_SHA256 = "6df93f915abef12de2ef882838c308dc1e2576615f2b075d9d1ef47b90793fb4"
AGE_LABELS = ("0_to_25ms","25_to_50ms","50_to_75ms","75_to_100ms","at_least_100ms","before_first_correction")
SPEED_LABELS = ("under_1","1_to_20","20_to_60","60_to_120","at_least_120")


def angles(left, right):
    cross = np.linalg.norm(np.cross(left, right), axis=-1)
    dot = np.einsum("ij,ij->i", left, right)
    return np.arctan2(cross, dot)


def slices(labels, codes, world_errors, body_errors):
    out = {}
    for i, label in enumerate(labels):
        mask = codes == i
        out[label] = {"frames":int(mask.sum())}
        for key, errors in (("world_rmse_deg",world_errors),("sensor_rmse_deg",body_errors)):
            out[label][key] = float(np.sqrt(np.mean(errors[mask]**2))) if mask.any() else None
    return out


def replay_record(cert, parent, config):
    name = cert["name"]
    fold = next(row for row in parent["folds"] if row["axis"] == cert["fold_axis"] and row["fold"] == cert["fold"])
    assert name in fold["validation"] and name not in fold["training"] and fold["training"] == cert["training"]
    raw = raw_shice_samples(name)
    imu = raw["imu"].astype(np.float64)
    native = VQF(float(raw["dt"]), **config["parameters"]).updateBatch(
        np.ascontiguousarray(imu[:, 3:]), np.ascontiguousarray(imu[:, :3]))["quat6D"]
    mount = Rotation.from_quat(fold["installation_quaternion_wxyz"], scalar_first=True)
    baseline = Rotation.from_quat(native, scalar_first=True) * mount
    baseline_up = baseline.inv().apply(UP)
    export = BATCH / cert["quaternion_file"]
    assert digest(export) == cert["quaternion_sha256"]
    with np.load(export, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_index"], np.arange(len(imu)))
        np.testing.assert_array_equal(saved["time_s"], raw["time_s"])
        head_up = Rotation.from_quat(saved["q_wxyz"], scalar_first=True).inv().apply(UP)

    # The SHICE frontend closes each 100 Hz input block after ten original rows.
    # P00022 starts at the first 10 Hz release with 64 complete input blocks.
    assert config["update_hz"] == 10 and config["context"] == 64
    release = np.arange(699, len(imu), 100)
    active = np.searchsorted(release, np.arange(len(imu)), side="right")
    update_p = head_up[release]
    update_u = baseline_up[release]

    world_at_update = baseline[release].apply(update_p)
    held_world = np.vstack((UP, world_at_update))[active]
    world_prediction = baseline.inv().apply(held_world)

    axes = np.cross(update_u, update_p)
    lengths = np.linalg.norm(axes, axis=1)
    fit_angles = angles(update_u, update_p)
    assert np.all(fit_angles < np.pi / 2)
    scales = np.divide(fit_angles, lengths, out=np.zeros_like(lengths), where=lengths > 1e-15)
    rotation_vectors = np.vstack((np.zeros(3), axes * scales[:, None]))
    body_prediction = Rotation.from_rotvec(rotation_vectors[active]).apply(baseline_up)

    match = np.rad2deg(angles(world_prediction, head_up))
    at_release = np.rad2deg(angles(body_prediction[release], update_p))
    assert match.max() < 0.0001 and at_release.max() < 1e-8, (name, match.max(), at_release.max())
    score = raw["score"]
    assert int(score.sum()) == cert["scored_frames"]
    true = raw["reference"][score]
    world_errors = np.rad2deg(angles(true, world_prediction[score]))
    body_errors = np.rad2deg(angles(true, body_prediction[score]))
    world_score = float(np.sqrt(np.mean(world_errors**2)))
    body_score = float(np.sqrt(np.mean(body_errors**2)))
    assert math.isclose(world_score, cert["direct_direction_rmse_deg"], abs_tol=2e-5)
    frame_age = (raw["time_s"][score] - raw["time_s"][release[(active[score]-1).clip(min=0)]]) * 1000
    age_bins = np.searchsorted((25,50,75,100), frame_age, side="right")
    age_bins[active[score] == 0] = len(AGE_LABELS) - 1
    angular_speed = np.rad2deg(np.linalg.norm(raw["imu"][:, 3:], axis=1))
    prefix = np.r_[0., np.cumsum(angular_speed, dtype=np.float64)]
    ends = np.flatnonzero(score) + 1
    starts = np.maximum(0, ends - round(0.1 / raw["dt"]))
    speed_bins = np.searchsorted((1,20,60,120), (prefix[ends]-prefix[starts]) / (ends-starts), side="right")
    return {"name":name, "axis":cert["fold_axis"], "fold":cert["fold"], "source":str(raw["source"]),
            "source_sha256":digest(raw["source"]), "export_sha256":cert["quaternion_sha256"],
            "native_frames":len(imu), "scored_frames":int(score.sum()), "releases":len(release),
            "score_indices_sha256":hashlib.sha256(np.flatnonzero(score).astype("<i8").tobytes()).hexdigest(),
            "control_max_direction_error_deg":float(match.max()),
            "body_max_release_error_deg":float(at_release.max()),
            "world_rmse_deg":world_score, "body_rmse_deg":body_score,
            "body_minus_world_rmse_deg":body_score - world_score,
            "score_by_update_age":slices(AGE_LABELS,age_bins,world_errors,body_errors),
            "score_by_speed":slices(SPEED_LABELS,speed_bins,world_errors,body_errors)}


def replay():
    parent = read(BATCH / "rounds/P00022/result.json")
    config = read(BATCH / "rounds/P00022/round.json")["config"]
    exports = read(BATCH / "rounds/V00001/result.json")
    assert exports["parent_round"] == "P00022"
    assert [(c["name"], c["fold"]) for c in exports["records"]] == [("R2-10","R2"), ("R4-2","M2")]
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(), "source_candidate":"P00022",
            "quaternion_round":"V00001", "new_head_fits":0,
            "records":[replay_record(c, parent, config) for c in exports["records"]]}


def check_job(row, result, frames):
    cfg = row["config"]
    protocol = read(STAGE)
    assert digest(STAGE) == INITIAL_PROTOCOL_SHA256 and protocol["development_only"]
    assert datetime.fromisoformat(protocol["created_at"]) <= datetime.fromisoformat(row["started_at"])
    assert result["round_id"] == row["round_id"] and result["parent_round"] == cfg["parent_round"]
    parent = read(BATCH / "rounds" / cfg["parent_round"] / "round.json")
    assert parent["seed"] == row["seed"] == result["paired_checkpoint_seed"]
    assert parent["config_sha256"] == cfg["head_sha256"] and result["ranked_separately_from_model_configs"]
    if row["round_id"] == protocol["queued_round"]:
        assert cfg["parent_round"] == protocol["source_round"] and cfg["head_sha256"] == protocol["checkpoint_reference_sha256"]
    source = read(BATCH / parent["result"])["sequences"]
    joint = read(BATCH / "rounds" / cfg["joint_round"] / "result.json")
    controls = {**source, "joint_holdout":joint["sequences"]["joint_holdout"]}
    axes = ("device","motion","joint_holdout")
    assert set(result["sequences"]) == set(axes) == set(result["metrics"])
    n_folds = {"device":4, "motion":4, "joint_holdout":16}
    assert len(result["folds"]) == result["real_head_fits"] == sum(n_folds.values())
    assert all(f["shared_head_fits"] == 1 for f in result["folds"])
    max_gap = 0.
    comparison = {}
    for axis in axes:
        scores = result["sequences"][axis]
        reference = {r["name"]:r for r in controls[axis]}
        folds = [f for f in result["folds"] if f["axis"] == axis]
        assert len(folds) == n_folds[axis] and len({f["fold"] for f in folds}) == n_folds[axis]
        memberships = [(name, f["fold"]) for f in folds for name in check_fold(f,axis,set(frames))]
        assert len(scores) == len(memberships) == len(frames)
        assert set(memberships) == {(r["name"], r["fold"]) for r in scores}
        values = result["metrics"][axis]
        check_scores(scores,values["sensor_hold"],frames)
        check_scores([{**r,"rmse_deg":r["baseline_deg"]} for r in scores],values["world_hold"],frames)
        for r in scores:
            old = reference[r["name"]]
            assert r["fold"] == old["fold"] and r["frames"] == old["frames"]
            assert math.isclose(old["rmse_deg"],r["world_historical_rmse_deg"],abs_tol=1e-9)
            gap = abs(r["baseline_deg"] - old["rmse_deg"])
            assert math.isclose(gap,r["world_replay_abs_delta_deg"],abs_tol=1e-9) and gap <= 1e-4
            max_gap = max(max_gap, gap)
            for field, labels in (("score_by_speed",SPEED_LABELS),("score_by_update_age",AGE_LABELS)):
                cells = r[field]
                assert set(cells) == set(labels) and sum(c["frames"] for c in cells.values()) == r["frames"]
                for arm, total in (("world",r["baseline_deg"]),("sensor",r["rmse_deg"])):
                    sse = 0.
                    for cell in cells.values():
                        sq = cell[f"{arm}_squared_error"]
                        n = cell["frames"]
                        assert n >= 0 and sq >= 0
                        value = cell[f"{arm}_rmse_deg"]
                        if n:
                            assert math.isclose(math.sqrt(sq/n), value, abs_tol=1e-7)
                        else:
                            assert value is None and sq == 0
                        sse += sq
                    assert math.isclose(math.sqrt(sse/r["frames"]),total,abs_tol=1e-7)
        world = values["world_hold"]["macro_rmse_deg"]
        body = values["sensor_hold"]["macro_rmse_deg"]
        assert math.isclose(body-world,values["sensor_minus_world_deg"],abs_tol=1e-8)
        comparison[axis] = {"world_deg":world, "sensor_deg":body, "delta_deg":body-world}
    limits = protocol["promotion"]
    passed = (comparison["joint_holdout"]["delta_deg"] <= -limits["minimum_joint_improvement_deg"]
              and all(comparison[a]["delta_deg"] <= limits["maximum_allowed_axis_regression_deg"] for a in ("device","motion")))
    assert passed == result["promotion_screen_passed"] and passed == (not result["nonpromotion_reasons"])
    assert math.isclose(max_gap,result["maximum_world_reproduction_delta_deg"],abs_tol=1e-8)
    assert result["synthetic_checks"] == protocol["quaternion_checks"]
    prefix = result["native_prefix_check"]
    assert prefix["input_inference_before_any_score_mask"] and prefix["prefix_frames"] > 0
    assert max(prefix["world_max_quat_difference_deg"],prefix["sensor_max_quat_difference_deg"]) <= .001
    summary_path = BATCH / "sensor-frame-hold-development-summary.json"
    if row["round_id"] == protocol["queued_round"] and summary_path.is_file():
        summary = read(summary_path)
        assert summary["round_id"] == row["round_id"] and set(summary["axes"]) == set(axes)
        for axis in axes:
            published = summary["axes"][axis]
            assert all(published[k] == value for k,value in result["metrics"][axis].items())
            for field, labels in (("score_by_speed",SPEED_LABELS),("score_by_update_age",AGE_LABELS)):
                assert set(published[field]) == set(labels)
                for label in labels:
                    cells = [r[field][label] for r in result["sequences"][axis] if r[field][label]["frames"]]
                    pooled = published[field][label]
                    total = sum(c["frames"] for c in cells)
                    assert total == pooled["frames"] and len(cells) == pooled["records"]
                    for arm in ("world","sensor"):
                        macro = pooled[f"{arm}_macro_rmse_deg"]
                        micro = pooled[f"{arm}_pooled_rmse_deg"]
                        if total:
                            assert math.isclose(macro,fmean(c[f"{arm}_rmse_deg"] for c in cells),abs_tol=1e-8)
                            assert math.isclose(micro,math.sqrt(sum(c[f"{arm}_squared_error"] for c in cells)/total),abs_tol=1e-8)
                        else:
                            assert macro is None and micro is None
                    change = pooled["macro_sensor_minus_world_deg"]
                    if total:
                        assert math.isclose(change,pooled["sensor_macro_rmse_deg"]-pooled["world_macro_rmse_deg"],abs_tol=1e-8)
                    else:
                        assert change is None
        assert summary["result_sha256"] == digest(BATCH / row["result"]) and summary["head_fits"] == result["real_head_fits"]
        assert summary["promotion_screen_passed"] == passed
        if not passed:
            assert summary["next_seeds_launched"] == 0
    raw_checks = []
    if cfg["parent_round"] == "P00022":
        evidence = read(HERE / "hold_geometry_replay.json")
        assert evidence["new_head_fits"] == 0 and evidence["source_candidate"] == "P00022"
        for raw in evidence["records"]:
            other = next(r for r in result["sequences"][raw["axis"]] if r["name"] == raw["name"])
            assert raw["scored_frames"] == other["frames"] and raw["fold"] == other["fold"]
            gaps = [abs(raw["world_rmse_deg"] - other["baseline_deg"]), abs(raw["body_rmse_deg"] - other["rmse_deg"])]
            for field in ("score_by_speed","score_by_update_age"):
                for label, actual in raw[field].items():
                    cell = other[field][label]
                    assert actual["frames"] == cell["frames"]
                    if cell["frames"]:
                        gaps.extend(abs(actual[k]-cell[k]) for k in ("world_rmse_deg","sensor_rmse_deg"))
            assert max(gaps) <= 2e-5, (raw["name"], max(gaps))
            raw_checks.append({"name":raw["name"], "frames":other["frames"], "max_rmse_gap_deg":max(gaps)})
    return {"id":row["round_id"], "parent":cfg["parent_round"], "seed":row["seed"], "fits":result["real_head_fits"],
            "axes":comparison, "screen_passed":passed, "source_reproduction_max_gap_deg":max_gap, "raw_checks":raw_checks}


if __name__ == "__main__":
    evidence = replay()
    write(HERE / "hold_geometry_replay.json", evidence)
    for row in evidence["records"]:
        print(row["name"], row["fold"], row["scored_frames"], row["world_rmse_deg"], row["body_rmse_deg"])
