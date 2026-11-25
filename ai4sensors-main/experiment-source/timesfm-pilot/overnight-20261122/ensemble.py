"""Predeclared equal-weight averages of independent native-frame tilt estimates."""

import os
from hashlib import sha256
from itertools import combinations

import numpy as np

from common import BATCH, EXCLUDED, HISTORY, aggregate, body_up, by_device, digest, errors, fingerprint, names, read_json, rmse, stamp, unit, write_json
from experiment import baseline_reference
from joint_holdout import joint_folds
import learning


PROTOCOL = {
    "id": "equal-weight-native-tilt-v1",
    "fixed_weights": [0.5, 0.5],
    "common_frame": "sensor/body coordinates at the original native score frame",
    "blend_rule": "normalize the weighted sum of unit gravity-direction vectors",
    "causal_validation_axes": ["device", "motion", "joint_holdout"],
    "r5_inclusion_min_joint_improvement_deg": 0.02,
    "r5_inclusion_max_single_axis_degradation_deg": 0.02,
}
DISTINCT_FILTER_PROTOCOL = BATCH / "independent-filter-ensemble-development-protocol.json"
THREE_SEED_PROTOCOL = BATCH / "three-seed-ensemble-development-protocol.json"


def validate(config):
    members = config["members"]
    independent = config.get("independent_filters", False)
    three = config.get("shared_head_seed_ensemble", False)
    weights = [1. / 3.] * 3 if three else PROTOCOL["fixed_weights"]
    if [m["weight"] for m in members] != weights:
        raise ValueError("The ensemble must use the frozen equal-weight members")
    if three:
        settings = [{key: value for key, value in member["config"].items() if key != "seed"} for member in members]
        if independent or any(value != settings[0] for value in settings[1:]) or len({m["config"]["seed"] for m in members}) != 3:
            raise ValueError("The three heads must share one configuration and use distinct initialization seeds")
    for member in members:
        kinds = {"legacy_head"} if three else ({"legacy_head", "ridge"} if independent else {"legacy_head", "tcn", "install_rotation"})
        if member["config"]["kind"] not in kinds or (independent and member["config"].get("embedding", 0)):
            raise ValueError("Unsupported ensemble method")
        if not independent and member["config"]["parameters"] != config["parameters"]:
            raise ValueError("The two members must share native FullVQF filter parameters")
        source = read_json(BATCH / "rounds" / member["round_id"] / "round.json")
        if source["status"] != "completed" or fingerprint(member["config"]) != source["config_sha256"]:
            raise ValueError(f"Frozen component configuration changed: {member['round_id']}")
    return members


def fit(config, training, raw=None):
    from final_eval import fit_candidate

    members = validate(config)
    packed = {}
    if config.get("shared_head_seed_ensemble"):
        import legacy_head
        import torch

        stream = {name: learning.attach_native(name, config) for name in training} if raw is None else raw
        common = members[0]["config"]
        mount = learning.installation_fit([stream[name] for name in training], common)
        prepared = [legacy_head.prepare(learning.apply_install(stream[name], mount, include_reference=False), common)
                    for name in training]
        device = torch.device("cuda")
        for index, member in enumerate(members):
            state, _ = legacy_head.fit(prepared, member["config"], device, training)
            packed.update({f"member{index}__{key}": value for key, value in state.items()})
            packed[f"member{index}__installation_quaternion_wxyz"] = mount
        return packed
    for index, member in enumerate(members):
        state = fit_candidate(member["config"], training)
        packed.update({f"member{index}__{key}": value for key, value in state.items()})
    return packed


def member_state(state, index):
    prefix = f"member{index}__"
    return {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}


def build(config, state, device):
    result = []
    for index, member in enumerate(config["members"]):
        model_config = member["config"]
        own = member_state(state, index)
        model = None
        if model_config["kind"] == "legacy_head":
            from legacy_head import make_model

            model = make_model(own, device)
        if model_config["kind"] == "tcn":
            from temporal import make_model

            model = make_model(own, model_config, device)
        result.append((model_config, own, model))
    return result


def member_directions(data, config, state, model, device, delay_ms=0.):
    kind = config["kind"]
    if kind == "install_rotation":
        from install_rotation import frame_directions

        estimate, _ = frame_directions(data, state, config)
        if config["coordinates"] == "body":
            return estimate
        return body_up(data["q_scored"], estimate)
    mounting = state.get("installation_quaternion_wxyz")
    adjusted = learning.apply_install(data, mounting, include_reference=False) if mounting is not None else data
    frame_q = data["q_scored"]
    if mounting is not None:
        frame_q = learning.multiply_quaternion(frame_q, mounting)
    if kind == "legacy_head":
        import legacy_head

        direction, _ = legacy_head.frame_directions(legacy_head.prepare(adjusted, config), config, model, device, delay_ms)
    elif kind == "tcn":
        import temporal

        direction, _ = temporal.frame_directions(temporal.prepare(adjusted, config), config, model, device, delay_ms)
    elif kind == "ridge":
        direction, _ = learning.frame_directions(learning.prepare(adjusted, config), state, config, delay_ms)
    else:
        raise ValueError(f"Cannot produce a body-frame direction for {kind}")
    return body_up(frame_q, direction)


def blend(first, second, *others):
    combined = first + second
    for direction in others:
        combined += direction
    return np.where((np.linalg.norm(combined, axis=-1) > 1e-12)[..., None], unit(combined), first)


def shared_head_directions(data, branches, device, delay_ms=0.):
    import legacy_head

    mount = branches[0][1]["installation_quaternion_wxyz"]
    if any(not np.array_equal(own["installation_quaternion_wxyz"], mount) for _, own, _ in branches):
        raise ValueError("Shared-head ensemble has different fitted mounting rotations")
    prepared = legacy_head.prepare(learning.apply_install(data, mount, include_reference=False), branches[0][0])
    output_q = learning.multiply_quaternion(data["q_scored"], mount)
    return [unit(body_up(output_q, legacy_head.frame_directions(prepared, cfg, model, device, delay_ms)[0]))
            for cfg, _, model in branches]


def score(data, config, state, device, delay_ms=0.):
    model = build(config, state, device)
    if config.get("shared_head_seed_ensemble"):
        predictions = shared_head_directions(data, model, device, delay_ms)
    else:
        predictions = [member_directions(data, cfg, owned, branch, device, delay_ms) for cfg, owned, branch in model]
    return rmse(blend(*predictions), data["ref_body_scored"])


def planned_folds(records):
    stages = [(axis, fold, [n for n in records if n not in held], held) for axis, fold, held in learning.folds(records)]
    stages.extend(("joint_holdout", label, training, held) for label, training, held in joint_folds(records))
    return stages


def measured_prior(member):
    parent = member["round_id"]
    historical = read_json(BATCH / "rounds" / parent / "result.json")["sequences"]
    reference = read_json(BATCH / "rounds" / member["joint_round"] / "result.json")
    joint = reference["sequences"]
    expected = {axis: {row["name"]: row for row in rows} for axis, rows in historical.items()}
    if isinstance(joint, dict):
        joint = joint["joint_holdout"]
    expected["joint_holdout"] = {row["name"]: row for row in joint}
    return expected


def correlation(first, second, first_error, second_error, truth):
    angular_corr = None
    if np.std(first_error) > 1e-9 and np.std(second_error) > 1e-9:
        angular_corr = float(np.corrcoef(first_error, second_error)[0, 1])
    a = first - (first * truth).sum(axis=1, keepdims=True) * truth
    b = second - (second * truth).sum(axis=1, keepdims=True) * truth
    denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
    tangent_agreement = float(np.sum(a*b) / denominator) if denominator > 1e-10 else None
    return angular_corr, tangent_agreement


def bench(config, state, device, benchmark_record):
    from vqf import VQF

    import temporal
    from common import Z, record, rotate
    from features import physics
    from final_eval import timed
    from tilt_eval import predict_updates
    from tilt_model import physical_features

    branches = build(config, state, device)
    data = learning.attach_native(benchmark_record, config)
    streams = [learning.apply_install(data, part.get("installation_quaternion_wxyz"), False)
               if part.get("installation_quaternion_wxyz") is not None else data for _, part, _ in branches]
    positions = np.linspace(150, len(data["windows"])-1, 110, dtype=int)

    def update_one(index, cfg, own, model, source):
        if cfg["kind"] == "install_rotation":
            return None
        end = positions[index] + 1
        q = source["q"][end-1:end]
        if cfg["kind"] == "legacy_head":
            inputs = physical_features(source["windows"][None,end-64:end], source["up"][end-1:end],
                                       source["bias"][end-1:end], source["rest"][end-1:end])
            residual = predict_updates(model, inputs, None, device) * cfg["gain"]
        else:
            local = {key: source[key][end-70:end] for key in ("windows", "up", "q", "bias", "rest")}
            where, matrix, acc, gyr = physics(local, cfg)
            selected = where[-1:]
            inputs = {"history": temporal.windows(local["windows"], selected, cfg["context"]), "matrix": matrix[-1:],
                      "up": local["up"][selected], "acc": acc[-1:], "gyr": gyr[-1:]}
            residual = temporal.inference(inputs, *model, cfg, device)
        return unit(Z + rotate(unit(q), residual))[0]

    def periodic(i):
        return [update_one(i, cfg, own, model, stream) for (cfg, own, model), stream in zip(branches, streams, strict=True)]

    held = periodic(0)
    samples = record(benchmark_record)
    dt = float(samples["dt"])
    accel = np.ascontiguousarray(samples["imu"][:, :3], dtype=np.float64)
    gyro = np.ascontiguousarray(samples["imu"][:, 3:], dtype=np.float64)
    base = VQF(dt, **config["parameters"])
    start = 22000
    base.updateBatch(gyro[:start], accel[:start])

    def native_step(i):
        j = start + i
        base.update(gyro[j], accel[j])
        q = base.getQuat6D()
        values = []
        for (cfg, own, _), world in zip(branches, held, strict=True):
            if cfg["kind"] == "install_rotation":
                if cfg["coordinates"] == "body":
                    values.append(body_up(q) @ own["fitted"])
                else:
                    values.append(body_up(q, own["fitted"]))
                continue
            if own.get("installation_quaternion_wxyz") is not None:
                adjusted = learning.multiply_quaternion(q, own["installation_quaternion_wxyz"])
            else:
                adjusted = q
            values.append(body_up(adjusted, world))
        return blend(*values)

    return {"record": benchmark_record, "nominal_native_hz": 1./dt, "correction_hz": 1./(100*dt),
            "shared_native_vqf_filters": 1, "native_step": timed(native_step, 3000, 100),
            "correction_update": timed(periodic, 100, 10),
            "native_step_includes_held_corrections_and_body_direction_blend": True}


def run(item):
    import torch

    config = item["config"]
    members = validate(config)
    independent = config.get("independent_filters", False)
    three = config.get("shared_head_seed_ensemble", False)
    declared_protocol = {**PROTOCOL, "fixed_weights": [1. / 3.] * 3} if three else PROTOCOL
    expected = [measured_prior(member) for member in members]
    ids = [m["round_id"] for m in members]
    baseline = baseline_reference()
    records = names("dev")
    device = torch.device("cuda")
    if independent or three:
        path, key = ((DISTINCT_FILTER_PROTOCOL, "independent_filters_protocol_sha256") if independent
                     else (THREE_SEED_PROTOCOL, "three_seed_protocol_sha256"))
        if digest(path) != config[key]:
            raise ValueError("Frozen ensemble protocol changed")
        registered = read_json(path)
        if registered["members"] != members or registered["criteria"] != declared_protocol:
            raise ValueError("Registered members or thresholds do not match the running ensemble")
        for relative, sha in registered["anchor_result_sha256"].items():
            if digest(BATCH / relative) != sha:
                raise ValueError(f"Frozen historical member scores changed: {relative}")
        print("ensemble", item["round_id"], "python worker", os.getpid(), "starting", stamp(), flush=True)
    if independent:
        left, right = [member["config"] for member in members]
        if left["parameters"] == right["parameters"]:
            raise ValueError("Independent-filter experiment requires distinct FullVQF configurations")
        streams = {m["round_id"]: {name: learning.attach_native(name, m["config"]) for name in records} for m in members}
        for name in records:
            first_data, second_data = [streams[key][name] for key in ids]
            np.testing.assert_array_equal(first_data["index"], second_data["index"])
            np.testing.assert_array_equal(first_data["ref_body_scored"], second_data["ref_body_scored"])
    else:
        raw = {name: learning.attach_native(name, config) for name in records}
        streams = {key: raw for key in ids}
    common_index_sha = {}
    if three:
        for name, data in raw.items():
            with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as original:
                np.testing.assert_array_equal(data["index"], np.flatnonzero(original["score"]))
            common_index_sha[name] = sha256(np.asarray(data["index"], dtype="<i8").tobytes()).hexdigest()
    selections, folds = {}, []
    for axis, fold, training, held in planned_folds(records):
        state = fit(config, training, raw=raw if three else None)
        fitted = build(config, state, device)
        fold_rows = []
        for name in held:
            truth = streams[ids[0]][name]["ref_body_scored"]
            if three:
                directions = shared_head_directions(raw[name], fitted, device)
            else:
                directions = [member_directions(streams[key][name], cfg, owned, model, device)
                              for key, (cfg, owned, model) in zip(ids, fitted, strict=True)]
                if independent:
                    directions = [unit(direction) for direction in directions]
            angular_errors = [errors(direction, truth) for direction in directions]
            member_rmse = [float(np.sqrt(np.square(delta, dtype=np.float64).mean())) for delta in angular_errors]
            source = [entry[axis][name] for entry in expected]
            if any(row["frames"] != len(truth) or abs(row["rmse_deg"] - new) > 1e-4
                   for row, new in zip(source, member_rmse, strict=True)):
                raise ValueError(f"Body-frame conversion did not reproduce original member scores in {axis}/{name}")
            pairs = [correlation(directions[a], directions[b], angular_errors[a], angular_errors[b], truth)
                     for a, b in combinations(range(len(directions)), 2)]
            angular, tangent = (float(np.mean(values)) if values else None
                                for values in ([x for x in column if x is not None] for column in zip(*pairs, strict=True)))
            row = {"name": name, "fold": fold, "frames": len(truth), "baseline_deg": baseline[name],
                   "rmse_deg": rmse(blend(*directions), truth), "members_rmse_deg": dict(zip(ids, member_rmse, strict=True)),
                   "angular_error_pearson_r": angular, "tangent_error_cosine": tangent}
            if independent or three:
                row["historical_member_rmse_deg"] = {key: original["rmse_deg"] for key, original in zip(ids, source, strict=True)}
                row["member_reproduction_abs_delta_deg"] = {
                    key: abs(original["rmse_deg"] - new) for key, original, new in zip(ids, source, member_rmse, strict=True)}
            if three:
                row["score_frame_indices_sha256"] = common_index_sha[name]
            fold_rows.append(row)
        selections.setdefault(axis, []).extend(fold_rows)
        evidence = {"axis": axis, "fold": fold, "training": training, "validation": held, "metrics": aggregate(fold_rows)}
        if independent or three:
            evidence["actual_model_fits"] = {key: 1 for key in ids}
            evidence["body_installation_by_member_wxyz"] = {
                key: own["installation_quaternion_wxyz"].tolist() if "installation_quaternion_wxyz" in own else None
                for key, (_, own, _) in zip(ids, fitted, strict=True)}
        folds.append(evidence)
        print(axis, fold, folds[-1]["metrics"]["macro_rmse_deg"], flush=True)

    results, qualification = {}, {}
    for axis, rows in selections.items():
        if sorted(row["name"] for row in rows) != sorted(records):
            raise ValueError(f"Some original scored records are missing in {axis}")
        result = {**aggregate(rows), "per_device": by_device(rows)}
        member_scores = {"members_macro_rmse_deg": {key: float(np.mean([r["members_rmse_deg"][key] for r in rows])) for key in ids}}
        better = min(member_scores["members_macro_rmse_deg"].values())
        qualification[axis] = result["macro_rmse_deg"] - better
        for key in ("angular_error_pearson_r", "tangent_error_cosine"):
            values = [row[key] for row in rows if row[key] is not None]
            result[f"mean_per_record_{key}"] = float(np.mean(values)) if values else None
        results[axis] = {**result, **member_scores, "delta_vs_better_member_deg": qualification[axis]}
    results["selection_rmse_deg"] = (results["device"]["macro_rmse_deg"] + results["motion"]["macro_rmse_deg"]) / 2
    qualifies = qualification["joint_holdout"] <= -PROTOCOL["r5_inclusion_min_joint_improvement_deg"] and all(
        qualification[axis] <= PROTOCOL["r5_inclusion_max_single_axis_degradation_deg"] for axis in ("device", "motion"))
    result = {"dev_metrics": results, "sequences": selections, "folds": folds, "members": members, "protocol": declared_protocol,
              "eligible_for_r5": qualifies and not (independent or three), "relative_to_best_individual": qualification}
    if independent or three:
        result.update(accuracy_gate_passed=qualifies,
                      admission_pending_engineering=qualifies,
                      real_member_fits=sum(sum(fold["actual_model_fits"].values()) for fold in folds),
                      shared_filter_states=three, exact_matching_native_indices=True)
    if independent:
        result["independent_filters_protocol_sha256"] = config["independent_filters_protocol_sha256"]
    if three:
        result["three_seed_protocol_sha256"] = config["three_seed_protocol_sha256"]
    return result


def queue():
    pairings = ((("P00022", "J00016"), ("N00035", "J00010")), (("P00022", "J00016"), ("I00004", "J00002")))
    stage = []
    for i, pairing in enumerate(pairings, 1):
        identity = f"U{i:05d}"
        config = {"kind": "ensemble", "parameters": {"tauAcc": 0.2}, "seed": None, "members": []}
        for name, joint in pairing:
            source = read_json(BATCH / "rounds" / name / "round.json")
            config["members"].append({"round_id": name, "joint_round": joint, "config": source["config"], "weight": 0.5})
        validate(config)
        target = BATCH / "rounds" / identity / "round.json"
        if not target.is_file():
            write_json(target, {"round_id": identity, "config": config, "config_sha256": fingerprint(config), "seed": None,
                                "batch": "fixed-direction-ensemble", "relation": pairing[0][0],
                                "hypothesis": "Independent physical/temporal or calibration errors may cancel in an equal-weight body-frame average",
                                "partitions": {"development": names("dev"), "excluded": list(EXCLUDED)},
                                "planned_at": stamp(), "started_at": None, "ended_at": None, "status": "planned", "attempts": 0,
                                "dev_metrics": None, "result": f"rounds/{identity}/result.json", "stdout": f"rounds/{identity}/run.log"})
        stage.append({"round_id": identity, "config": config})
    write_json(BATCH / "ensemble-development-stage.json", {"planned_at": stamp(), "protocol": PROTOCOL, "rounds": stage})
    print([(r["round_id"], [m["round_id"] for m in r["config"]["members"]]) for r in stage], flush=True)


if __name__ == "__main__":
    queue()
