"""Verify causal native-rate IMU-to-world quaternion output independently of tilt scoring."""

import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import learning
from common import BATCH, SHICE, Z, body_up, digest, fingerprint, names, record, read_json, rmse, rotate, stamp, unit, vqf_outputs, write_json
from features import bins, block_indices, physics
from tilt_eval import predict_updates
from tilt_model import contexts, correction_quaternions, multiply, normalize, physical_features


IDENTITY = np.array([1., 0., 0., 0.])
PROTOCOL = {
    "format": "wxyz unit quaternion, IMU to estimated world with +Z up, native sample order",
    "composition": "q_output = q_world_correction * q_vqf * q_sensor_mount; Hamilton product, right factor applied first",
    "prefix_fractions": [0.27, 0.59, 0.81],
    "max_norm_error": 1e-6,
    "max_quaternion_prefix_rotation_deg": 0.001,
    "max_scored_rmse_difference_deg": 1e-4,
    "does_not_evaluate_heading": True,
}


def inertial_states(imu, dt, parameters):
    result = vqf_outputs(imu, dt, parameters)
    base = result["quat6D"]
    windows, actual, ready = bins(imu, dt, True)
    return base, {"windows": windows, "q": base[actual].astype(np.float32), "up": body_up(base[actual]).astype(np.float32),
                  "bias": result["bias"][actual].astype(np.float32), "rest": result["restDetected"][actual].astype(np.float32),
                  "ready": ready}


def quaternion_for_direction(base, body_direction):
    world_direction = rotate(unit(base), unit(body_direction))
    correction = correction_quaternions(IDENTITY, world_direction - Z)
    antipodal = (np.linalg.norm(correction, axis=-1) < 1e-8)[..., None]
    correction = np.where(antipodal, np.array([0., 1., 0., 0.]), correction)
    return normalize(multiply(correction, base))


def residual_updates(data, config, state, device, backbone):
    kind = config["kind"]
    at = block_indices(len(data["windows"]), config["update_hz"])
    if not len(at):
        return at, np.empty((0, 3))
    if kind == "stream_gru":
        from stream_gru import infer_delta,make_model

        x = physical_features(contexts(data["windows"],at),data["up"][at],data["bias"][at],data["rest"][at])
        delta = infer_delta(x,make_model(state,config,device),device)
        return at, config["gain"]*delta
    if kind == "legacy_head":
        from legacy_head import make_model

        head = make_model(state, device)
        x = physical_features(contexts(data["windows"], at), data["up"][at], data["bias"][at], data["rest"][at])
        return at, config["gain"] * predict_updates(head, x, None, device)
    if kind == "equivariant":
        import equivariant

        at, basis, scalars = equivariant.features(data, config)
        return at, equivariant.delta(basis, scalars, state, config)

    at, matrix, acc, gyr = physics(data, config)
    if kind == "ridge" and config.get("embedding"):
        from final_eval import embed_external

        latent = embed_external(data, config, backbone, device)
        matrix = np.concatenate((matrix, latent), axis=1)
    inputs = {"matrix": matrix, "up": data["up"][at], "acc": acc, "gyr": gyr}
    if kind == "ridge":
        return at, learning.delta_at_updates(inputs, state, config)
    if kind == "tcn":
        import temporal

        valid = at + 1 >= config["context"]
        at, inputs = at[valid], {key: value[valid] for key, value in inputs.items()}
        if not len(at):
            return at, np.empty((0, 3))
        inputs["history"] = temporal.windows(data["windows"], at, config["context"])
        return at, temporal.inference(inputs, *temporal.make_model(state, config, device), config, device)
    raise ValueError(f"No quaternion rollout available for {kind}")


def native_replay(imu, dt, config, state, device, backbone=None, audit_updates=False, packet_ok=None):
    kind = config["kind"]
    if kind == "bias_physical_head":
        from bias_calibration import online_filter
        from calibrated_head import verify_protocol

        verify_protocol(config)
        if packet_ok is None:
            raise ValueError("Causal bias replay requires the original packet validity")
        adjusted, _, _, _ = online_filter(imu, packet_ok, dt, config["minimum_online_poses"])
        return native_replay(adjusted, dt, config["head"], state, device, backbone, audit_updates)
    parameters = config.get("parameters", {})
    base, data = inertial_states(imu, dt, parameters)
    if kind in {"baseline", "filter"}:
        return base, None
    if kind == "install_rotation":
        if config["coordinates"] == "body":
            q_mount = Rotation.from_matrix(state["fitted"]).as_quat(scalar_first=True)
            return normalize(multiply(base, q_mount)), None
        correction = correction_quaternions(IDENTITY, state["fitted"] - Z)
        return normalize(multiply(correction, base)), None
    if kind == "hybrid":
        from hybrid_filter import gate, mix

        default, origin = inertial_states(imu, dt, {})
        block_weight = gate(origin["windows"], origin["up"], data["up"], config)
        available = np.searchsorted(origin["ready"], np.arange(len(base)), side="right")
        weight = np.r_[0., block_weight][available]
        fused = mix(body_up(default), body_up(base), weight)
        estimate = quaternion_for_direction(default, fused)
        return np.where(weight[:, None] == 0., default, estimate), None
    if kind == "ensemble":
        from ensemble import blend, member_state

        outputs = [native_replay(imu, dt, member["config"], member_state(state, i), device, backbone)[0]
                   for i, member in enumerate(config["members"])]
        fused = blend(*(body_up(output) for output in outputs))
        return quaternion_for_direction(base, fused), None

    mounting = state.get("installation_quaternion_wxyz")
    if mounting is not None:
        base = normalize(multiply(base, mounting))
        data = learning.apply_install(data, mounting, include_reference=False)
    indices, delta = residual_updates(data, config, state, device, backbone)
    when = data["ready"][indices]
    correction = correction_quaternions(data["q"][indices], delta)
    available = np.searchsorted(when, np.arange(len(base)), side="right")
    held = np.vstack((IDENTITY, correction))[available]
    output = normalize(multiply(held, base))
    single = None
    if audit_updates and kind == "legacy_head" and len(indices):
        from evaluate_tilt import correction_update
        from legacy_head import make_model

        head = make_model(state, device)
        checks = []
        for index in sorted({0, len(indices)//2, len(indices)-1}):
            at = indices[index]
            _, _, explicit = correction_update(None, head, data["windows"][at-63:at+1], data["q"][at],
                                                data["bias"][at], data["rest"][at], config["gain"], device)
            checks.append({"native_index": int(when[index]),
                           "quaternion_rotation_difference_deg": rotation_difference(output[when[index]:when[index]+1], explicit[None])[0]})
        single = checks
    return output, single


def rotation_difference(a, b):
    target = np.where((a*b).sum(axis=-1, keepdims=True) < 0., -b, b)
    return np.rad2deg(4. * np.arctan2(np.linalg.norm(a-target, axis=-1), np.linalg.norm(a+target, axis=-1)))


def source_time(name, n):
    source_file = SHICE / f"{name.split('-')[0]}_aligned_formal" / f"{name}.csv"
    with source_file.open(encoding="utf-8-sig") as stream:
        fields = stream.readline().strip().split(",")
        source = np.loadtxt(stream, delimiter=",", quotechar='"',
                            usecols=(fields.index("sample_index"), fields.index("time_s")), ndmin=2)
    np.testing.assert_array_equal(source[:, 0], np.arange(n))
    if np.any(np.diff(source[:, 1]) <= 0.):
        raise ValueError("Non-monotonic timestamps in original native source")
    return source[:, 1]


def verify(name, config, state, expected, output_path, device, backbone=None):
    source = record(name)
    imu, dt = source["imu"], float(source["dt"])
    packet_ok = None
    if config["kind"] == "bias_physical_head":
        from bias_calibration import read_native

        native = read_native(name)
        imu, packet_ok = native["untreated"], native["original_ok"]
    quat, explicit = native_replay(imu, dt, config, state, device, backbone, audit_updates=True, packet_ok=packet_ok)
    if quat.shape != (len(imu), 4) or not np.isfinite(quat).all():
        raise ValueError(f"Invalid native quaternion output shape or values in {name}")
    count = int(source["score"].sum())
    if expected["frames"] != count:
        raise ValueError(f"Native scoring mask count differs in {name}")
    q_norm = float(np.max(np.abs(np.linalg.norm(quat, axis=1)-1.0)))
    value = rmse(body_up(quat[source["score"]]), source["ref"][source["score"]])
    parts = []
    for fraction in PROTOCOL["prefix_fractions"]:
        cut = int(len(imu)*fraction)
        prefix_validity = packet_ok[:cut] if packet_ok is not None else None
        prefix, _ = native_replay(imu[:cut], dt, config, state, device, backbone, packet_ok=prefix_validity)
        difference = rotation_difference(quat[:cut], prefix)
        parts.append({"prefix_native_frames":cut, "max_rotation_delta_deg": float(np.max(difference)),
                      "p95_rotation_delta_deg": float(np.percentile(difference, 95)),
                      "last_rotation_delta_deg": float(difference[-1])})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, sample_index=np.arange(len(imu), dtype=np.int64),
                            time_s=source_time(name, len(imu)), q_wxyz=quat)
    os.replace(temporary, output_path)
    checks = {"name": name, "native_frames": len(imu), "scored_frames": count, "max_unit_norm_error": q_norm,
              "direct_direction_rmse_deg": expected["rmse_deg"], "quaternion_tilt_rmse_deg": value,
              "absolute_tilt_rmse_delta_deg": abs(value-expected["rmse_deg"]), "prefix_checks": parts,
              "online_update_comparison": explicit, "quaternion_file": str(output_path.relative_to(BATCH)),
              "quaternion_sha256": digest(output_path)}
    checks["passed"] = (q_norm <= PROTOCOL["max_norm_error"] and
                        checks["absolute_tilt_rmse_delta_deg"] <= PROTOCOL["max_scored_rmse_difference_deg"] and
                        all(p["max_rotation_delta_deg"] <= PROTOCOL["max_quaternion_prefix_rotation_deg"] for p in parts) and
                        (not explicit or all(p["quaternion_rotation_difference_deg"] <= PROTOCOL["max_quaternion_prefix_rotation_deg"]
                                             for p in explicit)))
    return checks


def run(job, destination):
    from final_eval import fit_candidate

    parent_id = job["config"]["parent_round"]
    previous = read_json(BATCH / "rounds" / parent_id / "round.json")
    config = previous["config"]
    if fingerprint(config) != job["config"]["parent_config_sha256"]:
        raise ValueError("Quaternion audit refers to a changed development configuration")
    source = read_json(BATCH / previous["result"])
    device = torch.device("cuda")
    all_names = names("dev")
    outputs = []
    for sample in job["config"]["records"]:
        name, axis = sample["name"], sample["axis"]
        _, fold, held = next(row for row in learning.folds(all_names) if row[0] == axis and name in row[2])
        training = [entry for entry in all_names if entry not in held]
        state = fit_candidate(config, training)
        if config["kind"] == "baseline":
            rows = source["dev_metrics"]["sequences"]
        elif config["kind"] == "hybrid":
            rows = source["sequences"]
        else:
            rows = source["sequences"][axis]
        expected = next(row for row in rows if row["name"] == name)
        result = verify(name, config, state, expected, destination.parent / f"{name}-q-wxyz.npz", device)
        outputs.append({"fold_axis": axis, "fold": fold, "training": training, "holdout": held, **result})
        print(parent_id, name, result["passed"], result["absolute_tilt_rmse_delta_deg"],
              max(p["max_rotation_delta_deg"] for p in result["prefix_checks"]), flush=True)
    write_json(destination, {"started_for_round": job["round_id"], "parent_round": parent_id,
                             "model_config_sha256": fingerprint(config), "protocol": PROTOCOL,
                             "same_existing_scored_frames": True, "output_status": "passed" if all(r["passed"] for r in outputs) else "failed",
                             "checked_at": stamp(), "records": outputs})


def queue():
    from optimizer_study import schedule

    choices = ("P00022", "N00035", "I00004", "R00193", "H00010", "R00001")
    planned = []
    for i, parent_id in enumerate(choices, 1):
        parent = read_json(BATCH / "rounds" / parent_id / "round.json")
        identity = f"V{i:05d}"
        config = {"kind": "quaternion_audit", "parent_round": parent_id, "parent_config_sha256": parent["config_sha256"],
                  "protocol_sha256": fingerprint(PROTOCOL), "seed": parent["seed"],
                  "records": [{"name":"R2-10","axis":"device"},{"name":"R4-2","axis":"motion"}]}
        saved = schedule(identity, config, "quat-output-audit", parent_id,
                         "Verify native quaternion magnitude, tilt parity and prefix causality with the existing world correction")
        planned.append({"round_id":saved,"parent":parent_id})
    write_json(BATCH / "quaternion-output-stage.json", {"created_at":stamp(),"protocol":PROTOCOL,"planned":planned})
    print(planned,flush=True)


if __name__ == "__main__":
    queue()
