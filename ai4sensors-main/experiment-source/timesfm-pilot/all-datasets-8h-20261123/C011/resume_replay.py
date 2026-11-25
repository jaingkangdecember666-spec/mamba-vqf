"""Resume the C011 audit and replay after the scalar was already frozen."""

from datetime import datetime
import os
from time import perf_counter

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import replay_long_state as source
from replay_long_state import (C001, C003, CAMPAIGN, GATES, HERE, MASKS, PILOT, TAU, TRANSFER, UP, base, c2, c4, c6,
                              digest, ema_at, front_end, link_units, read, replay, stamp, tilt_error, world_base, write)


def installation_checks(probes, cfg, core, mean, std, k):
    rotations = [*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
                 Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))]
    fixed = []
    for domain, (raw, observed, dt) in probes.items():
        _, q, at, _, history, _, acc, _ = front_end(raw, observed, dt, cfg)
        take = np.linspace(0, len(at) - 1, min(32, len(at)), dtype=int)
        at = at[take]
        engine = Rotation.from_quat(q, scalar_first=True)
        m = ema_at(q, acc, observed, dt, at)
        branch, rest = world_base(core, core.output, history[take], q[at], mean, std)
        world, _ = link_units(branch, m, rest, k)
        for i, change in enumerate(rotations):
            q_new = (engine * change.inv()).as_quat(scalar_first=True)
            moved = history[take].copy()
            moved[..., :12] = change.apply(moved[..., :12].reshape(-1, 3)).reshape(len(moved), moved.shape[1], 12)
            m_new = ema_at(q_new, change.apply(acc), observed, dt, at)
            b_new, r_new = world_base(core, core.output, moved, q_new[at], mean, std)
            np.testing.assert_array_equal(rest, r_new)
            world_new, _ = link_units(b_new, m_new, r_new, k)
            errors = [float(tilt_error(a, b).max()) for a, b in zip(world, world_new, strict=True)]
            ema_err = float(np.max(np.abs(m_new - m)))
            base.require(ema_err <= 2e-12 and max(errors) <= .0001, "Fixed-world installation rotation audit failed")
            fixed.append({"domain": domain, "rotation": i, "windows": len(take), "ema_max_abs": ema_err,
                          "max_control_deg": errors[0], "max_C011_deg": errors[1], "max_physical_only_deg": errors[2]})

    dt = .005
    time = np.arange(2400) * dt
    moving = time >= 2.
    raw = np.zeros((len(time), 6))
    raw[:, 2] = source.GRAVITY
    ph = time[moving] - 2.
    raw[moving, 0] = 1.3 * np.sin(1.1 * ph)
    raw[moving, 1] = .7 * np.sin(2.2 * ph)
    raw[moving, 3] = .15 * np.sin(.6 * ph)
    raw[moving, 4] = .14 * np.cos(.9 * ph)
    raw[moving, 5] = .18 * np.cos(.35 * ph)
    observed = np.ones(len(raw), dtype=bool)
    observed[510:517] = False

    def body_replay(imu):
        _, q, at, release, history, _, acc, _ = front_end(imu, observed, dt, cfg)
        state = ema_at(q, acc, observed, dt, at)
        world, rest = world_base(core, core.output, history, q[at], mean, std)
        links, _ = link_units(world, state, rest, k)
        inv = Rotation.from_quat(q[release], scalar_first=True).inv()
        # SciPy's native Rotation.apply requires a writable input buffer here.
        up = np.broadcast_to(UP, world.shape).copy()
        return release, rest, [inv.apply(u) for u in (up, *links)], inv.apply(state)

    release, rest, original, original_ema = body_replay(raw)
    full = []
    for i, change in enumerate(rotations):
        rotated = change.apply(raw.reshape(-1, 3)).reshape(raw.shape)
        release2, rest2, moved, moved_ema = body_replay(rotated)
        np.testing.assert_array_equal(release, release2)
        np.testing.assert_array_equal(rest, rest2)
        errors = [float(tilt_error(a, change.inv().apply(b)).max()) for a, b in zip(original, moved, strict=True)]
        state_error = float(np.max(np.abs(original_ema - change.inv().apply(moved_ema))))
        full.append({"rotation": i, "windows": len(release), "default_vqf_max_deg": errors[0], "C006_max_deg": errors[1],
                     "C011_max_deg": errors[2], "physical_only_max_deg": errors[3], "ema_body_max_abs": state_error})
    return {"fixed_world_gauge": fixed, "fixed_world_comparisons": sum(r["windows"] for r in fixed),
            "synthetic_raw_sensor_path": full, "synthetic_raw_max_C011_deg": max(r["C011_max_deg"] for r in full)}


def main():
    process, protocol, lock = (HERE / n for n in ("process.json", "protocol.json", "coefficient-lock.json"))
    rollback = read(process)
    p, fixed = read(protocol), read(lock)
    fit_path, witness = HERE / "external-selection.json", HERE / "training-windows.npz"
    record = HERE / "replay-recovery.json"
    base.require(rollback["status"] == "failed" and rollback["failure"] == "ValueError('buffer source array is read-only')"
                 and not (HERE / "results.json").exists() and not record.exists(),
                 "Replay recovery is only for the documented pre-replay audit error")
    base.require(rollback["protocol_sha256"] == fixed["protocol_sha256"] == digest(protocol)
                 and fixed["training_windows_sha256"] == digest(witness)
                 and fixed["external_selection_sha256"] == digest(fit_path)
                 and all(digest(PILOT / path) == sha for path, sha in p["files_sha256"].items()),
                 "A frozen C011 input or the scalar lock changed")
    k = fixed["k"]
    fit = read(fit_path)
    lock_sha = digest(lock)
    base.require(fit["selection"]["k"] == k and 0. <= k <= 1. and fixed["tau_s"] == TAU,
                 "Invalid persisted scalar")
    recovery = {"original_process": rollback, "recovery_started_at": stamp(), "pid": os.getpid(),
                "reason": "Use a writable copy of the broadcast world-up vector for the synthetic SciPy audit.",
                "recovery_source_sha256": digest(HERE / "resume_replay.py"), "coefficient_lock_sha256": lock_sha,
                "selection_executions_in_recovery": 0, "original_run_log": "run.log"}
    write(record, recovery)

    def status(phase, **extra):
        write(process, {"status": phase, "started_at": rollback["started_at"], "pid": os.getpid(),
                        "protocol_sha256": digest(protocol), "recovery_sha256": digest(record),
                        "coefficient_lock_sha256": lock_sha, **extra})

    status("rotation_audit_recovery")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    weights = source.C004 / read(source.C004 / "training.json")["arms"]["B"]["weights"]
    base.require(digest(weights) == p["C006_checkpoint_sha256"], "C006 weights changed")
    saved = torch.load(weights, weights_only=True, map_location="cpu")
    core = base.ResidualHead(saved["model"])
    core.load_state_dict(saved["state_dict"])
    core.eval().requires_grad_(False).to(device)
    mean, std = saved["mean"].to(device), saved["std"].to(device)
    cfg = source.c3.Settings(**read(C003 / "protocol.json")["thresholds"])
    inventory = read(C001 / "inventory.json")
    examples = {}
    mats = {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for spec, audit in zip(p["source_records"], fit["records"], strict=True):
            if spec["domain"] in examples:
                continue
            raw, observed, dt, _, _, hashes = c4.train_raw(spec, yt["train"], mats)
            base.require(hashes == audit["raw_hashes"], "A rotation probe input changed")
            examples[spec["domain"]] = (raw, observed, dt)
    prefix = HERE / "prefix-audit.json"
    evidence = read(prefix)
    base.require(evidence["missing_packet_holds_exactly"] and len(evidence["probes"]) == len(examples)
                 and all(x["exact_prefix_and_future_mutation"] for x in evidence["probes"]), "Recorded prefix audit did not pass")
    installation = HERE / "installation-audit.json"
    write(installation, installation_checks(examples, cfg, core, mean, std, k))
    base.require(digest(lock) == lock_sha, "Gain changed before replay")
    status("development_replay")
    print("C011 resuming with previously locked coefficient", k, flush=True)
    scored_at = perf_counter()
    old = read(source.C006 / "results.json")
    result = replay(old, inventory, read(CAMPAIGN / "protocol.json"), read(TRANSFER / "external-split.json"),
                    core, core.output, mean, std, cfg, k)
    completed = stamp()
    base.require(digest(lock) == lock_sha, "Locked coefficient changed during development replay")
    result.update({"started_at": rollback["started_at"], "completed_at": completed, "k": k, "neural_optimizer_steps": 0,
                   "protocol_sha256": digest(protocol), "coefficient_lock_sha256": lock_sha, "masks_sha256": digest(MASKS),
                   "gate_traces_sha256": digest(GATES), "recovery_sha256": digest(record), "prefix_audit_sha256": digest(prefix),
                   "installation_audit_sha256": digest(installation), "scoring_seconds": perf_counter() - scored_at,
                   "wall_elapsed_seconds_including_recovery": (datetime.fromisoformat(completed)
                                                               - datetime.fromisoformat(rollback["started_at"])).total_seconds()})
    path = HERE / "results.json"
    write(path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(path))
    print("C011 coverage", len(result["sequences"]), "failures", result["failures"], "scoring seconds", result["scoring_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process = HERE / "process.json"
        if process.exists():
            prior = read(process)
            if prior["pid"] == os.getpid():
                write(process, {**prior, "status": "failed", "phase": prior["status"],
                                "completed_at": stamp(), "failure": repr(error)})
        raise
