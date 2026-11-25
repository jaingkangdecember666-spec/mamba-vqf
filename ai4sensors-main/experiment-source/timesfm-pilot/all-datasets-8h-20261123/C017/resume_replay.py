"""Resume after the C017 coefficient lock without opening training sources or fitting."""
from pathlib import Path
from time import perf_counter
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_stable_force as run

H, R = run.HERE, run.ROOT
read, write, sha, stamp = run.read, run.write, run.digest, run.stamp
require = run.base.require


def main():
    tick = perf_counter()
    protocol, lock, old_failure, selection = (read(H / p) for p in ("protocol.json", "coefficient-lock.json",
                                                                     "failure.json", "external-selection.json"))
    p14 = read(R / "C014/protocol.json")
    lock_hash = sha(H / "coefficient-lock.json")
    proto_hash = sha(H / "protocol.json")
    require(lock["protocol_sha256"] == proto_hash and lock["external_selection_sha256"] == sha(H / "external-selection.json")
            and lock["k"] == selection["coefficient_k"] and lock["pid"] == protocol["pid"] == old_failure["pid"]
            and sha(H / "replay_stable_force.before_audit_fix.py.txt") ==
                protocol["files_sha256"][str((H / "replay_stable_force.py").relative_to(run.PILOT))],
            "Resume no longer matches the immutable C017 coefficient lock")
    k = lock["k"]
    recovery_path = H / "replay-recovery.json"
    recovery = {"pid": os.getpid(), "started_at": stamp(), "original_pid": lock["pid"],
                "coefficient_lock_sha256": lock_hash, "protocol_sha256": proto_hash,
                "failure_sha256": sha(H / "failure.json"), "fixed_replay_code_sha256": sha(H / "replay_stable_force.py"),
                "resume_code_sha256": sha(Path(__file__)), "fit_repeated": False,
                "scope": "Fix missing dt argument in synthetic prefix audit, add synthetic hold/rest/cap checks, then run same 213 development records."}
    write(recovery_path, recovery)
    print("C017 resume", recovery, flush=True)

    def status(phase, **more):
        write(H / "process.json", {"status": phase, "pid": os.getpid(), "original_pid": lock["pid"],
                                  "started_at": protocol["started_at"], "updated_at": stamp(), "k": k,
                                  "protocol_sha256": proto_hash, "coefficient_lock_sha256": lock_hash,
                                  "recovery_sha256": sha(recovery_path), **more})

    status("causal_rotation_audit")
    try:
        campaign = read(R / "protocol.json")
        cfg = run.c3.Settings(**read(run.C003 / "protocol.json")["thresholds"])
        for name, expected in protocol["files_sha256"].items():
            path = (R / name) if (R / name).is_file() else (run.PILOT / name)
            if path != H / "replay_stable_force.py":
                require(sha(path) == expected, f"Frozen resume input changed: {path}")
        models, mean, std, _, _ = run.integrity_models(p14, campaign)
        audit_path = H / "causal-rotation-audit.json"
        write(audit_path, {"coefficient_lock_sha256": lock_hash, "probes": run.audit(cfg, k),
                           "targeted_zero_invalid_rest_cap_checks": "passed"})
        status("development_replay")
        print("C017 causal and rotation audit complete", flush=True)
        before = perf_counter()
        result = run.replay(read(R / "C014/results.json"), read(run.C001 / "inventory.json"),
                            campaign, read(run.TRANSFER / "external-split.json"), cfg, models, mean, std, k)
        require(sha(H / "coefficient-lock.json") == lock_hash
                and all(sha(run.PILOT / p) == v for p, v in protocol["fixed_weights"].items()),
                "Frozen inputs changed after lock")
        out = H / "results.json"
        result.update({"pid": os.getpid(), "original_pid": lock["pid"], "started_at": protocol["started_at"],
                       "completed_at": stamp(), "k": k, "protocol_sha256": proto_hash,
                       "coefficient_lock_sha256": lock_hash, "causal_rotation_audit_sha256": sha(audit_path),
                       "recovery_sha256": sha(recovery_path), "development_replay_seconds": perf_counter() - before,
                       "resume_total_seconds": perf_counter() - tick, "scoring_masks_sha256": sha(run.MASKS),
                       "final_holdout_scored": False})
        write(out, result)
        good = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
        status("complete" if good else "incomplete", results_sha256=sha(out), scored=result["pooled_records"]["records"])
        print("C017 finished", result["pooled_records"], "failures", result["failures"], flush=True)
        return 0 if good else 1
    except BaseException as exc:
        write(H / "resume-failure.json", {"pid": os.getpid(), "failed_at": stamp(), "error": repr(exc)})
        raise


if __name__ == "__main__":
    sys.exit(main())
