"""Publish the interrupted C037 closeout exclusively from existing scored artifacts."""
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import fmean

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METHODS = (
    "full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C035_deg", "physical_only_deg",
    "C036_deg", "C037_deg", "fixed_tau1p5_deg", "fixed_tau3_deg", "fixed_tau6_deg",
)
WEIGHTS = ("fast", "slow", "original")
ARTIFACTS = (
    "sequences.csv", "array-sensor-poses.csv", "group-summaries.json",
    "fixed-vqf-comparison.json", "development-routing.npz",
)
FAILURE = ("failure.json", "failure-traceback.log")


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=HERE, delete=False) as stream:
        stage = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    stage.replace(path)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def array_sha(values, dtype):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes()).hexdigest()


def table(name):
    counts = {
        "scored_frames", "active_release_count", "total_release_count", "active_scored_frames",
        "inactive_scored_frames", "cap_affected_scored_frames", "cap_affected_release_count",
    }
    floats = {*METHODS, "control_max_abs_deg", "control_max_abs_delta_deg",
              "fixed_tau3_max_abs_vs_FullVQF", "max_parent_disagreement_deg", "seconds", "prediction_seconds"}
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return [
            {k: int(v) if k in counts else (float(v) if v else None) if k in floats or k.endswith("_mean") else v
             for k, v in row.items()}
            for row in csv.DictReader(stream)
        ]


def summarize(rows):
    means = {k: fmean(row[k] for row in rows) for k in METHODS}
    supported = [row for row in rows if row["active_release_count"] > 0]
    return {
        "records": len(rows), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in rows),
        "macro_tilt_rmse_deg": means, "active_releases": sum(r["active_release_count"] for r in rows),
        "cap_affected_scored_frames": sum(r["cap_affected_scored_frames"] for r in rows),
        "cap_affected_release_count": sum(r["cap_affected_release_count"] for r in rows),
        "mean_routing_prob_equal_records_with_support": {
            key: fmean(r[f"p_{key}_mean"] for r in supported) for key in WEIGHTS
        } if supported else {},
        "C037_delta_deg": {k: means["C037_deg"] - means[k] for k in METHODS if k != "C037_deg"},
        "C037_wins": {k: sum(row["C037_deg"] < row[k] for row in rows) for k in METHODS if k != "C037_deg"},
    }


def keys_match(left, right, keys):
    return all(left[k] == right[k] for k in keys)


def main():
    started = now()
    result_path = HERE / "results.json"
    check(not result_path.exists(), "C037 results already exist")
    protocol = read(HERE / "protocol.json")
    restored = {name: sha(ROOT / name) for name in protocol["inputs_sha256"]}
    check(restored == protocol["inputs_sha256"], "At least one frozen input has changed")
    check(sha(HERE / "run_physical_cap.py") == protocol["worker_sha256"], "Worker changed")
    original = read(HERE / "process.json")
    launch = read(HERE / "launch.json")
    failure = read(HERE / "failure.json")
    progress = read(HERE / "replay-progress.json")
    coefficient = read(HERE / "coefficient-lock.json")
    solver = read(HERE / "solver-receipt.json")
    certificate = read(HERE / "certificate.json")
    certificate_check = read(HERE / "certificate-verification.json")
    pid = launch["pid"]
    protocol_hash = sha(HERE / "protocol.json")
    lock_hash = sha(HERE / "coefficient-lock.json")
    check(all(row["pid"] == pid for row in (original, failure, progress, coefficient, solver, certificate)),
          "Attempt identity differs")
    check(all(row["protocol_sha256"] == protocol_hash for row in (launch, original, failure, coefficient))
          and lock_hash == original["coefficient_lock_sha256"]
          and original["status"] == "development_replay" and progress["scored"] == progress["attempted"] == 213
          and not progress["failures"] and "Frozen evidence changed during replay" in failure["error"],
          "Interrupted replay receipt differs")
    check(all(coefficient[field] == sha(HERE / name) for field, name in (
        ("solver_receipt_sha256", "solver-receipt.json"), ("evaluation_log_sha256", "solver-evaluations.jsonl"),
        ("training_io_sha256", "training-io.npz"), ("certificate_sha256", "certificate.json"),
    )), "Frozen coefficient evidence differs")
    check(certificate["branches_sha256"] == sha(HERE / "solver-branches.jsonl")
          and certificate["evaluations_sha256"] == coefficient["evaluation_log_sha256"]
          and certificate_check["certificate_sha256"] == coefficient["certificate_sha256"]
          and certificate["gap"] == certificate_check["gap"] <= protocol["objective_gap_target"]
          and coefficient["cap_deg"] == certificate["incumbent_cap_deg"] == solver["cap_deg"]
          and coefficient["objective"] == certificate["upper_bound"] == solver["objective"]
          and certificate_check["coverage_to_both_endpoints_without_gap"]
          and solver["solve_count"] == coefficient["solve_count"] == 1
          and solver["neural_fits"] == coefficient["neural_fits"] == 0, "Branch-and-bound certificate differs")

    artifact_sha = {name: sha(HERE / name) for name in ARTIFACTS}
    evidence_sha = {name: sha(HERE / name) for name in (*FAILURE, "launch.json", "replay-progress.json",
                    "penn-training-source-scores.json", "penn-A-native-masks.npz", "certificate-verification.json")}
    previous = read(ROOT / "C036/results.json")
    physical = read(ROOT / "C034/results.json")
    old_poses = read(ROOT / "C022/results.json")["array_sensor_poses"]
    fixed = read(ROOT / "review/vqf_response_controls_development.json")
    fixed_rows = {(r["dataset"], r["record"]): r for r in fixed["records"]}
    sequences, poses = table("sequences.csv"), table("array-sensor-poses.csv")
    check(len(sequences) == len(fixed_rows) == 213 and len(poses) == 96, "Archived table size differs")
    check(sum(row["scored_frames"] for row in sequences) == 16110258
          and [[r["dataset"], r["record"]] for r in sequences] == protocol["development_records"],
          "Development record order or frame denominator differs")
    identity = ("dataset", "record", "scored_frames", "scoring_indices_sha256")
    archived_columns = (
        "full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C035_deg", "physical_only_deg", "C036_deg",
    )
    native_hashes = ("native_input_last_sha256", "native_release_sha256", "C023_active_routing_sha256",
                     "C035_active_routing_sha256", "C023_delta_sha256", "C035_delta_sha256")
    with (
        np.load(ROOT / "C005/scoring-masks.npz", allow_pickle=False) as masks,
        np.load(ROOT / "C006/gate-traces.npz", allow_pickle=False) as gates,
        np.load(HERE / "development-routing.npz", allow_pickle=False) as routes,
    ):
        route = {key: routes[key] for key in routes.files}
        offset = 0
        for ordinal, (row, prior, control) in enumerate(
            zip(sequences, previous["sequences"], physical["sequences"], strict=True)
        ):
            key = row["dataset"], row["record"]
            fixed_row = fixed_rows[key]
            check(row["status"] == "scored" and all(keys_match(row, old, identity) for old in (prior, control, fixed_row))
                  and keys_match(row, prior, native_hashes)
                  and row["mask_key"] == prior["mask_key"] == control["mask_key"], f"Record identity differs: {key}")
            indices = np.flatnonzero(masks[row["mask_key"]])
            release = gates[row["mask_key"]][:, 0]
            check(len(indices) == row["scored_frames"] and array_sha(indices, "<i8") == row["scoring_indices_sha256"]
                  and len(release) == row["total_release_count"] and array_sha(release, "<i8") == row["native_release_sha256"],
                  f"Native indices differ: {key}")
            check(all(math.isfinite(row[k]) and row[k] >= 0. for k in METHODS)
                  and row["physical_only_deg"] == row["C036_deg"] == control["physical_only_deg"]
                  and all(abs(row[k] - prior[k]) <= 1e-6 for k in archived_columns)
                  and all(row[k] == fixed_row[f] == prior[k] for k, f in (
                      ("fixed_tau1p5_deg", "fast_deg"), ("fixed_tau3_deg", "default_deg"), ("fixed_tau6_deg", "slow_deg")))
                  and row["active_scored_frames"] + row["inactive_scored_frames"] == row["scored_frames"],
                  f"Archived controls differ: {key}")
            n = row["active_release_count"]
            active = slice(offset, offset + n)
            active_at = route["native_release"][active]
            slots = np.searchsorted(release, active_at)
            check(np.all(route["ordinal"][active] == ordinal) and np.array_equal(release[slots], active_at)
                  and np.all(route["native_input_last"][active] <= active_at)
                  and all(array_sha(route[m][active], "<f8") == row[f"{m}_active_routing_sha256"] for m in ("C023", "C035")),
                  f"Archived active routing differs: {key}")
            offset += n
        check(all(len(values) == offset for values in route.values()), "Routing ledger is incomplete")
    indexed_poses = {(r["record"], r["pose"]): r for r in old_poses}
    check(len({(r["record"], r["pose"]) for r in poses}) == 96, "Duplicate poses")
    for row, prior, control in zip(poses, previous["array_sensor_poses"], physical["array_sensor_poses"], strict=True):
        pose_key = row["record"], row["pose"]
        check(all(keys_match(row, old, ("record", "pose", "scored_frames", "scoring_indices_sha256"))
                  for old in (prior, control, indexed_poses[pose_key]))
              and all(abs(row[k] - prior[k]) <= 1e-6 for k in archived_columns)
              and math.isfinite(row["C037_deg"]) and row["C037_deg"] >= 0.
              and row["physical_only_deg"] == row["C036_deg"] == control["physical_only_deg"],
              f"Pose controls differ: {pose_key}")

    penn = read(HERE / "penn-training-source-scores.json")
    penn_prior = read(ROOT / "C036/penn-training-source-scores.json")
    check(penn["pid"] == pid and penn["coefficient_lock_sha256"] == lock_hash
          and penn["native_mask_file_sha256"] == evidence_sha["penn-A-native-masks.npz"]
          and [r["record"] for r in penn["records"]] == protocol["penn_training_side_only"], "Penn A receipt differs")
    with np.load(HERE / "penn-A-native-masks.npz", allow_pickle=False) as masks:
        for row, prior in zip(penn["records"], penn_prior["records"], strict=True):
            key = row["record"]
            with np.load(ROOT / "C033" / f"{key}-native-mask-evidence.npz", allow_pickle=False) as old:
                for name in ("scoring_mask", "release", "native_input_last"):
                    np.testing.assert_array_equal(masks[f"{key}_{name}"], old[name])
            indices = np.flatnonzero(masks[f"{key}_scoring_mask"])
            np.testing.assert_array_equal(indices, masks[f"{key}_scoring_indices"])
            check(keys_match(row, prior, ("record", "mask_sha256", "scored_frames"))
                  and len(indices) == row["scored_frames"] and array_sha(indices, "<i8") == row["mask_sha256"]
                  and all(abs(row["full_record_tilt_RMSE_deg"][m] - prior["full_record_tilt_RMSE_deg"][m]) <= 1e-6
                          for m in archived_columns), f"Penn A controls differ: {key}")
    check(all(fmean(r["full_record_tilt_RMSE_deg"][k] for r in penn["records"]) == v
              for k, v in penn["record_equal_mean"].items()), "Penn A mean differs")

    by_group, by_domain = defaultdict(list), defaultdict(list)
    for row in sequences:
        by_group[row["dataset"], row["group"]].append(row)
        by_domain[row["dataset"]].append(row)
    overall = summarize(sequences)
    domains = {name: summarize(by_domain[name]) for name in read(ROOT / "protocol.json")["datasets"]}
    tum = {r["record"]: summarize([r]) for r in by_domain["TUM-VI"]}
    oxford = {g: summarize(rows) for (d, g), rows in sorted(by_group.items()) if d == "OxIOD"}
    groups = {f"{d}/{g}": summarize(rows) for (d, g), rows in sorted(by_group.items())}
    check(read(HERE / "group-summaries.json") == {"TUM_trajectories": tum, "Oxford_carry": oxford, "group": groups},
          "Independent group aggregation disagrees")
    fixed_out = read(HERE / "fixed-vqf-comparison.json")
    check(fixed_out == {"archived_controls_sha256": restored["review/vqf_response_controls_development.json"],
                         "matched_legacy_source_C020_csv_sha256": restored["C020/sequences.csv"],
                         "archived_fixed_results_reused": True, "each_record_mask_identity_checked": True,
                         "global": overall, "domains": domains, "TUM_trajectories": tum, "Oxford_carry": oxford},
          "Independent archived VQF aggregation disagrees")
    check(all(sha(ROOT / name) == expected for name, expected in restored.items())
          and all(sha(HERE / name) == expected for name, expected in {**artifact_sha, **evidence_sha}.items())
          and sha(HERE / "coefficient-lock.json") == lock_hash, "Evidence changed during closeout")
    finished = now()
    recovery = {
        "original_pid": pid, "recovery_pid": os.getpid(), "recovery_started_at": started,
        "recovery_completed_at": finished, "fit_repeated": False, "replay_rows": 0,
        "verified_existing_development_rows": len(sequences), "verified_existing_pose_rows": len(poses),
        "verified_existing_penn_training_rows": len(penn["records"]), "verified_native_frames": overall["frames"],
        "verified_locked_input_count": len(restored), "restored_input_sha256": restored["review/physical_cap_certificate.py"],
        "original_process": original, "original_process_sha256": sha(HERE / "process.json"), "original_failure": failure,
        "original_evidence_sha256": evidence_sha, "script_sha256": sha(Path(__file__)),
        "original_replay_completed_at": progress["updated_at"],
    }
    result = {
        "pooled_records": overall, "domains": domains, "TUM_trajectories": tum, "Oxford_carry": oxford,
        "sequences": sequences, "array_sensor_poses": poses, "failures": [], "artifact_sha256": artifact_sha, "pid": pid,
        "cap_deg": coefficient["cap_deg"], "coefficient_lock_sha256": lock_hash, "protocol_sha256": protocol_hash,
        "scoring_masks_sha256": restored["C005/scoring-masks.npz"],
        "penn_training_source_scores_sha256": evidence_sha["penn-training-source-scores.json"],
        "solve_count": 1, "neural_fits": 0, "solver_receipt_sha256": coefficient["solver_receipt_sha256"],
        "certificate_sha256": coefficient["certificate_sha256"],
        "replay_seconds": (datetime.fromisoformat(failure["failed_at"]) - datetime.fromisoformat(original["updated_at"])).total_seconds(),
        "replay_seconds_source": "Original process development-replay status to the final hash-check failure",
        "completed_at": finished, "final_holdout_scored": False, "supplemental_holdout_scored": False, **recovery,
    }
    write(result_path, result)
    results_hash = sha(result_path)
    write(HERE / "recovery.json", {**recovery, "results_sha256": results_hash})
    write(HERE / "process.json", {**original, "status": "complete", "updated_at": finished,
                                  "scored": len(sequences), "results_sha256": results_hash, "original_pid": pid,
                                  "recovery_pid": os.getpid(), "recovery_sha256": sha(HERE / "recovery.json")})
    print(json.dumps({"results_sha256": results_hash, "recovery": {
        key: recovery[key] for key in ("original_pid", "recovery_pid", "recovery_started_at",
                                      "recovery_completed_at", "fit_repeated", "replay_rows")
    }, "macro": overall["macro_tilt_rmse_deg"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
