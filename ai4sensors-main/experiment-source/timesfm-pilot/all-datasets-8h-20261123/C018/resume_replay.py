"""Continue C018 after both arms have been irrevocably fixed. No optimization."""
from pathlib import Path
from time import perf_counter
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pair as c18
import numpy as np
import torch
from torch import nn

H, R = c18.HERE, c18.ROOT
sha, read, write, stamp, require = c18.sha, c18.read, c18.write, c18.stamp, c18.base.require


def recover_probes(inventory, cfg, selection, prep_hash):
    path = H / "training-inputs.npz"
    require(sha(path) == prep_hash, "Frozen C018 probe slot evidence changed")
    p15 = read(R / "C015/protocol.json")
    p15prep = read(R / "C015/training-preparation.json")
    sources = p15["legacy_train_sources"] + p15["ridi_sources"]
    keys = {}
    with np.load(path, allow_pickle=False) as a:
        for i, spec in enumerate(sources):
            domain = spec.get("domain", "RIDI")
            if domain in keys:
                continue
            use = np.flatnonzero(a["source_record_ordinal"] == i)
            unique, first = np.unique(a["native_release"][use], return_index=True)
            slots = use[first]
            eligible = np.flatnonzero(~a["rest"][slots] & a["valid"][slots])
            if len(eligible) >= 3:
                j = eligible[np.linspace(0, len(eligible) - 1, 3, dtype=int)]
                keys[domain] = (i, spec, unique[j], a["center"][slots[j]], a["native_input_last"][slots[j]])
    require(set(keys) == set(c18.DOMAINS), "Original probe selection unavailable")
    rows, mats = [], {}
    with c18.c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for domain, (i, spec, target, centers, last) in keys.items():
            if domain == "RIDI":
                raw, observed, dt, _, _, hashes = c18.c17.trained_phone(spec, p15prep["ridi_eligibility"][i - 63])
            else:
                raw, observed, dt, _, _, hashes = c18.c4.train_raw(spec, yt["train"], mats)
            old = selection["records"][i]
            require(old["source_sha256"] == hashes and old["record"] == spec["record"], "Original probe input changed")
            _, q, at, release, windows, _, _, _ = c18.c17.front_end(raw, observed, dt, cfg)
            pos = np.searchsorted(release, target)
            np.testing.assert_array_equal(release[pos], target)
            np.testing.assert_array_equal(at[pos], last)
            for j, n in enumerate(pos):
                rows.append({"domain": domain, "record": spec["record"], "native_release": int(target[j]),
                             "history": windows[n], "q_last": q[at[n]], "center": centers[j]})
    return rows


def main():
    began = perf_counter()
    protocol, pair, train, failure = (read(H / f) for f in ("protocol.json", "pair-lock.json", "training.json", "failure.json"))
    protocol_sha, pair_sha = sha(H / "protocol.json"), sha(H / "pair-lock.json")
    require(pair["protocol_sha256"] == protocol_sha and pair["training_sha256"] == sha(H / "training.json")
            and pair["pid"] == train["single_pid"] == protocol["pid"] == failure["pid"]
            and sha(H / "run_pair.before_audit_fix.py.txt") == protocol["source_worker_sha256"],
            "Locked A/B provenance mismatch")
    require(sha(R / "C017/coefficient-lock.json") == pair["C017_coefficient_lock_sha256"]
            and train["fit_count"] == 2
            and all(sha(H / f"{m}-readout.pt") == pair["readout_sha256"][m] for m in "AB"),
            "C017 lock or trained readout changed")
    recovery_path = H / "replay-recovery.json"
    recovery = {"pid": os.getpid(), "started_at": stamp(), "original_fitting_pid": pair["pid"], "pair_lock_sha256": pair_sha,
                "protocol_sha256": protocol_sha, "C017_coefficient_lock_sha256": pair["C017_coefficient_lock_sha256"],
                "original_failure_sha256": sha(H / "failure.json"), "corrected_worker_sha256": sha(H / "run_pair.py"),
                "resume_worker_sha256": sha(Path(__file__)), "optimizer_steps_repeated": 0,
                "change": "Pass single probe angles as one-row arrays. Restore audited original probe windows and replay the frozen A/B."}
    write(recovery_path, recovery)
    print("C018 recovery", recovery, flush=True)

    def status(name, **details):
        write(H / "process.json", {"status": name, "pid": os.getpid(), "original_fitting_pid": pair["pid"],
                                  "started_at": protocol["started_at"], "updated_at": stamp(), "protocol_sha256": protocol_sha,
                                  "pair_lock_sha256": pair_sha, "recovery_sha256": sha(recovery_path), **details})

    status("learned_branch_audits")
    try:
        require(all(sha(R / name) == h for name, h in protocol["C017_preflight_files_sha256"].items())
                and all(sha(c18.PILOT / name) == h for name, h in protocol["inherited_input_sha256"].items()),
                "A frozen inherited input changed after both fits")
        campaign, p14, lock = (read(R / n) for n in ("protocol.json", "C014/protocol.json", "C017/coefficient-lock.json"))
        cfg = c18.c3.Settings(**read(R / "C003/protocol.json")["thresholds"])
        models, mean, std, _, _ = c18.c17.integrity_models(p14, campaign)
        layers, averages, scales = [], [], []
        for arm in "AB":
            saved = torch.load(H / f"{arm}-readout.pt", map_location=mean.device, weights_only=True)
            require(saved["protocol_sha256"] == protocol_sha and saved["steps"] == 160 and saved["gain"] == c18.GAIN,
                    "Locked readout does not match C018")
            layer = nn.Linear(c18.WIDTH, 3, device=mean.device)
            layer.load_state_dict(saved["state_dict"])
            layers.append(layer.eval().requires_grad_(False))
            averages.append(saved["feature_mean"])
            scales.append(saved["feature_std"])
        averages, scales = torch.stack(averages), torch.stack(scales)
        prep_path = H / "training-preparation.json"
        require(sha(prep_path) == train["training_preparation_sha256"], "Frozen training preparation changed")
        prep = read(prep_path)
        inventory = read(R / "C001/inventory.json")
        selection = read(R / "C017/external-selection.json")
        probes = recover_probes(inventory, cfg, selection, prep["training_inputs_sha256"])
        probe_path = H / "old-context-intervention.json"
        write(probe_path, {"pair_lock_sha256": pair_sha, **c18.old_context_probes(probes, models[0], mean, std, layers, averages, scales)})
        audit_path = H / "causal-rotation-audit.json"
        write(audit_path, {"pair_lock_sha256": pair_sha, **c18.audits(cfg, models, mean, std, lock["k"], layers, averages, scales)})
        status("development_replay")
        print("C018 learned-branch audits passed; scoring 213 original development records", flush=True)
        tick = perf_counter()
        result = c18.replay(read(R / "C017/results.json"), inventory, campaign, cfg, read(c18.TRANSFER / "external-split.json"),
                            models, mean, std, lock["k"], layers, averages, scales)
        require(sha(R / "C017/coefficient-lock.json") == pair["C017_coefficient_lock_sha256"] and sha(H / "pair-lock.json") == pair_sha
                and all(sha(H / f"{m}-readout.pt") == pair["readout_sha256"][m] for m in "AB"), "Locked readouts changed during replay")
        result.update({"pid": os.getpid(), "original_fitting_pid": pair["pid"], "started_at": protocol["started_at"],
                       "completed_at": stamp(), "protocol_sha256": protocol_sha, "pair_lock_sha256": pair_sha,
                       "C017_coefficient_lock_sha256": pair["C017_coefficient_lock_sha256"],
                       "scoring_masks_sha256": sha(c18.MASKS), "causal_rotation_audit_sha256": sha(audit_path),
                       "old_context_intervention_sha256": sha(probe_path), "recovery_sha256": sha(recovery_path),
                       "development_replay_seconds": perf_counter() - tick, "resume_total_seconds": perf_counter() - began,
                       "final_holdout_scored": False})
        output = H / "results.json"
        write(output, result)
        success = not result["failures"] and len(result["array_sensor_poses"]) == 96 and result["pooled_records"]["records"] == 213
        status("complete" if success else "incomplete", results_sha256=sha(output), scored=result["pooled_records"]["records"])
        print("C018 finished", result["pooled_records"], "failures", result["failures"], flush=True)
        return 0 if success else 1
    except BaseException as exc:
        write(H / "resume-failure.json", {"pid": os.getpid(), "failed_at": stamp(), "error": repr(exc)})
        raise


if __name__ == "__main__":
    sys.exit(main())
