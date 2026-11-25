"""C038: one arithmetic readout midpoint and common native development replay."""
import json
import os
import sys
import traceback
from pathlib import Path
from statistics import fmean
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "C035"))
import run_trust_router as parent

np, torch, loadmat = parent.np, parent.torch, parent.loadmat
c33, c31, c23, c22, c17, c3, c2, c14 = (
    parent.c33, parent.c31, parent.c23, parent.c22, parent.c17, parent.c3, parent.c2, parent.c14
)
base, intrinsic, Penn, held = parent.base, parent.intrinsic, parent.Penn, parent.held
read, write, sha, stamp, array_sha = parent.read, parent.write, parent.sha, parent.stamp, parent.array_sha
require, add_baselines = parent.require, parent.add_baselines
UP, DATA, TRANSFER, MASKS, GATES = parent.UP, parent.DATA, parent.TRANSFER, parent.MASKS, parent.GATES
WEIGHTS, PENN = parent.WEIGHTS, parent.PENN
ROUTERS = ("C023", "C035", "C038")
METHODS = (*c23.METHODS[:4], "C035_deg", "C038_deg", *c23.FIXED)


def status(launch, phase, **details):
    write(HERE / "process.json", {**launch, "status": phase, "updated_at": parent.local_now(), **details})


def synthesize(protocol):
    states = {name: c33.frozen_state(name) for name in ROUTERS[:2]}
    left, right = states.values()
    metadata = ("dtype", "C022_coefficient_lock_sha256")
    common = (*c33.STATE_KEYS, *metadata)
    for key in c33.STATE_KEYS:
        torch.testing.assert_close(left[key], right[key], rtol=0., atol=0.)
    require(all(left[key] == right[key] for key in metadata), "Router deployment metadata differs")
    for state in (left, right):
        require(set(state["state_dict"]) == {"weight", "bias"} and parent.flat_state(state).size == 50,
                "Expected a single 24-to-2 float64 readout")
    midpoint = {
        key: (left["state_dict"][key] + right["state_dict"][key]) / 2
        for key in ("weight", "bias")
    }
    stored = {**{key: left[key] for key in common}, "state_dict": midpoint, "optimizer_steps": 0,
              "fit_count": 0, "synthesis_fraction": .5, "protocol_sha256": sha(HERE / "protocol.json")}
    target = HERE / "router.pt"
    with target.open("xb") as stream:
        torch.save(stored, stream)
    actual = c33.frozen_state("C038")
    for key in c33.STATE_KEYS:
        torch.testing.assert_close(actual[key], left[key], rtol=0., atol=0.)
    require(all(actual[key] == left[key] for key in metadata), "Saved router deployment metadata differs")
    np.testing.assert_array_equal(parent.flat_state(actual), (parent.flat_state(left) + parent.flat_state(right)) / 2)
    states["C038"] = actual
    return states


def source_audit(protocol, states):
    ledger = parent.archive(ROOT / "C033/training-inputs.npz")
    original = parent.archive(ROOT / "C022/training-slots.npz")
    penn = parent.archive(ROOT / "C033/penn-training-slots.npz")
    expected = parent.archive(ROOT / "C035/training-io.npz")
    original_anchor = parent.archive(ROOT / "C023/training-io.npz")
    joined = ("center", "fast", "slow", "truth", "enabled", "source_record_ordinal", "source_record_name",
              "domain", "native_release", "native_input_last")
    require(ledger["z"].shape == (160, 98, 24), "Original source slot shape changed")
    for key in joined:
        np.testing.assert_array_equal(ledger[key][:, :96].reshape(original[key].shape), original[key])
        np.testing.assert_array_equal(ledger[key][:, 96:], penn[key])
    np.testing.assert_array_equal(ledger["z"][:, :96].reshape(-1, 24), original_anchor["normalized_input"])
    np.testing.assert_array_equal(ledger["z"][:, 96:], penn["z"])
    z = ledger["z"].reshape(-1, 24)
    fields = {key: ledger[key].ravel() for key in joined[4:]}
    for key, values in fields.items():
        np.testing.assert_array_equal(values, expected[key])
    ordinals = fields["source_record_ordinal"]
    ids = np.unique(ordinals)
    require(len(ids) == 141 and len(z) == 15680, "Unexpected original source record or slot count")
    directions = {key: ledger[key].reshape(-1, 3) for key in ("center", "fast", "slow", "truth")}
    enabled = fields["enabled"]
    prior = states["C023"]["p0"].numpy()
    arrays = {}
    gaps = {}
    for name, state in states.items():
        direction, _, _, p = parent.prediction(
            z, prior, parent.flat_state(state), directions["center"], directions["fast"], directions["slow"], enabled
        )
        parent.check_probabilities(p)
        with torch.inference_mode():
            deployed = c23.weights(c31.frozen_linear(state), torch.from_numpy(z), state["log_p0"]).numpy()
        np.testing.assert_allclose(deployed, p, rtol=0., atol=2e-15)
        gaps[name] = float(np.max(np.abs(deployed - p)))
        if name in ROUTERS[:2]:
            np.testing.assert_array_equal(p, expected[name + "_prob"])
            np.testing.assert_array_equal(direction, expected[name + "_direction"])
        else:
            np.testing.assert_array_equal(direction[~enabled], expected["C023_direction"][~enabled])
        truth = directions["truth"]
        angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(direction, truth), axis=1),
                                     np.einsum("ij,ij->i", direction, truth)))
        arrays.update({name + "_prob": p, name + "_direction": direction, name + "_angle_deg": angle})
    rows = []
    for ordinal in ids:
        use = ordinals == ordinal
        row = {"record_ordinal": int(ordinal), "domain": str(fields["domain"][use][0]),
               "record": str(fields["source_record_name"][use][0]), "slots": int(use.sum())}
        row["rmse_deg"] = {name: float(np.sqrt(np.mean(arrays[name + "_angle_deg"][use] ** 2))) for name in ROUTERS}
        rows.append(row)
    output = HERE / "source-replay.npz"
    np.savez_compressed(output, **fields, **arrays)
    write(HERE / "synthesis-verification.json", {
        "pid": os.getpid(), "fraction": .5, "parameter_count": 50, "new_optimizer_steps": 0, "fit_count": 0,
        "original_forward_audit_records": len(ids), "original_forward_audit_slots": len(z),
        "parent_sha256": {name: protocol["inputs_sha256"][name + "/router.pt"] for name in ROUTERS[:2]},
        "checkpoint_sha256": sha(HERE / "router.pt"), "shared_normalization_and_prior_exact": True,
        "midpoint_checkpoint_exact": True, "archived_parent_predictions_and_probabilities_exact": True,
        "deployed_forward_probability_max_abs_difference": gaps,
        "original_source_slot_bank_sha256": protocol["inputs_sha256"]["C033/training-inputs.npz"],
        "source_replay_sha256": sha(output), "domain_key_present": True, "source_records": rows,
        "domains": {str(name): {model: fmean(row["rmse_deg"][model] for row in rows if row["domain"] == name)
                               for model in ROUTERS} for name in np.unique(fields["domain"])},
    })





def metrics(rows):
    keep = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in keep) for k in METHODS} if keep else {}
    active = sum(r["active_release_count"] for r in keep)
    weighted = {}
    if active:
        weighted = {key: fmean(r[f"p_{key}_mean"] for r in keep if r["active_release_count"] > 0) for key in WEIGHTS}
    return {"records": len(keep), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in keep),
            "macro_tilt_rmse_deg": macro, "active_releases": active, "mean_routing_prob_equal_records_with_support": weighted,
            "C038_delta_deg": {m: macro["C038_deg"] - macro[m] for m in METHODS if m != "C038_deg"} if keep else {},
            "C038_wins": {m: sum(r["C038_deg"] < r[m] for r in keep) for m in METHODS if m != "C038_deg"}}


def replay(prior, fixed_scores, inventory, campaign, split, cfg, k, fixed, router):
    c1r, c1p, c3r, c2p = (read(ROOT / n) for n in ("C001/results.json", "C001/protocol.json", "C003/results.json", "C002/protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in x["sequences"]} for x in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    spec = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    locked_fixed = {(r["dataset"], r["record"]): r for r in fixed_scores["records"]}
    base.require(len(keys) == len(locked_fixed) == 213 and set(spec) == keys == set(locked_fixed)
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in spec.values())
                 and fixed_scores["source_csv_sha256"] == sha(ROOT / "C020/sequences.csv")
                 and fixed_scores["source_candidate_column"] == "C020_deg" and fixed_scores["selection_lock_sha256"] is None,
                 "C023 or archived VQF control has different scope")
    mp, ap = DATA / "Sassari/medium_v5.mat", DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp) == split["source_file_sha256"][str(mp)] and sha(ap) == evidence["array_reference_sha256"], "Archived references changed")
    medium = loadmat(mp, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array = c2.reference_array(ap)
    old_poses = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    rows, poses, failures, ledger = [], [], [], []
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for i, old in enumerate(prior["sequences"], 1):
            start = perf_counter()
            key = old["dataset"], old["record"]
            identity = {f: old[f] for f in c17.IDENTITY}
            try:
                packet, raw, observed, dt, ref, mask, axes, _ = c17.payload(key, spec[key], map3, map1, medium, yt["valid"],
                                                                            split, c1p, array, evidence, cfg)
                q0, q, at, release, hist, _, acc, _ = packet
                np.testing.assert_array_equal(mask, masks[old["mask_key"]])
                ix = np.flatnonzero(mask)
                fixed_ref = locked_fixed[key]
                base.require(len(ix) == old["scored_frames"] == fixed_ref["scored_frames"]
                             and array_sha(ix, "<i8") == old["scoring_indices_sha256"] == fixed_ref["scoring_indices_sha256"]
                             and array_sha(at, "<i8") == old["native_input_last_sha256"]
                             and array_sha(release, "<i8") == old["native_release_sha256"], "Archived fixed VQF/control native scoring mismatch")
                tick = perf_counter()
                predictions, probabilities, deltas, active, rest = held(q, at, release, hist, acc, raw, observed, dt, k, fixed, router)
                pred_seconds = perf_counter() - tick
                require(int(active.sum()) == old["active_release_count"] and
                        array_sha(probabilities["C023"][active], "<f8") == old["C023_active_routing_sha256"] and
                        array_sha(probabilities["C035"][active], "<f8") == old["C035_active_routing_sha256"] and
                        array_sha(deltas["C023"], "<f8") == old["C023_delta_sha256"] and
                        array_sha(deltas["C035"], "<f8") == old["C035_delta_sha256"],
                        "Frozen C023/C035 release count, probabilities, or deltas changed")
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C035_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C035 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C035_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C038_deg"][~active_scored], errs["C020_deg"][~active_scored])
                active_probs = {m:dict(zip(WEIGHTS, p[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                                for m,p in probabilities.items()}
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C035_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, parent_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs["C038"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C035_deg"], score["C038_deg"], flush=True)
            except (AssertionError, RuntimeError, OSError, ValueError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - start})
                print("FAILED", *key, repr(exc), flush=True)
            if i % 10 == 0 or i == len(prior["sequences"]):
                write(HERE / "replay-progress.json", {"pid": os.getpid(), "attempted": i, "scored": i-len(failures),
                                                     "failures": failures, "last_record": list(key), "updated_at": stamp()})
    seq, pos, group, fixed_file = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json", "fixed-vqf-comparison.json"))
    fields = dict.fromkeys(f for r in rows for f in r)
    intrinsic.shice.save_csv(seq, [{f: r.get(f, "") for f in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pos, poses)
    routing = HERE/"development-routing.npz"
    if ledger:
        np.savez_compressed(routing,**{k:np.concatenate([p[k] for p in ledger]) for k in ledger[0]})
    groups = {"TUM_trajectories": {r["record"]: metrics([r]) for r in rows if r["dataset"] == "TUM-VI"},
              "Oxford_carry": {g: metrics([r for r in rows if r["dataset"] == "OxIOD" and r["group"] == g])
                               for g in sorted({r["group"] for r in rows if r["dataset"] == "OxIOD"})},
              "group": {f"{d}/{g}": metrics([r for r in rows if r["dataset"] == d and r["group"] == g])
                        for d, g in sorted({(r["dataset"], r["group"]) for r in rows})}}
    write(group, groups)
    overall, by_domain = metrics(rows), {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]}
    write(fixed_file, {"archived_controls_sha256": sha(ROOT / "review/vqf_response_controls_development.json"),
                       "matched_legacy_source_C020_csv_sha256": fixed_scores["source_csv_sha256"],
                       "archived_fixed_results_reused": True, "each_record_mask_identity_checked": True,
                       "global": overall, "domains": by_domain, "TUM_trajectories": groups["TUM_trajectories"], "Oxford_carry": groups["Oxford_carry"]})
    return {"pooled_records": overall, "domains": by_domain, "TUM_trajectories": groups["TUM_trajectories"], "Oxford_carry": groups["Oxford_carry"],
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifact_sha256": {p.name: sha(p) for p in (seq, pos, group, fixed_file, routing) if p.is_file()}}












def score_penn(protocol,fixed,cfg,routers):
    ds=Penn()
    require(not ds.formal and all(ds.roles[name]=="external_training" for name in PENN),
            "Only the explicitly registered Penn A training records are numerically accessible")
    controls={r["record"]:r for r in read(ROOT/"C037/penn-training-source-scores.json")["records"]}
    rows,masks=[],{}
    for name in PENN:
        payload,quality=ds.native(name)
        raw,hz,ref,mask=payload
        observed=np.isfinite(raw).all(axis=1)
        q0,q,at,release,hist,_,acc,_=c17.front_end(raw,observed,1./hz,cfg)
        dirs,probs,_,active,_=held(q,at,release,hist,acc,raw,observed,1./hz,protocol["C022_k"],fixed,routers)
        ix=np.flatnonzero(mask)
        old=controls[name]
        require(len(ix)==old["scored_frames"] and array_sha(ix,"<i8")==old["mask_sha256"]==quality["scoring_indices_sha256"],
                "Penn A training score mask differs from the original")
        with np.load(ROOT/"C033"/f"{name}-native-mask-evidence.npz",allow_pickle=False) as native:
            np.testing.assert_array_equal(mask,native["scoring_mask"])
            np.testing.assert_array_equal(release,native["release"])
            np.testing.assert_array_equal(at,native["native_input_last"])
        errors=c17.measure(ix,ref,q0,q,release,{k:np.vstack((UP,d)) for k,d in dirs.items()})
        results=c17.scores(errors)
        old_scores=old["full_record_tilt_RMSE_deg"]
        for now,prior in (("C023_deg","C023_deg"),("C035_deg","C035_deg"),("full_vqf_deg","full_vqf_deg")):
            require(abs(results[now]-old_scores[prior])<=1e-6,"Archived Penn A training control changes")
        gyro=np.ascontiguousarray(raw[:,3:])
        accel=np.ascontiguousarray(raw[:,:3])
        for tau,key,old_key in ((1.5,"fixed_tau1p5_deg","fixed_tau1p5_deg"),
                                (3.,"fixed_tau3_deg","full_vqf_deg"),
                                (6.,"fixed_tau6_deg","fixed_tau6_deg")):
            qtau=c22.vf(gyro,accel,1./hz,tau)
            up=c14.rotate(c14.normalize(qtau[ix]),ref[ix])
            scores=c14.tilt_error(np.broadcast_to(UP,up.shape),up)
            results[key]=float(np.sqrt(np.mean(scores**2)))
            require(abs(results[key]-old_scores[old_key])<=1e-6,"Penn A fixed VQF changed")
        masks[name+"_scoring_mask"]=mask
        masks[name+"_scoring_indices"]=ix
        masks[name+"_release"]=release
        masks[name+"_native_input_last"]=at
        rows.append({"record":name,"scored_frames":len(ix),"mask_sha256":array_sha(ix,"<i8"),
                     "native_release_sha256":array_sha(release,"<i8"),
                     "native_input_last_sha256":array_sha(at,"<i8"),
                     "full_record_tilt_RMSE_deg":results,
                     "active_release_probabilities":{
                         n:dict(zip(WEIGHTS,p[active].mean(axis=0).tolist(),strict=True)) for n,p in probs.items()
                     }})
    np.savez_compressed(HERE/"penn-A-native-masks.npz",**masks)
    write(HERE/"penn-training-source-scores.json",{"pid":os.getpid(),"scope":"AF/AS native full-track training-side score",
          "checkpoint_sha256":sha(HERE/"router.pt"),"native_mask_file_sha256":sha(HERE/"penn-A-native-masks.npz"),
          "records":rows,"record_equal_mean":{m:fmean(row["full_record_tilt_RMSE_deg"][m] for row in rows)
                                              for m in rows[0]["full_record_tilt_RMSE_deg"]}})




def run(launch, protocol):
    torch.set_num_threads(2)
    status(launch, "synthesizing_one_readout")
    states = synthesize(protocol)
    source_audit(protocol, states)
    files = ("router.pt", "synthesis-verification.json", "source-replay.npz")
    fingerprints = {name: sha(HERE / name) for name in files}
    lock = {"pid": os.getpid(), "locked_at": parent.local_now(), "synthesis_fraction": .5, "fit_count": 0,
            "neural_fits": 0, "solve_count": 0, "new_optimizer_steps": 0, "checkpoint_sha256": fingerprints["router.pt"],
            "protocol_sha256": launch["protocol_sha256"], "artifact_sha256": fingerprints}
    with (HERE / "router-lock.json").open("x", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2)
        stream.write("\n")
    lock_hash = sha(HERE / "router-lock.json")
    anchor = states["C023"]
    models = {name: c31.frozen_linear(state) for name, state in states.items()}
    routers = models, anchor["feature_mean"].numpy(), anchor["feature_std"].numpy(), anchor["log_p0"]
    fixed, campaign = c22.load_fixed(read(ROOT / "C022/protocol.json"))
    config = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    status(launch, "scoring_Penn_A_training_source", router_lock_sha256=lock_hash)
    score_penn(protocol, fixed, config, routers)
    prior = read(ROOT / "C037/results.json")
    require([[r["dataset"], r["record"]] for r in prior["sequences"]] == protocol["development_records"],
            "C037 original development records differ")
    start = perf_counter()
    status(launch, "development_replay", router_lock_sha256=lock_hash, expected_records=213)
    print("C038 locked; starting the only development replay", flush=True)
    result = replay(prior, read(ROOT / "review/vqf_response_controls_development.json"),
                    read(ROOT / "C001/inventory.json"), campaign, read(TRANSFER / "external-split.json"),
                    config, protocol["C022_k"], fixed, routers)
    verify_protocol(protocol)
    require(sha(HERE / "router-lock.json") == lock_hash and all(sha(HERE / name) == h for name, h in fingerprints.items()),
            "Synthesized checkpoint or evidence changed during replay")
    result.update(pid=os.getpid(), router_lock_sha256=lock_hash, protocol_sha256=launch["protocol_sha256"],
                  scoring_masks_sha256=sha(MASKS), penn_training_source_scores_sha256=sha(HERE / "penn-training-source-scores.json"),
                  synthesis_verification_sha256=fingerprints["synthesis-verification.json"], fit_count=0, neural_fits=0,
                  solve_count=0, new_optimizer_steps=0, replay_seconds=perf_counter() - start, completed_at=parent.local_now(),
                  final_holdout_scored=False, supplemental_holdout_scored=False)
    scored = result["pooled_records"]["records"]
    complete = scored == 213 and not result["failures"] and len(result["array_sensor_poses"]) == 96
    output = HERE / "results.json"
    write(output, result)
    status(launch, "complete" if complete else "incomplete", scored=scored, router_lock_sha256=lock_hash,
           results_sha256=sha(output))
    require(complete, "C038 development replay incomplete")
    print("C038 complete", result["pooled_records"]["macro_tilt_rmse_deg"], flush=True)


def verify_protocol(protocol):
    require(sha(Path(__file__)) == protocol["worker_sha256"], "C038 worker changed")
    for name, expected in protocol["inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"Frozen C038 input changed: {name}")


def main():
    launch = {"pid": os.getpid(), "started_at": parent.local_now(), "command": "python -B -u C038/run_midpoint.py",
              "protocol_sha256": sha(HERE / "protocol.json")}
    with (HERE / "launch.json").open("x", encoding="utf-8") as stream:
        json.dump(launch, stream, indent=2)
        stream.write("\n")
    print("C038 registered PID", launch["pid"], flush=True)
    try:
        protocol = read(HERE / "protocol.json")
        verify_protocol(protocol)
        run(launch, protocol)
    except Exception as error:
        write(HERE / "failure.json", {**launch, "failed_at": parent.local_now(), "error": repr(error)})
        (HERE / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

