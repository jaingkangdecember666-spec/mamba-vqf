"""C031: synthesize a single frozen C023-family router by averaging logged weights."""
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0,str(ROOT/"C027"))

import numpy as np
from scipy.io import loadmat
import torch
from torch import nn

import run_order_repeat as c27

c23,c22,c18,c17,c3,c2=c27.c23,c27.c22,c27.c18,c27.c17,c27.c3,c27.c2
base,intrinsic=c27.base,c27.intrinsic
read,write,sha,stamp,array_sha=c27.read,c27.write,c27.sha,c27.stamp,c27.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c27.UP,c27.DATA,c27.TRANSFER,c27.MASKS,c27.GATES
require,archive=c27.require,c27.np_archive
check_probabilities,add_baselines=c27.check_probabilities,c27.add_baselines
WEIGHTS=c23.WEIGHTS
PARENTS=("C023","C027")
ROUTERS=(*PARENTS,"C031")
METHODS=(*c23.METHODS[:4],"C027_deg","C031_deg",*c23.METHODS[4:])
BATCH_KEYS=("input_feature_sha256","native_release_sha256","native_input_last_sha256",
            "source_record_ordinal_sha256","legacy_slot_identity_sha256")
STATE_KEYS=("feature_mean","feature_std","p0","log_p0")


def status(protocol,phase,**details):
    write(HERE/"process.json",{"pid":os.getpid(),"started_at":protocol["started_at"],"updated_at":stamp(),
                               "status":phase,"protocol_sha256":sha(HERE/"protocol.json"),**details})


def verify_inputs():
    c023,files=c27.verify_c023()
    prior,receipt,lock=(read(ROOT/"C027"/n) for n in ("protocol.json","process.json","router-lock.json"))
    require(receipt["status"]=="complete" and receipt["scored"]==213 and lock["fit_count"]==1 and
            lock["optimizer_steps"]==160 and c023["development_records"]==prior["development_records"] and
            prior["scoring_masks_sha256"]==c023["scoring_masks_sha256"]==sha(MASKS) and
            prior["C022_k"]==c023["C022_k"],"C027 and C023 protocol comparability changed")
    for n,digest in prior["inputs_sha256"].items():
        require(files.setdefault(n,digest)==digest,"C023 and C027 disagree on a frozen input hash")
    files.update({n:h for n,h in (
        ("C027/router-lock.json",receipt["router_lock_sha256"]),("C027/results.json",receipt["results_sha256"]),
        ("C027/router.pt",lock["checkpoint_sha256"]),("C027/run_order_repeat.py",prior["worker_sha256"]),
        ("C027/process.json",sha(ROOT/"C027/process.json")),
    )})
    files.update({f"C027/{n}":h for n,h in lock["artifact_sha256"].items()})
    require(all(sha(ROOT/n)==h for n,h in files.items()),"A frozen C023/C027 source or checkpoint changed")
    return c023,prior,files


def checkpoint(name):
    return torch.load(ROOT/name/"router.pt",map_location="cpu",weights_only=True)


def frozen_linear(state):
    model=nn.Linear(24,2,dtype=torch.float64)
    model.load_state_dict(state["state_dict"])
    return model.eval().requires_grad_(False)


def load_trajectory(name,state,parent_pid):
    path=ROOT/name/"optimizer-steps.jsonl"
    with path.open("rb") as stream:
        raw_lines=stream.read().splitlines()
    lines=[json.loads(raw) for raw in raw_lines]
    require(len(lines)==160,"Each parent must have exactly 160 archived optimizer updates")
    previous=np.zeros(50,dtype=np.float64)
    for step,row in enumerate(lines,1):
        require(row["step"]==step and row["pid"]==parent_pid,"Invalid original optimizer step/PID")
        np.testing.assert_array_equal(row["params_before"],previous)
        previous=np.asarray(row["params_after"],dtype=np.float64)
        require(previous.shape==(50,) and np.isfinite(previous).all(),"Invalid post-update parameters")
    saved=torch.cat([state["state_dict"][name].flatten() for name in ("weight","bias")]).numpy()
    np.testing.assert_array_equal(previous,saved)
    return lines,raw_lines


def synthesize(states,original,source,x,files):
    schedule=read(ROOT/"C027/sampling-lock.json")
    require(schedule["source_batch_index_base"]==0 and len(schedule["permutation"])==160,
            "C027 archived permutation is missing")
    np.testing.assert_array_equal(np.sort(schedule["permutation"]),np.arange(160))
    trajectories={}
    for name in PARENTS:
        state=states[name]
        require(state["optimizer_steps"]==160 and state["state_dict"]["weight"].shape==(2,24) and
                state["state_dict"]["bias"].shape==(2,),"Unexpected parent architecture")
        trajectories[name]=load_trajectory(name,state,read(ROOT/name/"router-lock.json")["pid"])
        for key in STATE_KEYS:
            torch.testing.assert_close(state[key],states["C023"][key],rtol=0.,atol=0.)
    steps23=trajectories["C023"][0]
    steps27=trajectories["C027"][0]
    for i,(row23,row27) in enumerate(zip(steps23,steps27,strict=True)):
        cut=slice(i*96,(i+1)*96)
        fingerprints={k:array_sha(v,dtype) for k,v,dtype in (
            ("input_feature_sha256",x["z"][cut].numpy(),"<f8"),
            ("native_release_sha256",source["native_release"][cut],"<i8"),
            ("native_input_last_sha256",source["native_input_last"][cut],"<i8"),
            ("source_record_ordinal_sha256",source["source_record_ordinal"][cut],"<i8"),
            ("legacy_slot_identity_sha256",original["global_window"][cut],"<i8"),
        )}
        index=row27["source_batch_index"]
        require(index==schedule["permutation"][i] and all(row23[k]==v for k,v in fingerprints.items()) and
                all(row27[key]==steps23[index][key] for key in BATCH_KEYS),
                "Archived optimizer batch identity disagrees with the common source slots")
    selected=[]
    origin=[]
    for name in PARENTS:
        rows,raw=trajectories[name]
        for i in range(80,160):
            row=rows[i]
            after=np.asarray(row["params_after"],dtype=np.float64)
            source_batch=row["source_batch_index"] if name=="C027" else i
            selected.append(after)
            origin.append({
                "vector_index":len(selected)-1,"parent":name,"optimizer_step":i+1,"source_batch_index":source_batch,
                "archived_optimizer_pid":row["pid"],"draw_mass":1./160.,
                "archived_step_line_sha256":hashlib.sha256(raw[i]).hexdigest(),
                "params_after_sha256":array_sha(after,"<f8"),**{k:row[k] for k in BATCH_KEYS},
            })
    vector_bank=np.stack(selected,axis=0)
    require(vector_bank.shape==(160,50),"Expected 80 after-update vectors from each parent")
    average=np.mean(vector_bank,axis=0,dtype=np.float64)
    np.savez_compressed(HERE/"synthesis-vectors.npz",params_after=vector_bank,arithmetic_mean=average,
                        parent=np.asarray([r["parent"] for r in origin]),
                        optimizer_step=np.asarray([r["optimizer_step"] for r in origin]),
                        source_batch_index=np.asarray([r["source_batch_index"] for r in origin]))
    write(HERE/"synthesis-lock.json",{
        "pid":os.getpid(),"locked_at":stamp(),"operation":"Single float64 arithmetic mean of 160 post-update vectors",
        "steps_inclusive":[81,160],"source_batch_index_base":0,
        "vectors_per_parent":dict.fromkeys(PARENTS,80),"mass_per_parent":dict.fromkeys(PARENTS,.5),
        "sample_mass":1./160.,"optimizers_run":0,"new_optimizer_steps":0,"new_training_records":0,
        "parameter_order":["weight[2,24] row-major","bias[2]"],
        "parent_logs_sha256":{name:files[f"{name}/optimizer-steps.jsonl"] for name in PARENTS},
        "parent_checkpoint_sha256":{name:files[f"{name}/router.pt"] for name in PARENTS},
        "vector_bank_sha256":array_sha(vector_bank,"<f8"),
        "averaged_params_sha256":array_sha(average,"<f8"),
        "vectors_archive_sha256":sha(HERE/"synthesis-vectors.npz"),"contributors":origin,
    })
    state={**{k:states["C023"][k] for k in (*STATE_KEYS,"dtype","C022_coefficient_lock_sha256")},
           "state_dict":{"weight":torch.from_numpy(average[:48].reshape(2,24)),"bias":torch.from_numpy(average[48:])},
           "optimizer_steps":0,"fit_count":0,"synthesis_vectors":160,
           "synthesis_lock_sha256":sha(HERE/"synthesis-lock.json"),
           "protocol_sha256":sha(HERE/"protocol.json")}
    torch.save(state,HERE/"router.pt")
    return state


def verify_source(states,original,source,x):
    """Replay only the 139 frozen shared source acquisitions to audit the synthetic readout."""
    baselines={name:archive(ROOT/name/"training-io.npz") for name in PARENTS}
    for name,old in baselines.items():
        np.testing.assert_array_equal(x["z"].numpy(),old["normalized_input"])
        np.testing.assert_array_equal(x["prob0"].numpy(),old["p0"])
        np.testing.assert_array_equal(x["enabled"].numpy(),old["enabled"])
        np.testing.assert_array_equal(source["native_release"],old["native_release"])
    shape=x["z"].shape
    require(shape==(15360,24) and np.unique(source["source_record_ordinal"]).size==139,
            "Unexpected frozen source audit shape")
    truth=x["truth"].numpy()
    inactive=~x["enabled"]
    arrays={}
    errors={}
    for name,state in states.items():
        model=frozen_linear(state)
        with torch.inference_mode():
            chord, up, shift, prob=c23.losses(model,x,slice(None))
            torch.testing.assert_close(up[inactive],c18.unit(x["center"])[inactive],rtol=0.,atol=0.)
            require(torch.count_nonzero(shift[inactive]).item()==0,"Readout modifies a gated source slot")
        p, direction, correction=prob.numpy(),up.numpy(),shift.numpy()
        check_probabilities(p)
        require(np.max(np.linalg.norm(correction,axis=1))<=c22.CAP+2e-15,"Readout exceeded capped filter responses")
        if name in PARENTS:
            saved=baselines[name]
            np.testing.assert_array_equal(p,saved["prob_trained"])
            np.testing.assert_array_equal(direction,saved[f"{name}_prediction"])
            np.testing.assert_array_equal(chord.numpy(),saved[f"{name}_loss"])
            np.testing.assert_array_equal(correction,saved[f"{name}_delta"])
        angle=np.degrees(np.arctan2(np.linalg.norm(np.cross(direction,truth),axis=1),np.einsum("ij,ij->i",direction,truth)))
        arrays.update({f"{name}_prob":p,f"{name}_direction":direction,f"{name}_chord":chord.numpy(),f"{name}_tilt_deg":angle})
        errors[name]=angle

    rows=[]
    ordinal=source["source_record_ordinal"]
    for i in np.unique(ordinal):
        ix=ordinal==i
        active=ix&source["enabled"]
        entry={"record_ordinal":int(i),"record":str(source["source_record_name"][ix][0]),
               "domain":str(source["domain"][ix][0]),"slots":int(ix.sum()),"active_slots":int(active.sum())}
        for name in ROUTERS:
            entry[name]={"tilt_RMSE_deg":float(np.sqrt(np.mean(errors[name][ix]**2))),
                         "active_probability_mean":dict(zip(WEIGHTS,arrays[f"{name}_prob"][active].mean(0).tolist(),strict=True))
                                                   if active.any() else None}
        rows.append(entry)
    def summarize(items):
        return {"records":len(items),"slots":sum(row["slots"] for row in items),
                "record_macro_tilt_RMSE_deg":{name:fmean(r[name]["tilt_RMSE_deg"] for r in items) for name in ROUTERS}}
    np.savez_compressed(HERE/"source-replay.npz",**arrays,
                        **{k:source[k] for k in ("domain","source_record_ordinal","source_record_name","native_release","native_input_last","enabled")})
    write(HERE/"source-verification.json",{
        "pid":os.getpid(),"audit_scope":"Forward-only comparison on the existing C023/C027 shared source slots",
        "parent_predictions_and_probabilities_exact":True,"optimizer_steps":0,"new_training_records":0,
        "all":summarize(rows),"domains":{d:summarize([r for r in rows if r["domain"]==d]) for d in c23.DOMAINS},
        "records":rows,"source_replay_sha256":sha(HERE/"source-replay.npz"),
        "shared_source_slot_identity_sha256":array_sha(original["global_window"],"<i8"),
    })



@torch.inference_mode()
def held(q, at, release, history, acc, raw, observed, dt, k, fixed, routers):
    """Score both routers against a common B6 and filter packet."""
    original, center, _, rest, valid = c22.base_packet(q, at, release, history, acc, observed, dt, fixed)
    fast, slow, mix, *_ = c22.filters(raw, observed, acc, q, dt, at)
    active = valid & ~rest
    models, average, scale, log0 = routers
    backbones, mean, std, *_ = fixed
    features = c18.features(history, q[at], backbones[0], mean, std)
    np.testing.assert_array_equal(features.valid, valid)
    normalized = torch.from_numpy((features.pooled[:, 1].astype(np.float64) - average) / scale)
    directions = {"C020_deg": original, "C022_deg": c23.augment(original, center, k * np.where(active[:, None], mix, 0.))}
    probabilities, deltas = {}, {}
    for name, model in models.items():
        p = c23.weights(model, normalized, log0).numpy()
        check_probabilities(p)
        extra = np.where(active[:, None], p[:, :1] * fast + p[:, 1:2] * slow, 0.)
        require(np.max(np.linalg.norm(extra, axis=1)) <= c22.CAP + 2e-15 and np.all(extra[:, 2] == 0.),
                "Convex auxiliary correction exceeds the prescribed cap")
        probabilities[name] = p
        deltas[name] = extra
        directions[name + "_deg"] = c23.augment(original, center, extra)
    return directions, probabilities, deltas, active, rest



def metrics(rows):
    keep = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in keep) for k in METHODS} if keep else {}
    active = sum(r["active_release_count"] for r in keep)
    weighted = {}
    if active:
        weighted = {key: fmean(r[f"p_{key}_mean"] for r in keep if r["active_release_count"] > 0) for key in WEIGHTS}
    return {"records": len(keep), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in keep),
            "macro_tilt_rmse_deg": macro, "active_releases": active, "mean_routing_prob_equal_records_with_support": weighted,
            "C031_delta_deg": {m: macro["C031_deg"] - macro[m] for m in METHODS if m != "C031_deg"} if keep else {},
            "C031_wins": {m: sum(r["C031_deg"] < r[m] for r in keep) for m in METHODS if m != "C031_deg"}}


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
                        array_sha(probabilities["C027"][active], "<f8") == old["active_routing_sha256"] and
                        array_sha(deltas["C023"], "<f8") == old["C023_delta_sha256"] and
                        array_sha(deltas["C027"], "<f8") == old["C027_delta_sha256"],
                        "Frozen C023/C027 release count, probabilities, or deltas changed")
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C027_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C027 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C027_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C031_deg"][~active_scored], errs["C020_deg"][~active_scored])
                active_probs = {m:dict(zip(WEIGHTS, p[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                                for m,p in probabilities.items()}
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C027_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, parent_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs["C031"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C027_deg"], score["C031_deg"], flush=True)
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






def replay_locked(protocol):
    lock=read(HERE/"router-lock.json")
    digest=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["fit_count"]==0 and lock["optimizer_steps"]==0 and
            lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/name)==h for name,h in lock["artifact_sha256"].items()),
            "Replay requires the runtime's original C031 synthesized checkpoint")
    states={name:checkpoint(name) for name in ROUTERS}
    own=states["C031"]
    for parent in PARENTS:
        for key in STATE_KEYS:
            torch.testing.assert_close(own[key],states[parent][key],rtol=0.,atol=0.)
    require(own["optimizer_steps"]==0 and own["fit_count"]==0 and own["synthesis_vectors"]==160 and
            own["synthesis_lock_sha256"]==sha(HERE/"synthesis-lock.json"),"A synthesized C031 checkpoint was expected")
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    prior=read(ROOT/"C027/results.json")
    anchor=read(ROOT/"C023/results.json")
    require([[r["dataset"],r["record"]] for r in prior["sequences"]]==protocol["development_records"] and
            [[r["dataset"],r["record"]] for r in anchor["sequences"]]==protocol["development_records"],
            "Parent development replay orders differ")
    models={name:frozen_linear(state) for name,state in states.items()}
    routers=models,own["feature_mean"].numpy(),own["feature_std"].numpy(),own["log_p0"]
    status(protocol,"development_replay",router_lock_sha256=digest,expected_records=213)
    print("C031 synthesized checkpoint locked; replaying the common 213 development records",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),
                  read(ROOT/"C001/inventory.json"),campaign,read(TRANSFER/"external-split.json"),cfg,
                  protocol["C022_k"],fixed,routers)
    require(sha(HERE/"router-lock.json")==digest and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/name)==h for name,h in protocol["inputs_sha256"].items()),"Frozen evidence changed")
    result.update(pid=os.getpid(),router_lock_sha256=digest,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),weights_origin="single arithmetic mean of 160 logged parent vectors",
                  fit_count=0,optimizer_steps=0,new_training_records=0,started_at=protocol["started_at"],
                  completed_at=stamp(),development_replay_seconds=perf_counter()-start,
                  final_holdout_scored=False,supplemental_holdout_scored=False)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(protocol,"complete" if complete else "incomplete",scored=scored,
           router_lock_sha256=digest,results_sha256=sha(out))
    require(complete,"C031 replay incomplete; synthesized checkpoint remains locked")
    print("C031 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def run(launch):
    anchor,prior,files=verify_inputs()
    states={name:checkpoint(name) for name in PARENTS}
    c022=read(ROOT/"C022/coefficient-lock.json")
    require(c022["k"]==anchor["C022_k"]==prior["C022_k"] and
            states["C023"]["C022_coefficient_lock_sha256"]==states["C027"]["C022_coefficient_lock_sha256"]==
            files["C022/coefficient-lock.json"],"C023 or C027 C022 prior changed")
    torch.testing.assert_close(states["C023"]["p0"],c23.prior_probs(c022["k"]),rtol=0.,atol=0.)
    mean,std=(states["C023"][k].numpy() for k in ("feature_mean","feature_std"))
    protocol={**launch,"worker_sha256":sha(Path(__file__)),"inputs_sha256":files,"C022_k":c022["k"],
              "parameters":50,"input_width":24,"C023_feature_definition":anchor["feature"],
              "synthesis_operation":"float64 arithmetic mean of updates 81 through 160 inclusive from each parent",
              "parent_order":list(PARENTS),"parent_weights":[.5,.5],"vector_count":160,
              "fit_count":0,"optimizer_steps":0,"new_training_records":0,
              "source_forward_audit_existing_records":139,"source_forward_audit_existing_slots":15360,
              "development_records":anchor["development_records"],"scoring_masks_sha256":sha(MASKS),
              "formal_holdouts_authorized":False,"supplemental_holdouts_authorized":False}
    write(HERE/"protocol.json",protocol)
    torch.set_num_threads(2)
    status(protocol,"verifying_parent_trajectories")
    original,source,x=c23.load_train(c022,files,mean,std)
    states["C031"]=synthesize(states,original,source,x,files)
    status(protocol,"source_forward_audit",synthesis_lock_sha256=sha(HERE/"synthesis-lock.json"))
    verify_source(states,original,source,x)
    artifacts=("protocol.json","synthesis-lock.json","synthesis-vectors.npz","source-replay.npz","source-verification.json")
    with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
        json.dump({"pid":os.getpid(),"locked_at":stamp(),"fit_count":0,"optimizer_steps":0,"new_training_records":0,
                   "weight_syntheses":1,"vectors":160,"checkpoint_sha256":sha(HERE/"router.pt"),
                   "artifact_sha256":{name:sha(HERE/name) for name in artifacts}},handle,indent=2)
        handle.write("\n")
    replay_locked(protocol)


def main():
    launch={"pid":os.getpid(),"started_at":stamp(),"command":"python -B -u C031/run_average_router.py",
            "scope":"C031 only; one arithmetic synthesis of archived C023/C027 weights, no training, common development replay"}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        run(launch)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":stamp(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

