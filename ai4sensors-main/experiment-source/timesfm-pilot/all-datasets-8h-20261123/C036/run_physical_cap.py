"""C036: fit one sphere-angle cap over the frozen C023 correction."""
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"C035"),str(ROOT/"review")]

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize_scalar
import torch

import run_trust_router as c35
from anchor_blend_features import included_degrees
from penn_handheld import Penn
from physical_angle_cap import cap_direction

c33,c31,c23,c22,c18,c17,c3,c2,c14=c35.c33,c35.c31,c35.c23,c35.c22,c35.c18,c35.c17,c35.c3,c35.c2,c35.c14
base,intrinsic=c35.base,c35.intrinsic
read,write,sha,stamp,array_sha=c35.read,c35.write,c35.sha,c35.stamp,c35.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c35.UP,c35.DATA,c35.TRANSFER,c35.MASKS,c35.GATES
require,archive,check_probabilities,add_baselines=c35.require,c35.archive,c35.check_probabilities,c35.add_baselines
ROUTERS=("C023","C035")
METHODS=(*c23.METHODS[:4],"C035_deg","physical_only_deg","C036_deg",*c23.FIXED)
WEIGHTS=c23.WEIGHTS
PENN=("af","as")
XTOL=1e-7


def local_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def status(launch,phase,**fields):
    write(HERE/"process.json",{"pid":os.getpid(),"started_at":launch["started_at"],"status":phase,"updated_at":local_now(),
                              "protocol_sha256":sha(HERE/"protocol.json"),**fields})


def separation_degrees(physical,learned):
    return included_degrees(physical,learned)


def capped(physical,learned,disagreement,cap):
    """Use the frozen C023 vector unchanged on the no-cap branch."""
    binding=disagreement>cap
    out=learned.copy()
    if cap==0.:
        out[binding]=physical[binding]
    elif binding.any():
        out[binding]=cap_direction(physical[binding],learned[binding],cap)
    return out,binding


class FitBank:
    def __init__(self,protocol):
        raw="review/raw-anchor-training-141.npz"
        physical="review/external-physical-anchor-141.npz"
        summary=read(ROOT/"review/external-physical-anchor-141.json")
        require(summary["primary_external_bank_sha256"]==protocol["inputs_sha256"][raw] and
                summary["physical_bank_sha256"]==protocol["inputs_sha256"][physical],"Physical and raw anchor archives differ")
        self.raw=archive(ROOT/raw)
        self.physical=archive(ROOT/physical)
        for key in ("domain","record","native_release"):
            np.testing.assert_array_equal(self.raw[key],self.physical[key])
        with np.load(ROOT/"C033/training-inputs.npz",allow_pickle=False) as ledger:
            for key,source in (("domain","domain"),("record","source_record_name"),("record_id","source_record_ordinal"),
                               ("native_release","native_release"),("truth","truth")):
                value=ledger[source]
                np.testing.assert_array_equal(self.raw[key],value.reshape((-1,)+value.shape[2:]))
        self.truth=self.raw["truth"]
        self.reference=self.raw["raw_default"]
        self.learned=self.raw["C023"]
        self.control=self.physical["physical"]
        self.ordinal=self.raw["record_id"]
        self.domain=self.raw["domain"]
        self.record=self.raw["record"]
        require(len(self.ordinal)==15680 and np.unique(self.ordinal).size==141,"Expected exactly 15,680 training slots and 141 acquisitions")
        for value in (self.truth,self.reference,self.learned,self.control):
            require(value.shape==(15680,3) and np.isfinite(value).all() and
                    np.allclose(np.linalg.norm(value,axis=1),1.,rtol=0.,atol=2e-12),"Fit-bank vectors are not finite unit vectors")
        self.disagreement=separation_degrees(self.control,self.learned)
        self.count=np.bincount(self.ordinal)
        self.groups=sorted(set(self.domain))
        self.members=[np.unique(self.ordinal[self.domain==g]) for g in self.groups]
        require(len(self.groups)==5 and len(self.count)==141 and np.all(self.count>0),"Fit-bank source membership changed")
        self.ref,self.learned_risk,self.physical_risk=(self.group_rmse(x)[0] for x in (self.reference,self.learned,self.control))
        for j,d in enumerate(self.groups):
            expected=summary["group_record_means_deg"][d]
            np.testing.assert_allclose([self.ref[j],self.learned_risk[j],self.physical_risk[j]],
                                       [expected[k] for k in ("default","C023","physical")],rtol=0.,atol=2e-13)
        require(np.all(self.ref>0.) and np.max(self.disagreement)>0.,"Degenerate C036 angle search or default denominator")
        strata=read(ROOT/"review/physical-correction-strata.json")
        require(strata["source_sha256"]==protocol["inputs_sha256"][raw] and strata["physical_sha256"]==protocol["inputs_sha256"][physical],
                "Archived separation strata refer to different source vectors")
        thresholds=np.quantile(self.disagreement,[.25,.5,.75,.9])
        np.testing.assert_allclose(thresholds,strata["joint_quartiles_and_p90_deg"],rtol=0.,atol=2e-13)
        penn=(self.domain=="PennCOSYVIO")&(self.disagreement>thresholds[2])
        sqdiff=included_degrees(self.learned,self.truth)**2-included_degrees(self.control,self.truth)**2
        quartile=fmean(float(sqdiff[penn&(self.ordinal==i)].mean()) for i in np.unique(self.ordinal[penn]))
        recorded=strata["groups"]["PennCOSYVIO"]["by_joint_quartile"][-1]["mean_sq_error_model_minus_physical_deg2_across_present_records"]
        require(abs(quartile-recorded)<=2e-12,"Penn upper-quartile evidence changed")
        fit_sources={g:{"records":len(self.members[j]),"slots":int((self.domain==g).sum()),
                        "default_rmse_deg":float(self.ref[j]),"C023_rmse_deg":float(self.learned_risk[j]),
                        "physical_rmse_deg":float(self.physical_risk[j])}
                     for j,g in enumerate(self.groups)}
        write(HERE/"training-identity.json",{"pid":os.getpid(),"slots":15680,"records":141,"source_groups":fit_sources,
          "archive_identity_fields_exact":["domain","record","native_release","record_id","truth"],
          "joint_disagreement_quantiles_deg":thresholds.tolist(),"Penn_top_quartile_C023_minus_physical_deg2":quartile,
          "raw_archive_sha256":protocol["inputs_sha256"][raw],"physical_archive_sha256":protocol["inputs_sha256"][physical]})

    def group_rmse(self,v):
        angles=included_degrees(v,self.truth)
        row=np.sqrt(np.bincount(self.ordinal,weights=angles**2)/self.count)
        return np.asarray([row[ids].mean() for ids in self.members]),row,angles

    def evaluate(self,cap):
        out,binding=capped(self.control,self.learned,self.disagreement,cap)
        group,record,angles=self.group_rmse(out)
        ratio=group/self.ref
        by_group={}
        for j,g in enumerate(self.groups):
            idx=self.domain==g
            n=int(idx.sum())
            by_group[g]={"RMSE_deg":float(group[j]),"ratio_to_default":float(ratio[j]),"cap_slots":int(binding[idx].sum()),
                         "cap_fraction":float(binding[idx].sum()/n),"slots":n}
        return float(ratio.max()),by_group,out,binding,record,angles


def solve(protocol,bank):
    upper=float(np.max(bank.disagreement))
    evaluated=[]
    started_at=local_now()
    start=perf_counter()
    with (HERE/"solver-evaluations.jsonl").open("x",encoding="utf-8") as log:
        def evaluate(x,stage):
            maximum,groups,*_=bank.evaluate(float(x))
            entry={"pid":os.getpid(),"evaluation":len(evaluated)+1,"stage":stage,
                   "cap_deg":float(x),"objective":maximum,"groups":groups}
            print(json.dumps(entry,allow_nan=False),file=log,flush=True)
            evaluated.append(entry)
            return entry

        lower=evaluate(0.,"lower_endpoint")
        high=evaluate(upper,"upper_endpoint")
        original_lower=float((bank.physical_risk/bank.ref).max())
        original_upper=float((bank.learned_risk/bank.ref).max())
        require(abs(lower["objective"]-original_lower)<=2e-13 and abs(high["objective"]-original_upper)<=2e-13,
                "C036 optimization endpoints do not reproduce the locked physical/C023 controls")
        result=minimize_scalar(lambda x:evaluate(x,"bounded_interior")["objective"],bounds=(0.,upper),
                               method="bounded",options={"xatol":XTOL})
        interior=next(row for row in reversed(evaluated) if row["cap_deg"]==float(result.x))
        require(interior["objective"]==float(result.fun) and 0.<=float(result.x)<=upper,"Bounded optimum receipt does not match evaluated point")
        choices={"lower_endpoint":lower,"upper_endpoint":high,"bounded_result":interior}
        name,chosen=min(choices.items(),key=lambda x:(x[1]["objective"],x[1]["cap_deg"]))
    cap=chosen["cap_deg"]
    value,by_group,out,binding,record,angles=bank.evaluate(cap)
    require(abs(chosen["objective"]-value)<=2e-15,"Final cap replay of training score differs")
    np.testing.assert_array_equal(out[~binding],bank.learned[~binding])
    post=separation_degrees(bank.control,out)
    require(np.max(np.abs(post-np.minimum(bank.disagreement,cap)))<=1e-9,"Spherical cap did not preserve requested angular distance")
    np.savez_compressed(HERE/"training-io.npz",C036=out,cap_active=binding,C023_physical_disagreement_deg=bank.disagreement,
                        C036_physical_disagreement_deg=post,C036_angle_error_deg=angles,domain=bank.domain,
                        source_record_ordinal=bank.ordinal,record=bank.record,native_release=bank.raw["native_release"])
    receipt={"pid":os.getpid(),"solver_kind":"physical_angle_cap","solve_count":1,"neural_fits":0,"parameters":1,
             "method":"scipy.optimize.minimize_scalar bounded","scipy_version":scipy.__version__,"xatol":XTOL,
             "bounds_deg":[0.,upper],"started_at":started_at,"completed_at":local_now(),"fit_seconds":perf_counter()-start,
             "result_success":bool(result.success),"result_status":int(result.status),"result_message":str(result.message),
             "nit":int(result.nit),"nfev":int(result.nfev),"evaluations_with_endpoints":len(evaluated),
             "choice":name,"choices":choices,"cap_deg":cap,"objective":value,"group_order":bank.groups,"groups":by_group,
             "active_slots":int(binding.sum()),"cap_active_fraction":float(binding.mean()),"uncapped_slots_exact":int((~binding).sum()),
             "training_record_RMSE_deg":[{"record_id":int(i),"source":str(bank.domain[bank.ordinal==i][0]),
                 "record":str(bank.record[bank.ordinal==i][0]),"slots":int(bank.count[i]),"C036_RMSE_deg":float(record[i])}
                 for i in range(len(bank.count))]}
    write(HERE/"solver-receipt.json",receipt)
    write(HERE/"coefficient-lock.json",{"pid":os.getpid(),"locked_at":local_now(),"cap_deg":cap,
          "objective":value,"cap_bounds_deg":[0.,upper],"group_order":bank.groups,"groups":by_group,
          "cap_active_fraction":float(binding.mean()),"solve_count":1,"neural_fits":0,"choice":name,
          "C023_checkpoint_sha256":protocol["inputs_sha256"]["C023/router.pt"],
          "physical_bank_sha256":protocol["inputs_sha256"]["review/external-physical-anchor-141.npz"],
          "raw_bank_sha256":protocol["inputs_sha256"]["review/raw-anchor-training-141.npz"],
          "protocol_sha256":sha(HERE/"protocol.json"),"solver_receipt_sha256":sha(HERE/"solver-receipt.json"),
          "evaluation_log_sha256":sha(HERE/"solver-evaluations.jsonl"),"training_io_sha256":sha(HERE/"training-io.npz")})
    print("C036 locked cap",cap,"max ratio",value,"choice",name,flush=True)
    return cap




@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,routers,cap):
    original,center,physical,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    active=valid&~rest
    models,average,scale,log0=routers
    backbones,mean,std,*_=fixed
    features=c18.features(history,q[at],backbones[0],mean,std)
    np.testing.assert_array_equal(features.valid,valid)
    z=torch.from_numpy((features.pooled[:,1].astype(np.float64)-average)/scale)
    extra=k*np.where(active[:,None],mix,0.)
    out={"C020_deg":original,"C022_deg":c23.augment(original,center,extra),
         "physical_only_deg":c14.normalize(physical+extra)}
    probabilities,deltas={},{}
    for name,model in models.items():
        p=c23.weights(model,z,log0).numpy()
        check_probabilities(p)
        delta=np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        require(np.max(np.linalg.norm(delta,axis=1))<=c22.CAP+2e-15 and np.all(delta[:,2]==0.),
                "Frozen C023/C035 response escaped the prescribed filter cap")
        out[name+"_deg"]=c23.augment(original,center,delta)
        probabilities[name]=p
        deltas[name]=delta
    dis=separation_degrees(out["physical_only_deg"],out["C023_deg"])
    out["C036_deg"],binding=capped(out["physical_only_deg"],out["C023_deg"],dis,cap)
    return out,probabilities,deltas,active,rest,dis,binding



def metrics(rows):
    keep = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in keep) for k in METHODS} if keep else {}
    active = sum(r["active_release_count"] for r in keep)
    weighted = {}
    if active:
        weighted = {key: fmean(r[f"p_{key}_mean"] for r in keep if r["active_release_count"] > 0) for key in WEIGHTS}
    return {"records": len(keep), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in keep),
            "macro_tilt_rmse_deg": macro, "active_releases": active,
            "cap_affected_scored_frames":sum(r["cap_affected_scored_frames"] for r in keep),
            "cap_affected_release_count":sum(r["cap_affected_release_count"] for r in keep), "mean_routing_prob_equal_records_with_support": weighted,
            "C036_delta_deg": {m: macro["C036_deg"] - macro[m] for m in METHODS if m != "C036_deg"} if keep else {},
            "C036_wins": {m: sum(r["C036_deg"] < r[m] for r in keep) for m in METHODS if m != "C036_deg"}}


def replay(prior, fixed_scores, inventory, campaign, split, cfg, k, fixed, router, cap):
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
                predictions, probabilities, deltas, active, rest, disagreement, binding = held(q, at, release, hist, acc, raw, observed, dt, k, fixed, router, cap)
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
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C035_deg","physical_only_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C035 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C035_deg"][~active_scored], errs["C020_deg"][~active_scored])
                cap_scored=np.r_[False,binding][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C036_deg"][~cap_scored],errs["C023_deg"][~cap_scored])
                active_probs = {m:dict(zip(WEIGHTS, p[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                                for m,p in probabilities.items()}
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C035_deg", "physical_only_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, parent_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs["C023"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "cap_affected_scored_frames":int(cap_scored.sum()),"cap_affected_release_count":int(binding.sum()),
                             "cap_release_mask_sha256":array_sha(binding,"|u1"),"max_parent_disagreement_deg":float(disagreement.max()),
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C035_deg"], score["C036_deg"], flush=True)
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












def score_penn(protocol,fixed,cfg,routers,cap):
    ds=Penn()
    require(not ds.formal and all(ds.roles[name]=="external_training" for name in PENN),
            "Only the explicitly registered Penn A training records are numerically accessible")
    controls={r["record"]:r for r in read(ROOT/"C035/penn-training-source-scores.json")["records"]}
    rows,masks=[],{}
    for name in PENN:
        payload,quality=ds.native(name)
        raw,hz,ref,mask=payload
        observed=np.isfinite(raw).all(axis=1)
        q0,q,at,release,hist,_,acc,_=c17.front_end(raw,observed,1./hz,cfg)
        dirs,probs,_,active,_,disagreement,binding=held(q,at,release,hist,acc,raw,observed,1./hz,protocol["C022_k"],fixed,routers,cap)
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
                     "full_record_tilt_RMSE_deg":results,"cap_affected_release_count":int(binding.sum()),
                     "cap_affected_scored_frames":int(np.r_[False,binding][np.searchsorted(release,ix,side="right")].sum()),
                     "max_parent_disagreement_deg":float(disagreement.max()),
                     "active_release_probabilities":{
                         n:dict(zip(WEIGHTS,p[active].mean(axis=0).tolist(),strict=True)) for n,p in probs.items()
                     }})
    np.savez_compressed(HERE/"penn-A-native-masks.npz",**masks)
    write(HERE/"penn-training-source-scores.json",{"pid":os.getpid(),"scope":"AF/AS native full-track training-side score",
          "coefficient_lock_sha256":sha(HERE/"coefficient-lock.json"),"native_mask_file_sha256":sha(HERE/"penn-A-native-masks.npz"),
          "records":rows,"record_equal_mean":{m:fmean(row["full_record_tilt_RMSE_deg"][m] for row in rows)
                                              for m in rows[0]["full_record_tilt_RMSE_deg"]}})





def replay_locked(protocol,launch,cap):
    path=HERE/"coefficient-lock.json"
    lock=read(path)
    lock_hash=sha(path)
    require(lock["pid"]==os.getpid() and lock["cap_deg"]==cap and lock["solve_count"]==1 and lock["neural_fits"]==0 and
            lock["protocol_sha256"]==sha(HERE/"protocol.json") and lock["solver_receipt_sha256"]==sha(HERE/"solver-receipt.json") and
            lock["evaluation_log_sha256"]==sha(HERE/"solver-evaluations.jsonl") and
            lock["training_io_sha256"]==sha(HERE/"training-io.npz"),"C036 coefficient lock or solver evidence changed")
    models,states={},{}
    for name in ROUTERS:
        state=c33.frozen_state(name)
        states[name]=state
        models[name]=c31.frozen_linear(state)
    anchor=states["C023"]
    for key in c33.STATE_KEYS:
        torch.testing.assert_close(anchor[key],states["C035"][key],rtol=0.,atol=0.)
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    routers=models,anchor["feature_mean"].numpy(),anchor["feature_std"].numpy(),anchor["log_p0"]
    status(launch,"scoring_Penn_A_training_source",coefficient_lock_sha256=lock_hash)
    score_penn(protocol,fixed,cfg,routers,cap)
    prior=read(ROOT/"C035/results.json")
    independent=read(ROOT/"C034/results.json")
    require([[r["dataset"],r["record"]] for r in prior["sequences"]]==protocol["development_records"] and
            prior["scoring_masks_sha256"]==independent["scoring_masks_sha256"]==sha(MASKS),"Reference panel masks or order changed")
    for a,b in zip(prior["sequences"],independent["sequences"],strict=True):
        require(all(a[k]==b[k] for k in ("dataset","record","scored_frames","scoring_indices_sha256")),
                "Physical-only archive score mask differs")
        a["physical_only_deg"]=b["physical_only_deg"]
    for a,b in zip(prior["array_sensor_poses"],independent["array_sensor_poses"],strict=True):
        require(all(a[k]==b[k] for k in ("record","pose","scored_frames","scoring_indices_sha256")),
                "Physical-only archive pose mask differs")
        a["physical_only_deg"]=b["physical_only_deg"]
    status(launch,"development_replay",coefficient_lock_sha256=lock_hash,expected_records=213)
    print("C036 locked; replaying common development panel",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,routers,cap)
    require(sha(path)==lock_hash and sha(HERE/"solver-receipt.json")==lock["solver_receipt_sha256"] and
            all(sha(ROOT/name)==h for name,h in protocol["inputs_sha256"].items()),"Frozen evidence changed during replay")
    result.update(pid=os.getpid(),cap_deg=cap,coefficient_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),penn_training_source_scores_sha256=sha(HERE/"penn-training-source-scores.json"),
                  solve_count=1,neural_fits=0,solver_receipt_sha256=lock["solver_receipt_sha256"],
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",scored=scored,coefficient_lock_sha256=lock_hash,results_sha256=sha(out))
    require(complete,"C036 replay incomplete; fitted coefficient retained")
    print("C036 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C036/run_physical_cap.py",
            "scope":"C036 single bounded cap solve, zero neural fits, common development replay and Penn A training audit",
            "protocol_sha256":sha(HERE/"protocol.json")}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()) and
                protocol["xatol"]==XTOL and protocol["solve_count"]==1 and protocol["neural_fits"]==0,
                "C036 registered artifact or solver specification changed")
        torch.set_num_threads(2)
        status(launch,"verifying_C023_and_physical_source_bank")
        bank=FitBank(protocol)
        status(launch,"single_bounded_scalar_solve")
        cap=solve(protocol,bank)
        replay_locked(protocol,launch,cap)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

