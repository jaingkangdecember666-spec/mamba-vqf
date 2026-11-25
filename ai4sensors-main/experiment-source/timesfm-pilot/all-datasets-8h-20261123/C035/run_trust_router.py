"""C035: one C023-centered neural-readout trust-region minimax fit."""
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
sys.path[:0]=[str(ROOT/"C033"),str(ROOT/"review")]

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize
import torch
from torch import nn

import run_penn_router as c33
from check_history_router import prediction
from penn_handheld import BASE,Penn
from source_router_trust import TrustRisk

c31,c23,c22,c18,c17,c3,c2,c14=c33.c31,c33.c23,c33.c22,c33.c18,c33.c17,c33.c3,c33.c2,c33.c14
base,intrinsic=c33.base,c33.intrinsic
read,write,sha,stamp,array_sha=c33.read,c33.write,c33.sha,c33.stamp,c33.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c33.UP,c33.DATA,c33.TRANSFER,c33.MASKS,c33.GATES
require,archive,check_probabilities,add_baselines=c33.require,c33.archive,c33.check_probabilities,c33.add_baselines
ROUTERS=("C023","C033","C035")
METHODS=(*c23.METHODS[:4],"C033_deg","C035_deg",*c23.FIXED)
WEIGHTS=c23.WEIGHTS
PENN=("af","as")
RADIUS=.4693045062626217
OPTIONS={"ftol":1e-11,"maxiter":200,"disp":False}
ATOL=1e-9


def local_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def status(launch,phase,**fields):
    write(HERE/"process.json",{"pid":os.getpid(),"started_at":launch["started_at"],"status":phase,"updated_at":local_now(),
                              "protocol_sha256":sha(HERE/"protocol.json"),**fields})


def flat_state(state):
    p=state["state_dict"]
    require(p["weight"].shape==(2,24) and p["bias"].shape==(2,) and p["weight"].dtype==torch.float64,
            "C023-family router requires fifty float64 parameters")
    return torch.cat((p["weight"].flatten(),p["bias"])).numpy()


def verify_slots(protocol,anchor,other,risk):
    ledger=archive(ROOT/"C033/training-inputs.npz")
    original=archive(ROOT/"C022/training-slots.npz")
    p=archive(ROOT/"C033/penn-training-slots.npz")
    review=read(ROOT/"review/held_router_trust_preflight.json")
    require(ledger["z"].shape==(160,98,24) and len(risk.ids)==15680 and len(risk.n)==141 and
            float(risk.radius)==RADIUS and review["C023_to_its_own_order_replicate_distance"]==RADIUS and
            review["recorded_source_panel_sha256"]==protocol["inputs_sha256"]["C033/training-inputs.npz"],
            "C035 radius or original C033 source panel changed")
    for key in ("center","fast","slow","truth","enabled","rest","source_record_ordinal","source_record_name","native_release",
                "native_input_last","record_equal_weights","domain"):
        np.testing.assert_array_equal(ledger[key][:,:96].reshape(original[key].shape),original[key])
        np.testing.assert_array_equal(ledger[key][:,96:],p[key])
    np.testing.assert_array_equal(ledger["z"][:,96:],p["z"])
    archive23=archive(ROOT/"C023/training-io.npz")
    np.testing.assert_array_equal(ledger["z"][:,:96].reshape(-1,24),archive23["normalized_input"])
    np.testing.assert_array_equal(risk.z,ledger["z"].reshape(-1,24))
    np.testing.assert_array_equal(risk.ids,ledger["source_record_ordinal"].ravel())
    np.testing.assert_array_equal(risk.start,flat_state(anchor))
    np.testing.assert_array_equal(risk.prior,anchor["p0"].numpy())
    np.testing.assert_allclose(risk.ref,review["C023_macro_deg"],rtol=0.,atol=2e-13)
    value,_=risk.relative(flat_state(other))
    np.testing.assert_allclose(value,review["C033_over_C023_ratios"],rtol=0.,atol=2e-13)
    for j,name in enumerate(PENN):
        receipt=read(ROOT/"C033/sampling-lock.json")["records"][j]
        require(receipt["record"]==name,"C033 Penn source order changed")
        for slot_key in ("native_release","native_input_last"):
            np.testing.assert_array_equal(ledger[slot_key][:,96+j],[r[slot_key] for r in receipt["selected"]])
        with np.load(ROOT/"C033"/f"{name}-native-mask-evidence.npz",allow_pickle=False) as evidence:
            idx=evidence["selected"]
            np.testing.assert_array_equal(ledger["native_release"][:,96+j],evidence["release"][idx])
            np.testing.assert_array_equal(ledger["native_input_last"][:,96+j],evidence["native_input_last"][idx])
            require(evidence["eligible"].size>=160 and np.all(evidence["complete_history"][idx]) and
                    np.all(evidence["enabled"][idx]) and np.unique(idx).size==160,"Penn source selection changed")
    audit={"pid":os.getpid(),"original_train_slots":15360,"Penn_AF_AS_slots":320,"train_slots":15680,"train_records":141,
           "domain_records":dict(zip(risk.domains,[len(m) for m in risk.members],strict=True)),
           "C023_source_macro_deg":risk.ref.tolist(),"C033_to_C023_ratio":value.tolist(),"radius_verified_from_C027":risk.radius,
           "C033_slot_bank_sha256":protocol["inputs_sha256"]["C033/training-inputs.npz"],
           "source_record_ordinal_sha256":array_sha(ledger["source_record_ordinal"],"<i8"),
           "native_release_sha256":array_sha(ledger["native_release"],"<i8"),
           "native_input_last_sha256":array_sha(ledger["native_input_last"],"<i8"),"old_C022_slot_and_Penn_slot_exact":True}
    write(HERE/"training-identity.json",audit)
    return ledger


def check_differences(risk,param,directions):
    ratio,jac=risk.relative(param)
    samples={}
    max_error=0.
    for name,vector in directions:
        h=1e-5
        observed=(risk.relative(param+h*vector)[0]-risk.relative(param-h*vector)[0])/(2*h)
        wanted=jac@vector
        error=float(np.max(np.abs(observed-wanted)))
        max_error=max(max_error,error)
        samples[name]={"direction":vector.tolist(),"analytic":wanted.tolist(),"numeric":observed.tolist(),"max_abs_error":error}
    require(max_error<=1e-7,"TrustRisk analytic derivative differs from finite difference")
    return {"ratios":ratio.tolist(),"analytic_jacobian":jac.tolist(),"checks":samples,"max_abs_error":max_error,
            "central_difference_step":1e-5}


def neural_fit(risk,protocol,anchor,previous):
    start=np.r_[risk.start,1.]
    dimension=50
    fixed_directions=[("unit_C023_to_C033",(flat_state(previous)-risk.start)/np.linalg.norm(flat_state(previous)-risk.start)),
                      *[(f"coordinate_{i}",np.eye(dimension)[i]) for i in (0,23,24,47,48,49)]]
    before=check_differences(risk,risk.start,fixed_directions)
    np.testing.assert_array_equal(before["ratios"],np.ones(len(risk.domains)))
    write(HERE/"zero-step-audit.json",{"pid":os.getpid(),"x0":start.tolist(),"domain_order":risk.domains,
          "C023_raw_baseline_rmse_deg":risk.ref.tolist(),"radius":RADIUS,"finite_difference":before})
    callbacks,evaluations,updates=0,0,0
    cached_p,cached_r,cached_j=None,None,None
    prior_param=risk.start.copy()
    unique=set()
    with ((HERE/"solver-evaluations.jsonl").open("x",encoding="utf-8") as pointlog,
          (HERE/"solver-iterations.jsonl").open("x",encoding="utf-8") as iterlog,
          (HERE/"neural-steps.jsonl").open("x",encoding="utf-8") as steplog):
        def values(x):
            nonlocal evaluations,cached_p,cached_r,cached_j
            w=x[:dimension]
            if cached_p is None or not np.array_equal(w,cached_p):
                cached_r,cached_j=risk.relative(w)
                cached_p=w.copy()
            d=w-risk.start
            ball=RADIUS**2-float(d@d)
            c=np.r_[x[-1]-cached_r,ball]
            jac=np.vstack((np.column_stack((-cached_j,np.ones(len(risk.domains)))),np.r_[-2.*d,0.]))
            key=tuple(float(v) for v in x)
            if key not in unique:
                unique.add(key)
                evaluations+=1
                print(json.dumps({"evaluation":evaluations,"pid":os.getpid(),"x":list(key),"objective":x[-1],
                                  "source_ratios":cached_r.tolist(),"relative_risk_jacobian":cached_j.tolist(),
                                  "constraint_slacks":c.tolist(),"constraint_jacobian":jac.tolist(),
                                  "trust_distance":float(np.linalg.norm(d))},allow_nan=False),file=pointlog,flush=True)
            return cached_r,cached_j,c,jac

        def objective(x):
            values(x)
            return float(x[-1])

        def log_iter(x,stage,index):
            nonlocal updates,prior_param
            r,jac,slack,_=values(x)
            includes={"kind":stage,"iteration":index,"pid":os.getpid(),"t":float(x[-1]),"source_ratios":r.tolist(),
                      "source_ratio_jacobian":jac.tolist(),"constraint_slacks":slack.tolist(),
                      "constraint_max_violation":float(max(0.,-slack.min())),
                      "trust_distance":float(np.linalg.norm(x[:dimension]-risk.start))}
            print(json.dumps({**includes,"params":x[:dimension].tolist()},allow_nan=False),file=iterlog,flush=True)
            if stage!="initial" and not np.array_equal(x[:dimension],prior_param):
                updates+=1
                diff=x[:dimension]-prior_param
                row={**includes,"step":updates,"params_before":prior_param.tolist(),"params_after":x[:dimension].tolist(),
                     "actual_update":diff.tolist(),"update_l2":float(np.linalg.norm(diff)),
                     "learned_weight_sha256":array_sha(x[:dimension],"<f8")}
                print(json.dumps(row,allow_nan=False),file=steplog,flush=True)
                prior_param=x[:dimension].copy()

        def callback(x):
            nonlocal callbacks
            callbacks+=1
            log_iter(x,"SLSQP_major_iterate",callbacks)
            if callbacks in (1,10,25,50,100,150,200):
                print("C035 iteration",callbacks,"t",float(x[-1]),flush=True)

        log_iter(start,"initial",0)
        clock=perf_counter()
        solve_started_at=local_now()
        found=minimize(objective,start,jac=lambda _:np.r_[np.zeros(dimension),1.],method="SLSQP",
                       constraints=({"type":"ineq","fun":lambda x:values(x)[2],"jac":lambda x:values(x)[3]},),
                       callback=callback,options=OPTIONS)
        fit_seconds=perf_counter()-clock
        solve_completed_at=local_now()
        terminal=np.asarray(found.x,dtype=np.float64)
        require(terminal.shape==(51,) and np.isfinite(terminal).all(),"Nonfinite terminal SLSQP point")
        log_iter(terminal,"terminal",callbacks+1)
    coefficient=terminal[:dimension]
    after=check_differences(risk,coefficient,fixed_directions)
    end,jac=risk.relative(coefficient)
    delta=coefficient-risk.start
    distance=float(np.linalg.norm(delta))
    ball=RADIUS**2-float(delta@delta)
    slack=terminal[-1]-end
    maximum_violation=float(max(0.,-ball,-slack.min()))
    feasible=maximum_violation<=ATOL
    write(HERE/"finite-difference.json",{"pid":os.getpid(),"domain_order":risk.domains,"initial":before,"terminal":after})
    result={"pid":os.getpid(),"solver_kind":"neural_readout","neural_fits":1,"solve_count":1,"method":"SLSQP",
            "scipy_version":scipy.__version__,"options":OPTIONS,"constraints":["five source epigraph ratios","squared Euclidean trust region"],
            "starts_at_C023":True,"initial_x":start.tolist(),"terminal_x":terminal.tolist(),"radius":RADIUS,
            "result_success":bool(found.success),"result_status":int(found.status),"result_message":str(found.message),
            "result_objective":float(found.fun),"nit":int(found.nit),"nfev":int(found.nfev),"njev":int(found.njev),
            "callback_count":callbacks,"distinct_solver_point_count":evaluations,
            "distinct_a_evaluation_count":len({point[:-1] for point in unique}),
            "real_parameter_updates":updates,"fit_seconds":fit_seconds,"solve_started_at":solve_started_at,
            "solve_completed_at":solve_completed_at,"reported_multipliers":np.asarray(getattr(found,"multipliers",[])).tolist(),
            "terminal_feasible":bool(feasible),"feasibility_atol":ATOL,"terminal_max_constraint_violation":maximum_violation,
            "terminal_epigraph_max_group_ratio":float(end.max()),"terminal_trust_distance":distance,"terminal_trust_squared_slack":ball,
            "terminal_group_slack":dict(zip(risk.domains,slack.tolist(),strict=True)),
            "terminal_source_ratio_jacobian":jac.tolist(),"group_order":risk.domains,
            "parent_C023_checkpoint_sha256":protocol["inputs_sha256"]["C023/router.pt"],
            "C033_training_input_sha256":protocol["inputs_sha256"]["C033/training-inputs.npz"],
            "checkpoint_choice":"Single SLSQP terminal point if feasible; retain actual receipt on non-convergence"}
    write(HERE/"solver-receipt.json",result)
    require(feasible,"Infeasible C035 solver endpoint; receipt retained")
    state={**{k:anchor[k] for k in (*c33.STATE_KEYS,"dtype","C022_coefficient_lock_sha256")},
           "state_dict":{"weight":torch.from_numpy(coefficient[:48].reshape(2,24)),"bias":torch.from_numpy(coefficient[48:])},
           "optimizer_steps":updates,"solver_kind":"neural_readout","neural_fits":1,
           "protocol_sha256":sha(HERE/"protocol.json"),"solver_receipt_sha256":sha(HERE/"solver-receipt.json")}
    torch.save(state,HERE/"router.pt")
    print("C035 SLSQP finished",found.message,"nit",found.nit,"real_updates",updates,"ratio",float(end.max()),flush=True)
    return state,result


def training_scores(risk,ledger,states,receipt):
    ordinals=ledger["source_record_ordinal"].ravel()
    names=ledger["source_record_name"].ravel()
    domain=ledger["domain"].ravel()
    active=ledger["enabled"].ravel()
    outcomes={}
    for name,state in states.items():
        direction,_,_,p=prediction(risk.z,risk.prior,flat_state(state),risk.center,risk.f,risk.s,risk.enabled)
        check_probabilities(p)
        angle=np.rad2deg(np.arctan2(np.linalg.norm(np.cross(direction,risk.truth),axis=1),
                                  np.einsum("ij,ij->i",direction,risk.truth)))
        outcomes[name]=(angle,p,direction)
    np.testing.assert_array_equal(outcomes["C023"][2][~active],outcomes["C035"][2][~active])
    groups={}
    for d in risk.domains:
        use=domain==d
        ids=np.unique(ordinals[use])
        rows=[use&(ordinals==i) for i in ids]
        groups[d]={"records":len(rows),"slots":int(use.sum()),"active_slots":int((use&active).sum()),
                   "source_record_macro_RMSE_deg":{m:fmean(float(np.sqrt(np.mean(angle[k]**2))) for k in rows)
                                                   for m,(angle,_,_) in outcomes.items()}}
    ratios=[groups[d]["source_record_macro_RMSE_deg"]["C035"]/groups[d]["source_record_macro_RMSE_deg"]["C023"] for d in risk.domains]
    np.testing.assert_allclose(ratios,np.asarray(receipt["terminal_x"][-1])-
                               np.array([receipt["terminal_group_slack"][d] for d in risk.domains]),rtol=0.,atol=2e-13)
    path=HERE/"training-io.npz"
    np.savez_compressed(path,**{m+"_"+key:value for m,vals in outcomes.items() for key,value in zip(("angle_deg","prob","direction"),vals,strict=True)},
                        source_record_ordinal=ordinals,source_record_name=names,domain=domain,enabled=active,
                        native_release=ledger["native_release"].ravel(),native_input_last=ledger["native_input_last"].ravel())
    write(HERE/"training.json",{"pid":os.getpid(),"steps":receipt["real_parameter_updates"],"fit_seconds":receipt["fit_seconds"],
          "solver_kind":"neural_readout","solver_receipt_sha256":sha(HERE/"solver-receipt.json"),"neural_fits":1,"solve_count":1,
          "training_slots":15680,"original_training_records":141,"parameters":50,
          "group_order":risk.domains,"C035_to_C023_ratios":ratios,"domains":groups,"training_io_sha256":sha(path)})



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
            "C035_delta_deg": {m: macro["C035_deg"] - macro[m] for m in METHODS if m != "C035_deg"} if keep else {},
            "C035_wins": {m: sum(r["C035_deg"] < r[m] for r in keep) for m in METHODS if m != "C035_deg"}}


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
                        array_sha(probabilities["C033"][active], "<f8") == old["C033_active_routing_sha256"] and
                        array_sha(deltas["C023"], "<f8") == old["C023_delta_sha256"] and
                        array_sha(deltas["C033"], "<f8") == old["C033_delta_sha256"],
                        "Frozen C023/C033 release count, probabilities, or deltas changed")
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C033_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C033 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C033_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C035_deg"][~active_scored], errs["C020_deg"][~active_scored])
                active_probs = {m:dict(zip(WEIGHTS, p[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                                for m,p in probabilities.items()}
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C033_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, parent_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs["C035"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C033_deg"], score["C035_deg"], flush=True)
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
    controls={r["record"]:r for r in read(ROOT/"C033/penn-training-source-scores.json")["records"]}
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
        for now,prior in (("C023_deg","C023"),("C033_deg","C033"),("full_vqf_deg","full_vqf_deg")):
            require(abs(results[now]-old_scores[prior])<=1e-6,"Archived Penn A training control changes")
        gyro=np.ascontiguousarray(raw[:,3:])
        accel=np.ascontiguousarray(raw[:,:3])
        for tau,key,old_key in ((1.5,"fixed_tau1p5_deg","archived_fixed_fast"),
                                (3.,"fixed_tau3_deg","full_vqf_deg"),
                                (6.,"fixed_tau6_deg","archived_fixed_slow")):
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


def replay_locked(protocol,launch,fixed,campaign,cfg,states):
    lock=read(HERE/"router-lock.json")
    lock_hash=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["neural_fits"]==1 and lock["solve_count"]==1 and
            lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/name)==digest for name,digest in lock["artifact_sha256"].items()),"C035 locked checkpoint changed")
    anchor,own=(states[name] for name in ("C023","C035"))
    for name in ("C033","C035"):
        for key in c33.STATE_KEYS:
            torch.testing.assert_close(states[name][key],anchor[key],rtol=0.,atol=0.)
    require(own["protocol_sha256"]==sha(HERE/"protocol.json") and
            own["solver_receipt_sha256"]==sha(HERE/"solver-receipt.json"),"Checkpoint belongs to a different solver run")
    models={name:c31.frozen_linear(state) for name,state in states.items()}
    routers=models,anchor["feature_mean"].numpy(),anchor["feature_std"].numpy(),anchor["log_p0"]
    status(launch,"scoring_Penn_A_training_source",router_lock_sha256=lock_hash)
    score_penn(protocol,fixed,cfg,routers)
    prior=read(ROOT/"C033/results.json")
    require([[r["dataset"],r["record"]] for r in prior["sequences"]]==protocol["development_records"],
            "C033 development records differ from the registered panel")
    status(launch,"development_replay",router_lock_sha256=lock_hash,expected_records=213)
    print("C035 locked; running common native development replay",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,routers)
    require(sha(HERE/"router-lock.json")==lock_hash and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/name)==h for name,h in protocol["inputs_sha256"].items()),"Frozen evidence changed during replay")
    result.update(pid=os.getpid(),router_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),penn_training_source_scores_sha256=sha(HERE/"penn-training-source-scores.json"),
                  fit_count=1,neural_fits=1,solve_count=1,solver_receipt_sha256=own["solver_receipt_sha256"],
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False)
    dest=HERE/"results.json"
    write(dest,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",scored=scored,router_lock_sha256=lock_hash,
           results_sha256=sha(dest))
    require(complete,"C035 replay incomplete; frozen solver checkpoint retained")
    print("C035 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def run(protocol,launch):
    require(protocol["options"]==OPTIONS and protocol["radius"]==RADIUS and protocol["solve_count"]==1 and
            protocol["neural_fits"]==1 and protocol["trainable_readout_parameters"]==50,"C035 registered constraints changed")
    torch.set_num_threads(2)
    status(launch,"verifying_C033_slots_and_C023_origin")
    states={n:c33.frozen_state(n) for n in ("C023","C033")}
    anchor,prior=(states[n] for n in ("C023","C033"))
    for key in c33.STATE_KEYS:
        torch.testing.assert_close(anchor[key],prior[key],rtol=0.,atol=0.)
    k=read(ROOT/"C022/coefficient-lock.json")["k"]
    require(k==protocol["C022_k"],"C022 prior coefficient differs")
    torch.testing.assert_close(anchor["p0"],c23.prior_probs(k),rtol=0.,atol=0.)
    risk=TrustRisk()
    ledger=verify_slots(protocol,anchor,prior,risk)
    status(launch,"single_neural_readout_SLSQP")
    state,receipt=neural_fit(risk,protocol,anchor,prior)
    states["C035"]=state
    training_scores(risk,ledger,states,receipt)
    artifacts=("protocol.json","training-identity.json","zero-step-audit.json","solver-iterations.jsonl","solver-evaluations.jsonl",
               "solver-receipt.json","finite-difference.json","neural-steps.jsonl","training.json","training-io.npz")
    with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
        json.dump({"pid":os.getpid(),"locked_at":local_now(),"solve_count":1,"neural_fits":1,
                   "neural_steps":receipt["real_parameter_updates"],"checkpoint_sha256":sha(HERE/"router.pt"),
                   "artifact_sha256":{n:sha(HERE/n) for n in artifacts}},handle,indent=2)
        handle.write("\n")
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    replay_locked(protocol,launch,fixed,campaign,cfg,states)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C035/run_trust_router.py",
            "scope":"C035 only; one C023-centered fifty-parameter trust-region SLSQP fit and native development replay",
            "protocol_sha256":sha(HERE/"protocol.json")}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"C035 registered artifact changed")
        run(protocol,launch)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

