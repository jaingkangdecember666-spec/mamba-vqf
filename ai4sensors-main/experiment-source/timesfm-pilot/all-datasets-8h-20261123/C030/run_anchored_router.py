"""C030: exact group-angular-RMSE minimax fit of four antisymmetric C023 corrections."""
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C029"), str(ROOT / "review")]

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F

import run_geometry_router as c29
from causal_response_geometry import measures

c26, c25, c23, c22, c18, c17 = c29.c26, c29.c25, c29.c23, c29.c22, c29.c18, c29.c17
c3, c2, base, intrinsic = c29.c3, c29.c2, c29.base, c29.intrinsic
read, write, sha, stamp, array_sha = c29.read, c29.write, c29.sha, c29.stamp, c29.array_sha
UP, DATA, TRANSFER, MASKS, GATES = c29.UP, c29.DATA, c29.TRANSFER, c29.MASKS, c29.GATES
WEIGHTS, PROB_TOL, NORM_KEYS = c29.WEIGHTS, c29.PROB_TOL, c29.NORM_KEYS
archive, require, weights_state = c29.archive, c29.require, c29.weights_state
ROUTERS = ("C023", "C029", "C030")
FROZEN_SCORES = ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C029_deg")
METHODS = (*FROZEN_SCORES, "C030_deg", *c23.FIXED)
GROUPS = ("BROAD", "Sassari", "YT_Server", "RIDI", "DAVIS", "raw", "fixed_prefix_bias_removed")
GEOMETRY_NAMES = ("D_over_6", "D_over_6_times_bias_z", "D_over_6_times_force_norm_z", "D_over_6_times_force_spread_z")
OPTIONS = {"ftol": 1e-12, "maxiter": 200, "disp": False}
FEASIBILITY_ATOL = 1e-10


def status(protocol, phase, **details):
    write(HERE/"process.json", {"pid":os.getpid(), "started_at":protocol["started_at"], "updated_at":stamp(),
                                "status":phase, "protocol_sha256":sha(HERE/"protocol.json"), **details})


def verify_inputs():
    prior, state, lock = [read(ROOT/"C029"/n) for n in ("protocol.json","process.json","router-lock.json")]
    incumbent, c023_lock = [read(ROOT/"C023"/n) for n in ("process.json","router-lock.json")]
    require(state["status"] == incumbent["status"] == "complete" and state["scored"] == incumbent["scored"] == 213
            and prior["slots"] == 17600 and prior["recordings"] == 148 and prior["scoring_masks_sha256"] == sha(MASKS),
            "C029 and C023 must have completed the same development panel")
    files = {**prior["inputs_sha256"], **{f"C029/{n}": h for n,h in lock["artifact_sha256"].items()},
             "C029/router-lock.json":state["router_lock_sha256"], "C029/results.json":state["results_sha256"],
             "C029/router.pt":lock["checkpoint_sha256"], "C029/run_geometry_router.py":prior["worker_sha256"],
             "C023/router-lock.json":incumbent["router_lock_sha256"], "C023/results.json":incumbent["results_sha256"]}
    require(files["C023/router.pt"] == c023_lock["checkpoint_sha256"], "C023 anchor hash differs")
    for n in ("C029/process.json","C023/process.json","review/anchored_response_gradient.py","review/anchored_response_gradient.json"):
        files[n] = sha(ROOT/n)
    require(all(sha(ROOT/n) == h for n,h in files.items()), "A frozen input changed")
    ref = read(ROOT/"review/anchored_response_gradient.json")
    require(ref["source_sha256"] == files["C026/training-inputs.npz"] and ref["base_router_sha256"] == files["C023/router.pt"]
            and set(ref["groups"]) == set(GROUPS), "Anchor derivative review is from a different archive")
    return prior, files, ref


def geometry_basis(fast, slow, normalized28):
    """Use the same causal capped filter responses and motion normalizers as C029."""
    z = np.asarray(normalized28)
    require(z.ndim == 2 and z.shape[1] == 28, "Expected the original 28 causal inputs")
    angle = measures(fast, slow, z[:, 27])[:, 0]
    phi = (angle[:, None] / 6.) * np.column_stack((np.ones(len(z)), z[:,27], z[:,25], z[:,26]))
    require(np.isfinite(phi).all() and np.all(angle >= 0.), "Nonfinite response geometry")
    return angle, phi


def design_from_training(data):
    z = data["features"].reshape(-1, 28)
    f, s = (data[k].reshape(-1, 3) for k in ("fast","slow"))
    require(z.shape == (17600,28) and np.isfinite(z).all() and np.all(f[:,2] == 0.) and np.all(s[:,2] == 0.) and
            max(np.linalg.norm(f,axis=1).max(),np.linalg.norm(s,axis=1).max()) <= c22.CAP+2e-15,
            "Input is not the C026 horizontal capped response archive")
    angle, phi = geometry_basis(f,s,z)
    with np.load(ROOT/"C029/geometry-training.npz", allow_pickle=False) as prior:
        np.testing.assert_array_equal(prior["features"][:,:,:28],data["features"])
        np.testing.assert_array_equal(prior["geometry_raw"][:,:,0].ravel(),angle)
    return z, f, s, angle, phi


class AngularGroups:
    """C023 probabilities and analytic derivatives of the true per-original angle RMSE."""

    def __init__(self, data, design, anchor, records):
        z, self.fast, self.slow, self.angle, self.phi = design
        self.center, self.truth = (data[k].reshape(-1, 3) for k in ("center","truth"))
        self.active = data["enabled"].ravel()
        self.group = np.where(data["domain"] == "EuRoC", data["condition"], data["domain"]).ravel()
        self.ordinal = data["source_record_ordinal"].ravel()
        require(np.unique(self.ordinal).size == 148 and set(self.group) == set(GROUPS) and
                np.isfinite(self.truth).all() and np.allclose(np.linalg.norm(self.truth,axis=1),1.,rtol=0.,atol=2e-12),
                "C026 record identities or body-world truth changed")
        self.rows, self.record_ids = [], np.full(len(z), -1, dtype=np.int64)
        self.record_groups = []
        for j, name in enumerate(GROUPS):
            mask = self.group == name
            for ordinal in np.unique(self.ordinal[mask]):
                ids = mask & (self.ordinal == ordinal)
                mapping = records[name, int(ordinal)]
                require(int(ids.sum()) == mapping["slots"], "Original acquisition slot population changed")
                self.record_groups.append(j)
                self.record_ids[ids] = len(self.rows)
                self.rows.append({"group":name,"source":mapping["source"],"ordinal":int(ordinal),
                                  "record":mapping["record"],"condition":mapping["condition"],
                                  "slots":int(ids.sum()),"active_slots":int((ids & self.active).sum())})
        require(len(self.rows) == 151 and np.all(self.record_ids >= 0), "C030 requires every training slot including inactive slots")
        self.record_groups = np.asarray(self.record_groups,dtype=np.int64)
        self.slot_counts = np.bincount(self.record_ids)
        self.group_counts = np.bincount(self.record_groups)
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(z[:,:24]))
            original = anchor["state_dict"]
            l = F.linear(x, original["weight"], original["bias"]) + anchor["log_p0"][:2]
            self.logits = torch.cat((l,anchor["log_p0"][2].expand(len(z),1)),dim=1).numpy()

    def angular(self, a):
        # The common vector norm cancels inside atan2; the direction is still normalized for replay.
        h = self.phi @ np.asarray(a)
        logits = self.logits.copy()
        logits[:,0] += h
        logits[:,1] -= h
        probability = softmax(logits,axis=1)
        mixture = (probability[:,:1]*self.fast + probability[:,1:2]*self.slow)*self.active[:,None]
        u = self.center + mixture
        cross = np.cross(u,self.truth)
        sn = np.linalg.norm(cross,axis=1)
        cs = np.einsum("ij,ij->i",u,self.truth)
        theta = np.arctan2(sn,cs)
        require(not np.any((sn == 0.) & (cs < 0.)), "An exactly antipodal sample has an undefined angular gradient")
        du = (probability[:,:1]*(self.fast-mixture)-probability[:,1:2]*(self.slow-mixture))*self.active[:,None]
        cross_derivative = np.cross(du,self.truth)
        sn_derivative = np.divide(np.einsum("ij,ij->i",cross,cross_derivative),sn,out=np.zeros_like(sn),where=sn>0.)
        cs_derivative = np.einsum("ij,ij->i",du,self.truth)
        theta_derivative = (cs*sn_derivative - sn*cs_derivative)/(sn**2+cs**2)
        require(np.isfinite(theta).all() and np.isfinite(theta_derivative).all(), "Nonfinite angular loss or derivative")
        return theta, theta_derivative, probability, u

    def evaluate(self, a):
        theta, dtheta, _, _ = self.angular(a)
        count = len(self.rows)
        rms = np.sqrt(np.bincount(self.record_ids,weights=theta**2,minlength=count)/self.slot_counts)
        require(np.all(rms > 0.), "An original recording has zero angular RMS and no unique derivative")
        derivative = (theta*dtheta)[:, None]*self.phi
        rms_gradient = np.column_stack([
            np.bincount(self.record_ids,weights=derivative[:,i],minlength=count) for i in range(4)
        ]) / (self.slot_counts*rms)[:,None]
        mean = np.bincount(self.record_groups,weights=rms,minlength=len(GROUPS))/self.group_counts
        jacobian = np.column_stack([
            np.bincount(self.record_groups,weights=rms_gradient[:,i],minlength=len(GROUPS)) for i in range(4)
        ]) / self.group_counts[:,None]
        return mean,jacobian,rms,rms_gradient


def finite_difference(bank, a, baseline):
    value, analytic, _, _ = bank.evaluate(a)
    audit = {"a": a.tolist(),"group_RMSE_deg":np.degrees(value).tolist(),"group_ratio_analytic_gradient":(analytic/baseline[:,None]).tolist(),
             "central_difference":{}}
    for step in (1e-5,1e-6):
        numerical=[]
        for i in range(4):
            delta=np.eye(4)[i]*step
            plus=bank.evaluate(a+delta)[0]
            minus=bank.evaluate(a-delta)[0]
            numerical.append((plus-minus)/(2*step*baseline))
        gradient=np.asarray(numerical).T
        gap=float(np.max(np.abs(gradient-analytic/baseline[:,None])))
        audit["central_difference"][str(step)]={"ratio_gradient":gradient.tolist(),"max_abs_error":gap}
        require(gap <= 2e-8, "True-angular objective gradient finite-difference check failed")
    return audit


def packed_state(anchor, prior, a):
    require(anchor["state_dict"]["weight"].shape == (2,24) and anchor["state_dict"]["bias"].shape == (2,),
            "C023 checkpoint must be the locked 24-dimensional anchor")
    old_weight = anchor["state_dict"]["weight"]
    extra = torch.from_numpy(np.vstack((a,-a)))
    weight = torch.cat((old_weight,torch.zeros((2,4),dtype=torch.float64),extra),dim=1)
    return {**{k:prior[k] for k in ("context_mean","context_std","feature_names")},
            "old_feature_mean":anchor["feature_mean"],"old_feature_std":anchor["feature_std"],
            "p0":anchor["p0"],"log_p0":anchor["log_p0"],
            "geometry_mean":torch.zeros(4,dtype=torch.float64),"geometry_std":torch.ones(4,dtype=torch.float64),
            "geometry_names":list(GEOMETRY_NAMES),"a":torch.from_numpy(a.copy()),
            "state_dict":{"weight":weight,"bias":anchor["state_dict"]["bias"]},"dtype":"float64",
            "solve_count":1,"protocol_sha256":sha(HERE/"protocol.json"),
            "anchor_checkpoint_sha256":sha(ROOT/"C023/router.pt")}


def solve(bank, z, anchor, prior, reference):
    started=perf_counter()
    a0=np.zeros(4)
    group0, grad0, records0, record_grad0 = bank.evaluate(a0)
    ref_error={}
    for j,name in enumerate(GROUPS):
        ref=reference["groups"][name]
        rmse_gap=abs(float(np.degrees(group0[j]))-ref["source_record_macro_C023_deg"])
        grad_gap=float(np.max(np.abs(grad0[j]/group0[j]-ref["fractional_RMSE_gradient"])))
        require(bank.group_counts[j] == ref["record_conditions"] and rmse_gap <= 2e-11 and grad_gap <= 2e-9,
                f"C023 external source gradient differs from review for {name}")
        ref_error[name]={"macro_deg_abs_gap":rmse_gap,"ratio_gradient_max_abs_gap":grad_gap}
    initial=finite_difference(bank,a0,group0)
    write(HERE/"zero-step-audit.json", {"pid":os.getpid(),"group_order":list(GROUPS),"x0":[0.,0.,0.,0.,1.],
          "zero_step_matches_C023_diagnostic":ref_error,"group_baseline_RMSE_deg":np.degrees(group0).tolist(),
          "analytic_and_numerical_ratios":initial})
    archive_path=HERE/"anchored-inputs.npz"
    np.savez_compressed(archive_path, physical_basis=bank.phi, fast_slow_angle_deg=bank.angle,
                        record_index=bank.record_ids, original_record_ordinal=bank.ordinal, group=bank.group,
                        group_index_by_record=bank.record_groups, active=bank.active, C023_group_RMSE=group0,
                        C023_logits=bank.logits, C023_squared_angular_RMSE_by_record=records0**2)
    cache_a, cache_r, cache_jac = None,None,None
    evaluations, callbacks = 0,0
    with ((HERE/"solver-evaluations.jsonl").open("x",encoding="utf-8") as eval_log,
          (HERE/"solver-iterations.jsonl").open("x",encoding="utf-8") as iter_log):
        def constraint_data(x):
            nonlocal cache_a,cache_r,cache_jac,evaluations
            a=x[:4]
            if cache_a is None or not np.array_equal(a,cache_a):
                value, jac,*_=bank.evaluate(a)
                cache_a,cache_r,cache_jac=a.copy(),value/group0,jac/group0[:,None]
                evaluations+=1
                print(json.dumps({"evaluation":evaluations,"pid":os.getpid(),"a":a.tolist(),
                                  "true_group_RMSE_deg":np.degrees(value).tolist(),"group_RMSE_ratio":cache_r.tolist(),
                                  "group_ratio_jacobian":cache_jac.tolist()},allow_nan=False),file=eval_log,flush=True)
            return cache_r,cache_jac

        def inequalities(x):
            return x[4]-constraint_data(x)[0]

        def inequality_jac(x):
            return np.column_stack((-constraint_data(x)[1],np.ones(len(GROUPS))))

        def log_iteration(x, label, index):
            ratios,jac=constraint_data(x)
            slack=x[4]-ratios
            print(json.dumps({"stage":label,"iteration":index,"pid":os.getpid(),"a":x[:4].tolist(),"t":float(x[4]),
                              "ratios":dict(zip(GROUPS,ratios.tolist(),strict=True)),
                              "gradient_by_group":dict(zip(GROUPS,jac.tolist(),strict=True)),
                              "constraint_slacks":dict(zip(GROUPS,slack.tolist(),strict=True)),
                              "max_constraint_violation":float(max(0.,-slack.min()))},allow_nan=False),file=iter_log,flush=True)

        def callback(x):
            nonlocal callbacks
            callbacks+=1
            log_iteration(x,"SLSQP_callback",callbacks)
            print("C030 SLSQP iteration",callbacks,float(x[4]),flush=True)

        x0=np.r_[a0,1.]
        log_iteration(x0,"start",0)
        result=minimize(lambda x: float(x[4]),x0,jac=lambda _:np.array([0.,0.,0.,0.,1.]),method="SLSQP",
                        bounds=[(-1.,1.)]*4+[(None,None)],
                        constraints=({"type":"ineq","fun":inequalities,"jac":inequality_jac},),
                        callback=callback,options=OPTIONS)
        terminal=np.asarray(result.x,dtype=np.float64)
        require(terminal.shape == (5,) and np.isfinite(terminal).all(), "SLSQP returned a nonfinite endpoint")
        log_iteration(terminal,"terminal",callbacks+1)
    end_group,end_grad,end_rec,end_rec_grad = bank.evaluate(terminal[:4])
    ratios=end_group/group0
    slacks=terminal[4]-ratios
    bound_violation=float(np.max(np.maximum(np.abs(terminal[:4])-1.,0.)))
    feasible=bound_violation<=FEASIBILITY_ATOL and float(slacks.min())>=-FEASIBILITY_ATOL
    improved=bool(feasible and np.max(ratios)<1.-FEASIBILITY_ATOL)
    a=terminal[:4].copy() if improved else a0.copy()
    choice="SLSQP_terminal" if improved else "C023_identity_due_to_no_feasible_strict_improvement"
    end_fd=finite_difference(bank,terminal[:4],group0)
    write(HERE/"finite-difference.json",{"pid":os.getpid(),"group_order":list(GROUPS),"zero_step":initial,
          "terminal":end_fd,"scope":"Central derivative checks only, including two-sided perturbation at a coefficient bound"})
    deployed_group,deployed_grad,deployed_record,deployed_record_grad=bank.evaluate(a)
    theta,_,prob,q=bank.angular(a)
    theta0,_,p0,q0=bank.angular(a0)
    c26.require_simplex(prob)
    np.testing.assert_array_equal(q[~bank.active],q0[~bank.active])
    checkpoint=packed_state(anchor,prior,a)
    with torch.inference_mode():
        observed=torch.from_numpy(np.column_stack((z,bank.phi)))
        model=nn.Linear(32,2,dtype=torch.float64)
        model.load_state_dict(checkpoint["state_dict"])
        packed=c23.weights(model,observed,checkpoint["log_p0"]).numpy()
    packed_gap=float(np.max(np.abs(prob-packed)))
    require(packed_gap <= 2e-13, "Packed checkpoint is not numerically equivalent to the solved logit correction")
    np.savez_compressed(HERE/"training-io.npz", C023_prob=p0,C030_prob=prob,archive_packed_prob=packed,
                        C023_tilt_deg=np.degrees(theta0),C030_tilt_deg=np.degrees(theta),
                        C023_direction=q0/np.linalg.norm(q0,axis=1,keepdims=True),
                        C030_direction=q/np.linalg.norm(q,axis=1,keepdims=True))
    torch.save(checkpoint,HERE/"router.pt")
    group_report={}
    for i,name in enumerate(GROUPS):
        group_report[name]={"records":int(bank.group_counts[i]),"slots":int((bank.group==name).sum()),
                            "C023_macro_RMSE_deg":float(np.degrees(group0[i])),
                            "SLSQP_terminal_macro_RMSE_deg":float(np.degrees(end_group[i])),
                            "SLSQP_terminal_ratio":float(ratios[i]),"SLSQP_terminal_constraint_slack":float(slacks[i]),
                            "C030_locked_macro_RMSE_deg":float(np.degrees(deployed_group[i])),
                            "C030_locked_ratio":float(deployed_group[i]/group0[i]),
                            "initial_ratio_gradient":(grad0[i]/group0[i]).tolist(),
                            "terminal_ratio_gradient":(end_grad[i]/group0[i]).tolist(),
                            "locked_ratio_gradient":(deployed_grad[i]/group0[i]).tolist()}
    row_report=[{**record,"C023_RMSE_deg":float(np.degrees(records0[i])),
                 "SLSQP_terminal_RMSE_deg":float(np.degrees(end_rec[i])),
                 "C030_locked_RMSE_deg":float(np.degrees(deployed_record[i])),
                 "initial_RMSE_gradient_deg":np.degrees(record_grad0[i]).tolist(),
                 "terminal_RMSE_gradient_deg":np.degrees(end_rec_grad[i]).tolist(),
                 "locked_RMSE_gradient_deg":np.degrees(deployed_record_grad[i]).tolist()}
                for i,record in enumerate(bank.rows)]
    return {"pid":os.getpid(),"solve_count":1,"scipy_version":scipy.__version__,"method":"SLSQP","options":OPTIONS,
            "initial_x":x0.tolist(),"coefficient_bounds":[-1.,1.],"group_order":list(GROUPS),
            "terminal_x":terminal.tolist(),"result_success":bool(result.success),"result_status":int(result.status),
            "result_message":str(result.message),"result_objective":float(result.fun),
            "nit":int(result.nit),"nfev":int(result.nfev),"njev":int(result.njev),
            "reported_multipliers":np.asarray(getattr(result,"multipliers",[])).tolist(),
            "callback_count":callbacks,"distinct_a_evaluation_count":evaluations,
            "terminal_feasible":bool(feasible),"terminal_strictly_improves_all_groups":improved,
            "feasibility_atol":FEASIBILITY_ATOL,"terminal_max_bound_violation":bound_violation,
            "terminal_max_constraint_violation":float(max(0.,-slacks.min())),
            "terminal_epigraph_max_group_ratio":float(ratios.max()),"locked_choice":choice,"locked_a":a.tolist(),
            "packed_checkpoint_max_prob_abs_difference":packed_gap,"groups":group_report,
            "original_record_conditions":row_report,"solve_and_archive_seconds":perf_counter()-started}




@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,models,feature_norm,ctx_norm,prior_geometry,a,log0):
    original,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    active=valid&~rest
    inherited,bmean,bstd,*_=fixed
    feat=c18.features(history,q[at],inherited[0],bmean,bstd)
    np.testing.assert_array_equal(feat.valid,valid)
    mean,std=feature_norm
    z0=(feat.pooled[:,1].astype(np.float64)-mean)/std
    context=c25.from_frontend(raw,observed,acc,dt)[at]
    z28=np.column_stack((z0,(context-ctx_norm[0])/ctx_norm[1]))
    _,prior_z=c29.geometric_features(fast,slow,z28[:,27],*prior_geometry)
    angle,phi=geometry_basis(fast,slow,z28)
    logits=torch.cat((models["C023"](torch.from_numpy(z0))+log0[:2],log0[2].expand(len(z0),1)),dim=1)
    displacement=torch.from_numpy(phi@a)
    anchored_logits=logits.clone()
    anchored_logits[:,0]+=displacement
    anchored_logits[:,1]-=displacement
    np.testing.assert_array_equal(anchored_logits[:,2].numpy(),logits[:,2].numpy())
    probabilities={"C023":torch.softmax(logits,dim=1).numpy(),
                   "C029":c23.weights(models["C029"],torch.from_numpy(np.column_stack((z28,prior_z))),log0).numpy(),
                   "C030":torch.softmax(anchored_logits,dim=1).numpy()}
    packed=c23.weights(models["C030"],torch.from_numpy(np.column_stack((z28,phi))),log0).numpy()
    packed_gap=float(np.max(np.abs(probabilities["C030"]-packed)))
    require(packed_gap<=2e-13,"Packed C030 changes inference beyond roundoff")
    directions={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(active[:,None],mix,0.))}
    for name,p in probabilities.items():
        c26.require_simplex(p)
        shift=np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        require(np.max(np.linalg.norm(shift,axis=1))<=c22.CAP+2e-15 and np.all(shift[:,2]==0.),
                "Anchored router correction escaped the capped horizontal responses")
        directions[name+"_deg"]=c23.augment(original,center,shift)
    return directions,probabilities,active,rest,context,angle,phi,displacement.numpy(),packed_gap



def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ROUTERS} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C030_delta_deg":{k:macro["C030_deg"]-macro[k] for k in METHODS if k!="C030_deg"} if good else {},
            "C030_wins":{k:sum(r["C030_deg"]<r[k] for r in good) for k in METHODS if k!="C030_deg"}}


def replay(previous,control,inventory,campaign,split,cfg,k,fixed,models,norm,ctx_norm,prior_geometry,a,log0):
    c1r,c1p,c3r,c2p=(read(ROOT/n) for n in ("C001/results.json","C001/protocol.json","C003/results.json","C002/protocol.json"))
    evidence=c2p["frozen_evidence"]
    map1,map3=({(r["dataset"],r["record"]):r for r in s["sequences"]} for s in (c1r,c3r))
    keys={(r["dataset"],r["record"]) for r in previous["sequences"]}
    spec={(r["dataset"],r["record"]):r for r in inventory["records"] if (r["dataset"],r["record"]) in keys}
    table={(r["dataset"],r["record"]):r for r in control["records"]}
    base.require(len(keys)==213 and set(spec)==keys==set(table)
                 and all(r["side"] in ("development","development_previously_used_for_selection") for r in spec.values())
                 and control["source_csv_sha256"]==sha(ROOT/"C020/sequences.csv") and control["selection_lock_sha256"] is None,
                 "C030 replay or fixed controls leave development scope")
    mp,ap=DATA/"Sassari/medium_v5.mat",DATA/"IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp)==split["source_file_sha256"][str(mp)] and sha(ap)==evidence["array_reference_sha256"],"Frozen reference changed")
    medium=loadmat(mp,variable_names=("Qs","XS1","XS2","AP1","AP2","SH1","SH2"))
    array=c2.reference_array(ap)
    poses_old={(r["record"],r["pose"]):r for r in previous["array_sensor_poses"]}
    seq,poses,failed,ledger=[],[],[],[]
    frozen=FROZEN_SCORES
    with c2.c1.h5py.File(inventory["yt"]["path"],"r") as yt,np.load(MASKS,allow_pickle=False) as masks,np.load(GATES,allow_pickle=False) as gates:
        for i,old in enumerate(previous["sequences"]):
            start=perf_counter()
            key=old["dataset"],old["record"]
            identity={f:old[f] for f in c17.IDENTITY}
            try:
                payload,raw,observed,dt,truth,mask,axes,_=c17.payload(key,spec[key],map3,map1,medium,yt["valid"],split,c1p,array,evidence,cfg)
                q0,q,at,release,w,_,acc,_=payload
                np.testing.assert_array_equal(mask,masks[old["mask_key"]])
                ix=np.flatnonzero(mask)
                row=table[key]
                base.require(len(ix)==old["scored_frames"]==row["scored_frames"]
                             and array_sha(ix,"<i8")==old["scoring_indices_sha256"]==row["scoring_indices_sha256"]
                             and array_sha(at,"<i8")==old["native_input_last_sha256"]
                             and array_sha(release,"<i8")==old["native_release_sha256"],"C029 or fixed native indices changed")
                run=perf_counter()
                dirs,probs,active,rest,context,angle,phi,correction,packed_gap=held(q,at,release,w,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,prior_geometry,a,log0)
                calc=perf_counter()-run
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(active.sum()==old["active_release_count"] and
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ROUTERS[:-1]),
                             "Archived C023 or C029 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C030_deg"][same],errs["C020_deg"][same])
                detail=[]
                for pose,mark in axes.items():
                    use=mark[ix]
                    prior=poses_old[key[1],pose]
                    sig=array_sha(ix[use],"<i8")
                    base.require(sig==prior["scoring_indices_sha256"] and int(use.sum())==prior["scored_frames"],"Original array pose changed")
                    values=c17.scores(errs,use)
                    base.require(max(abs(values[m]-prior[m]) for m in frozen)<=1e-6,"Original array pose controls changed")
                    detail.append({"record":key[1],"pose":pose,"scored_frames":int(use.sum()),"scoring_indices_sha256":sig,**values})
                means={f"{arm}_p_{w}_mean":n for arm,p in probs.items() for w,n in
                       zip(WEIGHTS,p[active].mean(0).tolist() if active.any() else [None]*3,strict=True)}
                seq.append({**identity,**score,**references,**means,"status":"scored","control_max_abs_deg":gap,
                            "native_input_last_sha256":old["native_input_last_sha256"],"native_release_sha256":old["native_release_sha256"],
                            "active_release_count":int(active.sum()),"inactive_scored_frames":int(same.sum()),
                            **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probs.items()},
                            "C030_active_physical_basis_sha256":array_sha(phi[active],"<f8"),
                            "C030_active_fast_slow_angle_mean_deg":float(angle[active].mean()) if active.any() else None,
                            "C030_packed_probability_max_abs_difference":packed_gap,
                            "prediction_seconds":calc,"seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i),"release":release[active],"input_last":at[active],
                               "context":context[active],"fast_slow_angle_deg":angle[active],"physical_basis":phi[active],
                               "antisymmetric_logit_shift":correction[active],**{m:p[active] for m,p in probs.items()}})
                poses.extend(detail)
                print("development",*key,score["C023_deg"],score["C029_deg"],score["C030_deg"],flush=True)
            except (AssertionError,RuntimeError,OSError,ValueError,KeyError,IndexError) as ex:
                failed.append({"dataset":key[0],"record":key[1],"error":repr(ex)})
                seq.append({**identity,"status":"failed","error":repr(ex),"seconds":perf_counter()-start})
                print("FAILED",*key,repr(ex),flush=True)
            if (i + 1) % 10 == 0 or i + 1 == len(previous["sequences"]):
                write(HERE / "replay-progress.json", {"pid": os.getpid(), "attempted": i + 1,
                      "scored": i + 1 - len(failed), "failures": failed, "last_record": list(key), "updated_at": stamp()})
    csv,pos,group,pred=(HERE/n for n in ("sequences.csv","array-sensor-poses.csv","group-summaries.json","development-routing.npz"))
    columns=dict.fromkeys(k for row in seq for k in row)
    intrinsic.shice.save_csv(csv,[{k:r.get(k,"") for k in columns} for r in seq])
    if poses:
        intrinsic.shice.save_csv(pos,poses)
    if ledger:
        np.savez_compressed(pred,**{k:np.concatenate([r[k] for r in ledger]) for k in ledger[0]})
    groups={"SHICE_devices":{g:metrics([v for v in seq if v["dataset"]=="SHICE" and v["group"]==g])
                             for g in sorted({r["group"] for r in seq if r["dataset"]=="SHICE"})},
            "TUM_trajectories":{r["record"]:metrics([r]) for r in seq if r["dataset"]=="TUM-VI"},
            "Oxford_carry":{g:metrics([v for v in seq if v["dataset"]=="OxIOD" and v["group"]==g])
                            for g in sorted({r["group"] for r in seq if r["dataset"]=="OxIOD"})}}
    write(group,groups)
    return {"pooled_records":metrics(seq),"domains":{d:metrics([r for r in seq if r["dataset"]==d]) for d in campaign["datasets"]},
            **groups,"sequences":seq,"array_sensor_poses":poses,"failures":failed,
            "artifact_sha256":{item.name:sha(item) for item in (csv,pos,group,pred) if item.is_file()}}












def replay_locked(protocol):
    lock=read(HERE/"router-lock.json")
    lock_hash=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["solve_count"]==1 and lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/name)==digest for name,digest in lock["artifact_sha256"].items()),
            "Replay requires the original locked C030 SLSQP solution")
    states=[weights_state(name) for name in ROUTERS]
    anchor, prior, own=states
    models={}
    for name,state,width in zip(ROUTERS,states,(24,30,32),strict=True):
        model=nn.Linear(width,2,dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name]=model.eval().requires_grad_(False)
        for k in ("p0","log_p0"):
            torch.testing.assert_close(own[k],state[k],rtol=0.,atol=0.)
    torch.testing.assert_close(own["state_dict"]["weight"][:,:24],anchor["state_dict"]["weight"],rtol=0.,atol=0.)
    torch.testing.assert_close(own["state_dict"]["bias"],anchor["state_dict"]["bias"],rtol=0.,atol=0.)
    require(not own["state_dict"]["weight"][:,24:28].count_nonzero().item() and
            tuple(own["geometry_names"])==GEOMETRY_NAMES and tuple(own["feature_names"])==c25.OBSERVERS,
            "Packed C030 does not freeze the C023 anchor or geometry names")
    np.testing.assert_array_equal(own["geometry_mean"].numpy(),np.zeros(4))
    np.testing.assert_array_equal(own["geometry_std"].numpy(),np.ones(4))
    torch.testing.assert_close(own["state_dict"]["weight"][:,28:],torch.stack((own["a"],-own["a"])),rtol=0.,atol=0.)
    a=own["a"].numpy()
    for key in NORM_KEYS:
        torch.testing.assert_close(own[key],prior[key],rtol=0.,atol=0.)
    mean,std,ctx_mean,ctx_std=(own[k].numpy() for k in NORM_KEYS)
    prior_geometry=tuple(prior[k].numpy() for k in ("geometry_mean","geometry_std"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    previous,control,inventory,split=(read(p) for p in (
        ROOT/"C029/results.json",ROOT/"review/vqf_response_controls_development.json",
        ROOT/"C001/inventory.json",TRANSFER/"external-split.json",
    ))
    require([[r["dataset"],r["record"]] for r in previous["sequences"]]==protocol["development_records"],
            "C029 development replay order changed")
    status(protocol,"development_replay",router_lock_sha256=lock_hash,expected_records=213)
    print("C030 locked; replaying the common 213 development records",flush=True)
    start=perf_counter()
    result=replay(previous,control,inventory,campaign,split,cfg,protocol["C022_k"],fixed,
                  models,(mean,std),(ctx_mean,ctx_std),prior_geometry,a,own["log_p0"])
    require(sha(HERE/"router-lock.json")==lock_hash and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/name)==digest for name,digest in protocol["inputs_sha256"].items()),"Frozen evidence changed")
    result.update(pid=os.getpid(),original_fit_pid=lock["pid"],router_lock_sha256=lock_hash,
                  protocol_sha256=sha(HERE/"protocol.json"),scoring_masks_sha256=sha(MASKS),
                  replay_seconds=perf_counter()-start,completed_at=stamp(),
                  final_holdout_scored=False,supplemental_holdout_scored=False)
    dest=HERE/"results.json"
    write(dest,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(protocol,"complete" if complete else "incomplete",scored=scored,
           router_lock_sha256=lock_hash,results_sha256=sha(dest))
    require(complete,"C030 replay incomplete; checkpoint remains locked")
    print("C030 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def run(launch):
    prior_protocol,inputs,ref=verify_inputs()
    anchor,prior=weights_state("C023"),weights_state("C029")
    k=read(ROOT/"C022/coefficient-lock.json")["k"]
    require(k==prior_protocol["C022_k"] and tuple(prior["feature_names"])==c25.OBSERVERS,"C022 prior or motion features changed")
    for key in ("mean","std"):
        torch.testing.assert_close(anchor["feature_"+key],prior["old_feature_"+key],rtol=0.,atol=0.)
    for key in ("p0","log_p0"):
        torch.testing.assert_close(anchor[key],prior[key],rtol=0.,atol=0.)
    torch.testing.assert_close(anchor["p0"],c23.prior_probs(k),rtol=0.,atol=0.)
    protocol={**launch,"inputs_sha256":inputs,"worker_sha256":sha(Path(__file__)),"C022_k":k,
              "trainable_physical_coefficients":4,"SLSQP_solve_count":1,"solver_options":OPTIONS,
              "start":[0.,0.,0.,0.,1.],"bounds_a":[-1.,1.],"geometry_names":list(GEOMETRY_NAMES),"group_order":list(GROUPS),
              "objective":"Minimize epigraph t subject to each group's mean of original measurement-record angular RMSEs over C023 being <= t",
              "record_loss":"Squared true angle at every frozen training slot, regardless of active gate; RMSE within record condition, then equal record average",
              "endpoint_policy":"Archive the endpoint verbatim; replay it if feasible and improving all groups, otherwise replay the recorded C023 identity; no retry",
              "slots":17600,"source_records":148,"record_conditions":151,
              "development_records":prior_protocol["development_records"],"scoring_masks_sha256":sha(MASKS),
              "formal_holdouts_authorized":False,"supplemental_holdouts_authorized":False}
    write(HERE/"protocol.json",protocol)
    torch.set_num_threads(2)
    status(protocol,"anchor_and_gradient_verification")
    data=archive(ROOT/"C026/training-inputs.npz")
    design=design_from_training(data)
    record_rows=read(ROOT/"C028/source-record-map.json")["records"]
    record_map={(r["group"],r["ordinal"]):r for r in record_rows}
    require(len(record_map)==151,"Acquisition names and conditions differ from the locked source map")
    bank=AngularGroups(data,design,anchor,record_map)
    status(protocol,"single_SLSQP_solve")
    solution=solve(bank,design[0],anchor,prior,ref)
    write(HERE/"solver-receipt.json",solution)
    artifacts=("protocol.json","solver-receipt.json","zero-step-audit.json","finite-difference.json",
               "solver-evaluations.jsonl","solver-iterations.jsonl","anchored-inputs.npz","training-io.npz")
    with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
        json.dump({"pid":os.getpid(),"solve_count":1,"locked_at":stamp(),"locked_choice":solution["locked_choice"],
                   "checkpoint_sha256":sha(HERE/"router.pt"),
                   "artifact_sha256":{name:sha(HERE/name) for name in artifacts}},handle,indent=2)
        handle.write("\n")
    replay_locked(protocol)


def main():
    launch={"pid":os.getpid(),"started_at":stamp(),"command":"python -B -u C030/run_anchored_router.py",
            "scope":"C030 only; one SLSQP solve of four C023-anchored physical coefficients; common development replay"}
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

