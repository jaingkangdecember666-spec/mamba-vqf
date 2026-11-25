"""C034: one four-coefficient raw FullVQF/C023 native interpolation."""
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
from scipy.signal import lfilter
from scipy.special import expit
from scipy.spatial.transform import Rotation
import torch
from vqf import VQF

import run_penn_router as c33
from anchor_blend_features import blend,design,included_degrees
from anchor_blend_risk import AnchorRisk
from check_history_router import prediction
from penn_handheld import BASE,Penn
from raw_anchor_sources import source_inputs

c31,c23,c22,c18,c17,c3,c2,c14=c33.c31,c33.c23,c33.c22,c33.c18,c33.c17,c33.c3,c33.c2,c33.c14
base,intrinsic=c33.base,c33.intrinsic
read,write,sha,stamp,array_sha=c33.read,c33.write,c33.sha,c33.stamp,c33.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c33.UP,c33.DATA,c33.TRANSFER,c33.MASKS,c33.GATES
require,archive,check_probabilities,add_baselines=c33.require,c33.archive,c33.check_probabilities,c33.add_baselines
WEIGHTS=c23.WEIGHTS
PENN=("af","as")
ROUTERS=("C023","C033")
METHODS=(*c23.METHODS[:4],"C033_deg","C034_deg","physical_only_deg",*c23.FIXED)
FEATURES=("intercept","separation_deg_over_6_capped","relative_1s_causal_force_support","tanh_raw_gyro_bias_deg_s")
OPTIONS={"ftol":1e-11,"maxiter":300,"disp":False}
BANK=ROOT/"review/raw-anchor-training-141.npz"
ATOL=1e-9


def local_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def status(launch,phase,**fields):
    write(HERE/"process.json",{"pid":os.getpid(),"started_at":launch["started_at"],"status":phase,"updated_at":local_now(),
                              "protocol_sha256":sha(HERE/"protocol.json"),**fields})


def raw_native(signal,observed,dt,indices=None):
    """Unmodified native VQF plus one causal second of unit specific force in its world frame."""
    x=base.causal_fill(signal,observed)
    acc=np.ascontiguousarray(x[:,:3])
    gyr=np.ascontiguousarray(x[:,3:])
    vqf=VQF(dt).updateBatch(gyr,acc)
    r=Rotation.from_quat(vqf["quat6D"],scalar_first=True)
    unit=acc/np.maximum(np.linalg.norm(acc,axis=1,keepdims=True),1e-12)
    alpha=-np.expm1(-dt)
    mean=lfilter([alpha],[1.,alpha-1.],r.apply(unit),axis=0)
    if indices is None:
        keep=r.inv()
        bias=vqf["bias"]
    else:
        keep=r[indices].inv()
        mean=mean[indices]
        bias=vqf["bias"][indices]
    mean/=np.maximum(np.linalg.norm(mean,axis=1,keepdims=True),1e-12)
    return keep.apply(UP),keep.apply(mean),np.linalg.norm(bias,axis=1)


def verify_slots(protocol,anchor):
    bank=archive(BANK)
    ledger=archive(ROOT/"C033/training-inputs.npz")
    previous=archive(ROOT/"C022/training-slots.npz")
    original=archive(ROOT/"review/external-raw-anchor-139.npz")
    require(bank["features"].shape==(15680,4) and ledger["enabled"].shape==(160,98)
            and np.unique(bank["record_id"]).size==141 and np.isfinite(bank["features"]).all(),"C034 locked training bank shape differs")
    fields={"domain":"domain","record":"source_record_name","record_id":"source_record_ordinal",
            "native_release":"native_release","truth":"truth"}
    for target,source in fields.items():
        value=ledger[source]
        np.testing.assert_array_equal(bank[target],value.reshape((-1,)+value.shape[2:]))
        np.testing.assert_array_equal(bank[target].reshape((160,98)+value.shape[2:])[:,:96].reshape(previous[source].shape),
                                      previous[source])
    for key in ("native_input_last","center","fast","slow","enabled","rest"):
        np.testing.assert_array_equal(ledger[key][:,:96].reshape(previous[key].shape),previous[key])
    frozen_old=(np.arange(15680)%98)<96
    for new,key in (("raw_default","raw_default"),("past_force","past_force"),("raw_gyro_bias_rad_s","estimated_bias_rad_s")):
        np.testing.assert_array_equal(bank[new][frozen_old],original[key])
    np.testing.assert_allclose(bank["C023"][frozen_old],original["C023"],rtol=1e-12,atol=1e-13)
    target=ledger["truth"].reshape(-1,3)
    for key in ("raw_default","C023","truth","past_force"):
        require(np.isfinite(bank[key]).all() and np.allclose(np.linalg.norm(bank[key],axis=1),1.,rtol=0.,atol=2e-12),
                "An archived endpoint, reference or observer is not a finite unit vector")
    params=torch.cat([anchor["state_dict"][key].ravel() for key in ("weight","bias")]).numpy()
    candidate,*_=prediction(ledger["z"].reshape(-1,24),anchor["p0"].numpy(),params,
                            *(ledger[k].reshape(-1,3) for k in ("center","fast","slow")),ledger["enabled"].ravel())
    np.testing.assert_allclose(candidate,bank["C023"],rtol=1e-12,atol=1e-13)
    recomputed=design(bank["raw_default"],bank["C023"],bank["past_force"],bank["raw_gyro_bias_rad_s"])
    np.testing.assert_array_equal(recomputed,bank["features"])
    require(np.array_equal(bank["truth"],target) and len(set(zip(bank["domain"],bank["record"])))==141,
            "C033 annotations or identities changed")
    require(sha(BANK)==protocol["inputs_sha256"]["review/raw-anchor-training-141.npz"],"Training bank hash mismatch")
    np.savez_compressed(HERE/"original-slot-identities.npz",source_batch_index=np.repeat(np.arange(160),98),
                        slot_in_batch=np.tile(np.arange(98),160),native_input_last=ledger["native_input_last"].ravel(),
                        enabled=ledger["enabled"].ravel(),**{k:bank[k] for k in ("domain","record","record_id","native_release")})
    return bank,previous,ledger,float(np.max(np.abs(candidate-bank["C023"])))


def rebuild_sources(bank,previous,ledger,cfg):
    """Rebuild sampled original approved acquisitions and both Penn A releases from raw files."""
    pairs={(str(d),str(r)) for d,r in zip(previous["domain"],previous["source_record_name"],strict=True)}
    chosen={min((d,r) for d,r in pairs if d==source) for source in ("BROAD","Sassari","YT_Server")}
    seen=set()
    proof=[]
    for domain,name,payload in source_inputs(chosen):
        key=domain,name
        require(key in pairs and key not in seen and (key in chosen or domain=="RIDI"),"A non-training original source entered C034 rebuild")
        try:
            signal,hz,reference,valid=payload
            record=(previous["domain"]==domain)&(previous["source_record_name"]==name)
            idx=previous["native_release"][record]
            require(valid[idx].all() and np.isfinite(signal[idx]).all(),"A selected original source reference or IMU row is invalid")
            r=Rotation.from_quat(previous["q_default_release"][record],scalar_first=True)
            rb,force,bias=raw_native(signal,np.isfinite(signal).all(axis=1),1./hz,idx)
            old=bank["native_release"].reshape(160,98)[:,:96].ravel()
            np.testing.assert_array_equal(old[record],idx)
            keep=np.flatnonzero(np.tile(np.arange(98)<96,160))[record]
            raw_error=float(np.max(np.abs(r.apply(rb)-bank["raw_default"][keep])))
            force_error=float(np.max(np.abs(r.apply(force)-bank["past_force"][keep])))
            bias_error=float(np.max(np.abs(bias-bank["raw_gyro_bias_rad_s"][keep])))
            truth_error=float(np.max(np.abs(r.apply(reference[idx])-bank["truth"][keep])))
            require(max(raw_error,force_error,bias_error)<=1e-10 and truth_error<=2e-6,
                    f"Independent native rebuild differs for {domain}/{name}")
            proof.append({"source":domain,"record":name,"slots":len(idx),"native_hz":hz,"native_release_sha256":array_sha(idx,"<i8"),
                          "raw_direction_max_abs_error":raw_error,"force_max_abs_error":force_error,
                          "gyro_bias_max_abs_error":bias_error,"target_max_abs_error":truth_error})
        finally:
            seen.add(key)
    require({d for d,_ in seen}==set(c23.DOMAINS) and chosen<=seen,"Missing an original source audit")
    ds=Penn()
    require(not ds.formal and all(ds.roles[name]=="external_training" for name in PENN),"Only Penn A may be read numerically")
    selection=read(ROOT/"C033/sampling-lock.json")["records"]
    penn={}
    for j,name in enumerate(PENN):
        require(selection[j]["record"]==name,"Penn source order changed")
        payload,quality=ds.native(name)
        signal,hz,reference,mask=payload
        observed=np.isfinite(signal).all(axis=1)
        idx=ledger["native_release"][:,96+j]
        last=ledger["native_input_last"][:,96+j]
        keep=np.arange(96+j,15680,98)
        require(len(np.unique(idx))==160 and mask[idx].all() and observed[idx].all() and observed[last].all(),
                "Invalid Penn archived native release or incomplete observations")
        for key,rows in (("native_release",idx),("native_input_last",last)):
            np.testing.assert_array_equal(rows,[v[key] for v in selection[j]["selected"]])
        require(all(v["complete_history"] and v["enabled"] and not v["rest"] for v in selection[j]["selected"]),
                "Penn training-slot gate evidence changed")
        with np.load(ROOT/"C033"/(name+"-native-mask-evidence.npz"),allow_pickle=False) as origin:
            np.testing.assert_array_equal(mask,origin["scoring_mask"])
            np.testing.assert_array_equal(idx,origin["release"][origin["selected"]])
            np.testing.assert_array_equal(last,origin["native_input_last"][origin["selected"]])
        packet=c17.front_end(signal,observed,1./hz,cfg)
        q=packet[1]
        r=Rotation.from_quat(q[idx],scalar_first=True)
        rb,force,bias=raw_native(signal,observed,1./hz,idx)
        raw_error=float(np.max(np.abs(r.apply(rb)-bank["raw_default"][keep])))
        force_error=float(np.max(np.abs(r.apply(force)-bank["past_force"][keep])))
        bias_error=float(np.max(np.abs(bias-bank["raw_gyro_bias_rad_s"][keep])))
        truth_error=float(np.max(np.abs(r.apply(reference[idx])-bank["truth"][keep])))
        require(max(raw_error,force_error,bias_error,truth_error)<=2e-10,f"Penn raw observer rebuild differs for {name}")
        proof.append({"source":"PennCOSYVIO","record":name,"slots":len(idx),"native_hz":hz,
                      "native_release_sha256":array_sha(idx,"<i8"),"original_mask_sha256":quality["scoring_indices_sha256"],
                      "raw_direction_max_abs_error":raw_error,"force_max_abs_error":force_error,
                      "gyro_bias_max_abs_error":bias_error,"target_max_abs_error":truth_error})
        penn[name]=(payload,packet,observed,quality)
    return proof,penn


class AngularRisk:
    def __init__(self,bank):
        self.z=bank["features"]
        self.raw=bank["raw_default"]
        self.model=bank["C023"]
        self.ref=bank["truth"]
        self.diff=self.model-self.raw
        names=bank["domain"].tolist()
        records=bank["record"].tolist()
        self.ids=sorted(set(zip(names,records)))
        ordinal={pair:i for i,pair in enumerate(self.ids)}
        self.row_id=np.asarray([ordinal[pair] for pair in zip(names,records,strict=True)])
        self.count=np.bincount(self.row_id)
        self.domains=sorted(set(names))
        ranks={name:i for i,name in enumerate(self.domains)}
        self.group=np.asarray([ranks[d] for d,_ in self.ids])
        self.by_group=np.bincount(self.group)
        self.raw_record=self.record_rms(included_degrees(self.raw,self.ref))
        self.den=self.macro(self.raw_record)
        require(len(self.ids)==141 and len(self.domains)==5 and np.all(self.den>0.),"Invalid five-source record denominators")
        self.cross_raw=np.cross(self.raw,self.ref)
        self.cross_delta=np.cross(self.diff,self.ref)
        self.dot_raw=np.einsum("ij,ij->i",self.raw,self.ref)
        self.dot_delta=np.einsum("ij,ij->i",self.diff,self.ref)

    def record_rms(self,angle_deg):
        return np.sqrt(np.bincount(self.row_id,weights=angle_deg**2)/self.count)

    def macro(self,record_values):
        return np.bincount(self.group,weights=record_values)/self.by_group

    def evaluate(self,beta):
        g=expit(self.z@beta)
        cross=self.cross_raw+g[:,None]*self.cross_delta
        sn=np.linalg.norm(cross,axis=1)
        cs=self.dot_raw+g*self.dot_delta
        theta=np.arctan2(sn,cs)
        require(not np.any((sn==0.)&(cs<0.)),"Antipodal blend has an undefined joint derivative")
        ds=np.divide(np.einsum("ij,ij->i",cross,self.cross_delta),sn,out=np.zeros_like(sn),where=sn>0.)
        slope=(cs*ds-sn*self.dot_delta)/(sn**2+cs**2)
        weighted_theta=theta*slope*g*(1.-g)
        record=self.record_rms(np.rad2deg(theta))
        scale=np.divide((180./np.pi)**2,self.count*record,out=np.zeros_like(record),where=record>1e-15)
        dj=np.column_stack([np.bincount(self.row_id,weights=weighted_theta*col)*scale for col in self.z.T])
        return self.macro(record)/self.den,np.column_stack([self.macro(dj[:,j])/self.den for j in range(4)]),record,dj


def derivative_check(risk,reference,beta):
    ratios,analytic,*_=risk.evaluate(beta)
    expected,expected_jac=reference.risk(beta)
    np.testing.assert_allclose(ratios,expected,rtol=0.,atol=2e-13)
    np.testing.assert_allclose(analytic,expected_jac,rtol=0.,atol=2e-11)
    diffs={}
    for h in (1e-4,1e-5,1e-6):
        grad=np.column_stack([(risk.evaluate(beta+delta)[0]-risk.evaluate(beta-delta)[0])/(2*h) for delta in np.eye(4)*h])
        gap=float(np.max(np.abs(grad-analytic)))
        require(gap<1e-7,"C034 analytic ratio gradient does not match a direct finite difference")
        diffs[str(h)]={"numeric_gradient":grad.tolist(),"max_abs_gap":gap}
    return {"beta":beta.tolist(),"ratio":ratios.tolist(),"analytic_jacobian":analytic.tolist(),
            "independent_jacobian_max_abs_gap":float(np.max(np.abs(analytic-expected_jac))),"finite_difference":diffs}


def solve(risk,protocol):
    start=perf_counter()
    independent=AnchorRisk(BANK)
    initial_beta=np.zeros(4,dtype=np.float64)
    before=derivative_check(risk,independent,initial_beta)
    pre=read(ROOT/"review/raw-anchor-risk-preflight.json")
    require(risk.domains==pre["domains"] and pre["bank_sha256"]==sha(BANK),"Published risk diagnostic differs from the locked bank")
    np.testing.assert_allclose(risk.den,pre["raw_default_record_macro_deg"],rtol=0.,atol=2e-13)
    np.testing.assert_allclose(before["ratio"],pre["initial_zero_coefficients_ratio"],rtol=0.,atol=2e-13)
    np.testing.assert_allclose(before["analytic_jacobian"],pre["initial_jacobian"],rtol=0.,atol=2e-11)
    t0=max(before["ratio"])
    x0=np.r_[initial_beta,t0]
    write(HERE/"zero-step-audit.json",{"pid":os.getpid(),"group_order":risk.domains,"raw_default_group_RMSE_deg":risk.den.tolist(),
                                       "beta_initial":[0.]*4,"epigraph_initial_t":t0,"equal_blend_check":before})
    callbacks,evaluations=0,0
    oldbeta,cached_r,cached_j=None,None,None
    seen=set()
    with ((HERE/"solver-evaluations.jsonl").open("x",encoding="utf-8") as point_file,
          (HERE/"solver-iterations.jsonl").open("x",encoding="utf-8") as iter_file):
        def data(x):
            nonlocal oldbeta,cached_r,cached_j,evaluations
            beta=x[:4]
            if oldbeta is None or not np.array_equal(beta,oldbeta):
                cached_r,cached_j,*_=risk.evaluate(beta)
                oldbeta=beta.copy()
            key=tuple(map(float,x))
            if key not in seen:
                seen.add(key)
                evaluations+=1
                print(json.dumps({"evaluation":evaluations,"pid":os.getpid(),"x":list(key),"ratios":cached_r.tolist(),
                                  "ratio_jacobian":cached_j.tolist(),"constraint_slack":(x[4]-cached_r).tolist(),
                                  "constraint_jacobian":np.column_stack((-cached_j,np.ones(len(risk.domains)))).tolist()},allow_nan=False),
                      file=point_file,flush=True)
            return cached_r,cached_j

        def objective(x):
            data(x)
            return float(x[4])

        def constraints(x):
            return x[4]-data(x)[0]

        def jacobian(x):
            return np.column_stack((-data(x)[1],np.ones(len(risk.domains))))

        def log_iteration(x,kind,index):
            ratio,jac=data(x)
            print(json.dumps({"kind":kind,"iteration":index,"pid":os.getpid(),"x":x.tolist(),
                              "ratios":ratio.tolist(),"ratio_jacobian":jac.tolist(),
                              "constraint_slack":(x[4]-ratio).tolist()},allow_nan=False),file=iter_file,flush=True)

        def callback(x):
            nonlocal callbacks
            callbacks+=1
            log_iteration(x,"SLSQP_callback",callbacks)
            if callbacks in (1,10,25,50,100,200,300):
                print("C034 iteration",callbacks,float(x[4]),flush=True)

        log_iteration(x0,"start",0)
        opt=minimize(objective,x0,jac=lambda _:np.array([0.,0.,0.,0.,1.]),method="SLSQP",
                     bounds=[(-4.,4.)]*4+[(None,None)],
                     constraints=({"type":"ineq","fun":constraints,"jac":jacobian},),
                     callback=callback,options=OPTIONS)
        terminal=np.asarray(opt.x,dtype=np.float64)
        require(terminal.shape==(5,) and np.isfinite(terminal).all(),"Nonfinite SLSQP endpoint")
        log_iteration(terminal,"terminal",callbacks+1)
    after=derivative_check(risk,independent,terminal[:4])
    ratios,jac,per_record,record_grad=risk.evaluate(terminal[:4])
    slack=terminal[4]-ratios
    bound_gap=float(np.maximum(np.abs(terminal[:4])-4.,0.).max())
    violation=float(max(0.,-slack.min()))
    feasible=bound_gap<=ATOL and violation<=ATOL
    beta=terminal[:4].copy()
    write(HERE/"finite-difference.json",{"pid":os.getpid(),"group_order":risk.domains,"zero":before,"terminal":after})
    g=expit(risk.z@beta)
    y=blend(risk.raw,risk.model,g)
    theta=included_degrees(y,risk.ref)
    np.testing.assert_allclose(risk.record_rms(theta),per_record,rtol=0.,atol=2e-13)
    source=np.array([d for d,_ in risk.ids])
    source_only_model=risk.macro(risk.record_rms(included_degrees(risk.model,risk.ref)))
    np.savez_compressed(HERE/"training-risk.npz",blend_weight=g,body_or_rotated_direction=y,angle_deg=theta,
                        record_index=risk.row_id,record_default_RMSE_deg=risk.raw_record,record_blend_RMSE_deg=per_record)
    result={"pid":os.getpid(),"solve_count":1,"neural_fits":0,"optimizer_steps":0,"method":"SLSQP",
            "scipy_version":scipy.__version__,"options":OPTIONS,"coefficient_bounds":[-4.,4.],"initial_x":x0.tolist(),
            "terminal_x":terminal.tolist(),"locked_beta":beta.tolist(),"result_success":bool(opt.success),
            "result_status":int(opt.status),"result_message":str(opt.message),"result_objective":float(opt.fun),
            "nit":int(opt.nit),"nfev":int(opt.nfev),"njev":int(opt.njev),"callback_count":callbacks,
            "distinct_solver_point_count":evaluations,"distinct_a_evaluation_count":len({x[:4] for x in seen}),
            "reported_multipliers":np.asarray(getattr(opt,"multipliers",[])).tolist(),"terminal_feasible":bool(feasible),
            "terminal_max_constraint_violation":violation,"terminal_max_bound_violation":bound_gap,
            "terminal_epigraph_max_group_ratio":float(ratios.max()),"group_order":risk.domains,
            "selection_note":"One terminal point only; preserve success, convergence and feasibility as reported",
            "groups":{d:{"records":int(risk.by_group[i]),"slots":int(risk.count[source==d].sum()),
                         "raw_default_macro_RMSE_deg":float(risk.den[i]),"C023_macro_RMSE_deg":float(source_only_model[i]),
                         "C034_macro_RMSE_deg":float(ratios[i]*risk.den[i]),"C034_ratio":float(ratios[i]),
                         "initial_ratio_gradient":before["analytic_jacobian"][i],"terminal_ratio_gradient":jac[i].tolist(),
                         "constraint_slack":float(slack[i])} for i,d in enumerate(risk.domains)},
            "training_record_RMSE_deg":[{"source":d,"record":n,"slots":int(risk.count[i]),
                                         "raw":float(risk.raw_record[i]),"C034":float(per_record[i]),
                                         "C034_RMSE_gradient_deg":record_grad[i].tolist()} for i,(d,n) in enumerate(risk.ids)],
            "fit_bank_sha256":sha(BANK),"parent_C023_checkpoint_sha256":protocol["inputs_sha256"]["C023/router.pt"],
            "solve_and_archive_seconds":perf_counter()-start}
    write(HERE/"solver-receipt.json",result)
    require(feasible,"C034 terminal epigraph point is infeasible; receipt retained")
    torch.save({"beta":torch.from_numpy(beta),"feature_names":list(FEATURES),"parent_checkpoint_sha256":protocol["inputs_sha256"]["C023/router.pt"],
                "protocol_sha256":sha(HERE/"protocol.json"),"solver_receipt_sha256":sha(HERE/"solver-receipt.json"),
                "solve_count":1,"neural_fits":0},HERE/"router.pt")
    return beta




@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,routers):
    original,center,physical,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    active=valid&~rest
    models,average,scale,log0=routers
    backbones,mean,std,*_=fixed
    features=c18.features(history,q[at],backbones[0],mean,std)
    np.testing.assert_array_equal(features.valid,valid)
    normalized=torch.from_numpy((features.pooled[:,1].astype(np.float64)-average)/scale)
    extra=k*np.where(active[:,None],mix,0.)
    out={"C020_deg":original,"C022_deg":c23.augment(original,center,extra),
         "physical_only_deg":c14.normalize(physical+extra)}
    probabilities,deltas={},{}
    for name,model in models.items():
        p=c23.weights(model,normalized,log0).numpy()
        check_probabilities(p)
        delta=np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        require(np.max(np.linalg.norm(delta,axis=1))<=c22.CAP+2e-15 and np.all(delta[:,2]==0.),
                "C023 or C033 gain escaped the C022 capped response")
        out[name+"_deg"]=c23.augment(original,center,delta)
        deltas[name]=delta
        probabilities[name]=p
    return out,probabilities,deltas,active,rest


def native_blend(raw,observed,dt,q0,q,release,central,scoring_indices,beta):
    """Compute the blend independently at each native row, then select only the frozen scoring mask."""
    u0,force,bias=raw_native(raw,observed,dt)
    n=len(raw)
    require(len(q0)==len(q)==n and np.all(np.diff(release)>0),"C034 native timelines differ")
    use=np.searchsorted(release,np.arange(n),side="right")
    model=Rotation.from_quat(q,scalar_first=True).inv().apply(np.vstack((UP,central))[use])
    check_raw=Rotation.from_quat(q0[scoring_indices],scalar_first=True).inv().apply(UP)
    native_gap=float(np.max(np.abs(check_raw-u0[scoring_indices])))
    require(native_gap<=2e-8,"C034 raw anchor differs from the native unmodified FullVQF control")
    weights=expit(design(u0,model,force,bias)@beta)
    require(weights.shape==(n,) and np.isfinite(weights).all() and np.all((weights>0.)&(weights<1.)),
            "Invalid C034 blend weights")
    y=blend(u0,model,weights)
    return y[scoring_indices],weights[scoring_indices],native_gap,n


def score_penn(penn,k,fixed,routers,beta):
    records=[]
    previous={r["record"]:r for r in read(ROOT/"C033/penn-training-source-scores.json")["records"]}
    for name in PENN:
        payload,packet,observed,quality=penn[name]
        raw,hz,ref,mask=payload
        q0,q,at,release,hist,_,acc,_=packet
        estimates,*_=held(q,at,release,hist,acc,raw,observed,1./hz,k,fixed,routers)
        ix=np.flatnonzero(mask)
        native,weights,raw_gap,count=native_blend(raw,observed,1./hz,q0,q,release,estimates["C023_deg"],ix,beta)
        errors=c17.measure(ix,ref,q0,q,release,{m:np.vstack((UP,p)) for m,p in estimates.items()})
        errors["C034_deg"]=included_degrees(native,ref[ix])**2
        score=c17.scores(errors)
        old=previous[name]
        require(len(ix)==old["scored_frames"] and array_sha(ix,"<i8")==old["mask_sha256"]==quality["scoring_indices_sha256"],
                "Penn A source score mask changed")
        for current,archived in (("C023_deg","C023"),("C033_deg","C033"),("full_vqf_deg","full_vqf_deg")):
            require(abs(score[current]-old["full_record_tilt_RMSE_deg"][archived])<=1e-6,"Penn A source control changed")
        score.update(fixed_tau1p5_deg=old["full_record_tilt_RMSE_deg"]["archived_fixed_fast"],
                     fixed_tau3_deg=old["full_record_tilt_RMSE_deg"]["full_vqf_deg"],
                     fixed_tau6_deg=old["full_record_tilt_RMSE_deg"]["archived_fixed_slow"])
        records.append({"record":name,"scored_frames":len(ix),"mask_sha256":array_sha(ix,"<i8"),
                        "native_packets_blended":count,"raw_anchor_max_abs_gap":raw_gap,
                        "blend_weight_mean":float(weights.mean()),"full_record_tilt_RMSE_deg":score})
    mean={k:fmean(row["full_record_tilt_RMSE_deg"][k] for row in records) for k in records[0]["full_record_tilt_RMSE_deg"]}
    write(HERE/"penn-training-source-scores.json",{"pid":os.getpid(),"scope":"Penn A only, in-sample full native training-source score",
          "records":records,"record_equal_mean":mean,"checkpoint_sha256":sha(HERE/"router.pt")})



def metrics(rows):
    keep = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in keep) for k in METHODS} if keep else {}
    active = sum(r["active_release_count"] for r in keep)
    weighted = {}
    if active:
        weighted = {key: fmean(r[f"p_{key}_mean"] for r in keep if r["active_release_count"] > 0) for key in WEIGHTS}
    return {"records": len(keep), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in keep),
            "macro_tilt_rmse_deg": macro, "active_releases": active, "mean_routing_prob_equal_records_with_support": weighted,
            "C034_delta_deg": {m: macro["C034_deg"] - macro[m] for m in METHODS if m != "C034_deg"} if keep else {},
            "C034_wins": {m: sum(r["C034_deg"] < r[m] for r in keep) for m in METHODS if m != "C034_deg"}}


def replay(prior, fixed_scores, inventory, campaign, split, cfg, k, fixed, router, beta):
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
                baked, blend_weights, raw_gap, native_count = native_blend(raw, observed, dt, q0, q, release,
                                                                            predictions["C023_deg"], ix, beta)
                errs["C034_deg"] = included_degrees(baked, ref[ix])**2
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C033_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C033 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C033_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C033_deg"][~active_scored], errs["C020_deg"][~active_scored])
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
                             **{f"p_{k}_mean": v for k, v in active_probs["C023"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "C034_native_packets_blended":native_count,
                             "C034_scored_blend_weight_mean":float(blend_weights.mean()),
                             "C034_scored_blend_weight_sha256":array_sha(blend_weights,"<f8"),
                             "C034_raw_anchor_max_abs_gap":raw_gap,
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C033_deg"], score["C034_deg"], flush=True)
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












def replay_locked(protocol,launch,fixed,campaign,cfg,models,beta,penn):
    lock=read(HERE/"router-lock.json")
    lock_hash=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["solve_count"]==1 and lock["neural_fits"]==0 and
            lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/name)==digest for name,digest in lock["artifact_sha256"].items()),"Replay requires original C034 locked blend")
    stored=torch.load(HERE/"router.pt",map_location="cpu",weights_only=True)
    np.testing.assert_array_equal(beta,stored["beta"].numpy())
    require(stored["parent_checkpoint_sha256"]==protocol["inputs_sha256"]["C023/router.pt"] and
            stored["solver_receipt_sha256"]==sha(HERE/"solver-receipt.json") and
            stored["protocol_sha256"]==sha(HERE/"protocol.json"),"C034 checkpoint or parent evidence differs")
    anchor=c33.frozen_state("C023")
    routers=models,anchor["feature_mean"].numpy(),anchor["feature_std"].numpy(),anchor["log_p0"]
    status(launch,"scoring_Penn_A_training_source",router_lock_sha256=lock_hash)
    score_penn(penn,protocol["C022_k"],fixed,routers,beta)
    prior=read(ROOT/"C033/results.json")
    require([[r["dataset"],r["record"]] for r in prior["sequences"]]==protocol["development_records"],
            "Development panel or order changed")
    status(launch,"development_replay",router_lock_sha256=lock_hash,expected_records=213)
    print("C034 locked; native-packet replay on the common development panel",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,routers,beta)
    require(sha(HERE/"router-lock.json")==lock_hash and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/name)==h for name,h in protocol["inputs_sha256"].items()),"Frozen evidence changed during replay")
    result.update(pid=os.getpid(),router_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),penn_training_source_scores_sha256=sha(HERE/"penn-training-source-scores.json"),
                  solver_receipt_sha256=sha(HERE/"solver-receipt.json"),solve_count=1,neural_fits=0,
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",scored=scored,router_lock_sha256=lock_hash,results_sha256=sha(out))
    require(complete,"C034 replay incomplete; solver and checkpoint retained")
    print("C034 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def run(protocol,launch):
    require(protocol["options"]==OPTIONS and protocol["beta_initial"]==[0.]*4 and protocol["beta_bounds"]==[-4.,4.] and
            protocol["solver_calls"]==1 and not protocol["neural_fits"],"Registered solver recipe changed")
    torch.set_num_threads(2)
    status(launch,"verifying_raw_anchor_bank")
    states={name:c33.frozen_state(name) for name in ROUTERS}
    anchor=states["C023"]
    for key in c33.STATE_KEYS:
        torch.testing.assert_close(anchor[key],states["C033"][key],atol=0.,rtol=0.)
    bank,original,ledger,delta=verify_slots(protocol,anchor)
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    rebuild,penn=rebuild_sources(bank,original,ledger,cfg)
    write(HERE/"source-rebuild.json",{"pid":os.getpid(),"bitwise_slot_matches":15680,"record_count":141,
          "frozen_C023_direction_max_abs_error":delta,"sources_rebuilt_from_approved_native_inputs":rebuild,
          "original_slot_identities_sha256":sha(HERE/"original-slot-identities.npz"),
          "fit_bank_sha256":sha(BANK),"Penn_B_numeric_reads":0})
    status(launch,"single_SLSQP_solve",source_rebuild_sha256=sha(HERE/"source-rebuild.json"))
    beta=solve(AngularRisk(bank),protocol)
    artifacts=("protocol.json","source-rebuild.json","original-slot-identities.npz","zero-step-audit.json",
               "solver-iterations.jsonl","solver-evaluations.jsonl","solver-receipt.json","finite-difference.json","training-risk.npz")
    with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
        json.dump({"pid":os.getpid(),"locked_at":local_now(),"solve_count":1,"neural_fits":0,"optimizer_steps":0,
                   "checkpoint_sha256":sha(HERE/"router.pt"),
                   "artifact_sha256":{n:sha(HERE/n) for n in artifacts}},handle,indent=2)
        handle.write("\n")
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    models={name:c31.frozen_linear(state) for name,state in states.items()}
    replay_locked(protocol,launch,fixed,campaign,cfg,models,beta,penn)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C034/run_raw_anchor_blend.py",
            "scope":"One C034 SLSQP over four fixed-bank coefficients, native development replay, separate Penn A training score",
            "protocol_sha256":sha(HERE/"protocol.json")}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"C034 registered artifact changed")
        run(protocol,launch)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

