"""C037: fit one sphere-angle cap over the frozen C023 correction."""
from datetime import datetime,timedelta,timezone
import heapq
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
import torch

import run_trust_router as c35
from anchor_blend_features import included_degrees
from penn_handheld import Penn
from physical_angle_cap import cap_direction
from physical_cap_certificate import ExternalCap

c33,c31,c23,c22,c18,c17,c3,c2,c14=c35.c33,c35.c31,c35.c23,c35.c22,c35.c18,c35.c17,c35.c3,c35.c2,c35.c14
base,intrinsic=c35.base,c35.intrinsic
read,write,sha,stamp,array_sha=c35.read,c35.write,c35.sha,c35.stamp,c35.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c35.UP,c35.DATA,c35.TRANSFER,c35.MASKS,c35.GATES
require,archive,check_probabilities,add_baselines=c35.require,c35.archive,c35.check_probabilities,c35.add_baselines
ROUTERS=("C023","C035")
METHODS=(*c23.METHODS[:4],"C035_deg","physical_only_deg","C036_deg","C037_deg",*c23.FIXED)
WEIGHTS=c23.WEIGHTS
PENN=("af","as")


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
        require(np.all(self.ref>0.) and np.max(self.disagreement)>0.,"Degenerate C037 angle search or default denominator")
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





CERTIFICATE_TOL=1e-5
SLOPE_EDGE_EPS=1e-11
CHECK_TOL=2e-9


def solve(protocol,bank):
    reference=ExternalCap()
    np.testing.assert_array_equal(reference.disagreement,bank.disagreement)
    np.testing.assert_allclose(reference.den,bank.ref,rtol=0.,atol=2e-13)
    require(reference.groups==bank.groups and
            reference.provenance=={"raw":protocol["inputs_sha256"]["review/raw-anchor-training-141.npz"],
                                   "physical":protocol["inputs_sha256"]["review/external-physical-anchor-141.npz"]},
            "Independent branch-and-bound source reference differs from the registered data")
    upper_endpoint=float(bank.disagreement.max())
    points={}
    slope_cache={}
    best=None
    leaf_heap=[]
    splits=0
    max_risk_reference_gap=0.
    max_bound_reference_gap=0.
    start=perf_counter()
    started_at=local_now()
    with ((HERE/"solver-evaluations.jsonl").open("x",encoding="utf-8") as point_log,
          (HERE/"solver-branches.jsonl").open("x",encoding="utf-8") as branch_log):
        def point(x):
            nonlocal best,max_risk_reference_gap
            if x in points:
                return points[x]
            objective,group,*_=bank.evaluate(x)
            ratios=np.array([group[d]["ratio_to_default"] for d in bank.groups])
            independent=reference.risks(x)
            delta=float(np.max(np.abs(ratios-independent)))
            max_risk_reference_gap=max(max_risk_reference_gap,delta)
            require(delta<=2e-13 and objective==float(ratios.max()),"Independent source risks disagree")
            label=len(points)+1
            points[x]=(label,ratios,objective)
            if best is None or (objective,x)<(best[0],best[1]):
                best=(objective,x,label)
            print(json.dumps({"evaluation":label,"pid":os.getpid(),"cap_deg":x,"objective":objective,
                              "ratios_by_source":dict(zip(bank.groups,ratios.tolist(),strict=True)),
                              "groups":group,"incumbent_cap_deg":best[1],"incumbent_upper_bound":best[0]},allow_nan=False),
                  file=point_log,flush=True)
            return points[x]

        def slope(a):
            if a not in slope_cache:
                moving=bank.disagreement>=a-SLOPE_EDGE_EPS
                record=np.sqrt(np.bincount(bank.ordinal,weights=moving.astype(np.float64))/bank.count)
                slope_cache[a]=np.array([record[ids].mean() for ids in bank.members])/bank.ref
            return slope_cache[a]

        def bound(left,right):
            nonlocal max_bound_reference_gap
            _,rl,_=point(left)
            _,rr,_=point(right)
            sl=slope(left)
            width=right-left
            require(width>0 and np.all(np.abs(rr-rl)<=sl*width+CHECK_TOL),"An interval violates its source Lipschitz bound")
            by_source=.5*(rl+rr-sl*width)
            lower=float(by_source.max())
            independent=reference.interval_bound(left,right,rl,rr)
            # The independent helper uses strict > at the tolerance edge.  C037 uses the registered >=.
            ref_slopes=reference.slopes(left)
            gap=abs(lower-independent)
            max_bound_reference_gap=max(max_bound_reference_gap,gap)
            if np.array_equal(bank.disagreement>=left-SLOPE_EDGE_EPS,bank.disagreement>left-SLOPE_EDGE_EPS):
                require(gap<=2e-13,"Independent interval lower bound differs")
            else:
                require(np.all(sl+2e-13>=ref_slopes) and lower<=independent+2e-13,
                        "Inclusive edge count is not conservative")
            return (lower,left,right),by_source,sl

        point(0.)
        point(upper_endpoint)
        node,_,_=bound(0.,upper_endpoint)
        heapq.heappush(leaf_heap,node)
        while best[0]-leaf_heap[0][0]>CERTIFICATE_TOL:
            lower,left,right=heapq.heappop(leaf_heap)
            mid=left+(right-left)/2.
            require(left<mid<right,"Binary floating-point resolution reached before the gap criterion")
            label,_,mid_value=point(mid)
            children=[]
            for a,b in ((left,mid),(mid,right)):
                child,_,_=bound(a,b)
                heapq.heappush(leaf_heap,child)
                children.append({"left":a,"right":b,"lower_bound":child[0]})
            splits+=1
            print(json.dumps({"split":splits,"pid":os.getpid(),"parent":{"left":left,"right":right,"lower_bound":lower},
                              "midpoint_evaluation":label,"midpoint_objective":mid_value,"children":children,
                              "global_upper_bound":best[0],"global_lower_bound":leaf_heap[0][0],
                              "gap":best[0]-leaf_heap[0][0]},allow_nan=False),file=branch_log,flush=True)
            if splits in (1,10,25,50,100,250,500,1000) or splits%2000==0:
                print("C037 split",splits,"gap",best[0]-leaf_heap[0][0],flush=True)

    leaves=[]
    previous=0.
    for lower,left,right in sorted(leaf_heap,key=lambda leaf:leaf[1]):
        require(left==previous and left<right,"C037 certificate leaves contain a gap or overlap")
        checked,by_source,sl=bound(left,right)
        require(checked[0]==lower,"The terminal leaf bound does not reproduce")
        leaves.append({"left":left,"right":right,"lower_bound":lower,
                       "left_evaluation":points[left][0],"right_evaluation":points[right][0],
                       "L_by_source":dict(zip(bank.groups,sl.tolist(),strict=True)),
                       "lower_bound_by_source":dict(zip(bank.groups,by_source.tolist(),strict=True))})
        previous=right
    require(previous==upper_endpoint and len(leaves)==splits+1 and len(points)==splits+2,
            "Endpoint coverage or shared-evaluation accounting failed")
    upper,cap,evaluation_id=best
    lower=leaf_heap[0][0]
    gap=upper-lower
    require(-2e-13<=gap<=CERTIFICATE_TOL and lower==min(r["lower_bound"] for r in leaves),
            "The registered global gap has not been certified")
    cert={"pid":os.getpid(),"method":"binary best-lower-bound branch and bound","solve_count":1,"neural_fits":0,
          "search_interval_deg":[0.,upper_endpoint],"slope_comparison_epsilon_deg":SLOPE_EDGE_EPS,
          "objective_gap_target":CERTIFICATE_TOL,"upper_bound":upper,"lower_bound":lower,"gap":gap,
          "incumbent_cap_deg":cap,"incumbent_evaluation":evaluation_id,"coverage_no_gaps_or_overlaps":True,
          "evaluated_points":len(points),"binary_splits":splits,"terminal_leaf_count":len(leaves),
          "independent_risk_max_abs_gap":max_risk_reference_gap,"independent_bound_max_abs_gap":max_bound_reference_gap,
          "evaluations_sha256":sha(HERE/"solver-evaluations.jsonl"),"branches_sha256":sha(HERE/"solver-branches.jsonl"),
          "leaves":leaves}
    write(HERE/"certificate.json",cert)
    objective,groups,out,binding,record,angles=bank.evaluate(cap)
    require(objective==upper,"Locked cap score differs from certified incumbent")
    np.testing.assert_array_equal(out[~binding],bank.learned[~binding])
    post=separation_degrees(bank.control,out)
    require(np.max(np.abs(post-np.minimum(bank.disagreement,cap)))<=1e-9,"Locked cap violates the shortest spherical arc")
    np.savez_compressed(HERE/"training-io.npz",C037=out,cap_active=binding,C023_physical_disagreement_deg=bank.disagreement,
                        C037_physical_disagreement_deg=post,C037_angle_error_deg=angles,domain=bank.domain,
                        source_record_ordinal=bank.ordinal,record=bank.record,native_release=bank.raw["native_release"],
                        C037_record_RMSE_deg=record)
    receipt={"pid":os.getpid(),"solver_kind":"physical_angle_cap","solve_count":1,"neural_fits":0,"parameters":1,
             "method":"binary interval branch and bound","started_at":started_at,"completed_at":local_now(),
             "fit_seconds":perf_counter()-start,"result_success":True,"nit":splits,"nfev":len(points),
             "bounds_deg":[0.,upper_endpoint],"cap_deg":cap,"objective":objective,"upper_bound":upper,"lower_bound":lower,
             "gap":gap,"group_order":bank.groups,"groups":groups,"cap_active_fraction":float(binding.mean()),
             "cap_active_slots":int(binding.sum()),"uncapped_slots_exact":int((~binding).sum()),
             "certificate_sha256":sha(HERE/"certificate.json"),
             "training_record_RMSE_deg":[{"record_id":int(i),"source":str(bank.domain[bank.ordinal==i][0]),
                 "record":str(bank.record[bank.ordinal==i][0]),"slots":int(bank.count[i]),"C037_RMSE_deg":float(record[i])}
                 for i in range(len(bank.count))]}
    write(HERE/"solver-receipt.json",receipt)
    write(HERE/"coefficient-lock.json",{
        "pid":os.getpid(),"locked_at":local_now(),"cap_deg":cap,"objective":objective,"cap_bounds_deg":[0.,upper_endpoint],
        "group_order":bank.groups,"groups":groups,"cap_active_fraction":float(binding.mean()),
        "solve_count":1,"neural_fits":0,"choice":"certified_branch_and_bound_incumbent",
        "C023_checkpoint_sha256":protocol["inputs_sha256"]["C023/router.pt"],
        "physical_bank_sha256":protocol["inputs_sha256"]["review/external-physical-anchor-141.npz"],
        "raw_bank_sha256":protocol["inputs_sha256"]["review/raw-anchor-training-141.npz"],
        "protocol_sha256":sha(HERE/"protocol.json"),"solver_receipt_sha256":sha(HERE/"solver-receipt.json"),
        "evaluation_log_sha256":sha(HERE/"solver-evaluations.jsonl"),"training_io_sha256":sha(HERE/"training-io.npz"),
        "certificate_sha256":sha(HERE/"certificate.json")})
    print("C037 locked cap",cap,"objective",objective,"certificate gap",gap,flush=True)
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
    out["C036_deg"]=out["physical_only_deg"]
    dis=separation_degrees(out["physical_only_deg"],out["C023_deg"])
    out["C037_deg"],binding=capped(out["physical_only_deg"],out["C023_deg"],dis,cap)
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
            "C037_delta_deg": {m: macro["C037_deg"] - macro[m] for m in METHODS if m != "C037_deg"} if keep else {},
            "C037_wins": {m: sum(r["C037_deg"] < r[m] for r in keep) for m in METHODS if m != "C037_deg"}}


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
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C035_deg","physical_only_deg","C036_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C035 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C035_deg"][~active_scored], errs["C020_deg"][~active_scored])
                cap_scored=np.r_[False,binding][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C037_deg"][~cap_scored],errs["C023_deg"][~cap_scored])
                active_probs = {m:dict(zip(WEIGHTS, p[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                                for m,p in probabilities.items()}
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C035_deg", "physical_only_deg", "C036_deg"))
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
                print("development", *key, score["C023_deg"], score["C035_deg"], score["C037_deg"], flush=True)
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
    controls={r["record"]:r for r in read(ROOT/"C036/penn-training-source-scores.json")["records"]}
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
        for now,prior in (("C023_deg","C023_deg"),("C035_deg","C035_deg"),("C036_deg","C036_deg"),("full_vqf_deg","full_vqf_deg")):
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
            lock["training_io_sha256"]==sha(HERE/"training-io.npz") and lock["certificate_sha256"]==sha(HERE/"certificate.json"),"C037 coefficient lock or solver evidence changed")
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
    prior=read(ROOT/"C036/results.json")
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
    print("C037 locked; replaying common development panel",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,routers,cap)
    require(sha(path)==lock_hash and sha(HERE/"solver-receipt.json")==lock["solver_receipt_sha256"] and
            all(sha(ROOT/name)==h for name,h in protocol["inputs_sha256"].items()),"Frozen evidence changed during replay")
    result.update(pid=os.getpid(),cap_deg=cap,coefficient_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),penn_training_source_scores_sha256=sha(HERE/"penn-training-source-scores.json"),
                  solve_count=1,neural_fits=0,solver_receipt_sha256=lock["solver_receipt_sha256"],certificate_sha256=lock["certificate_sha256"],
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",scored=scored,coefficient_lock_sha256=lock_hash,results_sha256=sha(out))
    require(complete,"C037 replay incomplete; fitted coefficient retained")
    print("C037 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C037/run_physical_cap.py",
            "scope":"C037 single certified cap branch-and-bound solve, zero neural fits, common development replay and Penn A training audit",
            "protocol_sha256":sha(HERE/"protocol.json")}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()) and
                protocol["objective_gap_target"]==CERTIFICATE_TOL and protocol["slope_edge_epsilon_deg"]==SLOPE_EDGE_EPS and protocol["solve_count"]==1 and protocol["neural_fits"]==0,
                "C037 registered artifact or solver specification changed")
        torch.set_num_threads(2)
        status(launch,"verifying_C023_and_physical_source_bank")
        bank=FitBank(protocol)
        status(launch,"single_certified_scalar_solve")
        cap=solve(protocol,bank)
        replay_locked(protocol,launch,cap)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

