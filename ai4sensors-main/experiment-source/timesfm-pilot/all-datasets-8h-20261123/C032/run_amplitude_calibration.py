"""C032: calibrate one C030 amplitude against the frozen C023/C027 order variation."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"C030"))

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F

import run_anchored_router as c30

c26,c25,c23,c22,c18,c17=c30.c26,c30.c25,c30.c23,c30.c22,c30.c18,c30.c17
c3,c2,base,intrinsic=c30.c3,c30.c2,c30.base,c30.intrinsic
read,write,sha,stamp,array_sha=c30.read,c30.write,c30.sha,c30.stamp,c30.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c30.UP,c30.DATA,c30.TRANSFER,c30.MASKS,c30.GATES
WEIGHTS,NORM_KEYS,GEOMETRY_NAMES=c30.WEIGHTS,c30.NORM_KEYS,c30.GEOMETRY_NAMES
archive,require,weights_state=c30.archive,c30.require,c30.weights_state
ROUTERS=("C023","C030","C031","C032")
FROZEN_SCORES=("full_vqf_deg","C020_deg","C022_deg","C023_deg","C030_deg","C031_deg")
METHODS=(*FROZEN_SCORES,"C032_deg",*c23.FIXED)
BATCH_KEYS=("input_feature_sha256","native_release_sha256","native_input_last_sha256",
            "source_record_ordinal_sha256","legacy_slot_identity_sha256")


def local_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def status(launch,phase,**fields):
    write(HERE/"process.json",{"pid":os.getpid(),"status":phase,"started_at":launch["started_at"],
                              "updated_at":local_now(),"protocol_sha256":sha(HERE/"protocol.json"),**fields})


def checkpoint_pair():
    c023,c027,c030=(weights_state(n) for n in ("C023","C027","C030"))
    for key in ("feature_mean","feature_std","p0","log_p0"):
        torch.testing.assert_close(c023[key],c027[key],rtol=0.,atol=0.)
    for key in ("mean","std"):
        torch.testing.assert_close(c023["feature_"+key],c030["old_feature_"+key],rtol=0.,atol=0.)
    for key in ("p0","log_p0"):
        torch.testing.assert_close(c023[key],c030[key],rtol=0.,atol=0.)
    torch.testing.assert_close(c023["state_dict"]["weight"],c030["state_dict"]["weight"][:,:24],rtol=0.,atol=0.)
    torch.testing.assert_close(c023["state_dict"]["bias"],c030["state_dict"]["bias"],rtol=0.,atol=0.)
    torch.testing.assert_close(c030["state_dict"]["weight"][:,28:],torch.stack((c030["a"],-c030["a"])),rtol=0.,atol=0.)
    require(c030["state_dict"]["weight"][:,24:28].count_nonzero().item()==0 and tuple(c030["geometry_names"])==GEOMETRY_NAMES,
            "C030 packed physical correction must preserve the C023 anchor")
    np.testing.assert_array_equal(c030["geometry_mean"].numpy(),np.zeros(4))
    np.testing.assert_array_equal(c030["geometry_std"].numpy(),np.ones(4))
    return c023,c027,c030


def verify_common_slots(original,source,x):
    z24=x["z"].numpy()
    expected_order=read(ROOT/"C027/sampling-lock.json")["permutation"]
    np.testing.assert_array_equal(np.sort(expected_order),np.arange(160))
    prior={}
    for name in ("C023","C027"):
        state=archive(ROOT/name/"training-io.npz")
        np.testing.assert_array_equal(z24,state["normalized_input"])
        np.testing.assert_array_equal(source["enabled"],state["enabled"])
        np.testing.assert_array_equal(source["native_release"],state["native_release"])
        ordinal_key="record_ordinal" if name=="C023" else "source_record_ordinal"
        np.testing.assert_array_equal(source["source_record_ordinal"],state[ordinal_key])
        if name=="C027":
            np.testing.assert_array_equal(source["native_input_last"],state["native_input_last"])
        with (ROOT/name/"optimizer-steps.jsonl").open(encoding="utf-8") as handle:
            prior[name]=[json.loads(line) for line in handle]
    require(len(prior["C023"])==len(prior["C027"])==160,"Missing frozen batch evidence")
    for j,row in enumerate(prior["C023"]):
        cut=slice(96*j,96*(j+1))
        fingerprints={
            "input_feature_sha256": array_sha(z24[cut],"<f8"),
            "native_release_sha256":array_sha(source["native_release"][cut],"<i8"),
            "native_input_last_sha256":array_sha(source["native_input_last"][cut],"<i8"),
            "source_record_ordinal_sha256":array_sha(source["source_record_ordinal"][cut],"<i8"),
            "legacy_slot_identity_sha256":array_sha(original["global_window"][cut],"<i8"),
        }
        require(row["step"]==j+1 and all(row[k]==h for k,h in fingerprints.items()),"C023 slot fingerprints differ")
    for j,row in enumerate(prior["C027"]):
        i=row["source_batch_index"]
        require(row["step"]==j+1 and i==expected_order[j] and
                all(row[k]==prior["C023"][i][k] for k in BATCH_KEYS),"C027 slots differ from C023 under its permutation")
    with np.load(ROOT/"C026/training-inputs.npz",allow_pickle=False) as mixed:
        for key in ("native_release","native_input_last","enabled","source_record_ordinal","domain","fast","slow","truth"):
            blocks=mixed[key][:,:96]
            np.testing.assert_array_equal(blocks.reshape((-1,)+blocks.shape[2:]),source[key])
        z28=mixed["features"][:,:96,:].reshape(-1,28).copy()
    np.testing.assert_array_equal(z28[:,:24],z24)
    angle,phi=c30.geometry_basis(source["fast"],source["slow"],z28)
    with np.load(ROOT/"C030/anchored-inputs.npz",allow_pickle=False) as parent:
        np.testing.assert_array_equal(phi,parent["physical_basis"].reshape(160,110,4)[:,:96].reshape(-1,4))
        np.testing.assert_array_equal(angle,parent["fast_slow_angle_deg"].reshape(160,110)[:,:96].ravel())
    return z28,angle,phi


def group_weights(source):
    names=source["domain"]
    ordinal=source["source_record_ordinal"]
    target=source["truth"]
    valid=(source["intrinsic_valid"] & np.isfinite(target).all(axis=1) &
           (np.abs(np.linalg.norm(target,axis=1)-1.)<=2e-12))
    enabled=source["enabled"]
    use=valid & enabled
    w=np.zeros(len(use),dtype=np.float64)
    details,rows={},[]
    require(len(use)==15360 and np.unique(ordinal).size==139 and set(names)==set(c23.DOMAINS),
            "C032 requires the common 139 recordings and four original sources")
    for d in c23.DOMAINS:
        domain=names==d
        all_ids=np.unique(ordinal[domain])
        eligible_ids=np.unique(ordinal[domain & use])
        require(len(eligible_ids)>0,f"No eligible original records in {d}")
        for i in all_ids:
            rec=domain & (ordinal==i)
            included=rec & use
            n=int(included.sum())
            if n:
                w[included]=1./(4*len(eligible_ids)*n)
            rows.append({"domain":d,"record_ordinal":int(i),"record":str(source["source_record_name"][rec][0]),
                         "total_slots":int(rec.sum()),"valid_target_slots":int((rec & valid).sum()),
                         "eligible_slots":n,"invalid_target_slots":int((rec & ~valid).sum()),
                         "valid_target_gate_off_slots":int((rec & valid & ~enabled).sum()),"weight_mass":float(w[rec].sum())})
        details[d]={"records":len(all_ids),"eligible_records":len(eligible_ids),
                    "empty_record_ordinals":[int(i) for i in np.setdiff1d(all_ids,eligible_ids)],
                    "total_slots":int(domain.sum()),"eligible_slots":int((domain & use).sum()),
                    "invalid_target_slots":int((domain & ~valid).sum()),"gate_off_slots":int((domain & valid & ~enabled).sum()),
                    "weight_mass":float(w[domain].sum())}
        require(abs(details[d]["weight_mass"]-.25)<=2e-15,f"Unequal C032 mass for {d}")
    require(abs(w.sum()-1.)<=2e-15 and np.array_equal(w>0.,use),"C032 support and weights disagree")
    return w,use,valid,details,rows


@torch.inference_mode()
def parent_logits(features,state):
    return (F.linear(features,state["state_dict"]["weight"],state["state_dict"]["bias"])+state["log_p0"][:2]).numpy()


def calibrate(protocol,anchor,c027,c030):
    c022=read(ROOT/"C022/coefficient-lock.json")
    require(c022["k"]==protocol["C022_k"],"The C022 prior changed")
    torch.testing.assert_close(anchor["p0"],c23.prior_probs(c022["k"]),rtol=0.,atol=0.)
    original,source,x=c23.load_train(c022,protocol["inputs_sha256"],anchor["feature_mean"].numpy(),anchor["feature_std"].numpy())
    z28,angle,phi=verify_common_slots(original,source,x)
    w,use,valid,details,records=group_weights(source)
    logits23=parent_logits(x["z"],anchor)
    logits27=parent_logits(x["z"],c027)
    order=(logits23[:,0]-logits23[:,1])-(logits27[:,0]-logits27[:,1])
    a=c030["a"].numpy()
    physical=2.*(phi@a)
    require(np.isfinite(order[use]).all() and np.isfinite(physical[use]).all(),"Nonfinite eligible logit difference")
    order_rms=float(np.sqrt(w[use] @ order[use]**2))
    physical_rms=float(np.sqrt(w[use] @ physical[use]**2))
    gamma=min(1.,order_rms/physical_rms) if physical_rms>0. else 1.
    deployed=gamma*a
    require(np.isfinite(gamma) and 0.<=gamma<=1.,"Invalid gamma")
    for d in c23.DOMAINS:
        support=(source["domain"]==d)&use
        details[d].update(order_squared_contribution=float(w[support]@order[support]**2),
                          physical_squared_contribution=float(w[support]@physical[support]**2))
    evidence=HERE/"calibration-slots.npz"
    np.savez_compressed(evidence,slot_index=np.arange(15360,dtype=np.int64),
                        source_batch_index=np.repeat(np.arange(160),96),within_batch_index=np.tile(np.arange(96),160),
                        legacy_global_window=original["global_window"],
                        **{k:source[k] for k in ("source_record_ordinal","source_record_name","domain","native_release",
                                               "native_input_last","enabled","intrinsic_valid")},
                        valid_target=valid,eligible=use,weight=w,features28=z28,fast_slow_angle_deg=angle,physical_basis=phi,
                        C023_fast_slow_logits=logits23,C027_fast_slow_logits=logits27,
                        C023_minus_C027_log_odds=order,C030_minus_C023_log_odds=physical)
    equivalent=np.array_equal(deployed,a)
    write(HERE/"calibration.json",{"pid":os.getpid(),"calibrated_at":local_now(),"gamma":gamma,
          "RMS_order_log_odds":order_rms,"RMS_unscaled_C030_correction_log_odds":physical_rms,
          "RMS_scaled_correction_log_odds":gamma*physical_rms,"C030_a":a.tolist(),"C032_a":deployed.tolist(),
          "equivalent_to_C030":equivalent,"classification":"C030 replay" if equivalent else "one externally calibrated scalar",
          "zero_denominator_fallback":physical_rms==0.,"eligible_slots":int(use.sum()),
          "invalid_target_slots":int((~valid).sum()),"valid_target_gate_off_slots":int((valid & ~source["enabled"]).sum()),
          "eligible_records":int(np.unique(source["source_record_ordinal"][use]).size),"original_records":139,"original_slots":15360,
          "sources":details,"records":records,"calibration_slots_sha256":sha(evidence),
          "parent_hashes":{name:protocol["inputs_sha256"][name+"/router.pt"] for name in ("C023","C027","C030","C031")},
          "optimizer_steps":0,"new_training_records":0})
    state=c30.packed_state(anchor,c030,deployed)
    state.pop("solve_count")
    state.update(protocol_sha256=sha(HERE/"protocol.json"),gamma=gamma,fit_count=0,optimizer_steps=0,scalar_calibrations=1,equivalent_to_C030=equivalent,
                 calibration_sha256=sha(HERE/"calibration.json"),baseline_C030_checkpoint_sha256=sha(ROOT/"C030/router.pt"))
    torch.save(state,HERE/"router.pt")
    return gamma, equivalent




@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,models,feature_norm,ctx_norm,a30,a32,log0):
    original,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    active=valid&~rest
    inherited,bmean,bstd,*_=fixed
    feat=c18.features(history,q[at],inherited[0],bmean,bstd)
    np.testing.assert_array_equal(feat.valid,valid)
    mean,std=feature_norm
    z24=(feat.pooled[:,1].astype(np.float64)-mean)/std
    context=c25.from_frontend(raw,observed,acc,dt)[at]
    z28=np.column_stack((z24,(context-ctx_norm[0])/ctx_norm[1]))
    angle,phi=c30.geometry_basis(fast,slow,z28)
    x=torch.from_numpy(z24)
    logits=torch.cat((models["C023"](x)+log0[:2],log0[2].expand(len(x),1)),dim=1)
    probabilities={"C023":torch.softmax(logits,dim=1).numpy(),"C031":c23.weights(models["C031"],x,log0).numpy()}
    packed_x=torch.from_numpy(np.column_stack((z28,phi)))
    packed_gap=0.
    for name,a in (("C030",a30),("C032",a32)):
        correction=torch.from_numpy(phi@a)
        corrected=logits.clone()
        corrected[:,0]+=correction
        corrected[:,1]-=correction
        probabilities[name]=torch.softmax(corrected,dim=1).numpy()
        packed=c23.weights(models[name],packed_x,log0).numpy()
        packed_gap=max(packed_gap,float(np.max(np.abs(probabilities[name]-packed))))
    require(packed_gap<=2e-13,"Packed model changes the explicit anchored probabilities beyond roundoff")
    directions={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(active[:,None],mix,0.))}
    for name,p in probabilities.items():
        c26.require_simplex(p)
        correction=np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        require(np.max(np.linalg.norm(correction,axis=1))<=c22.CAP+2e-15 and np.all(correction[:,2]==0.),
                "Router correction escaped the capped horizontal responses")
        directions[name+"_deg"]=c23.augment(original,center,correction)
    return directions,probabilities,active,rest,context,angle,phi,phi@a32,packed_gap



def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ROUTERS} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C032_delta_deg":{k:macro["C032_deg"]-macro[k] for k in METHODS if k!="C032_deg"} if good else {},
            "C032_wins":{k:sum(r["C032_deg"]<r[k] for r in good) for k in METHODS if k!="C032_deg"}}


def replay(previous,control,inventory,campaign,split,cfg,k,fixed,models,norm,ctx_norm,a30,a32,log0):
    c1r,c1p,c3r,c2p=(read(ROOT/n) for n in ("C001/results.json","C001/protocol.json","C003/results.json","C002/protocol.json"))
    evidence=c2p["frozen_evidence"]
    map1,map3=({(r["dataset"],r["record"]):r for r in s["sequences"]} for s in (c1r,c3r))
    keys={(r["dataset"],r["record"]) for r in previous["sequences"]}
    spec={(r["dataset"],r["record"]):r for r in inventory["records"] if (r["dataset"],r["record"]) in keys}
    table={(r["dataset"],r["record"]):r for r in control["records"]}
    base.require(len(keys)==213 and set(spec)==keys==set(table)
                 and all(r["side"] in ("development","development_previously_used_for_selection") for r in spec.values())
                 and control["source_csv_sha256"]==sha(ROOT/"C020/sequences.csv") and control["selection_lock_sha256"] is None,
                 "C032 replay or fixed controls leave development scope")
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
                             and array_sha(release,"<i8")==old["native_release_sha256"],"C030 or fixed native indices changed")
                run=perf_counter()
                dirs,probs,active,rest,context,angle,phi,correction,packed_gap=held(q,at,release,w,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,a30,a32,log0)
                calc=perf_counter()-run
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(active.sum()==old["active_release_count"] and
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ROUTERS[:-1]),
                             "Archived C023, C030 or C031 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C032_deg"][same],errs["C020_deg"][same])
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
                            "C032_active_physical_basis_sha256":array_sha(phi[active],"<f8"),
                            "C032_active_fast_slow_angle_mean_deg":float(angle[active].mean()) if active.any() else None,
                            "C032_packed_probability_max_abs_difference":packed_gap,
                            "prediction_seconds":calc,"seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i),"release":release[active],"input_last":at[active],
                               "context":context[active],"fast_slow_angle_deg":angle[active],"physical_basis":phi[active],
                               "antisymmetric_logit_shift":correction[active],**{m:p[active] for m,p in probs.items()}})
                poses.extend(detail)
                print("development",*key,score["C023_deg"],score["C030_deg"],score["C031_deg"],score["C032_deg"],flush=True)
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















def merge_parent_panel():
    previous, other=(read(ROOT/name/"results.json") for name in ("C030","C031"))
    require(previous["scoring_masks_sha256"]==other["scoring_masks_sha256"]==sha(MASKS),"C030/C031 scoring masks differ")
    for a,b in zip(previous["sequences"],other["sequences"],strict=True):
        require(all(a[k]==b[k] for k in ("dataset","record","scoring_indices_sha256","scored_frames",
                                        "native_input_last_sha256","native_release_sha256","active_release_count",
                                        "C023_active_routing_sha256","C023_deg")),"C031 does not match the common panel")
        a.update(C031_deg=b["C031_deg"],C031_active_routing_sha256=b["C031_active_routing_sha256"])
    for a,b in zip(previous["array_sensor_poses"],other["array_sensor_poses"],strict=True):
        require(all(a[k]==b[k] for k in ("record","pose","scoring_indices_sha256","scored_frames")),"C031 array poses changed")
        a["C031_deg"]=b["C031_deg"]
    return previous


def replay_locked(protocol,launch):
    lock=read(HERE/"router-lock.json")
    lock_hash=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["scalar_calibrations"]==1 and lock["optimizer_steps"]==0 and
            lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/n)==d for n,d in lock["artifact_sha256"].items()),"The same worker must replay the original locked C032")
    states={name:weights_state(name) for name in ROUTERS}
    anchor,base30,own=(states[n] for n in ("C023","C030","C032"))
    models={}
    for name,width in zip(ROUTERS,(24,32,24,32),strict=True):
        state=states[name]
        model=nn.Linear(width,2,dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name]=model.eval().requires_grad_(False)
        for k in ("p0","log_p0"):
            torch.testing.assert_close(state[k],anchor[k],rtol=0.,atol=0.)
    for key in NORM_KEYS:
        torch.testing.assert_close(own[key],base30[key],rtol=0.,atol=0.)
    for key in ("mean","std"):
        torch.testing.assert_close(own["old_feature_"+key],anchor["feature_"+key],rtol=0.,atol=0.)
        torch.testing.assert_close(own["old_feature_"+key],states["C031"]["feature_"+key],rtol=0.,atol=0.)
    torch.testing.assert_close(own["state_dict"]["weight"][:,:24],anchor["state_dict"]["weight"],rtol=0.,atol=0.)
    torch.testing.assert_close(own["state_dict"]["bias"],anchor["state_dict"]["bias"],rtol=0.,atol=0.)
    torch.testing.assert_close(own["a"],own["gamma"]*base30["a"],rtol=0.,atol=0.)
    require(own["protocol_sha256"]==sha(HERE/"protocol.json") and own["calibration_sha256"]==sha(HERE/"calibration.json"),
            "C032 frozen protocol/calibration does not match checkpoint")
    mean,std,ctx_mean,ctx_std=(own[k].numpy() for k in NORM_KEYS)
    a30,a32=(state["a"].numpy() for state in (base30,own))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    previous=merge_parent_panel()
    require([[r["dataset"],r["record"]] for r in previous["sequences"]]==protocol["development_records"],
            "Development replay order changed")
    status(launch,"development_replay",gamma=own["gamma"],equivalent_to_C030=own["equivalent_to_C030"],
           router_lock_sha256=lock_hash,expected_records=213)
    print("C032 locked; replaying common development panel",own["gamma"],flush=True)
    start=perf_counter()
    result=replay(previous,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,models,(mean,std),
                  (ctx_mean,ctx_std),a30,a32,own["log_p0"])
    require(sha(HERE/"router-lock.json")==lock_hash and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"Frozen evidence changed")
    if own["equivalent_to_C030"]:
        require(all(r["C032_deg"]==r["C030_deg"] for r in result["sequences"] if r["status"]=="scored"),
                "C032 equivalent replay differs from C030")
    result.update(pid=os.getpid(),router_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),gamma=own["gamma"],equivalent_to_C030=own["equivalent_to_C030"],
                  calibration_sha256=own["calibration_sha256"],fit_count=0,optimizer_steps=0,new_training_records=0,
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",gamma=own["gamma"],equivalent_to_C030=own["equivalent_to_C030"],
           scored=scored,router_lock_sha256=lock_hash,results_sha256=sha(out))
    require(complete,"C032 replay incomplete; locked checkpoint retained")
    print("C032 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C032/run_amplitude_calibration.py",
            "protocol_sha256":sha(HERE/"protocol.json"),"scope":"C032 single scalar calibration, zero neural updates, common development replay"}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"Registered C032 inputs or script changed")
        torch.set_num_threads(2)
        status(launch,"calibrating")
        gamma,equivalent=calibrate(protocol,*checkpoint_pair())
        artifacts=("protocol.json","calibration.json","calibration-slots.npz")
        with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
            json.dump({"pid":os.getpid(),"locked_at":local_now(),"gamma":gamma,"equivalent_to_C030":equivalent,
                       "scalar_calibrations":1,"fit_count":0,"optimizer_steps":0,"new_training_records":0,
                       "checkpoint_sha256":sha(HERE/"router.pt"),
                       "artifact_sha256":{n:sha(HERE/n) for n in artifacts}},handle,indent=2)
            handle.write("\n")
        replay_locked(protocol,launch)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

