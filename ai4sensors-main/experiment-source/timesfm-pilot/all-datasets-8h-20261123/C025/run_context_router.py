"""C025: same C024 training doses with four causal physical observers appended."""
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"C024"),str(ROOT/"review")]

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn
import run_davis_router as c24
from router_motion_context import NAMES as OBSERVERS, series

c23,c22,c20,c18,c17,c14,c3,c2=c24.c23,c24.c22,c24.c20,c24.c18,c24.c17,c24.c14,c24.c3,c24.c2
base,intrinsic=c24.base,c24.intrinsic
read,write,sha,stamp,array_sha=c24.read,c24.write,c24.sha,c24.stamp,c24.array_sha
UP,DATA,PILOT,TRANSFER,MASKS,GATES=c24.UP,c24.DATA,c24.PILOT,c24.TRANSFER,c24.MASKS,c24.GATES
DOMAINS,NAMES,STEPS,BATCH,OLD=c24.DOMAINS,c24.NAMES,c24.STEPS,c24.BATCH,c24.OLD_BATCH
LR,WD,CLIP,WEIGHTS,PROB_TOL=c24.LR,c24.WD,c24.CLIP,c24.WEIGHTS,c24.PROB_TOL
METHODS=(*c24.METHODS[:5],"C025_deg",*c24.METHODS[5:])


def corrected(raw, observed, dt, cfg):
    bias, active, _, _ = c3.estimate(raw, observed, dt, cfg)
    filled = base.causal_fill(raw, observed)
    if active.any():
        filled = filled.copy()
        filled[:,:3] -= bias
    return np.ascontiguousarray(filled,dtype=np.float64)


def from_frontend(raw,observed,acc,dt):
    data=np.column_stack((acc,base.causal_fill(raw,observed)[:,3:]))
    return series(data,1./dt)


def inspect_native(name,raw,observed,dt,cfg,indices,cached):
    all_features=series(corrected(raw,observed,dt,cfg),1./dt)
    rebuilt=all_features[indices]
    max_cache=float(np.max(np.abs(rebuilt-cached)))
    base.require(max_cache<=2e-8,f"Archived motion context differs from original raw data at {name}: {max_cache}")
    j=int(indices[len(indices)//2])
    changed=raw.copy()
    changed[j+1:,:3]+=np.array([13.,-9.,7.])
    changed[j+1:,3:]+=1.1
    masks=observed.copy()
    masks[j+2::2]=False
    prefix=series(corrected(raw[:j+1],observed[:j+1],dt,cfg),1./dt)
    future=series(corrected(changed,masks,dt,cfg),1./dt)[:j+1]
    pref_error=float(np.max(np.abs(prefix-all_features[:j+1])))
    future_error=float(np.max(np.abs(future-all_features[:j+1])))
    base.require(max(pref_error,future_error)<=2e-10,"Observer failed native prefix or future-packet invariance")
    r=Rotation.from_rotvec([0.,0.,np.pi/2.])
    rotated=r.apply(raw[:j+1].reshape(-1,3)).reshape(j+1,6)
    g=series(corrected(rotated,observed[:j+1],dt,cfg),1./dt)
    # Yaw signed-axis change checks the production VQF/bias/observer chain, no reference label used.
    spin=np.max(np.abs(prefix-g),axis=0)
    base.require(np.isfinite(spin).all(), "Observer rotation produced nonfinite values")
    return {"record":name,"audited_slots":len(indices),"native_prefix_cut_inclusive":j,"cache_max_abs":max_cache,
            "direct_prefix_max_abs":pref_error,"mutated_future_max_abs":future_error,
            "yaw_90_reinitialized_chain_max_abs_by_feature":dict(zip(OBSERVERS,spin.tolist(),strict=True))}


def sample_audit(data, cached_old, cache_new, cfg):
    p15=read(ROOT/"C015/protocol.json")
    p15prep=read(ROOT/"C015/training-preparation.json")
    p19=read(ROOT/"C019/training-preparation.json")
    inventory=read(ROOT/"C001/inventory.json")
    old_sources, *_ = c22.c19.frozen_panel(
       inventory,read(ROOT/"review/yt_training_panel.json"),p15)
    names=p19["source_records"]
    allowed=("BROAD","Sassari","YT_Server","RIDI")
    old_id=data["source_record_ordinal"][:,:OLD].ravel()
    old_domains=data["domain"][:,:OLD].ravel()
    at=data["native_input_last"][:,:OLD].ravel()
    proof=[]
    with c2.c1.h5py.File(inventory["yt"]["path"],"r") as yt:
        for d in allowed:
            rec=int(old_id[np.flatnonzero(old_domains==d)[0]])
            base.require(rec<len(old_sources),"Spot audit outside original external source cohort")
            spec=old_sources[rec]
            if d=="RIDI":
                raw,obs,dt,_,_,hashes=c17.trained_phone(spec,p15prep["ridi_eligibility"][rec-63])
            else:
                raw,obs,dt,_,_,hashes=c22.c4.train_raw(spec,yt["train"],{})
            use=old_id==rec
            row=inspect_native(names[rec],raw,obs,dt,cfg,at[use],cached_old[use])
            proof.append({**row,"domain":d,"source_sha256":hashes})
    prep=read(ROOT/"C024/training-preparation.json")
    davis_plan=read(c24.DAVIS/"preprocessing-lock.json")
    index={r["record"]:r for r in davis_plan["source_records"]}
    base.require(set(index)==set(NAMES) and davis_plan["unseen_numerical_records"]==3,"DAVIS train-only set changed")
    for j,name in enumerate(NAMES):
        path=(ROOT/index[name]["prepared"]).resolve()
        base.require(path==(c24.DAVIS/"prepared-train"/(name+".npz")).resolve()
                     and sha(path)==index[name]["sha256"]==prep["rows"][j]["prepared_npz_sha256"],"Prepared DAVIS source not the archived train-only file")
        with np.load(path,allow_pickle=False) as a:
            raw=a["six_axis"]; hz=float(a["native_hz"])
        indices=data["native_input_last"][:,OLD:]
        use=data["source_record_ordinal"][:,OLD:]==139+j
        row=inspect_native(name,raw,np.isfinite(raw).all(1),1./hz,cfg,indices[use],cache_new[use])
        proof.append({**row,"domain":"DAVIS","prepared_sha256":sha(path),"raw_member_sha256":prep["rows"][j]["raw_member_sha256"]})
    return proof


def prepare(protocol):
    a=ROOT/"C024/training-inputs.npz"
    with np.load(a,allow_pickle=False) as f:
        data={k:f[k] for k in f.files}
    with np.load(ROOT/"C024/training-io.npz",allow_pickle=False) as f:
        c024_io={k:f[k] for k in f.files}
    base.require(data["z"].shape==(160,104,24) and len(np.unique(data["source_record_ordinal"]))==145 and
                 c024_io["effective_loss_weights"].shape==(16640,), "C024 batch or record count changed")
    for k in ("native_release","source_record_ordinal","enabled"):
        np.testing.assert_array_equal(data[k].ravel(),c024_io[k])
    np.testing.assert_array_equal(data["old_slot"],np.broadcast_to(np.arange(BATCH)<OLD,(STEPS,BATCH)))
    old_path=ROOT/"review/source_motion_router_features.npz"
    diagnostic=read(ROOT/"review/source_motion_router_diagnostic.json")
    base.require(tuple(diagnostic["feature_names"])==OBSERVERS and diagnostic["code_sha256"]==sha(ROOT/"review/router_motion_context.py")
                 and diagnostic["native_features_sha256"]==sha(old_path) and set(diagnostic["davis"])==set(NAMES),
                 "Archived feature semantics changed")
    with np.load(old_path,allow_pickle=False) as a:
        for k in ("native_release","native_input_last","source_record_ordinal"):
            np.testing.assert_array_equal(a[k],data[k][:,:OLD].ravel())
        assert tuple(a["feature_names"])==OBSERVERS
        cached_old=a["features"]
    base.require(cached_old.shape==(15360,4) and cached_old.dtype==np.float64,"Invalid four original context columns")
    cached_new=np.empty((STEPS,8,4),dtype=np.float64)
    for i,name in enumerate(NAMES):
        entry=diagnostic["davis"][name]
        p=(ROOT/entry["file"]).resolve()
        base.require(p==(ROOT/"review/davis_source_motion_context"/(name+".npy")).resolve()
                     and sha(p)==entry["sha256"],"DAVIS motion cache path/hash differs")
        v=np.load(p,mmap_mode="r",allow_pickle=False)
        base.require(v.shape==(entry["native_rows"],4) and v.dtype==np.float64,"Bad DAVIS motion cache shape")
        take=data["source_record_ordinal"][:,OLD:]==139+i
        ids=data["native_input_last"][:,OLD:][take]
        base.require(len(ids) in (213,214) and ids.min()>=0 and ids.max()<len(v),"DAVIS motion cache indices outside record")
        cached_new[take]=v[ids]
    context=np.concatenate((cached_old.reshape(STEPS,OLD,4),cached_new),axis=1)
    base.require(np.isfinite(context).all() and np.all(context>=0.),"Nonfinite or negative physical observer inputs")
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    proof=sample_audit(data,cached_old,cached_new,cfg)
    weights=data["record_equal_weights"].reshape(-1)
    domains=data["domain"].ravel()
    source_sums={d:float(weights[domains==d].sum()) for d in DOMAINS}
    base.require(all(abs(v-1.)<=2e-12 for v in source_sums.values()),"Weighted context moments use unequal source measure")
    p=weights/len(DOMAINS)
    v=context.reshape(-1,4)
    mean=p@v
    sd=np.sqrt(p@np.square(v-mean))
    base.require(np.all(sd>0.) and np.isfinite(sd).all(),"A new feature has no finite variation")
    normalized=(context-mean)/sd
    features=np.concatenate((data["z"],normalized),axis=-1)
    np.testing.assert_array_equal(features[:,:,:24],data["z"])
    with (ROOT/"C024/optimizer-steps.jsonl").open() as f: old_steps=[json.loads(s) for s in f]
    base.require(len(old_steps)==160,"C024 step evidence unavailable")
    for j,r in enumerate(old_steps):
        assert r["step"]==j+1
        for side,selector in (("original",slice(0,OLD)),("new",slice(OLD,None))):
            base.require(array_sha(data["native_release"][j,selector],"<i8")==r[side+"_release_sha256"],"C025 native step order changed")
    target=HERE/"training-context.npz"
    np.savez_compressed(target,features=features,context_raw=context,new_feature_mean=mean,new_feature_std=sd,
                        native_release=data["native_release"],native_input_last=data["native_input_last"],
                        source_record_ordinal=data["source_record_ordinal"],domain=data["domain"],record_equal_weights=data["record_equal_weights"])
    write(HERE/"context-normalization.json",{"names":list(OBSERVERS),"weighting":"Five sources equally weighted, each record equal within source, slots equal within record, include all 16640 slots.",
                                             "population_mean":mean.tolist(),"population_std":sd.tolist(),"source_weight_sums_before_dividing_by_5":source_sums,
                                             "old_24_preserved_bitwise":True,"old_24_sha256":array_sha(data["z"],"<f8"),
                                             "training_context_sha256":sha(target)})
    write(HERE/"source-context-audit.json",{"records":proof,"cache_file_sha256":sha(old_path),"poster_numeric_file_reads":0,
                                            "all_16640_native_slots_aligned":True,"training_context_sha256":sha(target)})
    return data,c024_io,features,mean,sd

def fit(data, archived, features, mean, std, protocol):
    tick=perf_counter()
    model=nn.Linear(28,2,dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    tensor={k:torch.from_numpy(data[k].reshape((-1,)+data[k].shape[2:])) for k in ("center","fast","slow","truth","enabled")}
    x=torch.from_numpy(features.reshape(-1,28))
    p0=torch.from_numpy(data["p0"])
    log0=p0.log()
    w=torch.from_numpy(archived["effective_loss_weights"])
    domain=data["domain"].ravel()
    prior_receipt=read(ROOT/"C024/zero-step-audit.json")
    denom=prior_receipt["C022_denominator_by_domain"]
    expected_weights=data["record_equal_weights"].ravel()/np.asarray([denom[d] for d in domain])
    np.testing.assert_array_equal(expected_weights,w.numpy())
    base.require(prior_receipt["loss_multiplier"]==32. and sum(t.numel() for t in model.parameters())==58,"C025 changed fixed optimizer recipe")
    def forward(use):
        y,delta,p=c23.position(model,x[use],log0,tensor["center"][use],tensor["fast"][use],tensor["slow"][use],tensor["enabled"][use])
        return (y-tensor["truth"][use]).square().sum(-1),y,delta,p

    with torch.inference_mode():
        initial, zero_direction, zero_delta, prior=forward(slice(None))
        last_max=float(np.max(np.abs(initial.numpy()-archived["baseline_chord"])))
        base.require(last_max <=3e-17 and torch.count_nonzero(zero_delta[~tensor["enabled"]]).item()==0
                     and torch.max((prior-p0).abs()).item()<=PROB_TOL,"Initial model differs from original C022")
    pre_path=HERE/"zero-step-audit.json"
    write(pre_path, {**prior_receipt,"frozen_C024_baseline_chord_max_abs":last_max,"all_16640_inactive_match_C020":True,
                      "effective_slot_loss_weights_sha256":array_sha(w.numpy(),"<f8")})
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD,foreach=False)
    prior_log=ROOT/"C024/optimizer-steps.jsonl"
    with prior_log.open(encoding="utf8") as handle: old_steps=[json.loads(l) for l in handle]
    log_path=HERE/"optimizer-steps.jsonl"
    with log_path.open("x",encoding="utf8") as out:
        for j in range(STEPS):
            batch=slice(j*BATCH,(j+1)*BATCH)
            optimizer.zero_grad(set_to_none=True)
            prev=c18.flat_params(model)
            e,_,_,p=forward(batch)
            weighted=32.*w[batch]*e
            value=weighted.sum()
            if j==0:
                base.require(abs(float(value.detach())-old_steps[0]["objective"])<=2e-15,"Old C024 first optimizer loss differs")
            value.backward()
            grad=c18.flat_params(model,use_grad=True)
            mag=float(torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP))
            clipped=c18.flat_params(model,use_grad=True)
            optimizer.step()
            updated=c18.flat_params(model)
            base.require(np.isfinite(np.r_[float(value.detach()),mag,grad,updated]).all(),"Invalid C025 optimizer evolution")
            receipts={"original_release_sha256":array_sha(data["native_release"][j,:OLD],"<i8"),
                      "new_release_sha256":array_sha(data["native_release"][j,OLD:],"<i8")}
            base.require(all(old_steps[j][k]==v for k,v in receipts.items()) and old_steps[j]["step"]==j+1,"C025 altered batch membership")
            row={"step":j+1,"pid":os.getpid(),"objective":float(value.detach()),"gradient_l2":mag,
                 "gradient_before_clip":grad.tolist(),"gradient_after_clip":clipped.tolist(),
                 "weights_before":prev.tolist(),"weights_after":updated.tolist(),"actual_update":(updated-prev).tolist(),
                 "domain_contribution":{d:float(weighted[domain[batch]==d].sum().detach()) for d in DOMAINS},
                 "source_record_ordinal_sha256":array_sha(data["source_record_ordinal"][j],"<i8"),
                 "native_input_last_sha256":array_sha(data["native_input_last"][j],"<i8"),
                 "normalized_features_sha256":array_sha(features[j],"<f8"),"routing_probability_sha256":array_sha(p.detach().numpy(),"<f8"),
                 **receipts}
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
            if j+1 in (1,40,80,120,160):
                print("C025 fit",j+1,row["objective"],mag,flush=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        final,pred,delta,p=forward(slice(None))
        prob=p.numpy()
        base.require(np.isfinite(prob).all() and np.all(prob>=0.) and np.max(np.abs(prob.sum(1)-1.))<=PROB_TOL,"Invalid final simplex weights")
        torch.testing.assert_close(pred[~tensor["enabled"]],c18.unit(tensor["center"])[~tensor["enabled"]],rtol=0.,atol=0.)
        base.require(torch.count_nonzero(delta[~tensor["enabled"]]).item()==0,"C025 bypassed original gate")
    train_path=HERE/"training-io.npz"
    np.savez_compressed(train_path,C024_prob=archived["trained_prob"],C025_prob=prob,C022_chord=initial.numpy(),
                        C024_chord=archived["trained_chord"],C025_chord=final.numpy(),active=tensor["enabled"].numpy(),
                        native_release=data["native_release"],native_input_last=data["native_input_last"],
                        source_record_ordinal=data["source_record_ordinal"],p0=p0.numpy())
    saved=HERE/"router.pt"
    prior24=torch.load(ROOT/"C024/router.pt",map_location="cpu",weights_only=True)
    torch.save({"state_dict":{k:p.detach().cpu() for k,p in model.state_dict().items()},
                "p0":p0,"log_p0":log0,"old_feature_mean":prior24["feature_mean"],"old_feature_std":prior24["feature_std"],
                "context_mean":torch.from_numpy(mean),"context_std":torch.from_numpy(std),
                "feature_names":list(OBSERVERS),"steps":160,"dtype":"float64","protocol_sha256":sha(HERE/"protocol.json"),
                "C024_sampling_lock_sha256":protocol["input_file_sha256"]["C024/sampling-lock.json"]},saved)
    w0=data["record_equal_weights"].ravel()
    enabled=tensor["enabled"].numpy()
    summary={}
    for d in DOMAINS:
        use=domain==d
        pre=float(w0[use]@initial.numpy()[use])
        a24=float(w0[use]@archived["trained_chord"][use]); a25=float(w0[use]@final.numpy()[use])
        summary[d]={"slots":int(use.sum()),"active_slots":int((use&enabled).sum()),"C022_chord":pre,"C024_chord":a24,"C025_chord":a25,
                    "C024_relative_error":a24/denom[d],"C025_relative_error":a25/denom[d],
                    "p_C024":c24.quantiles(archived["trained_prob"][use&enabled]),"p_C025":c24.quantiles(prob[use&enabled])}
    return {"pid":os.getpid(),"fit_count":1,"steps":160,"parameters":58,"seconds":perf_counter()-tick,
            "probability_sum_max_abs_error":float(np.max(np.abs(prob.sum(1)-1.))),"probability_sum_tolerance":PROB_TOL,
            "domains":summary,"artifacts_sha256":{p.name:sha(p) for p in (saved,train_path,log_path,pre_path)}}

@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,models,feature_norm,ctx_norm,log0):
    original,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    on=valid&~rest
    inherited,bmean,bstd,*_=fixed
    feat=c18.features(history,q[at],inherited[0],bmean,bstd)
    np.testing.assert_array_equal(feat.valid,valid)
    norms,scales=feature_norm
    f0=(feat.pooled[:,1].astype(np.float64)-norms)/scales
    context=from_frontend(raw,observed,acc,dt)[at]
    z=np.concatenate((f0,(context-ctx_norm[0])/ctx_norm[1]),axis=1)
    probs={name:c23.weights(model,torch.from_numpy(features),log0).numpy()
           for (name,model),features in zip(models.items(),(f0,f0,z),strict=True)}
    dirs={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(on[:,None],mix,0.))}
    shifts={}
    for name,p in probs.items():
        base.require(np.isfinite(p).all() and np.all(p>=0.) and np.max(np.abs(p.sum(1)-1.))<=PROB_TOL,
                     "C025 probabilities violate simplex")
        shift=np.where(on[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        dirs[name+"_deg"]=c23.augment(original,center,shift)
        shifts[name]=shift
    base.require(np.max(np.linalg.norm(shifts["C025"],axis=1))<=c22.CAP+2e-15,"C025 combined filter correction exceeded the cap")
    return dirs,probs,on,rest,context


def audit(cfg,k,fixed,models,norm,ctx_norm,log0):
    rows=[]
    for hz in (48.,100.,200.):
        dt=1./hz
        optical=hz if hz<100. else None
        raw,observed=c14.synthetic(hz)
        _,q,at,release,hist,_,acc,_=c17.front_end(raw,observed,dt,cfg,optical)
        base_output,base_p,base_on,base_rest,base_context=held(q,at,release,hist,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,log0)
        late=np.flatnonzero(release>at)
        cuts={len(at)//2,len(at)-2}
        if len(late): cuts.add(int(late[len(late)//2]))
        context_error=np.zeros(4)
        p_error,theta=0.,0.
        for i in sorted(cuts):
            edited=raw.copy()
            edited[at[i]+1:,:3]+=[16.,-7.,8.]
            edited[at[i]+1:,3:]+=1.2
            perturbed=observed.copy()
            perturbed[at[i]+2::2]=False
            end,stop=i+1,int(release[i])+1
            for signal,obs in ((raw[:stop],observed[:stop]),(edited,perturbed)):
                _,qr,ar,rr,h,_,ac,_=c17.front_end(signal,obs,dt,cfg,optical)
                np.testing.assert_array_equal(ar[:end],at[:end])
                np.testing.assert_array_equal(rr[:end],release[:end])
                values,p,on,rest,c=held(qr,ar[:end],rr[:end],h[:end],ac,signal,obs,dt,k,fixed,models,norm,ctx_norm,log0)
                np.testing.assert_array_equal(on,base_on[:end])
                np.testing.assert_array_equal(rest,base_rest[:end])
                context_error=np.maximum(context_error,np.max(np.abs(c-base_context[:end]),axis=0))
                p_error=max(p_error,float(np.max(np.abs(p["C025"]-base_p["C025"][:end]))))
                theta=max(theta,float(c20.angles(values["C025_deg"],base_output["C025_deg"][:end]).max()))
        base.require(max(context_error)<=2e-10 and p_error<=5e-7 and theta<=1e-5,"C025 future changes affected observers or output")
        rotations=[]
        qorig=Rotation.from_quat(q[release],scalar_first=True)
        for label,rot in zip(("90z","90x","63_1_2_-1"),
               (Rotation.from_rotvec([0.,0.,np.pi/2]),Rotation.from_rotvec([np.pi/2,0.,0.]),
                Rotation.from_rotvec(np.deg2rad(63.)*np.array([1.,2.,-1.])/np.sqrt(6.))),strict=True):
            transformed=rot.apply(raw.reshape(-1,3)).reshape(raw.shape)
            _,qr,ar,rr,h,_,ac,_=c17.front_end(transformed,observed,dt,cfg,optical)
            np.testing.assert_array_equal(at,ar)
            np.testing.assert_array_equal(release,rr)
            values,p,on,rest,c=held(qr,ar,rr,h,ac,transformed,observed,dt,k,fixed,models,norm,ctx_norm,log0)
            native=rot.inv().apply(Rotation.from_quat(qr[rr],scalar_first=True).inv().apply(values["C025_deg"]))
            delta_context=np.max(np.abs(c-base_context),axis=0)
            delta_prob=float(np.max(np.abs(p["C025"]-base_p["C025"])))
            delta_tilt=float(c20.angles(native,qorig.inv().apply(base_output["C025_deg"])).max())
            base.require(np.isfinite(np.r_[delta_context,delta_prob,delta_tilt]).all(),"Invalid rotation diagnostic")
            rotations.append({"rotation":label,"observer_max_abs":dict(zip(OBSERVERS,delta_context.tolist(),strict=True)),
                              "probability_max_abs":delta_prob,"body_tilt_max_deg":delta_tilt,"gate_changes":int(np.count_nonzero(on!=base_on))})
        rows.append({"native_hz":hz,"windows":len(hist),"cuts":sorted(cuts),"releases_after_input_last":len(late),
                     "future_observer_max_abs":dict(zip(OBSERVERS,context_error.tolist(),strict=True)),"future_probability_max_abs":p_error,
                     "future_body_tilt_max_deg":theta,"rotation_from_raw_imu":rotations})
    return rows


def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ("C023","C024","C025")} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C025_delta_deg":{k:macro["C025_deg"]-macro[k] for k in METHODS if k!="C025_deg"} if good else {},
            "C025_wins":{k:sum(r["C025_deg"]<r[k] for r in good) for k in METHODS if k!="C025_deg"}}


def replay(previous,control,inventory,campaign,split,cfg,k,fixed,models,norm,ctx_norm,log0):
    c1r,c1p,c3r,c2p=(read(ROOT/n) for n in ("C001/results.json","C001/protocol.json","C003/results.json","C002/protocol.json"))
    evidence=c2p["frozen_evidence"]
    map1,map3=({(r["dataset"],r["record"]):r for r in s["sequences"]} for s in (c1r,c3r))
    keys={(r["dataset"],r["record"]) for r in previous["sequences"]}
    spec={(r["dataset"],r["record"]):r for r in inventory["records"] if (r["dataset"],r["record"]) in keys}
    table={(r["dataset"],r["record"]):r for r in control["records"]}
    base.require(len(keys)==213 and set(spec)==keys==set(table)
                 and all(r["side"] in ("development","development_previously_used_for_selection") for r in spec.values())
                 and control["source_csv_sha256"]==sha(ROOT/"C020/sequences.csv") and control["selection_lock_sha256"] is None,
                 "C025 replay or fixed controls leave development scope")
    mp,ap=DATA/"Sassari/medium_v5.mat",DATA/"IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp)==split["source_file_sha256"][str(mp)] and sha(ap)==evidence["array_reference_sha256"],"Frozen reference changed")
    medium=loadmat(mp,variable_names=("Qs","XS1","XS2","AP1","AP2","SH1","SH2"))
    array=c2.reference_array(ap)
    poses_old={(r["record"],r["pose"]):r for r in previous["array_sensor_poses"]}
    seq,poses,failed,ledger=[],[],[],[]
    frozen=c24.METHODS[:5]
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
                             and array_sha(release,"<i8")==old["native_release_sha256"],"C024 or fixed native indices changed")
                run=perf_counter()
                dirs,probs,active,rest,context=held(q,at,release,w,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,log0)
                calc=perf_counter()-run
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(active.sum()==old["active_release_count"] and
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ("C023","C024")),
                             "Archived C023 or C024 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C025_deg"][same],errs["C020_deg"][same])
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
                            "prediction_seconds":calc,"seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i),"release":release[active],"input_last":at[active],
                               "context":context[active],**{m:p[active] for m,p in probs.items()}})
                poses.extend(detail)
                print("development",*key,score["C023_deg"],score["C024_deg"],score["C025_deg"],flush=True)
            except (AssertionError,RuntimeError,OSError,ValueError,KeyError,IndexError) as ex:
                failed.append({"dataset":key[0],"record":key[1],"error":repr(ex)})
                seq.append({**identity,"status":"failed","error":repr(ex),"seconds":perf_counter()-start})
                print("FAILED",*key,repr(ex),flush=True)
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

def status(protocol,phase,**detail):
    write(HERE/"process.json",{"pid":os.getpid(),"original_fit_pid":protocol["pid"],"status":phase,
                              "started_at":protocol["started_at"],"updated_at":stamp(),"protocol_sha256":sha(HERE/"protocol.json"),**detail})


def after_lock(protocol,recovery=None):
    branch,checkpoint=HERE/"router-lock.json",HERE/"router.pt"
    lock=read(branch)
    sig=sha(branch)
    base.require(lock["pid"]==protocol["pid"] and lock["fit_count"]==1 and lock["checkpoint_sha256"]==sha(checkpoint)
                 and lock["train_context_sha256"]==sha(HERE/"training-context.npz")
                 and lock["training_sha256"]==sha(HERE/"training.json") and lock["protocol_sha256"]==sha(HERE/"protocol.json")
                 and all(sha(ROOT/n)==h for n,h in protocol["input_file_sha256"].items()),"C025 lock/control changed")
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    payload=[torch.load(p,map_location="cpu",weights_only=True) for p in (ROOT/"C023/router.pt",ROOT/"C024/router.pt",checkpoint)]
    models={}
    for name,state,width in zip(("C023","C024","C025"),payload,(24,24,28),strict=True):
        model=nn.Linear(width,2,dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name]=model.eval().requires_grad_(False)
    old, prior, stored=payload
    for k in ("p0","log_p0"):
        torch.testing.assert_close(old[k],stored[k],rtol=0.,atol=0.)
        torch.testing.assert_close(prior[k],stored[k],rtol=0.,atol=0.)
    for name in ("mean","std"):
        torch.testing.assert_close(old["feature_"+name],stored["old_feature_"+name],rtol=0.,atol=0.)
        torch.testing.assert_close(prior["feature_"+name],stored["old_feature_"+name],rtol=0.,atol=0.)
    norm=tuple(stored[k].numpy() for k in ("old_feature_mean","old_feature_std"))
    context=tuple(stored[k].numpy() for k in ("context_mean","context_std"))
    log0=stored["log_p0"]
    k=protocol["C022_k"]
    tags={"router_lock_sha256":sig,"recovery_sha256":recovery}
    status(protocol,"native_context_audits",**tags)
    ad=HERE/"causal-rotation-audit.json"
    write(ad,{**tags,"native_source_audit_sha256":sha(HERE/"source-context-audit.json"),
              "meaning":"All observer inputs rebuilt from raw native IMU under causal fill and online debias. Rotation includes reinitialization of the VQF chain.",
              "checks":audit(cfg,k,fixed,models,norm,context,log0)})
    status(protocol,"development_replay",**tags)
    print("C025 learned checkpoint locked. Starting 213-record development replay.",flush=True)
    start=perf_counter()
    result=replay(read(ROOT/"C024/results.json"),read(ROOT/"review/vqf_response_controls_development.json"),
                  read(ROOT/"C001/inventory.json"),campaign,read(TRANSFER/"external-split.json"),cfg,k,fixed,models,norm,context,log0)
    base.require(sha(branch)==sig and sha(checkpoint)==lock["checkpoint_sha256"]
                 and all(sha(ROOT/n)==h for n,h in protocol["input_file_sha256"].items()),"A locked C025 source changed")
    result.update({**tags,"pid":os.getpid(),"original_fit_pid":protocol["pid"],"started_at":protocol["started_at"],
                   "completed_at":stamp(),"protocol_sha256":sha(HERE/"protocol.json"),"scoring_masks_sha256":sha(MASKS),
                   "C024_sampling_lock_sha256":protocol["input_file_sha256"]["C024/sampling-lock.json"],
                   "causal_rotation_audit_sha256":sha(ad),"development_replay_seconds":perf_counter()-start,
                   "final_holdout_scored":False,"poster_numeric_file_reads":0})
    out=HERE/"results.json"
    write(out,result)
    completed=not result["failures"] and result["pooled_records"]["records"]==213 and len(result["array_sensor_poses"])==96
    status(protocol,"complete" if completed else "incomplete",**tags,scored=result["pooled_records"]["records"],results_sha256=sha(out))
    print("C025 finished",result["pooled_records"],"failures",result["failures"],flush=True)
    if not completed:
        raise RuntimeError("C025 replay incomplete; router remains locked")


def main():
    launch,initial=(read(HERE/n) for n in ("launch.json","input-verification.json"))
    files=initial["files_sha256"]
    base.require(os.getpid()==launch["pid"]==initial["pid"] and all(sha(ROOT/n)==h for n,h in files.items()),"Registered C025 inputs changed")
    p24,lock22,diag=(read(ROOT/n) for n in ("C024/protocol.json","C022/coefficient-lock.json","review/source_motion_router_diagnostic.json"))
    base.require(p24["optimizer"]=={"kind":"AdamW","lr":LR,"weight_decay":WD,"clip_global_norm":CLIP,"steps":160,
                                  "batch_size":104,"foreach":False}
                 and p24["C022_k"]==lock22["k"] and p24["scoring_masks_sha256"]==sha(MASKS)
                 and len(p24["development_records"])==213,"C024 training or replay recipe differs")
    for n,artifact in diag["davis"].items():
        p=(ROOT/artifact["file"]).resolve()
        base.require(n in NAMES and p==(ROOT/"review/davis_source_motion_context"/(n+".npy")).resolve()
                     and sha(p)==artifact["sha256"],"DAVIS context-cache receipt changed")
        files[p.relative_to(ROOT).as_posix()]=artifact["sha256"]
    for name,h in p24["inputs_sha256"].items():
        if name.startswith("external-sources/davis-handheld/"):
            base.require(sha(ROOT/name)==h,"A frozen DAVIS training file changed")
            files[name]=h
    protocol={**launch,"worker_sha256":sha(Path(__file__)),"input_file_sha256":files,
              "C022_k":lock22["k"],"optimizer":p24["optimizer"],"fit_count":1,"parameters":58,"input_width":28,
              "dimensions_preserved":"C024 archived 24 normalized columns, exact bytes",
              "novel_columns":list(OBSERVERS),"observer_tau_s":2.,
              "observer_source":"Same fully causal filled and online-debiased native six-axis stream, observer state read at native_input_last.",
              "normalizer":"Append only four columns. Population moments, old equal source/record/draw weights with total mass divided by five.",
              "objective":"Unchanged C024 per-slot effective weight and 32 multiplier; division by original full-source C022 denominators.",
              "development_records":p24["development_records"],"scoring_masks_sha256":sha(MASKS),"allow_poster_numeric_reads":False}
    path=HERE/"protocol.json"
    write(path,protocol)
    torch.set_num_threads(2)
    status(protocol,"training_context_preparation")
    data,archived,features,mean,std=prepare(protocol)
    status(protocol,"single_58_parameter_fit",training_context_sha256=sha(HERE/"training-context.npz"))
    print("C025 features aligned for all 16640 slots; fifty-eight parameters, one fit.",flush=True)
    trained=fit(data,archived,features,mean,std,protocol)
    train_path,branch=HERE/"training.json",HERE/"router-lock.json"
    write(train_path,{**trained,"context_normalization_sha256":sha(HERE/"context-normalization.json"),
                      "source_context_audit_sha256":sha(HERE/"source-context-audit.json")})
    with branch.open("x",encoding="utf-8") as f:
        json.dump({"pid":os.getpid(),"locked_at":stamp(),"steps":160,"fit_count":1,"protocol_sha256":sha(path),
                   "checkpoint_sha256":sha(HERE/"router.pt"),"training_sha256":sha(train_path),
                   "train_context_sha256":sha(HERE/"training-context.npz"),"C024_router_lock_sha256":files["C024/router-lock.json"]},f,indent=2)
        f.write("\n")
    after_lock(protocol)


def resume():
    protocol,lock,state=(read(HERE/n) for n in ("protocol.json","router-lock.json","process.json"))
    base.require(state["status"]!="complete" and protocol["pid"]==lock["pid"]==state["original_fit_pid"],"Cannot replay an unlocked or already complete C025")
    path=HERE/"replay-recovery.json"
    write(path,{"pid":os.getpid(),"original_fit_pid":lock["pid"],"started_at":stamp(),"fit_repeated":False,
                "router_lock_sha256":sha(HERE/"router-lock.json"),
                "original_failure_sha256":{n:sha(HERE/n) for n in ("failure.json","failure-traceback.log","results.json") if (HERE/n).is_file()}})
    after_lock(protocol,sha(path))


if __name__=="__main__":
    if "--resume" in sys.argv[1:]:
        resume()
    else:
        main()
