"""C033: append two PennCOSYVIO source slots to each frozen C023 minibatch."""
from datetime import datetime,timedelta,timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path[:0]=[str(ROOT/"C031"),str(ROOT/"review")]

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

import run_average_router as c31
from penn_handheld import BASE, Penn

c27,c23,c22,c18,c17,c3,c2=c31.c27,c31.c23,c31.c22,c31.c18,c31.c17,c31.c3,c31.c2
base,intrinsic=c31.base,c31.intrinsic
c14=c23.c14
read,write,sha,stamp,array_sha=c31.read,c31.write,c31.sha,c31.stamp,c31.array_sha
UP,DATA,TRANSFER,MASKS,GATES=c31.UP,c31.DATA,c31.TRANSFER,c31.MASKS,c31.GATES
require,archive,check_probabilities,add_baselines=c31.require,c31.archive,c31.check_probabilities,c31.add_baselines
PENN=("af","as")
DOMAINS=(*c23.DOMAINS,"PennCOSYVIO")
ROUTERS=("C023","C031","C033")
METHODS=(*c23.METHODS[:4],"C031_deg","C033_deg",*c23.METHODS[4:])
WEIGHTS=c23.WEIGHTS
STEPS,OLD,BATCH=160,96,98
MULTIPLIER=STEPS/len(DOMAINS)
FIELDS=("center","fast","slow","truth","enabled","rest","native_release","native_input_last","source_record_ordinal",
        "source_record_name","record_equal_weights","domain")
STATE_KEYS=("feature_mean","feature_std","p0","log_p0")


def local_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="milliseconds")


def status(launch,phase,**fields):
    write(HERE/"process.json",{"pid":os.getpid(),"started_at":launch["started_at"],"status":phase,"updated_at":local_now(),
                              "protocol_sha256":sha(HERE/"protocol.json"),**fields})


def frozen_state(name):
    return torch.load(ROOT/name/"router.pt",map_location="cpu",weights_only=True)


def release_key(name,native_release):
    return hashlib.sha256(f"C033|{name}|{native_release}".encode("utf-8")).digest()


def prepare_old(protocol,anchor):
    lock=read(ROOT/"C022/coefficient-lock.json")
    require(lock["k"]==protocol["C022_k"],"C022 coefficient lock changed")
    original,source,x=c23.load_train(lock,protocol["inputs_sha256"],anchor["feature_mean"].numpy(),anchor["feature_std"].numpy())
    previous=archive(ROOT/"C023/training-io.npz")
    for own,expected in ((x["z"].numpy(),previous["normalized_input"]),(x["prob0"].numpy(),anchor["p0"].numpy()),
                         (x["log0"].numpy(),anchor["log_p0"].numpy()),(source["enabled"],previous["enabled"]),
                         (source["native_release"],previous["native_release"]),
                         (source["source_record_ordinal"],previous["record_ordinal"])):
        np.testing.assert_array_equal(own,expected)
    with (ROOT/"C023/optimizer-steps.jsonl").open(encoding="utf-8") as f:
        steps=[json.loads(line) for line in f]
    require(len(steps)==STEPS and len(np.unique(source["source_record_ordinal"]))==139,"Missing C023 schedule evidence")
    for j,step in enumerate(steps):
        cut=slice(j*OLD,(j+1)*OLD)
        hashes={
            "input_feature_sha256":array_sha(x["z"][cut].numpy(),"<f8"),
            "native_release_sha256":array_sha(source["native_release"][cut],"<i8"),
            "native_input_last_sha256":array_sha(source["native_input_last"][cut],"<i8"),
            "source_record_ordinal_sha256":array_sha(source["source_record_ordinal"][cut],"<i8"),
            "legacy_slot_identity_sha256":array_sha(original["global_window"][cut],"<i8"),
        }
        require(step["step"]==j+1 and all(step[k]==h for k,h in hashes.items()),f"C023 batch {j+1} was changed")
    source["z"]=x["z"].numpy()
    return source,previous,original["global_window"]


def source_packet(name,ds,expected,fixed,cfg,anchor,ordinal):
    require(name in PENN and ds.roles[name]=="external_training","A sealed Penn traversal is outside this protocol")
    (raw,hz,up,score_mask),quality=ds.native(name)
    require({**quality,"record":name,"prepared":expected["prepared"],"sha256":expected["sha256"]}==expected,
            f"Penn native quality differs from the frozen {name} record")
    with np.load(ROOT/expected["prepared"],allow_pickle=False) as checked:
        for k,v in (("six_axis",raw),("native_hz",hz),("optical_up",up),("mask",score_mask)):
            np.testing.assert_array_equal(checked[k],v)
    clock=np.loadtxt(BASE/name/"imu.txt",usecols=0)
    observed=np.isfinite(raw).all(axis=1)
    ref_valid=score_mask & np.isfinite(up).all(axis=1) & (np.abs(np.linalg.norm(up,axis=1)-1.)<=2e-12)
    np.testing.assert_array_equal(ref_valid,score_mask)
    require(len(clock)==len(raw) and np.all(np.diff(clock)>0.) and np.all(clock[score_mask]-clock[0]>=10.),
            f"Penn native time or post-prefix reference changed for {name}")
    dt=1./hz
    q0,q,at,release,history,complete,acc,calibration=c17.front_end(raw,observed,dt,cfg)
    original,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    intrinsic_valid=intrinsic.build_frame(history).valid
    np.testing.assert_array_equal(valid,intrinsic_valid)
    active=valid & ~rest
    eligible=np.flatnonzero(complete & active & ref_valid[release] & observed[at] & observed[release])
    require(len(eligible)>=STEPS and np.all(at<=release) and np.all(np.diff(release)>0.),
            f"Penn {name} has fewer than 160 admissible observed complete active releases")
    order=sorted(eligible.tolist(),key=lambda i:(release_key(name,int(release[i])),int(release[i])))
    positions=np.asarray(order[:STEPS],dtype=np.int64)
    rel,cut=release[positions],at[positions]
    require(np.unique(rel).size==STEPS and np.all(active[positions]),"Penn sampling must have no repeats and active nonrest support")
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    _,_,_,_,_,check,check_tau3=c22.filters(raw,observed,acc,q,dt,cut,independence=True,check_default=True)
    models,mean,std,*_=fixed
    feat=c18.features(history,q[at],models[0],mean,std)
    np.testing.assert_array_equal(feat.valid,valid)
    z=(feat.pooled[:,1].astype(np.float64)-anchor["feature_mean"].numpy())/anchor["feature_std"].numpy()
    truth=c14.rotate(c14.normalize(q[rel]),up[rel])
    require(np.isfinite(z).all() and np.allclose(np.linalg.norm(truth,axis=1),1.,atol=2e-12,rtol=0.)
            and np.all(center[:,2]==1.),"Invalid Penn feature or target orientation")
    arrays={"center":center[positions],"fast":fast[positions],"slow":slow[positions],"truth":truth,
            "enabled":active[positions],"rest":rest[positions],"native_release":rel,"native_input_last":cut,
            "source_record_ordinal":np.full(STEPS,ordinal,dtype=np.int64),"source_record_name":np.full(STEPS,name),
            "record_equal_weights":np.full(STEPS,1./(2*STEPS)),"domain":np.full(STEPS,"PennCOSYVIO"),"z":z[positions]}
    receipt={"record":name,"slot_in_batch":OLD+ordinal-139,"source_record_ordinal":ordinal,"native_hz":hz,
             "native_rows":len(raw),"valid_reference_frames":int(score_mask.sum()),
             "scoring_indices_sha256":array_sha(np.flatnonzero(score_mask),"<i8"),
             "complete_window_releases":int(complete.sum()),"active_releases":int(active.sum()),
             "eligible_releases":len(eligible),"eligible_native_release_rows":release[eligible].tolist(),
             "eligible_native_release_sha256":array_sha(release[eligible],"<i8"),
             "numeric_source_preparation_sha256":expected["sha256"],"train_slots":len(positions),
             "debias_active_native_frames":int(calibration["active_native_frames"]),
             "independent_filter_response_max_abs":check,"default_tau3_max_abs":check_tau3,
             "selected":[{"step":j+1,"native_release":int(r),"native_input_last":int(a),
                          "native_release_timestamp_s":float(clock[r]),"native_input_last_timestamp_s":float(clock[a]),
                          "seconds_since_start":float(clock[r]-clock[0]),"reference_valid":bool(ref_valid[r]),
                          "release_observed":bool(observed[r]),"latest_input_observed":bool(observed[a]),
                          "complete_history":bool(complete[pos]),"intrinsic_valid":bool(valid[pos]),"rest":bool(rest[pos]),
                          "enabled":bool(active[pos]),"literal_sha256":release_key(name,int(r)).hex()}
                         for j,(pos,r,a) in enumerate(zip(positions,rel,cut,strict=True))]}
    np.savez_compressed(HERE/f"{name}-native-mask-evidence.npz",native_timestamp_s=clock,scoring_mask=score_mask,
                        observed=observed,reference_valid=ref_valid,release=release,native_input_last=at,
                        complete_history=complete,rest=rest,intrinsic_valid=valid,enabled=active,eligible=eligible,selected=positions)
    evaluation={"name":name,"q0":q0,"q":q,"up":up,"mask":score_mask,"release":release,"original":original,
                "center":center,"fast":fast,"slow":slow,"mixture":mix,"active":active,"z":z}
    return arrays,receipt,evaluation


def prepare_penn(protocol,anchor,fixed,cfg):
    pre=read(BASE/"preprocessing-lock.json")
    ds=Penn()
    require(not ds.formal and tuple(r["record"] for r in pre["source_records"])==PENN,"Only the two Penn A training records are registered")
    streams,receipts,score_packets=[],[],[]
    for j,expected in enumerate(pre["source_records"],139):
        arrays,receipt,scoring=source_packet(expected["record"],ds,expected,fixed,cfg,anchor,j)
        streams.append(arrays)
        receipts.append(receipt)
        score_packets.append(scoring)
        print("C033 sampled",expected["record"],len(receipt["selected"]),"from",receipt["eligible_releases"],flush=True)
    packet={key:np.stack([array[key] for array in streams],axis=1) for key in streams[0]}
    np.savez_compressed(HERE/"penn-training-slots.npz",**packet)
    write(HERE/"sampling-lock.json",{"pid":os.getpid(),"locked_at":local_now(),"seed_free_rule":"Literal C033|<canonical lower-case published record>|<decimal native release index>",
          "order":"256-bit SHA256 digest lexicographic, ties by native release index; fixed af then as slots after the old 96 in each step",
          "C023_slots_per_step":96,"Penn_slots_per_step":2,"total_steps":160,"records":receipts,
          "Penn_preprocessing_sha256":protocol["Penn_preprocessing_sha256"],"penn_slots_sha256":sha(HERE/"penn-training-slots.npz"),
          "native_masks_sha256":{name:sha(HERE/f"{name}-native-mask-evidence.npz") for name in PENN}})
    return packet,score_packets


def score_penn(packets,models,k,protocol):
    archived={row["record"]:row for row in read(ROOT/"review/penn_training_controls.json")["rows"]}
    rows=[]
    for item in packets:
        original,center,active=(item[v] for v in ("original","center","active"))
        fast,slow,mix=(item[k] for k in ("fast","slow","mixture"))
        out={"C020":original,"C022":c23.augment(original,center,k*np.where(active[:,None],mix,0.))}
        probs={}
        for name,model in models.items():
            with torch.inference_mode():
                p=c23.weights(model,torch.from_numpy(item["z"]),c23.prior_probs(k).log()).numpy()
            check_probabilities(p)
            out[name]=c23.augment(original,center,np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.))
            probs[name]=dict(zip(WEIGHTS,p[active].mean(axis=0).tolist(),strict=True))
        ix=np.flatnonzero(item["mask"])
        squared=c17.measure(ix,item["up"],item["q0"],item["q"],item["release"],
                            {name:np.vstack((UP,v)) for name,v in out.items()})
        deg=c17.scores(squared)
        record=item["name"]
        old=archived[record]
        gap=max(abs(deg["full_vqf_deg"]-old["default"]),*(abs(deg[m]-old[m]) for m in ("C023","C031")))
        require(abs(deg["full_vqf_deg"]-old["default"])<=1e-5 and
                array_sha(ix,"<i8")==old["quality"]["scoring_indices_sha256"],"Penn A native scoring reference changed")
        rows.append({"record":record,"scored_frames":len(ix),"mask_sha256":array_sha(ix,"<i8"),
                     "full_record_tilt_RMSE_deg":{**deg,"archived_fixed_fast":old["fixed_fast"],"archived_fixed_slow":old["fixed_slow"]},
                     "active_release_probabilities":probs,"prior_control_max_abs_diff_deg":gap})
    mean={m:fmean(r["full_record_tilt_RMSE_deg"][m] for r in rows) for m in rows[0]["full_record_tilt_RMSE_deg"]}
    write(HERE/"penn-training-source-scores.json",{"pid":os.getpid(),"scope":"AF and AS in-sample training source only, full frozen native optical masks",
          "checkpoint_sha256":sha(HERE/"router.pt"),"preprocessing_sha256":protocol["Penn_preprocessing_sha256"],
          "record_equal_mean":mean,"records":rows})


def fit(old,previous,legacy,penn,anchor):
    started=perf_counter()
    data={k:np.concatenate((old[k].reshape((STEPS,OLD)+old[k].shape[1:]),penn[k]),axis=1) for k in (*FIELDS,"z")}
    for key in (*FIELDS,"z"):
        np.testing.assert_array_equal(data[key][:,:OLD].reshape(old[key].shape),old[key])
    require(data["z"].shape==(STEPS,BATCH,24) and np.unique(data["source_record_ordinal"]).size==141 and
            data["enabled"][:,OLD:].all() and not data["rest"][:,OLD:].any(),"Expected 15,680 slots, 141 records, and active Penn slots")
    flat={k:v.reshape((-1,)+v.shape[2:]) for k,v in data.items()}
    inputs={k:torch.from_numpy(flat[k]) for k in ("z","center","slow","fast","truth","enabled")}
    inputs["log0"]=anchor["log_p0"]
    model=nn.Linear(24,2,dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    params=tuple(model.parameters())
    require(sum(p.numel() for p in params)==50 and not c18.flat_params(model).any(),"Bad 50-parameter zero initialization")
    with torch.inference_mode():
        before,pred0,delta0,p0=c23.losses(model,inputs,slice(None))
    initial=before.numpy()
    zero_gap=float(np.max(np.abs(initial.reshape(STEPS,BATCH)[:,:OLD].ravel()-previous["C022_loss"])))
    require(zero_gap<=3e-17 and torch.count_nonzero(delta0[~inputs["enabled"]]).item()==0 and
            torch.max((p0-anchor["p0"]).abs()).item()<=8*np.finfo(np.float64).eps,"C023 zero-step prior or chord changed")
    torch.testing.assert_close(pred0[~inputs["enabled"]],c18.unit(inputs["center"])[~inputs["enabled"]],rtol=0.,atol=0.)
    frozen=read(ROOT/"C023/zero-step-audit.json")["C022_source_init_gradients"]
    denom={}
    effective=np.empty(len(initial))
    for domain in DOMAINS:
        mask=flat["domain"]==domain
        w=flat["record_equal_weights"][mask]
        own=float(w@initial[mask])
        denom[domain]=own if domain=="PennCOSYVIO" else frozen[domain]["denominator"]
        require(abs(w.sum()-1.)<=2e-12 and denom[domain]>0. and abs(own-denom[domain])<=2e-14,
                "A source has different record-equivalent mass or frozen C022 denominator")
        effective[mask]=w/denom[domain]
    np.testing.assert_array_equal(effective.reshape(STEPS,BATCH)[:,:OLD].ravel(),previous["relative_loss_weight"])
    require(MULTIPLIER==32. and abs(float(effective@initial)/5.-1.)<=5e-13,"Five-source normalization failed")
    write(HERE/"zero-step-audit.json",{"pid":os.getpid(),"C022_denominator_by_domain":denom,"loss_multiplier":MULTIPLIER,
          "C023_prior":anchor["p0"].tolist(),"C023_zero_step_chord_max_abs_difference":zero_gap,
          "old_effective_weights_preserved_bitwise":True,"normalized_full_initial_loss":float(effective@initial)/5.})
    np.savez_compressed(HERE/"training-inputs.npz",**data,effective_loss_weights=effective.reshape(STEPS,BATCH),
                        C022_chord=initial.reshape(STEPS,BATCH),C023_legacy_global_window=legacy.reshape(STEPS,OLD))
    w=torch.from_numpy(effective)
    optimizer=torch.optim.AdamW(params,lr=c23.LR,weight_decay=c23.WD,foreach=False)
    peak={n:0. for n in PENN}
    max_gap=0.
    with (HERE/"optimizer-steps.jsonl").open("x",encoding="utf-8") as log:
        for j in range(STEPS):
            cut=slice(j*BATCH,(j+1)*BATCH)
            optimizer.zero_grad(set_to_none=True)
            before_step=c18.flat_params(model)
            chord,_,_,p=c23.losses(model,inputs,cut)
            weighted=MULTIPLIER*w[cut]*chord
            objective=weighted.sum()
            term_masks={d:flat["domain"][cut]==d for d in DOMAINS}
            term_masks.update({n:flat["source_record_ordinal"][cut]==139+i for i,n in enumerate(PENN)})
            grad_book={}
            for name,mask in term_masks.items():
                part=weighted[mask].sum()
                g=torch.cat([u.ravel() for u in torch.autograd.grad(part,params,retain_graph=True)]).detach().numpy()
                gn=float(np.linalg.norm(g))
                grad_book[name]={"slots":int(mask.sum()),"objective":float(part.detach()),"gradient":g.tolist(),"gradient_l2":gn}
                if name in peak:
                    peak[name]=max(peak[name],gn)
            objective.backward()
            gradient=c18.flat_params(model,use_grad=True)
            gap=float(np.max(np.abs(sum(np.asarray(grad_book[d]["gradient"]) for d in DOMAINS)-gradient)))
            require(gap<=2e-14 and
                    abs(sum(grad_book[d]["objective"] for d in DOMAINS)-float(objective.detach()))<=2e-15,
                    "Actual source objectives or gradients did not reconstruct the optimizer update")
            require(np.max(np.abs(sum(np.asarray(grad_book[n]["gradient"]) for n in PENN)-
                                  grad_book["PennCOSYVIO"]["gradient"]))<=2e-14,"Penn record gradients did not sum to the source gradient")
            max_gap=max(max_gap,gap)
            length=float(torch.nn.utils.clip_grad_norm_(params,c23.CLIP))
            clipped=c18.flat_params(model,use_grad=True)
            optimizer.step()
            after=c18.flat_params(model)
            prob=p.detach().numpy()
            check_probabilities(prob)
            require(np.isfinite(np.r_[after,gradient,clipped,length,float(objective.detach())]).all(),"Nonfinite optimizer step")
            entry={"pid":os.getpid(),"step":j+1,"source_batch_index":j,"objective":float(objective.detach()),
                   "gradient_before_clip":gradient.tolist(),"gradient_after_clip":clipped.tolist(),"gradient_l2":length,
                   "weights_before":before_step.tolist(),"weights_after":after.tolist(),"actual_update":(after-before_step).tolist(),
                   "source_and_penn_record_gradients":grad_book,"gradient_sum_max_abs_difference":gap,
                   "C023_release_sha256":array_sha(data["native_release"][j,:OLD],"<i8"),
                   "Penn_release_sha256":array_sha(data["native_release"][j,OLD:],"<i8"),
                   "native_input_last_sha256":array_sha(data["native_input_last"][j],"<i8"),
                   "normalized_feature_sha256":array_sha(data["z"][j],"<f8"),"probability_sha256":array_sha(prob,"<f8")}
            print(json.dumps(entry,allow_nan=False),file=log,flush=True)
            if j+1 in (1,40,80,120,160):
                print("C033 fit",j+1,entry["objective"],length,flush=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        chord,pred,delta,prob=c23.losses(model,inputs,slice(None))
        torch.testing.assert_close(pred[~inputs["enabled"]],c18.unit(inputs["center"])[~inputs["enabled"]],rtol=0.,atol=0.)
        require(torch.count_nonzero(delta[~inputs["enabled"]]).item()==0,"C033 gate was bypassed")
    check_probabilities(prob.numpy())
    require(all(g>0. for g in peak.values()),"Both Penn records must contribute actual gradients")
    final=chord.numpy()
    np.savez_compressed(HERE/"training-io.npz",C022_chord=initial.reshape(STEPS,BATCH),C033_chord=final.reshape(STEPS,BATCH),
                        C033_prob=prob.numpy().reshape(STEPS,BATCH,3),C033_direction=pred.numpy().reshape(STEPS,BATCH,3))
    torch.save({**{k:anchor[k] for k in STATE_KEYS},"state_dict":model.state_dict(),"optimizer_steps":160,"dtype":"float64",
                "C022_coefficient_lock_sha256":sha(ROOT/"C022/coefficient-lock.json"),"protocol_sha256":sha(HERE/"protocol.json"),
                "sampling_lock_sha256":sha(HERE/"sampling-lock.json")},HERE/"router.pt")
    domains={}
    angle=lambda value:np.degrees(2*np.arcsin(np.sqrt(np.clip(value,0.,4.))/2.))
    for d in DOMAINS:
        sel=flat["domain"]==d
        weights=flat["record_equal_weights"][sel]
        pre=float(weights@initial[sel])
        post=float(weights@final[sel])
        ids=np.unique(flat["source_record_ordinal"][sel])
        records=[sel&(flat["source_record_ordinal"]==i) for i in ids]
        domains[d]={"slots":int(sel.sum()),"records":len(ids),"C022_record_equal_chord":pre,"C033_record_equal_chord":post,
                    "C022_record_macro_tilt_RMSE_deg":fmean(float(np.sqrt(np.mean(angle(initial[m])**2))) for m in records),
                    "C033_record_macro_tilt_RMSE_deg":fmean(float(np.sqrt(np.mean(angle(final[m])**2))) for m in records),
                    "relative_chord":post/pre}
    return {"pid":os.getpid(),"fit_count":1,"steps":STEPS,"parameters":50,"slots":STEPS*BATCH,"source_records":141,
            "training_seconds":perf_counter()-started,"C022_loss_multiplier":MULTIPLIER,
            "Penn_actual_gradient_l2_peaks":peak,"source_gradient_sum_max_abs_gap":max_gap,"domains":domains}



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
            "C033_delta_deg": {m: macro["C033_deg"] - macro[m] for m in METHODS if m != "C033_deg"} if keep else {},
            "C033_wins": {m: sum(r["C033_deg"] < r[m] for r in keep) for m in METHODS if m != "C033_deg"}}


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
                        array_sha(probabilities["C031"][active], "<f8") == old["C031_active_routing_sha256"] and
                        array_sha(deltas["C023"], "<f8") == old["C023_delta_sha256"] and
                        array_sha(deltas["C031"], "<f8") == old["C031_delta_sha256"],
                        "Frozen C023/C031 release count, probabilities, or deltas changed")
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                parent_gap = max(abs(score[m] - old[m]) for m in ("C023_deg","C031_deg"))
                require(parent_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023/C031 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C031_deg"][~active_scored], errs["C020_deg"][~active_scored])
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
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C031_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, parent_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs["C033"].items()},
                             **{f"{m}_p_{k}_mean": v for m, arm in active_probs.items() for k,v in arm.items()},
                             **{f"{m}_active_routing_sha256":array_sha(p[active],"<f8") for m,p in probabilities.items()},
                             **{f"{m}_delta_sha256":array_sha(d,"<f8") for m,d in deltas.items()},
                             "prediction_seconds":pred_seconds, "seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i-1),"native_release":release[active],
                               "native_input_last":at[active],**{m:p[active] for m,p in probabilities.items()}})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C031_deg"], score["C033_deg"], flush=True)
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









def replay_locked(protocol,launch,fixed,campaign,cfg,packets):
    lock=read(HERE/"router-lock.json")
    lock_hash=sha(HERE/"router-lock.json")
    require(lock["pid"]==os.getpid() and lock["fit_count"]==1 and lock["optimizer_steps"]==STEPS and
            lock["checkpoint_sha256"]==sha(HERE/"router.pt") and
            all(sha(HERE/name)==digest for name,digest in lock["artifact_sha256"].items()),
            "Replay requires original C033 locked fit")
    states={name:frozen_state(name) for name in ROUTERS}
    anchor,own=(states[name] for name in ("C023","C033"))
    for name in ("C031","C033"):
        for key in STATE_KEYS:
            torch.testing.assert_close(states[name][key],anchor[key],rtol=0.,atol=0.)
    require(own["optimizer_steps"]==STEPS and own["protocol_sha256"]==sha(HERE/"protocol.json") and
            own["sampling_lock_sha256"]==sha(HERE/"sampling-lock.json"),"C033 model receipt changed")
    models={name:c31.frozen_linear(state) for name,state in states.items()}
    routers=models,anchor["feature_mean"].numpy(),anchor["feature_std"].numpy(),anchor["log_p0"]
    status(launch,"scoring_Penn_training_records",router_lock_sha256=lock_hash)
    score_penn(packets,models,protocol["C022_k"],protocol)
    prior=read(ROOT/"C031/results.json")
    require([[r["dataset"],r["record"]] for r in prior["sequences"]]==protocol["development_records"],
            "Development replay order changed")
    status(launch,"development_replay",router_lock_sha256=lock_hash,expected_records=213)
    print("C033 locked; replaying common development panel",flush=True)
    start=perf_counter()
    result=replay(prior,read(ROOT/"review/vqf_response_controls_development.json"),read(ROOT/"C001/inventory.json"),
                  campaign,read(TRANSFER/"external-split.json"),cfg,protocol["C022_k"],fixed,routers)
    require(sha(HERE/"router-lock.json")==lock_hash and sha(HERE/"router.pt")==lock["checkpoint_sha256"] and
            all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"Frozen evidence changed")
    result.update(pid=os.getpid(),router_lock_sha256=lock_hash,protocol_sha256=sha(HERE/"protocol.json"),
                  scoring_masks_sha256=sha(MASKS),penn_training_source_scores_sha256=sha(HERE/"penn-training-source-scores.json"),
                  sampling_lock_sha256=own["sampling_lock_sha256"],fit_count=1,optimizer_steps=160,
                  replay_seconds=perf_counter()-start,completed_at=local_now(),final_holdout_scored=False,
                  supplemental_holdout_scored=False,unapproved_penn_numeric_reads=0)
    out=HERE/"results.json"
    write(out,result)
    scored=result["pooled_records"]["records"]
    complete=scored==213 and not result["failures"] and len(result["array_sensor_poses"])==96
    status(launch,"complete" if complete else "incomplete",scored=scored,router_lock_sha256=lock_hash,results_sha256=sha(out))
    require(complete,"C033 replay incomplete; checkpoint remains locked")
    print("C033 complete",scored,result["pooled_records"]["macro_tilt_rmse_deg"],flush=True)


def main():
    launch={"pid":os.getpid(),"started_at":local_now(),"command":"python -B -u C033/run_penn_router.py",
            "scope":"C033 only; AF and AS paired scheduling, one 50-parameter source fit, common development replay",
            "protocol_sha256":sha(HERE/"protocol.json")}
    with (HERE/"launch.json").open("x",encoding="utf-8") as handle:
        json.dump(launch,handle,indent=2)
        handle.write("\n")
    try:
        protocol=read(HERE/"protocol.json")
        require(sha(Path(__file__))==protocol["worker_sha256"] and
                all(sha(ROOT/n)==h for n,h in protocol["inputs_sha256"].items()),"Registered C033 artifact changed")
        require(protocol["penn_allowed_training_records"]==list(PENN) and
                protocol["optimizer"]=={"kind":"AdamW","lr":c23.LR,"weight_decay":c23.WD,"clip_global_norm":c23.CLIP,
                                        "steps":STEPS,"batch_size":BATCH,"foreach":False},"C033 recipe changed")
        torch.set_num_threads(2)
        status(launch,"locking_Penn_training_slots")
        anchor=frozen_state("C023")
        old,previous,legacy=prepare_old(protocol,anchor)
        cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
        fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
        penn,packets=prepare_penn(protocol,anchor,fixed,cfg)
        status(launch,"single_fit",sampling_lock_sha256=sha(HERE/"sampling-lock.json"))
        training=fit(old,previous,legacy,penn,anchor)
        write(HERE/"training.json",training)
        artifacts=("protocol.json","sampling-lock.json","penn-training-slots.npz","af-native-mask-evidence.npz",
                   "as-native-mask-evidence.npz","training-inputs.npz","training-io.npz","training.json",
                   "optimizer-steps.jsonl","zero-step-audit.json")
        with (HERE/"router-lock.json").open("x",encoding="utf-8") as handle:
            json.dump({"pid":os.getpid(),"locked_at":local_now(),"fit_count":1,"optimizer_steps":STEPS,
                       "checkpoint_sha256":sha(HERE/"router.pt"),
                       "artifact_sha256":{n:sha(HERE/n) for n in artifacts}},handle,indent=2)
            handle.write("\n")
        replay_locked(protocol,launch,fixed,campaign,cfg,packets)
    except BaseException as error:
        write(HERE/"failure.json",{**launch,"failed_at":local_now(),"error":repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__=="__main__":
    main()

