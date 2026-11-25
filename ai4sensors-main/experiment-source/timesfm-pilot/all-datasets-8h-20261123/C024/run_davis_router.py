"""C024: one router fit augmenting each C023 batch with eight frozen DAVIS draws."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C023"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
import run_router as c23
from davis_handheld import Davis

c22, c20, c18, c17, c14, c3, c2 = c23.c22, c23.c20, c23.c18, c23.c17, c23.c14, c23.c3, c23.c2
base, intrinsic = c23.base, c23.intrinsic
read, write, sha, stamp, array_sha = c23.read, c23.write, c23.sha, c23.stamp, c23.array_sha
UP, DATA, PILOT, TRANSFER, MASKS, GATES = c23.UP, c23.DATA, c23.PILOT, c23.TRANSFER, c23.MASKS, c23.GATES
STEPS, OLD_BATCH, NEW_BATCH, BATCH = c23.STEPS, 96, 8, 104
LR, WD, CLIP, WEIGHTS, QUANTILES = c23.LR, c23.WD, c23.CLIP, c23.WEIGHTS, c23.QUANTILES
DAVIS = ROOT / "external-sources/davis-handheld"
NAMES = ("shapes_rotation", "shapes_translation", "shapes_6dof", "boxes_rotation", "boxes_translation", "boxes_6dof")
DOMAINS = (*c23.DOMAINS, "DAVIS")
METHODS = (*c23.METHODS[:4], "C024_deg", *c23.METHODS[4:])
PROB_TOL = 8. * np.finfo(np.float64).eps
CORE_ARRAYS = ("center", "fast", "slow", "truth", "enabled", "rest", "native_release", "native_input_last", "source_record_ordinal")


def rows_for_davis(lock):
    # Restrict every numerical load to the six literal external_train names.
    by_name = {item["record"]: item for item in lock["source_records"]}
    manifest = read(DAVIS / "source-files.json")
    protocol = read(DAVIS / "protocol.json")
    manifest_rows = {r["record"]: r for r in manifest["records"] if r["role"] == "external_train"}
    role_rows = {r["record"]: r for r in protocol["records"] if r["role"] == "external_train"}
    base.require(tuple(r["record"] for r in lock["source_records"]) == NAMES
                 and set(by_name) == set(manifest_rows) == set(role_rows) == set(NAMES)
                 and set(lock["sealed_records"]) == {"poster_rotation","poster_translation","poster_6dof"}
                 and lock["unseen_numerical_records"] == 3, "DAVIS role or fixed planned order changed")
    for name in NAMES:
        prepared = ROOT / by_name[name]["prepared"]
        base.require(prepared.resolve() == (DAVIS / "prepared-train" / (name + ".npz")).resolve()
                     and sha(prepared) == by_name[name]["sha256"], "Prepared training data changed")
        item = manifest_rows[name]
        base.require(set(item["members"]) == {"imu.txt","groundtruth.txt"}, "DAVIS input manifest changed")
        for key, digest in item["members"].items():
            path = DAVIS / digest["path"]
            base.require(path.resolve() == (DAVIS / "external_train" / name / key).resolve()
                         and path.stat().st_size == digest["bytes"] and sha(path) == digest["sha256"],
                         "DAVIS member differs or is outside external_train")
    return by_name, manifest_rows


def davis_input(name, meta, fixed_record, loader, manifest):
    base.require(name in NAMES and loader.roles[name] == "external_train" and loader.formal is False, "DAVIS role check failed")
    restored, calibration = loader.native(name)
    path = ROOT / meta["prepared"]
    with np.load(path, allow_pickle=False) as f:
        base.require(set(f.files) == {"six_axis","native_hz","optical_up","mask"}, "Unexpected DAVIS training schema")
        imu, hz, up, mask = (f[k] for k in ("six_axis","native_hz","optical_up","mask"))
    for x, y in zip((imu, hz, up, mask), restored, strict=True):
        np.testing.assert_array_equal(x, y)
    base.require(all(meta[k] == v for k, v in calibration.items()) and imu.shape == (meta["native_rows"],6)
                 and mask.dtype == np.bool_ and mask.sum() == meta["scored_rows"] and fixed_record["scored_frames"] == mask.sum()
                 and array_sha(np.flatnonzero(mask),"<i8") == fixed_record["scoring_indices_sha256"]
                 and fixed_record["fast_deg"] < fixed_record["default_deg"], "Locked DAVIS calibration, mask or reported fixed VQF contrasts changed")
    bound = {key: v["sha256"] for key, v in manifest["members"].items()}
    return imu, float(hz), up, mask, bound


def build_new(cfg, fixed):
    plan = read(DAVIS / "preprocessing-lock.json")
    meta, raw_records = rows_for_davis(plan)
    source_scores = read(DAVIS / "external-fixed-vqf.json")
    base.require(source_scores["protocol_sha256"] == plan["protocol_sha256"], "DAVIS control receipt has another protocol")
    legacy = {r["record"]: r for r in source_scores["rows"]}
    base.require(set(legacy) == set(NAMES), "DAVIS fixed response report leaves six training records")
    indexing = np.arange(STEPS * NEW_BATCH, dtype=np.int64).reshape(STEPS, NEW_BATCH) % len(NAMES)
    counts = np.bincount(indexing.ravel(), minlength=6)
    base.require(counts.tolist() == [214,214,213,213,213,213], "Unexpected fixed DAVIS dose")
    models, mean, std, *_ = fixed
    shape = indexing.shape + (3,)
    emitted = {k: np.empty(shape) for k in ("center","fast","slow","truth")}
    emitted.update({key: np.empty(indexing.shape, dtype=bool) for key in ("enabled","rest")})
    emitted.update({key: np.empty(indexing.shape, dtype=np.int64) for key in ("native_release","native_input_last","source_record_ordinal")})
    emitted["pooled_B"] = np.empty(indexing.shape + (24,), dtype=np.float32)
    emitted["record_equal_weights"] = (1. / (6 * counts[indexing])).astype(np.float64)
    emitted["q_input_last"] = np.empty(indexing.shape + (4,))
    emitted["q_release"] = np.empty(indexing.shape + (4,))
    emitted["q_fast"] = np.empty(indexing.shape + (4,))
    emitted["q_slow"] = np.empty(indexing.shape + (4,))
    details = []
    loader = Davis()
    for j, name in enumerate(NAMES):
        started = perf_counter()
        imu, hz, up, mask, raw_hashes = davis_input(name, meta[name], legacy[name], loader, raw_records[name])
        dt = 1. / hz
        observed = np.isfinite(imu).all(axis=1)
        _, q, at, release, hist, full, acceleration, cal = c17.front_end(imu, observed, dt, cfg)
        frame = intrinsic.build_frame(hist)
        eligible = np.flatnonzero(full & mask[release] & frame.valid)
        base.require(len(eligible) >= counts[j] and np.all(at <= release), f"Too few complete native releases for {name}")
        ordered = sorted(eligible.tolist(), key=lambda i: (hashlib.sha256(f"C024|{name}|{int(release[i])}".encode("utf-8")).digest(), int(release[i])))
        pos = np.asarray(ordered[:counts[j]], dtype=np.int64)
        r = release[pos]
        cuts = at[pos]
        base.require(len(np.unique(r)) == counts[j] and np.all(mask[r] & full[pos] & frame.valid[pos]),
                     "DAVIS selection violates the locked no-replacement sampling rule")
        _, c020, _, rest, valid = c22.base_packet(q, at, release, hist, acceleration, observed, dt, fixed)
        np.testing.assert_array_equal(valid, frame.valid)
        f, s, _, fq, sq, independent, default_gap = c22.filters(imu, observed, acceleration, q, dt, cuts, independence=True, check_default=True)
        feat = c18.features(hist[pos], q[cuts], models[0], mean, std)
        np.testing.assert_array_equal(feat.valid, valid[pos])
        target = c14.rotate(c14.normalize(q[r]), up[r])
        base.require(np.allclose(np.linalg.norm(target, axis=1), 1., rtol=0., atol=2e-12)
                     and np.all(c020[pos,2] == 1.), "DAVIS direction is not in the common corrected VQF world")
        selector = indexing == j
        arrays = {"center": c020[pos], "fast": f, "slow": s, "truth": target, "enabled": valid[pos] & ~rest[pos],
                  "rest": rest[pos], "native_release": r, "native_input_last": cuts, "source_record_ordinal": np.full(len(r),139+j),
                  "pooled_B": feat.pooled[:,1], "q_input_last": q[cuts], "q_release": q[r], "q_fast": fq, "q_slow": sq}
        for key, value in arrays.items():
            emitted[key][selector] = value
        details.append({"record": name, "ordinal": 139+j, "source_role": "external_train", "slots": len(r),
                         "unique_releases": len(np.unique(r)), "eligible_releases": len(eligible), "native_hz": hz,
                         "native_rows": len(imu), "valid_reference_mask_frames": int(mask.sum()), "active_slots": int(arrays["enabled"].sum()),
                         "debias_active_native_frames": int(cal["active_native_frames"]),
                         "all_eligible_release_sha256": array_sha(release[eligible], "<i8"), "selected_release_sha256": array_sha(r, "<i8"),
                         "selected_input_last_sha256": array_sha(cuts, "<i8"), "selected_truth_sha256": array_sha(target,"<f8"),
                         "native_score_index_sha256": legacy[name]["scoring_indices_sha256"], "raw_member_sha256": raw_hashes,
                         "prepared_npz_sha256": meta[name]["sha256"], "C020_center_sha256": array_sha(c020[pos],"<f8"),
                         "B6_feature_sha256": array_sha(feat.pooled[:,1],"<f4"), "independent_response_max_abs": independent,
                         "default_tau3_max_abs": default_gap, "seconds": perf_counter()-started})
        print("C024 DAVIS prepared", name, len(r), len(eligible), int(arrays["enabled"].sum()), flush=True)
    base.require(abs(emitted["record_equal_weights"].sum()-1.) < 1e-15 and np.all(emitted["native_input_last"]<=emitted["native_release"]),
                 "DAVIS record-equal weights or chronology failed")
    archive = HERE/"davis-new-slots.npz"
    np.savez_compressed(archive, **emitted, fixed_cycle_index=indexing, source_names=np.asarray(NAMES))
    return emitted, {"rows":details, "new_slots":STEPS*NEW_BATCH, "six_source_counts":dict(zip(NAMES,counts.tolist(),strict=True)),
                     "prepared_source_array_sha256":sha(archive), "fixed_cycle_sha256":array_sha(indexing,"<i8"),
                     "poster_numeric_file_reads":0, "release_selection": "SHA256 UTF-8 C024|record|native decimal release, without replacement"}

def extend(original, source, x, additional, average, scale):
    def join(a, b):
        return np.concatenate((a.reshape((STEPS, OLD_BATCH)+a.shape[1:]), b), axis=1)
    joined = {key: join(source[key], additional[key]) for key in CORE_ARRAYS}
    joined["legacy_global_window"] = join(original["global_window"], np.full((STEPS, NEW_BATCH), -1, dtype=np.int64))
    joined["record_equal_weights"] = join(source["record_equal_weights"], additional["record_equal_weights"])
    joined["z"] = join(x["z"].numpy(), (additional["pooled_B"].astype(np.float64)-average)/scale)
    joined["domain"] = join(source["domain"], np.full((STEPS, NEW_BATCH), "DAVIS"))
    joined["p0"] = x["prob0"].numpy()
    joined["old_slot"] = np.broadcast_to(np.arange(BATCH)<OLD_BATCH, (STEPS, BATCH)).copy()
    for key in (*CORE_ARRAYS, "record_equal_weights", "domain"):
        np.testing.assert_array_equal(joined[key][:,:OLD_BATCH].reshape(source[key].shape), source[key])
    np.testing.assert_array_equal(joined["z"][:,:OLD_BATCH].reshape(-1,24), x["z"].numpy())
    np.testing.assert_array_equal(joined["legacy_global_window"][:,:OLD_BATCH].ravel(), original["global_window"])
    with (ROOT / "C023/optimizer-steps.jsonl").open(encoding="utf-8") as f:
        logs = [json.loads(s) for s in f]
    base.require(len(logs)==STEPS, "C023 does not have the expected complete prior step ledger")
    manifest = []
    for j, old in enumerate(logs):
        proof = {name + "_sha256": array_sha(joined[name][j,:OLD_BATCH], "<i8")
                 for name in ("native_release","native_input_last","source_record_ordinal")}
        proof["legacy_slot_identity_sha256"] = array_sha(joined["legacy_global_window"][j,:OLD_BATCH],"<i8")
        proof["input_feature_sha256"] = array_sha(joined["z"][j,:OLD_BATCH],"<f8")
        base.require(old["step"]==j+1 and all(old[k]==v for k,v in proof.items()), f"C024 changed C023 step {j+1}")
        manifest.append({"step":j+1,**proof,"new_release_sha256":array_sha(joined["native_release"][j,OLD_BATCH:],"<i8"),
                         "new_ordinal_sha256":array_sha(joined["source_record_ordinal"][j,OLD_BATCH:],"<i8")})
    base.require(np.all(joined["native_input_last"]<=joined["native_release"]) and joined["z"].shape==(STEPS,BATCH,24)
                 and np.isfinite(joined["z"]).all(), "Augmented batch shape or causality changed")
    path = HERE/"training-inputs.npz"
    np.savez_compressed(path,**joined)
    step_file = HERE/"sampling-steps.jsonl"
    with step_file.open("x",encoding="utf-8") as f:
        for row in manifest:
            print(json.dumps(row),file=f)
    return joined, {"training_inputs_sha256":sha(path),"original_96_match_steps":len(manifest),
                    "sampling_steps_sha256":sha(step_file), "original_legacy_window_sha256":array_sha(joined["legacy_global_window"][:,:OLD_BATCH],"<i8")}


def evaluate(model, tensor, use):
    pred, delta, p = c23.position(model, tensor["z"][use], tensor["log0"], tensor["center"][use], tensor["fast"][use], tensor["slow"][use],
                                   tensor["enabled"][use])
    return (pred-tensor["truth"][use]).square().sum(-1), pred, delta, p


def quantiles(p):
    q = np.quantile(p, QUANTILES, axis=0)
    return {k: {"mean":float(p[:,i].mean()), "quantiles":dict(zip(map(str,QUANTILES),q[:,i].tolist(),strict=True))}
            for i,k in enumerate(WEIGHTS)}


def fit(data, protocol, sample_hash, checkpoint23):
    tick = perf_counter()
    model = nn.Linear(24,2,dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    params = tuple(model.parameters())
    base.require(sum(p.numel() for p in params)==50, "Unexpected number of C024 parameters")
    a = {key: arr.reshape((STEPS*BATCH,)+arr.shape[2:]) for key,arr in data.items() if key!="p0"}
    t = {key:torch.from_numpy(a[key]) for key in ("z","center","fast","slow","truth","enabled","record_equal_weights")}
    t["log0"] = c23.prior_probs(protocol["C022_k"]).log()
    np.testing.assert_array_equal(t["log0"].exp().numpy(), checkpoint23["log_p0"].exp().numpy())
    with torch.inference_mode():
        before, direction, delta, prior = evaluate(model,t,slice(None))
    target_k = read(ROOT/"C022/external-selection.json")
    p0 = data["p0"]
    explicit = a["center"] + np.where(a["enabled"][:,None],p0[0]*a["fast"]+p0[1]*a["slow"],0.)
    zero_gap = float(np.max(np.abs(c14.normalize(explicit)-direction.numpy())))
    base.require(zero_gap <= 5e-16 and torch.count_nonzero(delta[~t["enabled"]]).item()==0
                 and np.max(np.abs(prior.numpy()-p0))<=PROB_TOL, "Zero-step C022 routing fails")
    denominator = np.empty(len(before))
    baseline = before.numpy()
    for d in DOMAINS:
        use = a["domain"]==d
        w = a["record_equal_weights"][use]
        val = float(w@baseline[use])
        base.require(abs(w.sum()-1.)<1e-12 and val>0.,"Invalid C024 equal source weight or zero initial loss")
        if d!="DAVIS":
            frozen = target_k["domains"][d]["C022_loss_at_lock"]
            base.require(abs(val-frozen)<=2e-14, "C024 baseline on C023 slots differs from its locked C022 loss")
            val = frozen
        denominator[use]=val
    effective = torch.from_numpy(a["record_equal_weights"]/denominator)
    scale = len(before)/(len(DOMAINS)*BATCH)
    base.require(scale==32. and abs(float(effective@before)/5.-1.)<=5e-12,"C024 five-source objective does not reproduce zero-step")
    init_file=HERE/"zero-step-audit.json"
    write(init_file,{"zero_C022_direction_max_abs_gap":zero_gap,"C022_denominator_by_domain":{d:float(denominator[a["domain"]==d][0]) for d in DOMAINS},
                     "loss_multiplier":scale,"source_weights_sum":{d:float(a["record_equal_weights"][a["domain"]==d].sum()) for d in DOMAINS},
                     "zero_weight_proportions":p0.tolist(), "normalized_full_initial_loss":float(effective@before)/5.})
    optimizer = torch.optim.AdamW(params,lr=LR,weight_decay=WD,foreach=False)
    trace=HERE/"optimizer-steps.jsonl"
    gradients=[]

    def decomposition(term,r):
        grad=torch.autograd.grad(term,params,retain_graph=True)
        vec=torch.cat([u.flatten() for u in grad]).detach().numpy()
        return {"slots":int(r.sum()),"objective":float(term.detach()),"gradient_l2":float(np.linalg.norm(vec)),
                "true_gradient":vec.tolist()}

    with trace.open("x",encoding="utf-8") as out:
        for step in range(STEPS):
            use=slice(step*BATCH,(step+1)*BATCH)
            optimizer.zero_grad(set_to_none=True)
            old=c18.flat_params(model)
            errors, _, _, p = evaluate(model,t,use)
            losses=scale*effective[use]*errors
            loss=losses.sum()
            if step==0:
                base.require(abs(float(loss.detach())-float(scale*(effective[use]@before[use])))<=2e-15, "Initial 104-slot real loss changed")
            i=a["source_record_ordinal"][use]
            six={}
            for j,name in enumerate(NAMES):
                yes=i==139+j
                six[name]=decomposition(losses[yes].sum(),yes)
            legacy=a["old_slot"][use]
            prev=decomposition(losses[legacy].sum(),legacy)
            loss.backward()
            raw=c18.flat_params(model,use_grad=True)
            gap=float(np.max(np.abs(np.sum([a["true_gradient"] for a in six.values()],axis=0)+prev["true_gradient"]-raw)))
            base.require(gap <= 2e-14, "Per-capture true gradients do not reconstruct the actual optimizer update")
            norm=float(torch.nn.utils.clip_grad_norm_(params,CLIP))
            clipped=c18.flat_params(model,use_grad=True)
            optimizer.step()
            after=c18.flat_params(model)
            base.require(np.isfinite(np.r_[float(loss.detach()),norm,raw,after]).all(), "C024 non-finite optimizer step")
            gradients.append([s["gradient_l2"] for s in six.values()])
            row={"pid":os.getpid(),"step":step+1,"objective":float(loss.detach()),"global_gradient_norm":norm,
                 "gradient_before_clip":raw.tolist(),"gradient_after_clip":clipped.tolist(),"weights_before":old.tolist(),"weights_after":after.tolist(),
                 "original_96":prev,"DAVIS_per_capture":six,"gradient_sum_max_abs_difference":gap,
                 "source_loss_contribution":{d:float(losses[a["domain"][use]==d].sum().detach()) for d in DOMAINS},
                 "original_release_sha256":array_sha(data["native_release"][step,:OLD_BATCH],"<i8"),
                 "new_release_sha256":array_sha(data["native_release"][step,OLD_BATCH:],"<i8"),
                 "routing_weights_sha256":array_sha(p.detach().numpy(),"<f8")}
            print(json.dumps(row,allow_nan=False),file=out,flush=True)
            if step+1 in (1,40,80,120,160):
                print("C024 fit",step+1,row["objective"],norm,flush=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        loss,pred,delta,p=evaluate(model,t,slice(None))
        prob=p.numpy()
        err_sum=float(np.max(np.abs(prob.sum(axis=1)-1.)))
        base.require(np.isfinite(prob).all() and np.all(prob>=0.) and err_sum<=PROB_TOL
                     and torch.count_nonzero(delta[~t["enabled"]]).item()==0, "Router probability or gating invariant failed")
        torch.testing.assert_close(pred[~t["enabled"]],c18.unit(t["center"])[~t["enabled"]],atol=0.,rtol=0.)
    final=loss.numpy()
    domain_metrics={}
    for d in DOMAINS:
        sel=a["domain"]==d
        w=a["record_equal_weights"][sel]
        old,new=float(w@baseline[sel]),float(w@final[sel])
        domain_metrics[d]={"slots":int(sel.sum()),"records":len(np.unique(a["source_record_ordinal"][sel])),
                           "C022_chord_loss":old,"C024_chord_loss":new,"relative_error":new/old,
                           "active_prob":quantiles(prob[sel&a["enabled"]])}
    capture_metrics=[]
    grad_by_capture=np.asarray(gradients)
    for j,name in enumerate(NAMES):
        sel=a["source_record_ordinal"]==139+j
        nonzero=grad_by_capture[:,j]>0.
        base.require(nonzero.any(),f"No measured DAVIS contribution from {name}")
        capture_metrics.append({"record":name,"slots":int(sel.sum()),"active_slots":int((sel&a["enabled"]).sum()),
                                "C022_sample_mean_chord":float(baseline[sel].mean()),"C024_sample_mean_chord":float(final[sel].mean()),
                                "steps_with_nonzero_gradient":int(nonzero.sum()),"max_gradient_l2":float(grad_by_capture[:,j].max())})
    path, weights_path = HERE/"training-io.npz",HERE/"router.pt"
    np.savez_compressed(path,trained_prob=prob,baseline_chord=baseline,trained_chord=final,effective_loss_weights=effective.numpy(),
                        enabled=a["enabled"],source_record_ordinal=a["source_record_ordinal"],native_release=a["native_release"])
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"p0":torch.from_numpy(p0),"log_p0":t["log0"],
                "feature_mean":checkpoint23["feature_mean"],"feature_std":checkpoint23["feature_std"],
                "steps":160,"dtype":"float64","sampling_lock_sha256":sample_hash,"protocol_sha256":sha(HERE/"protocol.json")},weights_path)
    return {"pid":os.getpid(),"fit_count":1,"steps":160,"parameters":50,"training_and_archive_seconds":perf_counter()-tick,
            "probability_sum_check":{"actual_max_abs":err_sum,"tolerance_abs":PROB_TOL,"finite":True,"nonnegative":True},
            "domains":domain_metrics,"DAVIS_captures":capture_metrics,"checkpoint_sha256":sha(weights_path),
            "artifacts_sha256":{p.name:sha(p) for p in (trace,init_file,path,weights_path)}}

@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,routers):
    base_direction,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    active=valid & ~rest
    model23,model24,mean,std,prior=routers
    models,B6mean,B6std,*_=fixed
    feat=c18.features(history,q[at],models[0],B6mean,B6std)
    np.testing.assert_array_equal(feat.valid,valid)
    z=torch.from_numpy((feat.pooled[:,1].astype(np.float64)-mean)/std)
    p23,p24=(c23.weights(model,z,prior).numpy() for model in (model23,model24))
    for p in (p23,p24):
        base.require(np.isfinite(p).all() and np.all(p>=0.) and np.max(np.abs(p.sum(1)-1.))<=PROB_TOL,
                     "C024 row has invalid simplex probabilities")
    delta23,delta24=(np.where(active[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.) for p in (p23,p24))
    near=k*np.where(active[:,None],mix,0.)
    base.require(np.max(np.linalg.norm(delta24,axis=1))<=c22.CAP+2e-15,"C024 weighted direction exceeded cap")
    dirs = {"C020_deg":base_direction,"C022_deg":c23.augment(base_direction,center,near),
            "C023_deg":c23.augment(base_direction,center,delta23),"C024_deg":c23.augment(base_direction,center,delta24)}
    return dirs,p23,p24,active,rest,delta24


def metrics(rows):
    ok=[a for a in rows if a["status"]=="scored"]
    macro={k:fmean(a[k] for a in ok) for k in METHODS} if ok else {}
    per_arm={}
    if any(a["active_release_count"] for a in ok):
        for arm in ("C023","C024"):
            per_arm[arm]={k:fmean(a[f"{arm}_p_{k}_mean"] for a in ok if a["active_release_count"]>0) for k in WEIGHTS}
    return {"records":len(ok),"expected_records":len(rows),"frames":sum(a["scored_frames"] for a in ok),
            "macro_tilt_rmse_deg":macro,"mean_active_release_probabilities_record_equal":per_arm,
            "C024_delta_deg":{m:macro["C024_deg"]-macro[m] for m in METHODS if m!="C024_deg"} if ok else {},
            "C024_wins":{m:sum(a["C024_deg"]<a[m] for a in ok) for m in METHODS if m!="C024_deg"}}


def replay(previous,fixed_table,inventory,campaign,split,cfg,k,fixed,routers):
    c1r,c1p,c3r,c2p=(read(ROOT/x) for x in ("C001/results.json","C001/protocol.json","C003/results.json","C002/protocol.json"))
    evidence=c2p["frozen_evidence"]
    map1,map3=({(r["dataset"],r["record"]):r for r in s["sequences"]} for s in (c1r,c3r))
    key_set={(r["dataset"],r["record"]) for r in previous["sequences"]}
    specs={(r["dataset"],r["record"]):r for r in inventory["records"] if (r["dataset"],r["record"]) in key_set}
    comparative={(r["dataset"],r["record"]):r for r in fixed_table["records"]}
    base.require(len(key_set)==213 and set(specs)==set(comparative)==key_set
                 and all(r["side"] in ("development","development_previously_used_for_selection") for r in specs.values())
                 and fixed_table["source_csv_sha256"]==sha(ROOT/"C020/sequences.csv") and fixed_table["selection_lock_sha256"] is None,
                 "C024 development or control scope differs")
    ms,ra=DATA/"Sassari/medium_v5.mat",DATA/"IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(ms)==split["source_file_sha256"][str(ms)] and sha(ra)==evidence["array_reference_sha256"],
                 "A frozen development reference changed")
    medium=loadmat(ms,variable_names=("Qs","XS1","XS2","AP1","AP2","SH1","SH2"))
    array=c2.reference_array(ra)
    reference_poses={(r["record"],r["pose"]):r for r in previous["array_sensor_poses"]}
    rows,poses,failures=[],[],[]
    frozen_methods=c23.METHODS[:4]
    with c2.c1.h5py.File(inventory["yt"]["path"],"r") as yt,np.load(MASKS,allow_pickle=False) as masks,np.load(GATES,allow_pickle=False) as gates:
        for old in previous["sequences"]:
            start=perf_counter()
            key=old["dataset"],old["record"]
            identity={label:old[label] for label in c17.IDENTITY}
            try:
                packet,raw,observed,dt,truth,mask,axes,_=c17.payload(key,specs[key],map3,map1,medium,yt["valid"],
                                                                    split,c1p,array,evidence,cfg)
                q0,q,at,release,hist,_,acc,_=packet
                np.testing.assert_array_equal(mask,masks[old["mask_key"]])
                ix=np.flatnonzero(mask)
                table=comparative[key]
                base.require(len(ix)==old["scored_frames"]==table["scored_frames"]
                             and array_sha(ix,"<i8")==old["scoring_indices_sha256"]==table["scoring_indices_sha256"]
                             and array_sha(at,"<i8")==old["native_input_last_sha256"]
                             and array_sha(release,"<i8")==old["native_release_sha256"], "C024 misaligned native score masks")
                tick=perf_counter()
                dirs,p23,p24,active,rest,delta=held(q,at,release,hist,acc,raw,observed,dt,k,fixed,routers)
                infer_time=perf_counter()-tick
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(array_sha(p23[active],"<f8")==old["active_routing_sha256"]
                             and active.sum()==old["active_release_count"], "Frozen C023 routing output differs")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                values=c17.scores(errs)
                gap=max(abs(values[m]-old[m]) for m in frozen_methods)
                archive={key:table[original] for key,original in c23.FIXED.items()}
                base.require(gap<=1e-6 and all(archive[label]==old[label] for label in c23.FIXED)
                             and np.isfinite(list(values.values())).all(),"Frozen C023/fixed control differs")
                handled=np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C024_deg"][~handled],errs["C020_deg"][~handled])
                detail=[]
                for name,subset in axes.items():
                    use=subset[ix]
                    old_pose=reference_poses[key[1],name]
                    sig=array_sha(ix[use],"<i8")
                    base.require(sig==old_pose["scoring_indices_sha256"] and int(use.sum())==old_pose["scored_frames"],
                                 "Frozen C023 array pose mask differs")
                    val=c17.scores(errs,use)
                    score_error=max(abs(val[m]-old_pose[m]) for m in frozen_methods)
                    base.require(score_error<=1e-6,"Frozen pose scores changed")
                    detail.append({"record":key[1],"pose":name,"scored_frames":int(use.sum()),"scoring_indices_sha256":sig,
                                   **val,"control_max_abs_delta_deg":score_error})
                pmean={f"{arm}_p_{weight}_mean": value for arm,prob in (("C023",p23),("C024",p24))
                       for weight,value in zip(WEIGHTS,prob[active].mean(0).tolist() if active.any() else [None]*3,strict=True)}
                rows.append({**identity,**values,**archive,**pmean,"status":"scored","control_max_abs_deg":gap,
                             "native_input_last_sha256":old["native_input_last_sha256"],"native_release_sha256":old["native_release_sha256"],
                             "active_release_count":int(active.sum()),"active_scored_frames":int(handled.sum()),
                             "inactive_scored_frames":int((~handled).sum()),"C024_delta_sha256":array_sha(delta,"<f8"),
                             "C023_active_routing_sha256":array_sha(p23[active],"<f8"),"C024_active_routing_sha256":array_sha(p24[active],"<f8"),
                             "prediction_seconds":infer_time,"seconds":perf_counter()-start})
                poses.extend(detail)
                print("development",*key,*(values[m] for m in METHODS[:5]),flush=True)
            except (AssertionError,RuntimeError,ValueError,OSError,KeyError,IndexError) as ex:
                failures.append({"dataset":key[0],"record":key[1],"error":repr(ex)})
                rows.append({**identity,"status":"failed","error":repr(ex),"seconds":perf_counter()-start})
                print("FAILED",*key,repr(ex),flush=True)
    seq,pos,group=(HERE/n for n in ("sequences.csv","array-sensor-poses.csv","group-summaries.json"))
    fields=dict.fromkeys(f for r in rows for f in r)
    intrinsic.shice.save_csv(seq,[{f:r.get(f,"") for f in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pos,poses)
    groups={"SHICE_devices":{d:metrics([r for r in rows if r["dataset"]=="SHICE" and r["group"]==d])
                             for d in sorted({r["group"] for r in rows if r["dataset"]=="SHICE"})},
            "TUM_rooms":{r["record"]:metrics([r]) for r in rows if r["dataset"]=="TUM-VI"},
            "Oxford_carry":{g:metrics([r for r in rows if r["dataset"]=="OxIOD" and r["group"]==g])
                            for g in sorted({r["group"] for r in rows if r["dataset"]=="OxIOD"})}}
    write(group,groups)
    return {"pooled_records":metrics(rows),"domains":{d:metrics([r for r in rows if r["dataset"]==d]) for d in campaign["datasets"]},
            **groups,"sequences":rows,"array_sensor_poses":poses,"failures":failures,
            "artifact_sha256":{p.name:sha(p) for p in (seq,pos,group) if p.is_file()}}

def status(protocol, phase, **details):
    write(HERE/"process.json",{"pid":os.getpid(),"original_fit_pid":protocol["pid"],"status":phase,
                              "started_at":protocol["started_at"],"updated_at":stamp(),"protocol_sha256":sha(HERE/"protocol.json"),**details})


def setup(protocol):
    base.require(all(sha(ROOT/n)==v for n,v in protocol["inputs_sha256"].items()),"One frozen C024 input changed")
    fixed,campaign=c22.load_fixed(read(ROOT/"C022/protocol.json"))
    cfg=c3.Settings(**read(ROOT/"C003/protocol.json")["thresholds"])
    return fixed,campaign,cfg


def after_lock(protocol,fixed,campaign,cfg,recovery=None):
    path,ck=HERE/"router-lock.json",HERE/"router.pt"
    lock=read(path)
    sig=sha(path)
    base.require(lock["checkpoint_sha256"]==sha(ck) and lock["training_sha256"]==sha(HERE/"training.json")
                 and lock["sampling_lock_sha256"]==sha(HERE/"sampling-lock.json") and lock["fit_count"]==1
                 and lock["protocol_sha256"]==sha(HERE/"protocol.json")
                 and lock["C023_router_lock_sha256"]==sha(ROOT/"C023/router-lock.json"),
                 "C024 archived model, sampling or training receipt changed")
    old=torch.load(ROOT/"C023/router.pt",map_location="cpu",weights_only=True)
    new=torch.load(ck,map_location="cpu",weights_only=True)
    for field in ("p0","log_p0","feature_mean","feature_std"):
        torch.testing.assert_close(new[field],old[field],rtol=0.,atol=0.)
    base.require(new["protocol_sha256"]==lock["protocol_sha256"] and new["sampling_lock_sha256"]==lock["sampling_lock_sha256"]
                 and new["steps"]==160,"Wrong learned artifact")
    mods=[nn.Linear(24,2,dtype=torch.float64) for _ in range(2)]
    for model,source in zip(mods,(old,new),strict=True):
        model.load_state_dict(source["state_dict"])
        model.eval().requires_grad_(False)
    router=(*mods,new["feature_mean"].numpy(),new["feature_std"].numpy(),new["log_p0"])
    k=protocol["C022_k"]
    tag={"router_lock_sha256":sig,"recovery_sha256":recovery}
    status(protocol,"causal_rotation_audits",**tag)
    audits=HERE/"causal-rotation-audit.json"
    write(audits,{**tag,"probes":c23.audit(cfg,k,fixed,router[1:])})
    status(protocol,"development_replay",**tag)
    print("C024 checkpoint frozen; replay of original 213 records started",flush=True)
    begin=perf_counter()
    replayed=replay(read(ROOT/"C023/results.json"),read(ROOT/"review/vqf_response_controls_development.json"),
                    read(ROOT/"C001/inventory.json"),campaign,read(TRANSFER/"external-split.json"),cfg,k,fixed,router)
    base.require(sha(path)==sig and sha(ck)==lock["checkpoint_sha256"]
                 and all(sha(ROOT/n)==v for n,v in protocol["inputs_sha256"].items()),
                 "C024 locked artifact or baseline input changed during replay")
    replayed.update({**tag,"pid":os.getpid(),"original_fit_pid":protocol["pid"],"started_at":protocol["started_at"],
                     "completed_at":stamp(),"protocol_sha256":sha(HERE/"protocol.json"),"sampling_lock_sha256":lock["sampling_lock_sha256"],
                     "C023_router_lock_sha256":lock["C023_router_lock_sha256"],"scoring_masks_sha256":sha(MASKS),
                     "causal_rotation_audit_sha256":sha(audits),"development_replay_seconds":perf_counter()-begin,
                     "archived_fixed_vqf_controls_reused_with_masks":True,"poster_numeric_file_reads":0,"final_holdout_scored":False})
    output=HERE/"results.json"
    write(output,replayed)
    good=(not replayed["failures"]) and replayed["pooled_records"]["records"]==213 and len(replayed["array_sensor_poses"])==96
    status(protocol,"complete" if good else "incomplete",**tag,scored=replayed["pooled_records"]["records"],results_sha256=sha(output))
    print("C024 finished",replayed["pooled_records"],"failures",replayed["failures"],flush=True)
    if not good:
        raise RuntimeError("C024 incomplete replay; learned weights remain locked")


def main():
    launch,verify=(read(HERE/n) for n in ("launch.json","input-verification.json"))
    files=verify["files_sha256"]
    base.require(os.getpid()==launch["pid"]==verify["pid"] and all(sha(ROOT/n)==v for n,v in files.items()),"Frozen C024 registration changed")
    original,lock22,lock23=(read(ROOT/n) for n in ("C023/protocol.json","C022/coefficient-lock.json","C023/router-lock.json"))
    base.require(original["optimizer"]=={"kind":"AdamW","lr":LR,"weight_decay":WD,"clip_global_norm":CLIP,"steps":160,"batch_size":96}
                 and original["C022_k"]==lock22["k"] and lock23["optimizer_steps"]==160
                 and original["scoring_masks_sha256"]==sha(MASKS) and len(original["development_records"])==213,"C023 fixed recipe or panel changed")
    davis_lock=read(DAVIS/"preprocessing-lock.json")
    base.require(davis_lock["loader_sha256"]==sha(ROOT/"review/davis_handheld.py") and davis_lock["protocol_sha256"]==sha(DAVIS/"protocol.json")
                 and davis_lock["source_bytes_manifest_sha256"]==sha(DAVIS/"source-files.json"),"DAVIS upstream lock no longer matches")
    by_name, manifests=rows_for_davis(davis_lock)
    extra=[ROOT/r["prepared"] for r in by_name.values()]
    extra.extend(DAVIS/m["path"] for r in manifests.values() for m in r["members"].values())
    sources={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in extra}
    recipe={"kind":"AdamW","lr":LR,"weight_decay":WD,"clip_global_norm":CLIP,"steps":160,"batch_size":104,"foreach":False}
    protocol={**launch,"code_sha256":sha(Path(__file__)),"inputs_sha256":{**files,**sources},
              "C022_k":lock22["k"],"C022_coefficient_lock_sha256":files["C022/coefficient-lock.json"],
              "C023_router_lock_sha256":files["C023/router-lock.json"],"gain_prior":"exact C023 softmax p0 and fixed third logit",
              "optimizer":recipe,"dtype":"float64 CPU","parameters":50,"fit_count":1,"original_slots_per_step":96,"DAVIS_slots_per_step":8,
              "fixed_cycle":list(NAMES),"candidate_rule":"Native recording prefix with complete fine input window, official optical mask at release and valid C020 intrinsic frame.",
              "DAVIS_selection":"Each trial independently sorted by SHA256 of UTF-8 C024|record|native release as unpadded decimal; original release number as collision tie break. Take its fixed cycle quota without replacement.",
              "objective":"32 * sum(w_i * native squared chord error_i / full source C022 mean loss). Each source has record-equal weights, five sources total.",
              "feature_normalization":"Frozen C019 history difference normalization, no re-estimation.",
              "scoring_masks_sha256":sha(MASKS),"development_records":original["development_records"],"poster_numeric_file_reads_allowed":0}
    write(HERE/"protocol.json",protocol)
    fixed,campaign,cfg=setup(protocol)
    old_weights=torch.load(ROOT/"C023/router.pt",map_location="cpu",weights_only=True)
    xmean,xstd=old_weights["feature_mean"].numpy(),old_weights["feature_std"].numpy()
    c19=torch.load(ROOT/"C019/readout.pt",map_location="cpu",weights_only=True)
    np.testing.assert_array_equal(xmean,c19["feature_mean"].numpy())
    np.testing.assert_array_equal(xstd,c19["feature_std"].numpy())
    status(protocol,"external_DAVIS_preparation")
    original_slots,source,x=c23.load_train(lock22,files,xmean,xstd)
    new,proof=build_new(cfg,fixed)
    data,schedule=extend(original_slots,source,x,new,xmean,xstd)
    prep=HERE/"training-preparation.json"
    write(prep,{**proof,**schedule})
    sample=HERE/"sampling-lock.json"
    write(sample,{"pid":os.getpid(),"locked_at":stamp(),"protocol_sha256":sha(HERE/"protocol.json"),
                  "preparation_sha256":sha(prep),"original_slots":15360,"DAVIS_slots":1280,
                  "original_96_equal_steps":160,"C023_router_lock_sha256":files["C023/router-lock.json"],**schedule})
    sample_hash=sha(sample)
    status(protocol,"single_augmented_router_fit",sampling_lock_sha256=sample_hash)
    print("C024 sampling locked",sample_hash,proof["six_source_counts"],flush=True)
    training=fit(data,protocol,sample_hash,old_weights)
    train_path,branch=HERE/"training.json",HERE/"router-lock.json"
    write(train_path,{**training,"sampling_lock_sha256":sample_hash})
    with branch.open("x",encoding="utf-8") as f:
        json.dump({"pid":os.getpid(),"locked_at":stamp(),"fit_count":1,"steps":160,
                   "checkpoint_sha256":training["checkpoint_sha256"],"training_sha256":sha(train_path),
                   "protocol_sha256":sha(HERE/"protocol.json"),"sampling_lock_sha256":sample_hash,
                   "C023_router_lock_sha256":files["C023/router-lock.json"]},f,indent=2)
        f.write("\n")
    after_lock(protocol,fixed,campaign,cfg)


def resume():
    protocol,lock,state=(read(HERE/n) for n in ("protocol.json","router-lock.json","process.json"))
    base.require(state["status"]!="complete" and protocol["pid"]==lock["pid"]==state["original_fit_pid"],
                 "Only a previously locked incomplete C024 replay can resume")
    r=HERE/"replay-recovery.json"
    write(r,{"pid":os.getpid(),"original_fit_pid":lock["pid"],"fit_repeated":False,"started_at":stamp(),
             "router_lock_sha256":sha(HERE/"router-lock.json"),
             "prior_failure_evidence_sha256":{n:sha(HERE/n) for n in ("failure.json","failure-traceback.log","results.json") if (HERE/n).is_file()}})
    fixed,campaign,cfg=setup(protocol)
    after_lock(protocol,fixed,campaign,cfg,sha(r))


if __name__=="__main__":
    if "--resume" in sys.argv[1:]:
        resume()
    else:
        main()
