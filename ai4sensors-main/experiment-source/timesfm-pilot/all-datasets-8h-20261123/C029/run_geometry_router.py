"""C029: add only two causal response-geometry inputs to C028."""
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C028"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F

import run_response_teacher as c28
from causal_response_geometry import NAMES as GEOMETRY_NAMES, measures

c26, c25, c23, c22, c18, c17 = c28.c26, c28.c25, c28.c23, c28.c22, c28.c18, c28.c17
c3, c2, base, intrinsic = c28.c3, c28.c2, c28.base, c28.intrinsic
read, write, sha, stamp, array_sha = c28.read, c28.write, c28.sha, c28.stamp, c28.array_sha
UP, DATA, TRANSFER, MASKS, GATES = c28.UP, c28.DATA, c28.TRANSFER, c28.MASKS, c28.GATES
WEIGHTS, PROB_TOL, NORM_KEYS = c28.WEIGHTS, c28.PROB_TOL, c28.NORM_KEYS
STEPS, BATCH, OPTIMIZER = c28.STEPS, c28.BATCH, c28.OPTIMIZER
GROUPS, SOURCES, CONDITIONS, SHARES = c28.GROUPS, c28.SOURCES, c28.CONDITIONS, c28.SHARES
archive, require, weights_state = c28.archive, c28.require, c28.weights_state
training_metrics, angular_degrees = c28.training_metrics, c28.angular_degrees
ROUTERS = ("C023", "C028", "C029")
FROZEN_SCORES = ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C028_deg")
METHODS = (*FROZEN_SCORES, "C029_deg", *c23.FIXED)


def status(protocol, phase, **details):
    write(HERE / "process.json", {
        "pid": os.getpid(), "started_at": protocol["started_at"], "updated_at": stamp(),
        "status": phase, "protocol_sha256": sha(HERE / "protocol.json"), **details,
    })


def verify_inputs():
    previous, receipt, lock = [read(ROOT / "C028" / n) for n in ("protocol.json", "process.json", "router-lock.json")]
    require(receipt["status"] == "complete" and receipt["scored"] == 213 and lock["fit_count"] == 1 and lock["steps"] == STEPS == 160
            and previous["optimizer"] == OPTIMIZER and previous["input_width"] == 28 and previous["parameters"] == 58
            and previous["zero_initialized"] and previous["slots"] == 17600 and previous["recordings"] == 148
            and previous["scoring_masks_sha256"] == sha(MASKS) and previous["worker_sha256"] == sha(ROOT / "C028/run_response_teacher.py"),
            "C028 schedule, architecture, or optimizer did not match")
    inputs = {**previous["inputs_sha256"], **{f"C028/{n}": h for n, h in lock["artifact_sha256"].items()},
              "C028/router-lock.json": receipt["router_lock_sha256"], "C028/router.pt": lock["checkpoint_sha256"],
              "C028/results.json": receipt["results_sha256"], "C028/run_response_teacher.py": previous["worker_sha256"]}
    for n in ("C028/process.json", "review/causal_response_geometry.py", "review/geometry_source_preflight.py",
              "review/geometry_source_preflight.json"):
        inputs[n] = sha(ROOT / n)
    require(all(sha(ROOT / name) == digest for name, digest in inputs.items()), "A frozen C028 artifact changed")
    geometry = read(ROOT / "review/geometry_source_preflight.json")
    require(geometry["source_sha256"] == inputs["C026/training-inputs.npz"] and
            tuple(geometry["names"]) == GEOMETRY_NAMES, "Geometry review describes a different source archive")
    return previous, inputs, geometry


def geometric_features(fast, slow, normalized_bias, mean, std):
    raw = measures(fast, slow, normalized_bias)
    normalized = (raw - mean) / std
    require(np.isfinite(normalized).all() and np.all(raw[:, 0] >= 0.), "Invalid fast/slow response-angle features")
    return raw, normalized


def lock_training_data(data, inputs, preflight):
    locked = archive(ROOT / "C028/teacher-slots.npz")
    teacher_lock = read(ROOT / "C028/teacher-lock.json")
    require(teacher_lock["teacher_slots_sha256"] == inputs["C028/teacher-slots.npz"] and
            teacher_lock["original_slot_archive_sha256"] == inputs["C026/training-inputs.npz"],
            "Teacher labels came from a different source archive")
    old = data["features"]
    require(old.shape == (STEPS, BATCH, 28) and old.dtype == np.float64 and np.isfinite(old).all(), "C028 feature archive changed")
    for k in ("enabled", "rest", "domain", "condition", "source_record_ordinal", "native_input_last", "native_release"):
        np.testing.assert_array_equal(data[k], locked[k])
    labels, ce_weights, groups = (locked[k].ravel() for k in ("label", "global_weight", "group"))
    np.testing.assert_array_equal(groups, np.where(data["domain"] == "EuRoC", data["condition"], data["domain"]).ravel())
    require(len(labels) == 17600 and set(groups) == set(GROUPS) and np.isfinite(ce_weights).all() and
            abs(ce_weights.sum() - 1.) <= 5e-16 and np.all((labels >= 0) & (labels < 3)) and
            array_sha(labels, "<i8") == teacher_lock["label_sha256"] and
            array_sha(ce_weights, "<f8") == teacher_lock["global_weight_sha256"], "C028 teaching labels or weights changed")
    np.testing.assert_array_equal(ce_weights > 0., locked["enabled"].ravel() & (locked["confidence_deg2"].ravel() > 0.))
    with (ROOT / "C028/optimizer-steps.jsonl").open(encoding="utf-8") as handle:
        prior_steps = [json.loads(line) for line in handle]
    require(len(prior_steps) == STEPS, "C028 steps missing")
    for j, row in enumerate(prior_steps):
        require(row["step"] == j + 1 and row["source_batch_index"] == j and all(
            row[k] == array_sha(v, dtype) for k, v, dtype in (
                ("features_sha256", old[j], "<f8"),
                ("labels_sha256", locked["label"][j], "<i8"),
                ("global_weights_sha256", locked["global_weight"][j], "<f8"),
                ("source_record_ordinal_sha256", data["source_record_ordinal"][j], "<i8"),
                ("native_release_sha256", data["native_release"][j], "<i8"),
                ("native_input_last_sha256", data["native_input_last"][j], "<i8"),
            )), f"C028 ordered slot fingerprint changed at step {j+1}")

    z = old.reshape(-1, 28)
    fast, slow = (data[k].reshape(-1, 3) for k in ("fast", "slow"))
    require(np.all(fast[:, 2] == 0.) and np.all(slow[:, 2] == 0.) and
            max(np.linalg.norm(fast, axis=1).max(), np.linalg.norm(slow, axis=1).max()) <= c22.CAP + 2e-15,
            "The geometry inputs must be the existing capped filter responses")
    obs = measures(fast, slow, z[:, -1])
    geom_weight = np.zeros(len(z), dtype=np.float64)
    ordinals = data["source_record_ordinal"].ravel()
    details = {}
    require(np.unique(ordinals).size == 148, "Acquisition count differs")
    for g in GROUPS:
        selected = groups == g
        records = np.unique(ordinals[selected])
        details[g] = {"recordings": len(records), "slots": int(selected.sum()), "mass_per_record": SHARES[g] / len(records)}
        for i in records:
            use = selected & (ordinals == i)
            geom_weight[use] = SHARES[g] / (len(records) * use.sum())
        details[g]["geometry_mass"] = float(geom_weight[selected].sum())
        require(abs(details[g]["geometry_mass"] - SHARES[g]) <= 2e-16, "Geometry mass differs by source or condition")
    require(np.all(geom_weight > 0.) and abs(geom_weight.sum() - 1.) <= 5e-16,
            "Every slot must take part in geometry normalization")
    mean = geom_weight @ obs
    std = np.sqrt(geom_weight @ (obs - mean)**2)
    np.testing.assert_allclose(mean, preflight["source_mean"], rtol=0., atol=2e-12)
    np.testing.assert_allclose(std, preflight["source_std"], rtol=0., atol=2e-12)
    _, normalized = geometric_features(fast, slow, z[:, -1], mean, std)
    design = np.column_stack((np.ones(len(z)), z))
    mass_root = np.sqrt(geom_weight[:, None])
    projection = np.linalg.lstsq(design * mass_root, normalized * mass_root, rcond=None)[0]
    residual = normalized - design @ projection
    unexplained = geom_weight @ residual**2
    np.testing.assert_allclose(unexplained, preflight["fraction_of_feature_variance_remaining_after_projection_on_old28"],
                               rtol=0., atol=2e-12)
    cols = normalized.reshape(STEPS, BATCH, 2)
    combined = np.concatenate((old, cols), axis=2)
    np.testing.assert_array_equal(combined[:, :, :28], old)
    geometry_path = HERE / "geometry-training.npz"
    np.savez_compressed(geometry_path, features=combined, geometry_raw=obs.reshape(STEPS, BATCH, 2),
                        geometry_normalized=cols, geometry_stat_weight=geom_weight.reshape(STEPS, BATCH),
                        geometry_mean=mean, geometry_std=std)
    write(HERE / "geometry-lock.json", {
        "pid": os.getpid(), "locked_at": stamp(), "geometry_names": list(GEOMETRY_NAMES),
        "geometry_mean": mean.tolist(), "geometry_std": std.tolist(), "geometry_groups": details,
        "normalization": "Equal six-source mass, equal original acquisition mass within source/condition, equal weight per slot of acquisition; EuRoC conditions each half",
        "normalization_includes_all_17600_slots": True, "uses_confidence_for_geometry_normalization": False,
        "linear_projection_diagnostic_residual_fraction": unexplained.tolist(), "projection_used_for_fit_features": False,
        "geometry_archive_sha256": sha(geometry_path), "geometry_stat_weight_sha256": array_sha(geom_weight, "<f8"),
        "uncorrected_C028_28_columns_sha256": array_sha(old, "<f8"), "appended_features_sha256": array_sha(combined, "<f8"),
        "inherited_label_sha256": teacher_lock["label_sha256"], "inherited_CE_weight_sha256": teacher_lock["global_weight_sha256"],
        "C028_teacher_lock_sha256": inputs["C028/teacher-lock.json"], "C028_teacher_slots_sha256": inputs["C028/teacher-slots.npz"],
        "C028_order_160_batches_verified": True,
    })
    return combined, (labels, ce_weights, groups), (mean, std), prior_steps



def fit(data, baseline, teacher_data, state, expanded, geometry_norm, previous_steps):
    started = perf_counter()
    labels, weights, groups = teacher_data
    label = torch.from_numpy(labels)
    weight = torch.from_numpy(weights)
    x = torch.from_numpy(expanded.reshape(-1, 30))
    model = nn.Linear(30, 2, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    params = tuple(model.parameters())
    require(sum(p.numel() for p in params) == 62 and not c18.flat_params(model).any(), "The fit must start with 62 zeros")
    log0 = state["log_p0"]
    center, fast, slow, truth = (data[k].reshape(-1, 3) for k in ("center", "fast", "slow", "truth"))
    active = data["enabled"].ravel()

    def values(part):
        logits = torch.cat((model(x[part]) + log0[:2], log0[2].expand(len(x[part]), 1)), dim=1)
        return F.cross_entropy(logits, label[part], reduction="none"), logits

    center_t, fast_t, slow_t, truth_t, active_t = (torch.from_numpy(v) for v in (center, fast, slow, truth, active))

    @torch.inference_mode()
    def predict(prob):
        p = torch.from_numpy(prob)
        correction = (p[:, :1] * fast_t + p[:, 1:2] * slow_t).masked_fill(~active_t[:, None], 0.)
        require(float(torch.linalg.vector_norm(correction, dim=1).max()) <= c22.CAP + 2e-15 and
                torch.count_nonzero(correction[:, 2]).item() == 0,
                "Training diagnostic violates the deployed convex-response cap")
        pred = c18.unit(center_t + correction)
        torch.testing.assert_close(pred[~active_t], c18.unit(center_t)[~active_t], rtol=0., atol=0.)
        return pred.numpy(), (pred - truth_t).square().sum(-1).numpy()

    with torch.inference_mode():
        ce0, zero_logits = values(slice(None))
        ce0 = ce0.numpy()
        p0 = torch.softmax(zero_logits, dim=1).numpy()
        np.testing.assert_allclose(p0, np.broadcast_to(state["p0"].numpy(), p0.shape), rtol=0., atol=2e-16)
        p0dir, p0chord = predict(p0)
        zero_chord_gap = float(np.max(np.abs(p0chord - data["c022_chord"].ravel())))
        require(zero_chord_gap <= 3e-17 and np.isfinite(ce0).all(), "Zero-step direction differs from frozen C022")
    np.testing.assert_array_equal(ce0, baseline["C022_CE"].ravel())
    np.testing.assert_array_equal(p0, baseline["C022_p"].reshape(-1, 3))
    old_prob = baseline["C028_p"].reshape(-1, 3)
    old_ce = baseline["C028_CE"].ravel()
    baseline_direction, _ = predict(old_prob)
    np.testing.assert_array_equal(baseline_direction, baseline["C028_direction"].reshape(-1, 3))
    angle0 = angular_degrees(p0dir, truth)
    angle28 = angular_degrees(baseline_direction, truth)
    np.testing.assert_array_equal(angle28, baseline["C028_tilt_deg"].ravel())
    before = training_metrics(data, labels, weights, groups, {"C022_zero": (ce0, p0, angle0), "C028": (old_ce, old_prob, angle28)})
    write(HERE / "zero-step-audit.json", {
        "pid": os.getpid(), "C022_log_prior": log0.tolist(), "C022_prior": state["p0"].tolist(),
        "C022_chord_reproduction_max_abs": zero_chord_gap, "C028_zero_probabilities_and_cross_entropy_exact": True,
        "artifacts": {"teacher_lock_sha256": sha(ROOT/"C028/teacher-lock.json")},
        "optimizer_initial_62_parameters": c18.flat_params(model).tolist(), "training_before": before,
    })
    c028_flat = torch.cat([state["state_dict"][k].ravel() for k in ("weight", "bias")]).numpy()
    optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=.01, foreach=False)
    max_split_gap, max_gradient_gap = 0., 0.
    with (HERE / "optimizer-steps.jsonl").open("x", encoding="utf-8") as log:
        for j in range(STEPS):
            part = slice(j*BATCH, (j+1)*BATCH)
            optimizer.zero_grad(set_to_none=True)
            before_step = c18.flat_params(model)
            ce, logits = values(part)
            weighted = STEPS * (weight[part] * ce)
            objective = weighted.sum()
            parts = {g: weighted[groups[part] == g].sum() for g in GROUPS}
            split_gap = abs(float(objective.detach()) - sum(float(p.detach()) for p in parts.values()))
            gradients = {g: torch.cat([v.ravel() for v in torch.autograd.grad(
                subtotal, params, retain_graph=True,
            )]).detach().numpy() for g, subtotal in parts.items()}
            objective.backward()
            gradient = c18.flat_params(model, use_grad=True)
            if j == 0:
                common = np.r_[gradient[:60].reshape(2, 30)[:, :28].ravel(), gradient[-2:]]
                np.testing.assert_allclose(common, previous_steps[0]["gradient_before_clip"], rtol=0., atol=2e-14)
                require(abs(float(objective.detach()) - previous_steps[0]["objective"]) <= 2e-15,
                        "C029 zero step changed the C028 first minibatch objective")
            gradient_gap = float(np.max(np.abs(sum(gradients.values()) - gradient)))
            require(split_gap <= 2e-14 and gradient_gap <= 2e-14, "Domain or condition objective/gradient decomposition failed")
            max_split_gap = max(max_split_gap, split_gap)
            max_gradient_gap = max(max_gradient_gap, gradient_gap)
            length = float(torch.nn.utils.clip_grad_norm_(params, 2.))
            clipped = c18.flat_params(model, use_grad=True)
            optimizer.step()
            after = c18.flat_params(model)
            require(np.isfinite(np.r_[after, gradient, length, float(objective.detach())]).all(), "Nonfinite optimizer update")
            probabilities = torch.softmax(logits.detach(), dim=1).numpy()
            c26.require_simplex(probabilities)
            book = {}
            for g in GROUPS:
                marked = groups[part] == g
                book[g] = {"slots": int(marked.sum()), "positive_weight_slots": int(np.count_nonzero(weights[part][marked])),
                           "mass": float(weights[part][marked].sum()), "objective_contribution": float(parts[g].detach()),
                           "gradient": gradients[g].tolist()}
            domains = {g: book[g]["objective_contribution"] for g in SOURCES[:-1]}
            domains["EuRoC"] = sum(book[g]["objective_contribution"] for g in CONDITIONS)
            row = {
                "pid": os.getpid(), "step": j+1, "source_batch_index": j, "objective": float(objective.detach()),
                "objective_multiplier": STEPS, "batch_weight_mass": float(weights[part].sum()),
                "gradient_before_clip": gradient.tolist(), "gradient_after_clip": clipped.tolist(), "gradient_l2": length,
                "weights_before": before_step.tolist(), "weights_after": after.tolist(), "actual_update": (after-before_step).tolist(),
                "domain_objective_contribution": domains, "domain_or_EuRoC_condition_ledger": book,
                "decomposition_objective_gap": split_gap, "decomposition_gradient_max_abs_gap": gradient_gap,
                "source_record_ordinal_sha256": array_sha(data["source_record_ordinal"][j], "<i8"),
                "native_release_sha256": array_sha(data["native_release"][j], "<i8"),
                "native_input_last_sha256": array_sha(data["native_input_last"][j], "<i8"),
                "features_sha256": array_sha(expanded[j], "<f8"),
                "inherited_28_features_sha256": array_sha(data["features"][j], "<f8"),
                "added_2_features_sha256": array_sha(expanded[j, :, 28:], "<f8"),
                "labels_sha256": array_sha(labels[part], "<i8"), "global_weights_sha256": array_sha(weights[part], "<f8"),
                "routing_probabilities_sha256": array_sha(probabilities, "<f8"),
            }
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            if j+1 in (1, 40, 80, 120, 160):
                print("C029 fit", j+1, row["objective"], length, flush=True)

    model.eval().requires_grad_(False)
    with torch.inference_mode():
        ce, logits = values(slice(None))
        after_ce = ce.numpy()
        p29 = torch.softmax(logits, dim=1).numpy()
        np.testing.assert_array_equal(p29, c23.weights(model, x, log0).numpy())
    c26.require_simplex(p29)
    p29dir, _ = predict(p29)
    angle29 = angular_degrees(p29dir, truth)
    modes = {"C022_zero": (ce0, p0, angle0), "C028": (old_ce, old_prob, angle28), "C029": (after_ce, p29, angle29)}
    shape = (STEPS, BATCH)
    np.savez_compressed(HERE/"training-io.npz", C022_p=p0.reshape(*shape,3), C028_p=old_prob.reshape(*shape,3),
                        C029_p=p29.reshape(*shape,3), C022_CE=ce0.reshape(shape), C028_CE=old_ce.reshape(shape),
                        C029_CE=after_ce.reshape(shape), C022_tilt_deg=angle0.reshape(shape),
                        C028_tilt_deg=angle28.reshape(shape), C029_tilt_deg=angle29.reshape(shape),
                        C029_direction=p29dir.reshape(*shape,3))
    own_weights = model.weight.detach().numpy()
    difference = np.r_[own_weights[:, :28].ravel(), model.bias.detach().numpy()] - c028_flat
    geometry_mean, geometry_std = geometry_norm
    torch.save({**state, "state_dict": model.state_dict(), "protocol_sha256": sha(HERE/"protocol.json"),
                "teacher_lock_sha256": sha(ROOT/"C028/teacher-lock.json"), "baseline_C028_checkpoint_sha256": sha(ROOT/"C028/router.pt"),
                "geometry_mean": torch.from_numpy(geometry_mean), "geometry_std": torch.from_numpy(geometry_std),
                "geometry_names": list(GEOMETRY_NAMES), "geometry_lock_sha256": sha(HERE/"geometry-lock.json")},
               HERE/"router.pt")
    return {"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "parameters": 62,
            "slots": STEPS*BATCH, "original_recordings": 148, "seconds": perf_counter()-started,
            "global_objective_multiplier": STEPS, "objective_decomposition_max_abs_gap": max_split_gap,
            "gradient_decomposition_max_abs_gap": max_gradient_gap, "C029_minus_C028_shared_final_params": difference.tolist(), "geometry_final_coefficients": own_weights[:, 28:].tolist(),
            "parameter_delta_l2": float(np.linalg.norm(difference)),
            "parameter_delta_max_abs": float(np.max(np.abs(difference))),
            "training": training_metrics(data, labels, weights, groups, modes)}





@torch.inference_mode()
def held(q,at,release,history,acc,raw,observed,dt,k,fixed,models,feature_norm,ctx_norm,geometry_norm,log0):
    original,center,_,rest,valid=c22.base_packet(q,at,release,history,acc,observed,dt,fixed)
    fast,slow,mix,*_=c22.filters(raw,observed,acc,q,dt,at)
    on=valid&~rest
    inherited,bmean,bstd,*_=fixed
    feat=c18.features(history,q[at],inherited[0],bmean,bstd)
    np.testing.assert_array_equal(feat.valid,valid)
    norms,scales=feature_norm
    f0=(feat.pooled[:,1].astype(np.float64)-norms)/scales
    context=c25.from_frontend(raw,observed,acc,dt)[at]
    z=np.concatenate((f0,(context-ctx_norm[0])/ctx_norm[1]),axis=1)
    geom_raw,geom_z=geometric_features(fast,slow,z[:,27],*geometry_norm)
    expanded=np.column_stack((z,geom_z))
    probs={name:c23.weights(model,torch.from_numpy(features),log0).numpy()
           for (name,model),features in zip(models.items(),(f0,z,expanded),strict=True)}
    dirs={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(on[:,None],mix,0.))}
    shifts={}
    for name,p in probs.items():
        base.require(np.isfinite(p).all() and np.all(p>=0.) and np.max(np.abs(p.sum(1)-1.))<=PROB_TOL,
                     "C029 probabilities violate simplex")
        shift=np.where(on[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        dirs[name+"_deg"]=c23.augment(original,center,shift)
        shifts[name]=shift
    base.require(np.max(np.linalg.norm(shifts["C029"],axis=1))<=c22.CAP+2e-15,"C029 combined filter correction exceeded the cap")
    return dirs,probs,on,rest,context,geom_raw,geom_z,z[:,27]



def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ROUTERS} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C029_delta_deg":{k:macro["C029_deg"]-macro[k] for k in METHODS if k!="C029_deg"} if good else {},
            "C029_wins":{k:sum(r["C029_deg"]<r[k] for r in good) for k in METHODS if k!="C029_deg"}}


def replay(previous,control,inventory,campaign,split,cfg,k,fixed,models,norm,ctx_norm,geometry_norm,log0):
    c1r,c1p,c3r,c2p=(read(ROOT/n) for n in ("C001/results.json","C001/protocol.json","C003/results.json","C002/protocol.json"))
    evidence=c2p["frozen_evidence"]
    map1,map3=({(r["dataset"],r["record"]):r for r in s["sequences"]} for s in (c1r,c3r))
    keys={(r["dataset"],r["record"]) for r in previous["sequences"]}
    spec={(r["dataset"],r["record"]):r for r in inventory["records"] if (r["dataset"],r["record"]) in keys}
    table={(r["dataset"],r["record"]):r for r in control["records"]}
    base.require(len(keys)==213 and set(spec)==keys==set(table)
                 and all(r["side"] in ("development","development_previously_used_for_selection") for r in spec.values())
                 and control["source_csv_sha256"]==sha(ROOT/"C020/sequences.csv") and control["selection_lock_sha256"] is None,
                 "C029 replay or fixed controls leave development scope")
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
                             and array_sha(release,"<i8")==old["native_release_sha256"],"C028 or fixed native indices changed")
                run=perf_counter()
                dirs,probs,active,rest,context,geom_raw,geom_z,bias_z=held(q,at,release,w,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,geometry_norm,log0)
                calc=perf_counter()-run
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(active.sum()==old["active_release_count"] and
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ROUTERS[:-1]),
                             "Archived C023 or C028 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C029_deg"][same],errs["C020_deg"][same])
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
                            "C029_active_geometry_raw_sha256":array_sha(geom_raw[active],"<f8"),
                            "C029_active_geometry_standardized_sha256":array_sha(geom_z[active],"<f8"),
                            "C029_active_fast_slow_angle_mean_deg":float(geom_raw[active,0].mean()) if active.any() else None,
                            "prediction_seconds":calc,"seconds":perf_counter()-start})
                ledger.append({"ordinal":np.full(active.sum(),i),"release":release[active],"input_last":at[active],
                               "context":context[active],"geometry_raw":geom_raw[active],"geometry_normalized":geom_z[active],
                               "normalized_gyro_bias":bias_z[active],**{m:p[active] for m,p in probs.items()}})
                poses.extend(detail)
                print("development",*key,score["C023_deg"],score["C028_deg"],score["C029_deg"],flush=True)
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
    lock = read(HERE / "router-lock.json")
    lock_hash = sha(HERE / "router-lock.json")
    require(lock["fit_count"] == 1 and lock["pid"] == os.getpid() and
            lock["checkpoint_sha256"] == sha(HERE/"router.pt") and
            all(sha(HERE / n) == h for n, h in lock["artifact_sha256"].items()),
            "The original locked C029 must be used for replay")
    states = [weights_state(name) for name in ROUTERS]
    incumbent, prior, own = states
    models = {}
    for name, state, width in zip(ROUTERS, states, (24, 28, 30), strict=True):
        model = nn.Linear(width, 2, dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name] = model.eval().requires_grad_(False)
        for key in ("p0", "log_p0"):
            torch.testing.assert_close(own[key], state[key], rtol=0., atol=0.)
    for key in NORM_KEYS:
        torch.testing.assert_close(own[key], prior[key], rtol=0., atol=0.)
    for key in ("mean", "std"):
        torch.testing.assert_close(own["old_feature_" + key], incumbent["feature_" + key], rtol=0., atol=0.)
    geom_lock = read(HERE/"geometry-lock.json")
    require(tuple(own["geometry_names"]) == GEOMETRY_NAMES and tuple(own["feature_names"]) == c25.OBSERVERS and
            own["geometry_lock_sha256"] == sha(HERE/"geometry-lock.json"), "Geometry checkpoint signature differs")
    geometry_norm = tuple(own[k].numpy() for k in ("geometry_mean", "geometry_std"))
    for key, value in zip(("geometry_mean", "geometry_std"), geometry_norm, strict=True):
        np.testing.assert_array_equal(value, geom_lock[key])
    mean, std, ctx_mean, ctx_std = (own[k].numpy() for k in NORM_KEYS)
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    fixed, campaign = c22.load_fixed(read(ROOT / "C022/protocol.json"))
    previous, control, inventory, split = (read(p) for p in (
        ROOT / "C028/results.json", ROOT / "review/vqf_response_controls_development.json",
        ROOT / "C001/inventory.json", TRANSFER / "external-split.json",
    ))
    require([[r["dataset"], r["record"]] for r in previous["sequences"]] == protocol["development_records"],
            "C028 panel or replay order changed")
    status(protocol, "development_replay", router_lock_sha256=lock_hash, expected_records=213)
    print("C029 locked; replaying the common 213 development records from native inputs", flush=True)
    start = perf_counter()
    result = replay(previous, control, inventory, campaign, split, cfg, protocol["C022_k"], fixed,
                    models, (mean, std), (ctx_mean, ctx_std), geometry_norm, own["log_p0"])
    require(sha(HERE/"router-lock.json") == lock_hash and sha(HERE/"router.pt") == lock["checkpoint_sha256"] and
            all(sha(ROOT / n) == h for n, h in protocol["inputs_sha256"].items()), "Frozen evidence changed")
    result.update(pid=os.getpid(), original_fit_pid=lock["pid"], router_lock_sha256=lock_hash,
                  protocol_sha256=sha(HERE/"protocol.json"), scoring_masks_sha256=sha(MASKS),
                  geometry_lock_sha256=own["geometry_lock_sha256"], replay_seconds=perf_counter()-start,
                  completed_at=stamp(), final_holdout_scored=False, supplemental_holdout_scored=False)
    destination = HERE / "results.json"
    write(destination, result)
    scored = result["pooled_records"]["records"]
    complete = scored == 213 and not result["failures"] and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if complete else "incomplete", scored=scored,
           router_lock_sha256=lock_hash, results_sha256=sha(destination))
    require(complete, "C029 replay incomplete; checkpoint remains locked")
    print("C029 complete", scored, result["pooled_records"]["macro_tilt_rmse_deg"], flush=True)


def run(launch):
    previous, inputs, preflight = verify_inputs()
    state = weights_state("C028")
    k = read(ROOT / "C022/coefficient-lock.json")["k"]
    torch.testing.assert_close(state["p0"], c23.prior_probs(k), rtol=0., atol=0.)
    torch.testing.assert_close(state["log_p0"], state["p0"].log(), rtol=0., atol=0.)
    require(state["steps"] == STEPS and k == previous["C022_k"] and tuple(state["feature_names"]) == c25.OBSERVERS,
            "C028 observer names or C022 prior changed")
    original = weights_state("C026")
    for key in NORM_KEYS:
        torch.testing.assert_close(state[key], original[key], rtol=0., atol=0.)
    protocol = {**launch, "inputs_sha256": inputs, "worker_sha256": sha(Path(__file__)),
                "optimizer": OPTIMIZER, "fit_count": 1, "C022_k": k, "parameters": 62,
                "input_width": 30, "zero_initialized": True, "slots": STEPS*BATCH, "recordings": 148,
                "feature_names": state["feature_names"], "geometry_names": list(GEOMETRY_NAMES),
                "sole_change": "Two appended causal columns: angle between up+fast and up+slow, and its product with existing normalized gyro bias",
                "objective": previous["objective"], "training_only": previous["training_only"],
                "development_input": previous["development_input"], "formal_holdouts_authorized": False,
                "supplemental_holdouts_authorized": False, "development_records": previous["development_records"],
                "scoring_masks_sha256": sha(MASKS)}
    write(HERE/"protocol.json", protocol)
    torch.set_num_threads(2)
    status(protocol, "locking_geometry_and_C028_inputs")
    data = archive(ROOT / "C026/training-inputs.npz")
    baseline = archive(ROOT / "C028/training-io.npz")
    expanded, teacher_data, geometry_norm, previous_steps = lock_training_data(data, inputs, preflight)
    status(protocol, "single_fit", geometry_lock_sha256=sha(HERE/"geometry-lock.json"))
    training = fit(data, baseline, teacher_data, state, expanded, geometry_norm, previous_steps)
    write(HERE/"training.json", training)
    artifacts = ("protocol.json", "geometry-lock.json", "geometry-training.npz", "zero-step-audit.json",
                 "training.json", "training-io.npz", "optimizer-steps.jsonl")
    with (HERE/"router-lock.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "locked_at": stamp(),
                   "checkpoint_sha256": sha(HERE/"router.pt"),
                   "artifact_sha256": {n: sha(HERE/n) for n in artifacts}}, handle, indent=2)
        handle.write("\n")
    replay_locked(protocol)


def main():
    launch = {"pid": os.getpid(), "started_at": stamp(), "command": "python -B -u C029/run_geometry_router.py",
              "scope": "C029 only; append two observables to C028, one fit, and common development replay"}
    with (HERE/"launch.json").open("x", encoding="utf-8") as handle:
        json.dump(launch, handle, indent=2)
        handle.write("\n")
    try:
        run(launch)
    except BaseException as error:
        write(HERE/"failure.json", {**launch, "failed_at": stamp(), "error": repr(error)})
        (HERE/"failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

