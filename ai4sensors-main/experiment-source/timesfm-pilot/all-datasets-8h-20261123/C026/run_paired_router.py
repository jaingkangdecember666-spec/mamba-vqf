"""C026: append paired EuRoC training slots to the frozen C025 schedule."""
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C025"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

import run_context_router as c25
from euroc_paired_sources import NAMES, paired

c24, c23, c22, c18, c17 = c25.c24, c25.c23, c25.c22, c25.c18, c25.c17
c14, c3, c2 = c25.c14, c25.c3, c25.c2
base, intrinsic = c25.base, c25.intrinsic
read, write, sha, stamp, array_sha = c25.read, c25.write, c25.sha, c25.stamp, c25.array_sha
UP, DATA, TRANSFER, MASKS, GATES = c25.UP, c25.DATA, c25.TRANSFER, c25.MASKS, c25.GATES
DOMAINS, WEIGHTS, PROB_TOL = c25.DOMAINS, c25.WEIGHTS, c25.PROB_TOL
STEPS, EXISTING, BATCH = 160, 104, 110
CONDITIONS = ("raw", "fixed_prefix_bias_removed")
ROUTERS = ("C023", "C024", "C025", "C026")
METHODS = (*c25.METHODS[:6], "C026_deg", *c25.METHODS[6:])
SLOT_FIELDS = ("center", "fast", "slow", "truth", "enabled", "rest",
               "native_release", "native_input_last", "source_record_ordinal")
NORM_KEYS = ("old_feature_mean", "old_feature_std", "context_mean", "context_std")
MULTIPLIER = STEPS / 6
EUROC_WEIGHT = 1 / (2 * len(NAMES) * STEPS)


def require(test, detail):
    base.require(test, detail)


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def weights_state(name):
    return torch.load(ROOT / name / "router.pt", map_location="cpu", weights_only=True)


def status(protocol, phase, **detail):
    write(HERE / "process.json", {
        "pid": os.getpid(), "started_at": protocol["started_at"], "updated_at": stamp(),
        "status": phase, "protocol_sha256": sha(HERE / "protocol.json"), **detail,
    })


def prepare_existing(state):
    old, context, io, prior = (archive(ROOT / path) for path in (
        "C024/training-inputs.npz", "C025/training-context.npz",
        "C024/training-io.npz", "C025/training-io.npz",
    ))
    features = context["features"]
    require(features.shape == (STEPS, EXISTING, 28) and features.dtype == np.float64,
            "C025 feature shape or dtype was changed")
    for key in ("native_release", "native_input_last", "source_record_ordinal", "domain", "record_equal_weights"):
        np.testing.assert_array_equal(old[key], context[key])
    np.testing.assert_array_equal(features[:, :, :24], old["z"])
    np.testing.assert_array_equal(state["context_mean"].numpy(), context["new_feature_mean"])
    np.testing.assert_array_equal(state["context_std"].numpy(), context["new_feature_std"])
    np.testing.assert_array_equal(
        features[:, :, 24:],
        (context["context_raw"] - state["context_mean"].numpy()) / state["context_std"].numpy(),
    )
    for name, expected_size in zip(NORM_KEYS, (24, 24, 4, 4), strict=True):
        value = state[name].numpy()
        require(value.shape == (expected_size,) and np.isfinite(value).all() and
                ("std" not in name or np.all(value > 0)), "Invalid frozen C025 normalizer")
    for name in ("p0", "log_p0"):
        require(state[name].shape == (3,) and torch.isfinite(state[name]).all(), "Invalid C022 prior")
    np.testing.assert_array_equal(old["p0"], state["p0"].numpy())

    denominators = read(ROOT / "C024/zero-step-audit.json")["C022_denominator_by_domain"]
    c25_initial = read(ROOT / "C025/zero-step-audit.json")
    require(c25_initial["C022_denominator_by_domain"] == denominators and set(denominators) == set(DOMAINS),
            "Original source normalization was changed")
    weights = io["effective_loss_weights"]
    expected = old["record_equal_weights"].ravel() / np.array([denominators[d] for d in old["domain"].ravel()])
    np.testing.assert_array_equal(weights, expected)
    require(array_sha(weights, "<f8") == c25_initial["effective_slot_loss_weights_sha256"],
            "C025 effective loss weights differ")
    np.testing.assert_array_equal(prior["C022_chord"], io["baseline_chord"])
    for key in ("native_release", "native_input_last", "source_record_ordinal"):
        np.testing.assert_array_equal(prior[key], old[key])
    np.testing.assert_array_equal(prior["active"], old["enabled"].ravel())
    with (ROOT / "C025/optimizer-steps.jsonl").open(encoding="utf-8") as handle:
        steps = [json.loads(line) for line in handle]
    require(len(steps) == STEPS and np.unique(old["source_record_ordinal"]).size == 145,
            "C025 schedule receipt or source count changed")
    for j, row in enumerate(steps):
        fingerprints = {
            "original_release_sha256": array_sha(old["native_release"][j, :c25.OLD], "<i8"),
            "new_release_sha256": array_sha(old["native_release"][j, c25.OLD:], "<i8"),
            "native_input_last_sha256": array_sha(old["native_input_last"][j], "<i8"),
            "source_record_ordinal_sha256": array_sha(old["source_record_ordinal"][j], "<i8"),
            "normalized_features_sha256": array_sha(features[j], "<f8"),
        }
        require(row["step"] == j + 1 and all(row[key] == value for key, value in fingerprints.items()),
                f"C025 step {j + 1} does not match the archived fingerprints")
    return old, context, weights.reshape(STEPS, EXISTING), prior["C022_chord"], denominators


def eligible_pair(name, cfg):
    inputs = paired(name)
    original, altered = inputs
    require(original[1] == altered[1] == 200., "EuRoC native rate changed")
    np.testing.assert_array_equal(original[2], altered[2])
    np.testing.assert_array_equal(original[3], altered[3])
    np.testing.assert_array_equal(original[0][:, :3], altered[0][:, :3])
    require(not original[3][:2000].any(), "EuRoC prefix used for training supervision")
    chains = []
    for raw, rate, up, mask in inputs:
        observed = np.isfinite(raw).all(axis=1)
        dt = 1 / rate
        _, q, at, release, history, complete, acc, calibration = c17.front_end(raw, observed, dt, cfg)
        valid = intrinsic.build_frame(history).valid
        require(np.all(at <= release) and np.all(np.diff(release) > 0), "Noncausal or repeated release")
        chains.append((raw, dt, up, mask, observed, q, at, release, history, complete, acc, calibration, valid))
    first, second = chains
    np.testing.assert_array_equal(first[6], second[6])
    release = first[7]
    np.testing.assert_array_equal(release, second[7])
    eligible = np.flatnonzero(first[3][release] & first[9] & second[9] & first[12] & second[12])
    require(len(eligible) >= STEPS, f"Fewer than 160 shared eligible releases for {name}")
    order = sorted(eligible.tolist(), key=lambda i: (release_key(name, int(release[i])), int(release[i])))
    selected = np.asarray(order[:STEPS], dtype=np.int64)
    require(np.unique(release[selected]).size == STEPS, f"Repeated training release for {name}")
    return chains, eligible, selected


def release_key(name, release):
    return hashlib.sha256(f"C026|{name}|{release}".encode("utf-8")).digest()


def make_condition(chain, selected, ordinal, state, fixed):
    raw, dt, up, _, observed, q, at, release, history, _, acc, calibration, frame_valid = chain
    _, center, _, rest, valid = c22.base_packet(q, at, release, history, acc, observed, dt, fixed)
    np.testing.assert_array_equal(valid, frame_valid)
    cuts, emitted = at[selected], release[selected]
    fast, slow, _, q_fast, q_slow, independent, default = c22.filters(
        raw, observed, acc, q, dt, cuts, independence=True, check_default=True,
    )
    models, bmean, bstd, *_ = fixed
    feat = c18.features(history[selected], q[cuts], models[0], bmean, bstd)
    np.testing.assert_array_equal(feat.valid, valid[selected])
    context = c25.from_frontend(raw, observed, acc, dt)[cuts]
    norm, scale, ctx_mean, ctx_scale = (state[key].numpy() for key in NORM_KEYS)
    features = np.column_stack(((feat.pooled[:, 1].astype(np.float64) - norm) / scale,
                                (context - ctx_mean) / ctx_scale))
    truth = c14.rotate(c14.normalize(q[emitted]), up[emitted])
    require(np.isfinite(features).all() and np.all(context >= 0) and
            np.allclose(np.linalg.norm(truth, axis=1), 1., rtol=0., atol=2e-12) and
            np.all(center[selected, 2] == 1.), "EuRoC features or world directions are invalid")
    out = dict(center=center[selected], fast=fast, slow=slow, truth=truth, enabled=valid[selected] & ~rest[selected],
               rest=rest[selected], native_release=emitted, native_input_last=cuts,
               source_record_ordinal=np.full(STEPS, ordinal, dtype=np.int64), features=features, context_raw=context,
               q_input_last=q[cuts], q_release=q[emitted], q_fast=q_fast, q_slow=q_slow,
               body_reference=up[emitted])
    detail = {"active_slots": int(out["enabled"].sum()), "native_sixaxis_sha256": array_sha(raw, "<f8"),
              "normalized_features_sha256": array_sha(features, "<f8"), "truth_sha256": array_sha(truth, "<f8"),
              "causal_acceleration_debias_active_frames": int(calibration["active_native_frames"]),
              "independent_response_max_abs": independent, "default_tau3_max_abs": default}
    return out, detail


def prepare_pairs(cfg, state, fixed):
    slots, records = [], []
    audit_records = {row["record"]: row for row in read(ROOT / "review/euroc_bias_response.json")["records"]}
    for ordinal, name in enumerate(NAMES, start=145):
        chains, eligible, selected = eligible_pair(name, cfg)
        release = chains[0][7]
        emitted = release[selected]
        details, pair = {}, []
        for condition, chain in zip(CONDITIONS, chains, strict=True):
            packet, details[condition] = make_condition(chain, selected, ordinal, state, fixed)
            pair.append(packet)
        for field in ("native_release", "native_input_last", "source_record_ordinal", "body_reference"):
            np.testing.assert_array_equal(pair[0][field], pair[1][field])
        slots.extend(pair)
        record = audit_records[name]
        records.append({
            "record": name, "ordinal": ordinal, "unique_paired_releases": len(emitted), "eligible_releases": len(eligible),
            "common_eligible_release_rows": release[eligible].tolist(),
            "common_eligible_sha256": array_sha(release[eligible], "<i8"),
            "common_optical_indices_sha256": array_sha(np.flatnonzero(chains[0][3]), "<i8"),
            "prefix_bias_deg_s": record["prefix_bias_deg_s"],
            "source_npz_sha256": record["original_source_sha256"], "source_hdf5_sha256": record["conversion_sha256"],
            "selected": [{"step": j + 1, "native_release": int(r), "native_input_last": int(a),
                          "literal_sha256": release_key(name, int(r)).hex()}
                         for j, (r, a) in enumerate(zip(emitted, pair[0]["native_input_last"], strict=True))],
            "conditions": details,
        })
        print("C026 paired releases", name, len(emitted), len(eligible), os.getpid(), flush=True)
    appended = {key: np.stack([item[key] for item in slots], axis=1) for key in slots[0]}
    np.savez_compressed(HERE / "euroc-paired-inputs.npz", **appended)
    write(HERE / "sampling-lock.json", {
        "pid": os.getpid(), "locked_at": stamp(), "selection_key": "C026|<name>|<decimal native release>",
        "ordering": "SHA256 digest lexicographic, then numeric native release; no replacement",
        "eligible": "Shared post-prefix optical mask and complete intrinsic windows in both chains",
        "slot_order": [{"record": n, "condition": c} for n in NAMES for c in CONDITIONS],
        "paired_inputs_sha256": sha(HERE / "euroc-paired-inputs.npz"), "records": records,
    })
    return appended


def require_simplex(prob):
    require(np.isfinite(prob).all() and np.all(prob >= 0) and
            np.max(np.abs(prob.sum(axis=1) - 1.)) <= PROB_TOL, "Invalid router probabilities")


def fit(old, context, weights, old_initial, denominators, appended, state):
    start = perf_counter()
    values = {key: np.concatenate((old[key], appended[key]), axis=1) for key in SLOT_FIELDS}
    features = np.concatenate((context["features"], appended["features"]), axis=1)
    domains = np.concatenate((old["domain"], np.full((STEPS, 6), "EuRoC")), axis=1)
    conditions = np.full((STEPS, BATCH), "frozen", dtype="<U32")
    conditions[:, EXISTING:] = list(CONDITIONS) * 3
    for key in SLOT_FIELDS:
        np.testing.assert_array_equal(values[key][:, :EXISTING], old[key])
    np.testing.assert_array_equal(features[:, :EXISTING], context["features"])
    require(features.shape == (160, 110, 28) and np.unique(values["source_record_ordinal"]).size == 148,
            "C026 slot schedule does not contain exactly 17,600 slots and 148 recordings")
    inputs = {key: torch.from_numpy(values[key].reshape((-1,) + values[key].shape[2:]))
              for key in ("center", "fast", "slow", "truth", "enabled")}
    x = torch.from_numpy(features.reshape(-1, 28))
    model = nn.Linear(28, 2, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    params = tuple(model.parameters())
    require(sum(p.numel() for p in params) == 58, "Unexpected router parameter count")

    def forward(rows):
        pred, delta, prob = c23.position(model, x[rows], state["log_p0"], inputs["center"][rows],
                                        inputs["fast"][rows], inputs["slow"][rows], inputs["enabled"][rows])
        return (pred - inputs["truth"][rows]).square().sum(-1), pred, delta, prob

    whole = slice(None)
    with torch.inference_mode():
        zero, _, delta, prob = forward(whole)
        zero = zero.numpy().reshape(STEPS, BATCH)
        discrepancy = float(np.max(np.abs(zero[:, :EXISTING].ravel() - old_initial)))
        require(discrepancy <= 3e-17 and torch.count_nonzero(delta[~inputs["enabled"]]).item() == 0 and
                torch.max((prob - state["p0"]).abs()).item() <= PROB_TOL, "Zero step does not reproduce C022")
    new_denominators = {condition: float(zero[conditions == condition].mean()) for condition in CONDITIONS}
    require(all(np.isfinite(d) and d > 0 for d in new_denominators.values()), "Invalid EuRoC zero-step denominators")
    effective = np.empty((STEPS, BATCH), dtype=np.float64)
    effective[:, :EXISTING] = weights
    for condition, denominator in new_denominators.items():
        effective[conditions == condition] = EUROC_WEIGHT / denominator
    np.testing.assert_array_equal(effective[:, :EXISTING], weights)
    w = torch.from_numpy(effective)
    old_mass = {d: float(old["record_equal_weights"][old["domain"] == d].sum()) for d in DOMAINS}
    new_mass = {c: float((conditions == c).sum() * EUROC_WEIGHT) for c in CONDITIONS}
    require(all(abs(v - 1.) <= 2e-12 for v in old_mass.values()) and all(v == .5 for v in new_mass.values()) and
            abs(float(np.sum(effective * zero)) / 6. - 1.) <= 3e-14, "C026 source shares are not normalized")
    write(HERE / "zero-step-audit.json", {
        "pid": os.getpid(), "frozen_source_denominators": denominators, "EuRoC_condition_denominators": new_denominators,
        "old_slot_C022_max_abs_chord_error": discrepancy, "old_source_mass": old_mass, "EuRoC_condition_mass": new_mass,
        "normalized_zero_step_objective": float(np.sum(effective * zero)) / 6, "loss_multiplier": MULTIPLIER,
        "old_effective_weights_sha256": array_sha(weights, "<f8"),
        "effective_weights_sha256": array_sha(effective, "<f8"),
    })
    np.savez_compressed(HERE / "training-inputs.npz", **values, features=features, domain=domains, condition=conditions,
                        effective_loss_weights=effective, c022_chord=zero)
    optimizer = torch.optim.AdamW(params, lr=.01, weight_decay=.01, foreach=False)
    gradient_peaks = dict.fromkeys(CONDITIONS, 0.)
    max_gradient_gap = 0.
    with (HERE / "optimizer-steps.jsonl").open("x", encoding="utf-8") as handle:
        for j in range(STEPS):
            optimizer.zero_grad(set_to_none=True)
            before = c18.flat_params(model)
            losses, _, _, probabilities = forward(slice(j * BATCH, (j + 1) * BATCH))
            weighted = MULTIPLIER * w[j] * losses
            objective = weighted.sum()
            components = {"frozen": weighted[:EXISTING].sum(),
                          **{c: weighted[conditions[j] == c].sum() for c in CONDITIONS}}
            gradients = {name: torch.cat([g.reshape(-1) for g in torch.autograd.grad(
                component, params, retain_graph=True,
            )]).detach().numpy() for name, component in components.items()}
            objective.backward()
            gradient = c18.flat_params(model, use_grad=True)
            gap = float(np.max(np.abs(sum(gradients.values()) - gradient)))
            require(gap <= 2e-14, f"Condition gradients do not sum to the actual step {j + 1} gradient")
            max_gradient_gap = max(max_gradient_gap, gap)
            for c in CONDITIONS:
                gradient_peaks[c] = max(gradient_peaks[c], float(np.linalg.norm(gradients[c])))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(params, 2.))
            after_clip = c18.flat_params(model, use_grad=True)
            optimizer.step()
            after = c18.flat_params(model)
            require(np.isfinite(np.r_[after, gradient, gradient_norm, float(objective.detach())]).all(),
                    f"Nonfinite optimizer step {j + 1}")
            row = {
                "step": j + 1, "pid": os.getpid(), "objective": float(objective.detach()),
                "gradient_l2": gradient_norm, "gradient_before_clip": gradient.tolist(),
                "gradient_after_clip": after_clip.tolist(), "weights_before": before.tolist(),
                "weights_after": after.tolist(), "actual_update": (after - before).tolist(),
                "domain_contribution": {d: float(weighted[domains[j] == d].sum().detach()) for d in (*DOMAINS, "EuRoC")},
                "optimizer_gradient_components": {name: g.tolist() for name, g in gradients.items()},
                "objective_components": {name: float(v.detach()) for name, v in components.items()},
                "gradient_component_max_abs_gap": gap,
                "native_input_last_sha256": array_sha(values["native_input_last"][j], "<i8"),
                "native_release_sha256": array_sha(values["native_release"][j], "<i8"),
                "C025_release_sha256": array_sha(values["native_release"][j, :EXISTING], "<i8"),
                "EuRoC_release_sha256": array_sha(values["native_release"][j, EXISTING:], "<i8"),
                "source_record_ordinal_sha256": array_sha(values["source_record_ordinal"][j], "<i8"),
                "normalized_features_sha256": array_sha(features[j], "<f8"),
                "routing_probability_sha256": array_sha(probabilities.detach().numpy(), "<f8"),
            }
            print(json.dumps(row, allow_nan=False), file=handle, flush=True)
            if j + 1 in (1, 40, 80, 120, 160):
                print("C026 fit", j + 1, row["objective"], gradient_norm, flush=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        final, pred, delta, prob = forward(whole)
        inactive = ~inputs["enabled"]
        torch.testing.assert_close(pred[inactive], c18.unit(inputs["center"])[inactive], rtol=0., atol=0.)
        require(torch.count_nonzero(delta[inactive]).item() == 0, "Router bypassed the original gate")
    final, prob = final.numpy().reshape(STEPS, BATCH), prob.numpy()
    require_simplex(prob)
    torch.save({**{key: state[key] for key in (*NORM_KEYS, "p0", "log_p0", "feature_names")},
                "state_dict": model.state_dict(), "steps": STEPS, "dtype": "float64",
                "C025_checkpoint_sha256": sha(ROOT / "C025/router.pt"),
                "sampling_lock_sha256": sha(HERE / "sampling-lock.json")}, HERE / "router.pt")
    np.savez_compressed(HERE / "training-io.npz", C022_chord=zero, C026_chord=final,
                        C026_prob=prob.reshape(STEPS, BATCH, 3), active=values["enabled"],
                        domain=domains, condition=conditions)
    summary = {}
    for label, subset in [(d, domains == d) for d in DOMAINS] + [(c, conditions == c) for c in CONDITIONS]:
        active = subset & values["enabled"]
        natural = (np.where(subset[:, :EXISTING], old["record_equal_weights"], 0.).sum()
                   if label in DOMAINS else new_mass[label])
        summary[label] = {
            "slots": int(subset.sum()), "active_slots": int(active.sum()), "mass": float(natural),
            "C022_normalized_loss": float(np.sum(effective[subset] * zero[subset]) / natural),
            "C026_normalized_loss": float(np.sum(effective[subset] * final[subset]) / natural),
            "active_probabilities": c24.quantiles(prob[active.ravel()]) if active.any() else {},
        }
    return {"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "parameters": 58, "slots": STEPS * BATCH,
            "source_records": 148, "seconds": perf_counter() - start, "grad_component_max_abs_gap": max_gradient_gap,
            "EuRoC_condition_gradient_l2_peak": gradient_peaks, "groups": summary,
            "probability_sum_max_abs_error": float(np.max(np.abs(prob.sum(1) - 1.)))}



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
    context=c25.from_frontend(raw,observed,acc,dt)[at]
    z=np.concatenate((f0,(context-ctx_norm[0])/ctx_norm[1]),axis=1)
    probs={name:c23.weights(model,torch.from_numpy(features),log0).numpy()
           for (name,model),features in zip(models.items(),(f0,f0,z,z),strict=True)}
    dirs={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(on[:,None],mix,0.))}
    shifts={}
    for name,p in probs.items():
        base.require(np.isfinite(p).all() and np.all(p>=0.) and np.max(np.abs(p.sum(1)-1.))<=PROB_TOL,
                     "C026 probabilities violate simplex")
        shift=np.where(on[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        dirs[name+"_deg"]=c23.augment(original,center,shift)
        shifts[name]=shift
    base.require(np.max(np.linalg.norm(shifts["C026"],axis=1))<=c22.CAP+2e-15,"C026 combined filter correction exceeded the cap")
    return dirs,probs,on,rest,context



def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ROUTERS} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C026_delta_deg":{k:macro["C026_deg"]-macro[k] for k in METHODS if k!="C026_deg"} if good else {},
            "C026_wins":{k:sum(r["C026_deg"]<r[k] for r in good) for k in METHODS if k!="C026_deg"}}


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
                 "C026 replay or fixed controls leave development scope")
    mp,ap=DATA/"Sassari/medium_v5.mat",DATA/"IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(sha(mp)==split["source_file_sha256"][str(mp)] and sha(ap)==evidence["array_reference_sha256"],"Frozen reference changed")
    medium=loadmat(mp,variable_names=("Qs","XS1","XS2","AP1","AP2","SH1","SH2"))
    array=c2.reference_array(ap)
    poses_old={(r["record"],r["pose"]):r for r in previous["array_sensor_poses"]}
    seq,poses,failed,ledger=[],[],[],[]
    frozen=c25.METHODS[:6]
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
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ROUTERS[:-1]),
                             "Archived C023, C024, or C025 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C026_deg"][same],errs["C020_deg"][same])
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
                print("development",*key,score["C023_deg"],score["C025_deg"],score["C026_deg"],flush=True)
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



def replay_locked(protocol, fixed, campaign, cfg):
    lock = read(HERE / "router-lock.json")
    lock_hash = sha(HERE / "router-lock.json")
    require(lock["fit_count"] == 1 and lock["pid"] == os.getpid() and
            lock["checkpoint_sha256"] == sha(HERE / "router.pt"), "Replay requires the original locked C026 fit")
    require(all(sha(HERE / file) == digest for file, digest in lock["artifact_sha256"].items()),
            "A C026 locked training artifact changed")
    states = [weights_state(name) for name in ROUTERS]
    own = states[-1]
    models = {}
    for name, state, width in zip(ROUTERS, states, (24, 24, 28, 28), strict=True):
        linear = nn.Linear(width, 2, dtype=torch.float64)
        linear.load_state_dict(state["state_dict"])
        models[name] = linear.eval().requires_grad_(False)
        for key in ("p0", "log_p0"):
            torch.testing.assert_close(own[key], state[key], rtol=0., atol=0.)
        for suffix in ("mean", "std"):
            old_key = ("old_feature_" if width == 28 else "feature_") + suffix
            torch.testing.assert_close(own["old_feature_" + suffix], state[old_key], rtol=0., atol=0.)
    for key in NORM_KEYS:
        torch.testing.assert_close(own[key], states[-2][key], rtol=0., atol=0.)
    norm, ctx_norm = tuple(own[k].numpy() for k in NORM_KEYS[:2]), tuple(own[k].numpy() for k in NORM_KEYS[2:])
    previous, control, inventory, split = (read(path) for path in (
        ROOT / "C025/results.json", ROOT / "review/vqf_response_controls_development.json",
        ROOT / "C001/inventory.json", TRANSFER / "external-split.json",
    ))
    require([[row["dataset"], row["record"]] for row in previous["sequences"]] == protocol["development_records"],
            "Development replay order changed since C025")
    status(protocol, "development_replay", router_lock_sha256=lock_hash, expected_records=213)
    print("C026 locked; replaying 213 development records", flush=True)
    start = perf_counter()
    result = replay(previous, control, inventory, campaign, split, cfg, protocol["C022_k"], fixed,
                    models, norm, ctx_norm, own["log_p0"])
    require(sha(HERE / "router-lock.json") == lock_hash and sha(HERE / "router.pt") == lock["checkpoint_sha256"] and
            all(sha(ROOT / file) == digest for file, digest in protocol["inputs_sha256"].items()),
            "A replay control or checkpoint changed")
    result.update(pid=os.getpid(), original_fit_pid=lock["pid"], router_lock_sha256=lock_hash,
                  protocol_sha256=sha(HERE / "protocol.json"), scoring_masks_sha256=sha(MASKS),
                  replay_seconds=perf_counter() - start, completed_at=stamp(), final_holdout_scored=False)
    results = HERE / "results.json"
    write(results, result)
    scored = result["pooled_records"]["records"]
    complete = scored == 213 and not result["failures"] and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if complete else "incomplete", scored=scored, router_lock_sha256=lock_hash,
           results_sha256=sha(results))
    require(complete, "C026 replay incomplete; checkpoint remains locked")
    print("C026 finished", scored, result["pooled_records"]["macro_tilt_rmse_deg"], flush=True)


def main():
    launch, checked = (read(HERE / name) for name in ("launch.json", "input-verification.json"))
    files = checked["files_sha256"]
    require(os.getpid() == launch["pid"] == checked["pid"] and
            all(sha(ROOT / file) == digest for file, digest in files.items()),
            "C026 inherited registration or input verification changed")
    require(NAMES == ("V1_01_easy", "V1_02_medium", "V1_03_difficult"), "Disallowed training source")
    previous = read(ROOT / "C025/protocol.json")
    recipe = {"kind": "AdamW", "lr": .01, "weight_decay": .01, "clip_global_norm": 2.,
              "steps": STEPS, "batch_size": EXISTING, "foreach": False}
    k = read(ROOT / "C022/coefficient-lock.json")["k"]
    require(previous["optimizer"] == recipe and previous["C022_k"] == k and
            previous["scoring_masks_sha256"] == sha(MASKS) and len(previous["development_records"]) == 213,
            "C025 inherited optimizer recipe, C022 center, or development panel changed")
    for key in ("zero-step-audit.json", "optimizer-steps.jsonl", "training-io.npz", "router.pt"):
        file = "C025/" + key
        digest = read(ROOT / "C025/training.json")["artifacts_sha256"][key]
        require(sha(ROOT / file) == digest and files.get(file, digest) == digest, "C025 training evidence changed")
        files[file] = digest
    state = weights_state("C025")
    write(HERE / "normalizers.json", {
        "source_checkpoint_sha256": files["C025/router.pt"],
        "frozen_dimensions": 28, "observer_names": state["feature_names"],
        **{key: {"values": state[key].tolist(), "sha256": array_sha(state[key].numpy(), "<f8")} for key in NORM_KEYS},
    })
    protocol = {**launch, "worker_sha256": sha(Path(__file__)), "inputs_sha256": files,
                "optimizer": {**recipe, "batch_size": BATCH}, "C022_k": k, "zero_initialized": True,
                "input_width": 28, "parameters": 58, "fits": 1, "old_slots_per_step": EXISTING,
                "normalizers_sha256": sha(HERE / "normalizers.json"), "development_records": previous["development_records"],
                "training_sources": [*DOMAINS, "EuRoC"], "euroc_records": list(NAMES),
                "euroc_condition_unnormalized_slot_weight": EUROC_WEIGHT,
                "objective": "160/6 times sum of frozen old or condition-normalized EuRoC slot weights times squared chord",
                "euroc_augmentation": "Training only, using raw and first ten seconds' fixed reference-input bias removal",
                "development_input": "Native six-axis IMU, causal online debias and physical observers; no reference-input bias removal",
                "scoring_masks_sha256": sha(MASKS), "formal_holdouts_authorized": False}
    write(HERE / "protocol.json", protocol)
    torch.set_num_threads(2)
    status(protocol, "locking_training_samples")
    old, context, effective, prior_chord, denom = prepare_existing(state)
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    fixed, campaign = c22.load_fixed(read(ROOT / "C022/protocol.json"))
    new = prepare_pairs(cfg, state, fixed)
    sampling_hash = sha(HERE / "sampling-lock.json")
    status(protocol, "single_fit", sampling_lock_sha256=sampling_hash)
    training = fit(old, context, effective, prior_chord, denom, new, state)
    write(HERE / "training.json", training)
    paths = ("euroc-paired-inputs.npz", "sampling-lock.json", "normalizers.json", "training-inputs.npz",
             "zero-step-audit.json", "optimizer-steps.jsonl", "training-io.npz", "training.json", "protocol.json")
    with (HERE / "router-lock.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "locked_at": stamp(), "steps": STEPS, "fit_count": 1,
                   "checkpoint_sha256": sha(HERE / "router.pt"), "sampling_lock_sha256": sampling_hash,
                   "artifact_sha256": {name: sha(HERE / name) for name in paths}}, handle, indent=2)
        handle.write("\n")
    replay_locked(protocol, fixed, campaign, cfg)


if __name__ == "__main__":
    main()

