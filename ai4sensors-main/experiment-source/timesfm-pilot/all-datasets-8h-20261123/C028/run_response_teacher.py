"""C028: train the frozen C026 router architecture on source-only response labels."""
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "C026"), str(ROOT / "review")]

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.nn import functional as F

import run_paired_router as c26
from external_response_teacher import teacher

c25, c23, c22, c18, c17 = c26.c25, c26.c23, c26.c22, c26.c18, c26.c17
c14, c3, c2, base, intrinsic = c26.c14, c26.c3, c26.c2, c26.base, c26.intrinsic
read, write, sha, stamp, array_sha = c26.read, c26.write, c26.sha, c26.stamp, c26.array_sha
UP, DATA, TRANSFER, MASKS, GATES = c26.UP, c26.DATA, c26.TRANSFER, c26.MASKS, c26.GATES
WEIGHTS, PROB_TOL, NORM_KEYS = c26.WEIGHTS, c26.PROB_TOL, c26.NORM_KEYS
STEPS, BATCH = c26.STEPS, c26.BATCH
ROUTERS = ("C023", "C026", "C028")
FROZEN_SCORES = ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "C026_deg")
METHODS = (*FROZEN_SCORES, "C028_deg", *c23.FIXED)
SOURCES = (*c26.DOMAINS, "EuRoC")
CONDITIONS = c26.CONDITIONS
GROUPS = (*c26.DOMAINS, *CONDITIONS)
SHARES = {g: 1. / (12 if g in CONDITIONS else 6) for g in GROUPS}
OPTIMIZER = dict(kind="AdamW", lr=.01, weight_decay=.01, clip_global_norm=2., steps=160, batch_size=110, foreach=False)
archive, require, weights_state = c26.archive, c26.require, c26.weights_state


def status(protocol, phase, **details):
    write(HERE / "process.json", {
        "pid": os.getpid(), "started_at": protocol["started_at"], "updated_at": stamp(),
        "status": phase, "protocol_sha256": sha(HERE / "protocol.json"), **details,
    })


def verify_inputs():
    previous, receipt, lock = [read(ROOT / "C026" / n) for n in ("protocol.json", "process.json", "router-lock.json")]
    require(receipt["status"] == "complete" and receipt["scored"] == 213 and lock["fit_count"] == 1 and lock["steps"] == STEPS == 160
            and previous["optimizer"] == OPTIMIZER and previous["input_width"] == 28 and previous["parameters"] == 58
            and previous["zero_initialized"] and tuple(previous["training_sources"]) == SOURCES
            and previous["scoring_masks_sha256"] == sha(MASKS)
            and previous["worker_sha256"] == sha(ROOT / "C026/run_paired_router.py"),
            "C026 architecture, optimizer or protocol did not match")
    inputs = {**{f"C026/{n}": h for n, h in lock["artifact_sha256"].items()},
              **{n: h for n, h in previous["inputs_sha256"].items()
                 if not n.startswith("external-sources/euroc-")},
              "C026/router-lock.json": receipt["router_lock_sha256"],
              "C026/router.pt": lock["checkpoint_sha256"],
              "C026/results.json": receipt["results_sha256"],
              "C026/run_paired_router.py": previous["worker_sha256"],
              "C022/training-slots.npz": read(ROOT / "C022/coefficient-lock.json")["training_slots_sha256"],
              "C024/davis-new-slots.npz": read(ROOT / "C024/training-preparation.json")["prepared_source_array_sha256"]}
    for n in ("review/external_response_teacher.py", "review/external_response_teacher.json", "C026/process.json"):
        inputs[n] = sha(ROOT / n)
    require(all(sha(ROOT / name) == digest for name, digest in inputs.items()),
            "C026 or another frozen input differs from its receipt")
    review = read(ROOT / "review/external_response_teacher.json")
    require(review["inputs_sha256"] == inputs["C026/training-inputs.npz"] and
            review["router_sha256"] == inputs["C026/router.pt"], "Teacher review is from a different C026 archive")
    return previous, inputs, review


def original_names(data):
    """Resolve names through the 139 C022, six DAVIS, and three paired EuRoC identities."""
    old = archive(ROOT / "C022/training-slots.npz")
    davis = archive(ROOT / "C024/davis-new-slots.npz")
    euroc = read(ROOT / "C026/sampling-lock.json")
    keys = ("native_release", "native_input_last", "source_record_ordinal")
    for key in keys:
        np.testing.assert_array_equal(data[key][:, :96].ravel(), old[key])
        np.testing.assert_array_equal(data[key][:, 96:104], davis[key])
    mapping = {}
    for ordinal, name, domain in zip(old["source_record_ordinal"], old["source_record_name"], old["domain"], strict=True):
        item = str(name), str(domain)
        require(mapping.setdefault(int(ordinal), item) == item, "Ambiguous original recorded name")
    for j, name in enumerate(davis["source_names"], start=139):
        mapping[j] = str(name), "DAVIS"
    for j, record in enumerate(euroc["records"]):
        ordinal = record["ordinal"]
        require(ordinal == 145 + j, "EuRoC original recording identity changed")
        mapping[ordinal] = record["record"], "EuRoC"
        releases = [p["native_release"] for p in record["selected"]]
        cuts = [p["native_input_last"] for p in record["selected"]]
        for slot, condition in enumerate(CONDITIONS, start=104 + 2*j):
            np.testing.assert_array_equal(data["native_release"][:, slot], releases)
            np.testing.assert_array_equal(data["native_input_last"][:, slot], cuts)
            require(np.all(data["source_record_ordinal"][:, slot] == ordinal) and
                    np.all(data["condition"][:, slot] == condition), "EuRoC pair or order changed")
    require(set(mapping) == set(np.unique(data["source_record_ordinal"])) == set(range(148)),
            "Expected 148 original records spanning six training sources")
    names = np.array([mapping[i][0] for i in range(148)])[data["source_record_ordinal"]]
    domains = np.array([mapping[i][1] for i in range(148)])[data["source_record_ordinal"]]
    np.testing.assert_array_equal(domains, data["domain"])
    return names, mapping


def lock_labels(data, c026, review, inputs):
    require(data["features"].shape == (STEPS, BATCH, 28) and data["features"].dtype == np.float64 and
            np.isfinite(data["features"]).all() and np.all(data["native_input_last"] <= data["native_release"]),
            "C026 normalized features or chronology changed")
    for k1, k2 in (("enabled", "active"), ("domain", "domain"), ("condition", "condition"), ("c022_chord", "C022_chord")):
        np.testing.assert_array_equal(data[k1], c026[k2])
    with (ROOT / "C026/optimizer-steps.jsonl").open(encoding="utf-8") as f:
        prior_steps = [json.loads(row) for row in f]
    require(len(prior_steps) == STEPS, "C026 optimizer steps missing")
    for j, old in enumerate(prior_steps):
        require(old["step"] == j + 1 and all(
            array_sha(data[k][j], dtype) == old[h] for k, h, dtype in (
                ("native_release", "native_release_sha256", "<i8"),
                ("native_input_last", "native_input_last_sha256", "<i8"),
                ("source_record_ordinal", "source_record_ordinal_sha256", "<i8"),
                ("features", "normalized_features_sha256", "<f8"),
            )), f"C026 slot order changed in batch {j + 1}")
    names, mapping = original_names(data)
    vectors = [data[k].reshape(-1, 3) for k in ("center", "fast", "slow", "truth")]
    enabled = data["enabled"].ravel()
    label, gap, angles_sq = teacher(*vectors, enabled)
    require(np.isfinite(angles_sq).all() and np.isfinite(gap).all() and np.all(gap >= 0) and
            not np.any(gap[~enabled]), "Invalid teacher confidence or original gate")
    source = data["domain"].ravel()
    condition = data["condition"].ravel()
    groups = np.where(source == "EuRoC", condition, source)
    require(set(groups) == set(GROUPS), "Unregistered training source or condition")
    ordinal = data["source_record_ordinal"].ravel()
    prior_bank = {(row["group"], row["ordinal"]): row for row in review["per_original_record_condition"]}
    require(len(prior_bank) == 151, "Teacher review lacks an original record or EuRoC condition")
    weights = np.zeros(len(label), dtype=np.float64)
    c026_prob = c026["C026_prob"].reshape(-1, 3)
    c26.require_simplex(c026_prob)
    c026_target = c026_prob[np.arange(len(label)), label]
    c026_correct = c026_prob.argmax(axis=1) == label
    records, summaries = [], {}
    for group in GROUPS:
        mask = groups == group
        ids = np.unique(ordinal[mask])
        group_review = review["groups"][group]
        require(len(ids) == group_review["record_conditions"] == group_review["records_with_confident_slots"],
                f"Missing confident original recording in {group}")
        require(int(mask.sum()) == group_review["raw_slots"], f"Slot count changed for {group}")
        share = SHARES[group] / len(ids)
        for i in ids:
            k = mask & (ordinal == i)
            amount = float(gap[k].sum())
            expected = prior_bank[group, int(i)]
            require(amount > 0 and int(k.sum()) == expected["slots"] and int((gap[k] > 0).sum()) == expected["positive_gap_slots"]
                    and abs(amount - expected["sum_angular_squared_gap"]) <= 2e-12 * max(1., amount),
                    f"Independent teacher mismatch for {group}/{i}")
            weights[k] = (gap[k] / amount) * share
            name, domain = mapping[int(i)]
            records.append({"group": group, "condition": str(condition[k][0]), "ordinal": int(i),
                            "record": name, "source": domain, "slots": int(k.sum()), "positive_gap_slots": int((gap[k] > 0).sum()),
                            "raw_confidence_deg2": amount, "normalized_weight": float(weights[k].sum())})
            require(abs(float(weights[k].sum()) - share) <= 2e-16, "Equal within-group record mass was lost")
        group_weights = weights[mask] / SHARES[group]
        before = float(group_weights @ c026_correct[mask])
        first = float(group_weights @ c026_target[mask])
        require(abs(before - group_review["record_balanced_argmax_accuracy"]) <= 2e-14 and
                abs(first - group_review["mean_C026_probability_for_externally_best_choice"]) <= 2e-14,
                "Rebuilt teacher/C026 comparison differs")
        mass = {name: float(weights[mask & (label == j)].sum() / SHARES[group]) for j, name in enumerate(WEIGHTS)}
        require(all(abs(m - group_review["classes"][k]["record_balanced_confidence_mass"]) <= 2e-14 for k,m in mass.items()),
                "Independent teacher class masses differ")
        summaries[group] = {"records": len(ids), "slots": int(mask.sum()), "weight": float(weights[mask].sum()),
                            "within_group_class_mass": mass, "C026_teacher_argmax_accuracy": before}
    require(abs(float(weights.sum()) - 1.) <= 5e-16 and not np.any(weights[~enabled | (gap == 0.)]) and
            np.all(weights[enabled & (gap > 0.)] > 0.), "Weights violate normalized mass, gate or tie")
    for source_name in SOURCES:
        require(abs(float(weights[source == source_name].sum()) - 1/6) <= 2e-16, "Sources must have equal mass")
    for cond in CONDITIONS:
        require(abs(float(weights[(source == "EuRoC") & (condition == cond)].sum()) - 1/12) <= 2e-16, "EuRoC half share lost")

    write(HERE / "source-record-map.json", {"original_record_count": len(mapping), "record_condition_count": len(records),
          "original_slot_archive_sha256": inputs["C026/training-inputs.npz"], "records": records, "groups": summaries})
    label_file = HERE / "teacher-slots.npz"
    shape = (STEPS, BATCH)
    np.savez_compressed(label_file, label=label.reshape(shape), confidence_deg2=gap.reshape(shape),
                        candidate_squared_angle_deg2=angles_sq.reshape(*shape, 3), global_weight=weights.reshape(shape),
                        source_record_name=names, group=groups.reshape(shape),
                        **{k: data[k] for k in ("enabled", "rest", "domain", "condition", "source_record_ordinal",
                                                 "native_input_last", "native_release")})
    write(HERE / "teacher-lock.json", {
        "pid": os.getpid(), "locked_at": stamp(), "slots": int(len(label)), "class_order": list(WEIGHTS),
        "normalized_global_weight": float(weights.sum()), "positive_weight_slots": int((weights > 0).sum()),
        "original_gate_off_slots": int((~enabled).sum()), "active_ties": int((enabled & (gap == 0.)).sum()),
        "groups": summaries, "teacher_slots_sha256": sha(label_file), "source_record_map_sha256": sha(HERE/"source-record-map.json"),
        "original_slot_archive_sha256": inputs["C026/training-inputs.npz"],
        "independent_teacher_review_sha256": inputs["review/external_response_teacher.json"],
        "label_sha256": array_sha(label, "<i8"), "confidence_sha256": array_sha(gap, "<f8"),
        "global_weight_sha256": array_sha(weights, "<f8")
    })
    return label, weights, groups


def angular_degrees(direction, truth):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(direction, truth), axis=1),
                                 np.einsum("ij,ij->i", direction, truth)))


def training_metrics(data, labels, weights, groups, modes):
    source = data["domain"].ravel()
    ordinal = data["source_record_ordinal"].ravel()
    armed = data["enabled"].ravel()
    result = {}
    for name in (*GROUPS, "EuRoC", "all"):
        use = groups == name if name in GROUPS else (source == name if name == "EuRoC" else np.ones(len(source), dtype=bool))
        norm = weights[use] / weights[use].sum()
        record_conditions = [(ordinal == i) & (groups == g) & use for g in set(groups[use]) for i in np.unique(ordinal[(groups == g) & use])]
        result[name] = {"slots": int(use.sum()), "active_slots": int((use & armed).sum()),
                        "weighted_slots": int(np.count_nonzero(weights[use])), "confidence_mass": float(weights[use].sum())}
        for method, (ce, prob, angle) in modes.items():
            selected = prob[use].argmax(axis=1)
            square = angle**2
            result[name][method] = {
                "confidence_weighted_CE_nats": float(norm @ ce[use]),
                "confidence_weighted_tilt_RMSE_deg": float(np.sqrt(norm @ square[use])),
                "record_condition_equal_tilt_RMSE_deg": fmean(float(np.sqrt(square[m].mean())) for m in record_conditions),
                "confidence_weighted_argmax_accuracy": float(norm @ (selected == labels[use])),
                "confidence_weighted_teacher_probability": float(norm @ prob[use, labels[use]]),
                "active_mean_probabilities": dict(zip(WEIGHTS, prob[use & armed].mean(axis=0).tolist(), strict=True)),
            }
    return result


def fit(data, baseline, teacher_data, state):
    started = perf_counter()
    labels, weights, groups = teacher_data
    label = torch.from_numpy(labels)
    weight = torch.from_numpy(weights)
    x = torch.from_numpy(data["features"].reshape(-1, 28))
    model = nn.Linear(28, 2, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    params = tuple(model.parameters())
    require(sum(p.numel() for p in params) == 58 and not c18.flat_params(model).any(), "The fit must start with 58 zeros")
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
    old_prob = baseline["C026_prob"].reshape(-1, 3)
    old_ce = -np.log(old_prob[np.arange(len(labels)), labels])
    p26dir, p26chord = predict(old_prob)
    old_chord_gap = float(np.max(np.abs(p26chord - baseline["C026_chord"].ravel())))
    require(old_chord_gap <= 3e-17 and np.isfinite(old_ce).all(), "C026 archived training directions differ")
    angle0 = angular_degrees(p0dir, truth)
    angle26 = angular_degrees(p26dir, truth)
    before = training_metrics(data, labels, weights, groups, {"C022_zero": (ce0, p0, angle0), "C026": (old_ce, old_prob, angle26)})
    write(HERE / "zero-step-audit.json", {
        "pid": os.getpid(), "C022_log_prior": log0.tolist(), "C022_prior": state["p0"].tolist(),
        "C022_chord_reproduction_max_abs": zero_chord_gap, "C026_chord_reproduction_max_abs": old_chord_gap,
        "artifacts": {"teacher_lock_sha256": sha(HERE/"teacher-lock.json")},
        "optimizer_initial_58_parameters": c18.flat_params(model).tolist(), "training_before": before,
    })
    c026_flat = torch.cat([state["state_dict"][k].ravel() for k in ("weight", "bias")]).numpy()
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
                "features_sha256": array_sha(data["features"][j], "<f8"),
                "labels_sha256": array_sha(labels[part], "<i8"), "global_weights_sha256": array_sha(weights[part], "<f8"),
                "routing_probabilities_sha256": array_sha(probabilities, "<f8"),
            }
            print(json.dumps(row, allow_nan=False), file=log, flush=True)
            if j+1 in (1, 40, 80, 120, 160):
                print("C028 fit", j+1, row["objective"], length, flush=True)

    model.eval().requires_grad_(False)
    with torch.inference_mode():
        ce, logits = values(slice(None))
        after_ce = ce.numpy()
        p28 = torch.softmax(logits, dim=1).numpy()
        np.testing.assert_array_equal(p28, c23.weights(model, x, log0).numpy())
    c26.require_simplex(p28)
    p28dir, _ = predict(p28)
    angle28 = angular_degrees(p28dir, truth)
    modes = {"C022_zero": (ce0, p0, angle0), "C026": (old_ce, old_prob, angle26), "C028": (after_ce, p28, angle28)}
    shape = (STEPS, BATCH)
    np.savez_compressed(HERE/"training-io.npz", C022_p=p0.reshape(*shape,3), C026_p=old_prob.reshape(*shape,3),
                        C028_p=p28.reshape(*shape,3), C022_CE=ce0.reshape(shape), C026_CE=old_ce.reshape(shape),
                        C028_CE=after_ce.reshape(shape), C022_tilt_deg=angle0.reshape(shape),
                        C026_tilt_deg=angle26.reshape(shape), C028_tilt_deg=angle28.reshape(shape),
                        C028_direction=p28dir.reshape(*shape,3))
    difference = c18.flat_params(model) - c026_flat
    torch.save({**state, "state_dict": model.state_dict(), "protocol_sha256": sha(HERE/"protocol.json"),
                "teacher_lock_sha256": sha(HERE/"teacher-lock.json"), "baseline_C026_checkpoint_sha256": sha(ROOT/"C026/router.pt")},
               HERE/"router.pt")
    return {"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "parameters": 58,
            "slots": STEPS*BATCH, "original_recordings": 148, "seconds": perf_counter()-started,
            "global_objective_multiplier": STEPS, "objective_decomposition_max_abs_gap": max_split_gap,
            "gradient_decomposition_max_abs_gap": max_gradient_gap, "C028_minus_C026_final_params": difference.tolist(),
            "parameter_delta_l2": float(np.linalg.norm(difference)),
            "parameter_delta_max_abs": float(np.max(np.abs(difference))),
            "training": training_metrics(data, labels, weights, groups, modes)}



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
           for (name,model),features in zip(models.items(),(f0,z,z),strict=True)}
    dirs={"C020_deg":original,"C022_deg":c23.augment(original,center,k*np.where(on[:,None],mix,0.))}
    shifts={}
    for name,p in probs.items():
        base.require(np.isfinite(p).all() and np.all(p>=0.) and np.max(np.abs(p.sum(1)-1.))<=PROB_TOL,
                     "C028 probabilities violate simplex")
        shift=np.where(on[:,None],p[:,:1]*fast+p[:,1:2]*slow,0.)
        dirs[name+"_deg"]=c23.augment(original,center,shift)
        shifts[name]=shift
    base.require(np.max(np.linalg.norm(shifts["C028"],axis=1))<=c22.CAP+2e-15,"C028 combined filter correction exceeded the cap")
    return dirs,probs,on,rest,context



def metrics(rows):
    good=[r for r in rows if r["status"]=="scored"]
    macro={k:fmean(r[k] for r in good) for k in METHODS} if good else {}
    armed=[r for r in good if r["active_release_count"]]
    probs={m:{k:fmean(r[f"{m}_p_{k}_mean"] for r in armed) for k in WEIGHTS} for m in ROUTERS} if armed else {}
    return {"records":len(good),"expected_records":len(rows),"frames":sum(r["scored_frames"] for r in good),
            "macro_tilt_rmse_deg":macro,"active_release_probabilities_record_equal":probs,
            "C028_delta_deg":{k:macro["C028_deg"]-macro[k] for k in METHODS if k!="C028_deg"} if good else {},
            "C028_wins":{k:sum(r["C028_deg"]<r[k] for r in good) for k in METHODS if k!="C028_deg"}}


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
                 "C028 replay or fixed controls leave development scope")
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
                             and array_sha(release,"<i8")==old["native_release_sha256"],"C026 or fixed native indices changed")
                run=perf_counter()
                dirs,probs,active,rest,context=held(q,at,release,w,acc,raw,observed,dt,k,fixed,models,norm,ctx_norm,log0)
                calc=perf_counter()-run
                gate=gates[old["mask_key"]]
                np.testing.assert_array_equal(release,gate[:,0])
                np.testing.assert_array_equal(rest,gate[:,3])
                base.require(active.sum()==old["active_release_count"] and
                             all(array_sha(probs[arm][active],"<f8")==old[f"{arm}_active_routing_sha256"] for arm in ROUTERS[:-1]),
                             "Archived C023 or C026 native probabilities changed")
                errs=c17.measure(ix,truth,q0,q,release,{m:np.vstack((UP,d)) for m,d in dirs.items()})
                score=c17.scores(errs)
                gap=max(abs(score[m]-old[m]) for m in frozen)
                references={m:row[k] for m,k in c23.FIXED.items()}
                base.require(gap<=1e-6 and np.isfinite(list(score.values())).all()
                             and all(old[m]==v for m,v in references.items()),"Frozen control score differs")
                same=~np.r_[False,active][np.searchsorted(release,ix,side="right")]
                np.testing.assert_array_equal(errs["C028_deg"][same],errs["C020_deg"][same])
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
                print("development",*key,score["C023_deg"],score["C026_deg"],score["C028_deg"],flush=True)
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
            lock["checkpoint_sha256"] == sha(HERE / "router.pt") and
            all(sha(HERE / n) == digest for n, digest in lock["artifact_sha256"].items()),
            "Only the original locked C028 model may be replayed")
    states = [weights_state(name) for name in ROUTERS]
    incumbent, prior, own = states
    models = {}
    for name, state, width in zip(ROUTERS, states, (24, 28, 28), strict=True):
        model = nn.Linear(width, 2, dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name] = model.eval().requires_grad_(False)
        for k in ("p0", "log_p0"):
            torch.testing.assert_close(own[k], state[k], rtol=0., atol=0.)
    for k in NORM_KEYS:
        torch.testing.assert_close(own[k], prior[k], rtol=0., atol=0.)
    for k in ("mean", "std"):
        torch.testing.assert_close(own["old_feature_" + k], incumbent["feature_" + k], rtol=0., atol=0.)
    mean, std, c_mean, c_std = (own[k].numpy() for k in NORM_KEYS)
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    fixed, campaign = c22.load_fixed(read(ROOT / "C022/protocol.json"))
    previous, control, inventory, split = (read(p) for p in (
        ROOT / "C026/results.json", ROOT / "review/vqf_response_controls_development.json",
        ROOT / "C001/inventory.json", TRANSFER / "external-split.json",
    ))
    require([[r["dataset"], r["record"]] for r in previous["sequences"]] == protocol["development_records"],
            "C026 development replay order changed")
    status(protocol, "development_replay", router_lock_sha256=lock_hash, expected_records=213)
    print("C028 locked; replaying the common 213 development records", flush=True)
    start = perf_counter()
    result = replay(previous, control, inventory, campaign, split, cfg, protocol["C022_k"], fixed,
                    models, (mean, std), (c_mean, c_std), own["log_p0"])
    require(sha(HERE/"router-lock.json") == lock_hash and sha(HERE/"router.pt") == lock["checkpoint_sha256"] and
            all(sha(ROOT / n) == digest for n, digest in protocol["inputs_sha256"].items()), "Frozen evidence changed")
    result.update(pid=os.getpid(), original_fit_pid=lock["pid"], router_lock_sha256=lock_hash,
                  protocol_sha256=sha(HERE/"protocol.json"), scoring_masks_sha256=sha(MASKS),
                  replay_seconds=perf_counter()-start, completed_at=stamp(), final_holdout_scored=False)
    dest = HERE / "results.json"
    write(dest, result)
    scored = result["pooled_records"]["records"]
    complete = scored == 213 and not result["failures"] and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if complete else "incomplete", scored=scored,
           router_lock_sha256=lock_hash, results_sha256=sha(dest))
    require(complete, "C028 replay incomplete; checkpoint remains locked")
    print("C028 complete", scored, result["pooled_records"]["macro_tilt_rmse_deg"], flush=True)


def run(launch):
    previous, inputs, review = verify_inputs()
    state = weights_state("C026")
    k = read(ROOT / "C022/coefficient-lock.json")["k"]
    require(k == previous["C022_k"] and state["steps"] == STEPS and
            tuple(state["feature_names"]) == c25.OBSERVERS, "C026 weights or feature definitions differ")
    torch.testing.assert_close(state["p0"], c23.prior_probs(k), rtol=0., atol=0.)
    torch.testing.assert_close(state["log_p0"], state["p0"].log(), rtol=0., atol=0.)
    normalizers = read(ROOT / "C026/normalizers.json")
    for key, width in zip(NORM_KEYS, (24, 24, 4, 4), strict=True):
        value = state[key].numpy()
        require(value.shape == (width,) and np.isfinite(value).all() and
                ("std" not in key or np.all(value > 0.)) and array_sha(value, "<f8") == normalizers[key]["sha256"],
                "Inherited normalizer differs")
    protocol = {**launch, "inputs_sha256": inputs, "worker_sha256": sha(Path(__file__)),
                "optimizer": OPTIMIZER, "fits": 1, "C022_k": k, "parameters": 58,
                "input_width": 28, "zero_initialized": True, "slots": STEPS*BATCH, "recordings": 148,
                "condition_shares_of_global_weight": SHARES, "training_source_shares": {s: 1/6 for s in SOURCES},
                "class_order": list(WEIGHTS),
                "labels": "Smallest angular error of normalized center+fast, center+slow, or center, against same-time source truth",
                "confidence": "Squared angle gap in deg^2; zero at a tie or when the frozen C026 gate is off",
                "objective": "160 times sum of globally normalized slot weight times categorical cross-entropy, per original minibatch",
                "training_only": "Teacher labels, squared angular gap, and EuRoC fixed-prefix bias removal",
                "development_input": previous["development_input"], "formal_holdouts_authorized": False,
                "development_records": previous["development_records"], "scoring_masks_sha256": sha(MASKS)}
    write(HERE / "protocol.json", protocol)
    torch.set_num_threads(2)
    status(protocol, "locking_teacher_labels")
    data = archive(ROOT / "C026/training-inputs.npz")
    baseline = archive(ROOT / "C026/training-io.npz")
    labels = lock_labels(data, baseline, review, inputs)
    status(protocol, "single_fit", teacher_lock_sha256=sha(HERE/"teacher-lock.json"))
    training = fit(data, baseline, labels, state)
    write(HERE / "training.json", training)
    artifacts = ("protocol.json", "teacher-lock.json", "teacher-slots.npz", "source-record-map.json",
                 "zero-step-audit.json", "training.json", "training-io.npz", "optimizer-steps.jsonl")
    with (HERE / "router-lock.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "locked_at": stamp(),
                   "checkpoint_sha256": sha(HERE / "router.pt"),
                   "artifact_sha256": {n: sha(HERE / n) for n in artifacts}}, handle, indent=2)
        handle.write("\n")
    replay_locked(protocol)


def main():
    launch = {"pid": os.getpid(), "started_at": stamp(), "command": "python -B -u C028/run_response_teacher.py",
              "scope": "C028 only; one response-supervised fit using the frozen C026 slots; common development replay"}
    with (HERE / "launch.json").open("x", encoding="utf-8") as handle:
        json.dump(launch, handle, indent=2)
        handle.write("\n")
    try:
        run(launch)
    except BaseException as error:
        write(HERE / "failure.json", {**launch, "failed_at": stamp(), "error": repr(error)})
        (HERE / "failure-traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()

