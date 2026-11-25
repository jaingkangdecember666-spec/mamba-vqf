"""C027: repeat the locked C023 router with one predeclared minibatch permutation."""
import hashlib
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "C023"))

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn

import run_router as c23

c22, c18, c17, c14, c3, c2 = c23.c22, c23.c18, c23.c17, c23.c14, c23.c3, c23.c2
read, write, sha, stamp, array_sha = c23.read, c23.write, c23.sha, c23.stamp, c23.array_sha
UP, DATA, TRANSFER, MASKS, GATES = c23.UP, c23.DATA, c23.TRANSFER, c23.MASKS, c23.GATES
DOMAINS, STEPS, BATCH = c23.DOMAINS, c23.STEPS, c23.BATCH
METHODS = (*c23.METHODS[:4], "C027_deg", *c23.METHODS[4:])
SEED = 53129
TOL = 8 * np.finfo(np.float64).eps
require = c23.base.require


def status(protocol, phase, **details):
    write(HERE / "process.json", {
        "pid": os.getpid(), "started_at": protocol["started_at"], "updated_at": stamp(),
        "status": phase, "protocol_sha256": sha(HERE / "protocol.json"), **details,
    })


def verify_c023():
    state, lock, prior, training = (read(ROOT / "C023" / name) for name in (
        "process.json", "router-lock.json", "protocol.json", "training.json",
    ))
    required = {
        "C023/process.json": sha(ROOT / "C023/process.json"),
        "C023/router-lock.json": state["router_lock_sha256"],
        "C023/protocol.json": state["protocol_sha256"],
        "C023/results.json": state["results_sha256"],
        "C023/training.json": lock["training_sha256"],
        "C023/run_router.py": prior["code_sha256"],
        **{"C023/" + name: digest for name, digest in training["artifacts_sha256"].items()},
        **prior["inputs_sha256"],
    }
    require(state["status"] == "complete" and state["scored"] == 213 and
            lock["fit_count"] == training["fit_count"] == 1 and lock["optimizer_steps"] == STEPS == 160 and
            lock["checkpoint_sha256"] == required["C023/router.pt"] and lock["protocol_sha256"] == required["C023/protocol.json"] and
            all(sha(ROOT / name) == digest for name, digest in required.items()), "C023 archived evidence changed")
    return prior, required


def np_archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_prior_router():
    return torch.load(ROOT / "C023/router.pt", map_location="cpu", weights_only=True)


def adam_parameters():
    return {"lr": c23.LR, "weight_decay": c23.WD, "foreach": False}


def lock_schedule(original, source, x, state, files):
    io = np_archive(ROOT / "C023/training-io.npz")
    with (ROOT / "C023/optimizer-steps.jsonl").open(encoding="utf-8") as stream:
        prior_steps = [json.loads(line) for line in stream]
    require(len(prior_steps) == STEPS and BATCH == 96 and x["z"].shape == (15360, 24) and
            np.unique(source["source_record_ordinal"]).size == 139, "C023 train schedule changed")
    for own, expected in ((x["z"].numpy(), io["normalized_input"]),
                          (x["prob0"].numpy(), io["p0"]),
                          (x["prob0"].numpy(), state["p0"].numpy()),
                          (x["log0"].numpy(), state["log_p0"].numpy()),
                          (source["enabled"], io["enabled"]),
                          (source["source_record_ordinal"], io["record_ordinal"]),
                          (source["native_release"], io["native_release"])):
        np.testing.assert_array_equal(own, expected)
    denominators = {d: v["denominator"] for d, v in
                    read(ROOT / "C023/zero-step-audit.json")["C022_source_init_gradients"].items()}
    external = read(ROOT / "C022/external-selection.json")["domains"]
    require(tuple(denominators) == DOMAINS and all(v == external[d]["C022_loss_at_lock"] for d, v in denominators.items()),
            "C023 domain denominators changed")
    recomputed = source["record_equal_weights"] / np.array([denominators[d] for d in source["domain"]])
    np.testing.assert_array_equal(recomputed, io["relative_loss_weight"])
    permutation = np.random.Generator(np.random.PCG64(SEED)).permutation(STEPS)
    np.testing.assert_array_equal(np.sort(permutation), np.arange(STEPS))
    receipt = []
    for step, source_index in enumerate(permutation, start=1):
        index = int(source_index)
        part = slice(index * BATCH, (index + 1) * BATCH)
        legacy = prior_steps[index]
        hashes = {
            "input_feature_sha256": array_sha(io["normalized_input"][part], "<f8"),
            "native_release_sha256": array_sha(source["native_release"][part], "<i8"),
            "native_input_last_sha256": array_sha(source["native_input_last"][part], "<i8"),
            "source_record_ordinal_sha256": array_sha(source["source_record_ordinal"][part], "<i8"),
            "legacy_slot_identity_sha256": array_sha(original["global_window"][part], "<i8"),
        }
        require(legacy["step"] == index + 1 and all(legacy[k] == d for k, d in hashes.items()), "C023 minibatch identity changed")
        hashes["effective_loss_weight_sha256"] = array_sha(io["relative_loss_weight"][part], "<f8")
        hashes["source_slot_sha256"] = {
            k: array_sha(source[k][part], dt) for k, dt in (
                ("center", "<f8"), ("truth", "<f8"), ("fast", "<f8"), ("slow", "<f8"),
                ("enabled", "|u1"), ("rest", "|u1"), ("record_equal_weights", "<f8"),
            )
        }
        hashes["source_domain_sha256"] = hashlib.sha256(
            json.dumps(source["domain"][part].tolist(), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        receipt.append({"step": step, "source_batch_index": index, **hashes})
    write(HERE / "sampling-lock.json", {
        "pid": os.getpid(), "locked_at": stamp(), "seed": SEED, "numpy_version": np.__version__,
        "generator": "numpy.random.Generator(numpy.random.PCG64(53129)).permutation(160)",
        "source_batch_index_base": 0, "permutation": permutation.tolist(),
        "permutation_sha256": array_sha(permutation, "<i8"), "batch_size": BATCH, "steps": STEPS,
        "source_records": 139, "slots": STEPS * BATCH, "original_within_batch_order_verified": True,
        "C022_training_slots_sha256": files["C022/training-slots.npz"],
        "C019_training_inputs_sha256": files["C019/training-inputs.npz"],
        "C023_training_io_sha256": files["C023/training-io.npz"],
        "C023_optimizer_steps_sha256": files["C023/optimizer-steps.jsonl"],
        "feature_mean_sha256": array_sha(state["feature_mean"].numpy(), "<f8"),
        "feature_std_sha256": array_sha(state["feature_std"].numpy(), "<f8"),
        "C022_prior_sha256": array_sha(state["p0"].numpy(), "<f8"),
        "frozen_C022_domain_denominators": denominators, "steps_with_source_hashes": receipt,
    })
    return receipt, prior_steps, io, denominators


def check_probabilities(p):
    require(np.isfinite(p).all() and np.all(p >= 0.) and
            np.max(np.abs(p.sum(axis=1) - 1.)) <= TOL, "Non-convex routing probabilities")


def fit(source, x, state, schedule, prior_steps, io, denominators, k):
    tick = perf_counter()
    model = nn.Linear(24, 2, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    require(sum(p.numel() for p in model.parameters()) == 50 and not c18.flat_params(model).any(),
            "C027 must start with exactly 50 zero parameters")
    w = torch.from_numpy(io["relative_loss_weight"])
    coef = len(w) / (len(DOMAINS) * BATCH)
    require(coef == 40., "C023 objective multiplier changed")
    with torch.inference_mode():
        loss0, direction0, shift0, prob0 = c23.losses(model, x, slice(None))
        loss0 = loss0.numpy()
        initial_error = float(np.max(np.abs(loss0 - io["C022_loss"])))
        torch.testing.assert_close(prob0, x["prob0"][None].expand_as(prob0), rtol=0., atol=2e-16)
        target = c14.normalize(source["center"] + k * source["response"])
        direction_gap = float(np.max(np.abs(direction0.numpy() - target)))
        require(initial_error <= 3e-17 and direction_gap <= 5e-16 and
                torch.count_nonzero(shift0[~x["enabled"]]).item() == 0, "Zero step differs from C023/C022")
    write(HERE / "zero-step-audit.json", {
        "pid": os.getpid(), "zero_C022_direction_max_abs_error": direction_gap,
        "archived_C023_initial_chord_max_abs_error": initial_error,
        "C022_denominator_by_domain": denominators, "minibatch_scale": coef,
        "normalized_full_zero_step_loss": float(w.numpy() @ loss0) / len(DOMAINS),
    })
    optimizer = torch.optim.AdamW(model.parameters(), **adam_parameters())
    with (HERE / "optimizer-steps.jsonl").open("x", encoding="utf-8") as log:
        for entry, c023_at_step in zip(schedule, prior_steps, strict=True):
            index, step = entry["source_batch_index"], entry["step"]
            part = slice(index * BATCH, (index + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            before = c18.flat_params(model)
            loss, _, _, p = c23.losses(model, x, part)
            objective = coef * (w[part] * loss).sum()
            objective.backward()
            raw = c18.flat_params(model, use_grad=True)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), c23.CLIP))
            clipped = c18.flat_params(model, use_grad=True)
            optimizer.step()
            after = c18.flat_params(model)
            require(np.isfinite(np.r_[raw, clipped, after, norm, float(objective.detach())]).all(), "Nonfinite optimizer trace")
            observed, probability = loss.detach().numpy(), p.detach().numpy()
            check_probabilities(probability)
            if step == 1:
                expected = float(coef * np.dot(w[part].numpy(), loss0[part]))
                require(abs(float(objective.detach()) - expected) <= 5e-16, "First permuted minibatch objective changed")
            by_domain = {}
            for name in DOMAINS:
                domain = source["domain"][part] == name
                by_domain[name] = {
                    "count": int(domain.sum()),
                    "normalized_objective_contribution": float(coef * np.dot(w[part].numpy()[domain], observed[domain])),
                    "active": int(np.count_nonzero(domain & source["enabled"][part])),
                }
            require(abs(sum(v["normalized_objective_contribution"] for v in by_domain.values()) -
                        float(objective.detach())) <= 2e-15, "Domain loss ledger differs from the optimized objective")
            record = {
                **entry, "pid": os.getpid(), "objective": float(objective.detach()), "raw_gradient_l2": norm,
                "raw_gradient": raw.tolist(), "clipped_gradient": clipped.tolist(),
                "params_before": before.tolist(), "params_after": after.tolist(), "actual_update": (after - before).tolist(),
                "params_after_minus_C023_at_same_update": (after - np.array(c023_at_step["params_after"])).tolist(),
                "domains": by_domain, "prediction_weights_sha256": array_sha(probability, "<f8"),
            }
            print(json.dumps(record, allow_nan=False), file=log, flush=True)
            if step in (1, 40, 80, 120, 160):
                print("C027 fit", step, "source_batch_index", index, "objective", record["objective"], flush=True)

    model.eval().requires_grad_(False)
    with torch.inference_mode():
        final, pred, delta, p = c23.losses(model, x, slice(None))
        inactive = ~x["enabled"]
        torch.testing.assert_close(pred[inactive], c18.unit(x["center"])[inactive], rtol=0., atol=0.)
        require(torch.count_nonzero(delta[inactive]).item() == 0, "C027 bypassed the original gate")
    check_probabilities(p.numpy())
    archive = HERE / "training-io.npz"
    np.savez_compressed(archive, normalized_input=x["z"].numpy(), p0=state["p0"].numpy(),
                        prob_trained=p.numpy(), C022_loss=loss0, C023_loss=io["C023_loss"], C027_loss=final.numpy(),
                        C027_prediction=pred.numpy(), C027_delta=delta.numpy(), enabled=source["enabled"],
                        relative_loss_weight=w.numpy(), source_record_ordinal=source["source_record_ordinal"],
                        native_release=source["native_release"], native_input_last=source["native_input_last"])
    frozen_params = torch.cat([state["state_dict"][key].ravel() for key in ("weight", "bias")]).numpy()
    np.testing.assert_array_equal(frozen_params, prior_steps[-1]["params_after"])
    difference = c18.flat_params(model) - frozen_params
    summary = c23.summaries(final.numpy(), loss0, p.numpy(), source)
    for domain, item in summary.items():
        item["C027_record_equal_chord"] = item.pop("C023_record_equal_chord")
        use = source["domain"] == domain
        item["C023_record_equal_chord"] = float(source["record_equal_weights"][use] @ io["C023_loss"][use])
    sampling_hash = sha(HERE / "sampling-lock.json")
    torch.save({**state, "state_dict": model.state_dict(), "protocol_sha256": sha(HERE / "protocol.json"),
                "sampling_lock_sha256": sampling_hash, "baseline_C023_checkpoint_sha256": sha(ROOT / "C023/router.pt")},
               HERE / "router.pt")
    return {
        "pid": os.getpid(), "fit_count": 1, "steps": STEPS, "parameters": 50, "slots": STEPS * BATCH,
        "source_records": 139, "seconds": perf_counter() - tick, "sampling_lock_sha256": sampling_hash, "domains": summary,
        "parameter_order": ["weight[2,24] row-major", "bias[2]"], "final_delta_vs_C023": difference.tolist(),
        "final_delta_l2": float(np.linalg.norm(difference)), "final_delta_max_abs": float(np.abs(difference).max()),
        "route_probability_sum_max_abs_error": float(np.max(np.abs(p.numpy().sum(axis=1) - 1.))),
    }




WEIGHTS = c23.WEIGHTS
add_baselines = c23.add_baselines
base, intrinsic = c23.base, c23.intrinsic


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
            "C027_delta_deg": {m: macro["C027_deg"] - macro[m] for m in METHODS if m != "C027_deg"} if keep else {},
            "C027_wins": {m: sum(r["C027_deg"] < r[m] for r in keep) for m in METHODS if m != "C027_deg"}}


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
    rows, poses, failures = [], [], []
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
                probes, delta = probabilities["C027"], deltas["C027"]
                require(int(active.sum()) == old["active_release_count"] and
                        array_sha(probabilities["C023"][active], "<f8") == old["active_routing_sha256"] and
                        array_sha(deltas["C023"], "<f8") == old["C023_delta_sha256"],
                        "Frozen C023 release count, probabilities, or deltas changed")
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                c023_gap = abs(score["C023_deg"] - old["C023_deg"])
                require(c023_gap <= 1e-6 and all(score[m] == old[m] for m in c23.FIXED),
                        "Archived C023 or fixed VQF score changed")
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C027_deg"][~active_scored], errs["C020_deg"][~active_scored])
                active_probs = dict(zip(WEIGHTS, probes[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": max(gap, c023_gap), "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs.items()}, "active_routing_sha256": array_sha(probes[active], "<f8"),
                             "C027_delta_sha256": array_sha(delta, "<f8"),
                             "C023_active_routing_sha256": old["active_routing_sha256"],
                             "C023_delta_sha256": old["C023_delta_sha256"], "prediction_seconds": pred_seconds, "seconds": perf_counter() - start})
                poses.extend(detail)
                print("development", *key, score["C023_deg"], score["C027_deg"], flush=True)
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
            "artifact_sha256": {p.name: sha(p) for p in (seq, pos, group, fixed_file) if p.is_file()}}



def load_locked_routers():
    models = {}
    states = {}
    for name in ("C023", "C027"):
        state = torch.load(ROOT / name / "router.pt", map_location="cpu", weights_only=True)
        model = nn.Linear(24, 2, dtype=torch.float64)
        model.load_state_dict(state["state_dict"])
        models[name] = model.eval().requires_grad_(False)
        states[name] = state
    prior, new = states["C023"], states["C027"]
    require(prior["optimizer_steps"] == new["optimizer_steps"] == STEPS and
            prior["C022_coefficient_lock_sha256"] == new["C022_coefficient_lock_sha256"], "Router controls changed")
    for key in ("feature_mean", "feature_std", "p0", "log_p0"):
        torch.testing.assert_close(prior[key], new[key], rtol=0., atol=0.)
    return models, prior["feature_mean"].numpy(), prior["feature_std"].numpy(), prior["log_p0"]


def after_lock(protocol):
    lock = read(HERE / "router-lock.json")
    lock_hash = sha(HERE / "router-lock.json")
    require(lock["pid"] == os.getpid() and lock["fit_count"] == 1 and lock["optimizer_steps"] == STEPS and
            lock["checkpoint_sha256"] == sha(HERE / "router.pt") and
            all(sha(HERE / name) == digest for name, digest in lock["artifact_sha256"].items()),
            "Development replay requires the original C027 locked fit")
    fixed, campaign = c22.load_fixed(read(ROOT / "C022/protocol.json"))
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    prior = read(ROOT / "C023/results.json")
    require([[r["dataset"], r["record"]] for r in prior["sequences"]] == protocol["development_records"],
            "Development panel or order differs from C023")
    routers = load_locked_routers()
    status(protocol, "development_replay", router_lock_sha256=lock_hash, expected_records=213)
    print("C027 locked; replaying 213 development records", flush=True)
    start = perf_counter()
    result = replay(prior, read(ROOT / "review/vqf_response_controls_development.json"),
                    read(ROOT / "C001/inventory.json"), campaign, read(TRANSFER / "external-split.json"),
                    cfg, protocol["C022_k"], fixed, routers)
    require(sha(HERE / "router-lock.json") == lock_hash and sha(HERE / "router.pt") == lock["checkpoint_sha256"]
            and all(sha(ROOT / n) == h for n, h in protocol["inputs_sha256"].items()), "A frozen replay control changed")
    result.update(pid=os.getpid(), original_fit_pid=lock["pid"], router_lock_sha256=lock_hash,
                  protocol_sha256=sha(HERE / "protocol.json"), scoring_masks_sha256=sha(MASKS),
                  started_at=protocol["started_at"], completed_at=stamp(), development_replay_seconds=perf_counter()-start,
                  archived_fixed_controls_reused=True, final_holdout_scored=False)
    out = HERE / "results.json"
    write(out, result)
    scored = result["pooled_records"]["records"]
    complete = scored == 213 and not result["failures"] and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if complete else "incomplete", scored=scored,
           router_lock_sha256=lock_hash, results_sha256=sha(out))
    require(complete, "C027 replay incomplete; checkpoint remains locked")
    print("C027 complete", scored, result["pooled_records"]["macro_tilt_rmse_deg"], flush=True)


def run(launch):
    prior, files = verify_c023()
    optimizer = {"kind": "AdamW", "lr": .01, "weight_decay": .01, "clip_global_norm": 2.,
                 "steps": 160, "batch_size": 96}
    require(prior["optimizer"] == optimizer and adam_parameters() == {"lr": .01, "weight_decay": .01, "foreach": False}
            and c23.CLIP == 2. and prior["scoring_masks_sha256"] == sha(MASKS)
            and len(prior["development_records"]) == 213, "Unexpected C023 optimizer or replay panel")
    c022 = read(ROOT / "C022/coefficient-lock.json")
    require(c022["k"] == prior["C022_k"] and c022["fit_count"] == 1 and
            c022["C019_sampling_lock_sha256"] == files["C019/sampling-lock.json"], "Frozen C022 or C019 lock changed")
    state = load_prior_router()
    source_norm = torch.load(ROOT / "C019/readout.pt", map_location="cpu", weights_only=True)
    for key in ("feature_mean", "feature_std"):
        torch.testing.assert_close(state[key], source_norm[key].to(torch.float64), rtol=0., atol=0.)
    mean, std = state["feature_mean"].numpy(), state["feature_std"].numpy()
    require(mean.shape == std.shape == (24,) and np.isfinite(mean).all() and np.isfinite(std).all() and np.all(std > 0.),
            "C023 B6 feature normalization changed")
    protocol = {**launch, "worker_sha256": sha(Path(__file__)), "inputs_sha256": files,
                "C022_k": c022["k"], "parameters": 50, "fit_count": 1, "zero_initialized": True,
                "input_width": 24, "feature": prior["feature"], "optimizer": {**optimizer, "foreach": False},
                "loss": prior["loss"], "four_domains": list(DOMAINS), "training_records": 139,
                "sole_change": "PCG64(53129) permutation of 160 complete minibatches; no within-batch changes",
                "seed_changes_optimizer_batch_order_only": True, "frozen_Mamba_initialization_seeds_reused": True,
                "development_records": prior["development_records"], "scoring_masks_sha256": sha(MASKS),
                "formal_holdouts_authorized": False}
    write(HERE / "protocol.json", protocol)
    torch.set_num_threads(2)
    status(protocol, "locking_batch_permutation")
    original, source, x = c23.load_train(c022, files, mean, std)
    schedule, prior_steps, io, denom = lock_schedule(original, source, x, state, files)
    sampling_hash = sha(HERE / "sampling-lock.json")
    status(protocol, "single_fit", sampling_lock_sha256=sampling_hash)
    training = fit(source, x, state, schedule, prior_steps, io, denom, c022["k"])
    write(HERE / "training.json", training)
    artifacts = ("protocol.json", "sampling-lock.json", "training.json", "zero-step-audit.json",
                 "optimizer-steps.jsonl", "training-io.npz")
    with (HERE / "router-lock.json").open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "fit_count": 1, "optimizer_steps": STEPS, "locked_at": stamp(),
                   "sampling_lock_sha256": sampling_hash, "checkpoint_sha256": sha(HERE / "router.pt"),
                   "artifact_sha256": {name: sha(HERE / name) for name in artifacts}}, handle, indent=2)
        handle.write("\n")
    after_lock(protocol)


def main():
    launch = {"pid": os.getpid(), "started_at": stamp(), "command": "python -B -u C027/run_order_repeat.py",
              "scope": "C027 only; one permuted C023 fit followed by the original development replay"}
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

