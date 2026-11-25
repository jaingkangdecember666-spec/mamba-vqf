"""C023: one fresh 50-parameter softmax router on frozen C022 training slots."""
import json
import os
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "C022"))

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from torch import nn
import run_filter_pair as c22

c20, c18, c17, c14, c6, c3, c2 = c22.c20, c22.c18, c22.c17, c22.c14, c22.c6, c22.c3, c22.c2
base, intrinsic = c22.base, c22.intrinsic
read, write, sha, stamp, array_sha = c22.read, c22.write, c22.sha, c22.stamp, c22.array_sha
UP, DATA, PILOT, TRANSFER, MASKS, GATES = c22.UP, c22.DATA, c22.PILOT, c22.TRANSFER, c22.MASKS, c22.GATES
DOMAINS, STEPS, BATCH = c22.DOMAINS, c22.STEPS, c22.BATCH
LR, WD, CLIP = .01, .01, 2.
METHODS = ("full_vqf_deg", "C020_deg", "C022_deg", "C023_deg", "fixed_tau1p5_deg", "fixed_tau3_deg", "fixed_tau6_deg")
FIXED = dict(zip(METHODS[4:], ("fast_deg", "default_deg", "slow_deg"), strict=True))
WEIGHTS = ("fast", "slow", "original")
QUANTILES = (0., .05, .25, .5, .75, .95, 1.)


def prior_probs(k):
    p0 = torch.tensor([2. * k / 3., k / 3., 1. - k], dtype=torch.float64)
    base.require(torch.all(p0 > 0.).item() and abs(float(p0.sum()) - 1.) <= 2e-16, "Invalid C022 prior")
    return p0


def weights(model, z, base_logits):
    extra = model(z)
    logits = torch.cat((extra + base_logits[:2], base_logits[2].expand(len(z), 1)), dim=1)
    return torch.softmax(logits, dim=1)


def position(model, z, base_logits, center, fast, slow, enabled):
    p = weights(model, z, base_logits)
    correction = (p[:, :1] * fast + p[:, 1:2] * slow).masked_fill(~enabled[:, None], 0.)
    return c18.unit(center + correction), correction, p


def load_train(lock, files, average, scale):
    original = c20.load_schedule(files)
    archive = ROOT / "C022/training-slots.npz"
    base.require(sha(archive) == lock["training_slots_sha256"] == files["C022/training-slots.npz"], "C022 training vectors changed")
    with np.load(archive, allow_pickle=False) as a:
        source = {key: a[key] for key in a.files}
    for k in ("truth", "native_release", "native_input_last", "source_record_ordinal", "source_record_name", "record_equal_weights", "domain", "rest"):
        np.testing.assert_array_equal(source[k], original[k])
    np.testing.assert_array_equal(source["intrinsic_valid"], original["valid"])
    np.testing.assert_array_equal(source["enabled"], original["valid"] & ~original["rest"])
    np.testing.assert_array_equal(source["response"], np.where(source["enabled"][:, None], (2. * source["fast"] + source["slow"]) / 3., 0.))
    z = (original["pooled"][:, 1].astype(np.float64) - average) / scale
    base.require(z.shape == (15360, 24) and np.isfinite(z).all() and len(np.unique(source["source_record_ordinal"])) == 139
                 and np.all(source["native_input_last"] <= source["native_release"]), "Locked training support or shape changed")
    x = {"z": torch.from_numpy(z)}
    for k in ("center", "fast", "slow", "truth", "enabled", "record_equal_weights"):
        x[k] = torch.from_numpy(source[k])
    x["prob0"] = prior_probs(lock["k"])
    x["log0"] = x["prob0"].log()
    base.require(all(t.device.type == "cpu" for t in x.values()), "C023 optimizer must only see CPU archived training tensors")
    return original, source, x


def losses(model, x, use):
    y, delta, p = position(model, x["z"][use], x["log0"], x["center"][use], x["fast"][use], x["slow"][use], x["enabled"][use])
    return (y - x["truth"][use]).square().sum(-1), y, delta, p


def summaries(loss, prior_loss, probability, source):
    result = {}
    for d in DOMAINS:
        m = source["domain"] == d
        active = m & source["enabled"]
        w = source["record_equal_weights"][m]
        before, after = float(w @ prior_loss[m]), float(w @ loss[m])
        q = np.quantile(probability[active], QUANTILES, axis=0)
        result[d] = {"slots": int(m.sum()), "active_slots": int(active.sum()),
                     "records": len(np.unique(source["source_record_ordinal"][m])), "C022_record_equal_chord": before,
                     "C023_record_equal_chord": after, "relative_error": after / before,
                     "active_prob_mean": dict(zip(WEIGHTS, probability[active].mean(axis=0).tolist(), strict=True)),
                     "active_prob_quantiles": {k: dict(zip(map(str, QUANTILES), q[:, j].tolist(), strict=True)) for j, k in enumerate(WEIGHTS)}}
    return result


def train(model, original, source, x, protocol, lock_hash):
    tick = perf_counter()
    xall = slice(None)
    p0 = x["prob0"].numpy()
    ya = source["center"] + protocol["k"] * source["response"]
    expected = c14.normalize(ya)
    opt_ref = read(ROOT / "C022/external-selection.json")
    review = read(ROOT / "review/source_response_routing.json")
    base.require(review["coefficient"] == protocol["k"], "Independent router slope uses different k")
    pre, zero_y, zero_d, zero_p = losses(model, x, xall)
    tol = float(np.max(np.abs(zero_y.detach().numpy() - expected)))
    with torch.no_grad():
        torch.testing.assert_close(zero_p, x["prob0"][None].expand_as(zero_p), atol=2e-16, rtol=0.)
        torch.testing.assert_close(zero_y[~x["enabled"]], c18.unit(x["center"])[~x["enabled"]], rtol=0., atol=0.)
        base.require(tol <= 5e-16 and torch.count_nonzero(zero_d[~x["enabled"]]).item() == 0, "Zero-step C022 reproduction failed")
    original_loss = pre.detach().numpy()
    denominators, zero_grads = np.empty(len(pre)), {}
    for d in DOMAINS:
        use = source["domain"] == d
        w = x["record_equal_weights"][use]
        target = opt_ref["domains"][d]["C022_loss_at_lock"]
        value = float(w @ pre[use].detach())
        base.require(abs(value - target) <= 2e-14 and abs(float(w.sum()) - 1.) <= 2e-12, "Source denominator differs from C022")
        denominators[use] = target
        # Independent initialization audit with no optimizer or model update.
        g, b = torch.autograd.grad((w * pre[use]).sum() / target, tuple(model.parameters()), retain_graph=True)
        wanted = review["domains"][d]
        err = max(abs(float(g.norm()) - wanted["context_weight_gradient_norm"]),
                  float(np.max(np.abs(b.numpy() - wanted["two_constant_logit_gradients"]))))
        base.require(err <= 2e-7, f"Independent C022 router-gradient discrepancy {d}: {err}")
        zero_grads[d] = {"denominator": target, "weight_grad_l2": float(g.norm()), "bias_grad": b.tolist(),
                         "max_abs_review_difference": err}
    del pre, zero_y, zero_d, zero_p
    normalization = x["record_equal_weights"] / torch.from_numpy(denominators)
    coef = len(original_loss) / (len(DOMAINS) * BATCH)
    base.require(coef == 40. and sum(p.numel() for p in model.parameters()) == 50 and model.weight.count_nonzero().item() == 0
                 and model.bias.count_nonzero().item() == 0, "Bad C023 parameter count, loss multiplier or init")
    init_audit = {"p0": p0.tolist(), "C022_direction_max_abs_difference": tol, "disabled_direction_exact_C020": True,
                  "initial_normalized_loss": float(np.dot(normalization.numpy(), original_loss) / len(DOMAINS)),
                  "minibatch_scale": coef, "C022_source_init_gradients": zero_grads, "dtype": "float64 CPU"}
    init_file = HERE / "zero-step-audit.json"
    write(init_file, init_audit)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, foreach=False)
    trace = HERE / "optimizer-steps.jsonl"
    with trace.open("x", encoding="utf-8") as out:
        for ix in range(STEPS):
            use = slice(ix * BATCH, (ix + 1) * BATCH)
            optimizer.zero_grad(set_to_none=True)
            before = c18.flat_params(model)
            err, _, _, probs = losses(model, x, use)
            obj = coef * (normalization[use] * err).sum()
            obj.backward()
            g = c18.flat_params(model, use_grad=True)
            n = float(torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP))
            clipped = c18.flat_params(model, use_grad=True)
            optimizer.step()
            after = c18.flat_params(model)
            actual, probs = err.detach().numpy(), probs.detach().numpy()
            base.require(np.isfinite(np.r_[float(obj.detach()), n, g, after]).all(), "C023 optimizer has nonfinite values")
            if ix == 0:
                exact_first = float(coef * np.dot(normalization[use].numpy(), original_loss[use]))
                base.require(abs(float(obj.detach()) - exact_first) <= 5e-16, "Lost zero-init C022 minibatch loss")
            totals = {}
            for domain in DOMAINS:
                m = source["domain"][use] == domain
                totals[domain] = {"count": int(m.sum()),
                                  "normalized_objective_contribution": float(coef * np.dot(normalization[use].numpy()[m], actual[m])),
                                  "active": int(np.count_nonzero(m & source["enabled"][use]))}
            row = {"step": ix + 1, "pid": os.getpid(), "objective": float(obj.detach()), "raw_gradient_l2": n,
                   "raw_gradient": g.tolist(), "clipped_gradient": clipped.tolist(),
                   "params_before": before.tolist(), "params_after": after.tolist(), "domains": totals,
                   "active_weights_mean_before_step": probs[source["enabled"][use]].mean(axis=0).tolist(),
                   "input_feature_sha256": array_sha(x["z"][use].numpy(), "<f8"), "prediction_weights_sha256": array_sha(probs, "<f8"),
                   "native_release_sha256": array_sha(source["native_release"][use], "<i8"),
                   "native_input_last_sha256": array_sha(source["native_input_last"][use], "<i8"),
                   "source_record_ordinal_sha256": array_sha(source["source_record_ordinal"][use], "<i8"),
                   "legacy_slot_identity_sha256": array_sha(original["global_window"][use], "<i8")}
            base.require(abs(sum(v["normalized_objective_contribution"] for v in totals.values()) - row["objective"]) <= 2e-15,
                         "Domain loss ledger diverges from actual optimized objective")
            print(json.dumps(row, allow_nan=False), file=out, flush=True)
            if ix + 1 in (1, 40, 80, 120, 160):
                print("C023 fit", ix + 1, row["objective"], n, flush=True)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        after, pred, delta, p = losses(model, x, xall)
        np.testing.assert_allclose(p.sum(-1).numpy(), np.ones(len(p)), rtol=0., atol=5e-16)
        base.require(torch.all(p >= 0.).item() and torch.count_nonzero(delta[~x["enabled"]]).item() == 0, "Router violates convexity or rest")
        torch.testing.assert_close(pred[~x["enabled"]], c18.unit(x["center"])[~x["enabled"]], rtol=0., atol=0.)
    io = HERE / "training-io.npz"
    np.savez_compressed(io, normalized_input=x["z"].numpy(), p0=p0, prob_trained=p.numpy(), enabled=source["enabled"],
                        C022_loss=original_loss, C023_loss=after.numpy(), C023_prediction=pred.numpy(),
                        C023_delta=delta.numpy(), relative_loss_weight=normalization.numpy(),
                        record_ordinal=source["source_record_ordinal"], native_release=source["native_release"])
    checkpoint = HERE / "router.pt"
    torch.save({"state_dict": {k: t.detach().cpu() for k, t in model.state_dict().items()}, "p0": x["prob0"], "log_p0": x["log0"],
                "feature_mean": torch.from_numpy(protocol["feature_mean"]), "feature_std": torch.from_numpy(protocol["feature_std"]),
                "optimizer_steps": STEPS, "dtype": "float64", "protocol_sha256": protocol["sha"], "C022_coefficient_lock_sha256": lock_hash}, checkpoint)
    return {"pid": os.getpid(), "fit_count": 1, "steps": STEPS, "parameters": 50, "fit_and_archive_seconds": perf_counter() - tick,
            "domains": summaries(after.numpy(), original_loss, p.numpy(), source), "parameter_order": ["weight[2,24] row-major", "bias[2]"],
            "feature_shape": list(x["z"].shape), "training_projection_sha256": array_sha(pred.numpy(), "<f8"),
            "artifacts_sha256": {p.name: sha(p) for p in (trace, io, init_file, checkpoint)}}

@torch.inference_mode()
def infer_probs(window, q_last, fixed, router):
    model, average, scale, log0 = router
    models, mean, std, *_ = fixed
    feat = c18.features(window, q_last, models[0], mean, std)
    p = weights(model, torch.from_numpy((feat.pooled[:, 1].astype(np.float64) - average) / scale), log0).numpy()
    base.require(p.shape == (len(window), 3) and np.isfinite(p).all() and np.all(p >= 0.)
                 and np.max(np.abs(p.sum(axis=1) - 1.)) <= 5e-16, "Invalid causal routing weights")
    return p, feat.valid


def augment(original, unnormalized, extra):
    out = original.copy()
    changed = np.any(extra != 0., axis=1)
    out[changed] = c14.normalize(unnormalized[changed] + extra[changed])
    np.testing.assert_array_equal(out[~changed], original[~changed])
    return out


def held(q, at, release, w, acc, raw, observed, dt, k, fixed, router):
    base_direction, center, _, rest, valid = c22.base_packet(q, at, release, w, acc, observed, dt, fixed)
    fast, slow, mix, fq, sq, _, _ = c22.filters(raw, observed, acc, q, dt, at)
    active = valid & ~rest
    a = k * np.where(active[:, None], mix, 0.)
    p, valid2 = infer_probs(w, q[at], fixed, router)
    np.testing.assert_array_equal(valid2, valid)
    extra = np.where(active[:, None], p[:, :1] * fast + p[:, 1:2] * slow, 0.)
    base.require(np.max(np.linalg.norm(extra, axis=1)) <= c22.CAP + 2e-15 and np.all(extra[:, 2] == 0.),
                 "Convex auxiliary correction exceeds the prescribed cap")
    results = {"C020_deg": base_direction, "C022_deg": augment(base_direction, center, a), "C023_deg": augment(base_direction, center, extra)}
    return results, p, extra, active, rest, fq, sq


def audit(cfg, k, fixed, router):
    notes = []
    for hz in (48., 100., 200.):
        dt = 1. / hz
        optical = hz if hz < 100. else None
        raw, observed = c14.synthetic(hz)
        _, q, at, release, w, _, acc, _ = c17.front_end(raw, observed, dt, cfg, optical)
        result, p, vectors, allowed, rest, fq, sq = held(q, at, release, w, acc, raw, observed, dt, k, fixed, router)
        cuts = {len(at) // 2, len(at) - 2}
        late = np.flatnonzero(release > at)
        if len(late):
            cuts.add(int(late[len(late) // 2]))
        future_error, future_p = 0., 0.
        for j in sorted(cuts):
            changed = raw.copy()
            changed[at[j] + 1:, :3] += [17., -8., 7.]
            changed[at[j] + 1:, 3:] += 1.1
            observation = observed.copy()
            observation[at[j] + 2::2] = False
            end, trunc = j + 1, int(release[j]) + 1
            for x, obs in ((changed, observation), (raw[:trunc], observed[:trunc])):
                _, q2, a2, r2, w2, _, ac2, _ = c17.front_end(x, obs, dt, cfg, optical)
                np.testing.assert_array_equal(a2[:end], at[:end])
                np.testing.assert_array_equal(r2[:end], release[:end])
                out, p2, shift, active, static, _, _ = held(q2, a2[:end], r2[:end], w2[:end], ac2, x, obs, dt, k, fixed, router)
                np.testing.assert_array_equal(active, allowed[:end])
                np.testing.assert_array_equal(static, rest[:end])
                future_error = max(future_error, float(c20.angles(out["C023_deg"], result["C023_deg"][:end]).max()))
                future_p = max(future_p, float(np.max(np.abs(p2 - p[:end]))))
                np.testing.assert_allclose(shift, vectors[:end], rtol=0., atol=3e-8)
        base.require(future_error <= 1e-5 and future_p <= 5e-7, "Router uses the unobservable future")
        rotated_rows = []
        rg, rf, rs = (Rotation.from_quat(x, scalar_first=True) for x in (q[at], fq, sq))
        rp = Rotation.from_quat(q[release], scalar_first=True)
        for name, rot in zip(("90x", "90y", "90z", "63_1_2_-1"),
             (*Rotation.from_rotvec(np.eye(3) * np.pi / 2.),
              Rotation.from_rotvec(np.deg2rad(63.) * np.array([1., 2., -1.]) / np.sqrt(6.))), strict=True):
            changed_w = w.copy()
            changed_w[..., :12] = rot.apply(w[..., :12].reshape(-1, 3)).reshape(len(w), 64, 12)
            qa, qr_f, qr_s = ((x * rot.inv()).as_quat(scalar_first=True) for x in (rg, rf, rs))
            rotated_probs, valid = infer_probs(changed_w, qa, fixed, router)
            np.testing.assert_array_equal(valid & ~rest, allowed)
            f, s = c22.responses(qa, qr_f, qr_s)
            aux = np.where(allowed[:, None], rotated_probs[:, :1] * f + rotated_probs[:, 1:2] * s, 0.)
            p_error = float(np.max(np.abs(p - rotated_probs)))
            d_error = float(np.max(np.abs(aux - vectors)))
            base.require(p_error <= 5e-7 and d_error <= 5e-8, "Installation change affects the router in a fixed world")
            y = rot.apply(raw.reshape(-1, 3)).reshape(raw.shape)
            _, q3, a3, r3, w3, _, acc3, _ = c17.front_end(y, observed, dt, cfg, optical)
            np.testing.assert_array_equal(r3, release)
            out3, _, _, _, _, _, _ = held(q3, a3, r3, w3, acc3, y, observed, dt, k, fixed, router)
            mapped = Rotation.from_quat(q3[r3], scalar_first=True).inv().apply(out3["C023_deg"])
            raw_error = float(c20.angles(rp.inv().apply(result["C023_deg"]), rot.inv().apply(mapped)).max())
            rotated_rows.append({"rotation": name, "fixed_world_probability_max_abs": p_error, "fixed_world_delta_max_abs": d_error,
                                 "reinitialized_native_chain_max_deg": raw_error})
        notes.append({"native_hz": hz, "releases": len(w), "active_releases": int(allowed.sum()), "cuts": sorted(cuts),
                      "publication_later_than_input_count": len(late), "future_output_max_deg": future_error,
                      "future_prob_max_abs": future_p, "rotations": rotated_rows})
    return notes


def metrics(rows):
    keep = [r for r in rows if r["status"] == "scored"]
    macro = {k: fmean(r[k] for r in keep) for k in METHODS} if keep else {}
    active = sum(r["active_release_count"] for r in keep)
    weighted = {}
    if active:
        weighted = {key: fmean(r[f"p_{key}_mean"] for r in keep if r["active_release_count"] > 0) for key in WEIGHTS}
    return {"records": len(keep), "expected_records": len(rows), "frames": sum(r["scored_frames"] for r in keep),
            "macro_tilt_rmse_deg": macro, "active_releases": active, "mean_routing_prob_equal_records_with_support": weighted,
            "C023_delta_deg": {m: macro["C023_deg"] - macro[m] for m in METHODS if m != "C023_deg"} if keep else {},
            "C023_wins": {m: sum(r["C023_deg"] < r[m] for r in keep) for m in METHODS if m != "C023_deg"}}


def add_baselines(score, archived, fixed_row):
    gap = max(abs(score[key] - archived[key]) for key in ("full_vqf_deg", "C020_deg", "C022_deg"))
    base.require(gap <= 1e-6 and np.isfinite(list(score.values())).all(), f"Frozen C022 controls changed by {gap}")
    values = {m: fixed_row[prior] for m, prior in FIXED.items()}
    archive_gap = abs(values["fixed_tau3_deg"] - score["full_vqf_deg"])
    base.require(archive_gap <= 2e-4 and abs(fixed_row["candidate_deg"] - score["C020_deg"]) <= 1e-6,
                 "Archived fixed VQF table differs from published C020 references")
    return {**score, **values}, gap, archive_gap


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
        for old in prior["sequences"]:
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
                predictions, probes, delta, active, rest, _, _ = held(q, at, release, hist, acc, raw, observed, dt, k, fixed, router)
                pred_seconds = perf_counter() - tick
                gate = gates[old["mask_key"]]
                np.testing.assert_array_equal(release, gate[:, 0])
                np.testing.assert_array_equal(rest, gate[:, 3])
                errs = c17.measure(ix, ref, q0, q, release, {m: np.vstack((UP, p)) for m, p in predictions.items()})
                score, gap, fixed_gap = add_baselines(c17.scores(errs), old, fixed_ref)
                active_scored = np.r_[False, active][np.searchsorted(release, ix, side="right")]
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C020_deg"][~active_scored])
                np.testing.assert_array_equal(errs["C023_deg"][~active_scored], errs["C022_deg"][~active_scored])
                active_probs = dict(zip(WEIGHTS, probes[active].mean(axis=0).tolist(), strict=True)) if active.any() else dict.fromkeys(WEIGHTS)
                detail = []
                for pose, selector in axes.items():
                    use = selector[ix]
                    archived = old_poses[key[1], pose]
                    sig = array_sha(ix[use], "<i8")
                    base.require(sig == archived["scoring_indices_sha256"] and int(use.sum()) == archived["scored_frames"], "Frozen C022 array pose differs")
                    value = c17.scores(errs, use)
                    cgap = max(abs(value[k] - archived[k]) for k in ("full_vqf_deg", "C020_deg", "C022_deg"))
                    base.require(cgap <= 1e-6, "Pose control comparison failed")
                    detail.append({"record": key[1], "pose": pose, "scored_frames": int(use.sum()), "scoring_indices_sha256": sig,
                                   **value, "control_max_abs_delta_deg": cgap})
                rows.append({**identity, **score, "status": "scored", "control_max_abs_deg": gap, "fixed_tau3_max_abs_vs_FullVQF": fixed_gap,
                             "native_input_last_sha256": old["native_input_last_sha256"], "native_release_sha256": old["native_release_sha256"],
                             "active_release_count": int(active.sum()), "total_release_count": len(release),
                             "active_scored_frames": int(active_scored.sum()), "inactive_scored_frames": int((~active_scored).sum()),
                             **{f"p_{k}_mean": v for k, v in active_probs.items()}, "active_routing_sha256": array_sha(probes[active], "<f8"),
                             "C023_delta_sha256": array_sha(delta, "<f8"), "prediction_seconds": pred_seconds, "seconds": perf_counter() - start})
                poses.extend(detail)
                print("development", *key, *(score[m] for m in METHODS[:4]), flush=True)
            except (AssertionError, RuntimeError, OSError, ValueError, KeyError, IndexError) as exc:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(exc)})
                rows.append({**identity, "status": "failed", "error": repr(exc), "seconds": perf_counter() - start})
                print("FAILED", *key, repr(exc), flush=True)
    seq, pos, group, fixed_file = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json", "fixed-vqf-comparison.json"))
    fields = dict.fromkeys(f for r in rows for f in r)
    intrinsic.shice.save_csv(seq, [{f: r.get(f, "") for f in fields} for r in rows])
    if poses:
        intrinsic.shice.save_csv(pos, poses)
    groups = {"TUM_rooms": {r["record"]: metrics([r]) for r in rows if r["dataset"] == "TUM-VI"},
              "Oxford_carry": {g: metrics([r for r in rows if r["dataset"] == "OxIOD" and r["group"] == g])
                               for g in sorted({r["group"] for r in rows if r["dataset"] == "OxIOD"})},
              "group": {f"{d}/{g}": metrics([r for r in rows if r["dataset"] == d and r["group"] == g])
                        for d, g in sorted({(r["dataset"], r["group"]) for r in rows})}}
    write(group, groups)
    overall, by_domain = metrics(rows), {d: metrics([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]}
    write(fixed_file, {"archived_controls_sha256": sha(ROOT / "review/vqf_response_controls_development.json"),
                       "matched_legacy_source_C020_csv_sha256": fixed_scores["source_csv_sha256"],
                       "archived_fixed_results_reused": True, "each_record_mask_identity_checked": True,
                       "global": overall, "domains": by_domain, "TUM_rooms": groups["TUM_rooms"], "Oxford_carry": groups["Oxford_carry"]})
    return {"pooled_records": overall, "domains": by_domain, "TUM_rooms": groups["TUM_rooms"], "Oxford_carry": groups["Oxford_carry"],
            "sequences": rows, "array_sensor_poses": poses, "failures": failures,
            "artifact_sha256": {p.name: sha(p) for p in (seq, pos, group, fixed_file) if p.is_file()}}

def status(protocol, phase, **details):
    write(HERE / "process.json", {"pid": os.getpid(), "original_fit_pid": protocol["pid"], "status": phase,
                                  "started_at": protocol["started_at"], "updated_at": stamp(),
                                  "protocol_sha256": sha(HERE / "protocol.json"), **details})


def after_lock(protocol, recovery_sha=None):
    branch, weights_path, train_path = (HERE / n for n in ("router-lock.json","router.pt","training.json"))
    frozen = read(branch)
    sig = sha(branch)
    base.require(frozen["checkpoint_sha256"] == sha(weights_path) and frozen["training_sha256"] == sha(train_path)
                 and frozen["protocol_sha256"] == sha(HERE / "protocol.json") and frozen["fit_count"] == 1
                 and frozen["C022_coefficient_lock_sha256"] == sha(ROOT / "C022/coefficient-lock.json")
                 and all(sha(ROOT / n) == h for n, h in protocol["inputs_sha256"].items()), "Archived and fitted controls changed")
    c22_protocol = read(ROOT / "C022/protocol.json")
    fixed, campaign = c22.load_fixed(c22_protocol)
    ckpt = torch.load(weights_path, weights_only=True, map_location="cpu")
    model = nn.Linear(24, 2, dtype=torch.float64)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().requires_grad_(False)
    k = protocol["C022_k"]
    np.testing.assert_array_equal(ckpt["p0"].numpy(), prior_probs(k).numpy())
    base.require(ckpt["protocol_sha256"] == sha(HERE / "protocol.json") and ckpt["optimizer_steps"] == STEPS
                 and ckpt["C022_coefficient_lock_sha256"] == frozen["C022_coefficient_lock_sha256"], "Saved C023 readout differs")
    router = model, ckpt["feature_mean"].numpy(), ckpt["feature_std"].numpy(), ckpt["log_p0"]
    cfg = c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
    tag = {"router_lock_sha256": sig, "recovery_sha256": recovery_sha}
    status(protocol, "causal_rotation_audits", **tag)
    path = HERE / "causal-rotation-audit.json"
    write(path, {**tag, "checks": audit(cfg, k, fixed, router)})
    status(protocol, "development_replay", **tag)
    print("C023 router locked. Replaying the original 213 records.", flush=True)
    begin = perf_counter()
    result = replay(read(ROOT / "C022/results.json"), read(ROOT / "review/vqf_response_controls_development.json"),
                    read(ROOT / "C001/inventory.json"), campaign, read(TRANSFER / "external-split.json"), cfg, k, fixed, router)
    base.require(sha(branch) == sig and sha(weights_path) == frozen["checkpoint_sha256"]
                 and all(sha(ROOT / file) == h for file, h in protocol["inputs_sha256"].items()), "A fixed receipt changed in replay")
    result.update({**tag, "pid": os.getpid(), "original_fit_pid": protocol["pid"], "started_at": protocol["started_at"],
                   "completed_at": stamp(), "protocol_sha256": sha(HERE / "protocol.json"), "scoring_masks_sha256": sha(MASKS),
                   "C022_coefficient_lock_sha256": frozen["C022_coefficient_lock_sha256"],
                   "causal_rotation_audit_sha256": sha(path), "development_replay_seconds": perf_counter() - begin,
                   "archived_fixed_controls_reused": True, "final_holdout_scored": False})
    out = HERE / "results.json"
    write(out, result)
    good = not result["failures"] and result["pooled_records"]["records"] == 213 and len(result["array_sensor_poses"]) == 96
    status(protocol, "complete" if good else "incomplete", **tag, scored=result["pooled_records"]["records"], results_sha256=sha(out))
    print("C023 finished", result["pooled_records"], "failures", result["failures"], flush=True)
    if not good:
        raise RuntimeError("C023 incomplete development replay; router remains locked")


def main():
    launch, init = (read(HERE / n) for n in ("launch.json", "input-verification.json"))
    files = init["files_sha256"]
    base.require(os.getpid() == launch["pid"] == init["pid"] and all(sha(ROOT / n) == h for n,h in files.items()),
                 "C023 registration or frozen input differs")
    c22_protocol, c22_lock = (read(ROOT / "C022" / n) for n in ("protocol.json", "coefficient-lock.json"))
    base.require(c22_lock["fit_count"] == 1 and c22_lock["C019_sampling_lock_sha256"] == files["C019/sampling-lock.json"]
                 and c22_protocol["scoring_masks_sha256"] == sha(MASKS) and len(c22_protocol["development_records"]) == 213,
                 "Frozen C022 schedule or mask differs")
    c19_readout = torch.load(ROOT / "C019/readout.pt", map_location="cpu", weights_only=True)
    mean, std = (c19_readout[k].numpy().astype(np.float64) for k in ("feature_mean", "feature_std"))
    base.require(mean.shape == std.shape == (24,) and np.all(std > 0.) and np.isfinite(mean).all() and np.isfinite(std).all(),
                 "C019 causal B readout normalizer invalid")
    protocol_path = HERE / "protocol.json"
    protocol = {**launch, "code_sha256": sha(Path(__file__)), "inputs_sha256": files, "C022_k": c22_lock["k"],
                "C022_coefficient_lock_sha256": files["C022/coefficient-lock.json"],
                "C019_sampling_lock_sha256": files["C019/sampling-lock.json"],
                "feature": "Archived C019 pooled[:,1], frozen B6 mean(H[:56])-mean(H[-8:]), normalized by C019 feature_mean/std",
                "p0": prior_probs(c22_lock["k"]).tolist(), "architecture": "One zero-initialized 24 to 2 linear; fixed third logit; float64",
                "parameters": 50, "fit_count": 1, "optimizer": {"kind": "AdamW", "lr": LR, "weight_decay": WD,
                           "clip_global_norm": CLIP, "steps": STEPS, "batch_size": BATCH},
                "loss": "40 * sum(w_i * squared_chord_i / L_C022[domain_i]), the full-domain locked C022 chord losses",
                "inactive": "All extra correction zero when the original gate is off",
                "train_scope": "C022 training slots and C019 latent/weights only. Diagnostic audit before optimizer; no raw or development input during fit.",
                "development_records": c22_protocol["development_records"], "scoring_masks_sha256": sha(MASKS)}
    write(protocol_path, protocol)
    status(protocol, "archived_training_preparation")
    torch.set_num_threads(2)
    original, source, x = load_train(c22_lock, files, mean, std)
    model = nn.Linear(24, 2, dtype=torch.float64)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    status(protocol, "single_router_fit")
    training = train(model, original, source, x, {"sha": sha(protocol_path), "k": c22_lock["k"], "feature_mean": mean, "feature_std": std},
                     files["C022/coefficient-lock.json"])
    train_path, branch = HERE / "training.json", HERE / "router-lock.json"
    write(train_path, {**training, "C022_training_slots_sha256": files["C022/training-slots.npz"],
                       "normalized_feature_source_sha256": files["C019/training-inputs.npz"]})
    with branch.open("x", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "locked_at": stamp(), "fit_count": 1, "protocol_sha256": sha(protocol_path),
                   "training_sha256": sha(train_path), "checkpoint_sha256": sha(HERE / "router.pt"),
                   "C022_coefficient_lock_sha256": files["C022/coefficient-lock.json"], "optimizer_steps": STEPS}, f, indent=2)
        f.write("\n")
    after_lock(protocol)


def resume():
    protocol, lock, state = (read(HERE / n) for n in ("protocol.json","router-lock.json","process.json"))
    base.require(state["status"] != "complete" and protocol["pid"] == lock["pid"] == state["original_fit_pid"], "C023 may only replay a previously locked incomplete fit")
    path = HERE / "replay-recovery.json"
    write(path, {"pid": os.getpid(), "original_fit_pid": lock["pid"], "started_at": stamp(), "fit_repeated": False,
                 "router_lock_sha256": sha(HERE / "router-lock.json"),
                 "original_failure_sha256": {n: sha(HERE / n) for n in ("failure.json","failure-traceback.log","results.json") if (HERE / n).is_file()}})
    after_lock(protocol, sha(path))


if __name__ == "__main__":
    if "--resume" in sys.argv[1:]:
        resume()
    else:
        main()
