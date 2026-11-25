"""Rebuild the fixed C015 source slots, optimize one frozen readout, and replay 213 development records."""

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
C015 = CAMPAIGN / "C015"
sys.path.insert(0, str(C015))

import fit_phone as c15
import numpy as np
import torch
from scipy.io import loadmat

from pareto_readout import ARMIJO, MAX_HALVINGS, MAX_STEPS, QP_TOL, STEP_CAP, domain_loss, optimize

c12, c6, c5, c4, c3, c2 = c15.c12, c15.c6, c15.c5, c15.c4, c15.c3, c15.c2
base, intrinsic = c15.base, c15.intrinsic
read, write, digest, stamp, array_sha = c15.read, c15.write, c15.digest, c15.stamp, c15.array_sha
C001, C002, C003, C004, C006 = c15.C001, c15.C002, c15.C003, c15.C004, c15.C006
DATA, PILOT, TRANSFER, MASKS, GATES = c15.DATA, c15.PILOT, c15.TRANSFER, c15.MASKS, c15.GATES
DOMAINS = ("BROAD", "Sassari", "YT_Server", "RIDI")
METHODS = ("full_vqf_deg", "C006_deg", "C015_deg", "C016_deg")
SAMPLING = C015 / "sampling.npz"


def prepare_slots(old, inventory, cfg, core, mean, std):
    p4 = read(C004 / "protocol.json")
    sources, archived = old["legacy_train_sources"], read(C004 / "training-preparation.json")["records"]
    phones = old["ridi_sources"]
    prior_prep = read(C015 / "training-preparation.json")
    base.require(sources == p4["source_records"] and len(sources) == 63
                 and tuple(r["record"] for r in phones) == c15.ORDER, "C015 source identities changed")
    with np.load(SAMPLING, allow_pickle=False) as file:
        past = {k: file[k] for k in file.files}
    with np.load(C004 / "schedules.npz", allow_pickle=False) as file:
        schedule, offsets, releases = (file[k] for k in ("B", "record_offsets", "release_indices"))
    np.testing.assert_array_equal(past["C004_B_global_window"], schedule)
    mask = past["replacement_mask"]
    np.testing.assert_array_equal(mask, np.arange(96)[None, :] % 12 == np.arange(160)[:, None] % 12)
    draws = past["C015_global_window"]
    np.testing.assert_array_equal(draws[~mask], schedule[~mask])
    np.testing.assert_array_equal(past["C015_native_release"][~mask], releases[schedule[~mask]])
    pool, rests, input_last, names, mats = [], [], [], [], {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, (spec, pre) in enumerate(zip(sources, archived, strict=True)):
            data, gate, at, details, _ = c12.point_source(spec, pre, releases[offsets[i]:offsets[i + 1]], yt["train"],
                                                         mats, core, mean, std, cfg)
            expected = prior_prep["original_sources"][i]
            base.require(all(expected[k] == v for k, v in details.items()), f"Frozen C015 readout inputs changed: {spec['record']}")
            pool.append(data)
            rests.append(gate)
            input_last.append(at)
            names.append((spec["domain"], spec["record"]))
    phone_tracks, local = (past[k] for k in ("RIDI_track_ordinal", "RIDI_drawn_index_within_track"))
    phone_offsets = past["RIDI_track_eligible_offsets"]
    base.require(np.all((phone_tracks >= 0) & (phone_tracks < 4)) and len(phone_tracks) == len(local) == 1280
                 and np.all((local >= 0) & (local < np.diff(phone_offsets)[phone_tracks])),
                 "Invalid original C015 RIDI slot addresses")
    phone_release_list = []
    for i, spec in enumerate(phones):
        data, masks, audit, (raw, observed, dt) = c15.phone_points(spec, cfg, core, mean, std)
        base.require(audit == prior_prep["ridi_eligibility"][i], f"Frozen C015 phone data changed: {spec['record']}")
        for key, array in masks.items():
            np.testing.assert_array_equal(array, past[spec["record"] + "_" + key])
        _, at, release, window, _, _ = c4.inertial_replay(raw, observed, dt, cfg, training=True)
        eligible = masks["eligible_release"]
        j = np.searchsorted(release, eligible)
        np.testing.assert_array_equal(release[j], eligible)
        np.testing.assert_array_equal(at[j], masks["input_last"])
        base.require(len(j) == phone_offsets[i + 1] - phone_offsets[i], "C015 phone eligibility order changed")
        pool.append(data)
        rests.append(c6.rest_flags(window)[j])
        input_last.append(at[j])
        phone_release_list.append(eligible)
        names.append(("RIDI", spec["record"]))
    index = phone_offsets[phone_tracks] + local
    np.testing.assert_array_equal(draws[mask], offsets[-1] + index)
    np.testing.assert_array_equal(past["C015_native_release"][mask], np.concatenate(phone_release_list)[index])
    np.testing.assert_array_equal(past["C015_record_ordinal"][mask], len(sources) + phone_tracks)
    np.testing.assert_array_equal(past["C015_record_ordinal"][~mask], np.searchsorted(offsets[1:], schedule[~mask], side="right"))
    flat = draws.ravel()
    values = np.concatenate(pool)[flat]
    gate = np.concatenate(rests)[flat]
    mapping = past["C015_record_ordinal"].ravel()
    domain = np.asarray([DOMAINS.index(d) for d, _ in names])[mapping]
    counts = np.bincount(mapping, minlength=len(names))
    base.require(draws.shape == (160, 96) and np.all(counts > 0) and len(values) == 15360
                 and counts[-4:].tolist() == [320] * 4, "C015 original record or slot population differs")
    counts_domain = np.bincount([DOMAINS.index(d) for d, _ in names], minlength=4)
    weight = 1. / (counts[mapping] * counts_domain[domain])
    np.testing.assert_allclose(np.bincount(domain, weights=weight), np.ones(4), rtol=0., atol=1e-12)
    last = np.concatenate(input_last)[flat]
    release = past["C015_native_release"].ravel()
    base.require(np.all(last <= release), "A source input uses a future native frame")
    evidence = {"C015_sampling_sha256": digest(SAMPLING), "schedule_sha256": array_sha(draws, "<i8"),
                "native_releases_sha256": array_sha(release, "<i8"), "rest_flags_sha256": array_sha(gate, "?"),
                "draw_count": len(flat), "record_count": len(names),
                "domains": {d: {"records": int(counts_domain[i]), "draws": int(np.count_nonzero(domain == i)),
                                "rest_draws": int(np.count_nonzero((domain == i) & gate))}
                            for i, d in enumerate(DOMAINS)},
                "source_records": [{"domain": d, "record": name, "draws": int(counts[i]),
                                    "rest_draws": int(np.count_nonzero((mapping == i) & gate))}
                                   for i, (d, name) in enumerate(names)]}
    path = HERE / "training-windows.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, values=values, rest_gate=gate, per_draw_weight=weight, domain_ordinal=domain,
                            source_record_ordinal=mapping, C015_global_window=flat, native_release=release, native_input_last=last)
    evidence["training_windows_sha256"] = digest(path)
    return values, gate, domain, weight, evidence, path


def domain_indices(domain, weight, dtype, device):
    return [(torch.from_numpy(ix).to(device), torch.from_numpy(weight[ix]).to(device=device, dtype=dtype))
            for ix in (np.flatnonzero(domain == j) for j in range(len(DOMAINS)))]


@torch.inference_mode()
def verify_deployment(initial, final, values, rest, domain, weight, device):
    x = torch.from_numpy(values).to(device)
    g = torch.from_numpy(rest).to(device)
    indices = domain_indices(domain, weight, x.dtype, device)
    before = domain_loss(initial.to(device=device, dtype=x.dtype), x, g, indices, base.CAP, c4.GAIN)
    after = domain_loss(final.to(device=device, dtype=x.dtype), x, g, indices, base.CAP, c4.GAIN)
    base.require(torch.isfinite(before).all() and torch.isfinite(after).all() and torch.all(before > 0),
                 "Deployment verification has invalid source loss")
    return {"initial_loss": before.tolist(), "deployed_loss": after.tolist(),
            "relative_to_C015_float32": (after / before).tolist(),
            "not_worse_by_domain": (after <= before).tolist(), "strictly_lower_by_domain": (after < before).tolist(),
            "all_four_not_worse": bool((after <= before).all())}


def metric_summary(rows):
    good = [r for r in rows if r["status"] == "scored"]
    values = {m: float(np.mean([r[m] for r in good])) for m in METHODS} if good else {}
    return {"denominator": len(rows), "scored_records": len(good), "scored_frames": sum(r["scored_frames"] for r in good),
            "macro_rmse_deg": values,
            "C016_delta_macro_deg": {m: values["C016_deg"] - values[m] for m in METHODS[:-1]} if good else {},
            "C016_wins_vs": {m: sum(r["C016_deg"] < r[m] for r in good) for m in METHODS[:-1]}}


def score_record(old, spec, map3, map1, medium, yt, split, c1p, array, evidence, masks, gates, core, layers, mean, std, cfg):
    key = old["dataset"], old["record"]
    if key[0] in ("OxIOD", "RepoIMU"):
        raw, q, at, release, history, mask, ref, qa, _, _ = c5.optical_payload(spec, cfg)
        raw_q, q = raw.as_quat(scalar_first=True), q.as_quat(scalar_first=True)
        base.require(qa["quality_flags"] == old["reference_quality_flags"], "Optical QA changed")
        axes = {}
    else:
        raw, dt, observed, ref, mask, axes = c3.sources(map3[key], spec, map1.get(key), medium, yt, split, c1p, array,
                                                        evidence["array_raw_sha256"])
        clean = base.causal_fill(raw, observed)
        raw_q = base.VQF(dt).updateBatch(np.ascontiguousarray(clean[:, 3:]), np.ascontiguousarray(clean[:, :3]))["quat6D"]
        q, at, release, history, _, bias = c4.inertial_replay(raw, observed, dt, cfg)
        base.require(bias["first_active_native_index"] == map3[key]["bias"]["first_active_native_index"], "C003 bias activation changed")
    np.testing.assert_array_equal(mask, masks[old["mask_key"]])
    base.require(intrinsic.mask_digest(mask) == old["scoring_indices_sha256"] and int(mask.sum()) == old["scored_frames"],
                 "C015 development mask changed")
    rest = c6.rest_flags(history)
    frozen_gate = gates[old["mask_key"]]
    np.testing.assert_array_equal(release, frozen_gate[:, 0])
    np.testing.assert_array_equal(rest, frozen_gate[:, 3])
    delta = c4.evaluate_windows(core, layers, history, mean, std)
    for v in delta.values():
        v[rest] = 0.
    candidates = {k: (k, c4.GAIN, False) for k in layers}

    def scorer(include, prior):
        pure = intrinsic.score_attitudes(raw_q, ref, include, raw_q[at], release, {}, {}, 1.)["full_vqf"]
        out = intrinsic.score_attitudes(q, ref, include, q[at], release, delta, candidates, 1.)
        values = {"full_vqf_deg": pure, **{name + "_deg": out[name] for name in layers}}
        gap = max(abs(values[k] - prior[k]) for k in METHODS[:-1])
        base.require(np.isfinite(list(values.values())).all() and gap <= 1e-6, f"C015 control changed by {gap}")
        return {**values, "control_max_abs_error_deg": gap}

    return scorer(mask, old), scorer, axes, mask


def replay(prior, inventory, campaign, split, core, layers, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in obj["sequences"]} for obj in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and set(specs) == keys
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "Scoring scope is outside the frozen development panel")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(path)
    old_poses = {(a["record"], a["pose"]): a for a in prior["array_sensor_poses"]}
    rows, poses, failures = [], [], []
    identity = ("dataset", "record", "group", "instrument", "scenario", "reference_QA", "reference_status",
                "mask_key", "scored_frames", "scoring_indices_sha256")
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            key = old["dataset"], old["record"]
            try:
                scores, score, axes, mask = score_record(old, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                                        evidence, masks, gates, core, layers, mean, std, cfg)
                row = {**{k: old[k] for k in identity}, **scores, "status": "scored", "rest_scored_frames": old["rest_scored_frames"]}
                detail = []
                for axis, mask_pose in axes.items():
                    use = mask_pose & mask
                    reference = old_poses[key[1], axis]
                    base.require(intrinsic.mask_digest(use) == reference["scoring_indices_sha256"]
                                 and int(use.sum()) == reference["scored_frames"], "Array pose mask differs")
                    detail.append({**{k: reference[k] for k in ("record", "pose", "scored_frames", "scoring_indices_sha256")},
                                   **score(use, reference)})
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as error:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(error)})
                rows.append({**{k: old[k] for k in identity}, "status": "failed", "error": repr(error)})
                print("FAILED", *key, repr(error), flush=True)
    csv_path, group_path, pose_path = (HERE / n for n in ("sequences.csv", "group-summaries.json", "array-sensor-poses.csv"))
    fields = (*identity, "status", "rest_scored_frames", *METHODS, "control_max_abs_error_deg", "error")
    intrinsic.shice.save_csv(csv_path, [{k: row.get(k, "") for k in fields} for row in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    groups = {field: {f"{d}/{label}": metric_summary([r for r in rows if r["dataset"] == d and r[field] == label])
                      for d, label in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("reference_QA", "instrument", "scenario")}
    write(group_path, groups)
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": {d: metric_summary([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "optical_subsets": {k: metric_summary(v) for k, v in subsets.items()}, "csv_sha256": digest(csv_path),
            "pose_csv_sha256": digest(pose_path) if poses else None, "group_summaries_sha256": digest(group_path),
            "control_max_abs_delta_deg": max((r["control_max_abs_error_deg"] for r in rows if r["status"] == "scored"), default=None),
            "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    checkpoint, iterations = HERE / "common-descent-readout.pt", HERE / "iterations.jsonl"
    base.require(not protocol_path.exists() and not checkpoint.exists() and not iterations.exists(), "C016 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p15, t15, r15, s15 = (read(C015 / n) for n in ("protocol.json", "training.json", "results.json", "process.json"))
    p6, r6 = (read(C006 / n) for n in ("protocol.json", "results.json"))
    p3 = read(C003 / "protocol.json")
    inventory_path, split_path = C001 / "inventory.json", TRANSFER / "external-split.json"
    inventory = read(inventory_path)
    base.require(s15["status"] == "complete" and not r15["failures"] and len(r15["sequences"]) == 213
                 and digest(C015 / "results.json") == s15["results_sha256"]
                 and digest(C015 / "protocol.json") == r15["protocol_sha256"] == s15["protocol_sha256"]
                 and digest(C015 / "training.json") == r15["training_sha256"]
                 and digest(C015 / "sequences.csv") == r15["csv_sha256"]
                 and digest(C015 / "array-sensor-poses.csv") == r15["pose_csv_sha256"]
                 and all(digest(C015 / p) == h for p, h in t15["files_sha256"].items())
                 and all(digest(PILOT / p) == h for p, h in p15["files_sha256"].items())
                 and digest(SAMPLING) == t15["sampling_sha256"]
                 and digest(C006 / "protocol.json") == r6["protocol_sha256"]
                 and digest(MASKS) == r6["shared_masks_sha256"] == p15["scoring_masks_sha256"] == r15["scoring_masks_sha256"]
                 and digest(GATES) == r6["gate_traces_sha256"] == p15["gate_traces_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == p3["estimator_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and all(digest(c5.REVIEW / n) == h for n, h in p6["optical_reference_sha256"].items()),
                 "C015 source, protocol, checkpoint or scoring evidence changed")
    base.require([[r["dataset"], r["record"]] for r in r15["sequences"]] == p15["development_records"]
                 == [[r["dataset"], r["record"]] for r in r6["sequences"]], "Development panel changed")
    original = PILOT / campaign["incumbent"]
    control = C004 / read(C004 / "training.json")["arms"]["B"]["weights"]
    start = C015 / "phone-readout.pt"
    checks = {original: campaign["incumbent_sha256"], control: p6["B_checkpoint_sha256"],
              start: t15["files_sha256"][start.name]}
    base.require(all(digest(path) == h for path, h in checks.items()), "Frozen checkpoint hash changed")
    b6, c006, c015 = (torch.load(path, weights_only=True, map_location="cpu") for path in checks)
    backbone = {k: v for k, v in c015["state_dict"].items() if not k.startswith("output.")}
    norm = c4.tensor_hash({k: c015[k] for k in ("mean", "std")})
    base.require(c015["protocol_sha256"] == digest(C015 / "protocol.json") and c015["sampling_sha256"] == digest(SAMPLING)
                 and c015["C015_optimizer_steps"] == 160 and c015["model"] == "mamba_v1" and c015["seed"] == 1729
                 and norm == p15["fixed_normalization_sha256"] == t15["fixed_normalization_sha256"]
                 and c4.tensor_hash(backbone) == t15["B6_backbone_sha256"]
                 and all(torch.equal(c015[k], obj[k]) for obj in (b6, c006) for k in ("mean", "std"))
                 and all(torch.equal(v, obj["state_dict"][k]) for k, v in backbone.items() for obj in (b6, c006)),
                 "Fixed deployment weights or normalization differ")
    files = dict(p15["files_sha256"])
    extra = (Path(__file__).resolve(), HERE / "pareto_readout.py", C015 / "fit_phone.py", C015 / "protocol.json",
             C015 / "training.json", C015 / "training-preparation.json", C015 / "results.json", C015 / "process.json",
             C015 / "sequences.csv", C015 / "array-sensor-poses.csv", C006 / "results.json", SAMPLING, *checks.keys())
    files.update({str(p.relative_to(PILOT)): digest(p) for p in extra})
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": files,
                "initial_C015_checkpoint_sha256": digest(start), "fixed_normalization_sha256": norm,
                "fixed_backbone_sha256": c4.tensor_hash(backbone), "C015_sampling_sha256": digest(SAMPLING),
                "domains": DOMAINS, "source_weighting": "Mean squared direction chord length within a source record's sampled slots. Equal weight to each record within its domain. Divide each domain loss by its own fixed C015 initial loss.",
                "training_gate": "Original default VQF restDetected at the input-last block. Neural delta is exactly zero during rest, and the slot retains its original weight in the per-record loss.",
                "optimized": "Only 75 output parameters, packed as weight row-major then bias. Frozen C015 backbone features. CPU float64 full batches.",
                "QP": "Enumerate nonempty faces of the four-dimensional simplex and solve face KKT equations for the minimum L2 norm of the convex combination. Validate scaled primal/dual/complementarity residuals.",
                "QP_scaled_residual_tolerance": QP_TOL, "common_direction": "Negative normalized minimum-norm combination.",
                "line_search": {"initial_parameter_l2_upper": STEP_CAP, "Armijo": ARMIJO,
                                "halvings": MAX_HALVINGS, "each_domain_must_pass": True,
                                "maximum_accepted_steps": MAX_STEPS},
                "stop_scope": "External optimization only. Accept up to 80 steps or record numerically zero direction/no strict common direction/all-domain Armijo exhausted.",
                "deployment": "Cast once to original float32 readout and verify all four external losses. Freeze that candidate and score the original 213 development records with original C006 inference.",
                "development_records": p15["development_records"], "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES)}
    write(protocol_path, protocol)
    ph = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "pid": os.getpid(), "started_at": started, "protocol_sha256": ph, **extra})

    status("external_preparation")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "Exact C015 frozen feature path needs original CUDA device")
    device = torch.device("cuda")
    core = base.ResidualHead(c015["model"])
    core.load_state_dict(c015["state_dict"])
    core.eval().requires_grad_(False).to(device)
    mean, std = c015["mean"].to(device), c015["std"].to(device)
    cfg = c3.Settings(**p3["thresholds"])
    values, gate, domain, weight, sample, witness = prepare_slots(p15, inventory, cfg, core, mean, std)
    sample_path = HERE / "training-preparation.json"
    write(sample_path, sample)
    base.require(all(digest(PILOT / name) == h for name, h in files.items())
                 and digest(witness) == sample["training_windows_sha256"], "An input changed before C016 optimization")
    cpu = torch.device("cpu")
    inputs = torch.from_numpy(values).double()
    rest = torch.from_numpy(gate)
    indices = domain_indices(domain, weight, torch.float64, cpu)
    initial = torch.cat((c015["state_dict"]["output.weight"].flatten(), c015["state_dict"]["output.bias"])).double()
    base.require(initial.numel() == 75 and len(inputs) == 15360, "Unexpected C016 objective shape")
    status("external_optimization", training_windows_sha256=digest(witness))
    print("C016 actual C015 source slots and rest flags locked", sample["domains"], flush=True)
    output, fit, steps = optimize(initial, inputs, rest, indices, base.CAP, c4.GAIN, iterations, stamp,
                                  lambda a: array_sha(a, "<f8"))
    status("float32_verification", accepted_steps=fit["accepted_steps"])
    cpu32 = verify_deployment(initial, output, values, gate, domain, weight, cpu)
    gpu32 = verify_deployment(initial, output, values, gate, domain, weight, device)
    stage = checkpoint.with_suffix(".tmp")
    size = c015["state_dict"]["output.weight"].numel()
    weight32 = output[:size].reshape_as(c015["state_dict"]["output.weight"]).float()
    bias32 = output[size:].float()
    state = {**c015["state_dict"], "output.weight": weight32, "output.bias": bias32}
    torch.save({**c015, "state_dict": state, "C016_start_checkpoint_sha256": digest(start), "C016_protocol_sha256": ph,
                "C016_accepted_steps": fit["accepted_steps"]}, stage)
    stage.replace(checkpoint)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    changed = {k for k, v in saved["state_dict"].items() if not torch.equal(v, c015["state_dict"][k])}
    base.require(changed <= {"output.weight", "output.bias"} and all(torch.equal(saved[k], c015[k]) for k in ("mean", "std")),
                 "C016 changed a frozen tensor")
    layer = deepcopy(core.output)
    layer.load_state_dict({"weight": saved["state_dict"]["output.weight"], "bias": saved["state_dict"]["output.bias"]})
    control_layer = deepcopy(core.output)
    control_layer.load_state_dict({"weight": c006["state_dict"]["output.weight"], "bias": c006["state_dict"]["output.bias"]})
    active = [x for x in steps if "accepted_step" in x]
    training_path, deployment_path = HERE / "training.json", HERE / "float32-verification.json"
    write(deployment_path, {"domain_order": DOMAINS, "cpu_float32": cpu32, "original_cuda_float32": gpu32,
                           "parameter_change_l2_float32": float((output.float().double() - initial).norm()),
                           "changed_readout_elements_float32": int(torch.count_nonzero(output.float() - initial.float())),
                           "candidate_checkpoint_sha256": digest(checkpoint)})
    write(training_path, {"fit_count": 1, "domain_order": DOMAINS, "fit": fit, "changed_tensors": sorted(changed),
                          "max_scaled_qp_residual": max(x["qp"]["max_scaled_kkt_residual"] for x in steps),
                          "accepted_halvings": [x["accepted_halvings"] for x in active],
                          "artifacts_sha256": {p.name: digest(p) for p in (witness, sample_path, iterations, checkpoint, deployment_path)}})
    base.require(digest(start) == protocol["initial_C015_checkpoint_sha256"], "C015 starting checkpoint was modified")
    status("development_replay", training_sha256=digest(training_path), accepted_steps=fit["accepted_steps"])
    print("C016 float32 candidate fixed; scoring original 213 development records", gpu32, flush=True)
    tick = perf_counter()
    result = replay(r15, inventory, campaign, read(split_path), core,
                    {"C006": control_layer, "C015": core.output, "C016": layer}, mean, std, cfg)
    completed = stamp()
    result.update({"started_at": started, "completed_at": completed, "protocol_sha256": ph,
                   "training_sha256": digest(training_path), "scoring_masks_sha256": digest(MASKS),
                   "accepted_optimizer_steps": fit["accepted_steps"], "scoring_seconds": perf_counter() - tick,
                   "total_seconds": perf_counter() - wall})
    path = HERE / "results.json"
    write(path, result)
    status("partial" if result["failures"] else "complete", completed_at=completed, results_sha256=digest(path))
    print("C016 coverage", len(result["sequences"]), "failures", result["failures"], "seconds", result["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process = HERE / "process.json"
        if process.exists():
            previous = read(process)
            if previous["pid"] == os.getpid():
                write(process, {**previous, "phase": previous["status"], "status": "failed",
                                "completed_at": stamp(), "failure": repr(error)})
        raise
