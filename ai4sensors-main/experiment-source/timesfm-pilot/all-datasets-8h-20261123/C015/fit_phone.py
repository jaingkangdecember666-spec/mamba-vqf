"""Fit one frozen-B6 readout with eight fixed RIDI source slots in each C004-B step."""

import csv
from collections import Counter
from copy import deepcopy
from datetime import datetime
import inspect
import os
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
sys.path.insert(0, str(CAMPAIGN / "C012"))

import fit_rest_gated as c12
import numpy as np
import torch
from scipy.io import loadmat
from tilt_eval import normalize, rotate

c7, c6, c5, c4, c3, c2 = c12.c7, c12.c6, c12.c5, c12.c4, c12.c3, c12.c2
base, intrinsic = c12.base, c12.intrinsic
read, write, digest, stamp, array_sha = c12.read, c12.write, c12.digest, c12.stamp, c12.array_sha
C001, C002, C003, C004, C006 = c12.C001, c12.C002, c12.C003, c12.C004, c12.C006
DATA, PILOT, TRANSFER, MASKS, GATES = c12.DATA, c12.PILOT, c12.TRANSFER, c12.MASKS, c12.GATES
RIDI = CAMPAIGN / "external-sources" / "ridi-phone-train"
ORDER = ("hao_bag1", "hao_body1", "hao_handheld1", "hao_leg1")
RATE, WARMUP_NS = 200, 10_000_000_000
METHODS = ("full_vqf_deg", "C006_deg", "C015_deg")


def verify_phone_input(record, signal, observed, time):
    """Prove NPZ input is a one-sided hold of the original accelerometer and gyro readings."""
    columns, valid, ages = [], np.ones(len(time), dtype=bool), []
    for field in ("acce", "gyro"):
        path = RIDI / "original" / record["record"] / (field + ".txt")
        data = np.loadtxt(path, dtype=[("time", "<i8"), ("sensor", "<f8", (3,))], comments="#")
        native, first = np.unique(data["time"], return_index=True)
        previous = np.searchsorted(native, time, side="right") - 1
        base.require(previous.min() >= 0, "A prepared phone sample precedes the first sensor observation")
        age = time - native[previous]
        data_aligned = data["sensor"][first[previous]]
        valid &= np.isfinite(data_aligned).all(axis=1) & (age >= 0) & (age <= 20_000_000)
        columns.append(data_aligned)
        ages.append(int(age[observed].max()))
    actual = np.column_stack(columns)
    actual[~valid] = np.nan
    np.testing.assert_array_equal(valid, observed)
    np.testing.assert_array_equal(actual, signal)
    np.testing.assert_allclose(np.asarray(ages) / 1.e6, record["max_used_acc_and_gyro_age_ms"], rtol=0., atol=1e-9)
    return ages


def phone_points(record, cfg, core, mean, std):
    path = RIDI / record["file"]
    n = record["native_200hz_rows"]
    with np.load(path, allow_pickle=False) as data:
        base.require(set(data.files) == {"time_ns", "acc_m_s2", "gyr_rad_s", "imu_observed", "tango_wxyz", "reference_valid"},
                     f"Invalid phone schema: {path.name}")
        time, acc, gyro, observed = (data[k] for k in ("time_ns", "acc_m_s2", "gyr_rad_s", "imu_observed"))
        base.require(time.shape == observed.shape == (n,) and time.dtype == np.int64 and observed.dtype == np.bool_
                     and acc.shape == gyro.shape == (n, 3) and np.all(np.diff(time) == 5_000_000), "Invalid phone IMU grid")
        imu = np.column_stack((acc, gyro))
        base.require(np.array_equal(np.isfinite(imu).all(axis=1), observed) and observed.sum() == record["observed_imu_rows"],
                     "Stored phone IMU mask changed")
        ages = verify_phone_input(record, imu, observed, time)
        # The front end consumes only acceleration, gyro, validity, and the regular sampling period.
        q, at, release, history, complete, bias = c4.inertial_replay(imu, observed, 1. / RATE, cfg, training=True)
        tango, valid = data["tango_wxyz"], data["reference_valid"]
    base.require(tango.shape == (n, 4) and valid.shape == (n,) and valid.dtype == np.bool_,
                 "Invalid Tango reference array")
    reference, pose = base.reference_direction(tango)
    base.require(int((valid & pose & observed).sum()) == record["paired_pose_imu_rows"], "Original Tango mask changed")
    after10 = (time[release] - time[0]) >= WARMUP_NS
    usable_time = complete & observed[release] & valid[release] & pose[release] & after10
    selected = np.flatnonzero(usable_time)
    base.require(len(selected) > 0 and np.all(at[selected] <= release[selected]), "No causal phone release with a label")
    last_rotation = normalize(q[at[selected]])
    inverse = last_rotation.copy()
    inverse[:, 1:] *= -1.
    target_world = rotate(normalize(q[release[selected]]), reference[release[selected]])
    aligned = rotate(inverse, target_world)
    error = float(np.max(np.abs(rotate(last_rotation, aligned) - target_world)))
    latent, up, basis = c4.features(core, history[selected], mean, std)
    target = intrinsic.transform_directions(aligned, basis)
    use = basis.valid
    base.require(use.any() and error <= 2e-12, "Phone point-target transform failed")
    np.testing.assert_allclose(np.einsum("ni,nji->nj", target[use], basis.basis[use]), aligned[use], rtol=0., atol=1e-7)
    positions = selected[use]
    if np.array_equal(at[positions], release[positions]):
        np.testing.assert_allclose(aligned[use], reference[release[positions]], rtol=0., atol=1e-12)
    points = np.c_[latent[use], up[use], target[use]]
    archive = {"observed_imu": observed, "reference_valid": valid, "eligible_release": release[positions],
               "input_last": at[positions], "grid_relative_time_ns": time - time[0]}
    audit = {**record, "raw_imu_sha256": array_sha(imu, "<f8"), "original_accelerometer_sha256": array_sha(acc, "<f8"),
             "original_gyro_sha256": array_sha(gyro, "<f8"), "tango_wxyz_sha256": array_sha(tango, "<f8"),
             "original_imu_mask_sha256": array_sha(observed, "?"), "original_reference_mask_sha256": array_sha(valid, "?"),
             "one_sided_original_IMU_exact": True, "max_original_hold_age_ns": ages,
             "scheduled_windows": len(release), "source_eligible_before_geometry": len(selected),
             "eligible_windows": int(use.sum()), "excluded_degenerate": int((~use).sum()),
             "first_eligible_release": int(release[positions[0]]), "last_eligible_release": int(release[positions[-1]]),
             "eligible_release_sha256": array_sha(release[positions], "<i8"), "point_targets_sha256": array_sha(target[use], "<f4"),
             "target_world_roundtrip_max_abs": error, "C003_active_native_frames": bias["active_native_frames"]}
    return points, archive, audit, (imu, observed, 1. / RATE)


def design_sampling(schedule, offsets, original_releases, phones):
    mask = np.arange(96)[None, :] % 12 == np.arange(160)[:, None] % 12
    base.require(schedule.shape == mask.shape and np.all(mask.sum(axis=1) == 8), "Wrong replacement mask")
    phone = np.tile(np.arange(len(ORDER), dtype=np.int64), 320)
    base.require(len(phone) == mask.sum(), "Replacement track assignment differs")
    rng = np.random.default_rng(c4.SEED)
    indices = np.array([rng.integers(len(phones[j][0])) for j in phone], dtype=np.int64)
    phone_offsets = np.r_[0, np.cumsum([len(row[0]) for row in phones])]
    selected_pool = phone_offsets[phone] + indices
    archive = {"replacement_mask": mask, "C004_B_global_window": schedule,
               "C004_B_record_ordinal": np.searchsorted(offsets[1:], schedule, side="right"),
               "C004_B_native_release": original_releases[schedule]}
    actual = schedule.copy()
    actual[mask] = offsets[-1] + selected_pool
    new_ordinal, new_release = (archive[k].copy() for k in ("C004_B_record_ordinal", "C004_B_native_release"))
    new_ordinal[mask] = len(offsets) - 1 + phone
    eligible = np.concatenate([a["eligible_release"] for _, a, _, _ in phones])
    new_release[mask] = eligible[selected_pool]
    archive.update({"C015_global_window": actual, "C015_record_ordinal": new_ordinal,
                    "C015_native_release": new_release, "RIDI_track_ordinal": phone,
                    "RIDI_drawn_index_within_track": indices, "RIDI_track_eligible_offsets": phone_offsets})
    for m, (_, detail, _, _) in enumerate(phones):
        archive.update({f"{ORDER[m]}_{key}": value for key, value in detail.items()})
    np.testing.assert_array_equal(actual[~mask], schedule[~mask])
    np.testing.assert_array_equal(new_release[~mask], original_releases[schedule[~mask]])
    np.testing.assert_array_equal(np.bincount(phone), np.full(len(ORDER), 320))
    return actual, mask, archive


def summarize(rows):
    valid = [r for r in rows if r["status"] == "scored"]
    values = {k: float(np.mean([r[k] for r in valid])) for k in METHODS} if valid else {}
    return {"denominator": len(rows), "scored_records": len(valid), "scored_frames": sum(r["scored_frames"] for r in valid),
            "macro_rmse_deg": values,
            "C015_delta_macro_deg": {k: values["C015_deg"] - values[k] for k in METHODS[:-1]} if valid else {},
            "C015_wins_vs": {k: sum(r["C015_deg"] < r[k] for r in valid) for k in METHODS[:-1]}}


def replay(prior, inventory, campaign, split, core, control, layer, mean, std, cfg):
    c1r, c1p, c3r, c2p = (read(p) for p in (C001 / "results.json", C001 / "protocol.json",
                                             C003 / "results.json", C002 / "protocol.json"))
    evidence = c2p["frozen_evidence"]
    map1, map3 = ({(r["dataset"], r["record"]): r for r in item["sequences"]} for item in (c1r, c3r))
    keys = {(r["dataset"], r["record"]) for r in prior["sequences"]}
    specs = {(r["dataset"], r["record"]): r for r in inventory["records"] if (r["dataset"], r["record"]) in keys}
    base.require(len(keys) == 213 and keys == set(specs)
                 and all(r["side"] in ("development", "development_previously_used_for_selection") for r in specs.values()),
                 "Score scope is outside the frozen development panel")
    medium_path = DATA / "Sassari/medium_v5.mat"
    base.require(digest(medium_path) == split["source_file_sha256"][str(medium_path)], "Sassari medium source changed")
    medium = loadmat(medium_path, variable_names=("Qs", "XS1", "XS2", "AP1", "AP2", "SH1", "SH2"))
    array_path = DATA / "IMU-Array-main/Calibration/Group1_ref_data.bin"
    base.require(digest(array_path) == evidence["array_reference_sha256"], "IMU-Array reference changed")
    array = c2.reference_array(array_path)
    poses_prior = {(r["record"], r["pose"]): r for r in prior["array_sensor_poses"]}
    layers = {"c003": core.output, "C006": control.output, "C007": layer}
    rows, poses, failures = [], [], []
    identity = ("dataset", "record", "group", "instrument", "scenario", "reference_QA", "reference_status",
                "mask_key", "scored_frames", "scoring_indices_sha256")
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt, np.load(MASKS, allow_pickle=False) as masks, np.load(GATES, allow_pickle=False) as gates:
        for old in prior["sequences"]:
            key = old["dataset"], old["record"]
            try:
                # The frozen C007 replay slot uses the same C006 rest inference rule needed here.
                row, detail = c7.evaluate(old, specs[key], map3, map1, medium, yt["valid"], split, c1p, array,
                                         evidence["array_raw_sha256"], masks, gates, core, layers, mean, std, cfg)
                row["C015_deg"] = row.pop("C007_deg")
                row.pop("c003_deg")
                row["reference_QA"] = old["reference_QA"]
                row["rest_scored_frames"] = old["rest"]["scored_frames_with_rest"]
                for pose in detail:
                    control_pose = poses_prior[pose["record"], pose["pose"]]
                    base.require(pose["scoring_indices_sha256"] == control_pose["scoring_indices_sha256"]
                                 and abs(pose["C006_deg"] - control_pose["B_rest_gated_deg"]) <= 1e-6
                                 and abs(pose["full_vqf_deg"] - control_pose["full_vqf_deg"]) <= 1e-6,
                                 "C006 IMU-Array pose score or mask changed")
                    pose["C015_deg"] = pose.pop("C007_deg")
                    pose.pop("c003_deg")
                rows.append(row)
                poses.extend(detail)
                print(*key, row["scored_frames"], *(row[m] for m in METHODS), flush=True)
            except (AssertionError, RuntimeError, ValueError, OSError, KeyError, IndexError) as error:
                failures.append({"dataset": key[0], "record": key[1], "error": repr(error)})
                rows.append({**{k: old[k] for k in identity}, "status": "failed", "error": repr(error)})
                print("FAILED", *key, repr(error), flush=True)
    csv_path, pose_path, groups_path = (HERE / n for n in ("sequences.csv", "array-sensor-poses.csv", "group-summaries.json"))
    fields = (*identity, "status", "rest_scored_frames", *METHODS, "error")
    intrinsic.shice.save_csv(csv_path, [{k: row.get(k, "") for k in fields} for row in rows])
    if poses:
        intrinsic.shice.save_csv(pose_path, poses)
    groups = {field: {f"{d}/{label}": summarize([r for r in rows if r["dataset"] == d and r[field] == label])
                      for d, label in sorted({(r["dataset"], r[field]) for r in rows})}
              for field in ("reference_QA", "instrument", "scenario")}
    write(groups_path, groups)
    subsets = {"Oxford_verified": [r for r in rows if r["dataset"] == "OxIOD" and r["reference_QA"] == "verified"],
               "Repo_geometry_verified": [r for r in rows if r["dataset"] == "RepoIMU"
                                          and r["reference_QA"] in ("verified", "geometry_verified_relative_level")]}
    return {"summary": {d: summarize([r for r in rows if r["dataset"] == d]) for d in campaign["datasets"]},
            "optical_subsets": {k: summarize(v) for k, v in subsets.items()}, "csv_sha256": digest(csv_path),
            "pose_csv_sha256": digest(pose_path) if poses else None, "group_summaries_sha256": digest(groups_path),
            "control_max_abs_delta_deg": max((abs(v) for r in rows for v in r.get("controls_minus_prior_deg", {}).values()), default=None),
            "sequences": rows, "array_sensor_poses": poses, "failures": failures}


def main():
    wall, started = perf_counter(), stamp()
    protocol_path, process_path = HERE / "protocol.json", HERE / "process.json"
    checkpoint, sampling_path = HERE / "phone-readout.pt", HERE / "sampling.npz"
    base.require(not protocol_path.exists() and not checkpoint.exists() and not sampling_path.exists(), "C015 already started")
    campaign = read(CAMPAIGN / "protocol.json")
    base.require(datetime.fromisoformat(campaign["start"]) <= datetime.fromisoformat(started)
                 < datetime.fromisoformat(campaign["end"]), "Outside campaign window")
    p4, r4, t4, prep4 = (read(C004 / n) for n in ("protocol.json", "results.json", "training.json", "training-preparation.json"))
    p6, prior = (read(C006 / n) for n in ("protocol.json", "results.json"))
    bias_protocol = read(C003 / "protocol.json")
    inventory_path, split_path = C001 / "inventory.json", TRANSFER / "external-split.json"
    inventory = read(inventory_path)
    schedule_path, old_steps = C004 / "schedules.npz", C004 / "training-steps.csv"
    incumbent = PILOT / campaign["incumbent"]
    control_path = C004 / t4["arms"]["B"]["weights"]
    c12_path = CAMPAIGN / "C012"
    witness = c12_path / "training-windows.npz"
    base.require(len(prior["sequences"]) == 213 and not prior["failures"]
                 and digest(C006 / "protocol.json") == prior["protocol_sha256"]
                 and digest(C004 / "protocol.json") == r4["protocol_sha256"]
                 and digest(C004 / "training-preparation.json") == r4["training_preparation_sha256"]
                 and digest(schedule_path) == r4["schedules_sha256"] == t4["schedules_sha256"]
                 and digest(old_steps) == t4["training_steps_csv_sha256"]
                 and digest(MASKS) == prior["shared_masks_sha256"] and digest(GATES) == prior["gate_traces_sha256"]
                 and digest(inventory_path) == p6["development_inventory_sha256"]
                 and digest(C003 / "protocol.json") == p6["C003_bias_protocol_sha256"]
                 and digest(C003 / "online_bias.py") == bias_protocol["estimator_sha256"]
                 and digest(split_path) == p4["source_split_sha256"] == inventory["prior_external_split_sha256"]
                 and digest(incumbent) == p4["base_checkpoint_sha256"] == p6["B6_checkpoint_sha256"] == campaign["incumbent_sha256"]
                 and digest(control_path) == p6["B_checkpoint_sha256"] == t4["arms"]["B"]["weights_sha256"]
                 and digest(witness) == read(c12_path / "training.json")["files_sha256"][witness.name],
                 "Frozen source, C006 comparison or point-label witness changed")
    base.require(all(digest(c5.REVIEW / n) == h for n, h in p6["optical_reference_sha256"].items()),
                 "Optical reference implementation changed")
    base.require((p4["seed"], p4["steps_per_arm"], p4["batch_size"], p4["lr"], p4["weight_decay"], p4["inference_gain"]) ==
                 (c4.SEED, c4.STEPS, c4.BATCH, c4.LR, c4.WD, c4.GAIN), "C004 optimizer changed")
    legacy, archived = p4["source_records"], prep4["records"]
    base.require(Counter(r["domain"] for r in legacy) == {"BROAD": 27, "Sassari": 12, "YT_Server": 24}
                 and all(all(r[k] == v for k, v in spec.items()) for spec, r in zip(legacy, archived, strict=True))
                 and p4["yt_prefix_exclusion"]["test_overlap"] == p4["yt_prefix_exclusion"]["diagnostic_valid_overlap"] == [],
                 "C004-B external manifest changed")
    with np.load(schedule_path, allow_pickle=False) as file:
        schedule, offsets, releases = (file[k] for k in ("B", "record_offsets", "release_indices"))
    base.require(schedule.shape == (160, 96) and np.array_equal(np.diff(offsets), [r["usable"] for r in archived]),
                 "B scheduling changed")
    source_audit = RIDI / "source-audit.json"
    directions_path, agreement_path = (c5.REVIEW / n for n in ("ridi_training_directions.json", "source_gradient_agreement.json"))
    ridi, directions, agreement = read(source_audit), read(directions_path), read(agreement_path)
    phones = ridi["rows"]
    base.require(tuple(row["record"] for row in phones) == ORDER and ridi["selected_subject"] == "hao"
                 and not ridi["subject_intersection_with_official_test_and_cross"]
                 and directions["source_sha256"] == digest(source_audit)
                 and tuple(row["record"] for row in directions["rows"]) == ORDER, "RIDI source-only provenance changed")
    list_files = {}
    sets = {}
    for partition, record in ridi["lists"].items():
        name = Path(record["archive_entry"]).name
        source = RIDI / "original" / name
        base.require(digest(source) == record["sha256"], "RIDI split registry changed")
        list_files[partition] = source
        sets[partition] = {line.split(",", 1)[0].strip() for line in source.read_text(encoding="utf-8").splitlines() if line.strip()}
    base.require(set(ORDER) <= sets["train"] and not set(ORDER) & (sets["test"] | sets["cross"])
                 and not any(k.startswith("hao_") for k in sets["test"] | sets["cross"])
                 and {k: len(v) for k, v in sets.items()} == ridi["official_list_counts"], "RIDI selection crosses a source partition")
    phone_files = {}
    for phone in phones:
        base.require(phone["subject"] == "hao" and phone["phase"] == "external_train_only"
                     and phone["file"] == phone["record"] + ".npz", "Invalid RIDI source identity")
        phone_files[RIDI / phone["file"]] = phone["file_sha256"]
        for field in ("acce", "gyro", "pose"):
            phone_files[RIDI / "original" / phone["record"] / (field + ".txt")] = phone["source_files"][field]["sha256"]
    base.require(all(digest(p) == h for p, h in phone_files.items()), "RIDI source file hash changed")
    frozen, control_data = (torch.load(p, map_location="cpu", weights_only=True) for p in (incumbent, control_path))
    base.require(frozen["seed"] == c4.SEED == 1729 and frozen["epoch"] == 6 and frozen["model"] == "mamba_v1"
                 and control_data["c004_arm"] == "B" and control_data["c004_gain"] == c4.GAIN == .25
                 and all(torch.equal(frozen[k], control_data[k]) for k in ("mean", "std"))
                 and all(torch.equal(v, control_data["state_dict"][k]) for k, v in frozen["state_dict"].items() if not k.startswith("output.")),
                 "Original B6 initialization or C006 control changed")
    norms = c4.tensor_hash({k: frozen[k] for k in ("mean", "std")})
    backbone = c4.tensor_hash({k: v for k, v in frozen["state_dict"].items() if not k.startswith("output.")})
    base.require(norms == p4["frozen_normalization_sha256"] and backbone == p4["frozen_backbone_parameter_sha256"],
                 "Frozen B6 features or input normalization differs")
    core = base.ResidualHead(frozen["model"])
    core.load_state_dict(frozen["state_dict"])
    base.require(sum(p.numel() for p in core.output.parameters()) == 75
                 and c4.tensor_hash(core.output.state_dict()) == p4["initial_B_output_parameter_sha256"], "B6 readout initialization changed")
    dependencies = {Path(inspect.getfile(m)).resolve() for m in (c12, c7, c6, c5, c4, c3, c2, c2.c1, base,
                                                                base.ResidualHead, type(core.sequence), intrinsic, intrinsic.build_frame, rotate)}
    dependencies.update(c5.REVIEW / name for name in p6["optical_reference_sha256"])
    dependencies.update((Path(__file__).resolve(), CAMPAIGN / "protocol.json", incumbent, control_path, schedule_path,
                         old_steps, split_path, inventory_path, MASKS, GATES, C004 / "protocol.json",
                         C004 / "training-preparation.json", C004 / "training.json", C006 / "protocol.json",
                         C006 / "results.json", C003 / "protocol.json", C003 / "online_bias.py", C003 / "results.json",
                         C001 / "protocol.json", C001 / "results.json", C002 / "protocol.json", c12_path / "training.json",
                         witness, source_audit, directions_path, agreement_path, *list_files.values(), *phone_files.keys()))
    fingerprints = {str(p.relative_to(PILOT)): digest(p) for p in sorted(dependencies)}
    protocol = {"started_at": started, "pid": os.getpid(), "files_sha256": fingerprints,
                "initial_B6_sha256": digest(incumbent), "initial_B6_seed": frozen["seed"],
                "fixed_backbone_sha256": backbone, "fixed_normalization_sha256": norms,
                "C004_B_draw_matrix_sha256": array_sha(schedule, "<i8"), "C004_B_shape": list(schedule.shape),
                "C006_checkpoint_sha256": digest(control_path), "development_records": [[r["dataset"], r["record"]] for r in prior["sequences"]],
                "scoring_masks_sha256": digest(MASKS), "gate_traces_sha256": digest(GATES),
                "optimizer": {"kind": "AdamW", "steps": c4.STEPS, "batch_size": c4.BATCH, "seed": c4.SEED,
                              "lr": c4.LR, "weight_decay": c4.WD, "clip_grad_norm": 2., "trainable_parameters": 75},
                "sampling": {"replacement": "Zero-based slot j%12==step_index%12. Every other slot retained exactly.",
                             "draw_seed": 1729, "draw_rng": "numpy.default_rng", "retained": 14080, "RIDI": 1280,
                             "per_track_total": 320, "per_step_each_track": 2, "track_order": ORDER,
                             "ridi_slot_assignment": "Manifest order twice per step, in ascending replaced-slot order. Uniform draws with replacement from each fixed eligible pool."},
                "RIDI_input": "Only NPZ acc_m_s2 and gyr_rad_s numerical sensor values. imu_observed supplies the independent packet mask. Audit against original raw acc and gyro files. Arm uses the unchanged causal C003 front end; no reference supplied to it.",
                "RIDI_targets": "Original same-handset Tango wxyz with world +Z. Use pose only to form the original C004 release-point supervision. Release must have valid reference and observed IMU after 10 s, with a complete observed causal history and valid intrinsic basis.",
                "training_and_inference": "C004-B ungated point loss at gain 0.25. Only the original B6 output layer trains. Replay uses unchanged C006 static release policy and original gain.",
                "legacy_train_sources": legacy, "ridi_sources": phones,
                "source_diagnostic_RIDI_BROAD_cosine_C006": agreement["checkpoints"]["C006"]["gradient_cosines"]["BROAD"]["RIDI"],
                "scope": "One readout fit with a fixed eight-slot substitution on the original C004-B schedule. The 213 archived development records only."}
    write(protocol_path, protocol)
    ph = digest(protocol_path)

    def status(phase, **extra):
        write(process_path, {"status": phase, "pid": os.getpid(), "started_at": started, "protocol_sha256": ph, **extra})

    status("external_preparation")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    base.require(torch.cuda.is_available(), "Original B6 CUDA path unavailable")
    device = torch.device("cuda")
    core.eval().requires_grad_(False).to(device)
    control = deepcopy(core)
    control.load_state_dict(control_data["state_dict"])
    mean, std = frozen["mean"].to(device), frozen["std"].to(device)
    cfg = c3.Settings(**bias_protocol["thresholds"])
    pool, old_at, prep, probes, mats = [], [], [], {}, {}
    with c2.c1.h5py.File(inventory["yt"]["path"], "r") as yt:
        for i, (record, old) in enumerate(zip(legacy, archived, strict=True)):
            data, _, at, audit, probe = c12.point_source(record, old, releases[offsets[i]:offsets[i+1]], yt["train"],
                                                        mats, core, mean, std, cfg)
            pool.append(data)
            old_at.append(at)
            prep.append(audit)
    original = np.concatenate(pool)
    original_at = np.concatenate(old_at)
    with np.load(witness, allow_pickle=False) as old:
        np.testing.assert_array_equal(old["original_B_schedule"], schedule)
        np.testing.assert_array_equal(old["native_release"], releases[schedule])
        np.testing.assert_array_equal(old["native_input_last"], original_at[schedule])
        np.testing.assert_array_equal(old["C004_points"], original[schedule])
    phone_data = [phone_points(spec, cfg, core, mean, std) for spec in phones]
    for spec, item in zip(phones, phone_data, strict=True):
        probes[spec["record"]] = item[3]
    mixed, mask, ledger = design_sampling(schedule, offsets, releases, phone_data)
    pool_data = np.concatenate([original, *(item[0] for item in phone_data)])
    np.testing.assert_array_equal(pool_data[mixed[~mask]], original[schedule[~mask]])
    base.require(int(mask.sum()) == 1280 and int((~mask).sum()) == 14080, "Dose changed")
    retained_ordinals = ledger["C015_record_ordinal"]
    for i, row in enumerate(prep):
        row["original_draws"] = int(np.count_nonzero(ledger["C004_B_record_ordinal"] == i))
        row["retained_draws"] = int(np.count_nonzero(retained_ordinals == i))
        base.require(row["original_draws"] == archived[i]["B_draws"], "Original source row draw count changed")
    with sampling_path.open("wb") as stream:
        np.savez_compressed(stream, **ledger)
    prep_path = HERE / "training-preparation.json"
    write(prep_path, {"protocol_sha256": ph, "fixed_normalization_sha256": norms, "sampling_sha256": digest(sampling_path),
                      "retained_draws": int((~mask).sum()), "ridi_draws": int(mask.sum()),
                      "ridi_draws_per_track": dict.fromkeys(ORDER, 320), "retained_points_exact_to_C004_B": True,
                      "original_sources": prep, "ridi_eligibility": [item[2] for item in phone_data]})
    status("sample_frozen", sampling_sha256=digest(sampling_path), preparation_sha256=digest(prep_path))
    print("C015 draw lock passed", "retained", int((~mask).sum()), "phone", int(mask.sum()),
          "eligible phone pools", [item[2]["eligible_windows"] for item in phone_data], flush=True)
    with old_steps.open(encoding="utf-8", newline="") as stream:
        first_b = next(row for row in csv.DictReader(stream) if row["arm"] == "B" and row["step"] == "1")
    tensor = torch.from_numpy(pool_data).to(device)
    with torch.inference_mode():
        ungated_old_loss = float(c12.per_draw_error(core.output, tensor[torch.from_numpy(schedule[0]).to(device)]).mean())
        mixed_first = float(c12.per_draw_error(core.output, tensor[torch.from_numpy(mixed[0]).to(device)]).mean())
    base.require(ungated_old_loss == float(first_b["loss"]), "Original C004 first batch does not reproduce")
    base.require(all(digest(PILOT / path) == sha for path, sha in fingerprints.items())
                 and digest(sampling_path) == read(prep_path)["sampling_sha256"], "An input changed before training")
    status("fitting", optimizer_steps_planned=160, sampling_sha256=digest(sampling_path))
    out, history, timing = c4.optimize("C015", core.output, tensor, mixed, core.output.in_features, device)
    stage = checkpoint.with_suffix(".tmp")
    tensor_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    tensor_state.update({"output." + k: v.detach().cpu().clone() for k, v in out.state_dict().items()})
    torch.save({**frozen, "state_dict": tensor_state, "C015_optimizer_steps": len(history), "sampling_sha256": digest(sampling_path),
                "protocol_sha256": ph, "fixed_normalization_sha256": norms}, stage)
    stage.replace(checkpoint)
    ordinal = ledger["C015_record_ordinal"]
    domain_codes = np.asarray([r["domain"] for r in legacy] + list(ORDER))[ordinal]
    for i, row in enumerate(history):
        counts = {f"retained_{d}": int(np.count_nonzero(domain_codes[i] == d)) for d in c12.DOMAINS}
        phone_counts = {name: int(np.count_nonzero(domain_codes[i] == name)) for name in ORDER}
        base.require(sum(counts.values()) == 88 and set(phone_counts.values()) == {2}, "A fitted batch changed source dose")
        row.update(counts, **phone_counts, retained=88, RIDI=8,
                   record_ordinals_sha256=array_sha(ordinal[i], "<i8"),
                   native_releases_sha256=array_sha(ledger["C015_native_release"][i], "<i8"))
    steps_path = HERE / "training-steps.csv"
    intrinsic.shice.save_csv(steps_path, history)
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    changed = {k for k, v in saved["state_dict"].items() if not torch.equal(v, frozen["state_dict"][k])}
    base.require(changed == {"output.weight", "output.bias"} and all(torch.equal(saved[k], frozen[k]) for k in ("mean", "std"))
                 and history[0]["loss"] == mixed_first, "Initialization, normalization or non-readout weights changed")
    out.load_state_dict({k.removeprefix("output."): saved["state_dict"][k] for k in changed})
    prefix_path = HERE / "prefix-audit.json"
    write(prefix_path, c7.causal_check(probes, core, out, mean, std, cfg))
    tpath = HERE / "training.json"
    write(tpath, {"fit_count": 1, "fit": timing, "B6_backbone_sha256": backbone, "fixed_normalization_sha256": norms,
                  "updated_parameter_tensors": sorted(changed), "C004_B_first_loss_replayed_exactly": ungated_old_loss,
                  "C015_first_loss": mixed_first, "sampling_sha256": digest(sampling_path),
                  "files_sha256": {p.name: digest(p) for p in (checkpoint, sampling_path, prep_path, steps_path, prefix_path)}})
    del tensor, phone_data, probes, mats, pool, pool_data, original
    status("development_replay", training_sha256=digest(tpath))
    print("C015 one readout fit persisted; scoring the original 213 development records", flush=True)
    scoring = perf_counter()
    final = replay(prior, inventory, campaign, read(split_path), core, control, out, mean, std, cfg)
    completed = stamp()
    final.update({"started_at": started, "completed_at": completed, "protocol_sha256": ph, "optimizer_steps": len(history),
                  "training_sha256": digest(tpath), "scoring_masks_sha256": digest(MASKS),
                  "scoring_seconds": perf_counter() - scoring, "total_seconds": perf_counter() - wall})
    result_path = HERE / "results.json"
    write(result_path, final)
    status("partial" if final["failures"] else "complete", completed_at=completed, results_sha256=digest(result_path))
    print("C015 coverage", len(final["sequences"]), "failures", final["failures"], "seconds", final["total_seconds"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        process = HERE / "process.json"
        if process.exists():
            old = read(process)
            if old["pid"] == os.getpid():
                write(process, {**old, "phase": old["status"], "status": "failed", "completed_at": stamp(), "failure": repr(error)})
        raise
