"""Reconcile the prior tilt baselines and replay selected FullVQF records from disk."""

import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from vqf import VQF


HERE = Path(__file__).resolve().parent
PILOT = HERE.parent.parent
DATA = Path("E:/数据集")
SHICE = DATA / "IMU_dynamic_data_SHICE"
AUDIT = SHICE / "Label_quality_audit_20261116"
VERTICAL = (0, 0, 1)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay(path, imu, packet_ok, reference, reference_axis, dt, eligible, recorded):
    observed = packet_ok & np.isfinite(imu).all(axis=1)
    last_observed = np.maximum.accumulate(np.where(observed, np.arange(len(imu)), -1))
    signal = imu[last_observed.clip(min=0)].copy()
    signal[last_observed < 0] = 0
    quats = VQF(dt).updateBatch(np.ascontiguousarray(signal[:, 3:]),
                                np.ascontiguousarray(signal[:, :3]))["quat6D"]
    scored = eligible & observed & np.isfinite(reference).all(axis=1) & (np.linalg.norm(reference, axis=1) > 0)
    predicted = Rotation.from_quat(quats[scored], scalar_first=True).inv().apply(VERTICAL)
    truth = Rotation.from_quat(reference[scored], scalar_first=True).inv().apply(reference_axis)
    angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(predicted, truth), axis=1),
                                    np.einsum("ij,ij->i", predicted, truth)))
    measured = float(np.sqrt(np.mean(angles ** 2)))
    baseline = recorded["rmse_deg"]["full_vqf"]
    assert int(scored.sum()) == recorded["scoring_frames"]
    assert math.isclose(measured, baseline, abs_tol=2e-4), (path, measured, baseline)
    return {"source": str(path), "sha256": digest(path), "scoring_frames": int(scored.sum()),
            "score_index_sha256":hashlib.sha256(np.flatnonzero(scored).astype("<i8").tobytes()).hexdigest(),
            "raw_replay_full_vqf_deg": measured, "historical_deg": baseline}


def replay_shice(name, recorded):
    device = name.split("-")[0]
    path = SHICE / f"{device}_aligned_formal" / f"{name}.csv"
    columns = ["time_s", "is_interpolated", *(f"imu_{kind}_{axis}" for kind in ("acc", "gyr") for axis in "xyz"),
               *(f"opt_quat_{axis}" for axis in "wxyz")]
    with path.open(encoding="utf-8-sig") as f:
        fields = next(csv.reader(f))
        raw = np.loadtxt(f, delimiter=",", quotechar='"', usecols=[fields.index(k) for k in columns])
    with np.load(AUDIT / name / "screening_masks.npz", allow_pickle=False) as mask:
        np.testing.assert_array_equal(mask["sample_index"], np.arange(len(raw)))
        eligible = (raw[:, 0] >= 10) & mask["context_ok"].astype(bool) & mask["observed_loss_ok"].astype(bool)
    dt = float(np.median(np.diff(raw[:, 0])))
    return replay(path, raw[:, 2:8], raw[:, 1] == 0, raw[:, 8:12], (0, -1, 0), dt, eligible, recorded)


def reconcile():
    correction = PILOT / "tilt-correction"
    files = [correction / name for name in ("validation.json", "test.json", "external_test.json")]
    files.append(PILOT / "gru-tilt" / "test.json")
    valid, held, external, gru = map(read, files)
    heads = {name: digest(correction / name) for name in external["checkpoint_sha256"]}
    assert heads == external["checkpoint_sha256"]
    assert digest(Path(gru["selected_checkpoint"])) == gru["selected_sha256"]
    with (AUDIT / "recording_recommendations.csv").open(encoding="utf-8-sig", newline="") as f:
        audit = list(csv.DictReader(f))
    unique = {r["recording"] for r in audit}
    assert len(unique) == len(audit) == 59
    partitions = [set(valid["training_records"]), set(valid["validation_records"]),
                  set(held["test_records"]), set(valid["excluded_records"])]
    assert set.union(*partitions) == unique and sum(map(len, partitions)) == len(unique)
    assert partitions[-1] == {"R1-3", "R1-4"}
    assert all(s.split("-")[0] in {"R1", "R2", "R3", "R4"} for s in partitions[0])
    assert all(s.startswith("R5-") for s in partitions[1]) and all(s.startswith("R6-") for s in partitions[2])

    corrected = {"R6": held["r6_records"]}
    means = {"R6": held["r6_macro_rmse_deg"]}
    for name, data in external["datasets"].items():
        corrected[name] = [dict(recording=r["recording"], **r["protocols"]["main"]) for r in data["records"]]
        means[name] = data["protocols"]["main"]["macro_rmse_deg"]
    tables, rows_by_set = {}, {}
    for name, records in corrected.items():
        rows = {r["recording"]: r for r in records}
        rows_by_set[name] = rows
        g = gru["tests"][name]
        assert len(rows) == len(records) == len(g["per_sequence"]) == g["records"]
        assert set(rows) == {r["recording"] for r in g["per_sequence"]}
        for other in g["per_sequence"]:
            r = rows[other["recording"]]
            assert r["scoring_frames"] == other["scoring_frames"]
            for key in ("full_vqf", "physical_only"):
                assert math.isclose(r["rmse_deg"][key], other["rmse_deg"][key], abs_tol=1e-7)
        n_frames = sum(r["scoring_frames"] for r in records)
        assert n_frames == g["scoring_frames"]
        macro = {k: fmean(r["rmse_deg"][k] for r in records) for k in ("full_vqf", "physical_only", "timesfm")}
        for k, v in macro.items():
            assert math.isclose(v, means[name][k], abs_tol=1e-10)
        macro["gru"] = fmean(r["rmse_deg"]["gru"] for r in g["per_sequence"])
        assert math.isclose(macro["gru"], g["macro_rmse_deg"]["gru"], abs_tol=1e-10)
        tables[name] = {"sequences": len(rows), "scoring_frames": n_frames, "macro_rmse_deg": macro}

    samples = {}
    r5 = next(r for r in valid["r5_records"] if r["recording"] == "R5-7")
    samples["R5-7"] = replay_shice("R5-7", r5)
    samples["R6-1"] = replay_shice("R6-1", rows_by_set["R6"]["R6-1"])
    trial = "01_undisturbed_slow_rotation_A"
    path = DATA / "broad-main" / "broad-main" / "data_mat" / f"{trial}.mat"
    mat = loadmat(path, variable_names=["imu_acc", "imu_gyr", "opt_quat", "movement", "sampling_rate"])
    n = len(mat["opt_quat"])
    samples["BROAD/" + trial] = replay(path, np.column_stack((mat["imu_acc"], mat["imu_gyr"])), np.ones(n, bool),
                                       mat["opt_quat"], VERTICAL, 1 / mat["sampling_rate"].item(),
                                       mat["movement"].ravel().astype(bool), rows_by_set["BROAD"][trial])
    path = DATA / "Sassari" / "slow_v5.mat"
    mat = loadmat(path, variable_names=["XS1", "Qs"])
    include = np.ones(len(mat["Qs"]), bool)
    samples["Sassari/slow/XS1"] = replay(path, mat["XS1"][:, 1:7], include, mat["Qs"], VERTICAL,
                                         0.01, include, rows_by_set["Sassari"]["slow/XS1"])
    return {"reviewed_utc": datetime.now(timezone.utc).isoformat(), "historical_only": True,
            "source_sha256": {str(p.relative_to(PILOT)): digest(p) for p in files},
            "checkpoint_sha256": {**heads, "selected_gru.pt": gru["selected_sha256"]},
            "source_counts": {"train": len(partitions[0]), "valid": len(partitions[1]), "test": len(partitions[2]),
                              "excluded": sorted(partitions[-1]), "audit_status": dict(Counter(r["status"] for r in audit)),
                              "absolute_1deg_certified": sum(r["absolute_1deg_certified"].lower() == "true" for r in audit)},
            "historical_baselines": tables, "raw_replay_samples": samples}


if __name__ == "__main__":
    report = reconcile()
    destination = HERE / "baseline_audit.json"
    staging = destination.with_suffix(".tmp")
    staging.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    staging.replace(destination)
    print(json.dumps(report["historical_baselines"], ensure_ascii=False, indent=2))
