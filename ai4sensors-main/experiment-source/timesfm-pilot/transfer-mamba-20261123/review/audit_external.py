"""Independently cross-check external split membership and default FullVQF from the source MAT files."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from vqf import VQF


HERE = Path(__file__).resolve().parent
BATCH = HERE.parent
PILOT = BATCH.parent
UP = np.array([0., 0., 1.])


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_external(row):
    path = Path(row["path"])
    if row["domain"] == "BROAD":
        src = loadmat(path, variable_names=["imu_acc", "imu_gyr", "opt_quat", "movement", "sampling_rate"])
        imu = np.column_stack((src["imu_acc"], src["imu_gyr"]))
        truth = src["opt_quat"]
        eligible = src["movement"].ravel().astype(bool)
        hz = float(src["sampling_rate"].item())
    else:
        sensor = row["name"].split("/")[1]
        src = loadmat(path, variable_names=["Qs", sensor])
        imu, truth = src[sensor][:, 1:7], src["Qs"]
        eligible = np.ones(len(truth), dtype=bool)
        hz = 100.
    return imu, truth, eligible, hz


def score_raw(row):
    imu, truth, eligible, hz = load_external(row)
    observed = np.isfinite(imu).all(axis=1)
    previous = np.maximum.accumulate(np.where(observed, np.arange(len(imu)), -1))
    signal = imu[previous.clip(min=0)].copy()
    signal[previous < 0] = 0.
    est = VQF(1./hz).updateBatch(np.ascontiguousarray(signal[:,3:], dtype=np.float64),
                                  np.ascontiguousarray(signal[:,:3], dtype=np.float64))["quat6D"]
    take = observed & eligible & np.isfinite(truth).all(axis=1) & np.any(truth != 0, axis=1)
    idx = np.flatnonzero(take)
    ref = Rotation.from_quat(truth[idx], scalar_first=True).inv().apply(UP)
    up = Rotation.from_quat(est[idx], scalar_first=True).inv().apply(UP)
    rad = np.arctan2(np.linalg.norm(np.cross(ref,up),axis=1), np.einsum("ij,ij->i",ref,up))
    angles = np.rad2deg(rad)
    warm = idx/hz >= 10.
    return {"domain":row["domain"],"name":row["name"],"group":row["group"],"hz":hz,
            "frames":len(idx),"index_sha256":hashlib.sha256(idx.astype("<i8").tobytes()).hexdigest(),
            "default_rmse_deg":float(np.sqrt(np.mean(angles**2))),"warm_frames":int(warm.sum()),
            "warm_index_sha256":hashlib.sha256(idx[warm].astype("<i8").tobytes()).hexdigest(),
            "warm_rmse_deg":float(np.sqrt(np.mean(angles[warm]**2)))}


def main():
    split_file = BATCH/"external-split.json"
    split = read(split_file)
    all_paths = {Path(r["path"]) for r in split["records"]}
    assert {str(n):digest(n) for n in all_paths} == split["source_file_sha256"]
    sides = defaultdict(set)
    physical_files = defaultdict(set)
    for r in split["records"]:
        if r["domain"] == "BROAD":
            family = re.sub(r"_(?:[A-D]|[0-9]+s|[0-9]+cm)$", "", r["name"].split("_",1)[1])
        else:
            assert r["domain"] == "Sassari"
            family = r["name"].split("/")[0]
        sides[r["domain"],family].add(r["split"])
        physical_files[r["path"]].add(r["split"])
    assert all(len(v)==1 for v in [*sides.values(),*physical_files.values()])
    historical = read(PILOT/"overnight-20261122/rounds/R01826/result.json")["datasets"]
    expected = {(domain,r["name"]):r for domain in ("BROAD","Sassari") for r in historical[domain]["sequences"]}
    assert len(split["records"]) == len(expected) == len({(r["domain"],r["name"]) for r in split["records"]})
    assert set(expected) == {(r["domain"],r["name"]) for r in split["records"]}
    validation = [score_raw(r) for r in split["records"] if r["split"] == "validation"]
    for r in validation:
        original = expected[r["domain"],r["name"]]
        assert r["frames"] == original["frames"] and r["index_sha256"] == original["score_frame_indices_sha256"]
        assert abs(r["default_rmse_deg"]-original["rmse_deg"]["full_vqf"]) < 1.e-4
    counts = {d:dict(Counter(r["split"] for r in split["records"] if r["domain"]==d)) for d in ("BROAD","Sassari")}
    result = {"split_sha256":digest(split_file),"raw_file_count":len(all_paths),"records_by_domain":counts,
              "native_validation":validation}
    (HERE/"external_source_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"counts":counts,"scored_validation":len(validation),"frames":sum(r["frames"] for r in validation)}))


if __name__ == "__main__":
    main()
