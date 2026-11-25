"""Shared IO, data partition and tilt metrics for the overnight batch."""

import csv
import hashlib
import json
import os
import sys
from time import sleep
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


BATCH = Path(__file__).resolve().parent
PILOT = BATCH.parent
sys.path.insert(0, str(PILOT))
DATA = Path("E:/数据集")
SHICE = DATA / "IMU_dynamic_data_SHICE"
HISTORY = PILOT / "tilt-correction"
CACHE = PILOT / "gru-tilt" / "data"
TZ = timezone(timedelta(hours=8))
START = datetime.fromisoformat("2026-11-22T23:03:02+08:00")
END = datetime.fromisoformat("2026-11-23T09:03:02+08:00")
LOCK = datetime.fromisoformat("2026-11-23T08:03:02+08:00")
Z = np.array([0., 0., 1.])
EXCLUDED = ("R1-3", "R1-4")


def now():
    return datetime.now(TZ)


def stamp():
    return now().isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    staged.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(staged, path)
            return
        except PermissionError as error:
            if os.name != "nt" or error.winerror not in {5, 32, 33} or attempt == 19:
                raise
            sleep(min(0.05 * (attempt + 1), 0.3))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def names(split):
    groups = {"dev": {"R1", "R2", "R3", "R4"}, "r5": {"R5"}, "r6": {"R6"}}
    audit = SHICE / "Label_quality_audit_20261116/recording_recommendations.csv"
    with audit.open(encoding="utf-8-sig", newline="") as stream:
        return [item["recording"] for item in csv.DictReader(stream)
                if item["sensor"] in groups[split] and item["recording"] not in EXCLUDED]


def record(name):
    with np.load(CACHE / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in ("imu", "ref", "score", "context", "dt")}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(min=1e-12)


def rotate(q, v):
    xyz = q[..., 1:]
    cross = np.cross(xyz, v)
    return v + 2 * (q[..., :1] * cross + np.cross(xyz, cross))


def body_up(q, world=Z):
    inverse = q.copy()
    inverse[..., 1:] *= -1
    return rotate(unit(inverse), world)


def errors(a, b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1)))


def rmse(a, b):
    return float(np.sqrt(np.mean(np.square(errors(a, b), dtype=np.float64))))


def aggregate(rows):
    if not rows:
        raise ValueError("No eligible scoring records")
    worst = max(rows, key=lambda r: r["rmse_deg"])
    result = {"macro_rmse_deg": float(np.mean([r["rmse_deg"] for r in rows])),
              "records": len(rows), "scoring_frames": int(sum(r["frames"] for r in rows)),
              "worst_record": worst["name"], "worst_rmse_deg": worst["rmse_deg"]}
    comparable = [r for r in rows if "baseline_deg" in r]
    if comparable:
        worst_delta = max(comparable, key=lambda r: r["rmse_deg"] - r["baseline_deg"])
        result["worse_records"] = sum(r["rmse_deg"] > r["baseline_deg"] + 1e-9 for r in comparable)
        result["worst_delta_deg"] = worst_delta["rmse_deg"] - worst_delta["baseline_deg"]
        result["worst_delta_record"] = worst_delta["name"]
    return result


def by_device(rows):
    devices = sorted({row["name"].split("-")[0] for row in rows})
    result = {}
    for device in devices:
        own = [row for row in rows if row["name"].startswith(device + "-")]
        value = aggregate(own)
        if all("baseline_deg" in row for row in own):
            value["macro_delta_deg"] = value["macro_rmse_deg"] - float(np.mean([row["baseline_deg"] for row in own]))
        result[device] = value
    return result


def vqf_outputs(imu, dt, parameters):
    from vqf import VQF

    return VQF(float(dt), **parameters).updateBatch(np.ascontiguousarray(imu[:, 3:], dtype=np.float64),
                                                   np.ascontiguousarray(imu[:, :3], dtype=np.float64))
