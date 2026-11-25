"""Reconstruct BROAD and Sassari native score frames and replay locked comparators."""

import hashlib
import sys
from collections import defaultdict
from math import isclose
from statistics import fmean

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH
from verify_baselines import DATA, VERTICAL, replay


SENSORS = ("XS1","XS2","AP1","AP2","SH1","SH2")


def motion_groups(dataset, rows, methods):
    """Use the acquisition filename to assign each paired record to one fixed group."""
    expected = ({"rotation":9,"translation":9,"combined":5,"disturbed":16} if dataset == "BROAD"
                else {"slow":6,"medium":6,"fast":6})
    assert dataset in {"BROAD","Sassari"} and len({r["name"] for r in rows}) == len(rows)
    groups = defaultdict(list)
    for row in rows:
        name = row["name"]
        if dataset == "Sassari":
            group, sensor = name.split("/")
            assert sensor in SENSORS
        else:
            parts = name.split("_")
            assert len(parts) >= 3 and parts[0].isdigit()
            if parts[1] == "disturbed":
                group = "disturbed"
            else:
                assert parts[1] == "undisturbed" and parts[2] in {"slow","fast"} and len(parts) >= 4
                group = parts[3]
                assert group != "disturbed"
        assert group in expected, name
        groups[group].append(row)
    assert {group:len(members) for group,members in groups.items()} == expected
    result = {}
    for group in expected:
        members = groups[group]
        result[group] = {"records":len(members),"names":[r["name"] for r in members],
                         "frames":sum(r["frames"] for r in members),
                         "mean_deg":{m:fmean(r["rmse_deg"][m] for r in members) for m in methods}}
    for method in methods:
        aggregate = sum(g["records"] * g["mean_deg"][method] for g in result.values()) / len(rows)
        assert isclose(aggregate,fmean(r["rmse_deg"][method] for r in rows),rel_tol=0,abs_tol=1e-10)
    return result


def raw_records(dataset, names):
    if dataset == "BROAD":
        folder = DATA / "broad-main/broad-main/data_mat"
        for path in sorted(folder.glob("*.mat")):
            if path.stem in names:
                raw = loadmat(path,variable_names=["imu_acc","imu_gyr","opt_quat","sampling_rate","movement"])
                yield (path.stem,path,np.column_stack((raw["imu_acc"],raw["imu_gyr"])),raw["opt_quat"],
                       raw["movement"].ravel().astype(bool),1/float(raw["sampling_rate"].item()))
    elif dataset == "Sassari":
        for speed in ("slow","medium","fast"):
            sensors = [s for s in SENSORS if f"{speed}/{s}" in names]
            if sensors:
                path = DATA / "Sassari" / f"{speed}_v5.mat"
                raw = loadmat(path,variable_names=["Qs",*sensors])
                for sensor in sensors:
                    yield f"{speed}/{sensor}",path,raw[sensor][:,1:7],raw["Qs"],np.ones(len(raw["Qs"]),bool),.01


def replay_locked_record(imu, quats, eligible, dt, config, weights, backbone, device):
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from final_eval import cache_external_candidate, embed_external, score_candidate

    valid = np.isfinite(imu).all(axis=1)
    last = np.maximum.accumulate(np.where(valid,np.arange(len(valid)),-1))
    signal = imu[last.clip(min=0)].astype(np.float32)
    signal[last<0] = 0
    label = np.isfinite(quats).all(axis=1) & np.any(quats != 0,axis=1)
    reference = np.zeros((len(quats),3),dtype=np.float64)
    reference[label] = Rotation.from_quat(quats[label],scalar_first=True).inv().apply(VERTICAL)
    mask = valid & label & eligible
    cached = cache_external_candidate(signal,dt,mask,valid,reference,config)
    np.testing.assert_array_equal(cached["index"],np.flatnonzero(mask))
    latent = embed_external(cached,config,backbone,device)
    score = float(score_candidate(cached,config,weights,latent))
    indices_hash = hashlib.sha256(np.asarray(cached["index"],dtype="<i8").tobytes()).hexdigest()
    return score, indices_hash


def external_frame_evidence(dataset, names):
    proofs = {}
    for name,_,imu,quats,eligible,_ in raw_records(dataset,names):
        mask = np.isfinite(imu).all(axis=1) & np.isfinite(quats).all(axis=1) & np.any(quats != 0,axis=1) & eligible
        indices = np.flatnonzero(mask).astype("<i8",copy=False)
        proofs[name] = {"frames":len(indices), "score_frame_indices_sha256":hashlib.sha256(indices.tobytes()).hexdigest()}
    assert set(proofs) == set(names), dataset
    return proofs


def replay_suites(datasets, config, weights):
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    backbone = device = None
    if config.get("embedding"):
        import torch
        from tilt_model import load_backbone
        device = torch.device("cuda")
        backbone = load_backbone(BATCH.parent / "checkpoint",device)
    suites = {}
    for dataset, expected_count in (("BROAD",39),("Sassari",18)):
        rows = {r["name"]:r for r in datasets[dataset]["sequences"]}
        assert len(rows)==len(datasets[dataset]["sequences"])==expected_count
        results = []
        for name,path,imu,quats,eligible,dt in raw_records(dataset,rows):
            row = rows[name]
            assert len(imu) == len(quats) == len(eligible)
            scored = replay(path,imu,np.ones(len(imu),bool),quats,VERTICAL,dt,eligible,
                            {"rmse_deg":row["rmse_deg"],"scoring_frames":row["frames"]})
            assert scored["score_index_sha256"] == row["score_frame_indices_sha256"], (dataset,name)
            result = {"name":name,"source":str(path),"source_sha256":scored["sha256"],"frames":scored["scoring_frames"],
                      "score_index_sha256":scored["score_index_sha256"],"raw_pure_deg":scored["raw_replay_full_vqf_deg"],
                      "pure_abs_difference_deg":abs(scored["raw_replay_full_vqf_deg"]-row["rmse_deg"]["full_vqf"])}
            locked, frame_hash = replay_locked_record(imu,quats,eligible,dt,config,weights,backbone,device)
            gap = abs(locked-row["rmse_deg"]["locked"])
            assert gap < 2e-4 and frame_hash == result["score_index_sha256"], (name,gap)
            result.update(raw_locked_deg=locked,locked_abs_difference_deg=gap)
            results.append(result)
        assert len(results) == len(rows) and {r["name"] for r in results} == set(rows)
        suites[dataset] = results
    return suites
