"""Reconstruct H00010 from the native SHICE IMU packets and the two VQF branches."""

from datetime import datetime, timezone
from statistics import fmean

import numpy as np
from scipy.spatial.transform import Rotation
from vqf import VQF

from audit_rounds import BATCH, HERE, read, write
from verify_final import digest, raw_shice_samples, score_mask_evidence


def angle(a,b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.einsum("ij,ij->i",a,b)))


def replay(name, config):
    data = raw_shice_samples(name)
    signal = data["imu"]
    acc = np.ascontiguousarray(signal[:,:3],dtype=np.float64)
    gyro = np.ascontiguousarray(signal[:,3:],dtype=np.float64)
    score = np.flatnonzero(data["score"])
    ends = np.arange(9,len(signal),10)
    endpoints,scored = [],[]
    for params in ({},config["parameters"]):
        q = VQF(data["dt"],**params).updateBatch(gyro,acc)["quat6D"]
        endpoints.append(Rotation.from_quat(q[ends],scalar_first=True).inv().apply([0,0,1]))
        scored.append(Rotation.from_quat(q[score],scalar_first=True).inv().apply([0,0,1]))
    low,high = config["thresholds"]["disagreement_deg"]
    assert set(config["thresholds"]) == {"disagreement_deg"} and config["max_fast_weight"]==1 and config["mode"]=="gated"
    discrepancy = angle(*endpoints)
    blocks = (score+1)//10
    latest = np.maximum(blocks-1,0)
    fallback = (blocks < config["history_blocks"]) | (discrepancy[latest] >= high)
    t = np.clip((discrepancy[latest]-low)/(high-low),0.,1.)
    fast_weight = (1.-t)**2 * (1.+2.*t)
    fast_weight[fallback] = 0.
    predicted = scored[0]+fast_weight[:,None]*(scored[1]-scored[0])
    errors = angle(predicted,data["reference"][score])
    return {"name":name,"source_sha256":digest(data["source"]),**score_mask_evidence(data["score"]),
            "fallback_frames":int(fallback.sum()),"rmse_deg":float(np.sqrt(np.mean(errors**2)))}


def main():
    path = BATCH/"rounds/H00010/result.json"
    saved = read(path)
    config = saved["config"]
    expected = {r["name"]:r for r in saved["sequences"]}
    dev = read(BATCH/"rounds/H00010/round.json")["partitions"]["development"]
    assert len(expected)==len(dev)==37 and set(dev)==set(expected)
    assert {name.split("-")[0] for name in dev}=={"R1","R2","R3","R4"}
    rows = []
    for name in sorted(dev):
        row = replay(name,config)
        old = expected[name]
        row["abs_rmse_difference_deg"] = abs(row["rmse_deg"]-old["rmse_deg"])
        assert row["frames"]==old["frames"] and row["fallback_frames"]==old["default_fallback_frames"], (row,old)
        assert row["abs_rmse_difference_deg"] < 1e-5, (name,row["abs_rmse_difference_deg"])
        rows.append(row)
    total = sum(r["frames"] for r in rows)
    hits = sum(r["fallback_frames"] for r in rows)
    proof = {"at":datetime.now(timezone.utc).isoformat(),"source_result_sha256":digest(path),"records":rows,
             "frames":total,"fallback_frames":hits,"pooled_fraction":hits/total,
             "macro_fraction":fmean(r["fallback_frames"]/r["frames"] for r in rows),
             "max_abs_rmse_difference_deg":max(r["abs_rmse_difference_deg"] for r in rows)}
    write(HERE/"gate_fallback_raw_replay.json",proof)
    print({k:v for k,v in proof.items() if k!="records"},flush=True)


if __name__=="__main__":
    main()
