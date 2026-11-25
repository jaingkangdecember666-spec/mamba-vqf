"""Measure the frame mismatch between batch-1 training labels and the native-frame scoring objective."""

import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation
from vqf import VQF


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UP = np.array([0.,0.,1.])


def describe(angles):
    return {"rmse_deg":float(np.sqrt(np.mean(angles**2))),"median_deg":float(np.median(angles)),
            "p95_deg":float(np.quantile(angles,.95)),"max_deg":float(angles.max())}


def main():
    split = json.loads((ROOT/"external-split.json").read_text(encoding="utf-8"))
    records, pooled = [], {}
    for row in split["records"]:
        path = Path(row["path"])
        if row["domain"] == "BROAD":
            src=loadmat(path,variable_names=("imu_acc","imu_gyr","opt_quat","movement","sampling_rate"))
            imu=np.column_stack((src["imu_acc"],src["imu_gyr"]))
            truth,move,fs=src["opt_quat"],src["movement"].ravel().astype(bool),float(src["sampling_rate"].item())
        else:
            device=row["name"].split("/")[1]
            src=loadmat(path,variable_names=("Qs",device))
            imu,truth,fs=src[device][:,1:7],src["Qs"],100.
            move=np.ones(len(imu),dtype=bool)
        valid=np.isfinite(imu).all(axis=1)
        at=np.maximum.accumulate(np.where(valid,np.arange(len(imu)),-1))
        signal=imu[at.clip(min=0)].copy()
        signal[at<0]=0
        quat=VQF(1/fs).updateBatch(np.ascontiguousarray(signal[:,3:],dtype=np.float64),
                                    np.ascontiguousarray(signal[:,:3],dtype=np.float64))["quat6D"]
        blocks=np.arange(70,int(np.floor(len(imu)*100/fs+1e-9))+1,10)
        end=np.floor(blocks*fs/100+1e-9).astype(np.int64)
        start=np.floor((blocks-64)*fs/100+1e-9).astype(np.int64)
        release=np.ceil(blocks*fs/100-1e-9).astype(np.int64)-1
        last=end-1
        bad=np.r_[0,np.cumsum(~valid)]
        use=(bad[end]==bad[start])&valid[release]&move[release]&np.isfinite(truth[release]).all(axis=1)&np.any(truth[release]!=0,axis=1)
        release,last=release[use],last[use]
        origin=Rotation.from_quat(quat[last],scalar_first=True)
        delivery=Rotation.from_quat(quat[release],scalar_first=True)
        old=Rotation.from_quat(truth[release],scalar_first=True).inv().apply(UP)
        score_world=delivery.apply(old)
        aligned=origin.inv().apply(score_world)
        np.testing.assert_allclose(origin.apply(aligned),score_world,rtol=0,atol=2e-12)
        diff=np.degrees(np.arctan2(np.linalg.norm(np.cross(old,aligned),axis=1),np.einsum("ij,ij->i",old,aligned)))
        mismatch=release!=last
        records.append({"domain":row["domain"],"name":row["name"],"split":row["split"],"eligible":len(last),
                        "different_sample":int(mismatch.sum()),"max_gap_ms":float((release-last).max()*1e3/fs),**describe(diff)})
        bucket=row["domain"]+"/"+row["split"]
        pooled.setdefault(bucket,[]).append(diff)
    result={"records":records,"pooled":{k:describe(np.concatenate(a)) for k,a in pooled.items()}}
    for key,stats in result["pooled"].items():
        selected=[r for r in records if r["domain"]+"/"+r["split"]==key]
        stats.update(eligible=sum(r["eligible"] for r in selected),different_sample=sum(r["different_sample"] for r in selected))
    (HERE/"training_target_audit.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result["pooled"],ensure_ascii=False))


if __name__=="__main__":
    main()
