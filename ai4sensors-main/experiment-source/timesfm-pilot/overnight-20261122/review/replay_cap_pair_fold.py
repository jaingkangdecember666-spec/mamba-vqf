"""Check the effect of clipping on the R4/M4 fold with a QR fit and the original IMU packets."""

import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, read, write
from replay_mounted_alpha import training_inputs, standardize_records, fit_by_qr
from verify_baselines import digest
from verify_cap_pair import PLAN, PLAN_SHA
from verify_final import raw_shice_samples, score_mask_evidence
from verify_mount_pair import REFERENCE


def run():
    sys.path.insert(0,str(BATCH))
    import learning
    from features import cache_record

    assert digest(PLAN)==PLAN_SHA
    plan, reference = read(PLAN),read(REFERENCE)
    result_path = BATCH/"rounds/O00001/result.json"
    original = read(result_path)
    fold = reference["folds"]["joint_holdout:R4/M4"]
    seed = 20261124
    config = {**plan["shared_config"],"seed":seed,"cap_deg":4.}
    assert config["hidden"]==128 and config["embedding"]==0 and config["gyro_gate"]==0 and config["acc_gate"]>0
    mount = np.asarray(fold["wxyz"],dtype=np.float32)
    rows,_ = training_inputs(fold["training"],config,mount)
    state = standardize_records(rows)
    width = len(state["mean"])
    rng = np.random.default_rng(seed)
    state.update(random=rng.standard_normal((width,config["hidden"]),dtype=np.float32)/np.sqrt(width),
                 shift=rng.uniform(-.8,.8,config["hidden"]).astype(np.float32))
    coef, = fit_by_qr(rows,state,[config["alpha"]])
    published = {r["name"]:r for r in original["sequences_by_seed"][str(seed)]["joint_holdout"] if r["fold"]=="R4/M4"}
    assert sorted(published)==fold["validation"]==["R4-4","R4-8"]
    evidence = {}
    for name,expected in published.items():
        raw = raw_shice_samples(name)
        masks = score_mask_evidence(raw["score"])
        assert masks["score_frame_indices_sha256"]==expected["score_frame_indices_sha256"] and masks["frames"]==expected["frames"]
        native = cache_record(raw["imu"],raw["dt"],raw["score"],raw["context"],raw["reference"],config["parameters"],True,keep_scored_quaternion=True)
        part = learning.prepare(learning.apply_install(native,mount),config)
        prediction = (learning.design(part["matrix"],state)@coef).astype(np.float64)
        u = part["up"].astype(np.float64)
        tangent = prediction - np.einsum("ij,ij->i",prediction,u)[:,None]*u
        lengths = np.linalg.norm(tangent,axis=1)
        rotations = Rotation.from_quat(part["q"],scalar_first=True)
        gate = config["gain"]*np.exp(-.5*(part["acc"].astype(np.float64)/config["acc_gate"])**2)
        release = np.searchsorted(part["ready"],part["index"],side="right")
        used = np.bincount(release,minlength=len(prediction)+1)[1:]
        scores = {}
        for cap in (4.,6.):
            limit = np.tan(np.deg2rad(cap))
            over = lengths>limit
            scale = np.minimum(1.,limit/np.maximum(lengths,1e-12))*gate
            direction = rotations.apply(tangent*scale[:,None])+[0,0,1]
            scored = np.vstack(([0.,0.,1.],direction))[release]
            ref = part["ref_world"]
            angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(scored,ref),axis=1),np.einsum("ij,ij->i",scored,ref)))
            rmse = float(np.sqrt(np.mean(angles**2)))
            counts = {"all_candidate_updates":len(lengths),"updates_over_cap":int(over.sum()),
                      "updates_over_cap_used_on_score_frames":int(np.count_nonzero(over & (used>0))),
                      "original_score_frames_using_clipped_update":int(used[over].sum())}
            assert counts==expected["clip"][str(cap)], (name,cap,counts,expected["clip"][str(cap)])
            gap = abs(rmse-expected["rmse_deg"][str(cap)])
            assert gap < 2e-5,(name,cap,gap)
            scores[str(cap)] = {"rmse_deg":rmse,"abs_rmse_difference_deg":gap,**counts}
        evidence[name] = {"source_sha256":digest(raw["source"]),**masks,"caps":scores}
    proof = {"at":datetime.now(timezone.utc).isoformat(),"fold":"R4/M4","seed":seed,"training":fold["training"],
             "training_updates":sum(len(x) for x,_ in rows),"records":evidence,"source_result_sha256":digest(result_path),"plan_sha256":PLAN_SHA}
    write(HERE/"ridge_cap_pair_raw_replay.json",proof)
    print({n:r["caps"] for n,r in evidence.items()},flush=True)


if __name__=="__main__":
    run()
