"""Rebuild native-frame predictions from saved weights and time-stamped raw IMU blocks."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from audit_external import UP, load_external

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/".deps"),str(ROOT)]
from models import ResidualHead


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def weights(path, kind, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = ResidualHead(kind)
    head.load_state_dict(checkpoint["state_dict"])
    return head.to(device).eval(), checkpoint["mean"].to(device), checkpoint["std"].to(device), float(checkpoint["cap_deg"])


def features(signal, hz):
    filt = VQF(1/hz).updateBatch(np.ascontiguousarray(signal[:,3:],dtype=np.float64),
                                 np.ascontiguousarray(signal[:,:3],dtype=np.float64))
    nblock = int(np.floor(len(signal)*100/hz+1e-9))
    edges = np.arange(nblock+1)*(hz/100)
    stops = np.floor(edges+1e-9).astype(np.int64)
    n_interval = np.diff(stops)
    assert np.all(n_interval>=1)
    sampled = np.add.reduceat(signal[:stops[-1]].astype(np.float64),stops[:-1],axis=0)/n_interval[:,None]
    native = Rotation.from_quat(filt["quat6D"],scalar_first=True)
    at = native[stops[1:]-1]
    block = np.column_stack((sampled.astype(np.float32),at.inv().apply(UP),filt["bias"][stops[1:]-1],
                             filt["restDetected"][stops[1:]-1])).astype(np.float32)
    indices = np.arange(69,nblock,10)
    windows = block[indices[:,None]+np.arange(-63,1)[None,:]]
    last = stops[indices+1]-1
    release = np.ceil(edges[indices+1]-1e-9).astype(np.int64)-1
    return native, windows, last, release


def metric(up_world, target_world):
    angles = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(up_world,target_world),axis=1),
                                     np.sum(up_world*target_world,axis=1)))
    return float(np.sqrt(np.mean(angles**2)))


def canonical_frame(windows):
    """A right-handed basis from gravity and a transverse signal in the observed window."""
    z = windows[:, -1, 6:9].astype(np.float64)
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    row = np.arange(len(z))

    def strongest(vectors):
        transverse = vectors - np.einsum("ntj,nj->nt", vectors, z)[:, :, None] * z[:, None, :]
        length = np.linalg.norm(transverse, axis=2)
        i = length.argmax(axis=1)
        return transverse[row, i], length[row, i]

    a, an = strongest(windows[:, :, :3].astype(float))
    w, wn = strongest(windows[:, :, 3:6].astype(float) - windows[:, :, 9:12].astype(float))
    from_acc = an >= .01
    valid = from_acc | (wn >= .001)
    x = np.where(from_acc[:, None], a, w) / np.maximum(np.where(from_acc, an, wn)[:, None], 1e-12)
    axes = np.stack((x, np.cross(z, x), z), axis=2)
    axes[~valid] = np.eye(3)
    result = windows.copy()
    vec = windows[:, :, :12].reshape(len(z), -1, 3)
    result[:, :, :12] = (vec @ axes).reshape(len(z), windows.shape[1], 12)
    return result, axes, valid


@torch.inference_mode()
def score(signal, hz, reference_body, valid_rows, models, policies, omega0, canonical_models=()):
    native, windows, last, release = features(signal,hz)
    native_ids = np.flatnonzero(valid_rows)
    truth = native[native_ids].apply(reference_body[native_ids])
    q_at_update = native[last]
    up = q_at_update.inv().apply(UP)
    active = np.searchsorted(release,native_ids,side="right")
    answer = {"full_vqf":metric(np.broadcast_to(UP,truth.shape),truth)}
    angular = np.linalg.norm(windows[:,:,3:6].astype(np.float64),axis=2).mean(axis=1)
    gates = angular/(angular+omega0)
    for name,(head,mean,std,cap) in models.items():
        body_axes = None
        model_windows = windows
        if name in canonical_models:
            model_windows, body_axes, active_basis = canonical_frame(windows)
        pieces = []
        for start in range(0,len(windows),128):
            x=torch.from_numpy(model_windows[start:start+128]).to(mean.device)
            pieces.append(head(((x-mean)/std).clamp(-12,12)).cpu().numpy())
        raw = np.concatenate(pieces).astype(np.float64)*.1
        if body_axes is not None:
            raw = np.einsum("nij,nj->ni", body_axes, raw)
            raw[~active_basis] = 0.
        delta = raw - np.sum(raw*up,axis=1)[:,None]*up
        delta *= np.minimum(1.,np.tan(np.deg2rad(cap))/np.maximum(1e-8,np.linalg.norm(delta,axis=1)))[:,None]
        delta_world = q_at_update.apply(delta)
        for label,gain in policies[name].items():
            multiplier = gates[:,None] if gain == "angular" else float(gain)
            corrected = UP[None,:]+multiplier*delta_world
            corrected /= np.linalg.norm(corrected,axis=1)[:,None]
            native_world = np.vstack((UP,corrected))[active]
            answer[label]=metric(native_world,truth)
    return {"frames":len(native_ids),"indices_sha256":hashlib.sha256(native_ids.astype("<i8").tobytes()).hexdigest(),
            "updates":len(last),"rmse_deg":answer}


def external_case(row):
    imu,quat,eligible,hz = load_external(row)
    observed = np.isfinite(imu).all(axis=1)
    prev = np.maximum.accumulate(np.where(observed,np.arange(len(imu)),-1))
    signal = imu[prev.clip(min=0)].copy()
    signal[prev<0]=0
    labelled = np.isfinite(quat).all(axis=1) & np.any(quat!=0,axis=1)
    ref = np.zeros((len(imu),3))
    ref[labelled] = Rotation.from_quat(quat[labelled],scalar_first=True).inv().apply(UP)
    return signal,hz,ref,eligible & observed & labelled


def audit_external_models():
    torch.set_num_threads(2)
    batch1, batch2 = read(ROOT/"results.json"),read(ROOT/"batch2/results.json")
    models={"mamba_v1":weights(ROOT/"mamba_v1.pt","mamba_v1"),
            "tcn":weights(ROOT/"tcn.pt","tcn","cuda"),"tcn_cpu":weights(ROOT/"tcn.pt","tcn")}
    policies={"mamba_v1":{"mamba_full":1.,"mamba_constant":batch2["constant_gain"],"mamba_angular_gate":"angular"},
              "tcn":{"tcn":1.},"tcn_cpu":{"tcn_cpu":1.}}
    before={r["sequence"]:r for r in batch1["sequences"]}
    after={r["sequence"]:r for r in batch2["sequences"]}
    checks = []
    for row in read(ROOT/"external-split.json")["records"]:
        if row["split"]!="validation":
            continue
        name=row["name"]
        observed=score(*external_case(row),models,policies,batch2["omega0_rad_s"])
        expected={**after[name]["rmse_deg"],"tcn":before[name]["rmse_deg"]["tcn"]}
        assert observed["indices_sha256"]==after[name]["scoring_indices_sha256"]
        errors={k:abs(observed["rmse_deg"][k]-expected[k]) for k in expected}
        assert max(errors.values()) < 1e-4, (name,errors)
        checks.append({"domain":row["domain"],"name":name,**observed,"max_abs_delta":max(errors.values()),
                       "tcn_cpu_vs_cuda_deg":abs(observed["rmse_deg"]["tcn"]-observed["rmse_deg"]["tcn_cpu"])})
    target=ROOT/"review"/"external_models_replay.json"
    target.write_text(json.dumps(checks,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(len(checks),sum(r["frames"] for r in checks),max(r["max_abs_delta"] for r in checks))


if __name__=="__main__":
    audit_external_models()
