"""Independent body-frame replay of the first R5 and R6 clock-sensitivity anchors."""
import hashlib
import json
from datetime import datetime

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.transform import Rotation
from vqf import VQF

from alternate_native import OriginalForward, abl, mdl, torch
from lock_from_frontier import ROOT, REVIEW, read, digest
from native_catalog import Catalog


OFFSETS = (-5, -1, 0, 1, 5)


def hash_indices(idx):
    return hashlib.sha256(np.asarray(idx, dtype="<i8").tobytes()).hexdigest()


def angles(estimates, ref):
    result = {}
    for name, v in estimates.items():
        radians = np.arctan2(np.linalg.norm(np.cross(v, ref), axis=1), (v * ref).sum(axis=1))
        result[name] = float(np.sqrt(np.mean(np.degrees(radians) ** 2)))
    return result


@torch.inference_mode()
def score(forward, payload):
    signal, hz, truth, valid = payload
    seen = np.isfinite(signal).all(axis=1)
    mask = valid & seen
    nominal = np.flatnonzero(mask)
    # Each nonwrapping length-eleven window is all true; its sixth index is the center.
    joint = 5 + np.flatnonzero(sliding_window_view(mask, 11).all(axis=1))
    q0, q, at, release, history, _, acc, _ = mdl.c17.front_end(signal, seen, 1./hz, forward.cfg, None)
    directions, *_ = abl.held(q, at, release, history, acc, signal, seen, 1./hz, forward.k, forward.fixed, forward.routers, 0.)
    world = np.vstack((mdl.UP, directions["C038_deg"]))
    previous = np.maximum.accumulate(np.where(seen, np.arange(len(seen)), -1))
    filled = signal[previous.clip(min=0)].copy()
    filled[previous < 0] = 0.
    fast_quat = VQF(1./hz, tauAcc=1.5).updateBatch(np.ascontiguousarray(filled[:, 3:]), np.ascontiguousarray(filled[:, :3]))["quat6D"]

    def body(quaternions, rows, direction):
        return Rotation.from_quat(quaternions[rows], scalar_first=True).inv().apply(direction)

    def predictions(rows):
        return {"default_deg":body(q0, rows, mdl.UP), "fast_deg":body(fast_quat, rows, mdl.UP),
                "candidate_deg":body(q, rows, world[np.searchsorted(release, rows, side="right")])}

    estimates = predictions(joint)
    return {"rate_hz":hz, "formal_frames":len(nominal), "formal_mask":hash_indices(nominal),
            "common_frames":len(joint), "common_mask":hash_indices(joint),
            "nominal":angles(predictions(nominal), truth[nominal]),
            "offsets":{str(k):angles(estimates, truth[joint+k]) for k in OFFSETS}}


def main():
    lock = ROOT / "selection-lock.json"
    plan = ROOT / "final/shice-clock-sensitivity/protocol.json"
    policy = read(plan)
    assert tuple(policy["shift_native_samples"]) == OFFSETS and policy["source_sha256"]["selection-lock.json"] == digest(lock)
    torch.set_num_threads(2)
    forward, catalog = OriginalForward(lock), Catalog(lock)
    controls = {r["record"]:r for r in read(REVIEW / "vqf_response_controls_final.json")["records"] if r["dataset"]=="SHICE"}
    rows = {}
    for record in ("R5-1", "R6-1"):
        row = score(forward, catalog.native("SHICE", record)[0])
        old = controls[record]
        assert row["formal_frames"]==old["scored_frames"] and row["formal_mask"]==old["scoring_indices_sha256"]
        assert max(abs(v-old[k]) for k,v in row["nominal"].items()) < 2e-4
        rows[record] = row
    outcome = {"replayed_at":datetime.now().astimezone().isoformat(), "protocol_sha256":digest(plan),
               "selection_sha256":digest(lock), "records":rows}
    (REVIEW / "shice-clock-anchors.json").write_text(json.dumps(outcome, indent=2)+"\n", encoding="utf8")
    print([(name, data["common_frames"]) for name, data in rows.items()])


if __name__ == "__main__":
    main()
