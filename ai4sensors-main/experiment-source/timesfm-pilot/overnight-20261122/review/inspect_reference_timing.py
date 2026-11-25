"""Measure ± one native sample of encoder-label timing on existing development exports."""

import hashlib
from datetime import datetime, timezone
from itertools import combinations
from math import isclose
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, read, write
from verify_baselines import digest
from verify_final import raw_shice_samples


RECORDS = ("R2-10","R4-2")
OFFSETS = {"earlier":-1,"original":0,"later":1}
TRIALS = {f"V{i:05}" for i in range(1,10)}


def centers(mask):
    return np.flatnonzero(mask[:-2] & mask[1:-1] & mask[2:]) + 1


def tilt(up, truth):
    radians = np.arctan2(np.linalg.norm(np.cross(up,truth),axis=1),np.einsum("ij,ij->i",up,truth))
    return float(np.sqrt(np.mean(np.rad2deg(radians)**2)))


def shifted(up, truth, rows):
    predicted = up[rows]
    return {k:tilt(predicted,truth[rows+offset]) for k,offset in OFFSETS.items()}


def describe(rows):
    result = {}
    for timing in ("earlier","later"):
        differences = [r["rmse_deg"][timing]-r["rmse_deg"]["original"] for r in rows]
        reversals, changes_from_tie, fixed_pairs = 0, 0, 0
        for a,b in combinations(rows,2):
            before = a["rmse_deg"]["original"]-b["rmse_deg"]["original"]
            after = a["rmse_deg"][timing]-b["rmse_deg"][timing]
            if abs(before) <= 1e-8:
                changes_from_tie += int(abs(after) > 1e-8)
            else:
                fixed_pairs += 1
                reversals += int(abs(after) > 1e-8 and before*after < 0)
        result[timing] = {"min_change_deg":min(differences),"max_change_deg":max(differences),
                          "originally_ordered_pairs":fixed_pairs,"reversed_pairs":reversals,"ties_resolved":changes_from_tie}
    return result


def check_saved(evidence, exports):
    included = {(r["round_id"],r["name"]):r for r in exports if r["round_id"] in TRIALS}
    assert set(included) == {(t,n) for t in TRIALS for n in RECORDS}
    assert evidence["reference_sample_offsets"] == OFFSETS and set(evidence["records"]) == set(RECORDS)
    for name, item in evidence["records"].items():
        entries = item["outputs"]
        assert len(entries) == len(TRIALS) and {r["trial"] for r in entries} == TRIALS
        assert 0 < item["common_frames"] <= item["original_frames"] <= item["native_frames"]
        assert digest(Path(item["source"])) == item["source_sha256"]
        for row in entries:
            original = included[row["trial"],name]
            assert original["candidate"] == row["candidate"] and original["quaternion_file"] == row["file"]
            assert original["source"] == item["source"] and digest(BATCH/row["file"]) == row["sha256"]
            assert item["original_frames"] == original["scored_frames"] and item["native_frames"] == original["native_frames"]
            assert isclose(row["full_mask_zero_deg"],original["raw_quaternion_rmse_deg"],abs_tol=2e-7)
            assert set(row["rmse_deg"]) == set(OFFSETS) and all(0 <= v <= 180 for v in row["rmse_deg"].values())
        assert item["summary"] == describe(entries)


def audit():
    exports = read(HERE/"development_quaternion_audit.json")["outputs"]
    evidence = {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"reference_sample_offsets":OFFSETS,"records":{}}
    for name in RECORDS:
        raw = raw_shice_samples(name)
        time, score = raw["time_s"], raw["score"]
        common = centers(score)
        intervals = np.concatenate((time[common]-time[common-1],time[common+1]-time[common]))
        assert np.min(intervals) > 0
        entries = []
        for original in exports:
            if original["name"] != name or original["round_id"] not in TRIALS:
                continue
            path = BATCH/original["quaternion_file"]
            with np.load(path,allow_pickle=False) as data:
                np.testing.assert_array_equal(data["time_s"],time)
                np.testing.assert_array_equal(data["sample_index"],np.arange(len(time)))
                up = Rotation.from_quat(data["q_wxyz"],scalar_first=True).inv().apply([0,0,1])
            entries.append({"trial":original["round_id"],"candidate":original["candidate"],"file":original["quaternion_file"],
                            "sha256":digest(path),"full_mask_zero_deg":tilt(up[score],raw["reference"][score]),
                            "rmse_deg":shifted(up,raw["reference"],common)})
        evidence["records"][name] = {"source":str(raw["source"]),"source_sha256":digest(raw["source"]),
            "native_frames":len(time),"original_frames":int(score.sum()),"common_frames":len(common),
            "center_index_sha256":hashlib.sha256(np.asarray(common,dtype="<i8").tobytes()).hexdigest(),
            "nearby_step_ms":{"minimum":float(intervals.min()*1e3),"maximum":float(intervals.max()*1e3)},
            "outputs":entries,"summary":describe(entries)}
    check_saved(evidence,exports)
    return evidence


if __name__ == "__main__":
    result = audit()
    write(HERE/"development_reference_timing.json",result)
    for name,item in result["records"].items():
        print(name,item["common_frames"],item["summary"])
