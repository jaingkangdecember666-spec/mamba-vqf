"""Audit every shifted SHICE score and independently rebuild the common masks from the old archive."""
import csv
import hashlib
import json
from datetime import datetime
from statistics import fmean

import numpy as np

from lock_from_frontier import ROOT, REVIEW, read, digest

OUT = ROOT / "final/shice-clock-sensitivity"
PREVIOUS = ROOT.parent / "transfer-mamba-20261123/batch8"
FIELDS = {"c038_deg":"candidate_deg", "vqf3_deg":"default_deg", "vqf1p5_deg":"fast_deg"}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def hash_ids(a):
    return hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()


def main():
    protocol = read(OUT / "protocol.json")
    summary = read(OUT / "summary.json")
    process = read(OUT / "process.json")
    assert process["status"] == summary["status"] == "complete" and process["summary_sha256"] == digest(OUT / "summary.json")
    assert summary["protocol_sha256"] == digest(OUT / "protocol.json") and not summary["failures"]
    assert list(protocol["shift_native_samples"]) == [-5, -1, 0, 1, 5]
    for path, checksum in protocol["source_sha256"].items():
        assert digest(ROOT / path) == checksum, path
    for name, checksum in summary["outputs_sha256"].items():
        assert digest(OUT / name) == checksum, name
    controls = {r["record"]:r for r in read(REVIEW / "vqf_response_controls_final.json")["records"] if r["dataset"]=="SHICE"}
    data = csv_rows(OUT / "sequences.csv")
    full = {r["record"]:r for r in csv_rows(OUT / "nominal-full.csv")}
    assert set(controls) == set(full) == set(protocol["authorized_records"]) and len(full)==20
    assert len(data) == len({(r["record"], r["offset_native_samples"]) for r in data}) == 100
    native = read(REVIEW / "shice-clock-anchors.json")
    assert native["protocol_sha256"] == digest(OUT / "protocol.json") and set(native["records"]) == {"R5-1", "R6-1"}
    assert datetime.fromisoformat(native["replayed_at"]) < datetime.fromisoformat(process["started_at"])
    old_masks = PREVIOUS / "scoring-masks.npz"
    assert digest(old_masks) == read(PREVIOUS / "results.json")["masks_sha256"]
    anchor_gaps = []
    common_counts = {}
    with np.load(old_masks, allow_pickle=False) as originals:
        for name, current in full.items():
            old = controls[name]
            nominal = np.asarray(originals[name], dtype=bool)
            nn = np.flatnonzero(nominal)
            assert hash_ids(nn) == current["scoring_indices_sha256"] == old["scoring_indices_sha256"]
            assert len(nn) == int(current["scored_frames"]) == old["scored_frames"]
            assert all(abs(float(current["replay_"+key])-old[value]) <= protocol["parity_tolerance_deg"] for key,value in FIELDS.items())
            # Sum invalid rows over each full eleven-element interval, using native indices.
            prefix = np.insert(np.cumsum(~nominal, dtype="i8"), 0, 0)
            common = np.flatnonzero(prefix[11:] == prefix[:-11])+5
            common_counts[name] = len(common)
            mask_hash = hash_ids(common)
            evidence = read(OUT / "receipts" / f"{name}.json")
            assert summary["receipt_sha256"][f"{name}.json"] == digest(OUT / "receipts" / f"{name}.json")
            assert evidence["status"]=="scored" and evidence["common_scoring_indices_sha256"] == mask_hash
            cases = {r["offset_native_samples"]:r for r in evidence["offset_scores"]}
            rows = {int(r["offset_native_samples"]):r for r in data if r["record"]==name}
            assert set(rows) == set(cases) == set(protocol["shift_native_samples"])
            for shift, row in rows.items():
                assert int(row["common_scored_frames"]) == len(common) and row["common_scoring_indices_sha256"] == mask_hash
                assert all(float(row[key]) == cases[shift][key] and np.isfinite(float(row[key])) for key in FIELDS)
                assert abs(float(row["offset_ms"]) - 1000*shift/float(row["rate_hz"])) < 1e-10
                if name in native["records"]:
                    independently = native["records"][name]
                    assert independently["common_mask"] == mask_hash and independently["common_frames"] == len(common)
                    anchor_gaps.extend(abs(float(row[k])-independently["offsets"][str(shift)][v]) for k,v in FIELDS.items())
    assert len(anchor_gaps)==30 and max(anchor_gaps)<2e-4
    stats = {}
    for group in ("R5", "R6", "R5+R6"):
        names = {n for n in full if group=="R5+R6" or n.split("-")[0]==group}
        worker = summary["groups"][group]
        m = sum(common_counts[n] for n in names)
        n = sum(int(full[a]["scored_frames"]) for a in names)
        assert m == worker["common_scored_frames"] and n == worker["nominal_full_scored_frames"]
        scores = {}
        for shift in protocol["shift_native_samples"]:
            subset = [r for r in data if int(r["offset_native_samples"])==shift and r["record"] in names]
            means = {k:fmean(float(r[k]) for r in subset) for k in FIELDS}
            checked = worker["common_by_offset_native_samples"][f"{shift:+d}"]
            assert means == checked["record_equal_rmse_deg"] and len(subset)==checked["records"]
            wins = {k:sum(float(r["c038_deg"])<float(r[k]) for r in subset) for k in ("vqf3_deg","vqf1p5_deg")}
            assert wins["vqf3_deg"] == checked["c038_wins_vs_vqf3"] and wins["vqf1p5_deg"] == checked["c038_wins_vs_vqf1p5"]
            scores[str(shift)] = {"means":means, "wins":wins, "gain_default_pct":100*(1-means["c038_deg"]/means["vqf3_deg"])}
        stats[group] = {"records":len(names), "common_frames":m, "formal_frames":n, "offsets":scores}
    out = {"verified_at":datetime.now().astimezone().isoformat(), "selection_sha256":digest(ROOT/"selection-lock.json"),
           "protocol_sha256":digest(OUT/"protocol.json"), "worker_summary_sha256":digest(OUT/"summary.json"),
           "archive_mask_sha256":digest(old_masks), "independent_anchor_sha256":digest(REVIEW/"shice-clock-anchors.json"),
           "raw_anchor_comparisons":len(anchor_gaps), "raw_anchor_max_gap_deg":max(anchor_gaps),
           "reconstructed_masks":len(common_counts), "records_per_offset":20, "groups":stats}
    (REVIEW/"shice-clock-verified.json").write_text(json.dumps(out, indent=2)+"\n", encoding="utf8")
    print("Masks", len(common_counts), "native angle checks", len(anchor_gaps), "max gap", max(anchor_gaps))


if __name__=="__main__":
    main()
