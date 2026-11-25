"""Describe frozen performance on all fifteen designated source-side records for the four supplementary domains."""
import csv
import json
from statistics import fmean

import numpy as np
import torch

from davis_handheld import Davis, BASE as DAVIS
from lock_from_frontier import ROOT, REVIEW, digest, read
from locked_replay import SelectedPredictor
from native_catalog import body_reference
from penn_handheld import Penn, BASE as PENN


def archived_arrays(domain):
    directory = ROOT / "external-sources" / {"EuRoC": "euroc-original-inputs", "RIDI": "ridi-phone-train"}[domain]
    manifest = read(directory / "source-audit.json")
    for row in manifest["records" if domain == "EuRoC" else "rows"]:
        name = row["record"]
        path = directory / (name + ".npz")
        recorded = row["prepared_sha256" if domain == "EuRoC" else "file_sha256"]
        assert digest(path) == recorded
        with np.load(path) as f:
            x = np.column_stack((f["acc_m_s2"], f["gyr_rad_s"]))
            if domain == "RIDI":
                clock = f["time_ns"]
                x[~f["imu_observed"]] = np.nan
                truth, valid = body_reference(f["tango_wxyz"], [0., 0., 1.])
                valid &= f["reference_valid"]
            else:
                clock = f["utc_ns"]
                assert np.max(abs(np.diff(clock) - 5000000)) < 1000
                truth, valid = body_reference(f["original_wxyz"], [0., 0., 1.])
            mask = valid & np.isfinite(x).all(1) & ((clock - clock[0]) >= 10000000000)
            yield name, (x, 200., truth, mask), recorded


def native_records():
    for domain in ("EuRoC", "RIDI"):
        for name, native, checksum in archived_arrays(domain):
            yield domain, name, native, checksum
    for domain, loader, files, role in (("DAVIS", Davis(), DAVIS, "external_train"),
                                        ("PennCOSYVIO", Penn(), PENN, "external_training")):
        locked = {r["record"]: r for r in read(files / "preprocessing-lock.json")["source_records"]}
        assert set(locked) == {n for n, use in loader.roles.items() if use == role}
        for name, row in locked.items():
            native, _ = loader.native(name)
            path = ROOT / row["prepared"]
            assert digest(path) == row["sha256"]
            with np.load(path) as data:
                for value, key in zip(native, ("six_axis","native_hz","optical_up","mask"), strict=True):
                    np.testing.assert_array_equal(value, data[key])
            yield domain, name, native, row["sha256"]


def main():
    torch.set_num_threads(2)
    lock = ROOT / "selection-lock.json"
    model = SelectedPredictor(lock)
    selected = read(REVIEW / "selected-training-sources.json")["unique_by_source"]
    source = []
    for domain, name, native, checksum in native_records():
        score = model.score(native)
        source.append({"dataset": domain, "record": name, "source_sha256": checksum, **score})
    target = []
    paths = [REVIEW / sub / "sequences.csv" for sub in ("final-supplemental", "final-davis", "final-penn")]
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as f:
            target.extend(r for r in csv.DictReader(f) if r.get("candidate_deg"))
    assert len(source) == 15 and len(target) == 31
    keys = lambda rows: {(r["dataset"], r["record"]) for r in rows}
    assert len(keys(source)) == 15 and len(keys(target)) == 31 and not keys(source) & keys(target)
    summary = {}
    for domain, count in {"EuRoC":3, "RIDI":4, "DAVIS":6, "PennCOSYVIO":2}.items():
        record = {"source_used_by_winner": selected.get(domain, 0)}
        for side, rows in (("source_side", source),("formal_transfer", target)):
            group = [r for r in rows if r["dataset"] == domain]
            if side == "source_side": assert len(group) == count
            m = {k: fmean(float(r[k]) for r in group) for k in ("full_vqf_deg","physical_deg","candidate_deg")}
            record[side] = {"records": len(group), "frames": sum(int(r["scored_frames"]) for r in group), **m,
                            "gain_percent": 100 * (1.-m["candidate_deg"]/m["full_vqf_deg"]),
                            "records_improved": sum(float(r["candidate_deg"]) < float(r["full_vqf_deg"]) for r in group)}
        summary[domain] = record
    proof = {"selection_lock_sha256": digest(lock), "source_side": source, "domains": summary,
             "selected_training_records_filename": "review/selected-training-sources.json",
             "test_tables_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
             "scope": "Post-selection diagnostic of fifteen complete source-side captures. Four RIDI and both Penn A tracks contributed to the winner. The three EuRoC V1 and six DAVIS boxes/shapes recordings were available in development of other candidates. Room 1 uses reconstructed acquisition gyro and the originally published attitude, as in the Room 2 reference convention. No target-side fit, outcome-based source sampling, or changes to formal scores."}
    (REVIEW / "source-side-transfer.json").write_text(json.dumps(proof,indent=2)+"\n",encoding="utf8")
    print({d:(round(a["source_side"]["gain_percent"],3),round(a["formal_transfer"]["gain_percent"],3)) for d,a in summary.items()})


if __name__ == "__main__":
    main()
