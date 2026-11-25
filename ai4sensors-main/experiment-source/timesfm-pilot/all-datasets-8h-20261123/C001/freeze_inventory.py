"""Freeze grouping from source IDs, file metadata and documentation before scoring."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE.parent
PILOT = CAMPAIGN.parent
TRANSFER = PILOT / "transfer-mamba-20261123"
DATA = Path("E:/数据集")
sys.path[:0] = [str(HERE / ".deps"), str(TRANSFER)]

import h5py
from freeze_split import digest, stamp, write


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def add(rows, domain, name, group, side, use, **extra):
    rows.append({"dataset": domain, "record": name, "group": group, "side": side, "c001": use, **extra})


def indexed_yt(rows):
    path = DATA / "YT_server/YT_server_200Hz.h5"
    groups = defaultdict(list)
    totals = {"train": 1479, "valid": 422, "test": 213}
    with h5py.File(path, "r") as file:
        for split, total in totals.items():
            assert len(file[split]) == total
            for name in sorted(file[split]):
                assert re.fullmatch(r"[^_]+_[^_]+_server_\d+_\d+", name)
                groups[name.rsplit("_", 1)[0]].append((split, name, int(file[split][name]["ts"].shape[0])))
    test = {g for g, rs in groups.items() if any(s == "test" for s, _, _ in rs)}
    clean_valid = {g for g, rs in groups.items() if g not in test and any(s == "valid" for s, _, _ in rs)}
    chosen_groups = sorted(clean_valid, key=lambda g: (key("C001|YT_Server|" + g), g))[:12]
    selected = [min((n for s, n, _ in groups[g] if s == "valid"),
                    key=lambda n: (key("C001|YT_Server|valid|" + n), n)) for g in chosen_groups]
    selected_ids = set(selected)
    audit_path = CAMPAIGN / "review/yt_group_audit.json"
    audit = read(audit_path)
    clean = {s: sum(1 for g in groups.keys() - test for ss, _, _ in groups[g] if ss == s) for s in ("train", "valid")}
    assert len(groups) == audit["trials_if_prefix_is_trial"] == 474 and len(test) == audit["families_touched_by_official_test"] == 169
    assert clean == audit["records_in_families_without_official_test"] == {"train": 951, "valid": 273}
    assert len(selected_ids) == len(set(chosen_groups)) == 12 and len(clean_valid) >= 12
    for name, members in sorted(groups.items()):
        for split, record, frame_count in members:
            if name in test:
                side = "sealed_test" if split == "test" else "excluded_test_sibling"
            elif name in chosen_groups:
                side = "development" if record in selected_ids else "excluded_development_sibling"
            else:
                side = "external_training_candidate" if split == "train" else "unused_training_side_sibling"
            add(rows, "YT_Server", record, name, side, "new_score" if record in selected_ids else "none",
                official_split=split, native_frames=frame_count, source=f"/{split}/{record}")
    return {"path": str(path), "file_size": path.stat().st_size, "official_counts": totals, "groups": len(groups),
            "test_groups": len(test), "clean_record_counts_before_dev_exclusion": clean,
            "clean_valid_groups": len(clean_valid), "selected_groups": chosen_groups, "selected_valid": selected,
            "group_hash_rule": "ascending SHA256 UTF-8 C001|YT_Server|<prefix>, then lexical tie; first 12 safe groups",
            "record_hash_rule": "lowest SHA256 UTF-8 C001|YT_Server|valid|<record> within each selected group",
            "metadata_audit_sha256": digest(audit_path), "cross_split_prefix_pairs": audit["cross_split_same_prefix_pairs"],
            "minimum_adjacent_gap_s": audit["closest_nonoverlapping"][0][0],
            "group_identity_limit": "The filename prefix is a conservative recording-family proxy. True session and cross-prefix acquisition IDs are not documented."}


def other_domains(rows):
    old_split_path = TRANSFER / "external-split.json"
    old_split = read(old_split_path)
    for row in old_split["records"]:
        source = "historical_source_train" if row["split"] == "train" else "development_previously_used_for_selection"
        add(rows, row["domain"], row["name"], row["group"], source,
            "reused_score" if row["split"] == "validation" else "none", path=row["path"])
    for device in range(1, 7):
        group = f"R{device}"
        for path in sorted((DATA / f"IMU_dynamic_data_SHICE/{group}_aligned_formal").glob(f"{group}-*.csv")):
            side = "sealed_current_campaign_historically_exposed" if device >= 5 else "development"
            if path.stem in ("R1-3", "R1-4"):
                side = "excluded_label_audit"
            add(rows, "SHICE", path.stem, group, side, "reused_score" if side == "development" else "none", path=str(path))
    for room in range(1, 7):
        paths = list((DATA / f"TUM-VI/dataset-room{room}_512_16").rglob("mav0/imu0/data.csv"))
        assert len(paths) == 1
        add(rows, "TUM-VI", f"room{room}", f"room{room}", "development" if room <= 2 else "sealed_current_campaign",
            "new_score" if room <= 2 else "none", path=str(paths[0]))
    array = DATA / "IMU-Array-main"
    for group, relative, side in (
        ("calibration_trial", "Calibration/original data/Group1CalibrateData", "development"),
        ("vehicle_trial", "NavigationTest/Group1OriginData", "sealed_current_campaign"),
    ):
        paths = sorted((array / relative).glob("ImuArray_*_IMU.bin"))
        assert len(paths) == 16
        for path in paths:
            size = path.stat().st_size
            assert size and size % 56 == 0
            add(rows, "IMU-Array", f"{group}/{path.stem}", group, side, "none", path=str(path), native_frames=size // 56)
    ox = DATA / "Oxford Inertial Odometry Dataset_2.0/Oxford Inertial Odometry Dataset"
    ox_dev = {"handbag", "handheld", "multi devices", "running"}
    reference_pairs = sorted(p for p in ox.glob("**/syn/vi*.csv") if re.fullmatch(r"vi\d+\.csv", p.name))
    assert len(reference_pairs) == 132
    for label in reference_pairs:
        imu = label.with_name("imu" + label.name[2:])
        assert imu.is_file()
        relative = imu.relative_to(ox)
        group = relative.parts[0]
        side = "development" if group in ox_dev else "sealed_current_campaign"
        add(rows, "OxIOD", relative.as_posix(), group, side, "none", path=str(imu), reference_path=str(label))
    for scenario, reference in (("handbag", "hand1.csv"), ("pocket", "vi1.csv")):
        folder = ox / "test/multi-attachments" / scenario
        assert (folder / "imu1.csv").is_file() and (folder / reference).is_file()
        # Keep the short attachment trial with the same attachment family, conservatively.
        add(rows, "OxIOD", f"test/multi-attachments/{scenario}/imu1.csv", scenario,
            "development" if scenario in ox_dev else "sealed_current_campaign", "none",
            path=str(folder / "imu1.csv"), reference_path=str(folder / reference), supplement=True)
    tango = sorted(ox.glob("large scale/**/syn/tango*.csv"))
    assert len(tango) == 28
    for path in tango:
        add(rows, "OxIOD", path.relative_to(ox).as_posix(), "large scale", "excluded_different_body_reference", "none", path=str(path))

    repo = DATA / "RepoIMU-main/RepoIMU-main"
    sources = [(discipline, path) for discipline in ("Pendulum", "TStick")
               for path in sorted((repo / discipline).glob("*.csv"))]
    assert len(sources) == 95
    ids = {}
    for discipline in ("Pendulum", "TStick"):
        groups = {f"{discipline}/Test{match[1]}" for d, path in sources if d == discipline and "Static" not in path.stem
                  if (match := re.search(r"_Test(\d+)", path.stem))}
        chosen = sorted(groups, key=lambda group: (key("C001|RepoIMU|" + group), group))
        ids[discipline] = set(chosen[:max(1, (len(chosen) + 3) // 4)])
    for discipline, path in sources:
        match = re.search(r"_Test(\d+)", path.stem)
        assert match
        group = f"{discipline}/Test{match[1]}"
        side = "excluded_no_pose" if "Static" in path.stem else "sealed_current_campaign" if group in ids[discipline] else "development"
        add(rows, "RepoIMU", f"{discipline}/{path.stem}", group, side, "none", path=str(path))
    return digest(old_split_path)


NOTES = {
    "SHICE": {"grouping": "Whole device. R1–R4 diagnostic only. R5/R6 sealed for the new lock; historical scores exposed. R1-3/R1-4 excluded, R1-2 absent.",
              "reference": "Turntable encoder pose in same CSV row, world -Y; score observed_loss_ok and context_ok, valid original IMU/pose, after 10 s.",
              "training": "No SHICE fitting, normalization or calibration.", "denominator": "59 present; 57 potentially scored."},
    "BROAD": {"grouping": "Frozen earlier group split by program and repeated variants; 27 source training, 12 development used in prior checkpoint selection.",
              "reference": "Same-row optical opt_quat wxyz/world +Z, official movement mask and finite original IMU/pose.",
              "training": "Only original 27 source groups. Development 12 are historically exposed; there is no new unused holdout in this split.",
              "denominator": "39 trials, MAT/HDF5 copies count once."},
    "Sassari": {"grouping": "Slow/fast source training, medium development. All six sensors of one motion stay together.",
                "reference": "Shared same-row Vicon Qs wxyz/world +Z, 100 Hz. Use previous main alignment, with prior one-row sensitivity documented separately.",
                "training": "12 streams of two motions only. Medium previously used for checkpoint selection; no new unused holdout.",
                "denominator": "18 streams from 3 motions."},
    "TUM-VI": {"grouping": "One room per group: room1/2 development; room3–6 sealed for this campaign. Prior overnight room scoring existed.",
               "reference": "Calibrated T_WI mocap CSV, built-in mounting/time alignment, +Z. SLERP consecutive valid brackets <=25 ms, no extrapolation, 10 s initial exclusion. Do not refit mounting or time offset.",
               "training": "Eligible development source only if explicitly approved in a later batch; label contract available.",
               "denominator": "6 rooms; report native frames and dense valid frames separately."},
    "IMU-Array": {"grouping": "Whole calibration trial is development, whole vehicle trial sealed, each with 16 simultaneous raw sensor channels. No time-block split.",
                  "reference": "Calibration uses same-time finite nonzero static reference specific force directions; navigation uses Leador-A15/GNSS RTS-smoothed roll/pitch/yaw, synchronized in coverage, NED -Z and installed raw axis conversion. Vehicle mount source still needs audit.",
                  "training": "Only development calibration is structurally eligible, static reference windows only after audit. No driving training in this split.",
                  "denominator": "32 raw streams; 2 physical trials. Array combined output and compensated copies excluded."},
    "OxIOD": {"grouping": "Entire top-level Vicon capture scenario on one side. Development is handbag, handheld, multi devices and running. The short attachment diagnostics inherit their attachment side. Deduplicate documented test copies and keep Tango out of phone scoring.",
              "reference": "Same-row synced viN, Vicon marker xyzw, world +Z; nominal 100 Hz. Phone-marker mounting transform and supplement references must be verified, frozen without held-out score feedback.",
              "training": "Development groups only after mount/time audit. Cross-scenario person identity is not guaranteed disjoint.",
              "denominator": "132 standard Vicon, 2 supplements, 28 other-body Tango excluded from phone tilt."},
    "RepoIMU": {"grouping": "Group by Pendulum/TStick plus numeric TestNN, including all trials, a/b and all three pendulum segments. Seal lowest SHA256(C001|RepoIMU|<group>) ceil(25% of dynamic groups) within each apparatus.",
                "reference": "Same-row Vicon marker wxyz/world +Z; sensor installation per apparatus/segment and native clock require audit. Use finite nonzero pose. T-stick 100 Hz nominal, pendulum 90 or 166 Hz.",
                "training": "Development groups only after pose/installation audit. Static files with no pose excluded from tilt labels.",
                "denominator": "95 streams; 88 with pose, 7 IMU-only. Physical grouping further accounts for repeated motions."},
    "YT_Server": {"grouping": "Group by record ID before final underscore/segment number. Official test 213 fixed; exclude every test-family sibling from dev/train. Hold selected development families together and exclude their training siblings.",
                  "reference": "HDF5 acc/gyr/ts/qua same row, wxyz/world +Z convention under diagnosis; quaternion producer, independence and synchronization provenance undocumented. Low-dynamic gravity consistency only.",
                  "training": "Structurally eligible official train families only after both exclusions. No supervised use until reference provenance/quality is accepted.",
                  "denominator": "2,114 file records; explicitly distinguish 12 clean valid C001 and 213 official sealed test."},
}


def main():
    path = HERE / "inventory.json"
    assert not path.exists(), "C001 inventory already frozen"
    rows = []
    split_sha = other_domains(rows)
    yt = indexed_yt(rows)
    expected = {"BROAD": 39, "Sassari": 18, "SHICE": 59, "TUM-VI": 6, "IMU-Array": 32,
                "OxIOD": 162, "RepoIMU": 95, "YT_Server": 2114}
    assert Counter(row["dataset"] for row in rows) == expected
    assert Counter(row["c001"] for row in rows) == {"none": len(rows)-69, "reused_score": 55, "new_score": 14}
    coverage = {domain: {"records": count, "by_side": dict(Counter(r["side"] for r in rows if r["dataset"] == domain))}
                for domain, count in expected.items()}
    write(path, {"frozen_at": stamp(), "campaign_protocol_sha256": digest(CAMPAIGN / "protocol.json"),
                 "prior_external_split_sha256": split_sha, "policy_by_dataset": NOTES,
                 "record_denominators": coverage, "yt": yt, "records": rows})
    print(coverage, flush=True)
    print("selected YT valid", yt["selected_valid"], flush=True)


if __name__ == "__main__":
    main()
