"""Compare the original two checkpoints with C038 on the exact official native rows."""
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from statistics import fmean

from lock_from_frontier import ROOT, REVIEW, digest, read


FINAL = ROOT / "final"
EXTRA = FINAL / "historical-models"
PREVIOUS = ROOT.parent / "transfer-mamba-20261123"
MODELS = ("transfer_mamba", "strict_tcn")
IDENTITY = ("dataset", "record", "group", "side", "setting")
METRICS = ("full_vqf_deg", "transfer_mamba_deg", "strict_tcn_deg", "candidate_deg")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows):
    mapping = {(row["dataset"], row["record"]):row for row in rows}
    assert len(mapping) == len(rows)
    return mapping


def check_inputs():
    worker = read(EXTRA / "run.json")
    process = read(EXTRA / "process.json")
    anchor_plan = read(REVIEW / "historical-baseline-anchors.json")
    anchors = read(REVIEW / "previous-model-native-anchors.json")
    old_lock = PREVIOUS / "review/final-transfer-lock.json"
    current_lock = ROOT / "selection-lock.json"
    assert worker["selection_sha256"] == anchors["selection_sha256"] == digest(current_lock)
    assert worker["historical_selection_sha256"] == anchors["baseline_lock_sha256"] == digest(old_lock)
    assert anchors["preregistered_anchor_sha256"] == digest(REVIEW / "historical-baseline-anchors.json")
    assert datetime.fromisoformat(anchor_plan["registered_at"]) < datetime.fromisoformat(process["started_at"])
    assert set(index(anchors["records"])) == {(d, r) for d, rows in anchor_plan["anchors"].items() for r in rows}
    assert len(anchors["records"]) == anchor_plan["count"] == 18
    for path, checksum in {**worker["source_sha256"], **anchors["forward_scripts"]}.items():
        assert digest(ROOT / path) == checksum, path
    locked = read(old_lock)
    for name in MODELS:
        options = locked["models"][name]
        assert options == worker["models"][name] and options["gain"] == 1.
        assert options["sha256"] == anchors["source_checkpoint_hashes"][name] == digest(PREVIOUS / options["path"])
    assert process["status"] in ("complete", "complete_with_errors")
    assert process["summary_sha256"] == digest(EXTRA / "summary.json")
    return worker, process, anchors


def backend_check(shice, old, actual):
    folder = EXTRA / "tcn-backend-attribution"
    result = read(folder / "summary.json")
    assert digest(folder / "sequences.csv") == result["sequences_sha256"]
    for path, checksum in result["source_sha256"].items():
        assert digest(ROOT / path) == checksum, path
    flags = {r["record"] for r in shice if not r["within_original_tolerances"]}
    by_name = {row["record"]:row for row in result["rows"]}
    assert flags == set(by_name) and len(flags) == result["first_cpu_outside_tolerance"]
    archived_delta = []
    current_delta = []
    for record, row in by_name.items():
        receipt = read(folder / f"{record}.json")
        assert receipt == row and row["native_mask_and_clock_exact"]
        initial = actual["SHICE",record]
        assert row["scoring_indices_sha256"] == initial["mask"] and row["scored_frames"] == initial["scored_frames"]
        assert row["archive_cuda_deg"] == float(old[record]["strict_tcn"])
        assert row["first_cpu_deg"] == initial["strict_tcn_deg"]
        gap = row["replayed_original_cuda_deg"] - row["archive_cuda_deg"]
        assert gap == row["original_cuda_minus_archived_deg"] and abs(gap) <= result["preset_tolerance_deg"]
        archived_delta.append(abs(gap))
        current_delta.append(abs(row["current_input_cuda_deg"] - row["replayed_original_cuda_deg"]))
    probe = by_name[result["controlled_tf32_probe_record"]]
    no_tf32 = abs(probe["first_cpu_deg"] - probe["controlled_current_cuda_no_tf32_deg"])
    assert result["max_abs_original_cuda_vs_archive_deg"] == max(archived_delta)
    assert result["max_abs_current_cuda_vs_original_cuda_deg"] == max(current_delta)
    return {"first_run_flags":len(flags), "original_backend_matching":sum(d <= result["preset_tolerance_deg"] for d in archived_delta),
            "max_archive_gap_deg":max(archived_delta), "max_same_backend_processed_input_gap_deg":max(current_delta),
            "probe":probe["record"], "probe_cpu_vs_cuda_deg":abs(probe["cpu_minus_current_cuda_deg"]),
            "probe_cpu_vs_cuda_no_tf32_deg":no_tf32, "summary_sha256":digest(folder / "summary.json")}


def main():
    worker, process, anchors = check_inputs()
    old = {r["sequence"]:r for r in read_csv(PREVIOUS / "batch8/all-sequences.csv")}
    actual = {}
    counts = Counter()
    missing = []
    shice = []
    source_hashes = {}
    run_sha256 = digest(EXTRA / "run.json")
    for panel, expected in (("main", 365), ("transfer", 33)):
        source, output = FINAL / f"{panel}.csv", EXTRA / f"{panel}.csv"
        assert digest(output) == process["tables"][panel]["sha256"]
        source_hashes.update({p.relative_to(ROOT).as_posix():digest(p) for p in (source, output)})
        official, baselines = index(read_csv(source)), index(read_csv(output))
        assert len(official) == expected and official.keys() == baselines.keys()
        assert not actual.keys() & official.keys()
        evidence = sorted((EXTRA / "receipts" / panel).glob("*.json"))
        assert len(evidence) == expected
        receipts = index([read(p) for p in evidence])
        assert receipts.keys() == baselines.keys()
        for key, row in baselines.items():
            first, raw = official[key], receipts[key]
            assert all(row[k] == first[k] for k in IDENTITY), key
            assert raw["run_sha256"] == run_sha256
            assert raw["status"] == row["status"]
            counts[row["status"]] += 1
            if first["status"] == "source_unavailable":
                assert row["status"] == first["status"] and all(not row[k] for k in METRICS)
                missing.append(key)
                continue
            assert first["status"] == "scored" and row["status"] in ("scored", "batch8_score_mismatch"), key
            assert row["scored_frames"] == first["scored_frames"] and int(first["scored_frames"]) > 0
            assert row["scoring_indices_sha256"] == first["scoring_indices_sha256"] == raw["scoring_indices_sha256"]
            assert len(first["scoring_indices_sha256"]) == 64
            scores = {k:float(row[k]) for k in METRICS}
            assert all(math.isfinite(v) and v >= 0 and v == raw[k] for k, v in scores.items())
            assert scores["candidate_deg"] == float(first["candidate_deg"])
            assert abs(scores["full_vqf_deg"] - float(first["full_vqf_deg"])) <= worker["tolerances_deg"]["full_vqf"]
            if key[0] == "SHICE":
                prior = old[key[1]]
                assert prior["scoring_indices_sha256"] == first["scoring_indices_sha256"] and prior["frames"] == first["scored_frames"]
                gaps = {m:scores[m + "_deg"] - float(prior[m]) for m in ("full_vqf", *MODELS)}
                assert all(abs(v - raw["batch8_differences_deg"][m]) < 1e-12 for m, v in gaps.items())
                ok = (abs(gaps["full_vqf"]) <= worker["tolerances_deg"]["full_vqf"] and
                      all(abs(gaps[m]) <= worker["tolerances_deg"]["historical_r5_r6_models"] for m in MODELS))
                assert (row["status"] == "scored") == ok
                shice.append({"record":key[1], "within_original_tolerances":ok, "signed_gap_deg":gaps})
            else:
                assert row["status"] == "scored", key
            actual[key] = {**scores, "scored_frames":int(first["scored_frames"]), "mask":first["scoring_indices_sha256"]}
    assert len(actual) == 396 and set(missing) == {("RIDI", "hang_leg_new3"), ("RIDI", "huayi_handheld_test1")}
    assert len(shice) == 20
    anchor_differences = []
    for expected in anchors["records"]:
        key = expected["dataset"], expected["record"]
        row = actual[key]
        assert row["scored_frames"] == expected["scored_frames"] and row["mask"] == expected["scoring_indices_sha256"], key
        gaps = {m:abs(row[m + "_deg"] - expected[m]) for m in MODELS}
        gaps["full_vqf"] = abs(row["full_vqf_deg"] - expected["full_vqf_deg"])
        assert max(gaps.values()) < 2e-4, (key, gaps)
        anchor_differences.append({"dataset":key[0], "record":key[1], "abs_difference_deg":gaps})

    by_domain = defaultdict(list)
    for (domain, _), row in actual.items():
        by_domain[domain].append(row)
    assert len(by_domain) == 12
    domains = {}
    for domain, rows in by_domain.items():
        domains[domain] = {
            "requested":len(rows) + sum(d == domain for d, _ in missing), "scored":len(rows),
            "scored_frames":sum(r["scored_frames"] for r in rows),
            "means_deg":{k:fmean(r[k] for r in rows) for k in METRICS},
            "records_c038_beats":{k:sum(r["candidate_deg"] < r[k] for r in rows) for k in METRICS[:-1]},
        }
    wins = {k:sum(d["means_deg"]["candidate_deg"] < d["means_deg"][k] for d in domains.values()) for k in METRICS[:-1]}
    answer = {
        "verified_at":datetime.now().astimezone().isoformat(), "selection_sha256":worker["selection_sha256"],
        "previous_lock_sha256":worker["historical_selection_sha256"], "previous_checkpoint_sha256":anchors["source_checkpoint_hashes"],
        "worker_process":process, "requested":398, "scored_native_rows":len(actual), "worker_status_counts":dict(counts),
        "missing":missing, "domains":domains, "wins_by_domain":wins, "native_anchor_count":len(anchor_differences),
        "native_anchor_max_gap_deg":max(max(x["abs_difference_deg"].values()) for x in anchor_differences),
        "original_archive_records":len(shice), "original_archive_tolerances_deg":worker["tolerances_deg"],
        "original_archive_rows_passing":sum(x["within_original_tolerances"] for x in shice),
        "original_archive_max_gap_deg":{m:max(abs(x["signed_gap_deg"][m]) for x in shice) for m in ("full_vqf", *MODELS)},
        "paired_vs_original_archive":shice, "independent_anchors":anchor_differences,
        "source_hashes":source_hashes,
        "tcn_backend":backend_check(shice, old, actual),
    }
    (REVIEW / "historical-models-verified.json").write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print("Cross-source comparisons", wins, "Anchor max", answer["native_anchor_max_gap_deg"],
          "SHICE archive flags", len(shice) - answer["original_archive_rows_passing"])


if __name__ == "__main__":
    main()
