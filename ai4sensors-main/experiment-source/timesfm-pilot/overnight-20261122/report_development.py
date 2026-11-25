"""Refresh the development comparisons and selection preview from the round ledger."""

from statistics import mean, stdev

from common import BATCH, read_json, stamp, write_json
from final_eval import repetitions, rows_now
from selection_rules import build_shortlist, model_family


def read_completed(row):
    if row["status"] == "completed":
        return read_json(BATCH / row["result"])
    return None


def joint_summary(index):
    plan = read_json(BATCH / "joint-holdout-stage.json")
    base = read_completed(index[plan["pure_fullvqf_ref"]])["dev_metrics"]
    data = {}
    for item in plan["candidates"]:
        round_id = item["round_id"]
        payload = read_completed(index[round_id])
        if payload is None:
            continue
        score = payload["dev_metrics"]["joint_holdout"]
        if score["records"] != base["records"] or score["scoring_frames"] != base["scoring_frames"]:
            raise ValueError(f"Scoring denominator differs in {round_id}")
        data[round_id] = {**item, "scores": score}
    first_tcn = ("J00004", "J00007", "J00008")
    family_sets = {"TCN_32_blocks": first_tcn, "physical-random-ridge-three-seed": ["J00003"]}
    for item in data.values():
        if item.get("seed_family_audit"):
            family_sets.setdefault(item["seed_family_audit"], []).append(item["round_id"])
    summaries = {}
    for name, ids in family_sets.items():
        recorded = [i for i in ids if i in data]
        if len(recorded) < 2:
            continue
        scores = [data[i]["scores"]["macro_rmse_deg"] for i in recorded]
        summaries[name] = {"mean_deg": mean(scores), "sample_std_deg": stdev(scores),
                           "range_deg": [min(scores), max(scores)], "rounds": recorded}
    result = {"updated_at": stamp(), "pure_fullvqf_macro_rmse_deg": base["macro_rmse_deg"], "protocol": plan,
              "rows": list(data.values()), "seed_family_summary": summaries,
              "uncompleted_audit_rounds": [item["round_id"] for item in plan["candidates"] if item["round_id"] not in data]}
    if all(i in data for i in ("J00005", "J00006")):
        emb, off = (data[i]["scores"] for i in ("J00005", "J00006"))
        result["matched_timesfm_embedding_delta_deg"] = emb["macro_rmse_deg"] - off["macro_rmse_deg"]
        result["matched_timesfm_per_device_delta_deg"] = {
            name: metric["macro_rmse_deg"] - off["per_device"][name]["macro_rmse_deg"]
            for name, metric in emb["per_device"].items()}
    write_json(BATCH / "joint-holdout-summary.json", result)
    return result


def hybrid_summary(index):
    plan = read_json(BATCH / "hybrid-development-stage.json")
    summary = []
    for entry in plan["rounds"]:
        row = index[entry["round"]]
        result = read_completed(row)
        current = {"round": row["round_id"], "role": entry["role"], "status": row["status"]}
        if result:
            scores = result["dev_metrics"]["device"]
            worse = sum(r["rmse_deg"] > r["baseline_deg"] + 1e-5 for r in result["sequences"])
            current.update(rmse_deg=scores["macro_rmse_deg"], worse_records=scores["worse_records"],
                           worse_records_above_precision_tolerance=worse, worst_delta_deg=scores["worst_delta_deg"],
                           worst_delta_record=scores["worst_delta_record"],
                           default_fallback_fraction=result["mean_default_fallback_fraction"],
                           fast_only_fraction=result["mean_fast_only_fraction"], mean_fast_weight=result["mean_fast_weight"])
        summary.append(current)
    result = {"at": stamp(), "rounds": summary, "protocol": "R1-R4 only; fixed label-free gate",
              "reported_worse_records_precision_tolerance_deg": 1e-5}
    write_json(BATCH / "hybrid-development-summary.json", result)
    return result


def report():
    rounds = rows_now()
    index = {row["round_id"]: row for row in rounds}
    joint = joint_summary(index)
    hybrid = hybrid_summary(index)
    shortlist = build_shortlist(rounds)
    by_seed = repetitions(rounds)
    selected = [entry for entry in shortlist if index[entry["round_id"]]["config"]["kind"] in {"ridge", "tcn", "legacy_head"}]
    groups = {model_family(index[row["round_id"]]) for row in selected}
    envelope = {"created_at": stamp(), "candidates": shortlist,
                "seed_repeats_for_shortlisted_families": {key: by_seed[key] for key in sorted(groups) if key in by_seed},
                "r5_opened": any((BATCH / "rounds").glob("*/selection-progress.json"))}
    write_json(BATCH / "shortlist-preview.json", envelope)
    print({"reviewed_joints": len(joint["rows"]), "joint_seed_families": joint["seed_family_summary"],
           "completed_hybrid": sum(r["status"] == "completed" for r in hybrid["rounds"]), "shortlist": len(shortlist)},
          flush=True)


if __name__ == "__main__":
    report()
