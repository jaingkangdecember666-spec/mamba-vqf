"""Collect the scored-frame regime breakdowns without promoting them to selection metrics."""

from collections import Counter
from statistics import mean
from math import sqrt

from common import BATCH, read_json, stamp, write_json


HARD = {"device": "R2-10", "motion": "R4-2"}


def merge(records):
    result = {}
    if not records:
        return result
    total = sum(row["frames"] for row in records)
    for axis, labels in records[0]["slices"].items():
        sse_total = sum(bucket["squared_error"] for row in records for bucket in row["slices"][axis].values())
        buckets = {}
        for key in labels:
            values = [r["slices"][axis][key] for r in records if r["slices"][axis][key]["frames"]]
            count = sum(r["frames"] for r in values)
            new_error = sum(r["squared_error"] for r in values)
            base_error = sum(r["baseline_squared_error"] for r in values)
            if count:
                delta_by_record = [r["rmse_deg"] - r["baseline_rmse_deg"] for r in values]
                buckets[key] = {"frame_count": count, "frame_share": count / total, "sequence_count": len(values),
                                "error_energy_share": new_error / sse_total if sse_total else None,
                                "rmse_deg_frame_weighted": sqrt(new_error / count), "baseline_rmse_deg_frame_weighted": sqrt(base_error / count),
                                "delta_deg_macro_over_supported_records": mean(delta_by_record),
                                "worse_record_count": sum(d > 1e-6 for d in delta_by_record)}
            else:
                buckets[key] = {"frame_count": 0, "frame_share": 0., "sequence_count": 0}
        result[axis] = buckets
    return result


def pair(record_a, record_b):
    reference = {row["name"]: row for row in record_b}
    result = {}
    for axis, buckets in record_a[0]["slices"].items():
        result[axis] = {}
        for label in buckets:
            effects = []
            for row in record_a:
                a = row["slices"][axis][label]
                b = reference[row["name"]]["slices"][axis][label]
                if a["frames"] != b["frames"]:
                    raise ValueError(f"Different scoring breakdown for {row['name']} {axis}/{label}")
                if a["frames"]:
                    effects.append((row["name"], a["rmse_deg"] - b["rmse_deg"]))
            by_device = {}
            for name, value in effects:
                by_device.setdefault(name.split("-")[0], []).append(value)
            result[axis][label] = {"records": len(effects),
                                  "macro_delta_deg": mean(value for _, value in effects) if effects else None,
                                  "direction_counts": dict(Counter("worse" if value > 1e-6 else "better" if value < -1e-6 else "tied"
                                                                   for _, value in effects)),
                                  "device_macro_delta_deg": {device: mean(val) for device, val in by_device.items()}}
    return result


def report():
    plan = read_json(BATCH / "regime-development-stage.json")
    evidence = {}
    outstanding = []
    for entry in plan["rounds"]:
        identity = entry["audit_round"]
        row = read_json(BATCH / "rounds" / identity / "round.json")
        if row["status"] != "completed":
            outstanding.append({"round": identity, "source": entry["parent_round"], "status": row["status"]})
            continue
        evidence[entry["parent_round"]] = read_json(BATCH / row["result"])
    result = {"updated_at": stamp(), "stage": plan, "outstanding": outstanding, "audits": {}}
    rotation = evidence.get("I00004")
    controls = {row["parent_round"]: row["paired_parent"] for row in plan["rounds"] if row.get("paired_parent")}
    for method, values in evidence.items():
        method_rows = {}
        for axis, rows in values["comparisons"].items():
            hard = next(r for r in rows if r["name"] == HARD[axis])
            other = [r for r in rows if r["name"] != hard["name"]]
            nearby = [r for r in other if r["fold"] == hard["fold"]]
            method_rows[axis] = {"original_frame_count": sum(row["frames"] for row in rows),
                                 "record_count": len(rows), "hard_record": hard,
                                 "hard_record_summary": merge([hard]),
                                 "all_records": merge(rows), "remaining_records": merge(other),
                                 "other_records_in_same_fold": merge(nearby),
                                 "same_fold_control_records": [r["name"] for r in nearby]}
            if rotation is not None and method != "I00004":
                reference = rotation["comparisons"][axis]
                method_rows[axis]["matched_delta_vs_rotation"] = pair(rows, reference)
            if controls.get(method) in evidence:
                control = controls[method]
                method_rows[axis]["matched_training_variant_delta"] = {"control": control,
                    "slices": pair(rows, evidence[control]["comparisons"][axis])}
        result["audits"][method] = method_rows
    write_json(BATCH / "regime-development-summary.json", result)
    print({"updated_at": result["updated_at"], "completed": list(evidence), "outstanding": outstanding}, flush=True)


if __name__ == "__main__":
    report()
