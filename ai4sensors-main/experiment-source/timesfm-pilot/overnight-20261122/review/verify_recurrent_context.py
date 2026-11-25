"""Locate forbidden training context between the original SHICE development labels."""

from datetime import datetime, timezone
import math
import sys

import numpy as np

from audit_rounds import BATCH, HERE, read, write
from verify_baselines import AUDIT


def audit_context():
    records = read(BATCH / "rounds/R00001/round.json")["partitions"]["development"]
    results = []
    for name in records:
        with np.load(BATCH.parent / "tilt-correction/prepared" / f"{name}.npz", allow_pickle=False) as prepared:
            eligible = prepared["updates"][prepared["train_ok"]]
        with np.load(AUDIT / name / "screening_masks.npz", allow_pickle=False) as masks:
            context = masks["context_ok"].astype(bool)
            np.testing.assert_array_equal(masks["sample_index"], np.arange(len(context)))
        assert len(eligible) and np.all(np.diff(eligible) > 0)
        # Each 10 Hz update uses 64 complete 100 Hz features, ten native SHICE rows per feature.
        end = eligible + 1
        bad_count = np.r_[0, np.cumsum(~context,dtype=np.int64)]
        assert np.all(end >= 640) and np.all(bad_count[end] == bad_count[end-640])
        forbidden = np.flatnonzero(~context)
        interior = forbidden[(forbidden > eligible[0]) & (forbidden < eligible[-1])]
        spans = 1 + int((np.diff(interior) > 1).sum()) if len(interior) else 0
        after = int((eligible > interior[0]).sum()) if len(interior) else 0
        labeled_segments = 1 + int((np.diff(bad_count[end]) > 0).sum())
        results.append({"name":name,"eligible_updates":len(eligible),"interior_native_forbidden_spans":spans,
                        "eligible_updates_after_first_interior_forbidden_span":after,
                        "disjoint_context_periods_containing_eligible_updates":labeled_segments})
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(),"records":results,
            "eligible_updates":sum(r["eligible_updates"] for r in results),
            "affected_records":sum(r["interior_native_forbidden_spans"] > 0 for r in results),
            "interior_native_forbidden_spans":sum(r["interior_native_forbidden_spans"] for r in results),
            "eligible_updates_after_first_interior_forbidden_span":sum(r["eligible_updates_after_first_interior_forbidden_span"] for r in results)}


def audit_perturbation(result):
    sys.path.insert(0,str(BATCH))
    import torch
    import stream_gru
    from features import fetch_record

    evidence = read(BATCH / "stream-gru-context-intervention.json")
    assert evidence["labeled_updates_total"] == result["eligible_updates"]
    reference = {r["name"]:r for r in result["records"] if r["interior_native_forbidden_spans"]}
    assert set(reference) == {r["name"] for r in evidence["internal_gap_records"]}
    for observed in evidence["internal_gap_records"]:
        expected = reference[observed["name"]]
        assert observed["native_forbidden_runs_between_labeled_updates"] == expected["interior_native_forbidden_spans"]
        assert observed["qualified_labels_after_first_native_gap"] == expected["eligible_updates_after_first_interior_forbidden_span"]
        assert observed["independent_state_segments"] == expected["disjoint_context_periods_containing_eligible_updates"]
    config = read(BATCH / "rounds/C00011/round.json")["config"]
    name = evidence["intervention"]["record"]
    item = stream_gru.prepare(fetch_record(name,config["parameters"]),config,for_training=True)
    changed = item["matrix"].copy()
    forbidden = ~item["state_input_ok"]
    changed[forbidden] = 10.0
    torch.set_num_threads(1)
    torch.manual_seed(1701)
    original, mean, std, log = stream_gru.training_arrays([item],"cpu")
    altered, mean2, std2, log2 = stream_gru.training_arrays([{**item,"matrix":changed}],"cpu")
    assert log == log2 and log["labels"] == reference[name]["eligible_updates"]
    assert all(torch.equal(a,b) for a,b in zip((*original,mean,std),(*altered,mean2,std2),strict=True))
    segments = list(stream_gru.training_segments(item))
    start, end = segments[1]
    previous_forbidden = np.flatnonzero(forbidden[:start])
    assert len(previous_forbidden) and previous_forbidden[-1] == start-1
    model = stream_gru.ResidualGRU(mean,std,config["width"]).eval()
    through = min(start+3,end)
    physical = torch.tensor(item["matrix"],requires_grad=True)
    _, hidden = model(physical[None,start:through])
    gradient, = torch.autograd.grad(hidden.square().sum(),physical)
    before = float(gradient[:start].abs().max())
    within = float(gradient[start:through].abs().max())
    with torch.no_grad():
        a = physical[None,start-1:through]
        b = torch.as_tensor(changed)[None,start-1:through]
        unsegmented = float((model(a)[1]-model(b)[1]).abs().max())
    assert before == 0 and within > 0 and unsegmented > 0
    return {"record":name,"model_round_config":"C00011","revision":config["recurrent_training_context"],
            "forbidden_updates_perturbed":int(forbidden.sum()),"packed_inputs_targets_weights_unchanged":True,
            "training_normalization_unchanged":True,"max_input_gradient_before_segment":before,
            "max_input_gradient_within_segment":within,"unsegmented_hidden_change_positive_control":unsegmented}


def review_state_gains(completed_ids):
    summary = read(BATCH / "stream-gru-development-summary.json")
    rule = summary["stage"]["repeat_rule"]
    reviewed = []
    for stage in summary["results"]:
        identity = stage["round_id"]
        if identity not in completed_ids or stage["status"] != "completed":
            continue
        result = read(BATCH / "rounds" / identity / "result.json")
        seed = stage["seed"]
        light, joint = summary["legacy_controls_same_seed"][str(seed)]
        grouped = read(BATCH / "rounds" / light / "round.json")
        simultaneous = read(BATCH / "rounds" / joint / "round.json")
        assert grouped["config"]["seed"] == seed and simultaneous["config"]["parent_round"] == light
        values = {}
        for axis in ("device","motion","joint_holdout"):
            source = simultaneous if axis == "joint_holdout" else grouped
            score = result["state_controls"][axis]
            control = source["dev_metrics"][axis]["macro_rmse_deg"]
            observed = stage["axes"][axis]
            diffs = {"continuous_deg":score["continuous"]["macro_rmse_deg"],
                     "reset_every_2_seconds_deg":score["reset_2_seconds"]["macro_rmse_deg"],
                     "reset_every_update_deg":score["reset_each_update"]["macro_rmse_deg"],
                     "same_seed_legacy_control_deg":control}
            assert all(math.isclose(value,observed[key],abs_tol=1e-8) for key,value in diffs.items())
            values[axis] = diffs
        j = values["joint_holdout"]
        device = values["device"]
        passes = (j["reset_every_update_deg"]-j["continuous_deg"] >= rule["min_continuous_joint_improvement_over_reset_each_deg"]
                  and j["continuous_deg"]-j["same_seed_legacy_control_deg"] <= rule["max_joint_degradation_vs_physical_control_deg"]
                  and device["continuous_deg"]-device["same_seed_legacy_control_deg"] <= rule["max_device_degradation_vs_physical_control_deg"])
        assert passes == stage["joint_state_repeat_qualified"]
        reviewed.append({"id":identity,"seed":seed,"qualifies":passes,"joint":j["continuous_deg"],
                         "same_seed_control":j["same_seed_legacy_control_deg"],"control_id":light})
    return reviewed


if __name__ == "__main__":
    result = audit_context()
    if (BATCH / "stream-gru-context-intervention.json").is_file():
        result["independent_intervention"] = audit_perturbation(result)
    write(HERE / "recurrent_context_audit.json",result)
    print({k:v for k,v in result.items() if k != "records"})
