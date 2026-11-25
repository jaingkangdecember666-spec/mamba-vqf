"""Report matched no-memory and reset contrasts for the frozen fixed-state study."""

from common import BATCH, read_json, stamp, write_json


STAGE = BATCH/"ema-state-development-protocol.json"
AXES = ("device","motion","joint_holdout")


def completed(identity):
    row = read_json(BATCH/"rounds"/identity/"round.json")
    return row,read_json(BATCH/row["result"]) if row["status"]=="completed" else None


def original(grouped, simultaneous):
    _, a = completed(grouped)
    _, b = completed(simultaneous)
    return {axis:(b if axis=="joint_holdout" else a)["dev_metrics"][axis] for axis in AXES}


def report():
    stage=read_json(STAGE)
    _, memoryless=completed("D00001")
    fixed=original(*stage["axis_controls"]["rotation"])
    physical=original(*stage["axis_controls"]["legacy_physical"])
    temporal=original(*stage["axis_controls"]["trainable_temporal_reference"])
    found=[]
    for identity in stage["configurations"]:
        row,value=completed(identity)
        meta={"round_id":identity,"status":row["status"],"horizons_seconds":row["config"]["horizons"],"deterministic_fit":True}
        if value and memoryless:
            axes={}
            for axis in AXES:
                current=value["dev_metrics"][axis]
                vanilla=memoryless["dev_metrics"][axis]
                for reference in (vanilla,fixed[axis],physical[axis],temporal[axis]):
                    if (current["records"],current["scoring_frames"]) != (reference["records"],reference["scoring_frames"]):
                        raise ValueError(f"EMA study differs on original score coverage: {identity}/{axis}")
                score=current["macro_rmse_deg"]
                reset=value["same_weight_reset"][axis]["macro_rmse_deg"]
                axes[axis]={"continuous_deg":score,"same_fitted_weights_reset_deg":reset,
                            "fitted_memoryless_deg":vanilla["macro_rmse_deg"],
                            "fixed_installation_control_deg":fixed[axis]["macro_rmse_deg"],
                            "physical_head_control_deg":physical[axis]["macro_rmse_deg"],
                            "tcn_control_deg":temporal[axis]["macro_rmse_deg"],
                            "difference_vs_fitted_memoryless_deg":score-vanilla["macro_rmse_deg"],
                            "difference_vs_reset_deg":score-reset,
                            "difference_vs_physical_head_deg":score-physical[axis]["macro_rmse_deg"],
                            "native_score_frames":current["scoring_frames"],
                            "per_device_deg":{k:row["macro_rmse_deg"] for k,row in current["per_device"].items()},
                            "worst_record":current["worst_record"],"worst_record_deg":current["worst_rmse_deg"]}
            limits=stage["promotion"]
            joint=axes["joint_holdout"]
            decision=(bool(meta["horizons_seconds"]) and
                      joint["difference_vs_fitted_memoryless_deg"]<=-limits["minimum_joint_improvement_over_fitted_no_memory_deg"] and
                      joint["difference_vs_reset_deg"]<=-limits["minimum_joint_improvement_over_same_weight_reset_deg"] and
                      axes["device"]["difference_vs_fitted_memoryless_deg"]<=limits["max_leave_device_degradation_vs_no_memory_deg"] and
                      joint["difference_vs_physical_head_deg"]<=limits["max_joint_degradation_vs_legacy_head_deg"])
            meta.update(axes=axes,promoted=decision,elapsed_seconds=value["elapsed_seconds"],
                        real_deterministic_fits=len(value["folds"]))
        found.append(meta)
    write_json(BATCH/"ema-state-development-summary.json",{"updated_at":stamp(),"protocol":stage,"results":found,
                        "r5_qualified_rounds":[r["round_id"] for r in found if r.get("promoted")]})
    for row in found:
        print(row["round_id"],row["status"],row.get("promoted"),
              {k:(round(v["continuous_deg"],5),round(v["same_fitted_weights_reset_deg"],5))
               for k,v in row.get("axes",{}).items()},flush=True)


if __name__=="__main__":
    report()
