"""New-seed checks of frozen ridge configurations, outside all selection pools."""

import os
from statistics import mean
from time import perf_counter

import learning
from common import BATCH, EXCLUDED, aggregate, by_device, digest, fingerprint, names, read_json, stamp, write_json
from experiment import baseline_reference
from features import VERSION
from joint_holdout import joint_folds
from optimizer_study import round_record, schedule


STAGE = BATCH/"ridge-independent-seed-protocol.json"
FAMILIES = ("R00915","R00900","R00193")
SEEDS = (104729,130363,155921)
AXES = ("device","motion","joint_holdout")


def plans(records):
    grouped = [(axis,label,[name for name in records if name not in held],held) for axis,label,held in learning.folds(records)]
    return grouped + [("joint_holdout",label,train,held) for label,train,held in joint_folds(records)]


def queue():
    host = read_json(BATCH/"status.json")
    records = names("dev")
    frozen = []
    for identity in FAMILIES:
        round_ = round_record(identity)
        config = round_["config"]
        stripped = {k:v for k,v in config.items() if k!="seed"}
        original = read_json(BATCH/round_["result"])
        if (config["kind"]!="ridge" or not config["hidden"] or config["embedding"] or config.get("installation") or
                original.get("mask_version")!=VERSION):
            raise ValueError(f"{identity} is not a completed, context-valid physical random-feature ridge")
        frozen.append({"source_round":identity,"without_seed":stripped,"without_seed_sha256":fingerprint(stripped),
                       "original_config_sha256":round_["config_sha256"],"original_result_sha256":digest(BATCH/round_["result"])})
    records_to_queue = []
    for i,anchor in enumerate(frozen):
        for j,seed in enumerate(SEEDS,1):
            identity = f"B{3*i+j:05d}"
            records_to_queue.append({"round_id":identity,"source_round":anchor["source_round"],"seed":seed,
                                     "without_seed_sha256":anchor["without_seed_sha256"]})
    protocol = {"created_at":stamp(),"supervisor_pid":host["supervisor_pid"],"requested_after_round":host["round_id"],
                "first_planned_round":records_to_queue[0]["round_id"],"frozen_families":frozen,
                "matched_new_seeds":list(SEEDS),"dev_names":records,"excluded_records":list(EXCLUDED),
                "context_mask_version":VERSION,
                "folds":[{"axis":axis,"label":label,"fit":train,"score":held} for axis,label,train,held in plans(records)],
                "development_only":True,"allowed_in_adaptive_parent_pool":False,"allowed_in_r5_shortlist":False,
                "development_diagnostics_only_no_config_counting":True,
                "scoring_code_sha256":{n:digest(BATCH/n) for n in ("common.py","learning.py","features.py","joint_holdout.py")},
                "pending_rounds":records_to_queue}
    write_json(STAGE,protocol)
    for row in records_to_queue:
        entry = {"kind":"ridge_seed_diagnostic","source_round":row["source_round"],"seed":row["seed"],
                 "without_seed_sha256":row["without_seed_sha256"]}
        observed = schedule(row["round_id"],entry,"independent-ridge-seed-diagnostic",row["source_round"],
                            "Measure the frozen ridge family at a predeclared independent seed and three original fold axes")
        if observed != row["round_id"]:
            raise ValueError(f"Diagnostic {row['round_id']} reused an unexpected round")
    print([(r["round_id"],r["source_round"],r["seed"]) for r in records_to_queue],flush=True)


def run(job,destination):
    selected = job["config"]
    protocol = read_json(STAGE)
    frozen = next(row for row in protocol["frozen_families"] if row["source_round"]==selected["source_round"])
    if fingerprint(frozen["without_seed"]) != selected["without_seed_sha256"]:
        raise ValueError("Frozen seed-free fingerprint differs")
    if any(digest(BATCH/name)!=sha for name,sha in protocol["scoring_code_sha256"].items()):
        raise ValueError("Scoring implementation drifted after pre-registration")
    if selected["seed"] not in protocol["matched_new_seeds"] or names("dev") != protocol["dev_names"]:
        raise ValueError("Unexpected seed or development partition")
    config = {**frozen["without_seed"],"seed":selected["seed"]}
    started = perf_counter()
    observations = {n:learning.for_record(n,config) for n in protocol["dev_names"]}
    actual = {axis:[] for axis in AXES}
    baseline = baseline_reference()
    folds_completed, failures = [], []
    fitted = 0
    attempted = 0
    adjacent = destination.parent/"diagnostic-progress.json"

    def progress():
        write_json(adjacent,{"round_id":job["round_id"],"at":stamp(),"fit_attempts":attempted,
                  "actual_successful_fits":fitted,"folds":folds_completed,"failures":failures})
    progress()
    for fold in protocol["folds"]:
        axis,label = fold["axis"],fold["label"]
        train,held = fold["fit"],fold["score"]
        if set(train) & set(held):
            raise ValueError("Diagnostic training and score sets overlap")
        attempted += 1
        fitted_started = perf_counter()
        try:
            state = learning.fit([observations[n] for n in train],config)
        except Exception as error:
            failures.append({"axis":axis,"fold":label,"phase":"fit","error":repr(error)})
            print(selected["source_round"],selected["seed"],axis,label,"fit_failed",repr(error),flush=True)
            progress()
            continue
        fitted += 1
        scoring_started = perf_counter()
        records = []
        for name in held:
            part = observations[name]
            row = {"name":name,"fold":label,"frames":len(part["index"]),"baseline_deg":baseline[name]}
            try:
                row["rmse_deg"] = learning.evaluate(part,state,config)
                records.append(row)
            except Exception as error:
                failures.append({"axis":axis,"fold":label,"phase":"score","record":name,"frames":row["frames"],"error":repr(error)})
        actual[axis].extend(records)
        folds_completed.append({"axis":axis,"fold":label,"fitted":True,"expected_records":len(held),
                                "scored_records":len(records),"fit_seconds":scoring_started-fitted_started,
                                "score_seconds":perf_counter()-scoring_started})
        progress()
    by_axis = {}
    original = read_json(BATCH/"rounds/R00001/result.json")["dev_metrics"]
    for axis,rows in actual.items():
        if sorted(row["name"] for row in rows)!=sorted(protocol["dev_names"]) or sum(row["frames"] for row in rows)!=original["scoring_frames"]:
            failures.append({"axis":axis,"phase":"coverage","scored_records":len(rows),
                             "frames":sum(row["frames"] for row in rows),"expected_frames":original["scoring_frames"]})
        by_axis[axis] = {**aggregate(rows),"per_device":by_device(rows)} if rows else None
    diagnostic = {"round_id":job["round_id"],"worker_pid":os.getpid(),"supervisor_pid":protocol["supervisor_pid"],
                  "configuration_source":selected["source_round"],"without_seed_sha256":selected["without_seed_sha256"],
                  "evaluated_config":config,"evaluated_config_sha256":fingerprint(config),"seed":selected["seed"],
                  "status":"complete" if not failures else "incomplete","fit_attempts":attempted,"actual_successful_fits":fitted,
                  "scoring_folds_complete":sum(x["scored_records"]==x["expected_records"] for x in folds_completed),
                  "folds":folds_completed,"failures":failures,"scored_sequences":actual,"diagnostic_metrics":by_axis,
                  "recorded_at":stamp(),"elapsed_seconds":perf_counter()-started,
                  "adaptive_parent_pool":False,"r5_shortlist":False,"training_mask_version":VERSION,
                  "training_context_counts":{n:part["training_context"] for n,part in observations.items()}}
    write_json(destination,diagnostic)
    print(diagnostic["configuration_source"],diagnostic["seed"],diagnostic["status"],
          {axis:x["macro_rmse_deg"] for axis,x in by_axis.items() if x},fitted,flush=True)


def report():
    stage = read_json(STAGE)
    row_results = []
    for queued in stage["pending_rounds"]:
        round_ = round_record(queued["round_id"])
        result_file = BATCH/round_["result"]
        file = read_json(result_file) if result_file.is_file() else {}
        partial_path = result_file.parent/"diagnostic-progress.json"
        partial = read_json(partial_path) if partial_path.is_file() else {}
        item = {**queued,"status":round_["status"],"started_at":round_["started_at"],"ended_at":round_["ended_at"],
                "diagnostic_status":file.get("status"),"fit_attempts":file.get("fit_attempts",partial.get("fit_attempts",0)),
                "actual_successful_fits":file.get("actual_successful_fits",partial.get("actual_successful_fits",0)),
                "failures":file.get("failures",partial.get("failures",[])),"diagnostic_metrics":file.get("diagnostic_metrics"),
                "result":round_["result"]}
        row_results.append(item)
    hashes = {anchor["without_seed_sha256"] for anchor in stage["frozen_families"]}
    original_results = []
    for path in (BATCH/"rounds").glob("*/round.json"):
        row = read_json(path)
        config = row["config"]
        if config["kind"]!="ridge" or row["status"]!="completed" or not row["dev_metrics"]:
            continue
        family = fingerprint({k:v for k,v in config.items() if k!="seed"})
        if family in hashes:
            old = row["dev_metrics"]
            original_results.append({"round_id":row["round_id"],"seed":row["seed"],"family_sha256":family,
                         "device_deg":old["device"]["macro_rmse_deg"],"motion_deg":old["motion"]["macro_rmse_deg"],
                         "combined_deg":old["selection_rmse_deg"]})
    aggregates = {}
    for family in FAMILIES:
        finished = [row for row in row_results if row["source_round"]==family and row["diagnostic_status"]=="complete"]
        common = [row for row in row_results if row["source_round"]=="R00193" and row["diagnostic_status"]=="complete"]
        early = {row["seed"]:row for row in common}
        axes = {}
        originals = []
        family_hash = next(anchor["without_seed_sha256"] for anchor in stage["frozen_families"] if anchor["source_round"]==family)
        for record in original_results:
            if record["family_sha256"]==family_hash:
                originals.append(record)
        if finished:
            for axis in (*AXES,"combined_separate_axes"):
                def value(row):
                    measures = row["diagnostic_metrics"]
                    return ((measures["device"]["macro_rmse_deg"]+measures["motion"]["macro_rmse_deg"])/2.
                            if axis=="combined_separate_axes" else measures[axis]["macro_rmse_deg"])
                values = [value(row) for row in finished]
                deltas = {row["seed"]:value(row)-value(early[row["seed"]]) for row in finished if row["seed"] in early}
                axes[axis] = {"mean_deg":mean(values),"seed_min_max_deg":[min(values),max(values)],
                              "seed_range_deg":max(values)-min(values),
                              "matched_minus_R00193_by_seed_deg":deltas,
                              "matched_minus_R00193_mean_deg":mean(deltas.values()) if deltas else None}
        aggregates[family] = {"successful_seeds":[r["seed"] for r in finished],"axes":axes,
                              "already_observed_original_seeds":sorted(originals,key=lambda r:r["seed"])}
    write_json(BATCH/"ridge-independent-seed-summary.json",{"updated_at":stamp(),"protocol":stage,
           "nine_rounds":row_results,"by_family":aggregates,"first_started_diagnostic_round":
               next((row["round_id"] for row in row_results if row["started_at"]),None)})
    for family,payload in aggregates.items():
        print(family,len(payload["successful_seeds"]),payload["axes"],flush=True)


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv:
        report()
    else:
        queue()
