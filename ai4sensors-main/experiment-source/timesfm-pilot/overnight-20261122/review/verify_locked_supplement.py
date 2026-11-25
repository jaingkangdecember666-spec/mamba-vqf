"""Verify the post-lock R6 component scores, timing samples and CPU/CUDA output agreement."""

import sys
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_scores, read, write
from replay_locked_components import angular_error, check_saved as check_components
from verify_baselines import digest
from verify_final import raw_shice_samples, score_mask_evidence


FOLDER = BATCH/"locked-tcn-explanation"
PLAN_SHA = "f6a1d2f08c4131d746c19f8ef83cd75453cf34d401bdba73b7d20c659fa6614b"


def replay_parity(config,weights):
    sys.path.insert(0,str(BATCH))
    import learning
    import temporal
    import torch
    from features import cache_record

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    raw = raw_shice_samples("R6-3")
    base = cache_record(raw["imu"],raw["dt"],raw["score"],raw["context"],raw["reference"],config["parameters"],True,keep_scored_quaternion=True)
    part = temporal.prepare(learning.apply_install(base,weights["installation_quaternion_wxyz"]),config)
    rotations = Rotation.from_quat(part["q"],scalar_first=True)
    scored = np.flatnonzero(raw["score"])
    # SHICE contains ten native packets per 100 Hz block; at 10 Hz the first complete 64-block history is released after 70 blocks.
    ready = np.arange(699,(len(raw["imu"])//10)*10,100,dtype=np.int64)
    np.testing.assert_array_equal(ready,part["ready"])
    np.testing.assert_array_equal(scored,part["index"])
    active = np.searchsorted(ready,scored,side="right")
    release,output,rms = {},{},{}
    for backend in ("cpu","cuda"):
        device = torch.device(backend)
        model,normal = temporal.make_model(weights,config,device)
        residual = temporal.inference(part,model,normal,config,device)
        direction = rotations.apply(residual)+[0,0,1]
        direction /= np.linalg.norm(direction,axis=1,keepdims=True)
        release[backend] = direction
        output[backend] = np.vstack(([0.,0.,1.],direction))[active]
        rms[backend] = float(np.sqrt(np.mean(angular_error(output[backend],part["ref_world"])**2)))
    gap = angular_error(output["cpu"],output["cuda"])
    release_gap = angular_error(release["cpu"],release["cuda"])
    choices = np.floor(np.linspace(0,len(ready)-1,110)).astype(np.int64)[10:]
    return {"record":"R6-3","record_source_sha256":digest(raw["source"]),**score_mask_evidence(raw["score"]),
            "releases":len(ready),"sampled_release_indices":ready[choices].tolist(),"rmse_deg":rms,
            "mean_direction_delta_deg":float(np.mean(gap)),"p95_direction_delta_deg":float(np.percentile(gap,95)),
            "max_direction_delta_deg":float(np.max(gap)),"max_release_direction_delta_deg":float(np.max(release_gap))}


def timing_summary(item, count):
    samples = np.asarray(item["samples_ms"],dtype=np.float64)
    assert samples.shape==(count,) and np.isfinite(samples).all() and (samples>=0).all()
    measured = {"p50_ms":float(np.percentile(samples,50)),"p95_ms":float(np.percentile(samples,95)),"max_ms":float(samples.max())}
    assert all(abs(item["summary"][k]-v)<1e-12 for k,v in measured.items())
    return measured


def audit(audit_final,components,parity):
    plan = read(FOLDER/"protocol.json")
    result = read(FOLDER/"result.json")
    status = read(FOLDER/"status.json")
    watcher = read(HERE/"completion-watch.json")
    assert digest(FOLDER/"protocol.json")==result["protocol_sha256"]==PLAN_SHA
    assert status["phase"]=="completed" and all(status[k]==result[k] for k in ("started_at","completed_at","pid"))
    assert result["refits"]==0 and result["formal_artifact_hashes_unchanged"] and result["gpu"]==audit_final["latency"]["accelerator"]
    assert result["torch_cpu_threads"]==plan["benchmark"]["cpu_threads"]==2
    assert watcher["status"]=="finished" and watcher["verified_at"]==result["observer_finished_at"]
    status_formal = read(BATCH/"status.json")
    times = [plan["formal_completed_at"],plan["registered_at"],watcher["verified_at"],result["started_at"],result["completed_at"],status_formal["window_end"]]
    times = list(map(datetime.fromisoformat,times))
    assert times==sorted(times) and plan["formal_round"]==audit_final["final_id"]
    folder = BATCH/"rounds"/audit_final["final_id"]
    for p,key in ((BATCH/"selection-lock.json","selection_lock_file_sha256"),(BATCH/plan["weights_path"],"weights_sha256"),
                  (folder/"result.json","formal_result_sha256"),(folder/"final-records-progress.json","formal_scoring_manifest_sha256"),
                  (BATCH/"ledger.jsonl","formal_ledger_sha256_before_supplement"),(BATCH/"experiment-index.json","formal_index_sha256_before_supplement")):
        assert digest(p)==plan[key],p
    assert result["lock_sha256"]==audit_final["lock_sha256"]==plan["frozen_selection_lock_sha256"]
    assert result["locked_weight_sha256"]==audit_final["weights_sha256"]
    scores = result["r6"]["records"]
    expected = components["datasets"]["R6"]["records"]
    source = {r["name"]:r for r in plan["r6_records"]}
    frames = {n:r["frames"] for n,r in expected.items()}
    assert len(scores)==len(source)==len(expected)==10 and {r["name"] for r in scores}==set(source)==set(expected)
    gaps = []
    for key,arm in (("fast_fullvqf","fast"),("fast_plus_locked_mount","mounted"),("locked_tcn","locked")):
        check_scores([{**r,"rmse_deg":r["rmse_deg"][key]} for r in scores],result["r6"]["summary"][key],frames)
        for r in scores:
            n = r["name"]
            assert r["score_frame_indices_sha256"]==source[n]["score_frame_indices_sha256"]==expected[n]["score_frame_indices_sha256"]
            assert r["full_arm_absolute_reproduction_difference_deg"]==abs(r["rmse_deg"]["locked_tcn"]-expected[n]["rmse_deg"]["locked"])<1e-6
            gaps.append(abs(r["rmse_deg"][key]-expected[n]["rmse_deg"][arm]))
    assert max(gaps)<2e-4
    plan_bench = plan["benchmark"]
    measurements = result["R6-3"]
    passed,nt,ct = (measurements[k] for k in ("output_parity","shared_native_timing","correction_update_timing"))
    assert plan_bench["record"]==parity["record"]=="R6-3" and parity["frames"]==passed["scoring_frames"]==frames["R6-3"]
    assert parity["releases"]==passed["actual_correction_releases"] and passed["native_frames_scoring_indices_sha256"]==parity["score_frame_indices_sha256"]
    assert ct["sampled_original_release_indices"]==parity["sampled_release_indices"]
    assert nt["samples"]==plan_bench["native_frames"]==3000 and nt["warmups"]==plan_bench["native_warmup_frames"]==100
    assert nt["first_timed_native_index"]==plan_bench["native_first_frame"]+nt["warmups"]
    assert ct["warmups_each"]==plan_bench["warmup_updates_per_backend"]==10 and ct["samples_each"]==plan_bench["timed_correction_updates_per_backend"]==100
    assert ct["synchronous_single_stream"] and ct["preexisting_native_blocks"] and ct["cuda_post_forward_synchronization_included"]
    assert ct["sample_scope"]==plan_bench["correction_scope"] and nt["sample_scope"]==plan_bench["native_scope"]
    for backend in ("cpu","cuda"):
        assert abs(parity["rmse_deg"][backend]-passed["same_frozen_weight_tilt_rmse_deg"][backend])<2e-5
    for k in ("mean_direction_delta_deg","p95_direction_delta_deg","max_direction_delta_deg","max_release_direction_delta_deg"):
        assert abs(parity[k]-passed[k])<2e-5,(k,parity[k],passed[k])
    timings = {"native":timing_summary(nt,nt["samples"]),**{b:timing_summary(v,ct["samples_each"]) for b,v in ct["backends"].items()}}
    return {"started_at":result["started_at"],"ended_at":result["completed_at"],"pid":result["pid"],"protocol_sha256":PLAN_SHA,
            "result_sha256":digest(FOLDER/"result.json"),"r6_max_raw_difference_deg":max(gaps),"timing":timings,
            "native_samples":nt["samples"],"update_samples_per_backend":ct["samples_each"],"sampled_release_indices":parity["sampled_release_indices"],
            "parity":parity,"reported_max_direction_delta_deg":passed["max_direction_delta_deg"]}


if __name__=="__main__":
    official = read(HERE/"final_audit.json")
    lock = read(BATCH/"selection-lock.json")
    with np.load(BATCH/lock["weights"],allow_pickle=False) as stored:
        weights = {k:stored[k] for k in stored.files}
    components = check_components(read(HERE/"locked_component_replay.json"),official)
    parity = replay_parity(lock["selected"]["config"],weights)
    result = audit(official,components,parity)
    write(HERE/"locked_tcn_supplement_audit.json",result)
    print(result["timing"],result["r6_max_raw_difference_deg"],parity["max_direction_delta_deg"],flush=True)
