"""Post-fit development diagnostic of the actual sensor predictions saved for F00030."""

import hashlib
from datetime import datetime, timezone
from statistics import fmean

import numpy as np

from audit_rounds import BATCH, HERE, read, write
from verify_final import digest, raw_shice_samples
from verify_forecast_probe import PROTOCOL, PROTOCOL_SHA256


MODELS = ("timesfm_3","timestral_209m")
HORIZONS = (1,4,8)
METHODS = ("persistence","linear_4blocks",*MODELS)


def eligible_windows(indices, observed, context):
    # Every history and its future targets stay within audited context. The last
    # block and the three scored targets must each contain ten original readings.
    release = indices[indices + 10*max(HORIZONS) < len(observed)]
    bad_context = np.r_[0,np.cumsum(~context,dtype=np.int64)]
    bad_packet = np.r_[0,np.cumsum(~observed,dtype=np.int64)]
    ok = bad_context[release+81] == bad_context[release-639]
    for offset in (0,*HORIZONS):
        end = release + 10*offset + 1
        ok &= bad_packet[end] == bad_packet[end-10]
    return np.flatnonzero(ok)


def inspect_record(name, provenance):
    raw = raw_shice_samples(name)
    imu = raw["imu"]
    n_blocks = len(imu)//10
    means = imu[:10*n_blocks].reshape(n_blocks,10,6).mean(axis=1,dtype=np.float64)
    indices = np.arange(699,len(imu),100)
    assert len(indices) == provenance["windows"]
    eligible = eligible_windows(indices,raw["observed"],raw["context"])
    reference_indices = indices[eligible]
    base = reference_indices//10
    prediction_errors = {}
    for model in METHODS:
        if model == "persistence":
            changes = np.zeros((len(base),len(HORIZONS),6))
        elif model == "linear_4blocks":
            # Fixed four-block least-squares slope, anchored to the last observation.
            past = means[base[:,None] + np.arange(-3,1)]
            slope = np.einsum("i,nij->nj",np.array([-1.5,-.5,.5,1.5])/5,past)
            changes = slope[:,None,:] * np.asarray(HORIZONS)[None,:,None]
        else:
            path = BATCH/"forecast-development-cache"/model/(name+".npz")
            assert digest(path) == provenance["feature_hashes"][model]
            with np.load(path,allow_pickle=False) as cache:
                assert str(cache["protocol"]) == PROTOCOL_SHA256 and str(cache["model"]) == model
                np.testing.assert_array_equal(cache["indices"],indices)
                changes = cache["features"][eligible].reshape(-1,len(HORIZONS),6)
        prediction_errors[model] = {}
        for j,h in enumerate(HORIZONS):
            error = changes[:,j] - (means[base+h] - means[base])
            assert np.isfinite(error).all()
            prediction_errors[model][str(h)] = {
                group:float(np.sqrt(np.mean(error[:,section]**2))) if len(base) else None
                for group,section in (("acc",slice(0,3)),("gyro",slice(3,6)))}
    return {"name":name,"issued":len(indices),"eligible":len(base),
            "eligible_native_indices_sha256":hashlib.sha256(reference_indices.astype("<i8").tobytes()).hexdigest(),
            "rmse":prediction_errors}


def summarize(records):
    valid = [r for r in records if r["eligible"]]
    summary = []
    for group,unit in (("acc","m/s²"),("gyro","rad/s")):
        for h in HORIZONS:
            errors = {model:fmean(r["rmse"][model][str(h)][group] for r in valid) for model in METHODS}
            wins = {model:sum(r["rmse"][model][str(h)][group]<r["rmse"]["persistence"][str(h)][group] for r in valid)
                    for model in MODELS}
            trend_wins = {model:sum(r["rmse"][model][str(h)][group]<r["rmse"]["linear_4blocks"][str(h)][group] for r in valid)
                          for model in MODELS}
            summary.append({"signal":group,"unit":unit,"horizon_blocks":h,"horizon_ms":h*10,
                            "record_macro_rmse":errors,"record_wins_vs_persistence":wins,"record_wins_vs_linear":trend_wins})
    return summary


def inspect():
    stage = read(PROTOCOL)
    assert digest(PROTOCOL) == PROTOCOL_SHA256 and stage["horizon_steps"] == list(HORIZONS)
    trial = BATCH/"rounds/F00030/result.json"
    result = read(trial)
    assert result["all_original_development_coverage"] and set(result["records"]) == set(stage["development_records"])
    records = [inspect_record(name,result["records"][name]) for name in stage["development_records"]]
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(), "source_round":"F00030","source_sha256":digest(trial),
            "kind":"post_fit_development_sensor_diagnostic","new_fits":0,"used_for_candidate_selection":False,
            "simple_baselines":"Hold the last 10 ms block constant, or extrapolate from it using the least-squares slope over the last four 10 ms blocks, separately for each sensor axis.",
            "eligibility":"All 64 history blocks through the 8th future block stay within audited context; the last observed block and the three selected target blocks each contain ten real, finite six-axis packets.",
            "aggregation":"Within each record, root mean squared component error over eligible windows and three axes; arithmetic mean across records, separately for acceleration and angular velocity.",
            "issued_windows":sum(r["issued"] for r in records),"eligible_windows":sum(r["eligible"] for r in records),
            "covered_records":sum(r["eligible"]>0 for r in records),"records":records,"summary":summarize(records)}


def check_saved(evidence):
    trial = BATCH/"rounds/F00030/result.json"
    assert evidence["source_round"] == "F00030" and evidence["source_sha256"] == digest(trial)
    assert evidence["new_fits"] == 0 and not evidence["used_for_candidate_selection"]
    reference = read(trial)["records"]
    rows = evidence["records"]
    assert len(rows) == len(reference) and {r["name"] for r in rows} == set(reference)
    for r in rows:
        assert 0 <= r["eligible"] <= r["issued"] == reference[r["name"]]["windows"]
        assert set(r["rmse"]) == set(METHODS) and len(bytes.fromhex(r["eligible_native_indices_sha256"])) == 32
        for scores in r["rmse"].values():
            assert set(scores) == {str(h) for h in HORIZONS}
            for group in scores.values():
                assert set(group) == {"acc","gyro"}
                assert all((v is not None and np.isfinite(v) and v>=0) if r["eligible"] else v is None for v in group.values())
    assert evidence["issued_windows"] == sum(r["issued"] for r in rows)
    assert evidence["eligible_windows"] == sum(r["eligible"] for r in rows)
    assert evidence["covered_records"] == sum(r["eligible"]>0 for r in rows)
    assert evidence["summary"] == summarize(rows)


if __name__ == "__main__":
    result = inspect()
    check_saved(result)
    write(HERE/"sensor_forecast_audit.json",result)
    print(result["covered_records"],result["eligible_windows"],result["issued_windows"],flush=True)
    for r in result["summary"]:
        print(r["signal"],r["horizon_ms"],r["record_macro_rmse"],r["record_wins_vs_persistence"])
