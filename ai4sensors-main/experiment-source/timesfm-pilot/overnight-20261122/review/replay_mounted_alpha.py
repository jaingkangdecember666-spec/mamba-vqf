"""Reproduce two regularization strengths with weighted QR on one development fold."""

import hashlib
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.spatial.transform import Rotation

from audit_rounds import BATCH, HERE, check_fold, fingerprint, read, write
from verify_baselines import SHICE, digest
from verify_final import raw_shice_samples, score_mask_evidence
from verify_mount_pair import REFERENCE


IDS = ("M00102", "R01651")
ALPHAS = (0.0018789, 0.0036222)
FOLD = "R1"
SEED = 20261123


def solve_compressed(blocks, alphas):
    """Each QR factor represents the design and targets of one equally weighted record."""
    compressed = np.linalg.qr(np.vstack(blocks), mode="r")
    design, targets = compressed[:, :-3], compressed[:, -3:]
    p = design.shape[1]
    coefficients = []
    for alpha in alphas:
        penalties = np.full(p, alpha, dtype=np.float64)
        penalties[0] = 1e-9
        weights, _, rank, _ = np.linalg.lstsq(
            np.vstack((design, np.diag(np.sqrt(penalties)))),
            np.vstack((targets, np.zeros((p, 3)))), rcond=None)
        assert rank == p and np.isfinite(weights).all()
        coefficients.append(weights.astype(np.float32))
    return coefficients


def training_inputs(train, config, q):
    import learning
    from features import fetch_record

    rows, support = [], {}
    for name in train:
        cached = fetch_record(name, config["parameters"])
        part = learning.prepare(learning.apply_install(cached,q,include_reference=False),config)
        mask = part["train"]
        assert mask.any()
        rows.append((part["matrix"][mask],part["target"][mask]))
        support[name] = part["training_context"]
    return rows, support


def standardize_records(rows):
    mean = np.mean([x.mean(axis=0,dtype=np.float64) for x,_ in rows], axis=0)
    second = np.mean([np.square(x,dtype=np.float64).mean(axis=0) for x,_ in rows], axis=0)
    return {"mean":mean.astype(np.float32),"std":np.sqrt(np.maximum(second-mean**2,9e-6)).astype(np.float32)}


def fit_by_qr(rows, state, alphas):
    import learning

    factors = []
    for x,y in rows:
        matrix = np.column_stack((learning.design(x,state),y)).astype(np.float64)
        matrix /= np.sqrt(len(rows)*len(x))
        factors.append(np.linalg.qr(matrix,mode="r"))
    return solve_compressed(factors,alphas)


def replay_raw_scores(names, reference, q, configs, state, coeffs):
    import learning
    from features import cache_record

    config = configs[0]
    assert all(c["parameters"] == config["parameters"] for c in configs)
    evidence = {}
    for name in names:
        raw = raw_shice_samples(name)
        coverage = score_mask_evidence(raw["score"])
        assert coverage["score_frame_indices_sha256"] == reference[name]["score_index_sha256"]
        cached = cache_record(raw["imu"],raw["dt"],raw["score"],raw["context"],raw["reference"],config["parameters"],True,keep_scored_quaternion=True)
        prepared = learning.prepare(learning.apply_install(cached,q),config)
        np.testing.assert_array_equal(prepared["index"], np.flatnonzero(raw["score"]))
        scores = [float(learning.evaluate(prepared,{**state,"coef":w},cfg)) for w,cfg in zip(coeffs,configs,strict=True)]
        evidence[name] = {"source":str(raw["source"]),"source_sha256":digest(raw["source"]),**coverage,"qr_rmse_deg":scores}
    return evidence


def replay():
    sys.path.insert(0, str(BATCH))

    jobs = [read(BATCH / "rounds" / rid / "round.json") for rid in IDS]
    configs = [j["config"] for j in jobs]
    low, high = configs
    assert low["seed"] == SEED and low["alpha"] == ALPHAS[0] and high == {**low, "alpha": ALPHAS[1]}
    assert low["hidden"] == 128 and low["installation"] == "all" and low["parameters"] == {} and low["embedding"] == 0
    results = [read(BATCH / job["result"]) for job in jobs]
    folds = [next(f for f in data["folds"] if (f["axis"], f["fold"]) == ("device", FOLD)) for data in results]
    target = read(REFERENCE)
    ref_fold = target["folds"]["device:" + FOLD]
    train, valid = ref_fold["training"], ref_fold["validation"]
    for f in folds:
        check_fold(f, "device", set(target["source_records"]))
        assert sorted(f["training"]) == train and sorted(f["validation"]) == valid
    assert folds[0]["installation_quaternion_wxyz"] == folds[1]["installation_quaternion_wxyz"]
    q = np.asarray(folds[0]["installation_quaternion_wxyz"], dtype=np.float32)
    rotation, reference = (Rotation.from_quat(v, scalar_first=True) for v in (q, ref_fold["wxyz"]))
    rotation_delta = float(np.rad2deg((rotation.inv()*reference).magnitude()))
    assert rotation_delta < 2e-5
    rows, support = training_inputs(train, low, q)
    for name in train:
        assert results[0]["training_context_audit"][name] == support[name] == results[1]["training_context_audit"][name]
    state = standardize_records(rows)
    width = len(state["mean"])
    rng = np.random.default_rng(SEED)
    state.update(random=rng.standard_normal((width,low["hidden"]),dtype=np.float32)/np.sqrt(width),
                 shift=rng.uniform(-.8,.8,low["hidden"]).astype(np.float32))
    coeffs = fit_by_qr(rows,state,ALPHAS)
    norms = [float(np.linalg.norm(c[1:].astype(np.float64))) for c in coeffs]
    assert norms[1] < norms[0]
    evidence = replay_raw_scores(valid,target["source_records"],q,configs,state,coeffs)
    return {"reviewed_utc":datetime.now(timezone.utc).isoformat(), "rounds":list(IDS), "alphas":list(ALPHAS), "seed":SEED,
            "held_device":FOLD,"training_records":train,"training_updates":dict(zip(train,[len(x) for x,_ in rows],strict=True)),
            "validation_records":valid,"raw_records":evidence, "mount_reference_delta_deg":rotation_delta,
            "audit_refits":len(coeffs),"intercept_penalty":1e-9,"penalized_coefficient_norms":norms,
            "coefficient_sha256":[hashlib.sha256(c.astype("<f4").tobytes()).hexdigest() for c in coeffs],
            "config_sha256":{rid:fingerprint(cfg) for rid,cfg in zip(IDS,configs,strict=True)},
            "result_sha256":{job["round_id"]:digest(BATCH/job["result"]) for job in jobs}}


def check_saved(receipt):
    assert receipt["rounds"] == list(IDS) and receipt["alphas"] == list(ALPHAS) and receipt["seed"] == SEED
    assert receipt["audit_refits"] == 2 and receipt["held_device"] == FOLD and receipt["intercept_penalty"] == 1e-9
    ref = read(REFERENCE)
    split = ref["folds"]["device:"+FOLD]
    assert receipt["training_records"] == split["training"] and receipt["validation_records"] == split["validation"]
    assert set(receipt["training_updates"]) == set(split["training"]) and all(n>0 for n in receipt["training_updates"].values())
    assert set(receipt["raw_records"]) == set(split["validation"]) and 0<=receipt["mount_reference_delta_deg"]<2e-5
    assert 0<receipt["penalized_coefficient_norms"][1]<receipt["penalized_coefficient_norms"][0]
    official_scores = []
    for rid in IDS:
        job = read(BATCH/"rounds"/rid/"round.json")
        assert job["config_sha256"] == receipt["config_sha256"][rid] == fingerprint(job["config"])
        path = BATCH/job["result"]
        assert digest(path) == receipt["result_sha256"][rid]
        official_scores.append({r["name"]:r for r in read(path)["sequences"]["device"] if r["fold"] == FOLD})
        assert set(official_scores[-1]) == set(split["validation"])
    gaps = []
    for n,row in receipt["raw_records"].items():
        source = SHICE/(FOLD+"_aligned_formal")/(n+".csv")
        assert str(source) == row["source"] and digest(source) == row["source_sha256"]
        assert len(row["qr_rmse_deg"]) == len(IDS)
        for arm,scores in enumerate(official_scores):
            recorded = scores[n]
            assert row["frames"] == recorded["frames"] == ref["source_records"][n]["frames"] and recorded["fold"] == FOLD
            assert row["score_frame_indices_sha256"] == ref["source_records"][n]["score_index_sha256"]
            value = row["qr_rmse_deg"][arm]
            assert np.isfinite(value) and 0<=value<=180
            gaps.append(abs(value-recorded["rmse_deg"]))
    assert max(gaps) < 2e-4
    return {"score_comparisons":len(gaps),"records":len(receipt["raw_records"]),"total_frames":sum(r["frames"] for r in receipt["raw_records"].values()),
            "max_abs_difference_deg":max(gaps),"training_updates":sum(receipt["training_updates"].values()),"norms":receipt["penalized_coefficient_norms"]}


if __name__ == "__main__":
    result = replay()
    print(check_saved(result))
    write(HERE/"mounted_alpha_qr_replay.json",result)
