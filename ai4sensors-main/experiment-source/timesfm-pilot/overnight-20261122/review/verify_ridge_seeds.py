"""Independently check the prespecified fresh-seed experiments on R1–R4."""

import hashlib
import math
from collections import Counter
from datetime import datetime
from statistics import fmean

from audit_rounds import BATCH, HERE, check_fold, check_scores, fingerprint, read, read_bytes_shared, write


PROTOCOL_SHA256 = "37cbb9f03fbf718dd3ff28a1f9735692a99a3cf9b89517983164ac1bed8ca4ec"
FAMILIES = ("R00915", "R00900", "R00193")
SEEDS = (104729, 130363, 155921)
AXES = ("device", "motion", "joint_holdout")


def check_job(row, result, reference):
    protocol_path = BATCH / "ridge-independent-seed-protocol.json"
    assert hashlib.sha256(read_bytes_shared(protocol_path)).hexdigest() == PROTOCOL_SHA256
    protocol = read(protocol_path)
    cfg = row["config"]
    trial = next(r for r in protocol["pending_rounds"] if r["round_id"] == row["round_id"])
    seed, family = trial["seed"], trial["source_round"]
    assert set((r["source_round"], r["seed"]) for r in protocol["pending_rounds"]) == {(f, s) for f in FAMILIES for s in SEEDS}
    assert len(protocol["pending_rounds"]) == 9
    assert protocol["development_only"] and not protocol["allowed_in_adaptive_parent_pool"] and not protocol["allowed_in_r5_shortlist"]
    assert set(row["partitions"]) == {"development","excluded"} and not row["dev_metrics"]
    assert not result["adaptive_parent_pool"] and not result["r5_shortlist"]
    assert row["round_id"] == result["round_id"] and row["seed"] == cfg["seed"] == result["seed"] == seed
    assert cfg["source_round"] == result["configuration_source"] == family
    assert datetime.fromisoformat(protocol["created_at"]) <= datetime.fromisoformat(row["started_at"])
    assert datetime.fromisoformat(row["started_at"]) <= datetime.fromisoformat(result["recorded_at"]) <= datetime.fromisoformat(row["ended_at"])
    parent = read(BATCH / "rounds" / family / "round.json")
    frozen = next(r for r in protocol["frozen_families"] if r["source_round"] == family)
    assert frozen["original_config_sha256"] == parent["config_sha256"] == fingerprint(parent["config"])
    parent_result = BATCH / parent["result"]
    assert frozen["original_result_sha256"] == hashlib.sha256(read_bytes_shared(parent_result)).hexdigest()
    stripped = {k:v for k,v in parent["config"].items() if k != "seed"}
    assert frozen["without_seed"] == stripped and fingerprint(stripped) == cfg["without_seed_sha256"] == result["without_seed_sha256"]
    assert result["evaluated_config"] == {**stripped,"seed":seed}
    assert result["evaluated_config_sha256"] == fingerprint(result["evaluated_config"])
    original = read(parent_result)
    assert result["training_mask_version"] == original["mask_version"] == protocol["context_mask_version"]
    assert result["training_context_counts"] == original["training_context_audit"]

    names = set(protocol["dev_names"])
    frames = {r["name"]:r["frames"] for r in reference}
    base = {r["name"]:r["rmse_deg"] for r in reference}
    assert names == set(frames) == set(row["partitions"]["development"]) == set(result["training_context_counts"])
    assert len(names) == 37 and set(protocol["excluded_records"]) == {"R1-3","R1-4"}
    planned = {}
    for fold in protocol["folds"]:
        axis, label = fold["axis"], fold["label"]
        planned[axis,label] = check_fold({"fold":label, "training":fold["fit"], "validation":fold["score"]},axis,names)
    assert len(planned) == result["fit_attempts"] == 24 and Counter(k[0] for k in planned) == {"device":4,"motion":4,"joint_holdout":16}
    fitted = {(r["axis"],r["fold"]):r for r in result["folds"]}
    assert len(fitted) == len(result["folds"]) == result["actual_successful_fits"]
    failures = result["failures"]
    assert set(planned) - set(fitted) == {(r["axis"],r["fold"]) for r in failures if r["phase"] == "fit"}
    assert (result["status"] == "complete") == (not failures)
    assert result["status"] in {"complete","incomplete"} and set(fitted) <= set(planned)

    scores, coverage = {}, {}
    assert set(result["diagnostic_metrics"]) == set(result["scored_sequences"]) == set(AXES)
    for axis in AXES:
        rows = result["scored_sequences"][axis]
        metrics = result["diagnostic_metrics"][axis]
        assert bool(rows) == (metrics is not None)
        if rows:
            check_scores(rows,metrics,frames)
            scores[axis] = fmean(r["rmse_deg"] for r in rows)
            for r in rows:
                assert (axis,r["fold"]) in fitted and r["name"] in planned[axis,r["fold"]]
                assert math.isclose(r["baseline_deg"],base[r["name"]],abs_tol=1e-9)
        for (fold_axis, label), item in fitted.items():
            if fold_axis != axis:
                continue
            held = planned[axis,label]
            scored = {r["name"] for r in rows if r["fold"] == label}
            assert item["fitted"] and len(held) == item["expected_records"] and len(scored) == item["scored_records"]
            missed = {r["record"] for r in failures if r["phase"] == "score" and r["axis"] == axis and r["fold"] == label}
            assert held - scored == missed
        assert (len(rows) < len(names)) == any(r["phase"] == "coverage" and r["axis"] == axis for r in failures)
        coverage[axis] = {"records":len(rows), "frames":sum(r["frames"] for r in rows)}
    assert result["scoring_folds_complete"] == sum(r["scored_records"] == r["expected_records"] for r in fitted.values())
    return {"round_id":row["round_id"],"source":family,"seed":seed,"status":result["status"],
            "successful_fits":len(fitted),"fit_attempts":result["fit_attempts"],"failures":failures,
            "coverage":coverage,"macro_deg":scores,"result":row["result"],
            "joint_by_record_deg":{r["name"]:r["rmse_deg"] for r in result["scored_sequences"]["joint_holdout"]}}


def matched_summary(rows):
    valid = {(r["source"],r["seed"]):r for r in rows if r["status"] == "complete"}
    summaries = []
    for family in FAMILIES:
        tested = [s for s in SEEDS if (family,s) in valid]
        if not tested:
            continue
        axes = {}
        for axis in AXES:
            vals = [valid[family,s]["macro_deg"][axis] for s in tested]
            paired = {str(s):valid[family,s]["macro_deg"][axis]-valid["R00193",s]["macro_deg"][axis]
                      for s in tested if ("R00193",s) in valid}
            axes[axis] = {"mean_deg":fmean(vals), "min_deg":min(vals), "max_deg":max(vals),
                          "paired_differences_deg":paired, "paired_mean_deg":fmean(paired.values()) if paired else None}
        matched_seeds = [s for s in tested if ("R00193",s) in valid]
        per_record = {}
        if matched_seeds:
            for name in valid[family,matched_seeds[0]]["joint_by_record_deg"]:
                per_record[name] = fmean(valid[family,s]["joint_by_record_deg"][name]-valid["R00193",s]["joint_by_record_deg"][name]
                                         for s in matched_seeds)
            assert math.isclose(fmean(per_record.values()),axes["joint_holdout"]["paired_mean_deg"],abs_tol=1e-9)
        summaries.append({"source":family,"seeds":tested,"axes":axes,"joint_paired_per_record_deg":per_record})
    return summaries


def compare_complete_summary(families, published):
    assert published["protocol"] == read(BATCH / "ridge-independent-seed-protocol.json")
    assert {r["source"] for r in families} == set(published["by_family"]) == set(FAMILIES)
    residuals = []
    for family in families:
        target = published["by_family"][family["source"]]
        assert set(family["seeds"]) == set(target["successful_seeds"]) == set(SEEDS)
        for name, ours in family["axes"].items():
            theirs = target["axes"][name]
            paired = theirs["matched_minus_R00193_by_seed_deg"]
            assert set(ours["paired_differences_deg"]) == set(paired)
            residuals.extend(abs(ours["paired_differences_deg"][s]-paired[s]) for s in paired)
            pairs = [(ours["mean_deg"],theirs["mean_deg"]), (ours["min_deg"],theirs["seed_min_max_deg"][0]),
                     (ours["max_deg"],theirs["seed_min_max_deg"][1]), (ours["paired_mean_deg"],theirs["matched_minus_R00193_mean_deg"])]
            residuals.extend(abs(a-b) for a,b in pairs)
    assert max(residuals) < 1e-8
    return max(residuals)


def replay_first_joint_fold():
    """Fit the first registered trial and joint fold by weighted QR on its saved feature inputs."""
    import sys
    import numpy as np
    from scipy.linalg import lstsq
    from scipy.spatial.transform import Rotation

    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from learning import for_record

    protocol = read(BATCH / "ridge-independent-seed-protocol.json")
    for name, digest in protocol["scoring_code_sha256"].items():
        assert hashlib.sha256(read_bytes_shared(BATCH / name)).hexdigest() == digest
    job = read(BATCH / "rounds" / protocol["first_planned_round"] / "round.json")
    source = BATCH / job["result"]
    measured = read(source)
    check_job(job,measured,read(BATCH / "rounds/R00001/result.json")["dev_metrics"]["sequences"])
    cfg = measured["evaluated_config"]
    assert not cfg["embedding"] and not cfg["acc_gate"] and not cfg["gyro_gate"]
    fold = next(f for f in protocol["folds"] if f["axis"] == "joint_holdout")
    train_x, train_y, weights = [], [], []
    for name in fold["fit"]:
        data = for_record(name,cfg)
        mask = data["train"]
        train_x.append(data["matrix"][mask])
        train_y.append(data["target"][mask])
        weights.append(np.full(int(mask.sum()),1 / (len(fold["fit"]) * int(mask.sum()))))
    x, y, w = np.vstack(train_x), np.vstack(train_y), np.concatenate(weights)
    center = w @ x.astype(np.float64)
    variance = w @ np.square(x.astype(np.float64) - center)
    scale = np.sqrt(np.maximum(variance, 9e-6)).astype(np.float32)
    center = center.astype(np.float32)
    rng = np.random.default_rng(cfg["seed"])
    random_basis = rng.standard_normal((x.shape[1],cfg["hidden"]),dtype=np.float32) / np.sqrt(x.shape[1])
    shift = rng.uniform(-.8,.8,cfg["hidden"]).astype(np.float32)

    def features(matrix):
        norm = np.clip((matrix - center) / scale,-12,12)
        return np.column_stack((np.ones(len(norm)),norm,np.tanh(norm @ random_basis + shift)))

    design = features(x)
    dims = design.shape[1]
    penalty = np.full(dims,cfg["alpha"])
    penalty[0] = 1e-9
    a = np.vstack((np.sqrt(w)[:,None] * design,np.diag(np.sqrt(penalty))))
    b = np.vstack((np.sqrt(w)[:,None] * y, np.zeros((dims,3))))
    coef, _, rank, _ = lstsq(a,b,cond=None,lapack_driver="gelsy",check_finite=False)
    assert rank == dims
    expected = {r["name"]:r for r in measured["scored_sequences"]["joint_holdout"] if r["fold"] == fold["label"]}
    checked = []
    for name in fold["score"]:
        test = for_record(name,cfg)
        up = test["up"].astype(np.float64)
        residual = features(test["matrix"]) @ coef.astype(np.float32)
        residual -= np.einsum("ij,ij->i",residual,up)[:,None] * up
        magnitude = np.maximum(np.linalg.norm(residual,axis=1),1e-9)
        residual *= (cfg["gain"] * np.minimum(1,np.tan(np.deg2rad(cfg["cap_deg"])) / magnitude))[:,None]
        vertical = np.array([0.,0.,1.])
        updated = Rotation.from_quat(test["q"],scalar_first=True).apply(residual) + vertical
        updated /= np.linalg.norm(updated,axis=1)[:,None]
        last = np.searchsorted(test["ready"],test["index"],side="right")
        direction = np.vstack((vertical,updated))[last]
        ref = test["ref_world"].astype(np.float64)
        angle = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(direction,ref),axis=1),np.einsum("ij,ij->i",direction,ref)))
        rmse = float(np.sqrt(np.mean(angle**2)))
        delta = abs(rmse-expected[name]["rmse_deg"])
        assert len(angle) == expected[name]["frames"] and delta < 2e-4, (name,delta)
        checked.append({"record":name,"frames":len(angle),"qr_deg":rmse,"reported_deg":expected[name]["rmse_deg"],"abs_delta_deg":delta})
    return {"round_id":job["round_id"],"seed":cfg["seed"],"fold":fold["label"],"fit_records":fold["fit"],
            "train_labels":len(x),"design_rank":int(rank),"checked":checked,"result_sha256":hashlib.sha256(read_bytes_shared(source)).hexdigest(),
            "method":"Existing causal feature inputs, independent per-record weighting, augmented pivoted QR, SciPy quaternion rotations."}


def check_matched_joint_rates(index, completed_ids, frames):
    import numpy as np
    from verify_baselines import AUDIT, digest

    path = BATCH / "ridge-25hz-50hz-joint-protocol.json"
    if not path.is_file():
        return None
    assert digest(path) == "60a1f2cd0c0a2f2488c963fa706605c6a955080ce241b71674ab678f4f62bf07"
    plan = read(path)
    jobs = plan["joint_rounds"]
    if any(job["joint_round"] not in completed_ids for job in jobs):
        return None
    assert set(plan["developments"]) == set(frames) and len(jobs) == 6
    assert len({j["joint_round"] for j in jobs}) == 6 and len({j["parent_round"] for j in jobs}) == 6
    assert plan["validation_source_sha256"] == digest(BATCH / "joint_holdout.py")
    for fold in plan["folds"]:
        check_fold(fold,"joint_holdout",set(frames))
    folds = {f["fold"]:f for f in plan["folds"]}
    assert len(folds) == len(plan["folds"]) == 16
    frozen = {r["parent_round"]:r for r in plan["exact_parents"]}
    context = {25:{},50:{}}
    score_hashes = {}
    cache = BATCH / "cache" / fingerprint({"version":plan["mask_version"],"parameters":{}})[:16]
    for name, count in frames.items():
        with np.load(BATCH.parent/"gru-tilt/data"/f"{name}.npz",allow_pickle=False) as raw:
            score = raw["score"].astype(bool)
        with np.load(AUDIT/name/"screening_masks.npz",allow_pickle=False) as screening:
            usable = screening["context_ok"].astype(bool)
            np.testing.assert_array_equal(screening["sample_index"],np.arange(len(score)))
        assert len(score) == len(usable) and int(score.sum()) == count
        indices = np.flatnonzero(score).astype("<i8",copy=False)
        with np.load(cache/f"{name}.npz",allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["index"],indices)
        score_hashes[name] = {"frames":count,"score_index_sha256":hashlib.sha256(indices.tobytes()).hexdigest()}
        previous_invalid = np.maximum.accumulate(np.where(usable,-1,np.arange(len(usable))))
        for rate in context:
            ends = np.arange(640,len(score)+1,1000//rate)
            potential = score[ends-1]
            eligible = potential & (ends >= 960) & (previous_invalid[ends-1] < ends-960)
            context[rate][name] = {"update_labels_available":int(potential.sum()),"training_labels_eligible":int(eligible.sum()),
                                   "excluded_for_context_or_history":int(potential.sum()-eligible.sum())}
    sequence = {}
    trial_ids = {}
    for job in jobs:
        parent = index[job["parent_round"]]
        cfg = parent["config"]
        original = frozen[parent["round_id"]]
        assert original["config"] == cfg and original["config_sha256"] == parent["config_sha256"] == fingerprint(cfg)
        row = index[job["joint_round"]]
        assert row["config"]["parent_round"] == parent["round_id"] and row["config"]["parent_config_sha256"] == parent["config_sha256"]
        assert cfg["seed"] == row["seed"] == original["seed"] and cfg["context"] == 96 and cfg["parameters"] == {}
        if not job["preexisting"]:
            assert datetime.fromisoformat(row["started_at"]) >= datetime.fromisoformat(plan["registered_at"])
        result = read(BATCH / row["result"])
        assert result["mask_version"] == plan["mask_version"] and set(result["training_context_audit"]) == set(frames)
        for name, expected in context[cfg["update_hz"]].items():
            evidence = result["training_context_audit"][name]
            assert evidence["mask_version"] == plan["mask_version"] and evidence["support_blocks_100hz"] == 96
            assert evidence["min_native_input_rows"] == evidence["max_native_input_rows"] == 960
            assert all(evidence[k] == v for k,v in expected.items()), (row["round_id"],name)
        assert len(result["folds"]) == len(folds) and {f["fold"] for f in result["folds"]} == set(folds)
        for f in result["folds"]:
            assert all(f[k] == folds[f["fold"]][k] for k in ("training","validation"))
        key = (cfg["update_hz"],cfg["seed"])
        assert key not in sequence
        data = result["sequences"]
        scored = data if isinstance(data,list) else data["joint_holdout"]
        sequence[key] = {s["name"]:s["rmse_deg"] for s in scored}
        trial_ids[key] = row["round_id"]
    seeds = (20261122,20261123,20261124)
    assert set(sequence) == {(hz,s) for hz in context for s in seeds}
    pairs = []
    for s in seeds:
        slow, fast = sequence[25,s],sequence[50,s]
        deltas = {n:fast[n]-slow[n] for n in frames}
        pairs.append({"seed":s,"round_25":trial_ids[25,s],"round_50":trial_ids[50,s],"rmse_25":fmean(slow.values()),
                      "rmse_50":fmean(fast.values()),"mean_difference_deg":fmean(deltas.values()),"difference_by_record":deltas})
    published = BATCH / "ridge-25hz-50hz-joint-summary.json"
    if published.is_file():
        summary = read(published)
        assert summary["protocol_sha256"] == digest(path) and summary["common_frames_by_record"] == score_hashes
        assert (summary["new_joint_rounds"],summary["reused_joint_rounds"]) == (
            sum(not j["preexisting"] for j in jobs),sum(j["preexisting"] for j in jobs))
        assert summary["new_fits"] == len(folds)*summary["new_joint_rounds"]
        announced = {p["seed"]:p for p in summary["paired"]}
        assert set(announced) == set(seeds) and len(summary["paired"]) == len(seeds)
        for p in pairs:
            item = announced[p["seed"]]
            assert math.isclose(p["mean_difference_deg"],item["delta_50_minus_25_deg"],abs_tol=1e-12)
            for rate in context:
                rid = p[f"round_{rate}"]
                details = item[f"{rate}hz"]
                assert details["joint_round"] == rid and details["parent_round"] == index[rid]["config"]["parent_round"]
                assert details["result_sha256"] == digest(BATCH/index[rid]["result"])
            assert len(item["sequences"]) == len(frames) and {r["name"] for r in item["sequences"]} == set(frames)
            for r in item["sequences"]:
                n = r["name"]
                assert r["frames"] == frames[n] and math.isclose(r["delta_50_minus_25_deg"],p["difference_by_record"][n],abs_tol=1e-12)
                assert all(math.isclose(r[f"rmse_{rate}_hz_deg"],sequence[rate,p["seed"]][n],abs_tol=1e-12) for rate in context)
        diff = [p["mean_difference_deg"] for p in pairs]
        np.testing.assert_allclose([fmean(diff),min(diff),max(diff)],
                                   [summary["mean_delta_50_minus_25_deg"],*summary["range_delta_50_minus_25_deg"]],rtol=0,atol=1e-12)
        for rate in context:
            values = [p[f"rmse_{rate}"] for p in pairs]
            family = summary["family"][str(rate)]
            np.testing.assert_allclose([fmean(values),min(values),max(values)],[family["mean_deg"],*family["range_deg"]],rtol=0,atol=1e-12)
    by_record = {n:fmean(p["difference_by_record"][n] for p in pairs) for n in frames}
    by_device = {n:n.split("-")[0] for n in frames}
    by_program = {n:f["fold"].split("/")[1] for f in folds.values() for n in f["validation"]}
    strata = {}
    for axis, labels in (("device",by_device),("program",by_program)):
        assert set(labels) == set(frames)
        strata[axis] = {}
        for label in sorted(set(labels.values())):
            members = [n for n in by_record if labels[n] == label]
            remaining = [v for n,v in by_record.items() if labels[n] != label]
            strata[axis][label] = {"records":len(members),"lower_at_50hz":sum(by_record[n]<-1e-8 for n in members),
                                   "mean_50_minus_25_deg":fmean(by_record[n] for n in members),
                                   "mean_omitting_group_deg":fmean(remaining)}
        assert sum(g["records"] for g in strata[axis].values()) == len(frames)
        assert math.isclose(sum(g["records"]*g["mean_50_minus_25_deg"] for g in strata[axis].values())/len(frames),
                            fmean(by_record.values()),abs_tol=1e-12)
    return {"pairs":pairs,"records_per_seed":len(frames),"fits":len(folds)*len(jobs),
            "eligible_updates_by_rate":{str(rate):sum(v["training_labels_eligible"] for v in counts.values()) for rate,counts in context.items()},
            "scoring_frames_per_seed_and_setting":sum(frames.values()),"paired_mean_deg":fmean(p["mean_difference_deg"] for p in pairs),
            "records_lower_at_50hz":sum(v < -1e-8 for v in by_record.values()),"paired_by_group":strata,
            "published_summary_reconciled":published.is_file()}


def regularization_pair(index, completed_ids, frames):
    earlier = ("R01028","R01029","R01022")
    later = ("R01453","R01454","R01442")
    if not set(earlier+later) <= completed_ids:
        return None
    comparisons, by_record, snapshots = [], {}, {}
    expected = set(frames)
    for seed, old_id, new_id in zip((20261122,20261123,20261124),earlier,later,strict=True):
        old, new = (index[i] for i in (old_id,new_id))
        a, b = old["config"], new["config"]
        assert old["seed"] == new["seed"] == a["seed"] == b["seed"] == seed
        assert b == {**a,"alpha":.0017121,"gain":.966} and (a["alpha"],a["gain"]) == (.0024752,.98)
        assert (b["update_hz"],b["context"],b["hidden"],b["embedding"]) == (25,96,128,0)
        records = [read(BATCH/entry["result"]) for entry in (old,new)]
        first,last = records
        assert first["training_context_audit"] == last["training_context_audit"] and set(first["training_context_audit"]) == expected
        assert {r["mask_version"] for r in records} == {"causal-bin-v2-full-window"}
        def folds(r):
            return {(f["axis"],f["fold"]):(sorted(f["training"]),sorted(f["validation"]),f.get("installation_quaternion_wxyz")) for f in r["folds"]}
        assert len(first["folds"]) == len(last["folds"]) == 8 and folds(first) == folds(last)
        axes = {}
        for axis in ("device","motion"):
            old_by, new_by = ({r["name"]:r for r in part["sequences"][axis]} for part in records)
            assert set(old_by) == set(new_by) == expected
            for n in expected:
                assert old_by[n]["fold"] == new_by[n]["fold"] and old_by[n]["frames"] == new_by[n]["frames"] == frames[n]
                assert old_by[n]["baseline_deg"] == new_by[n]["baseline_deg"]
            differences = {n:new_by[n]["rmse_deg"]-old_by[n]["rmse_deg"] for n in sorted(expected)}
            by_record.setdefault(axis,[]).append(differences)
            axes[axis] = fmean(differences.values())
        scores = [r["dev_metrics"]["selection_rmse_deg"] for r in (old,new)]
        assert math.isclose(scores[1]-scores[0],fmean(axes.values()),abs_tol=1e-10)
        comparisons.append({"seed":seed,"old":old_id,"new":new_id,"old_deg":scores[0],"new_deg":scores[1],"axes":axes})
        for item in (old,new):
            snapshots[item["round_id"]] = item["config_sha256"]
    devices = {}
    for axis, seeds in by_record.items():
        per_record = {n:fmean(s[n] for s in seeds) for n in sorted(expected)}
        groups = {}
        for device in sorted({n.split("-")[0] for n in expected}):
            values = [v for n,v in per_record.items() if n.split("-")[0] == device]
            groups[device] = {"records":len(values),"wins":sum(v < -1e-8 for v in values),"mean_deg":fmean(values)}
        devices[axis] = {"delta_deg":fmean(per_record.values()),"by_device":groups,"per_record":per_record}
    return {"pairs":comparisons,"records":len(expected),"frames":sum(frames.values()),"config_sha256":snapshots,"axes":devices,
            "old_mean_deg":fmean(r["old_deg"] for r in comparisons),"new_mean_deg":fmean(r["new_deg"] for r in comparisons)}


if __name__ == "__main__":
    receipt = replay_first_joint_fold()
    write(HERE / "ridge_seed_fold_replay.json",receipt)
    print(receipt["round_id"],receipt["fold"],receipt["train_labels"],receipt["checked"])
