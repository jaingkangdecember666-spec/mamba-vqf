"""Test ordered short-window DCT coefficients against fixed permutation and physical controls."""

import os
from hashlib import sha256
from time import perf_counter

import numpy as np

import learning
from common import BATCH, HISTORY, aggregate, by_device, digest, names, read_json, stamp
from ensemble import measured_prior
from experiment import baseline_reference
from features import SCALE, VERSION, block_indices
from stream_gru import planned_folds, prepare
from tilt_model import contexts


STAGE = BATCH / "dct-development-protocol.json"
WINDOW = 64
ORDERS = np.arange(1, 5)
PERMUTATION_SEED = 641004
ARMS = ("ordered_dct", "permuted_dct", "physical_zero")


def operators():
    k = ORDERS[:, None]
    t = np.arange(WINDOW)[None, :] + .5
    basis = np.sqrt(2. / WINDOW) * np.cos(np.pi * k * t / WINDOW)
    perm = np.random.default_rng(PERMUTATION_SEED).permutation(WINDOW)
    return basis, perm


def coefficients(windows, basis):
    normalized = windows.astype(np.float64) / SCALE
    return np.einsum("kt,btc->bkc", basis, normalized, optimize=False).reshape(len(windows), 24).astype(np.float32)


def verify_operator(basis, perm):
    from scipy.fft import dct

    if sorted(perm.tolist()) != list(range(WINDOW)) or np.array_equal(perm, np.arange(WINDOW)):
        raise ValueError("DCT control requires a fixed bijective nonidentity permutation")
    rng = np.random.default_rng(61003)
    histories = rng.standard_normal((7, WINDOW, 6), dtype=np.float32)
    expected = dct(histories.astype(np.float64) / SCALE, type=2, norm="ortho", axis=1)[:, 1:5]
    ordinary = coefficients(histories, basis)
    np.testing.assert_allclose(ordinary, expected.reshape(7, 24), atol=2e-7, rtol=1e-7)
    reversed_ = coefficients(histories[:, ::-1], basis).reshape(7, 4, 6)
    np.testing.assert_allclose(reversed_, ordinary.reshape(7, 4, 6) * (-1)**ORDERS[None, :, None], atol=1e-7)
    np.testing.assert_allclose(coefficients(np.ones((1, WINDOW, 6)), basis), 0., atol=1e-12)
    permuted = coefficients(histories[:, perm], basis)
    if float(np.max(np.abs(ordinary - permuted))) < .1:
        raise ValueError("Time-shuffling control did not change ordered features")
    np.testing.assert_array_equal(coefficients(histories[:4], basis), ordinary[:4])
    return {"scipy_dct_ii_ortho_matches": True, "reversal_flips_odd_orders_only": True,
            "constant_signal_zero_non_dc": True, "prefix_exact": True, "time_shuffle_changes_coefficients": True}


def collect_raw(records, config, basis, perm):
    raw, attached, evidence = {}, {}, {}
    for name in records:
        data = learning.attach_native(name, config)
        updates = block_indices(len(data["windows"]), config["update_hz"])
        native = data["ready"][updates]
        with np.load(HISTORY / "prepared" / f"{name}.npz", allow_pickle=False) as old:
            np.testing.assert_array_equal(updates, old["update_blocks"])
            np.testing.assert_array_equal(native, old["updates"])
            np.testing.assert_array_equal(data["index"], np.flatnonzero(old["score"]))
        view = contexts(data["windows"], updates)
        np.testing.assert_array_equal(view[:, -1], data["windows"][updates])
        ordered = coefficients(view, basis)
        permuted = coefficients(view[:, perm, :], basis)
        raw[name] = data
        attached[name] = dict(zip(ARMS, (ordered, permuted, np.zeros_like(ordered)), strict=True))
        evidence[name] = {
            "native_release_count": len(native), "frames": len(data["index"]),
            "score_frame_indices_sha256": sha256(np.asarray(data["index"], dtype="<i8").tobytes()).hexdigest(),
            "native_release_indices_sha256": sha256(np.asarray(native, dtype="<i8").tobytes()).hexdigest(),
            "earliest_history_native_release": int(native[0]) if len(native) else None,
        }
    return raw, attached, evidence


def mounted_prepare(data, mount, config, training):
    part = prepare(learning.apply_install(data, mount, include_reference=not training), config, for_training=training)
    up = part["matrix"][:, :3]
    part["up"] = up
    if training:
        delta = part["target"] - up
        part["target"] = delta - (delta * up).sum(axis=1, keepdims=True) * up
    return part


def compare(rows):
    base = {row["name"]: row["rmse_deg"]["physical_zero"] for row in rows}
    metrics = {}
    for arm in (*ARMS, "legacy_head"):
        part = [{"name": row["name"], "frames": row["frames"], "rmse_deg": row["rmse_deg"][arm],
                 "baseline_deg": base[row["name"]]} for row in rows]
        metrics[arm] = {**aggregate(part), "per_device": by_device(part)}
    ordered = metrics["ordered_dct"]["macro_rmse_deg"]
    for arm in ("permuted_dct", "physical_zero", "legacy_head"):
        metrics[f"ordered_minus_{arm}_deg"] = ordered - metrics[arm]["macro_rmse_deg"]
    return metrics


def run(item):
    started = perf_counter()
    config = item["config"]
    stage = read_json(STAGE)
    if item["round_id"] != stage["round_id"] or config["protocol_sha256"] != digest(STAGE):
        raise ValueError("DCT readout requires the original registration")
    path = BATCH / stage["operator_file"]
    if digest(path) != stage["operator_file_sha256"]:
        raise ValueError("Frozen DCT operators changed")
    for filename, sha in stage["anchor_result_sha256"].items():
        if digest(BATCH / filename) != sha:
            raise ValueError(f"Archived P00022 reference changed: {filename}")
    with np.load(path, allow_pickle=False) as saved:
        basis, perm = saved["basis"], saved["permutation"]
    checks = verify_operator(basis, perm)
    print("DCT", item["round_id"], "python worker", os.getpid(), "starting", stamp(), flush=True)
    records = names("dev")
    raw, coefficients_by_record, evidence = collect_raw(records, config, basis, perm)
    expected = measured_prior({"round_id": "P00022", "joint_round": "J00016"})
    baseline = baseline_reference()
    fold_results, sequences = [], {}
    for axis, fold, train, held in planned_folds(records):
        q = learning.installation_fit([raw[name] for name in train], config)
        data = {name: mounted_prepare(raw[name], q, config, name in train) for name in train + held}
        rows = {name: {"name": name, "fold": fold, "frames": evidence[name]["frames"],
                       "score_frame_indices_sha256": evidence[name]["score_frame_indices_sha256"],
                       "baseline_deg": baseline[name], "rmse_deg": {"legacy_head": expected[axis][name]["rmse_deg"]}}
                for name in held}
        fit_log = {}
        for arm in ARMS:
            matrices = {name: {**part, "matrix": np.column_stack((part["matrix"], coefficients_by_record[name][arm]))}
                        for name, part in data.items()}
            fit_at = perf_counter()
            readout = learning.fit([matrices[name] for name in train], config)
            fit_log[arm] = {"fits": 1, "seconds": perf_counter() - fit_at, "readout_inputs": len(readout["std"])}
            for name in held:
                row = rows[name]
                if row["frames"] != expected[axis][name]["frames"] or row["frames"] != len(matrices[name]["index"]):
                    raise ValueError(f"Original scoring support differs in {axis}/{name}")
                row["rmse_deg"][arm] = learning.evaluate(matrices[name], readout, config)
        for row in rows.values():
            row["ordered_delta_deg"] = {arm: row["rmse_deg"]["ordered_dct"] - row["rmse_deg"][arm]
                                        for arm in ("permuted_dct", "physical_zero", "legacy_head")}
        sequences.setdefault(axis, []).extend(rows.values())
        fold_results.append({"axis": axis, "fold": fold, "training": train, "validation": held,
                             "installation_quaternion_wxyz": q.tolist(), "fit_log": fit_log,
                             "training_qualifications": {name: data[name]["training_context"] for name in train},
                             "eligible_training_updates": {name: int(data[name]["train"].sum()) for name in train}})
        print(axis, fold, {arm: float(np.mean([row["rmse_deg"][arm] for row in rows.values()])) for arm in ARMS}, flush=True)
    axis_metrics = {}
    for axis, rows in sequences.items():
        if sorted(row["name"] for row in rows) != sorted(records) or sum(row["frames"] for row in rows) != 8328711:
            raise ValueError(f"Incomplete original native scoring in {axis}")
        axis_metrics[axis] = compare(rows)
    minimum = stage["criteria"]["joint_minimum_improvement_deg"]
    maximum = stage["criteria"]["single_axis_legacy_maximum_regression_deg"]
    criteria = {f"joint_beats_{arm}": axis_metrics["joint_holdout"][f"ordered_minus_{arm}_deg"] <= -minimum
                for arm in ("permuted_dct", "physical_zero", "legacy_head")}
    criteria.update({f"{axis}_acceptable": axis_metrics[axis]["ordered_minus_legacy_head_deg"] <= maximum
                     for axis in ("device", "motion")})
    return {"dev_metrics": None, "axis_metrics": axis_metrics, "folds": fold_results, "sequences": sequences,
            "fit_count": sum(sum(a["fits"] for a in f["fit_log"].values()) for f in fold_results),
            "mount_fit_count": len(fold_results), "real_window_count": sum(v["native_release_count"] for v in evidence.values()),
            "mask_version": VERSION, "scoring_evidence": evidence, "operator_checks": checks, "failed_fits": 0,
            "promotion_checks": criteria, "worth_expanding": all(criteria.values()), "admitted_for_r5": [],
            "python_pid": os.getpid(), "elapsed_seconds": perf_counter() - started}
