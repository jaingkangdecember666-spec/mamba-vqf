"""Replay two fixed gain rules on Batch 1's external validation records."""

import csv
import hashlib
import json
import sys
from pathlib import Path

BATCH2 = Path(__file__).resolve().parent
BATCH1 = BATCH2.parent
sys.path.insert(0, str(BATCH1))

from train_pair import (
    CAP, DOMAINS, HOP, WINDOW, ResidualHead, digest, load, macro, normalized, np,
    require, score_attitudes, stamp, tangent, torch, write,
)

METHODS = ("full_vqf", "mamba_full", "mamba_constant", "mamba_angular_gate")
TOLERANCE = 1e-6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def omega(part):
    # Columns 3:6 hold the causal gyro block means in rad/s, before normalization.
    gyro = part.histories[:, :, 3:6].astype(np.float64)
    return np.linalg.norm(gyro, axis=-1).mean(axis=1)


def calibrate(training, fit_scope):
    counts = {(row["domain"], row["sequence"]): row["eligible_training_windows"]
              for row in fit_scope["datasets"] if row["split"] == "train"}
    train_omega, records = [], []
    for part in training:
        key = (part.domain, part.item.name)
        values = omega(part)[part.eligible_training]
        require(len(values) == counts.pop(key) and len(values) > 0,
                f"Unexpected training eligibility for {key}")
        train_omega.append(values)
        records.append({"domain": part.domain, "sequence": part.item.name,
                        "eligible_updates": len(values), "omega_median_rad_s": float(np.median(values))})
    require(not counts, "Missing training records")
    domains = {domain: {"omega_median_rad_s": float(np.median([
        row["omega_median_rad_s"] for row in records if row["domain"] == domain
    ]))} for domain in DOMAINS}
    omega0 = float(np.mean([row["omega_median_rad_s"] for row in domains.values()]))
    require(np.isfinite(omega0) and omega0 > 0, "Invalid omega0")
    for row, values in zip(records, train_omega, strict=True):
        row["g_mean"] = float(np.mean(values / (values + omega0)))
    for domain, row in domains.items():
        subset = [r for r in records if r["domain"] == domain]
        row["records"] = len(subset)
        row["g_mean"] = float(np.mean([r["g_mean"] for r in subset]))
    constant = float(np.mean([row["g_mean"] for row in domains.values()]))
    require(0 < constant < 1, "Invalid constant gain")
    return {"omega0_rad_s": omega0, "constant_gain": constant,
            "train_domains": domains, "train_records": records}


@torch.inference_mode()
def replay(validation, checkpoint, calibration, archived, device):
    model = ResidualHead("mamba_v1")
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().requires_grad_(False).to(device)
    mean, std = checkpoint["mean"].to(device), checkpoint["std"].to(device)
    expected = {(row["domain"], row["sequence"]): row for row in archived}
    rows = []
    max_diff = {"full_vqf": 0., "mamba_full": 0.}
    for part in validation:
        key = (part.domain, part.item.name)
        # Match the batch size and forward path used to produce the original full-gain score.
        deltas = []
        for start in range(0, len(part.histories), 256):
            subset = slice(start, start + 256)
            features = torch.from_numpy(part.histories[subset]).to(device)
            up = torch.from_numpy(part.up[subset]).to(device)
            deltas.append(tangent(model(normalized(features, mean, std)), up, CAP).cpu().numpy())
        delta = np.concatenate(deltas)
        values = omega(part)
        gate = values / (values + calibration["omega0_rad_s"])
        require(np.isfinite(gate).all() and np.isfinite(delta).all(), f"Nonfinite inference for {key}")
        constant = calibration["constant_gain"]
        scores = score_attitudes(
            part.base, part.reference, part.score, part.base[part.index], part.scheduled,
            {"base_delta": delta, "gated_delta": delta * gate[:, None]},
            {"mamba_full": ("base_delta", 1., False),
             "mamba_constant": ("base_delta", constant, False),
             "mamba_angular_gate": ("gated_delta", 1., False)},
            1. / part.item.frequency,
        )
        old = expected.pop(key)
        indices = np.flatnonzero(part.score).astype("<i8")
        require(len(indices) == old["frames"], f"Scoring frame count changed for {key}")
        for current, prior in (("full_vqf", "full_vqf"), ("mamba_full", "mamba_v1")):
            error = abs(scores[current] - old["rmse_deg"][prior])
            require(error <= TOLERANCE, f"{current} failed Batch 1 reproduction for {key}: {error}")
            max_diff[current] = max(max_diff[current], error)
        require(all(np.isfinite(list(scores.values()))), f"Invalid RMSE for {key}")
        rows.append({"domain": part.domain, "sequence": part.item.name,
                     "frames": len(indices), "scoring_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
                     "correction_updates": len(gate), "mean_g": float(gate.mean()), "rmse_deg": scores})
        print(key, scores, flush=True)
    require(not expected, "Missing validation records")
    return rows, max_diff


def summarize(rows):
    summary = {}
    for domain in DOMAINS:
        records = [row for row in rows if row["domain"] == domain]
        summary[domain] = {"sequences": len(records), "frames": sum(row["frames"] for row in records),
                           "macro_rmse_deg": {method: macro(records, method) for method in METHODS}}
    summary["equal_domain_macro_rmse_deg"] = {
        method: float(np.mean([summary[domain]["macro_rmse_deg"][method] for domain in DOMAINS]))
        for method in METHODS
    }
    return summary


def main():
    protocol_path = BATCH2 / "protocol.json"
    calibration_path = BATCH2 / "calibration.json"
    results_path = BATCH2 / "results.json"
    require(not results_path.exists(), "Batch 2 already finished")
    batch1_results = read(BATCH1 / "results.json")
    batch1_protocol = read(BATCH1 / "run-protocol.json")
    sources = {name: digest(BATCH1 / name) for name in (
        "external-split.json", "fit-scope.json", "mamba_v1.pt", "results.json", "run-protocol.json",
        "train_pair.py", "models.py", ".deps/transformers/models/mamba/modeling_mamba.py",
    )}
    require(sources["external-split.json"] == batch1_protocol["split_sha256"], "Batch 1 split changed")
    require(sources["run-protocol.json"] == batch1_results["protocol_sha256"], "Batch 1 protocol changed")
    require(sources["mamba_v1.pt"] == batch1_results["training"]["mamba_v1"]["weights_sha256"],
            "Batch 1 Mamba weights changed")
    require(sources[".deps/transformers/models/mamba/modeling_mamba.py"] ==
            batch1_protocol["mamba_implementation"]["module_sha256"], "Batch 1 Mamba implementation changed")

    protocol = {
        "frozen_at": stamp(), "batch1_artifact_sha256": sources,
        "scope": "Only Batch 1 BROAD/Sassari train and validation. No optimizer steps.",
        "omega": "Mean of the L2 norms of raw gyro block means in the same 64 completed 100 Hz input blocks.",
        "omega0": "Use Batch 1 eligible training updates. Median within record, then median of record medians per domain, then arithmetic mean of the two domains.",
        "angular_gate": "omega / (omega + omega0) for each scheduled update. Multiply existing capped tangent delta before the original quaternion correction.",
        "constant_gain": "Mean angular gain across eligible training updates per record, mean of record means per domain, then arithmetic mean of the two domains.",
        "controls": {"full_vqf": "Default FullVQF with zero correction", "mamba_full": "Batch 1 Mamba with gain 1"},
        "window_blocks": WINDOW, "block_hz": 100, "update_hop_blocks": HOP,
        "score": "Original native frame masks, update release and hold; per-record tilt RMSE, equal records per domain, equal domain weights for combined RMSE.",
        "mean_g": "Arithmetic mean across all original scheduled updates in each validation record.",
        "control_reproduction_tolerance_deg": TOLERANCE,
    }
    if protocol_path.exists():
        previous = read(protocol_path)
        require({k: v for k, v in previous.items() if k != "frozen_at"} ==
                {k: v for k, v in protocol.items() if k != "frozen_at"}, "Frozen protocol changed")
    else:
        write(protocol_path, protocol)

    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    split = load(read(BATCH1 / "external-split.json"))
    calibration = calibrate(split["train"], read(BATCH1 / "fit-scope.json"))
    if calibration_path.exists():
        require(read(calibration_path) == calibration, "Frozen training gain parameters changed")
    else:
        write(calibration_path, calibration)
    print("train-only gain parameters", calibration["omega0_rad_s"], calibration["constant_gain"], flush=True)

    checkpoint = torch.load(BATCH1 / "mamba_v1.pt", weights_only=True, map_location="cpu")
    require(checkpoint["model"] == "mamba_v1" and checkpoint["split_sha256"] == sources["external-split.json"]
            and checkpoint["window"] == WINDOW and checkpoint["cap_deg"] == CAP, "Incompatible checkpoint")
    rows, control_diff = replay(split["validation"], checkpoint, calibration, batch1_results["sequences"], device)
    summary = summarize(rows)
    csv_path = BATCH2 / "validation-sequences.csv"
    staging = csv_path.with_suffix(".tmp")
    fields = ("domain", "sequence", "frames", "scoring_indices_sha256", "correction_updates", "mean_g")
    with staging.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, (*fields, *METHODS))
        writer.writeheader()
        for row in rows:
            writer.writerow({**{name: row[name] for name in fields}, **row["rmse_deg"]})
    staging.replace(csv_path)
    write(results_path, {"completed_at": stamp(), "protocol_sha256": digest(protocol_path),
                         "calibration_sha256": digest(calibration_path), "optimizer_steps": 0,
                         "validation_csv_sha256": digest(csv_path), "device": torch.cuda.get_device_name(device),
                         "omega0_rad_s": calibration["omega0_rad_s"], "constant_gain": calibration["constant_gain"],
                         "batch1_control_max_abs_diff_deg": control_diff, "summary": summary, "sequences": rows})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
