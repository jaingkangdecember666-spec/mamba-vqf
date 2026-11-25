"""Score the locked main, coverage, and transfer panels in one resumable job."""
import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from check_sources import ROOT
from davis_handheld import Davis
from locked_replay import SelectedPredictor, sha256
from mixture_replay import replay
from native_catalog import Catalog
from optical_score import metric
from penn_handheld import BASE as PENN_BASE, Penn
from supplemental_data import Supplemental


PANELS = {"main": ("formal_cases", 365), "additional": ("full_coverage_additional_cases", 76)}
IDENTITY = ("dataset", "record", "group", "side")
ANGLES = ("candidate_deg", "physical_deg", "full_vqf_deg")
COLUMNS = (*IDENTITY, "setting", "status", "scored_frames", "scoring_indices_sha256", *ANGLES, "detail")
PHYSICS = ("C022/protocol.json", "C022/coefficient-lock.json")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@contextmanager
def atomic_output(path, encoding="utf-8"):
    with tempfile.NamedTemporaryFile(
        mode="w", encoding=encoding, newline="", dir=path.parent, delete=False
    ) as stream:
        staging = Path(stream.name)
        try:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            staging.replace(path)
        finally:
            stream.close()
            staging.unlink(missing_ok=True)


def write_receipt(path, payload):
    with atomic_output(path) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def single_job(path):
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another process owns the final scoring directory") from exc
        yield  # Closing the handle releases the lock on completion or process exit.


def plan_cases(lock, inventory):
    main, additional = [], []
    for row in inventory.values():
        dataset, side = row["dataset"], row["side"]
        case = {key: row[key] for key in IDENTITY}
        if side.startswith("sealed") or side == "development_previously_used_for_selection":
            main.append(case)
        elif (dataset == "SHICE" and side == "development") or (
            dataset in ("BROAD", "Sassari") and side == "historical_source_train"
        ):
            additional.append(case)
    planned = {}
    for tag, expected in (("main", main), ("additional", additional)):
        field, count = PANELS[tag]
        locked = lock[field]
        by_key = {(case["dataset"], case["record"]): case for case in expected}
        actual = [(case["dataset"], case["record"]) for case in locked]
        if len(locked) != count or len(by_key) != count or set(actual) != by_key.keys():
            raise ValueError(f"Locked {tag} membership differs from the frozen inventory")
        planned[tag] = [by_key[key] for key in actual]
        if any(any(case[k] != ref[k] for k in IDENTITY) for case, ref in zip(locked, planned[tag])):
            raise ValueError(f"Locked {tag} case metadata differs from the frozen inventory")
    return planned


def physical_gains(root):
    force, bank = (read(root / name) for name in PHYSICS)
    return force["C017_k"], bank["k"]


class TransferCatalog:
    def __init__(self, lock_path):
        self.supplemental = Supplemental(lock_path)
        self.davis = Davis(lock_path)
        self.penn = Penn(lock_path)
        self.penn_files = {row["record"]: row for row in read(PENN_BASE / "source-files.json")["records"]}
        self.missing = {
            key: row["exception"] for key, row in self.supplemental.missing.items()
        }

    def cases(self):
        davis = sorted(name for name, role in self.davis.roles.items() if role == "sealed_poster")
        penn = read(PENN_BASE / "preprocessing-lock.json")["sealed_records"]
        assert set(penn) == {name for name, role in self.penn.roles.items() if role == "sealed_after_model_lock"}
        entries = [
            *self.supplemental.cases(),
            *(("DAVIS", name, "poster", name.split("_")[1]) for name in davis),
            *(("PennCOSYVIO", name, "B", name) for name in penn),
        ]
        assert Counter(item[0] for item in entries) == {"RIDI": 25, "EuRoC": 3, "DAVIS": 3, "PennCOSYVIO": 2}
        assert len({item[:2] for item in entries}) == 33
        assert len(self.missing) == 2 and set(self.missing) <= {item[:2] for item in entries if item[0] == "RIDI"}
        return [
            {"dataset": dataset, "record": record, "group": group, "setting": setting, "side": "sealed_transfer"}
            for dataset, record, group, setting in entries
        ]

    def native(self, dataset, record):
        if dataset == "DAVIS":
            return self.davis.native(record)
        if dataset == "PennCOSYVIO":
            for key in ("imu", "pose"):
                entry = self.penn_files[record][key]
                if sha256(PENN_BASE / entry["path"]) != entry["sha256"]:
                    raise ValueError(f"Penn source checksum changed: {entry['path']}")
            return self.penn.native(record)
        return self.supplemental.native(dataset, record)


def score_native(predictor, raw, gains):
    signal, hz, truth, mask = raw
    indices = np.flatnonzero(mask & np.isfinite(signal).all(axis=1)).astype("<i8", copy=False)
    indices_hash = hashlib.sha256(indices.tobytes()).hexdigest()
    if not len(indices):
        return {"status": "no_scoring_frames", "scored_frames": 0, "scoring_indices_sha256": indices_hash,
                "detail": "No paired native scoring rows"}
    paired = predictor.score(raw)
    if paired["scored_frames"] != len(indices) or paired["scoring_indices_sha256"] != indices_hash:
        raise RuntimeError("Independent predictions used different native indices")
    force, bank = gains
    if predictor.force.get("physical_deg") != force or predictor.bank.get("physical_deg") != bank:
        physical = replay(
            signal, hz, {}, {"physical_deg": {}}, indices,
            stable_force_gains={"physical_deg": force}, filter_bank_gains={"physical_deg": bank},
        )
        paired["physical_deg"] = metric(physical["physical_deg"], truth[indices])
    if any(not math.isfinite(paired[key]) or paired[key] < 0 for key in ANGLES):
        raise ValueError("Expected finite nonnegative tilt RMSE")
    return {"status": "scored", **paired}


def evaluate_case(path, case, catalog, predictor, gains, selection_hash):
    identity = {**case, "selection_sha256": selection_hash}
    attempt = 1
    if path.exists():
        old = read(path)
        if any(old[key] != value for key, value in identity.items()):
            raise ValueError(f"Receipt does not belong to this locked case: {path}")
        if old["status"] in ("scored", "no_scoring_frames", "source_unavailable"):
            return old
        attempt = old["attempt"] + 1
    key = case["dataset"], case["record"]
    missing = getattr(catalog, "missing", {}).get(key)
    if missing:
        result = {"status": "source_unavailable", "detail": missing}
    else:
        try:
            raw, quality = catalog.native(*key)
            result = score_native(predictor, raw, gains)
            if "scoring_indices_sha256" in quality and quality["scoring_indices_sha256"] != result["scoring_indices_sha256"]:
                raise ValueError("Adapter and predictor scoring indices differ")
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, AssertionError) as exc:
            result = {"status": "evaluation_failed", "detail": f"{type(exc).__name__}: {exc}"}
    row = {**identity, **result, "attempt": attempt, "completed_at": datetime.now().astimezone().isoformat()}
    write_receipt(path, row)
    print(case["dataset"], case["record"], row["status"], flush=True)
    return row


def write_panel(path, rows):
    with atomic_output(path, encoding="utf-8-sig") as stream:
        table = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        table.writeheader()
        table.writerows(rows)


def score_panel(out, tag, cases, catalog, predictor, gains, lock_path, lock_hash):
    receipts = out / "receipts" / tag
    receipts.mkdir(parents=True, exist_ok=True)
    rows = [
        evaluate_case(receipts / f"{index:03d}.json", case, catalog, predictor, gains, lock_hash)
        for index, case in enumerate(cases, 1)
    ]
    if sha256(lock_path) != lock_hash:
        raise ValueError("Selection lock changed before publishing the panel")
    path = out / f"{tag}.csv"
    write_panel(path, rows)
    return {
        "rows": len(rows), "status_counts": dict(Counter(row["status"] for row in rows)),
        "scored_frames": sum(row.get("scored_frames", 0) for row in rows), "sha256": sha256(path),
        "published_at": datetime.now().astimezone().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection_lock", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--expected-selection-sha256")
    args = parser.parse_args()
    lock_path = (ROOT / "selection-lock.json").resolve()
    if args.selection_lock.resolve() != lock_path or not lock_path.is_file():
        parser.error("The canonical selection-lock.json must exist before execution")

    lock_hash = sha256(lock_path)
    if args.expected_selection_sha256 and lock_hash != args.expected_selection_sha256:
        parser.error("The canonical selection lock does not match the authorized checksum")
    lock = read(lock_path)
    for name, expected in lock["evidence_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Locked evidence has changed: {name}")
    catalog = Catalog(lock_path)
    plan = plan_cases(lock, catalog.rows)
    gains = physical_gains(ROOT)
    evidence = {
        "selection_sha256": lock_hash, "device": args.device, "cases": plan,
        "sha256": {name: sha256(ROOT / name) for name in (*PHYSICS, "review/score_final.py")},
    }

    out = ROOT / "final"
    out.mkdir(exist_ok=True)
    with single_job(out / "scoring.lock"):
        torch.set_num_threads(2)
        predictor = SelectedPredictor(lock_path, args.device)
        if predictor.lock != lock or catalog.lock != lock or sha256(lock_path) != lock_hash:
            raise ValueError("Selection lock changed during preflight")
        manifest = out / "run.json"
        if manifest.exists():
            if read(manifest) != evidence:
                raise ValueError("Existing outputs belong to a different scoring run")
        else:
            write_receipt(manifest, evidence)
        process_path = out / "process.json"
        process = {
            "pid": os.getpid(), "started_at": datetime.now().astimezone().isoformat(),
            "script": str(Path(__file__).resolve()), "selection_sha256": lock_hash, "device": args.device,
            "router_sha256": lock["review_inference"]["response_router"]["sha256"], "tables": {},
        }
        for tag, cases in plan.items():
            write_receipt(process_path, {**process, "status": f"scoring_{tag}"})
            process["tables"][tag] = score_panel(out, tag, cases, catalog, predictor, gains, lock_path, lock_hash)
            print("published", tag, process["tables"][tag], flush=True)
        write_receipt(process_path, {**process, "status": "scoring_transfer"})
        transfer = TransferCatalog(lock_path)
        process["tables"]["transfer"] = score_panel(
            out, "transfer", transfer.cases(), transfer, predictor, gains, lock_path, lock_hash
        )
        has_failures = any("evaluation_failed" in value["status_counts"] for value in process["tables"].values())
        write_receipt(process_path, {**process, "status": "complete_with_errors" if has_failures else "complete",
                                     "finished_at": datetime.now().astimezone().isoformat()})
        print("published", "transfer", process["tables"]["transfer"], flush=True)
    return int(has_failures)


if __name__ == "__main__":
    raise SystemExit(main())
