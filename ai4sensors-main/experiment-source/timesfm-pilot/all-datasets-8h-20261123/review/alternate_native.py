"""Replay frozen anchors with the development implementation of C038 and its quaternion error path."""
import csv
import hashlib
import json
import sys
from pathlib import Path

from native_catalog import Catalog
from supplemental_data import Supplemental
from davis_handheld import Davis
from penn_handheld import Penn

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "review"
sys.path[:0] = [str(ROOT / "C038"), str(ROOT / "C036")]
import run_midpoint as mdl
import run_physical_cap as abl

np, torch = mdl.np, mdl.torch
COLUMNS = ("dataset", "record", "status", "scored_frames", "scoring_indices_sha256",
           "full_vqf_deg", "physical_deg", "candidate_deg")


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class OriginalForward:
    def __init__(self, lock):
        if read(lock)["selected"]["batch"] != "C038":
            raise ValueError("This native adapter is specific to C038")
        self.k = read(ROOT / "C022/coefficient-lock.json")["k"]
        self.fixed, _ = mdl.c22.load_fixed(read(ROOT / "C022/protocol.json"))
        self.cfg = mdl.c3.Settings(**read(ROOT / "C003/protocol.json")["thresholds"])
        states = {n: mdl.c33.frozen_state(n) for n in ("C023", "C038")}
        selected = states["C038"]
        self.routers = ({n: mdl.c31.frozen_linear(s) for n,s in states.items()},
                        selected["feature_mean"].numpy(), selected["feature_std"].numpy(), selected["log_p0"])

    @torch.inference_mode()
    def score(self, native, dataset):
        signal, hz, truth, provided_mask = native
        seen = np.isfinite(signal).all(axis=1)
        ix = np.asarray(np.flatnonzero(provided_mask & seen), dtype="<i8")
        if not len(ix):
            return {"status": "no_scoring_frames", "scored_frames": 0,
                    "scoring_indices_sha256": hashlib.sha256(ix.tobytes()).hexdigest()}
        dt = 1. / hz
        optical = hz if dataset in {"OxIOD", "RepoIMU"} or hz < 100. else None
        q0, q, at, release, history, _, acc, _ = mdl.c17.front_end(signal, seen, dt, self.cfg, optical)
        # This is the native development implementation used for C036. Its first two outputs
        # are unchanged by the C036 cap, and can replay an arbitrary frozen C038 readout.
        directions, *_ = abl.held(q, at, release, history, acc, signal, seen, dt, self.k,
                                  self.fixed, self.routers, 0.)
        estimates = {"candidate_deg": np.vstack((mdl.UP, directions["C038_deg"])),
                     "physical_deg": np.vstack((mdl.UP, directions["physical_only_deg"]))}
        errors = mdl.c17.measure(ix, truth, q0, q, release, estimates)
        values = mdl.c17.scores(errors)
        assert np.isfinite(list(values.values())).all()
        return {"status": "scored", "scored_frames": len(ix),
                "scoring_indices_sha256": hashlib.sha256(ix.tobytes()).hexdigest(), **values}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(2)
    lock = ROOT / "selection-lock.json"
    backend = OriginalForward(lock)
    primary = Catalog(lock)
    trained = { (r["dataset"], r["record"]): r for r in read(ROOT / "C038/results.json")["sequences"]}
    physical = { (r["dataset"], r["record"]): r for r in read(ROOT / "C036/results.json")["sequences"]}
    anchors = read(REVIEW / "C038_audit.json")["native_development_rows"]
    preflight = []
    for item in anchors:
        key = item["dataset"], item["record"]
        check = backend.score(primary.native(*key)[0], key[0])
        a, b = trained[key], physical[key]
        assert check["scoring_indices_sha256"] == a["scoring_indices_sha256"] == b["scoring_indices_sha256"]
        gaps = [abs(check["full_vqf_deg"] - a["full_vqf_deg"]), abs(check["candidate_deg"] - a["C038_deg"]),
                abs(check["physical_deg"] - b["physical_only_deg"])]
        assert max(gaps) < .0002, (key, gaps)
        preflight.append({"dataset": key[0], "record": key[1], "max_gap_deg": max(gaps)})
    panels = {"main": [], "transfer": []}
    for dataset, names in read(REVIEW / "final-anchor-spec.json")["anchors"].items():
        for name in names:
            result = backend.score(primary.native(dataset, name)[0], dataset)
            panels["main"].append(dict(dataset=dataset, record=name, **result))
            print(dataset, name, result.get("candidate_deg", result["status"]), flush=True)
    assert len(panels["main"]) == 14
    write_csv(REVIEW / "native-alternate-main.csv", panels["main"])
    extra, davis, penn = Supplemental(lock), Davis(lock), Penn(lock)
    sources = [(d, n) for d, n, *_ in extra.cases()]
    sources.extend(("DAVIS",n) for n, role in sorted(davis.roles.items()) if role == "sealed_poster")
    sources.extend(("PennCOSYVIO",n) for n, role in sorted(penn.roles.items()) if role == "sealed_after_model_lock")
    assert len(sources) == len(set(sources)) == 33
    for dataset, name in sources:
        if (dataset, name) in extra.missing:
            result = {"status": "source_unavailable"}
        else:
            if dataset == "DAVIS": native = davis.native(name)[0]
            elif dataset == "PennCOSYVIO": native = penn.native(name)[0]
            else: native = extra.native(dataset, name)[0]
            result = backend.score(native, dataset)
        panels["transfer"].append(dict(dataset=dataset, record=name, **result))
        print(dataset, name, result.get("candidate_deg", result["status"]), flush=True)
    write_csv(REVIEW / "native-alternate-transfer.csv", panels["transfer"])
    files = [Path(__file__), Path(abl.__file__), Path(mdl.__file__), Path(mdl.c22.__file__),
             Path(mdl.c17.__file__), Path(mdl.c31.__file__), ROOT / "C038/router.pt",
             REVIEW / "native-alternate-main.csv", REVIEW / "native-alternate-transfer.csv"]
    summary = {"lock_sha256": sha(lock), "development_preflight": preflight,
               "hashes": {p.resolve().relative_to(ROOT).as_posix(): sha(p) for p in files}}
    (REVIEW / "native-alternate-summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()
