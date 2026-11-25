"""Expand the frozen two-implementation cross-check to every already-scored case and preserve its paired evidence."""
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter

from alternate_native import OriginalForward, ROOT, REVIEW, read, sha, torch
from native_catalog import Catalog

MEASURES = ("full_vqf_deg", "physical_deg", "candidate_deg")
TOLERANCE_DEG = .0002
OUTPUT = REVIEW / "full-native-verification"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as src:
        return list(csv.DictReader(src))


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, payload):
    staged = path.with_suffix(".tmp")
    staged.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,indent=2)+"\n", encoding="utf8")
    staged.replace(path)


def main():
    locked = ROOT / "selection-lock.json"
    plan = read(locked)
    if plan["selected"]["batch"] != "C038":
        raise ValueError("The original development forward belongs to C038")
    torch.set_num_threads(2)
    inputs = [ROOT / "final" / name for name in ("main.csv","additional.csv","transfer.csv")]
    existing = [REVIEW / "native-alternate-main.csv", REVIEW / "native-alternate-transfer.csv"]
    dependencies = [Path(__file__), REVIEW/"alternate_native.py", locked, *inputs, *existing]
    fingerprints = {p.relative_to(ROOT).as_posix(): sha(p) for p in dependencies}
    groups = {p.stem:rows(p) for p in inputs}
    assert [len(groups[p.stem]) for p in inputs] == [365,76,33]
    pairs = [(tag, row["dataset"], row["record"]) for tag, group in groups.items() for row in group]
    assert len({(d,n) for _,d,n in pairs}) == len(pairs) == 474
    registered = {"identity_sha256":fingerprints,"tolerance_deg":TOLERANCE_DEG,"expected_rows":474,
                  "scope":"Every official row, irrespective of score, using the C036/C038 development forward, C017 quaternion error, same pre-aligned inputs and locked weights. Prior independently calculated rows are reused."}
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT / "plan.json"
    if manifest.exists():
        assert read(manifest)["plan"] == registered
    else:
        write(manifest,{"registered_at":stamp(),"plan":registered})
    archive = {(r["dataset"],r["record"]):r for p in existing for r in rows(p)}
    assert len(archive) == 47
    predictor = OriginalForward(locked)
    source = Catalog(locked)
    out = []
    start = perf_counter()
    for tag, group in groups.items():
        for original in group:
            domain, name = original["dataset"], original["record"]
            key = domain, name
            i = len(out) + 1
            dest = OUTPUT / f"{i:03d}.json"
            if dest.exists():
                check = read(dest)
                assert (check["panel"],check["dataset"],check["record"]) == (tag,domain,name)
                out.append(check)
                continue
            cached = key in archive
            if cached:
                independent = archive[key]
            else:
                assert tag in ("main","additional")
                independent = predictor.score(source.native(domain,name)[0],domain)
            check = {"panel":tag,"dataset":domain,"record":name,"source":"earlier_alternate" if cached else "expanded_alternate",
                     "official_status":original["status"],"native_status":independent["status"]}
            assert independent["status"] == original["status"], check
            if independent["status"] == "scored":
                assert independent["scoring_indices_sha256"] == original["scoring_indices_sha256"], check
                assert int(independent["scored_frames"]) == int(original["scored_frames"]),check
                gaps = {k:abs(float(independent[k])-float(original[k])) for k in MEASURES}
                check.update(scored_frames=int(original["scored_frames"]),indices_sha256=original["scoring_indices_sha256"],
                             alternate_deg={k:float(independent[k]) for k in MEASURES},absolute_gaps_deg=gaps,
                             within_tolerance=max(gaps.values())<TOLERANCE_DEG)
            else:
                assert tag=="transfer" and domain=="RIDI" and independent["status"]=="source_unavailable"
            write(dest,check)
            out.append(check)
            if i%40 == 0:
                print(i,domain,name,round(perf_counter()-start,1),flush=True)
    summaries = {}
    for tag in groups:
        rr = [r for r in out if r["panel"]==tag]
        scored = [r for r in rr if r["official_status"]=="scored"]
        summary = {"requested":len(rr),"scored":len(scored),"verified":sum(r["within_tolerance"] for r in scored),
                   "native_frames":sum(r["scored_frames"] for r in scored),
                   "reused_alternate_rows":sum(r["source"]=="earlier_alternate" for r in rr),
                   "max_difference_deg":{k:max(r["absolute_gaps_deg"][k] for r in scored) for k in MEASURES}}
        summaries[tag] = summary
    result = {"lock_sha256":sha(locked),"completed_at":stamp(),"pid":os.getpid(),"last_execution_seconds":perf_counter()-start,
              "plan_sha256":sha(manifest),"panels":summaries,"requested":len(out),
              "scored":sum(v["scored"] for v in summaries.values()),"native_frames":sum(v["native_frames"] for v in summaries.values()),
              "verified":sum(v["verified"] for v in summaries.values()),"score_missing":sum(r["official_status"]!="scored" for r in out),
              "max_absolute_difference_deg":max(q for v in summaries.values() for q in v["max_difference_deg"].values()),
              "per_record_hashes":{f"{i:03d}.json":sha(OUTPUT/f"{i:03d}.json") for i in range(1,len(out)+1)}}
    assert result["scored"]==472 and result["score_missing"]==2
    write(REVIEW/"all-native-verification.json",result)
    print({k:v for k,v in result.items() if k in ("scored","verified","max_absolute_difference_deg","native_frames")},flush=True)
    if result["verified"] != result["scored"]:
        raise ValueError("A full-panel record exceeds the registered angular tolerance")


if __name__=="__main__":
    main()
