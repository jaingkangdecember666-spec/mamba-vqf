"""Retrospective check of the three global VQF settings using only the 213 development records."""
import csv
import json
from collections import Counter
from statistics import fmean

from lock_from_frontier import ROOT, REVIEW, read, digest


def table(path):
    with path.open(encoding="utf-8-sig",newline="") as source:
        return list(csv.DictReader(source))


def key(row):
    return row["dataset"],row["record"],row["scoring_indices_sha256"],int(row["scored_frames"])


def main():
    base=REVIEW/"vqf_response_controls_development.json"
    frontier=REVIEW/"development_frontier.json"
    controls=read(base)
    choices=read(frontier)
    selected=read(ROOT/"selection-lock.json")
    assert controls["selection_lock_sha256"] is None
    assert digest(ROOT/"C020/sequences.csv")==controls["source_csv_sha256"]
    assert digest(frontier)==selected["evidence_sha256"]["review/development_frontier.json"]
    assert digest(ROOT/"C006/sequences.csv")==selected["evidence_sha256"]["C006/sequences.csv"]
    cases=controls["records"]
    assert len(cases)==len({key(r) for r in cases})==213
    assert {key(r) for r in cases}=={key(r) for r in table(ROOT/"C006/sequences.csv")}
    assert {key(r) for r in cases}=={key(r) for r in table(ROOT/"C020/sequences.csv")}
    complete=choices["full_coverage"]
    assert len(complete)==len({v["signature"] for v in complete})
    selected_record=next(v for v in complete if v["batch"]==selected["selected"]["batch"] and v["column"]==selected["selected"]["column"])
    fields={"1.5":"fast_deg","3":"default_deg","6":"slow_deg"}
    limits={}
    for d,original in controls["domains"].items():
        group=[r for r in cases if r["dataset"]==d]
        means={k:fmean(r[col] for r in group)for k,col in fields.items()}
        assert all(abs(means[k]-original["means"][col])<1e-11 for k,col in fields.items())
        best=min(means,key=means.get)
        for candidate in complete:
            stats=candidate["domains"][d]
            assert len(group)==stats["records"] and abs(stats["baseline_deg"]-means["3"])<.0002
        observed=[r["domains"][d]["candidate_deg"] for r in complete]
        limits[d]={"records":len(group),"three_vqf_deg":means,"strongest_tau_s":best,"threshold_deg":means[best],
                   "selected_deg":selected_record["domains"][d]["candidate_deg"],"minimum_any_tested_deg":min(observed),
                   "number_of_policies_beating_all_three":sum(value<means[best] for value in observed)}
    record=[]
    for v in complete:
        passed=[d for d,vv in limits.items() if v["domains"][d]["candidate_deg"]<vv["threshold_deg"]]
        record.append({"batch":v["batch"],"column":v["column"],"signature":v["signature"],"pass_count":len(passed),"passed":passed})
    histogram=Counter(r["pass_count"]for r in record)
    sources=[base,frontier,ROOT/"C006/sequences.csv",ROOT/"C020/sequences.csv",ROOT/"selection-lock.json"]
    out={"selection_sha256":digest(sources[-1]),"compared":len(complete),"partial_excluded":len(choices["incomplete_coverage"]),
         "histogram":dict(sorted(histogram.items())),"highest_pass_count":max(histogram),"domains":limits,
         "selected":next(r for r in record if r["signature"]==selected_record["signature"]),
         "all_configurations":record,"input_sha256":{p.relative_to(ROOT).as_posix():digest(p) for p in sources},
         "scope":"Each of the three pre-existing fixed VQF settings is evaluated on a common 213-record panel. Within each domain use the lowest of three group means, not a per-record oracle. This checks the development evidence after the campaign and changes no candidate or official evaluation."}
    (REVIEW/"development-vqf-thresholds.json").write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")
    print(out["histogram"],[d for d,v in limits.items() if v["number_of_policies_beating_all_three"]==0])


if __name__=="__main__":
    main()
