"""Score the same shortlisted development predictions under alternate record weights."""

import sys
from collections import defaultdict
from datetime import datetime
from math import isclose, isfinite
from statistics import fmean

from audit_rounds import BATCH, fingerprint, read


PROGRAMS = ((1,5,9),(2,6,10),(3,7),(4,8))


def scoring_rows(row, frames):
    if row["config"]["kind"] == "baseline":
        data = row["dev_metrics"]["sequences"]
    else:
        data = read(BATCH/row["result"])["sequences"]
    axes = {}
    for axis in ("device","motion"):
        records = data[axis] if isinstance(data,dict) else data
        scores = {r["name"]:r["rmse_deg"] for r in records}
        assert len(records) == len(scores) and {r["name"]:r["frames"] for r in records} == frames
        assert all(isfinite(v) and 0 <= v <= 180 for v in scores.values())
        axes[axis] = scores
    paired = {n:(axes["device"][n]+axes["motion"][n])/2 for n in frames}
    metrics = row["dev_metrics"]
    assert isclose(fmean(paired.values()),metrics.get("selection_rmse_deg",metrics.get("macro_rmse_deg")),abs_tol=1e-8)
    return paired


def ranking(scores, groups):
    assert groups and all(groups)
    values = {key:fmean(fmean(column[n] for n in group) for group in groups) for key,column in scores.items()}
    order = sorted(values,key=lambda k:(values[k],k))
    first = values[order[0]]
    equal = [k for k in order if abs(values[k]-first) <= 1e-9]
    second = next((k for k in order if k not in equal),None)
    return {"leaders":equal,"mean_deg":first,"next":second,"gap_deg":values[second]-first if second else None,
            "candidate_mean_deg":values}


def review_shortlist(index, completed_ids, cutoff, frames):
    if str(BATCH) not in sys.path:
        sys.path.insert(0,str(BATCH))
    from verify_final import replay_with_development_fields
    from verify_shortlist_reasons import audit as audit_reasons, verify_wording_change

    filed = []
    for row in index.values():
        if row["config"]["kind"] == "selection":
            path = BATCH/"rounds"/row["round_id"]/"shortlist.json"
            if path.is_file():
                manifest = read(path)
                filed.append((datetime.fromisoformat(manifest["at"]),manifest))
    if filed:
        cutoff, manifest = max(filed,key=lambda pair:pair[0])
    available = [index[i] for i in completed_ids if index[i]["config"]["kind"] not in {"selection","final"}
                 and datetime.fromisoformat(index[i]["ended_at"]) <= cutoff]
    shortlist, ledger = replay_with_development_fields(available,cutoff)
    if filed:
        assert shortlist == manifest["selected_before_r5"] and fingerprint(shortlist) == manifest["shortlist_sha256"]
    scores = {c["round_id"]:scoring_rows(index[c["round_id"]],frames) for c in shortlist}
    controls = {kind:[c["round_id"] for c in shortlist if c["kind"] == kind] for kind in ("baseline","install_rotation")}
    assert len(scores) == len(shortlist) and len(controls["baseline"]) == 1
    if any(r["config"]["kind"] == "install_rotation" for r in available):
        assert controls["install_rotation"]
    nominations = audit_reasons(available,shortlist,frames,cutoff)
    if (BATCH/"hybrid-fallback-shortlist-reason-audit.json").is_file():
        nominations["wording_change"] = verify_wording_change(index,frames)
    devices, programs = defaultdict(list), defaultdict(list)
    for name in sorted(frames):
        device,number = name.split("-")
        devices[device].append(name)
        label, = [f"M{i}" for i,group in enumerate(PROGRAMS,1) if int(number) in group]
        programs[label].append(name)
    assert len(devices) == len(programs) == 4
    ordered = {"record":ranking(scores,[list(frames)]),"device":ranking(scores,list(devices.values())),
               "program":ranking(scores,list(programs.values()))}
    omissions = {}
    for label,members in {**devices,**programs}.items():
        omissions[label] = ranking(scores,[[n for n in frames if n not in members]])
    return {"at":cutoff.isoformat(),"filed":bool(filed),"count":len(shortlist),"shortlist_sha256":fingerprint(shortlist),"controls":controls,
            "permitted_input_sha256":ledger["input_sha256"],"records":len(frames),"device_counts":{k:len(v) for k,v in sorted(devices.items())},
            "program_counts":{k:len(v) for k,v in sorted(programs.items())},"config_sha256":{k:index[k]["config_sha256"] for k in scores},
            "weighting":ordered,"omissions":omissions,"nomination_evidence":nominations}
