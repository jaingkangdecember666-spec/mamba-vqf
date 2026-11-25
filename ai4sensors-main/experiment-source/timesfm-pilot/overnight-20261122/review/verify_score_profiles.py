"""Compare unrounded per-record development scores across directly evaluated configurations."""

import hashlib
import math
import struct
from itertools import combinations

from audit_rounds import fingerprint, parameter_identity


def score_vector(sequences, frames):
    values = []
    for axis in ("device","motion"):
        records = sequences[axis] if isinstance(sequences,dict) else sequences
        by_name = {r["name"]:r for r in records}
        assert len(records) == len(frames) and {n:r["frames"] for n,r in by_name.items()} == frames
        values.extend(float(by_name[n]["rmse_deg"]) for n in sorted(frames))
    assert all(math.isfinite(v) and 0 <= v <= 180 for v in values)
    return tuple(values)


def summarize(profiles, index, direct_ids, shortlist_ids, frames):
    all_ids = {rid for members in profiles.values() for rid in members}
    assert all_ids == direct_ids and sum(map(len,profiles.values())) == len(all_ids)
    duplicates, manifest = [], {}
    for vector,members in profiles.items():
        assert len(vector) == 2*len(frames)
        h = hashlib.sha256(struct.pack(f"<{len(vector)}d",*vector)).hexdigest()
        manifest.update({rid:h for rid in members})
        if len(members) < 2:
            continue
        ids = sorted(members)
        configs = [index[rid]["config"] for rid in ids]
        keys = set.union(*(set(c) for c in configs))
        varied = sorted(k for k in keys if any((k in c,c.get(k)) != (k in configs[0],configs[0].get(k)) for c in configs[1:]))
        duplicates.append({"rounds":ids,"scored_vector_sha256":h,"varying_config_fields":varied,
                           "distinct_settings_excluding_seed":len({parameter_identity(c) for c in configs}),
                           "config_sha256":{rid:index[rid]["config_sha256"] for rid in ids},
                           "shortlisted":sorted(set(ids)&shortlist_ids)})
    duplicates.sort(key=lambda g:g["rounds"])
    return {"direct_rounds":len(all_ids),"distinct_score_vectors":len(profiles),"records":len(frames),
            "axes":["device","motion"],"sorted_record_order":sorted(frames),"manifest_sha256":fingerprint(manifest),
            "extra_rounds_with_identical_scores":len(all_ids)-len(profiles),"groups":duplicates,
            "different_setting_groups":sum(g["distinct_settings_excluding_seed"]>1 for g in duplicates),
            "cap_only_groups":sum(g["varying_config_fields"] == ["cap_deg"] for g in duplicates),
            "shortlist_groups":sum(len(g["shortlisted"])>=2 for g in duplicates)}


def compare_r5(groups, selection):
    evaluated = {r["round_id"]:r for r in selection["r5_candidates"]}
    failed = {r["round_id"] for r in selection["failed_candidates"]}
    comparisons = []
    for group in groups:
        for left,right in combinations(group["shortlisted"],2):
            if left not in evaluated or right not in evaluated:
                assert all(r in failed for r in (left,right) if r not in evaluated)
                comparisons.append({"rounds":[left,right],"status":"missing_candidate_score"})
                continue
            first,second = [{s["name"]:s for s in evaluated[r]["r5_sequences"]} for r in (left,right)]
            assert set(first) == set(second) and first
            for name,item in first.items():
                other = second[name]
                assert item["frames"] == other["frames"] and item["score_frame_indices_sha256"] == other["score_frame_indices_sha256"]
            diffs = {n:second[n]["rmse_deg"]-first[n]["rmse_deg"] for n in sorted(first)}
            comparisons.append({"rounds":[left,right],"status":"compared","r5_records":len(first),"identical":all(v == 0 for v in diffs.values()),
                                "maximum_absolute_difference_deg":max(map(abs,diffs.values())),"second_minus_first_deg":diffs})
    return comparisons
