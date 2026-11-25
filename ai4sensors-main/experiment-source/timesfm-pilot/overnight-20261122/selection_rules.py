"""Deterministic shortlist construction using completed development evidence only."""

from statistics import median, mean

from common import BATCH, fingerprint, read_json


def eligible(round_):
    config = round_["config"]
    if round_["status"] != "completed":
        return False
    if config["kind"] in {"ridge", "tcn", "equivariant"} and config["context"] > 64:
        return read_json(BATCH / round_["result"]).get("mask_version") == "causal-bin-v2-full-window"
    if config["kind"] == "stream_gru":
        return (config.get("recurrent_training_context") == "split-state-at-forbidden-context-v2"
                and read_json(BATCH / round_["result"]).get("recurrent_training_context") == config["recurrent_training_context"])
    if config["kind"] in {"ensemble", "bias_physical_head"}:
        return read_json(BATCH / round_["result"]).get("eligible_for_r5") is True
    return config["kind"] in {"baseline", "filter", "ridge", "install_rotation", "tcn", "equivariant", "hybrid", "legacy_head"}


def dev_value(row):
    result = row["dev_metrics"] or {}
    return result.get("selection_rmse_deg", result.get("macro_rmse_deg", float("inf")))


def model_family(row):
    return fingerprint({key: value for key, value in row["config"].items() if key != "seed"})


def complete_seed_families(rows, minimum=3):
    groups = {}
    for row in rows:
        groups.setdefault(model_family(row), {}).setdefault(row["seed"], row)
    return [list(members.values()) for members in groups.values() if len(members) >= minimum]


def middle_seed(family):
    middle = median(map(dev_value, family))
    return min(family, key=lambda row: (abs(dev_value(row) - middle), row["round_id"]))


def joint_shortlist(items, pool, maximum=3):
    eligible_parents = {row["round_id"]: row for row in pool}
    groups = {}
    for row in items:
        config = row["config"]
        if (row["status"] != "completed" or config["kind"] != "joint"
                or config.get("matched_without_embedding") or config.get("audit_only")
                or config["parent_round"] not in eligible_parents):
            continue
        parent = eligible_parents[config["parent_round"]]
        if config["parent_config_sha256"] != parent["config_sha256"]:
            continue
        value = row["dev_metrics"]["joint_holdout"]["macro_rmse_deg"]
        groups.setdefault(model_family(parent), {})[parent["seed"]] = (parent, value, row["round_id"])
    families = sorted(groups.values(), key=lambda rows: (
        mean(value for _, value, _ in rows.values()), min(audit for _, _, audit in rows.values())))
    selection, represented_kinds = [], set()
    for family in families:
        matches = list(family.values())
        row = middle_seed([parent for parent, _, _ in matches])
        kind = row["config"]["kind"]
        if kind in represented_kinds:
            continue
        audits = ", ".join(sorted(audit for _, _, audit in matches))
        average = mean(value for _, value, _ in matches)
        selection.append((row, f"leading joint unseen-device and unseen-program family for {kind}, {average:.6f} deg across {audits}"))
        represented_kinds.add(kind)
        if len(selection) == maximum:
            break
    return selection


def build_shortlist(items):
    items = sorted(items, key=lambda row: row["round_id"])
    pool = [row for row in items if eligible(row)]
    baselines = [row for row in pool if row["config"]["kind"] == "baseline"]
    if len(baselines) != 1:
        raise ValueError("Exactly one completed, recomputed FullVQF baseline is required")
    baseline = baselines[0]
    eligible_by_kind = {kind: sorted([row for row in pool if row["config"]["kind"] == kind],
                                     key=lambda row: (dev_value(row), row["round_id"]))
                        for kind in ("filter", "ridge", "install_rotation", "tcn", "equivariant", "hybrid", "legacy_head")}
    rotations = eligible_by_kind["install_rotation"]
    learners = eligible_by_kind["ridge"]
    choices = [(baseline, "native pure FullVQF recomputed in this batch")]
    for row in rotations[:2]:
        choices.append((row, "best fixed installation rotations by grouped development"))
    for row in eligible_by_kind["filter"][:3]:
        choices.append((row, "best unlearned filters by grouped development"))
    for row in learners[:4]:
        choices.append((row, "best residual candidates by combined development macro RMSE"))
    for axis in ("device", "motion"):
        if learners:
            row = min(learners, key=lambda item: (item["dev_metrics"][axis]["macro_rmse_deg"], item["round_id"]))
            choices.append((row, f"best residual under {axis} holdout"))
    best_physical = next((row for row in learners if not row["config"]["embedding"]), None)
    if best_physical:
        choices.append((best_physical, "best physical-only readout"))
    best_embedding = next((row for row in learners if row["config"]["embedding"]), None)
    if best_embedding:
        choices.append((best_embedding, "best frozen TimesFM readout"))
    replicated = complete_seed_families(eligible_by_kind["tcn"])
    competitive = [rows for rows in replicated if best_physical is None or mean(map(dev_value, rows)) <= dev_value(best_physical) + 0.15]
    if competitive:
        family = min(competitive, key=lambda rows: mean(map(dev_value, rows)))
        choices.append((middle_seed(family), "median-development seed of a competitive, replicated causal TCN family"))
    random_physical = [row for row in learners if row["config"]["hidden"] and not row["config"]["embedding"]]
    repeated_ridge = complete_seed_families(random_physical)
    if repeated_ridge:
        observed_control = next((row for row in random_physical if row["round_id"] == "R00193"), None)
        for target, label in ((min(repeated_ridge, key=lambda rows: mean(map(dev_value, rows))), "best replicated random-feature ridge"),
                              (next((rows for rows in repeated_ridge if observed_control in rows), None),
                               "original leading R00193 random-feature ridge")):
            if target is None:
                continue
            choices.append((middle_seed(target), f"median-development seed of the {label} family"))
            choices.append((min(target, key=dev_value), f"best individual seed of the {label} family"))
    initial_legacy = [row for row in eligible_by_kind["legacy_head"] if row["batch"] in
                      {"legacy-causal-head-screen", "legacy-causal-head-seed-repeat"}]
    legacy_families = complete_seed_families(initial_legacy)
    if legacy_families:
        ranked_legacy = sorted(legacy_families, key=lambda group: mean(map(dev_value, group)))
        cutoff = mean(map(dev_value, ranked_legacy[0])) + 0.15
        for group in ranked_legacy[:2]:
            average = mean(map(dev_value, group))
            if average <= cutoff:
                choices.append((middle_seed(group), f"median seed of an original causal physical head family averaging {average:.6f} deg"))
    choices.extend(joint_shortlist(items, pool))
    recurrent_report = BATCH / "stream-gru-development-summary.json"
    if recurrent_report.is_file():
        recurrent = {row["round_id"]: row for row in pool if row["config"]["kind"] == "stream_gru"}
        for result in read_json(recurrent_report).get("r5_approved", []):
            if result["round_id"] in recurrent:
                choices.append((recurrent[result["round_id"]], "replicated correctly segmented recurrent residual with joint memory benefit"))
    gated = [row for row in eligible_by_kind["hybrid"] if row["config"]["mode"] == "gated"]
    if gated:
        choices.append((gated[0], "best development error among default-fast causal gates"))
        nearby = [row for row in gated if dev_value(row) <= dev_value(gated[0]) + 0.1]
        conservative = min(nearby, key=lambda row: (row["dev_metrics"]["device"]["worst_delta_deg"], dev_value(row)))
        choices.append((conservative, "best worst-record delta among competitive default-fast causal gates"))
        observed_fallback = [row for row in gated if read_json(BATCH / row["result"])["mean_default_fallback_fraction"] >= 0.01]
        if observed_fallback:
            choices.append((observed_fallback[0], "best causal gate with a development macro-average full-fallback fraction of at least 1% across recordings"))
    ensemble = sorted((row for row in pool if row["config"]["kind"] == "ensemble"),
                      key=lambda row: (row["dev_metrics"]["joint_holdout"]["macro_rmse_deg"], dev_value(row)))
    for row in ensemble:
        choices.append((row, "predeclared equal-weight ensemble met joint-gain and single-axis safety criteria"))
    bias = [row for row in pool if row["config"]["kind"] == "bias_physical_head"]
    for row in bias:
        choices.append((row, "record-local causal accelerometer offset passed three matched development seeds"))
    invariant = eligible_by_kind["equivariant"]
    if invariant and (best_physical is None or dev_value(invariant[0]) <= dev_value(best_physical) + 0.15):
        choices.append((invariant[0], "competitive invariant physical correction"))
    by_round = {}
    for row, reason in choices:
        key = row["round_id"]
        if key not in by_round:
            by_round[key] = {"round_id": key, "kind": row["config"]["kind"],
                             "development_rmse_deg": dev_value(row), "reasons": []}
        by_round[key]["reasons"].append(reason)
    return list(by_round.values())
