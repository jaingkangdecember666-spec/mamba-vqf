"""Fit the C038 response router from the archived external training slots."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import softmax


HERE = Path(__file__).resolve().parent
WIDTH = 24
PARAMETERS = 2 * WIDTH + 2


def sha256(file):
    with file.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_c023(panel, reference, denominators, original):
    # The first 96 slots of each optimizer step are the original C023 schedule.
    def original_slots(key):
        values = panel[key][:, :96]
        return values.reshape((-1, *values.shape[2:]))

    z, center, fast, slow, truth, enabled = (
        torch.from_numpy(original_slots(key)) for key in ("z", "center", "fast", "slow", "truth", "enabled")
    )
    domains = original_slots("domain")
    weights = original_slots("record_equal_weights")
    normalization = torch.from_numpy(weights / np.array([denominators[str(name)] for name in domains]))
    prior = reference["log_p0"]
    model = torch.nn.Linear(WIDTH, 2, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=original["lr"], weight_decay=original["weight_decay"], foreach=False)
    batch = original["batch_size"]
    for i in range(original["steps"]):
        rows = slice(i * batch, (i + 1) * batch)
        optimizer.zero_grad(set_to_none=True)
        logits = torch.cat((model(z[rows]) + prior[:2], prior[2].expand(batch, 1)), dim=1)
        probs = logits.softmax(dim=1)
        adjust = probs[:, :1] * fast[rows] + probs[:, 1:2] * slow[rows]
        direction = center[rows] + adjust.masked_fill(~enabled[rows, None], 0.)
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        errors = (direction - truth[rows]).square().sum(-1)
        (40. * (errors * normalization[rows]).sum()).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), original["clip_global_norm"])
        optimizer.step()
    return np.r_[model.weight.detach().numpy().ravel(), model.bias.detach().numpy()]


class SourceRisk:
    """Equal-record tilt RMSE and analytic gradients over the five external sources."""
    def __init__(self, panel, prior, start):
        def flat(key):
            values = panel[key]
            return values.reshape((-1, *values.shape[2:]))

        self.z, self.center, self.fast, self.slow, self.truth, self.enabled = (
            flat(key) for key in ("z", "center", "fast", "slow", "truth", "enabled")
        )
        self.prior = prior
        self.ids, domains = flat("source_record_ordinal"), flat("domain")
        self.counts = np.bincount(self.ids)
        self.members = [np.unique(self.ids[domains == name]) for name in sorted(set(domains))]
        assert len(self.counts) == 141 and len(self.ids) == 15680 and len(self.members) == 5
        self.reference, _ = self.evaluate(start)

    def prediction(self, params):
        logits = np.broadcast_to(np.log(self.prior), (len(self.z), 3)).copy()
        logits[:, :2] += self.z @ params[:2*WIDTH].reshape(2, WIDTH).T + params[2*WIDTH:]
        probs = softmax(logits, axis=1)
        mix = (probs[:, :1] * self.fast + probs[:, 1:2] * self.slow) * self.enabled[:, None]
        direction = self.center + mix
        length = np.linalg.norm(direction, axis=1, keepdims=True)
        return direction / length, length, mix, probs

    def evaluate(self, params):
        direction, length, mix, probs = self.prediction(params)
        dot = np.sum(direction * self.truth, axis=1)
        angle = np.arctan2(np.linalg.norm(np.cross(direction, self.truth), axis=1), dot)
        rmse = np.sqrt(np.bincount(self.ids, weights=angle**2) / self.counts)
        means = np.array([rmse[group].mean() for group in self.members])
        scale = np.divide(angle, np.sin(angle), out=np.ones_like(angle), where=angle > 1e-9)
        d_direction = -2. * scale[:, None] * (self.truth - direction * dot[:, None]) / length
        gate = np.column_stack([
            probs[:, i] * np.sum(d_direction * (alternative - mix), axis=1) * self.enabled
            for i, alternative in enumerate((self.fast, self.slow))
        ])
        features = np.column_stack((self.z, np.ones(len(self.z))))
        derivatives = []
        for group in self.members:
            mask = np.isin(self.ids, group)
            ids = self.ids[mask]
            weight = 1. / (2. * self.counts[ids] * rmse[ids] * len(group))
            grad = (gate[mask] * weight[:, None]).T @ features[mask]
            derivatives.append(np.r_[grad[:, :-1].ravel(), grad[:, -1]])
        return np.degrees(means), np.degrees(np.vstack(derivatives))

    def relative(self, params):
        loss, derivatives = self.evaluate(params)
        return loss / self.reference, derivatives / self.reference[:, None]


def fit_c035(risk, start, radius, options):
    def constraints(point):
        margin = point[:PARAMETERS] - start
        values, jacobian = risk.relative(point[:PARAMETERS])
        slacks = np.r_[point[-1] - values, radius**2 - float(margin @ margin)]
        gradients = np.vstack((np.column_stack((-jacobian, np.ones(len(values)))), np.r_[-2.*margin, 0.]))
        return slacks, gradients

    outcome = minimize(lambda p: float(p[-1]), np.r_[start, 1.],
                       jac=lambda _: np.r_[np.zeros(PARAMETERS), 1.], method="SLSQP",
                       constraints=({"type":"ineq", "fun":lambda p: constraints(p)[0],
                                     "jac":lambda p: constraints(p)[1]},), options=options)
    if not outcome.success or min(constraints(outcome.x)[0]) < -1e-9:
        raise RuntimeError(f"External source fit failed: {outcome.message}")
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "trained")
    args = parser.parse_args()
    spec = json.loads((HERE / "recipe.json").read_bytes())
    for name, checksum in spec["sha256"].items():
        if sha256(HERE / name) != checksum:
            raise ValueError(f"Input hash mismatch: {name}")
    torch.set_num_threads(2)
    frozen = torch.load(HERE / "weights/c038-router.pt", map_location="cpu", weights_only=True)
    with np.load(HERE / "external-training.npz", allow_pickle=False) as archive:
        panel = {name: archive[name] for name in archive.files}
    start = fit_c023(panel, frozen, spec["c023_chord_denominators"], spec["c023_optimizer"])
    risk = SourceRisk(panel, frozen["p0"].numpy(), start)
    result = fit_c035(risk, start, spec["c035_radius"], spec["c035_solver_options"])
    params = (start + result.x[:PARAMETERS]) / 2.
    archived = np.r_[frozen["state_dict"]["weight"].numpy().ravel(), frozen["state_dict"]["bias"].numpy()]
    maximum = float(np.max(np.abs(params - archived)))
    model_direction = risk.prediction(params)[0]
    reference_direction = risk.prediction(archived)[0]
    angle = np.degrees(np.arctan2(np.linalg.norm(np.cross(model_direction, reference_direction), axis=1),
                                 np.sum(model_direction * reference_direction, axis=1)))
    if maximum > 1e-5 or float(angle.max()) > 1e-4:
        raise RuntimeError(f"Refitted C038 differs from archive: parameters {maximum:g}, angles {angle.max():g} degrees")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "router.pt"
    receipt = args.output / "verification.json"
    if output.exists() or receipt.exists():
        raise FileExistsError("Choose a new --output folder to keep the previous training run.")
    fitted = copy.copy(frozen)
    fitted["state_dict"] = {"weight":torch.from_numpy(params[:2*WIDTH].reshape(2, WIDTH)),
                            "bias":torch.from_numpy(params[2*WIDTH:])}
    torch.save(fitted, output)
    evidence = {"c023_steps":spec["c023_optimizer"]["steps"], "c035_iterations":int(result.nit),
                "external_source_records":len(risk.counts), "external_slots":len(risk.ids),
                "parameter_max_abs_difference":maximum, "prediction_max_angle_deg":float(angle.max()),
                "checkpoint_sha256":sha256(output)}
    receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
