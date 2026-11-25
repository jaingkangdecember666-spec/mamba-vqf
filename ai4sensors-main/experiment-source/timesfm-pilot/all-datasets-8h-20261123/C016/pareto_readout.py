"""Deterministic four-domain common-descent readout solver."""

import itertools
import json
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

DOMAIN_COUNT = 4
STEP_CAP = .02
ARMIJO = 1e-4
MAX_HALVINGS = 20
MAX_STEPS = 80
QP_TOL = 2e-11


def solve_qp(gradients):
    """Minimize ||sum_i alpha_i*g_i|| on the four-dimensional probability simplex."""
    gram = gradients @ gradients.T
    scale = float(np.diag(gram).max())
    q = gram / scale if scale else gram
    best_alpha, best_score = None, np.inf
    # Every minimum has a simplex face on which the equality-constrained KKT equations hold.
    for size in range(1, DOMAIN_COUNT + 1):
        for subset in itertools.combinations(range(DOMAIN_COUNT), size):
            ids = list(subset)
            mat = np.zeros((size + 1, size + 1), dtype=np.float64)
            mat[:size, :size] = q[np.ix_(ids, ids)]
            mat[:size, size] = -1.
            mat[size, :size] = 1.
            rhs = np.r_[np.zeros(size), 1.]
            solution, _, _, _ = np.linalg.lstsq(mat, rhs, rcond=None)
            if np.max(np.abs(mat @ solution - rhs)) > QP_TOL or solution[:size].min() < -QP_TOL:
                continue
            alpha = np.zeros(DOMAIN_COUNT, dtype=np.float64)
            alpha[ids] = solution[:size].clip(min=0.)
            alpha /= alpha.sum()
            objective = float(alpha @ q @ alpha)
            if objective < best_score:
                best_alpha, best_score = alpha, objective
    if best_alpha is None:
        raise ValueError("No feasible simplex QP solution")
    multiplier = q @ best_alpha - best_score
    residuals = {"sum_equality_abs": abs(float(best_alpha.sum()) - 1.),
                 "primal_nonnegative_violation": max(0., -float(best_alpha.min())),
                 "dual_nonnegative_violation_scaled": max(0., -float(multiplier.min())),
                 "complementarity_scaled_max_abs": float(np.max(np.abs(best_alpha * multiplier)))}
    max_residual = max(residuals.values())
    if max_residual > QP_TOL:
        raise ValueError(f"Simplex QP KKT residual {max_residual:.5g} exceeds {QP_TOL:g}")
    mixed = best_alpha @ gradients
    return mixed, {"coefficients": best_alpha.tolist(), "active_domains": np.flatnonzero(best_alpha > QP_TOL).tolist(),
                   "gradient_gram": gram.tolist(), "gram_scale": scale, "min_squared_gradient_norm": float(mixed @ mixed),
                   "scaled_dual_multipliers": multiplier.tolist(), "residuals": residuals, "max_scaled_kkt_residual": max_residual}


def domain_loss(theta, inputs, gate, weighted_indices, cap, gain):
    width = inputs.shape[1] - 6
    x, up, target = inputs.split((width, 3, 3), dim=1)
    weights = theta[:3 * width].reshape(3, width)
    bias = theta[3 * width:]
    delta = .1 * F.linear(x, weights, bias)
    delta = delta - (delta * up).sum(dim=1, keepdim=True) * up
    limit = torch.tensor(cap * torch.pi / 180, dtype=inputs.dtype, device=inputs.device).tan()
    limited = delta * (limit / delta.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp_max(1.)
    allowed = torch.where(gate[:, None], torch.zeros_like(limited), limited)
    errors = (F.normalize(up + gain * allowed, dim=1) - target).square().sum(dim=1)
    return torch.stack([(errors[idx] * weights).sum() for idx, weights in weighted_indices])


def optimize(initial, inputs, gate, weighted_indices, cap, gain, log_path, stamp, fingerprint):
    """One deterministic run, requiring each source to satisfy its own Armijo bound."""
    params = initial.detach().clone()
    with torch.no_grad():
        baseline = domain_loss(params, inputs, gate, weighted_indices, cap, gain)
    if not torch.isfinite(baseline).all() or torch.any(baseline <= 0.):
        raise ValueError("Invalid C015 starting domain loss")
    start, started_at = perf_counter(), stamp()
    accepted = 0
    finish = "accepted_step_limit"
    step_records = []
    with log_path.open("x", encoding="utf-8", newline="\n") as stream:
        while accepted < MAX_STEPS:
            parameter = params.detach().requires_grad_(True)
            raw_loss = domain_loss(parameter, inputs, gate, weighted_indices, cap, gain)
            relative = raw_loss / baseline
            individual = torch.stack([torch.autograd.grad(relative[j], parameter, retain_graph=j < DOMAIN_COUNT - 1)[0]
                                      for j in range(DOMAIN_COUNT)]).detach().numpy()
            objective = relative.detach().numpy()
            if not np.isfinite(individual).all() or not np.isfinite(objective).all():
                raise ValueError("Nonfinite full-batch source objective or gradient")
            mixture, qp = solve_qp(individual)
            mix_norm = float(np.linalg.norm(mixture))
            norms = np.linalg.norm(individual, axis=1)
            stationary_tol = float(np.finfo(np.float64).eps * 128. * max(norms))
            row = {"attempt": len(step_records) + 1, "accepted_before": accepted, "parameter_sha256_before": fingerprint(params.numpy()),
                   "absolute_loss_before": raw_loss.detach().tolist(), "relative_loss_before": objective.tolist(),
                   "relative_domain_gradients": individual.tolist(), "domain_gradient_norms": norms.tolist(),
                   "convex_combination": mixture.tolist(), "combination_norm": mix_norm,
                   "numerical_stationarity_tolerance": stationary_tol, "qp": qp, "trials": [],
                   "accepted_halvings": None}
            if not mix_norm > stationary_tol:
                finish = "zero_common_descent_within_float64_precision"
            else:
                direction = -mixture / mix_norm
                slopes = individual @ direction
                row["directional_derivative_by_domain"] = slopes.tolist()
                if float(slopes.max()) >= 0.:
                    finish = "no_strict_common_direction_resolved_in_float64"
                else:
                    delta = torch.from_numpy(direction)
                    for halving in range(MAX_HALVINGS + 1):
                        length = STEP_CAP * 2.**(-halving)
                        trial = params + length * delta
                        with torch.no_grad():
                            absolute = domain_loss(trial, inputs, gate, weighted_indices, cap, gain)
                            measured = (absolute / baseline).numpy()
                        rhs = objective + (ARMIJO * length) * slopes
                        conditions = np.isfinite(measured) & (measured <= rhs) & (measured < objective)
                        row["trials"].append({"halvings": halving, "requested_l2": length,
                                              "actual_l2": float((trial - params).norm()),
                                              "absolute_loss": absolute.tolist(), "relative_loss": measured.tolist(),
                                              "armijo_upper_by_domain": rhs.tolist(), "accepted_by_domain": conditions.tolist()})
                        if conditions.all():
                            params = trial
                            accepted += 1
                            row["accepted_halvings"] = halving
                            row["accepted_step"] = accepted
                            row["parameter_sha256_after"] = fingerprint(params.numpy())
                            row["absolute_loss_after"] = absolute.tolist()
                            row["relative_loss_after"] = measured.tolist()
                            break
                    else:
                        finish = "all_domain_armijo_exhausted_20_halvings"
            if row["accepted_halvings"] is None:
                row["stop_reason"] = finish
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            step_records.append(row)
            if row["accepted_halvings"] is None:
                break
            if accepted == 1 or accepted == MAX_STEPS or accepted % 10 == 0:
                print("accepted", accepted, "halvings", row["accepted_halvings"],
                      "relative losses", row["relative_loss_after"], flush=True)
    with torch.no_grad():
        final = domain_loss(params, inputs, gate, weighted_indices, cap, gain)
    return params, {"started_at": started_at, "completed_at": stamp(), "seconds": perf_counter() - start,
                     "accepted_steps": accepted, "attempts": len(step_records), "stop_reason": finish,
                     "initial_loss": baseline.tolist(), "final_loss": final.tolist(),
                     "final_relative_loss": (final / baseline).tolist(),
                     "parameter_change_l2": float((params - initial).norm())}, step_records
