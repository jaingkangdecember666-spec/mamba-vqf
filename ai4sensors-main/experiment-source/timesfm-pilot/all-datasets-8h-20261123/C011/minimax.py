"""Bounded scalar minimax for capped unit-direction chord losses."""

import heapq

import numpy as np
from scipy.optimize import minimize_scalar

TIE_TOL = 1e-12
CERTIFICATE_TOL = 1e-10
GRID_INTERVALS = 512


class Curve:
    """One domain, with equal weight for each record and equal weight for its sampled draws."""

    def __init__(self, base, ema, target, weights, cap):
        self.a = np.einsum("ij,ij->i", base, target)
        self.b = np.einsum("ij,ij->i", ema, target)
        self.c = np.einsum("ij,ij->i", base, base)
        self.d = np.einsum("ij,ij->i", base, ema)
        self.e = np.einsum("ij,ij->i", ema, ema)
        self.target = 1. + np.einsum("ij,ij->i", target, target)
        self.cap_k = np.divide(cap, np.sqrt(self.e), out=np.full(len(self.e), np.inf), where=self.e > 0)
        self.p = self.b * self.c - self.a * self.d
        self.q = self.b * self.d - self.a * self.e
        self.turn = np.divide(-self.p, self.q, out=np.zeros(len(self.q)), where=self.q != 0.)
        self.base_loss = self.loss(np.zeros(len(self.a)))
        self.base_mean = float(weights @ self.base_loss)
        if not np.isfinite(self.base_mean) or self.base_mean <= 0. or not np.all(ema[:, 2] == 0.):
            raise ValueError("Invalid training directions for scalar minimax")
        self.w = weights / self.base_mean
        # Squared norm of base + k*ema stays above its unchanged vertical component squared.
        z = np.abs(base[:, 2])
        if np.any(z < .99):
            raise ValueError("Unnormalized C006 direction no longer has unit-scale vertical component")
        kmax = np.minimum(self.cap_k, 1.)
        max_num = np.maximum(np.abs(self.p), np.abs(self.p + self.q * kmax))
        max_d = np.maximum(np.abs(self.d), np.abs(self.d + self.e * kmax))
        self.curvature = 2. * np.abs(self.q) / z**3 + 6. * max_num * max_d / z**5
        self.curvature_mean = float(self.w @ self.curvature)

    def loss(self, clipped_k, mask=slice(None)):
        a, b = self.a[mask], self.b[mask]
        c, d, e = self.c[mask], self.d[mask], self.e[mask]
        return self.target[mask] - 2. * (a + b * clipped_k) / np.sqrt(c + clipped_k * (2. * d + e * clipped_k))

    def ratio(self, k):
        return 1. + float(self.w @ (self.loss(np.minimum(k, self.cap_k)) - self.base_loss))

    def interval_lower(self, low, high, f_low, f_high):
        crossing = (low < self.cap_k) & (self.cap_k < high)
        # Unclipped terms are bounded below by their chords minus the curvature error.
        # For a term crossing its cap, subtract the endpoint contributions and add
        # its exact interval minimum, including the stationary point if present.
        if crossing.any():
            lo = self.loss(low, crossing)
            clipped_hi = self.cap_k[crossing]
            hi = self.loss(clipped_hi, crossing)
            turn = np.clip(self.turn[crossing], low, clipped_hi)
            best = np.minimum(np.minimum(lo, hi), self.loss(turn, crossing))
            weight = self.w[crossing]
            f_low -= float(weight @ (lo - best))
            f_high -= float(weight @ (hi - best))
        return min(f_low, f_high) - self.curvature_mean * (high - low)**2 / 8. - 2e-12


def select_scalar(curves):
    """Return the smallest numerically tied minimizing coefficient on [0, 1]."""
    trials = {}
    best_value = np.inf

    def evaluate(k):
        nonlocal best_value
        k = float(k)
        if k not in trials:
            trials[k] = np.array([curve.ratio(k) for curve in curves])
            best_value = min(best_value, float(trials[k].max()))
        return trials[k]

    def objective(k):
        return float(evaluate(k).max())

    grid = np.linspace(0., 1., GRID_INTERVALS + 1)
    grid_y = np.array([objective(k) for k in grid])
    for j in range(1, len(grid) - 1):
        a, b, c = grid_y[j - 1:j + 2]
        if (b <= a and b < c) or (b < a and b <= c):
            result = minimize_scalar(objective, bounds=(grid[j - 1], grid[j + 1]), method="bounded",
                                     options={"xatol": 2e-13})
            evaluate(result.x)

    def interval(a, b):
        f_a, f_b = evaluate(a), evaluate(b)
        bound = max(curve.interval_lower(a, b, x, y) for curve, x, y in zip(curves, f_a, f_b, strict=True))
        return float(bound), float(a), float(b)

    queue = [interval(a, b) for a, b in zip(grid[:-1], grid[1:], strict=True)]
    # A globally flat optimum can start exactly when the limiting contribution clips.
    # Include the cap locations in every interval that could still contain an optimum.
    caps = np.unique(np.concatenate([curve.cap_k for curve in curves]))
    for bound, a, b in queue:
        if bound <= best_value + TIE_TOL:
            lo, hi = np.searchsorted(caps, (a, b))
            for k in caps[lo:hi]:
                evaluate(k)
    heapq.heapify(queue)
    subdivided = 0
    while queue[0][0] < best_value - CERTIFICATE_TOL:
        _, a, b = heapq.heappop(queue)
        mid = (a + b) * .5
        if not a < mid < b or subdivided >= 100_000:
            raise ValueError("Scalar minimax interval certificate did not converge")
        heapq.heappush(queue, interval(a, mid))
        heapq.heappush(queue, interval(mid, b))
        subdivided += 1
    chosen = min(k for k, ratios in trials.items() if float(ratios.max()) <= best_value + TIE_TOL)
    chosen_y = objective(chosen)
    return chosen, {"k": chosen, "objective": chosen_y, "best_evaluated_objective": best_value,
                    "lower_bound": queue[0][0], "certificate_gap": max(0., chosen_y - queue[0][0]),
                    "certificate_tolerance": CERTIFICATE_TOL, "tie_tolerance": TIE_TOL,
                    "grid_intervals": GRID_INTERVALS, "evaluations": len(trials), "subdivided_intervals": subdivided,
                    "objective_at_zero": objective(0.), "objective_at_one": objective(1.)}
