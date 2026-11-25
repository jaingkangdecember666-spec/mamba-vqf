"""Estimate accelerometer bias causally from directionally diverse stationary blocks."""

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Settings:
    block_s: float = 1.
    max_gyro_rad_s: float = .05
    max_acc_deviation_m_s2: float = .10
    minimum_acc_norm_m_s2: float = 8.
    maximum_acc_norm_m_s2: float = 12.
    direction_separation_deg: float = 30.
    minimum_blocks_in_direction: int = 3
    minimum_directions: int = 4
    minimum_centered_singular_m_s2: float = .75
    minimum_singular_ratio: float = .15
    minimum_radius_m_s2: float = 8.8
    maximum_radius_m_s2: float = 10.8
    maximum_sphere_residual_m_s2: float = .20
    maximum_center_norm_m_s2: float = 1.5
    smoothing: float = .20
    maximum_step_m_s2: float = .025


def sphere(centers, cfg):
    mean = centers.mean(axis=0)
    centered = centers - mean
    singular = np.linalg.svd(centered, compute_uv=False)
    scale = float(singular[-1] / np.sqrt(len(centers)))
    ratio = float(singular[-1] / max(singular[0], 1e-15))
    geometry = {"smallest_centered_singular_m_s2": scale, "smallest_to_largest_singular_ratio": ratio}
    if scale < cfg.minimum_centered_singular_m_s2 or ratio < cfg.minimum_singular_ratio:
        return None, "ill_conditioned_directions", geometry
    power = np.sum(centered**2, axis=1)
    offset = np.linalg.lstsq(2. * centered, power - power.mean(), rcond=None)[0]
    center = mean + offset
    distances = np.linalg.norm(centers - center, axis=1)
    radius = float(np.sqrt(np.mean(distances**2)))
    residual = float(np.sqrt(np.mean((distances - radius)**2)))
    geometry.update({"sphere_radius_m_s2": radius, "sphere_residual_rms_m_s2": residual,
                     "sphere_center_xyz_m_s2": center.tolist()})
    if not cfg.minimum_radius_m_s2 <= radius <= cfg.maximum_radius_m_s2:
        return None, "radius_out_of_range", geometry
    if np.linalg.norm(center) > cfg.maximum_center_norm_m_s2:
        return None, "center_exceeds_bound", geometry
    if residual > cfg.maximum_sphere_residual_m_s2:
        return None, "sphere_residual_exceeds_bound", geometry
    return center, "accepted", geometry


def estimate(raw, observed, dt, cfg=Settings()):
    """Return bias available at each native index, active flags and causal diagnostics.

    The signal is in m/s² and rad/s. Observation flags refer to real packets,
    including when the caller has forward-filled values for downstream replay.
    No reference, record identity or end-of-record statistics are used.
    """
    width = max(1, round(cfg.block_s / dt))
    n = len(raw)
    bias = np.zeros((n, 3), dtype=np.float64)
    active_mask = np.zeros(n, dtype=bool)
    current = np.zeros(3, dtype=np.float64)
    first = None
    accepted_blocks, fit_attempts = Counter(), Counter()
    geometry = last_good = None
    anchors, centers, counts = [], [], []
    changes = []
    similarity_limit = np.cos(np.deg2rad(cfg.direction_separation_deg))

    for start in range(0, n, width):
        end = min(start + width, n)
        bias[start:end] = current
        active_mask[start:end] = first is not None
        if end - start < width:
            break
        if not observed[start:end].all() or not np.isfinite(raw[start:end]).all():
            accepted_blocks["missing_or_nonfinite"] += 1
            continue
        accel, gyro = raw[start:end, :3], raw[start:end, 3:]
        if np.linalg.norm(gyro, axis=1).max() > cfg.max_gyro_rad_s:
            accepted_blocks["gyro_motion"] += 1
            continue
        acceleration = accel.mean(axis=0)
        size = np.linalg.norm(acceleration)
        if not cfg.minimum_acc_norm_m_s2 <= size <= cfg.maximum_acc_norm_m_s2:
            accepted_blocks["magnitude_out_of_range"] += 1
            continue
        if np.linalg.norm(accel - acceleration, axis=1).max() > cfg.max_acc_deviation_m_s2:
            accepted_blocks["acceleration_unstable"] += 1
            continue
        accepted_blocks["stationary_accepted"] += 1
        direction = acceleration / size
        if anchors:
            similarities = np.asarray(anchors) @ direction
            closest = int(np.argmax(similarities))
        if anchors and similarities[closest] >= similarity_limit:
            counts[closest] += 1
            centers[closest] += (acceleration - centers[closest]) / counts[closest]
        else:
            anchors.append(direction)
            centers.append(acceleration)
            counts.append(1)
        qualified = np.asarray(counts) >= cfg.minimum_blocks_in_direction
        if qualified.sum() < cfg.minimum_directions:
            fit_attempts["too_few_confirmed_directions"] += 1
            continue
        center, reason, geometry = sphere(np.asarray(centers)[qualified], cfg)
        fit_attempts[reason] += 1
        if center is None:
            continue
        change = cfg.smoothing * (center - current)
        step_size = np.linalg.norm(change)
        if step_size > cfg.maximum_step_m_s2:
            change *= cfg.maximum_step_m_s2 / step_size
        current += change
        last_good = geometry
        if first is None:
            first = end
        changes.append({"effective_from_native_index": end, "bias_xyz_m_s2": current.tolist()})

    status = "active" if active_mask.any() else "fit_at_record_end" if first is not None else "insufficient_or_rejected"
    diagnostics = {"status": status, "samples_per_block": width, "complete_blocks": n // width,
                   "block_outcomes": dict(accepted_blocks), "fit_outcomes": dict(fit_attempts),
                   "direction_clusters": len(counts), "confirmed_directions": sum(c >= cfg.minimum_blocks_in_direction for c in counts),
                   "blocks_per_direction": counts, "first_valid_estimate_next_index": first,
                   "first_active_native_index": first if status == "active" else None,
                   "active_native_frames": int(active_mask.sum()), "last_attempt_geometry": geometry,
                   "last_accepted_sphere": last_good, "accepted_update_count": len(changes),
                   "last_applied_bias_xyz_m_s2": bias[-1].tolist(), "last_applied_bias_norm_m_s2": float(np.linalg.norm(bias[-1]))}
    return bias, active_mask, diagnostics, changes


def synthetic_checks(cfg=Settings()):
    dt = .01
    width = round(cfg.block_s / dt)
    truth = np.array([.18, -.13, .09])
    axes = np.r_[np.eye(3), -np.eye(3)]
    levels = np.repeat(9.80665 * axes + truth, 12 * width, axis=0)
    signal = np.column_stack((levels, np.zeros_like(levels)))
    observed = np.ones(len(signal), dtype=bool)
    bias, active, info, changes = estimate(signal, observed, dt, cfg)
    one = np.tile(signal[0], (len(signal), 1))
    one_bias, one_active, one_info, _ = estimate(one, observed, dt, cfg)
    assert not one_active.any() and np.count_nonzero(one_bias) == 0 and one_info["confirmed_directions"] == 1
    first = info["first_active_native_index"]
    assert first is not None and first == 39 * width and np.all(bias[:first] == 0.)
    assert info["last_accepted_sphere"]["sphere_residual_rms_m_s2"] < 1e-10
    assert np.linalg.norm(bias[-1] - truth) < .002
    step = np.diff(np.r_[np.zeros((1, 3)), np.asarray([x["bias_xyz_m_s2"] for x in changes])], axis=0)
    assert np.linalg.norm(step, axis=1).max() <= cfg.maximum_step_m_s2 + 1e-12

    rotation, _ = np.linalg.qr(np.array([[1., -2., 3.], [2., 3., 1.], [-1., 1., 5.]]))
    if np.linalg.det(rotation) < 0.:
        rotation[:, 0] *= -1.
    changed = np.column_stack((signal[:, :3] @ rotation, signal[:, 3:] @ rotation))
    rotated_bias, rotated_active, _, _ = estimate(changed, observed, dt, cfg)
    np.testing.assert_array_equal(active, rotated_active)
    eq = float(np.max(np.linalg.norm(rotated_bias - bias @ rotation, axis=1)))
    assert eq < 1e-12

    stop = int(first + 3 * width + width // 2)
    suffix = signal.copy()
    suffix[stop:, :3] += np.array([1., -3., 2.])
    changed_bias, changed_active, _, _ = estimate(suffix, observed, dt, cfg)
    short_bias, short_active, _, _ = estimate(signal[:stop], observed[:stop], dt, cfg)
    np.testing.assert_array_equal(short_bias, bias[:stop])
    np.testing.assert_array_equal(changed_bias[:stop], bias[:stop])
    np.testing.assert_array_equal(short_active, active[:stop])
    np.testing.assert_array_equal(changed_active[:stop], active[:stop])
    return {"single_pose_active_frames": int(one_active.sum()), "diverse_first_active_native_index": first,
            "synthetic_final_bias_error_m_s2": float(np.linalg.norm(bias[-1] - truth)),
            "estimation_step_max_m_s2": float(np.linalg.norm(step, axis=1).max()),
            "rotation_determinant": float(np.linalg.det(rotation)), "rotation_covariance_max_error_m_s2": eq,
            "truncated_prefix_max_error_m_s2": 0., "changed_future_prefix_max_error_m_s2": 0.}
