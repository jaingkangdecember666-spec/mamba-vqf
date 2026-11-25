"""Measure how much a frozen correction changes when the same external IMU window is re-expressed."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from replay_native import ROOT, canonical_frame, external_case, features, read, weights


def angle(a, b):
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), (a * b).sum(axis=-1)))


@torch.inference_mode()
def estimate(model, x, intrinsic=False):
    head, mean, std, cap = model
    model_input = x
    if intrinsic:
        model_input, axes, valid = canonical_frame(x)
    raw = head(((torch.from_numpy(model_input) - mean) / std).clamp(-12, 12)).numpy().astype(float) * 0.1
    if intrinsic:
        raw = np.einsum("nij,nj->ni", axes, raw)
        raw[~valid] = 0.
    up = x[:, -1, 6:9].astype(float)
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    delta = raw - (raw * up).sum(axis=1, keepdims=True) * up
    delta *= np.minimum(1, np.tan(np.deg2rad(cap)) / np.maximum(1e-8, np.linalg.norm(delta, axis=1, keepdims=True)))
    tip = up + delta
    return tip / np.linalg.norm(tip, axis=1, keepdims=True), up


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--canonical', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    model = weights(args.checkpoint, 'mamba_v1')
    transforms = list(Rotation.from_rotvec(np.eye(3) * np.pi / 2))
    transforms.append(Rotation.from_rotvec(np.deg2rad(63) * np.array([1, 2, -1]) / np.sqrt(6)))
    records = []
    for record in read(ROOT / 'external-split.json')['records']:
        if record['split'] != 'validation':
            continue
        imu, hz, _, valid = external_case(record)
        _, windows, _, release = features(imu, hz)
        eligible = np.flatnonzero(valid[release])
        take = eligible[np.linspace(0, len(eligible) - 1, 32, dtype=int)]
        x = windows[take]
        base, up = estimate(model, x, args.canonical)
        errors = []
        for rot in transforms:
            moved = x.copy()
            for start in (0, 3, 6, 9):
                group = x[:, :, start:start + 3]
                moved[:, :, start:start + 3] = rot.apply(group.reshape(-1, 3)).reshape(group.shape)
            transformed, _ = estimate(model, moved, args.canonical)
            errors.extend(angle(base, rot.inv().apply(transformed)).tolist())
        records.append({'domain': record['domain'], 'sequence': record['name'], 'angles_deg': errors,
                        'original_correction_deg': angle(base, up).tolist()})
    error = np.concatenate([r['angles_deg'] for r in records])
    strength = np.concatenate([r['original_correction_deg'] for r in records])
    out = {'checkpoint': str(args.checkpoint), 'label': args.label, 'records': len(records), 'windows': len(strength),
           'rotated_comparisons': len(error), 'frame_change_rms_deg': float(np.mean(error ** 2) ** 0.5),
           'frame_change_p95_deg': float(np.quantile(error, .95)), 'correction_rms_deg': float(np.mean(strength ** 2) ** 0.5)}
    print(json.dumps(out))
    (ROOT / 'review' / ('axis_response_' + args.label + '.json')).write_text(
        json.dumps({**out, 'sequences': records}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
