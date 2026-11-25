"""Per-release intrinsic frame for the existing 64-block Mamba residual head."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_pair as original


ACC_FLOOR = 0.01
GYRO_FLOOR = 0.001
ACC, GYRO, DEGENERATE = 1, 2, 0


@dataclass
class Frame:
    # Columns are the intrinsic x, y, z axes expressed in the original body frame.
    basis: np.ndarray
    source: np.ndarray
    anchor: np.ndarray

    @property
    def valid(self):
        return self.source != DEGENERATE


def transverse(vectors, z):
    return vectors - np.einsum("btc,bc->bt", vectors, z)[..., None] * z[:, None]


def build_frame(histories):
    """Choose one right-handed basis from each complete release window."""
    if histories.ndim != 3 or histories.shape[1:] != (original.WINDOW, 13):
        raise ValueError("Expected histories with shape [batch, 64, 13]")
    values = np.asarray(histories, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("A release window contains nonfinite features")
    up = values[:, -1, 6:9]
    lengths = np.linalg.norm(up, axis=-1)
    if np.any(lengths <= 0):
        raise ValueError("A release window has no VQF up direction")
    z = up / lengths[:, None]
    acc = transverse(values[..., :3], z)
    gyro = transverse(values[..., 3:6] - values[..., 9:12], z)
    rows = np.arange(len(values))
    acc_norm = np.linalg.norm(acc, axis=-1)
    gyro_norm = np.linalg.norm(gyro, axis=-1)
    acc_anchor = np.argmax(acc_norm, axis=1)
    gyro_anchor = np.argmax(gyro_norm, axis=1)
    use_acc = acc_norm[rows, acc_anchor] >= ACC_FLOOR
    use_gyro = (~use_acc) & (gyro_norm[rows, gyro_anchor] >= GYRO_FLOOR)
    source = np.select((use_acc, use_gyro), (ACC, GYRO), default=DEGENERATE).astype(np.uint8)
    valid = source != DEGENERATE
    anchor = np.where(use_acc, acc_anchor, np.where(use_gyro, gyro_anchor, -1))
    choice = np.where(use_acc[:, None], acc[rows, acc_anchor], gyro[rows, gyro_anchor])
    x = np.zeros_like(z)
    y = np.zeros_like(z)
    x[valid] = choice[valid] / np.linalg.norm(choice[valid], axis=1, keepdims=True)
    y[valid] = np.cross(z[valid], x[valid])
    y[valid] /= np.linalg.norm(y[valid], axis=1, keepdims=True)
    x[valid] = np.cross(y[valid], z[valid])
    basis = np.stack((x, y, z), axis=-1)
    basis[~valid] = 0.
    return Frame(basis, source, anchor)


def transform_inputs(histories, frame):
    vectors = histories[..., :12].reshape(len(histories), original.WINDOW, 4, 3)
    rotated = np.einsum("ntvi,nij->ntvj", vectors, frame.basis)
    return np.concatenate((rotated.reshape(len(histories), original.WINDOW, 12), histories[..., 12:]), axis=-1
                          ).astype(np.float32)


def transform_directions(directions, frame):
    return np.einsum("ni,nij->nj", directions, frame.basis).astype(np.float32)


@torch.inference_mode()
def predict_delta(model, histories, mean, std, *, frame=None):
    """Return the body-frame tangent correction at release, and zero on degenerate windows.

    Pass the unchanged 13-channel histories used by Batch 3 and the training-only
    moments from this checkpoint. The caller retains its existing release/hold logic.
    """
    if frame is None:
        frame = build_frame(histories)
    valid = frame.valid
    body_delta = np.zeros((len(histories), 3), dtype=np.float64)
    if valid.any():
        selected = Frame(frame.basis[valid], frame.source[valid], frame.anchor[valid])
        canonical = torch.from_numpy(transform_inputs(histories[valid], selected)).to(mean.device)
        tangent = original.tangent(model(original.normalized(canonical, mean, std)),
                                   canonical[:, -1, 6:9], original.CAP).cpu().numpy()
        body_delta[valid] = np.einsum("ni,nji->nj", tangent, selected.basis)
    return body_delta
