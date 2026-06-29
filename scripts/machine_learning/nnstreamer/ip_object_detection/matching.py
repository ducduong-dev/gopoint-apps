#!/usr/bin/env python3

"""
SPDX-License-Identifier: MIT

Association helpers (IoU cost + linear assignment) for ByteTrack.

Vendored from the ByteTrack project:
    https://github.com/ifzhang/ByteTrack  (Copyright (c) 2021 Yifu Zhang, MIT)

Simplified to depend only on numpy, with an optional fast path: if SciPy is
available we use scipy.optimize.linear_sum_assignment for the Hungarian solve;
otherwise we fall back to a pure-numpy greedy assignment. This keeps the tracker
working on a minimal i.MX BSP where SciPy may not be installed (the Kalman
filter still needs SciPy, but the launcher degrades to a no-Kalman path there).
"""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on target image
    _HAVE_SCIPY = False


def linear_assignment(cost_matrix, thresh):
    """Solve assignment, returning (matches, unmatched_a, unmatched_b).

    matches      -- (K, 2) array of matched (row, col) index pairs;
    unmatched_a  -- rows (tracks) with no match;
    unmatched_b  -- cols (detections) with no match.
    Pairs whose cost exceeds ``thresh`` are rejected.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )

    if _HAVE_SCIPY:
        rows, cols = _assign_scipy(cost_matrix, thresh)
    else:
        rows, cols = _assign_greedy(cost_matrix, thresh)

    matches = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(rows, cols):
        if cost_matrix[r, c] <= thresh:
            matches.append([r, c])
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_a = tuple(r for r in range(cost_matrix.shape[0]) if r not in matched_rows)
    unmatched_b = tuple(c for c in range(cost_matrix.shape[1]) if c not in matched_cols)
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    return matches, unmatched_a, unmatched_b


def _assign_scipy(cost_matrix, thresh):
    """Optimal Hungarian assignment via SciPy."""
    # Cap costs above the gate so the solver never prefers an invalid pair.
    capped = np.where(cost_matrix > thresh, thresh + 1e-4, cost_matrix)
    return linear_sum_assignment(capped)


def _assign_greedy(cost_matrix, thresh):
    """Greedy fallback: repeatedly take the lowest remaining cost pair."""
    rows, cols = [], []
    used_r, used_c = set(), set()
    flat = np.argsort(cost_matrix, axis=None)
    n_cols = cost_matrix.shape[1]
    for idx in flat:
        r, c = divmod(int(idx), n_cols)
        if r in used_r or c in used_c:
            continue
        if cost_matrix[r, c] > thresh:
            break
        rows.append(r)
        cols.append(c)
        used_r.add(r)
        used_c.add(c)
    return np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)


def ious(atlbrs, btlbrs):
    """Pairwise IoU between two lists of boxes in (x1, y1, x2, y2) form."""
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float32)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float32)
    if atlbrs.size == 0 or btlbrs.size == 0:
        return np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)

    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    lt = np.maximum(atlbrs[:, None, :2], btlbrs[None, :, :2])
    rb = np.minimum(atlbrs[:, None, 2:], btlbrs[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def iou_distance(atracks, btracks):
    """IoU cost matrix (1 - IoU) between two track/detection lists."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    return 1.0 - ious(atlbrs, btlbrs)


def fuse_score(cost_matrix, detections):
    """Fuse detection confidence into the IoU cost (ByteTrack fuse_score)."""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1.0 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1.0 - fuse_sim
