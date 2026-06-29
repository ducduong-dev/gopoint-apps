#!/usr/bin/env python3

"""
SPDX-License-Identifier: MIT

ByteTrack multi-object tracker.

Vendored and lightly trimmed from the ByteTrack project:
    https://github.com/ifzhang/ByteTrack  (Copyright (c) 2021 Yifu Zhang, MIT)
    "ByteTrack: Multi-Object Tracking by Associating Every Detection Box",
    Zhang et al., ECCV 2022.

Changes from upstream:
  * numpy/scipy only -- the lap/cython_bbox dependencies are dropped in favor of
    matching.linear_assignment (SciPy Hungarian with a numpy greedy fallback);
  * each track carries a class id / label so multi-class YOLOv8 detections keep
    their category through association;
  * the argparse-style ``args`` bag is replaced by explicit constructor params.

The tracker is detector-agnostic: feed update() an (N, 5) array of
[x1, y1, x2, y2, score] boxes plus a parallel list of class ids, and it returns
the list of currently active STrack objects, each with a stable track_id.
"""

import numpy as np

from kalman_filter import KalmanFilter
import matching


class TrackState:
    """Lifecycle states of a single track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Bookkeeping shared by all tracks (id allocation, state, frame counts)."""

    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    start_frame = 0
    frame_id = 0
    time_since_update = 0

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id():
        BaseTrack._count = 0

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    """A single tracked object, with a shared-class-level Kalman filter."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, class_id):
        # Wait to allocate id/state until activate().
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.class_id = int(class_id)
        self.tracklen = 0

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance
        )

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) == 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a brand-new track."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh)
        )

        self.tracklen = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-bind a lost track to a fresh detection."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklen = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id

    def update(self, new_track, frame_id):
        """Correct an active track with its matched detection."""
        self.frame_id = frame_id
        self.tracklen += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.class_id = new_track.class_id

    @property
    def tlwh(self):
        """Current box as (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Current box as (x1, y1, x2, y2)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert (tl x, tl y, w, h) -> (center x, center y, aspect, h)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def __repr__(self):
        return f"OT_{self.track_id}_({self.start_frame}-{self.frame_id})"


class BYTETracker:
    """Two-stage association tracker (high-score then low-score detections)."""

    def __init__(
        self,
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    ):
        self.tracked_stracks = []  # active tracks
        self.lost_stracks = []  # temporarily lost tracks
        self.removed_stracks = []

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        # Low-score detections above this are used in the second association.
        self.det_thresh = track_thresh + 0.1
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

    def update(self, boxes, scores, class_ids):
        """Advance the tracker by one frame.

        boxes      -- (N, 4) array of [x1, y1, x2, y2] in pixels;
        scores     -- (N,) confidences;
        class_ids  -- (N,) integer class ids.
        Returns the list of active STrack objects for this frame.
        """
        self.frame_id += 1
        activated = []
        refind = []
        lost = []
        removed = []

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        class_ids = np.asarray(class_ids).reshape(-1)

        remain = scores > self.track_thresh
        low = (scores > 0.1) & (scores <= self.track_thresh)

        boxes_high, scores_high, cls_high = (
            boxes[remain],
            scores[remain],
            class_ids[remain],
        )
        boxes_low, scores_low, cls_low = boxes[low], scores[low], class_ids[low]

        detections = [
            STrack(STrack.tlwh_from_tlbr(b), s, c)
            for b, s, c in zip(boxes_high, scores_high, cls_high)
        ]

        # Split current tracks into confirmed vs. not-yet-confirmed.
        unconfirmed = []
        tracked = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked.append(track)

        # --- First association: high-score detections vs. tracked + lost. ---
        strack_pool = join_stracks(tracked, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching.iou_distance(strack_pool, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itrack, idet in matches:
            track = strack_pool[itrack]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # --- Second association: low-score detections vs. remaining tracks. ---
        detections_low = [
            STrack(STrack.tlwh_from_tlbr(b), s, c)
            for b, s, c in zip(boxes_low, scores_low, cls_low)
        ]
        r_tracked = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = matching.iou_distance(r_tracked, detections_low)
        # Unmatched low-score detections are intentionally discarded (ByteTrack).
        matches, u_track, _ = matching.linear_assignment(
            dists, thresh=0.5
        )
        for itrack, idet in matches:
            track = r_tracked[itrack]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        for i in u_track:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # --- Handle unconfirmed tracks (only matched to high-score dets). ---
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7
        )
        for itrack, idet in matches:
            unconfirmed[itrack].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itrack])
        for i in u_unconfirmed:
            track = unconfirmed[i]
            track.mark_removed()
            removed.append(track)

        # --- Init new tracks from unmatched high-score detections. ---
        for idet in u_detection:
            track = detections[idet]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # --- Age out lost tracks. ---
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # --- Merge bookkeeping lists. ---
        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked
        ]
        self.tracked_stracks = join_stracks(self.tracked_stracks, activated)
        self.tracked_stracks = join_stracks(self.tracked_stracks, refind)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks
        )
        return [t for t in self.tracked_stracks if t.is_activated]


# STrack expects tlwh in __init__; add a small helper for tlbr inputs.
def _tlwh_from_tlbr(tlbr):
    ret = np.asarray(tlbr, dtype=np.float32).copy()
    ret[2:] -= ret[:2]
    return ret


STrack.tlwh_from_tlbr = staticmethod(_tlwh_from_tlbr)


def join_stracks(tlista, tlistb):
    """Union of two track lists, keyed by track_id."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        if exists.get(t.track_id, 0) == 0:
            exists[t.track_id] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Set difference tlista - tlistb, keyed by track_id."""
    stracks = {t.track_id: t for t in tlista}
    for t in tlistb:
        stracks.pop(t.track_id, None)
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Drop near-identical tracks (IoU > 0.85), keeping the longer-lived one."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
