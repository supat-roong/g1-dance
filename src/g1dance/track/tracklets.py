"""Within-shot association, and the descriptors that survive a cut.

Inside one take, identity is a geometry problem: a dancer moves a little
between frames, so overlapping boxes with similar keypoints are the same
person. That is all this module does with motion.

What it produces for the next module is different in kind. A descriptor has to
be comparable across a cut, where nothing geometric carries over, so it holds
only what a change of camera leaves intact: the costume's colours and the
dancer's body proportions. `full_body_score` is the third output and exists
because the subject should be picked in the shot where retargeting will
actually work, not merely the first shot they appear in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..motion.ir import MP_POSE
from .detect import APPEARANCE_DIM, Detection

#: Landmarks that decide whether a body is usable for retargeting. Ankles and
#: feet are in here deliberately: a waist-up shot is worthless for whole-body
#: tracking however crisp the arms are.
KEY_LANDMARKS = tuple(sorted(MP_POSE.values()))
FOOT_LANDMARKS = tuple(MP_POSE[n] for n in
                       ("left_ankle", "right_ankle", "left_heel", "right_heel",
                        "left_foot_index", "right_foot_index"))

#: A dancer taller than half the frame is framed well enough to retarget.
FULL_BODY_HEIGHT_FRAC = 0.5
#: Association is refused above this cost, which starts a new tracklet rather
#: than welding two people together.
MAX_MATCH_COST = 0.7
#: Frames a tracklet may go unmatched before it is closed.
MAX_GAP_FRAMES = 12
#: Weight of appearance against body proportions in the cross-shot descriptor.
#: Colour dominates because two dancers often have near-identical proportions.
APPEARANCE_WEIGHT = 0.8

N_PROPORTIONS = 7


@dataclass
class Tracklet:
    """One person, continuously observed, inside one shot."""

    id: int
    shot: int
    frames: np.ndarray                  # (n,) absolute frame indices
    bboxes: np.ndarray                  # (n, 4)
    landmarks: np.ndarray               # (n, 33, 2)
    visibility: np.ndarray              # (n, 33)
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(APPEARANCE_DIM, np.float32))
    proportions: np.ndarray = field(default_factory=lambda: np.zeros(N_PROPORTIONS, np.float32))
    full_body_score: float = 0.0

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, int)
        for name in ("bboxes", "landmarks", "visibility", "appearance", "proportions"):
            setattr(self, name, np.asarray(getattr(self, name), np.float32))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        """Half open, so it composes with shot ranges."""
        return int(self.frames[-1]) + 1

    @property
    def duration(self) -> int:
        return self.end - self.start

    @property
    def descriptor(self) -> np.ndarray:
        """Cross-shot identity vector: weighted, unit norm, cosine comparable."""
        return np.concatenate([
            APPEARANCE_WEIGHT * _unit(self.appearance),
            (1.0 - APPEARANCE_WEIGHT) * _unit(self.proportions),
        ]).astype(np.float32)

    def bbox_at(self, frame: int) -> np.ndarray:
        """Box at an arbitrary frame inside the tracklet, linearly filled in.

        Detection may have been strided, so most frames were never seen.
        """
        f = np.clip(frame, self.start, self.end - 1)
        return np.array([np.interp(f, self.frames, self.bboxes[:, i])
                         for i in range(4)], np.float32)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def keypoint_distance(a: Detection, b: Detection, scale: float) -> float:
    """Mean landmark displacement in units of body size, clipped to [0, 1].

    Scaling by body size rather than frame size keeps the cost meaningful for
    a dancer far from the camera, whose real motion is small in pixels.
    """
    weight = np.minimum(a.visibility, b.visibility)
    if weight.sum() <= 0 or scale <= 0:
        return 1.0
    dist = np.linalg.norm(a.landmarks - b.landmarks, axis=1)
    mean = float((dist * weight).sum() / weight.sum())
    return float(np.clip(mean / scale, 0.0, 1.0))


def match_cost(a: Detection, b: Detection, iou_weight: float = 0.6) -> float:
    """Cost of calling two detections the same person, in [0, 1]."""
    scale = max(a.height, b.height, 1.0)
    geometric = 1.0 - iou(a.bbox, b.bbox)
    kinematic = keypoint_distance(a, b, scale)
    return iou_weight * geometric + (1.0 - iou_weight) * kinematic


@dataclass
class _OpenTrack:
    """Mutable accumulator; becomes a Tracklet once the shot ends."""

    id: int
    frames: list[int] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    last_frame: int = -1

    def add(self, frame: int, detection: Detection) -> None:
        self.frames.append(frame)
        self.detections.append(detection)
        self.last_frame = frame


def build_tracklets(frames: list[int], detections: list[list[Detection]],
                    shot: int, frame_height: int, *, first_id: int = 0,
                    max_cost: float = MAX_MATCH_COST,
                    max_gap: int = MAX_GAP_FRAMES,
                    min_length: int = 2) -> list[Tracklet]:
    """Associate per-frame detections into tracklets inside a single shot.

    `frames` and `detections` are parallel and sorted. Association is a
    rectangular assignment per frame, gated by `max_cost`, so a detection that
    matches nothing opens a new tracklet instead of being forced onto the
    nearest track.
    """
    if len(frames) != len(detections):
        raise ValueError("frames and detections must be the same length")
    open_tracks: list[_OpenTrack] = []
    closed: list[_OpenTrack] = []
    next_id = first_id

    for frame, frame_dets in zip(frames, detections):
        alive, expired = [], []
        for track in open_tracks:
            (alive if frame - track.last_frame <= max_gap else expired).append(track)
        closed.extend(expired)
        open_tracks = alive

        matched_dets: set[int] = set()
        if open_tracks and frame_dets:
            cost = np.array([[match_cost(track.detections[-1], det)
                              for det in frame_dets] for track in open_tracks])
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= max_cost:
                    open_tracks[r].add(frame, frame_dets[c])
                    matched_dets.add(int(c))

        for i, det in enumerate(frame_dets):
            if i in matched_dets:
                continue
            track = _OpenTrack(id=next_id)
            next_id += 1
            track.add(frame, det)
            open_tracks.append(track)

    closed.extend(open_tracks)
    out = [_finalise(t, shot, frame_height) for t in closed
           if len(t.frames) >= min_length]
    out.sort(key=lambda t: (t.start, t.id))
    return out


def _finalise(track: _OpenTrack, shot: int, frame_height: int) -> Tracklet:
    dets = track.detections
    bboxes = np.stack([d.bbox for d in dets])
    landmarks = np.stack([d.landmarks for d in dets])
    visibility = np.stack([d.visibility for d in dets])
    tracklet = Tracklet(
        id=track.id, shot=shot, frames=np.array(track.frames, int),
        bboxes=bboxes, landmarks=landmarks, visibility=visibility,
        appearance=mean_appearance(dets),
        proportions=body_proportions(landmarks, visibility, bboxes))
    tracklet.full_body_score = full_body_score(visibility, bboxes, frame_height)
    return tracklet


def mean_appearance(detections: list[Detection]) -> np.ndarray:
    """Average of the per-frame histograms, renormalised."""
    hists = [d.appearance for d in detections if d.appearance.size == APPEARANCE_DIM]
    if not hists:
        return np.zeros(APPEARANCE_DIM, np.float32)
    mean = np.mean(np.stack(hists), axis=0).astype(np.float32)
    total = float(mean.sum())
    return mean / total if total > 0 else mean


def body_proportions(landmarks: np.ndarray, visibility: np.ndarray,
                     bboxes: np.ndarray) -> np.ndarray:
    """Limb lengths relative to torso length, plus the box aspect ratio.

    Ratios rather than lengths, so the descriptor does not change when the
    camera moves closer. The median over frames absorbs single bad frames.
    """
    idx = MP_POSE
    shoulder = _midpoint(landmarks, idx["left_shoulder"], idx["right_shoulder"])
    hip = _midpoint(landmarks, idx["left_hip"], idx["right_hip"])
    torso = np.linalg.norm(shoulder - hip, axis=1)
    good = torso > 1e-3
    if not good.any():
        return np.zeros(N_PROPORTIONS, np.float32)

    def ratio(a: int, b: int) -> np.ndarray:
        seg = np.linalg.norm(landmarks[:, a] - landmarks[:, b], axis=1)
        return seg / np.maximum(torso, 1e-3)

    width = np.maximum(bboxes[:, 2] - bboxes[:, 0], 1e-3)
    features = np.stack([
        ratio(idx["left_shoulder"], idx["right_shoulder"]),
        ratio(idx["left_hip"], idx["right_hip"]),
        ratio(idx["left_shoulder"], idx["left_elbow"]),
        ratio(idx["left_elbow"], idx["left_wrist"]),
        ratio(idx["left_hip"], idx["left_knee"]),
        ratio(idx["left_knee"], idx["left_ankle"]),
        (bboxes[:, 3] - bboxes[:, 1]) / width,
    ], axis=1)
    value = np.median(features[good], axis=0)
    return np.nan_to_num(value, nan=0.0, posinf=0.0).astype(np.float32)


def _midpoint(landmarks: np.ndarray, a: int, b: int) -> np.ndarray:
    return (landmarks[:, a] + landmarks[:, b]) / 2.0


def full_body_score(visibility: np.ndarray, bboxes: np.ndarray,
                    frame_height: int) -> float:
    """How usable this person is for retargeting, in [0, 1].

    Two independent ways to be useless: joints the detector cannot see, and a
    dancer so small in frame that the pixels carry no pose. Both are folded in
    here so subject picking is one comparison rather than a policy at each
    call site.
    """
    if visibility.size == 0 or frame_height <= 0:
        return 0.0
    key = np.clip(visibility[:, list(KEY_LANDMARKS)], 0.0, 1.0)
    feet = np.clip(visibility[:, list(FOOT_LANDMARKS)], 0.0, 1.0)
    seen = float(np.median(key.mean(axis=1)))
    grounded = float(np.median(feet.mean(axis=1)))
    height_frac = float(np.median(bboxes[:, 3] - bboxes[:, 1])) / frame_height
    size = float(np.clip(height_frac / FULL_BODY_HEIGHT_FRAC, 0.0, 1.0))
    return float(np.clip(0.45 * seen + 0.2 * grounded + 0.35 * size, 0.0, 1.0))


def similarity(a: Tracklet, b: Tracklet) -> float:
    """Cosine similarity of two descriptors, in [0, 1]."""
    da, db = a.descriptor, b.descriptor
    norm = float(np.linalg.norm(da) * np.linalg.norm(db))
    if norm <= 0:
        return 0.0
    return float(np.clip(float(da @ db) / norm, 0.0, 1.0))
