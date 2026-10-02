"""Stage 2: find the dancer, keep hold of them across cuts, say when they left.

The public surface is deliberately small. Callers want `track()` and the
`TrackResult` it returns; everything else here exists so the four steps can be
tested and replaced one at a time.
"""

from __future__ import annotations

from .detect import (
    DETECTORS,
    Detection,
    DetectError,
    ModelDownloadError,
    UnsupportedRuntimeError,
    crop_hist,
    detect_people,
    ensure_pose_model,
    get_detector,
    register_detector,
)
from .propagate import (
    ABSENT,
    ASSIGNED,
    SIMILARITY_THRESHOLD,
    SUBJECT,
    ShotAssignment,
    TrackAnalysis,
    TrackResult,
    analyse,
    propagate,
    track,
    write_contact_sheet,
)
from .shots import detect_shots, shot_of
from .tracklets import (
    Tracklet,
    build_tracklets,
    body_proportions,
    full_body_score,
    iou,
    similarity,
)

__all__ = [
    "ABSENT", "ASSIGNED", "SUBJECT", "SIMILARITY_THRESHOLD",
    "DETECTORS", "DetectError", "Detection", "ModelDownloadError",
    "UnsupportedRuntimeError",
    "ShotAssignment", "TrackAnalysis", "TrackResult", "Tracklet", "analyse",
    "body_proportions", "build_tracklets", "crop_hist", "detect_people",
    "detect_shots", "ensure_pose_model", "full_body_score", "get_detector",
    "iou", "propagate", "register_detector", "shot_of", "similarity",
    "track", "write_contact_sheet",
]
