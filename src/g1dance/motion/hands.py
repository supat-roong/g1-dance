"""Name keyed registry of hand estimators: the fifth seam.

Four stages were swappable and hands were not. `HandLandmarker` was constructed
inside `mediapipe_fe.estimate`, so the only way to change how fingers were
estimated was to edit the body frontend, which is the coupling the platform
design removed everywhere else.

It matters more than "expressive garnish" suggests. `fingers.py` opens by
conceding that hand landmarks in real dance footage drop out constantly and that
everything in it degrades into a neutral hand. That is a claim about the
estimator rather than about hands, and it was untestable while there was only
one.

The seam is sequence level: an estimator gets the clip and returns whole tracks.
A per frame contract would have to hand its callee pixel landmarks and crop
geometry, which is more intimate than a seam should be, and would force a model
that localises hands itself to work through somebody else's region of interest.
The cost is reading the video twice, measured at 0.61 s per 360 frames against
minutes for the estimate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .ir import N_HAND_LANDMARKS


@dataclass
class HandTracks:
    """Both hands over a clip, in the IR frame, wrist relative, metres.

    The 21 point topology is the common one rather than a MediaPipe choice: MANO
    and OpenPose use the same points, and `fingers.py` consumes only flexion
    angles built from wrist, MCP, PIP and tip, which are dimensionless.

    `conf` of zero means the estimator did not see that hand on that frame, which
    `fingers.py` already reads as the signal to hold a neutral hand. A frame
    reported at zero confidence carries zero landmarks, which is the same
    convention the body contract uses for a landmark it could not see.
    """

    left: np.ndarray                # (T, 21, 3)
    right: np.ndarray               # (T, 21, 3)
    conf: np.ndarray                # (T, 2) left, right
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.left = np.asarray(self.left, np.float32)
        self.right = np.asarray(self.right, np.float32)
        self.conf = np.asarray(self.conf, np.float32)
        self.validate()

    @property
    def n_frames(self) -> int:
        return len(self.conf)

    def validate(self) -> None:
        T = self.n_frames
        for name, arr, shape in (("left", self.left, (T, N_HAND_LANDMARKS, 3)),
                                 ("right", self.right, (T, N_HAND_LANDMARKS, 3)),
                                 ("conf", self.conf, (T, 2))):
            if arr.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {arr.shape}")
        if self.conf.size and (self.conf.min() < 0.0 or self.conf.max() > 1.0):
            raise ValueError("conf must lie in [0, 1]")

    @classmethod
    def empty(cls, n_frames: int, **meta) -> HandTracks:
        z = np.zeros((n_frames, N_HAND_LANDMARKS, 3), np.float32)
        return cls(left=z.copy(), right=z.copy(),
                   conf=np.zeros((n_frames, 2), np.float32), meta=dict(meta))


@runtime_checkable
class HandEstimator(Protocol):
    def __call__(self, source, subject, pose, **opts) -> HandTracks: ...


def attach(pose, tracks: HandTracks):
    """A copy of `pose` carrying `tracks`.

    Order matters at the call sites: hands are attached before `fill_gaps`,
    because `clean.fill_gaps` interpolates `hand_left`, `hand_right` and
    `hand_conf` along with the body. Filling first and attaching after would
    leave the hands unfilled while the body was filled, which is a sequence no
    stage downstream expects.
    """
    from dataclasses import replace
    if tracks.n_frames != len(pose.valid):
        raise ValueError(
            f"hand tracks cover {tracks.n_frames} frames, the pose has "
            f"{len(pose.valid)}")
    # `replace`, not a constructor call naming the fields: the constructor
    # version silently dropped the SMPL-X parameters, which every parameter
    # frontend goes through here, so GMR refused everything `run` produced.
    return replace(
        pose, hand_left=tracks.left, hand_right=tracks.right,
        hand_conf=tracks.conf,
        meta={**(pose.meta or {}), "hands": dict(tracks.meta)})


def _load_none() -> HandEstimator:
    def estimate_hands(source, subject, pose, **opts) -> HandTracks:
        """No hands, stated rather than left open."""
        return HandTracks.empty(len(pose.valid), estimator="none")

    return estimate_hands


def _load_mediapipe() -> HandEstimator:
    from .frontends.mediapipe_hands import estimate_hands
    return estimate_hands


_LOADERS: dict[str, Callable[[], HandEstimator]] = {
    "none": _load_none,
    "mediapipe": _load_mediapipe,
}

DESCRIPTIONS: dict[str, str] = {
    "none": "no hands; the fingers hold their neutral pose",
    "mediapipe": "one landmarker per side, region of interest from the wrist",
}

HAND_ESTIMATORS = _LOADERS

#: What each hand estimator needs from the body frontend, in the vocabulary of
#: `frontends.PROVIDES`. The mediapipe one crops around pixel landmarks.
NEEDS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "mediapipe": frozenset({"pixels"}),
}


def register_hand_estimator(name: str, loader: Callable[[], HandEstimator],
                            description: str = "") -> None:
    if name in _LOADERS:
        raise ValueError(f"hand estimator {name!r} is already registered")
    _LOADERS[name] = loader
    DESCRIPTIONS[name] = description


def available_hand_estimators() -> list[str]:
    return sorted(_LOADERS)


def get_hand_estimator(name: str) -> HandEstimator:
    try:
        return _LOADERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown hand estimator {name!r}; "
            f"available: {', '.join(available_hand_estimators())}") from None


__all__ = ["DESCRIPTIONS", "HAND_ESTIMATORS", "HandEstimator", "HandTracks",
           "attach",
           "available_hand_estimators", "get_hand_estimator",
           "register_hand_estimator"]
