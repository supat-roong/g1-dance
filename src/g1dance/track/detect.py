"""Per-frame multi-person detection, behind a name-keyed registry.

The default detector is MediaPipe Pose with `num_poses > 1`. It is Apache-2.0
and is already a dependency of the estimate stage, so the default path adds no
new heavyweight requirement. Ultralytics YOLO with BoT-SORT is worth having for
crowded footage but is AGPL-3.0, so it stays an opt-in extra; the registry is
what lets it be added later without any caller changing.

Detections carry an appearance histogram computed here rather than in
tracklets.py. The detector is the only stage holding the decoded frame, and
keeping the crop at that point means the rest of the pipeline never has to
decode the video a second time.
"""

from __future__ import annotations

import os
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from ..motion.ir import N_POSE_LANDMARKS

#: Where downloaded task bundles live. One cache for the whole project.
CACHE_DIR = Path(os.environ.get("G1DANCE_CACHE", Path.home() / ".cache" / "g1dance"))
MODEL_DIR = CACHE_DIR / "models"

#: MediaPipe publishes these as loose `.task` bundles; there is no pip package
#: that ships them, so they are fetched once and cached.
POSE_MODELS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}

#: Colour histogram shape. Hue and saturation only: value swings with stage
#: lighting far more than a dancer's costume does.
HIST_BINS = (16, 8)
APPEARANCE_DIM = HIST_BINS[0] * HIST_BINS[1]


class DetectError(RuntimeError):
    """Detection could not be set up or run."""


class ModelDownloadError(DetectError):
    """The task bundle is neither cached nor reachable."""


@dataclass
class Detection:
    """One person in one frame, in pixel coordinates.

    Pixels rather than normalised coordinates because IoU, crop extraction and
    the contact sheet all want pixels, and only the frame size converts back.
    """

    bbox: np.ndarray                       # (4,) x1, y1, x2, y2
    landmarks: np.ndarray                  # (33, 2) pixels
    visibility: np.ndarray                 # (33,)
    score: float = 1.0
    appearance: np.ndarray = field(default_factory=lambda: np.zeros(APPEARANCE_DIM, np.float32))

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, np.float32).reshape(4)
        self.landmarks = np.asarray(self.landmarks, np.float32)
        self.visibility = np.asarray(self.visibility, np.float32)
        self.appearance = np.asarray(self.appearance, np.float32).reshape(-1)
        if self.landmarks.shape != (N_POSE_LANDMARKS, 2):
            raise ValueError(f"landmarks: expected (33, 2), got {self.landmarks.shape}")
        if self.visibility.shape != (N_POSE_LANDMARKS,):
            raise ValueError(f"visibility: expected (33,), got {self.visibility.shape}")

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2,
                         (self.bbox[1] + self.bbox[3]) / 2], np.float32)


class PersonDetector(Protocol):
    """Anything that can turn one BGR frame into a list of people."""

    def detect_frame(self, image: np.ndarray) -> list[Detection]: ...


DETECTORS: dict[str, Callable[..., PersonDetector]] = {}


def register_detector(name: str) -> Callable[[Callable[..., PersonDetector]],
                                             Callable[..., PersonDetector]]:
    """Decorator putting a detector factory in the registry under `name`."""

    def wrap(factory: Callable[..., PersonDetector]) -> Callable[..., PersonDetector]:
        DETECTORS[name] = factory
        return factory

    return wrap


def get_detector(name: str = "mediapipe", **kwargs) -> PersonDetector:
    if name not in DETECTORS:
        raise DetectError(
            f"unknown detector {name!r}; registered: {sorted(DETECTORS)}")
    return DETECTORS[name](**kwargs)


# ---------------------------------------------------------------- appearance
def crop_hist(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """L1 normalised hue/saturation histogram of a bbox crop.

    The costume is the one cue that survives a cut, so this is what identity
    propagation across shots ultimately rests on.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_bbox(bbox, w, h)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(APPEARANCE_DIM, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HIST_BINS), [0, 180, 0, 256])
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    return hist / total if total > 0 else hist


def _clip_bbox(bbox: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(max(0, min(width - 1, round(float(bbox[0])))))
    y1 = int(max(0, min(height - 1, round(float(bbox[1])))))
    x2 = int(max(x1 + 1, min(width, round(float(bbox[2])))))
    y2 = int(max(y1 + 1, min(height, round(float(bbox[3])))))
    return x1, y1, x2, y2


def bbox_from_landmarks(points: np.ndarray, visibility: np.ndarray,
                        width: int, height: int, margin: float = 0.06,
                        min_visibility: float = 0.2) -> np.ndarray:
    """Tight box around the landmarks worth trusting, padded a little."""
    keep = visibility >= min_visibility
    pts = points[keep] if keep.any() else points
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    pad_x = (x2 - x1) * margin
    pad_y = (y2 - y1) * margin
    return np.array([max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                     min(float(width), x2 + pad_x),
                     min(float(height), y2 + pad_y)], np.float32)


# ---------------------------------------------------------------- mediapipe
def ensure_pose_model(variant: str = "lite", cache_dir: Path | None = None) -> Path:
    """Path to a cached `.task` bundle, downloading it once if need be."""
    if variant not in POSE_MODELS:
        raise DetectError(
            f"unknown pose model {variant!r}; known: {sorted(POSE_MODELS)}")
    directory = Path(cache_dir) if cache_dir else MODEL_DIR
    dest = directory / f"pose_landmarker_{variant}.task"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    directory.mkdir(parents=True, exist_ok=True)
    url = POSE_MODELS[variant]
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            tmp.write_bytes(response.read())
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"the MediaPipe pose model is not cached at {dest} and could not be "
            f"downloaded from {url} ({exc}). Fetch it once while online, or "
            f"place the file there by hand.") from exc
    tmp.replace(dest)
    return dest


class UnsupportedRuntimeError(DetectError):
    """The installed MediaPipe cannot run this task on this machine."""


#: The only runtime configuration that works on mediapipe 1.0.x / macOS arm64.
#:
#: Two failure modes, both verified on this machine, and both SIGABRT rather
#: than an exception, so neither can be caught and retried:
#:
#:   * `Delegate.CPU` dies in the detector subgraph with `DrishtiMetalHelper
#:     ... Service is unavailable`. It is not a Holistic-only problem and
#:     MEDIAPIPE_DISABLE_GPU does not help.
#:   * `Delegate.GPU` with a 3-channel `ImageFormat.SRGB` dies with
#:     `unsupported ImageFrame format: 1`.
#:
#: GPU plus 4-channel SRGBA works, and is what both this detector and the
#: mediapipe frontend use. Because the alternatives abort the process, these
#: are deliberately not configurable.
_DELEGATE_GPU = True
_IMAGE_FORMAT = "SRGBA"


def _check_mediapipe_runtime(mp) -> None:
    """Kept as an explicit escape hatch, not a version gate.

    An earlier revision refused mediapipe 1.0.x outright on macOS arm64 after
    seeing the CPU-delegate abort. That diagnosis was too broad: the build is
    fine, the CPU delegate is not. Setting G1DANCE_FORCE_CPU_DELEGATE=1 restores
    the old path for anyone debugging it, and will abort the process.
    """
    if os.environ.get("G1DANCE_FORCE_CPU_DELEGATE"):
        raise UnsupportedRuntimeError(
            "G1DANCE_FORCE_CPU_DELEGATE is set. The CPU delegate aborts the "
            "process on mediapipe 1.0.x / macOS arm64. Unset it to use Metal.")


@register_detector("mediapipe")
@dataclass
class MediaPipeDetector:
    """MediaPipe Tasks PoseLandmarker in IMAGE mode with `num_poses > 1`.

    IMAGE mode rather than VIDEO mode: VIDEO mode carries tracking state
    between calls, which makes results depend on how frames were sampled.
    Association is tracklets.py's job, and it should get the same detections
    whatever stride the caller chose.
    """

    #: Defaults tuned on a six-dancer studio clip, where the old ones
    #: (lite model, 5 people, 0.5 confidence) found 0 to 2 people per frame out
    #: of six and shattered every tracklet, leaving 12 percent subject
    #: coverage. `full` plus a 0.3 gate finds 1 to 4. The pose detector's limit
    #: here is subject size: these dancers are about 400 px tall in a 1080p
    #: frame, near the bottom of what multi-pose resolves, which is also why
    #: `detect_scale` exists.
    max_people: int = 8
    variant: str = "full"
    min_detection_confidence: float = 0.3
    min_presence_confidence: float = 0.3
    #: Upscale factor applied before detection. Costs time, buys small
    #: subjects: the detector resolves people it otherwise misses entirely.
    detect_scale: float = 1.0
    model_path: Path | None = None
    _landmarker: object | None = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                PoseLandmarker, PoseLandmarkerOptions, RunningMode)
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise DetectError(f"mediapipe is not importable: {exc}") from exc
        _check_mediapipe_runtime(mp)
        self._mp = mp
        path = Path(self.model_path) if self.model_path else ensure_pose_model(self.variant)
        delegate = BaseOptions.Delegate.GPU if _DELEGATE_GPU else BaseOptions.Delegate.CPU
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path), delegate=delegate),
            running_mode=RunningMode.IMAGE,
            num_poses=int(self.max_people),
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_presence_confidence,
            output_segmentation_masks=False)
        self._landmarker = PoseLandmarker.create_from_options(options)

    def detect_frame(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        src = image
        if self.detect_scale != 1.0:
            src = cv2.resize(image, None, fx=self.detect_scale, fy=self.detect_scale,
                             interpolation=cv2.INTER_CUBIC)
        rgba = np.ascontiguousarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGBA))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGBA, data=rgba)
        result = self._landmarker.detect(mp_image)
        out: list[Detection] = []
        for pose in getattr(result, "pose_landmarks", []) or []:
            points = np.array([[lm.x * w, lm.y * h] for lm in pose], np.float32)
            vis = np.array([_confidence(lm) for lm in pose], np.float32)
            bbox = bbox_from_landmarks(points, vis, w, h)
            out.append(Detection(bbox=bbox, landmarks=points, visibility=vis,
                                 score=float(vis.mean()),
                                 appearance=crop_hist(image, bbox)))
        return out

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None


def _confidence(landmark) -> float:
    """MediaPipe reports visibility and presence separately; the useful signal
    for us is "is this joint really here", which is the weaker of the two."""
    vis = getattr(landmark, "visibility", None)
    pres = getattr(landmark, "presence", None)
    values = [v for v in (vis, pres) if v is not None]
    return float(min(values)) if values else 1.0


# ---------------------------------------------------------------- driving
def detect_people(video: str | Path, frames: list[int] | None = None,
                  detector: str | PersonDetector = "mediapipe",
                  **kwargs) -> list[list[Detection]]:
    """Run a detector over `frames` of `video`, in the order given.

    `frames` must be sorted; the video is decoded once, forwards. Returns one
    list of detections per requested frame, empty where nobody was found.
    """
    video = Path(video)
    det = get_detector(detector, **kwargs) if isinstance(detector, str) else detector
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise DetectError(f"could not open {video} for detection")
    try:
        wanted = _frame_plan(frames, cap)
        results: list[list[Detection]] = []
        want_set = set(wanted)
        index = 0
        seen: dict[int, list[Detection]] = {}
        while want_set and index <= max(wanted):
            ok = cap.grab()
            if not ok:
                break
            if index in want_set:
                ok, image = cap.retrieve()
                if ok:
                    seen[index] = det.detect_frame(image)
                want_set.discard(index)
            index += 1
        results = [seen.get(f, []) for f in wanted]
    finally:
        cap.release()
        if isinstance(detector, str):
            close = getattr(det, "close", None)
            if close:
                close()
    return results


def _frame_plan(frames: list[int] | None, cap) -> list[int]:
    if frames is None:
        total = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        return list(range(max(total, 0)))
    wanted = sorted({int(f) for f in frames if int(f) >= 0})
    if not wanted:
        raise DetectError("no frames requested")
    return wanted
