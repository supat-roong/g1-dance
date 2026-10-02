"""MediaPipe Tasks frontend: video plus a subject box in, PoseSequence out.

Three facts about the installed stack shape everything below, all of them
established by probing mediapipe 1.0.1 on macOS arm64 rather than by reading
documentation, because the Tasks API changed repeatedly across versions.

1. `HolisticLandmarker` exists in `mediapipe.tasks.python.vision` but cannot
   run here. Its face blendshapes subgraph fails to bind its input tensors to
   the Metal delegate, and the CPU delegate aborts the whole process inside
   `DrishtiMetalHelper` before any Python exception can be raised. The design
   asks for Holistic; the pipeline therefore composes a pose model with a
   separate hand model, and reproduces the one thing Holistic gave for free by
   deriving the hand region of interest from the pose wrists. That derivation
   now lives behind the hand seam in `motion/hands.py`; this module's part in it
   is `landmarks_px`, which records where each landmark sat in the image so the
   region of interest can be built without running pose a second time.

2. The only working runtime configuration is the GPU (Metal) delegate fed with
   four channel `SRGBA` images. `Delegate.CPU` aborts the process; `SRGB` input
   on the GPU delegate aborts it too, with "unsupported ImageFrame format: 1".
   A hard abort cannot be caught, so these two choices are not configurable.

3. `pose_world_landmarks` already arrives in metres with the hip midpoint at
   the origin, which is exactly the `PoseSequence` contract. Measured on a
   reference frame: hip midpoint at (0.002, 0.000, 0.003) m, shoulder to hip
   0.49 m. We re-centre anyway, because a contract that is only true by
   accident of the upstream model is not a contract.

Landmarks are rotated 180 degrees about x on the way out, turning MediaPipe's
y-down frame into the y-up frame `PoseSequence` documents while preserving
handedness.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from ..ir import MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence

#: Google's public model bundle host. Every URL below was verified to return
#: 200 with the byte count recorded beside it.
_MODEL_HOST = "https://storage.googleapis.com/mediapipe-models"

#: bundle name -> (download url, expected size in bytes)
MODEL_BUNDLES: dict[str, tuple[str, int]] = {
    "pose_lite": (
        f"{_MODEL_HOST}/pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_lite.task", 5_777_746),
    "pose_full": (
        f"{_MODEL_HOST}/pose_landmarker/pose_landmarker_full/float16/latest/"
        "pose_landmarker_full.task", 9_398_198),
    "pose_heavy": (
        f"{_MODEL_HOST}/pose_landmarker/pose_landmarker_heavy/float16/latest/"
        "pose_landmarker_heavy.task", 30_664_242),
    "hand": (
        f"{_MODEL_HOST}/hand_landmarker/hand_landmarker/float16/latest/"
        "hand_landmarker.task", 7_819_105),
    # Present for completeness and to document that it was tried. Loading it
    # raises on this platform; see the module docstring.
    "holistic": (
        f"{_MODEL_HOST}/holistic_landmarker/holistic_landmarker/float16/latest/"
        "holistic_landmarker.task", 13_683_609),
}

#: where bundles are cached. Overridable so CI can point at a warmed cache.
DEFAULT_MODEL_DIR = Path(
    os.environ.get("G1DANCE_MODEL_DIR",
                   Path(__file__).resolve().parents[4] / "assets" / "mediapipe"))

#: MediaPipe pose indices for the landmarks that must be present for a frame to
#: count. Head and hand landmarks come and go; a torso that vanishes means the
#: detection is junk.
_CORE_LANDMARKS = tuple(MP_POSE[k] for k in (
    "left_shoulder", "right_shoulder", "left_hip", "right_hip"))

#: pose landmark indices per side: wrist, elbow, and the three coarse hand
#: points the pose model itself emits (pinky, index, thumb)
_SIDE_LANDMARKS = {
    "left": {"wrist": 15, "elbow": 13, "hand": (17, 19, 21)},
    "right": {"wrist": 16, "elbow": 14, "hand": (18, 20, 22)},
}

#: 180 degree rotation about x: MediaPipe y-down becomes PoseSequence y-up,
#: without mirroring the subject the way a bare sign flip on y would.
_TO_IR_AXES = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


class ModelUnavailableError(RuntimeError):
    """A required `.task` bundle is missing and download was not permitted."""


def model_path(bundle: str, model_dir: Path | str | None = None) -> Path:
    """Local path a bundle is cached at, whether or not it exists yet."""
    if bundle not in MODEL_BUNDLES:
        raise ValueError(
            f"unknown model bundle {bundle!r}; "
            f"available: {', '.join(sorted(MODEL_BUNDLES))}")
    url, _ = MODEL_BUNDLES[bundle]
    return Path(model_dir or DEFAULT_MODEL_DIR) / url.rsplit("/", 1)[-1]


def ensure_model(bundle: str, model_dir: Path | str | None = None,
                 download: bool = True) -> Path:
    """Return the local bundle path, fetching it once if it is not cached.

    Raises rather than downloading silently when `download` is False, so tests
    and offline runs fail with a message naming the URL instead of hanging on a
    socket.
    """
    path = model_path(bundle, model_dir)
    if path.exists():
        return path
    url, size = MODEL_BUNDLES[bundle]
    if not download:
        raise ModelUnavailableError(
            f"model bundle {bundle!r} is not cached at {path}. "
            f"Fetch it with: curl -L -o {path} {url}")
    import urllib.request
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    got = tmp.stat().st_size
    if got != size:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailableError(
            f"{bundle} downloaded from {url} is {got} bytes, expected {size}")
    tmp.rename(path)
    return path


# --------------------------------------------------------------------------
# geometry. Pure numpy, no mediapipe, so it can be unit tested without models.
# --------------------------------------------------------------------------

def expand_bbox(bbox: Sequence[float], frame_w: int, frame_h: int,
                expand: float = 0.15) -> tuple[int, int, int, int]:
    """Grow an xyxy box by `expand` of its size, clipped to the frame.

    The tracker's box hugs the subject, and a hugging box clips the very
    landmarks that matter most: hands and feet at the extremes of a dance move.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    dx, dy = w * expand * 0.5, h * expand * 0.5
    return (int(np.floor(max(0.0, x0 - dx))), int(np.floor(max(0.0, y0 - dy))),
            int(np.ceil(min(float(frame_w), x1 + dx))),
            int(np.ceil(min(float(frame_h), y1 + dy))))


def letterbox(crop: np.ndarray, size: int = 256) -> tuple[np.ndarray, float, int, int]:
    """Resize a crop into a square canvas without distorting it.

    Squashing a tall dancer into a square changes every limb ratio the pose
    model was trained on, so we pad instead. Returns the canvas plus the scale
    and padding needed to undo the transform.
    """
    import cv2
    h, w = crop.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"empty crop of shape {crop.shape}")
    scale = size / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size) + crop.shape[2:], dtype=crop.dtype)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def unletterbox_points(pts_norm: np.ndarray, box: tuple[int, int, int, int],
                       scale: float, pad_x: int, pad_y: int,
                       size: int = 256) -> np.ndarray:
    """Map landmarks normalised to the letterboxed crop back to frame pixels."""
    pts = np.asarray(pts_norm, np.float64).reshape(-1, 2) * size
    pts[:, 0] = (pts[:, 0] - pad_x) / scale + box[0]
    pts[:, 1] = (pts[:, 1] - pad_y) / scale + box[1]
    return pts


def wrist_roi(pose_px: np.ndarray, visibility: np.ndarray, side: str,
              frame_w: int, frame_h: int, expand: float = 1.1,
              min_size: float = 48.0,
              min_visibility: float = 0.5) -> tuple[int, int, int, int] | None:
    """Square box around one hand, derived from the pose wrist and elbow.

    This is the piece Holistic would have done internally. The forearm sets the
    scale because it is the one nearby segment whose length is proportional to
    the hand, and the box is pushed past the wrist along the forearm because
    that is where the hand actually is. When the pose model also emitted its
    coarse hand points we widen the box to contain them, which recovers a hand
    held out at an angle the forearm alone does not predict.

    Returns None when the wrist or elbow is not trustworthy, which the caller
    reports as zero hand confidence rather than a guessed hand.
    """
    ids = _SIDE_LANDMARKS[side]
    wrist_i, elbow_i = ids["wrist"], ids["elbow"]
    if min(visibility[wrist_i], visibility[elbow_i]) < min_visibility:
        return None
    wrist = np.asarray(pose_px[wrist_i], np.float64)
    elbow = np.asarray(pose_px[elbow_i], np.float64)
    forearm = float(np.linalg.norm(wrist - elbow))
    half = max(forearm * expand, min_size)
    centre = wrist + 0.25 * (wrist - elbow)
    seen = [pose_px[i] for i in ids["hand"] if visibility[i] >= min_visibility]
    if seen:
        reach = float(np.abs(np.asarray(seen, np.float64) - centre).max())
        half = max(half, reach * 1.6, min_size)
    x0, y0 = int(np.floor(centre[0] - half)), int(np.floor(centre[1] - half))
    x1, y1 = int(np.ceil(centre[0] + half)), int(np.ceil(centre[1] + half))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(int(frame_w), x1), min(int(frame_h), y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return x0, y0, x1, y1


def to_ir_frame(xyz: np.ndarray) -> np.ndarray:
    """MediaPipe world axes to PoseSequence axes (x right, y up, z to camera)."""
    return np.asarray(xyz, np.float32) @ _TO_IR_AXES.T


# --------------------------------------------------------------------------
# mediapipe plumbing
# --------------------------------------------------------------------------

def _wrap_image(rgba: np.ndarray):
    """Wrap an HxWx4 uint8 array as an mp.Image the Metal delegate accepts."""
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGBA,
                    data=np.ascontiguousarray(rgba, dtype=np.uint8))


def _base_options(path: Path):
    """BaseOptions pinned to the GPU delegate. See fact 2 in the docstring."""
    from mediapipe.tasks.python import BaseOptions
    return BaseOptions(model_asset_path=str(path),
                       delegate=BaseOptions.Delegate.GPU)


def _read_frames(video: Path | str, count: int | None = None
                 ) -> Iterator[tuple[int, np.ndarray | None, int, int]]:
    """Yield (index, BGR frame or None, width, height) over a video."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {video}")
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        i = 0
        while count is None or i < count:
            ok, frame = cap.read()
            if not ok:
                return
            yield i, frame, w, h
            i += 1
    finally:
        cap.release()


def probe_fps(video: Path | str) -> float:
    """Container fps, used only when the caller does not supply one."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video} reports no usable fps; pass fps= explicitly")
    return fps


def estimate(video: Path | str, bbox_per_frame: np.ndarray,
             valid_mask: np.ndarray | None = None, fps: float | None = None,
             *, pose_bundle: str = "pose_full", model_dir: Path | str | None = None,
             download: bool = True, crop_size: int = 256, hand_size: int = 224,
             bbox_expand: float = 0.15, min_pose_confidence: float = 0.5,
             min_landmark_visibility: float = 0.5, hands: bool = True,
             ) -> PoseSequence:
    """Run pose and hand landmarking on the tracked subject's crop.

    `bbox_per_frame` is (T, 4) xyxy in frame pixels from the track stage and
    `valid_mask` is its (T,) companion. Frames the tracker already gave up on
    are never decoded through a model, and frames where the pose comes back
    missing or below `min_pose_confidence` are marked invalid here, so the
    validity mask that reaches `clean` means one thing throughout: this frame
    carries a pose we believe.
    """
    bbox_per_frame = np.asarray(bbox_per_frame, np.float64)
    if bbox_per_frame.ndim != 2 or bbox_per_frame.shape[1] != 4:
        raise ValueError(
            f"bbox_per_frame must be (T, 4) xyxy, got {bbox_per_frame.shape}")
    T = len(bbox_per_frame)
    valid_mask = (np.ones(T, bool) if valid_mask is None
                  else np.asarray(valid_mask, bool))
    if valid_mask.shape != (T,):
        raise ValueError(
            f"valid_mask must be ({T},) to match bbox_per_frame, "
            f"got {valid_mask.shape}")
    fps = float(fps) if fps else probe_fps(video)

    from mediapipe.tasks.python import vision as V

    pose_file = ensure_model(pose_bundle, model_dir, download)

    landmarks = np.zeros((T, N_POSE_LANDMARKS, 3), np.float32)
    visibility = np.zeros((T, N_POSE_LANDMARKS), np.float32)
    hand_left = np.zeros((T, N_HAND_LANDMARKS, 3), np.float32)
    hand_right = np.zeros((T, N_HAND_LANDMARKS, 3), np.float32)
    hand_conf = np.zeros((T, 2), np.float32)
    # Where each landmark sat in the image. Kept because the hand seam builds
    # its region of interest from it; recomputing it would mean running pose
    # twice. See motion/hands.py.
    landmarks_px = np.zeros((T, N_POSE_LANDMARKS, 2), np.float32)
    valid = np.zeros(T, bool)

    pose_opts = V.PoseLandmarkerOptions(
        base_options=_base_options(pose_file),
        running_mode=V.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=min_pose_confidence,
        min_pose_presence_confidence=min_pose_confidence)
    pose_lm = V.PoseLandmarker.create_from_options(pose_opts)
    stamp = -1
    try:
        import cv2
        for t, frame, frame_w, frame_h in _read_frames(video, T):
            if not valid_mask[t]:
                continue
            stamp = max(stamp + 1, int(round(t * 1000.0 / fps)))
            box = expand_bbox(bbox_per_frame[t], frame_w, frame_h, bbox_expand)
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            crop = rgba[box[1]:box[3], box[0]:box[2]]
            canvas, scale, pad_x, pad_y = letterbox(crop, crop_size)
            result = pose_lm.detect_for_video(_wrap_image(canvas), stamp)
            if not result.pose_world_landmarks or not result.pose_landmarks:
                continue

            vis = np.array([lm.visibility for lm in result.pose_landmarks[0]],
                           np.float32)
            if float(vis[list(_CORE_LANDMARKS)].mean()) < min_pose_confidence:
                continue
            world = np.array([[lm.x, lm.y, lm.z]
                              for lm in result.pose_world_landmarks[0]], np.float32)
            hip = 0.5 * (world[MP_POSE["left_hip"]] + world[MP_POSE["right_hip"]])
            landmarks[t] = to_ir_frame(world - hip)
            visibility[t] = vis
            valid[t] = True

            if not hands:
                continue
            norm = np.array([[lm.x, lm.y] for lm in result.pose_landmarks[0]])
            landmarks_px[t] = unletterbox_points(
                norm, box, scale, pad_x, pad_y, crop_size)
    finally:
        pose_lm.close()

    return PoseSequence(
        landmarks_px=landmarks_px,
        fps=fps, landmarks=landmarks, visibility=visibility,
        hand_left=hand_left, hand_right=hand_right, hand_conf=hand_conf,
        valid=valid,
        meta={
            "frontend": "mediapipe",
            "frontend_version": _mediapipe_version(),
            "pose_bundle": pose_bundle,
            "hands": bool(hands),
            "hand_source": "pose_wrist_roi",
            "delegate": "gpu",
            "axes": "x_right_y_up_z_toward_camera",
            "units": "metres_hip_centred",
            "source": str(video),
            "coverage": float(valid.mean()) if T else 0.0,
        })


def _mediapipe_version() -> str:
    import mediapipe as mp
    return str(mp.__version__)


__all__ = ["MODEL_BUNDLES", "ModelUnavailableError", "ensure_model", "estimate",
           "expand_bbox", "letterbox", "model_path", "probe_fps", "to_ir_frame",
           "unletterbox_points", "wrist_roi"]
