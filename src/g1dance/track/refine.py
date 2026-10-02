"""Recover the frames the global detection pass missed.

The pose detector resolves a person by how many pixels they occupy, and in a
wide studio shot of a dance group nobody occupies many. Measured on a
six-dancer clip at 1080p, where each dancer is about 400 px tall, the global
pass finds roughly 1.8 of 6 people per frame, so the chosen subject is missing
from a large fraction of frames however good the association is.

Those frames are not hard, they are small. Once the subject's approximate
location is known from the frames either side, cropping to it and upscaling
makes them fill the image, and the same detector finds them: 33 of 33 recovered
on the frames the global pass had dropped.

So this is a second pass, not a better first pass. It runs after a subject has
been chosen, predicts where they should be by interpolating the bounding box
across each gap, re-detects inside that window, and accepts the result only if
it lands close to the prediction at a plausible size. Rejecting on those two
checks is what keeps it from quietly locking onto a different dancer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detect import DETECTORS, get_detector
from .propagate import TrackResult

#: How far outside the predicted box to look, as a fraction of its larger side.
ROI_MARGIN = 0.85
#: Upscale applied to the crop. The whole point is to make the subject big.
ROI_SCALE = 2.0
#: A recovered detection whose centre is further than this from the prediction,
#: relative to the predicted size, is a different person.
MAX_CENTRE_DRIFT = 0.75
#: Likewise for size: a detection this much larger or smaller is not the subject.
MAX_SIZE_RATIO = 2.2


#: Median window and low-pass cutoff for the bounding box trajectory, in
#: frames and Hz. A person's box cannot move discontinuously, so anything
#: high-frequency in it is detector noise, and it matters more than it looks:
#: the box is the crop the pose estimator sees, so a jumping box shifts and
#: rescales the subject inside its own input and the world landmarks jitter
#: with it. Measured on a six-dancer clip, the raw box centre moved 22.6 px
#: per frame on average and 125 px at p95 for a subject 225 px wide.
#: Median only by default. A low-pass here is a trap: the box is the crop the
#: estimator sees, so filtering it makes the crop lag a fast subject and the
#: extracted motion flattens. Measured on the six-dancer clip, a 4 Hz low-pass
#: cut bbox jitter from 125 to 43 px p95 but took reference fidelity from
#: 0.692 to 0.284 and motion std from 0.797 to 0.179. Spikes are noise; the
#: rest of the box trajectory is the dancer.
BBOX_MEDIAN_WINDOW = 5
BBOX_LOWPASS_HZ = 0.0


def smooth_bbox(bbox: np.ndarray, valid: np.ndarray, fps: float,
                window: int = BBOX_MEDIAN_WINDOW,
                cutoff: float = BBOX_LOWPASS_HZ) -> np.ndarray:
    """Median then low-pass each box coordinate, within valid runs only."""
    from scipy.ndimage import median_filter
    from scipy.signal import butter, filtfilt

    out = np.asarray(bbox, float).copy()
    from ..motion.ir import _segments_from_valid
    for lo, hi in _segments_from_valid(np.asarray(valid, bool)):
        n = hi - lo
        if n < 3:
            continue
        seg = out[lo:hi]
        w = min(window if window % 2 else window + 1, n if n % 2 else n - 1)
        if w >= 3:
            seg = median_filter(seg, size=(w, 1), mode="nearest")
        nyq = 0.5 * fps
        if cutoff > 0.0 and cutoff < nyq and n > 12:
            b, a = butter(2, cutoff / nyq, btype="low")
            seg = filtfilt(b, a, seg, axis=0)
        out[lo:hi] = seg
    return out


@dataclass
class RefineStats:
    attempted: int = 0
    recovered: int = 0
    rejected_drift: int = 0
    rejected_size: int = 0
    coverage_before: float = 0.0
    coverage_after: float = 0.0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _supported(name: str, kwargs: dict) -> dict:
    """Drop keyword arguments the registered detector does not take."""
    import inspect
    factory = DETECTORS.get(name)
    if factory is None:
        return {}
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in params}


def _predict(bbox: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Bounding box for every frame, interpolated across the invalid ones."""
    idx = np.flatnonzero(valid)
    if idx.size < 2:
        return bbox.copy()
    t = np.arange(len(valid))
    return np.stack([np.interp(t, idx, bbox[idx, k]) for k in range(4)], axis=1)


def _accept(box: np.ndarray, want: np.ndarray, stats: RefineStats) -> bool:
    cw, ch = want[2] - want[0], want[3] - want[1]
    size = max(cw, ch, 1e-6)
    dc = np.linalg.norm([(box[0] + box[2] - want[0] - want[2]) * 0.5,
                         (box[1] + box[3] - want[1] - want[3]) * 0.5])
    if dc > MAX_CENTRE_DRIFT * size:
        stats.rejected_drift += 1
        return False
    got = max(box[2] - box[0], box[3] - box[1], 1e-6)
    if not (1.0 / MAX_SIZE_RATIO) <= got / size <= MAX_SIZE_RATIO:
        stats.rejected_size += 1
        return False
    return True


def refine(video: str | Path, result: TrackResult, *, detector: str = "mediapipe",
           margin: float = ROI_MARGIN, scale: float = ROI_SCALE,
           **detector_kwargs) -> tuple[TrackResult, RefineStats]:
    """Fill in the subject's missing frames by re-detecting on a crop."""
    stats = RefineStats(coverage_before=result.coverage)
    missing = np.flatnonzero(~result.valid)
    if missing.size == 0 or result.valid.sum() < 2:
        stats.coverage_after = result.coverage
        return result, stats

    # Tuning knobs are specific to the MediaPipe detector; a registry can hold
    # others (the tests register a colour-blob one) that take none of them, so
    # only pass what the chosen detector actually accepts.
    wanted = {"max_people": 2, "min_detection_confidence": 0.2,
              "min_presence_confidence": 0.2, **detector_kwargs}
    det = get_detector(detector, **_supported(detector, wanted))
    pred = _predict(result.bbox, result.valid)
    bbox = result.bbox.copy()
    valid = result.valid.copy()

    cap = cv2.VideoCapture(str(video))
    try:
        for f in missing:
            x0, y0, x1, y1 = pred[f]
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            half = max(x1 - x0, y1 - y0) * (0.5 + margin)
            a = int(max(cx - half, 0)); b = int(max(cy - half, 0))
            c = int(min(cx + half, result.width)); d = int(min(cy + half, result.height))
            if c - a < 24 or d - b < 24:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, frame = cap.read()
            if not ok:
                continue
            crop = frame[b:d, a:c]
            if scale != 1.0:
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            stats.attempted += 1
            dets = det.detect_frame(crop)
            if not dets:
                continue
            best = max(dets, key=lambda x: x.score)
            box = np.asarray(best.bbox, float) / scale + np.array([a, b, a, b], float)
            if not _accept(box, pred[f], stats):
                continue
            bbox[f] = box
            valid[f] = True
            stats.recovered += 1
    finally:
        cap.release()

    bbox = smooth_bbox(bbox, valid, result.fps)
    out = TrackResult(video=result.video, fps=result.fps, width=result.width,
                      height=result.height, bbox=bbox, valid=valid,
                      shots=result.shots, assignments=result.assignments,
                      meta={**result.meta, "refine": stats.as_dict()})
    stats.coverage_after = out.coverage
    out.meta["refine"] = stats.as_dict()
    return out, stats
