"""Gap filling, outlier rejection, filtering and resampling for a PoseSequence.

Monocular landmarks arrive with three defects that all become invisible once
they reach the trainer: single frame dropouts where the detector lost the
subject, velocity spikes where it snapped to the wrong limb, and per frame
jitter that a position tracking reward faithfully reproduces as a tremor. Each
gets one pass here, in that order, because filling a gap before rejecting the
spike that caused it would interpolate through the spike.

The rule the whole module is built around is that nothing crosses a segment
boundary. `valid` marks where the tracker actually had the subject, and the
frames either side of an invalid run are usually either side of a cut. A filter
run over the concatenated array smears the end of one shot into the start of
the next, producing a motion the dancer never performed, at exactly the moments
a reviewer is least likely to look. So every operation here works on one
contiguous valid run at a time.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.signal import butter, filtfilt

from .ir import PoseSequence, _segments_from_valid

#: control rate the RL environment runs at; see design section 7.2
TARGET_FPS = 50.0

#: fields carrying (T, ..., 3) positions, the things worth filtering
_POSITION_FIELDS = ("landmarks", "hand_left", "hand_right")
#: fields carried along by interpolation but never filtered, because smoothing
#: a confidence turns a clean "not seen" into a soft "half seen"
_SCALAR_FIELDS = ("visibility", "hand_conf")
_ALL_FIELDS = _POSITION_FIELDS + _SCALAR_FIELDS


def _interp_axis(values: np.ndarray, x_new: np.ndarray,
                 x_old: np.ndarray) -> np.ndarray:
    """np.interp over the leading axis of an arbitrarily shaped array."""
    flat = values.reshape(len(values), -1)
    out = np.empty((len(x_new), flat.shape[1]), np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(x_new, x_old, flat[:, c])
    return out.reshape((len(x_new),) + values.shape[1:])


def fill_gaps(seq: PoseSequence, max_gap_s: float = 0.2) -> PoseSequence:
    """Interpolate across invalid runs no longer than `max_gap_s`.

    Only interior gaps are filled: a leading or trailing run has nothing to
    interpolate from and extrapolating a dancer's limbs is fiction. Longer gaps
    are left invalid on purpose, so they split the motion into segments that
    the rest of the pipeline keeps apart.
    """
    max_gap = max(0, int(round(max_gap_s * seq.fps)))
    valid = seq.valid.copy()
    if max_gap == 0 or valid.all() or not valid.any():
        return replace(seq, valid=valid, meta=dict(seq.meta))

    arrays = {name: getattr(seq, name).copy() for name in _ALL_FIELDS}
    filled = 0
    first, last = int(np.argmax(valid)), len(valid) - 1 - int(np.argmax(valid[::-1]))
    i = first
    while i <= last:
        if valid[i]:
            i += 1
            continue
        j = i
        while j <= last and not valid[j]:
            j += 1
        if j - i <= max_gap:
            lo, hi = i - 1, j                 # both valid by construction
            w = (np.arange(i, j, dtype=np.float32) - lo) / float(hi - lo)
            for name, arr in arrays.items():
                shape = (len(w),) + (1,) * (arr.ndim - 1)
                a, b = arr[lo], arr[hi]
                arr[i:j] = a + (b - a) * w.reshape(shape)
            valid[i:j] = True
            filled += j - i
        i = j

    meta = dict(seq.meta)
    meta["gap_filled_frames"] = int(filled)
    meta["gap_fill_max_frames"] = max_gap
    return replace(seq, valid=valid, meta=meta, **arrays)


def reject_outliers(seq: PoseSequence, max_speed: float = 12.0,
                    min_visibility: float = 0.3) -> PoseSequence:
    """Invalidate frames where a visible landmark teleports.

    A real wrist peaks around 8 m/s in fast choreography; a detector that has
    snapped to the other dancer's arm moves it metres in one frame. Speed is
    measured only against the previous valid frame, so a spike is not hidden by
    averaging across a gap, and only landmarks the detector claimed to see are
    judged, since an unseen landmark sits at the origin and would register as a
    jump every time it reappears.
    """
    valid = seq.valid.copy()
    if valid.sum() < 2:
        return replace(seq, valid=valid, meta=dict(seq.meta))
    dt = 1.0 / seq.fps
    lm, vis = seq.landmarks, seq.visibility
    rejected = 0
    prev = None
    for t in range(len(valid)):
        if not valid[t]:
            prev = None                        # a gap resets the reference
            continue
        if prev is None:
            prev = t
            continue
        seen = (vis[t] >= min_visibility) & (vis[prev] >= min_visibility)
        if seen.any():
            gap = (t - prev) * dt
            speed = np.linalg.norm(lm[t, seen] - lm[prev, seen], axis=-1) / gap
            if float(speed.max()) > max_speed:
                valid[t] = False
                rejected += 1
                continue
        prev = t
    meta = dict(seq.meta)
    meta["outlier_frames"] = int(rejected)
    meta["outlier_max_speed"] = float(max_speed)
    return replace(seq, valid=valid, meta=meta)


def lowpass(seq: PoseSequence, cutoff_hz: float = 6.0, order: int = 4
            ) -> PoseSequence:
    """Zero phase Butterworth low pass, applied per valid segment.

    `filtfilt` is used rather than `lfilter` because a causal filter delays the
    motion, and a reference motion that lags the music is a different dance.
    Human limb motion lives below about 6 Hz; above that is detector jitter.

    Segments too short for the filter's transient are passed through untouched.
    Padding them out to make the maths work would invent the very samples the
    segment boundary says we do not have.
    """
    nyquist = 0.5 * seq.fps
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff {cutoff_hz} Hz must be in (0, {nyquist}) for {seq.fps} fps")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    min_len = 3 * max(len(a), len(b)) + 1

    arrays = {name: getattr(seq, name).copy() for name in _POSITION_FIELDS}
    filtered = 0
    for start, end in seq.segments:
        n = end - start
        if n < min_len:
            continue
        padlen = min(3 * max(len(a), len(b)), n - 1)
        for arr in arrays.values():
            chunk = arr[start:end]
            flat = chunk.reshape(n, -1)
            arr[start:end] = filtfilt(
                b, a, flat, axis=0, padlen=padlen).reshape(chunk.shape)
        filtered += n

    meta = dict(seq.meta)
    meta["lowpass_hz"] = float(cutoff_hz)
    meta["lowpass_order"] = int(order)
    meta["lowpass_frames"] = int(filtered)
    return replace(seq, meta=meta, **arrays)


def resample(seq: PoseSequence, target_fps: float = TARGET_FPS) -> PoseSequence:
    """Resample onto a uniform `target_fps` grid, segment by segment.

    The output grid spans exactly the input's duration, so the first and last
    output samples land on the first and last input samples and a resample to
    the source rate is the identity. Output frames are valid only where they
    fall inside a valid input segment, which is what keeps interpolation from
    bridging a cut.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    T = seq.n_frames
    if T < 2 or float(target_fps) == float(seq.fps):
        return replace(seq, fps=float(target_fps), meta=dict(seq.meta))

    ratio = float(target_fps) / float(seq.fps)
    T_new = int(np.floor((T - 1) * ratio)) + 1
    t_new = np.arange(T_new, dtype=np.float64) / float(target_fps)

    out = {name: np.zeros((T_new,) + getattr(seq, name).shape[1:], np.float32)
           for name in _ALL_FIELDS}
    valid = np.zeros(T_new, bool)
    for start, end in _segments_from_valid(seq.valid, min_len=1):
        x_old = np.arange(start, end, dtype=np.float64) / float(seq.fps)
        if end - start == 1:
            continue                       # nothing to interpolate between
        sel = (t_new >= x_old[0] - 1e-9) & (t_new <= x_old[-1] + 1e-9)
        if not sel.any():
            continue
        x_sel = np.clip(t_new[sel], x_old[0], x_old[-1])
        for name in _ALL_FIELDS:
            out[name][sel] = _interp_axis(
                getattr(seq, name)[start:end], x_sel, x_old)
        valid[sel] = True

    meta = dict(seq.meta)
    meta["resampled_from_fps"] = float(seq.fps)
    return PoseSequence(fps=float(target_fps), valid=valid, meta=meta, **out)


def clean(seq: PoseSequence, *, max_gap_s: float = 0.2,
          max_speed: float = 12.0, cutoff_hz: float = 6.0, order: int = 4,
          target_fps: float | None = TARGET_FPS) -> PoseSequence:
    """The full pass, in the only order that makes sense.

    Reject first so a spike is never interpolated through, fill next so the
    filter sees long unbroken segments, filter before resampling so the low
    pass runs at the source rate where the jitter actually is, and resample
    last onto the control rate the environment steps at.
    """
    seq = reject_outliers(seq, max_speed=max_speed)
    seq = fill_gaps(seq, max_gap_s=max_gap_s)
    seq = lowpass(seq, cutoff_hz=cutoff_hz, order=order)
    if target_fps is not None:
        seq = resample(seq, target_fps=target_fps)
    return seq


__all__ = ["TARGET_FPS", "clean", "fill_gaps", "lowpass", "reject_outliers",
           "resample"]
