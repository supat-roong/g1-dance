"""Tests for motion cleanup.

Every case here is built from synthetic trajectories with a known answer,
because the defects this module removes are the ones nobody can see in a
rendered preview. A filter that quietly smears one shot into the next looks
fine frame by frame and only shows up as a policy that learned a move the
dancer never made.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.clean import (
    clean, fill_gaps, lowpass, reject_outliers, resample)
from g1dance.motion.ir import MP_POSE, N_HAND_LANDMARKS, N_POSE_LANDMARKS, PoseSequence

WRIST = MP_POSE["left_wrist"]


def arm_swing(n_frames: int, fps: float, swing_hz: float = 1.0,
              noise_hz: float = 0.0, noise_amp: float = 0.0,
              valid: np.ndarray | None = None) -> PoseSequence:
    """A sine driven left arm swing, optionally with a high frequency ripple."""
    t = np.arange(n_frames, dtype=np.float64) / fps
    swing = 0.3 * np.sin(2 * np.pi * swing_hz * t)
    if noise_amp:
        swing = swing + noise_amp * np.sin(2 * np.pi * noise_hz * t)

    landmarks = np.zeros((n_frames, N_POSE_LANDMARKS, 3), np.float32)
    landmarks[:, MP_POSE["left_shoulder"]] = (0.2, 0.5, 0.0)
    landmarks[:, MP_POSE["right_shoulder"]] = (-0.2, 0.5, 0.0)
    landmarks[:, WRIST, 0] = 0.2
    landmarks[:, WRIST, 1] = 0.1 + swing
    landmarks[:, WRIST, 2] = 0.0

    hand = np.zeros((n_frames, N_HAND_LANDMARKS, 3), np.float32)
    hand[:, 8, 1] = swing
    return PoseSequence(
        fps=fps, landmarks=landmarks,
        visibility=np.ones((n_frames, N_POSE_LANDMARKS), np.float32),
        hand_left=hand, hand_right=hand.copy(),
        hand_conf=np.ones((n_frames, 2), np.float32),
        valid=np.ones(n_frames, bool) if valid is None else np.asarray(valid, bool))


def analytic(times: np.ndarray, swing_hz: float = 1.0) -> np.ndarray:
    return 0.1 + 0.3 * np.sin(2 * np.pi * swing_hz * times)


# --------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------

def test_resample_changes_length_and_keeps_the_duration():
    seq = arm_swing(101, 25.0)
    out = resample(seq, 50.0)
    assert out.fps == 50.0
    # the grid spans the same duration, so endpoints coincide exactly
    assert out.n_frames == int(np.floor((101 - 1) * 50.0 / 25.0)) + 1 == 201
    assert out.valid.all()


def test_resample_down_to_50_from_60():
    seq = arm_swing(121, 60.0)
    out = resample(seq, 50.0)
    assert out.fps == 50.0
    assert out.n_frames == int(np.floor(120 * 50.0 / 60.0)) + 1 == 101


def test_resample_preserves_trajectory_shape():
    seq = arm_swing(101, 25.0, swing_hz=1.0)
    out = resample(seq, 50.0)
    t_new = np.arange(out.n_frames) / out.fps
    got = out.landmarks[:, WRIST, 1]
    assert np.max(np.abs(got - analytic(t_new))) < 0.01
    # endpoints land on the original samples, not somewhere between them
    assert got[0] == pytest.approx(seq.landmarks[0, WRIST, 1], abs=1e-6)
    assert got[-1] == pytest.approx(seq.landmarks[-1, WRIST, 1], abs=1e-6)


def test_resample_to_the_same_rate_is_the_identity():
    seq = arm_swing(60, 50.0)
    out = resample(seq, 50.0)
    assert np.array_equal(out.landmarks, seq.landmarks)
    assert out.n_frames == seq.n_frames


def test_resample_does_not_bridge_an_invalid_run():
    valid = np.ones(101, bool)
    valid[40:60] = False
    seq = arm_swing(101, 25.0, valid=valid)
    out = resample(seq, 50.0)
    t_new = np.arange(out.n_frames) / out.fps
    # every output frame inside the hole is invalid, none of it interpolated
    hole = (t_new > 39 / 25.0) & (t_new < 60 / 25.0)
    assert not out.valid[hole].any()
    assert out.valid[t_new < 39 / 25.0].all()
    assert np.allclose(out.landmarks[hole], 0.0)


# --------------------------------------------------------------------------
# low pass
# --------------------------------------------------------------------------

def test_lowpass_removes_high_frequency_noise_and_keeps_the_signal():
    fps, n = 60.0, 360
    clean_seq = arm_swing(n, fps, swing_hz=1.0)
    noisy = arm_swing(n, fps, swing_hz=1.0, noise_hz=20.0, noise_amp=0.03)
    out = lowpass(noisy, cutoff_hz=6.0)

    interior = slice(30, n - 30)             # skip the filter's own transient
    truth = clean_seq.landmarks[interior, WRIST, 1]
    before = np.sqrt(np.mean((noisy.landmarks[interior, WRIST, 1] - truth) ** 2))
    after = np.sqrt(np.mean((out.landmarks[interior, WRIST, 1] - truth) ** 2))
    assert before > 0.015
    assert after < 0.002
    assert after < before / 10
    # the 1 Hz component survives with its amplitude intact
    assert np.ptp(out.landmarks[interior, WRIST, 1]) == pytest.approx(0.6, abs=0.02)


def test_lowpass_filters_the_hands_too():
    n = 360
    out = lowpass(arm_swing(n, 60.0, noise_hz=20.0, noise_amp=0.03), cutoff_hz=6.0)
    truth = arm_swing(n, 60.0).hand_left[30:-30, 8, 1]
    assert np.sqrt(np.mean((out.hand_left[30:-30, 8, 1] - truth) ** 2)) < 0.002


def test_lowpass_does_not_smear_across_a_segment_boundary():
    """Two segments holding different constants must stay at their constants."""
    n = 240
    valid = np.ones(n, bool)
    valid[118:122] = False                   # the cut
    seq = arm_swing(n, 60.0, valid=valid)
    lm = seq.landmarks.copy()
    lm[:, WRIST, 1] = 0.0
    lm[122:, WRIST, 1] = 1.0                 # a step across the cut
    seq = PoseSequence(fps=seq.fps, landmarks=lm, visibility=seq.visibility,
                       hand_left=seq.hand_left, hand_right=seq.hand_right,
                       hand_conf=seq.hand_conf, valid=valid)
    out = lowpass(seq, cutoff_hz=6.0)
    assert np.max(np.abs(out.landmarks[:118, WRIST, 1])) < 1e-5
    assert np.max(np.abs(out.landmarks[122:, WRIST, 1] - 1.0)) < 1e-5


def test_lowpass_leaves_segments_shorter_than_the_transient_alone():
    # a 4th order Butterworth needs 16 samples before filtfilt has room to pad
    n = 12
    seq = arm_swing(n, 60.0, noise_hz=20.0, noise_amp=0.03)
    out = lowpass(seq, cutoff_hz=6.0)
    assert np.array_equal(out.landmarks, seq.landmarks)
    assert out.meta["lowpass_frames"] == 0


def test_lowpass_rejects_a_cutoff_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist|must be in"):
        lowpass(arm_swing(120, 10.0), cutoff_hz=8.0)


# --------------------------------------------------------------------------
# gap filling
# --------------------------------------------------------------------------

def test_fill_gaps_fills_short_holes_and_leaves_long_ones():
    n, fps = 200, 50.0
    valid = np.ones(n, bool)
    valid[50:53] = False                     # 3 frames = 60 ms, short
    valid[100:130] = False                   # 30 frames = 600 ms, long
    seq = arm_swing(n, fps, valid=valid)
    out = fill_gaps(seq, max_gap_s=0.2)      # 10 frames at 50 fps

    assert out.valid[50:53].all()
    assert not out.valid[100:130].any()
    assert out.meta["gap_filled_frames"] == 3
    t = np.arange(n) / fps
    assert np.max(np.abs(out.landmarks[50:53, WRIST, 1] - analytic(t[50:53]))) < 0.01


def test_fill_gaps_interpolates_linearly_between_the_bracketing_frames():
    n = 20
    seq = arm_swing(n, 50.0)
    lm = seq.landmarks.copy()
    lm[:, WRIST, 1] = np.arange(n)           # a ramp with an obvious answer
    valid = np.ones(n, bool)
    valid[10:13] = False
    seq = PoseSequence(fps=50.0, landmarks=lm, visibility=seq.visibility,
                       hand_left=seq.hand_left, hand_right=seq.hand_right,
                       hand_conf=seq.hand_conf, valid=valid)
    out = fill_gaps(seq, max_gap_s=0.2)
    assert np.allclose(out.landmarks[10:13, WRIST, 1], [10.0, 11.0, 12.0], atol=1e-5)


def test_fill_gaps_does_not_extrapolate_at_the_ends():
    n = 100
    valid = np.ones(n, bool)
    valid[:4] = False
    valid[-4:] = False
    out = fill_gaps(arm_swing(n, 50.0, valid=valid), max_gap_s=0.2)
    assert not out.valid[:4].any()
    assert not out.valid[-4:].any()
    assert out.meta["gap_filled_frames"] == 0


# --------------------------------------------------------------------------
# outlier rejection
# --------------------------------------------------------------------------

def test_reject_outliers_drops_a_teleporting_landmark():
    n = 100
    seq = arm_swing(n, 50.0)
    lm = seq.landmarks.copy()
    lm[42, WRIST, 0] += 3.0                  # 150 m/s at 50 fps
    seq = PoseSequence(fps=50.0, landmarks=lm, visibility=seq.visibility,
                       hand_left=seq.hand_left, hand_right=seq.hand_right,
                       hand_conf=seq.hand_conf, valid=seq.valid)
    out = reject_outliers(seq, max_speed=12.0)
    assert not out.valid[42]
    assert out.valid.sum() == n - 1
    assert out.meta["outlier_frames"] == 1


def test_reject_outliers_keeps_a_fast_but_plausible_swing():
    seq = arm_swing(200, 50.0, swing_hz=3.0)   # peak wrist speed ~5.7 m/s
    assert reject_outliers(seq, max_speed=12.0).valid.all()


def test_reject_outliers_ignores_landmarks_the_detector_did_not_see():
    n = 60
    seq = arm_swing(n, 50.0)
    lm, vis = seq.landmarks.copy(), seq.visibility.copy()
    vis[:, 5] = 0.0                          # never seen
    lm[30, 5] = (9.0, 9.0, 9.0)              # so its jump must not count
    seq = PoseSequence(fps=50.0, landmarks=lm, visibility=vis,
                       hand_left=seq.hand_left, hand_right=seq.hand_right,
                       hand_conf=seq.hand_conf, valid=seq.valid)
    assert reject_outliers(seq, max_speed=12.0).valid.all()


# --------------------------------------------------------------------------
# the whole pass
# --------------------------------------------------------------------------

def test_clean_pipeline_lands_at_50_hz_with_the_signal_intact():
    n, fps = 480, 60.0
    valid = np.ones(n, bool)
    valid[200:203] = False
    seq = arm_swing(n, fps, swing_hz=1.0, noise_hz=20.0, noise_amp=0.03,
                    valid=valid)
    out = clean(seq, max_gap_s=0.2, cutoff_hz=6.0, target_fps=50.0)

    assert out.fps == 50.0
    assert out.n_frames == int(np.floor((n - 1) * 50.0 / fps)) + 1
    assert out.valid.mean() > 0.99           # the short hole was filled
    t = np.arange(out.n_frames) / out.fps
    interior = slice(40, out.n_frames - 40)
    err = out.landmarks[interior, WRIST, 1] - analytic(t[interior])
    assert np.sqrt(np.mean(err ** 2)) < 0.01


def test_clean_keeps_a_long_hole_as_two_segments():
    n = 480
    valid = np.ones(n, bool)
    valid[200:260] = False
    out = clean(arm_swing(n, 60.0, valid=valid), max_gap_s=0.2, target_fps=50.0)
    assert len(out.segments) == 2
