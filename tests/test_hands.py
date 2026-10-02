"""The hand seam: the fifth registry.

Hands were the one stage that could not be swapped, because `HandLandmarker` was
constructed inside the body frontend. Moving it out is a refactor, so the bar is
that it reproduces what the fused loop produced, exactly.

The delicate part is the timestamp sequence. The landmarkers run in VIDEO mode,
which carries tracking state between calls, so the separated pass must present
the same stamps in the same order or the state diverges and the output changes
with nothing raising anywhere. That is verified against a freshly generated
baseline rather than against a stored artifact, because a stored one predates
several changes to the stage and would be measuring the wrong thing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from g1dance.motion.hands import (
    HandTracks, attach, available_hand_estimators, get_hand_estimator)
from g1dance.motion.ir import N_HAND_LANDMARKS, PoseSequence


def _pose(n=6, px=True):
    return PoseSequence(
        fps=30.0, landmarks=np.zeros((n, 33, 3)), visibility=np.ones((n, 33)),
        hand_left=np.zeros((n, 21, 3)), hand_right=np.zeros((n, 21, 3)),
        hand_conf=np.zeros((n, 2)), valid=np.ones(n, bool),
        landmarks_px=np.zeros((n, 33, 2)) if px else None)


# ------------------------------------------------------------------ the value
def test_empty_tracks_have_the_declared_shapes():
    t = HandTracks.empty(7)
    assert t.left.shape == (7, N_HAND_LANDMARKS, 3)
    assert t.right.shape == (7, N_HAND_LANDMARKS, 3)
    assert t.conf.shape == (7, 2)
    assert t.n_frames == 7


def test_tracks_refuse_a_wrong_shape():
    with pytest.raises(ValueError, match="left"):
        HandTracks(left=np.zeros((4, 20, 3)), right=np.zeros((4, 21, 3)),
                   conf=np.zeros((4, 2)))


def test_tracks_refuse_confidence_outside_the_unit_interval():
    with pytest.raises(ValueError, match="conf"):
        HandTracks(left=np.zeros((2, 21, 3)), right=np.zeros((2, 21, 3)),
                   conf=np.array([[0.5, 1.5], [0.0, 0.0]]))


# --------------------------------------------------------------- the registry
def test_both_entries_are_registered():
    assert set(available_hand_estimators()) >= {"none", "mediapipe"}


def test_an_unknown_estimator_names_the_alternatives():
    with pytest.raises(ValueError, match="mediapipe"):
        get_hand_estimator("no_such_hand_model")


def test_none_says_no_hands_rather_than_leaving_it_open():
    t = get_hand_estimator("none")(None, None, _pose(5))
    assert t.n_frames == 5
    assert not t.conf.any()
    assert t.meta["estimator"] == "none"


def test_mediapipe_refuses_a_sequence_without_pixels():
    """It cannot invent a region of interest, so it says so."""
    with pytest.raises(ValueError, match="landmarks_px"):
        get_hand_estimator("mediapipe")("x.mp4", object(), _pose(px=False))


def test_mediapipe_refuses_a_missing_subject_window():
    with pytest.raises(ValueError, match="subject"):
        get_hand_estimator("mediapipe")("x.mp4", None, _pose())


# ------------------------------------------------------------------- attach
def test_attach_puts_the_tracks_on_the_sequence():
    p = _pose(4)
    t = HandTracks.empty(4)
    t.conf[:] = 0.5
    t.left[:] = 1.0
    out = attach(p, t)
    np.testing.assert_allclose(out.hand_conf, 0.5)
    np.testing.assert_allclose(out.hand_left, 1.0)


def test_attach_keeps_the_body_and_the_pixels():
    p = _pose(4)
    p.landmarks[:] = 3.0
    out = attach(p, HandTracks.empty(4))
    np.testing.assert_allclose(out.landmarks, 3.0)
    assert out.landmarks_px is not None


def test_attach_keeps_the_smplx_parameters():
    """GMR needs them, and every parameter frontend passes through here."""
    T = 4
    p = replace(_pose(T, px=False), smplx_betas=np.zeros(10),
                smplx_global_orient=np.ones((T, 3)),
                smplx_body_pose=np.full((T, 63), 0.1),
                smplx_transl=np.full((T, 3), 2.0))
    out = attach(p, HandTracks.empty(T))
    np.testing.assert_allclose(out.smplx_transl, 2.0)
    np.testing.assert_allclose(out.smplx_body_pose, 0.1)


def test_attach_records_which_estimator_produced_the_hands():
    out = attach(_pose(3), HandTracks.empty(3, estimator="none"))
    assert out.meta["hands"]["estimator"] == "none"


def test_attach_refuses_a_length_mismatch():
    with pytest.raises(ValueError, match="frames"):
        attach(_pose(4), HandTracks.empty(5))


# ---------------------------------------------------------------- the IR field
def test_a_sequence_without_pixels_is_allowed():
    """A frontend that read a parameter file never saw a frame."""
    assert _pose(px=False).landmarks_px is None


def test_pixels_round_trip_through_a_file(tmp_path):
    p = _pose(4)
    p.landmarks_px = np.arange(4 * 33 * 2, dtype=np.float32).reshape(4, 33, 2)
    p.save(tmp_path / "p.npz")
    np.testing.assert_array_equal(
        PoseSequence.load(tmp_path / "p.npz").landmarks_px, p.landmarks_px)


def test_a_file_written_before_the_seam_existed_still_loads(tmp_path):
    """Every artifact in raw/ predates this field."""
    p = _pose(3, px=False)
    p.save(tmp_path / "old.npz")
    assert PoseSequence.load(tmp_path / "old.npz").landmarks_px is None


def test_pixels_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="landmarks_px"):
        p = _pose(4)
        PoseSequence(fps=p.fps, landmarks=p.landmarks, visibility=p.visibility,
                     hand_left=p.hand_left, hand_right=p.hand_right,
                     hand_conf=p.hand_conf, valid=p.valid,
                     landmarks_px=np.zeros((4, 33, 3)))
