"""The finger round trip, and the guarantee that fingers never break the run.

The same argument as tests/test_retarget.py applies one level down: a permuted
finger order or a left hand map applied to the right hand is invisible in a
rendered frame and fatal to the reference. So a random valid Inspire pose is run
through forward kinematics, MediaPipe-layout hand landmarks are synthesised at
the corresponding anatomical points, and the original six joint angles are
required back.

The coarse backend is asserted exactly, because it is defined as the inversion of
the model's own flexion curve and so has a right answer. DexPilot is asserted
only for ordering, range and monotonicity: it optimises the distances between
fingertips and the thumb under a closing prior, not joint angles, and it
deliberately does not reproduce a curl exactly. It is also optional, and the
tests below run with or without it, because the pipeline has to as well.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion import fingers as F
from g1dance.motion.ir import N_HAND_LANDMARKS
from g1dance.sim.model import FINGER_ORDER, HAND_JOINTS, G1Model
from test_retarget import (forward_kinematics, hand_points, random_valid_fingers,
                           random_valid_pose, synth_pose)

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def model() -> G1Model:
    return G1Model()


def landmarks_for(model: G1Model, hand: dict, side: str,
                  n_frames: int = 3) -> np.ndarray:
    d = forward_kinematics(model, model.default_qpos, hand)
    return np.tile(hand_points(model, d, side), (n_frames, 1, 1))


# --------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("side", SIDES)
def test_coarse_round_trip_recovers_all_six_dof(model, side):
    errors = []
    for seed in range(12):
        rng = np.random.default_rng(seed)
        hand = random_valid_fingers(model, rng)
        lm = landmarks_for(model, hand, side)
        out = F.coarse_fingers(lm, np.ones(len(lm)), side, model)
        errors.append(np.abs(out[-1] - hand[side]))
    errors = np.asarray(errors)
    summary = (f"{side} finger error: median {np.median(errors):.2e} "
               f"max {errors.max():.2e} rad, per DoF "
               f"{dict(zip(FINGER_ORDER, np.round(errors.max(0), 6)))}")
    assert errors.max() < 1e-4, summary


@pytest.mark.parametrize("side", SIDES)
def test_a_left_hand_map_is_not_reused_for_the_right(model, side):
    """The two hands mirror, so the same landmarks must not give the same angles
    unless the pose really is symmetric. This is the cheap guard against a
    copy-paste of the left branch into the right."""
    rng = np.random.default_rng(1)
    hand = random_valid_fingers(model, rng)
    lm = landmarks_for(model, hand, side)
    other = "right" if side == "left" else "left"
    right_way = F.coarse_fingers(lm, np.ones(len(lm)), side, model)[-1]
    wrong_way = F.coarse_fingers(lm, np.ones(len(lm)), other, model)[-1]
    assert np.abs(right_way - hand[side]).max() < 1e-4
    assert np.abs(wrong_way - hand[side]).max() > 0.05


def test_flexion_calibration_is_monotone_and_spans_the_range(model):
    """The coarse map inverts this curve, so a fold in it would be a silent
    two-to-one mapping from human curl to joint angle."""
    for side in SIDES:
        kin = F.hand_kinematics(model, side)
        for i, finger in enumerate(F.CURL_FINGERS):
            flex, qs = kin.calibration[finger]
            assert (np.diff(flex) > 0).all(), (side, finger)
            assert qs.min() == pytest.approx(kin.ranges[i, 0])
            assert qs.max() == pytest.approx(kin.ranges[i, 1])
            # a closed Inspire finger turns through more than pi, which is why
            # the flexion measure is a sum of two signed angles
            assert flex.max() - flex.min() > np.pi


def test_finger_order_matches_the_model(model):
    assert HAND_JOINTS == [f"{p}_{j}_joint" for p in ("L", "R")
                           for j in FINGER_ORDER]
    for side in SIDES:
        kin = F.hand_kinematics(model, side)
        prefix = "L" if side == "left" else "R"
        assert kin.joints == [f"{prefix}_{j}_joint" for j in FINGER_ORDER]
    assert F.CURL_FINGERS == tuple(n.split("_")[0] for n in FINGER_ORDER[:4])


def test_mimic_coefficients_come_from_the_model(model):
    """mj_kinematics ignores equality constraints, so the coupled joints are
    driven by hand and the ratios must be read from the model, not assumed."""
    mimic = F.mimic_coefficients(model)
    assert mimic["L_index_intermediate_joint"][0] == "L_index_proximal_joint"
    assert mimic["L_index_intermediate_joint"][1][1] == pytest.approx(1.0)
    assert mimic["R_thumb_distal_joint"][0] == "R_thumb_proximal_pitch_joint"
    assert mimic["R_thumb_distal_joint"][1][1] == pytest.approx(2.4)
    assert len(mimic) == 12


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------

def test_low_confidence_falls_back_to_the_neutral_hand(model):
    """Hand tracking drops out constantly in real footage. A dropout has to
    relax the hand, not freeze whatever the detector produced last."""
    rng = np.random.default_rng(2)
    hand = random_valid_fingers(model, rng)
    lm = landmarks_for(model, hand, "left", n_frames=6)
    kin = F.hand_kinematics(model, "left")
    conf = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    out = F.coarse_fingers(lm, conf, "left", model)
    assert np.allclose(out[2], kin.neutral())
    assert np.abs(out[0] - hand["left"]).max() < 1e-4


def test_non_finite_landmarks_do_not_propagate(model):
    lm = landmarks_for(model, random_valid_fingers(model, np.random.default_rng(3)),
                       "right", n_frames=4)
    lm[1, 5] = np.nan
    out = F.coarse_fingers(lm, np.ones(4), "right", model)
    assert np.isfinite(out).all()
    assert np.allclose(out[1], F.hand_kinematics(model, "right").neutral())


def test_pipeline_works_without_dex_retargeting(model, monkeypatch):
    monkeypatch.setattr(F, "dex_available", lambda *a, **k: False)
    pose = _pose_with_hands(model)
    out, backend = F.retarget_fingers(pose, model, {"finger_backend": "auto"})
    assert backend == "coarse"
    assert out.shape == (pose.n_frames, len(HAND_JOINTS))


def test_dexpilot_is_preferred_when_available(model):
    pose = _pose_with_hands(model)
    _, backend = F.retarget_fingers(pose, model, {"finger_backend": "auto"})
    assert backend == ("dexpilot" if F.dex_available() else "coarse")


# --------------------------------------------------------------------------
# the whole-sequence entry point
# --------------------------------------------------------------------------

def _pose_with_hands(model: G1Model, n_frames: int = 6):
    rng = np.random.default_rng(4)
    return synth_pose(model, random_valid_pose(model, rng), n_frames=n_frames,
                      hand=random_valid_fingers(model, rng))


def test_retarget_fingers_fills_both_hands_in_model_order(model):
    rng = np.random.default_rng(5)
    hand = random_valid_fingers(model, rng)
    pose = synth_pose(model, random_valid_pose(model, rng), n_frames=6, hand=hand)
    out, backend = F.retarget_fingers(pose, model, {"finger_backend": "coarse"})
    assert backend == "coarse"
    assert out.shape == (6, len(HAND_JOINTS))
    assert np.abs(out[-1, :6] - hand["left"]).max() < 1e-3
    assert np.abs(out[-1, 6:] - hand["right"]).max() < 1e-3


def test_targets_stay_inside_the_joint_limits(model):
    pose = _pose_with_hands(model)
    out, _ = F.retarget_fingers(pose, model, {"finger_backend": "auto"})
    for h, side in enumerate(SIDES):
        r = F.hand_kinematics(model, side).ranges
        block = out[:, 6 * h: 6 * (h + 1)]
        assert (block >= r[:, 0] - 1e-9).all() and (block <= r[:, 1] + 1e-9).all()


def test_missing_hands_give_a_neutral_hand(model):
    rng = np.random.default_rng(6)
    pose = synth_pose(model, random_valid_pose(model, rng), n_frames=6)
    assert (pose.hand_conf == 0).all()
    out, _ = F.retarget_fingers(pose, model, {"finger_backend": "coarse"})
    expect = np.concatenate([F.hand_kinematics(model, s).neutral() for s in SIDES])
    assert np.abs(out - expect).max() < 1e-6


# --------------------------------------------------------------------------
# dex-retargeting, when it is installed
# --------------------------------------------------------------------------

@pytest.mark.skipif(not F.dex_available(),
                    reason="dex-retargeting or its Inspire URDFs are not present")
@pytest.mark.parametrize("side", SIDES)
def test_dexpilot_is_ordered_bounded_and_monotone_in_the_curl(model, side):
    """Not an exact round trip: DexPilot solves fingertip-to-thumb distances
    under a closing prior, so it under curls a synthetic hand by design. What has
    to hold is that its six outputs land in our six slots, inside our limits, and
    move the right way."""
    kin = F.hand_kinematics(model, side)
    out = []
    for curl in (0.0, 0.8, 1.6):
        hand = {s: np.array([curl, curl, curl, curl, 0.3, 0.6]) for s in SIDES}
        lm = landmarks_for(model, hand, side)
        out.append(F.dexpilot_fingers(lm, np.ones(len(lm)), side, model)[-1])
    out = np.asarray(out)
    assert (out >= kin.ranges[:, 0] - 1e-6).all()
    assert (out <= kin.ranges[:, 1] + 1e-6).all()
    for i in range(4):                       # pinky, ring, middle, index
        assert (np.diff(out[:, i]) > 0.05).all(), (FINGER_ORDER[i], out[:, i])


@pytest.mark.skipif(not F.dex_available(),
                    reason="dex-retargeting or its Inspire URDFs are not present")
def test_dexpilot_joint_names_map_by_name_not_by_position(model):
    retgt, _ = F._dex_retargeting("left", str(F.ASSET_HANDS))
    for joint in FINGER_ORDER:
        assert f"{joint}_joint" in retgt.joint_names


def test_hand_frame_is_equivariant(model):
    """Both sides of the finger map build this frame, one from landmarks and one
    from the model. A rotated hand must give the rotated frame or the two are not
    comparable."""
    rng = np.random.default_rng(7)
    pts = rng.normal(size=(4, 3))
    frame = F.hand_frame(*pts)
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    rot *= np.sign(np.linalg.det(rot))
    assert np.allclose(F.hand_frame(*(pts @ rot.T)), rot @ frame, atol=1e-9)
    assert np.allclose(frame.T @ frame, np.eye(3), atol=1e-9)


def test_landmark_layout_matches_mediapipe(model):
    assert F.MP_HAND["thumb"] == (1, 2, 3, 4)
    assert F.MP_HAND["pinky"] == (17, 18, 19, 20)
    assert max(max(v) for v in F.MP_HAND.values()) == N_HAND_LANDMARKS - 1
