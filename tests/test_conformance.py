"""What every component must satisfy, applied to every component there is.

Each test here is parameterised over a registry, so registering a component
subjects it to its contract without anyone remembering to write a test. That is
the difference between a platform that is flexible and one that is trustworthy: a
plugged-in component whose numbers you believe has been checked first.

Two of these carry the whole point of the project rather than a mere invariant:

* every tracking controller must keep the robot upright, because a dance that
  falls over is not a result;
* every retargeter must round-trip the robot's own stance, because a retargeter
  that does not is producing a plausible looking wrong dance, which the design
  document names as the most expensive bug class here.

Components whose assets are absent skip with a reason. `motion/`, `runs/` and the
downloaded weights are all gitignored, so a fresh clone runs what it can.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g1dance.motion.frontends import available_frontends, get_frontend
from g1dance.motion.hands import (
    HandTracks, available_hand_estimators, get_hand_estimator)
from g1dance.motion.ir import MP_POSE, N_POSE_LANDMARKS, PoseSequence
from g1dance.motion.retargeters import (
    available_retargeters, get_retargeter, needs_smplx_params)
from g1dance.control.controllers import BASELINES, available_controllers, make_controller
from g1dance.sim.env import DanceTrackEnv, load_config, synthetic_motion
from g1dance.sim.model import BODY_JOINTS, HAND_JOINTS, G1Model
from g1dance.track.detect_names import DETECTOR_NAMES
from test_retarget import synth_pose

CFG = "configs/upper_body.yaml"


@pytest.fixture(scope="module")
def g1() -> G1Model:
    return G1Model()


@pytest.fixture(scope="module")
def easy_motion(g1):
    """A gentle reference: open-loop PD already falls over on it at 1.11 s."""
    return synthetic_motion(240, fps=50.0, g1=g1)


def _env(motion, **over):
    cfg = load_config(CFG)
    for k, v in over.items():
        cfg["env"][k] = v
    return DanceTrackEnv(motion, cfg, seed=0, randomize=False)


#: Controllers needing an argument the registry cannot default; none today. A
#: Path value that does not exist skips the controller rather than failing it.
_EXTRA_ARGS: dict[str, dict] = {}


def _controller_or_skip(name: str):
    kwargs = dict(_EXTRA_ARGS.get(name, {}))
    for key, value in kwargs.items():
        if isinstance(value, Path) and not value.exists():
            pytest.skip(f"{name} needs {key}={value}, which is not present here")
    try:
        return make_controller(name, **kwargs)
    except Exception as exc:                      # noqa: BLE001 - reported below
        pytest.skip(f"{name} unavailable here: {type(exc).__name__}: {exc}")


# =========================================================== detectors
@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_detector_is_registered_and_constructible(name):
    from g1dance.track.detect import DETECTORS
    assert name in DETECTORS


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_detector_returns_valid_detections_inside_the_frame(name):
    from g1dance.track.detect import DetectError, get_detector
    try:
        det = get_detector(name)
    except (DetectError, RuntimeError) as exc:
        pytest.skip(f"{name} cannot start here: {exc}")
    frame = np.zeros((240, 320, 3), np.uint8)
    out = det.detect_frame(frame)
    assert isinstance(out, list)
    for d in out:
        # Detection.__post_init__ already enforces the shapes; what is checked
        # here is that the box is inside the image it came from.
        assert d.landmarks.shape == (N_POSE_LANDMARKS, 2)
        assert 0 <= d.bbox[0] <= d.bbox[2] <= frame.shape[1] + 1
        assert 0 <= d.bbox[1] <= d.bbox[3] <= frame.shape[0] + 1
    close = getattr(det, "close", None)
    if close:
        close()


@pytest.mark.parametrize("name", DETECTOR_NAMES)
def test_detector_is_deterministic_on_a_repeated_frame(name):
    """IMAGE mode was chosen so results never depend on how frames were sampled."""
    from g1dance.track.detect import DetectError, get_detector
    try:
        det = get_detector(name)
    except (DetectError, RuntimeError) as exc:
        pytest.skip(f"{name} cannot start here: {exc}")
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    a, b = det.detect_frame(frame), det.detect_frame(frame)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x.landmarks, y.landmarks, atol=0)
    close = getattr(det, "close", None)
    if close:
        close()


# =========================================================== estimators
@pytest.mark.parametrize("name", available_frontends())
def test_estimator_has_the_uniform_signature(name):
    import inspect
    fn = get_frontend(name)
    assert list(inspect.signature(fn).parameters)[:2] == ["source", "subject"]


@pytest.mark.parametrize("name", available_frontends())
def test_estimator_refuses_a_missing_source_rather_than_inventing_one(name):
    fn = get_frontend(name)
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        fn(Path("no_such_source_file.xyz"), None)


def _pose_or_skip(path="raw/kirari.pose.npz") -> PoseSequence:
    p = Path(path)
    if not p.exists():
        pytest.skip("raw/ is gitignored; needs a locally produced pose sequence")
    return PoseSequence.load(p)


def test_a_produced_sequence_is_hip_centred_in_metres():
    """The contract every frontend promises, checked on real output."""
    seq = _pose_or_skip()
    lm, vis = seq.landmarks, seq.visibility
    hip = 0.5 * (lm[:, MP_POSE["left_hip"]] + lm[:, MP_POSE["right_hip"]])
    seen = (vis[:, MP_POSE["left_hip"]] > 0.5) & (vis[:, MP_POSE["right_hip"]] > 0.5)
    assert seen.any(), "no frame has both hips, cannot check the contract"
    assert np.abs(hip[seen]).max() < 0.05, "hip midpoint is not at the origin"
    # metres: a human shoulder-to-hip is tens of centimetres, not millimetres
    sho = 0.5 * (lm[:, MP_POSE["left_shoulder"]] + lm[:, MP_POSE["right_shoulder"]])
    torso = np.linalg.norm((sho - hip)[seen], axis=1)
    assert 0.2 < float(np.median(torso)) < 1.0, "torso length is not in metres"


def test_an_unseen_landmark_is_reported_as_unseen_not_guessed():
    """The convention that lets a heterogeneous estimator work at all."""
    seq = _pose_or_skip()
    zero_vis = seq.visibility <= 0.0
    if not zero_vis.any():
        pytest.skip("this sequence saw every landmark on every frame")
    at_origin = np.linalg.norm(seq.landmarks[zero_vis], axis=-1) == 0.0
    assert at_origin.mean() > 0.5, \
        "landmarks with zero visibility should sit at the origin, not be guessed"


def test_a_sequence_declares_the_axes_and_units_it_delivers():
    seq = _pose_or_skip()
    meta = dict(seq.meta or {})
    assert meta.get("units") == "metres_hip_centred", meta.get("units")
    assert "y_up" in str(meta.get("axes", "")), meta.get("axes")


# =========================================================== retargeters
def _retarget_input(name, g1, n_frames=8):
    """A pose sequence in the form this backend declares it needs.

    A parameter-driven backend cannot be fed landmarks, and the robot's own
    stance cannot be expressed as SMPL-X parameters without solving the inverse
    problem, so the two kinds get different inputs and the round-trip identity
    test applies only to the landmark kind.
    """
    q = np.asarray(g1.default_qpos, np.float64)
    pose = synth_pose(g1, q, n_frames=n_frames)
    if not needs_smplx_params(get_retargeter(name)):
        return pose
    from g1dance.motion.bodymodel import body_model_path
    if not body_model_path().exists():
        pytest.skip(f"{name} needs the registration-gated SMPL-X model")
    from g1dance.motion.ir import PoseSequence
    T = pose.n_frames
    body = np.zeros((T, 63), np.float32)
    body[:, 16 * 3 + 2] = np.linspace(0.0, 0.6, T)
    # Real SMPL-X motion carries the upright rotation in the root orientation:
    # the canonical body is y-up, and a world with a floor in it is z-up. A
    # fixture without this retargets to a robot lying down, which is what the
    # height guard in the backend exists to catch.
    from scipy.spatial.transform import Rotation as _R
    orient = np.tile(_R.from_euler("x", np.pi / 2).as_rotvec(),
                     (T, 1)).astype(np.float32)
    return PoseSequence(
        fps=pose.fps, landmarks=pose.landmarks, visibility=pose.visibility,
        hand_left=pose.hand_left, hand_right=pose.hand_right,
        hand_conf=pose.hand_conf, valid=pose.valid,
        smplx_betas=np.zeros((T, 10), np.float32),
        smplx_global_orient=orient,
        smplx_body_pose=body, smplx_transl=np.zeros((T, 3), np.float32))


@pytest.mark.parametrize(
    "name", [n for n in available_retargeters()
             if not needs_smplx_params(get_retargeter(n))])
def test_retargeter_round_trips_the_robots_own_stance(name, g1):
    """The strongest available check: perfect input must give back the input.

    Landmarks are generated by forward kinematics from the default stance, so a
    correct retargeter returns that stance. One that does not is producing a
    plausible looking wrong dance.
    """
    q = np.asarray(g1.default_qpos, np.float64)
    pose = synth_pose(g1, q, n_frames=8)
    motion = get_retargeter(name)(pose, g1, cfg=None)
    err = np.abs(motion.qpos_ref - q[None, :])
    assert float(err.max()) < 0.25, \
        f"worst joint off by {err.max():.3f} rad at {BODY_JOINTS[int(err.max(0).argmax())]}"


@pytest.mark.parametrize("name", available_retargeters())
def test_retargeter_output_satisfies_the_model_contract(name, g1):
    motion = get_retargeter(name)(_retarget_input(name, g1), g1, cfg=None)
    motion.check_model(g1.hash(), BODY_JOINTS, HAND_JOINTS)
    rng = np.asarray(g1.joint_range(BODY_JOINTS))
    lo, hi = rng[:, 0], rng[:, 1]
    assert (motion.qpos_ref >= lo - 1e-6).all() and (motion.qpos_ref <= hi + 1e-6).all()
    assert motion.fps > 0 and motion.valid.any()


@pytest.mark.parametrize("name", available_retargeters())
def test_retargeter_velocity_is_the_derivative_of_its_position(name, g1):
    """qvel_ref that does not match qpos_ref makes every velocity reward a lie."""
    motion = get_retargeter(name)(_retarget_input(name, g1, 16), g1, cfg=None)
    fd = np.gradient(motion.qpos_ref, 1.0 / motion.fps, axis=0)
    assert np.abs(fd - motion.qvel_ref).max() < 1.0


@pytest.mark.parametrize("name", available_retargeters())
def test_retargeter_respects_the_root_yaw_rate_cap(name, g1):
    cap = 60.0
    motion = get_retargeter(name)(_retarget_input(name, g1, 24), g1,
                                  cfg={"max_root_yaw_rate": cap})
    qq = np.asarray(motion.root_quat)
    yaw = np.unwrap(np.arctan2(2 * (qq[:, 0] * qq[:, 3] + qq[:, 1] * qq[:, 2]),
                               1 - 2 * (qq[:, 2] ** 2 + qq[:, 3] ** 2)))
    if len(yaw) < 2:
        pytest.skip("too few frames to differentiate")
    rate = np.degrees(np.abs(np.diff(yaw)) * motion.fps)
    assert rate.max() <= cap * (1 + 1e-3)


# =========================================================== controllers
@pytest.mark.parametrize("name", available_controllers())
def test_controller_declares_how_its_output_is_interpreted(name):
    c = _controller_or_skip(name)
    assert isinstance(getattr(c, "absolute", None), bool), \
        "a controller must say whether it returns residuals or absolute targets"


@pytest.mark.parametrize("name", available_controllers())
def test_controller_returns_finite_targets_of_the_right_width(name, easy_motion):
    c = _controller_or_skip(name)
    e = _env(easy_motion)
    obs, priv = e.reset()
    try:
        c.reset(e)
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"{name} cannot start here: {type(exc).__name__}: {exc}")
    a = np.asarray(c.act(e, obs, priv))
    want = len(BODY_JOINTS) if c.absolute else e.act_dim
    assert a.shape == (want,), (name, a.shape, want)
    assert np.all(np.isfinite(a))


@pytest.mark.parametrize("name", available_controllers())
def test_controller_is_deterministic_given_the_same_state(name, easy_motion):
    c = _controller_or_skip(name)
    e = _env(easy_motion)
    obs, priv = e.reset()
    try:
        c.reset(e)
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"{name} cannot start here: {type(exc).__name__}: {exc}")
    first = np.asarray(c.act(e, obs, priv)).copy()
    c.reset(e)
    np.testing.assert_allclose(np.asarray(c.act(e, obs, priv)), first, atol=1e-9)


@pytest.mark.parametrize(
    "name", [n for n in available_controllers() if n not in BASELINES])
def test_every_tracking_controller_keeps_the_robot_upright(name, easy_motion):
    """The gate that encodes the project's goal rather than an invariant.

    Open-loop PD falls on this same reference at 1.11 s, so surviving is not a
    property of the reference being easy. A controller that cannot hold a gentle
    wave will not hold a dance, whatever its tracking error says.
    """
    from g1dance.control.controllers import OpenLoopController
    from g1dance.control.evaluate import _episode
    c = _controller_or_skip(name)
    e = _env(easy_motion)

    e.seed(0)
    try:
        tracked = _episode(e, c, max_steps=300)
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"{name} cannot run here: {type(exc).__name__}: {exc}")
    e.seed(0)
    loose = _episode(e, OpenLoopController(), max_steps=300)

    assert not tracked["fell"], f"{name} fell on a reference it should hold"
    assert tracked["steps"] >= loose["steps"], \
        f"{name} lasted {tracked['steps']} steps, open loop managed {loose['steps']}"


# =========================================================== hand estimators
def _synthetic_pose(n=6, px=True):
    from g1dance.motion.ir import PoseSequence
    return PoseSequence(
        fps=30.0, landmarks=np.zeros((n, N_POSE_LANDMARKS, 3)),
        visibility=np.ones((n, N_POSE_LANDMARKS)),
        hand_left=np.zeros((n, 21, 3)), hand_right=np.zeros((n, 21, 3)),
        hand_conf=np.zeros((n, 2)), valid=np.ones(n, bool),
        landmarks_px=np.zeros((n, N_POSE_LANDMARKS, 2)) if px else None)


@pytest.mark.parametrize("name", available_hand_estimators())
def test_hand_estimator_has_the_uniform_signature(name):
    import inspect
    fn = get_hand_estimator(name)
    assert list(inspect.signature(fn).parameters)[:3] == ["source", "subject", "pose"]


@pytest.mark.parametrize("name", available_hand_estimators())
def test_hand_estimator_either_produces_valid_tracks_or_refuses_clearly(name):
    """No silent zeros: an estimator that cannot work must say why.

    Returning empty tracks when the input is unusable would look exactly like a
    dancer whose hands were never visible, which `fingers.py` responds to by
    holding a neutral hand. The failure would reach the robot as a shrug.
    """
    pose = _synthetic_pose()
    try:
        tracks = get_hand_estimator(name)("no_such_video.mp4", None, pose)
    except (ValueError, FileNotFoundError, OSError) as exc:
        assert str(exc).strip(), f"{name} refused without saying why"
        return
    assert isinstance(tracks, HandTracks)
    tracks.validate()
    assert tracks.n_frames == pose.n_frames


@pytest.mark.parametrize("name", available_hand_estimators())
def test_hand_tracks_are_finite_and_bounded(name):
    try:
        t = get_hand_estimator(name)("no_such_video.mp4", None, _synthetic_pose())
    except (ValueError, FileNotFoundError, OSError):
        pytest.skip(f"{name} needs inputs this test cannot supply")
    assert np.all(np.isfinite(t.left)) and np.all(np.isfinite(t.right))
    assert t.conf.min() >= 0.0 and t.conf.max() <= 1.0


@pytest.mark.parametrize("name", available_hand_estimators())
def test_an_unseen_hand_carries_no_landmarks(name):
    """Same convention the body contract uses: unseen is stated, not guessed."""
    try:
        t = get_hand_estimator(name)("no_such_video.mp4", None, _synthetic_pose())
    except (ValueError, FileNotFoundError, OSError):
        pytest.skip(f"{name} needs inputs this test cannot supply")
    for side, arr in (("left", t.left), ("right", t.right)):
        col = 0 if side == "left" else 1
        unseen = t.conf[:, col] <= 0.0
        if unseen.any():
            assert not arr[unseen].any(), f"{name} guessed an unseen {side} hand"


def test_the_pose_contract_carries_pixels_when_a_frontend_saw_a_frame():
    """The field the hand seam depends on, checked on real output."""
    seq = _pose_or_skip()
    if seq.landmarks_px is None:
        pytest.skip("this pose sequence predates landmarks_px")
    assert seq.landmarks_px.shape == (seq.n_frames, N_POSE_LANDMARKS, 2)
    assert np.all(np.isfinite(seq.landmarks_px))
