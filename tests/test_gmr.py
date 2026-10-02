"""GMR as a second retargeting backend.

The retargeter seam had exactly one entry, and it is the stage the measured
ceiling lives behind: source to reference sits at 0.729 and its dominant cost is
self collision resolution firing on two thirds of the frames. A second entry is
what makes "is our IK the limit" a question that can be answered rather than
argued.

What is tested here is the boundary, not GMR itself. Its joint order agreeing
with ours is checked rather than assumed, because a silent disagreement there
produces a plausible looking wrong dance; and a pose sequence without SMPL-X
parameters is refused rather than approximated, because GMR reconstructs joint
orientations from the pose and a joint position cannot supply one.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.motion.bodymodel import body_model_path
from g1dance.motion.ir import PoseSequence
from g1dance.motion.retargeters import available_retargeters, get_retargeter
from g1dance.sim.model import BODY_JOINTS, HAND_JOINTS, G1Model

HAVE_GMR = True
try:
    import general_motion_retargeting  # noqa: F401
except ImportError:
    HAVE_GMR = False

needs_gmr = pytest.mark.skipif(
    not (HAVE_GMR and body_model_path().exists()),
    reason="needs GMR and the registration-gated SMPL-X model")


@pytest.fixture(scope="module")
def g1() -> G1Model:
    return G1Model()


def _pose(T=40, params=True, fps=30.0):
    kw = dict(fps=fps, landmarks=np.zeros((T, 33, 3)),
              visibility=np.ones((T, 33)), hand_left=np.zeros((T, 21, 3)),
              hand_right=np.zeros((T, 21, 3)), hand_conf=np.zeros((T, 2)),
              valid=np.ones(T, bool))
    if not params:
        return PoseSequence(**kw)
    body = np.zeros((T, 63), np.float32)
    body[:, 16 * 3 + 2] = np.linspace(0.0, 0.8, T)
    # Real SMPL-X motion carries the upright rotation in the root orientation:
    # the canonical body is y-up, and a world with a floor in it is z-up. A
    # fixture without this retargets to a robot lying down, which is what the
    # height guard in the backend exists to catch.
    from scipy.spatial.transform import Rotation as _R
    orient = np.tile(_R.from_euler("x", np.pi / 2).as_rotvec(),
                     (T, 1)).astype(np.float32)
    return PoseSequence(
        **kw, smplx_betas=np.zeros((T, 10), np.float32),
        smplx_global_orient=orient,
        smplx_body_pose=body, smplx_transl=np.zeros((T, 3), np.float32))


# ------------------------------------------------------------------ registry
def test_gmr_is_registered_beside_the_builtin():
    assert {"builtin", "gmr"} <= set(available_retargeters())


def test_gmr_refuses_landmark_only_input(g1):
    """It cannot get an orientation from a position, so it says so."""
    from g1dance.motion.gmr_backend import GmrUnavailable
    if not HAVE_GMR:
        pytest.skip("GMR is not installed here")
    with pytest.raises(GmrUnavailable, match="SMPL-X parameters"):
        get_retargeter("gmr")(_pose(params=False), g1)


# -------------------------------------------------------------- the contract
@needs_gmr
def test_gmr_agrees_with_us_about_joint_order():
    """A silent disagreement here is the most expensive bug in this project."""
    import mujoco

    import general_motion_retargeting as gmr
    from g1dance.motion.gmr_backend import GMR_ROBOT
    m = mujoco.MjModel.from_xml_path(str(gmr.ROBOT_XML_DICT[GMR_ROBOT]))
    hinges = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
              for i in range(m.njnt)
              if m.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    assert hinges == list(BODY_JOINTS)


@needs_gmr
def test_gmr_produces_a_reference_the_model_accepts(g1):
    ref = get_retargeter("gmr")(_pose(), g1)
    ref.check_model(g1.hash(), BODY_JOINTS, HAND_JOINTS)
    assert ref.fps == 50.0
    assert ref.n_frames > 0 and ref.valid.any()
    assert ref.meta["retargeter"] == "gmr"


@needs_gmr
def test_gmr_output_stays_inside_the_joint_limits(g1):
    ref = get_retargeter("gmr")(_pose(), g1)
    lo, hi = np.asarray(g1.joint_range(BODY_JOINTS)).T
    assert (ref.qpos_ref >= lo - 1e-6).all() and (ref.qpos_ref <= hi + 1e-6).all()


@needs_gmr
def test_gmr_honours_the_root_yaw_rate_cap(g1):
    """The cap is a property of a reference, so every backend must respect it."""
    ref = get_retargeter("gmr")(_pose(), g1, cfg={"max_root_yaw_rate": 30.0})
    q = np.asarray(ref.root_quat)
    yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                               1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
    if len(yaw) > 1:
        assert np.degrees(np.abs(np.diff(yaw)) * ref.fps).max() <= 30.0 * 1.001


@needs_gmr
def test_gmr_actually_follows_the_input(g1):
    """A backend returning the default stance would pass every test above."""
    ref = get_retargeter("gmr")(_pose(), g1)
    span = float(np.abs(ref.qpos_ref.max(0) - ref.qpos_ref.min(0)).max())
    assert span > 0.2, f"the reference barely moves: {span:.3f} rad"


@needs_gmr
def test_gmr_velocity_is_the_derivative_of_its_position(g1):
    ref = get_retargeter("gmr")(_pose(), g1)
    fd = np.gradient(ref.qpos_ref, 1.0 / ref.fps, axis=0)
    assert np.abs(fd - ref.qvel_ref).max() < 1.0


@needs_gmr
def test_gmr_puts_the_robot_on_the_floor(g1):
    """The failure this backend is most likely to produce, and it never raises.

    SMPL-X's rest pelvis sits near the origin with the feet nearly a metre
    below, so a source whose translation is not ground referenced retargets to a
    robot underneath the floor. Measured before the fix: a pelvis at 0.010 m.
    Nothing downstream complains; the robot just falls.
    """
    ref = get_retargeter("gmr")(_pose(), g1)
    h = float(np.median(np.asarray(ref.root_pos)[:, 2]))
    assert 0.3 <= h <= 1.5, f"pelvis median height {h:.3f} m"


@needs_gmr
def test_an_implausible_height_is_refused_rather_than_returned(g1, monkeypatch):
    """If the guard ever stops firing, this is what notices."""
    from g1dance.motion import gmr_backend
    import general_motion_retargeting as gmr_pkg

    real = gmr_pkg.GeneralMotionRetargeting

    class Sunk(real):
        def retarget(self, frame, offset_to_ground=False):
            q = super().retarget(frame, offset_to_ground=offset_to_ground)
            q = np.asarray(q).copy()
            q[2] = 0.01                      # a pelvis under the floor
            return q

    monkeypatch.setattr(gmr_pkg, "GeneralMotionRetargeting", Sunk)
    with pytest.raises(gmr_backend.GmrUnavailable, match="standing on a floor"):
        get_retargeter("gmr")(_pose(), g1)
