"""Each reward term against a value computed by hand.

A reward term that is wrong by a sign or a normalization does not raise; it
produces a policy that has learned something other than what was intended, and
the only way that is caught is here. Every expected value below is written out
arithmetically rather than by calling the function a second time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from g1dance.sim.rewards import REGISTRY, RewardContext, RewardFunction


def ctx(**over) -> RewardContext:
    n = 29
    base = dict(
        q=np.zeros(n), qd=np.zeros(n), ref_q=np.zeros(n), ref_qd=np.zeros(n),
        tracked=np.arange(n), link_pos=np.zeros((3, 3)),
        ref_link_pos=np.zeros((3, 3)), link_weight=np.ones(3),
        joint_weight=np.ones(n), root_height=0.69, ref_root_height=0.69,
        upright=1.0, root_xy=np.zeros(2), ref_root_xy=np.zeros(2), yaw=0.0,
        ref_yaw=0.0, foot_pos=np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        foot_vel=np.zeros((2, 3)), foot_contact=np.zeros(2, bool),
        action=np.zeros(4), prev_action=np.zeros(4), torque=np.zeros(n),
        joint_range=np.tile([-1.0, 1.0], (n, 1)), n_self_contacts=0,
        track_root_xy=False)
    base.update(over)
    return RewardContext(**base)


def test_registry_holds_every_named_term():
    assert set(REGISTRY) == {
        "track_joint", "track_link", "track_vel", "track_root", "alive",
        "feet_distance", "feet_slip", "stance_width", "leg_posture",
        "track_yaw", "self_collision", "action_rate", "torque", "joint_limit"}


def test_track_joint_is_one_when_perfect():
    assert REGISTRY["track_joint"](ctx(), k=2.0) == pytest.approx(1.0)


def test_track_joint_normalizes_by_the_weight_sum():
    # two of 29 DoF off by 0.5 each, unit weights:
    # weighted squared error = (0.25 + 0.25) / 29
    q = np.zeros(29)
    q[3] = 0.5
    q[7] = -0.5
    got = REGISTRY["track_joint"](ctx(q=q), k=2.0)
    assert got == pytest.approx(math.exp(-2.0 * 0.5 / 29))


def test_track_joint_only_sees_tracked_dof():
    q = np.zeros(29)
    q[0] = 3.0                       # a leg DoF, outside the tracked set
    c = ctx(q=q, tracked=np.arange(12, 29), joint_weight=np.ones(17))
    assert REGISTRY["track_joint"](c, k=2.0) == pytest.approx(1.0)


def test_track_link_uses_squared_euclidean_distance():
    lp = np.zeros((3, 3))
    lp[1] = [0.03, 0.04, 0.0]        # |d| = 0.05, |d|^2 = 0.0025
    got = REGISTRY["track_link"](ctx(link_pos=lp), k=20.0)
    assert got == pytest.approx(math.exp(-20.0 * 0.0025 / 3))


def test_track_link_ignores_zero_weighted_links():
    lp = np.zeros((3, 3))
    lp[2] = [1.0, 0.0, 0.0]
    w = np.array([1.0, 1.0, 0.0])
    assert REGISTRY["track_link"](ctx(link_pos=lp, link_weight=w),
                                  k=20.0) == pytest.approx(1.0)


def test_track_vel_matches_the_joint_form():
    qd = np.zeros(29)
    qd[5] = 2.0
    got = REGISTRY["track_vel"](ctx(qd=qd), k=0.05)
    assert got == pytest.approx(math.exp(-0.05 * 4.0 / 29))


def test_track_root_penalizes_height_and_tilt():
    c = ctx(root_height=0.79, upright=0.8)      # dh = 0.10, 1 - upright = 0.2
    want = math.exp(-(20.0 * 0.01 + 5.0 * 0.04))
    assert REGISTRY["track_root"](c) == pytest.approx(want)


def test_track_root_adds_xy_and_yaw_only_in_mode_b():
    c = ctx(root_xy=np.array([0.2, 0.0]), yaw=0.5)
    assert REGISTRY["track_root"](c) == pytest.approx(1.0)
    c2 = ctx(root_xy=np.array([0.2, 0.0]), yaw=0.5, track_root_xy=True)
    want = math.exp(-(2.0 * 0.04 + 2.0 * 0.25))
    assert REGISTRY["track_root"](c2) == pytest.approx(want)


def test_track_root_wraps_the_yaw_difference():
    # 3.0 rad vs -3.0 rad is 0.2832 apart the short way, not 6.0
    c = ctx(yaw=3.0, ref_yaw=-3.0, track_root_xy=True)
    dyaw = 2.0 * math.pi - 6.0
    assert REGISTRY["track_root"](c) == pytest.approx(
        math.exp(-2.0 * dyaw ** 2))


def test_alive_is_a_constant():
    assert REGISTRY["alive"](ctx()) == 1.0


def test_feet_distance_is_a_one_sided_hinge():
    far = ctx(foot_pos=np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]]))
    assert REGISTRY["feet_distance"](far, min_dist=0.15) == 0.0
    near = ctx(foot_pos=np.array([[0.0, 0.03, 0.0], [0.0, -0.03, 0.0]]))
    assert REGISTRY["feet_distance"](near, min_dist=0.15) == pytest.approx(0.09)


def test_feet_slip_counts_only_loaded_feet():
    v = np.array([[0.3, 0.4, 9.0], [1.0, 0.0, 0.0]])
    c = ctx(foot_vel=v, foot_contact=np.array([True, False]))
    assert REGISTRY["feet_slip"](c) == pytest.approx(0.25)   # 0.3^2 + 0.4^2


def test_self_collision_counts_contacts():
    assert REGISTRY["self_collision"](ctx(n_self_contacts=3)) == 3.0


def test_action_rate_is_the_mean_squared_change():
    c = ctx(action=np.array([1.0, 0.0, -1.0, 0.0]),
            prev_action=np.array([0.0, 0.0, 0.0, 0.0]))
    assert REGISTRY["action_rate"](c) == pytest.approx(2.0 / 4)


def test_torque_is_the_mean_square():
    t = np.zeros(29)
    t[0], t[1] = 3.0, 4.0
    assert REGISTRY["torque"](ctx(torque=t)) == pytest.approx(25.0 / 29)


def test_joint_limit_measures_excursion_past_the_soft_limit():
    # range [-1, 1], margin 0.05 -> soft limits at +-0.9
    q = np.zeros(29)
    q[0] = 0.95          # 0.05 over
    q[1] = -1.10         # 0.20 under
    q[2] = 0.80          # inside
    assert REGISTRY["joint_limit"](ctx(q=q), margin=0.05) == pytest.approx(0.25)


def test_reward_function_sums_weighted_terms_and_records_components():
    fn = RewardFunction.from_config({
        "alive": {"weight": 0.5},
        "action_rate": {"weight": -0.1},
    })
    c = ctx(action=np.array([1.0, 1.0, 0.0, 0.0]), prev_action=np.zeros(4))
    # 0.5 * 1.0 + (-0.1) * (2/4)
    assert fn(c) == pytest.approx(0.5 - 0.05)
    assert fn.names == ["alive", "action_rate"]
    assert fn.components.tolist() == pytest.approx([1.0, 0.5])


def test_zero_weight_terms_are_dropped():
    fn = RewardFunction.from_config({"alive": {"weight": 0.0},
                                     "torque": {"weight": -1.0}})
    assert fn.names == ["torque"]


def test_unknown_term_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError, match="unknown reward term"):
        RewardFunction.from_config({"track_evrything": {"weight": 1.0}})


def test_stance_width_is_zero_inside_the_bound_and_hinges_outside():
    c = ctx()
    c.foot_pos = np.array([[0.0, 0.15, 0.0], [0.0, -0.15, 0.0]])   # 0.30 apart
    assert REGISTRY["stance_width"](c, max_dist=0.55) == pytest.approx(0.0)
    c.foot_pos = np.array([[0.0, 0.40, 0.0], [0.0, -0.40, 0.0]])   # 0.80 apart
    assert REGISTRY["stance_width"](c, max_dist=0.55) == pytest.approx(0.25)


def test_stance_width_and_feet_distance_bound_opposite_sides():
    """Together they define a band; neither should fire inside it."""
    c = ctx()
    c.foot_pos = np.array([[0.0, 0.15, 0.0], [0.0, -0.15, 0.0]])   # 0.30 apart
    assert REGISTRY["feet_distance"](c, min_dist=0.15) == pytest.approx(0.0)
    assert REGISTRY["stance_width"](c, max_dist=0.55) == pytest.approx(0.0)


def test_leg_posture_is_zero_when_untracked_dof_match_their_reference():
    # the default fixture tracks all 29 DoF, which would make every assertion
    # here pass trivially; mode A tracks 17 of them
    c = ctx(tracked=np.arange(12, 29))
    assert REGISTRY["leg_posture"](c) == pytest.approx(0.0)


def test_leg_posture_ignores_the_tracked_dof():
    """Error on a tracked DoF belongs to track_joint, not here."""
    c = ctx(tracked=np.arange(12, 29))
    c.q = c.q.copy()
    c.q[c.tracked[0]] += 1.0
    assert REGISTRY["leg_posture"](c) == pytest.approx(0.0)


def test_leg_posture_grows_with_untracked_error():
    c = ctx(tracked=np.arange(12, 29))
    c.q = c.q.copy()
    c.q[0] += 0.5
    small = REGISTRY["leg_posture"](c)
    c.q[0] += 0.5
    assert REGISTRY["leg_posture"](c) > small > 0.0


def test_leg_posture_is_zero_when_every_dof_is_tracked():
    """Nothing is untracked in mode B, so the prior must not fire at all."""
    c = ctx(tracked=np.arange(29))
    c.q = c.q.copy()
    c.q[0] += 1.0
    assert REGISTRY["leg_posture"](c) == pytest.approx(0.0)


def test_track_yaw_is_one_when_heading_matches():
    c = ctx(yaw=0.7, ref_yaw=0.7)
    assert REGISTRY["track_yaw"](c, k=2.0) == pytest.approx(1.0)


def test_track_yaw_wraps_across_pi():
    """A heading just past pi is close to one just below it, not 2pi away."""
    import math
    near = REGISTRY["track_yaw"](ctx(yaw=math.pi - 0.05, ref_yaw=-math.pi + 0.05), k=2.0)
    far = REGISTRY["track_yaw"](ctx(yaw=0.0, ref_yaw=math.pi), k=2.0)
    assert near > 0.9
    assert far < 0.01


def test_track_yaw_decreases_as_heading_diverges():
    a = REGISTRY["track_yaw"](ctx(yaw=0.0, ref_yaw=0.2), k=2.0)
    b = REGISTRY["track_yaw"](ctx(yaw=0.0, ref_yaw=0.8), k=2.0)
    assert 1.0 > a > b > 0.0
