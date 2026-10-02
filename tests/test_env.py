"""Environment tests that need no video and no GPU.

The reference motion is synthesized (a slow sinusoidal arm wave around the
default stance) rather than loaded, which keeps the RL half of the pipeline
testable without the video half, which is the whole reason `ReferenceMotion` is
the contract between them.

What is checked here is the set of things that fail silently: the action and
observation dimensions in each mode, the exclusion of the fingers from both,
determinism under a fixed seed, and each termination condition firing for the
reason it was meant to.
"""

from __future__ import annotations

import numpy as np
import pytest
import mujoco

from g1dance.sim.env import DanceTrackEnv, load_config, synthetic_motion
from g1dance.sim.model import BODY_JOINTS, HAND_JOINTS, UPPER_JOINTS, G1Model

CFG_A = "configs/upper_body.yaml"
CFG_B = "configs/whole_body.yaml"


@pytest.fixture(scope="module")
def g1() -> G1Model:
    return G1Model()


@pytest.fixture(scope="module")
def motion(g1):
    return synthetic_motion(240, fps=30.0, g1=g1)


def env(motion, cfg=CFG_A, seed=0, randomize=False, **over):
    c = load_config(cfg)
    for k, v in over.items():
        if isinstance(v, dict):
            c["env"][k] = {**c["env"].get(k, {}), **v}
        else:
            c["env"][k] = v
    return DanceTrackEnv(motion, c, seed=seed, randomize=randomize)


# --------------------------------------------------------------------- #
# dimensions and the finger exclusion
# --------------------------------------------------------------------- #
def test_mode_a_tracks_17_dof(motion):
    """What distinguishes the modes is the tracked set, tested on the mode.

    Asserting it through a config file would make the invariant only as stable
    as somebody's reward retune.
    """
    e = DanceTrackEnv(motion, {"mode": "upper_body"})
    assert len(e.tracked) == 17 == len(UPPER_JOINTS)
    assert [BODY_JOINTS[i] for i in e.tracked] == UPPER_JOINTS


def test_mode_a_action_drives_every_body_dof_by_default(motion):
    """The legs are in the action even though they are not in the reward.

    This is the correction the measurement forced: held by PD alone the legs do
    not hold, so mode A is IK upper body over RL *lower* body, which needs the
    legs actuated. The reward still sees only the upper 17.
    """
    e = DanceTrackEnv(motion, {"mode": "upper_body"})
    assert e.act_dim == 29
    assert len(e.tracked) == 17
    # +2 phase, +2 heading error as sine and cosine
    assert e.obs_dim == 2 * 29 + 3 + 3 + 29 + 2 + 2 + e.window * (17 + 21)


def test_control_set_tracked_narrows_the_action_to_the_reward_set(motion):
    """The original 17-DoF action is still reachable, as the control that
    demonstrates why the legs belong in the action."""
    e = DanceTrackEnv(motion, {"mode": "upper_body",
                               "env": {"control_set": "tracked"}})
    assert e.act_dim == 17
    assert [BODY_JOINTS[i] for i in e.controlled] == UPPER_JOINTS


def test_mode_b_action_is_29_dof(motion):
    e = DanceTrackEnv(motion, {"mode": "whole_body"})
    assert e.act_dim == 29 == len(BODY_JOINTS)
    assert [BODY_JOINTS[i] for i in e.tracked] == BODY_JOINTS


def test_observation_dims_match_the_documented_layout(motion):
    for md, a in (("upper_body", 17), ("whole_body", 29)):
        e = DanceTrackEnv(motion, {"mode": md,
                                   "env": {"control_set": "tracked"}})
        w = e.window * (a + 3 * len(e.key_ids))
        assert e.obs_dim == 2 * 29 + 3 + 3 + a + 2 + 2 + w
        assert e.priv_dim == e.obs_dim + 15
        o, p = e.reset()
        assert o.shape == (e.obs_dim,) and p.shape == (e.priv_dim,)
        assert e.spec_dims() == (e.obs_dim, e.priv_dim, e.act_dim)


def test_fingers_are_never_in_the_action(motion):
    """The hand command must be the open-loop reference whatever the action is.

    Twelve finger DoF that quietly appeared in the action would take mode A
    from 29 to 41, which is the whole reason Inspire hands cost this RL problem
    nothing.
    """
    e = env(motion, CFG_A)
    e.seed(0)
    e.reset()
    f0 = e.frame
    e.step(np.ones(e.act_dim))
    hi = e.data.ctrl[e.hand_act].copy()
    e.seed(0)
    e.reset()
    assert e.frame == f0
    e.step(-np.ones(e.act_dim))
    lo = e.data.ctrl[e.hand_act].copy()
    assert np.array_equal(hi, lo)
    assert np.allclose(hi, e.track.finger[e.frame])
    assert e.act_dim <= len(BODY_JOINTS) < len(BODY_JOINTS) + len(HAND_JOINTS)


def test_fingers_are_never_in_the_observation(motion):
    """A finger moving must not move a single number the policy sees."""
    e = env(motion, CFG_A)
    o, _ = e.reset()
    e.data.qpos[e.hand_qpos] += 0.4
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    o2, _ = e._observe()
    assert np.array_equal(o, o2)


# --------------------------------------------------------------------- #
# control law
# --------------------------------------------------------------------- #
def test_control_target_is_reference_plus_scaled_action(motion):
    e = env(motion, CFG_A)
    e.reset()
    a = np.full(e.act_dim, 0.5)
    e.step(a)
    ref = e.track.ref_q[e.frame]
    want = ref.copy()
    want[e.controlled] += 0.5 * e.action_scale[e.controlled]
    want = np.clip(want, e.joint_range[:, 0], e.joint_range[:, 1])
    assert np.allclose(e.data.ctrl[e.body_act], want)


def test_untracked_dof_are_commanded_to_the_default_stance(motion, g1):
    """Mode A legs have no reference, so their target is the default stance.

    With `control_set: all` the policy adds a residual on top of that target;
    with a zero action the target must be exactly the stance.
    """
    e = DanceTrackEnv(motion, {"mode": "upper_body"})
    e.reset()
    e.step(np.zeros(e.act_dim))
    leg = np.array([i for i in range(29) if i not in set(e.tracked.tolist())])
    assert np.allclose(e.data.ctrl[e.body_act][leg], g1.default_qpos[leg])


# --------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------- #
def test_two_envs_with_the_same_seed_produce_the_same_trajectory(motion):
    outs = []
    for _ in range(2):
        e = env(motion, CFG_A, seed=7, randomize=True)
        acts = [np.sin(np.arange(e.act_dim) + i) * 0.3 for i in range(40)]
        e.seed(7)
        o, p = e.reset()
        traj = [o.copy()]
        for a in acts:
            o, p, r, term, trunc, _ = e.step(a)
            traj.append(o.copy())
            if term or trunc:
                o, p = e.reset()
        outs.append(np.concatenate(traj))
    assert np.array_equal(outs[0], outs[1])


def test_different_seeds_start_at_different_frames(motion):
    e = env(motion, CFG_A, randomize=True)
    frames = set()
    for s in range(20):
        e.seed(s)
        e.reset()
        frames.add(e.frame)
    assert len(frames) > 1


# --------------------------------------------------------------------- #
# reference state initialization
# --------------------------------------------------------------------- #
def test_reset_starts_inside_a_valid_segment(g1):
    m = synthetic_motion(240, fps=30.0, g1=g1, gap=(80, 120))
    e = env(m, CFG_A, randomize=True)
    assert len(e.track.segments) == 2
    for s in range(30):
        e.seed(s)
        e.reset()
        a, b = e.seg
        assert a <= e.frame < b
        assert e.track.valid[e.frame]


def test_reset_can_start_at_a_chosen_frame(g1):
    """For playing a whole clip: the frame is pinned, the segment follows."""
    m = synthetic_motion(240, fps=30.0, g1=g1, gap=(80, 120))
    e = env(m, CFG_A)
    e.reset(start_frame=0)
    assert e.frame == 0 and e.seg[0] == 0
    late = e.track.segments[1][0] + 5
    e.reset(start_frame=late)
    assert e.frame == late and e.seg == e.track.segments[1]


def test_a_start_frame_in_a_gap_is_refused(g1):
    m = synthetic_motion(240, fps=30.0, g1=g1, gap=(80, 120))
    e = env(m, CFG_A)
    gap = e.track.segments[0][1] + 1
    with pytest.raises(ValueError, match="no valid segment"):
        e.reset(start_frame=gap)


def test_reset_places_the_robot_on_the_reference_pose(motion):
    e = env(motion, CFG_A, randomize=False)
    e.reset()
    q = e.data.qpos[e.body_qpos]
    assert np.allclose(q, e.track.ref_q[e.frame], atol=1e-6)


def test_resampling_lands_on_the_control_rate(motion):
    e = env(motion, CFG_A)
    assert e.control_rate == pytest.approx(50.0)
    assert e.dt == pytest.approx(0.02)
    # 240 frames at 30 fps is 7.967 s of span -> 399 frames at 50 Hz
    assert len(e.track.ref_q) == 399


# --------------------------------------------------------------------- #
# termination and truncation
# --------------------------------------------------------------------- #
def test_termination_on_low_pelvis(motion):
    e = env(motion, CFG_A)
    e.reset()
    e.data.qpos[2] = 0.30
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    assert e._terminated() == (True, "height")


def test_termination_on_tipping_over(motion):
    e = env(motion, CFG_A)
    e.reset()
    e.data.qpos[2] = 1.0                     # high enough that height is fine
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)   # 90 deg about x
    e.data.qpos[3:7] = [c, s, 0.0, 0.0]
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    assert e._terminated() == (True, "upright")


def test_termination_on_joint_divergence(motion):
    e = env(motion, CFG_A)
    e.reset()
    q = e.data.qpos[e.body_qpos].copy()
    q[e.tracked] = np.clip(q[e.tracked] + 1.5, e.joint_range[e.tracked, 0],
                           e.joint_range[e.tracked, 1])
    e.data.qpos[e.body_qpos] = q
    e.data.qpos[2] = 1.0
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    assert e._terminated() == (True, "joint_divergence")


def test_termination_on_link_divergence(motion):
    """Same perturbation, with the joint threshold lifted out of the way."""
    e = env(motion, CFG_A,
            termination={"max_joint_error": 100.0, "max_link_error": 0.05})
    e.reset()
    q = e.data.qpos[e.body_qpos].copy()
    q[e.tracked] = np.clip(q[e.tracked] + 1.5, e.joint_range[e.tracked, 0],
                           e.joint_range[e.tracked, 1])
    e.data.qpos[e.body_qpos] = q
    e.data.qpos[2] = 1.0
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    assert e._terminated() == (True, "link_divergence")


def test_no_termination_in_the_nominal_pose(motion):
    e = env(motion, CFG_A, randomize=False)
    e.reset()
    assert e._terminated() == (False, "")


def test_truncation_at_the_segment_end(motion):
    """Running out of reference frames is a truncation, never a termination.

    The distinction is the whole reason PPO bootstraps at a segment boundary;
    if this reported `term` the value function would be capped at every phrase
    end in the dance.
    """
    e = env(motion, CFG_A, randomize=False, max_episode_steps=10_000)
    e.reset()
    e.frame = e.seg[1] - 2
    o, p, r, term, trunc, info = e.step(np.zeros(e.act_dim))
    assert (term, trunc) == (False, True)
    assert info["reason"] == "truncated"


def test_truncation_at_the_step_cap(motion):
    e = env(motion, CFG_A, randomize=False, max_episode_steps=3)
    e.reset()
    for i in range(3):
        o, p, r, term, trunc, info = e.step(np.zeros(e.act_dim))
    assert (term, trunc) == (False, True)


# --------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------- #
def test_an_episode_runs_and_rewards_are_finite(motion):
    for cfg in (CFG_A, CFG_B):
        e = env(motion, cfg, randomize=True)
        e.reset()
        for _ in range(60):
            o, p, r, term, trunc, _ = e.step(
                np.random.default_rng(0).normal(0, 0.1, e.act_dim))
            assert np.isfinite(o).all() and np.isfinite(p).all()
            assert np.isfinite(r)
            if term or trunc:
                e.reset()
        assert set(e.reward_components) == set(e.reward_fn.names)


def test_a_motion_built_for_another_model_is_rejected(motion, g1):
    import dataclasses
    bad = dataclasses.replace(motion, meta={"model_hash": "deadbeefdeadbeef"})
    with pytest.raises(ValueError, match="re-run retargeting|Re-run"):
        DanceTrackEnv(bad, load_config(CFG_A))


def test_a_motion_with_no_usable_segment_is_rejected(g1):
    m = synthetic_motion(24, fps=30.0, g1=g1)
    m.valid[:] = False
    m.valid[:3] = True
    with pytest.raises(ValueError, match="no valid segment"):
        DanceTrackEnv(m, load_config(CFG_A))


def test_the_shipped_configs_build_an_environment(motion):
    """Whatever the weights have been retuned to, both files must still load,
    name only known reward terms and produce a steppable environment."""
    for cfg in (CFG_A, CFG_B):
        e = env(motion, cfg, randomize=True)
        assert e.reward_fn.names
        o, p, r, term, trunc, _ = e.step(np.zeros(e.act_dim))
        assert np.isfinite(r)


def test_finger_collision_geoms_are_off_in_the_training_model(motion):
    """Spec 7.1. The grafted Inspire thumb intersects its own wrist link in the
    rest pose, so leaving those geoms on charges `self_collision` several
    contacts per step for something no action can change."""
    import mujoco as mj
    e = DanceTrackEnv(motion, {"mode": "upper_body"})
    assert e._n_finger_geoms > 0
    m = e.model
    roots = {int(m.jnt_bodyid[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j)])
             for j in HAND_JOINTS}
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b > 0:
            if b in roots:
                assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
                break
            b = int(m.body_parentid[b])


def test_finger_collisions_can_be_turned_back_on_for_rendering(motion):
    e = DanceTrackEnv(motion, {"mode": "upper_body",
                               "env": {"finger_collisions": True}})
    m = e.model
    on = sum(1 for g in range(m.ngeom)
             if m.geom_contype[g] or m.geom_conaffinity[g])
    off = DanceTrackEnv(motion, {"mode": "upper_body"})
    mo = off.model
    on_off = sum(1 for g in range(mo.ngeom)
                 if mo.geom_contype[g] or mo.geom_conaffinity[g])
    assert on > on_off


def test_the_observation_can_see_heading_error(motion):
    """Everything else in the observation is yaw-invariant.

    Projected gravity does not change when the robot spins about the vertical,
    and the reference window is in the pelvis frame. Without an explicit
    heading term the track_yaw reward asks for something the policy cannot
    perceive, so this asserts the observation actually responds to yaw.
    """
    e = DanceTrackEnv(motion, {"mode": "upper_body"})
    o0, _ = e.reset()
    base = o0.copy()
    # rotate the floating base about the vertical, changing nothing else
    e.data.qpos[3:7] = [np.cos(0.6), 0.0, 0.0, np.sin(0.6)]
    mujoco.mj_forward(e.model, e.data)
    e._build_ctx()
    o1, _ = e._observe()
    assert not np.allclose(base, o1), "observation is blind to heading"
    e.close()
