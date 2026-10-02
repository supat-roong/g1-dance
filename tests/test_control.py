"""The controller seam that lets an external tracker drive this environment.

`step` was written for training, where the action is a residual the policy adds
to the reference and is clipped to the unit box. A general tracking policy does
not work that way: it emits absolute motor targets in its own units, and routing
those through the residual path would clip them to plus or minus one action
scale around the reference. That does not fail, it quietly saturates, which is
the same failure mode as a wrong joint order.

So what is tested here is that the absolute path exists, that it agrees with the
residual path when handed equivalent targets, and that saturation is counted and
reported rather than absorbed.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1dance.control.controllers import (
    Controller, OpenLoopController, ZeroActionController, make_controller)
from g1dance.sim.env import DanceTrackEnv, load_config, synthetic_motion
from g1dance.sim.model import BODY_JOINTS, G1Model

CFG_A = "configs/upper_body.yaml"


@pytest.fixture(scope="module")
def g1() -> G1Model:
    return G1Model()


@pytest.fixture(scope="module")
def motion(g1):
    return synthetic_motion(240, fps=30.0, g1=g1)


def env(motion, cfg=CFG_A, seed=0, **over):
    c = load_config(cfg)
    for k, v in over.items():
        c["env"][k] = v
    return DanceTrackEnv(motion, c, seed=seed, randomize=False)


# ------------------------------------------------------------ absolute path
def test_absolute_targets_reach_the_actuators(motion):
    e = env(motion)
    e.reset()
    tgt = e.track.ref_q[e.frame + 1].copy()
    tgt[:] = np.clip(tgt + 0.05, e.joint_range[:, 0], e.joint_range[:, 1])
    e.step(tgt, absolute=True)
    np.testing.assert_allclose(e.data.ctrl[e.body_act], tgt, atol=1e-9)


def test_absolute_and_residual_agree_on_equivalent_targets(motion):
    """The clamp test: the two paths must land on the same actuator targets."""
    a = np.full(len(BODY_JOINTS), 0.3)

    e1, e2 = env(motion), env(motion)
    e1.reset()
    e2.reset()
    assert e1.frame == e2.frame, "same seed must start from the same frame"

    f1 = min(e1.frame + 1, len(e1.track.ref_q) - 1)
    expected = e1.track.ref_q[f1].copy()
    expected[e1.controlled] += e1.action_scale[e1.controlled] * a
    expected = np.clip(expected, e1.joint_range[:, 0], e1.joint_range[:, 1])

    e1.step(a)
    e2.step(expected, absolute=True)

    np.testing.assert_allclose(e1.data.ctrl[e1.body_act],
                              e2.data.ctrl[e2.body_act], atol=1e-9)


def test_absolute_path_is_not_clipped_to_the_unit_box(motion):
    """A target of 0.8 rad must survive; through the residual path it would not."""
    e = env(motion)
    e.reset()
    j = BODY_JOINTS.index("left_shoulder_pitch_joint")
    tgt = e.track.ref_q[e.frame + 1].copy()
    tgt[j] = np.clip(0.8, e.joint_range[j, 0], e.joint_range[j, 1])
    e.step(tgt, absolute=True)
    assert e.data.ctrl[e.body_act][j] == pytest.approx(tgt[j], abs=1e-9)


def test_out_of_range_targets_are_clipped_and_counted(motion):
    e = env(motion)
    e.reset()
    tgt = np.full(len(BODY_JOINTS), 1e3)
    e.step(tgt, absolute=True)
    ctrl = e.data.ctrl[e.body_act]
    assert np.all(ctrl <= e.joint_range[:, 1] + 1e-9)
    # A per-step attribute rather than an info key: `info` is deliberately
    # empty except at episode end, for the allocation cost in training.
    assert e.target_clip_frac == pytest.approx(1.0)


def test_in_range_targets_report_no_clipping(motion):
    e = env(motion)
    e.reset()
    tgt = e.track.ref_q[e.frame + 1].copy()
    e.step(tgt, absolute=True)
    assert e.target_clip_frac == pytest.approx(0.0)


def test_residual_path_still_clips_the_action(motion):
    """Training behaviour must be untouched by the new keyword."""
    e = env(motion)
    e.reset()
    f = min(e.frame + 1, len(e.track.ref_q) - 1)
    expected = e.track.ref_q[f].copy()
    expected[e.controlled] += e.action_scale[e.controlled] * 1.0
    expected = np.clip(expected, e.joint_range[:, 0], e.joint_range[:, 1])
    e.step(np.full(len(BODY_JOINTS), 5.0))       # far outside the unit box
    np.testing.assert_allclose(e.data.ctrl[e.body_act], expected, atol=1e-9)


# ---------------------------------------------------------------- controllers
def test_open_loop_controller_is_a_zero_residual(motion):
    e = env(motion)
    obs, priv = e.reset()
    c = OpenLoopController()
    c.reset(e)
    assert c.absolute is False
    np.testing.assert_allclose(c.act(e, obs, priv), 0.0)


def test_zero_action_controller_targets_the_default_stance(motion):
    e = env(motion)
    obs, priv = e.reset()
    c = ZeroActionController()
    c.reset(e)
    assert c.absolute is True
    np.testing.assert_allclose(c.act(e, obs, priv), e.g1.default_qpos, atol=1e-9)


def test_controllers_satisfy_the_protocol():
    for c in (OpenLoopController(), ZeroActionController()):
        assert isinstance(c, Controller)


def test_make_controller_names_the_alternatives():
    with pytest.raises(ValueError, match="open_loop"):
        make_controller("no_such_controller")


def test_an_episode_runs_end_to_end_under_a_controller(motion):
    """The loop eval and render both use: reset, then act and step to the end."""
    e = env(motion)
    obs, priv = e.reset()
    c = OpenLoopController()
    c.reset(e)
    for _ in range(20):
        obs, priv, _, term, trunc, _ = e.step(c.act(e, obs, priv),
                                              absolute=c.absolute)
        if term or trunc:
            break
    assert e.t > 0
